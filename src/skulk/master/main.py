import copy
import os
import time
from collections import deque
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal, cast, final

import anyio
import yaml
from loguru import logger

from skulk.master.placement import (
    PlacementError,
    PlacementInfoPendingError,
    PlacementModelCardIdentityError,
    add_instance_to_placements,
    cancel_unnecessary_downloads,
    delete_instance,
    fallback_command_for_refused_instance,
    get_transition_events,
    place_instance,
    replacement_command_for_download_failed_instance,
    replacement_command_for_refused_instance,
    require_instance_model_card_identity,
)
from skulk.master.placement_utils import (
    reserve_instance_vram,
    unified_memory_gpu_node_ids,
    usable_vram_by_node,
)
from skulk.shared.apply import apply
from skulk.shared.constants import SKULK_EVENT_LOG_DIR, SKULK_TRACING_ENABLED
from skulk.shared.log_summaries import summarize_command_for_log
from skulk.shared.models.memory_estimate import (
    estimate_shard_footprint,
    shard_fraction_of_model,
)
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    get_card,
    get_current_registry_card,
    get_custom_card_storage_collision,
    get_model_cards,
    same_authorized_model_card,
)
from skulk.shared.types.commands import (
    AddCustomModelCard,
    AudioTranscription,
    CancelDownload,
    CreateInstance,
    DecideStewardAction,
    DeleteCustomModelCard,
    DeleteInstance,
    EvictStagedModel,
    FailInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    ProposeStewardAction,
    RealtimeAudioTranscription,
    RefuseInstancePlacement,
    RequestEventLog,
    SendInputChunk,
    SetModelTrustApproval,
    SetTracingEnabled,
    SpeechSynthesis,
    StartDownload,
    TaskCancelled,
    TaskFinished,
    TestCommand,
    TextEmbedding,
    TextGeneration,
)
from skulk.shared.types.common import CommandId, NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    InstanceFailureRecorded,
    LocalForwarderEvent,
    ModelTrustApprovalChanged,
    NodeDownloadProgress,
    NodeGatheredInfo,
    NodeTimedOut,
    NodeTimeoutEvidence,
    StagedModelEvicted,
    StateSnapshotHydrated,
    StewardActionProposalChanged,
    TaskCreated,
    TaskDeleted,
    TaskFailed,
    TaskStatusUpdated,
    TracingStateChanged,
    is_persistable_control_event,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import MemoryUsage
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSnapshot, StateSyncMessage
from skulk.shared.types.steward_actions import (
    StewardActionProposal,
    StewardActionProposalId,
    StewardCancelDownloadAction,
    StewardPlaceModelAction,
    StewardRestartInstanceAction,
    StewardStopInstanceAction,
    steward_action_proposal_is_prunable,
)
from skulk.shared.types.tasks import (
    AudioTranscription as AudioTranscriptionTask,
)
from skulk.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from skulk.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from skulk.shared.types.tasks import (
    RealtimeAudioTranscription as RealtimeAudioTranscriptionTask,
)
from skulk.shared.types.tasks import (
    SpeechSynthesis as SpeechSynthesisTask,
)
from skulk.shared.types.tasks import (
    TaskId,
    TaskStatus,
)
from skulk.shared.types.tasks import (
    TextEmbedding as TextEmbeddingTask,
)
from skulk.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from skulk.shared.types.telemetry import (
    NODE_LIVENESS_TIMEOUT,
    TelemetryView,
    record_membership_from_event,
)
from skulk.shared.types.worker.downloads import (
    DownloadAttemptId,
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from skulk.shared.types.worker.instances import (
    Instance,
    InstanceFailure,
    InstanceFailureCode,
    InstanceId,
    InstanceMeta,
)
from skulk.shared.types.worker.runners import RunnerReady, RunnerRunning
from skulk.shared.types.worker.shards import (
    RpcDonorShardMetadata,
    Sharding,
    ShardMetadata,
)
from skulk.store.config import (
    load_skulk_config,
    persist_model_trust_config,
    resolve_config_path,
)
from skulk.utils.channels import Receiver, Sender
from skulk.utils.disk_event_log import DiskEventLog
from skulk.utils.event_buffer import MultiSourceBuffer
from skulk.utils.state_snapshot_store import StateSnapshotStore
from skulk.utils.task_group import TaskGroup

EVENT_LOG_REPLAY_BATCH_SIZE = 10_000
EVENT_LOG_REPLAY_CHUNK_SIZE = 32
EVENT_LOG_REPLAY_CHUNK_INTERVAL_SECONDS = 0.25
SNAPSHOT_EVENT_CADENCE = 10_000
REPLAY_TAIL_RETENTION_EVENTS = SNAPSHOT_EVENT_CADENCE
EVENT_LOG_GROWTH_WINDOW_SECONDS = 60.0
EVENT_LOG_IDLE_GROWTH_WARNING_EVENTS_PER_MINUTE = 60.0
EVENT_LOG_GROWTH_WARNING_COOLDOWN_SECONDS = 300.0
NON_CONTROL_EVENT_WARNING_COOLDOWN_SECONDS = 60.0
NON_CONTROL_EVENT_WARNING_KEY_LIMIT = 256
STEWARD_UPGRADE_STABILITY_SECONDS = 300.0
STEWARD_UPGRADE_IDLE_SECONDS = 30.0
STEWARD_UPGRADE_RETRY_COOLDOWN_SECONDS = 1800.0


@final
@dataclass(slots=True)
class EventLogGrowthMonitor:
    """Detect sustained event-log growth while the master has no active work."""

    window_seconds: float = EVENT_LOG_GROWTH_WINDOW_SECONDS
    warning_rate_per_minute: float = EVENT_LOG_IDLE_GROWTH_WARNING_EVENTS_PER_MINUTE
    warning_cooldown_seconds: float = EVENT_LOG_GROWTH_WARNING_COOLDOWN_SECONDS
    _idle_event_times: deque[float] = field(default_factory=deque)
    _idle_since: float | None = None
    _last_warning_at: float | None = None

    def observe(self, *, now: float, idle: bool) -> float | None:
        """Record one indexed event and return its warning rate when elevated.

        Active placement, download, and inference work resets the idle window so
        legitimate lifecycle bursts cannot prime a later warning. The returned
        rate is limited by ``warning_cooldown_seconds``; callers may log it or
        expose it through operator diagnostics without retaining event payloads.
        """

        if not idle:
            self._idle_event_times.clear()
            self._idle_since = None
            return None

        if self._idle_since is None:
            self._idle_since = now
        self._idle_event_times.append(now)
        cutoff = now - self.window_seconds
        while self._idle_event_times and self._idle_event_times[0] < cutoff:
            self._idle_event_times.popleft()

        observed_seconds = now - self._idle_since
        if observed_seconds < self.window_seconds:
            return None
        if (
            self._last_warning_at is not None
            and now - self._last_warning_at < self.warning_cooldown_seconds
        ):
            return None

        rate = len(self._idle_event_times) * 60.0 / self.window_seconds
        if rate < self.warning_rate_per_minute:
            return None
        self._last_warning_at = now
        return rate


TOPOLOGY_SETTLE_GRACE_SECONDS = 60.0
"""How long after master start the plan loop trusts topology for pruning.

A new session's topology starts empty and is rebuilt from live gossip:
worker connection probes re-emit edges on a 10s cycle, plus router/mDNS
events. A failover-seeded master (#273) carries instances from the prior
session but deliberately NOT the prior topology (a dead node's out-edges
would persist forever — only their source node ever deletes them), so for
the first moments every carried instance's nodes look "disconnected".
Pruning during that window would delete the very placements the seed
preserved. 60s comfortably covers several probe cycles; the dead master's
instances are still pruned — just one minute later, once absence reflects
real liveness rather than an unsettled view."""

RECENTLY_FREED_MEMORY_GRACE_SECONDS = 0.0
"""How long the planner credits just-deleted instance memory during placement.

This is intentionally disabled by default. The old 30s credit bridged gossip
lag after teardown, but live MLX runs showed the inverse failure: Metal memory
can remain unwired locally after the master has deleted the instance, so
crediting the freed footprint over-admits the next placement and leaves the
worker to refuse it. Placement must prefer the observed telemetry over an
optimistic teardown credit; the worker's local guard should be a last resort,
not the normal correction path."""
JsonObject = dict[str, object]

# API-facing task types: the ones whose loss strands an open HTTP request.
# Worker lifecycle tasks (CreateRunner, LoadModel, ...) are reconciled by the
# worker's own plan loop and must not be failed from here.
_COMMAND_TASK_TYPES = (
    TextGenerationTask,
    ImageGenerationTask,
    ImageEditsTask,
    TextEmbeddingTask,
    SpeechSynthesisTask,
    AudioTranscriptionTask,
    RealtimeAudioTranscriptionTask,
)


NODE_HEARTBEAT_GAP_WARNING = timedelta(seconds=10)
_INSTANCE_FAILURE_MESSAGE_LIMIT = 2048


def _node_unavailable_failure_message(node_id: NodeId, reason: str) -> str:
    """Return a bounded liveness explanation for an unconstrained node id."""
    prefix = "The placement was torn down because assigned node "
    suffix = f" {reason}."
    available_node_characters = max(
        0, _INSTANCE_FAILURE_MESSAGE_LIMIT - len(prefix) - len(suffix)
    )
    return f"{prefix}{str(node_id)[:available_node_characters]}{suffix}"


def instance_failure_event(
    instance: Instance,
    *,
    error_code: InstanceFailureCode,
    error_message: str,
    recorded_at: datetime | None = None,
) -> InstanceFailureRecorded:
    """Build durable operator truth while the failed placement still exists.

    Args:
        instance: Placement whose terminal failure is being retained.
        error_code: Stable operator-facing category for the failure.
        error_message: Bounded, payload-safe explanation shown to operators.
        recorded_at: Optional authoritative occurrence time. Defaults to the
            current UTC time when omitted.

    Returns:
        A new failure event containing the placement identity and assigned
        nodes. Constructing the event does not mutate cluster state or emit it.
    """
    return InstanceFailureRecorded(
        failure=InstanceFailure(
            instance_id=instance.instance_id,
            model_id=instance.shard_assignments.model_id,
            system_role=instance.system_role,
            error_code=error_code,
            error_message=error_message,
            affected_node_ids=tuple(instance.shard_assignments.node_to_runner),
            recorded_at=recorded_at or datetime.now(tz=timezone.utc),
        )
    )


def dead_node_instance_failure_events(
    state: State,
    connected_node_ids: AbstractSet[NodeId],
    timed_out_node_ids: AbstractSet[NodeId],
) -> list[InstanceFailureRecorded]:
    """Return one retained failure for every placement affected by node loss.

    A timed-out node can remain in the replicated topology until
    :class:`NodeTimedOut` applies, so topology absence alone is insufficient.
    This helper intentionally considers both signals and leaves event emission
    and subsequent teardown to the planning loop.

    Args:
        state: Current immutable cluster state containing live placements.
        connected_node_ids: Nodes currently present in topology.
        timed_out_node_ids: Nodes whose liveness evidence has expired, even if
            their topology entry has not yet been removed.

    Returns:
        One payload-safe failure event per affected placement. The helper does
        not mutate state, emit events, or tear down placements.
    """
    failures: list[InstanceFailureRecorded] = []
    for instance in state.instances.values():
        unavailable_nodes = sorted(
            node_id
            for node_id in instance.shard_assignments.node_to_runner
            if node_id not in connected_node_ids or node_id in timed_out_node_ids
        )
        if not unavailable_nodes:
            continue
        node_id = unavailable_nodes[0]
        reason = (
            "timed out" if node_id in timed_out_node_ids else "left the live topology"
        )
        failures.append(
            instance_failure_event(
                instance,
                error_code="node_unavailable",
                error_message=_node_unavailable_failure_message(node_id, reason),
            )
        )
    return failures


def _aware_timestamp(when: datetime) -> datetime:
    """Return a timestamp that is safe to compare with UTC receipt times."""
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def _signal_age_seconds(*, now: datetime, seen_at: datetime) -> float:
    """Return a non-negative signal age, defensively tolerating clock drift."""
    return max(
        0.0,
        (_aware_timestamp(now) - _aware_timestamp(seen_at)).total_seconds(),
    )


def compute_node_timeout_evidence(
    last_seen: Mapping[NodeId, datetime],
    heartbeat_last_seen: Mapping[NodeId, datetime],
    telemetry_last_seen: Mapping[NodeId, datetime],
    *,
    now: datetime,
    timeout: timedelta = NODE_LIVENESS_TIMEOUT,
) -> dict[NodeId, NodeTimeoutEvidence]:
    """Build reproducible liveness evidence for every event-known node.

    Args:
        last_seen: Last indexed control-plane event per node.
        heartbeat_last_seen: Local receipt time of each dedicated heartbeat.
        telemetry_last_seen: Local receipt time of ordinary fallback telemetry.
        now: Decision wall clock, injected for deterministic tests.
        timeout: Configured node-prune timeout.

    Returns:
        Evidence keyed by every node represented in ``last_seen``.
    """
    evidence: dict[NodeId, NodeTimeoutEvidence] = {}
    for node_id, last_logged_event_at in last_seen.items():
        last_logged_event_age = _signal_age_seconds(
            now=now, seen_at=last_logged_event_at
        )
        heartbeat_at = heartbeat_last_seen.get(node_id)
        heartbeat_age = (
            _signal_age_seconds(now=now, seen_at=heartbeat_at)
            if heartbeat_at is not None
            else None
        )
        fallback_telemetry_at = telemetry_last_seen.get(node_id)
        fallback_telemetry_age = (
            _signal_age_seconds(now=now, seen_at=fallback_telemetry_at)
            if fallback_telemetry_at is not None
            else None
        )
        available_ages = [last_logged_event_age]
        if heartbeat_age is not None:
            available_ages.append(heartbeat_age)
        if fallback_telemetry_age is not None:
            available_ages.append(fallback_telemetry_age)
        evidence[node_id] = NodeTimeoutEvidence(
            last_logged_event_age_seconds=last_logged_event_age,
            heartbeat_age_seconds=heartbeat_age,
            fallback_telemetry_age_seconds=fallback_telemetry_age,
            effective_age_seconds=min(available_ages),
            timeout_seconds=timeout.total_seconds(),
        )
    return evidence


def _timed_out_nodes_from_evidence(
    evidence: Mapping[NodeId, NodeTimeoutEvidence],
) -> set[NodeId]:
    """Select nodes whose freshest liveness signal exceeds the timeout."""
    return {
        node_id
        for node_id, observation in evidence.items()
        if observation.effective_age_seconds > observation.timeout_seconds
    }


def compute_timed_out_nodes(
    last_seen: Mapping[NodeId, datetime],
    heartbeat_last_seen: Mapping[NodeId, datetime],
    telemetry_last_seen: Mapping[NodeId, datetime],
    *,
    now: datetime,
    timeout: timedelta = NODE_LIVENESS_TIMEOUT,
) -> set[NodeId]:
    """Return nodes whose event, heartbeat, and telemetry signals are stale.

    Args:
        last_seen: ``State.last_seen`` (last logged event per node).
        heartbeat_last_seen: Local receipt time of each dedicated heartbeat.
        telemetry_last_seen: Local receipt time of ordinary fallback telemetry.
        now: The wall clock (tz-aware); injected for testing.
        timeout: Staleness past which a node is considered gone.

    Returns:
        Node ids to time out.
    """
    return _timed_out_nodes_from_evidence(
        compute_node_timeout_evidence(
            last_seen,
            heartbeat_last_seen,
            telemetry_last_seen,
            now=now,
            timeout=timeout,
        )
    )


def compute_heartbeat_gap_nodes(
    last_seen: Mapping[NodeId, datetime],
    heartbeat_last_seen: Mapping[NodeId, datetime],
    *,
    now: datetime,
    warning_after: timedelta = NODE_HEARTBEAT_GAP_WARNING,
) -> set[NodeId]:
    """Return live members whose dedicated heartbeat is late or absent.

    Before a node's first heartbeat, its last indexed event anchors the warning
    window so a new member gets a full grace period without making a missing
    heartbeat invisible forever.
    """
    return {
        node_id
        for node_id, last_logged_event_at in last_seen.items()
        if _signal_age_seconds(
            now=now,
            seen_at=heartbeat_last_seen.get(node_id, last_logged_event_at),
        )
        >= warning_after.total_seconds()
    }


def instances_on_dead_nodes(
    state: State,
    connected_node_ids: AbstractSet[NodeId],
    timed_out_node_ids: AbstractSet[NodeId],
) -> set[InstanceId]:
    """Instances with at least one shard on a disconnected or timed-out node.

    Timed-out nodes matter even while still present in topology: NodeTimedOut
    removes the node's instances AND their tasks from state in one apply, so
    any TaskFailed for those tasks must be emitted before that event — a
    later plan pass would no longer see them (#224 review catch).
    """
    dying: set[InstanceId] = set()
    for instance_id, instance in state.instances.items():
        for node_id in instance.shard_assignments.node_to_runner:
            if node_id not in connected_node_ids or node_id in timed_out_node_ids:
                dying.add(instance_id)
                break
    return dying


def orphaned_task_failure_events(
    state: State,
    dying_instance_ids: AbstractSet[InstanceId],
) -> list[TaskFailed]:
    """Fail in-flight API tasks whose instance is gone or being torn down.

    Without this, a node death mid-generation leaves the task in state
    forever and the API's chunk queue never receives a terminal chunk — the
    client request hangs until its own timeout (issue #223). The master is
    the only component with the global view to declare these tasks dead.

    Pure function of the master's current state so it can be tested without
    channel plumbing; ``dying_instance_ids`` covers instances whose
    InstanceDeleted was emitted in the same plan pass (state still lists
    them until the event round-trips through indexing and apply).
    """
    events: list[TaskFailed] = []
    for task_id, task in state.tasks.items():
        if not isinstance(task, _COMMAND_TASK_TYPES):
            continue
        if task.task_status not in (TaskStatus.Pending, TaskStatus.Running):
            continue
        instance_gone = (
            task.instance_id not in state.instances
            or task.instance_id in dying_instance_ids
        )
        if not instance_gone:
            continue
        events.append(
            TaskFailed(
                task_id=task_id,
                error_type="instance_lost",
                error_message=(
                    "The instance executing this request was lost "
                    "(node disconnected or instance deleted)"
                ),
            )
        )
    return events


#: A lifecycle task whose instance no longer exists can only ever be
#: completed by the worker that was executing it. Healthy teardown completes
#: in seconds; past this grace the executor is presumed gone for good (a
#: killed node returns with a NEW ephemeral identity that knows nothing of
#: the old tasks, #647) and the master declares the task dead.
ORPHANED_LIFECYCLE_TASK_GRACE_SECONDS = 60.0


def stale_lifecycle_task_failures(
    state: State,
    first_seen_orphaned: dict[TaskId, float],
    *,
    now: float,
    grace_seconds: float = ORPHANED_LIFECYCLE_TASK_GRACE_SECONDS,
) -> list[TaskFailed]:
    """Fail lifecycle tasks orphaned past the grace by a vanished executor.

    The #223 pass covers API-facing command tasks via their instance. The
    complementary gap (#647): worker LIFECYCLE tasks (Shutdown, CreateRunner,
    LoadModel, ...) belonging to an already-deleted instance are normally
    reconciled by the executing worker's own plan loop, but when that worker
    died ungracefully its restarted process carries a new node identity and
    can never report; the task sits Pending/Running in state forever, and
    anything waiting on task convergence hangs. Instance deletion also
    removed the task-to-node attribution, so the reap is grace-based rather
    than membership-based: a lifecycle task that stays non-terminal with no
    instance for longer than a healthy teardown could possibly take has no
    living executor.

    Mutates ``first_seen_orphaned`` (master-local tracking, not state):
    stamps newly orphaned tasks, drops tasks that left the orphaned
    condition, and drops tasks it emits for (TaskFailed flips the status on
    apply, so each task is emitted at most once; if the emit were ever lost
    the task re-stamps and gets another full grace, which is self-healing).

    Args:
        state: The master's current applied state.
        first_seen_orphaned: Monotonic first-observation stamps, owned by
            the caller across ticks.
        now: Current monotonic time.
        grace_seconds: How long a task may stay orphaned before it is
            declared dead.

    Returns:
        Terminal ``TaskFailed`` events to index through the ordered path.
    """
    events: list[TaskFailed] = []
    orphaned_now: set[TaskId] = set()
    for task_id, task in state.tasks.items():
        if isinstance(task, _COMMAND_TASK_TYPES):
            continue  # the #223 pass owns API-facing tasks
        if task.task_status not in (TaskStatus.Pending, TaskStatus.Running):
            continue
        if task.instance_id in state.instances:
            continue
        orphaned_now.add(task_id)
        first_seen = first_seen_orphaned.setdefault(task_id, now)
        if now - first_seen >= grace_seconds:
            events.append(
                TaskFailed(
                    task_id=task_id,
                    error_type="executor_lost",
                    error_message=(
                        "Lifecycle task outlived its instance with no "
                        "surviving executor"
                    ),
                )
            )
    emitted = {event.task_id for event in events}
    for task_id in list(first_seen_orphaned):
        if task_id not in orphaned_now or task_id in emitted:
            del first_seen_orphaned[task_id]
    return events


def instances_wedged_by_download_failure(
    state: State,
) -> dict[InstanceId, tuple[frozenset[NodeId], str]]:
    """Not-yet-ready instances that can never load because a rank's download failed.

    A multi-node instance forms its ring and every rank waits for all ranks to
    become load-ready. If one rank's model download terminally fails (disk full,
    transient HF or network error) that rank never advances, so the whole
    instance sits at ``RunnerConnected`` forever with nothing to fail or recover
    it (#381). This detects exactly that wedge from the master's own state: an
    instance not all-ready whose any rank node carries a terminal
    ``DownloadFailed`` for the instance's model. Returns, per wedged instance,
    the failed node id(s) and a human-readable cause. Pure for testability.

    A fully-ready instance (all runners ``RunnerReady``/``RunnerRunning``) is
    never reported even if a stale ``DownloadFailed`` lingers in state, so a
    serving instance is never torn down by this path.
    """
    wedged: dict[InstanceId, tuple[frozenset[NodeId], str]] = {}
    for instance_id, instance in state.instances.items():
        runner_ids = list(instance.shard_assignments.runner_to_shard.keys())
        if not runner_ids:
            continue
        all_ready = all(
            isinstance(state.runners.get(runner_id), (RunnerReady, RunnerRunning))
            for runner_id in runner_ids
        )
        if all_ready:
            continue
        shards = list(instance.shard_assignments.runner_to_shard.values())
        model_id = shards[0].model_card.model_id
        failed_nodes: set[NodeId] = set()
        cause = ""
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
            # RPC donors never download the model, so a stale DownloadFailed
            # for this model on a donor node (from an earlier placement) must
            # not condemn a pooled instance that only needs the DRIVER's copy.
            shard = instance.shard_assignments.runner_to_shard.get(runner_id)
            if isinstance(shard, RpcDonorShardMetadata):
                continue
            for progress in state.downloads.get(node_id, []):
                if (
                    isinstance(progress, DownloadFailed)
                    and progress.shard_metadata.model_card.model_id == model_id
                ):
                    failed_nodes.add(node_id)
                    cause = progress.error_message
        if failed_nodes:
            wedged[instance_id] = (frozenset(failed_nodes), cause)
    return wedged


class Master:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        command_receiver: Receiver[ForwarderCommand],
        event_sender: Sender[Event],
        local_event_receiver: Receiver[LocalForwarderEvent],
        global_event_sender: Sender[GlobalForwarderEvent],
        state_sync_receiver: Receiver[StateSyncMessage],
        state_sync_sender: Sender[StateSyncMessage],
        download_command_sender: Sender[ForwarderDownloadCommand],
        snapshot_event_cadence: int = SNAPSHOT_EVENT_CADENCE,
        initial_state: State | None = None,
        telemetry_view: TelemetryView | None = None,
        state_sync_store_http_host: str | None = None,
        initial_model_trust_identities: tuple[str, ...] = (),
    ):
        self.node_id = node_id
        self.session_id = session_id
        # Live node telemetry off the event log (#279). Node-owned so it
        # survives this master's election: a freshly promoted master keeps the
        # cluster's current node_resources instead of starting blind and
        # risking a placement on a management node. None only in tests/standalone
        # construction; the planner falls back to "no telemetry constraints".
        self._telemetry_view = (
            telemetry_view if telemetry_view is not None else TelemetryView()
        )
        # A promoted master seeds its session from the node's prior
        # replicated state (shared/session_carryover.py) so placements
        # survive failover (#273) — previously every new session started
        # empty, the empty snapshot propagated, and every worker shut down
        # its healthy runners (a full serving outage from one master
        # restart). The seed is indexed as the FIRST EVENT of the new
        # session in run() (see _index_seed_event) rather than assigned
        # here: a pre-seeded snapshot at idx -1 is indistinguishable from
        # "fresh empty state" to the event router, which deliberately skips
        # hydration for idx < 0 — making the seed an ordinary logged event
        # gives every consumer exactly one delivery path. A genuinely fresh
        # node (cold start, or a rebooted node winning election before ever
        # hydrating) passes None and starts empty exactly as before — a
        # stale-boot winner cannot resurrect a cluster view it does not
        # have.
        initial_trust_identities = (
            initial_state.model_trust_approved_remote_code_identities
            if initial_state is not None
            else initial_model_trust_identities
        )
        # Commands are processed serially, but their indexed echoes return on a
        # separate task. Keep the master's authoritative decision set here so
        # back-to-back mutations cannot each read the same stale State snapshot
        # and accidentally resurrect a revocation or discard an approval.
        self._model_trust_approvals = set(initial_trust_identities)
        # Custom-card events are pass-through State events, so authoritative
        # ownership races need a master-local ordered view. Seed aliases lazily
        # from this node's converged card cache, then update the view before the
        # indexed event round-trips. Indexed echoes deliberately never rewrite
        # this view: an older echo may arrive after a newer command decision.
        # A promoted master starts a new view and lazily seeds each alias from
        # its node's already converged card cache before the first new command.
        self._ordered_model_cards: dict[ModelId, ModelCard | None] = {}
        # Local placement effects reach indexed State on another task. Reserve
        # their GPU capacity before queuing the event so consecutive decisions
        # cannot spend the same memory while that echo is outstanding.
        self._pending_instance_reservations: dict[InstanceId, Instance] = {}
        self._ordered_steward_proposals = dict(
            initial_state.steward_action_proposals if initial_state is not None else {}
        )
        # Restart approval and teardown are separate indexed steps. This
        # process-local marker prevents duplicate teardown while allowing a
        # promoted master to reissue it once when approval survived but the
        # original master's deletion did not reach replicated State.
        self._steward_restart_teardown_issued: set[StewardActionProposalId] = set()
        # A dispatched proposal is indexed before its action transitions. Keep
        # a process-local marker so the current master waits for that echo,
        # while a promoted master can reissue the exact command once.
        self._steward_dispatched_effect_issued: set[StewardActionProposalId] = set()
        self._steward_reserved_placements: dict[
            StewardActionProposalId, dict[InstanceId, Instance]
        ] = {}
        self.state = State(
            tracing_enabled=SKULK_TRACING_ENABLED,
            model_trust_approved_remote_code_identities=tuple(
                sorted(self._model_trust_approvals)
            ),
        )
        # A cold-start trust baseline needs the same indexed delivery path as
        # failover state. Leaving non-empty approvals only in this idx=-1 State
        # makes state-sync followers treat it as a fresh empty snapshot; their
        # first unrelated event then replaces the config fallback with an empty
        # replicated set. Index the baseline as event 0 before serving sync.
        self._seed_state = (
            initial_state
            if initial_state is not None
            else self.state
            if self._model_trust_approvals
            else None
        )
        self._started_monotonic = time.monotonic()
        self._tg: TaskGroup = TaskGroup()
        self.command_task_mapping: dict[CommandId, TaskId] = {}
        self._realtime_instance_by_command: dict[CommandId, InstanceId] = {}
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        self.state_sync_receiver = state_sync_receiver
        self.state_sync_sender = state_sync_sender
        self.download_command_sender = download_command_sender
        self.event_sender = event_sender
        self._state_sync_store_http_host = state_sync_store_http_host
        self._system_id = SystemId()
        self._multi_buffer = MultiSourceBuffer[SystemId, Event]()
        self._event_log = DiskEventLog(SKULK_EVENT_LOG_DIR / "master")
        self._snapshot_store = StateSnapshotStore(
            SKULK_EVENT_LOG_DIR / "master" / "snapshots"
        )
        self._snapshot_event_cadence = snapshot_event_cadence
        self._last_snapshot_idx = -1
        self._pending_replay_start_idx: int | None = None
        self._replay_worker_running = False
        self._active_replay_next_idx: int | None = None
        self._active_replay_end_idx: int | None = None
        self._event_log_growth_monitor = EventLogGrowthMonitor()
        self._non_control_event_warning_times: dict[
            tuple[SystemId, type[Event]], float
        ] = {}
        # Nodes with an active dedicated-heartbeat gap warning. Tracking the
        # transition makes a 10-second planning loop emit one warning and one
        # recovery message instead of repeating the same warning indefinitely.
        self._heartbeat_gap_warned_nodes: set[NodeId] = set()
        # First-observation stamps for lifecycle tasks orphaned by a deleted
        # instance (#647); master-local, feeds stale_lifecycle_task_failures.
        self._orphaned_lifecycle_first_seen: dict[TaskId, float] = {}
        # Instance ids whose memory-refusal re-placement has already been
        # initiated (#290). The command processor generates events but does not
        # apply them — self.state only updates when they round-trip through
        # _event_processor — so two ranks of the same instance refusing in the
        # same window would both still see the instance present and each spawn a
        # wider replacement. Deduping on the refused id makes re-placement
        # happen at most once per instance regardless of how many ranks refuse
        # or of redelivery. Grows only by refused ids (rare); never reused since
        # InstanceIds are unique.
        self._refusal_replaced: set[InstanceId] = set()
        # Instances minted by the anywhere-but-the-refuser FALLBACK (second
        # recovery hop). A refusal against one of these is TERMINAL: without
        # this, exclusions are lost across recovery cycles (the replacement
        # command is rebuilt from shard assignments alone) and a tight fleet
        # oscillates -- A refuses full width, fallback lands on B/C, B refuses,
        # the wider re-place returns to A at the share it already refused.
        # Two hops per original placement bounds recovery; ids are unique and
        # rare, matching _refusal_replaced's growth rationale.
        self._fallback_placed_instances: set[InstanceId] = set()
        # Instance ids whose download-failure recovery has already been initiated
        # (#381), same dedup rationale as _refusal_replaced: the plan pass emits
        # events but does not apply them, so the wedged instance stays visible
        # for several ticks until InstanceDeleted round-trips. Deduping on the
        # id makes recovery fire once per wedged instance. Grows only by wedged
        # ids (rare); never reused since InstanceIds are unique.
        self._download_failure_recovered: set[InstanceId] = set()
        # Steward invariant pacing: at most one placement attempt per minute,
        # so an unplaceable steward (no eligible node yet) logs and retries
        # calmly instead of hammering the planner every 10s tick.
        self._steward_last_attempt_monotonic: float = 0.0
        self._steward_upgrade_model: ModelId | None = None
        self._steward_upgrade_stable_since: float | None = None
        self._steward_upgrade_prestaged_model: ModelId | None = None
        self._steward_upgrade_idle_since: float | None = None
        self._steward_upgrade_replacing_instance: InstanceId | None = None
        self._steward_upgrade_retry_after: float = 0.0
        # Per-node memory (bytes) freed by a just-deleted instance. The grace
        # window is zero by default, so entries are normally pruned without being
        # applied; keeping the structure preserves one place to revisit this if
        # we later have a shutdown-complete signal that proves memory recovered.
        self._recently_freed_bytes: dict[NodeId, list[tuple[int, float]]] = {}

    def _effective_downloads(self) -> dict[NodeId, list[DownloadProgress]]:
        """Return durable outcomes overlaid with live download telemetry."""

        return self._telemetry_view.effective_downloads(self.state.downloads)

    def _should_warn_for_non_control_event(
        self,
        origin: SystemId,
        event_type: type[Event],
        *,
        now: float,
    ) -> bool:
        """Rate-limit rejected-event warnings without weakening fail-closed ingest."""

        key = (origin, event_type)
        last_warning_at = self._non_control_event_warning_times.get(key)
        if (
            last_warning_at is not None
            and now - last_warning_at < NON_CONTROL_EVENT_WARNING_COOLDOWN_SECONDS
        ):
            return False

        if (
            key not in self._non_control_event_warning_times
            and len(self._non_control_event_warning_times)
            >= NON_CONTROL_EVENT_WARNING_KEY_LIMIT
        ):
            oldest_key = min(
                self._non_control_event_warning_times,
                key=self._non_control_event_warning_times.__getitem__,
            )
            self._non_control_event_warning_times.pop(oldest_key)
        self._non_control_event_warning_times[key] = now
        return True

    def _apply_indexed_event(self, indexed: IndexedEvent) -> None:
        """Apply one durable event and synchronize the master's telemetry view."""

        self.state = apply(self.state, indexed)
        if isinstance(indexed.event, InstanceCreated):
            self._pending_instance_reservations.pop(indexed.event.instance.instance_id, None)
        elif isinstance(indexed.event, InstanceDeleted):
            self._pending_instance_reservations.pop(indexed.event.instance_id, None)
        record_membership_from_event(self._telemetry_view, indexed.event)

    def _ordered_model_card(self, model_id: ModelId) -> ModelCard | None:
        """Return model-card truth at the master's current command order."""
        if model_id not in self._ordered_model_cards:
            self._ordered_model_cards[model_id] = get_card(model_id)
        registry_card = get_current_registry_card(model_id)
        ordered = self._ordered_model_cards[model_id]
        if registry_card is not None and (
            ordered is None
            or ordered.qualification_only
            or ordered.registry_card_id is not None
        ):
            # Registry refreshes do not traverse the command/event stream.  Pull
            # newer signed truth into the service-ownership view without
            # replacing an operator-owned custom override.  Signed truth always
            # supersedes a lifecycle-owned temporary card for future mutations.
            self._ordered_model_cards[model_id] = registry_card
        return self._ordered_model_cards[model_id]

    def _ordered_placement_model_card(self, model_id: ModelId) -> ModelCard | None:
        """Return authorized card truth for a placement at command order.

        Registry publication can advance from generation A to B while complete
        installed bytes deliberately keep A active until B has been staged.
        ``get_card`` exposes that effective installed generation. Preserve it
        for placement revalidation without letting an out-of-band custom card
        bypass the master's ordered add/delete ownership boundary.

        Args:
            model_id: Alias whose placement card is being revalidated.

        Returns:
            The command-ordered custom card, active signed installed card, or
            current authorized catalog card; ``None`` after an ordered removal.
        """
        ordered_card = self._ordered_model_card(model_id)
        if (
            ordered_card is None
            or ordered_card.is_custom
            or ordered_card.qualification_only
        ):
            return ordered_card

        effective_card = get_card(model_id)
        if (
            effective_card is not None
            and not effective_card.is_custom
            and effective_card.registry_card_id is not None
        ):
            return effective_card
        return ordered_card

    def _order_custom_model_card_add(
        self, command: AddCustomModelCard
    ) -> CustomModelCardAdded | None:
        """Enforce service ownership before ordering a custom-card addition."""
        model_id = command.model_card.model_id
        storage_collision = get_custom_card_storage_collision(model_id)
        if storage_collision is None:
            storage_collision = next(
                (
                    ordered
                    for ordered_model_id, ordered in self._ordered_model_cards.items()
                    if ordered is not None
                    and ordered.is_custom
                    and ordered_model_id != model_id
                    and ordered_model_id.normalize() == model_id.normalize()
                ),
                None,
            )
        if storage_collision is not None:
            logger.warning(
                "Rejected custom model card whose persistence key belongs to "
                f"another alias (model_id={model_id}, "
                f"owner={storage_collision.model_id})"
            )
            return None
        existing = self._ordered_model_card(model_id)
        if (
            command.requires_qualification_ownership
            and existing is not None
            and not existing.qualification_only
        ):
            logger.warning(
                "Rejected qualification card addition at authoritative ordering "
                f"boundary (model_id={model_id})"
            )
            return None
        self._ordered_model_cards[model_id] = command.model_card
        return CustomModelCardAdded(
            model_card=command.model_card,
            mutation_command_id=command.command_id,
        )

    def _order_custom_model_card_delete(
        self, command: DeleteCustomModelCard
    ) -> CustomModelCardDeleted | None:
        """Enforce service ownership before ordering a custom-card deletion."""
        existing = self._ordered_model_card(command.model_id)
        if command.requires_qualification_ownership and (
            existing is None
            or not existing.qualification_only
            or command.expected_qualification_card is None
            or existing != command.expected_qualification_card
        ):
            logger.warning(
                "Rejected qualification card deletion at authoritative ordering "
                f"boundary (model_id={command.model_id})"
            )
            return None
        self._ordered_model_cards[command.model_id] = None
        return CustomModelCardDeleted(
            model_id=command.model_id,
            mutation_command_id=command.command_id,
        )

    def _require_ordered_place_instance_card(self, command: PlaceInstance) -> None:
        """Require a quick-placement command to match master-ordered card truth.

        Args:
            command: Placement command carrying the API node's selected card.

        Raises:
            PlacementModelCardIdentityError: If the card was removed or replaced
                before the command reached the master's serialized order.
        """
        ordered_card = self._ordered_placement_model_card(command.model_card.model_id)
        if ordered_card is None or not same_authorized_model_card(
            command.model_card, ordered_card
        ):
            raise PlacementModelCardIdentityError(
                "Placement model-card identity no longer matches the authorized "
                f"catalog card for {command.model_card.model_id}. Refresh model "
                "truth and retry."
            )

    def _require_ordered_create_instance_card(self, command: CreateInstance) -> None:
        """Require an exact placement to match master-ordered card truth.

        Args:
            command: Exact placement command containing embedded shard cards.

        Raises:
            PlacementModelCardIdentityError: If the card was removed or replaced
                before the command reached the master's serialized order.
        """
        model_id = command.instance.shard_assignments.model_id
        ordered_card = self._ordered_placement_model_card(model_id)
        if ordered_card is None:
            raise PlacementModelCardIdentityError(
                "Exact placement model-card identity is no longer present in "
                f"the authorized catalog for {model_id}. Refresh model truth "
                "and retry."
            )
        require_instance_model_card_identity(command.instance, ordered_card)

    def _record_freed_instance(self, instance: Instance) -> None:
        """Record a deleted instance's per-node footprint for the grace window.

        The default grace is zero, which makes this a no-credit bookkeeping path.
        A future nonzero grace should be tied to a signal that the local worker
        has actually released the model memory, not merely to InstanceDeleted.
        """
        assignments = instance.shard_assignments
        deadline = time.monotonic() + RECENTLY_FREED_MEMORY_GRACE_SECONDS
        for node_id, runner_id in assignments.node_to_runner.items():
            shard = assignments.runner_to_shard.get(runner_id)
            if shard is None:
                continue
            fraction = shard_fraction_of_model(shard)
            if fraction is None or fraction <= 0.0:
                continue
            footprint = estimate_shard_footprint(shard.model_card, fraction)
            self._recently_freed_bytes.setdefault(node_id, []).append(
                (footprint.in_bytes, deadline)
            )

    def _freed_credit_by_node(self) -> dict[NodeId, int]:
        """Current non-expired freed-memory credit (bytes) per node, pruning the
        expired entries as a side effect (#314)."""
        now = time.monotonic()
        credit: dict[NodeId, int] = {}
        for node_id in list(self._recently_freed_bytes):
            live = [(b, d) for (b, d) in self._recently_freed_bytes[node_id] if d > now]
            if live:
                self._recently_freed_bytes[node_id] = live
                credit[node_id] = sum(b for (b, _) in live)
            else:
                del self._recently_freed_bytes[node_id]
        return credit

    async def _queue_control_event(self, event: Event) -> None:
        """Reserve a new placement before its asynchronous indexed echo returns."""
        if isinstance(event, InstanceCreated):
            self._pending_instance_reservations[event.instance.instance_id] = event.instance
        await self.event_sender.send(event)

    def _placement_reservations(
        self, instances: Mapping[InstanceId, Instance]
    ) -> dict[InstanceId, Instance]:
        """Overlay local committed creations on a real or hypothetical placement set."""
        reservations = dict(self._pending_instance_reservations)
        for reservation in self._steward_reserved_placements.values():
            reservations.update(reservation)
        reservations.update(instances)
        return reservations

    def _placement_memory_inputs(
        self,
        current_instances: Mapping[InstanceId, Instance] | None = None,
    ) -> tuple[
        Mapping[NodeId, MemoryUsage],
        Mapping[NodeId, Memory],
    ]:
        """Build the node-memory and usable-GPU placement inputs.

        Expired recently-freed entries are pruned before building the inputs.
        With the default zero grace, no speculative credit is applied: live MLX
        teardown can lag InstanceDeleted, and using observed telemetry produces
        placements whose concrete shard allocations match the worker's local
        pre-load guard.

        If an operator later enables a nonzero grace, both ``ram_available`` and
        the derived usable-GPU map are built from the credited memory snapshot;
        ``ram_total`` is never credited, so context-ceiling math stays anchored
        to physical capacity.
        """
        placements = self._placement_reservations(
            self.state.instances if current_instances is None else current_instances
        )
        credit = self._freed_credit_by_node()
        base_memory = self._telemetry_view.node_memory
        if not credit:
            base_vram = usable_vram_by_node(
                self._telemetry_view.node_system,
                self._telemetry_view.node_resources,
                node_memory=base_memory,
                current_instances=placements,
            )
            return base_memory, base_vram
        # Credit the freed bytes onto each node's ram_available, clamped to
        # ram_total so credited availability never exceeds capacity (telemetry
        # may already have partly caught up, or the footprint estimate may be
        # conservative).
        memory = {
            node_id: (
                usage.model_copy(
                    update={
                        "ram_available": Memory.from_bytes(
                            min(
                                usage.ram_total.in_bytes,
                                usage.ram_available.in_bytes + credit[node_id],
                            )
                        )
                    }
                )
                if node_id in credit
                else usage
            )
            for node_id, usage in base_memory.items()
        }
        # Derive VRAM from the credited memory rather than crediting the VRAM
        # figure directly: usable_vram_by_node applies its own working-set /
        # GTT ceiling, so the credited VRAM is naturally capped and can never
        # exceed the ceiling or total VRAM.
        vram = usable_vram_by_node(
            self._telemetry_view.node_system,
            self._telemetry_view.node_resources,
            node_memory=memory,
            current_instances=placements,
        )
        return memory, vram

    def _place_for_steward_action(
        self,
        command: PlaceInstance,
        current_instances: Mapping[InstanceId, Instance],
    ) -> dict[InstanceId, Instance]:
        """Compute one approved steward placement using authoritative inputs."""
        self._require_ordered_place_instance_card(command)
        credited_memory, credited_vram = self._placement_memory_inputs(
            current_instances
        )
        return place_instance(
            command,
            self.state.topology,
            current_instances,
            credited_memory,
            self.state.node_network,
            download_status=self._effective_downloads(),
            excluded_nodes=set(command.excluded_nodes),
            node_resources=self._telemetry_view.node_resources,
            node_vram=credited_vram,
            unified_memory_gpu_nodes=unified_memory_gpu_node_ids(
                self._telemetry_view.node_system,
                self._telemetry_view.node_resources,
                node_memory=credited_memory,
            ),
            approved_remote_code_identities=self._model_trust_approvals,
        )

    async def _execute_approved_steward_action(
        self, proposal: StewardActionProposal
    ) -> tuple[list[Event], CommandId, Literal["approved", "dispatched"]]:
        """Execute one approved basic action through existing typed machinery.

        This method owns no free-form effects. It translates the proposal's
        validated action union into the same placement, deletion, and download
        commands used by ordinary operator endpoints.
        """
        action = proposal.action
        if isinstance(action, StewardPlaceModelAction):
            command = PlaceInstance(
                model_card=action.model_card,
                sharding=action.sharding,
                instance_meta=action.instance_meta,
                min_nodes=action.min_nodes,
                excluded_nodes=list(action.excluded_nodes),
            )
            ordered_instances = dict(self.state.instances)
            for reservation in self._steward_reserved_placements.values():
                ordered_instances.update(reservation)
            placement = self._place_for_steward_action(command, ordered_instances)
            self._steward_reserved_placements[proposal.proposal_id] = {
                instance_id: instance
                for instance_id, instance in placement.items()
                if instance_id not in ordered_instances
            }
            self._steward_dispatched_effect_issued.add(proposal.proposal_id)
            return (
                list(
                    get_transition_events(
                        ordered_instances, placement, self.state.tasks
                    )
                ),
                command.command_id,
                "dispatched",
            )

        if isinstance(action, StewardCancelDownloadAction):
            active_download = any(
                isinstance(progress, (DownloadPending, DownloadOngoing))
                and progress.shard_metadata.model_card.model_id == action.model_id
                and progress.attempt_id == action.attempt_id
                for progress in self._effective_downloads().get(action.node_id, ())
            )
            if not active_download:
                raise ValueError("The proposed download is no longer active")
            command = CancelDownload(
                target_node_id=action.node_id,
                model_id=action.model_id,
                attempt_id=action.attempt_id,
            )
            return [], command.command_id, "approved"

        instance_id = action.instance.instance_id
        model_id = action.instance.shard_assignments.model_id
        instance = self.state.instances.get(instance_id)
        if instance is None:
            raise ValueError("The proposed instance no longer exists")
        if instance.system_role is not None:
            raise ValueError("System placements cannot be changed by steward actions")
        if instance.shard_assignments.model_id != model_id:
            raise ValueError("The proposed instance now serves different model truth")
        if instance != action.instance:
            raise ValueError("The proposed instance intent no longer matches current state")
        if any(
            other.proposal_id != proposal.proposal_id
            and other.status in {"approved", "dispatched"}
            and isinstance(
                other.action,
                (StewardStopInstanceAction, StewardRestartInstanceAction),
            )
            and other.action.instance.instance_id == instance_id
            for other in self._ordered_steward_proposals.values()
        ):
            raise ValueError("Another steward action already owns this instance")

        delete_command = DeleteInstance(instance_id=instance_id)
        if isinstance(action, StewardStopInstanceAction):
            return [], delete_command.command_id, "dispatched"

        assert isinstance(action, StewardRestartInstanceAction)
        replacement_command = replacement_command_for_download_failed_instance(
            action.instance, frozenset()
        )
        self._require_ordered_place_instance_card(replacement_command)
        return [], delete_command.command_id, "approved"

    def _prune_ordered_steward_action_proposals(self) -> None:
        """Bound local proposals without dropping actionable recovery work."""
        active_restart_ids = {
            proposal.proposal_id
            for proposal in self._ordered_steward_proposals.values()
            if proposal.status in {"approved", "dispatched"}
            and isinstance(proposal.action, StewardRestartInstanceAction)
        }
        self._steward_restart_teardown_issued.intersection_update(
            active_restart_ids
        )
        dispatched_ids = {
            proposal.proposal_id
            for proposal in self._ordered_steward_proposals.values()
            if proposal.status == "dispatched"
        }
        self._steward_dispatched_effect_issued.intersection_update(dispatched_ids)
        self._steward_reserved_placements = {
            proposal_id: reservation
            for proposal_id, reservation in self._steward_reserved_placements.items()
            if proposal_id in dispatched_ids
        }
        excess = len(self._ordered_steward_proposals) - 128
        if excess <= 0:
            return
        terminal = sorted(
            (
                proposal
                for proposal in self._ordered_steward_proposals.values()
                if steward_action_proposal_is_prunable(
                    proposal, datetime.now(tz=timezone.utc)
                )
            ),
            key=lambda proposal: proposal.created_at,
        )
        for proposal in terminal[:excess]:
            self._ordered_steward_proposals.pop(proposal.proposal_id, None)

    def _expire_steward_action_proposals(
        self, now: datetime
    ) -> list[StewardActionProposalChanged]:
        """Order terminal expiry for pending proposals whose deadline passed."""
        changes: list[StewardActionProposalChanged] = []
        for proposal_id, proposal in tuple(self._ordered_steward_proposals.items()):
            if proposal.status != "pending" or proposal.expires_at > now:
                continue
            expired = proposal.model_copy(
                update={
                    "status": "expired",
                    "decided_at": now,
                    "decided_by": "fabric_expiry",
                    "outcome": "The proposal expired without an operator decision.",
                }
            )
            self._ordered_steward_proposals[proposal_id] = expired
            changes.append(StewardActionProposalChanged(proposal=expired))
        self._prune_ordered_steward_action_proposals()
        return changes

    async def _resume_approved_steward_restarts(self, now: datetime) -> None:
        """Place approved restarts only after teardown and capacity converge."""
        for proposal_id, proposal in tuple(self._ordered_steward_proposals.items()):
            action = proposal.action
            if proposal.status != "approved" or not isinstance(
                action, StewardRestartInstanceAction
            ):
                continue
            replicated_proposal = self.state.steward_action_proposals.get(proposal_id)
            if (
                replicated_proposal is None
                or replicated_proposal.status != "approved"
            ):
                continue
            decided_at = proposal.decided_at
            if decided_at is None:
                failed = proposal.model_copy(
                    update={
                        "status": "failed",
                        "outcome": "Approved restart is missing its decision time.",
                    }
                )
            elif os.getenv("SKULK_FABRIC_CAPABILITIES_DISABLE") == "1":
                failed = proposal.model_copy(
                    update={
                        "status": "failed",
                        "outcome": "Fabric actions are disabled by the global kill switch.",
                    }
                )
            elif now > decided_at + timedelta(minutes=5):
                failed = proposal.model_copy(
                    update={
                        "status": "failed",
                        "outcome": (
                            "Restart capacity did not become available within five minutes."
                        ),
                    }
                )
            elif action.instance.instance_id in self.state.instances:
                current_instance = self.state.instances[action.instance.instance_id]
                if current_instance.system_role is not None:
                    failed = proposal.model_copy(
                        update={
                            "status": "failed",
                            "outcome": (
                                "System placements cannot be changed by steward actions."
                            ),
                        }
                    )
                elif current_instance != action.instance:
                    failed = proposal.model_copy(
                        update={
                            "status": "failed",
                            "outcome": (
                                "The approved restart intent no longer matches current state."
                            ),
                        }
                    )
                elif proposal_id in self._steward_restart_teardown_issued:
                    continue
                else:
                    replacement_command = (
                        replacement_command_for_download_failed_instance(
                            action.instance, frozenset()
                        )
                    )
                    try:
                        self._require_ordered_place_instance_card(replacement_command)
                    except PlacementModelCardIdentityError as error:
                        failed = proposal.model_copy(
                            update={
                                "status": "failed",
                                "outcome": str(error)[:1024],
                            }
                        )
                        self._ordered_steward_proposals[proposal_id] = failed
                        await self._queue_control_event(
                            StewardActionProposalChanged(proposal=failed)
                        )
                        continue
                    # A promoted master can inherit the durable approval before
                    # inheriting its predecessor's deletion. Reissue the exact
                    # teardown once; its instance identity makes this safe and
                    # preserves forward progress across that failover window.
                    self._record_freed_instance(current_instance)
                    delete_command = DeleteInstance(
                        instance_id=current_instance.instance_id
                    )
                    after_delete = delete_instance(
                        delete_command, self.state.instances
                    )
                    self._steward_restart_teardown_issued.add(proposal_id)
                    for cancel_command in cancel_unnecessary_downloads(
                        after_delete, self._effective_downloads()
                    ):
                        await self.download_command_sender.send(
                            ForwarderDownloadCommand(
                                origin=self._system_id, command=cancel_command
                            )
                        )
                    for event in get_transition_events(
                        self.state.instances, after_delete, self.state.tasks
                    ):
                        await self._queue_control_event(event)
                    continue
            else:
                replace_command = replacement_command_for_download_failed_instance(
                    action.instance, frozenset()
                )
                try:
                    self._require_ordered_place_instance_card(replace_command)
                    ordered_instances = dict(self.state.instances)
                    for reservation in self._steward_reserved_placements.values():
                        ordered_instances.update(reservation)
                    replacement = self._place_for_steward_action(
                        replace_command, ordered_instances
                    )
                    self._steward_reserved_placements[proposal_id] = {
                        instance_id: instance
                        for instance_id, instance in replacement.items()
                        if instance_id not in ordered_instances
                    }
                except PlacementModelCardIdentityError as error:
                    failed = proposal.model_copy(
                        update={
                            "status": "failed",
                            "outcome": str(error)[:1024],
                        }
                    )
                    self._ordered_steward_proposals[proposal_id] = failed
                    await self._queue_control_event(
                        StewardActionProposalChanged(proposal=failed)
                    )
                    continue
                except PlacementError:
                    # Teardown and memory telemetry converge independently. A
                    # normal large model is temporarily unplaceable until the
                    # old runner releases its allocation.
                    continue
                except ValueError as error:
                    failed = proposal.model_copy(
                        update={
                            "status": "failed",
                            "outcome": str(error)[:1024],
                        }
                    )
                    self._ordered_steward_proposals[proposal_id] = failed
                    await self._queue_control_event(
                        StewardActionProposalChanged(proposal=failed)
                    )
                    continue
                for cancel_command in cancel_unnecessary_downloads(
                    replacement, self._effective_downloads()
                ):
                    await self.download_command_sender.send(
                        ForwarderDownloadCommand(
                            origin=self._system_id, command=cancel_command
                        )
                    )
                dispatched = proposal.model_copy(
                    update={
                        "status": "dispatched",
                        "dispatched_at": now,
                        "command_id": replace_command.command_id,
                        "outcome": (
                            "Restart replacement was dispatched after teardown "
                            "capacity became available."
                        ),
                    }
                )
                self._ordered_steward_proposals[proposal_id] = dispatched
                self._steward_restart_teardown_issued.discard(proposal_id)
                self._steward_dispatched_effect_issued.add(proposal_id)
                await self._queue_control_event(
                    StewardActionProposalChanged(proposal=dispatched)
                )
                for event in get_transition_events(
                    ordered_instances, replacement, self.state.tasks
                ):
                    await self._queue_control_event(event)
                continue
            self._ordered_steward_proposals[proposal_id] = failed
            self._steward_restart_teardown_issued.discard(proposal_id)
            await self._queue_control_event(
                StewardActionProposalChanged(proposal=failed)
            )
        self._prune_ordered_steward_action_proposals()

    async def _reconcile_dispatched_steward_actions(self, now: datetime) -> None:
        """Reissue an unreflected dispatched action once after master failover."""
        for proposal_id, reservation in tuple(
            self._steward_reserved_placements.items()
        ):
            if all(instance_id in self.state.instances for instance_id in reservation):
                self._steward_reserved_placements.pop(proposal_id, None)
        for proposal_id, proposal in tuple(self._ordered_steward_proposals.items()):
            if (
                proposal.status != "dispatched"
                or proposal.command_id is None
                or (
                    (dispatch_started_at := proposal.dispatched_at or proposal.decided_at)
                    is None
                )
                or now > dispatch_started_at + timedelta(minutes=5)
                or proposal_id in self._steward_dispatched_effect_issued
            ):
                continue
            replicated_proposal = self.state.steward_action_proposals.get(proposal_id)
            if (
                replicated_proposal is None
                or replicated_proposal.status != "dispatched"
            ):
                continue
            if os.getenv("SKULK_FABRIC_CAPABILITIES_DISABLE") == "1":
                failed = proposal.model_copy(
                    update={
                        "status": "failed",
                        "outcome": (
                            "Fabric action recovery was blocked by the global "
                            "kill switch."
                        ),
                    }
                )
                self._ordered_steward_proposals[proposal_id] = failed
                self._steward_reserved_placements.pop(proposal_id, None)
                self._steward_restart_teardown_issued.discard(proposal_id)
                self._steward_dispatched_effect_issued.add(proposal_id)
                await self._queue_control_event(
                    StewardActionProposalChanged(proposal=failed)
                )
                continue
            action = proposal.action
            events: list[Event] = []
            try:
                if isinstance(action, StewardPlaceModelAction):
                    expected_instance_id = InstanceId(str(proposal.command_id))
                    if expected_instance_id in self.state.instances:
                        continue
                    command = PlaceInstance(
                        command_id=proposal.command_id,
                        model_card=action.model_card,
                        sharding=action.sharding,
                        instance_meta=action.instance_meta,
                        min_nodes=action.min_nodes,
                        excluded_nodes=list(action.excluded_nodes),
                    )
                    ordered_instances = dict(self.state.instances)
                    for reservation in self._steward_reserved_placements.values():
                        ordered_instances.update(reservation)
                    placement = self._place_for_steward_action(command, ordered_instances)
                    self._steward_reserved_placements[proposal_id] = {
                        instance_id: instance
                        for instance_id, instance in placement.items()
                        if instance_id not in ordered_instances
                    }
                    events = list(
                        get_transition_events(
                            ordered_instances, placement, self.state.tasks
                        )
                    )
                elif isinstance(action, StewardCancelDownloadAction):
                    node_downloads = self._effective_downloads().get(
                        action.node_id, ()
                    )
                    active_download = any(
                        isinstance(progress, (DownloadPending, DownloadOngoing))
                        and progress.shard_metadata.model_card.model_id
                        == action.model_id
                        and progress.attempt_id == action.attempt_id
                        for progress in node_downloads
                    )
                    if not active_download:
                        if any(
                            isinstance(progress, (DownloadPending, DownloadOngoing))
                            and progress.shard_metadata.model_card.model_id
                            == action.model_id
                            for progress in node_downloads
                        ):
                            raise ValueError(
                                "The approved download attempt has been replaced"
                            )
                        continue
                    self._steward_dispatched_effect_issued.add(proposal_id)
                    await self.download_command_sender.send(
                        ForwarderDownloadCommand(
                            origin=self._system_id,
                            command=CancelDownload(
                                command_id=proposal.command_id,
                                target_node_id=action.node_id,
                                model_id=action.model_id,
                                attempt_id=action.attempt_id,
                            ),
                        )
                    )
                    continue
                elif isinstance(action, StewardStopInstanceAction):
                    instance = self.state.instances.get(action.instance.instance_id)
                    if instance is None:
                        self._steward_dispatched_effect_issued.add(proposal_id)
                        for cancel_command in cancel_unnecessary_downloads(
                            self.state.instances, self._effective_downloads()
                        ):
                            await self.download_command_sender.send(
                                ForwarderDownloadCommand(
                                    origin=self._system_id, command=cancel_command
                                )
                            )
                        continue
                    if instance.system_role is not None:
                        raise ValueError(
                            "System placements cannot be changed by steward actions"
                        )
                    if instance != action.instance:
                        raise ValueError(
                            "The dispatched stop intent no longer matches current state"
                        )
                    after_delete = delete_instance(
                        DeleteInstance(
                            command_id=proposal.command_id,
                            instance_id=action.instance.instance_id,
                        ),
                        self.state.instances,
                    )
                    self._steward_dispatched_effect_issued.add(proposal_id)
                    for cancel_command in cancel_unnecessary_downloads(
                        after_delete, self._effective_downloads()
                    ):
                        await self.download_command_sender.send(
                            ForwarderDownloadCommand(
                                origin=self._system_id, command=cancel_command
                            )
                        )
                    events = list(
                        get_transition_events(
                            self.state.instances, after_delete, self.state.tasks
                        )
                    )
                else:
                    assert isinstance(action, StewardRestartInstanceAction)
                    expected_instance_id = InstanceId(str(proposal.command_id))
                    if expected_instance_id in self.state.instances:
                        continue
                    original = self.state.instances.get(action.instance.instance_id)
                    if original is not None:
                        if original.system_role is not None or original != action.instance:
                            raise ValueError(
                                "The dispatched restart intent no longer matches current state"
                            )
                        replacement_command = (
                            replacement_command_for_download_failed_instance(
                                action.instance, frozenset()
                            ).model_copy(update={"command_id": proposal.command_id})
                        )
                        self._require_ordered_place_instance_card(replacement_command)
                        if proposal_id in self._steward_restart_teardown_issued:
                            continue
                        after_delete = delete_instance(
                            DeleteInstance(instance_id=original.instance_id),
                            self.state.instances,
                        )
                        self._steward_restart_teardown_issued.add(proposal_id)
                        for cancel_command in cancel_unnecessary_downloads(
                            after_delete, self._effective_downloads()
                        ):
                            await self.download_command_sender.send(
                                ForwarderDownloadCommand(
                                    origin=self._system_id, command=cancel_command
                                )
                            )
                        for event in get_transition_events(
                            self.state.instances, after_delete, self.state.tasks
                        ):
                            await self._queue_control_event(event)
                        continue
                    replacement_command = (
                        replacement_command_for_download_failed_instance(
                            action.instance, frozenset()
                        ).model_copy(update={"command_id": proposal.command_id})
                    )
                    ordered_instances = dict(self.state.instances)
                    for reservation in self._steward_reserved_placements.values():
                        ordered_instances.update(reservation)
                    replacement = self._place_for_steward_action(
                        replacement_command, ordered_instances
                    )
                    self._steward_reserved_placements[proposal_id] = {
                        instance_id: instance
                        for instance_id, instance in replacement.items()
                        if instance_id not in ordered_instances
                    }
                    self._steward_dispatched_effect_issued.add(proposal_id)
                    for cancel_command in cancel_unnecessary_downloads(
                        replacement, self._effective_downloads()
                    ):
                        await self.download_command_sender.send(
                            ForwarderDownloadCommand(
                                origin=self._system_id, command=cancel_command
                            )
                        )
                    events = list(
                        get_transition_events(
                            ordered_instances, replacement, self.state.tasks
                        )
                    )
            except PlacementModelCardIdentityError as error:
                failed = proposal.model_copy(
                    update={"status": "failed", "outcome": str(error)[:1024]}
                )
                self._ordered_steward_proposals[proposal_id] = failed
                await self._queue_control_event(
                    StewardActionProposalChanged(proposal=failed)
                )
                continue
            except PlacementError:
                # Capacity telemetry can lag the state seed after promotion.
                # Retry on a later planning tick within the bounded window.
                continue
            except ValueError as error:
                failed = proposal.model_copy(
                    update={"status": "failed", "outcome": str(error)[:1024]}
                )
                self._ordered_steward_proposals[proposal_id] = failed
                await self._queue_control_event(
                    StewardActionProposalChanged(proposal=failed)
                )
                continue
            self._steward_dispatched_effect_issued.add(proposal_id)
            for event in events:
                await self._queue_control_event(event)
        self._prune_ordered_steward_action_proposals()

    async def _arm_approved_steward_download_cancellations(self) -> None:
        """Persist cancel dispatch intent before forwarding its side effect."""
        for proposal_id, proposal in tuple(self._ordered_steward_proposals.items()):
            if proposal.status != "approved" or not isinstance(
                proposal.action, StewardCancelDownloadAction
            ):
                continue
            replicated_proposal = self.state.steward_action_proposals.get(proposal_id)
            if (
                replicated_proposal is None
                or replicated_proposal.status != "approved"
            ):
                # The local proposal map is updated before its event is indexed.
                # Only arm the cancellation after approval is recoverable from
                # replicated state, including across master promotion.
                continue
            if os.getenv("SKULK_FABRIC_CAPABILITIES_DISABLE") == "1":
                failed = proposal.model_copy(
                    update={
                        "status": "failed",
                        "outcome": (
                            "Fabric actions are disabled by the global kill switch."
                        ),
                    }
                )
                self._ordered_steward_proposals[proposal_id] = failed
                await self._queue_control_event(
                    StewardActionProposalChanged(proposal=failed)
                )
                continue
            dispatched = proposal.model_copy(
                update={
                    "status": "dispatched",
                    "dispatched_at": datetime.now(tz=timezone.utc),
                    "outcome": (
                        "Approved cancellation was durably armed for exact-attempt "
                        "dispatch."
                    ),
                }
            )
            self._ordered_steward_proposals[proposal_id] = dispatched
            await self._queue_control_event(
                StewardActionProposalChanged(proposal=dispatched)
            )
        self._prune_ordered_steward_action_proposals()

    async def _index_seed_event(self) -> None:
        """Index failover or cold-start trust seed as this session's first event.

        Making the carried state an ordinary logged ``StateSnapshotHydrated``
        event gives every consumer exactly one delivery path: followers that
        snapshot-bootstrap after this point receive it inside the snapshot;
        followers that bootstrapped against the momentarily-empty state (the
        promotion race — including this node's own worker) receive it as the
        live event at index 0 and apply it like any other event. A seeded
        snapshot at idx ``-1`` instead looked identical to "fresh empty
        state", which the event router deliberately skips hydrating — the
        first live deployment of the seed lost it to exactly that race on
        the promoted node while a later-bootstrapping follower kept it.
        """
        if self._seed_state is None:
            return
        idx = len(self._event_log)
        seed = self._seed_state.model_copy(update={"last_event_applied_idx": idx})
        # Release the pre-index reference so the seed's object graph can be
        # collected as state evolves past it.
        self._seed_state = None
        indexed = IndexedEvent(event=StateSnapshotHydrated(state=seed), idx=idx)
        self._apply_indexed_event(indexed)
        self._append_event_log(indexed.event)
        await self._send_event(indexed)
        logger.info(
            f"Indexed startup seed as event {idx}: "
            f"{len(seed.instances)} carried instance(s), "
            f"{len(seed.model_trust_approved_remote_code_identities)} "
            "model trust decision(s)"
        )

    async def run(self):
        logger.info("Starting Master")

        try:
            await self._index_seed_event()
            async with self._tg as tg:
                tg.start_soon(self._event_processor)
                tg.start_soon(self._command_processor)
                tg.start_soon(self._state_sync_processor)
                tg.start_soon(self._plan)
        finally:
            await self._persist_snapshot(force=True)
            self._event_log.close()
            self.global_event_sender.close()
            self.local_event_receiver.close()
            self.command_receiver.close()
            self.state_sync_receiver.close()

    async def shutdown(self):
        logger.info("Stopping Master")
        self._tg.cancel_tasks()

    async def _command_processor(self) -> None:
        with self.command_receiver as commands:
            async for forwarder_command in commands:
                try:
                    logger.info(
                        "Executing command: "
                        f"{summarize_command_for_log(forwarder_command.command)}"
                    )

                    generated_events: list[Event] = []
                    command = forwarder_command.command
                    instance_task_counts: dict[InstanceId, int] = {}
                    match command:
                        case TestCommand():
                            pass
                        case TextGeneration():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            # A pinned request (steward/canary) must fail its
                            # task cleanly even when NO instance serves the
                            # model anymore (the pinned instance vanished
                            # between caller lookup and processing); raising
                            # here would leave the caller hanging with no
                            # terminal event. Mirrors the SpeechSynthesis
                            # guard.
                            if (
                                not instance_task_counts
                                and command.target_instance_id is None
                            ):
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            task_id = TaskId()
                            target_unavailable = False
                            if command.target_instance_id is not None:
                                # Steward/canary path: pin to the requested
                                # instance (mirrors SpeechSynthesis); a miss
                                # fails the task instead of silently landing
                                # on another placement.
                                if (
                                    command.target_instance_id
                                    not in instance_task_counts
                                ):
                                    target_unavailable = True
                                selected_instance_id = command.target_instance_id
                            else:
                                available_instance_ids = sorted(
                                    instance_task_counts.keys(),
                                    key=lambda instance_id: instance_task_counts[
                                        instance_id
                                    ],
                                )
                                selected_instance_id = available_instance_ids[0]
                            trace_enabled = self.state.tracing_enabled
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=TextGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        # Carry the owning API node onto the task
                                        # so the rank-0 supervisor can address
                                        # output over the Zenoh data plane (#279
                                        # Phase 2).
                                        owner_node=command.owner_node,
                                        instance_id=selected_instance_id,
                                        task_status=(
                                            TaskStatus.Failed
                                            if target_unavailable
                                            else TaskStatus.Pending
                                        ),
                                        task_params=command.task_params,
                                        trace_enabled=trace_enabled,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id
                            if target_unavailable:
                                generated_events.append(
                                    TaskFailed(
                                        task_id=task_id,
                                        error_type="instance_unavailable",
                                        error_message=(
                                            "Requested text-generation instance "
                                            "is unavailable or does not serve "
                                            "the requested model"
                                        ),
                                    )
                                )
                        case ImageGeneration():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            trace_enabled = self.state.tracing_enabled
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        owner_node=command.owner_node,  # #279 Phase 2
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                        trace_enabled=trace_enabled,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id
                        case ImageEdits():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            trace_enabled = self.state.tracing_enabled
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageEditsTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        owner_node=command.owner_node,  # #279 Phase 2
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                        trace_enabled=trace_enabled,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id
                        case TextEmbedding():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            trace_enabled = self.state.tracing_enabled
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=TextEmbeddingTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        owner_node=command.owner_node,  # #279 Phase 2
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                        trace_enabled=trace_enabled,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id
                        case SpeechSynthesis():
                            if command.task_params.reference_audio_data is not None:
                                raise ValueError(
                                    "Reference audio bytes must not enter commands or State"
                                )
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if (
                                not instance_task_counts
                                and command.target_instance_id is None
                            ):
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            task_id = TaskId()
                            target_unavailable = False
                            if command.target_instance_id is not None:
                                if (
                                    command.target_instance_id
                                    not in instance_task_counts
                                ):
                                    target_unavailable = True
                                selected_instance_id = command.target_instance_id
                            else:
                                available_instance_ids = sorted(
                                    instance_task_counts.keys(),
                                    key=lambda instance_id: instance_task_counts[
                                        instance_id
                                    ],
                                )
                                selected_instance_id = available_instance_ids[0]
                            trace_enabled = self.state.tracing_enabled
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=SpeechSynthesisTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        owner_node=command.owner_node,  # #279 Phase 2
                                        instance_id=selected_instance_id,
                                        task_status=(
                                            TaskStatus.Failed
                                            if target_unavailable
                                            else TaskStatus.Pending
                                        ),
                                        task_params=command.task_params,
                                        trace_enabled=trace_enabled,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id
                            if target_unavailable:
                                generated_events.append(
                                    TaskFailed(
                                        task_id=task_id,
                                        error_type="instance_unavailable",
                                        error_message=(
                                            "Requested TTS instance is unavailable or "
                                            "does not serve the requested model"
                                        ),
                                    )
                                )
                        case AudioTranscription():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            available_instance_ids = sorted(
                                instance_task_counts.keys(),
                                key=lambda instance_id: instance_task_counts[
                                    instance_id
                                ],
                            )

                            task_id = TaskId()
                            selected_instance_id = available_instance_ids[0]
                            trace_enabled = self.state.tracing_enabled
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=AudioTranscriptionTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        owner_node=command.owner_node,  # #279 Phase 2
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                        trace_enabled=trace_enabled,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id
                        case RealtimeAudioTranscription():
                            instance = self.state.instances.get(
                                command.target_instance_id
                            )
                            if instance is None:
                                raise ValueError(
                                    "No target instance found for realtime STT "
                                    f"command {command.command_id}"
                                )
                            if (
                                instance.shard_assignments.model_id
                                != command.task_params.model
                            ):
                                raise ValueError(
                                    "Realtime STT target instance model does not "
                                    f"match {command.task_params.model}"
                                )
                            if len(instance.shard_assignments.node_to_runner) != 1:
                                raise ValueError(
                                    "Realtime STT requires a single-host target "
                                    f"instance, got {command.target_instance_id}"
                                )
                            instance_busy = command.target_instance_id in (
                                self._realtime_instance_by_command.values()
                            ) or any(
                                # Runner readiness can precede lifecycle-task
                                # convergence. Only transcription inference
                                # consumes mounted STT capacity once ready.
                                isinstance(
                                    task,
                                    (
                                        AudioTranscriptionTask,
                                        RealtimeAudioTranscriptionTask,
                                    ),
                                )
                                and task.instance_id == command.target_instance_id
                                and task.task_status
                                in (TaskStatus.Pending, TaskStatus.Running)
                                for task in self.state.tasks.values()
                            )

                            task_id = TaskId()
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=RealtimeAudioTranscriptionTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        owner_node=command.owner_node,
                                        instance_id=command.target_instance_id,
                                        task_status=(
                                            TaskStatus.Failed
                                            if instance_busy
                                            else TaskStatus.Pending
                                        ),
                                        task_params=command.task_params,
                                        # Realtime STT does not emit trace
                                        # sessions yet. Do not register ranks
                                        # that can never report completion.
                                        trace_enabled=False,
                                    ),
                                )
                            )
                            self.command_task_mapping[command.command_id] = task_id
                            if instance_busy:
                                generated_events.append(
                                    TaskFailed(
                                        task_id=task_id,
                                        error_type="instance_busy",
                                        error_message=(
                                            "Realtime STT target instance is already "
                                            "reserved"
                                        ),
                                    )
                                )
                            else:
                                self._realtime_instance_by_command[
                                    command.command_id
                                ] = command.target_instance_id
                        case SetTracingEnabled():
                            generated_events.append(
                                TracingStateChanged(enabled=command.enabled)
                            )
                        case SetModelTrustApproval():
                            if command.approved:
                                self._model_trust_approvals.add(command.trust_identity)
                            else:
                                self._model_trust_approvals.discard(
                                    command.trust_identity
                                )
                            try:
                                persist_model_trust_config(
                                    resolve_config_path(),
                                    self._model_trust_approvals,
                                )
                            except (OSError, ValueError):
                                # Replicated State is authoritative. A degraded
                                # local disk must not kill the master command
                                # processor; workers gate new runner creation
                                # and loading from that State, while YAML remains
                                # only the durable restart/child-process fallback.
                                logger.exception(
                                    "Master failed to persist model trust locally; "
                                    "continuing with the indexed cluster decision"
                                )
                            generated_events.append(
                                ModelTrustApprovalChanged(
                                    trust_identity=command.trust_identity,
                                    approved=command.approved,
                                )
                            )
                        case ProposeStewardAction():
                            proposal = command.proposal
                            now = datetime.now(tz=timezone.utc)
                            generated_events.extend(
                                self._expire_steward_action_proposals(now)
                            )
                            existing = self._ordered_steward_proposals.get(
                                proposal.proposal_id
                            )
                            if existing is not None:
                                logger.info(
                                    "Ignoring redelivered steward proposal "
                                    f"{proposal.proposal_id}"
                                )
                            else:
                                if proposal.status != "pending":
                                    raise ValueError(
                                        "A new steward proposal must be pending"
                                    )
                                if proposal.expires_at <= now:
                                    raise ValueError(
                                        "A new steward proposal must not already be expired"
                                    )
                                if proposal.created_at > now + timedelta(seconds=30):
                                    raise ValueError(
                                        "A new steward proposal cannot be future-dated"
                                    )
                                if (
                                    proposal.expires_at - proposal.created_at
                                    > timedelta(minutes=15)
                                ):
                                    raise ValueError(
                                        "Steward proposals may live for at most 15 minutes"
                                    )
                                pending_count = sum(
                                    item.status == "pending"
                                    for item in self._ordered_steward_proposals.values()
                                )
                                if pending_count >= 32:
                                    raise ValueError(
                                        "Too many steward proposals are awaiting approval"
                                    )
                                self._ordered_steward_proposals[
                                    proposal.proposal_id
                                ] = proposal
                                self._prune_ordered_steward_action_proposals()
                                generated_events.append(
                                    StewardActionProposalChanged(proposal=proposal)
                                )
                        case DecideStewardAction():
                            proposal = self._ordered_steward_proposals.get(
                                command.proposal_id
                            )
                            if proposal is None:
                                raise ValueError("Steward proposal not found")
                            if proposal.status != "pending":
                                raise ValueError(
                                    "Steward proposal has already been decided"
                                )
                            now = datetime.now(tz=timezone.utc)
                            if proposal.expires_at <= now:
                                decided = proposal.model_copy(
                                    update={
                                        "status": "expired",
                                        "decided_at": now,
                                        "decided_by": command.decided_by,
                                        "outcome": "Approval arrived after proposal expiry.",
                                    }
                                )
                            elif not command.approved:
                                decided = proposal.model_copy(
                                    update={
                                        "status": "rejected",
                                        "decided_at": now,
                                        "decided_by": command.decided_by,
                                        "outcome": "Rejected by the operator.",
                                    }
                                )
                            elif os.getenv("SKULK_FABRIC_CAPABILITIES_DISABLE") == "1":
                                decided = proposal.model_copy(
                                    update={
                                        "status": "failed",
                                        "decided_at": now,
                                        "decided_by": command.decided_by,
                                        "outcome": "Fabric actions are disabled by the global kill switch.",
                                    }
                                )
                            else:
                                try:
                                    action_events, action_command_id, action_status = (
                                        await self._execute_approved_steward_action(
                                            proposal
                                        )
                                    )
                                except (PlacementError, ValueError) as error:
                                    decided = proposal.model_copy(
                                        update={
                                            "status": "failed",
                                            "decided_at": now,
                                            "decided_by": command.decided_by,
                                            "outcome": str(error)[:1024],
                                        }
                                    )
                                else:
                                    decided = proposal.model_copy(
                                        update={
                                            "status": action_status,
                                            "decided_at": now,
                                            "dispatched_at": (
                                                now
                                                if action_status == "dispatched"
                                                else None
                                            ),
                                            "decided_by": command.decided_by,
                                            "command_id": action_command_id,
                                            "outcome": (
                                                "Download cancellation was approved and awaits "
                                                "durable dispatch."
                                                if isinstance(
                                                    proposal.action,
                                                    StewardCancelDownloadAction,
                                                )
                                                else "Restart teardown was dispatched; replacement "
                                                "waits for released capacity."
                                                if action_status == "approved"
                                                else "Approved action was dispatched through "
                                                "the typed command path."
                                            ),
                                        }
                                    )
                                    generated_events.extend(action_events)
                            self._ordered_steward_proposals[
                                command.proposal_id
                            ] = decided
                            self._prune_ordered_steward_action_proposals()
                            generated_events.insert(
                                0, StewardActionProposalChanged(proposal=decided)
                            )
                        case DeleteInstance():
                            # Credit the freed memory back to placement admission
                            # for a short grace window so a back-to-back placement
                            # is not refused on gossip-lagged node memory (#314).
                            deleted_instance = self.state.instances.get(
                                command.instance_id
                            )
                            if deleted_instance is not None:
                                self._record_freed_instance(deleted_instance)
                            placement = delete_instance(command, self.state.instances)
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            for cmd in cancel_unnecessary_downloads(
                                placement, self._effective_downloads()
                            ):
                                await self.download_command_sender.send(
                                    ForwarderDownloadCommand(
                                        origin=self._system_id, command=cmd
                                    )
                                )
                            generated_events.extend(transition_events)
                        case FailInstance():
                            # Failure teardown is deliberately distinct from an
                            # operator stop. Capture model, nodes, and cause while
                            # the instance still exists, then delete it through the
                            # same lifecycle path as an ordinary stop.
                            failed_instance = self.state.instances.get(
                                command.instance_id
                            )
                            if failed_instance is None:
                                logger.info(
                                    "FailInstance for unknown instance "
                                    f"{command.instance_id}; ignoring redelivery"
                                )
                            else:
                                self._record_freed_instance(failed_instance)
                                placement = delete_instance(
                                    DeleteInstance(instance_id=command.instance_id),
                                    self.state.instances,
                                )
                                generated_events.append(
                                    instance_failure_event(
                                        failed_instance,
                                        error_code=command.error_code,
                                        error_message=command.error_message,
                                    )
                                )
                                for cancel_command in cancel_unnecessary_downloads(
                                    placement, self._effective_downloads()
                                ):
                                    await self.download_command_sender.send(
                                        ForwarderDownloadCommand(
                                            origin=self._system_id,
                                            command=cancel_command,
                                        )
                                    )
                                generated_events.extend(
                                    get_transition_events(
                                        self.state.instances,
                                        placement,
                                        self.state.tasks,
                                    )
                                )
                        case RefuseInstancePlacement():
                            # A worker could not fit its shard at load time
                            # (#290). Delete the refused instance and re-place
                            # the model one node wider so each node holds a
                            # smaller share. If even a full-width split will not
                            # fit, place_instance raises PlacementError and we
                            # stop at the deletion — that terminal case bounds
                            # the refuse→re-place loop to the cluster size.
                            refused = self.state.instances.get(command.instance_id)
                            if (
                                command.instance_id in self._fallback_placed_instances
                                and refused is not None
                            ):
                                # Second recovery hop already used: tear down
                                # and stop. Re-placing again would oscillate
                                # (see _fallback_placed_instances).
                                self._refusal_replaced.add(command.instance_id)
                                logger.error(
                                    "Fallback placement "
                                    f"{command.instance_id} was itself refused "
                                    f"on {command.node_id}; giving up on this "
                                    "placement (two recovery hops used)."
                                )
                                after_delete = delete_instance(
                                    DeleteInstance(instance_id=command.instance_id),
                                    self.state.instances,
                                )
                                await self._queue_control_event(
                                    instance_failure_event(
                                        refused,
                                        error_code="placement_failed",
                                        error_message=(
                                            "Skulk exhausted placement recovery after "
                                            f"a node refused its shard: {command.reason}"
                                        )[:2048],
                                    )
                                )
                                # Same download hygiene as every other delete
                                # path: a rank still mid-download for the
                                # torn-down instance must be cancelled, or it
                                # wastes bandwidth/disk and can re-trigger
                                # wedged-download recovery later.
                                for cancel_cmd in cancel_unnecessary_downloads(
                                    after_delete, self._effective_downloads()
                                ):
                                    await self.download_command_sender.send(
                                        ForwarderDownloadCommand(
                                            origin=self._system_id,
                                            command=cancel_cmd,
                                        )
                                    )
                                for event in get_transition_events(
                                    self.state.instances,
                                    after_delete,
                                    self.state.tasks,
                                ):
                                    await self._queue_control_event(event)
                            elif command.instance_id in self._refusal_replaced:
                                # Another rank of the same instance already
                                # triggered re-placement (self.state lags command
                                # processing, so both ranks can still see the
                                # instance present) — re-place exactly once.
                                logger.info(
                                    "RefuseInstancePlacement for "
                                    f"{command.instance_id} already handled; ignoring"
                                )
                            elif refused is None:
                                # Already gone (operator delete or redelivery) —
                                # no-op.
                                logger.info(
                                    "RefuseInstancePlacement for unknown instance "
                                    f"{command.instance_id}; ignoring"
                                )
                            else:
                                self._refusal_replaced.add(command.instance_id)
                                after_delete = delete_instance(
                                    DeleteInstance(instance_id=command.instance_id),
                                    self.state.instances,
                                )
                                replace_command = (
                                    replacement_command_for_refused_instance(refused)
                                )
                                try:
                                    final_placement = place_instance(
                                        replace_command,
                                        self.state.topology,
                                        after_delete,
                                        self._telemetry_view.node_memory,
                                        self.state.node_network,
                                        download_status=self._effective_downloads(),
                                        excluded_nodes=set(
                                            replace_command.excluded_nodes
                                        ),
                                        stamped_exclusions=set(refused.excluded_nodes),
                                        node_resources=self._telemetry_view.node_resources,
                                        node_vram=usable_vram_by_node(
                                            self._telemetry_view.node_system,
                                            self._telemetry_view.node_resources,
                                            node_memory=self._telemetry_view.node_memory,
                                            current_instances=self._placement_reservations(after_delete),
                                        ),
                                        unified_memory_gpu_nodes=unified_memory_gpu_node_ids(
                                            self._telemetry_view.node_system,
                                            self._telemetry_view.node_resources,
                                            node_memory=self._telemetry_view.node_memory,
                                        ),
                                        approved_remote_code_identities=self._model_trust_approvals,
                                    )
                                    logger.warning(
                                        "Re-placing "
                                        f"{replace_command.model_card.model_id} at "
                                        f"min_nodes={replace_command.min_nodes} after "
                                        f"{command.node_id} refused its shard "
                                        f"({command.reason})"
                                    )
                                except PlacementInfoPendingError as err:
                                    # Telemetry for a needed node has not arrived
                                    # yet (rare during a load-time refusal, since
                                    # the cluster is already running). Distinct
                                    # from a true shortfall: the refused instance
                                    # still can't stay on the node that rejected
                                    # it, so it is torn down, but log the
                                    # transient cause rather than "cannot fit".
                                    final_placement = after_delete
                                    logger.error(
                                        "Cannot re-place "
                                        f"{replace_command.model_card.model_id} after "
                                        f"refusal on {command.node_id}: cluster info "
                                        f"still gossiping ({err}). Torn down."
                                    )
                                except PlacementError as err:
                                    # The wider width can be unsatisfiable by
                                    # construction on a heterogeneous fleet (an
                                    # MLX model refused at the full Mac width
                                    # cannot add an AMD node). Fall back to
                                    # anywhere-but-the-refuser at min_nodes=1:
                                    # the memory fit-check, not the width,
                                    # decides. Only a second failure is
                                    # terminal.
                                    fallback = fallback_command_for_refused_instance(
                                        refused, command.node_id
                                    )
                                    try:
                                        final_placement = place_instance(
                                            fallback,
                                            self.state.topology,
                                            after_delete,
                                            self._telemetry_view.node_memory,
                                            self.state.node_network,
                                            download_status=self._effective_downloads(),
                                            excluded_nodes=set(fallback.excluded_nodes),
                                            stamped_exclusions=set(
                                                refused.excluded_nodes
                                            ),
                                            node_resources=self._telemetry_view.node_resources,
                                            node_vram=usable_vram_by_node(
                                                self._telemetry_view.node_system,
                                                self._telemetry_view.node_resources,
                                                node_memory=self._telemetry_view.node_memory,
                                                current_instances=self._placement_reservations(after_delete),
                                            ),
                                            unified_memory_gpu_nodes=unified_memory_gpu_node_ids(
                                                self._telemetry_view.node_system,
                                                self._telemetry_view.node_resources,
                                                node_memory=self._telemetry_view.node_memory,
                                            ),
                                            approved_remote_code_identities=self._model_trust_approvals,
                                        )
                                        for new_id in final_placement:
                                            if new_id not in after_delete:
                                                self._fallback_placed_instances.add(
                                                    new_id
                                                )
                                        logger.warning(
                                            "Re-placing "
                                            f"{fallback.model_card.model_id} "
                                            f"excluding refusing node "
                                            f"{command.node_id} (wider width "
                                            f"min_nodes={replace_command.min_nodes} "
                                            f"was unplaceable: {err})"
                                        )
                                    except (
                                        PlacementError,
                                        PlacementInfoPendingError,
                                    ) as fallback_err:
                                        final_placement = after_delete
                                        logger.error(
                                            "Cannot re-place "
                                            f"{replace_command.model_card.model_id} after "
                                            f"refusal on {command.node_id} (tried "
                                            f"min_nodes={replace_command.min_nodes}, then "
                                            f"excluding the refuser: {fallback_err}). "
                                            "Giving up on this placement."
                                        )
                                replacement_created = any(
                                    instance_id not in after_delete
                                    for instance_id in final_placement
                                )
                                generated_events.append(
                                    instance_failure_event(
                                        refused,
                                        error_code="placement_failed",
                                        error_message=(
                                            (
                                                (
                                                    "Skulk replaced this placement "
                                                    "after a node refused its shard: "
                                                )
                                                if replacement_created
                                                else (
                                                    "Skulk could not recover a "
                                                    "placement after a node refused "
                                                    "its shard: "
                                                )
                                            )
                                            + command.reason
                                        )[:2048],
                                    )
                                )
                                transition_events = get_transition_events(
                                    self.state.instances,
                                    final_placement,
                                    self.state.tasks,
                                )
                                for cmd in cancel_unnecessary_downloads(
                                    final_placement, self._effective_downloads()
                                ):
                                    await self.download_command_sender.send(
                                        ForwarderDownloadCommand(
                                            origin=self._system_id, command=cmd
                                        )
                                    )
                                generated_events.extend(transition_events)
                        case PlaceInstance():
                            self._require_ordered_place_instance_card(command)
                            # node_memory/node_vram come from the telemetry plane
                            # (#279 slice 2). Recently-freed credit is pruned
                            # here and disabled by default (#314); the usable-GPU
                            # map admits discrete/UMA GPU nodes against the pool
                            # their backend can actually allocate from.
                            credited_memory, credited_vram = (
                                self._placement_memory_inputs()
                            )
                            placement = place_instance(
                                command,
                                self.state.topology,
                                self.state.instances,
                                credited_memory,
                                self.state.node_network,
                                download_status=self._effective_downloads(),
                                excluded_nodes=set(command.excluded_nodes),
                                node_resources=self._telemetry_view.node_resources,
                                node_vram=credited_vram,
                                unified_memory_gpu_nodes=unified_memory_gpu_node_ids(
                                    self._telemetry_view.node_system,
                                    self._telemetry_view.node_resources,
                                    node_memory=credited_memory,
                                ),
                                approved_remote_code_identities=self._model_trust_approvals,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case CreateInstance():
                            self._require_ordered_create_instance_card(command)
                            # Placement inputs come from telemetry (#279 slice 2);
                            # the recently-freed bookkeeping path is pruned here
                            # and normally contributes no speculative credit.
                            credited_memory, credited_vram = (
                                self._placement_memory_inputs()
                            )
                            placement = add_instance_to_placements(
                                command,
                                self.state.topology,
                                self.state.instances,
                                credited_memory,
                                node_vram=credited_vram,
                                unified_memory_gpu_nodes=unified_memory_gpu_node_ids(
                                    self._telemetry_view.node_system,
                                    self._telemetry_view.node_resources,
                                    node_memory=credited_memory,
                                ),
                                approved_remote_code_identities=self._model_trust_approvals,
                                node_resources=self._telemetry_view.node_resources,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case SendInputChunk():
                            logger.warning(
                                "Rejected legacy payload input command; media must "
                                "use a node-addressed data transport"
                            )
                        case TaskCancelled():
                            self._realtime_instance_by_command.pop(
                                command.cancelled_command_id, None
                            )
                            if (
                                task_id := self.command_task_mapping.get(
                                    command.cancelled_command_id
                                )
                            ) is not None:
                                generated_events.append(
                                    TaskStatusUpdated(
                                        task_status=TaskStatus.Cancelled,
                                        task_id=task_id,
                                    )
                                )
                            else:
                                logger.warning(
                                    f"Nonexistent command {command.cancelled_command_id} cancelled"
                                )
                        case TaskFinished():
                            self._realtime_instance_by_command.pop(
                                command.finished_command_id, None
                            )
                            if (
                                task_id := self.command_task_mapping.pop(
                                    command.finished_command_id, None
                                )
                            ) is not None:
                                generated_events.append(TaskDeleted(task_id=task_id))
                            else:
                                logger.warning(
                                    f"Finished command {command.finished_command_id} finished"
                                )

                        case AddCustomModelCard():
                            if (
                                event := self._order_custom_model_card_add(command)
                            ) is not None:
                                generated_events.append(event)
                        case DeleteCustomModelCard():
                            if (
                                event := self._order_custom_model_card_delete(command)
                            ) is not None:
                                generated_events.append(event)
                        case EvictStagedModel():
                            # Broadcast a fleet-wide eviction of the store-deleted
                            # model: apply() drops its download entries; workers
                            # remove their staged copies on disk (#427).
                            generated_events.append(
                                StagedModelEvicted(model_id=command.model_id)
                            )
                        case RequestEventLog():
                            self._schedule_event_log_replay(command.since_idx)
                    for event in generated_events:
                        await self._queue_control_event(event)
                except PlacementError as error:
                    if (
                        isinstance(forwarder_command.command, CreateInstance)
                        and forwarder_command.command.instance.instance_id
                        not in self._placement_reservations(self.state.instances)
                    ):
                        # Exact-create acknowledgements precede master admission.
                        # Retain failure identity so controllers can clean up a
                        # refused placement instead of waiting for a missing runner.
                        await self._queue_control_event(
                            instance_failure_event(
                                forwarder_command.command.instance,
                                error_code="placement_failed",
                                error_message="Exact placement failed current admission checks.",
                            )
                        )
                    elif isinstance(forwarder_command.command, PlaceInstance):
                        # Quick-launch preflight also precedes ordered admission.
                        # Its acknowledged instance ID is derived from the command,
                        # even when no concrete placement could be constructed.
                        refused = forwarder_command.command
                        instance_id = InstanceId(str(refused.command_id))
                        if instance_id not in self._placement_reservations(
                            self.state.instances
                        ):
                            await self._queue_control_event(
                                InstanceFailureRecorded(
                                    failure=InstanceFailure(
                                        instance_id=instance_id,
                                        model_id=refused.model_card.model_id,
                                        system_role=refused.system_role,
                                        error_code="placement_failed",
                                        error_message="Quick placement failed current admission checks.",
                                        recorded_at=datetime.now(tz=timezone.utc),
                                    )
                                )
                            )
                    logger.opt(exception=error).warning("Placement command refused")
                except ValueError as e:
                    logger.opt(exception=e).warning("Error in command processor")

    def _report_heartbeat_gap_changes(self, *, now: datetime) -> None:
        """Log dedicated-heartbeat degradation and recovery once per transition."""
        current_nodes = set(self.state.last_seen)
        gap_nodes = compute_heartbeat_gap_nodes(
            self.state.last_seen,
            self._telemetry_view.node_last_heartbeat,
            now=now,
        )
        # Once absence has produced a warning, only an actual heartbeat receipt
        # clears it. A later control event grants initial grace to a new node but
        # must not masquerade as heartbeat recovery.
        gap_nodes |= {
            node_id
            for node_id in self._heartbeat_gap_warned_nodes & current_nodes
            if node_id not in self._telemetry_view.node_last_heartbeat
        }
        observations = compute_node_timeout_evidence(
            self.state.last_seen,
            self._telemetry_view.node_last_heartbeat,
            self._telemetry_view.node_last_telemetry,
            now=now,
        )
        for node_id in sorted(gap_nodes - self._heartbeat_gap_warned_nodes, key=str):
            evidence = observations[node_id]
            logger.bind(
                liveness_event="heartbeat_gap",
                node_id=str(node_id),
                heartbeat_age_seconds=evidence.heartbeat_age_seconds,
                fallback_telemetry_age_seconds=(
                    evidence.fallback_telemetry_age_seconds
                ),
                last_logged_event_age_seconds=(evidence.last_logged_event_age_seconds),
            ).warning(
                f"Dedicated heartbeat from node {node_id} is late or absent; "
                "ordinary telemetry and logged events remain liveness fallbacks"
            )
        recovered_nodes = (self._heartbeat_gap_warned_nodes - gap_nodes) & current_nodes
        for node_id in sorted(recovered_nodes, key=str):
            heartbeat_at = self._telemetry_view.node_last_heartbeat[node_id]
            logger.bind(
                liveness_event="heartbeat_recovered",
                node_id=str(node_id),
                heartbeat_age_seconds=_signal_age_seconds(
                    now=now, seen_at=heartbeat_at
                ),
            ).info(f"Dedicated heartbeat from node {node_id} recovered")
        self._heartbeat_gap_warned_nodes = gap_nodes

    # These plan loops are the cracks showing in our event sourcing architecture - more things could be commands
    async def _plan(self) -> None:
        while True:
            connected_node_ids = set(self.state.topology.list_nodes())
            now = datetime.now(tz=timezone.utc)
            for expiry_event in self._expire_steward_action_proposals(now):
                await self._queue_control_event(expiry_event)
            self._report_heartbeat_gap_changes(now=now)
            # ALL liveness-based action is suppressed while this session's
            # topology is still settling (#273): a failover-seeded master
            # carries instances but rebuilds topology and last_seen from
            # live gossip, so for the first probe cycles every carried node
            # looks "disconnected" — acting on that view would delete
            # exactly the placements the seed preserved. The suppression
            # must cover timed_out_node_ids too, not just the instance
            # pruning: NodeTimedOut's apply removes the node's instances
            # AND their tasks outright, and the TaskFailed-before-removal
            # invariant (#223/#224) is only upheld when the corresponding
            # dying_instance_ids pass ran in the same tick — a NodeTimedOut
            # emitted during the grace would strand in-flight API requests
            # without a terminal chunk (review catch on #274). After the
            # grace, absence means absence and cleanup proceeds normally.
            topology_settled = (
                time.monotonic() - self._started_monotonic
                >= TOPOLOGY_SETTLE_GRACE_SECONDS
            )
            timeout_evidence = (
                compute_node_timeout_evidence(
                    self.state.last_seen,
                    self._telemetry_view.node_last_heartbeat,
                    self._telemetry_view.node_last_telemetry,
                    now=now,
                )
                if topology_settled
                else {}
            )
            timed_out_node_ids = _timed_out_nodes_from_evidence(timeout_evidence)
            dying_instance_ids: set[InstanceId] = (
                instances_on_dead_nodes(
                    self.state, connected_node_ids, timed_out_node_ids
                )
                if topology_settled
                else set()
            )

            # Fail in-flight API tasks stranded by a dead or dying instance
            # so open HTTP requests terminate with an error instead of
            # hanging (#223). Emitted BEFORE InstanceDeleted/NodeTimedOut so
            # TaskFailed indexes ahead of the applies that remove the task
            # from state (NodeTimedOut deletes its instances' tasks
            # outright). TaskFailed flips task_status to Failed on apply, so
            # each task is emitted at most once across passes.
            for task_failed in orphaned_task_failure_events(
                self.state, dying_instance_ids
            ):
                logger.warning(
                    f"Failing orphaned task {task_failed.task_id}: "
                    f"{task_failed.error_message}"
                )
                await self._queue_control_event(task_failed)

            # Reap lifecycle tasks whose executor died with its old node
            # identity (#647): grace-based because instance deletion already
            # removed the task-to-node attribution. Suppressed during the
            # topology-settle grace like every liveness-derived action.
            if topology_settled:
                for task_failed in stale_lifecycle_task_failures(
                    self.state,
                    self._orphaned_lifecycle_first_seen,
                    now=time.monotonic(),
                ):
                    logger.warning(
                        f"Failing stale lifecycle task {task_failed.task_id}: "
                        f"{task_failed.error_message}"
                    )
                    await self._queue_control_event(task_failed)

            # Retain the failure before either InstanceDeleted or NodeTimedOut
            # removes the placement. Timed-out nodes may still be present in
            # topology, so this must use the same combined liveness truth as
            # dying_instance_ids rather than topology absence alone.
            if topology_settled:
                for failure_event in dead_node_instance_failure_events(
                    self.state, connected_node_ids, timed_out_node_ids
                ):
                    await self._queue_control_event(failure_event)

            # Kill instances whose assigned node has already left topology.
            # NodeTimedOut below owns teardown for timed-out-but-still-present
            # nodes, preventing duplicate deletion events.
            if topology_settled:
                for instance_id, instance in self.state.instances.items():
                    for node_id in instance.shard_assignments.node_to_runner:
                        if node_id not in connected_node_ids:
                            await self._queue_control_event(
                                InstanceDeleted(instance_id=instance_id)
                            )
                            break

            # time out dead nodes
            for node_id in timed_out_node_ids:
                evidence = timeout_evidence[node_id]
                logger.bind(
                    liveness_event="node_timed_out",
                    node_id=str(node_id),
                    last_logged_event_age_seconds=(
                        evidence.last_logged_event_age_seconds
                    ),
                    heartbeat_age_seconds=evidence.heartbeat_age_seconds,
                    fallback_telemetry_age_seconds=(
                        evidence.fallback_telemetry_age_seconds
                    ),
                    effective_age_seconds=evidence.effective_age_seconds,
                    timeout_seconds=evidence.timeout_seconds,
                ).warning(
                    f"Removing node {node_id}: all liveness signals exceeded "
                    f"the {evidence.timeout_seconds:.0f}s timeout"
                )
                await self._queue_control_event(
                    NodeTimedOut(node_id=node_id, evidence=evidence)
                )

            # Recover instances wedged because a rank's download failed (#381).
            # Suppressed during the settle grace for the same reason as the
            # liveness passes above: a freshly seeded master sees stale download
            # state until live gossip refreshes it. A terminal DownloadFailed on
            # a not-yet-ready instance can never clear on its own, so fail the
            # instance (surfacing the cause to any waiting request) and re-place
            # it excluding the failed node(s). Deduped so it acts once per wedged
            # instance, not every tick while the events round-trip.
            if topology_settled:
                await self._recover_download_failed_instances()

            # Intelligent fabric (steward) invariant: while the mode is
            # enabled, exactly one steward system placement exists. Behind
            # the settle grace so a freshly failed-over master does not act
            # on a stale seeded view; because this runs on every planning
            # tick, a new master re-establishes the steward after election
            # without any dedicated failover machinery.
            if topology_settled:
                await self._reconcile_dispatched_steward_actions(now)
                await self._arm_approved_steward_download_cancellations()
                await self._resume_approved_steward_restarts(now)
                await self._maintain_steward_placement()

            await anyio.sleep(10)

    async def _recover_download_failed_instances(self) -> None:
        """Fail and re-place instances stuck because a rank's download failed (#381).

        See :func:`instances_wedged_by_download_failure`. For each newly-detected
        wedged instance: fail any in-flight API task bound to it (so an open
        request gets a clean error instead of hanging), tear the instance down,
        and re-place the model at the same width excluding the failed node(s),
        reusing the #290 placement machinery. If no healthy node set can host the
        width, placement raises and we stop at the teardown, which bounds
        recovery to the available nodes instead of looping.
        """
        wedged = instances_wedged_by_download_failure(self.state)
        for instance_id, (failed_nodes, cause) in wedged.items():
            if instance_id in self._download_failure_recovered:
                continue
            instance = self.state.instances.get(instance_id)
            if instance is None:
                continue
            self._download_failure_recovered.add(instance_id)
            failed_list = sorted(str(node_id) for node_id in failed_nodes)
            logger.error(
                f"Instance {instance_id} wedged: a rank's download failed on "
                f"{failed_list} ({cause}). Failing it and re-placing without the "
                "failed node(s)."
            )
            # Fail any in-flight API task bound to this instance before the
            # teardown events index (the TaskFailed-before-removal invariant,
            # #223): get_transition_events below emits InstanceDeleted, whose
            # apply drops the instance's tasks.
            for task_failed in orphaned_task_failure_events(self.state, {instance_id}):
                await self._queue_control_event(task_failed)
            after_delete = delete_instance(
                DeleteInstance(instance_id=instance_id), self.state.instances
            )
            final_placement = after_delete
            try:
                replace_command = replacement_command_for_download_failed_instance(
                    instance, failed_nodes
                )
                final_placement = place_instance(
                    replace_command,
                    self.state.topology,
                    after_delete,
                    self._telemetry_view.node_memory,
                    self.state.node_network,
                    download_status=self._effective_downloads(),
                    excluded_nodes=set(replace_command.excluded_nodes),
                    stamped_exclusions=set(instance.excluded_nodes),
                    node_resources=self._telemetry_view.node_resources,
                    node_vram=usable_vram_by_node(
                        self._telemetry_view.node_system,
                        self._telemetry_view.node_resources,
                        node_memory=self._telemetry_view.node_memory,
                        current_instances=self._placement_reservations(after_delete),
                    ),
                    unified_memory_gpu_nodes=unified_memory_gpu_node_ids(
                        self._telemetry_view.node_system,
                        self._telemetry_view.node_resources,
                        node_memory=self._telemetry_view.node_memory,
                    ),
                    approved_remote_code_identities=self._model_trust_approvals,
                )
                logger.warning(
                    f"Re-placing {replace_command.model_card.model_id} excluding "
                    f"{failed_list} after a download failure"
                )
            except (PlacementError, PlacementInfoPendingError) as err:
                final_placement = after_delete
                logger.error(
                    f"Cannot re-place {instance_id}'s model after a download "
                    f"failure on {failed_list}: {err}. Torn down."
                )
            transition_events = get_transition_events(
                self.state.instances, final_placement, self.state.tasks
            )
            await self._queue_control_event(
                instance_failure_event(
                    instance,
                    error_code="download_failed",
                    error_message=(
                        "A node could not stage the model, so Skulk tore down "
                        "this placement and attempted recovery. Inspect cluster "
                        "logs for the underlying storage or transport error."
                    ),
                )
            )
            for cmd in cancel_unnecessary_downloads(
                final_placement, self._effective_downloads()
            ):
                await self.download_command_sender.send(
                    ForwarderDownloadCommand(origin=self._system_id, command=cmd)
                )
            for event in transition_events:
                await self._queue_control_event(event)
            # Recovery CONSUMES the terminal failure record: reset each failed
            # node's download status for this model back to Pending. Without
            # this, the stale DownloadFailed lingers in session state and this
            # same scan condemns EVERY future placement of the model that
            # touches the node, long after the cause (disk full, network blip)
            # is gone -- one transient failure permanently poisoned the model
            # on that node until a whole-fleet restart (observed live: an
            # ENOSPC during a pooled placement kept killing fresh placements
            # an hour after the disk was freed). This recovery pass already
            # acted on the failure (teardown + re-place excluding the node);
            # if the cause persists, the next download fails afresh and
            # recovery repeats, so nothing is lost by clearing history.
            for node_id in failed_nodes:
                runner_id = instance.shard_assignments.node_to_runner.get(node_id)
                shard = (
                    instance.shard_assignments.runner_to_shard.get(runner_id)
                    if runner_id is not None
                    else None
                )
                if shard is None:
                    continue
                await self._queue_control_event(
                    NodeDownloadProgress(
                        download_progress=DownloadPending(
                            node_id=node_id,
                            shard_metadata=shard,
                            model_directory="",
                            attempt_id=DownloadAttemptId(),
                        )
                    )
                )
                logger.info(
                    f"Reset stale DownloadFailed for "
                    f"{shard.model_card.model_id} on {node_id} (consumed by "
                    "recovery)"
                )

    async def _maintain_steward_placement(self) -> None:
        """Re-establish the intelligent-fabric steward placement invariant.

        While ``intelligent_fabric.enabled`` is set in the cluster config,
        exactly one instance carrying ``system_role="steward"`` must exist.
        This pass places the first card from the configured preference list
        the cluster can serve, tears down accidental duplicates (possible
        across master handoffs), and does nothing while a steward exists.
        Node loss needs no special handling here: the dead steward's
        instance is torn down by the liveness pass and this invariant
        re-places it on the next tick. Attempts are paced to one per minute
        so an unplaceable steward logs calmly instead of spamming.
        """
        try:
            config = load_skulk_config()
        except Exception:
            # An unreadable config never breaks the planning loop; the mode
            # simply stays off until the config loads again.
            return
        fabric = config.intelligent_fabric if config is not None else None
        stewards = sorted(
            (
                instance_id
                for instance_id, instance in self.state.instances.items()
                if instance.system_role == "steward"
            ),
            key=str,
        )
        if fabric is None or not fabric.enabled:
            # Disable is a lifecycle transition, not a shrug: a steward left
            # behind would keep occupying memory while hidden from every
            # ordinary instance surface. Symmetric with enable.
            if stewards:
                logger.info(
                    "Intelligent fabric is disabled; removing the steward "
                    f"placement(s) {[str(s) for s in stewards]}"
                )
                await self._teardown_steward_instances(stewards)
            self._reset_steward_upgrade()
            return

        if len(stewards) > 1:
            # Duplicate stewards can appear when two masters each placed one
            # around a failover window. Keep the lowest id (stable across
            # replicas), tear down the rest.
            extras = stewards[1:]
            logger.warning(
                f"Removing duplicate steward placement(s) "
                f"{[str(e) for e in extras]} (keeping {stewards[0]})"
            )
            await self._teardown_steward_instances(extras)
            return
        if stewards:
            await self._maintain_steward_upgrade(
                stewards[0], tuple(fabric.steward_models)
            )
            return

        if self._steward_upgrade_replacing_instance is not None:
            # The old placement has now left replicated state. Make the
            # already-staged successor eligible immediately instead of waiting
            # behind the ordinary one-minute no-placement retry pace.
            self._steward_upgrade_replacing_instance = None
            self._steward_last_attempt_monotonic = 0.0

        now = time.monotonic()
        if now - self._steward_last_attempt_monotonic < 60:
            return
        self._steward_last_attempt_monotonic = now

        # The card cache is lazily filled; a fresh master process may not
        # have loaded it yet.
        await get_model_cards()
        for model_ref in fabric.steward_models:
            try:
                final_placement = await self._place_steward_model(
                    model_ref, self.state.instances
                )
            except (PlacementError, PlacementInfoPendingError) as err:
                logger.warning(
                    f"Steward placement with {model_ref} not possible yet: {err}"
                )
                continue
            if final_placement is None:
                continue
            logger.info(
                f"Establishing steward placement with {model_ref} (intelligent fabric)"
            )
            for event in get_transition_events(
                self.state.instances, final_placement, self.state.tasks
            ):
                await self._queue_control_event(event)
            return
        logger.warning(
            "Intelligent fabric is enabled but no configured steward model "
            "can be placed on the current topology; will retry"
        )

    async def _place_steward_model(
        self,
        model_ref: str,
        current_instances: Mapping[InstanceId, Instance],
        node_memory: Mapping[NodeId, MemoryUsage] | None = None,
        node_vram: Mapping[NodeId, Memory] | None = None,
    ) -> dict[InstanceId, Instance] | None:
        """Return a steward placement for one exact brain, without emitting it."""
        await get_model_cards()
        card = get_card(ModelId(model_ref))
        if card is None:
            logger.warning(f"Steward model {model_ref} has no model card; skipping")
            return None
        command = PlaceInstance(
            model_card=card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
            system_role="steward",
        )
        placement_memory = node_memory or self._telemetry_view.node_memory
        placement_vram = (
            node_vram
            if node_vram is not None
            else usable_vram_by_node(
                self._telemetry_view.node_system,
                self._telemetry_view.node_resources,
                node_memory=placement_memory,
                current_instances=self._placement_reservations(current_instances),
            )
        )
        return place_instance(
            command,
            self.state.topology,
            current_instances,
            placement_memory,
            self.state.node_network,
            download_status=self._effective_downloads(),
            node_resources=self._telemetry_view.node_resources,
            node_vram=placement_vram,
            unified_memory_gpu_nodes=unified_memory_gpu_node_ids(
                self._telemetry_view.node_system,
                self._telemetry_view.node_resources,
                node_memory=placement_memory,
            ),
            approved_remote_code_identities=self._model_trust_approvals,
        )

    def _steward_replacement_memory_inputs(
        self, current: Instance
    ) -> tuple[Mapping[NodeId, MemoryUsage], Mapping[NodeId, Memory]]:
        """Build hypothetical RAM and VRAM with only the steward reclaimed.

        The snapshot is used only to decide whether replacement shards are
        worth prestaging. Actual post-teardown placement still uses observed
        telemetry, avoiding optimistic admission while a worker releases the
        outgoing model asynchronously.
        """
        credit: dict[NodeId, int] = {}
        assignments = current.shard_assignments
        for node_id, runner_id in assignments.node_to_runner.items():
            shard = assignments.runner_to_shard.get(runner_id)
            if shard is None:
                continue
            fraction = shard_fraction_of_model(shard)
            if fraction is None or fraction <= 0.0:
                continue
            footprint = estimate_shard_footprint(shard.model_card, fraction)
            credit[node_id] = credit.get(node_id, 0) + footprint.in_bytes
        memory = {
            node_id: (
                usage.model_copy(
                    update={
                        "ram_available": Memory.from_bytes(
                            min(
                                usage.ram_total.in_bytes,
                                usage.ram_available.in_bytes + credit[node_id],
                            )
                        )
                    }
                )
                if node_id in credit
                else usage
            )
            for node_id, usage in self._telemetry_view.node_memory.items()
        }
        unified_nodes = unified_memory_gpu_node_ids(
            self._telemetry_view.node_system,
            self._telemetry_view.node_resources,
            node_memory=memory,
        )
        vram = dict(
            usable_vram_by_node(
                self._telemetry_view.node_system,
                self._telemetry_view.node_resources,
                node_memory=memory,
            )
        )
        for node_id, reclaimed_bytes in credit.items():
            if node_id in unified_nodes or node_id not in vram:
                continue
            accelerator = getattr(
                self._telemetry_view.node_system.get(node_id), "accelerator", None
            )
            total_bytes = getattr(accelerator, "vram_total_bytes", None)
            credited_bytes = vram[node_id].in_bytes + reclaimed_bytes
            if isinstance(total_bytes, int) and total_bytes > 0:
                credited_bytes = min(total_bytes, credited_bytes)
            vram[node_id] = Memory.from_bytes(credited_bytes)
        return memory, reserve_instance_vram(
            vram,
            self._telemetry_view.node_system,
            {
                identifier: instance
                for identifier, instance in self.state.instances.items()
                if identifier != current.instance_id
            },
            unified_memory_gpu_nodes=unified_nodes,
        )

    def _reset_steward_upgrade(self) -> None:
        """Forget one in-progress best-brain convergence attempt."""
        self._steward_upgrade_model = None
        self._steward_upgrade_stable_since = None
        self._steward_upgrade_prestaged_model = None
        self._steward_upgrade_idle_since = None

    def _steward_model_download_completed(
        self, node_id: NodeId, shard_metadata: ShardMetadata
    ) -> bool:
        """Whether one node reports this exact completed steward shard."""
        return any(
            isinstance(progress, DownloadCompleted)
            and progress.node_id == node_id
            and progress.shard_metadata == shard_metadata
            for progress in self._effective_downloads().get(node_id, ())
        )

    async def _maintain_steward_upgrade(
        self,
        steward_id: InstanceId,
        preference: tuple[str, ...],
    ) -> None:
        """Converge an existing steward upward without creating a standby.

        A better brain must remain placeable for five minutes. Its exact target
        shards are then staged while the current steward keeps serving. Once
        staging is complete and the current steward has been idle/Ready for
        thirty seconds, the old placement is removed; the ordinary exactly-one
        invariant places the already-staged successor on the following tick.
        """
        if self._steward_upgrade_replacing_instance == steward_id:
            return
        now = time.monotonic()
        if now < self._steward_upgrade_retry_after:
            return
        current = self.state.instances.get(steward_id)
        if current is None:
            self._reset_steward_upgrade()
            return
        current_model = str(current.shard_assignments.model_id)
        try:
            current_index = preference.index(current_model)
        except ValueError:
            current_index = len(preference)

        candidate_model: ModelId | None = None
        candidate_instance: Instance | None = None
        instances_without_steward = {
            instance_id: instance
            for instance_id, instance in self.state.instances.items()
            if instance_id != steward_id
        }
        replacement_memory, replacement_vram = self._steward_replacement_memory_inputs(
            current
        )
        for model_ref in preference[:current_index]:
            try:
                placed = await self._place_steward_model(
                    model_ref,
                    instances_without_steward,
                    replacement_memory,
                    replacement_vram,
                )
            except (PlacementError, PlacementInfoPendingError):
                continue
            if placed is None:
                continue
            new_instances = [
                instance
                for instance_id, instance in placed.items()
                if instance_id not in instances_without_steward
            ]
            if len(new_instances) != 1:
                continue
            candidate_model = ModelId(model_ref)
            candidate_instance = new_instances[0]
            break

        if candidate_model is None or candidate_instance is None:
            self._reset_steward_upgrade()
            return
        if self._steward_upgrade_model != candidate_model:
            self._reset_steward_upgrade()
            self._steward_upgrade_model = candidate_model
            self._steward_upgrade_stable_since = now
            logger.info(
                f"Better steward brain {candidate_model} is placeable; "
                "waiting for topology stability before staging"
            )
            return
        stable_since = self._steward_upgrade_stable_since
        if (
            stable_since is None
            or now - stable_since < STEWARD_UPGRADE_STABILITY_SECONDS
        ):
            return

        required_shards: list[tuple[NodeId, ShardMetadata]] = []
        for (
            node_id,
            runner_id,
        ) in candidate_instance.shard_assignments.node_to_runner.items():
            shard = candidate_instance.shard_assignments.runner_to_shard[runner_id]
            if isinstance(shard, RpcDonorShardMetadata):
                continue
            required_shards.append((node_id, shard))
        if self._steward_upgrade_prestaged_model != candidate_model:
            for node_id, shard in required_shards:
                if self._steward_model_download_completed(node_id, shard):
                    continue
                await self.download_command_sender.send(
                    ForwarderDownloadCommand(
                        origin=self._system_id,
                        command=StartDownload(
                            target_node_id=node_id,
                            shard_metadata=shard,
                        ),
                    )
                )
            self._steward_upgrade_prestaged_model = candidate_model
            logger.info(f"Prestaging better steward brain {candidate_model}")
            return
        if any(
            not self._steward_model_download_completed(node_id, shard)
            for node_id, shard in required_shards
        ):
            return

        runner_ids = current.shard_assignments.node_to_runner.values()
        idle_ready = bool(current.shard_assignments.node_to_runner) and all(
            isinstance(self.state.runners.get(runner_id), RunnerReady)
            for runner_id in runner_ids
        )
        if idle_ready:
            idle_ready = not any(
                getattr(task, "instance_id", None) == steward_id
                and getattr(task, "task_status", None)
                in (TaskStatus.Pending, TaskStatus.Running)
                for task in self.state.tasks.values()
            )
        if not idle_ready:
            self._steward_upgrade_idle_since = None
            return
        if self._steward_upgrade_idle_since is None:
            self._steward_upgrade_idle_since = now
            return
        if now - self._steward_upgrade_idle_since < STEWARD_UPGRADE_IDLE_SECONDS:
            return

        logger.info(
            f"Replacing steward {current_model} with staged better brain "
            f"{candidate_model}"
        )
        self._steward_upgrade_replacing_instance = steward_id
        self._steward_upgrade_retry_after = now + STEWARD_UPGRADE_RETRY_COOLDOWN_SECONDS
        await self._teardown_steward_instances([steward_id])
        self._reset_steward_upgrade()

    async def _teardown_steward_instances(
        self, instance_ids: Sequence[InstanceId]
    ) -> None:
        """Tear down steward placements with full lifecycle hygiene.

        Fails any in-flight tasks first (the TaskFailed-before-removal
        invariant), emits the deletion transition events, and forwards
        download cancellations so an in-flight steward-model download does
        not keep occupying disk and bandwidth after its instance is gone —
        the same steps the ordinary DeleteInstance path performs.
        """
        for task_failed in orphaned_task_failure_events(self.state, set(instance_ids)):
            await self._queue_control_event(task_failed)
        survivors = dict(self.state.instances)
        for instance_id in instance_ids:
            survivors = delete_instance(
                DeleteInstance(instance_id=instance_id), survivors
            )
        for cmd in cancel_unnecessary_downloads(survivors, self._effective_downloads()):
            await self.download_command_sender.send(
                ForwarderDownloadCommand(origin=self._system_id, command=cmd)
            )
        for event in get_transition_events(
            self.state.instances, survivors, self.state.tasks
        ):
            await self._queue_control_event(event)

    async def _event_processor(self) -> None:
        with self.local_event_receiver as local_events:
            async for local_event in local_events:
                # Discard all events not from our session
                if local_event.session != self.session_id:
                    continue
                if not is_persistable_control_event(local_event.event):
                    if self._should_warn_for_non_control_event(
                        local_event.origin,
                        type(local_event.event),
                        now=time.monotonic(),
                    ):
                        # Envelope events (NodeGatheredInfo and friends) all
                        # share one event name; without the payload type the
                        # warning cannot identify WHICH reading keeps landing
                        # on the wrong plane (#633).
                        rejected = local_event.event
                        payload: object | None = None
                        if isinstance(rejected, NodeGatheredInfo):
                            payload = rejected.info
                        elif isinstance(rejected, NodeDownloadProgress):
                            payload = rejected.download_progress
                        payload_note = (
                            f" carrying {type(payload).__name__}"
                            if payload is not None
                            else ""
                        )
                        logger.warning(
                            "Rejected non-control event before ordering/indexing: "
                            f"{type(local_event.event).__name__}{payload_note} "
                            f"from {local_event.origin}"
                        )
                    self._multi_buffer.skip(local_event.origin_idx, local_event.origin)
                else:
                    self._multi_buffer.ingest(
                        local_event.origin_idx,
                        local_event.event,
                        local_event.origin,
                    )
                for event in self._multi_buffer.drain():
                    if isinstance(event, TaskDeleted):
                        for command_id, task_id in list(
                            self.command_task_mapping.items()
                        ):
                            if task_id == event.task_id:
                                self.command_task_mapping.pop(command_id, None)
                                self._realtime_instance_by_command.pop(command_id, None)

                    if isinstance(event, TaskFailed) or (
                        isinstance(event, TaskStatusUpdated)
                        and event.task_status
                        not in (TaskStatus.Pending, TaskStatus.Running)
                    ):
                        for command_id, task_id in self.command_task_mapping.items():
                            if task_id == event.task_id:
                                # Terminal task state is authoritative even if
                                # the owning API disappears before TaskFinished.
                                # Preserve command mapping for eventual deletion.
                                self._realtime_instance_by_command.pop(command_id, None)

                    # Refuse to index task-lifecycle events that are state
                    # no-ops (the task is already gone). Without this cap a
                    # single misbehaving emitter could mint unbounded
                    # status/delete events for dead tasks — each one indexed,
                    # persisted, and broadcast cluster-wide — drowning
                    # replicas and starving liveness into election churn
                    # (#278; observed at ~800 events/s, 12k+ events for one
                    # task). Ordering makes this safe: TaskCreated is always
                    # indexed before any follower can reference the task, so
                    # an unknown task_id here is necessarily stale. The
                    # command-mapping sweep above still runs — it is
                    # in-memory hygiene, not amplification.
                    if (
                        isinstance(event, (TaskStatusUpdated, TaskDeleted, TaskFailed))
                        and event.task_id not in self.state.tasks
                    ):
                        logger.debug(
                            f"Dropping no-op task event for unknown task: "
                            f"{type(event).__name__}({event.task_id})"
                        )
                        continue

                    logger.debug(f"Master indexing event: {str(event)[:100]}")
                    indexed = IndexedEvent(event=event, idx=len(self._event_log))
                    self._apply_indexed_event(indexed)

                    event._master_time_stamp = datetime.now(tz=timezone.utc)  # pyright: ignore[reportPrivateUsage]
                    if isinstance(event, NodeGatheredInfo):
                        event.when = str(datetime.now(tz=timezone.utc))

                    self._append_event_log(event)
                    await self._send_event(indexed)
                    await self._persist_snapshot()

    def _schedule_event_log_replay(self, since_idx: int) -> None:
        """Coalesce replay requests onto one paced background worker."""

        active_next = self._active_replay_next_idx
        active_end = self._active_replay_end_idx
        if (
            active_next is not None
            and active_end is not None
            and active_next <= since_idx < active_end
        ):
            # The active replay has not sent this index yet, so its existing pass
            # will satisfy the request without another broadcast of the same tail.
            return
        if self._pending_replay_start_idx is None:
            self._pending_replay_start_idx = since_idx
        else:
            self._pending_replay_start_idx = min(
                self._pending_replay_start_idx, since_idx
            )
        if self._replay_worker_running:
            return
        self._replay_worker_running = True
        self._tg.start_soon(self._drain_event_log_replays)

    async def _drain_event_log_replays(self) -> None:
        """Serve coalesced replay requests without blocking command processing."""

        try:
            while self._pending_replay_start_idx is not None:
                requested_start = self._pending_replay_start_idx
                self._pending_replay_start_idx = None
                await self._serve_event_log_replay(requested_start)
        finally:
            self._active_replay_next_idx = None
            self._active_replay_end_idx = None
            self._replay_worker_running = False

    async def _serve_event_log_replay(self, requested_start: int) -> None:
        """Broadcast one retained replay tail in bounded, paced chunks."""

        retained_start = max(requested_start, self._event_log.start_idx)
        replay_start = min(retained_start, len(self._event_log))
        if requested_start < self._event_log.start_idx:
            logger.warning(
                "Requested replay index predates retained master tail; "
                f"serving from {replay_start} instead of {requested_start}"
            )
        elif requested_start > len(self._event_log):
            logger.debug(
                "Requested replay index is beyond the current master tail; "
                f"serving an empty range at {replay_start}"
            )
        replay_end = min(
            replay_start + EVENT_LOG_REPLAY_BATCH_SIZE,
            len(self._event_log),
        )
        self._active_replay_next_idx = replay_start
        self._active_replay_end_idx = replay_end
        replayed = 0
        for idx, event in enumerate(
            self._event_log.read_range(replay_start, replay_end),
            start=replay_start,
        ):
            await self._send_event(IndexedEvent(idx=idx, event=event))
            replayed += 1
            self._active_replay_next_idx = idx + 1
            if replayed % EVENT_LOG_REPLAY_CHUNK_SIZE == 0 and idx + 1 < replay_end:
                await anyio.sleep(EVENT_LOG_REPLAY_CHUNK_INTERVAL_SECONDS)
        logger.info(
            "Served paced event-log replay "
            f"(start={replay_start}, end={replay_end}, events={replayed})"
        )

    def _append_event_log(self, event: Event) -> None:
        """Append one decision and warn on sustained idle-state log growth."""

        self._event_log.append(event)
        active_download = any(
            isinstance(progress, DownloadOngoing)
            for progress_values in self._effective_downloads().values()
            for progress in progress_values
        )
        idle = not self.state.tasks and not active_download
        rate = self._event_log_growth_monitor.observe(
            now=time.monotonic(),
            idle=idle,
        )
        if rate is not None:
            logger.warning(
                "Event log is growing during an idle cluster state "
                f"({rate:.1f} events/min over "
                f"{self._event_log_growth_monitor.window_seconds:.0f}s); "
                "inspect periodic control-plane event sources before replay pressure accumulates"
            )

    # This function is re-entrant, take care!
    async def _send_event(self, event: IndexedEvent):
        # Convenience method since this line is ugly
        await self.global_event_sender.send(
            GlobalForwarderEvent(
                origin=self.node_id,
                origin_idx=event.idx,
                session=self.session_id,
                event=event.event,
            )
        )

    async def _state_sync_processor(self) -> None:
        with self.state_sync_receiver as messages:
            async for message in messages:
                if message.kind != "request":
                    continue
                if message.session_id != self.session_id:
                    continue

                config_yaml = self._load_state_sync_config_yaml()
                logger.info(
                    f"Serving state snapshot to {message.requester}: "
                    f"{len(self.state.instances)} instance(s), "
                    f"last_event_applied_idx={self.state.last_event_applied_idx}"
                )
                await self.state_sync_sender.send(
                    StateSyncMessage(
                        kind="response",
                        requester=message.requester,
                        session_id=self.session_id,
                        snapshot=StateSnapshot(
                            session_id=self.session_id,
                            last_event_applied_idx=self.state.last_event_applied_idx,
                            state=self.state,
                        ),
                        config_yaml=config_yaml,
                    )
                )

    def _load_state_sync_config_yaml(self) -> str | None:
        """Return a config payload for bootstrap responses.

        ``hf_token`` rides along: a node joining the fleet adopts the elected
        master's token so downloads work everywhere without per-node token
        entry (the fabric is PSK-encrypted and trusted by doctrine). A blank
        token is dropped so it can never clobber a joining node's real one.
        ``model_trust`` remains stripped as deprecated compatibility state.
        Read/parse failures are treated as non-fatal so bootstrap requests
        cannot crash master coordination.
        """

        config_path = resolve_config_path()
        if not config_path.exists():
            return None

        try:
            decoded_config = cast(object, yaml.safe_load(config_path.read_text()))
        except Exception as exc:
            logger.opt(exception=exc).warning(
                "Failed to read local config for state-sync response"
            )
            return None

        if decoded_config is None:
            return None

        if not isinstance(decoded_config, dict):
            logger.warning(
                "Ignoring non-object config while preparing state-sync response"
            )
            return None

        raw_config = cast(dict[object, object], decoded_config)
        sanitized_config: JsonObject = {
            str(key): copy.deepcopy(value) for key, value in raw_config.items()
        }
        from skulk.store.config import normalized_hf_token

        if normalized_hf_token(sanitized_config.get("hf_token")) is None:
            sanitized_config.pop("hf_token", None)
        sanitized_config.pop("model_trust", None)
        model_store = sanitized_config.get("model_store")
        if self._state_sync_store_http_host is not None and isinstance(
            model_store, dict
        ):
            model_store["store_http_host"] = self._state_sync_store_http_host
        return yaml.safe_dump(
            sanitized_config,
            default_flow_style=False,
            sort_keys=False,
        )

    async def _persist_snapshot(self, force: bool = False) -> None:
        snapshot_idx = self.state.last_event_applied_idx
        if snapshot_idx < 0:
            return
        if not force and (
            snapshot_idx - self._last_snapshot_idx < self._snapshot_event_cadence
        ):
            return
        if snapshot_idx == self._last_snapshot_idx:
            return

        snapshot = StateSnapshot(
            session_id=self.session_id,
            last_event_applied_idx=snapshot_idx,
            state=self.state,
        )
        try:
            self._snapshot_store.write(snapshot)
        except Exception as exc:
            logger.opt(exception=exc).warning("Failed to persist state snapshot")
            return

        # Keep a bounded overlap after the latest durable snapshot so a
        # follower that bootstrapped from a recently served snapshot can still
        # replay the missing tail even if another snapshot is persisted before
        # its replay request arrives.
        keep_from_idx = max(
            snapshot.last_event_applied_idx + 1 - REPLAY_TAIL_RETENTION_EVENTS,
            0,
        )
        self._event_log.compact(keep_from_idx)
        self._last_snapshot_idx = snapshot.last_event_applied_idx
