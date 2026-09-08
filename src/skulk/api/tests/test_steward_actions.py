"""Approval-gated intelligent-fabric basic-action contracts."""

import json
from datetime import datetime, timedelta, timezone
from typing import cast

import anyio
import pytest

from skulk.api.main import API
from skulk.api.steward import StewardHarness, steward_tool_definitions
from skulk.master.main import Master
from skulk.shared.apply import apply
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.tests.conftest import get_pipeline_shard_metadata
from skulk.shared.types.commands import (
    CancelDownload,
    DecideStewardAction,
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
    ProposeStewardAction,
)
from skulk.shared.types.common import CommandId, ModelId, NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    LocalForwarderEvent,
    StateSnapshotHydrated,
    StewardActionProposalChanged,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.steward_actions import (
    StewardActionProposal,
    StewardActionProposalId,
    StewardCancelDownloadAction,
    StewardPlaceModelAction,
    StewardRestartInstanceAction,
    StewardStopInstanceAction,
)
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.shared.types.worker.downloads import DownloadAttemptId, DownloadPending
from skulk.shared.types.worker.instances import (
    InstanceId,
    InstanceMeta,
    MlxRingInstance,
    ShardAssignments,
)
from skulk.shared.types.worker.runners import RunnerId
from skulk.shared.types.worker.shards import PipelineShardMetadata, Sharding
from skulk.utils.channels import channel


def _master_with_channels() -> Master:
    """Initialize actual master bookkeeping with inert test-owned channels."""
    node = NodeId("master")
    event_sender, _ = channel[Event]()
    global_sender, _ = channel[GlobalForwarderEvent]()
    _, local_receiver = channel[LocalForwarderEvent]()
    _, command_receiver = channel[ForwarderCommand]()
    state_sender, state_receiver = channel[StateSyncMessage]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    return Master(
        node, SessionId(master_node_id=node, election_clock=0),
        event_sender=event_sender,
        global_event_sender=global_sender,
        local_event_receiver=local_receiver,
        command_receiver=command_receiver,
        state_sync_sender=state_sender,
        state_sync_receiver=state_receiver,
        download_command_sender=download_sender,
    )


def _cancel_proposal() -> StewardActionProposal:
    now = datetime.now(tz=timezone.utc)
    return StewardActionProposal(
        action=StewardCancelDownloadAction(
            node_id=NodeId("worker"),
            node_name="Worker",
            model_id=ModelId("org/model"),
            attempt_id=DownloadAttemptId("attempt"),
        ),
        rationale="The transfer is stalled.",
        evidence=("No progress for ten minutes.",),
        expected_effect="Stop the active transfer without deleting stored data.",
        created_at=now,
        expires_at=now + timedelta(minutes=10),
    )


def _ordinary_instance() -> MlxRingInstance:
    """Return one minimal ordinary placement for restart lifecycle tests."""
    node_id = NodeId("worker")
    runner_id = RunnerId("runner")
    card = ModelCard(
        model_id=ModelId("org/restart-model"),
        storage_size=Memory.from_gb(8),
        n_layers=4,
        hidden_size=8,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    return MlxRingInstance(
        instance_id=InstanceId("original-instance"),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={
                runner_id: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=4,
                    n_layers=4,
                )
            },
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={node_id: []},
        ephemeral_port=52415,
    )


def _authorize_instance_card(master: Master, instance: MlxRingInstance) -> None:
    """Seed command-ordered catalog truth for a lightweight master fixture."""
    card = next(iter(instance.shard_assignments.runner_to_shard.values())).model_card
    master._ordered_model_cards = {card.model_id: card}  # pyright: ignore[reportPrivateUsage]


def test_action_tools_only_create_proposals() -> None:
    """The model receives proposal verbs, never direct mutating verbs."""
    names = {
        cast("dict[str, object]", item["function"])["name"]
        for item in steward_tool_definitions()
    }

    assert {
        "propose_place_model",
        "propose_stop_model",
        "propose_restart_model",
        "propose_cancel_download",
    } <= names
    assert "place_model" not in names
    assert "stop_model" not in names
    assert "restart_model" not in names


def test_untrusted_steward_tools_are_observation_only() -> None:
    """A chat request without mutation authority cannot expose proposal tools."""
    names = {
        cast("dict[str, object]", item["function"])["name"]
        for item in steward_tool_definitions(include_proposals=False)
    }

    assert "get_cluster_state" in names
    assert not {
        "propose_place_model",
        "propose_stop_model",
        "propose_restart_model",
        "propose_cancel_download",
    } & names


@pytest.mark.asyncio
async def test_untrusted_harness_refuses_hallucinated_proposal_call() -> None:
    """Defense in depth rejects a proposal call absent from the tool schema."""
    harness = StewardHarness(cast("API", object()), proposals_allowed=False)

    result = cast(
        "dict[str, object]",
        json.loads(await harness.execute_tool("propose_stop_model", {})),
    )

    assert "requires operator mutation authority" in str(result["error"])


@pytest.mark.asyncio
async def test_harness_proposal_is_inert_until_separate_decision() -> None:
    """Creating a proposal submits only proposal state and reports no execution."""
    submitted: list[StewardActionProposal] = []

    class _FakeAPI:
        async def submit_steward_action_proposal(
            self, proposal: StewardActionProposal
        ) -> None:
            submitted.append(proposal)

    harness = StewardHarness(cast("API", cast("object", _FakeAPI())))
    result = cast(
        "dict[str, object]",
        json.loads(
            await harness._propose_action(  # pyright: ignore[reportPrivateUsage]
                _cancel_proposal().action,
                {
                    "rationale": "The transfer is stalled.",
                    "evidence": ["No progress for ten minutes."],
                    "expected_effect": "Stop the transfer.",
                },
            )
        ),
    )

    assert len(submitted) == 1
    assert submitted[0].status == "pending"
    assert result["approvalRequired"] is True
    assert result["note"] == "No cluster action has executed."


def test_proposal_event_round_trips_through_replicated_state() -> None:
    """Proposal audit survives JSON snapshots and event replay."""
    proposal = _cancel_proposal()
    state = apply(
        State(),
        IndexedEvent(
            idx=0,
            event=StewardActionProposalChanged(proposal=proposal),
        ),
    )

    restored = State.model_validate_json(state.model_dump_json(by_alias=True))

    assert restored.steward_action_proposals[proposal.proposal_id] == proposal


def test_dispatched_proposal_survives_its_failover_recovery_window() -> None:
    """Audit pruning retains dispatched work until recovery can no longer run."""
    now = datetime.now(tz=timezone.utc)
    template = _cancel_proposal()
    recent_dispatch = template.model_copy(
        update={
            "proposal_id": StewardActionProposalId("recent-dispatch"),
            "status": "dispatched",
            "decided_at": now - timedelta(minutes=6),
            "dispatched_at": now,
            "decided_by": "trusted_fabric_operator",
            "command_id": CommandId("cancel-command"),
        }
    )
    pending = {
        StewardActionProposalId(f"pending-{index}"): template.model_copy(
            update={"proposal_id": StewardActionProposalId(f"pending-{index}")}
        )
        for index in range(127)
    }
    state = State(
        steward_action_proposals={
            **pending,
            recent_dispatch.proposal_id: recent_dispatch,
        }
    )
    rejected = template.model_copy(
        update={
            "proposal_id": StewardActionProposalId("prunable-rejection"),
            "status": "rejected",
            "decided_at": now + timedelta(seconds=1),
            "decided_by": "trusted_fabric_operator",
        }
    )

    within_window = apply(
        state,
        IndexedEvent(
            idx=0,
            event=StewardActionProposalChanged(proposal=rejected),
        ),
    )
    assert recent_dispatch.proposal_id in within_window.steward_action_proposals
    assert rejected.proposal_id not in within_window.steward_action_proposals

    later_pending = template.model_copy(
        update={
            "proposal_id": StewardActionProposalId("later-pending"),
            "created_at": now + timedelta(minutes=6),
            "expires_at": now + timedelta(minutes=16),
        }
    )
    after_window = apply(
        within_window,
        IndexedEvent(
            idx=1,
            event=StewardActionProposalChanged(proposal=later_pending),
        ),
    )
    assert recent_dispatch.proposal_id not in after_window.steward_action_proposals
    assert later_pending.proposal_id in after_window.steward_action_proposals


@pytest.mark.asyncio
async def test_cancel_approval_rejects_a_replacement_download_attempt() -> None:
    """Approval cannot cancel a retry the operator did not review."""
    proposal = _cancel_proposal()
    telemetry_view = TelemetryView()
    telemetry_view.apply(
        NodeTelemetry(
            node_id=NodeId("worker"),
            info=DownloadPending(
                node_id=NodeId("worker"),
                attempt_id=DownloadAttemptId("replacement-attempt"),
                shard_metadata=get_pipeline_shard_metadata(
                    ModelId("org/model"), device_rank=0
                ),
            ),
        )
    )
    download_sender, download_receiver = channel[ForwarderDownloadCommand]()
    master = _master_with_channels()
    master.state = State()
    master._telemetry_view = telemetry_view  # pyright: ignore[reportPrivateUsage]
    master._steward_dispatched_effect_issued = set()  # pyright: ignore[reportPrivateUsage]
    master.download_command_sender = download_sender

    with pytest.raises(ValueError, match="no longer active"):
        await master._execute_approved_steward_action(  # pyright: ignore[reportPrivateUsage]
            proposal
        )
    with anyio.move_on_after(0.01) as dispatch_scope:
        await download_receiver.receive()
    assert dispatch_scope.cancel_called


@pytest.mark.asyncio
async def test_kill_switch_blocks_failover_recovery_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A promoted master fails carried actions closed under the kill switch."""
    now = datetime.now(tz=timezone.utc)
    proposal = _cancel_proposal().model_copy(
        update={
            "status": "dispatched",
            "decided_at": now,
            "dispatched_at": now,
            "decided_by": "trusted_fabric_operator",
            "command_id": CommandId("carried-command"),
        }
    )
    telemetry_view = TelemetryView()
    telemetry_view.apply(
        NodeTelemetry(
            node_id=NodeId("worker"),
            info=DownloadPending(
                node_id=NodeId("worker"),
                attempt_id=DownloadAttemptId("attempt"),
                shard_metadata=get_pipeline_shard_metadata(
                    ModelId("org/model"), device_rank=0
                ),
            ),
        )
    )
    event_sender, event_receiver = channel[Event]()
    download_sender, download_receiver = channel[ForwarderDownloadCommand]()
    master = _master_with_channels()
    master.state = State(
        steward_action_proposals={proposal.proposal_id: proposal}
    )
    master._ordered_steward_proposals = {proposal.proposal_id: proposal}  # pyright: ignore[reportPrivateUsage]
    master._steward_reserved_placements = {}  # pyright: ignore[reportPrivateUsage]
    master._steward_restart_teardown_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_dispatched_effect_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._telemetry_view = telemetry_view  # pyright: ignore[reportPrivateUsage]
    master.event_sender = event_sender
    master.download_command_sender = download_sender
    monkeypatch.setenv("SKULK_FABRIC_CAPABILITIES_DISABLE", "1")

    await master._reconcile_dispatched_steward_actions(now)  # pyright: ignore[reportPrivateUsage]

    changed = await event_receiver.receive()
    assert isinstance(changed, StewardActionProposalChanged)
    assert changed.proposal.status == "failed"
    assert "kill switch" in (changed.proposal.outcome or "")
    with anyio.move_on_after(0.01) as dispatch_scope:
        await download_receiver.receive()
    assert dispatch_scope.cancel_called


@pytest.mark.asyncio
async def test_stop_approval_rejects_replaced_instance_state() -> None:
    """A reused instance id cannot redirect an approval to new placement truth."""
    reviewed = _ordinary_instance()
    replacement = reviewed.model_copy(update={"ephemeral_port": 52416})
    now = datetime.now(tz=timezone.utc)
    proposal = StewardActionProposal(
        action=StewardStopInstanceAction(instance=reviewed),
        rationale="The instance is no longer required.",
        evidence=("No active workload requires the instance.",),
        expected_effect="Stop the reviewed ordinary model instance.",
        created_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    master = _master_with_channels()
    master.state = State(instances={replacement.instance_id: replacement})
    master._ordered_steward_proposals = {proposal.proposal_id: proposal}  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(ValueError, match="intent no longer matches"):
        await master._execute_approved_steward_action(proposal)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate_kind", ["stop", "restart"])
async def test_stop_and_restart_approvals_share_target_reservation(
    candidate_kind: str,
) -> None:
    """An approved lifecycle action prevents a second action on its target."""
    instance = _ordinary_instance()
    now = datetime.now(tz=timezone.utc)
    candidate_action = (
        StewardStopInstanceAction(instance=instance)
        if candidate_kind == "stop"
        else StewardRestartInstanceAction(instance=instance)
    )
    competing_action = (
        StewardRestartInstanceAction(instance=instance)
        if candidate_kind == "stop"
        else StewardStopInstanceAction(instance=instance)
    )
    candidate = StewardActionProposal(
        action=candidate_action,
        rationale="The instance needs lifecycle work.",
        evidence=("The exact instance was reviewed.",),
        expected_effect="Apply the reviewed lifecycle action.",
        created_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    competing = StewardActionProposal(
        action=competing_action,
        rationale="Another lifecycle action was approved first.",
        evidence=("The exact instance was reviewed.",),
        expected_effect="Apply the first lifecycle action.",
        created_at=now,
        expires_at=now + timedelta(minutes=10),
        status="approved",
        decided_at=now,
        decided_by="trusted_fabric_operator",
        command_id=CommandId("reserved-command"),
    )
    master = _master_with_channels()
    master.state = State(instances={instance.instance_id: instance})
    master._ordered_steward_proposals = {  # pyright: ignore[reportPrivateUsage]
        candidate.proposal_id: candidate,
        competing.proposal_id: competing,
    }

    with pytest.raises(ValueError, match="already owns this instance"):
        await master._execute_approved_steward_action(candidate)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_restart_waits_for_teardown_before_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restart dispatches deletion first and resumes from approved audit state."""
    instance = _ordinary_instance()
    now = datetime.now(tz=timezone.utc)
    proposal = StewardActionProposal(
        action=StewardRestartInstanceAction(instance=instance),
        rationale="The runner is degraded.",
        evidence=("Three consecutive probes failed.",),
        expected_effect="Replace the ordinary model instance.",
        created_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    event_sender, event_receiver = channel[Event]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    master = _master_with_channels()
    master.state = State(instances={instance.instance_id: instance})
    master._ordered_steward_proposals = {  # pyright: ignore[reportPrivateUsage]
        proposal.proposal_id: proposal
    }
    master._steward_restart_teardown_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_dispatched_effect_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_reserved_placements = {}  # pyright: ignore[reportPrivateUsage]
    master._recently_freed_bytes = {}  # pyright: ignore[reportPrivateUsage]
    master._telemetry_view = TelemetryView()  # pyright: ignore[reportPrivateUsage]
    master._model_trust_approvals = set()  # pyright: ignore[reportPrivateUsage]
    master._system_id = SystemId("master")  # pyright: ignore[reportPrivateUsage]
    master.event_sender = event_sender
    master.download_command_sender = download_sender
    _authorize_instance_card(master, instance)

    events, command_id, status = (
        await master._execute_approved_steward_action(  # pyright: ignore[reportPrivateUsage]
            proposal
        )
    )

    assert status == "approved"
    assert command_id
    assert events == []
    assert proposal.proposal_id not in master._steward_restart_teardown_issued  # pyright: ignore[reportPrivateUsage]

    approved = proposal.model_copy(
        update={
            "status": "approved",
            "decided_at": now,
            "decided_by": "trusted_fabric_operator",
            "command_id": command_id,
        }
    )
    master.state = State(
        instances={instance.instance_id: instance},
        steward_action_proposals={proposal.proposal_id: approved},
    )
    master._ordered_steward_proposals = {  # pyright: ignore[reportPrivateUsage]
        proposal.proposal_id: approved
    }
    replacement = instance.model_copy(
        update={"instance_id": InstanceId("replacement-instance")}
    )

    def place_after_release(
        _command: object, current_instances: object
    ) -> dict[InstanceId, MlxRingInstance]:
        assert current_instances == {}
        return {replacement.instance_id: replacement}

    monkeypatch.setattr(master, "_place_for_steward_action", place_after_release)
    await master._resume_approved_steward_restarts(  # pyright: ignore[reportPrivateUsage]
        now + timedelta(seconds=30)
    )

    deleted = await event_receiver.receive()
    assert isinstance(deleted, InstanceDeleted)
    master.state = State(
        steward_action_proposals={proposal.proposal_id: approved}
    )
    await master._resume_approved_steward_restarts(  # pyright: ignore[reportPrivateUsage]
        now + timedelta(seconds=31)
    )

    changed = await event_receiver.receive()
    created = await event_receiver.receive()
    assert isinstance(changed, StewardActionProposalChanged)
    assert changed.proposal.status == "dispatched"
    assert changed.proposal.dispatched_at == now + timedelta(seconds=31)
    assert isinstance(created, InstanceCreated)
    assert created.instance.instance_id == replacement.instance_id


@pytest.mark.asyncio
async def test_restart_refuses_changed_card_before_teardown() -> None:
    """Restart preserves the live instance when captured card truth is stale."""
    instance = _ordinary_instance()
    now = datetime.now(tz=timezone.utc)
    proposal = StewardActionProposal(
        action=StewardRestartInstanceAction(instance=instance),
        rationale="The runner is degraded.",
        evidence=("Three consecutive probes failed.",),
        expected_effect="Replace the ordinary model instance.",
        created_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    approved = proposal.model_copy(
        update={
            "status": "approved",
            "decided_at": now,
            "decided_by": "trusted_fabric_operator",
            "command_id": CommandId("teardown-command"),
        }
    )
    event_sender, event_receiver = channel[Event]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    master = _master_with_channels()
    master.state = State(
        instances={instance.instance_id: instance},
        steward_action_proposals={proposal.proposal_id: approved},
    )
    master._ordered_steward_proposals = {proposal.proposal_id: approved}  # pyright: ignore[reportPrivateUsage]
    master._ordered_model_cards = {  # pyright: ignore[reportPrivateUsage]
        instance.shard_assignments.model_id: None
    }
    master._steward_restart_teardown_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_dispatched_effect_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_reserved_placements = {}  # pyright: ignore[reportPrivateUsage]
    master.event_sender = event_sender
    master.download_command_sender = download_sender

    await master._resume_approved_steward_restarts(now)  # pyright: ignore[reportPrivateUsage]

    changed = await event_receiver.receive()
    assert isinstance(changed, StewardActionProposalChanged)
    assert changed.proposal.status == "failed"
    assert "identity no longer matches" in (changed.proposal.outcome or "")
    assert master.state.instances == {instance.instance_id: instance}
    with anyio.move_on_after(0.01) as no_teardown:
        await event_receiver.receive()
    assert no_teardown.cancel_called


@pytest.mark.asyncio
async def test_approved_restart_reissues_teardown_once_after_master_failover() -> None:
    """A promoted master resumes an approval whose delete was not replicated."""
    instance = _ordinary_instance()
    now = datetime.now(tz=timezone.utc)
    proposal = StewardActionProposal(
        action=StewardRestartInstanceAction(instance=instance),
        rationale="The runner is degraded.",
        evidence=("Three consecutive probes failed.",),
        expected_effect="Replace the ordinary model instance.",
        created_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(minutes=9),
        status="approved",
        decided_at=now - timedelta(seconds=30),
        decided_by="trusted_fabric_operator",
    )
    event_sender, event_receiver = channel[Event]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    master = _master_with_channels()
    master.state = State(
        instances={instance.instance_id: instance},
        steward_action_proposals={proposal.proposal_id: proposal},
    )
    master._ordered_steward_proposals = {  # pyright: ignore[reportPrivateUsage]
        proposal.proposal_id: proposal
    }
    master._steward_restart_teardown_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_dispatched_effect_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_reserved_placements = {}  # pyright: ignore[reportPrivateUsage]
    master._recently_freed_bytes = {}  # pyright: ignore[reportPrivateUsage]
    master._telemetry_view = TelemetryView()  # pyright: ignore[reportPrivateUsage]
    master._model_trust_approvals = set()  # pyright: ignore[reportPrivateUsage]
    master._system_id = SystemId("promoted-master")  # pyright: ignore[reportPrivateUsage]
    master.event_sender = event_sender
    master.download_command_sender = download_sender
    _authorize_instance_card(master, instance)

    await master._resume_approved_steward_restarts(now)  # pyright: ignore[reportPrivateUsage]

    deleted = await event_receiver.receive()
    assert isinstance(deleted, InstanceDeleted)
    assert deleted.instance_id == instance.instance_id
    assert proposal.proposal_id in master._steward_restart_teardown_issued  # pyright: ignore[reportPrivateUsage]

    await master._resume_approved_steward_restarts(  # pyright: ignore[reportPrivateUsage]
        now + timedelta(seconds=1)
    )
    with anyio.move_on_after(0.01) as receive_scope:
        await event_receiver.receive()
    assert receive_scope.cancel_called


@pytest.mark.asyncio
async def test_stop_cleanup_waits_for_replicated_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop cannot cancel downloads before its dispatch intent is recoverable."""
    instance = _ordinary_instance()
    model_id = instance.shard_assignments.model_id
    now = datetime.now(tz=timezone.utc)
    proposal = StewardActionProposal(
        action=StewardStopInstanceAction(
            instance=instance,
        ),
        rationale="The instance is no longer required.",
        evidence=("No active workload requires the instance.",),
        expected_effect="Stop the ordinary model instance.",
        created_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    event_sender, event_receiver = channel[Event]()
    download_sender, download_receiver = channel[ForwarderDownloadCommand]()
    telemetry_view = TelemetryView()
    telemetry_view.apply(
        NodeTelemetry(
            node_id=NodeId("worker"),
            info=DownloadPending(
                node_id=NodeId("worker"),
                attempt_id=DownloadAttemptId("attempt"),
                shard_metadata=get_pipeline_shard_metadata(model_id, device_rank=0),
            ),
        )
    )
    master = _master_with_channels()
    master.state = State(instances={instance.instance_id: instance})
    master._ordered_steward_proposals = {proposal.proposal_id: proposal}  # pyright: ignore[reportPrivateUsage]
    master._steward_restart_teardown_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_dispatched_effect_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_reserved_placements = {}  # pyright: ignore[reportPrivateUsage]
    master._recently_freed_bytes = {}  # pyright: ignore[reportPrivateUsage]
    master._telemetry_view = telemetry_view  # pyright: ignore[reportPrivateUsage]
    master._model_trust_approvals = set()  # pyright: ignore[reportPrivateUsage]
    master._system_id = SystemId("master")  # pyright: ignore[reportPrivateUsage]
    master.event_sender = event_sender
    master.download_command_sender = download_sender

    def cleanup_commands(
        _instances: object, _downloads: object
    ) -> list[CancelDownload]:
        return [CancelDownload(target_node_id=NodeId("worker"), model_id=model_id)]

    monkeypatch.setattr(
        "skulk.master.main.cancel_unnecessary_downloads",
        cleanup_commands,
    )

    events, command_id, status = (
        await master._execute_approved_steward_action(proposal)  # pyright: ignore[reportPrivateUsage]
    )
    assert events == []
    assert status == "dispatched"
    with anyio.move_on_after(0.01) as premature_cleanup:
        await download_receiver.receive()
    assert premature_cleanup.cancel_called

    dispatched = proposal.model_copy(
        update={
            "status": "dispatched",
            "decided_at": now,
            "decided_by": "trusted_fabric_operator",
            "command_id": command_id,
        }
    )
    master.state = State(
        instances={instance.instance_id: instance},
        steward_action_proposals={proposal.proposal_id: dispatched},
    )
    master._ordered_steward_proposals = {proposal.proposal_id: dispatched}  # pyright: ignore[reportPrivateUsage]
    cleanup: ForwarderDownloadCommand | None = None
    deleted: Event | None = None
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            master._reconcile_dispatched_steward_actions,  # pyright: ignore[reportPrivateUsage]
            now,
        )
        cleanup = await download_receiver.receive()
        deleted = await event_receiver.receive()
    assert cleanup is not None
    assert deleted is not None
    assert isinstance(cleanup.command, CancelDownload)
    assert isinstance(deleted, InstanceDeleted)


@pytest.mark.asyncio
async def test_dispatched_restart_reissues_exact_replacement_after_failover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A promoted master recovers the action-event dispatch window."""
    original = _ordinary_instance()
    replacement_id = InstanceId("replacement-command")
    now = datetime.now(tz=timezone.utc)
    proposal = StewardActionProposal(
        action=StewardRestartInstanceAction(instance=original),
        rationale="The runner is degraded.",
        evidence=("Three consecutive probes failed.",),
        expected_effect="Replace the ordinary model instance.",
        created_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(minutes=9),
        status="dispatched",
        decided_at=now - timedelta(minutes=4, seconds=59),
        dispatched_at=now - timedelta(seconds=30),
        decided_by="trusted_fabric_operator",
        command_id=CommandId(str(replacement_id)),
    )
    replacement = original.model_copy(update={"instance_id": replacement_id})
    event_sender, event_receiver = channel[Event]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    master = _master_with_channels()
    master.state = State(
        instances={original.instance_id: original},
        steward_action_proposals={proposal.proposal_id: proposal},
    )
    master._ordered_steward_proposals = {  # pyright: ignore[reportPrivateUsage]
        proposal.proposal_id: proposal
    }
    master._steward_restart_teardown_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_dispatched_effect_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_reserved_placements = {}  # pyright: ignore[reportPrivateUsage]
    master._recently_freed_bytes = {}  # pyright: ignore[reportPrivateUsage]
    master._telemetry_view = TelemetryView()  # pyright: ignore[reportPrivateUsage]
    master._model_trust_approvals = set()  # pyright: ignore[reportPrivateUsage]
    master._system_id = SystemId("promoted-master")  # pyright: ignore[reportPrivateUsage]
    master.event_sender = event_sender
    master.download_command_sender = download_sender
    _authorize_instance_card(master, original)

    def place_exact_command(
        command: PlaceInstance, current_instances: object
    ) -> dict[InstanceId, MlxRingInstance]:
        assert command.command_id == proposal.command_id
        assert current_instances == {}
        return {replacement.instance_id: replacement}

    monkeypatch.setattr(master, "_place_for_steward_action", place_exact_command)
    await master._reconcile_dispatched_steward_actions(now)  # pyright: ignore[reportPrivateUsage]

    deleted = await event_receiver.receive()
    assert isinstance(deleted, InstanceDeleted)
    assert deleted.instance_id == original.instance_id
    await master._reconcile_dispatched_steward_actions(  # pyright: ignore[reportPrivateUsage]
        now + timedelta(seconds=1)
    )
    with anyio.move_on_after(0.01) as duplicate_teardown_scope:
        await event_receiver.receive()
    assert duplicate_teardown_scope.cancel_called

    master.state = State(steward_action_proposals={proposal.proposal_id: proposal})
    await master._reconcile_dispatched_steward_actions(  # pyright: ignore[reportPrivateUsage]
        now + timedelta(seconds=2)
    )
    created = await event_receiver.receive()
    assert isinstance(created, InstanceCreated)
    assert created.instance.instance_id == replacement_id
    assert proposal.proposal_id in master._steward_dispatched_effect_issued  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_dispatched_place_and_stop_reissue_after_master_failover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A promoted master recovers missing place and stop transition events."""
    original = _ordinary_instance()
    card = next(iter(original.shard_assignments.runner_to_shard.values())).model_card
    now = datetime.now(tz=timezone.utc)
    place_command_id = CommandId("place-command")
    place_proposal = StewardActionProposal(
        action=StewardPlaceModelAction(
            model_card=card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        ),
        rationale="Capacity is required.",
        evidence=("The requested model has no active instance.",),
        expected_effect="Place the requested model.",
        created_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(minutes=9),
        status="dispatched",
        decided_at=now - timedelta(seconds=30),
        decided_by="trusted_fabric_operator",
        command_id=place_command_id,
    )
    stop_proposal = StewardActionProposal(
        action=StewardStopInstanceAction(
            instance=original,
        ),
        rationale="The instance is no longer required.",
        evidence=("No active workload requires the instance.",),
        expected_effect="Stop the ordinary model instance.",
        created_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(minutes=9),
        status="dispatched",
        decided_at=now - timedelta(seconds=30),
        decided_by="trusted_fabric_operator",
        command_id=CommandId("stop-command"),
    )
    replacement = original.model_copy(
        update={"instance_id": InstanceId(str(place_command_id))}
    )
    event_sender, event_receiver = channel[Event]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    master = _master_with_channels()
    master.state = State(
        instances={original.instance_id: original},
        steward_action_proposals={
            place_proposal.proposal_id: place_proposal,
            stop_proposal.proposal_id: stop_proposal,
        },
    )
    master._ordered_steward_proposals = {  # pyright: ignore[reportPrivateUsage]
        place_proposal.proposal_id: place_proposal,
        stop_proposal.proposal_id: stop_proposal,
    }
    master._steward_restart_teardown_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_dispatched_effect_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_reserved_placements = {}  # pyright: ignore[reportPrivateUsage]
    master._recently_freed_bytes = {}  # pyright: ignore[reportPrivateUsage]
    master._telemetry_view = TelemetryView()  # pyright: ignore[reportPrivateUsage]
    master._model_trust_approvals = set()  # pyright: ignore[reportPrivateUsage]
    master._system_id = SystemId("promoted-master")  # pyright: ignore[reportPrivateUsage]
    master.event_sender = event_sender
    master.download_command_sender = download_sender

    def place_exact_command(
        command: PlaceInstance,
        current_instances: object,
    ) -> dict[InstanceId, MlxRingInstance]:
        assert command.command_id == place_command_id
        assert current_instances == {original.instance_id: original}
        return {
            original.instance_id: original,
            replacement.instance_id: replacement,
        }

    monkeypatch.setattr(master, "_place_for_steward_action", place_exact_command)
    await master._reconcile_dispatched_steward_actions(now)  # pyright: ignore[reportPrivateUsage]

    created = await event_receiver.receive()
    deleted = await event_receiver.receive()
    assert isinstance(created, InstanceCreated)
    assert created.instance.instance_id == replacement.instance_id
    assert isinstance(deleted, InstanceDeleted)
    assert deleted.instance_id == original.instance_id


@pytest.mark.asyncio
async def test_place_approvals_reserve_capacity_before_state_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back-to-back place approvals plan against earlier reserved instances."""
    template = _ordinary_instance()
    card = next(iter(template.shard_assignments.runner_to_shard.values())).model_card
    now = datetime.now(tz=timezone.utc)

    def proposal() -> StewardActionProposal:
        return StewardActionProposal(
            action=StewardPlaceModelAction(
                model_card=card,
                sharding=Sharding.Pipeline,
                instance_meta=InstanceMeta.MlxRing,
                min_nodes=1,
            ),
            rationale="Capacity is required.",
            evidence=("The workload requires another placement.",),
            expected_effect="Place another ordinary model instance.",
            created_at=now,
            expires_at=now + timedelta(minutes=10),
        )

    first = proposal()
    second = proposal()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    master = _master_with_channels()
    master.state = State()
    master._ordered_steward_proposals = {}  # pyright: ignore[reportPrivateUsage]
    master._steward_restart_teardown_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_dispatched_effect_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_reserved_placements = {}  # pyright: ignore[reportPrivateUsage]
    master.download_command_sender = download_sender
    observed_counts: list[int] = []

    def reserve_place(
        command: PlaceInstance,
        current_instances: object,
    ) -> dict[InstanceId, MlxRingInstance]:
        assert isinstance(current_instances, dict)
        typed_current = cast("dict[InstanceId, MlxRingInstance]", current_instances)
        observed_counts.append(len(typed_current))
        placed = template.model_copy(
            update={"instance_id": InstanceId(str(command.command_id))}
        )
        return {**typed_current, placed.instance_id: placed}

    monkeypatch.setattr(master, "_place_for_steward_action", reserve_place)
    first_events, _, _ = await master._execute_approved_steward_action(  # pyright: ignore[reportPrivateUsage]
        first
    )
    second_events, _, _ = await master._execute_approved_steward_action(  # pyright: ignore[reportPrivateUsage]
        second
    )

    assert observed_counts == [0, 1]
    assert len(first_events) == 1
    assert len(second_events) == 1
    assert isinstance(first_events[0], InstanceCreated)
    assert isinstance(second_events[0], InstanceCreated)


@pytest.mark.asyncio
async def test_restart_replacements_reserve_capacity_before_state_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back-to-back restart replacements include earlier reserved capacity."""
    template = _ordinary_instance()
    second_original = template.model_copy(
        update={"instance_id": InstanceId("second-original-instance")}
    )
    now = datetime.now(tz=timezone.utc)

    def proposal(instance: MlxRingInstance) -> StewardActionProposal:
        return StewardActionProposal(
            action=StewardRestartInstanceAction(instance=instance),
            rationale="The runner is degraded.",
            evidence=("Three consecutive probes failed.",),
            expected_effect="Replace the ordinary model instance.",
            created_at=now - timedelta(minutes=1),
            expires_at=now + timedelta(minutes=9),
            status="approved",
            decided_at=now - timedelta(seconds=30),
            decided_by="trusted_fabric_operator",
        )

    first = proposal(template)
    second = proposal(second_original)
    event_sender, _ = channel[Event]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    master = _master_with_channels()
    master.state = State(
        steward_action_proposals={
            first.proposal_id: first,
            second.proposal_id: second,
        }
    )
    master._ordered_steward_proposals = {  # pyright: ignore[reportPrivateUsage]
        first.proposal_id: first,
        second.proposal_id: second,
    }
    master._steward_restart_teardown_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_dispatched_effect_issued = set()  # pyright: ignore[reportPrivateUsage]
    master._steward_reserved_placements = {}  # pyright: ignore[reportPrivateUsage]
    master._recently_freed_bytes = {}  # pyright: ignore[reportPrivateUsage]
    master._telemetry_view = TelemetryView()  # pyright: ignore[reportPrivateUsage]
    master._model_trust_approvals = set()  # pyright: ignore[reportPrivateUsage]
    master._system_id = SystemId("master")  # pyright: ignore[reportPrivateUsage]
    master.event_sender = event_sender
    master.download_command_sender = download_sender
    _authorize_instance_card(master, template)
    observed_counts: list[int] = []

    def reserve_restart(
        command: PlaceInstance,
        current_instances: object,
    ) -> dict[InstanceId, MlxRingInstance]:
        assert isinstance(current_instances, dict)
        typed_current = cast("dict[InstanceId, MlxRingInstance]", current_instances)
        observed_counts.append(len(typed_current))
        replacement = template.model_copy(
            update={"instance_id": InstanceId(str(command.command_id))}
        )
        return {**typed_current, replacement.instance_id: replacement}

    monkeypatch.setattr(master, "_place_for_steward_action", reserve_restart)
    await master._resume_approved_steward_restarts(now)  # pyright: ignore[reportPrivateUsage]

    assert observed_counts == [0, 1]
    assert len(master._steward_reserved_placements) == 2  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_master_approves_a_proposal_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back-to-back approvals cannot dispatch the same action twice."""
    proposal = _cancel_proposal()
    node_id = NodeId("master")
    session_id = SessionId(master_node_id=node_id, election_clock=0)
    global_sender, global_receiver = channel[GlobalForwarderEvent]()
    command_sender, command_receiver = channel[ForwarderCommand]()
    _, local_event_receiver = channel[LocalForwarderEvent]()
    _, state_sync_receiver = channel[StateSyncMessage]()
    state_sync_sender, _ = channel[StateSyncMessage]()
    download_sender, download_receiver = channel[ForwarderDownloadCommand]()
    event_sender, event_receiver = channel[Event]()
    telemetry_view = TelemetryView()
    master = Master(
        node_id,
        session_id,
        event_sender=event_sender,
        global_event_sender=global_sender,
        local_event_receiver=local_event_receiver,
        command_receiver=command_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=download_sender,
        initial_state=State(
            steward_action_proposals={proposal.proposal_id: proposal}
        ),
        telemetry_view=telemetry_view,
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(master.run)
        # Initial state is indexed before commands so failover carries pending
        # proposals into the new master's serialized decision view.
        seed = await global_receiver.receive()
        assert isinstance(seed.event, StateSnapshotHydrated)
        telemetry_view.apply(
            NodeTelemetry(
                node_id=NodeId("worker"),
                info=DownloadPending(
                    node_id=NodeId("worker"),
                    attempt_id=DownloadAttemptId("attempt"),
                    shard_metadata=get_pipeline_shard_metadata(
                        ModelId("org/model"), device_rank=0
                    ),
                ),
            )
        )
        decision = DecideStewardAction(
            proposal_id=proposal.proposal_id,
            approved=True,
            decided_by="trusted_fabric_operator",
        )
        await command_sender.send(
            ForwarderCommand(origin=SystemId("api"), command=decision)
        )
        changed = await event_receiver.receive()
        assert isinstance(changed, StewardActionProposalChanged)
        assert changed.proposal.status == "approved", changed.proposal.outcome
        await master._arm_approved_steward_download_cancellations()  # pyright: ignore[reportPrivateUsage]
        with anyio.move_on_after(0.1) as premature_arm_wait:
            await event_receiver.receive()
        assert premature_arm_wait.cancel_called

        master.state = master.state.model_copy(
            update={
                "steward_action_proposals": {
                    proposal.proposal_id: changed.proposal
                }
            }
        )
        await master._arm_approved_steward_download_cancellations()  # pyright: ignore[reportPrivateUsage]
        armed = await event_receiver.receive()
        assert isinstance(armed, StewardActionProposalChanged)
        assert armed.proposal.status == "dispatched"
        await master._reconcile_dispatched_steward_actions(  # pyright: ignore[reportPrivateUsage]
            datetime.now(tz=timezone.utc)
        )
        with anyio.move_on_after(0.1) as premature_dispatch_wait:
            await download_receiver.receive()
        assert premature_dispatch_wait.cancel_called

        master.state = master.state.model_copy(
            update={
                "steward_action_proposals": {
                    proposal.proposal_id: armed.proposal
                }
            }
        )
        await master._reconcile_dispatched_steward_actions(  # pyright: ignore[reportPrivateUsage]
            datetime.now(tz=timezone.utc)
        )
        dispatched = await download_receiver.receive()
        assert isinstance(dispatched.command, CancelDownload)
        assert dispatched.command.model_id == ModelId("org/model")

        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("api"),
                command=DecideStewardAction(
                    proposal_id=proposal.proposal_id,
                    approved=True,
                    decided_by="trusted_fabric_operator",
                ),
            )
        )
        with anyio.move_on_after(0.1) as duplicate_wait:
            await download_receiver.receive()
        assert duplicate_wait.cancel_called

        blocked = _cancel_proposal()
        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("api"),
                command=ProposeStewardAction(proposal=blocked),
            )
        )
        blocked_pending = await event_receiver.receive()
        assert isinstance(blocked_pending, StewardActionProposalChanged)
        monkeypatch.setenv("SKULK_FABRIC_CAPABILITIES_DISABLE", "1")
        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("api"),
                command=DecideStewardAction(
                    proposal_id=blocked.proposal_id,
                    approved=True,
                    decided_by="trusted_fabric_operator",
                ),
            )
        )
        blocked_result = await event_receiver.receive()
        assert isinstance(blocked_result, StewardActionProposalChanged)
        assert blocked_result.proposal.status == "failed"
        assert "kill switch" in (blocked_result.proposal.outcome or "")
        with anyio.move_on_after(0.1) as blocked_wait:
            await download_receiver.receive()
        assert blocked_wait.cancel_called
        monkeypatch.delenv("SKULK_FABRIC_CAPABILITIES_DISABLE")

        expiring = _cancel_proposal()
        await command_sender.send(
            ForwarderCommand(
                origin=SystemId("api"),
                command=ProposeStewardAction(proposal=expiring),
            )
        )
        proposed = await event_receiver.receive()
        assert isinstance(proposed, StewardActionProposalChanged)
        assert proposed.proposal.status == "pending"
        expiry_events = master._expire_steward_action_proposals(  # pyright: ignore[reportPrivateUsage]
            expiring.expires_at + timedelta(seconds=1)
        )
        assert len(expiry_events) == 1
        expired = expiry_events[0]
        assert isinstance(expired, StewardActionProposalChanged)
        assert expired.proposal.status == "expired"
        assert expired.proposal.decided_by == "fabric_expiry"
        task_group.cancel_scope.cancel()
