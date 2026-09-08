import random
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from copy import deepcopy
from typing import ClassVar, Final, Literal, Sequence, TypeAlias

from skulk.master.placement_utils import (
    Cycle,
    CycleMemoryDiagnostics,
    filter_cycles_by_memory,
    get_llama_rpc_donor_endpoints,
    get_mlx_jaccl_coordinators,
    get_mlx_jaccl_devices_matrix,
    get_mlx_ring_hosts_by_node,
    get_shard_assignments,
    get_shard_assignments_for_llama_rpc,
    get_smallest_cycles,
)
from skulk.shared.backends import (
    EngineType,
    engine_of,
    engine_supports_multi_node,
    platform_compatible_backends,
    resolve_node_backend,
)
from skulk.shared.data_plane_health import zenoh_isolated_nodes
from skulk.shared.models.capabilities import (
    family_predates_in_process_llama_cpp,
    resolve_model_capability_profile,
)
from skulk.shared.models.memory_estimate import (
    KV_CONTEXT_BUDGET_TOKENS,
    backend_offloads_to_vram,
    estimate_shard_footprint,
    instance_context_token_limit,
    shard_fraction_of_model,
)
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    card_serves_speech,
    registry_supported_backends_for_node,
    same_authorized_model_card,
)
from skulk.shared.models.remote_code_approval import (
    remote_code_approval_required,
    remote_code_trust_identity,
)
from skulk.shared.topology import Topology
from skulk.shared.types.commands import (
    CancelDownload,
    CreateInstance,
    DeleteInstance,
    DownloadCommand,
    PlaceInstance,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import (
    Event,
    InstanceCreated,
    InstanceDeleted,
    TaskStatusUpdated,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import (
    MemoryUsage,
    NodeNetworkInfo,
    NodeResources,
)
from skulk.shared.types.tasks import Task, TaskId, TaskStatus
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from skulk.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    LlamaRpcInstance,
    MlxJacclInstance,
    MlxRingInstance,
    instance_meta_of,
)
from skulk.shared.types.worker.runners import ShardAssignments
from skulk.shared.types.worker.shards import Sharding, TensorShardMetadata

# Ring/coordinator/donor listener ports are drawn from a band BELOW every
# OS's default ephemeral range: macOS assigns outgoing-connection local ports
# from 49152-65535 and Linux from 32768-60999, so the band must sit under
# 32768 to be safe on a heterogeneous fleet. Picking listener ports inside
# the ephemeral range made bind collisions a background hazard: any outgoing
# socket on the placement node (store transfers run exactly when placements
# happen) could hold the chosen port, the port is immutable on the instance,
# so every runner retry failed with EADDRINUSE until the crash breaker gave
# up (observed live in the e2e battery: [ring] Couldn't bind socket
# (error: 48) on a node that was mid store-download). In this band the only
# possible collisions are other Skulk listeners, which the caller excludes
# via live-instance ports.
_PLACEMENT_PORT_RANGE: Final = (24000, 31999)


def random_ephemeral_port(
    in_use_ports: AbstractSet[int] | None = None,
) -> int:
    """Pick a listener port for a placement, avoiding known in-use ports.

    ``in_use_ports`` should carry every live instance's ports so concurrent
    placements cannot collide with each other; the band itself keeps OS
    outgoing sockets out of play. Random probing first (cheap, no bias), then
    a deterministic scan under pressure. A truly exhausted band (8k live
    listener ports, beyond any real deployment) raises PlacementError rather
    than knowingly returning an in-use port.
    """
    low, high = _PLACEMENT_PORT_RANGE
    for _ in range(64):
        port = random.randint(low, high)
        if not in_use_ports or port not in in_use_ports:
            return port
    assert in_use_ports is not None  # 64 misses require a non-empty set
    for port in range(low, high + 1):
        if port not in in_use_ports:
            return port
    raise PlacementError(
        f"No free placement listener port in {low}-{high}: "
        f"{len(in_use_ports)} ports are claimed by live instances."
    )


def _listener_ports_in_use(
    instances: Mapping[InstanceId, Instance],
) -> set[int]:
    """Every listener port claimed by live (or just-minted) instances.

    Fed to :func:`random_ephemeral_port` so concurrent placements cannot
    collide with each other inside the reserved placement port band.
    """
    ports: set[int] = set()
    for existing in instances.values():
        if isinstance(existing, MlxRingInstance):
            ports.add(existing.ephemeral_port)
            continue
        endpoints = (
            existing.jaccl_coordinators.values()
            if isinstance(existing, MlxJacclInstance)
            else existing.donor_endpoints.values()
        )
        for endpoint in endpoints:
            _, _, port_text = endpoint.rpartition(":")
            if port_text.isdigit():
                ports.add(int(port_text))
    return ports


def add_instance_to_placements(
    command: CreateInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_vram: Mapping[NodeId, Memory] | None = None,
    unified_memory_gpu_nodes: AbstractSet[NodeId] | None = None,
    approved_remote_code_identities: AbstractSet[str] | None = None,
) -> Mapping[InstanceId, Instance]:
    """Validate and add one caller-specified exact instance placement.

    Args:
        command: Exact instance creation request.
        topology: Current cluster topology.
        current_instances: Existing authoritative placements.
        node_memory: Current node memory observations.
        node_vram: Current discrete-GPU memory observations.
        unified_memory_gpu_nodes: Nodes whose GPU allocations use system RAM.
        approved_remote_code_identities: Deprecated legacy approval set retained
            for compatibility with older call sites.

    Returns:
        Existing placements plus the validated, memory-stamped instance.

    Raises:
        PlacementModelCardIdentityError: If a shard card identifies a model
            other than the assignment alias.
    """
    # TODO: validate against topology

    # Stamp the memory-derived context-admission ceiling, same as the
    # place_instance path (#279 slice 2). A client-supplied exact placement
    # usually omits contextTokenLimit; without this it would get only the
    # card-limit backfill (BaseInstance validator), and since runners now trust
    # the stamped value instead of recomputing from per-node RAM, a prompt that
    # fits the card but not the hosting node could reach MLX instead of a clean
    # context_length_exceeded. instance_context_token_limit falls back to the
    # card's advertised context_length whenever ANY hosting node's ram_total is
    # still unknown, so a transient missing reading yields the card guard rather
    # than None (review catch on #292).
    assignments = command.instance.shard_assignments
    require_instance_model_card_identity(command.instance)
    require_instance_model_code_approval(
        command.instance,
        approved_remote_code_identities,
    )
    fixed_memory_by_node: dict[NodeId, Memory] = {}
    first_shard = next(iter(assignments.runner_to_shard.values()), None)
    if first_shard is not None:
        card = first_shard.model_card
        projector_size = (
            card.vision.projector_size if card.vision is not None else None
        )
        if projector_size is not None:
            projector_memory = Memory.from_bytes(projector_size)
            if isinstance(command.instance, LlamaRpcInstance):
                fixed_memory_by_node[command.instance.driver_node] = projector_memory
            elif len(assignments.node_to_runner) == 1:
                fixed_memory_by_node[next(iter(assignments.node_to_runner))] = (
                    projector_memory
                )
    ceiling = instance_context_token_limit(
        assignments,
        {
            node_id: node_memory[node_id].ram_total
            for node_id in assignments.node_to_runner
            if node_id in node_memory
        },
        node_vram=node_vram,
        unified_memory_gpu_nodes=unified_memory_gpu_nodes,
        fixed_memory_by_node=fixed_memory_by_node,
    )
    requested_limit = command.instance.context_token_limit
    if requested_limit is not None:
        if requested_limit <= 0:
            raise PlacementError("The requested context token limit must be positive")
        # Exact placement callers may intentionally budget a smaller window than
        # the preview maximum. Raising it silently can multiply load-time KV
        # allocation and defeat the caller's resource plan.
        ceiling = requested_limit if ceiling is None else min(ceiling, requested_limit)
    instance = command.instance.model_copy(update={"context_token_limit": ceiling})
    if not isinstance(instance, LlamaRpcInstance):
        for node_id, runner_id in assignments.node_to_runner.items():
            shard = assignments.runner_to_shard[runner_id]
            available = (node_vram or {}).get(node_id)
            fraction = shard_fraction_of_model(shard)
            if not backend_offloads_to_vram(shard.resolved_backend):
                continue
            if available is None or fraction is None:
                raise PlacementError(
                    "GPU memory telemetry and a concrete shard are required for exact placement"
                )
            # A context ceiling alone is not a weights admission check: it can
            # become zero, or fall back to the card when KV geometry is absent.
            # Exact placements must not bypass the already-committed GPU budget.
            footprint = estimate_shard_footprint(
                shard.model_card,
                fraction,
                context_budget=ceiling
                if ceiling is not None
                else KV_CONTEXT_BUDGET_TOKENS,
            )
            if ceiling == 0 or footprint > available:
                raise PlacementError("Insufficient GPU memory for the exact placement")
    return {**current_instances, instance.instance_id: instance}


def _get_node_download_fraction(
    node_id: NodeId,
    model_id: ModelId,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> float:
    """Return the download fraction (0.0–1.0) for a model on a given node."""
    for progress in download_status.get(node_id, []):
        if progress.shard_metadata.model_card.model_id != model_id:
            continue
        match progress:
            case DownloadCompleted():
                return 1.0
            case DownloadOngoing():
                total = progress.download_progress.total.in_bytes
                return (
                    progress.download_progress.downloaded.in_bytes / total
                    if total > 0
                    else 0.0
                )
            case DownloadPending():
                total = progress.total.in_bytes
                return progress.downloaded.in_bytes / total if total > 0 else 0.0
            case DownloadFailed():
                return 0.0
    return 0.0


def _cycle_download_score(
    cycle: Cycle,
    model_id: ModelId,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> float:
    """Sum of download fractions across all nodes in a cycle."""
    return sum(
        _get_node_download_fraction(node_id, model_id, download_status)
        for node_id in cycle
    )


def _cycle_backend_preference_score(
    cycle: Cycle,
    node_resources: Mapping[NodeId, NodeResources],
    backend_preference: Sequence[str],
) -> int:
    """Rank a cycle by how well its nodes satisfy the card's backend preference.

    ``backend_preference`` is an ordered list of backend tags (most preferred
    first). The score is ``len(backend_preference) - i`` for the earliest index ``i``
    whose tag is advertised by every node in the cycle that has a resources
    entry, or ``0`` when no preferred tag is satisfiable (and when there is no
    preference at all). Higher is better, so a cycle that can serve the top
    preference outranks one that can only serve a fallback, which in turn
    outranks one that serves none (graceful degradation by construction).

    This is a SOFT signal: ``compatible_backends`` has already hard-filtered the
    candidates, so every cycle here is runnable; this only orders them. Nodes
    without a resources entry yet (gossip warming up) are not counted against a
    preference, matching the optimistic eligibility default elsewhere.
    """
    for index, tag in enumerate(backend_preference):
        if all(
            tag in node_resources[node_id].backends
            for node_id in cycle
            if node_id in node_resources
        ):
            return len(backend_preference) - index
    return 0


def _card_platform_backends(
    card: ModelCard,
    node_resources: NodeResources | None = None,
) -> frozenset[str]:
    """The card's compatible backends minus current platform limitations.

    Cards declare MODEL truth (what the model and its artifacts can do); which
    of those capabilities our runner implementations can currently exploit is
    PLATFORM truth and lives in code (``platform_compatible_backends``), so a
    platform gap is never encoded on a card. Every placement-side read of
    ``compatible_backends`` goes through this helper so eligibility, the
    common-engine cycle rule, and backend stamping all agree.
    """
    compatible = set(card.placement.compatible_backends)
    if node_resources is not None:
        compatible.update(
            registry_supported_backends_for_node(
                card,
                node_backends=node_resources.backends,
                engine_builds=node_resources.engine_builds,
                hardware_classes=node_resources.hardware_classes,
            )
        )
    profile = resolve_model_capability_profile(card.model_id, model_card=card)
    return platform_compatible_backends(
        frozenset(compatible),
        card_serves_vision=card.vision is not None,
        card_serves_speech=card_serves_speech(card),
        card_has_pinned_projector=(
            card.vision is not None and card.vision.has_pinned_projector
        ),
        card_supports_tool_calling=profile.supports_tool_calling,
        card_vllm_tool_call_parser=(
            card.runtime.vllm_tool_call_parser if card.runtime is not None else None
        ),
        card_family_predates_in_process_binding=(
            family_predates_in_process_llama_cpp(card)
        ),
    )


def _cycle_common_multi_node_engines(
    cycle: Cycle,
    card: ModelCard,
    node_resources: Mapping[NodeId, NodeResources],
) -> set[EngineType]:
    """Multi-node-capable engines advertised by EVERY node in the cycle.

    A multi-node placement runs ONE engine across the whole cycle, so a cycle
    is only viable for a card when some single engine is both (a) present in
    the intersection of the card's ``compatible_backends`` with every node's
    advertised backends and (b) multi-node capable. Checking per node
    independently instead (the pre-#414 behavior) admits a mixed cycle for a
    hybrid card — e.g. an MLX-only Mac plus a llama.cpp-only AMD node, two
    engines that cannot form one ring.

    A node with no resources entry yet (gossip warming up) is treated as
    advertising the ``NodeResources`` default (``{"mlx"}``), matching the
    optimistic eligibility default elsewhere: MLX cards keep placing during
    warmup, while engine-specific multi-node shapes (llama_server RPC) wait
    until the node's real advertisement arrives.
    """
    common: set[EngineType] | None = None
    for node_id in cycle.node_ids:
        resources = node_resources.get(node_id)
        backends = (
            resources.backends if resources is not None else frozenset({"mlx"})
        )
        card_backends = _card_platform_backends(card, resources)
        node_engines: set[EngineType] = {
            engine
            for tag in backends & card_backends
            if (engine := engine_of(tag)) is not None
        }
        common = node_engines if common is None else (common & node_engines)
        if not common:
            return set()
    return {
        engine for engine in (common or set()) if engine_supports_multi_node(engine)
    }


def _is_llama_rpc_cycle(
    cycle: Cycle,
    card: ModelCard,
    node_resources: Mapping[NodeId, NodeResources],
) -> bool:
    """Whether a multi-node cycle would serve this card via the RPC shape (#328).

    True when ``llama_server`` is a common multi-node engine of the cycle and
    ``mlx`` is not: the proven MLX shapes keep priority whenever they are an
    option (in practice the sets never overlap, since no node advertises both).
    Single-node cycles are never RPC (the served engine runs standalone there).
    """
    if len(cycle) <= 1:
        return False
    engines = _cycle_common_multi_node_engines(cycle, card, node_resources)
    return "llama_server" in engines and "mlx" not in engines


def _common_served_accelerator_tags(
    cycle: Cycle,
    card: ModelCard,
    node_resources: Mapping[NodeId, NodeResources],
) -> frozenset[str]:
    """Return exact served accelerator tags shared by every node in a cycle.

    Vision RPC is qualified only when all ranks execute one homogeneous
    llama.cpp accelerator backend. Bare and CPU tags are intentionally absent:
    donors do not execute the image path, but mixed accelerator builds have not
    yet been qualified for multimodal RPC.
    """

    allowed_suffixes = ("-cuda", "-rocm", "-vulkan")
    common: set[str] | None = None
    for node_id in cycle.node_ids:
        resources = node_resources.get(node_id)
        if resources is None:
            return frozenset()
        tags = {
            tag
            for tag in resources.backends & _card_platform_backends(card, resources)
            if tag.startswith("llama_server-") and tag.endswith(allowed_suffixes)
        }
        common = tags if common is None else common & tags
        if not common:
            return frozenset()
    return frozenset(common or set())


PlacementFailureCode: TypeAlias = Literal[
    "no_valid_placement",
    "placement_info_pending",
    "model_code_approval_required",
    "model_card_identity_mismatch",
]


class PlacementError(ValueError):
    """Placement is impossible for the requested command and current state.

    Subclasses ``ValueError`` so existing callers that catch ``ValueError``
    (the API preview endpoint, the master command processor) keep working.
    """

    code: ClassVar[PlacementFailureCode] = "no_valid_placement"


class PlacementInfoPendingError(PlacementError):
    """Placement cannot be judged yet because cluster info is still arriving.

    Covers both phases of the cluster-startup window: connection edges lag
    node identities by a few gossip rounds (enough nodes exist but no
    connected cycle is known yet), and per-node memory info lags the edges
    (a cycle exists but a node's first NodeGatheredInfo event has not
    arrived). Both are retry-shortly conditions — distinct from a real
    topology gap or memory shortfall so callers can wait instead of
    reporting a false error.
    """

    code: ClassVar[PlacementFailureCode] = "placement_info_pending"


class PlacementModelCodeApprovalError(PlacementError):
    """Legacy error retained for placement wire compatibility."""

    code: ClassVar[PlacementFailureCode] = "model_code_approval_required"


class PlacementModelCardIdentityError(PlacementError):
    """A caller-specified shard embeds a card for a different model alias."""

    code: ClassVar[PlacementFailureCode] = "model_card_identity_mismatch"


def require_instance_model_card_identity(
    instance: Instance,
    authorized_card: ModelCard | None = None,
) -> None:
    """Require embedded shard cards to match the assignment and catalog truth.

    Args:
        instance: Caller-specified exact placement containing shard cards.
        authorized_card: Effective card loaded from the authorized local catalog.
            When supplied, every embedded card must match it exactly.

    Raises:
        PlacementModelCardIdentityError: If a shard identifies another model or
            differs from the authorized catalog card.
    """

    expected_model_id = instance.shard_assignments.model_id
    mismatched_model_ids = sorted(
        {
            str(shard.model_card.model_id)
            for shard in instance.shard_assignments.runner_to_shard.values()
            if shard.model_card.model_id != expected_model_id
        }
    )
    if mismatched_model_ids:
        mismatches = ", ".join(mismatched_model_ids)
        raise PlacementModelCardIdentityError(
            "Exact placement model-card identity mismatch: assignment model "
            f"{expected_model_id} contains shard card(s) for {mismatches}."
        )
    if authorized_card is None:
        return
    if authorized_card.model_id != expected_model_id:
        raise PlacementModelCardIdentityError(
            "Exact placement catalog identity mismatch: assignment model "
            f"{expected_model_id} resolved to {authorized_card.model_id}."
        )
    if any(
        not same_authorized_model_card(shard.model_card, authorized_card)
        for shard in instance.shard_assignments.runner_to_shard.values()
    ):
        raise PlacementModelCardIdentityError(
            "Exact placement embeds model-card content that does not match the "
            f"authorized catalog card for {expected_model_id}. Recompute the "
            "placement from the current catalog before retrying."
        )


def _raise_model_code_approval_error(card: ModelCard) -> None:
    """Raise the stable actionable placement error for one exact card."""

    trust_identity = remote_code_trust_identity(card)
    raise PlacementModelCodeApprovalError(
        "The cluster operator has not approved repository code for immutable "
        f"model-card identity {trust_identity}. Approve this model in Settings "
        "before placing it."
    )


def require_instance_model_code_approval(
    instance: Instance,
    approved_remote_code_identities: AbstractSet[str] | None = None,
) -> None:
    """Apply the legacy model-approval hook to every embedded card.

    Current publication/addition authorization makes this a no-op for valid
    cards. The hook and error type remain during rolling compatibility with
    older callers and response schemas.

    Args:
        instance: Caller-specified exact placement containing shard cards.
        approved_remote_code_identities: Deprecated legacy approval set.

    Raises:
        PlacementModelCodeApprovalError: If any distinct shard card is blocked.
    """

    checked_identities: set[str] = set()
    for shard in instance.shard_assignments.runner_to_shard.values():
        card = shard.model_card
        trust_identity = remote_code_trust_identity(card)
        if trust_identity in checked_identities:
            continue
        checked_identities.add(trust_identity)
        if remote_code_approval_required(card, approved_remote_code_identities):
            _raise_model_code_approval_error(card)


def place_instance(
    command: PlaceInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_network: Mapping[NodeId, NodeNetworkInfo],
    required_nodes: set[NodeId] | None = None,
    download_status: Mapping[NodeId, Sequence[DownloadProgress]] | None = None,
    excluded_nodes: set[NodeId] | None = None,
    node_resources: Mapping[NodeId, NodeResources] | None = None,
    node_vram: Mapping[NodeId, Memory] | None = None,
    unified_memory_gpu_nodes: AbstractSet[NodeId] | None = None,
    stamped_exclusions: set[NodeId] | None = None,
    approved_remote_code_identities: AbstractSet[str] | None = None,
) -> dict[InstanceId, Instance]:
    if remote_code_approval_required(
        command.model_card, approved_remote_code_identities
    ):
        _raise_model_code_approval_error(command.model_card)

    cycles = topology.get_cycles()
    candidate_cycles = list(filter(lambda it: len(it) >= command.min_nodes, cycles))
    if not candidate_cycles:
        known_nodes = sum(1 for _ in topology.list_nodes())
        if known_nodes >= command.min_nodes:
            # Enough nodes exist for this placement — they just aren't
            # bidirectionally connected (yet). Right after cluster formation
            # the connection edges lag the node identities by a few gossip
            # rounds, so this is usually a retry-shortly condition rather
            # than a network problem.
            raise PlacementInfoPendingError(
                f"The topology knows {known_nodes} node(s) but none form a "
                f"connected cycle of at least {command.min_nodes} node(s) yet. "
                "Connection info may still be gossiping (typical right after "
                "cluster formation) — retry shortly. If this persists, check "
                "network connectivity between the nodes."
            )
        raise PlacementError(
            f"The topology has only {known_nodes} node(s); a placement with "
            f"min_nodes={command.min_nodes} is impossible. Multi-node "
            "placement requires bidirectional connectivity between the nodes."
        )

    # Drop any cycle that touches an operator-excluded node. Exclusion is the
    # operator's "don't pick this node for new placements" signal — already
    # placed instances on the excluded node remain unaffected (they live in
    # current_instances, which this function never mutates), but the planner
    # treats the excluded node as if it were absent from the topology when
    # scoring fresh placements.
    if excluded_nodes:
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if not (set(cycle.node_ids) & excluded_nodes)
        ]
        if not candidate_cycles:
            raise PlacementError(
                f"All cycles of at least {command.min_nodes} node(s) touch an "
                f"excluded node ({len(excluded_nodes)} node(s) excluded). "
                "Check the excluded_nodes list — node IDs change when a "
                "cluster session restarts."
            )

    # A healthy control-plane topology does not prove that generated tokens can
    # return from every candidate node. When telemetry positively reports a
    # live Zenoh node with zero peers, exclude every cycle touching it so the
    # planner cannot create a runner that loads successfully but never delivers
    # output. Unknown peer counts remain eligible during startup.
    if node_resources:
        isolated_nodes = zenoh_isolated_nodes(
            live_nodes=set(topology.list_nodes()),
            node_resources=node_resources,
        )
        if isolated_nodes:
            candidate_cycles = [
                cycle
                for cycle in candidate_cycles
                if not (set(cycle.node_ids) & isolated_nodes)
            ]
            if not candidate_cycles:
                raise PlacementError(
                    f"All cycles of at least {command.min_nodes} node(s) touch "
                    "a node whose Zenoh inference data plane is isolated. "
                    "Restore data-plane connectivity or exclude the affected "
                    "node before placing the model."
                )

    # Hard-filter on node participation and backend compatibility (#149/#845).
    # Repository-code trust is an operator decision for the exact model-card
    # identity and was checked once above; it is deliberately not a node axis.
    # A node is ineligible for an inference shard of THIS model when it
    # declares a non-``full`` participation role (e.g. a ``management`` /
    # edge node that serves the API but never joins a ring), or when its
    # advertised backends do not intersect the card's compatible_backends.
    # Nodes with no resources entry yet (gossip still warming up) retain the
    # legacy optimistic behavior only when the card has a static projection.
    # A matrix-only card needs exact build/hardware evidence and therefore waits
    # rather than falling through to an unproven default engine.
    resolved_node_resources = node_resources or {}
    matrix_only = not command.model_card.placement.compatible_backends
    pending_matrix_nodes: set[NodeId] = set()
    ineligible_nodes: set[NodeId] = set()
    compatible_backend_summary: set[str] = set()
    for node_id in topology.list_nodes():
        resources = resolved_node_resources.get(node_id)
        if resources is None:
            if matrix_only:
                pending_matrix_nodes.add(node_id)
            continue
        compatible_backends = _card_platform_backends(command.model_card, resources)
        compatible_backend_summary.update(compatible_backends)
        if resources.participation != "full" or not (
            resources.backends & compatible_backends
        ):
            ineligible_nodes.add(node_id)
    excluded_for_compatibility = ineligible_nodes | pending_matrix_nodes
    if excluded_for_compatibility:
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if not (set(cycle.node_ids) & excluded_for_compatibility)
        ]
        if not candidate_cycles:
            if pending_matrix_nodes:
                raise PlacementInfoPendingError(
                    "Exact engine-build and hardware info has not been gossiped "
                    "for a matrix-only model. Retry shortly."
                )
            raise PlacementError(
                f"All cycles of at least {command.min_nodes} node(s) touch a "
                f"node ineligible for this model: either a non-participating "
                f"(management/edge) node or one whose exact backend support does "
                f"not match ({sorted(compatible_backend_summary)})."
            )

    # Common-engine cycle rule. A multi-node placement runs ONE engine across
    # the whole cycle, so a multi-node cycle is admissible only when some
    # single engine is advertised by every node (within the card's
    # compatible_backends) AND that engine can shard across nodes
    # (`engine_supports_multi_node`: mlx via ring/jaccl, llama_server via the
    # RPC driver+donors shape, #328). This subsumes two prior checks in one
    # place: the single-node engine guard (a card whose only engine is the
    # in-process llama.cpp gets no multi-node cycles, since llama_cpp is never
    # multi-node capable) and the #414 mixed-engine hole (a hybrid card no
    # longer admits a cycle mixing an MLX-only node with a llama.cpp-only
    # node, which the per-node any() check used to allow). Single-node cycles
    # are untouched -- the per-node participation/backend filter above already
    # covers them. Cards with no declared backends (legacy) skip the rule.
    backend_constrained = bool(command.model_card.placement.compatible_backends) or (
        command.model_card.registry_card_id is not None
    )
    if backend_constrained:
        multi_node_candidate_cycles = [
            cycle for cycle in candidate_cycles if len(cycle) > 1
        ]
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if len(cycle) == 1
            or _cycle_common_multi_node_engines(
                cycle, command.model_card, resolved_node_resources
            )
        ]
        if not candidate_cycles:
            # A node with no NodeResources entry yet defaults to mlx in the
            # common-engine rule, so a served-only cycle evaluated during the
            # telemetry warm-up window is dropped for the WRONG reason. When
            # backends telemetry was provided but is missing for multi-node
            # candidates, surface the retry-shortly signal instead of a hard
            # error the caller would treat as terminal (observed live: a
            # pooled placement right after fleet bring-up 400s once, then
            # succeeds on retry).
            if node_resources is not None:
                resources_pending_nodes = {
                    node_id
                    for cycle in multi_node_candidate_cycles
                    for node_id in cycle.node_ids
                    if node_id not in resolved_node_resources
                }
                if resources_pending_nodes:
                    raise PlacementInfoPendingError(
                        "Backend info has not been gossiped yet for node(s) "
                        f"[{', '.join(str(n) for n in sorted(resources_pending_nodes))}] "
                        "— the cluster may still be starting up. Retry shortly."
                    )
            raise PlacementError(
                "No candidate cycle can serve this model: multi-node placement "
                "requires one engine common to every node in the cycle that "
                "supports sharding (MLX, or llama_server via RPC), and no "
                "single-node cycle is available/eligible. Place it on one "
                "node, or free a node that advertises a compatible backend."
            )

    # The RPC shape is Pipeline-shaped by construction (llama.cpp layer-splits
    # across the pooled devices); a Tensor request cannot be honored on an RPC
    # cycle, so those cycles drop out of a Tensor placement's candidates.
    if command.sharding == Sharding.Tensor:
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if not _is_llama_rpc_cycle(
                cycle, command.model_card, resolved_node_resources
            )
        ]
        if not candidate_cycles:
            raise PlacementError(
                "Requested Tensor sharding, but every candidate cycle would "
                "serve this model via multi-node llama.cpp (RPC), which is "
                "pipeline-only."
            )

    pinned_projector_size = (
        command.model_card.vision.projector_size
        if command.model_card.vision is not None
        else None
    )
    projector_memory = (
        Memory.from_bytes(pinned_projector_size)
        if pinned_projector_size is not None
        else Memory()
    )
    if projector_memory.in_bytes > 0:
        vision_rpc_candidates = [
            cycle
            for cycle in candidate_cycles
            if _is_llama_rpc_cycle(
                cycle, command.model_card, resolved_node_resources
            )
        ]
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if cycle not in vision_rpc_candidates
            or _common_served_accelerator_tags(
                cycle, command.model_card, resolved_node_resources
            )
        ]
        if not candidate_cycles:
            raise PlacementError(
                "Served vision RPC requires one exact llama_server accelerator "
                "backend tag (CUDA, ROCm, or Vulkan) shared by every node in "
                "the cycle. Mixed-backend multimodal RPC is not qualified."
            )

    # Filter to cycles containing all required nodes (subset matching)
    if required_nodes:
        candidate_cycles = [
            cycle
            for cycle in candidate_cycles
            if required_nodes.issubset(cycle.node_ids)
        ]
        if not candidate_cycles:
            raise PlacementError(
                "No candidate cycle contains all required nodes "
                f"[{', '.join(str(n) for n in required_nodes)}]."
            )
    # Memory admission. RPC cycles (multi-node llama.cpp) admit against the
    # same UMA-aware usable-GPU figure as single-node placements: on a
    # unified-memory APU the BIOS VRAM carve is not an allocation boundary
    # (the GPU maps system RAM through GTT), so a carve-only figure refuses
    # placements that load and serve fine. Proven live on the Strix pair:
    # pooled gpt-oss-120b, refused by the carve figure, loaded 40.6G/22.8G
    # and served at 44 tok/s once admitted against unified memory. RPC still
    # goes through its own filter call because its missing-telemetry
    # semantics differ (see below).
    standard_candidates = [
        cycle
        for cycle in candidate_cycles
        if not _is_llama_rpc_cycle(
            cycle, command.model_card, resolved_node_resources
        )
    ]
    rpc_candidates = [
        cycle
        for cycle in candidate_cycles
        if _is_llama_rpc_cycle(cycle, command.model_card, resolved_node_resources)
    ]
    cycles_with_sufficient_memory: list[Cycle] = []
    memory_diagnostics = CycleMemoryDiagnostics()
    for standard_cycle in standard_candidates:
        standard_fixed = (
            {standard_cycle.node_ids[0]: projector_memory}
            if len(standard_cycle) == 1 and projector_memory.in_bytes > 0
            else {}
        )
        standard_fit, standard_diagnostics = filter_cycles_by_memory(
            [standard_cycle],
            node_memory,
            command.model_card,
            command.sharding,
            node_vram=node_vram,
            fixed_memory_by_node=standard_fixed,
        )
        cycles_with_sufficient_memory.extend(standard_fit)
        memory_diagnostics.pending_info_node_ids.extend(
            node_id
            for node_id in standard_diagnostics.pending_info_node_ids
            if node_id not in memory_diagnostics.pending_info_node_ids
        )
        memory_diagnostics.rejection_reasons.extend(
            standard_diagnostics.rejection_reasons
        )
    rpc_driver_by_cycle: dict[tuple[NodeId, ...], NodeId] = {}
    resolved_download_status = download_status or {}
    if rpc_candidates:
        rpc_vram_map: Mapping[NodeId, Memory] = node_vram or {}
        # A cycle node missing from the usable-GPU map (telemetry warm-up:
        # NodeResources arrived, node_system's accelerator reading not yet)
        # must surface as info-pending, NOT fall back to system RAM inside
        # the memory filter — that would size the split off the wrong figure,
        # and the RPC path deliberately bypasses the worker's local fit guard.
        vram_pending_nodes = {
            node_id
            for cycle in rpc_candidates
            for node_id in cycle.node_ids
            if node_id not in rpc_vram_map
        }
        vram_known_rpc_candidates = [
            cycle
            for cycle in rpc_candidates
            if all(node_id in rpc_vram_map for node_id in cycle.node_ids)
        ]
        for rpc_cycle in vram_known_rpc_candidates:
            driver = max(
                rpc_cycle.node_ids,
                key=lambda node_id: (
                    max(
                        0,
                        rpc_vram_map[node_id].in_bytes - projector_memory.in_bytes,
                    ),
                    _get_node_download_fraction(
                        node_id,
                        command.model_card.model_id,
                        resolved_download_status,
                    ),
                ),
            )
            rpc_fixed = (
                {driver: projector_memory}
                if projector_memory.in_bytes > 0
                else {}
            )
            rpc_fit, rpc_diagnostics = filter_cycles_by_memory(
                [rpc_cycle],
                node_memory,
                command.model_card,
                Sharding.Pipeline,
                node_vram=rpc_vram_map,
                exact_pipeline_layers=False,
                fixed_memory_by_node=rpc_fixed,
            )
            cycles_with_sufficient_memory.extend(rpc_fit)
            if rpc_fit:
                rpc_driver_by_cycle[tuple(rpc_cycle.node_ids)] = driver
            memory_diagnostics.pending_info_node_ids.extend(
                node_id
                for node_id in rpc_diagnostics.pending_info_node_ids
                if node_id not in memory_diagnostics.pending_info_node_ids
            )
            memory_diagnostics.rejection_reasons.extend(
                rpc_diagnostics.rejection_reasons
            )
        memory_diagnostics.pending_info_node_ids.extend(
            node_id
            for node_id in sorted(vram_pending_nodes)
            if node_id not in memory_diagnostics.pending_info_node_ids
        )
    if len(cycles_with_sufficient_memory) == 0:
        if memory_diagnostics.pending_info_node_ids:
            raise PlacementInfoPendingError(
                "Memory info has not been gathered yet for node(s) "
                f"[{', '.join(str(n) for n in memory_diagnostics.pending_info_node_ids)}] "
                "— the cluster may still be starting up. Retry shortly."
            )
        detail = (
            "; ".join(memory_diagnostics.rejection_reasons)
            if memory_diagnostics.rejection_reasons
            else "no candidate cycles to evaluate"
        )
        raise PlacementError(
            f"No candidate cycle fits {command.model_card.model_id} "
            f"({command.model_card.storage_size.in_gb:.1f}GB of weights): {detail}"
        )

    if command.sharding == Sharding.Tensor:
        if not command.model_card.supports_tensor:
            raise PlacementError(
                f"Requested Tensor sharding but this model does not support tensor parallelism: {command.model_card.model_id}"
            )
        # TODO: the condition here for tensor parallel is not correct, but it works good enough for now.
        kv_heads = command.model_card.num_key_value_heads
        cycles_with_sufficient_memory = [
            cycle
            for cycle in cycles_with_sufficient_memory
            if command.model_card.hidden_size % len(cycle) == 0
            and (kv_heads is None or kv_heads % len(cycle) == 0)
        ]
        if not cycles_with_sufficient_memory:
            raise PlacementError(
                f"No tensor sharding found for model with "
                f"hidden_size={command.model_card.hidden_size}"
                f"{f', num_key_value_heads={kv_heads}' if kv_heads is not None else ''}"
                f" across candidate cycles"
            )
    if command.sharding == Sharding.Pipeline and command.model_card.model_id == ModelId(
        "mlx-community/DeepSeek-V3.1-8bit"
    ):
        raise PlacementError(
            "Pipeline parallelism is not supported for DeepSeek V3.1 (8-bit)"
        )

    smallest_cycles = get_smallest_cycles(cycles_with_sufficient_memory)

    smallest_rdma_cycles = [
        cycle for cycle in smallest_cycles if topology.is_rdma_cycle(cycle)
    ]

    if command.instance_meta == InstanceMeta.MlxJaccl:
        if not smallest_rdma_cycles:
            raise PlacementError(
                "Requested RDMA (MlxJaccl) but no RDMA-connected cycles available"
            )
        smallest_cycles = smallest_rdma_cycles

    cycles_with_leaf_nodes: list[Cycle] = [
        cycle
        for cycle in smallest_cycles
        if any(topology.node_is_leaf(node_id) for node_id in cycle)
    ]

    candidate_cycles = (
        cycles_with_leaf_nodes if cycles_with_leaf_nodes != [] else smallest_cycles
    )

    # Prefer a cycle that can serve the model's preferred backend (soft, ordered;
    # compatible_backends has already hard-filtered, so this only ranks). This
    # dominates the download/memory tie-breakers because serving a model on its
    # faster backend is the whole point of the preference; among cycles with the
    # same preference rank, download locality then free memory still decide.
    backend_preference = command.model_card.placement.backend_preference
    selected_cycle = max(
        candidate_cycles,
        key=lambda cycle: (
            _cycle_backend_preference_score(
                cycle, resolved_node_resources, backend_preference
            ),
            _cycle_download_score(
                cycle, command.model_card.model_id, resolved_download_status
            ),
            sum(
                (node_memory[node_id].ram_available for node_id in cycle),
                start=Memory(),
            ),
        ),
    )

    # Single-node: force Pipeline/Ring (Tensor and Jaccl require multi-node)
    if len(selected_cycle) == 1:
        command.instance_meta = InstanceMeta.MlxRing
        command.sharding = Sharding.Pipeline

    # The selected cycle's engine decides the instance shape. A multi-node
    # GGUF cycle (llama_server common, mlx not) resolves to the RPC
    # driver-plus-donors shape regardless of the requested instance_meta,
    # mirroring the single-node coercion above: instance_meta is an MLX-shape
    # vocabulary and the served engine has exactly one multi-node form.
    selected_is_rpc = _is_llama_rpc_cycle(
        selected_cycle, command.model_card, resolved_node_resources
    )
    if command.instance_meta == InstanceMeta.LlamaRpc and not selected_is_rpc:
        raise PlacementError(
            "Requested a multi-node llama.cpp (RPC) placement, but the "
            "selected cycle does not serve this model via llama_server on "
            "every node."
        )

    driver_node: NodeId | None = None
    selected_rpc_backend: str | None = None
    if selected_is_rpc:
        # Driver = the node with the largest usable VRAM (most layers land
        # locally, minimizing cross-network boundaries), tie-broken by which
        # node already has the model on disk (only the driver reads the GGUF).
        driver_node = rpc_driver_by_cycle.get(tuple(selected_cycle.node_ids))
        if driver_node is None:
            raise PlacementError(
                "The selected RPC cycle has no memory-qualified driver. Retry "
                "after accelerator telemetry converges."
            )
        common_tags = _common_served_accelerator_tags(
            selected_cycle,
            command.model_card,
            resolved_node_resources,
        )
        if projector_memory.in_bytes > 0:
            selected_rpc_backend = resolve_node_backend(
                common_tags,
                backend_preference,
                common_tags,
            )
        shard_assignments = get_shard_assignments_for_llama_rpc(
            command.model_card, selected_cycle, driver_node
        )
    else:
        shard_assignments = get_shard_assignments(
            command.model_card, selected_cycle, command.sharding, node_memory, node_vram
        )

    # Persist the per-node resolved backend tag on each shard (#330). The worker
    # then dispatches its runner from this replicated value instead of
    # re-probing its own backends at spawn time, which also lets a card resolve
    # to different engines per node on a heterogeneous cycle. Resolution uses the
    # same resolve_node_backend the worker would, against the master's telemetry
    # view of each node's advertised backends. A node absent from node_resources
    # leaves resolved_backend=None, and the worker falls back to its local probe.
    # An RPC placement restricts resolution to the llama_server tags: the whole
    # cycle runs the served RPC machinery by construction, and a card that also
    # allows in-process llama_cpp (with it earlier in preference or alphabetical
    # order) must not stamp the driver with a tag that would dispatch the
    # single-node in-process runner.
    stamped_runner_to_shard = dict(shard_assignments.runner_to_shard)
    for node_id, runner_id in shard_assignments.node_to_runner.items():
        resources = resolved_node_resources.get(node_id)
        compatible_backends = _card_platform_backends(command.model_card, resources)
        if selected_is_rpc:
            compatible_backends = (
                frozenset({selected_rpc_backend})
                if selected_rpc_backend is not None
                else frozenset(
                    tag
                    for tag in compatible_backends
                    if engine_of(tag) == "llama_server"
                )
            )
        resolved_backend = (
            resolve_node_backend(
                compatible_backends, backend_preference, resources.backends
            )
            if resources is not None
            else None
        )
        if resolved_backend is not None:
            shard = stamped_runner_to_shard[runner_id]
            stamped_runner_to_shard[runner_id] = shard.model_copy(
                update={"resolved_backend": resolved_backend}
            )
    shard_assignments = ShardAssignments(
        model_id=shard_assignments.model_id,
        runner_to_shard=stamped_runner_to_shard,
        node_to_runner=shard_assignments.node_to_runner,
    )

    # Stamp the context-admission ceiling into the placement decision (#279
    # slice 2). Computed once here from the hosting nodes' static ram_total, so
    # every rank reads the identical value off replicated state rather than
    # recomputing from the (now telemetry-plane, last-write-wins) node memory.
    context_token_limit = instance_context_token_limit(
        shard_assignments,
        {
            node_id: node_memory[node_id].ram_total
            for node_id in selected_cycle.node_ids
        },
        node_vram=node_vram,
        unified_memory_gpu_nodes=unified_memory_gpu_nodes,
        fixed_memory_by_node=(
            {driver_node: projector_memory}
            if driver_node is not None and projector_memory.in_bytes > 0
            else (
                {selected_cycle.node_ids[0]: projector_memory}
                if len(selected_cycle) == 1 and projector_memory.in_bytes > 0
                else {}
            )
        ),
    )

    cycle_digraph: Topology = topology.get_subgraph_from_nodes(selected_cycle.node_ids)

    # The accepted command identity is also the resulting placement identity.
    # This lets API clients correlate an acknowledgement with exactly one
    # runtime without guessing from model name or observation order. Repair
    # commands receive fresh command IDs, so replacement placements remain
    # independently identifiable.
    instance_id = InstanceId(str(command.command_id))
    # Persist the CALLER'S per-placement exclusions on the instance (#658):
    # repair re-placements reconstruct intent from the instance, and without
    # this record they widened eligibility back to the full topology. Repair
    # callers pass ``stamped_exclusions`` (the original intent) separately
    # from ``excluded_nodes`` (intent plus their own transiently failed
    # nodes), so one transient failure is not laundered into permanent
    # operator intent across repair generations (PR #667 review). Sorted so
    # the replicated placement event is deterministic.
    stamp_source = (
        stamped_exclusions if stamped_exclusions is not None else excluded_nodes
    )
    stamped_exclusions_list = sorted(stamp_source) if stamp_source else []
    target_instances = dict(deepcopy(current_instances))

    if selected_is_rpc:
        assert driver_node is not None
        # Donor endpoints are chosen once here and stamped on the instance:
        # the donor binds exactly this address and the driver dials it, so the
        # two sides can never disagree. Address selection reuses the ring's
        # observed-connection prioritiser (Thunderbolt first, VPN last, #265).
        used_ports = _listener_ports_in_use(target_instances)
        donor_ports: dict[NodeId, int] = {}
        for donor_node_id in selected_cycle.node_ids:
            if donor_node_id == driver_node:
                continue
            picked = random_ephemeral_port(used_ports)
            used_ports.add(picked)
            donor_ports[donor_node_id] = picked
        donor_endpoints = get_llama_rpc_donor_endpoints(
            selected_cycle=selected_cycle,
            driver_node=driver_node,
            cycle_digraph=cycle_digraph,
            node_network=node_network,
            donor_ports=donor_ports,
        )
        target_instances[instance_id] = LlamaRpcInstance(
            instance_id=instance_id,
            shard_assignments=shard_assignments,
            context_token_limit=context_token_limit,
            excluded_nodes=stamped_exclusions_list,
            system_role=command.system_role,
            driver_node=driver_node,
            donor_endpoints=donor_endpoints,
        )
        return target_instances

    match command.instance_meta:
        case InstanceMeta.LlamaRpc:
            # Unreachable: an explicit LlamaRpc request either resolved to the
            # RPC shape (returned above) or raised when the cycle cannot serve
            # it. Kept for match exhaustiveness.
            raise PlacementError(
                "Requested a multi-node llama.cpp (RPC) placement, but the "
                "selected cycle does not serve this model via llama_server."
            )
        case InstanceMeta.MlxJaccl:
            # TODO(evan): shard assignments should contain information about ranks, this is ugly
            def get_device_rank(node_id: NodeId) -> int:
                runner_id = shard_assignments.node_to_runner[node_id]
                shard_metadata = shard_assignments.runner_to_shard.get(runner_id)
                assert shard_metadata is not None
                return shard_metadata.device_rank

            zero_node_ids = [
                node_id
                for node_id in selected_cycle.node_ids
                if get_device_rank(node_id) == 0
            ]
            assert len(zero_node_ids) == 1
            coordinator_node_id = zero_node_ids[0]

            mlx_jaccl_devices = get_mlx_jaccl_devices_matrix(
                [node_id for node_id in selected_cycle],
                cycle_digraph,
            )
            mlx_jaccl_coordinators = get_mlx_jaccl_coordinators(
                coordinator=coordinator_node_id,
                coordinator_port=random_ephemeral_port(
                    _listener_ports_in_use(target_instances)
                ),
                cycle_digraph=cycle_digraph,
                node_network=node_network,
            )
            target_instances[instance_id] = MlxJacclInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                context_token_limit=context_token_limit,
                excluded_nodes=stamped_exclusions_list,
                system_role=command.system_role,
                jaccl_devices=mlx_jaccl_devices,
                jaccl_coordinators=mlx_jaccl_coordinators,
            )
        case InstanceMeta.MlxRing:
            ephemeral_port = random_ephemeral_port(
                _listener_ports_in_use(target_instances)
            )
            hosts_by_node = get_mlx_ring_hosts_by_node(
                selected_cycle=selected_cycle,
                cycle_digraph=cycle_digraph,
                ephemeral_port=ephemeral_port,
                node_network=node_network,
            )
            target_instances[instance_id] = MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                context_token_limit=context_token_limit,
                excluded_nodes=stamped_exclusions_list,
                system_role=command.system_role,
                hosts_by_node=hosts_by_node,
                ephemeral_port=ephemeral_port,
            )

    return target_instances


def _placement_intent_from_instance(
    instance: Instance,
) -> tuple[ModelCard, Sharding, InstanceMeta, int]:
    """Recover the placement intent (model, sharding, backend, width) of an instance.

    The model card, sharding family, and instance backend are read back off the
    instance's shards so a recovery re-placement reproduces the original intent.
    Returns the width (node count) too. Raises :class:`PlacementError` if the
    instance carries no shards (a structurally-allowed empty ``ShardAssignments``
    has no model to re-place), letting callers tear the husk down on the same
    path they use for a genuine no-fit rather than crashing.
    """
    shards = list(instance.shard_assignments.runner_to_shard.values())
    if not shards:
        raise PlacementError(
            f"Cannot re-place instance {instance.instance_id}: it has no shards"
        )
    # All shards of an instance share one model card.
    model_card = shards[0].model_card
    sharding = (
        Sharding.Tensor
        if any(isinstance(shard, TensorShardMetadata) for shard in shards)
        else Sharding.Pipeline
    )
    instance_meta = instance_meta_of(instance)
    width = len(instance.shard_assignments.node_to_runner)
    return model_card, sharding, instance_meta, width


def fallback_command_for_refused_instance(
    instance: Instance, refusing_node: NodeId
) -> PlaceInstance:
    """Build the anywhere-but-the-refuser fallback for a memory-refused instance.

    The primary #290 recovery re-places one node WIDER so every shard shrinks,
    but on a heterogeneous fleet the wider width can be unsatisfiable by
    construction: an MLX model refused at the full Mac width cannot add an AMD
    node, so the wider attempt raises and recovery used to give up even though
    a narrower cycle EXCLUDING the memory-constrained refuser would fit
    (observed live: a 3-Mac placement refused on the 16GB node dead-ended at
    min_nodes=4 while the 24GB node could serve the model alone). This
    fallback re-places at ``min_nodes=1`` with the refusing node excluded,
    letting the planner pick the best remaining cycle; the memory fit-check,
    not the width, decides. The caller treats a second failure as terminal.
    """
    model_card, sharding, instance_meta, _width = _placement_intent_from_instance(
        instance
    )
    return PlaceInstance(
        model_card=model_card,
        sharding=sharding,
        instance_meta=instance_meta,
        min_nodes=1,
        # The refusing node plus the instance's original per-placement
        # exclusions (#658): the anywhere-fallback still may not land on
        # nodes the caller excluded.
        excluded_nodes=sorted({*instance.excluded_nodes, refusing_node}),
        # System-role marker (intelligent-fabric steward): repair
        # re-placements re-stamp it so the flag survives node loss.
        system_role=instance.system_role,
    )


def replacement_command_for_refused_instance(instance: Instance) -> PlaceInstance:
    """Build a *wider* placement command to recover a memory-refused instance.

    When a worker refuses its shard for lack of GPU-wireable memory at load
    time (#290), the master re-places the same model one node wider so every
    node holds a smaller share. The placement intent (model card, sharding
    family, instance backend, per-placement node exclusions) is recovered
    from the instance itself; the caller's original exclusions are stamped
    on the instance at placement time (#658), so the wider re-placement
    keeps honoring them instead of searching the full topology.

    ``min_nodes`` is the refused width plus one. Raising it past the cluster
    size makes :func:`place_instance` raise :class:`PlacementError`, which the
    caller treats as the terminal "cannot fit anywhere" outcome — this bounds
    the refuse→re-place loop to at most ``cluster_size`` iterations.

    Raises :class:`PlacementError` if the instance carries no shards.
    """
    model_card, sharding, instance_meta, width = _placement_intent_from_instance(
        instance
    )
    return PlaceInstance(
        model_card=model_card,
        sharding=sharding,
        instance_meta=instance_meta,
        min_nodes=width + 1,
        # The instance's original per-placement exclusions (#658); the wider
        # re-placement keeps honoring the caller's intent.
        excluded_nodes=sorted(instance.excluded_nodes),
        # System-role marker (intelligent-fabric steward): repair
        # re-placements re-stamp it so the flag survives node loss.
        system_role=instance.system_role,
    )


def replacement_command_for_download_failed_instance(
    instance: Instance, excluded_nodes: AbstractSet[NodeId]
) -> PlaceInstance:
    """Build a *same-width* re-placement that avoids the download-failed nodes (#381).

    When a rank's model download terminally fails (disk full, transient HF or
    network error), the instance can never become ready. Recovery keeps the
    same world size but excludes the node(s) whose download failed, forcing the
    planner onto healthy nodes. If too few nodes remain to host the width,
    :func:`place_instance` raises :class:`PlacementError`, which the caller
    treats as the terminal "cannot recover" outcome and stops at the teardown —
    bounding recovery to the available healthy nodes rather than looping.

    The excluded set must be passed to :func:`place_instance` as well; it is
    carried on the command only as a record of intent.

    Raises :class:`PlacementError` if the instance carries no shards.
    """
    model_card, sharding, instance_meta, width = _placement_intent_from_instance(
        instance
    )
    return PlaceInstance(
        model_card=model_card,
        sharding=sharding,
        instance_meta=instance_meta,
        min_nodes=width,
        # The union of the instance's original per-placement exclusions
        # (#658) and the newly failed node(s): repair must not land on
        # nodes the caller excluded, nor on the nodes that just failed.
        excluded_nodes=sorted({*instance.excluded_nodes, *excluded_nodes}),
        # System-role marker (intelligent-fabric steward): repair
        # re-placements re-stamp it so the flag survives node loss.
        system_role=instance.system_role,
    )


def delete_instance(
    command: DeleteInstance,
    current_instances: Mapping[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    target_instances = dict(deepcopy(current_instances))
    if command.instance_id in target_instances:
        del target_instances[command.instance_id]
        return target_instances
    raise ValueError(f"Instance {command.instance_id} not found")


def get_transition_events(
    current_instances: Mapping[InstanceId, Instance],
    target_instances: Mapping[InstanceId, Instance],
    tasks: Mapping[TaskId, Task],
) -> Sequence[Event]:
    events: list[Event] = []

    # find instances to create
    for instance_id, instance in target_instances.items():
        if instance_id not in current_instances:
            events.append(
                InstanceCreated(
                    instance=instance,
                )
            )

    # find instances to delete
    for instance_id in current_instances:
        if instance_id not in target_instances:
            for task in tasks.values():
                if task.instance_id == instance_id and task.task_status in [
                    TaskStatus.Pending,
                    TaskStatus.Running,
                ]:
                    events.append(
                        TaskStatusUpdated(
                            task_status=TaskStatus.Cancelled,
                            task_id=task.task_id,
                        )
                    )

            events.append(
                InstanceDeleted(
                    instance_id=instance_id,
                )
            )

    return events


def cancel_unnecessary_downloads(
    instances: Mapping[InstanceId, Instance],
    download_status: Mapping[NodeId, Sequence[DownloadProgress]],
) -> Sequence[DownloadCommand]:
    commands: list[DownloadCommand] = []
    currently_downloading = [
        (k, v.shard_metadata.model_card.model_id)
        for k, vs in download_status.items()
        for v in vs
        if isinstance(v, (DownloadOngoing))
    ]
    active_models = set(
        (
            node_id,
            instance.shard_assignments.runner_to_shard[runner_id].model_card.model_id,
        )
        for instance in instances.values()
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items()
    )
    for pair in currently_downloading:
        if pair not in active_models:
            commands.append(CancelDownload(target_node_id=pair[0], model_id=pair[1]))

    return commands
