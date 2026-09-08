import ipaddress
from collections.abc import Generator, Mapping
from typing import Final, final

from loguru import logger
from pydantic import Field

from skulk.shared.models.memory_estimate import (
    GPU_VRAM_WORKING_SET_FRACTION,
    GPU_WORKING_SET_FRACTION,
    UMA_GPU_OS_HEADROOM,
    backend_offloads_to_vram,
    estimate_shard_footprint,
    memory_overhead_factor,
    shard_fraction_of_model,
)
from skulk.shared.models.memory_estimate import (
    KV_CONTEXT_BUDGET_TOKENS as PLACEMENT_KV_CONTEXT_BUDGET_TOKENS,
)
from skulk.shared.models.memory_estimate import (
    MEMORY_OVERHEAD_FLOOR as PLACEMENT_MEMORY_OVERHEAD_FLOOR,
)
from skulk.shared.models.memory_estimate import (
    estimate_kv_cache_bytes as _estimate_kv_cache_bytes,
)
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.topology import Topology
from skulk.shared.types.common import Host, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import (
    MemoryUsage,
    NodeNetworkInfo,
    NodeResources,
    SystemPerformanceProfile,
)
from skulk.shared.types.topology import Cycle, RDMAConnection, SocketConnection
from skulk.shared.types.worker.instances import Instance, InstanceId, LlamaRpcInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    RpcDonorShardMetadata,
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)
from skulk.utils.pydantic_ext import CamelCaseModel

# Memory-fit constants (overhead factor/floor, GPU working-set fraction, KV
# context budget) and the KV-cache estimator are imported above under the
# historical ``PLACEMENT_*`` / ``_estimate_kv_cache_bytes`` names. They live in
# skulk.shared.models.memory_estimate so this placement fit-check and the
# worker's local pre-spawn OOM guard share one source of truth.


@final
class CycleMemoryDiagnostics(CamelCaseModel):
    """Why candidate cycles were rejected by the memory filter.

    Carried alongside the surviving cycles so the caller can raise an error
    that names the actual cause instead of a generic "insufficient memory".
    """

    pending_info_node_ids: list[NodeId] = Field(default_factory=list)
    """Nodes that appeared in candidate cycles but have no memory info yet.

    This happens in the window between a node joining the topology (identity
    gossiped) and its first NodeGatheredInfo event arriving — placing during
    cluster startup lands here. It is a retry-shortly condition, not a
    memory shortfall.
    """

    rejection_reasons: list[str] = Field(default_factory=list)
    """One human-readable line per cycle rejected on real memory grounds."""


# Engines that offload weights + KV onto the GPU (so placement admits them
# against VRAM, not system RAM). Both the in-process llama.cpp runner and the
# served llama-server engine launch with ``-ngl`` full-GPU offload; vLLM would
# join here too. The bare CPU compute tag is excluded by the ``-cpu`` check.
_GPU_OFFLOAD_ENGINE_PREFIXES: Final = ("llama_cpp-", "llama_server-", "vllm-")


def _has_gpu_offload_backend(backends: frozenset[str]) -> bool:
    """Whether a node advertises a discrete-GPU-offload backend.

    True for a non-CPU GPU-offload compute tag (``llama_cpp-vulkan/rocm/cuda``,
    ``llama_server-vulkan/rocm/cuda``, or ``vllm-cuda/rocm``). A node that exposes a
    GPU but advertises only a ``-cpu`` tag (the default when
    ``SKULK_LLAMA_CPP_BACKENDS`` is unset, or after a GPU wheel is replaced) runs out
    of system RAM, so it must NOT be admitted against VRAM it will not use. The
    served ``llama_server`` engine is included because it launches
    ``llama-server -ngl 99``, and ``vllm`` because ``vllm serve`` allocates weights +
    KV from the GPU -- both use VRAM exactly like the in-process llama.cpp runner.
    """
    return any(
        tag.startswith(_GPU_OFFLOAD_ENGINE_PREFIXES) and not tag.endswith("-cpu")
        for tag in backends
    )


def usable_vram_by_node(
    node_system: Mapping[NodeId, SystemPerformanceProfile],
    node_resources: Mapping[NodeId, NodeResources] | None = None,
    node_memory: Mapping[NodeId, MemoryUsage] | None = None,
    current_instances: Mapping[InstanceId, Instance] | None = None,
) -> dict[NodeId, Memory]:
    """Per-node usable GPU memory for placement, keyed by node.

    A node appears here only when it both (a) reports a VRAM-bearing accelerator
    (AMD/NVIDIA with a nonzero ``vram_total_bytes`` in ``node_system``) and
    (b) advertises a GPU-offload llama.cpp backend (``_has_gpu_offload_backend``),
    so its engine actually allocates weights + KV from the GPU. Apple
    unified-memory nodes report no discrete VRAM and are absent, keeping the
    system-RAM path. When ``node_resources`` is omitted the backend gate is
    skipped (test/simple use).

    The usable figure depends on the node's memory architecture:

    * Discrete GPU: ``min(vram_total - vram_used, vram_total *
      GPU_VRAM_WORKING_SET_FRACTION)``. VRAM is a dedicated pool reported
      separately from system RAM, so placement admits against VRAM, not
      ``0.75 * system_ram``, and never assumes the whole card is free.
      When ``current_instances`` is supplied, committed concrete GPU shards
      reduce the static working-set ceiling by their estimated weights,
      overhead, and stamped context window. This covers allocations not yet
      visible in telemetry without subtracting loaded memory twice. RPC
      placements have no concrete per-device partition and retain observed
      memory admission; their driver shard is not a whole-model reservation.
    * Unified-memory APU (e.g. AMD Strix Halo), detected when the GTT aperture
      spans the whole system (``gtt_total > vram_total`` AND ``gtt_total >=
      ram_total``): the GPU addresses the BIOS VRAM carve-out PLUS system RAM
      mapped through GTT, so a model larger than the carve-out runs there. Usable
      GPU memory is the working-set-capped VRAM plus the system RAM the GPU can
      map via GTT, leaving ``UMA_GPU_OS_HEADROOM`` for the OS + worker + download
      staging: ``min(vram_avail, vram_total * GPU_VRAM_WORKING_SET_FRACTION) +
      min(max(0, ram_available - UMA_GPU_OS_HEADROOM), gtt_total)``. The UMA path
      needs the node's current memory from ``node_memory``; without that entry it
      falls back to the discrete path. A discrete GPU also exposes a GTT total
      (often ~= VRAM), so requiring GTT to cover all of system RAM is what keeps
      it off this path.

    The same figure serves single-node AND multi-node RPC admission. An
    earlier revision admitted RPC placements against a VRAM-carve-only
    variant, misreading a spike where llama.cpp's proportional split simply
    never asked a donor to exceed its carve (``gtt_used`` stayed 0 because
    the shares happened to fit, not because RPC cannot reach GTT). On a UMA
    APU the carve is not an allocation boundary; a pooled gpt-oss-120b was
    refused by the carve figure and then served fine on the same pair once
    admitted against unified memory.
    """
    node_memory = node_memory or {}
    usable: dict[NodeId, Memory] = {}
    for node_id, profile in node_system.items():
        accelerator = profile.accelerator
        if accelerator is None or accelerator.vendor not in ("amd", "nvidia"):
            continue
        total = accelerator.vram_total_bytes
        if not total or total <= 0:
            continue
        if node_resources is not None:
            resources = node_resources.get(node_id)
            if resources is None or not _has_gpu_offload_backend(resources.backends):
                continue
        used = accelerator.vram_used_bytes or 0
        vram_avail = max(0, total - used)
        ceiling = int(total * GPU_VRAM_WORKING_SET_FRACTION)
        vram_usable = min(vram_avail, ceiling)
        gtt_total = accelerator.gtt_total_bytes
        memory = node_memory.get(node_id)
        # UMA signature: the GTT aperture spans the WHOLE system, i.e. it both
        # exceeds the VRAM carve-out AND covers all of system RAM. A discrete AMD
        # GPU also exposes ``mem_info_gtt_total`` (its default can equal VRAM), so
        # ``gtt_total >= vram_total`` alone would misclassify a discrete card as
        # unified and over-admit; requiring ``gtt_total >= ram_total`` keeps a
        # discrete GPU (gtt ~= vram < system RAM) on the conservative VRAM-only
        # path.
        if (
            gtt_total is not None
            and gtt_total > total
            and memory is not None
            and gtt_total >= memory.ram_total.in_bytes
        ):
            # GTT spans host RAM, so the GPU can map system memory beyond the
            # BIOS VRAM carve-out; count it (minus OS headroom) toward the pool,
            # capped by what GTT itself can address. The VRAM portion still keeps
            # its working-set headroom for driver/fragmentation overhead.
            sys_for_gpu = min(
                max(0, memory.ram_available.in_bytes - UMA_GPU_OS_HEADROOM.in_bytes),
                gtt_total,
            )
            usable[node_id] = Memory.from_bytes(vram_usable + sys_for_gpu)
            continue
        usable[node_id] = Memory.from_bytes(vram_usable)
    return reserve_instance_vram(
        usable,
        node_system,
        current_instances or {},
        unified_memory_gpu_nodes=unified_memory_gpu_node_ids(
            node_system, node_resources, node_memory
        ),
    )


def reserve_instance_vram(
    node_vram: Mapping[NodeId, Memory],
    node_system: Mapping[NodeId, SystemPerformanceProfile],
    current_instances: Mapping[InstanceId, Instance],
    *,
    unified_memory_gpu_nodes: frozenset[NodeId] = frozenset(),
) -> dict[NodeId, Memory]:
    """Cap observed discrete VRAM by the capacity committed to concrete shards.

    Args:
        node_vram: Observed usable memory, or a hypothetical teardown-credit map.
        node_system: Accelerator telemetry containing physical VRAM totals.
        current_instances: Placements that still own their shard reservations.
        unified_memory_gpu_nodes: APUs using host memory; retain their existing
            combined-pool admission instead of treating the VRAM carve as a cap.

    Returns:
        A fresh memory map bounded by both observations and uncommitted physical
        working-set capacity. RPC partitions remain observation-only because
        their runtime-selected per-device shares are not represented by shards.
    """
    committed: dict[NodeId, int] = {}
    for instance in current_instances.values():
        if isinstance(instance, LlamaRpcInstance):
            # The driver nominally spans every layer but llama.cpp chooses the
            # actual pooled split at runtime. Charging that nominal shard would
            # invent a full-model allocation on the driver and none on donors.
            continue
        assignments = instance.shard_assignments
        for node_id, runner_id in assignments.node_to_runner.items():
            shard = assignments.runner_to_shard[runner_id]
            if not backend_offloads_to_vram(shard.resolved_backend):
                continue
            fraction = shard_fraction_of_model(shard)
            if fraction is None:
                continue
            footprint = estimate_shard_footprint(
                shard.model_card,
                fraction,
                context_budget=(
                    instance.context_token_limit
                    if instance.context_token_limit is not None
                    else PLACEMENT_KV_CONTEXT_BUDGET_TOKENS
                ),
            )
            committed[node_id] = committed.get(node_id, 0) + footprint.in_bytes
    usable = dict(node_vram)
    for node_id, observed in node_vram.items():
        profile = node_system.get(node_id)
        accelerator = profile.accelerator if profile is not None else None
        total = accelerator.vram_total_bytes if accelerator is not None else None
        if node_id in unified_memory_gpu_nodes or total is None or total <= 0:
            continue
        ceiling = int(total * GPU_VRAM_WORKING_SET_FRACTION)
        # Observed usage includes loaded instances already. Bound against the
        # remaining static budget instead of subtracting from observed free
        # memory, which would count those allocations twice. After deletion,
        # lagging telemetry still limits admission until memory is released.
        usable[node_id] = Memory.from_bytes(
            min(observed.in_bytes, max(0, ceiling - committed.get(node_id, 0)))
        )
    return usable


def unified_memory_gpu_node_ids(
    node_system: Mapping[NodeId, SystemPerformanceProfile],
    node_resources: Mapping[NodeId, NodeResources] | None = None,
    node_memory: Mapping[NodeId, MemoryUsage] | None = None,
) -> frozenset[NodeId]:
    """Return GPU-offload nodes whose accelerator pool includes host RAM.

    A Strix-class APU reports a BIOS VRAM carve-out plus a GTT aperture spanning
    all remaining system RAM. Placement may use that combined pool, but served
    engines that allocate a fixed context window at startup cannot treat it as
    discrete VRAM: amdgpu may need host pages while constructing the GPU
    buffers, so the steady-state combined-pool fit does not bound peak host
    memory. The returned set lets context admission preserve the conservative
    served-engine floor on those nodes without giving up UMA-aware placement.

    Args:
        node_system: Per-node accelerator telemetry.
        node_resources: Optional backend telemetry. When supplied, only nodes
            advertising a GPU-offload backend are considered.
        node_memory: Per-node system-memory telemetry required to prove that GTT
            spans the whole host-memory pool.

    Returns:
        Immutable IDs of confirmed unified-memory GPU-offload nodes.
    """
    node_memory = node_memory or {}
    unified: set[NodeId] = set()
    for node_id, profile in node_system.items():
        accelerator = profile.accelerator
        memory = node_memory.get(node_id)
        if (
            accelerator is None
            or accelerator.vendor != "amd"
            or not accelerator.vram_total_bytes
            or accelerator.vram_total_bytes <= 0
            or accelerator.gtt_total_bytes is None
            or memory is None
        ):
            continue
        if node_resources is not None:
            resources = node_resources.get(node_id)
            if resources is None or not _has_gpu_offload_backend(resources.backends):
                continue
        if (
            accelerator.gtt_total_bytes > accelerator.vram_total_bytes
            and accelerator.gtt_total_bytes >= memory.ram_total.in_bytes
        ):
            unified.add(node_id)
    return frozenset(unified)


def _per_node_required_memory(
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    model_card: ModelCard,
    sharding: Sharding,
    node_vram: Mapping[NodeId, Memory] | None = None,
    *,
    exact_pipeline_layers: bool = True,
    fixed_memory_by_node: Mapping[NodeId, Memory] | None = None,
) -> dict[NodeId, Memory]:
    """Estimate the weight bytes each node in the cycle must hold.

    Tensor parallelism splits the weights evenly across ranks, so every node
    carries ``required_memory / len(cycle)`` regardless of its capacity.
    Pipeline parallelism allocates integer layer counts proportionally to each
    node's usable memory (see ``allocate_layers_proportionally``), so this
    returns the exact post-rounding weight share the worker will be asked to
    load. CFG-parallel image models allocate layers across one pipeline group
    and mirror those ranges across the second CFG group, matching
    ``_get_shard_assignments_for_cfg_parallel``. A fractional estimate can pass
    a borderline heterogeneous cycle that later fails the worker's local guard
    once one node receives an extra layer. Callers that use ``Sharding.Pipeline``
    as a proportional-memory proxy rather than real Skulk layer ranges can set
    ``exact_pipeline_layers=False``.

    Both the split here and the per-node admission below weigh by
    ``_node_usable_memory`` (capped at the Metal working-set ceiling), not raw
    ``ram_available``. Splitting by raw available while admitting against the
    cap over-weights a node whose free RAM exceeds its GPU ceiling (e.g. a
    16 GB node with 15 GB free but a ~12 GB cap), pushing its share past the
    cap and rejecting cycles that would fit if sized by usable memory — exactly
    the heterogeneous clusters this check is meant to support.
    """
    node_vram = node_vram or {}
    fixed_memory_by_node = fixed_memory_by_node or {}
    required_memory = model_card.storage_size
    if sharding == Sharding.Tensor:
        even_share = required_memory / len(cycle.node_ids)
        return {node_id: even_share for node_id in cycle.node_ids}
    node_usable = {
        node_id: max(
            Memory(),
            _node_usable_memory(node_memory[node_id], node_vram.get(node_id))
            - fixed_memory_by_node.get(node_id, Memory()),
        )
        for node_id in cycle.node_ids
    }
    total_usable = sum(node_usable.values(), start=Memory())
    if total_usable.in_bytes == 0:
        # Degenerate: no node reports any memory; assign everything everywhere
        # so the fit check below rejects the cycle with a concrete reason.
        return {node_id: required_memory for node_id in cycle.node_ids}
    if not exact_pipeline_layers:
        return {
            node_id: required_memory * (node_usable[node_id] / total_usable)
            for node_id in cycle.node_ids
        }
    if _uses_cfg_parallel(model_card, len(cycle.node_ids)):
        cfg_world_size = 2
        pipeline_world_size = len(cycle.node_ids) // cfg_world_size
        pipeline_node_ids = cycle.node_ids[:pipeline_world_size]
        pipeline_usable = {
            node_id: node_usable[node_id] for node_id in pipeline_node_ids
        }
        pipeline_total_usable = sum(pipeline_usable.values(), start=Memory())
        if pipeline_total_usable.in_bytes == 0:
            return {node_id: required_memory for node_id in cycle.node_ids}
        layer_allocations = allocate_pipeline_layers(
            model_card=model_card,
            memory_fractions=[
                pipeline_usable[node_id] / pipeline_total_usable
                for node_id in pipeline_node_ids
            ],
        )
        pipeline_ranks = list(range(pipeline_world_size)) + list(
            reversed(range(pipeline_world_size))
        )
        return {
            node_id: (required_memory * layer_allocations[pipeline_ranks[index]])
            // model_card.n_layers
            for index, node_id in enumerate(cycle.node_ids)
        }
    layer_allocations = allocate_pipeline_layers(
        model_card=model_card,
        memory_fractions=[
            node_usable[node_id] / total_usable for node_id in cycle.node_ids
        ],
    )
    return {
        node_id: (required_memory * layer_allocations[index]) // model_card.n_layers
        for index, node_id in enumerate(cycle.node_ids)
    }


def _uses_cfg_parallel(model_card: ModelCard, world_size: int) -> bool:
    """Whether pipeline placement will use mirrored CFG groups for this cycle."""
    return model_card.uses_cfg and world_size >= 2 and world_size % 2 == 0


def _effective_pipeline_stage_count(model_card: ModelCard, world_size: int) -> int:
    """Count real pipeline stages for exact layer allocation guards."""
    if _uses_cfg_parallel(model_card, world_size):
        return world_size // 2
    return world_size


def _node_usable_memory(
    memory: MemoryUsage, vram_usable: Memory | None = None
) -> Memory:
    """Memory a node can actually dedicate to a shard.

    On a discrete-VRAM GPU node (``vram_usable`` provided, see
    ``usable_vram_by_node``), this is the usable VRAM pool the GPU-offload engine
    allocates from, which is distinct from system RAM. Otherwise (Apple unified
    memory) it is capped at the Metal GPU working-set ceiling
    (``GPU_WORKING_SET_FRACTION`` of total RAM): gossiped ``ram_available`` can
    exceed what the GPU is allowed to wire, so a shard that fits "available" RAM
    can still abort on the GPU working-set wall.
    """
    if vram_usable is not None:
        return vram_usable
    ceiling = memory.ram_total * GPU_WORKING_SET_FRACTION
    return min(memory.ram_available, ceiling)


def filter_cycles_by_memory(
    cycles: list[Cycle],
    node_memory: Mapping[NodeId, MemoryUsage],
    model_card: ModelCard,
    sharding: Sharding = Sharding.Pipeline,
    context_budget: int = PLACEMENT_KV_CONTEXT_BUDGET_TOKENS,
    node_vram: Mapping[NodeId, Memory] | None = None,
    *,
    exact_pipeline_layers: bool = True,
    fixed_memory_by_node: Mapping[NodeId, Memory] | None = None,
) -> tuple[list[Cycle], CycleMemoryDiagnostics]:
    """Keep cycles whose every node can hold its shard with runtime headroom.

    The fit test is per node, not summed across the cycle: a 16 GB + 24 GB
    pair summing 21 GB of available memory cannot run a Tensor-sharded model
    whose even split overloads the 16 GB node, and a sum check would happily
    admit it. Each node must satisfy::

        weights_share * memory_overhead_factor(model_card)
            + kv_share + PLACEMENT_MEMORY_OVERHEAD_FLOOR
            <= node usable memory

    where usable memory is the node's discrete VRAM pool when it has one
    (``node_vram``, see ``usable_vram_by_node``) or
    ``min(ram_available, GPU_WORKING_SET_FRACTION * ram_total)`` on Apple unified
    memory.

    ``kv_share`` reserves KV cache for ``context_budget`` tokens. It scales with
    a node's shard exactly as the weights do (per-layer for exact pipeline,
    proportional for RPC-style pipeline, per-rank for tensor), so it is derived
    from the node's weight share rather than recomputing the split.

    Cycles touching nodes with no memory info at all (cluster still starting
    up) are recorded in ``diagnostics.pending_info_node_ids`` instead of being
    silently dropped — the caller distinguishes "retry shortly" from "does
    not fit".

    Returns the surviving cycles plus diagnostics describing every rejection.
    """
    node_vram = node_vram or {}
    fixed_memory_by_node = fixed_memory_by_node or {}
    diagnostics = CycleMemoryDiagnostics()
    filtered_cycles: list[Cycle] = []
    required_memory = model_card.storage_size
    # ``context_budget`` (KV_CONTEXT_BUDGET_TOKENS) is the admission FLOOR, not the
    # served window: a node is admissible only if it can hold at least this much KV,
    # so every placement can serve at least this context. The runner then serves the
    # LARGER memory-fit window from ``instance_context_token_limit`` (up to the card
    # max), derived from the SAME working set this filter admits against (VRAM on a
    # GPU node), so loading KV at that larger window fits by construction and is
    # >= this floor -- except when the model's own advertised max context is smaller
    # (then it serves that smaller max), or when the fit is uncomputable for a gguf
    # card (then it clamps back to this floor to avoid a load-time OOM; see
    # instance_context_token_limit). (Lifting the floor itself, to admit a big model
    # at a reduced context rather than reject it, is a separate future lever.)
    total_kv = _estimate_kv_cache_bytes(model_card, model_card.n_layers, context_budget)
    kv_ratio = (
        total_kv.in_bytes / required_memory.in_bytes
        if required_memory.in_bytes
        else 0.0
    )
    for cycle in cycles:
        missing = [node for node in cycle.node_ids if node not in node_memory]
        if missing:
            for node_id in missing:
                if node_id not in diagnostics.pending_info_node_ids:
                    diagnostics.pending_info_node_ids.append(node_id)
            continue
        pipeline_stage_count = _effective_pipeline_stage_count(
            model_card, len(cycle.node_ids)
        )
        if (
            exact_pipeline_layers
            and sharding == Sharding.Pipeline
            and pipeline_stage_count > model_card.n_layers
        ):
            diagnostics.rejection_reasons.append(
                f"cycle [{', '.join(str(n) for n in cycle.node_ids)}] "
                f"({sharding.value} sharding): {pipeline_stage_count} pipeline "
                "stages exceed "
                f"{model_card.n_layers} model layers"
            )
            continue
        split_limit = model_card.placement.max_pipeline_split_layer
        if (
            exact_pipeline_layers
            and sharding == Sharding.Pipeline
            and split_limit is not None
            and pipeline_stage_count - 1 > split_limit
        ):
            diagnostics.rejection_reasons.append(
                f"cycle [{', '.join(str(n) for n in cycle.node_ids)}] "
                f"({sharding.value} sharding): {pipeline_stage_count} pipeline "
                f"stages cannot fit before safe split boundary {split_limit}"
            )
            continue

        node_shares = _per_node_required_memory(
            cycle,
            node_memory,
            model_card,
            sharding,
            node_vram,
            exact_pipeline_layers=exact_pipeline_layers,
            fixed_memory_by_node=fixed_memory_by_node,
        )
        # GGUF runs on the lighter llama.cpp C++ runtime, so its weight overhead
        # is smaller than MLX's; use the engine-aware factor for the fit decision.
        overhead_factor = memory_overhead_factor(model_card)
        overloaded: list[str] = []
        for node_id, share in node_shares.items():
            kv_share = share * kv_ratio
            required_with_overhead = (
                share * overhead_factor
                + kv_share
                + PLACEMENT_MEMORY_OVERHEAD_FLOOR
                + fixed_memory_by_node.get(node_id, Memory())
            )
            node_vram_usable = node_vram.get(node_id)
            available = _node_usable_memory(node_memory[node_id], node_vram_usable)
            if required_with_overhead > available:
                pool = (
                    "usable GPU VRAM"
                    if node_vram_usable is not None
                    else f"min of available and {GPU_WORKING_SET_FRACTION:.0%} of RAM"
                )
                overloaded.append(
                    f"node {node_id} needs ~{required_with_overhead.in_gb:.1f}GB "
                    f"({share.in_gb:.1f}GB weights + {kv_share.in_gb:.1f}GB "
                    f"KV@{context_budget}tok + runtime/fixed headroom) but can use "
                    f"{available.in_gb:.1f}GB ({pool})"
                )
        if overloaded:
            diagnostics.rejection_reasons.append(
                f"cycle [{', '.join(str(n) for n in cycle.node_ids)}] "
                f"({sharding.value} sharding): " + "; ".join(overloaded)
            )
            continue
        filtered_cycles.append(cycle)
    return filtered_cycles, diagnostics


def get_smallest_cycles(
    cycles: list[Cycle],
) -> list[Cycle]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]


def allocate_layers_proportionally(
    total_layers: int,
    memory_fractions: list[float],
) -> list[int]:
    n = len(memory_fractions)
    if n == 0:
        raise ValueError("Cannot allocate layers to an empty node list")
    if total_layers < n:
        raise ValueError(
            f"Cannot distribute {total_layers} layers across {n} nodes "
            "(need at least 1 layer per node)"
        )

    # Largest remainder: floor each, then distribute remainder by fractional part
    raw = [f * total_layers for f in memory_fractions]
    result = [int(r) for r in raw]
    by_remainder = sorted(range(n), key=lambda i: raw[i] - result[i], reverse=True)
    for i in range(total_layers - sum(result)):
        result[by_remainder[i]] += 1

    # Ensure minimum 1 per node by taking from the largest
    for i in range(n):
        if result[i] == 0:
            max_idx = max(range(n), key=lambda j: result[j])
            assert result[max_idx] > 1
            result[max_idx] -= 1
            result[i] = 1

    return result


def allocate_pipeline_layers(
    model_card: ModelCard,
    memory_fractions: list[float],
) -> list[int]:
    """Allocate pipeline layers while honoring model-specific safe boundaries.

    The ordinary proportional allocation remains the target. When the model's
    tail reuses KV from earlier concrete layers, its card caps the final split
    boundary so the last rank also owns every producer. Earlier boundaries are
    shifted left only when necessary, preserving at least one layer per rank.

    Args:
        model_card: Model whose layer and placement constraints are applied.
        memory_fractions: Per-rank shares of usable memory in pipeline order.

    Returns:
        Positive contiguous layer counts whose sum is ``model_card.n_layers``.

    Raises:
        ValueError: The safe boundary cannot accommodate the requested ranks.
    """
    allocations = allocate_layers_proportionally(
        total_layers=model_card.n_layers,
        memory_fractions=memory_fractions,
    )
    split_limit = model_card.placement.max_pipeline_split_layer
    stage_count = len(allocations)
    if split_limit is None or stage_count == 1:
        return allocations
    if stage_count - 1 > split_limit:
        raise ValueError(
            f"Cannot distribute {model_card.n_layers} layers across {stage_count} "
            f"pipeline ranks before safe split boundary {split_limit}"
        )

    desired_boundaries: list[int] = []
    layers_before = 0
    for layer_count in allocations[:-1]:
        layers_before += layer_count
        desired_boundaries.append(layers_before)

    safe_boundaries = [0] * len(desired_boundaries)
    next_boundary = split_limit + 1
    for index in reversed(range(len(desired_boundaries))):
        boundary = min(desired_boundaries[index], next_boundary - 1)
        minimum_boundary = index + 1
        if boundary < minimum_boundary:
            raise ValueError(
                f"Cannot assign positive pipeline shards before safe split "
                f"boundary {split_limit}"
            )
        safe_boundaries[index] = boundary
        next_boundary = boundary

    constrained: list[int] = []
    previous_boundary = 0
    for boundary in safe_boundaries:
        constrained.append(boundary - previous_boundary)
        previous_boundary = boundary
    constrained.append(model_card.n_layers - previous_boundary)
    return constrained


def _validate_cycle(cycle: Cycle) -> None:
    if not cycle.node_ids:
        raise ValueError("Cannot create shard assignments for empty node cycle")


def _compute_total_memory(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> Memory:
    # Weigh by usable (working-set-capped) memory, matching the admission check
    # in filter_cycles_by_memory: splitting by raw ram_available while admitting
    # against the cap over-weights nodes whose free RAM exceeds their GPU
    # ceiling and produces shards that don't fit.
    node_vram = node_vram or {}
    total_memory = sum(
        (
            _node_usable_memory(node_memory[node_id], node_vram.get(node_id))
            for node_id in node_ids
        ),
        start=Memory(),
    )
    if total_memory.in_bytes == 0:
        raise ValueError("Cannot create shard assignments: total usable memory is 0")
    return total_memory


def _allocate_and_validate_layers(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    total_memory: Memory,
    model_card: ModelCard,
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> list[int]:
    node_vram = node_vram or {}
    layer_allocations = allocate_pipeline_layers(
        model_card=model_card,
        memory_fractions=[
            _node_usable_memory(node_memory[node_id], node_vram.get(node_id))
            / total_memory
            for node_id in node_ids
        ],
    )

    total_storage = model_card.storage_size
    total_layers = model_card.n_layers
    for i, node_id in enumerate(node_ids):
        node_layers = layer_allocations[i]
        required_memory = (total_storage * node_layers) // total_layers
        usable_memory = _node_usable_memory(
            node_memory[node_id], node_vram.get(node_id)
        )
        if required_memory > usable_memory:
            pool = (
                "GPU VRAM"
                if node_vram.get(node_id) is not None
                else f"{GPU_WORKING_SET_FRACTION:.0%} of RAM cap"
            )
            raise ValueError(
                f"Node {i} ({node_id}) has insufficient memory: "
                f"requires {required_memory.in_gb:.2f} GB for {node_layers} layers, "
                f"but can only use {usable_memory.in_gb:.2f} GB ({pool})"
            )

    return layer_allocations


def get_shard_assignments_for_pipeline_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> ShardAssignments:
    """Create shard assignments for pipeline parallel execution."""
    world_size = len(cycle)
    use_cfg_parallel = model_card.uses_cfg and world_size >= 2 and world_size % 2 == 0

    if use_cfg_parallel:
        return _get_shard_assignments_for_cfg_parallel(
            model_card, cycle, node_memory, node_vram
        )
    else:
        return _get_shard_assignments_for_pure_pipeline(
            model_card, cycle, node_memory, node_vram
        )


def _get_shard_assignments_for_cfg_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> ShardAssignments:
    """Create shard assignments for CFG parallel execution.

    CFG parallel runs two independent pipelines. Group 0 processes the positive
    prompt, group 1 processes the negative prompt. The ring topology places
    group 1's ranks in reverse order so both "last stages" are neighbors for
    efficient CFG exchange.
    """
    _validate_cycle(cycle)

    world_size = len(cycle)
    cfg_world_size = 2
    pipeline_world_size = world_size // cfg_world_size

    # Allocate layers for one pipeline group (both groups run the same layers)
    pipeline_node_ids = cycle.node_ids[:pipeline_world_size]
    pipeline_memory = _compute_total_memory(pipeline_node_ids, node_memory, node_vram)
    layer_allocations = _allocate_and_validate_layers(
        pipeline_node_ids, node_memory, pipeline_memory, model_card, node_vram
    )

    # Ring topology: group 0 ascending [0,1,2,...], group 1 descending [...,2,1,0]
    # This places both last stages as neighbors for CFG exchange.
    position_to_cfg_pipeline = [(0, r) for r in range(pipeline_world_size)] + [
        (1, r) for r in reversed(range(pipeline_world_size))
    ]

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for device_rank, node_id in enumerate(cycle.node_ids):
        cfg_rank, pipeline_rank = position_to_cfg_pipeline[device_rank]
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = CfgShardMetadata(
            model_card=model_card,
            device_rank=device_rank,
            world_size=world_size,
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
            cfg_rank=cfg_rank,
            cfg_world_size=cfg_world_size,
            pipeline_rank=pipeline_rank,
            pipeline_world_size=pipeline_world_size,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def _get_shard_assignments_for_pure_pipeline(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> ShardAssignments:
    """Create shard assignments for pure pipeline execution."""
    _validate_cycle(cycle)
    total_memory = _compute_total_memory(cycle.node_ids, node_memory, node_vram)

    layer_allocations = _allocate_and_validate_layers(
        cycle.node_ids, node_memory, total_memory, model_card, node_vram
    )

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for pipeline_rank, node_id in enumerate(cycle.node_ids):
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = PipelineShardMetadata(
            model_card=model_card,
            device_rank=pipeline_rank,
            world_size=len(cycle),
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def get_shard_assignments_for_tensor_parallel(
    model_card: ModelCard,
    cycle: Cycle,
):
    total_layers = model_card.n_layers
    world_size = len(cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for i, node_id in enumerate(cycle):
        shard = TensorShardMetadata(
            model_card=model_card,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
        )

        runner_id = RunnerId()

        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments(
    model_card: ModelCard,
    cycle: Cycle,
    sharding: Sharding,
    node_memory: Mapping[NodeId, MemoryUsage],
    node_vram: Mapping[NodeId, Memory] | None = None,
) -> ShardAssignments:
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_card=model_card,
                cycle=cycle,
                node_memory=node_memory,
                node_vram=node_vram,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_card=model_card,
                cycle=cycle,
            )


def get_shard_assignments_for_llama_rpc(
    model_card: ModelCard,
    cycle: Cycle,
    driver_node: NodeId,
) -> ShardAssignments:
    """Shard assignments for a driver-plus-donors llama.cpp RPC placement (#328).

    The driver (rank 0) gets a shard nominally spanning ALL layers: llama.cpp
    itself splits weights/KV across the pooled local+remote devices, so the
    range is bookkeeping for Skulk (download targeting, engine dispatch, memory
    accounting), not a layer instruction. Donors get the degenerate
    :class:`RpcDonorShardMetadata` (no layers) so the one-runner-per-node
    ``ShardAssignments`` invariant holds and the worker can dispatch a donor
    runner off the shard type.

    Args:
        model_card: The model to serve; the driver reads its GGUF from disk.
        cycle: The selected placement cycle; must contain ``driver_node``.
        driver_node: The node that runs llama-server and holds the model file.

    Returns:
        Assignments with one runner per node: a pipeline shard on the driver,
        donor shards elsewhere.
    """
    _validate_cycle(cycle)
    if driver_node not in cycle.node_ids:
        raise ValueError(f"Driver node {driver_node} is not part of the selected cycle")
    world_size = len(cycle.node_ids)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}
    donor_rank = 1
    for node_id in cycle.node_ids:
        if node_id == driver_node:
            shard: ShardMetadata = PipelineShardMetadata(
                model_card=model_card,
                device_rank=0,
                world_size=world_size,
                start_layer=0,
                end_layer=model_card.n_layers,
                n_layers=model_card.n_layers,
            )
        else:
            shard = RpcDonorShardMetadata(
                model_card=model_card,
                device_rank=donor_rank,
                world_size=world_size,
                start_layer=0,
                end_layer=0,
                n_layers=model_card.n_layers,
            )
            donor_rank += 1
        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id
    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def get_llama_rpc_donor_endpoints(
    selected_cycle: Cycle,
    driver_node: NodeId,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    donor_ports: Mapping[NodeId, int],
) -> dict[NodeId, str]:
    """Choose the ``ip:port`` each RPC donor binds and the driver dials (#328).

    For every non-driver node in the cycle, pick the donor-side address the
    driver can reach from the OBSERVED libp2p connections between the pair,
    ranked by the donor's gossiped interface type (Thunderbolt first, VPN last;
    same prioritiser as the MLX ring, #265) with link-local and loopback
    candidates excluded outright (``_is_routable_rpc_donor_address``: a
    multi-TB-port host routes all of 169.254/16 out one port, so a link-local
    endpoint breaks asymmetrically; point-to-point RPC links need a static
    per-link subnet). The chosen address is stamped on the instance: the
    donor's ggml-rpc-server binds exactly this address (never 0.0.0.0; the
    fabric is trusted but endpoints stay interface-scoped) and the driver's
    ``--rpc`` flag dials it.

    Raises:
        ValueError: When no ROUTABLE observed connection exists between the
            driver and a donor (no path at all, or only link-local/loopback
            paths — e.g. an unconfigured Thunderbolt link).
    """
    endpoints: dict[NodeId, str] = {}
    for node_id in selected_cycle.node_ids:
        if node_id == driver_node:
            continue
        donor_ip = _find_ip_prioritised(
            driver_node,
            node_id,
            cycle_digraph,
            node_network,
            ring=True,
            require_routable=True,
        )
        if donor_ip is None:
            raise ValueError(
                "Multi-node llama.cpp (RPC) requires a ROUTABLE path between "
                f"the driver and every donor; none observed from {driver_node} "
                f"to {node_id} (link-local/loopback addresses are excluded — "
                "assign a static subnet to point-to-point links, e.g. a "
                "Thunderbolt pair)"
            )
        endpoints[node_id] = f"{donor_ip}:{donor_ports[node_id]}"
    return endpoints


def get_mlx_jaccl_devices_matrix(
    selected_cycle: list[NodeId],
    cycle_digraph: Topology,
) -> list[list[str | None]]:
    """Build connectivity matrix mapping device i to device j via RDMA interface names.

    The matrix element [i][j] contains the interface name on device i that connects
    to device j, or None if no connection exists or no interface name is found.
    Diagonal elements are always None.
    """
    num_nodes = len(selected_cycle)
    matrix: list[list[str | None]] = [
        [None for _ in range(num_nodes)] for _ in range(num_nodes)
    ]

    for i, node_i in enumerate(selected_cycle):
        for j, node_j in enumerate(selected_cycle):
            if i == j:
                continue

            for conn in cycle_digraph.get_all_connections_between(node_i, node_j):
                if isinstance(conn, RDMAConnection):
                    matrix[i][j] = conn.source_rdma_iface
                    break
            else:
                raise ValueError(
                    "Current jaccl backend requires all-to-all RDMA connections"
                )

    return matrix


def _find_connection_ip(
    node_i: NodeId,
    node_j: NodeId,
    cycle_digraph: Topology,
) -> Generator[str, None, None]:
    """Find all IP addresses that connect node i to node j.

    Session edges are skipped (#662): they prove connectivity, but their
    annotated address is the observed remote endpoint of a libp2p
    connection, which for a NAT'd or proxied member is not a dialable
    listener address and must never become a ring/coordinator host.
    """
    for connection in cycle_digraph.get_all_connections_between(node_i, node_j):
        if isinstance(connection, SocketConnection) and not connection.session:
            yield connection.sink_multiaddr.ip_address


def _is_vpn_address(ip: str) -> bool:
    """Whether an address belongs to a VPN/overlay network (Tailscale).

    Detected by ADDRESS, not by the gossiped interface label: utun
    interfaces never appear in ``networksetup`` output so their gossiped
    type is "unknown", and adding a new ``InterfaceType`` literal would
    break old nodes decoding gossiped ``NetworkInterfaceInfo`` in a
    mixed-version rollout. Tailscale assigns from CGNAT ``100.64.0.0/10``
    (IPv4) and ``fd7a:115c:a1e0::/48`` (IPv6).
    """
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv4Address):
        return address in ipaddress.IPv4Network("100.64.0.0/10")
    return address in ipaddress.IPv6Network("fd7a:115c:a1e0::/48")


# Transport ranking for ring placements: VPN/overlay addresses are the LAST
# resort, below even "unknown" — Tailscale exists for reaching a node from
# OUTSIDE its local network and may be DERP-relayed (observed live: a
# kite-pair ring link riding the Dallas relay between two machines on the
# same switch). A pair whose only observed path is the overlay (genuinely
# cross-network placement) still works; a pair with any TB/LAN candidate
# never selects it (#265).
_RING_TRANSPORT_PRIORITY: dict[str, int] = {
    "thunderbolt": 0,
    "maybe_ethernet": 1,
    "ethernet": 2,
    "wifi": 3,
    "unknown": 4,
    "vpn": 5,
}

# RDMA/jaccl coordinator preference (control plane, not the data path).
_JACCL_TRANSPORT_PRIORITY: dict[str, int] = {
    "ethernet": 0,
    "wifi": 1,
    "unknown": 2,
    "maybe_ethernet": 3,
    "thunderbolt": 4,
    "vpn": 5,
}


def _is_routable_rpc_donor_address(ip: str) -> bool:
    """Whether an address may be stamped as an RPC donor endpoint.

    Rejects link-local (169.254/16, fe80::/10) and loopback: a node with more
    than one Thunderbolt interface routes ALL of 169.254/16 out a single
    (lowest metric) port, so a link-local endpoint on the other port dials or
    replies asymmetrically and TCP dies even while ICMP appears fine
    (measured on the Strix pair). Point-to-point links that should carry RPC
    traffic get a static per-link subnet instead.

    Also rejects IPv6 outright: donor endpoints are stamped as bare
    ``host:port`` strings, which is ambiguous for IPv6 and not accepted by
    ``llama-server --rpc``'s endpoint parsing, so stamping one would mint an
    instance whose driver can never load. Bracketed-IPv6 support can be added
    end-to-end if a v6-only fabric ever materializes.
    """
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return (
        address.version == 4 and not address.is_link_local and not address.is_loopback
    )


def _find_ip_prioritised(
    node_id: NodeId,
    other_node_id: NodeId,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    ring: bool,
    require_routable: bool = False,
) -> str | None:
    """Find an IP address between nodes with transport prioritization.

    Candidates are the OBSERVED libp2p connections between the pair, ranked
    by the sink node's gossiped interface type — ring placements prefer the
    fastest local interconnect (TB first), and VPN/overlay addresses rank
    strictly last regardless of gossiped label (see ``_is_vpn_address``).
    ``require_routable`` drops link-local and loopback candidates entirely
    (RPC donor endpoints must be dialable both ways on multi-interface hosts;
    see ``_is_routable_rpc_donor_address``; IPv6 is also rejected because
    endpoints are stamped as bare ``host:port``).

    TODO: Profile and get actual connection speeds.
    """
    ips = list(_find_connection_ip(node_id, other_node_id, cycle_digraph))
    if require_routable:
        ips = [ip for ip in ips if _is_routable_rpc_donor_address(ip)]
    if not ips:
        return None
    other_network = node_network.get(other_node_id, NodeNetworkInfo())
    ip_to_type = {
        iface.ip_address: iface.interface_type for iface in other_network.interfaces
    }
    priority = _RING_TRANSPORT_PRIORITY if ring else _JACCL_TRANSPORT_PRIORITY

    def transport_rank(ip: str) -> int:
        if _is_vpn_address(ip):
            return priority["vpn"]
        return priority.get(ip_to_type.get(ip, "unknown"), priority["unknown"])

    return min(ips, key=transport_rank)


def get_mlx_ring_hosts_by_node(
    selected_cycle: Cycle,
    cycle_digraph: Topology,
    ephemeral_port: int,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, list[Host]]:
    """Generate per-node host lists for MLX ring backend.

    Each node gets a list where:
    - Self position: Host(ip="0.0.0.0", port=ephemeral_port)
    - Left/right neighbors: actual connection IPs
    - Non-neighbors: Host(ip="198.51.100.1", port=0) placeholder (RFC 5737 TEST-NET-2)
    """
    world_size = len(selected_cycle)
    if world_size == 0:
        return {}

    hosts_by_node: dict[NodeId, list[Host]] = {}

    for rank, node_id in enumerate(selected_cycle):
        left_rank = (rank - 1) % world_size
        right_rank = (rank + 1) % world_size

        hosts_for_node: list[Host] = []

        for idx, other_node_id in enumerate(selected_cycle):
            if idx == rank:
                hosts_for_node.append(Host(ip="0.0.0.0", port=ephemeral_port))
                continue

            if idx not in {left_rank, right_rank}:
                # Placeholder IP from RFC 5737 TEST-NET-2
                hosts_for_node.append(Host(ip="198.51.100.1", port=0))
                continue

            connection_ip = _find_ip_prioritised(
                node_id, other_node_id, cycle_digraph, node_network, ring=True
            )
            if connection_ip is None:
                raise ValueError(
                    "MLX ring backend requires connectivity between neighbouring nodes"
                )

            hosts_for_node.append(Host(ip=connection_ip, port=ephemeral_port))

        hosts_by_node[node_id] = hosts_for_node

    return hosts_by_node


def get_mlx_jaccl_coordinators(
    coordinator: NodeId,
    coordinator_port: int,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, str]:
    """Get the coordinator addresses for MLX JACCL (rank 0 device).

    Select an IP address that each node can reach for the rank 0 node. Returns
    address in format "X.X.X.X:PORT" per node.
    """
    logger.debug(f"Selecting coordinator: {coordinator}")

    def get_ip_for_node(n: NodeId) -> str:
        if n == coordinator:
            return "0.0.0.0"

        ip = _find_ip_prioritised(
            n, coordinator, cycle_digraph, node_network, ring=False
        )
        if ip is not None:
            return ip

        raise ValueError(
            "Current jaccl backend requires all participating devices to be able to communicate"
        )

    return {
        n: f"{get_ip_for_node(n)}:{coordinator_port}"
        for n in cycle_digraph.list_nodes()
    }
