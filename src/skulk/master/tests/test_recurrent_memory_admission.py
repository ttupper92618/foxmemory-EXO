"""Hybrid-model admission must reserve actual slots, rollback state and KV together."""

import pytest

from skulk.master.placement import (
    PlacementError,
    add_instance_to_placements,
    place_instance,
)
from skulk.master.placement_utils import (
    filter_cycles_by_memory,
    get_shard_assignments_for_llama_rpc,
    usable_vram_by_node,
)
from skulk.master.tests.conftest import create_node_memory, create_node_network
from skulk.master.tests.test_placement import place_instance_command
from skulk.shared.models.gguf_memory import qwen35_cache_geometry
from skulk.shared.models.llama_server_settings import LlamaServerSettings
from skulk.shared.models.memory_estimate import (
    estimate_recurrent_cache_bytes,
    estimate_shard_footprint,
    instance_context_token_limit,
    per_token_kv_bytes,
)
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    PlacementCardConfig,
    RuntimeCapabilityCardConfig,
)
from skulk.shared.models.tests.test_gguf_memory import metadata
from skulk.shared.topology import Topology
from skulk.shared.types.commands import CreateInstance
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import (
    AcceleratorMetrics,
    NodeResources,
    SystemPerformanceProfile,
)
from skulk.shared.types.topology import Cycle
from skulk.shared.types.worker.instances import (
    InstanceId,
    LlamaRpcInstance,
    MlxRingInstance,
)
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata


def hybrid_instance(node: NodeId, slots: int = 16) -> MlxRingInstance:
    """Build a synthetic admission input with source-derived tensor dimensions."""
    card = ModelCard(
        model_id=ModelId("test/hybrid"),
        storage_size=Memory.from_bytes(5868826976),
        n_layers=33,
        hidden_size=4096,
        supports_tensor=False,
        num_key_value_heads=4,
        context_length=262144,
        tasks=[ModelTask.TextGeneration],
        gguf_file="model.gguf",
        gguf_cache_geometry=qwen35_cache_geometry("qwen35", metadata()),
        runtime=RuntimeCapabilityCardConfig(
            served_spec_type="draft_mtp", served_spec_n_max=3
        ),
        placement=PlacementCardConfig(
            compatible_backends=frozenset({"llama_server-cuda"})
        ),
    )
    runner = RunnerId()
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            node_to_runner={node: runner},
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=33,
                    n_layers=33,
                    resolved_backend="llama_server-cuda",
                    llama_server_settings=LlamaServerSettings(parallel_slots=slots),
                )
            },
        ),
        hosts_by_node={},
        ephemeral_port=50000,
        context_token_limit=8192,
    )


@pytest.mark.parametrize("slots", [1, 16, 64])
def test_context_ceiling_and_footprint_agree_at_the_memory_boundary(slots: int) -> None:
    """Context can grow only after fixed per-slot state has been reserved."""
    node = NodeId()
    instance = hybrid_instance(node, slots)
    shard = next(iter(instance.shard_assignments.runner_to_shard.values()))
    available = Memory.from_gb(24)
    limit = instance_context_token_limit(
        instance.shard_assignments,
        {node: Memory.from_gb(64)},
        node_vram={node: available},
    )
    assert limit is not None and 8192 < limit <= 262144
    footprint = estimate_shard_footprint(
        shard.model_card,
        1.0,
        limit,
        resolved_backend=shard.resolved_backend,
        llama_server_settings=shard.llama_server_settings,
    )
    assert footprint <= available
    if limit < 262144:
        assert (
            footprint.in_bytes + per_token_kv_bytes(shard.model_card)
            > available.in_bytes
        )


def test_actual_slot_count_is_reserved_before_load_telemetry() -> None:
    """A pending 64-slot placement cannot expose the same free GPU budget as 16."""
    node = NodeId()
    system = {
        node: SystemPerformanceProfile(
            accelerator=AcceleratorMetrics(
                vendor="nvidia",
                vram_total_bytes=Memory.from_gb(48).in_bytes,
                vram_used_bytes=0,
            )
        )
    }
    ordinary, wide = hybrid_instance(node, 16), hybrid_instance(node, 64)
    ordinary_free = usable_vram_by_node(
        system, current_instances={ordinary.instance_id: ordinary}
    )[node]
    wide_free = usable_vram_by_node(system, current_instances={wide.instance_id: wide})[
        node
    ]
    assert ordinary_free.in_bytes - wide_free.in_bytes == 3 * 3372220416


def test_in_process_and_plain_served_caches_keep_their_distinct_costs() -> None:
    """Disabling speculation removes rollback rows, but does not remove slot state."""
    card = next(
        iter(hybrid_instance(NodeId()).shard_assignments.runner_to_shard.values())
    ).model_card
    assert (
        estimate_recurrent_cache_bytes(card, resolved_backend="llama_cpp-cuda").in_bytes
        == 52690944
    )
    assert per_token_kv_bytes(card, resolved_backend="llama_cpp-cuda") == 32768
    assert (
        estimate_recurrent_cache_bytes(
            card,
            resolved_backend="llama_server-cuda",
            llama_server_settings=LlamaServerSettings(speculation_enabled=False),
        ).in_bytes
        == 843055104
    )


def test_exact_placement_uses_observed_settings_instead_of_caller_settings() -> None:
    """A client cannot submit a cheap one-slot estimate for a 64-slot worker."""
    node = NodeId()
    instance = hybrid_instance(node, 1)
    resources = {
        node: NodeResources(
            backends=frozenset({"llama_server-cuda"}),
            llama_server_settings=LlamaServerSettings(parallel_slots=64),
        )
    }
    with pytest.raises(PlacementError, match="Insufficient GPU memory"):
        add_instance_to_placements(
            CreateInstance(instance=instance),
            Topology(),
            {},
            {node: create_node_memory(Memory.from_gb(64).in_bytes)},
            node_vram={node: Memory.from_gb(12)},
            node_resources=resources,
        )


def test_cycle_filter_uses_the_same_fixed_slot_cost() -> None:
    """Auto placement rejects a wide instance before selecting an impossible cycle."""
    node = NodeId()
    card = next(
        iter(hybrid_instance(node).shard_assignments.runner_to_shard.values())
    ).model_card
    memory = {node: create_node_memory(Memory.from_gb(64).in_bytes)}
    for slots, accepted in [(16, True), (64, False)]:
        cycles, _ = filter_cycles_by_memory(
            [Cycle(node_ids=[node])],
            memory,
            card,
            node_vram={node: Memory.from_gb(12)},
            resolved_backends={node: "llama_server-cuda"},
            llama_server_settings={node: LlamaServerSettings(parallel_slots=slots)},
        )
        assert bool(cycles) is accepted


def test_rpc_captures_only_the_driver_serving_settings() -> None:
    """RPC donors expose memory, not independent speculative server contexts."""
    driver, donor = NodeId(), NodeId()
    card = next(
        iter(hybrid_instance(driver).shard_assignments.runner_to_shard.values())
    ).model_card
    assignments = get_shard_assignments_for_llama_rpc(
        card, Cycle(node_ids=[driver, donor]), driver
    )
    instance = LlamaRpcInstance(
        instance_id=InstanceId(),
        shard_assignments=assignments,
        driver_node=driver,
        donor_endpoints={donor: "localhost:50000"},
        context_token_limit=8192,
    )
    settings = LlamaServerSettings(parallel_slots=64)
    result = add_instance_to_placements(
        CreateInstance(instance=instance),
        Topology(),
        {},
        {
            node: create_node_memory(Memory.from_gb(64).in_bytes)
            for node in [driver, donor]
        },
        node_vram={node: Memory.from_gb(24) for node in [driver, donor]},
        node_resources={
            driver: NodeResources(
                backends=frozenset({"llama_server-cuda"}),
                llama_server_settings=settings,
            ),
            donor: NodeResources(backends=frozenset({"llama_server-cuda"})),
        },
    )[instance.instance_id].shard_assignments
    assert (
        result.runner_to_shard[result.node_to_runner[driver]].llama_server_settings
        == settings
    )
    assert (
        result.runner_to_shard[result.node_to_runner[donor]].llama_server_settings
        is None
    )


def test_unresolved_exact_hybrid_placement_requires_backend_observation() -> None:
    """Legacy RAM-only admission cannot approve a hybrid with unknown serving slots."""
    node = NodeId()
    instance = hybrid_instance(node, 1)
    assignments = instance.shard_assignments
    instance = instance.model_copy(
        update={
            "shard_assignments": assignments.model_copy(
                update={
                    "runner_to_shard": {
                        runner: shard.model_copy(update={"resolved_backend": None})
                        for runner, shard in assignments.runner_to_shard.items()
                    }
                }
            )
        }
    )
    with pytest.raises(PlacementError, match="Backend telemetry.*recurrent"):
        add_instance_to_placements(
            CreateInstance(instance=instance),
            Topology(),
            {},
            {node: create_node_memory(Memory.from_gb(64).in_bytes)},
        )


def test_auto_hybrid_placement_waits_for_backend_observation() -> None:
    """A known topology and RAM reading cannot substitute for serving controls."""
    node = NodeId()
    topology = Topology()
    topology.add_node(node)
    card = next(
        iter(hybrid_instance(node).shard_assignments.runner_to_shard.values())
    ).model_card
    with pytest.raises(PlacementError, match="Backend telemetry.*recurrent"):
        place_instance(
            place_instance_command(card),
            topology,
            {},
            {node: create_node_memory(Memory.from_gb(64).in_bytes)},
            {node: create_node_network()},
        )
