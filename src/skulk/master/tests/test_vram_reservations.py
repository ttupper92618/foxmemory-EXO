"""Committed GPU capacity must constrain admission before load telemetry arrives."""

import pytest

from skulk.master.placement import PlacementError, add_instance_to_placements
from skulk.master.placement_utils import filter_cycles_by_memory, usable_vram_by_node
from skulk.master.tests.conftest import create_node_memory
from skulk.shared.models.memory_estimate import estimate_shard_footprint
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.topology import Topology
from skulk.shared.types.commands import CreateInstance
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import AcceleratorMetrics, SystemPerformanceProfile
from skulk.shared.types.topology import Cycle
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata


def _card(
    storage_gb: float,
    *,
    kv_heads: int | None = None,
    context_length: int = 0,
    gguf_file: str,
) -> ModelCard:
    return ModelCard(
        model_id=ModelId("reservation-model"),
        storage_size=Memory.from_gb(storage_gb),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        num_key_value_heads=kv_heads,
        context_length=context_length,
        gguf_file=gguf_file,
    )


def _instance(
    node_id: NodeId, *, context: int = 8192, backend: str = "llama_cpp-cuda"
) -> MlxRingInstance:
    card = _card(6, kv_heads=4, context_length=262144, gguf_file="model.gguf")
    runner_id = RunnerId()
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            node_to_runner={node_id: runner_id},
            runner_to_shard={
                runner_id: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=32,
                    n_layers=32,
                    resolved_backend=backend,
                )
            },
        ),
        hosts_by_node={},
        ephemeral_port=50000,
        context_token_limit=context,
    )


def _system(total: int, used: int = 0) -> SystemPerformanceProfile:
    return SystemPerformanceProfile(
        accelerator=AcceleratorMetrics(
            vendor="nvidia", vram_total_bytes=total, vram_used_bytes=used
        )
    )


@pytest.mark.parametrize("context", [8192, 32768])
@pytest.mark.parametrize("loaded", [False, True])
def test_gpu_reservations_cover_pending_and_loaded_instances_once(
    context: int, loaded: bool
) -> None:
    """The same commitment applies before and after its bytes reach telemetry."""
    node = NodeId()
    instances = {
        item.instance_id: item
        for item in (_instance(node, context=context), _instance(node))
    }
    reserved = sum(
        estimate_shard_footprint(
            next(iter(item.shard_assignments.runner_to_shard.values())).model_card,
            1.0,
            context_budget=item.context_token_limit or 8192,
        ).in_bytes
        for item in instances.values()
    )
    total = Memory.from_gb(48).in_bytes
    systems = {node: _system(total, reserved if loaded else 0)}
    result = usable_vram_by_node(systems, current_instances=instances)
    assert result[node].in_bytes == int(total * 0.9) - reserved


def test_gpu_reservation_release_still_waits_for_observed_memory() -> None:
    """Removing ownership cannot credit allocations which telemetry still sees."""
    node = NodeId()
    total, free = Memory.from_gb(24).in_bytes, Memory.from_gb(2).in_bytes
    systems = {node: _system(total, total - free)}
    assert usable_vram_by_node(systems, current_instances={})[node].in_bytes == free


def test_overcommitted_gpu_budget_clamps_to_zero() -> None:
    """A restored oversized placement cannot expose negative or fresh capacity."""
    node = NodeId()
    instance = _instance(node, context=262144)
    systems = {node: _system(Memory.from_gb(24).in_bytes)}
    assert usable_vram_by_node(
        systems, current_instances={instance.instance_id: instance}
    )[node] == Memory()


def test_cpu_instance_does_not_reserve_gpu_memory() -> None:
    """A CPU-resolved model on a GPU host consumes the separate system-RAM pool."""
    node = NodeId()
    instance = _instance(node, backend="llama_cpp-cpu")
    systems = {node: _system(Memory.from_gb(24).in_bytes)}
    assert usable_vram_by_node(
        systems, current_instances={instance.instance_id: instance}
    ) == usable_vram_by_node(systems)


def test_committed_gpu_memory_blocks_a_second_load_before_telemetry() -> None:
    """An indexed placement owns its capacity while download is pending."""
    node = NodeId()
    instance = _instance(node, context=65536)
    systems = {node: _system(Memory.from_gb(24).in_bytes)}
    memory = {node: create_node_memory(Memory.from_gb(64).in_bytes)}
    cycles = [Cycle(node_ids=[node])]
    candidate = _card(12, gguf_file="second.gguf")
    before, _ = filter_cycles_by_memory(
        cycles, memory, candidate, node_vram=usable_vram_by_node(systems)
    )
    after, _ = filter_cycles_by_memory(
        cycles,
        memory,
        candidate,
        node_vram=usable_vram_by_node(
            systems, current_instances={instance.instance_id: instance}
        ),
    )
    assert before == cycles
    assert after == []


@pytest.mark.parametrize("known_kv", [False, True])
def test_exact_gpu_placement_cannot_bypass_committed_capacity(known_kv: bool) -> None:
    """Missing KV geometry cannot turn a card fallback into GPU memory admission."""
    node = NodeId()
    instance = _instance(node)
    if not known_kv:
        instance = instance.model_copy(
            update={
                "shard_assignments": instance.shard_assignments.model_copy(
                    update={
                        "runner_to_shard": {
                            runner: shard.model_copy(
                                update={
                                    "model_card": shard.model_card.model_copy(
                                        update={"num_key_value_heads": None}
                                    )
                                }
                            )
                            for runner, shard in instance.shard_assignments.runner_to_shard.items()
                        }
                    }
                )
            }
        )
    with pytest.raises(PlacementError, match="Insufficient GPU memory"):
        add_instance_to_placements(
            CreateInstance(instance=instance),
            Topology(),
            {},
            {node: create_node_memory(Memory.from_gb(64).in_bytes)},
            node_vram={node: Memory.from_gb(1)},
        )
