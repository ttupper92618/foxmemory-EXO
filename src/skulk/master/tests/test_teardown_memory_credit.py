# pyright: reportPrivateUsage=false
"""Recently-freed memory credit bookkeeping for placement admission (#314).

The speculative credit is disabled by default because live MLX teardown can lag
``InstanceDeleted``: crediting memory back before the worker actually releases it
over-admits the next placement and forces the worker's local pre-load guard to
refuse. These tests keep the bookkeeping explicit and verify the default
placement inputs stay grounded in observed telemetry.
"""

import asyncio

import pytest

from skulk.master.main import Master
from skulk.routing.router import get_node_id_keypair
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.commands import (
    CreateInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from skulk.shared.types.common import NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    InstanceFailureRecorded,
    LocalForwarderEvent,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import (
    AcceleratorMetrics,
    MemoryUsage,
    NodeResources,
    SystemPerformanceProfile,
)
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.worker.instances import (
    InstanceId,
    MlxRingInstance,
    ShardAssignments,
)
from skulk.shared.types.worker.runners import RunnerId
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import channel


def _make_master() -> Master:
    node_id = NodeId(get_node_id_keypair().to_node_id())
    session_id = SessionId(master_node_id=node_id, election_clock=0)
    ge_sender, _ = channel[GlobalForwarderEvent]()
    _, co_receiver = channel[ForwarderCommand]()
    _, le_receiver = channel[LocalForwarderEvent]()
    state_sync_sender, state_sync_receiver = channel[StateSyncMessage]()
    fcds, _ = channel[ForwarderDownloadCommand]()
    ev_send, _ = channel[Event]()
    return Master(
        node_id,
        session_id,
        event_sender=ev_send,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=co_receiver,
        state_sync_receiver=state_sync_receiver,
        state_sync_sender=state_sync_sender,
        download_command_sender=fcds,
    )


def _instance(node_id: NodeId) -> tuple[MlxRingInstance, ModelCard]:
    card = ModelCard(
        model_id=ModelId("org/m"),
        storage_size=Memory.from_gb(8.0),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    runner_id = RunnerId()
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner_id: shard},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={node_id: []},
        ephemeral_port=12345,
    )
    return instance, card


def _mem(gb: float) -> MemoryUsage:
    return MemoryUsage(
        ram_total=Memory.from_gb(64.0),
        ram_available=Memory.from_gb(gb),
        swap_total=Memory(),
        swap_available=Memory(),
    )


def test_no_recent_free_leaves_memory_unchanged() -> None:
    master = _make_master()
    node_id = NodeId(get_node_id_keypair().to_node_id())
    master._telemetry_view.node_memory[node_id] = _mem(4.0)
    memory, _vram = master._placement_memory_inputs()
    assert memory[node_id].ram_available.in_gb == 4.0


def test_freed_instance_credit_is_disabled_by_default() -> None:
    master = _make_master()
    node_id = NodeId(get_node_id_keypair().to_node_id())
    instance, card = _instance(node_id)
    # Gossip still shows the loaded memory (deflated availability), but Skulk
    # must not speculate that deletion instantly unwired the model locally.
    master._telemetry_view.node_memory[node_id] = _mem(2.0)

    master._record_freed_instance(instance)
    memory, _vram = master._placement_memory_inputs()

    assert card.model_id == ModelId("org/m")
    assert memory[node_id].ram_available.in_gb == 2.0
    assert node_id not in master._recently_freed_bytes
    # ram_total is never credited (context-ceiling math reads it).
    assert memory[node_id].ram_total.in_gb == 64.0


def test_credit_expires_after_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    import skulk.master.main as master_main

    master = _make_master()
    node_id = NodeId(get_node_id_keypair().to_node_id())
    instance, _card = _instance(node_id)
    master._telemetry_view.node_memory[node_id] = _mem(2.0)

    base = master_main.time.monotonic()
    master._record_freed_instance(instance)
    # Jump past the grace window: the credit must expire and be pruned.
    monkeypatch.setattr(
        master_main.time,
        "monotonic",
        lambda: base + master_main.RECENTLY_FREED_MEMORY_GRACE_SECONDS + 1.0,
    )
    memory, _vram = master._placement_memory_inputs()
    assert memory[node_id].ram_available.in_gb == 2.0
    assert node_id not in master._recently_freed_bytes


def _gpu_master_with_instance() -> tuple[Master, MlxRingInstance, NodeId]:
    master = _make_master()
    node_id = NodeId()
    instance, _card = _instance(node_id)
    instance = instance.model_copy(
        update={
            "shard_assignments": instance.shard_assignments.model_copy(
                update={
                    "runner_to_shard": {
                        runner: shard.model_copy(
                            update={"resolved_backend": "llama_cpp-cuda"}
                        )
                        for runner, shard in instance.shard_assignments.runner_to_shard.items()
                    }
                }
            )
        }
    )
    master._telemetry_view.node_memory[node_id] = _mem(64.0)
    master._telemetry_view.node_resources[node_id] = NodeResources(
        backends=frozenset({"llama_cpp-cuda"})
    )
    master._telemetry_view.node_system[node_id] = SystemPerformanceProfile(
        accelerator=AcceleratorMetrics(
            vendor="nvidia", vram_total_bytes=Memory.from_gb(24).in_bytes
        )
    )
    return master, instance, node_id


async def test_local_creation_reserves_before_index_and_releases_only_after_delete() -> (
    None
):
    """Queued, indexed, and deleted phases cannot expose the same capacity twice."""
    master, instance, node_id = _gpu_master_with_instance()
    _, initial = master._placement_memory_inputs()
    receiver = master.event_sender.clone_receiver()
    created = InstanceCreated(instance=instance)
    await master._queue_control_event(created)
    assert receiver.receive_nowait() == created
    assert instance.instance_id not in master.state.instances
    _, queued = master._placement_memory_inputs()
    assert queued[node_id] < initial[node_id]
    master._apply_indexed_event(IndexedEvent(event=created, idx=0))
    assert master._pending_instance_reservations == {}
    assert master._placement_memory_inputs()[1] == queued
    deleted = InstanceDeleted(instance_id=instance.instance_id)
    await master._queue_control_event(deleted)
    assert master._placement_memory_inputs()[1] == queued
    master._apply_indexed_event(IndexedEvent(event=deleted, idx=1))
    assert master._placement_memory_inputs()[1] == initial


async def test_refused_exact_gpu_placement_retains_failure_identity() -> None:
    """A command accepted at the API cannot fail silently when master capacity changed."""
    master, instance, node_id = _gpu_master_with_instance()
    card = next(iter(instance.shard_assignments.runner_to_shard.values())).model_card
    master._ordered_model_cards[card.model_id] = card
    master._telemetry_view.node_system[node_id] = SystemPerformanceProfile(
        accelerator=AcceleratorMetrics(
            vendor="nvidia", vram_total_bytes=Memory.from_gb(2).in_bytes
        )
    )
    sender = master.command_receiver.clone_sender()
    receiver = master.event_sender.clone_receiver()
    task = asyncio.create_task(master._command_processor())
    try:
        await sender.send(
            ForwarderCommand(
                origin=SystemId(), command=CreateInstance(instance=instance)
            )
        )
        async with asyncio.timeout(2):
            event = await receiver.receive()
        assert isinstance(event, InstanceFailureRecorded)
        assert event.failure.instance_id == instance.instance_id
        assert event.failure.error_code == "placement_failed"
        assert master._pending_instance_reservations == {}
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
