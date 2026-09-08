"""Ordinary inference must reach executable capacity and terminate failed pins."""

import anyio
import pytest

from skulk.master.main import Master, text_generation_instances
from skulk.routing.router import get_node_id_keypair
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.commands import (
    CommandId,
    ForwarderCommand,
    ForwarderDownloadCommand,
    TextGeneration,
)
from skulk.shared.types.common import ModelId, NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    TaskCreated,
    TaskFailed,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.tasks import TaskId, TaskStatus
from skulk.shared.types.tasks import TextGeneration as TextGenerationTask
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    ShardAssignments,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import channel


def _state(steward_status: RunnerStatus, ordinary_status: RunnerStatus) -> State:
    runner_id = RunnerId("runner-a")
    model_id = ModelId("org/model")
    card = ModelCard(
        model_id=model_id,
        storage_size=Memory.from_mb(128),
        n_layers=4,
        hidden_size=256,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=4,
        n_layers=4,
    )
    ordinary = MlxRingInstance(
        instance_id=InstanceId("ordinary"),
        shard_assignments=ShardAssignments(
            model_id=model_id,
            node_to_runner={NodeId("node-a"): runner_id},
            runner_to_shard={runner_id: shard},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    runner = RunnerId("steward-runner")
    shard = next(iter(ordinary.shard_assignments.runner_to_shard.values()))
    steward = ordinary.model_copy(
        update={
            "instance_id": InstanceId("steward"),
            "system_role": "steward",
            "shard_assignments": ShardAssignments(
                model_id=ModelId("org/model"),
                node_to_runner={NodeId("node-a"): runner},
                runner_to_shard={runner: shard},
            ),
        }
    )
    return State(
        instances={steward.instance_id: steward, ordinary.instance_id: ordinary},
        runners={runner: steward_status, RunnerId("runner-a"): ordinary_status},
    )


@pytest.mark.parametrize(
    "steward_status",
    [
        RunnerFailed(),
        RunnerLoading(),
        RunnerReady(),
        RunnerRunning(),
        RunnerShutdown(),
        RunnerShuttingDown(),
    ],
)
async def test_master_routes_ordinary_request_to_ready_ordinary_instance(
    steward_status: RunnerStatus,
) -> None:
    """Exercise actual command processing with the failed-steward live topology."""
    events = await _dispatch(_state(steward_status, RunnerReady()), None)
    assert len(events) == 1 and isinstance(events[0], TaskCreated)
    assert events[0].task.instance_id == InstanceId("ordinary")
    assert events[0].task.task_status == TaskStatus.Pending


@pytest.mark.parametrize("state", [_state(RunnerFailed(), RunnerFailed()), State()])
async def test_unavailable_model_emits_terminal_task(state: State) -> None:
    """No eligible placement must fail the caller rather than log and abandon it."""
    events = await _dispatch(state, None, failed=True)
    assert isinstance(events[0], TaskCreated)
    assert events[0].task.task_status == TaskStatus.Failed
    assert (
        isinstance(events[1], TaskFailed)
        and events[1].error_type == "instance_unavailable"
    )
    assert events[1].task_id == events[0].task.task_id


async def test_failed_explicit_pin_does_not_fall_back_to_ready_sibling() -> None:
    """A steward pin retains its identity and fails despite other ready capacity."""
    events = await _dispatch(
        _state(RunnerFailed(), RunnerReady()), InstanceId("steward"), failed=True
    )
    assert isinstance(events[0], TaskCreated) and events[
        0
    ].task.instance_id == InstanceId("steward")
    assert isinstance(events[1], TaskFailed)


def test_ready_steward_precedes_cold_ordinary_capacity() -> None:
    """Readiness remains more important than role while ordinary capacity loads."""
    assert text_generation_instances(
        _state(RunnerReady(), RunnerLoading()), ModelId("org/model")
    ) == [InstanceId("steward"), InstanceId("ordinary")]


def test_cold_ordinary_capacity_can_still_queue() -> None:
    """Preserve existing queuing during initial model loading without failed ranks."""
    assert text_generation_instances(
        _state(RunnerFailed(), RunnerLoading()), ModelId("org/model")
    ) == [InstanceId("ordinary")]


async def _dispatch(
    state: State, target: InstanceId | None, *, failed: bool = False
) -> list[Event]:
    node_id = NodeId(get_node_id_keypair().to_node_id())
    sender, receiver = channel[ForwarderCommand]()
    events, out = channel[Event]()
    global_sender, _ = channel[GlobalForwarderEvent]()
    _, local = channel[LocalForwarderEvent]()
    sync_sender, sync_receiver = channel[StateSyncMessage]()
    downloads, _ = channel[ForwarderDownloadCommand]()
    master = Master(
        node_id,
        SessionId(master_node_id=node_id, election_clock=0),
        event_sender=events,
        global_event_sender=global_sender,
        local_event_receiver=local,
        command_receiver=receiver,
        state_sync_receiver=sync_receiver,
        state_sync_sender=sync_sender,
        download_command_sender=downloads,
    )
    master.state = state
    command = TextGeneration(
        target_instance_id=target,
        task_params=TextGenerationTaskParams(
            model=ModelId("org/model"),
            input=[InputMessage(role="user", content="2 + 2?")],
        ),
    )
    captured: list[Event] = []
    async with anyio.create_task_group() as group:
        group.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage] - actual dispatch boundary
        await sender.send(ForwarderCommand(origin=SystemId("API"), command=command))
        with anyio.fail_after(2):
            captured = await out.receive_at_least(2 if failed else 1)
        group.cancel_scope.cancel()
    first = captured[0]
    assert isinstance(first, TaskCreated)
    assert isinstance(first.task, TextGenerationTask)
    assert first.task.command_id == command.command_id
    return captured


async def test_completed_tasks_do_not_outweigh_active_inference() -> None:
    """Retained completed work must not route new work onto the busier sibling."""
    state = _state(RunnerReady(), RunnerReady())
    other = state.instances[InstanceId("steward")].model_copy(
        update={"system_role": None}
    )
    tasks = {}
    for number in range(6):
        identifier = TaskId()
        tasks[identifier] = TextGenerationTask(
            task_id=identifier,
            command_id=CommandId(),
            instance_id=InstanceId("ordinary") if number else other.instance_id,
            task_status=TaskStatus.Complete if number else TaskStatus.Running,
            task_params=TextGenerationTaskParams(model=ModelId("org/model"), input=[]),
        )
    state = state.model_copy(
        update={
            "instances": {**state.instances, other.instance_id: other},
            "tasks": tasks,
        }
    )
    events = await _dispatch(state, None)
    assert isinstance(events[0], TaskCreated)
    assert events[0].task.instance_id == InstanceId("ordinary")


async def test_cold_explicit_pin_keeps_its_target() -> None:
    """A loading steward pin must not be silently redirected to ready ordinary work."""
    events = await _dispatch(
        _state(RunnerLoading(), RunnerReady()), InstanceId("steward")
    )
    assert isinstance(events[0], TaskCreated)
    assert events[0].task.instance_id == InstanceId("steward")
    assert events[0].task.task_status == TaskStatus.Pending
