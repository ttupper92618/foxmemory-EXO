import pytest

import skulk.master.placement as placement_module
import skulk.shared.models.model_cards as model_cards_module
from skulk.master.placement import (
    PlacementError,
    PlacementInfoPendingError,
    PlacementModelCardIdentityError,
    add_instance_to_placements,
    fallback_command_for_refused_instance,
    get_transition_events,
    place_instance,
    replacement_command_for_download_failed_instance,
    replacement_command_for_refused_instance,
)
from skulk.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_rdma_connection,
    create_socket_connection,
)
from skulk.shared.models.memory_estimate import KV_CONTEXT_BUDGET_TOKENS
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    PlacementCardConfig,
    VisionCardConfig,
)
from skulk.shared.models.registry import (
    RegistryCapabilityClaim,
    RegistryEngineSupportClaim,
)
from skulk.shared.models.remote_code_approval import (
    remote_code_approval_required as actual_remote_code_approval_required,
)
from skulk.shared.topology import Topology
from skulk.shared.types.commands import CreateInstance, PlaceInstance
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import (
    InstanceCreated,
    InstanceDeleted,
    TaskStatusUpdated,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.multiaddr import Multiaddr
from skulk.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
    NodeResources,
)
from skulk.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.topology import Connection, SocketConnection
from skulk.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadProgressData,
)
from skulk.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata, Sharding


@pytest.fixture
def instance() -> Instance:
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("test-model"), runner_to_shard={}, node_to_runner={}
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


@pytest.fixture
def model_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_gb(1),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def place_instance_command(model_card: ModelCard) -> PlaceInstance:
    return PlaceInstance(
        command_id=CommandId(),
        model_card=model_card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )


@pytest.mark.parametrize(
    "available_memory,total_layers,expected_layers",
    [
        ((5.0, 5.0, 10.0), 12, (3, 3, 6)),
        ((5.0, 5.0, 5.0), 12, (4, 4, 4)),
        ((3.12, 4.68, 10.92), 12, (2, 3, 7)),
    ],
)
def test_get_instance_placements_create_instance(
    available_memory: tuple[float, float, float],
    total_layers: int,
    expected_layers: tuple[int, int, int],
    model_card: ModelCard,
):
    # arrange
    model_card.n_layers = total_layers
    # 65% of the cycle's total memory: large enough that no 2-node cycle or
    # singleton passes the per-node headroom check (forcing the 3-node
    # placement this test asserts on), small enough that the 3-node cycle
    # fits under the 1.30x weight-overhead factor. The layer split itself only
    # depends on the memory fractions, not this size.
    model_card.storage_size = Memory.from_gb(sum(available_memory) * 0.65)
    topology = Topology()

    cic = place_instance_command(model_card)
    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()

    # fully connected (directed) between the 3 nodes
    conn_a_b = Connection(
        source=node_id_a, sink=node_id_b, edge=create_socket_connection(1)
    )
    conn_b_c = Connection(
        source=node_id_b, sink=node_id_c, edge=create_socket_connection(2)
    )
    conn_c_a = Connection(
        source=node_id_c, sink=node_id_a, edge=create_socket_connection(3)
    )
    conn_c_b = Connection(
        source=node_id_c, sink=node_id_b, edge=create_socket_connection(4)
    )
    conn_a_c = Connection(
        source=node_id_a, sink=node_id_c, edge=create_socket_connection(5)
    )
    conn_b_a = Connection(
        source=node_id_b, sink=node_id_a, edge=create_socket_connection(6)
    )

    node_memory = {
        node_id_a: create_node_memory(Memory.from_gb(available_memory[0]).in_bytes),
        node_id_b: create_node_memory(Memory.from_gb(available_memory[1]).in_bytes),
        node_id_c: create_node_memory(Memory.from_gb(available_memory[2]).in_bytes),
    }
    node_network = {
        node_id_a: create_node_network(),
        node_id_b: create_node_network(),
        node_id_c: create_node_network(),
    }
    topology.add_node(node_id_a)
    topology.add_node(node_id_b)
    topology.add_node(node_id_c)
    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_c_b)
    topology.add_connection(conn_a_c)
    topology.add_connection(conn_b_a)

    # act
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    # assert
    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    assert instance_id == InstanceId(str(cic.command_id))
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == model_card.model_id

    runner_id_a = instance.shard_assignments.node_to_runner[node_id_a]
    runner_id_b = instance.shard_assignments.node_to_runner[node_id_b]
    runner_id_c = instance.shard_assignments.node_to_runner[node_id_c]

    shard_a = instance.shard_assignments.runner_to_shard[runner_id_a]
    shard_b = instance.shard_assignments.runner_to_shard[runner_id_b]
    shard_c = instance.shard_assignments.runner_to_shard[runner_id_c]

    assert shard_a.end_layer - shard_a.start_layer == expected_layers[0]
    assert shard_b.end_layer - shard_b.start_layer == expected_layers[1]
    assert shard_c.end_layer - shard_c.start_layer == expected_layers[2]

    shards = [shard_a, shard_b, shard_c]
    shards_sorted = sorted(shards, key=lambda s: s.start_layer)
    assert shards_sorted[0].start_layer == 0
    assert shards_sorted[-1].end_layer == total_layers


def test_get_instance_placements_one_node_exact_fit_is_rejected() -> None:
    """Weights exactly equal to available memory must be refused.

    An exact fit leaves nothing for KV cache, activations, or the runner
    itself — admitting it produces the silent-thrash failure observed in
    the 2026-06-05 launch smoke (12-token prefill in 1230s). The per-node
    headroom check turns that into an explicit placement error.
    """
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(Memory.from_gb(8).in_bytes)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_gb(8),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )
    with pytest.raises(ValueError, match="No candidate cycle fits"):
        place_instance(cic, topology, {}, node_memory, node_network)


def test_get_instance_placements_one_node_fits_with_extra_memory() -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(Memory.from_gb(8).in_bytes)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_gb(5),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]
    assert instance.shard_assignments.model_id == "test-model"
    assert len(instance.shard_assignments.node_to_runner) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1
    assert len(instance.shard_assignments.runner_to_shard) == 1


def test_placement_stamps_context_token_limit() -> None:
    # #279 slice 2: the master computes the context-admission ceiling once at
    # placement time and stamps it onto the instance, so every rank reads the
    # identical value instead of recomputing from (now telemetry-plane) memory.
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(Memory.from_gb(8).in_bytes)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_gb(5),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            context_length=4096,
        ),
    )
    placements = place_instance(cic, topology, {}, node_memory, node_network)
    instance = next(iter(placements.values()))
    # A card-advertised context_length makes the ceiling enforceable; the stamp
    # is present and never exceeds the advertised limit.
    assert instance.context_token_limit is not None
    assert instance.context_token_limit <= 4096


def test_placement_stamps_resolved_backend() -> None:
    # #330: the master resolves each node's backend at placement time and stamps
    # it on the shard, so the worker dispatches from replicated state instead of
    # re-probing. A node advertising {"mlx"} for an mlx card -> resolved "mlx".
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(Memory.from_gb(8).in_bytes)}
    node_network = {node_id: create_node_network()}
    node_resources = {
        node_id: NodeResources(
            backends=frozenset({"mlx", "mlx-metal"}), participation="full"
        )
    }
    cic = place_instance_command(
        ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_gb(5),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )
    placements = place_instance(
        cic, topology, {}, node_memory, node_network, node_resources=node_resources
    )
    instance = next(iter(placements.values()))
    shard = next(iter(instance.shard_assignments.runner_to_shard.values()))
    assert shard.resolved_backend == "mlx"


def test_placement_leaves_resolved_backend_none_without_node_resources() -> None:
    # When the master has no resources entry for the node (gossip warming up), it
    # cannot resolve a backend; resolved_backend stays None and the worker falls
    # back to its local probe.
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(Memory.from_gb(8).in_bytes)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_gb(5),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )
    placements = place_instance(cic, topology, {}, node_memory, node_network)
    instance = next(iter(placements.values()))
    shard = next(iter(instance.shard_assignments.runner_to_shard.values()))
    assert shard.resolved_backend is None


def test_get_instance_placements_one_node_not_fit() -> None:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    node_memory = {node_id: create_node_memory(Memory.from_gb(8).in_bytes)}
    node_network = {node_id: create_node_network()}
    cic = place_instance_command(
        model_card=ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_gb(9),
            n_layers=10,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
    )

    with pytest.raises(ValueError, match="No candidate cycle fits"):
        place_instance(cic, topology, {}, node_memory, node_network)


def _two_node_topology():
    """Two-node topology where either node can host a placement on its own.

    Used by the excluded-nodes tests to verify the planner picks the alternate
    node when one is excluded, and refuses to place when *all* candidates are
    excluded. Return type is left loose intentionally — the values match the
    `place_instance` signature without us recomputing the helper return types.
    """
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    # Each node has enough memory to host the small model on its own.
    node_memory = {
        node_a: create_node_memory(Memory.from_gb(8).in_bytes),
        node_b: create_node_memory(Memory.from_gb(8).in_bytes),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    return topology, node_a, node_b, node_memory, node_network


def _small_model_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_mb(500),
        n_layers=10,
        hidden_size=1000,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def test_excluding_one_node_routes_placement_to_the_other() -> None:
    """The planner must pick a candidate cycle that doesn't touch any excluded
    node when an alternative exists."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        excluded_nodes={node_a},
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == {node_b}


def test_excluding_all_candidate_nodes_fails_to_place() -> None:
    """If every cycle that would otherwise satisfy the placement is excluded,
    the planner raises rather than picking an excluded node anyway."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    with pytest.raises(ValueError, match="touch an excluded node"):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            excluded_nodes={node_a, node_b},
        )


def test_empty_exclusion_set_preserves_default_behavior() -> None:
    """Defensive: passing an empty set must not change which placements are
    legal — the filter should be a no-op."""
    topology, _node_a, _node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    without_filter = place_instance(command, topology, {}, node_memory, node_network)
    with_empty_filter = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        excluded_nodes=set(),
    )

    # Both calls must produce equivalent placement sets (same node count, same
    # model). We don't assert exact node identity because the planner may pick
    # either node when both qualify.
    assert len(without_filter) == 1
    assert len(with_empty_filter) == 1
    assert (
        next(iter(without_filter.values())).shard_assignments.model_id
        == next(iter(with_empty_filter.values())).shard_assignments.model_id
    )


def test_min_nodes_above_cluster_size_is_a_hard_error() -> None:
    """min_nodes greater than the number of known nodes can never succeed —
    hard PlacementError, no retry semantics."""
    topology, _node_a, _node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())
    command.min_nodes = 3

    with pytest.raises(PlacementError, match="min_nodes=3 is impossible"):
        place_instance(command, topology, {}, node_memory, node_network)


def test_unconnected_nodes_at_min_nodes_reports_info_pending() -> None:
    """Enough nodes exist but no connecting edges yet: right after cluster
    formation the connection edges lag node identities by a few gossip
    rounds, so this must surface as info-pending (retry shortly), not as a
    hard topology error — and never as the old 'insufficient memory' lie."""
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)  # no connections gossiped yet
    node_memory = {
        node_a: create_node_memory(Memory.from_gb(8).in_bytes),
        node_b: create_node_memory(Memory.from_gb(8).in_bytes),
    }
    node_network = {node_a: create_node_network(), node_b: create_node_network()}
    command = place_instance_command(_small_model_card())
    command.min_nodes = 2

    with pytest.raises(PlacementInfoPendingError, match="retry shortly"):
        place_instance(command, topology, {}, node_memory, node_network)


def test_missing_node_memory_reports_info_pending() -> None:
    """A connected pair where one node's memory info has not arrived yet is
    the startup race observed on 2026-06-06: it must surface as
    info-pending, not 'insufficient memory'."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )
    command = place_instance_command(_small_model_card())
    command.min_nodes = 2
    del node_memory[node_b]

    with pytest.raises(PlacementInfoPendingError, match="Memory info"):
        place_instance(command, topology, {}, node_memory, node_network)


def test_get_transition_events_no_change(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances = {instance_id: instance}
    target_instances = {instance_id: instance}

    # act
    events = get_transition_events(current_instances, target_instances, {})

    # assert
    assert len(events) == 0


def test_get_transition_events_create_instance(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {}
    target_instances: dict[InstanceId, Instance] = {instance_id: instance}

    # act
    events = get_transition_events(current_instances, target_instances, {})

    # assert
    assert len(events) == 1
    assert isinstance(events[0], InstanceCreated)


def test_get_transition_events_delete_instance(instance: Instance):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}

    # act
    events = get_transition_events(current_instances, target_instances, {})

    # assert
    assert len(events) == 1
    assert isinstance(events[0], InstanceDeleted)
    assert events[0].instance_id == instance_id


def test_placement_selects_leaf_nodes(
    model_card: ModelCard,
):
    # arrange
    topology = Topology()

    # 3 GB: too big for any singleton under the headroom check (largest node
    # has 3 GB available), so the planner must use a 2-node cycle — which is
    # what lets this test observe the leaf-node preference.
    model_card.storage_size = Memory.from_gb(3)

    node_id_a = NodeId()
    node_id_b = NodeId()
    node_id_c = NodeId()
    node_id_d = NodeId()

    node_memory = {
        node_id_a: create_node_memory(Memory.from_gb(2).in_bytes),
        node_id_b: create_node_memory(Memory.from_gb(3).in_bytes),
        node_id_c: create_node_memory(Memory.from_gb(3).in_bytes),
        node_id_d: create_node_memory(Memory.from_gb(2).in_bytes),
    }
    node_network = {
        node_id_a: create_node_network(),
        node_id_b: create_node_network(),
        node_id_c: create_node_network(),
        node_id_d: create_node_network(),
    }

    topology.add_node(node_id_a)
    topology.add_node(node_id_b)
    topology.add_node(node_id_c)
    topology.add_node(node_id_d)

    # Daisy chain topology (directed)
    topology.add_connection(
        Connection(source=node_id_a, sink=node_id_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_b, sink=node_id_a, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_b, sink=node_id_c, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_c, sink=node_id_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_c, sink=node_id_d, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_id_d, sink=node_id_c, edge=create_socket_connection(1))
    )

    cic = place_instance_command(model_card=model_card)

    # act
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    # assert
    assert len(placements) == 1
    instance = list(placements.values())[0]

    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == set((node_id_a, node_id_b)) or assigned_nodes == set(
        (
            node_id_c,
            node_id_d,
        )
    )


def test_tensor_rdma_backend_connectivity_matrix(
    model_card: ModelCard,
):
    # arrange
    topology = Topology()
    model_card.n_layers = 12
    model_card.storage_size = Memory.from_gb(1.5)

    node_a = NodeId()
    node_b = NodeId()
    node_c = NodeId()

    node_memory = {
        node_a: create_node_memory(Memory.from_gb(1).in_bytes),
        node_b: create_node_memory(Memory.from_gb(1).in_bytes),
        node_c: create_node_memory(Memory.from_gb(1).in_bytes),
    }

    ethernet_interface = NetworkInterfaceInfo(
        name="en0",
        ip_address="10.0.0.1",
    )
    ethernet_conn = SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/8000")
    )

    node_network = {
        node_a: NodeNetworkInfo(interfaces=[ethernet_interface]),
        node_b: NodeNetworkInfo(interfaces=[ethernet_interface]),
        node_c: NodeNetworkInfo(interfaces=[ethernet_interface]),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)

    # RDMA connections (directed)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_rdma_connection(3))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_rdma_connection(3))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_c, edge=create_rdma_connection(4))
    )
    topology.add_connection(
        Connection(source=node_c, sink=node_b, edge=create_rdma_connection(4))
    )
    topology.add_connection(
        Connection(source=node_a, sink=node_c, edge=create_rdma_connection(5))
    )
    topology.add_connection(
        Connection(source=node_c, sink=node_a, edge=create_rdma_connection(5))
    )

    # Ethernet connections (directed)
    topology.add_connection(Connection(source=node_a, sink=node_b, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_b, sink=node_c, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_c, sink=node_a, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_a, sink=node_c, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_b, sink=node_a, edge=ethernet_conn))
    topology.add_connection(Connection(source=node_c, sink=node_b, edge=ethernet_conn))

    cic = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        command_id=CommandId(),
        model_card=model_card,
        min_nodes=1,
    )

    # act
    placements = place_instance(cic, topology, {}, node_memory, node_network)

    # assert
    assert len(placements) == 1
    instance_id = list(placements.keys())[0]
    instance = placements[instance_id]

    assert isinstance(instance, MlxJacclInstance)

    assert instance.jaccl_devices is not None
    assert instance.jaccl_coordinators is not None

    matrix = instance.jaccl_devices
    assert len(matrix) == 3
    for i in range(3):
        assert matrix[i][i] is None

    assigned_nodes = list(instance.shard_assignments.node_to_runner.keys())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(assigned_nodes)}

    idx_a = node_to_idx[node_a]
    idx_b = node_to_idx[node_b]
    idx_c = node_to_idx[node_c]

    assert matrix[idx_a][idx_b] == "rdma_en3"
    assert matrix[idx_b][idx_c] == "rdma_en4"
    assert matrix[idx_c][idx_a] == "rdma_en5"

    # Verify coordinators are set for all nodes
    assert len(instance.jaccl_coordinators) == 3
    for node_id in assigned_nodes:
        assert node_id in instance.jaccl_coordinators
        coordinator = instance.jaccl_coordinators[node_id]
        assert ":" in coordinator
        # Rank 0 node should use 0.0.0.0, others should use connection-specific IPs
        if node_id == assigned_nodes[0]:
            assert coordinator.startswith("0.0.0.0:")
        else:
            ip_part = coordinator.split(":")[0]
            assert len(ip_part.split(".")) == 4


def _make_task(
    instance_id: InstanceId,
    status: TaskStatus = TaskStatus.Running,
) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(),
        task_status=status,
        instance_id=instance_id,
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content="hello")],
        ),
    )


def test_get_transition_events_delete_instance_cancels_running_tasks(
    instance: Instance,
):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}
    task = _make_task(instance_id, TaskStatus.Running)
    tasks = {task.task_id: task}

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert – cancellation event should come before the deletion event
    assert len(events) == 2
    assert isinstance(events[0], TaskStatusUpdated)
    assert events[0].task_id == task.task_id
    assert events[0].task_status == TaskStatus.Cancelled
    assert isinstance(events[1], InstanceDeleted)
    assert events[1].instance_id == instance_id


def test_get_transition_events_delete_instance_cancels_pending_tasks(
    instance: Instance,
):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}
    task = _make_task(instance_id, TaskStatus.Pending)
    tasks = {task.task_id: task}

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert
    assert len(events) == 2
    assert isinstance(events[0], TaskStatusUpdated)
    assert events[0].task_id == task.task_id
    assert events[0].task_status == TaskStatus.Cancelled
    assert isinstance(events[1], InstanceDeleted)


def test_get_transition_events_delete_instance_ignores_completed_tasks(
    instance: Instance,
):
    # arrange
    instance_id = InstanceId()
    current_instances: dict[InstanceId, Instance] = {instance_id: instance}
    target_instances: dict[InstanceId, Instance] = {}
    tasks = {
        t.task_id: t
        for t in [
            _make_task(instance_id, TaskStatus.Complete),
            _make_task(instance_id, TaskStatus.Failed),
            _make_task(instance_id, TaskStatus.TimedOut),
            _make_task(instance_id, TaskStatus.Cancelled),
        ]
    }

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert – only the InstanceDeleted event, no cancellations
    assert len(events) == 1
    assert isinstance(events[0], InstanceDeleted)


def test_get_transition_events_delete_instance_cancels_only_matching_tasks(
    instance: Instance,
):
    # arrange
    instance_id_a = InstanceId()
    instance_id_b = InstanceId()
    current_instances: dict[InstanceId, Instance] = {
        instance_id_a: instance,
        instance_id_b: instance,
    }
    # only delete instance A, keep instance B
    target_instances: dict[InstanceId, Instance] = {instance_id_b: instance}

    task_a = _make_task(instance_id_a, TaskStatus.Running)
    task_b = _make_task(instance_id_b, TaskStatus.Running)
    tasks = {task_a.task_id: task_a, task_b.task_id: task_b}

    # act
    events = get_transition_events(current_instances, target_instances, tasks)

    # assert – only task_a should be cancelled
    cancel_events = [e for e in events if isinstance(e, TaskStatusUpdated)]
    delete_events = [e for e in events if isinstance(e, InstanceDeleted)]
    assert len(cancel_events) == 1
    assert cancel_events[0].task_id == task_a.task_id
    assert cancel_events[0].task_status == TaskStatus.Cancelled
    assert len(delete_events) == 1
    assert delete_events[0].instance_id == instance_id_a


def _make_shard_metadata(model_card: ModelCard) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=model_card.n_layers,
        n_layers=model_card.n_layers,
    )


def test_legacy_instance_backfills_context_token_limit_from_card() -> None:
    # An instance hydrated without a stamped ceiling (pre-#279 slice 2 snapshot
    # / event / older CreateInstance) must fall back to the card's context
    # length so the #145 admission guard still applies (review catch on #292).
    node_id = NodeId()
    runner_id = RunnerId()
    card = ModelCard(
        model_id=ModelId("legacy-model"),
        storage_size=Memory.from_gb(1),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        context_length=8192,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("legacy-model"),
            runner_to_shard={runner_id: _make_shard_metadata(card)},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
        # context_token_limit deliberately omitted (the legacy/None case)
    )
    assert instance.context_token_limit == 8192


def test_legacy_gguf_instance_backfill_clamped_to_budget_floor() -> None:
    # A legacy GGUF instance hydrated without a stamped ceiling must NOT backfill
    # to the card's large advertised context: the served llama.cpp engine loads
    # the whole KV cache up front from this value, so a 128k backfill would
    # preallocate a fictitious window and OOM the node on load. The backfill is
    # clamped to the KV_CONTEXT_BUDGET_TOKENS floor for gguf. (P1 review, #585.)
    node_id = NodeId()
    runner_id = RunnerId()
    card = ModelCard(
        model_id=ModelId("legacy-gguf"),
        storage_size=Memory.from_gb(1),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        context_length=131072,
        gguf_file="legacy-Q4_K_M.gguf",
    )
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("legacy-gguf"),
            runner_to_shard={runner_id: _make_shard_metadata(card)},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    assert instance.context_token_limit == KV_CONTEXT_BUDGET_TOKENS


def test_legacy_gguf_instance_backfill_floor_when_context_length_unknown() -> None:
    # A gguf card's context_length is best-effort and can be 0 (unknown). A legacy
    # gguf instance must STILL get a ceiling (the runner preallocates KV at the
    # floor via serving_n_ctx), so it backfills to KV_CONTEXT_BUDGET_TOKENS rather
    # than None -- else the API would admit requests larger than the runner serves.
    node_id = NodeId()
    runner_id = RunnerId()
    card = ModelCard(
        model_id=ModelId("legacy-gguf-noctx"),
        storage_size=Memory.from_gb(1),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        gguf_file="legacy-Q4_K_M.gguf",
        # context_length omitted -> defaults to 0 (unknown)
    )
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("legacy-gguf-noctx"),
            runner_to_shard={runner_id: _make_shard_metadata(card)},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    assert instance.context_token_limit == KV_CONTEXT_BUDGET_TOKENS


@pytest.mark.parametrize(
    "requested,expected",
    [(512, 512), (4096, 4096), (999_999, 4096), (0, None), (-1, None)],
)
def test_create_instance_restamps_context_token_limit(
    requested: int, expected: int | None
) -> None:
    # The exact-control POST /instance path must stamp the master's
    # memory-derived ceiling too (#292 review) — runners trust the stamped
    # field now, so a client-supplied placement can't smuggle in an inflated
    # ceiling and reach MLX past the safe limit.
    node_id = NodeId()
    runner_id = RunnerId()
    card = ModelCard(
        model_id=ModelId("exact-model"),
        storage_size=Memory.from_gb(1),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        context_length=4096,
    )
    client_instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=ModelId("exact-model"),
            runner_to_shard={runner_id: _make_shard_metadata(card)},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
        context_token_limit=requested,
    )
    command = CreateInstance(command_id=CommandId(), instance=client_instance)
    node_memory = {node_id: create_node_memory(Memory.from_gb(8).in_bytes)}
    if expected is None:
        with pytest.raises(PlacementError, match="must be positive"):
            add_instance_to_placements(command, Topology(), {}, node_memory)
        return
    result = add_instance_to_placements(command, Topology(), {}, node_memory)
    stamped = next(iter(result.values())).context_token_limit
    assert stamped == expected


def test_exact_served_placement_preserves_requested_window() -> None:
    """A generous GPU preview cannot inflate a caller's load-time KV allocation."""
    node_id, runner_id = NodeId(), RunnerId()
    card = ModelCard(
        model_id=ModelId("bounded-gguf"),
        storage_size=Memory.from_gb(6),
        n_layers=32,
        hidden_size=4096,
        num_key_value_heads=4,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        gguf_file="model.gguf",
        context_length=262144,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            node_to_runner={node_id: runner_id},
            runner_to_shard={
                runner_id: _make_shard_metadata(card).model_copy(
                    update={"resolved_backend": "llama_cpp-cuda"}
                )
            },
        ),
        hosts_by_node={},
        ephemeral_port=50000,
        context_token_limit=8192,
    )
    node_memory = {node_id: create_node_memory(Memory.from_gb(64).in_bytes)}
    result = add_instance_to_placements(
        CreateInstance(command_id=CommandId(), instance=instance),
        Topology(),
        {},
        node_memory,
        node_vram={node_id: Memory.from_gb(22)},
    )
    assert result[instance.instance_id].context_token_limit == 8192


def test_create_instance_reserves_pinned_projector_on_exact_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact-control path cannot spend projector bytes on KV admission."""

    node_id = NodeId("served-driver")
    runner_id = RunnerId("served-runner")
    projector_size = Memory.from_gb(1).in_bytes
    card = ModelCard(
        model_id=ModelId("org/served-vision"),
        storage_size=Memory.from_gb(4),
        n_layers=10,
        hidden_size=30,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        gguf_file="model-Q4_K_M.gguf",
        source_revision="a" * 40,
        vision=VisionCardConfig(
            projector_file="mmproj-F16.gguf",
            projector_size=projector_size,
        ),
    )
    client_instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner_id: _make_shard_metadata(card)},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    captured: dict[str, object] = {}

    def fake_context_limit(*_args: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 2048

    monkeypatch.setattr(
        placement_module,
        "instance_context_token_limit",
        fake_context_limit,
    )

    add_instance_to_placements(
        CreateInstance(command_id=CommandId(), instance=client_instance),
        Topology(),
        {},
        {node_id: create_node_memory(Memory.from_gb(8).in_bytes)},
    )

    assert captured["fixed_memory_by_node"] == {
        node_id: Memory.from_bytes(projector_size)
    }


def test_placement_prefers_cycle_with_downloaded_model(
    model_card: ModelCard,
) -> None:
    """When two cycles are otherwise equal, prefer the one with the model already downloaded."""
    topology = Topology()

    model_card.storage_size = Memory.from_mb(500)

    node_a = NodeId()
    node_b = NodeId()

    node_memory = {
        node_a: create_node_memory(Memory.from_gb(2).in_bytes),
        node_b: create_node_memory(Memory.from_gb(2).in_bytes),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)
    # No connections between them — two single-node cycles

    shard_meta = _make_shard_metadata(model_card)

    # node_b has the model fully downloaded, node_a does not
    download_status = {
        node_b: [
            DownloadCompleted(
                node_id=node_b,
                shard_metadata=shard_meta,
                total=model_card.storage_size,
            ),
        ],
    }

    cic = place_instance_command(model_card)
    placements = place_instance(
        cic, topology, {}, node_memory, node_network, download_status=download_status
    )

    assert len(placements) == 1
    instance = list(placements.values())[0]
    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == {node_b}


def test_placement_prefers_cycle_with_higher_download_progress(
    model_card: ModelCard,
) -> None:
    """When two cycles are otherwise equal, prefer the one with more download progress."""
    topology = Topology()

    model_card.storage_size = Memory.from_gb(1)

    node_a = NodeId()
    node_b = NodeId()

    node_memory = {
        node_a: create_node_memory(Memory.from_gb(2).in_bytes),
        node_b: create_node_memory(Memory.from_gb(2).in_bytes),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)

    shard_meta = _make_shard_metadata(model_card)

    # node_a: 30% downloaded, node_b: 80% downloaded
    download_status = {
        node_a: [
            DownloadOngoing(
                node_id=node_a,
                shard_metadata=shard_meta,
                download_progress=DownloadProgressData(
                    total=Memory.from_bytes(1000),
                    downloaded=Memory.from_bytes(300),
                    downloaded_this_session=Memory.from_bytes(300),
                    completed_files=0,
                    total_files=1,
                    speed=0.0,
                    eta_ms=0,
                    files={},
                ),
            ),
        ],
        node_b: [
            DownloadOngoing(
                node_id=node_b,
                shard_metadata=shard_meta,
                download_progress=DownloadProgressData(
                    total=Memory.from_bytes(1000),
                    downloaded=Memory.from_bytes(800),
                    downloaded_this_session=Memory.from_bytes(800),
                    completed_files=0,
                    total_files=1,
                    speed=0.0,
                    eta_ms=0,
                    files={},
                ),
            ),
        ],
    }

    cic = place_instance_command(model_card)
    placements = place_instance(
        cic, topology, {}, node_memory, node_network, download_status=download_status
    )

    assert len(placements) == 1
    instance = list(placements.values())[0]
    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    assert assigned_nodes == {node_b}


def test_placement_does_not_prefer_cycle_with_failed_download(
    model_card: ModelCard,
) -> None:
    """A failed download should count as 0% — not preferred over a node with no download history."""
    topology = Topology()

    model_card.storage_size = Memory.from_mb(500)

    node_a = NodeId()
    node_b = NodeId()

    # node_a has slightly more RAM so it would win on the RAM tiebreaker
    node_memory = {
        node_a: create_node_memory(Memory.from_gb(2.001).in_bytes),
        node_b: create_node_memory(Memory.from_gb(2).in_bytes),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }

    topology.add_node(node_a)
    topology.add_node(node_b)

    shard_meta = _make_shard_metadata(model_card)

    # node_b has a failed download — should not be preferred
    download_status = {
        node_b: [
            DownloadFailed(
                node_id=node_b,
                shard_metadata=shard_meta,
                error_message="connection reset",
            ),
        ],
    }

    cic = place_instance_command(model_card)
    placements = place_instance(
        cic, topology, {}, node_memory, node_network, download_status=download_status
    )

    assert len(placements) == 1
    instance = list(placements.values())[0]
    assigned_nodes = set(instance.shard_assignments.node_to_runner.keys())
    # node_a should win on RAM tiebreaker since failed download scores 0.0
    assert assigned_nodes == {node_a}


def test_management_node_is_excluded_from_placement() -> None:
    """A node declaring participation="management" must never receive an
    inference shard, even though it is topologically present and has memory.
    The planner routes to the full-participation node instead (#149)."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_resources={
            node_a: NodeResources(participation="management"),
            node_b: NodeResources(participation="full"),
        },
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert set(instance.shard_assignments.node_to_runner.keys()) == {node_b}


def test_all_management_nodes_fail_to_place() -> None:
    """If every candidate node is management-only, placement raises rather
    than assigning a shard to a non-participating node."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    with pytest.raises(ValueError, match="ineligible for this model"):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            node_resources={
                node_a: NodeResources(participation="management"),
                node_b: NodeResources(participation="management"),
            },
        )


def test_backend_incompatible_node_is_excluded() -> None:
    """A node whose advertised backends do not intersect the card's
    compatible_backends (default {"mlx"}) is filtered out (#149)."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_resources={
            node_a: NodeResources(backends=frozenset({"llama_cpp"})),
            node_b: NodeResources(backends=frozenset({"mlx"})),
        },
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert set(instance.shard_assignments.node_to_runner.keys()) == {node_b}


def test_signed_engine_support_adds_backend_without_card_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exact matrix claim makes an already-known empty-projection card placeable."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    capability = RegistryCapabilityClaim.model_validate(
        {
            "capability_id": "text.generate",
            "scope": "model",
            "status": "observed",
            "source": "upstream_structured",
            "confidence": 1,
        },
        strict=False,
    )
    card = _small_model_card().model_copy(
        update={
            "model_id": ModelId("org/future-model"),
            "source_revision": "a" * 40,
            "registry_card_id": "card_" + "a" * 52,
            "registry_snapshot_id": "snapshot-test",
            "registry_provenance": "agent",
            "registry_architecture": "future_architecture_v1",
            "registry_artifact_format": "safetensors",
            "registry_capability_claims": (capability,),
            "placement": PlacementCardConfig(compatible_backends=frozenset()),
        }
    )
    support = RegistryEngineSupportClaim.model_validate(
        {
            "claim_id": "support_" + "b" * 52,
            "engine": "vllm",
            "engine_build": "vllm@9.9.9",
            "architecture": "future_architecture_v1",
            "artifact_format": "safetensors",
            "artifact_card_id": "card_" + "a" * 52,
            "quantization": "",
            "capability_id": "text.generate",
            "status": "supported",
            "evidence_kind": "load_qualification",
            "evidence_trust": "foxlight_observed",
            "source_url": "https://evidence.example/load",
            "source_sha256": "c" * 64,
            "rationale": "Exact engine build loaded the exact artifact.",
            "hardware_classes": ["nvidia"],
            "supersedes_claim_id": None,
            "recorded_by": "operator@example.com",
            "created_at": "2026-08-16T12:00:00Z",
        },
        strict=False,
    )
    monkeypatch.setattr(model_cards_module, "_registry_engine_support", (support,))

    placements = place_instance(
        place_instance_command(card),
        topology,
        {},
        node_memory,
        node_network,
        node_resources={
            node_a: NodeResources(
                backends=frozenset({"vllm", "vllm-cuda"}),
                engine_builds={"vllm": "vllm@9.9.9", "vllm-cuda": "vllm@9.9.9"},
                hardware_classes=frozenset({"nvidia"}),
            ),
            node_b: NodeResources(
                backends=frozenset({"vllm", "vllm-cuda"}),
                engine_builds={"vllm": "vllm@9.9.8", "vllm-cuda": "vllm@9.9.8"},
                hardware_classes=frozenset({"nvidia"}),
            ),
        },
    )

    instance = next(iter(placements.values()))
    assert set(instance.shard_assignments.node_to_runner) == {node_a}
    assert {
        shard.resolved_backend
        for shard in instance.shard_assignments.runner_to_shard.values()
    } <= {"vllm", "vllm-cuda"}

    with pytest.raises(PlacementError, match="No candidate cycle"):
        place_instance(
            place_instance_command(card),
            topology,
            {},
            node_memory,
            node_network,
            required_nodes={node_b},
            node_resources={
                node_a: NodeResources(
                    backends=frozenset({"vllm", "vllm-cuda"}),
                    engine_builds={
                        "vllm": "vllm@9.9.9",
                        "vllm-cuda": "vllm@9.9.9",
                    },
                    hardware_classes=frozenset({"nvidia"}),
                ),
                node_b: NodeResources(
                    backends=frozenset({"vllm", "vllm-cuda"}),
                    engine_builds={
                        "vllm": "vllm@9.9.8",
                        "vllm-cuda": "vllm@9.9.8",
                    },
                    hardware_classes=frozenset({"nvidia"}),
                ),
            },
        )


def test_missing_node_resources_is_treated_as_eligible() -> None:
    """Nodes with no resources entry yet (gossip still warming up) must remain
    placeable, matching the pre-#149 full/mlx default — no regression."""
    topology, _node_a, _node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_resources={},  # nothing gossiped yet
    )

    assert len(placements) == 1


def test_published_repository_code_card_is_adaptively_placeable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication authorization does not constrain planner node choice."""
    topology, _node_a, _node_b, node_memory, node_network = _two_node_topology()
    card = _small_model_card().model_copy(
        update={
            "source_revision": "a" * 40,
            "registry_card_id": "card_" + "a" * 52,
            "registry_snapshot_id": "snapshot-test",
            "registry_provenance": "agent",
            "trust_remote_code": True,
        }
    )
    monkeypatch.setattr(
        placement_module,
        "remote_code_approval_required",
        actual_remote_code_approval_required,
    )

    placements = place_instance(
        place_instance_command(card),
        topology,
        {},
        node_memory,
        node_network,
        approved_remote_code_identities=frozenset(),
    )

    assert len(placements) == 1


def test_placement_needs_no_secondary_model_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exact published registry card remains placeable with no allow-list."""
    topology, _node_a, _node_b, node_memory, node_network = _two_node_topology()
    card = _small_model_card().model_copy(
        update={
            "source_revision": "a" * 40,
            "registry_card_id": "card_" + "a" * 52,
            "registry_snapshot_id": "snapshot-test",
            "registry_provenance": "agent",
            "trust_remote_code": True,
        }
    )
    monkeypatch.setattr(
        placement_module,
        "remote_code_approval_required",
        actual_remote_code_approval_required,
    )

    placements = place_instance(
        place_instance_command(card),
        topology,
        {},
        node_memory,
        node_network,
        approved_remote_code_identities=frozenset(),
    )

    assert len(placements) == 1


def test_exact_instance_creation_needs_no_secondary_model_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact-placement path accepts publication-authorized shard cards."""

    node_id = NodeId()
    runner_id = RunnerId()
    card = _small_model_card().model_copy(
        update={
            "source_revision": "a" * 40,
            "registry_card_id": "card_" + "c" * 52,
            "registry_snapshot_id": "snapshot-test",
            "registry_provenance": "agent",
            "trust_remote_code": True,
        }
    )
    monkeypatch.setattr(
        placement_module,
        "remote_code_approval_required",
        actual_remote_code_approval_required,
    )
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            runner_to_shard={runner_id: _make_shard_metadata(card)},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    command = CreateInstance(instance=instance)
    node_memory = {node_id: create_node_memory(Memory.from_gb(8).in_bytes)}

    placements = add_instance_to_placements(
        command,
        Topology(),
        {},
        node_memory,
        approved_remote_code_identities=frozenset(),
    )
    assert instance.instance_id in placements


def test_exact_instance_creation_rejects_mismatched_shard_card() -> None:
    """The master independently rejects inconsistent exact placements."""

    node_id = NodeId()
    runner_id = RunnerId()
    assignment_card = _small_model_card()
    shard_card = assignment_card.model_copy(
        update={"model_id": ModelId("other-org/other-model")}
    )
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=assignment_card.model_id,
            runner_to_shard={runner_id: _make_shard_metadata(shard_card)},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )

    with pytest.raises(
        PlacementModelCardIdentityError,
        match="other-org/other-model",
    ):
        add_instance_to_placements(
            CreateInstance(instance=instance),
            Topology(),
            {},
            {node_id: create_node_memory(Memory.from_gb(8).in_bytes)},
        )


def test_foxlight_signed_card_needs_no_separate_operator_approval() -> None:
    """The signed pinned card remains the complete Foxlight trust decision."""
    topology, _node_a, _node_b, node_memory, node_network = _two_node_topology()
    card = _small_model_card().model_copy(
        update={
            "source_revision": "a" * 40,
            "registry_card_id": "card_" + "a" * 52,
            "registry_snapshot_id": "snapshot-test",
            "registry_provenance": "foxlight",
        }
    )

    placements = place_instance(
        place_instance_command(card),
        topology,
        {},
        node_memory,
        node_network,
    )

    assert len(placements) == 1


def test_zenoh_isolated_node_is_excluded_from_placement() -> None:
    """Positive data-plane isolation routes work to a connected peer."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_resources={
            node_a: NodeResources(
                data_transport="zenoh", zenoh_connected_peers=0
            ),
            node_b: NodeResources(
                data_transport="zenoh", zenoh_connected_peers=1
            ),
        },
    )

    instance = next(iter(placements.values()))
    assert set(instance.shard_assignments.node_to_runner) == {node_b}


def test_all_zenoh_isolated_nodes_fail_to_place() -> None:
    """Never create a runner when every candidate data plane is isolated."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    with pytest.raises(PlacementError, match="Zenoh inference data plane is isolated"):
        place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            node_resources={
                node_a: NodeResources(
                    data_transport="zenoh", zenoh_connected_peers=0
                ),
                node_b: NodeResources(
                    data_transport="zenoh", zenoh_connected_peers=0
                ),
            },
        )


def test_unknown_zenoh_peer_count_remains_placement_eligible() -> None:
    """Startup telemetry unknowns must not create a false hard failure."""
    topology, node_a, node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_small_model_card())

    placements = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        required_nodes={node_a},
        node_resources={
            node_a: NodeResources(
                data_transport="zenoh", zenoh_connected_peers=None
            ),
            node_b: NodeResources(
                data_transport="zenoh", zenoh_connected_peers=1
            ),
        },
    )

    instance = next(iter(placements.values()))
    assert set(instance.shard_assignments.node_to_runner) == {node_a}


def fully_connected_three_nodes(
    available_memory: tuple[float, float, float],
) -> tuple[
    Topology,
    dict[NodeId, MemoryUsage],
    dict[NodeId, NodeNetworkInfo],
    list[NodeId],
]:
    """Build a fully-connected 3-node topology with the given available RAM.

    Returns ``(topology, node_memory, node_network, node_ids)``. Used by the
    refused-placement re-placement tests (#290).
    """
    topology = Topology()
    node_ids = [NodeId(), NodeId(), NodeId()]
    for node_id in node_ids:
        topology.add_node(node_id)
    # directed full mesh
    port = 1
    for source in node_ids:
        for sink in node_ids:
            if source != sink:
                topology.add_connection(
                    Connection(
                        source=source,
                        sink=sink,
                        edge=create_socket_connection(port),
                    )
                )
                port += 1
    node_memory = {
        node_ids[i]: create_node_memory(Memory.from_gb(available_memory[i]).in_bytes)
        for i in range(3)
    }
    node_network = {node_id: create_node_network() for node_id in node_ids}
    return topology, node_memory, node_network, node_ids


def test_replacement_command_widens_refused_instance_by_one_node() -> None:
    """A refused instance re-places one node wider, preserving model + backend (#290)."""
    topology, node_memory, node_network, _ = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    card = ModelCard(
        model_id=ModelId("widen-model"),
        storage_size=Memory.from_gb(6),
        n_layers=12,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    two_node = PlaceInstance(
        model_card=card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=2,
    )
    placed = place_instance(two_node, topology, {}, node_memory, node_network)
    instance = next(iter(placed.values()))
    assert len(instance.shard_assignments.node_to_runner) == 2

    replacement = replacement_command_for_refused_instance(instance)
    assert replacement.min_nodes == 3
    assert replacement.model_card.model_id == card.model_id
    assert replacement.sharding == Sharding.Pipeline
    assert replacement.instance_meta == InstanceMeta.MlxRing


def test_refusal_fallback_excludes_refuser_at_any_width() -> None:
    """When the wider re-place is unsatisfiable, the fallback re-places
    anywhere but the refusing node at min_nodes=1 (#290 on a heterogeneous
    fleet: an MLX model refused at the full Mac width cannot go wider, but a
    remaining node can hold it alone)."""
    topology, node_memory, node_network, node_ids = fully_connected_three_nodes(
        (10.0, 10.0, 24.0)
    )
    card = ModelCard(
        model_id=ModelId("fallback-model"),
        storage_size=Memory.from_gb(9),
        n_layers=12,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    three_node = PlaceInstance(
        model_card=card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=3,
    )
    placed = place_instance(three_node, topology, {}, node_memory, node_network)
    instance = next(iter(placed.values()))
    refusing_node = node_ids[0]

    # The wider attempt is unsatisfiable by construction (only 3 nodes).
    wider = replacement_command_for_refused_instance(instance)
    assert wider.min_nodes == 4
    with pytest.raises(PlacementError):
        place_instance(wider, topology, {}, node_memory, node_network)

    # The fallback places on the remaining nodes with the refuser excluded.
    fallback = fallback_command_for_refused_instance(instance, refusing_node)
    assert fallback.min_nodes == 1
    assert fallback.excluded_nodes == [refusing_node]
    assert fallback.model_card.model_id == card.model_id
    result = place_instance(
        fallback,
        topology,
        {},
        node_memory,
        node_network,
        excluded_nodes={refusing_node},
    )
    fallback_instance = next(iter(result.values()))
    assert refusing_node not in fallback_instance.shard_assignments.node_to_runner


def test_refused_instance_replaces_onto_a_wider_split() -> None:
    """A refused 2-node placement re-places across all three nodes (#290).

    The master picks the smallest fitting cycle, so on a memory view where the
    2-node split looks fine it keeps choosing 2 nodes — exactly the view that
    over-admits the split a worker then refuses on its tighter live reading.
    Bumping ``min_nodes`` to 3 forces the wider floor, so the re-placement lands
    a 3-node split (smaller per-node share) instead of the doomed 2-node one.
    """
    topology, node_memory, node_network, _ = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    card = ModelCard(
        model_id=ModelId("widen-fit-model"),
        storage_size=Memory.from_gb(6),
        n_layers=12,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    two_node = PlaceInstance(
        model_card=card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=2,
    )
    placed = place_instance(two_node, topology, {}, node_memory, node_network)
    refused_instance = next(iter(placed.values()))
    # The same memory view keeps selecting the 2-node split.
    assert len(refused_instance.shard_assignments.node_to_runner) == 2

    replacement = replacement_command_for_refused_instance(refused_instance)
    assert replacement.min_nodes == 3
    widened = place_instance(replacement, topology, {}, node_memory, node_network)
    widened_instance = next(iter(widened.values()))
    assert len(widened_instance.shard_assignments.node_to_runner) == 3


def test_replacement_at_full_cluster_width_is_terminal() -> None:
    """Re-placing a full-width instance raises PlacementError (loop bound) (#290).

    A 3-node instance on a 3-node cluster widens to min_nodes=4, which cannot be
    satisfied — place_instance raises, and the master treats that as the terminal
    "cannot fit anywhere" outcome that stops the refuse→re-place loop.
    """
    topology, node_memory, node_network, _ = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    card = ModelCard(
        model_id=ModelId("full-width-model"),
        storage_size=Memory.from_gb(9),
        n_layers=12,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    three_node = PlaceInstance(
        model_card=card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=3,
    )
    placed = place_instance(three_node, topology, {}, node_memory, node_network)
    instance = next(iter(placed.values()))
    assert len(instance.shard_assignments.node_to_runner) == 3

    replacement = replacement_command_for_refused_instance(instance)
    assert replacement.min_nodes == 4
    with pytest.raises(PlacementError):
        place_instance(replacement, topology, {}, node_memory, node_network)


def _llama_cpp_card() -> ModelCard:
    """A GGUF-style card whose compatible backends are all llama_cpp tags."""
    from skulk.shared.models.model_cards import PlacementCardConfig

    return ModelCard(
        model_id=ModelId("test-gguf"),
        storage_size=Memory.from_mb(500),
        n_layers=10,
        hidden_size=1000,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        placement=PlacementCardConfig(
            compatible_backends=frozenset({"llama_cpp-vulkan", "llama_cpp-cpu"})
        ),
    )


def test_llama_cpp_multi_node_placement_is_rejected() -> None:
    """llama.cpp is single-node only; a forced multi-node placement of a
    llama.cpp-only card is rejected up front rather than crashing at runner
    startup (world_size != 1)."""
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )
    node_memory = {
        node_a: create_node_memory(Memory.from_gb(8).in_bytes),
        node_b: create_node_memory(Memory.from_gb(8).in_bytes),
    }
    node_network = {node_a: create_node_network(), node_b: create_node_network()}
    command = place_instance_command(_llama_cpp_card()).model_copy(
        update={"min_nodes": 2}
    )
    with pytest.raises(ValueError, match="requires one engine common"):
        place_instance(command, topology, {}, node_memory, node_network)


def test_llama_cpp_single_node_placement_succeeds() -> None:
    """The single-node guard must not block a legitimate single-node placement."""
    topology, _node_a, _node_b, node_memory, node_network = _two_node_topology()
    command = place_instance_command(_llama_cpp_card())
    placements = place_instance(command, topology, {}, node_memory, node_network)
    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert len(instance.shard_assignments.node_to_runner) == 1


def test_hybrid_card_with_multi_node_engine_places_multi_node() -> None:
    """The single-node guard keys on engine capability, not on llama.cpp by name:
    a card that also allows a multi-node engine (MLX) still places across nodes,
    serving via that engine. Guards #328's generalization of the old
    llama.cpp-only special-case."""
    from skulk.shared.models.model_cards import PlacementCardConfig

    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )
    node_memory = {
        node_a: create_node_memory(Memory.from_gb(8).in_bytes),
        node_b: create_node_memory(Memory.from_gb(8).in_bytes),
    }
    node_network = {node_a: create_node_network(), node_b: create_node_network()}
    hybrid_card = ModelCard(
        model_id=ModelId("test-hybrid"),
        storage_size=Memory.from_mb(500),
        n_layers=10,
        hidden_size=1000,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        placement=PlacementCardConfig(
            compatible_backends=frozenset({"mlx", "llama_cpp-vulkan"})
        ),
    )
    command = place_instance_command(hybrid_card).model_copy(update={"min_nodes": 2})
    # Must NOT raise the single-node guard: MLX is multi-node capable.
    placements = place_instance(command, topology, {}, node_memory, node_network)
    instance = next(iter(placements.values()))
    assert len(instance.shard_assignments.node_to_runner) == 2


def test_placement_ports_stay_outside_os_ephemeral_range() -> None:
    """Listener ports come from the reserved band below the OS ephemeral
    range: an outgoing socket on a placement node (store transfers run
    exactly when placements happen) could otherwise hold the chosen port and
    every runner retry dies on EADDRINUSE (observed live: [ring] Couldn't
    bind socket (error: 48) on a node mid store-download)."""
    from skulk.master.placement import (
        _PLACEMENT_PORT_RANGE,  # pyright: ignore[reportPrivateUsage]
        random_ephemeral_port,
    )

    low, high = _PLACEMENT_PORT_RANGE
    assert high < 32768  # below BOTH ephemeral floors (Linux 32768, macOS 49152)
    for _ in range(200):
        port = random_ephemeral_port()
        assert low <= port <= high


def test_placement_ports_avoid_live_instance_ports() -> None:
    from skulk.master.placement import (
        _PLACEMENT_PORT_RANGE,  # pyright: ignore[reportPrivateUsage]
        random_ephemeral_port,
    )

    low, high = _PLACEMENT_PORT_RANGE
    # Reserve everything except one port; the picker must find it.
    free = low + 7
    in_use = {p for p in range(low, high + 1) if p != free}
    assert random_ephemeral_port(in_use) == free


def test_repair_commands_preserve_original_exclusions() -> None:
    """Repair re-placements keep the caller's per-placement exclusions (#658).

    Placement stamps the effective exclusions on the instance; every repair
    shape (wider refusal re-place, anywhere fallback, download-failure
    replacement) must union them with its own newly excluded nodes. Before
    the stamp existed, a repaired instance could land on exactly the nodes
    the caller excluded (observed live: a placement pinned off five nodes
    was repaired onto one of them).
    """
    topology, node_memory, node_network, node_ids = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    operator_excluded = node_ids[2]
    card = ModelCard(
        model_id=ModelId("exclusion-carry-model"),
        storage_size=Memory.from_gb(6),
        n_layers=12,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    command = PlaceInstance(
        model_card=card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=2,
        excluded_nodes=[operator_excluded],
    )
    placed = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        excluded_nodes={operator_excluded},
    )
    instance = next(iter(placed.values()))
    # The effective exclusions are stamped on the placement decision.
    assert instance.excluded_nodes == [operator_excluded]
    assert operator_excluded not in instance.shard_assignments.node_to_runner

    wider = replacement_command_for_refused_instance(instance)
    assert operator_excluded in wider.excluded_nodes

    refusing_node = node_ids[0]
    fallback = fallback_command_for_refused_instance(instance, refusing_node)
    assert set(fallback.excluded_nodes) == {operator_excluded, refusing_node}

    download_failed = replacement_command_for_download_failed_instance(
        instance, {node_ids[1]}
    )
    assert set(download_failed.excluded_nodes) == {operator_excluded, node_ids[1]}

    # A repair re-placement searches with the union but stamps only the
    # ORIGINAL intent on the new instance, so a transiently failed node is
    # not laundered into permanent operator intent across repair
    # generations (PR #667 review). Use a width-1 placement so the union
    # (operator exclusion plus the failed node) still leaves a host in the
    # three-node topology: the failed node may be excluded for THIS
    # placement yet must not be remembered as caller intent.
    single = PlaceInstance(
        model_card=card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
        excluded_nodes=[operator_excluded],
    )
    placed_single = place_instance(
        single,
        topology,
        {},
        node_memory,
        node_network,
        excluded_nodes={operator_excluded},
    )
    single_instance = next(iter(placed_single.values()))
    failed_node = next(iter(single_instance.shard_assignments.node_to_runner))
    repair = replacement_command_for_download_failed_instance(
        single_instance, {failed_node}
    )
    repaired = place_instance(
        repair,
        topology,
        {},
        node_memory,
        node_network,
        excluded_nodes=set(repair.excluded_nodes),
        stamped_exclusions=set(single_instance.excluded_nodes),
    )
    repaired_instance = next(iter(repaired.values()))
    assert repaired_instance.excluded_nodes == [operator_excluded]
    assert failed_node not in repaired_instance.shard_assignments.node_to_runner
