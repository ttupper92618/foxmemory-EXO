"""API-side telemetry-plane advertise surface (fabric-citizenship Phase 1).

`ExtensionContext.advertise_capability` records a tag on the shared TelemetryView
outbound set (which the worker's gatherer later gossips), and the round trip is
observable locally: a node advertising a capability sees it in its own
`read_cluster` snapshot once the reading coalesces back into the view.
"""

from datetime import datetime, timezone
from typing import cast

import pytest

from skulk.api.main import API
from skulk.extensions import (
    CapabilityDescriptor,
    ExtensionContext,
    LoadedExtensions,
    descriptor_revision,
)
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.utils.channels import channel
from skulk.utils.info_gatherer.info_gatherer import NodeCapabilities


def _build_api(view: TelemetryView, extensions: LoadedExtensions | None = None) -> API:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    return API(
        NodeId("api-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
        telemetry_view=view,
        extensions=extensions,
    )


def test_advertise_capability_records_on_outbound_set() -> None:
    view = TelemetryView()
    api = _build_api(view)
    api._extension_context.advertise_capability("memory")  # pyright: ignore[reportPrivateUsage]
    assert view.local_advertised_capabilities == {"memory"}


def test_advertise_capability_is_additive_and_idempotent() -> None:
    view = TelemetryView()
    api = _build_api(view)
    advertise = api._extension_context.advertise_capability  # pyright: ignore[reportPrivateUsage]
    advertise("memory")
    advertise("memory")  # idempotent
    advertise("search")  # additive
    assert view.local_advertised_capabilities == {"memory", "search"}


def test_advertise_capability_ignores_blank_tags() -> None:
    view = TelemetryView()
    api = _build_api(view)
    advertise = api._extension_context.advertise_capability  # pyright: ignore[reportPrivateUsage]
    advertise("   ")
    advertise("")
    assert view.local_advertised_capabilities == set()
    # a surrounding-whitespace tag is trimmed, not dropped
    advertise("  memory  ")
    assert view.local_advertised_capabilities == {"memory"}


def test_advertised_capability_round_trips_into_read_cluster() -> None:
    # End to end within one node: advertise -> the gatherer would emit a
    # NodeCapabilities reading -> the view coalesces it -> read_cluster surfaces
    # it. We stand in for the gossip hop by applying the reading the gatherer
    # would send for this node.
    view = TelemetryView()
    api = _build_api(view)
    api._extension_context.advertise_capability("memory")  # pyright: ignore[reportPrivateUsage]
    view.apply(
        NodeTelemetry(
            node_id=NodeId("api-node"),
            info=NodeCapabilities(
                capabilities=frozenset(view.local_advertised_capabilities)
            ),
        )
    )
    snapshot = api._extension_context.read_cluster()  # pyright: ignore[reportPrivateUsage]
    node = next(n for n in snapshot if n.node_id == NodeId("api-node"))
    assert node.capabilities == ("memory",)


# --- Provider surface (fabric-citizenship Phase 2a) --------------------------

_ECHO = CapabilityDescriptor(
    id="echo",
    version="1.0.0",
    title="Echo",
    description="Returns the input text unchanged.",
    input_schema={"type": "object"},
)


class _ProviderExtension:
    """Minimal provider extension for API wiring tests."""

    name = "test-provider"
    skulk_requires = ">=0"

    def __init__(self) -> None:
        self.started_with: list[ExtensionContext] = []

    def chat_middleware(self) -> None:
        return None

    def capabilities(self) -> list[CapabilityDescriptor]:
        return [_ECHO]

    def on_start(self, context: ExtensionContext) -> None:
        self.started_with.append(context)


def test_provider_is_advertised_without_starting_background_work_in_constructor() -> (
    None
):
    provider = _ProviderExtension()
    view = TelemetryView()
    _build_api(view, extensions=LoadedExtensions([provider]))
    # The descriptor's id became the telemetry discovery tag without the
    # extension calling advertise itself. Startup waits for the serving loop.
    assert view.local_advertised_capabilities == {"echo"}
    assert provider.started_with == []


async def test_list_node_capabilities_serves_descriptors_and_revisions() -> None:
    api = _build_api(
        TelemetryView(), extensions=LoadedExtensions([_ProviderExtension()])
    )
    payload = await api.list_node_capabilities()
    assert payload["node_id"] == "api-node"
    capabilities = cast("list[object]", payload["capabilities"])
    assert isinstance(capabilities, list) and len(capabilities) == 1
    restored = CapabilityDescriptor.model_validate(capabilities[0])
    assert restored == _ECHO
    revisions = payload["revisions"]
    assert isinstance(revisions, dict)
    assert revisions["echo@1.0.0"] == descriptor_revision(_ECHO)


async def test_list_node_capabilities_empty_without_extensions() -> None:
    payload = await _build_api(TelemetryView()).list_node_capabilities()
    assert payload["capabilities"] == []
    assert payload["revisions"] == {}


async def test_describe_node_local_returns_descriptors() -> None:
    api = _build_api(
        TelemetryView(), extensions=LoadedExtensions([_ProviderExtension()])
    )
    descriptors = await api._extension_context.describe_node(NodeId("api-node"))  # pyright: ignore[reportPrivateUsage]
    assert descriptors == (_ECHO,)


async def test_describe_node_unknown_peer_returns_empty() -> None:
    api = _build_api(TelemetryView())
    # No topology, so the peer is unreachable: degrade to (), never raise.
    descriptors = await api._extension_context.describe_node(NodeId("n-ghost"))  # pyright: ignore[reportPrivateUsage]
    assert descriptors == ()


def test_withdraw_capability_removes_tag() -> None:
    view = TelemetryView()
    api = _build_api(view)
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    context.advertise_capability("memory")
    context.advertise_capability("search")
    context.withdraw_capability("memory")
    assert view.local_advertised_capabilities == {"search"}
    # Withdrawing an unknown or already-withdrawn tag is a no-op.
    context.withdraw_capability("memory")
    context.withdraw_capability("never-advertised")
    assert view.local_advertised_capabilities == {"search"}


async def test_list_node_capabilities_with_own_node_id_serves_local() -> None:
    api = _build_api(
        TelemetryView(), extensions=LoadedExtensions([_ProviderExtension()])
    )
    payload = await api.list_node_capabilities(node_id="api-node")
    assert payload["node_id"] == "api-node"
    assert len(cast("list[object]", payload["capabilities"])) == 1


async def test_list_node_capabilities_unreachable_peer_is_empty() -> None:
    api = _build_api(TelemetryView())
    payload = await api.list_node_capabilities(node_id="n-ghost")
    assert payload["node_id"] == "n-ghost"
    assert payload["capabilities"] == []


async def test_state_merge_surfaces_node_capabilities() -> None:
    # The light discovery layer is operator-visible: GET /state carries a
    # nodeCapabilities map for live nodes with a non-empty tag set (sorted).
    view = TelemetryView()
    api = _build_api(view)
    peer = NodeId("n-peer")
    dead = NodeId("n-dead")
    view.node_capabilities[peer] = frozenset({"tts", "memory"})
    view.node_capabilities[dead] = frozenset({"ghost"})
    api.state = api.state.model_copy(
        update={"last_seen": {peer: datetime.now(tz=timezone.utc)}}
    )
    payload = await api.get_cluster_state()
    # Only the live node appears; the dead node's tags are filtered out.
    assert payload["nodeCapabilities"] == {"n-peer": ["memory", "tts"]}


async def test_list_node_capabilities_blank_node_id_means_local() -> None:
    # A blank or whitespace-padded node_id describes this node rather than
    # proxying to a literal empty peer id.
    api = _build_api(
        TelemetryView(), extensions=LoadedExtensions([_ProviderExtension()])
    )
    for raw in ("", "   "):
        payload = await api.list_node_capabilities(node_id=raw)
        assert payload["node_id"] == "api-node"
        assert len(cast("list[object]", payload["capabilities"])) == 1


async def test_api_runtime_owns_extension_startup_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed serving startup still cleans hooks on the loop that started them."""
    import asyncio

    from skulk.utils.task_group import TaskGroup

    class LifecycleProvider(_ProviderExtension):
        def __init__(self) -> None:
            super().__init__()
            self.loop: asyncio.AbstractEventLoop | None = None
            self.stopped = False

        def on_start(self, context: ExtensionContext) -> None:
            self.loop = asyncio.get_running_loop()
            super().on_start(context)

        async def on_stop(self) -> None:
            assert asyncio.get_running_loop() is self.loop
            self.stopped = True

    async def fail_before_serving(self: TaskGroup) -> None:
        raise RuntimeError("injected serving startup failure")

    provider = LifecycleProvider()
    api = _build_api(TelemetryView(), extensions=LoadedExtensions([provider]))
    assert provider.started_with == []
    monkeypatch.setattr(TaskGroup, "__aenter__", fail_before_serving)
    with pytest.raises(RuntimeError, match="injected serving startup failure"):
        await api.run()
    assert len(provider.started_with) == 1
    assert provider.stopped
