"""Steward adapters cannot bypass per-request proposal permission or drift checks."""

from collections.abc import AsyncIterator, Sequence
from typing import final

from pydantic import JsonValue

from skulk.extensions import (
    CapabilityDescriptor,
    CapabilityResult,
    CapabilityStreamFrame,
    CapabilityStreamSession,
    ExtensionContext,
    LoadedExtensions,
    call_failure,
)
from skulk.extensions.steward import (
    StewardTool,
    collect_steward_tools,
    invoke_steward_tool,
)
from skulk.shared.types.common import ModelId, NodeId


async def embed_stub(
    texts: list[str], model_id: ModelId | None = None
) -> list[list[float]] | None:
    """Deterministic embeddings for tests."""
    return [[float(len(text))] for text in texts]


async def describe_stub(node_id: NodeId) -> tuple[CapabilityDescriptor, ...]:
    """Empty describe surface for tests."""
    return ()


async def call_stub(
    node_id: NodeId,
    capability_id: str,
    version: str,
    descriptor_revision: str,
    payload: dict[str, object],
    *,
    timeout_seconds: float | None = None,
) -> CapabilityResult:
    """Unreachable call surface for tests."""
    return call_failure("test-call", "unreachable", "no fabric in tests")


async def empty_stream() -> AsyncIterator[CapabilityStreamFrame]:
    if False:
        yield CapabilityStreamFrame(
            call_id="unused",
            direction="provider_to_caller",
            sequence=0,
            kind="started",
        )


async def stream_stub(
    node_id: NodeId,
    capability_id: str,
    version: str,
    descriptor_revision: str,
    payload: dict[str, object],
    *,
    timeout_seconds: float | None = None,
) -> CapabilityStreamSession:
    """Unreachable streaming surface for tests."""

    return CapabilityStreamSession(
        open_result=call_failure("test-stream", "unreachable", "no fabric in tests"),
        frames=empty_stream(),
    )


def context() -> ExtensionContext:
    return ExtensionContext(
        node_id=NodeId("test-node"),
        skulk_version="1.3.1",
        embed_texts=embed_stub,
        read_cluster=lambda: (),
        advertise_capability=lambda capability: None,  # noqa: ARG005
        withdraw_capability=lambda capability: None,  # noqa: ARG005
        describe_node=describe_stub,
        call_capability=call_stub,
        stream_capability=stream_stub,
    )


@final
class Adapter:
    """Record inert adapter calls without possessing an effect authority."""

    name = "fixture"
    skulk_requires = ">=0"

    def __init__(self, proposal: bool = False) -> None:
        self.calls = 0
        self.tools: tuple[StewardTool, ...] = (
            StewardTool(
                name="extension_fixture",
                description="Fixture tool",
                mode="proposal" if proposal else "read",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            ),
        )

    def chat_middleware(self) -> None:
        """No chat middleware is involved in this isolated adapter."""
        return None

    async def steward_tools(self, context: ExtensionContext) -> Sequence[StewardTool]:
        """Return current eligibility, including withdrawal after collection."""
        return self.tools

    async def handle_steward_tool(
        self,
        context: ExtensionContext,
        tool: StewardTool,
        arguments: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        """Only record an inert request in this fixture."""
        self.calls += 1
        return {"accepted": True}


async def test_readonly_sessions_cannot_offer_or_invoke_proposals() -> None:
    adapter = Adapter(proposal=True)
    assert (
        await collect_steward_tools([adapter], context(), proposals_allowed=False) == ()
    )
    bindings = await collect_steward_tools([adapter], context(), proposals_allowed=True)
    result = await invoke_steward_tool(
        bindings[0], context(), {"text": "request"}, proposals_allowed=False
    )
    assert "error" in result
    assert adapter.calls == 0


async def test_withdrawal_and_contract_drift_refuse_stale_model_calls() -> None:
    adapter = Adapter()
    binding = (
        await collect_steward_tools([adapter], context(), proposals_allowed=True)
    )[0]
    adapter.tools = ()
    assert "error" in await invoke_steward_tool(
        binding, context(), {"text": "request"}, proposals_allowed=True
    )
    adapter.tools = (
        binding.tool.model_copy(update={"description": "changed contract"}),
    )
    assert "error" in await invoke_steward_tool(
        binding, context(), {"text": "request"}, proposals_allowed=True
    )
    assert adapter.calls == 0


async def test_schema_refusal_does_not_reach_adapter() -> None:
    adapter = Adapter()
    binding = (
        await collect_steward_tools([adapter], context(), proposals_allowed=True)
    )[0]
    invalid: tuple[dict[str, object], ...] = (
        {"text": 42},
        {"text": "request", "approval": True},
    )
    for payload in invalid:
        assert "error" in await invoke_steward_tool(
            binding, context(), payload, proposals_allowed=True
        )
    assert adapter.calls == 0
    assert "accepted" in await invoke_steward_tool(
        binding, context(), {"text": "request"}, proposals_allowed=True
    )
    assert adapter.calls == 1


async def test_ambiguous_adapter_names_are_not_offered() -> None:
    assert (
        await collect_steward_tools(
            [Adapter(), Adapter()], context(), proposals_allowed=True
        )
        == ()
    )


async def test_shutdown_withdraws_previously_bound_adapter() -> None:
    adapter = Adapter()
    registry = LoadedExtensions([adapter])
    binding = (await registry.steward_tools(context(), proposals_allowed=True))[0]
    await registry.run_shutdown_hooks()
    assert await registry.steward_tools(context(), proposals_allowed=True) == ()
    assert "error" in await registry.invoke_steward_tool(
        binding, context(), {"text": "request"}, proposals_allowed=True
    )
    assert adapter.calls == 0
