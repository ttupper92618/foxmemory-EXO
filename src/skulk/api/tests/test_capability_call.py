"""Provider-side capability-call dispatch (fabric-citizenship Phase 2b).

`_dispatch_capability_call` is the single guarded chokepoint every call runs
through (the endpoint, the local fast path, and a future queryable transport
all land here), so the guard matrix is tested against it directly.
"""

from typing import TYPE_CHECKING

import anyio

if TYPE_CHECKING:
    import pytest

from skulk.api.main import API
from skulk.extensions import (
    CapabilityCall,
    CapabilityDescriptor,
    CapabilityResult,
    ExtensionContext,
    LoadedExtensions,
    descriptor_revision,
)
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.telemetry import TelemetryView
from skulk.utils.channels import channel

_ECHO = CapabilityDescriptor(
    id="echo",
    version="1.0.0",
    title="Echo",
    description="Returns the input text unchanged.",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)
_ECHO_REVISION = descriptor_revision(_ECHO)


class _EchoProvider:
    """Callable provider used across the dispatch tests."""

    name = "echo-test"
    skulk_requires = ">=0"

    def chat_middleware(self) -> None:
        return None

    def capabilities(self) -> list[CapabilityDescriptor]:
        return [_ECHO]

    async def handle_call(
        self, context: ExtensionContext, call: CapabilityCall
    ) -> dict[str, object]:
        return {"text": call.payload["text"]}


class _SlowProvider(_EchoProvider):
    """Provider that never finishes within a short deadline."""

    async def handle_call(
        self, context: ExtensionContext, call: CapabilityCall
    ) -> dict[str, object]:
        await anyio.sleep(60)
        return {"text": "late"}


class _RaisingProvider(_EchoProvider):
    """Provider whose handler raises."""

    async def handle_call(
        self, context: ExtensionContext, call: CapabilityCall
    ) -> dict[str, object]:
        raise RuntimeError("handler exploded")


class _BadResultProvider(_EchoProvider):
    """Provider whose result violates its own output schema."""

    async def handle_call(
        self, context: ExtensionContext, call: CapabilityCall
    ) -> dict[str, object]:
        return {"wrong_key": 42}


def _build_api(provider: object | None) -> API:
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
        telemetry_view=TelemetryView(),
        extensions=LoadedExtensions([provider]) if provider is not None else None,  # pyright: ignore[reportArgumentType]
    )


def _call(**overrides: object) -> CapabilityCall:
    fields: dict[str, object] = {
        "call_id": "c-1",
        "capability_id": "echo",
        "version": "1.0.0",
        "descriptor_revision": _ECHO_REVISION,
        "caller_node": "caller",
        "target_node": "api-node",
        "payload": {"text": "hello"},
    }
    fields.update(overrides)
    return CapabilityCall.model_validate(fields)


async def _dispatch(api: API, call: CapabilityCall) -> CapabilityResult:
    return await api._dispatch_capability_call(call)  # pyright: ignore[reportPrivateUsage]


async def test_call_round_trips_through_handler() -> None:
    result = await _dispatch(_build_api(_EchoProvider()), _call())
    assert result.ok and result.error is None
    assert result.result == {"text": "hello"}
    assert result.call_id == "c-1"


async def test_unknown_capability_is_not_found() -> None:
    result = await _dispatch(_build_api(_EchoProvider()), _call(capability_id="nope"))
    assert not result.ok and result.error is not None
    assert result.error.code == "not_found"


async def test_wrong_version_is_version_mismatch() -> None:
    result = await _dispatch(_build_api(_EchoProvider()), _call(version="2.0.0"))
    assert not result.ok and result.error is not None
    assert result.error.code == "version_mismatch"


async def test_drifted_revision_is_revision_mismatch() -> None:
    result = await _dispatch(
        _build_api(_EchoProvider()), _call(descriptor_revision="deadbeef00000000")
    )
    assert not result.ok and result.error is not None
    assert result.error.code == "revision_mismatch"


async def test_payload_failing_input_schema_is_invalid_payload() -> None:
    result = await _dispatch(_build_api(_EchoProvider()), _call(payload={"text": 42}))
    assert not result.ok and result.error is not None
    assert result.error.code == "invalid_payload"


async def test_oversized_payload_is_rejected() -> None:
    result = await _dispatch(
        _build_api(_EchoProvider()), _call(payload={"text": "x" * 1_100_000})
    )
    assert not result.ok and result.error is not None
    assert result.error.code == "payload_too_large"


async def test_deadline_yields_typed_timeout() -> None:
    result = await _dispatch(_build_api(_SlowProvider()), _call(timeout_seconds=0.05))
    assert not result.ok and result.error is not None
    assert result.error.code == "timeout"


async def test_raising_handler_yields_provider_error() -> None:
    result = await _dispatch(_build_api(_RaisingProvider()), _call())
    assert not result.ok and result.error is not None
    assert result.error.code == "provider_error"
    assert "handler exploded" in result.error.message


async def test_result_failing_output_schema_is_invalid_result() -> None:
    result = await _dispatch(_build_api(_BadResultProvider()), _call())
    assert not result.ok and result.error is not None
    assert result.error.code == "invalid_result"


async def test_no_extensions_is_not_found() -> None:
    result = await _dispatch(_build_api(None), _call())
    assert not result.ok and result.error is not None
    assert result.error.code == "not_found"


async def test_concurrency_bound_rejects_with_overloaded() -> None:
    api = _build_api(_SlowProvider())
    results: list[CapabilityResult] = []

    async def one(index: int) -> None:
        results.append(
            await _dispatch(api, _call(call_id=f"c-{index}", timeout_seconds=0.5))
        )

    async with anyio.create_task_group() as tg:
        for index in range(10):
            tg.start_soon(one, index)
    codes = sorted(r.error.code for r in results if r.error is not None)
    # 8 slots time out (slow provider); the 2 beyond the bound are rejected
    # immediately as overloaded.
    assert codes.count("overloaded") == 2
    assert codes.count("timeout") == 8


async def test_context_call_capability_local_fast_path() -> None:
    # The caller-side verb with target == self dispatches in process through
    # the same guards; a plugin can call a capability its own node serves.
    api = _build_api(_EchoProvider())
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    result = await context.call_capability(
        NodeId("api-node"), "echo", "1.0.0", _ECHO_REVISION, {"text": "loop"}
    )
    assert result.ok and result.result == {"text": "loop"}


async def test_context_call_capability_unreachable_peer() -> None:
    api = _build_api(_EchoProvider())
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    result = await context.call_capability(
        NodeId("n-ghost"), "echo", "1.0.0", _ECHO_REVISION, {"text": "hi"}
    )
    assert not result.ok and result.error is not None
    assert result.error.code == "unreachable"


class _NonDictResultProvider(_EchoProvider):
    """Provider whose handler returns a non-dict (protocol violation)."""

    async def handle_call(
        self, context: ExtensionContext, call: CapabilityCall
    ) -> dict[str, object]:
        return "not a dict"  # pyright: ignore[reportReturnType]


class _NanResultProvider(_EchoProvider):
    """Provider whose numeric result carries a non-finite float."""

    async def handle_call(
        self, context: ExtensionContext, call: CapabilityCall
    ) -> dict[str, object]:
        return {"text": "x", "score": float("nan")}


_STREAMING = CapabilityDescriptor(
    id="tts-demo",
    version="1.0.0",
    title="TTS demo",
    description="A streaming-only capability (not unary-callable).",
    input_schema={"type": "object"},
    io_mode="server_streaming",
    output_chunk_schema={"type": "object"},
)


class _StreamingOnlyProvider(_EchoProvider):
    """Provider whose only descriptor is a streaming mode."""

    def capabilities(self) -> list[CapabilityDescriptor]:
        return [_STREAMING]


async def test_non_serializable_payload_is_typed_invalid_payload() -> None:
    # A local fast-path caller can hand a payload JSON can't carry; the
    # never-raises contract demands a typed error, not a TypeError.
    result = await _dispatch(
        _build_api(_EchoProvider()), _call(payload={"text": "x", "blob": b"raw"})
    )
    assert not result.ok and result.error is not None
    assert result.error.code == "invalid_payload"


async def test_non_dict_result_is_typed_invalid_result() -> None:
    result = await _dispatch(_build_api(_NonDictResultProvider()), _call())
    assert not result.ok and result.error is not None
    assert result.error.code == "invalid_result"


async def test_nan_result_is_typed_invalid_result() -> None:
    # json.dumps accepts NaN by default but the HTTP response renderer refuses
    # non-finite JSON; allow_nan=False catches it as a typed error instead.
    result = await _dispatch(_build_api(_NanResultProvider()), _call())
    assert not result.ok and result.error is not None
    assert result.error.code == "invalid_result"


async def test_streaming_descriptor_is_not_unary_callable() -> None:
    api = _build_api(_StreamingOnlyProvider())
    result = await _dispatch(
        api, _call(capability_id="tts-demo", descriptor_revision="ignored")
    )
    assert not result.ok and result.error is not None
    assert result.error.code == "not_found"
    # ...but it is still discoverable (discovery-only descriptor).
    assert api._extensions is not None  # pyright: ignore[reportPrivateUsage]
    assert api._extensions.capability_descriptors == (_STREAMING,)  # pyright: ignore[reportPrivateUsage]


async def test_caller_side_rejects_non_serializable_payload_before_any_hop() -> None:
    api = _build_api(_EchoProvider())
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    result = await context.call_capability(
        NodeId("n-peer"), "echo", "1.0.0", _ECHO_REVISION, {"blob": b"raw"}
    )
    assert not result.ok and result.error is not None
    assert result.error.code == "invalid_payload"


async def test_dispatch_rejects_misaddressed_envelope() -> None:
    # The call is node-addressed: an envelope claiming a different target must
    # not execute here (honest logs, no misrouted execution).
    result = await _dispatch(
        _build_api(_EchoProvider()), _call(target_node="some-other-node")
    )
    assert not result.ok and result.error is not None
    assert result.error.code == "invalid_payload"
    assert "addressed to" in result.error.message


async def test_caller_side_enforces_size_cap_before_posting() -> None:
    api = _build_api(_EchoProvider())
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    result = await context.call_capability(
        NodeId("n-peer"), "echo", "1.0.0", _ECHO_REVISION, {"text": "x" * 1_100_000}
    )
    assert not result.ok and result.error is not None
    assert result.error.code == "payload_too_large"


async def test_caller_side_out_of_range_timeout_is_typed() -> None:
    # An envelope violation (timeout beyond the ceiling) is a typed error,
    # never a ValidationError raised out of call_capability.
    api = _build_api(_EchoProvider())
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    result = await context.call_capability(
        NodeId("api-node"),
        "echo",
        "1.0.0",
        _ECHO_REVISION,
        {"text": "x"},
        timeout_seconds=9_999.0,
    )
    assert not result.ok and result.error is not None
    assert result.error.code == "invalid_payload"


class _IntKeyResultProvider(_EchoProvider):
    """Provider whose result dict has a non-string key."""

    async def handle_call(
        self, context: ExtensionContext, call: CapabilityCall
    ) -> dict[str, object]:
        return {1: "x"}  # pyright: ignore[reportReturnType]


async def test_non_string_result_keys_are_typed_invalid_result() -> None:
    # json.dumps stringifies int keys (so the size check passes) but the
    # strict result model rejects them; must be a typed error, not a 500.
    result = await _dispatch(_build_api(_IntKeyResultProvider()), _call())
    assert not result.ok and result.error is not None
    assert result.error.code == "invalid_result"


async def test_caller_budget_spans_lookup_and_provider(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    # #513: one budget clock. A lookup that eats (nearly) the whole deadline
    # leaves no budget for the provider hop; the caller gets a typed timeout
    # instead of waiting up to twice the requested deadline.
    api = _build_api(_EchoProvider())

    async def slow_lookup(node_id: NodeId) -> str:
        # Finishes just inside the budget, leaving less than the 0.05s floor.
        await anyio.sleep(0.27)
        return "http://192.0.2.1:52415"  # TEST-NET, never reached

    monkeypatch.setattr(api, "_peer_api_url_for", slow_lookup)
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    started = anyio.current_time()
    result = await context.call_capability(
        NodeId("n-peer"),
        "echo",
        "1.0.0",
        _ECHO_REVISION,
        {"text": "x"},
        timeout_seconds=0.3,
    )
    elapsed = anyio.current_time() - started
    assert not result.ok and result.error is not None
    assert result.error.code == "timeout"
    assert "no budget remains" in result.error.message
    # The whole call stayed near the requested budget, not lookup + provider.
    assert elapsed < 1.5


async def test_caller_lookup_cancelled_at_deadline(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    # A blackholed lookup is cancelled at the deadline and degrades typed.
    api = _build_api(_EchoProvider())

    async def blackholed_lookup(node_id: NodeId) -> str | None:
        await anyio.sleep(60)
        return None

    monkeypatch.setattr(api, "_peer_api_url_for", blackholed_lookup)
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    started = anyio.current_time()
    result = await context.call_capability(
        NodeId("n-peer"),
        "echo",
        "1.0.0",
        _ECHO_REVISION,
        {"text": "x"},
        timeout_seconds=0.2,
    )
    elapsed = anyio.current_time() - started
    assert not result.ok and result.error is not None
    # Deadline exhaustion during resolution is a timeout, not a verdict that
    # the node is unreachable.
    assert result.error.code == "timeout"
    assert elapsed < 2.0


async def test_caller_invalid_timeout_fails_fast_on_remote_path() -> None:
    # An out-of-range timeout must fail fast as a typed error BEFORE it
    # becomes the reachability lookup budget on the remote path.
    api = _build_api(_EchoProvider())
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    for bad in (0.0, -5.0, 9_999.0):
        result = await context.call_capability(
            NodeId("n-peer"),
            "echo",
            "1.0.0",
            _ECHO_REVISION,
            {"text": "x"},
            timeout_seconds=bad,
        )
        assert not result.ok and result.error is not None
        assert result.error.code == "invalid_payload"


async def test_caller_non_numeric_timeout_is_typed() -> None:
    # An untyped extension can pass a non-numeric timeout; the comparison must
    # not raise TypeError out of call_capability.
    api = _build_api(_EchoProvider())
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    result = await context.call_capability(
        NodeId("n-peer"),
        "echo",
        "1.0.0",
        _ECHO_REVISION,
        {"text": "x"},
        timeout_seconds="10",  # pyright: ignore[reportArgumentType]
    )
    assert not result.ok and result.error is not None
    assert result.error.code == "invalid_payload"


async def test_deeply_nested_payload_is_typed_invalid_payload() -> None:
    # json.dumps raises RecursionError on very deep nesting; it must degrade
    # to a typed error at every guard site, never escape as an exception.
    # Depth far beyond any recursion limit so the failure is deterministic
    # regardless of the current stack depth; model_construct bypasses
    # pydantic's own recursion during test setup (the guard under test is the
    # dispatch's, not the envelope validator's).
    deep: dict[str, object] = {"leaf": 1}
    for _ in range(50_000):
        deep = {"nested": deep}
    call = CapabilityCall.model_construct(
        call_id="c-deep",
        capability_id="echo",
        version="1.0.0",
        descriptor_revision=_ECHO_REVISION,
        caller_node="caller",
        target_node="api-node",
        timeout_seconds=30.0,
        payload=deep,
    )
    result = await _dispatch(_build_api(_EchoProvider()), call)
    assert not result.ok and result.error is not None
    assert result.error.code == "invalid_payload"

    api = _build_api(_EchoProvider())
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    caller_result = await context.call_capability(
        NodeId("n-peer"), "echo", "1.0.0", _ECHO_REVISION, deep
    )
    assert not caller_result.ok and caller_result.error is not None
    assert caller_result.error.code == "invalid_payload"


async def test_cached_descriptor_cannot_invoke_withdrawn_provider() -> None:
    """A caller retaining a valid revision loses execution when readiness drops."""

    class DynamicProvider(_EchoProvider):
        ready = True
        calls = 0

        def capability_ready(self, qualified_id: str) -> bool:
            return self.ready

        async def handle_call(
            self, context: ExtensionContext, call: CapabilityCall
        ) -> dict[str, object]:
            self.calls += 1
            return await super().handle_call(context, call)

    provider = DynamicProvider()
    api = _build_api(provider)
    assert (await _dispatch(api, _call())).ok
    provider.ready = False
    result = await _dispatch(api, _call())
    assert not result.ok and result.error is not None
    assert result.error.code == "not_found"
    assert provider.calls == 1
    assert (await api.list_node_capabilities())["capabilities"] == []
    provider.ready = True
    assert (await _dispatch(api, _call())).ok
    assert provider.calls == 2
