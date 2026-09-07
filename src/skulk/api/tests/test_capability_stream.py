"""End-to-end provider server-streaming dispatch over the local DATA path."""

from collections.abc import AsyncIterator

import anyio
import pytest

from skulk.api.main import (
    API,
    _ActiveProviderStream,  # pyright: ignore[reportPrivateUsage]
)
from skulk.extensions import (
    CapabilityCall,
    CapabilityDescriptor,
    CapabilityError,
    CapabilityResult,
    CapabilityStreamError,
    CapabilityStreamFrame,
    CapabilityStreamInput,
    ExtensionContext,
    InlineMediaAttachment,
    LoadedExtensions,
    descriptor_revision,
)
from skulk.routing.provider_streams import ProviderStreamPacket
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.telemetry import TelemetryView
from skulk.utils.channels import channel

_TTS = CapabilityDescriptor(
    id="tts",
    version="1.0.0",
    title="Test speech synthesis",
    description="Streams deterministic PCM bytes for provider transport tests.",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
    io_mode="server_streaming",
    output_chunk_schema={
        "type": "object",
        "properties": {"format": {"const": "pcm_s16le"}},
        "required": ["format"],
        "additionalProperties": False,
    },
)
_TTS_REVISION = descriptor_revision(_TTS)

_BIDIRECTIONAL = CapabilityDescriptor(
    id="realtime-stt",
    version="1.0.0",
    title="Realtime STT",
    description="Consumes audio frames and emits progressive transcripts.",
    input_schema={"type": "object"},
    io_mode="bidirectional",
    input_chunk_schema={
        "type": "object",
        "properties": {"format": {"const": "pcm_s16le"}},
        "required": ["format"],
        "additionalProperties": False,
    },
    output_chunk_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
)

_CLIENT_STREAMING = CapabilityDescriptor(
    id="buffered-stt",
    version="1.0.0",
    title="Buffered STT",
    description="Consumes audio frames and returns one final transcript.",
    input_schema={"type": "object"},
    output_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
    io_mode="client_streaming",
    input_chunk_schema={"type": "object"},
)


class _TtsProvider:
    name = "tts-test"
    skulk_requires = ">=0"

    def chat_middleware(self) -> None:
        return None

    def capabilities(self) -> list[CapabilityDescriptor]:
        return [_TTS]

    async def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=1,
            kind="chunk",
            payload={"format": "pcm_s16le"},
            media=InlineMediaAttachment(
                data=b"\x00\xff\x80\x7f",
                media_type="audio/pcm",
                codec="pcm_s16le",
                sample_rate=24000,
                channels=1,
            ),
        )
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=2,
            kind="completed",
        )


class _InvalidChunkProvider(_TtsProvider):
    async def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=1,
            kind="chunk",
            payload={"format": "not-pcm"},
        )


class _TrailingFrameProvider(_TtsProvider):
    def __init__(self) -> None:
        self.cleanup_started = anyio.Event()
        self.release_cleanup = anyio.Event()

    async def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        try:
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=1,
                kind="completed",
            )
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=2,
                kind="chunk",
                payload={"format": "pcm_s16le"},
            )
        finally:
            self.cleanup_started.set()
            await self.release_cleanup.wait()


class _RejectingTtsProvider(_TtsProvider):
    def __init__(self) -> None:
        self.handler_called = False

    async def admit_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> CapabilityError | None:
        return CapabilityError(
            code="not_found",
            message="no eligible model is mounted",
        )

    async def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        self.handler_called = True
        if False:
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=1,
                kind="completed",
            )


class _BlockingAdmissionTtsProvider(_TtsProvider):
    def __init__(self) -> None:
        self.admission_started = anyio.Event()
        self.release_admission = anyio.Event()

    async def admit_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> CapabilityError | None:
        self.admission_started.set()
        await self.release_admission.wait()
        return None


class _CancellableProvider(_TtsProvider):
    def __init__(self) -> None:
        self.cancelled = anyio.Event()

    async def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        try:
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=1,
                kind="chunk",
                payload={"format": "pcm_s16le"},
            )
            await anyio.sleep_forever()
        finally:
            self.cancelled.set()


class _TerminalFinalizationProvider(_TtsProvider):
    def __init__(self) -> None:
        self.finalization_started = anyio.Event()
        self.release_finalization = anyio.Event()
        self.output_stream: AsyncIterator[CapabilityStreamFrame] | None = None

    def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        self.output_stream = self._frames(call)
        return self.output_stream

    async def _frames(
        self,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        try:
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=1,
                kind="completed",
            )
        finally:
            self.finalization_started.set()
            await self.release_finalization.wait()


class _BurstProvider(_TtsProvider):
    async def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        for sequence in range(1, 302):
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=sequence,
                kind="chunk",
                payload={"format": "pcm_s16le"},
            )
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=302,
            kind="completed",
        )


class _FloodProvider(_TtsProvider):
    def __init__(self) -> None:
        self.cancelled = anyio.Event()

    async def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        sequence = 1
        try:
            while True:
                yield CapabilityStreamFrame(
                    call_id=call.call_id,
                    direction="provider_to_caller",
                    sequence=sequence,
                    kind="chunk",
                    payload={"format": "pcm_s16le"},
                )
                sequence += 1
                await anyio.sleep(0)
        finally:
            self.cancelled.set()


class _BidirectionalProvider(_TtsProvider):
    def __init__(self) -> None:
        self.input_frames: list[CapabilityStreamFrame] = []

    def capabilities(self) -> list[CapabilityDescriptor]:
        return [_BIDIRECTIONAL]

    async def handle_input_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> AsyncIterator[CapabilityStreamFrame]:
        async for frame in input_frames:
            self.input_frames.append(frame)
            if frame.is_terminal:
                break
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=1,
            kind="chunk",
            payload={"text": "heard"},
        )
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=2,
            kind="completed",
        )


class _MixedStreamingProvider(_BidirectionalProvider):
    def capabilities(self) -> list[CapabilityDescriptor]:
        return [_TTS, _BIDIRECTIONAL]


class _ClientStreamingProvider(_BidirectionalProvider):
    def capabilities(self) -> list[CapabilityDescriptor]:
        return [_CLIENT_STREAMING]

    async def handle_input_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> AsyncIterator[CapabilityStreamFrame]:
        async for frame in input_frames:
            self.input_frames.append(frame)
            if frame.is_terminal:
                break
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=1,
            kind="completed",
            payload={"text": "buffered transcript"},
        )


class _EmptyClientStreamingProvider(_ClientStreamingProvider):
    async def handle_input_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> AsyncIterator[CapabilityStreamFrame]:
        async for frame in input_frames:
            if frame.is_terminal:
                break
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=1,
            kind="completed",
        )


class _ImmediateFailureInputProvider(_BidirectionalProvider):
    async def handle_input_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> AsyncIterator[CapabilityStreamFrame]:
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=1,
            kind="failed",
            error=CapabilityStreamError(
                code="provider_error",
                message="provider failed before reading input",
            ),
        )


def _build_api(provider: object, *, provider_transport_buffer: int = 256) -> API:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    provider_sender, provider_receiver = channel[ProviderStreamPacket](
        provider_transport_buffer
    )
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
        provider_stream_sender=provider_sender,
        provider_stream_receiver=provider_receiver,
        extensions=LoadedExtensions([provider]),  # pyright: ignore[reportArgumentType]
    )


async def _collect_local_stream(
    api: API,
) -> tuple[bool, list[CapabilityStreamFrame]]:
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    opened = False
    frames: list[CapabilityStreamFrame] = []
    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await context.stream_capability(
            NodeId("api-node"),
            "tts",
            "1.0.0",
            _TTS_REVISION,
            {"text": "hello"},
            timeout_seconds=2.0,
        )
        opened = session.open_result.ok
        assert session.input is None
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()
    return opened, frames


async def test_local_provider_stream_preserves_lifecycle_and_binary_media() -> None:
    api = _build_api(_TtsProvider())
    opened, frames = await _collect_local_stream(api)

    assert opened is True
    assert [frame.kind for frame in frames] == ["started", "chunk", "completed"]
    assert [frame.sequence for frame in frames] == [0, 1, 2]
    assert isinstance(frames[1].media, InlineMediaAttachment)
    assert frames[1].media.data == b"\x00\xff\x80\x7f"
    diagnostics = api._provider_observer.snapshot(  # pyright: ignore[reportPrivateUsage]
        active_unary_calls=0,
        stream_slots_in_use=0,
        unary_concurrency_limit=8,
        stream_concurrency_limit=8,
    )
    tts = diagnostics.capabilities["tts@1.0.0"]
    assert tts.admitted_streams == 1
    assert tts.output_frames == 3
    assert tts.output_media_bytes == 4
    assert tts.completed_streams == 1
    assert tts.missing_terminal_streams == 0


async def test_provider_terminal_waits_for_handler_finalization() -> None:
    provider = _TerminalFinalizationProvider()
    api = _build_api(provider)
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    terminal_received = anyio.Event()
    terminal_frames: list[CapabilityStreamFrame] = []

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await context.stream_capability(
            NodeId("api-node"),
            "tts",
            "1.0.0",
            _TTS_REVISION,
            {"text": "finalize before terminal"},
            timeout_seconds=2.0,
        )
        assert session.open_result.ok is True
        iterator = session.frames.__aiter__()
        assert (await iterator.__anext__()).kind == "started"

        async def receive_terminal() -> None:
            terminal_frames.append(await iterator.__anext__())
            terminal_received.set()

        task_group.start_soon(receive_terminal)
        try:
            with anyio.fail_after(1.0):
                await provider.finalization_started.wait()
            assert terminal_received.is_set() is False
        finally:
            provider.release_finalization.set()

        with anyio.fail_after(1.0):
            await terminal_received.wait()
        task_group.cancel_scope.cancel()

    assert [frame.kind for frame in terminal_frames] == ["completed"]


async def test_mixed_provider_uses_descriptor_io_mode_for_handler() -> None:
    opened, frames = await _collect_local_stream(_build_api(_MixedStreamingProvider()))

    assert opened is True
    assert [frame.kind for frame in frames] == ["started", "chunk", "completed"]
    assert isinstance(frames[1].media, InlineMediaAttachment)


async def test_local_bidirectional_media_half_close_keeps_output_active() -> None:
    provider = _BidirectionalProvider()
    api = _build_api(provider)
    frames: list[CapabilityStreamFrame] = []

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await api._extension_context.stream_capability(  # pyright: ignore[reportPrivateUsage]
            NodeId("api-node"),
            "realtime-stt",
            "1.0.0",
            descriptor_revision(_BIDIRECTIONAL),
            {},
            timeout_seconds=2.0,
        )
        assert session.open_result.ok is True
        assert session.input is not None
        await session.input.send_chunk(
            payload={"format": "pcm_s16le"},
            media=InlineMediaAttachment(
                data=b"\x00\x01\x02\x03",
                media_type="audio/pcm",
                codec="pcm_s16le",
                sample_rate=16000,
                channels=1,
            ),
        )
        await session.input.complete()
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()

    assert [frame.kind for frame in provider.input_frames] == [
        "started",
        "chunk",
        "completed",
    ]
    assert [frame.sequence for frame in provider.input_frames] == [0, 1, 2]
    input_media = provider.input_frames[1].media
    assert isinstance(input_media, InlineMediaAttachment)
    assert input_media.data == b"\x00\x01\x02\x03"
    assert [frame.kind for frame in frames] == ["started", "chunk", "completed"]
    assert frames[1].payload == {"text": "heard"}


async def test_invalid_input_chunk_fails_provider_output() -> None:
    provider = _BidirectionalProvider()
    api = _build_api(provider)
    frames: list[CapabilityStreamFrame] = []

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await api._extension_context.stream_capability(  # pyright: ignore[reportPrivateUsage]
            NodeId("api-node"),
            "realtime-stt",
            "1.0.0",
            descriptor_revision(_BIDIRECTIONAL),
            {},
            timeout_seconds=2.0,
        )
        assert session.input is not None
        await session.input.send_chunk(payload={"format": "not-pcm"})
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()

    assert [frame.kind for frame in frames] == ["started", "failed"]
    assert frames[-1].error is not None
    assert frames[-1].error.code == "invalid_frame"
    assert [frame.kind for frame in provider.input_frames] == [
        "started",
        "failed",
    ]
    assert provider.input_frames[-1].error is not None
    assert provider.input_frames[-1].error.code == "invalid_frame"


async def test_client_streaming_half_close_returns_structured_final_result() -> None:
    provider = _ClientStreamingProvider()
    api = _build_api(provider)
    frames: list[CapabilityStreamFrame] = []

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await api._extension_context.stream_capability(  # pyright: ignore[reportPrivateUsage]
            NodeId("api-node"),
            "buffered-stt",
            "1.0.0",
            descriptor_revision(_CLIENT_STREAMING),
            {},
            timeout_seconds=2.0,
        )
        assert session.input is not None
        await session.input.send_chunk(payload={"frame": 1})
        await session.input.complete()
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()

    assert [frame.kind for frame in frames] == ["started", "completed"]
    assert frames[-1].payload == {"text": "buffered transcript"}


async def test_client_streaming_rejects_empty_required_final_result() -> None:
    api = _build_api(_EmptyClientStreamingProvider())
    frames: list[CapabilityStreamFrame] = []

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await api._extension_context.stream_capability(  # pyright: ignore[reportPrivateUsage]
            NodeId("api-node"),
            "buffered-stt",
            "1.0.0",
            descriptor_revision(_CLIENT_STREAMING),
            {},
            timeout_seconds=2.0,
        )
        assert session.input is not None
        await session.input.complete()
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()

    assert [frame.kind for frame in frames] == ["started", "failed"]
    assert frames[-1].error is not None
    assert frames[-1].error.code == "invalid_frame"


async def test_caller_input_cancellation_terminates_provider_output() -> None:
    api = _build_api(_BidirectionalProvider())
    frames: list[CapabilityStreamFrame] = []

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await api._extension_context.stream_capability(  # pyright: ignore[reportPrivateUsage]
            NodeId("api-node"),
            "realtime-stt",
            "1.0.0",
            descriptor_revision(_BIDIRECTIONAL),
            {},
            timeout_seconds=2.0,
        )
        assert session.input is not None
        await session.input.cancel("microphone disconnected")
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()

    assert [frame.kind for frame in frames] == ["started", "cancelled"]
    assert frames[-1].error is not None
    assert frames[-1].error.code == "cancelled"
    assert frames[-1].error.message == "microphone disconnected"


async def test_terminal_output_racing_input_start_returns_typed_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_start = CapabilityStreamInput.start

    async def delayed_start(stream: CapabilityStreamInput) -> None:
        await anyio.sleep(0.05)
        await original_start(stream)

    monkeypatch.setattr(CapabilityStreamInput, "start", delayed_start)
    api = _build_api(_ImmediateFailureInputProvider())
    frames: list[CapabilityStreamFrame] = []

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await api._extension_context.stream_capability(  # pyright: ignore[reportPrivateUsage]
            NodeId("api-node"),
            "realtime-stt",
            "1.0.0",
            descriptor_revision(_BIDIRECTIONAL),
            {},
            timeout_seconds=2.0,
        )
        assert session.open_result.ok is True
        assert session.input is not None
        assert session.input.closed is True
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()

    assert [frame.kind for frame in frames] == ["started", "failed"]
    assert frames[-1].error is not None
    assert frames[-1].error.code == "provider_error"


async def test_remote_open_uses_peer_api_but_media_uses_provider_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fabric_sender, fabric_receiver = channel[ProviderStreamPacket](256)

    def build_node(
        node_id: str,
        *,
        provider: object | None,
        stream_sender: object | None,
        stream_receiver: object | None,
    ) -> API:
        command_sender, _ = channel[ForwarderCommand]()
        download_sender, _ = channel[ForwarderDownloadCommand]()
        _, event_receiver = channel[IndexedEvent]()
        _, election_receiver = channel[ElectionMessage]()
        return API(
            NodeId(node_id),
            port=52415,
            event_receiver=event_receiver,
            command_sender=command_sender,
            download_command_sender=download_sender,
            election_receiver=election_receiver,
            enable_event_log=False,
            mount_dashboard=False,
            telemetry_view=TelemetryView(),
            provider_stream_sender=stream_sender,  # type: ignore[arg-type]
            provider_stream_receiver=stream_receiver,  # type: ignore[arg-type]
            extensions=(
                LoadedExtensions([provider])  # pyright: ignore[reportArgumentType]
                if provider is not None
                else None
            ),
        )

    provider_api = build_node(
        "provider-node",
        provider=_TtsProvider(),
        stream_sender=fabric_sender,
        stream_receiver=None,
    )
    caller_api = build_node(
        "caller-node",
        provider=None,
        stream_sender=None,
        stream_receiver=fabric_receiver,
    )

    async def peer_url(node_id: NodeId) -> str | None:
        return "http://provider.test" if node_id == NodeId("provider-node") else None

    monkeypatch.setattr(caller_api, "_peer_api_url_for", peer_url)

    class _Response:
        def __init__(self, result: object) -> None:
            self._result = result

        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return self._result

    class _PeerClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "_PeerClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def post(self, url: str, *, json: object) -> _Response:
            assert url.endswith("/v1/capabilities/stream")
            call = CapabilityCall.model_validate(json)
            result = await provider_api.serve_capability_stream(call)
            return _Response(result.model_dump(mode="json"))

    monkeypatch.setattr("skulk.api.main.httpx.AsyncClient", _PeerClient)

    opened = False
    frames: list[CapabilityStreamFrame] = []
    async with (
        provider_api._tg as provider_tasks,  # pyright: ignore[reportPrivateUsage]
        caller_api._tg as caller_tasks,  # pyright: ignore[reportPrivateUsage]
    ):
        caller_tasks.start_soon(
            caller_api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await caller_api._extension_context.stream_capability(  # pyright: ignore[reportPrivateUsage]
            NodeId("provider-node"),
            "tts",
            "1.0.0",
            _TTS_REVISION,
            {"text": "remote hello"},
            timeout_seconds=2.0,
        )
        opened = session.open_result.ok
        assert session.input is None
        frames = [frame async for frame in session.frames]
        caller_tasks.cancel_scope.cancel()
        provider_tasks.cancel_scope.cancel()

    assert opened is True
    assert [frame.kind for frame in frames] == ["started", "chunk", "completed"]
    assert isinstance(frames[1].media, InlineMediaAttachment)
    assert frames[1].media.data == b"\x00\xff\x80\x7f"


async def test_remote_bidirectional_input_and_output_use_provider_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_send, provider_receive = channel[ProviderStreamPacket](256)
    provider_send, caller_receive = channel[ProviderStreamPacket](256)
    provider = _BidirectionalProvider()

    def build_node(
        node_id: str,
        *,
        extension: object | None,
        sender: object,
        receiver: object,
    ) -> API:
        command_sender, _ = channel[ForwarderCommand]()
        download_sender, _ = channel[ForwarderDownloadCommand]()
        _, event_receiver = channel[IndexedEvent]()
        _, election_receiver = channel[ElectionMessage]()
        return API(
            NodeId(node_id),
            port=52415,
            event_receiver=event_receiver,
            command_sender=command_sender,
            download_command_sender=download_sender,
            election_receiver=election_receiver,
            enable_event_log=False,
            mount_dashboard=False,
            telemetry_view=TelemetryView(),
            provider_stream_sender=sender,  # type: ignore[arg-type]
            provider_stream_receiver=receiver,  # type: ignore[arg-type]
            extensions=(
                LoadedExtensions([extension])  # pyright: ignore[reportArgumentType]
                if extension is not None
                else None
            ),
        )

    provider_api = build_node(
        "provider-node",
        extension=provider,
        sender=provider_send,
        receiver=provider_receive,
    )
    caller_api = build_node(
        "caller-node",
        extension=None,
        sender=caller_send,
        receiver=caller_receive,
    )

    async def peer_url(node_id: NodeId) -> str | None:
        return "http://provider.test" if node_id == NodeId("provider-node") else None

    monkeypatch.setattr(caller_api, "_peer_api_url_for", peer_url)

    class _Response:
        def __init__(self, result: CapabilityResult) -> None:
            self._result = result

        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return self._result.model_dump(mode="json")

    class _PeerClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "_PeerClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def post(self, url: str, *, json: object) -> _Response:
            assert url.endswith("/v1/capabilities/stream")
            return _Response(
                await provider_api.serve_capability_stream(
                    CapabilityCall.model_validate(json)
                )
            )

    monkeypatch.setattr("skulk.api.main.httpx.AsyncClient", _PeerClient)
    output: list[CapabilityStreamFrame] = []

    async with (
        provider_api._tg as provider_tasks,  # pyright: ignore[reportPrivateUsage]
        caller_api._tg as caller_tasks,  # pyright: ignore[reportPrivateUsage]
    ):
        provider_tasks.start_soon(
            provider_api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        caller_tasks.start_soon(
            caller_api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await caller_api._extension_context.stream_capability(  # pyright: ignore[reportPrivateUsage]
            NodeId("provider-node"),
            "realtime-stt",
            "1.0.0",
            descriptor_revision(_BIDIRECTIONAL),
            {},
            timeout_seconds=2.0,
        )
        assert session.open_result.ok is True
        assert session.input is not None
        await session.input.send_chunk(
            payload={"format": "pcm_s16le"},
            media=InlineMediaAttachment(
                data=b"remote-audio",
                media_type="audio/pcm",
                codec="pcm_s16le",
                sample_rate=16000,
                channels=1,
            ),
        )
        await session.input.complete()
        output = [frame async for frame in session.frames]
        caller_tasks.cancel_scope.cancel()
        provider_tasks.cancel_scope.cancel()

    assert [frame.kind for frame in provider.input_frames] == [
        "started",
        "chunk",
        "completed",
    ]
    media = provider.input_frames[1].media
    assert isinstance(media, InlineMediaAttachment)
    assert media.data == b"remote-audio"
    assert [frame.kind for frame in output] == ["started", "chunk", "completed"]


async def test_invalid_provider_chunk_becomes_typed_failed_terminal() -> None:
    opened, frames = await _collect_local_stream(_build_api(_InvalidChunkProvider()))

    assert opened is True
    assert [frame.kind for frame in frames] == ["started", "failed"]
    assert frames[-1].error is not None
    assert frames[-1].error.code == "invalid_frame"


async def test_provider_frame_after_terminal_becomes_typed_failure() -> None:
    provider = _TrailingFrameProvider()
    api = _build_api(provider)
    frames: list[CapabilityStreamFrame] = []
    stream_finished = anyio.Event()
    opened = False

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await api._extension_context.stream_capability(  # pyright: ignore[reportPrivateUsage]
            NodeId("api-node"),
            "tts",
            "1.0.0",
            _TTS_REVISION,
            {"text": "invalid trailing frame"},
            timeout_seconds=2.0,
        )
        opened = session.open_result.ok

        async def collect_frames() -> None:
            frames.extend([frame async for frame in session.frames])
            stream_finished.set()

        task_group.start_soon(collect_frames)
        try:
            with anyio.fail_after(1.0):
                await provider.cleanup_started.wait()
            assert stream_finished.is_set() is False
        finally:
            provider.release_cleanup.set()
        with anyio.fail_after(1.0):
            await stream_finished.wait()
        task_group.cancel_scope.cancel()

    assert opened is True
    assert [frame.kind for frame in frames] == ["started", "failed"]
    assert frames[-1].error is not None
    assert frames[-1].error.code == "invalid_frame"


async def test_dynamic_admission_rejection_emits_no_started_frame() -> None:
    provider = _RejectingTtsProvider()
    api = _build_api(provider)
    call = CapabilityCall(
        call_id="rejected-before-start",
        capability_id="tts",
        version="1.0.0",
        descriptor_revision=_TTS_REVISION,
        caller_node="api-node",
        target_node="api-node",
        timeout_seconds=2.0,
        payload={"text": "hello"},
    )

    result: CapabilityResult | None = None
    async with api._tg:  # pyright: ignore[reportPrivateUsage]
        result = await api.serve_capability_stream(call)

    assert result is not None
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "not_found"
    assert provider.handler_called is False
    assert api._active_capability_streams == {}  # pyright: ignore[reportPrivateUsage]
    diagnostics = api._provider_observer.snapshot(  # pyright: ignore[reportPrivateUsage]
        active_unary_calls=0,
        stream_slots_in_use=0,
        unary_concurrency_limit=8,
        stream_concurrency_limit=8,
    )
    assert diagnostics.capabilities["tts@1.0.0"].rejected_streams == 1


async def test_dynamic_admission_reserves_concurrency_slot_before_await(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skulk.api.main._MAX_CONCURRENT_CAPABILITY_STREAMS", 1)
    provider = _BlockingAdmissionTtsProvider()
    api = _build_api(provider)

    def call(call_id: str) -> CapabilityCall:
        return CapabilityCall(
            call_id=call_id,
            capability_id="tts",
            version="1.0.0",
            descriptor_revision=_TTS_REVISION,
            caller_node="api-node",
            target_node="api-node",
            timeout_seconds=2.0,
            payload={"text": "hello"},
        )

    first_results: list[CapabilityResult] = []

    async def open_first() -> None:
        first_results.append(await api.serve_capability_stream(call("first")))

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(open_first)
        await provider.admission_started.wait()

        second_result = await api.serve_capability_stream(call("second"))
        assert second_result.ok is False
        assert second_result.error is not None
        assert second_result.error.code == "overloaded"
        assert list(api._active_capability_streams) == ["first"]  # pyright: ignore[reportPrivateUsage]

        provider.release_admission.set()
        while not first_results:
            await anyio.sleep(0)
        assert first_results[0].ok is True
        task_group.cancel_scope.cancel()


def test_bidirectional_descriptor_registers_input_stream_handler() -> None:
    loaded = LoadedExtensions([_BidirectionalProvider()])

    assert loaded.capability_descriptors == (_BIDIRECTIONAL,)
    assert loaded.stream_handler(_BIDIRECTIONAL.qualified_id) is not None


async def test_early_caller_close_cancels_only_its_provider_stream() -> None:
    provider = _CancellableProvider()
    api = _build_api(provider)
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await context.stream_capability(
            NodeId("api-node"),
            "tts",
            "1.0.0",
            _TTS_REVISION,
            {"text": "cancel me"},
            timeout_seconds=5.0,
        )
        assert session.open_result.ok is True
        iterator = session.frames.__aiter__()
        assert (await iterator.__anext__()).kind == "started"
        assert (await iterator.__anext__()).kind == "chunk"
        await session.frames.aclose()  # type: ignore[attr-defined]
        with anyio.fail_after(1.0):
            await provider.cancelled.wait()
        task_group.cancel_scope.cancel()


async def test_caller_stream_can_be_finalized_by_a_different_task() -> None:
    """Async-generator finalization must not inherit an open deadline scope."""

    provider = _CancellableProvider()
    api = _build_api(provider)
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await context.stream_capability(
            NodeId("api-node"),
            "tts",
            "1.0.0",
            _TTS_REVISION,
            {"text": "finalize me elsewhere"},
            timeout_seconds=5.0,
        )
        assert session.open_result.ok is True
        iterator = session.frames.__aiter__()
        assert (await iterator.__anext__()).kind == "started"

        async def close_from_child_task() -> None:
            await session.frames.aclose()  # type: ignore[attr-defined]

        task_group.start_soon(close_from_child_task)
        with anyio.fail_after(1.0):
            await provider.cancelled.wait()
        task_group.cancel_scope.cancel()


async def test_cancel_racing_admission_still_emits_started_first() -> None:
    api = _build_api(_TtsProvider())
    call = CapabilityCall(
        call_id="cancel-before-start",
        capability_id="tts",
        version="1.0.0",
        descriptor_revision=_TTS_REVISION,
        caller_node="api-node",
        target_node="api-node",
        timeout_seconds=2.0,
        payload={"text": "cancel immediately"},
    )
    cancel_requested = anyio.Event()
    cancel_requested.set()
    active = _ActiveProviderStream(
        caller_node="api-node",
        cancel_requested=cancel_requested,
        descriptor=_TTS,
    )

    await api._run_capability_stream(  # pyright: ignore[reportPrivateUsage]
        call,
        "tts-test",
        _TtsProvider(),
        _TTS,
        active,
    )
    assert api._provider_stream_receiver is not None  # pyright: ignore[reportPrivateUsage]
    first = await api._provider_stream_receiver.receive()  # pyright: ignore[reportPrivateUsage]
    second = await api._provider_stream_receiver.receive()  # pyright: ignore[reportPrivateUsage]

    assert [first.frame.kind, second.frame.kind] == ["started", "cancelled"]
    assert [first.frame.sequence, second.frame.sequence] == [0, 1]


async def test_caller_queue_overflow_cannot_report_truncated_stream_complete() -> None:
    # The simulated transport must hold more than the caller's 256-frame queue;
    # otherwise upstream backpressure prevents the queue under test from ever
    # overflowing and the provider legitimately completes the whole stream.
    api = _build_api(_BurstProvider(), provider_transport_buffer=512)
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    opened = False
    frames: list[CapabilityStreamFrame] = []

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await context.stream_capability(
            NodeId("api-node"),
            "tts",
            "1.0.0",
            _TTS_REVISION,
            {"text": "overflow the bounded caller queue"},
            timeout_seconds=5.0,
        )
        opened = session.open_result.ok
        with anyio.fail_after(1.0):
            while api._provider_stream_receivers:  # pyright: ignore[reportPrivateUsage]
                await anyio.sleep(0.01)
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()

    assert opened is True
    assert frames[0].kind == "started"
    assert frames[-1].kind == "failed"
    assert frames[-1].error is not None
    assert frames[-1].error.code == "transport_error"
    assert all(frame.kind != "completed" for frame in frames)
    assert [frame.sequence for frame in frames] == list(range(len(frames)))


async def test_non_consuming_caller_overflow_cancels_provider_immediately() -> None:
    provider = _FloodProvider()
    api = _build_api(provider)
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await context.stream_capability(
            NodeId("api-node"),
            "tts",
            "1.0.0",
            _TTS_REVISION,
            {"text": "do not consume this stream"},
            timeout_seconds=5.0,
        )
        assert session.open_result.ok is True
        with anyio.fail_after(1.0):
            await provider.cancelled.wait()
        await session.frames.aclose()  # type: ignore[attr-defined]
        task_group.cancel_scope.cancel()


async def test_readiness_withdrawal_during_admission_prevents_stream_execution() -> (
    None
):
    """An accepted dynamic probe cannot reopen a provider disabled while it awaited."""

    class WithdrawingProvider(_TtsProvider):
        ready = True
        invoked = False

        def capability_ready(self, qualified_id: str) -> bool:
            return self.ready

        async def admit_stream(
            self, context: ExtensionContext, call: CapabilityCall
        ) -> CapabilityError | None:
            await anyio.sleep(0)
            self.ready = False
            return None

        async def handle_stream(
            self, context: ExtensionContext, call: CapabilityCall
        ) -> AsyncIterator[CapabilityStreamFrame]:
            self.invoked = True
            async for frame in super().handle_stream(context, call):
                yield frame

    provider = WithdrawingProvider()
    opened, frames = await _collect_local_stream(_build_api(provider))
    assert not opened
    assert frames == []
    assert not provider.invoked
