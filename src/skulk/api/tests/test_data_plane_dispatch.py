"""API data-plane demux coverage (#279 Phase 2).

The API routes each output chunk to the per-command stream queue by type. This
exercises `_dispatch_generation_chunk` directly (the shared routing used by the
DATA-plane consumer) without standing up the full gossip path.
"""

from datetime import datetime, timedelta, timezone
from typing import cast
from unittest.mock import AsyncMock

import anyio
import pytest

import skulk.api.main as api_main
from skulk.api.main import API
from skulk.shared.election import ElectionMessage
from skulk.shared.models.model_cards import AudioResponseFormat, ModelId
from skulk.shared.types.chunks import (
    AudioChunk,
    DataChunk,
    EmbeddingChunk,
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
    TranscriptionChunk,
)
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.telemetry import NODE_LIVENESS_TIMEOUT, NodeTelemetry
from skulk.utils.channels import channel


def _build_api(*, data_plane_zenoh: bool = False) -> API:
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
        data_plane_zenoh=data_plane_zenoh,
    )


def _data_frame(
    command_id: CommandId,
    sequence: int,
    chunk: TokenChunk,
) -> DataChunk:
    """Build one legacy-compatible payload frame for reorder tests."""

    return DataChunk(command_id=command_id, sequence=sequence, chunk=chunk)


def test_reorder_buffer_default_follows_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #279 Phase 3 (transport-conditional): the buffer defaults ON for gossipsub
    # (it reorders) and OFF for Zenoh (per-publisher FIFO). The env var overrides.
    monkeypatch.delenv("SKULK_DATA_REORDER_BUFFER", raising=False)
    assert _build_api()._reorder_buffer_enabled is True  # pyright: ignore[reportPrivateUsage]
    assert _build_api(data_plane_zenoh=True)._reorder_buffer_enabled is False  # pyright: ignore[reportPrivateUsage]
    # explicit override beats the transport default in both directions
    monkeypatch.setenv("SKULK_DATA_REORDER_BUFFER", "1")
    assert _build_api(data_plane_zenoh=True)._reorder_buffer_enabled is True  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setenv("SKULK_DATA_REORDER_BUFFER", "0")
    assert _build_api()._reorder_buffer_enabled is False  # pyright: ignore[reportPrivateUsage]


async def test_state_surfaces_split_data_transport_health() -> None:
    api = _build_api(data_plane_zenoh=True)
    management_node = NodeId("remote-management-node")
    worker_node = NodeId("worker-node")
    now = datetime.now(tz=timezone.utc)
    # Replicated membership tracks the worker but can omit a remote API-only
    # participant whose resource reading arrives exclusively on telemetry.
    api.state = api.state.model_copy(update={"last_seen": {worker_node: now}})
    api._telemetry_view.apply(  # pyright: ignore[reportPrivateUsage]
        NodeTelemetry(
            node_id=management_node,
            info=NodeResources(
                backends=frozenset(),
                participation="management",
                data_transport="zenoh",
            ),
        ),
        received_at=now,
    )
    api._telemetry_view.apply(  # pyright: ignore[reportPrivateUsage]
        NodeTelemetry(
            node_id=worker_node,
            info=NodeResources(data_transport="gossipsub"),
        ),
        received_at=now,
    )

    payload = await api.get_cluster_state()

    assert payload["nodeResources"] == {
        "remote-management-node": {
            "backends": [],
            "engineBuilds": {},
            "llamaServerSettings": None,
            "hardwareClasses": [],
            "participation": "management",
            "apiAvailable": True,
            "dataTransport": "zenoh",
            "zenohConnectedPeers": None,
            "capabilityConflicts": [],
        },
        "worker-node": {
            "backends": ["mlx"],
            "engineBuilds": {},
            "llamaServerSettings": None,
            "hardwareClasses": [],
            "participation": "full",
            "apiAvailable": True,
            "dataTransport": "gossipsub",
            "zenohConnectedPeers": None,
            "capabilityConflicts": [],
        },
    }
    health = cast("dict[str, object]", payload["nodeHealth"])
    assert isinstance(health, dict)
    for node_id in ("remote-management-node", "worker-node"):
        summary = cast("dict[str, object]", health[node_id])
        assert isinstance(summary, dict)
        assert summary["level"] == "error"
        reasons = cast("list[dict[str, object]]", summary["reasons"])
        assert isinstance(reasons, list)
        assert any(
            reason.get("code") == "data_transport_mismatch"
            for reason in reasons
        )


async def test_state_omits_stale_telemetry_only_participant() -> None:
    api = _build_api()
    stale_node = NodeId("stale-management-node")
    stale_at = (
        datetime.now(tz=timezone.utc)
        - NODE_LIVENESS_TIMEOUT
        - timedelta(seconds=1)
    )
    api._telemetry_view.apply(  # pyright: ignore[reportPrivateUsage]
        NodeTelemetry(
            node_id=stale_node,
            info=NodeResources(
                backends=frozenset(),
                participation="management",
            ),
        ),
        received_at=stale_at,
    )

    payload = await api.get_cluster_state()

    assert payload["nodeResources"] == {}
    assert payload["nodeHealth"] == {}


@pytest.mark.asyncio
async def test_reorder_buffer_delivers_chunks_in_sequence_order() -> None:
    # #279 Phase 2b: the DATA gossip topic has no total order, so a multi-node
    # producer's chunks can arrive out of order. _reorder_and_dispatch must
    # buffer by sequence and release in order. Feed 1,0,3,2 -> expect 0,1,2,3.
    api = _build_api()
    cmd = CommandId("cmd-reorder")
    send, recv = channel[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
    ]()
    api._text_generation_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]

    def tok(i: int) -> TokenChunk:
        return TokenChunk(
            model=ModelId("mlx-community/test"),
            text=f"t{i}",
            token_id=i,
            usage=None,
            finish_reason=None,
        )

    for seq in (1, 0, 3, 2):
        await api._reorder_and_dispatch(  # pyright: ignore[reportPrivateUsage]
            _data_frame(cmd, seq, tok(seq))
        )

    got: list[int] = []
    with recv as stream:
        for _ in range(4):
            c = stream.receive_nowait()
            assert isinstance(c, TokenChunk)
            got.append(c.token_id)
    assert got == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_reorder_buffer_drops_duplicate_and_late_sequences() -> None:
    # A sequence at/below the cursor (duplicate or late re-send) must be dropped,
    # not re-delivered.
    api = _build_api()
    cmd = CommandId("cmd-dup")
    send, recv = channel[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
    ]()
    api._text_generation_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]

    def tok(i: int) -> TokenChunk:
        return TokenChunk(
            model=ModelId("mlx-community/test"),
            text=f"t{i}",
            token_id=i,
            usage=None,
            finish_reason=None,
        )

    await api._reorder_and_dispatch(  # pyright: ignore[reportPrivateUsage]
        _data_frame(cmd, 0, tok(0))
    )
    await api._reorder_and_dispatch(  # pyright: ignore[reportPrivateUsage]
        _data_frame(cmd, 1, tok(1))
    )
    await api._reorder_and_dispatch(  # pyright: ignore[reportPrivateUsage]
        _data_frame(cmd, 0, tok(99))
    )

    got: list[int] = []
    with recv as stream:
        for _ in range(2):
            c = stream.receive_nowait()
            assert isinstance(c, TokenChunk)
            got.append(c.token_id)
        assert got == [0, 1]
        # the duplicate of seq 0 produced nothing more
        with pytest.raises(anyio.WouldBlock):
            stream.receive_nowait()


@pytest.mark.asyncio
async def test_reorder_gap_flush_fails_stream_after_a_dropped_sequence() -> None:
    # A gap that outlives the bounded repair window is a transport failure, not
    # permission to return incomplete or transposed output. Here seq 0 is lost;
    # seq 1 and 2 buffer until the sweep synthesizes one terminal ErrorChunk.
    api = _build_api()
    cmd = CommandId("cmd-gap")
    send, recv = channel[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
    ]()
    api._text_generation_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]

    def tok(i: int) -> TokenChunk:
        return TokenChunk(
            model=ModelId("mlx-community/test"),
            text=f"t{i}",
            token_id=i,
            usage=None,
            finish_reason=None,
        )

    # seq 0 never arrives; 1 and 2 buffer behind the gap -> nothing dispatched
    await api._reorder_and_dispatch(  # pyright: ignore[reportPrivateUsage]
        _data_frame(cmd, 1, tok(1))
    )
    await api._reorder_and_dispatch(  # pyright: ignore[reportPrivateUsage]
        _data_frame(cmd, 2, tok(2))
    )
    with recv as stream:
        with pytest.raises(anyio.WouldBlock):
            stream.receive_nowait()  # confirm the gap holds everything

        # age the gap past the flush window and run one sweep pass
        state = api._chunk_reorder[cmd]  # pyright: ignore[reportPrivateUsage]
        assert state.gap_since is not None
        flush_window = api_main._REORDER_GAP_FLUSH_SECONDS  # pyright: ignore[reportPrivateUsage]
        await api._flush_stale_reorder_gaps(  # pyright: ignore[reportPrivateUsage]
            state.gap_since + flush_window + 1.0
        )

        terminal = stream.receive_nowait()
        assert isinstance(terminal, ErrorChunk)
        assert "Data-plane transport failure" in terminal.error_message
        assert cmd not in api._chunk_reorder  # pyright: ignore[reportPrivateUsage]
        diagnostics = api._data_plane_observer.snapshot()  # pyright: ignore[reportPrivateUsage]
        assert diagnostics.transport_failures == 1


@pytest.mark.asyncio
async def test_reorder_gap_timer_not_refreshed_by_later_chunks() -> None:
    # #301 review (Codex P2): the gap timer must measure how long THIS gap has
    # been stuck, not refresh on every chunk that arrives behind it — else a
    # stream that keeps receiving later chunks (with a dropped early seq) never
    # ages to the flush window. Feeding 1 then 2 (gap at 0) must keep the same
    # gap_since.
    api = _build_api()
    cmd = CommandId("cmd-gap-timer")
    send, _recv = channel[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
    ]()
    api._text_generation_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]

    def tok(i: int) -> TokenChunk:
        return TokenChunk(
            model=ModelId("mlx-community/test"),
            text=f"t{i}",
            token_id=i,
            usage=None,
            finish_reason=None,
        )

    await api._reorder_and_dispatch(  # pyright: ignore[reportPrivateUsage]
        _data_frame(cmd, 1, tok(1))
    )
    first_gap_since = api._chunk_reorder[cmd].gap_since  # pyright: ignore[reportPrivateUsage]
    assert first_gap_since is not None
    assert api._chunk_reorder[cmd].gap_at == 0  # pyright: ignore[reportPrivateUsage]

    await api._reorder_and_dispatch(  # pyright: ignore[reportPrivateUsage]
        _data_frame(cmd, 2, tok(2))
    )
    # same head-of-line gap (still waiting on seq 0): timer preserved, not reset
    assert api._chunk_reorder[cmd].gap_since == first_gap_since  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_reorder_state_dropped_when_no_queue() -> None:
    # A chunk for a command with no live stream queue (finalized / client gone)
    # must be dropped without creating a lingering reorder buffer.
    api = _build_api()
    cmd = CommandId("cmd-gone")
    chunk = TokenChunk(
        model=ModelId("mlx-community/test"),
        text="late",
        token_id=5,
        usage=None,
        finish_reason="stop",
    )
    await api._reorder_and_dispatch(  # pyright: ignore[reportPrivateUsage]
        _data_frame(cmd, 3, chunk)
    )
    assert cmd not in api._chunk_reorder  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_dispatch_routes_token_chunk_to_text_queue() -> None:
    api = _build_api()
    cmd = CommandId("cmd-text")
    send, recv = channel[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
    ]()
    api._text_generation_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]

    chunk = TokenChunk(
        model=ModelId("mlx-community/test"),
        text="hi",
        token_id=1,
        usage=None,
        finish_reason=None,
    )
    await api._dispatch_generation_chunk(cmd, chunk)  # pyright: ignore[reportPrivateUsage]
    with recv as stream:
        assert stream.receive_nowait() is chunk


@pytest.mark.asyncio
async def test_dispatch_drops_speech_chunk_for_text_queue() -> None:
    api = _build_api()
    cmd = CommandId("cmd-text-speech")
    send, recv = channel[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
    ]()
    api._text_generation_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]

    chunk = AudioChunk(
        model=ModelId("mlx-community/kokoro-test"),
        data="UklGRg==",
        chunk_index=0,
        total_chunks=1,
        format=AudioResponseFormat.Wav,
        finish_reason="stop",
    )
    await api._dispatch_generation_chunk(cmd, chunk)  # pyright: ignore[reportPrivateUsage]
    with recv as stream, pytest.raises(anyio.WouldBlock):
        stream.receive_nowait()


@pytest.mark.asyncio
async def test_dispatch_routes_audio_chunk_to_speech_queue() -> None:
    api = _build_api()
    cmd = CommandId("cmd-speech")
    send, recv = channel[AudioChunk | ErrorChunk]()
    api._audio_speech_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]

    chunk = AudioChunk(
        model=ModelId("mlx-community/kokoro-test"),
        data="UklGRg==",
        chunk_index=0,
        total_chunks=1,
        format=AudioResponseFormat.Wav,
        finish_reason="stop",
    )
    await api._dispatch_generation_chunk(cmd, chunk)  # pyright: ignore[reportPrivateUsage]
    with recv as stream:
        assert stream.receive_nowait() is chunk


@pytest.mark.asyncio
async def test_dispatch_routes_transcription_chunk_to_transcription_queue() -> None:
    api = _build_api()
    cmd = CommandId("cmd-transcription")
    send, recv = channel[TranscriptionChunk | ErrorChunk]()
    api._audio_transcription_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]

    chunk = TranscriptionChunk(
        model=ModelId("mlx-community/whisper-test"),
        text="hello",
        language="en",
        finish_reason="stop",
    )
    await api._dispatch_generation_chunk(cmd, chunk)  # pyright: ignore[reportPrivateUsage]
    with recv as stream:
        assert stream.receive_nowait() is chunk


@pytest.mark.asyncio
async def test_dispatch_routes_embedding_chunk_to_embedding_queue() -> None:
    api = _build_api()
    cmd = CommandId("cmd-embed")
    send, recv = channel[EmbeddingChunk | ErrorChunk]()
    api._embedding_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]

    chunk = EmbeddingChunk(
        model=ModelId("mlx-community/embed"),
        embeddings=[[0.1, 0.2]],
        token_count=2,
    )
    await api._dispatch_generation_chunk(cmd, chunk)  # pyright: ignore[reportPrivateUsage]
    with recv as stream:
        assert stream.receive_nowait() is chunk


@pytest.mark.asyncio
async def test_dispatch_routes_error_chunk_to_image_queue() -> None:
    # Regression (#297 review): a runner failure on an ImageGeneration command
    # emits an ErrorChunk; it must reach the image client, not crash the data
    # loop on a too-narrow `assert isinstance(chunk, ImageChunk)`.
    from skulk.shared.types.chunks import ImageChunk

    api = _build_api()
    cmd = CommandId("cmd-img")
    send, recv = channel[ImageChunk | ErrorChunk]()
    api._image_generation_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]

    err = ErrorChunk(
        model=ModelId("mlx-community/image"),
        error_message="runner shutdown before completing command",
    )
    await api._dispatch_generation_chunk(cmd, err)  # pyright: ignore[reportPrivateUsage]
    with recv as stream:
        got = stream.receive_nowait()
        assert isinstance(got, ErrorChunk)
        assert got is err


@pytest.mark.asyncio
async def test_dispatch_survives_closed_sender() -> None:
    # Regression (#297 review): cancel_command() closes the queue's sender; a
    # late chunk then raises ClosedResourceError on send. The dispatcher must
    # swallow it (and drop the queue), not let it crash the whole data loop.
    api = _build_api()
    cmd = CommandId("cmd-closed")
    send, _recv = channel[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
    ]()
    api._text_generation_queues[cmd] = send  # pyright: ignore[reportPrivateUsage]
    send.close()  # simulate cancel_command() closing the sender

    chunk = TokenChunk(
        model=ModelId("mlx-community/test"),
        text="late",
        token_id=9,
        usage=None,
        finish_reason=None,
    )
    # must not raise
    await api._dispatch_generation_chunk(cmd, chunk)  # pyright: ignore[reportPrivateUsage]
    assert cmd not in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_dispatch_for_unknown_command_is_a_noop() -> None:
    # A chunk for a command with no registered queue (client already gone) must
    # not raise — the data loop has to survive late/orphan chunks.
    api = _build_api()
    chunk = TokenChunk(
        model=ModelId("mlx-community/test"),
        text="x",
        token_id=2,
        usage=None,
        finish_reason="stop",
    )
    await api._dispatch_generation_chunk(  # pyright: ignore[reportPrivateUsage]
        CommandId("nobody-home"), chunk
    )


@pytest.mark.asyncio
async def test_token_stream_stall_after_first_chunk_cancels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #279 Phase 2b backstop: once streaming has started, a dropped chunk would
    # leave _token_chunk_stream blocked on receive() forever. Feed one chunk
    # (started=True), then go silent: the idle timeout must fire, send a
    # TaskCancelled (not TaskFinished — don't orphan the runner), yield a terminal
    # ErrorChunk, and end the stream. fail_after is the anti-hang guard.
    monkeypatch.setattr(api_main, "_STREAM_IDLE_TIMEOUT_SECONDS", 0.15)
    api = _build_api()
    finish_send = AsyncMock()
    cancel_send = AsyncMock()
    api._send = finish_send  # pyright: ignore[reportPrivateUsage]  # finalize: suppress real channel send
    api.command_sender = cancel_send  # stall path sends TaskCancelled here
    cmd = CommandId("cmd-stall")

    chunks: list[object] = []
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:

            async def consume() -> None:
                async for ch in api._token_chunk_stream(cmd):  # pyright: ignore[reportPrivateUsage]
                    chunks.append(ch)

            tg.start_soon(consume)
            # wait for the generator to register its per-command queue, then feed
            # exactly one chunk so the stream is "started", then go silent.
            while cmd not in api._text_generation_queues:  # pyright: ignore[reportPrivateUsage]
                await anyio.sleep(0.005)
            await api._text_generation_queues[cmd].send(  # pyright: ignore[reportPrivateUsage]
                TokenChunk(
                    model=ModelId("mlx-community/test"),
                    text="hi",
                    token_id=1,
                    usage=None,
                    finish_reason=None,
                )
            )

    # first the delivered token, then the synthetic terminal error
    assert len(chunks) == 2
    assert isinstance(chunks[0], TokenChunk)
    assert isinstance(chunks[1], ErrorChunk)
    assert "stall" in chunks[1].error_message.lower()
    # stall is treated as a cancellation (TaskCancelled sent on the command
    # channel), NOT a clean finish (finalize's TaskFinished suppressed because the
    # command was marked cancelled) — so the runner isn't orphaned.
    cancel_send.send.assert_awaited()  # pyright: ignore[reportAny]
    finish_send.assert_not_awaited()
    assert cmd not in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_token_stream_stall_with_terminal_task_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #298 review: a mid-stream stall whose task has ALREADY reached a terminal
    # status is a dropped FINAL data-plane chunk, not a stuck runner. Sending
    # TaskCancelled there is a no-op on a completed runner and leaks the master's
    # task/command mapping forever — so the stall must clean up via the normal
    # TaskFinished path (no TaskCancelled, command not marked cancelled).
    from skulk.shared.types.state import State
    from skulk.shared.types.tasks import (
        TaskId,
        TaskStatus,
        TextGeneration,
    )
    from skulk.shared.types.text_generation import (
        InputMessage,
        TextGenerationTaskParams,
    )
    from skulk.shared.types.worker.instances import InstanceId

    monkeypatch.setattr(api_main, "_STREAM_IDLE_TIMEOUT_SECONDS", 0.15)
    api = _build_api()
    finish_send = AsyncMock()
    cancel_send = AsyncMock()
    api._send = finish_send  # pyright: ignore[reportPrivateUsage]
    api.command_sender = cancel_send
    cmd = CommandId("cmd-stall-done")

    # The master already considers this command's task Complete.
    task = TextGeneration(
        task_id=TaskId(),
        task_status=TaskStatus.Complete,
        instance_id=InstanceId(),
        command_id=cmd,
        task_params=TextGenerationTaskParams(
            model=ModelId("mlx-community/test"),
            input=[InputMessage(role="user", content="hi")],
        ),
    )
    api.state = State(tasks={task.task_id: task})

    chunks: list[object] = []
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:

            async def consume() -> None:
                async for ch in api._token_chunk_stream(cmd):  # pyright: ignore[reportPrivateUsage]
                    chunks.append(ch)

            tg.start_soon(consume)
            while cmd not in api._text_generation_queues:  # pyright: ignore[reportPrivateUsage]
                await anyio.sleep(0.005)
            await api._text_generation_queues[cmd].send(  # pyright: ignore[reportPrivateUsage]
                TokenChunk(
                    model=ModelId("mlx-community/test"),
                    text="hi",
                    token_id=1,
                    usage=None,
                    finish_reason=None,
                )
            )

    assert len(chunks) == 2
    assert isinstance(chunks[0], TokenChunk)
    assert isinstance(chunks[1], ErrorChunk)
    # terminal task -> no TaskCancelled, command NOT marked cancelled, so
    # _finalize_command_stream sends the normal TaskFinished (master cleans up).
    cancel_send.send.assert_not_awaited()  # pyright: ignore[reportAny]
    finish_send.assert_awaited()
    assert cmd not in api._cancelled_command_ids  # pyright: ignore[reportPrivateUsage]
    assert cmd not in api._text_generation_queues  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_token_stream_does_not_timeout_before_first_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A request queued behind a long decode (no chunk yet) must NOT be timed out
    # (Codex P1): with a tiny idle timeout and no chunk delivered, the stream
    # stays open until we close the producer (EndOfStream) — no spurious error.
    monkeypatch.setattr(api_main, "_STREAM_IDLE_TIMEOUT_SECONDS", 0.05)
    api = _build_api()
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    cmd = CommandId("cmd-queued")

    chunks: list[object] = []
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:

            async def consume() -> None:
                async for ch in api._token_chunk_stream(cmd):  # pyright: ignore[reportPrivateUsage]
                    chunks.append(ch)

            tg.start_soon(consume)
            while cmd not in api._text_generation_queues:  # pyright: ignore[reportPrivateUsage]
                await anyio.sleep(0.005)
            # stay silent well past the idle timeout; must NOT yield an error
            await anyio.sleep(0.3)
            assert chunks == []  # no spurious stall before the first chunk
            # close the producer to end the stream cleanly
            api._text_generation_queues[cmd].close()  # pyright: ignore[reportPrivateUsage]

    assert chunks == []  # EndOfStream -> clean return, no error chunk
    assert cmd not in api._cancelled_command_ids  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_prefill_progress_does_not_arm_idle_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #298 review: a PrefillProgressChunk is not real output. Receiving one must
    # NOT arm the idle timer (prefill of a huge prompt can outlast the inter-token
    # bound between progress updates). Send one progress chunk, then stay silent
    # well past a tiny idle timeout: no stall error must be emitted.
    monkeypatch.setattr(api_main, "_STREAM_IDLE_TIMEOUT_SECONDS", 0.05)
    api = _build_api()
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    cmd = CommandId("cmd-prefill")

    chunks: list[object] = []
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:

            async def consume() -> None:
                async for ch in api._token_chunk_stream(cmd):  # pyright: ignore[reportPrivateUsage]
                    chunks.append(ch)

            tg.start_soon(consume)
            while cmd not in api._text_generation_queues:  # pyright: ignore[reportPrivateUsage]
                await anyio.sleep(0.005)
            await api._text_generation_queues[cmd].send(  # pyright: ignore[reportPrivateUsage]
                PrefillProgressChunk(
                    model=ModelId("mlx-community/test"),
                    processed_tokens=10,
                    total_tokens=1000,
                )
            )
            await anyio.sleep(0.3)  # >> idle timeout, but timer must stay disarmed
            assert all(isinstance(c, PrefillProgressChunk) for c in chunks)
            api._text_generation_queues[cmd].close()  # pyright: ignore[reportPrivateUsage]

    # only the progress chunk(s); no synthetic stall ErrorChunk
    assert all(isinstance(c, PrefillProgressChunk) for c in chunks)
    assert cmd not in api._cancelled_command_ids  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_reorder_buffer_disabled_dispatches_in_arrival_order() -> None:
    # #279 Phase 3: with the reorder buffer disabled, _apply_data dispatches
    # chunks as they arrive without buffering/holding for a gap, relying on the
    # transport's per-command order (Zenoh FIFO). In-order input passes straight
    # through. Also confirms a chunk for a command with no live queue is dropped
    # without error.
    from skulk.shared.types.chunks import DataChunk

    data_send, data_recv = channel[DataChunk]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    api = API(
        NodeId("api-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
        data_receiver=data_recv,
    )
    assert api._reorder_buffer_enabled is True  # pyright: ignore[reportPrivateUsage]  # default
    api._reorder_buffer_enabled = False  # pyright: ignore[reportPrivateUsage]

    cmd = CommandId("cmd-arrival")
    owned = CommandId("cmd-no-queue")
    qsend, qrecv = channel[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
    ]()
    api._text_generation_queues[cmd] = qsend  # pyright: ignore[reportPrivateUsage]

    def tok(i: int) -> TokenChunk:
        return TokenChunk(
            model=ModelId("mlx-community/test"),
            text=f"t{i}",
            token_id=i,
            usage=None,
            finish_reason=None,
        )

    for seq in (0, 1, 2):
        await data_send.send(
            DataChunk(command_id=cmd, chunk=tok(seq), sequence=seq, owner_node=None)
        )
    # a chunk for a command with no live queue must be dropped, not crash
    await data_send.send(
        DataChunk(command_id=owned, chunk=tok(7), sequence=0, owner_node=None)
    )
    data_send.close()

    await api._apply_data()  # pyright: ignore[reportPrivateUsage]  # drains until channel closed

    got: list[int] = []
    with qrecv as stream:
        for _ in range(3):
            c = stream.receive_nowait()
            assert isinstance(c, TokenChunk)
            got.append(c.token_id)
    assert got == [0, 1, 2]  # passed straight through, no buffering


@pytest.mark.asyncio
async def test_reorder_buffer_disabled_fails_on_sequence_gap() -> None:
    """Zenoh arrival mode must not return output after a dropped frame."""

    from skulk.shared.types.chunks import DataChunk

    data_send, data_recv = channel[DataChunk]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    api = API(
        NodeId("api-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
        data_receiver=data_recv,
        data_plane_zenoh=True,
    )
    cmd = CommandId("cmd-zenoh-gap")
    qsend, qrecv = channel[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
    ]()
    api._text_generation_queues[cmd] = qsend  # pyright: ignore[reportPrivateUsage]

    def tok(sequence: int) -> TokenChunk:
        return TokenChunk(
            model=ModelId("mlx-community/test"),
            text=f"t{sequence}",
            token_id=sequence,
            usage=None,
        )

    await data_send.send(DataChunk(command_id=cmd, sequence=0, chunk=tok(0)))
    await data_send.send(DataChunk(command_id=cmd, sequence=2, chunk=tok(2)))
    data_send.close()
    await api._apply_data()  # pyright: ignore[reportPrivateUsage]

    with qrecv as stream:
        first = stream.receive_nowait()
        terminal = stream.receive_nowait()
    assert isinstance(first, TokenChunk)
    assert first.token_id == 0
    assert isinstance(terminal, ErrorChunk)
    assert "expected 1, received 2" in terminal.error_message


@pytest.mark.asyncio
async def test_reorder_buffer_disabled_dedupes_loopback_duplicates() -> None:
    # #311 review: with the buffer off (Zenoh), a same-node serve delivers each
    # chunk twice (local TopicRouter publish + Zenoh self-loopback). The O(1)
    # dedupe cursor must drop the duplicate sequence so tokens are not doubled.
    # Feed each sequence twice (0,0,1,1,2,2) -> expect 0,1,2 dispatched once.
    from skulk.shared.types.chunks import DataChunk

    data_send, data_recv = channel[DataChunk]()
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    api = API(
        NodeId("api-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
        data_receiver=data_recv,
        data_plane_zenoh=True,  # buffer auto-off
    )
    assert api._reorder_buffer_enabled is False  # pyright: ignore[reportPrivateUsage]

    cmd = CommandId("cmd-dup-loopback")
    qsend, qrecv = channel[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
    ]()
    api._text_generation_queues[cmd] = qsend  # pyright: ignore[reportPrivateUsage]

    def tok(i: int) -> TokenChunk:
        return TokenChunk(
            model=ModelId("mlx-community/test"),
            text=f"t{i}",
            token_id=i,
            usage=None,
            finish_reason=None,
        )

    for seq in (0, 0, 1, 1, 2, 2):
        await data_send.send(
            DataChunk(command_id=cmd, chunk=tok(seq), sequence=seq, owner_node=None)
        )
    data_send.close()

    await api._apply_data()  # pyright: ignore[reportPrivateUsage]

    got: list[int] = []
    with qrecv as stream:
        try:
            while True:
                c = stream.receive_nowait()
                assert isinstance(c, TokenChunk)
                got.append(c.token_id)
        except anyio.WouldBlock:
            pass
    assert got == [0, 1, 2]  # each delivered once, duplicates dropped
