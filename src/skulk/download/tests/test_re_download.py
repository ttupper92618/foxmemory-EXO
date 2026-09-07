# pyright: reportPrivateUsage=false
"""Tests that re-downloading a previously deleted model completes successfully."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable
from datetime import timedelta
from pathlib import Path
from typing import Callable
from unittest.mock import AsyncMock, patch

import anyio
import pytest

from skulk.download import coordinator as coordinator_module
from skulk.download.coordinator import DownloadCoordinator
from skulk.download.download_utils import RepoDownloadProgress
from skulk.download.impl_shard_downloader import (
    ResumableShardDownloader,
    SingletonShardDownloader,
)
from skulk.download.shard_downloader import ShardDownloader
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    ForwarderDownloadCommand,
    StartDownload,
)
from skulk.shared.types.common import NodeId, SystemId
from skulk.shared.types.events import Event, NodeDownloadProgress
from skulk.shared.types.memory import Memory
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.shared.types.worker.downloads import (
    DownloadAttemptId,
    DownloadCompleted,
    DownloadFailed,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from skulk.store.installed_cards import (
    build_installed_card_record,
    require_registry_installed_artifact,
    write_installed_card,
)
from skulk.utils.channels import Receiver, Sender, channel

NODE_ID = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
MODEL_ID = ModelId("test-org/test-model")


def _make_shard(
    model_id: ModelId = MODEL_ID,
    source_revision: str | None = None,
    *,
    gguf_file: str | None = None,
    registry_card_id: str | None = None,
) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_mb(100),
            n_layers=28,
            hidden_size=1024,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            source_revision=source_revision,
            gguf_file=gguf_file,
            registry_card_id=registry_card_id,
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=28,
        n_layers=28,
    )


class FakeShardDownloader(ShardDownloader):
    """Fake downloader that simulates a successful download by firing the
    progress callback with status='complete' when ensure_shard is called."""

    def __init__(self) -> None:
        self._progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self._progress_callbacks.append(callback)

    async def ensure_shard(
        self,
        shard: ShardMetadata,
        config_only: bool = False,  # noqa: ARG002
    ) -> Path:
        # Simulate a completed download by firing the progress callback
        progress = RepoDownloadProgress(
            repo_id=str(shard.model_card.model_id),
            repo_revision="main",
            shard=shard,
            completed_files=1,
            total_files=1,
            downloaded=Memory.from_mb(100),
            downloaded_this_session=Memory.from_mb(100),
            total=Memory.from_mb(100),
            overall_speed=0,
            overall_eta=timedelta(seconds=0),
            status="complete",
        )
        for cb in self._progress_callbacks:
            await cb(shard, progress)
        return Path("/fake/models") / shard.model_card.model_id.normalize()

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        if False:  # noqa: SIM108  # empty async generator
            yield (
                Path(),
                RepoDownloadProgress(  # pyright: ignore[reportUnreachable]
                    repo_id="",
                    repo_revision="",
                    shard=_make_shard(),
                    completed_files=0,
                    total_files=0,
                    downloaded=Memory.from_bytes(0),
                    downloaded_this_session=Memory.from_bytes(0),
                    total=Memory.from_bytes(0),
                    overall_speed=0,
                    overall_eta=timedelta(seconds=0),
                    status="not_started",
                ),
            )

    async def get_shard_download_status_for_shard(
        self,
        shard: ShardMetadata,
    ) -> RepoDownloadProgress:
        return RepoDownloadProgress(
            repo_id=str(shard.model_card.model_id),
            repo_revision="main",
            shard=shard,
            completed_files=0,
            total_files=1,
            downloaded=Memory.from_bytes(0),
            downloaded_this_session=Memory.from_bytes(0),
            total=Memory.from_mb(100),
            overall_speed=0,
            overall_eta=timedelta(seconds=0),
            status="not_started",
        )


class SlowCancellationShardDownloader(FakeShardDownloader):
    """Downloader whose first cancelled attempt holds cleanup until released."""

    def __init__(self) -> None:
        super().__init__()
        self.ensure_calls = 0
        self.ensure_revisions: list[str | None] = []
        self.first_started = anyio.Event()
        self.release_cleanup = anyio.Event()
        self.restarted = anyio.Event()

    async def ensure_shard(
        self,
        shard: ShardMetadata,
        config_only: bool = False,  # noqa: ARG002
    ) -> Path:
        self.ensure_calls += 1
        self.ensure_revisions.append(shard.model_card.source_revision)
        if self.ensure_calls == 1:
            self.first_started.set()
            try:
                await anyio.sleep_forever()
            finally:
                with anyio.CancelScope(shield=True):
                    await self.release_cleanup.wait()
        self.restarted.set()
        return await super().ensure_shard(shard)


@pytest.mark.parametrize("status", ["not_started", "in_progress"])
async def test_incomplete_transfer_cannot_return_an_installed_path(
    tmp_path: Path, status: str
) -> None:
    """A non-raising fetch failure must not authorize a completed installation."""

    shard = _make_shard()
    progress = await FakeShardDownloader().get_shard_download_status_for_shard(shard)
    progress = progress.model_copy(update={"status": status})
    downloader = ResumableShardDownloader()
    with patch.object(
        downloader,
        "_download_with_capacity",
        new=AsyncMock(return_value=(tmp_path, progress)),
    ), pytest.raises(RuntimeError, match="did not finish downloading"):
        await downloader.ensure_shard(shard)
    assert not (tmp_path / ".skulk" / "installed-card.json").exists()


@pytest.mark.parametrize(
    "outcome", ["complete", "failed", "cancelled", "cancelled_return"]
)
@pytest.mark.parametrize("file_result", [False, True])
async def test_transfer_completion_waits_for_installed_identity(
    tmp_path: Path, outcome: str, file_result: bool
) -> None:
    """Completed bytes cannot start a runner before identity finalization."""

    transferred = anyio.Event()
    finalize = anyio.Event()
    directory = tmp_path / "staged-model"
    directory.mkdir()
    shard = _make_shard(
        source_revision="a" * 40,
        gguf_file="model.gguf",
        registry_card_id=f"card_{'a' * 52}",
    )

    class FinalizingDownloader(FakeShardDownloader):
        """Hold installation after the real transfer-complete callback."""

        async def ensure_shard(
            self, shard: ShardMetadata, config_only: bool = False
        ) -> Path:
            """Publish signed identity only when the test releases finalization."""
            await super().ensure_shard(shard, config_only)
            transferred.set()
            try:
                await finalize.wait()
            except anyio.get_cancelled_exc_class():
                # Some installers shield finalization and return after their
                # caller cancels. Their result cannot resurrect cancelled work.
                if outcome != "cancelled_return":
                    raise
            if outcome == "failed":
                raise OSError("installed identity write failed")
            (directory / "model.gguf").write_bytes(b"weights")
            (directory / ".skulk-source-revision").write_text("a" * 40 + "\n")
            write_installed_card(
                directory, build_installed_card_record(directory, shard.model_card)
            )
            return directory / "model.gguf" if file_result else directory

    _, commands = channel[ForwarderDownloadCommand]()
    events, received_events = channel[Event]()
    telemetry, _ = channel[NodeTelemetry]()
    downloader = FinalizingDownloader()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=downloader,
        download_command_receiver=commands,
        event_sender=events,
        telemetry_sender=telemetry,
    )
    initial = await downloader.get_shard_download_status_for_shard(shard)
    with anyio.fail_after(5):
        async with coordinator._tg:
            coordinator._start_download_task(shard, initial)
            await transferred.wait()
            assert not isinstance(coordinator.download_status[MODEL_ID], DownloadCompleted)
            with pytest.raises(anyio.WouldBlock):
                received_events.receive_nowait()
            if outcome.startswith("cancelled"):
                coordinator.active_downloads[MODEL_ID].cancel()
            else:
                finalize.set()
        if outcome.startswith("cancelled"):
            with pytest.raises(anyio.WouldBlock):
                received_events.receive_nowait()
        else:
            event = received_events.receive_nowait()
            assert isinstance(event, NodeDownloadProgress)
            if outcome == "failed":
                assert isinstance(event.download_progress, DownloadFailed)
            else:
                completed = event.download_progress
                assert isinstance(completed, DownloadCompleted)
                assert completed.model_directory == str(directory)
                require_registry_installed_artifact(directory, shard.model_card)
            with pytest.raises(anyio.WouldBlock):
                received_events.receive_nowait()


async def test_nested_companion_progress_cannot_strand_an_unowned_download() -> None:
    """Incidental transfers cannot block a later explicit companion request."""

    parent = _make_shard()
    companion = _make_shard(ModelId("test-org/companion"))
    requested: list[ModelId] = []

    class CompanionDownloader(FakeShardDownloader):
        """Emit nested companion callbacks while installing the parent."""

        async def ensure_shard(
            self, shard: ShardMetadata, config_only: bool = False
        ) -> Path:
            """Report companion transfer progress without a separate task."""
            requested.append(shard.model_card.model_id)
            if shard.model_card.model_id == MODEL_ID:
                progress = await self.get_shard_download_status_for_shard(companion)
                progress = progress.model_copy(update={"status": "in_progress"})
                for callback in self._progress_callbacks:
                    await callback(companion, progress)
                await super().ensure_shard(companion, config_only)
            return await super().ensure_shard(shard, config_only)

    _, commands = channel[ForwarderDownloadCommand]()
    events, received_events = channel[Event]()
    telemetry, _ = channel[NodeTelemetry]()
    downloader = CompanionDownloader()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=downloader,
        download_command_receiver=commands,
        event_sender=events,
        telemetry_sender=telemetry,
    )
    with patch.object(
        coordinator, "_resolve_or_refresh_complete_artifact", new=AsyncMock(return_value=None)
    ), anyio.fail_after(5):
        async with coordinator._tg:
            await coordinator._start_download(parent)
            assert await _wait_for_download_completed(received_events, MODEL_ID)
            assert companion.model_card.model_id not in coordinator.download_status
            await coordinator._start_download(companion)
            assert await _wait_for_download_completed(
                received_events, companion.model_card.model_id
            )
    assert requested == [MODEL_ID, companion.model_card.model_id]


async def test_purge_all_unregisters_only_companion_artifact_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Purging a companion directory must preserve its owner's installed truth."""

    _command_send, command_receive = channel[ForwarderDownloadCommand]()
    event_send, _event_receive = channel[Event]()
    telemetry_send, _telemetry_receive = channel[NodeTelemetry]()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=FakeShardDownloader(),
        download_command_receiver=command_receive,
        event_sender=event_send,
        telemetry_sender=telemetry_send,
        offline=True,
    )
    owner_card = _make_shard(ModelId("org/base")).model_card
    companion_id = ModelId("org/base-mtp")
    companion_directory = tmp_path / companion_id.normalize()
    companion_directory.mkdir()
    (companion_directory / "weights.safetensors").write_bytes(b"companion")
    write_installed_card(
        companion_directory,
        build_installed_card_record(
            companion_directory,
            owner_card,
            artifact_model_id=companion_id,
            artifact_repository=str(companion_id),
            artifact_role="mtp_sidecar",
            owner_model_id=owner_card.model_id,
        ),
    )
    unregistered: list[ModelId] = []
    monkeypatch.setattr(
        coordinator_module,
        "unregister_installed_card_record",
        unregistered.append,
    )

    purged = await coordinator._purge_dir(tmp_path, "staging")

    assert purged == 1
    assert unregistered == [companion_id]
    assert not companion_directory.exists()


async def test_revision_change_restarts_failed_download_state() -> None:
    """A failed attempt for an old pin must not suppress a corrected pin."""

    _command_send, command_receive = channel[ForwarderDownloadCommand]()
    event_send, _event_receive = channel[Event]()
    telemetry_send, _telemetry_receive = channel[NodeTelemetry]()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=FakeShardDownloader(),
        download_command_receiver=command_receive,
        event_sender=event_send,
        telemetry_sender=telemetry_send,
        offline=True,
    )
    old_shard = _make_shard(source_revision="0" * 40)
    new_shard = _make_shard(source_revision="1" * 40)
    coordinator.download_status[MODEL_ID] = DownloadFailed(
        shard_metadata=old_shard,
        node_id=NODE_ID,
        error_message="old revision failed",
        model_directory="/missing",
    )

    await coordinator._start_download(new_shard)

    status = coordinator.download_status[MODEL_ID]
    assert isinstance(status, DownloadFailed)
    assert status.shard_metadata.model_card.source_revision == "1" * 40


async def test_registry_identity_change_restarts_same_revision_status() -> None:
    """A new signed quant cannot inherit stale same-alias completion state."""
    _command_send, command_receive = channel[ForwarderDownloadCommand]()
    event_send, _event_receive = channel[Event]()
    telemetry_send, _telemetry_receive = channel[NodeTelemetry]()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=FakeShardDownloader(),
        download_command_receiver=command_receive,
        event_sender=event_send,
        telemetry_sender=telemetry_send,
        offline=True,
    )
    old_shard = _make_shard(
        source_revision="a" * 40,
        gguf_file="model-q4.gguf",
        registry_card_id=f"card_{'a' * 52}",
    )
    new_shard = _make_shard(
        source_revision="a" * 40,
        gguf_file="model-q5.gguf",
        registry_card_id=f"card_{'b' * 52}",
    )
    coordinator.download_status[MODEL_ID] = DownloadFailed(
        shard_metadata=old_shard,
        node_id=NODE_ID,
        error_message="old artifact failed",
        model_directory="/missing",
    )

    await coordinator._start_download(new_shard)

    status = coordinator.download_status[MODEL_ID]
    assert isinstance(status, DownloadFailed)
    assert status.shard_metadata.model_card.registry_card_id == f"card_{'b' * 52}"
    assert status.shard_metadata.model_card.gguf_file == "model-q5.gguf"


@pytest.mark.parametrize("offline", [False, True])
async def test_complete_byte_probe_refreshes_signed_card_without_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline: bool,
) -> None:
    """A stale same-artifact sidecar refreshes locally, including offline."""

    import skulk.shared.constants as constants

    _command_send, command_receive = channel[ForwarderDownloadCommand]()
    event_send, _event_receive = channel[Event]()
    telemetry_send, _telemetry_receive = channel[NodeTelemetry]()
    downloader = FakeShardDownloader()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=downloader,
        download_command_receiver=command_receive,
        event_sender=event_send,
        telemetry_sender=telemetry_send,
        offline=offline,
    )
    revision = "a" * 40
    old_shard = _make_shard(
        source_revision=revision,
        gguf_file="model.gguf",
        registry_card_id=f"card_{'a' * 52}",
    )
    requested_shard = _make_shard(
        source_revision=revision,
        gguf_file="model.gguf",
        registry_card_id=f"card_{'b' * 52}",
    )
    model_directory = tmp_path / MODEL_ID.normalize()
    model_directory.mkdir()
    (model_directory / "model.gguf").write_bytes(b"weights")
    (model_directory / ".skulk-source-revision").write_text(f"{revision}\n")
    write_installed_card(
        model_directory,
        build_installed_card_record(model_directory, old_shard.model_card),
    )
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))

    async def byte_complete_status(
        shard: ShardMetadata,
    ) -> RepoDownloadProgress:
        return RepoDownloadProgress(
            repo_id=str(shard.model_card.model_id),
            repo_revision=revision,
            shard=shard,
            completed_files=1,
            total_files=1,
            downloaded=Memory.from_mb(100),
            downloaded_this_session=Memory.from_bytes(0),
            total=Memory.from_mb(100),
            overall_speed=0,
            overall_eta=timedelta(seconds=0),
            status="complete",
        )

    started: list[RepoDownloadProgress] = []
    monkeypatch.setattr(
        downloader,
        "get_shard_download_status_for_shard",
        byte_complete_status,
    )

    def record_download_start(
        _shard: ShardMetadata,
        progress: RepoDownloadProgress,
    ) -> None:
        started.append(progress)

    monkeypatch.setattr(
        coordinator,
        "_start_download_task",
        record_download_start,
    )

    await coordinator._start_download(requested_shard)

    assert started == []
    status = coordinator.download_status[MODEL_ID]
    assert isinstance(status, DownloadCompleted)
    assert status.model_directory == str(model_directory)
    require_registry_installed_artifact(model_directory, requested_shard.model_card)


async def test_startup_complete_probe_refreshes_signed_card_before_terminal_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restart discovery cannot retain a stale signed completion identity."""

    _command_send, command_receive = channel[ForwarderDownloadCommand]()
    event_send, _event_receive = channel[Event]()
    telemetry_send, _telemetry_receive = channel[NodeTelemetry]()
    downloader = FakeShardDownloader()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=downloader,
        download_command_receiver=command_receive,
        event_sender=event_send,
        telemetry_sender=telemetry_send,
        offline=True,
    )
    revision = "a" * 40
    old_shard = _make_shard(
        source_revision=revision,
        gguf_file="model.gguf",
        registry_card_id=f"card_{'a' * 52}",
    )
    requested_shard = _make_shard(
        source_revision=revision,
        gguf_file="model.gguf",
        registry_card_id=f"card_{'b' * 52}",
    )
    model_directory = tmp_path / MODEL_ID.normalize()
    model_directory.mkdir()
    (model_directory / "model.gguf").write_bytes(b"weights")
    (model_directory / ".skulk-source-revision").write_text(f"{revision}\n")
    write_installed_card(
        model_directory,
        build_installed_card_record(model_directory, old_shard.model_card),
    )
    progress = RepoDownloadProgress(
        repo_id=str(MODEL_ID),
        repo_revision=revision,
        shard=requested_shard,
        completed_files=1,
        total_files=1,
        downloaded=Memory.from_mb(100),
        downloaded_this_session=Memory.from_bytes(0),
        total=Memory.from_mb(100),
        overall_speed=0,
        overall_eta=timedelta(seconds=0),
        status="complete",
    )

    async def startup_status() -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        yield model_directory, progress

    async def no_catalog_cards() -> list[ModelCard]:
        return []

    monkeypatch.setattr(downloader, "get_shard_download_status", startup_status)
    monkeypatch.setattr(coordinator_module, "get_model_cards", no_catalog_cards)
    monkeypatch.setattr(coordinator_module, "SKULK_MODELS_DIR", tmp_path)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(coordinator._emit_existing_download_progress)
        with anyio.fail_after(1):
            while not isinstance(
                coordinator.download_status.get(MODEL_ID),
                DownloadCompleted,
            ):
                await anyio.sleep(0)
        task_group.cancel_scope.cancel()

    status = coordinator.download_status[MODEL_ID]
    assert isinstance(status, DownloadCompleted)
    assert status.model_directory == str(model_directory)
    require_registry_installed_artifact(model_directory, requested_shard.model_card)


async def test_revision_change_replaces_active_download_after_cleanup() -> None:
    """A corrected pin must cancel and replace an active old-pin download."""

    command_send, command_receive = channel[ForwarderDownloadCommand]()
    event_send, event_receive = channel[Event]()
    telemetry_send, _telemetry_receive = channel[NodeTelemetry]()
    downloader = SlowCancellationShardDownloader()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=downloader,
        download_command_receiver=command_receive,
        event_sender=event_send,
        telemetry_sender=telemetry_send,
    )
    old_revision = "0" * 40
    new_revision = "1" * 40
    old_shard = _make_shard(source_revision=old_revision)
    new_shard = _make_shard(source_revision=new_revision)
    origin = SystemId("test")
    coordinator_task = asyncio.create_task(coordinator.run())

    try:
        await command_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=StartDownload(
                    target_node_id=NODE_ID, shard_metadata=old_shard
                ),
            )
        )
        with anyio.fail_after(2):
            await downloader.first_started.wait()

        await command_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=StartDownload(
                    target_node_id=NODE_ID, shard_metadata=new_shard
                ),
            )
        )
        await anyio.sleep(0)
        assert downloader.ensure_calls == 1

        downloader.release_cleanup.set()
        with anyio.fail_after(2):
            await downloader.restarted.wait()
        completed = await _wait_for_download_completed(event_receive, MODEL_ID)

        assert completed is not None
        assert completed.shard_metadata.model_card.source_revision == new_revision
        assert downloader.ensure_revisions == [old_revision, new_revision]
    finally:
        downloader.release_cleanup.set()
        await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task


async def test_re_download_after_delete_completes() -> None:
    """A model that was downloaded, deleted, and then re-downloaded should
    reach DownloadCompleted status. This is an end-to-end test through
    the DownloadCoordinator."""
    cmd_send: Sender[ForwarderDownloadCommand]
    cmd_send, cmd_recv = channel[ForwarderDownloadCommand]()
    event_send, event_recv = channel[Event]()
    telemetry_send, _ = channel[NodeTelemetry]()

    fake_downloader = FakeShardDownloader()
    wrapped_downloader = SingletonShardDownloader(fake_downloader)
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=wrapped_downloader,
        download_command_receiver=cmd_recv,
        event_sender=event_send,
        telemetry_sender=telemetry_send,
    )

    shard = _make_shard()
    origin = SystemId("test")

    with patch("skulk.download.coordinator.delete_model", new_callable=AsyncMock):
        # Run the coordinator in the background
        coordinator_task = asyncio.create_task(coordinator.run())

        try:
            # 1. Start first download
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=origin,
                    command=StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
                )
            )

            # Wait for DownloadCompleted
            first_completed = await _wait_for_download_completed(event_recv, MODEL_ID)
            assert first_completed is not None, "First download should complete"

            # 2. Delete the model
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=origin,
                    command=DeleteDownload(target_node_id=NODE_ID, model_id=MODEL_ID),
                )
            )
            # Give the coordinator time to process the delete
            await asyncio.sleep(0.05)

            # 3. Re-download the same model
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=origin,
                    command=StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
                )
            )

            # Wait for second DownloadCompleted — this is the bug: it never arrives
            second_completed = await _wait_for_download_completed(event_recv, MODEL_ID)
            assert second_completed is not None, (
                "Re-download after deletion should complete"
            )
        finally:
            await coordinator.shutdown()
            coordinator_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await coordinator_task


async def test_start_during_cancellation_runs_after_cleanup() -> None:
    """A replacement start must survive the prior task's cancellation window."""

    cmd_send, cmd_recv = channel[ForwarderDownloadCommand]()
    event_send, event_recv = channel[Event]()
    telemetry_send, _ = channel[NodeTelemetry]()
    downloader = SlowCancellationShardDownloader()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=downloader,
        download_command_receiver=cmd_recv,
        event_sender=event_send,
        telemetry_sender=telemetry_send,
    )
    shard = _make_shard()
    origin = SystemId("test")
    coordinator_task = asyncio.create_task(coordinator.run())

    try:
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
            )
        )
        with anyio.fail_after(2):
            await downloader.first_started.wait()

        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=CancelDownload(target_node_id=NODE_ID, model_id=MODEL_ID),
            )
        )
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
            )
        )
        await anyio.sleep(0)
        assert downloader.ensure_calls == 1

        downloader.release_cleanup.set()
        with anyio.fail_after(2):
            await downloader.restarted.wait()
        completed = await _wait_for_download_completed(event_recv, MODEL_ID)
        assert completed is not None
        assert downloader.ensure_calls == 2
    finally:
        await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task


async def test_cancel_download_rejects_a_replaced_attempt() -> None:
    """An attempt-bound cancel cannot stop a newer retry of the same model."""

    cmd_send, cmd_recv = channel[ForwarderDownloadCommand]()
    event_send, _ = channel[Event]()
    telemetry_send, _ = channel[NodeTelemetry]()
    downloader = SlowCancellationShardDownloader()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=downloader,
        download_command_receiver=cmd_recv,
        event_sender=event_send,
        telemetry_sender=telemetry_send,
    )
    shard = _make_shard()
    origin = SystemId("test")
    coordinator_task = asyncio.create_task(coordinator.run())

    try:
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
            )
        )
        with anyio.fail_after(2):
            await downloader.first_started.wait()
        current_status = coordinator.download_status[MODEL_ID]
        assert current_status.attempt_id is not None

        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=CancelDownload(
                    target_node_id=NODE_ID,
                    model_id=MODEL_ID,
                    attempt_id=DownloadAttemptId("replaced-attempt"),
                ),
            )
        )
        await anyio.sleep(0.05)
        assert coordinator.active_downloads[MODEL_ID].cancel_called is False

        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=CancelDownload(
                    target_node_id=NODE_ID,
                    model_id=MODEL_ID,
                    attempt_id=current_status.attempt_id,
                ),
            )
        )
        with anyio.fail_after(2):
            while not coordinator.active_downloads[MODEL_ID].cancel_called:
                await anyio.sleep(0)
    finally:
        downloader.release_cleanup.set()
        await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task


async def test_delete_clears_restart_queued_during_cancellation() -> None:
    """An explicit delete must invalidate a replacement start awaiting cleanup."""

    cmd_send, cmd_recv = channel[ForwarderDownloadCommand]()
    event_send, _ = channel[Event]()
    telemetry_send, _ = channel[NodeTelemetry]()
    downloader = SlowCancellationShardDownloader()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=downloader,
        download_command_receiver=cmd_recv,
        event_sender=event_send,
        telemetry_sender=telemetry_send,
    )
    shard = _make_shard()
    origin = SystemId("test")
    delete_processed = anyio.Event()

    async def record_delete(_model_id: ModelId) -> bool:
        delete_processed.set()
        return True

    coordinator_task = asyncio.create_task(coordinator.run())
    with patch("skulk.download.coordinator.delete_model", side_effect=record_delete):
        try:
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=origin,
                    command=StartDownload(
                        target_node_id=NODE_ID, shard_metadata=shard
                    ),
                )
            )
            with anyio.fail_after(2):
                await downloader.first_started.wait()

            for command in (
                CancelDownload(target_node_id=NODE_ID, model_id=MODEL_ID),
                StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
                DeleteDownload(target_node_id=NODE_ID, model_id=MODEL_ID),
            ):
                await cmd_send.send(
                    ForwarderDownloadCommand(origin=origin, command=command)
                )
            with anyio.fail_after(2):
                await delete_processed.wait()

            downloader.release_cleanup.set()
            await anyio.sleep(0.05)

            assert downloader.ensure_calls == 1
            assert not downloader.restarted.is_set()
        finally:
            downloader.release_cleanup.set()
            await coordinator.shutdown()
            coordinator_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await coordinator_task


async def _wait_for_download_completed(
    event_recv: Receiver[Event], model_id: ModelId, timeout: float = 2.0
) -> DownloadCompleted | None:
    """Drain events until we see a DownloadCompleted for the given model, or timeout."""
    try:
        async with asyncio.timeout(timeout):
            while True:
                event = await event_recv.receive()
                if (
                    isinstance(event, NodeDownloadProgress)
                    and isinstance(event.download_progress, DownloadCompleted)
                    and event.download_progress.shard_metadata.model_card.model_id
                    == model_id
                ):
                    return event.download_progress
    except TimeoutError:
        return None
