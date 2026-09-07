import asyncio
import hashlib
import shutil
from asyncio import create_task
from collections.abc import Awaitable
from pathlib import Path
from typing import AsyncIterator, Callable

from loguru import logger

from skulk.download.download_utils import (
    DownloadCapacityPreflight,
    RepoDownloadProgress,
    companion_download_specs,
    download_shard,
)
from skulk.download.shard_downloader import ShardDownloader
from skulk.shared.constants import SKULK_MODELS_DIR
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    get_model_cards,
    register_installed_card_record,
)
from skulk.shared.types.worker.downloads import FileListEntry
from skulk.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
)
from skulk.store.installed_cards import (
    InstalledArtifactRole,
    build_installed_card_record,
    companion_artifact_role,
    read_installed_card_with_fallback,
    write_installed_card,
)
from skulk.store.staging_eviction import MINIMUM_STAGING_FREE_DISK_BYTES


class DirectDownloadCapacityError(RuntimeError):
    """Raised before a Hugging Face transfer that would consume headroom."""


def _replacement_identity_for_installed_card(
    model_directory: Path,
    model_card: ModelCard,
    *,
    artifact_model_id: str,
    artifact_role: InstalledArtifactRole,
) -> str | None:
    """Return a stable replacement key when retained card truth has changed."""

    try:
        record = read_installed_card_with_fallback(model_directory)
    except (OSError, ValueError):
        record = None
    if record is None:
        return None
    if artifact_role == "base":
        requested_card_id = model_card.registry_card_id
        card_matches = (
            record.model_card.registry_card_id == requested_card_id
            if requested_card_id is not None
            else record.model_card == model_card
        )
        generation_matches = (
            record.artifact_role == "base"
            and record.artifact_model_id == artifact_model_id
            and card_matches
        )
    else:
        owner_matches = (
            record.owner_card_id == model_card.registry_card_id
            if model_card.registry_card_id is not None
            else record.model_card == model_card
        )
        generation_matches = (
            record.artifact_role == artifact_role
            and record.artifact_model_id == artifact_model_id
            and owner_matches
        )
    if generation_matches:
        return None
    payload = (
        f"{artifact_role}\0{artifact_model_id}\0"
        f"{model_card.model_dump_json(exclude={'is_custom'})}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _remaining_direct_download_bytes(
    target_directory: Path,
    file_list: list[FileListEntry],
) -> int:
    """Return physical bytes still needed for one filtered direct download."""

    remaining_bytes = 0
    for file_entry in file_list:
        if file_entry.size is None:
            raise DirectDownloadCapacityError(
                "Cannot verify Hugging Face model-cache disk capacity because "
                "the selected manifest contains a file with unknown size. "
                "Retry after repository metadata is available."
            )
        expected_size = file_entry.size
        target = target_directory / file_entry.path
        partial = target_directory / f"{file_entry.path}.partial"
        target_bytes = target.stat().st_size if target.is_file() else 0
        if target.is_file() and target_bytes == expected_size:
            continue
        partial_bytes = partial.stat().st_size if partial.is_file() else 0
        remaining_bytes += max(
            0,
            expected_size - target_bytes - partial_bytes,
        )
    return remaining_bytes


def _has_local_download_state(model_card: ModelCard) -> bool:
    """Return whether the canonical cache contains model download data.

    The startup progress scan exists to recover downloads interrupted in an
    earlier process. Catalog entries with no local files have nothing to
    recover and must not consume Hugging Face metadata requests merely because
    they are present in the shipped model-card registry.

    Args:
        model_card: Catalog card whose canonical direct-download directory
            should be inspected.

    Returns:
        ``True`` when at least one file exists below the model's canonical
        cache directory, including resumable ``.partial`` files.
    """

    model_directory = SKULK_MODELS_DIR / model_card.model_id.normalize()
    if not model_directory.is_dir():
        return False
    return any(path.is_file() for path in model_directory.rglob("*"))


def skulk_shard_downloader(
    max_parallel_downloads: int = 8, offline: bool = False
) -> ShardDownloader:
    return SingletonShardDownloader(
        ResumableShardDownloader(max_parallel_downloads, offline=offline)
    )


async def build_base_shard(model_id: ModelId) -> ShardMetadata:
    model_card = await ModelCard.load_or_fetch_from_hf(model_id)
    return PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=model_card.n_layers,
        n_layers=model_card.n_layers,
    )


async def build_full_shard(model_id: ModelId) -> PipelineShardMetadata:
    base_shard = await build_base_shard(model_id)
    return PipelineShardMetadata(
        model_card=base_shard.model_card,
        device_rank=base_shard.device_rank,
        world_size=base_shard.world_size,
        start_layer=base_shard.start_layer,
        end_layer=base_shard.n_layers,
        n_layers=base_shard.n_layers,
    )


class SingletonShardDownloader(ShardDownloader):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard_downloader = shard_downloader
        self.active_downloads: dict[ShardMetadata, asyncio.Task[Path]] = {}

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self.shard_downloader.on_progress(callback)

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        if shard not in self.active_downloads:
            self.active_downloads[shard] = asyncio.create_task(
                self.shard_downloader.ensure_shard(shard, config_only)
            )
        try:
            return await self.active_downloads[shard]
        finally:
            if shard in self.active_downloads and self.active_downloads[shard].done():
                del self.active_downloads[shard]

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        async for path, status in self.shard_downloader.get_shard_download_status():
            yield path, status

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        return await self.shard_downloader.get_shard_download_status_for_shard(shard)


class ResumableShardDownloader(ShardDownloader):
    def __init__(self, max_parallel_downloads: int = 8, offline: bool = False):
        self.max_parallel_downloads = max_parallel_downloads
        self.offline = offline
        self._direct_transfer_lock = asyncio.Lock()
        self.on_progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []

    async def _ensure_direct_download_capacity(
        self,
        target_directory: Path,
        file_list: list[FileListEntry],
    ) -> None:
        """Fail before a direct transfer that cannot retain OS headroom."""

        additional_bytes = await asyncio.to_thread(
            _remaining_direct_download_bytes,
            target_directory,
            file_list,
        )
        if additional_bytes == 0:
            # Reusing an exact, complete artifact writes no model bytes. The
            # reserve prevents a transfer from filling the filesystem; it
            # must not make already-downloaded models unavailable after
            # coordinator state is rebuilt.
            return
        free_bytes = await asyncio.to_thread(
            lambda: shutil.disk_usage(target_directory).free
        )
        required_free_bytes = additional_bytes + MINIMUM_STAGING_FREE_DISK_BYTES
        if free_bytes < required_free_bytes:
            raise DirectDownloadCapacityError(
                f"Insufficient Hugging Face model-cache disk capacity: need "
                f"{required_free_bytes / 2**30:.1f} GiB free for the remaining "
                "transfer and operating-system reserve, but only "
                f"{free_bytes / 2**30:.1f} GiB is available. Free disk space "
                "or move the model cache."
            )

    async def _download_with_capacity(
        self,
        shard: ShardMetadata,
        *,
        allow_patterns: list[str] | None = None,
        replacement_identity: str | None = None,
    ) -> tuple[Path, RepoDownloadProgress]:
        """Serialize exact capacity admission with one direct transfer."""

        preflight: DownloadCapacityPreflight = self._ensure_direct_download_capacity
        async with self._direct_transfer_lock:
            if replacement_identity is not None:
                return await download_shard(
                    shard,
                    self.on_progress_wrapper,
                    max_parallel_downloads=self.max_parallel_downloads,
                    allow_patterns=allow_patterns,
                    skip_internet=self.offline,
                    capacity_preflight=preflight,
                    replacement_identity=replacement_identity,
                )
            return await download_shard(
                shard,
                self.on_progress_wrapper,
                max_parallel_downloads=self.max_parallel_downloads,
                allow_patterns=allow_patterns,
                skip_internet=self.offline,
                capacity_preflight=preflight,
            )

    async def on_progress_wrapper(
        self, shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        for callback in self.on_progress_callbacks:
            await callback(shard, progress)

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self.on_progress_callbacks.append(callback)

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        bundle_root = (
            shard.model_card.artifact_bundle.root
            if shard.model_card.artifact_bundle is not None
            else None
        )
        config_path = (
            f"{bundle_root}/config.json" if bundle_root is not None else "config.json"
        )
        allow_patterns = [config_path] if config_only else None

        # Companions download before the base so required sidecar failures
        # surface before spending bandwidth on the base. ensure_shard must
        # finish both artifacts and installed identities before the coordinator
        # advertises completion and permits the planner to dispatch model loads.
        # Criticality differs per companion: split vision weights are
        # load-bearing (their failure fails the base — a vision model
        # without them is broken), while MTP sidecars and assistants are
        # best-effort (the runtime degrades to run-without-speculation;
        # failures log loudly instead).
        if not config_only and not self.offline:
            for companion_shard, allow, required in companion_download_specs(
                shard.model_card
            ):
                try:
                    repository = str(companion_shard.model_card.model_id)
                    role = companion_artifact_role(shard.model_card, repository)
                    replacement_identity = _replacement_identity_for_installed_card(
                        SKULK_MODELS_DIR / companion_shard.model_card.model_id.normalize(),
                        shard.model_card,
                        artifact_model_id=repository,
                        artifact_role=role,
                    )
                    (
                        companion_path,
                        companion_progress,
                    ) = await self._download_with_capacity(
                        companion_shard,
                        allow_patterns=allow,
                        replacement_identity=replacement_identity,
                    )
                    # download_shard converts repo-level fetch failures
                    # (e.g. FileNotFoundError on the file list) into a
                    # not_started result instead of raising — a required
                    # companion must not slip through that hole.
                    if required and companion_progress.status != "complete":
                        raise RuntimeError(
                            f"Required companion repo "
                            f"{companion_shard.model_card.model_id} did not "
                            f"download (status="
                            f"{companion_progress.status!r})"
                        )
                    if companion_progress.status == "complete":
                        companion_directory = (
                            companion_path.parent
                            if companion_path.is_file()
                            else companion_path
                        )
                        record = await asyncio.to_thread(
                            build_installed_card_record,
                            companion_directory,
                            shard.model_card,
                            artifact_role=role,
                            artifact_model_id=repository,
                            owner_model_id=str(shard.model_card.model_id),
                            owner_card_id=shard.model_card.registry_card_id,
                            artifact_repository=repository,
                            artifact_revision=companion_shard.model_card.source_revision,
                        )
                        await asyncio.to_thread(
                            write_installed_card,
                            companion_directory,
                            record,
                        )
                        register_installed_card_record(record)
                except Exception as error:
                    if required:
                        # Split vision weights are load-bearing: a vision
                        # model without them is broken, not degraded.
                        raise
                    logger.warning(
                        f"Companion repo {companion_shard.model_card.model_id} "
                        f"for {shard.model_card.model_id} could not be fetched "
                        f"({error}); speculative decoding that depends on it "
                        "will be unavailable on this node."
                    )

        base_replacement_identity = (
            None
            if config_only
            else _replacement_identity_for_installed_card(
                SKULK_MODELS_DIR / shard.model_card.model_id.normalize(),
                shard.model_card,
                artifact_model_id=str(shard.model_card.model_id),
                artifact_role="base",
            )
        )
        target_dir, base_progress = await self._download_with_capacity(
            shard,
            allow_patterns=allow_patterns,
            replacement_identity=base_replacement_identity,
        )

        if base_progress.status != "complete":
            raise RuntimeError(
                f"Model {shard.model_card.model_id} did not finish downloading "
                f"(status={base_progress.status!r})"
            )

        if not config_only:
            artifact_directory = target_dir.parent if target_dir.is_file() else target_dir
            record = await asyncio.to_thread(
                build_installed_card_record,
                artifact_directory,
                shard.model_card,
            )
            await asyncio.to_thread(write_installed_card, artifact_directory, record)
            register_installed_card_record(record)

        return target_dir

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        async def _status_for_model(
            model_id: ModelId,
        ) -> tuple[Path, RepoDownloadProgress]:
            """Helper coroutine that builds the shard for a model and gets its download status."""
            shard = await build_full_shard(model_id)
            return await download_shard(
                shard,
                self.on_progress_wrapper,
                skip_download=True,
                skip_internet=self.offline,
            )

        semaphore = asyncio.Semaphore(self.max_parallel_downloads)

        async def download_with_semaphore(
            model_card: ModelCard,
        ) -> tuple[Path, RepoDownloadProgress]:
            async with semaphore:
                return await _status_for_model(model_card.model_id)

        model_cards = await get_model_cards()
        local_model_cards = await asyncio.to_thread(
            lambda: [
                model_card
                for model_card in model_cards
                if _has_local_download_state(model_card)
            ]
        )
        tasks = [
            create_task(download_with_semaphore(model_card))
            for model_card in local_model_cards
        ]

        for task in asyncio.as_completed(tasks):
            try:
                yield await task
            except Exception as e:
                logger.warning(f"Error downloading shard: {type(e).__name__}")

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        _, progress = await download_shard(
            shard,
            self.on_progress_wrapper,
            skip_download=True,
            skip_internet=self.offline,
        )
        # A base cached before its card declared companion repos
        # (mtp_sidecar_repo / assistant_model_repo) reports complete here and
        # the coordinator never calls ensure_shard — so the companion is
        # never fetched (phase-c spec gotcha, flagged on PR #185). Degrade
        # the reported status when a declared companion is missing on disk
        # so the download path runs and pulls it.
        # Offline mode degrades only for LOAD-BEARING companions (split
        # vision weights): optional companions can never be fetched there
        # and load_mlx_items degrades to run-without-speculation, so
        # degrading for them would turn a perfectly loadable cached base
        # into DownloadFailed on air-gapped nodes — but a vision model
        # without its weights is broken and must not report complete.
        if progress.status == "complete" and self._missing_companion(
            shard, required_only=self.offline
        ):
            return progress.model_copy(update={"status": "in_progress"})
        return progress

    @staticmethod
    def _missing_companion(shard: ShardMetadata, required_only: bool = False) -> bool:
        """True when the card declares a companion repo absent from disk."""
        from skulk.download.download_utils import model_companions_present_on_disk

        return not model_companions_present_on_disk(
            shard.model_card, required_only=required_only
        )
