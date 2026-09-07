import asyncio
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, cast

import anyio
import yaml
from anyio import current_time
from loguru import logger

from skulk.download.download_utils import (
    RepoDownloadProgress,
    delete_model,
    map_repo_download_progress_to_download_progress_data,
    model_companions_present_on_disk,
    resolve_model_in_path,
)
from skulk.download.shard_downloader import ShardDownloader
from skulk.routing.router import TelemetrySender
from skulk.shared.constants import SKULK_MODELS_DIR
from skulk.shared.models.model_cards import (
    ModelId,
    get_model_cards,
    register_installed_card_record,
    same_model_artifact,
    unregister_installed_card_record,
)
from skulk.shared.types.commands import (
    CancelDownload,
    DeleteDownload,
    ForwarderDownloadCommand,
    PurgeStagingCache,
    RestartNode,
    StartDownload,
    SyncConfig,
)
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import (
    Event,
    NodeDownloadProgress,
)
from skulk.shared.types.telemetry import NodeTelemetry
from skulk.shared.types.worker.downloads import (
    DownloadAttemptId,
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from skulk.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from skulk.store.config import resolve_config_path, update_skulk_config_atomic
from skulk.store.installed_cards import (
    refresh_registry_installed_card_if_same_artifact,
)
from skulk.utils.channels import Receiver, Sender
from skulk.utils.task_group import TaskGroup

JsonObject = dict[str, object]


def _coerce_json_object(value: object) -> JsonObject:
    """Normalize one decoded config object into a string-keyed mapping."""
    if not isinstance(value, dict):
        return {}
    raw_dict = cast(dict[object, object], value)
    return {str(key): item for key, item in raw_dict.items()}


def _installed_artifact_model_id(model_directory: Path) -> ModelId | None:
    """Return the installed artifact alias retained beside a model directory."""

    from skulk.store.installed_cards import read_installed_card_with_fallback

    try:
        record = read_installed_card_with_fallback(model_directory)
    except (OSError, ValueError):
        return None
    if record is None:
        return None
    return ModelId(record.artifact_model_id)


@dataclass
class DownloadCoordinator:
    node_id: NodeId
    shard_downloader: ShardDownloader
    download_command_receiver: Receiver[ForwarderDownloadCommand]
    event_sender: Sender[Event]
    telemetry_sender: TelemetrySender | Sender[NodeTelemetry]
    offline: bool = False
    staging_cache_path: Path | None = None
    config_applied_callback: Callable[[], None] | None = None

    # Local state
    download_status: dict[ModelId, DownloadProgress] = field(default_factory=dict)
    active_downloads: dict[ModelId, anyio.CancelScope] = field(default_factory=dict)

    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)
    _stopped: anyio.Event = field(init=False, default_factory=anyio.Event)

    # Per-model throttle for in-progress download readings. Raw downloaders fire
    # once per chunk across parallel files, so admitting every callback would
    # continuously replace telemetry and waste serialization/transport work even
    # though only the latest value matters. The fraction-delta gate bounds useful
    # samples independently of model size; terminal transitions bypass it.
    _last_progress_time: dict[ModelId, float] = field(default_factory=dict)
    _last_progress_fraction: dict[ModelId, float] = field(default_factory=dict)
    _attempt_ids: dict[ModelId, DownloadAttemptId] = field(default_factory=dict)
    _suppressed_progress: set[ModelId] = field(default_factory=set)
    _pending_download_starts: dict[ModelId, ShardMetadata] = field(
        default_factory=dict
    )

    # Emit an in_progress update only after the download advances this fraction
    # (so ~20 readings total), never faster than the min interval (no bursts on a
    # fast/cached download), and at least once per heartbeat (so a slow download
    # still updates). Dropping intermediate updates is loss-free because the
    # telemetry view itself retains only the latest per-(node, model) status.
    _PROGRESS_STEP: ClassVar[float] = 0.05
    _MIN_INTERVAL_SECS: ClassVar[float] = 1.0
    _HEARTBEAT_SECS: ClassVar[float] = 30.0

    def __post_init__(self) -> None:
        self.shard_downloader.on_progress(self._download_progress_callback)

    def _model_dir(self, model_id: ModelId) -> str:
        return str(SKULK_MODELS_DIR / model_id.normalize())

    async def _resolve_or_refresh_complete_artifact(
        self,
        shard: ShardMetadata,
        candidate: Path | None = None,
    ) -> Path | None:
        """Resolve exact installed truth or refresh a proven card-only update.

        Args:
            shard: Exact card generation requested by the coordinator.
            candidate: Optional byte-complete directory reported by the direct
                downloader's status probe.

        Returns:
            A directory carrying the exact requested installed-card identity,
            or ``None`` when bytes must still be downloaded or staged.

        Side effects:
            May atomically refresh only installed-card metadata when an older
            sidecar proves the same repository, revision, and selected files.
        """

        card = shard.model_card
        exact = resolve_model_in_path(
            card.model_id,
            card.source_revision,
            expected_card=card,
            artifact_root=(
                card.artifact_bundle.root if card.artifact_bundle is not None else None
            ),
        )
        if exact is not None:
            return exact

        candidates: list[Path] = []
        if candidate is not None:
            candidates.append(candidate.parent if candidate.is_file() else candidate)
        retained_path = resolve_model_in_path(
            card.model_id,
            card.source_revision,
            artifact_root=(
                card.artifact_bundle.root if card.artifact_bundle is not None else None
            ),
        )
        if retained_path is not None:
            candidates.append(retained_path)
        if card.registry_card_id is not None:
            candidates.append(Path(self._model_dir(card.model_id)))
        seen: set[Path] = set()
        for artifact_directory in candidates:
            resolved = artifact_directory.resolve()
            if resolved in seen or not artifact_directory.is_dir():
                continue
            seen.add(resolved)
            if card.registry_card_id is None:
                return artifact_directory
            refreshed = await asyncio.to_thread(
                refresh_registry_installed_card_if_same_artifact,
                artifact_directory,
                card,
            )
            if refreshed is not None:
                register_installed_card_record(refreshed)
                logger.info(
                    "DownloadCoordinator: refreshed installed-card identity for "
                    f"{card.model_id} without transferring model bytes"
                )
                return artifact_directory
        return None

    def _reset_progress_throttle(self, model_id: ModelId) -> None:
        """Start a fresh progress high-water mark for one download attempt."""

        self._last_progress_time.pop(model_id, None)
        self._last_progress_fraction.pop(model_id, None)

    def _begin_attempt(self, model_id: ModelId) -> DownloadAttemptId:
        """Create and retain a fresh ordering identity for one download attempt."""

        attempt_id = DownloadAttemptId()
        self._attempt_ids[model_id] = attempt_id
        self._suppressed_progress.discard(model_id)
        return attempt_id

    def _begin_reset(self, model_id: ModelId) -> DownloadAttemptId:
        """Create a reset boundary and suppress tail samples from prior work."""

        attempt_id = DownloadAttemptId()
        self._attempt_ids[model_id] = attempt_id
        self._suppressed_progress.add(model_id)
        return attempt_id

    def _attempt_for(self, model_id: ModelId) -> DownloadAttemptId:
        """Return the current attempt identity, creating one for discovered work."""

        return self._attempt_ids.setdefault(model_id, DownloadAttemptId())

    def _prepare_status(self, status: DownloadProgress) -> DownloadProgress:
        """Attach one stable attempt identity and retain the prepared local status."""

        model_id = status.shard_metadata.model_card.model_id
        prepared = status.model_copy(
            update={"attempt_id": status.attempt_id or self._attempt_for(model_id)}
        )
        self.download_status[model_id] = prepared
        return prepared

    async def _emit_status(self, status: DownloadProgress) -> DownloadProgress:
        """Route transient status to telemetry and decisions to the event log."""

        prepared = self._prepare_status(status)
        if isinstance(prepared, DownloadOngoing):
            await self.telemetry_sender.send(
                NodeTelemetry(node_id=self.node_id, info=prepared)
            )
        elif isinstance(prepared, DownloadPending):
            # Pending is both the live initial state and the rare durable decision
            # that clears an older terminal outcome before a new attempt.
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=prepared)
            )
            await self.telemetry_sender.send(
                NodeTelemetry(node_id=self.node_id, info=prepared)
            )
        else:
            await self.event_sender.send(
                NodeDownloadProgress(download_progress=prepared)
            )
        return prepared

    def _emit_ongoing_nowait(self, status: DownloadOngoing) -> DownloadOngoing:
        """Offer initial live progress from the synchronous task-start path."""

        prepared = cast(DownloadOngoing, self._prepare_status(status))
        self.telemetry_sender.send_nowait(
            NodeTelemetry(node_id=self.node_id, info=prepared)
        )
        return prepared

    def _should_emit_in_progress(self, model_id: ModelId, fraction: float) -> bool:
        """Whether to emit an in-progress reading given download advancement.

        Bounds the high-frequency in_progress stream so it cannot saturate the
        ordered control plane (#364): emit on a fraction-delta step (rate-floored
        to avoid bursts) or a slow-download heartbeat.
        """
        now = current_time()
        # A new attempt resets these maps at its lifecycle boundary. Treat a
        # lower fraction inside an attempt as stale/out-of-order instead of a
        # restart; accepting it would re-seat the baseline and let callback
        # jitter bypass the bounded event-count guarantee.
        if model_id not in self._last_progress_fraction:
            self._last_progress_time[model_id] = now
            self._last_progress_fraction[model_id] = fraction
            return True
        last_time = self._last_progress_time.get(model_id, 0.0)
        last_fraction = self._last_progress_fraction[model_id]
        if fraction < last_fraction:
            return False
        advanced = (
            fraction - last_fraction >= self._PROGRESS_STEP
            and now - last_time >= self._MIN_INTERVAL_SECS
        )
        heartbeat = now - last_time >= self._HEARTBEAT_SECS
        if advanced or heartbeat:
            self._last_progress_time[model_id] = now
            self._last_progress_fraction[model_id] = fraction
            return True
        return False

    async def _download_progress_callback(
        self, callback_shard: ShardMetadata, progress: RepoDownloadProgress
    ) -> None:
        model_id = callback_shard.model_card.model_id
        if model_id in self._suppressed_progress or model_id not in self.active_downloads:
            # Nested companion transfers belong to their parent's installation.
            # Recording them as separate ongoing jobs strands those entries:
            # only the requested shard has an owned finalization task.
            return

        if progress.status == "complete":
            # Transfer completion precedes installed-card hashing/persistence.
            # Only ensure_shard returning can authorize the planner to load.
            return
        elif progress.status == "in_progress":
            total_bytes = progress.total.in_bytes
            fraction = (
                progress.downloaded.in_bytes / total_bytes if total_bytes > 0 else 0.0
            )
            if not self._should_emit_in_progress(model_id, fraction):
                return
            ongoing = DownloadOngoing(
                node_id=self.node_id,
                shard_metadata=callback_shard,
                download_progress=map_repo_download_progress_to_download_progress_data(
                    progress, include_files=False
                ),
                model_directory=self._model_dir(model_id),
            )
            await self._emit_status(ongoing)

    async def run(self) -> None:
        logger.info(
            f"Starting DownloadCoordinator{' (offline mode)' if self.offline else ''}"
        )
        try:
            async with self._tg as tg:
                tg.start_soon(self._command_processor)
                tg.start_soon(self._emit_existing_download_progress)
        finally:
            self._stopped.set()

    async def shutdown(self) -> None:
        self._tg.cancel_tasks()
        await self._stopped.wait()

    async def _command_processor(self) -> None:
        with self.download_command_receiver as commands:
            async for cmd in commands:
                # Cluster-wide commands — every node handles these
                match cmd.command:
                    case SyncConfig(config_yaml=config_yaml):
                        await self._sync_config(config_yaml)
                        continue
                    case PurgeStagingCache(model_id=purge_model_id):
                        await self._purge_staging_cache(purge_model_id)
                        continue
                    case _:
                        pass  # Targeted commands handled below

                # Only process targeted commands for this node
                if cmd.command.target_node_id != self.node_id:
                    logger.debug(
                        f"DownloadCoordinator: ignoring command for {cmd.command.target_node_id} (we are {self.node_id})"
                    )
                    continue

                match cmd.command:
                    case StartDownload(shard_metadata=shard):
                        logger.info(
                            f"DownloadCoordinator: received StartDownload for {shard.model_card.model_id}"
                        )
                        await self._start_download(shard)
                    case DeleteDownload(model_id=model_id):
                        await self._delete_download(model_id)
                    case CancelDownload(model_id=model_id, attempt_id=attempt_id):
                        await self._cancel_download(model_id, attempt_id)
                    case RestartNode():
                        await self._restart_node()

    async def _cancel_download(
        self,
        model_id: ModelId,
        attempt_id: DownloadAttemptId | None = None,
    ) -> None:
        """Cancel the current model download when its attempt identity matches."""
        if model_id in self.active_downloads and model_id in self.download_status:
            current_status = self.download_status[model_id]
            if attempt_id is not None and current_status.attempt_id != attempt_id:
                logger.info(
                    "Ignoring cancellation for stale download attempt "
                    f"{attempt_id} of {model_id}"
                )
                return
            logger.info(f"Cancelling download for {model_id}")
            self.active_downloads[model_id].cancel()
            self._reset_progress_throttle(model_id)
            self._begin_reset(model_id)
            pending = DownloadPending(
                shard_metadata=current_status.shard_metadata,
                node_id=self.node_id,
                model_directory=self._model_dir(model_id),
            )
            await self._emit_status(pending)

    async def _restart_node(self) -> None:
        """Restart this node by replacing the current process image (in-place restart)."""
        from skulk.utils.restart import schedule_restart

        logger.info(
            "RestartNode command received — scheduling in-place process restart"
        )
        schedule_restart()

    async def _sync_config(self, config_yaml: str) -> None:
        """Write received config YAML to the local config file and
        apply runtime-effective settings (e.g., KV cache backend)."""
        config_path = resolve_config_path()
        try:
            received = _coerce_json_object(cast(object, yaml.safe_load(config_yaml)))

            def preserve_local_fields(
                existing: dict[str, object],
            ) -> dict[str, object]:
                """Merge node-local fields inside the write lock.

                An incoming ``hf_token`` is adopted (that is how a token
                entered in one node's Settings reaches the store host), but
                absent-or-blank must never erase a locally configured one.
                """

                from skulk.store.config import normalized_hf_token

                updated = dict(received)
                if normalized_hf_token(
                    updated.get("hf_token")
                ) is None and existing.get("hf_token"):
                    updated["hf_token"] = existing["hf_token"]
                if "model_trust" not in updated and "model_trust" in existing:
                    updated["model_trust"] = existing["model_trust"]
                return updated

            raw = update_skulk_config_atomic(config_path, preserve_local_fields)
            local_config_yaml = yaml.safe_dump(
                raw,
                default_flow_style=False,
                sort_keys=False,
            )
            logger.info(
                f"DownloadCoordinator: synced {config_path.name} from cluster ({len(local_config_yaml)} bytes)"
            )
            # Apply inference config to env var so next runner spawn picks it up
            inference = _coerce_json_object(raw.get("inference"))
            if "kv_cache_backend" in inference:
                # Don't overwrite if user provided the env var at launch
                if not os.environ.get("_SKULK_KV_BACKEND_USER_SET"):
                    os.environ["SKULK_KV_CACHE_BACKEND"] = str(
                        inference["kv_cache_backend"]
                    )
                    logger.info(
                        f"DownloadCoordinator: updated KV_CACHE_BACKEND={inference['kv_cache_backend']}"
                    )
                else:
                    logger.info(
                        "DownloadCoordinator: skipping KV backend update (user env var override active)"
                    )
            # Promote the merged token: replaces a config-derived value so
            # rotation converges, never an operator-supplied launch value,
            # and whitespace never lands in the environment.
            from skulk.store.config import promote_hf_token

            _ = promote_hf_token(raw.get("hf_token"), source="config sync")
            # Apply logging config — enable/disable structured stdout
            logging_cfg = _coerce_json_object(raw.get("logging"))
            if logging_cfg:
                from skulk.shared.logging import (
                    external_log_pipe_enabled,
                    set_structured_stdout,
                )

                # External-shipper mode is install-level (env var set by
                # the service wrapper) and overrides runtime sync. In
                # internal-subprocess mode (env var off), require both
                # `enabled` and a non-empty `ingest_url` so clearing the
                # URL at runtime disables shipping (legacy contract).
                ingest_url_str = str(logging_cfg.get("ingest_url", ""))
                log_enabled = external_log_pipe_enabled() or (
                    bool(logging_cfg.get("enabled", False)) and bool(ingest_url_str)
                )
                set_structured_stdout(log_enabled, ingest_url=ingest_url_str)
            if self.config_applied_callback is not None:
                self.config_applied_callback()
        except Exception as exc:
            logger.warning(f"DownloadCoordinator: failed to sync config: {exc}")

    async def _purge_dir(self, path: Path, label: str) -> int:
        """Remove all model subdirectories from *path*. Returns count purged."""
        if not path.exists():
            return 0
        purged = 0
        for entry in path.iterdir():
            if entry.is_dir():
                installed_model_id = _installed_artifact_model_id(entry)
                logger.info(f"PurgeStagingCache: removing {entry} ({label})")
                await asyncio.to_thread(shutil.rmtree, entry, True)
                if installed_model_id is not None and not entry.exists():
                    unregister_installed_card_record(installed_model_id)
                purged += 1
        return purged

    async def _purge_staging_cache(self, model_id: ModelId | None) -> None:
        """Remove staged and downloaded model files from the local node."""
        if model_id is None:
            self._pending_download_starts.clear()
        else:
            self._pending_download_starts.pop(model_id, None)

        # Collect all directories to purge: staging cache + standard models dir
        purge_dirs: list[tuple[Path, str]] = []
        if self.staging_cache_path is not None:
            purge_dirs.append((self.staging_cache_path.expanduser(), "staging"))
        purge_dirs.append((SKULK_MODELS_DIR, "models"))

        if model_id is not None:
            # Purge a specific model
            if model_id in self.active_downloads:
                self.active_downloads[model_id].cancel()
            sanitized = str(model_id).replace("/", "--")
            found = False
            for dir_path, label in purge_dirs:
                model_dir = dir_path / sanitized
                if model_dir.exists():
                    logger.info(f"PurgeStagingCache: removing {model_dir} ({label})")
                    await asyncio.to_thread(shutil.rmtree, model_dir, True)
                    found = True
            if not found:
                # Also try the normalized form used by SKULK_MODELS_DIR
                norm_dir = SKULK_MODELS_DIR / model_id.normalize()
                if norm_dir.exists():
                    logger.info(f"PurgeStagingCache: removing {norm_dir} (models)")
                    await asyncio.to_thread(shutil.rmtree, norm_dir, True)
                    found = True
            if found and model_id in self.download_status:
                unregister_installed_card_record(model_id)
                current = self.download_status[model_id]
                self._begin_reset(model_id)
                pending = DownloadPending(
                    shard_metadata=current.shard_metadata,
                    node_id=self.node_id,
                    model_directory=self._model_dir(model_id),
                )
                await self._emit_status(pending)
                del self.download_status[model_id]
            elif not found:
                logger.info(f"PurgeStagingCache: model {model_id} not found")
            elif found:
                unregister_installed_card_record(model_id)
        else:
            # Purge all models from all directories
            # Cancel all active downloads first
            for _mid, scope in list(self.active_downloads.items()):
                scope.cancel()

            total_purged = 0
            for dir_path, label in purge_dirs:
                total_purged += await self._purge_dir(dir_path, label)

            # Also clear the HF file list cache so stale entries don't linger
            hf_cache = SKULK_MODELS_DIR / "caches"
            if hf_cache.exists():
                logger.info(
                    f"PurgeStagingCache: clearing file list cache at {hf_cache}"
                )
                await asyncio.to_thread(shutil.rmtree, hf_cache, True)

            # Reset all download statuses (including read-only entries —
            # their files have been deleted so they are no longer valid)
            for mid, status in list(self.download_status.items()):
                self._begin_reset(mid)
                pending = DownloadPending(
                    shard_metadata=status.shard_metadata,
                    node_id=self.node_id,
                    model_directory=self._model_dir(mid),
                )
                await self._emit_status(pending)
                del self.download_status[mid]

            logger.info(f"PurgeStagingCache: purged {total_purged} model directories")

    async def _start_download(self, shard: ShardMetadata) -> None:
        model_id = shard.model_card.model_id
        status = self.download_status.get(model_id)
        artifact_changed = (
            status is not None
            and not same_model_artifact(
                status.shard_metadata.model_card,
                shard.model_card,
            )
        )

        active_scope = self.active_downloads.get(model_id)
        if active_scope is not None:
            if active_scope.cancel_called:
                # Cancellation cleanup can outlive the command-loop tick that
                # requested it. Retain the replacement command so the active
                # task's finally block can start it after releasing ownership.
                self._pending_download_starts[model_id] = shard
                logger.info(
                    f"DownloadCoordinator: queued restart for {model_id} until "
                    "the cancelled download task finishes"
                )
                return
            if artifact_changed:
                self._begin_reset(model_id)
                self._pending_download_starts[model_id] = shard
                active_scope.cancel()
                logger.info(
                    f"DownloadCoordinator: replacing active download for {model_id} "
                    "because its immutable artifact identity changed"
                )
                return
            logger.info(
                f"DownloadCoordinator: {model_id} still has an active download task"
            )
            return

        # Check if already downloading, complete, or recently failed
        if status is not None:
            if artifact_changed:
                logger.info(
                    f"DownloadCoordinator: {model_id} has stale "
                    f"{type(status).__name__} state from another artifact; "
                    "starting a fresh download"
                )
                del self.download_status[model_id]
            # If marked completed but model directory no longer exists on disk,
            # clear the stale status and re-download from scratch.
            elif isinstance(status, DownloadCompleted) and status.model_directory:
                model_dir = Path(status.model_directory)
                if (
                    not model_dir.is_dir()
                    or not (model_dir / "config.json").exists()
                ):
                    logger.info(
                        f"DownloadCoordinator: {model_id} was DownloadCompleted but "
                        "its directory is stale; re-downloading"
                    )
                    del self.download_status[model_id]
                    # Fall through to fresh download below
                else:
                    logger.info(
                        f"DownloadCoordinator: {model_id} already DownloadCompleted, re-emitting"
                    )
                    await self._emit_status(status)
                    return
            elif isinstance(status, (DownloadOngoing, DownloadFailed)):
                logger.info(
                    f"DownloadCoordinator: {model_id} already {type(status).__name__}, re-emitting"
                )
                await self._emit_status(status)
                return

        self._begin_attempt(model_id)

        # Check SKULK_MODELS_PATH for pre-downloaded models. A base model on
        # disk does NOT make the download complete when the card declares a
        # companion (MTP sidecar / assistant) that is missing — treating it
        # as complete here bypassed ensure_shard entirely and loaded staged
        # models with speculative decoding silently unavailable (codex
        # review on the launch-smoke fix). Fall through to the download
        # path instead so the companion gets fetched.
        found_path = await self._resolve_or_refresh_complete_artifact(shard)
        # In offline mode optional companions can never be fetched and the
        # runner degrades to run-without-speculation, so only load-bearing
        # companions (split vision weights) block the shortcut there —
        # hiding a loadable base behind a doomed download would produce
        # DownloadFailed for a usable model, but advertising a vision model
        # without its weights would hand out a broken one.
        if found_path is not None and not model_companions_present_on_disk(
            shard.model_card, required_only=self.offline
        ):
            logger.info(
                f"DownloadCoordinator: {model_id} base model is on disk at "
                f"{found_path} but a declared companion repo is missing — "
                "running the download path to fetch it"
            )
            found_path = None
        if found_path is not None:
            logger.info(
                f"DownloadCoordinator: Model {model_id} found in SKULK_MODELS_PATH at {found_path}"
            )
            completed = DownloadCompleted(
                shard_metadata=shard,
                node_id=self.node_id,
                total=shard.model_card.storage_size,
                model_directory=str(found_path),
                read_only=True,
            )
            await self._emit_status(completed)
            return

        # Emit pending status
        progress = DownloadPending(
            shard_metadata=shard,
            node_id=self.node_id,
            model_directory=self._model_dir(model_id),
        )
        await self._emit_status(progress)

        # Check initial status from downloader
        initial_progress = (
            await self.shard_downloader.get_shard_download_status_for_shard(shard)
        )

        completed_path = (
            await self._resolve_or_refresh_complete_artifact(
                shard,
                Path(self._model_dir(model_id)),
            )
            if initial_progress.status == "complete"
            else None
        )
        if initial_progress.status == "complete" and completed_path is not None:
            completed = DownloadCompleted(
                shard_metadata=shard,
                node_id=self.node_id,
                total=initial_progress.total,
                model_directory=str(completed_path),
            )
            await self._emit_status(completed)
            return
        if initial_progress.status == "complete":
            # Byte-only direct-download probes cannot distinguish a genuinely
            # different generation from a replacement card whose newly
            # selected files are absent. Keep those cases on the transfer path.
            initial_progress = initial_progress.model_copy(
                update={"status": "in_progress"}
            )

        if self.offline:
            logger.warning(
                f"Offline mode: model {model_id} is not fully available locally, cannot download"
            )
            failed = DownloadFailed(
                shard_metadata=shard,
                node_id=self.node_id,
                error_message=f"Model files not found locally in offline mode: {model_id}",
                model_directory=self._model_dir(model_id),
            )
            await self._emit_status(failed)
            return

        # Start actual download
        self._start_download_task(shard, initial_progress)

    def _start_download_task(
        self, shard: ShardMetadata, initial_progress: RepoDownloadProgress
    ) -> None:
        model_id = shard.model_card.model_id
        self._reset_progress_throttle(model_id)

        # Emit ongoing status
        status = DownloadOngoing(
            node_id=self.node_id,
            shard_metadata=shard,
            download_progress=map_repo_download_progress_to_download_progress_data(
                initial_progress, include_files=False
            ),
            model_directory=self._model_dir(model_id),
        )
        self._emit_ongoing_nowait(status)

        async def download_wrapper(cancel_scope: anyio.CancelScope) -> None:
            try:
                path: Path | None = None
                with cancel_scope:
                    path = await self.shard_downloader.ensure_shard(shard)
                if path is not None and not cancel_scope.cancel_called:
                    # The returned path and installed identity become visible in
                    # one terminal event, after all installation work succeeds.
                    completed = DownloadCompleted(
                        shard_metadata=shard,
                        node_id=self.node_id,
                        total=shard.model_card.storage_size,
                        model_directory=str(path.parent if path.is_file() else path),
                    )
                    await self._emit_status(completed)
                    self._reset_progress_throttle(model_id)
            except Exception as e:
                logger.error(f"Download failed for {model_id}: {e}")
                self._reset_progress_throttle(model_id)
                failed = DownloadFailed(
                    shard_metadata=shard,
                    node_id=self.node_id,
                    error_message=str(e),
                    model_directory=self._model_dir(model_id),
                )
                await self._emit_status(failed)
            except anyio.get_cancelled_exc_class():
                # ignore cancellation - let cleanup do its thing
                pass
            finally:
                self.active_downloads.pop(model_id, None)
                pending_start = self._pending_download_starts.pop(model_id, None)
                if pending_start is not None:
                    await self._start_download(pending_start)

        scope = anyio.CancelScope()
        self._tg.start_soon(download_wrapper, scope)
        self.active_downloads[model_id] = scope

    async def _delete_download(self, model_id: ModelId) -> None:
        # Delete is authoritative over an earlier replacement start that was
        # waiting for cancellation cleanup to release task ownership.
        self._pending_download_starts.pop(model_id, None)

        # Protect read-only models (from SKULK_MODELS_PATH) from deletion
        if model_id in self.download_status:
            current = self.download_status[model_id]
            if isinstance(current, DownloadCompleted) and current.read_only:
                logger.warning(
                    f"Refusing to delete read-only model {model_id} (from SKULK_MODELS_PATH)"
                )
                return

        # Cancel if active
        if model_id in self.active_downloads:
            logger.info(f"Cancelling active download for {model_id} before deletion")
            self.active_downloads[model_id].cancel()

        # Delete from disk
        logger.info(f"Deleting model files for {model_id}")
        deleted = await delete_model(model_id)
        staged_deleted = False

        if deleted:
            logger.info(f"Successfully deleted model {model_id}")
        else:
            logger.warning(f"Model {model_id} was not found on disk")

        # Also remove the staging directory if the model was downloaded from the
        # store into a non-default location (e.g. ~/.skulk/staging/<model>).
        # delete_model() only removes SKULK_MODELS_DIR, so the staged copy would
        # otherwise survive and make the model reappear on the next activation.
        if model_id in self.download_status:
            current_status = self.download_status[model_id]
            if isinstance(current_status, DownloadCompleted):
                staging_dir = Path(current_status.model_directory)
                standard_dir = Path(self._model_dir(model_id))
                if staging_dir != standard_dir and staging_dir.exists():
                    logger.info(f"Deleting staged model files at {staging_dir}")
                    await asyncio.to_thread(shutil.rmtree, staging_dir, True)
                    staged_deleted = not staging_dir.exists()

        if deleted or staged_deleted:
            unregister_installed_card_record(model_id)

        # Emit pending status to reset UI state, then remove from local tracking
        if model_id in self.download_status:
            current_status = self.download_status[model_id]
            self._begin_reset(model_id)
            pending = DownloadPending(
                shard_metadata=current_status.shard_metadata,
                node_id=self.node_id,
                model_directory=self._model_dir(model_id),
            )
            await self._emit_status(pending)
            del self.download_status[model_id]

    async def _emit_existing_download_progress(self) -> None:
        hf_scan_done = False
        models_path_logged = False
        while True:
            try:
                # HF download status scan — only run once at startup to detect
                # interrupted downloads. No need to poll HF every 60 seconds.
                if not hf_scan_done:
                    hf_scan_done = True
                    logger.debug(
                        "DownloadCoordinator: One-time scan for existing HF download progress..."
                    )
                    async for (
                        discovered_path,
                        progress,
                    ) in self.shard_downloader.get_shard_download_status():
                        model_id = progress.shard.model_card.model_id

                        # Skip models with no local data
                        if progress.downloaded.in_bytes == 0:
                            continue

                        logger.info(
                            f"DownloadCoordinator scan: {model_id} status={progress.status} "
                            f"downloaded={progress.downloaded.in_bytes} total={progress.total.in_bytes} "
                            f"completed_files={progress.completed_files}/{progress.total_files}"
                        )

                        if model_id in self.active_downloads:
                            continue

                        if progress.status == "complete":
                            completed_path = (
                                await self._resolve_or_refresh_complete_artifact(
                                    progress.shard,
                                    discovered_path,
                                )
                            )
                            companions_present = model_companions_present_on_disk(
                                progress.shard.model_card,
                                required_only=self.offline,
                            )
                            if completed_path is not None and companions_present:
                                status: DownloadProgress = DownloadCompleted(
                                    node_id=self.node_id,
                                    shard_metadata=progress.shard,
                                    total=progress.total,
                                    model_directory=str(completed_path),
                                )
                            else:
                                status = DownloadPending(
                                    node_id=self.node_id,
                                    shard_metadata=progress.shard,
                                    model_directory=self._model_dir(model_id),
                                    downloaded=progress.downloaded,
                                    total=progress.total,
                                )
                        elif progress.status in ["in_progress", "not_started"]:
                            if progress.downloaded_this_session.in_bytes == 0:
                                status = DownloadPending(
                                    node_id=self.node_id,
                                    shard_metadata=progress.shard,
                                    model_directory=self._model_dir(
                                        progress.shard.model_card.model_id
                                    ),
                                    downloaded=progress.downloaded,
                                    total=progress.total,
                                )
                            else:
                                status = DownloadOngoing(
                                    node_id=self.node_id,
                                    shard_metadata=progress.shard,
                                    download_progress=map_repo_download_progress_to_download_progress_data(
                                        progress, include_files=False
                                    ),
                                    model_directory=self._model_dir(
                                        progress.shard.model_card.model_id
                                    ),
                                )
                        else:
                            continue

                        await self._emit_status(status)

                # Scan SKULK_MODELS_PATH for pre-downloaded models (runs every
                # cycle). Log the search path once — repeating the full tuple
                # every 60s was pure log noise that bloated piped logs.
                if not models_path_logged:
                    import skulk.shared.constants as _dbg_constants

                    logger.debug(
                        f"DownloadCoordinator: SKULK_MODELS_PATH={_dbg_constants.SKULK_MODELS_PATH}"
                    )
                    models_path_logged = True
                for card in await get_model_cards():
                    mid = card.model_id
                    if mid in self.active_downloads:
                        continue
                    if isinstance(
                        self.download_status.get(mid),
                        (DownloadCompleted, DownloadOngoing, DownloadFailed),
                    ):
                        continue
                    found = resolve_model_in_path(
                        mid,
                        card.source_revision,
                        expected_card=card,
                        artifact_root=(
                            card.artifact_bundle.root
                            if card.artifact_bundle is not None
                            else None
                        ),
                    )
                    if found is not None and not model_companions_present_on_disk(
                        card, required_only=self.offline
                    ):
                        # Same companion gate as _start_download: a staged
                        # base missing a companion must not be advertised
                        # as complete (offline nodes check only the
                        # load-bearing vision weights — optional companions
                        # can never arrive there).
                        continue
                    if found is not None:
                        logger.info(
                            f"DownloadCoordinator: SKULK_MODELS_PATH hit: {mid} -> {found}"
                        )
                        path_shard = PipelineShardMetadata(
                            model_card=card,
                            device_rank=0,
                            world_size=1,
                            start_layer=0,
                            end_layer=card.n_layers,
                            n_layers=card.n_layers,
                        )
                        self._begin_attempt(mid)
                        path_completed: DownloadProgress = DownloadCompleted(
                            node_id=self.node_id,
                            shard_metadata=path_shard,
                            total=card.storage_size,
                            model_directory=str(found),
                            read_only=True,
                        )
                        await self._emit_status(path_completed)

            except Exception as e:
                logger.error(
                    f"DownloadCoordinator: Error emitting existing download progress: {e}"
                )
            await anyio.sleep(60)
