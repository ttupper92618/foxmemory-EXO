# pyright: reportAny=false
"""Tests for MTP sidecar download in ResumableShardDownloader."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from skulk.download.impl_shard_downloader import ResumableShardDownloader
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    RuntimeCapabilityCardConfig,
)
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata


def _make_card(
    *,
    mtp_sidecar_repo: str | None = None,
    mtp_heads: bool | None = None,
) -> ModelCard:
    runtime = (
        RuntimeCapabilityCardConfig(
            mtp_sidecar_repo=mtp_sidecar_repo,
            mtp_heads=mtp_heads,
        )
        if mtp_sidecar_repo is not None
        else None
    )
    return ModelCard(
        model_id=ModelId("test-org/test-model"),
        storage_size=Memory.from_bytes(0),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        runtime=runtime,
    )


def _make_shard(card: ModelCard) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


@pytest.fixture
def downloader() -> ResumableShardDownloader:
    return ResumableShardDownloader(max_parallel_downloads=4, offline=False)


class TestMtpSidecarDownload:
    async def test_downloads_mtp_sidecar_when_configured(
        self, downloader: ResumableShardDownloader, tmp_path: Path
    ) -> None:
        """When mtp_sidecar_repo is set, ensure_shard calls download_shard for the sidecar."""
        card = _make_card(mtp_sidecar_repo="test-org/test-mtp", mtp_heads=True)
        shard = _make_shard(card)

        download_calls: list[dict[str, Any]] = []

        async def fake_download_shard(
            shard: Any,
            on_progress: Any,
            *,
            max_parallel_downloads: int = 8,
            allow_patterns: list[str] | None = None,
            skip_internet: bool = False,
            skip_download: bool = False,
            capacity_preflight: Any = None,
        ) -> tuple[Path, Any]:
            download_calls.append(
                {
                    "model_id": str(shard.model_card.model_id),
                    "allow_patterns": allow_patterns,
                }
            )
            (tmp_path / "config.json").write_text("{}")
            return tmp_path, MagicMock(status="complete")

        with patch(
            "skulk.download.impl_shard_downloader.download_shard",
            side_effect=fake_download_shard,
        ):
            await downloader.ensure_shard(shard)

        sidecar_calls = [c for c in download_calls if "mtp" in c["model_id"]]
        assert len(sidecar_calls) == 1
        assert sidecar_calls[0]["model_id"] == "test-org/test-mtp"
        assert sidecar_calls[0]["allow_patterns"] == ["mtp.safetensors", "config.json"]

    async def test_skips_mtp_download_when_not_configured(
        self, downloader: ResumableShardDownloader, tmp_path: Path
    ) -> None:
        """When mtp_sidecar_repo is absent, no extra download_shard call is made."""
        card = _make_card()
        shard = _make_shard(card)

        download_calls: list[dict[str, Any]] = []

        async def fake_download_shard(
            shard: Any,
            on_progress: Any,
            *,
            max_parallel_downloads: int = 8,
            allow_patterns: list[str] | None = None,
            skip_internet: bool = False,
            skip_download: bool = False,
            capacity_preflight: Any = None,
        ) -> tuple[Path, Any]:
            download_calls.append({"model_id": str(shard.model_card.model_id)})
            (tmp_path / "config.json").write_text("{}")
            return tmp_path, MagicMock(status="complete")

        with patch(
            "skulk.download.impl_shard_downloader.download_shard",
            side_effect=fake_download_shard,
        ):
            await downloader.ensure_shard(shard)

        sidecar_calls = [c for c in download_calls if "mtp" in c["model_id"]]
        assert len(sidecar_calls) == 0

    async def test_skips_mtp_download_in_offline_mode(
        self, tmp_path: Path
    ) -> None:
        """When offline=True, the MTP sidecar download block is skipped."""
        offline_downloader = ResumableShardDownloader(max_parallel_downloads=4, offline=True)
        card = _make_card(mtp_sidecar_repo="test-org/test-mtp", mtp_heads=True)
        shard = _make_shard(card)

        download_calls: list[dict[str, Any]] = []

        async def fake_download_shard(
            shard: Any,
            on_progress: Any,
            *,
            max_parallel_downloads: int = 8,
            allow_patterns: list[str] | None = None,
            skip_internet: bool = False,
            skip_download: bool = False,
            capacity_preflight: Any = None,
        ) -> tuple[Path, Any]:
            download_calls.append({"model_id": str(shard.model_card.model_id)})
            (tmp_path / "config.json").write_text("{}")
            return tmp_path, MagicMock(status="complete")

        with patch(
            "skulk.download.impl_shard_downloader.download_shard",
            side_effect=fake_download_shard,
        ):
            await offline_downloader.ensure_shard(shard)

        sidecar_calls = [c for c in download_calls if "mtp" in c["model_id"]]
        assert len(sidecar_calls) == 0

    async def test_skips_mtp_download_in_config_only_mode(
        self, downloader: ResumableShardDownloader, tmp_path: Path
    ) -> None:
        """When config_only=True, the MTP sidecar download block is skipped."""
        card = _make_card(mtp_sidecar_repo="test-org/test-mtp", mtp_heads=True)
        shard = _make_shard(card)

        download_calls: list[dict[str, Any]] = []

        async def fake_download_shard(
            shard: Any,
            on_progress: Any,
            *,
            max_parallel_downloads: int = 8,
            allow_patterns: list[str] | None = None,
            skip_internet: bool = False,
            skip_download: bool = False,
            capacity_preflight: Any = None,
        ) -> tuple[Path, Any]:
            download_calls.append({"model_id": str(shard.model_card.model_id)})
            (tmp_path / "config.json").write_text("{}")
            return tmp_path, MagicMock(status="complete")

        with patch(
            "skulk.download.impl_shard_downloader.download_shard",
            side_effect=fake_download_shard,
        ):
            await downloader.ensure_shard(shard, config_only=True)

        sidecar_calls = [c for c in download_calls if "mtp" in c["model_id"]]
        assert len(sidecar_calls) == 0
