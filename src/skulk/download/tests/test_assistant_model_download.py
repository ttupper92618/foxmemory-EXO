# pyright: reportAny=false
"""Tests for Gemma 4 assistant-model download in ResumableShardDownloader."""

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


def _make_card(*, assistant_model_repo: str | None = None) -> ModelCard:
    runtime = (
        RuntimeCapabilityCardConfig(assistant_model_repo=assistant_model_repo)
        if assistant_model_repo is not None
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


def _capture(tmp_path: Path):
    calls: list[dict[str, Any]] = []

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
        calls.append(
            {
                "model_id": str(shard.model_card.model_id),
                "allow_patterns": allow_patterns,
            }
        )
        (tmp_path / "config.json").write_text("{}")
        return tmp_path, MagicMock(status="complete")

    return calls, fake_download_shard


class TestAssistantModelDownload:
    async def test_downloads_assistant_when_configured(
        self, downloader: ResumableShardDownloader, tmp_path: Path
    ) -> None:
        card = _make_card(assistant_model_repo="test-org/test-assistant")
        shard = _make_shard(card)
        calls, fake = _capture(tmp_path)

        with patch(
            "skulk.download.impl_shard_downloader.download_shard", side_effect=fake
        ):
            await downloader.ensure_shard(shard)

        assistant_calls = [c for c in calls if "assistant" in c["model_id"]]
        assert len(assistant_calls) == 1
        assert assistant_calls[0]["model_id"] == "test-org/test-assistant"
        assert assistant_calls[0]["allow_patterns"] == ["*.safetensors", "config.json"]

    async def test_skips_assistant_when_not_configured(
        self, downloader: ResumableShardDownloader, tmp_path: Path
    ) -> None:
        shard = _make_shard(_make_card())
        calls, fake = _capture(tmp_path)

        with patch(
            "skulk.download.impl_shard_downloader.download_shard", side_effect=fake
        ):
            await downloader.ensure_shard(shard)

        assert [c for c in calls if "assistant" in c["model_id"]] == []

    async def test_skips_assistant_in_offline_mode(self, tmp_path: Path) -> None:
        offline = ResumableShardDownloader(max_parallel_downloads=4, offline=True)
        shard = _make_shard(_make_card(assistant_model_repo="test-org/test-assistant"))
        calls, fake = _capture(tmp_path)

        with patch(
            "skulk.download.impl_shard_downloader.download_shard", side_effect=fake
        ):
            await offline.ensure_shard(shard)

        assert [c for c in calls if "assistant" in c["model_id"]] == []

    async def test_skips_assistant_in_config_only_mode(
        self, downloader: ResumableShardDownloader, tmp_path: Path
    ) -> None:
        shard = _make_shard(_make_card(assistant_model_repo="test-org/test-assistant"))
        calls, fake = _capture(tmp_path)

        with patch(
            "skulk.download.impl_shard_downloader.download_shard", side_effect=fake
        ):
            await downloader.ensure_shard(shard, config_only=True)

        assert [c for c in calls if "assistant" in c["model_id"]] == []
