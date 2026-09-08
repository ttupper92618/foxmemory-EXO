# pyright: reportPrivateUsage=false
"""Cluster model listings project authoritative installed-card state."""

import asyncio
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

import skulk.api.main as api_main
from skulk.api.main import API
from skulk.shared.models.memory_estimate import estimate_shard_footprint
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    VisionCardConfig,
    authorized_model_card_digest,
)
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory
from skulk.store.installed_cards import (
    InstalledCardRecord,
    build_installed_card_record,
)
from skulk.store.model_store_client import ModelStoreClient


class _RegistryStoreClient:
    """Minimal store client returning a controlled registry snapshot."""

    def __init__(self, entries: list[dict[str, object]]) -> None:
        """Retain the entries returned by ``fetch_registry``."""

        self.entries = entries
        self.fetch_count = 0
        self.block = False

    async def fetch_registry(
        self,
        *,
        raise_on_error: bool = False,
    ) -> list[dict[str, object]]:
        """Return one synthetic authoritative-store snapshot."""

        del raise_on_error
        self.fetch_count += 1
        if self.block:
            await asyncio.Event().wait()
        return self.entries


def _card(identity_character: str) -> ModelCard:
    """Create one immutable signed card for the projection tests."""

    return ModelCard(
        model_id=ModelId("example/model@q4"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        source_revision=identity_character * 40,
        registry_card_id=f"card_{identity_character * 52}",
        registry_snapshot_id="snapshot_test",
        registry_provenance="foxlight",
        quantization="Q4_K_M",
    )


def _installed_record(tmp_path: Path, card: ModelCard) -> InstalledCardRecord:
    """Build a revision-verified installed record for ``card``."""

    assert card.registry_card_id is not None
    artifact = tmp_path / card.registry_card_id
    artifact.mkdir()
    (artifact / "model.gguf").write_bytes(b"weights")
    assert card.source_revision is not None
    (artifact / ".skulk-source-revision").write_text(
        f"{card.source_revision}\n",
        encoding="utf-8",
    )
    return build_installed_card_record(artifact, card, artifact_format="gguf")


def _custom_installed_record(tmp_path: Path) -> InstalledCardRecord:
    """Build a custom installed record sharing the registry test alias."""

    card = ModelCard(
        model_id=ModelId("example/model@q4"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        is_custom=True,
        quantization="operator",
    )
    artifact = tmp_path / "custom"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}", encoding="utf-8")
    return build_installed_card_record(artifact, card, artifact_format="custom")


def _qualification_installed_record(tmp_path: Path) -> InstalledCardRecord:
    """Build a retained artifact whose temporary card is service-owned."""

    card = ModelCard(
        model_id=ModelId("example/model@q4"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        is_custom=True,
        qualification_only=True,
        quantization="Q4_K_M",
    )
    artifact = tmp_path / "qualification"
    artifact.mkdir()
    (artifact / "model.gguf").write_bytes(b"weights")
    return build_installed_card_record(artifact, card, artifact_format="gguf")


def _configure_model_list_test(
    monkeypatch: pytest.MonkeyPatch,
    *,
    catalog_card: ModelCard | None,
    current_registry_card_value: ModelCard | None,
    local_record: InstalledCardRecord | None,
) -> None:
    """Install deterministic catalog and local-record effects for one test."""

    async def model_cards() -> list[ModelCard]:
        return [catalog_card] if catalog_card is not None else []

    def cluster_approvals(_api: API) -> frozenset[str]:
        return frozenset()

    def intelligent_fabric_enabled(_api: API) -> bool:
        return False

    def local_record_lookup(_model_id: ModelId) -> InstalledCardRecord | None:
        return local_record

    def current_registry_card_id(_model_id: ModelId) -> str | None:
        return None

    def current_registry_card(_model_id: ModelId) -> ModelCard | None:
        return current_registry_card_value

    def model_advisories(_card: ModelCard) -> tuple[()]:
        return ()

    monkeypatch.setattr(api_main, "get_model_cards", model_cards)
    monkeypatch.setattr(api_main, "get_installed_card_record", local_record_lookup)
    monkeypatch.setattr(
        api_main,
        "get_current_registry_card_id",
        current_registry_card_id,
    )
    monkeypatch.setattr(
        api_main,
        "get_current_registry_card",
        current_registry_card,
    )
    monkeypatch.setattr(api_main, "get_model_advisories", model_advisories)
    monkeypatch.setattr(API, "_cluster_remote_code_approvals", cluster_approvals)
    monkeypatch.setattr(
        API,
        "_intelligent_fabric_enabled",
        intelligent_fabric_enabled,
    )


def _api_with_store(store_client: _RegistryStoreClient) -> API:
    """Create the minimal API state needed by the model-list projection."""

    api = object.__new__(API)
    api._store_client = cast(ModelStoreClient, cast(object, store_client))
    api._model_list_store_records_cache = {}
    api._model_list_store_records_cached_at = 0.0
    api._model_list_store_records_lock = asyncio.Lock()
    return api


async def test_models_use_cluster_store_installed_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every API node should expose the central store's active generation."""

    card = _card("a")
    record = _installed_record(tmp_path, card)
    store_client = _RegistryStoreClient(
        [
            {
                "model_id": str(card.model_id),
                "installed_card": record.model_dump(mode="json"),
            }
        ]
    )
    _configure_model_list_test(
        monkeypatch,
        catalog_card=card,
        current_registry_card_value=card,
        local_record=None,
    )
    api = _api_with_store(store_client)

    response = await api.get_models(status=None)

    assert store_client.fetch_count == 1
    assert len(response.data) == 1
    assert response.data[0].installed is True
    assert response.data[0].active_installed_identity == card.registry_card_id
    assert response.data[0].installed_verification == "registry_verified"
    assert response.data[0].update_available is False


async def test_models_fall_back_to_local_installed_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A missing store record must not erase a usable local generation."""

    installed_card = _card("a")
    current_card = _card("b")
    local_record = _installed_record(tmp_path, installed_card)
    store_client = _RegistryStoreClient([])
    _configure_model_list_test(
        monkeypatch,
        catalog_card=current_card,
        current_registry_card_value=current_card,
        local_record=local_record,
    )
    api = _api_with_store(store_client)

    response = await api.get_models(status=None)

    assert store_client.fetch_count == 1
    assert len(response.data) == 1
    assert response.data[0].installed is True
    assert response.data[0].active_installed_identity == installed_card.registry_card_id
    assert response.data[0].installed_verification == "registry_verified"
    assert response.data[0].current_registry_identity == current_card.registry_card_id
    assert response.data[0].update_available is True


async def test_custom_card_keeps_local_installed_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A same-alias store card must not override an operator's custom card."""

    registry_card = _card("a")
    store_record = _installed_record(tmp_path, registry_card)
    local_record = _custom_installed_record(tmp_path)
    custom_card = local_record.model_card
    store_client = _RegistryStoreClient(
        [
            {
                "model_id": str(registry_card.model_id),
                "installed_card": store_record.model_dump(mode="json"),
            }
        ]
    )
    _configure_model_list_test(
        monkeypatch,
        catalog_card=custom_card,
        current_registry_card_value=registry_card,
        local_record=local_record,
    )
    api = _api_with_store(store_client)

    response = await api.get_models(status=None)

    assert response.data[0].installed is True
    assert response.data[0].active_installed_identity == local_record.installed_identity
    assert response.data[0].installed_verification == "custom"


async def test_custom_card_uses_matching_cluster_store_installation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A store-backed custom generation must be installed across API nodes."""

    store_record = _custom_installed_record(tmp_path)
    custom_card = store_record.model_card
    store_client = _RegistryStoreClient(
        [
            {
                "model_id": str(custom_card.model_id),
                "installed_card": store_record.model_dump(mode="json"),
            }
        ]
    )
    _configure_model_list_test(
        monkeypatch,
        catalog_card=custom_card,
        current_registry_card_value=None,
        local_record=None,
    )
    api = _api_with_store(store_client)

    response = await api.get_models(status=None)

    assert len(response.data) == 1
    assert response.data[0].installed is True
    assert response.data[0].active_installed_identity == (
        store_record.installed_identity
    )
    assert response.data[0].installed_verification == "custom"


async def test_store_generation_supplies_active_metadata_and_current_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Active fields describe installed A while update truth identifies current B."""

    installed_card = _card("a")
    current_card = _card("b")
    store_record = _installed_record(tmp_path, installed_card)
    store_client = _RegistryStoreClient(
        [
            {
                "model_id": str(installed_card.model_id),
                "installed_card": store_record.model_dump(mode="json"),
            }
        ]
    )
    _configure_model_list_test(
        monkeypatch,
        catalog_card=current_card,
        current_registry_card_value=current_card,
        local_record=None,
    )
    api = _api_with_store(store_client)

    response = await api.get_models(status=None)

    entry = response.data[0]
    assert entry.registry_card_id == installed_card.registry_card_id
    assert entry.active_installed_identity == installed_card.registry_card_id
    assert entry.current_registry_identity == current_card.registry_card_id
    assert entry.update_available is True


async def test_store_only_installed_card_remains_in_model_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Registry removal must not hide a complete installed generation."""

    installed_card = _card("a")
    store_record = _installed_record(tmp_path, installed_card)
    store_client = _RegistryStoreClient(
        [
            {
                "model_id": str(installed_card.model_id),
                "installed_card": store_record.model_dump(mode="json"),
            }
        ]
    )
    _configure_model_list_test(
        monkeypatch,
        catalog_card=None,
        current_registry_card_value=None,
        local_record=None,
    )
    api = _api_with_store(store_client)

    response = await api.get_models(status=None)

    assert len(response.data) == 1
    assert response.data[0].id == str(installed_card.model_id)
    assert response.data[0].installed is True
    assert response.data[0].current_registry_identity is None
    assert response.data[0].update_available is False


async def test_store_only_qualification_card_stays_out_of_model_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Retained qualification bytes must not expose their temporary card."""

    store_record = _qualification_installed_record(tmp_path)
    store_client = _RegistryStoreClient(
        [
            {
                "model_id": str(store_record.artifact_model_id),
                "installed_card": store_record.model_dump(mode="json"),
            }
        ]
    )
    _configure_model_list_test(
        monkeypatch,
        catalog_card=None,
        current_registry_card_value=None,
        local_record=None,
    )
    api = _api_with_store(store_client)

    response = await api.get_models(status=None)

    assert response.data == []


async def test_store_qualification_card_cannot_override_signed_catalog_card(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A retained temporary record cannot replace a same-alias signed card."""

    catalog_card = _card("a")
    store_record = _qualification_installed_record(tmp_path)
    store_client = _RegistryStoreClient(
        [
            {
                "model_id": str(store_record.artifact_model_id),
                "installed_card": store_record.model_dump(mode="json"),
            }
        ]
    )
    _configure_model_list_test(
        monkeypatch,
        catalog_card=catalog_card,
        current_registry_card_value=catalog_card,
        local_record=None,
    )
    api = _api_with_store(store_client)

    response = await api.get_models(status=None)

    assert len(response.data) == 1
    assert response.data[0].registry_card_id == catalog_card.registry_card_id
    assert response.data[0].catalog_source == "registry"
    assert response.data[0].installed is False


async def test_local_qualification_card_cannot_override_signed_catalog_card(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A retained node-local temporary record cannot replace signed truth."""

    catalog_card = _card("a")
    local_record = _qualification_installed_record(tmp_path)
    _configure_model_list_test(
        monkeypatch,
        catalog_card=catalog_card,
        current_registry_card_value=catalog_card,
        local_record=local_record,
    )
    api = _api_with_store(_RegistryStoreClient([]))

    response = await api.get_models(status=None)

    assert len(response.data) == 1
    assert response.data[0].registry_card_id == catalog_card.registry_card_id
    assert response.data[0].catalog_source == "registry"
    assert response.data[0].installed is False


async def test_active_qualification_card_uses_matching_local_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The service-owned card remains installed while its lifecycle is active."""

    local_record = _qualification_installed_record(tmp_path)
    qualification_card = local_record.model_card
    _configure_model_list_test(
        monkeypatch,
        catalog_card=qualification_card,
        current_registry_card_value=None,
        local_record=local_record,
    )
    api = _api_with_store(_RegistryStoreClient([]))

    response = await api.get_models(status=None)

    assert len(response.data) == 1
    assert response.data[0].catalog_source == "custom"
    assert response.data[0].installed is True
    assert response.data[0].active_installed_identity == (
        local_record.installed_identity
    )


async def test_model_list_store_timeout_reuses_last_known_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unreachable store cannot stall or erase the last installed projection."""

    card = _card("a")
    record = _installed_record(tmp_path, card)
    store_client = _RegistryStoreClient(
        [
            {
                "model_id": str(card.model_id),
                "installed_card": record.model_dump(mode="json"),
            }
        ]
    )
    _configure_model_list_test(
        monkeypatch,
        catalog_card=card,
        current_registry_card_value=card,
        local_record=None,
    )
    api = _api_with_store(store_client)
    first = await api.get_models(status=None)
    store_client.block = True
    api._model_list_store_records_cached_at = 0.0
    monkeypatch.setattr(api_main, "_MODEL_LIST_STORE_FETCH_TIMEOUT_SECONDS", 0.01)

    second = await asyncio.wait_for(api.get_models(status=None), timeout=0.1)
    third = await asyncio.wait_for(api.get_models(status=None), timeout=0.1)

    assert store_client.fetch_count == 2
    assert second.data[0].active_installed_identity == (
        first.data[0].active_installed_identity
    )
    assert third.data[0].active_installed_identity == (
        first.data[0].active_installed_identity
    )


async def test_model_store_replacement_invalidates_installed_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Store convergence cannot retain another store's installed records."""

    card = _card("a")
    record = _installed_record(tmp_path, card)
    first_store = _RegistryStoreClient(
        [
            {
                "model_id": str(card.model_id),
                "installed_card": record.model_dump(mode="json"),
            }
        ]
    )
    replacement_store = _RegistryStoreClient([])
    _configure_model_list_test(
        monkeypatch,
        catalog_card=card,
        current_registry_card_value=card,
        local_record=None,
    )

    def refresh_capabilities(_api: API) -> None:
        """Keep the runtime convergence test independent of provider state."""

    monkeypatch.setattr(
        API,
        "refresh_config_dependent_capabilities",
        refresh_capabilities,
    )
    api = _api_with_store(first_store)

    first = await api.get_models(status=None)
    api.set_model_store_runtime(
        None,
        cast(ModelStoreClient, cast(object, replacement_store)),
    )
    second = await api.get_models(status=None)

    assert first.data[0].installed is True
    assert second.data[0].installed is False
    assert first_store.fetch_count == 1
    assert replacement_store.fetch_count == 1


def test_store_projection_rejects_mismatched_alias(
    tmp_path: Path,
) -> None:
    """An index key cannot assert installation for a different signed card alias."""

    card = _card("a")
    record = _installed_record(tmp_path, card)

    records = API._store_installed_records(
        [
            {
                "model_id": "example/other@q4",
                "installed_card": record.model_dump(mode="json"),
            }
        ]
    )

    assert records == {}


async def test_requirements_bind_store_card_and_context_estimate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sizing must use installed A even when the same alias now catalogs B."""
    installed = _card("a").model_copy(
        update={"num_key_value_heads": 8, "context_length": 16384}
    )
    current = _card("b").model_copy(update={"storage_size": Memory.from_gb(100)})
    record = _installed_record(tmp_path, installed)
    _configure_model_list_test(
        monkeypatch,
        catalog_card=current,
        current_registry_card_value=current,
        local_record=None,
    )
    api = _api_with_store(
        _RegistryStoreClient(
            [
                {
                    "model_id": str(installed.model_id),
                    "installed_card": record.model_dump(mode="json"),
                }
            ]
        )
    )
    first = await api.get_model_requirements(str(installed.model_id), 1024)
    larger = await api.get_model_requirements(str(installed.model_id), 8192)
    listing = await api.get_models(status=None)
    assert (
        first.registry_card_id
        == listing.data[0].registry_card_id
        == installed.registry_card_id
    )
    assert first.card_digest == authorized_model_card_digest(installed)
    assert first.storage_bytes == installed.storage_size.in_bytes
    assert (
        first.estimated_memory_bytes
        == estimate_shard_footprint(installed, 1.0, 1024).in_bytes
    )
    assert (
        larger.estimated_memory_bytes
        == estimate_shard_footprint(installed, 1.0, 8192).in_bytes
    )
    assert first.estimated_memory_bytes is not None
    assert larger.estimated_memory_bytes is not None
    assert first.estimated_memory_bytes < larger.estimated_memory_bytes
    assert first.card_digest == larger.card_digest
    assert first.estimate_only is True
    assert first.discrete_gpu_memory_fraction == 0.9


async def test_requirements_preserve_custom_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sizing follows the local custom override, not the store's signed alias."""
    store_record = _installed_record(tmp_path, _card("a"))
    local_record = _custom_installed_record(tmp_path)
    _configure_model_list_test(
        monkeypatch,
        catalog_card=local_record.model_card,
        current_registry_card_value=_card("a"),
        local_record=local_record,
    )
    api = _api_with_store(
        _RegistryStoreClient(
            [
                {
                    "model_id": str(store_record.model_card.model_id),
                    "installed_card": store_record.model_dump(mode="json"),
                }
            ]
        )
    )
    result = await api.get_model_requirements(str(local_record.model_card.model_id))
    assert result.card_digest == authorized_model_card_digest(local_record.model_card)
    assert result.registry_card_id is None
    assert result.estimated_memory_bytes is None


async def test_requirements_store_only_and_hidden_qualification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A complete store-only card is addressable; retained qualification is not."""
    record = _installed_record(tmp_path, _card("a"))
    _configure_model_list_test(
        monkeypatch,
        catalog_card=None,
        current_registry_card_value=None,
        local_record=None,
    )
    api = _api_with_store(
        _RegistryStoreClient(
            [
                {
                    "model_id": str(record.model_card.model_id),
                    "installed_card": record.model_dump(mode="json"),
                }
            ]
        )
    )
    result = await api.get_model_requirements(str(record.model_card.model_id))
    assert result.card_digest == authorized_model_card_digest(record.model_card)
    qualification = _qualification_installed_record(tmp_path)
    hidden_api = _api_with_store(
        _RegistryStoreClient(
            [
                {
                    "model_id": str(qualification.model_card.model_id),
                    "installed_card": qualification.model_dump(mode="json"),
                }
            ]
        )
    )
    with pytest.raises(HTTPException) as exc:
        await hidden_api.get_model_requirements(str(qualification.model_card.model_id))
    assert exc.value.status_code == 404


async def test_requirements_http_bounds_and_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HTTP edge rejects invalid context and never discovers unknown aliases."""
    card = _card("a").model_copy(update={"context_length": 4096})
    _configure_model_list_test(
        monkeypatch,
        catalog_card=card,
        current_registry_card_value=card,
        local_record=None,
    )
    api = _api_with_store(_RegistryStoreClient([]))
    app = FastAPI()
    app.get("/models/requirements")(api.get_model_requirements)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        for context in (0, -1, 4097, 1_048_577):
            response = await client.get(
                "/models/requirements",
                params={"model_id": str(card.model_id), "context_tokens": context},
            )
            assert response.status_code == 422
        unknown = await client.get(
            "/models/requirements", params={"model_id": "unknown/repo"}
        )
        assert unknown.status_code == 404
        valid = await client.get(
            "/models/requirements",
            params={"model_id": str(card.model_id), "context_tokens": 4096},
        )
        assert valid.status_code == 200
        assert valid.json()["estimated_memory_bytes"] is None
        assert valid.json()["context_tokens"] == 4096


def test_authorized_card_digest_covers_content_not_snapshot() -> None:
    """Copied registry IDs cannot disguise altered cards; publication can repeat."""
    card = _card("a")
    digest = authorized_model_card_digest(card)
    assert digest == authorized_model_card_digest(
        card.model_copy(update={"registry_snapshot_id": "snapshot_later"})
    )
    for update in (
        {"storage_size": Memory.from_gb(100)},
        {"source_revision": "b" * 40},
        {"num_key_value_heads": 8},
        {"context_length": 100},
    ):
        assert digest != authorized_model_card_digest(card.model_copy(update=update))
    assert digest == authorized_model_card_digest(
        ModelCard.model_validate_json(card.model_dump_json())
    )


@pytest.mark.parametrize("text_generation", [True, False])
async def test_requirements_projector_and_non_text_memory(
    monkeypatch: pytest.MonkeyPatch, text_generation: bool
) -> None:
    """Projector disk is included; text geometry cannot size a speech runner."""
    card = _card("a").model_copy(
        update={
            "gguf_file": "model.gguf",
            "num_key_value_heads": 8,
            "vision": VisionCardConfig(
                projector_file="mmproj.gguf", projector_size=1048576
            ),
            "tasks": [
                ModelTask.TextGeneration if text_generation else ModelTask.SpeechToText
            ],
        }
    )
    _configure_model_list_test(
        monkeypatch,
        catalog_card=card,
        current_registry_card_value=card,
        local_record=None,
    )
    result = await _api_with_store(_RegistryStoreClient([])).get_model_requirements(
        str(card.model_id)
    )
    assert result.storage_bytes == card.storage_size.in_bytes + 1048576
    if text_generation:
        assert (
            result.estimated_memory_bytes
            == estimate_shard_footprint(card, 1.0).in_bytes
        )
    else:
        assert result.estimated_memory_bytes is None
