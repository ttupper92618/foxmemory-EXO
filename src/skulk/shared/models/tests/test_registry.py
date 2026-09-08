# pyright: reportPrivateUsage=false
"""Tests for signed registry loading and artifact identity separation."""

import asyncio
import json
import threading
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from anyio import Path as AsyncPath

import skulk.download.download_utils as download_utils
import skulk.shared.constants as constants_module
import skulk.shared.models.model_cards as model_cards_module
import skulk.shared.models.registry as registry_module
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    load_cached_registry_engine_support,
    registry_model_cards,
    registry_supported_backends_for_node,
)
from skulk.shared.models.registry import (
    RegistryAdvisories,
    RegistryCatalog,
    RegistryEngineSupport,
    RegistryUnavailableError,
    TufRegistryClient,
)
from skulk.shared.models.registry_gguf_metadata import RegistryGgufMetadata
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.store.installed_cards import (
    InstalledCardRecord,
    build_installed_card_record,
    write_installed_card,
)


def _catalog_payload() -> bytes:
    return json.dumps(
        {
            "schema_version": 2,
            "snapshot_id": "snapshot_1_test",
            "generated_at": "2026-08-08T12:00:00Z",
            "published_by": "validator@example.com",
            "note": "test",
            "card_metadata": {
                f"card_{'a' * 52}": {
                    "provenance": "foxlight",
                    "architecture": "future_architecture_v1",
                    "capability_claims": [
                        {
                            "capability_id": "text.generate",
                            "scope": "model",
                            "status": "observed",
                            "source": "upstream_structured",
                            "confidence": 1,
                            "evidence_urls": [],
                            "reviewer_model": None,
                            "input_modalities": ["text"],
                            "output_modalities": ["text"],
                            "details": {"pipeline_tag": "text-generation"},
                        }
                    ],
                    "source_links": {
                        "repository_url": "https://huggingface.co/org/multi-gguf",
                        "revision_url": "https://huggingface.co/org/multi-gguf/tree/"
                        + "b" * 40,
                        "artifact_url": "https://huggingface.co/org/multi-gguf/blob/"
                        + "b" * 40
                        + "/model-Q4_K_M.gguf",
                        "download_url": "https://huggingface.co/org/multi-gguf/resolve/"
                        + "b" * 40
                        + "/model-Q4_K_M.gguf",
                    },
                }
            },
            "cards": [
                {
                    "schema_version": 1,
                    "card_id": f"card_{'a' * 52}",
                    "alias": "org/multi-gguf@q4-k-m",
                    "model_ref": "org/multi-gguf@q4-k-m",
                    "artifact": {
                        "repository": "org/multi-gguf",
                        "revision": "b" * 40,
                        "selected_file": "model-Q4_K_M.gguf",
                        "format": "gguf",
                        "quantization": "Q4_K_M",
                    },
                    "card": {
                        "model_id": "org/multi-gguf",
                        "source_revision": "b" * 40,
                        "storage_size": {"in_bytes": 1024},
                        "n_layers": 4,
                        "hidden_size": 64,
                        "supports_tensor": False,
                        "tasks": ["TextGeneration"],
                        "gguf_file": "model-Q4_K_M.gguf",
                        "quantization": "Q4_K_M",
                        "placement": {"compatible_backends": ["llama_server"]},
                    },
                }
            ],
        }
    ).encode()


def _advisories_payload() -> bytes:
    """Build one active signed-warning target for cache recovery tests."""

    return json.dumps(
        {
            "schema_version": 1,
            "generated_at": "2026-08-10T12:00:00Z",
            "enforcement": "warn",
            "advisories": [
                {
                    "schema_version": 1,
                    "advisory_id": "FLA-2026-0001",
                    "severity": "critical",
                    "title": "Test model warning",
                    "description": "Retain this warning during a registry outage.",
                    "affected_card_ids": (f"card_{'a' * 52}",),
                    "affected_model_aliases": (),
                    "active": True,
                    "created_at": "2026-08-10T12:00:00Z",
                    "updated_at": "2026-08-10T12:00:00Z",
                    "enforcement": "warn",
                }
            ],
        }
    ).encode()


def _engine_support_payload() -> bytes:
    """Build signed support history with one active exact-build decision."""
    base: dict[str, object] = {
        "engine": "llama_server",
        "engine_build": "llama.cpp@sha256:" + "1" * 64,
        "architecture": "future_architecture_v1",
        "artifact_format": "gguf",
        "artifact_card_id": "card_" + "a" * 52,
        "quantization": "Q4_K_M",
        "capability_id": "text.generate",
        "evidence_kind": "feature_qualification",
        "evidence_trust": "foxlight_observed",
        "source_url": "https://evidence.example/test",
        "rationale": "Qualified exact artifact and engine build.",
        "hardware_classes": list[str](),
        "recorded_by": "operator@example.com",
        "created_at": "2026-08-16T12:00:00Z",
    }
    return json.dumps(
        {
            "schema_version": 1,
            "matrix_version": 7,
            "generated_at": "2026-08-16T12:00:00Z",
            "claims": [
                {
                    **base,
                    "claim_id": "support_" + "a" * 52,
                    "status": "experimental",
                    "source_sha256": "2" * 64,
                    "supersedes_claim_id": None,
                },
                {
                    **base,
                    "claim_id": "support_" + "b" * 52,
                    "status": "supported",
                    "source_sha256": "3" * 64,
                    "supersedes_claim_id": "support_" + "a" * 52,
                },
            ],
        }
    ).encode()


def test_registry_alias_is_separate_from_artifact_repository() -> None:
    """Two quants can use distinct runtime ids while sharing one Hub repo."""
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)

    card = registry_model_cards(catalog)[0]

    assert str(card.model_id) == "org/multi-gguf@q4-k-m"
    assert str(card.artifact_repository) == "org/multi-gguf"
    assert card.gguf_file == "model-Q4_K_M.gguf"
    assert card.registry_snapshot_id == "snapshot_1_test"
    assert card.registry_provenance == "foxlight"
    assert card.registry_architecture == "future_architecture_v1"
    assert card.registry_artifact_format == "gguf"
    assert card.registry_capability_claims[0].capability_id == "text.generate"


def test_registry_v2_bundle_round_trips_to_runtime_card() -> None:
    """The signed envelope and runtime card must agree on exact bundle truth."""

    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    cards = cast("list[dict[str, object]]", payload["cards"])
    envelope = cards[0]
    envelope["schema_version"] = 2
    artifact = cast("dict[str, object]", envelope["artifact"])
    bundle = {
        "bundle_id": f"bundle_{'b' * 52}",
        "root": None,
        "files": [
            {
                "path": "model-Q4_K_M.gguf",
                "size_bytes": 1024,
                "object_id": f"sha256:{'c' * 64}",
            }
        ],
        "download_size": 1024,
        "alternate_locations": [
            {"root": "duplicate", "paths": ["duplicate/model-Q4_K_M.gguf"]}
        ],
    }
    artifact["bundle"] = bundle
    card_payload = cast("dict[str, object]", envelope["card"])
    card_payload["artifact_bundle"] = {
        key: value for key, value in bundle.items() if key != "alternate_locations"
    }

    card = registry_model_cards(
        RegistryCatalog.model_validate(payload, strict=False)
    )[0]

    assert card.artifact_bundle is not None
    assert card.artifact_bundle.bundle_id == f"bundle_{'b' * 52}"
    assert card.artifact_bundle.files[0].path == "model-Q4_K_M.gguf"


def test_registry_v2_rejects_card_envelope_bundle_disagreement() -> None:
    """A signed envelope cannot select different bytes from the runtime card."""

    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    cards = cast("list[dict[str, object]]", payload["cards"])
    envelope = cards[0]
    envelope["schema_version"] = 2
    artifact = cast("dict[str, object]", envelope["artifact"])
    artifact["bundle"] = {
        "bundle_id": f"bundle_{'b' * 52}",
        "root": None,
        "files": [{"path": "model-Q4_K_M.gguf", "size_bytes": 1024}],
        "download_size": 1024,
    }
    card_payload = cast("dict[str, object]", envelope["card"])
    card_payload["artifact_bundle"] = {
        "bundle_id": f"bundle_{'d' * 52}",
        "root": None,
        "files": [{"path": "model-Q4_K_M.gguf", "size_bytes": 1024}],
        "download_size": 1024,
    }
    catalog = RegistryCatalog.model_validate(payload, strict=False)

    with pytest.raises(ValueError, match="bundle disagrees"):
        registry_model_cards(catalog)


def test_signed_support_expands_only_exact_node_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A positive active claim adds a backend without rewriting card identity."""
    card = registry_model_cards(
        RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    )[0]
    support = RegistryEngineSupport.model_validate_json(
        _engine_support_payload(), strict=False
    )
    monkeypatch.setattr(
        model_cards_module, "_registry_engine_support", support.active_claims()
    )

    exact = registry_supported_backends_for_node(
        card,
        node_backends=frozenset({"llama_server", "llama_server-vulkan"}),
        engine_builds={
            "llama_server": "llama.cpp@sha256:" + "1" * 64,
            "llama_server-vulkan": "llama.cpp@sha256:" + "1" * 64,
        },
        hardware_classes=frozenset({"amd"}),
    )
    stale = registry_supported_backends_for_node(
        card,
        node_backends=frozenset({"llama_server-vulkan"}),
        engine_builds={"llama_server-vulkan": "llama.cpp@sha256:" + "9" * 64},
        hardware_classes=frozenset({"amd"}),
    )

    assert exact == frozenset({"llama_server", "llama_server-vulkan"})
    assert stale == frozenset()


def test_runner_process_restores_signed_support_from_verified_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unstamped child fallback recovers support not inherited from its parent."""
    card = registry_model_cards(
        RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    )[0]
    support = RegistryEngineSupport.model_validate_json(
        _engine_support_payload(), strict=False
    )
    monkeypatch.setattr(model_cards_module, "_registry_engine_support", ())
    monkeypatch.setattr(
        model_cards_module._registry_client,
        "load_cached_engine_support",
        lambda: support,
    )

    assert load_cached_registry_engine_support() is True
    assert registry_supported_backends_for_node(
        card,
        node_backends=frozenset({"llama_server-vulkan"}),
        engine_builds={"llama_server-vulkan": "llama.cpp@sha256:" + "1" * 64},
        hardware_classes=frozenset({"amd"}),
    ) == frozenset({"llama_server-vulkan"})


def test_signed_support_requires_positive_complete_artifact_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact qualification needs positive evidence for every artifact capability."""
    card = registry_model_cards(
        RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    )[0]
    support = RegistryEngineSupport.model_validate_json(
        _engine_support_payload(), strict=False
    )
    monkeypatch.setattr(
        model_cards_module, "_registry_engine_support", support.active_claims()
    )
    other_artifact = card.model_copy(update={"registry_card_id": "card_" + "c" * 52})
    incomplete_claims = card.registry_capability_claims + (
        card.registry_capability_claims[0].model_copy(
            update={"scope": "artifact", "status": "incomplete"}
        ),
    )
    incomplete_artifact = card.model_copy(
        update={"registry_capability_claims": incomplete_claims}
    )
    unknown_artifact = card.model_copy(
        update={
            "registry_capability_claims": (
                card.registry_capability_claims[0].model_copy(
                    update={"status": "unknown"}
                ),
            )
        }
    )
    second_capability = card.registry_capability_claims[0].model_copy(
        update={"capability_id": "vision.generate"}
    )
    multi_capability_artifact = card.model_copy(
        update={
            "registry_capability_claims": (
                *card.registry_capability_claims,
                second_capability,
            )
        }
    )

    assert (
        registry_supported_backends_for_node(
            other_artifact,
            node_backends=frozenset({"llama_server-vulkan"}),
            engine_builds={"llama_server-vulkan": "llama.cpp@sha256:" + "1" * 64},
            hardware_classes=frozenset({"amd"}),
        )
        == frozenset()
    )
    assert (
        registry_supported_backends_for_node(
            incomplete_artifact,
            node_backends=frozenset({"llama_server-vulkan"}),
            engine_builds={"llama_server-vulkan": "llama.cpp@sha256:" + "1" * 64},
            hardware_classes=frozenset({"amd"}),
        )
        == frozenset()
    )
    assert (
        registry_supported_backends_for_node(
            unknown_artifact,
            node_backends=frozenset({"llama_server-vulkan"}),
            engine_builds={"llama_server-vulkan": "llama.cpp@sha256:" + "1" * 64},
            hardware_classes=frozenset({"amd"}),
        )
        == frozenset()
    )
    assert (
        registry_supported_backends_for_node(
            multi_capability_artifact,
            node_backends=frozenset({"llama_server-vulkan"}),
            engine_builds={"llama_server-vulkan": "llama.cpp@sha256:" + "1" * 64},
            hardware_classes=frozenset({"amd"}),
        )
        == frozenset()
    )

    complete_support = support.claims[0].model_copy(
        update={
            "claim_id": "support_" + "d" * 52,
            "capability_id": "vision.generate",
            "status": "supported",
        }
    )
    monkeypatch.setattr(
        model_cards_module,
        "_registry_engine_support",
        (*support.active_claims(), complete_support),
    )
    assert registry_supported_backends_for_node(
        multi_capability_artifact,
        node_backends=frozenset({"llama_server-vulkan"}),
        engine_builds={"llama_server-vulkan": "llama.cpp@sha256:" + "1" * 64},
        hardware_classes=frozenset({"amd"}),
    ) == frozenset({"llama_server-vulkan"})


@pytest.mark.asyncio
async def test_registry_refresh_swaps_engine_support_after_verified_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent placement retains the prior support projection during refresh."""
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    refreshed_support = RegistryEngineSupport.model_validate_json(
        _engine_support_payload(), strict=False
    )
    prior_support = refreshed_support.claims[0].model_copy(
        update={"engine_build": "llama.cpp@sha256:" + "0" * 64}
    )
    prior_card = registry_model_cards(catalog)[0]
    load_started = threading.Event()
    allow_load = threading.Event()

    class BlockingClient:
        def load_catalog(self, _catalog_validator: object = None) -> RegistryCatalog:
            load_started.set()
            if not allow_load.wait(timeout=5):
                raise TimeoutError("test did not release catalog load")
            return catalog

        def load_advisories(self) -> RegistryAdvisories:
            return RegistryAdvisories.model_validate_json(
                _advisories_payload(), strict=False
            )

        def load_engine_support(self) -> RegistryEngineSupport:
            return refreshed_support

    original_support = model_cards_module._registry_engine_support
    original_current_cards = dict(model_cards_module._registry_current_cards)
    original_card_cache = dict(model_cards_module._card_cache)
    model_cards_module._registry_engine_support = (prior_support,)
    model_cards_module._registry_current_cards.clear()
    model_cards_module._registry_current_cards[prior_card.model_id] = prior_card
    monkeypatch.setattr(model_cards_module, "_registry_enabled", lambda: True)
    monkeypatch.setattr(model_cards_module, "_registry_client", BlockingClient())
    refresh = asyncio.create_task(model_cards_module._load_cards_from_registry())
    try:
        assert await asyncio.to_thread(load_started.wait, 2)
        assert model_cards_module._registry_engine_support == (prior_support,)
        assert (
            model_cards_module._registry_current_cards[prior_card.model_id]
            == prior_card
        )
        allow_load.set()
        assert await refresh
        assert model_cards_module._registry_engine_support == (
            refreshed_support.active_claims()
        )
    finally:
        allow_load.set()
        if not refresh.done():
            await refresh
        model_cards_module._registry_engine_support = original_support
        model_cards_module._registry_current_cards.clear()
        model_cards_module._registry_current_cards.update(original_current_cards)
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_card_cache)


def test_engine_support_rejects_cross_key_supersession() -> None:
    """A signed target cannot hide one compatibility key with another."""
    payload = cast("dict[str, object]", json.loads(_engine_support_payload()))
    claims = cast("list[dict[str, object]]", payload["claims"])
    claims[1]["architecture"] = "different_architecture"

    with pytest.raises(ValueError, match="supersedes a different key"):
        RegistryEngineSupport.model_validate(payload, strict=False)


def test_engine_support_rejects_supersession_cycles() -> None:
    """A signed replacement history cannot erase every active decision."""
    payload = cast("dict[str, object]", json.loads(_engine_support_payload()))
    claims = cast("list[dict[str, object]]", payload["claims"])
    claims[0]["supersedes_claim_id"] = claims[1]["claim_id"]

    with pytest.raises(ValueError, match="supersession cycle"):
        RegistryEngineSupport.model_validate(payload, strict=False)


@pytest.mark.parametrize("location", ["catalog", "card"])
def test_registry_rejects_unknown_schema_versions(location: str) -> None:
    """A client never interprets a future signed schema with v1 semantics."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    if location == "catalog":
        payload["schema_version"] = 3
    else:
        cards = cast("list[dict[str, object]]", payload["cards"])
        cards[0]["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        RegistryCatalog.model_validate(payload, strict=False)


@pytest.mark.parametrize("alias", [".", "..", "org/..", "../model", "org\\model"])
def test_registry_rejects_path_like_aliases(alias: str) -> None:
    """Signed aliases can never address a staging root or its parent."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    cards = cast("list[dict[str, object]]", payload["cards"])
    cards[0]["alias"] = alias

    with pytest.raises(ValueError, match="safe repository identifier"):
        RegistryCatalog.model_validate(payload, strict=False)


def test_registry_forces_signed_cards_to_non_custom() -> None:
    """Signed payload content cannot acquire local override semantics."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    cards = cast("list[dict[str, object]]", payload["cards"])
    card_payload = cast("dict[str, object]", cards[0]["card"])
    card_payload["is_custom"] = True
    catalog = RegistryCatalog.model_validate(payload, strict=False)

    assert not registry_model_cards(catalog)[0].is_custom


def test_registry_rejects_envelope_card_quantization_mismatch() -> None:
    """Support evidence cannot join a card to a different artifact quantization."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    cards = cast("list[dict[str, object]]", payload["cards"])
    artifact = cast("dict[str, object]", cards[0]["artifact"])
    artifact["quantization"] = "Q8_0"
    catalog = RegistryCatalog.model_validate(payload, strict=False)

    with pytest.raises(ValueError, match="envelope quantization disagrees"):
        registry_model_cards(catalog)


def test_registry_rejects_unpinned_separate_processor_repository() -> None:
    """A signed card cannot approve code that remains mutable upstream."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    cards = cast("list[dict[str, object]]", payload["cards"])
    card_payload = cast("dict[str, object]", cards[0]["card"])
    card_payload["vision"] = {
        "model_type": "test_vlm",
        "processor_repo": "org/processor",
    }
    catalog = RegistryCatalog.model_validate(payload, strict=False)

    with pytest.raises(ValueError, match="processor_revision"):
        registry_model_cards(catalog)


@pytest.mark.parametrize(
    ("section", "expected_field"),
    [
        (
            {"vision": {"model_type": "vlm", "weights_repo": "org/vision"}},
            "vision.weights_repo",
        ),
        (
            {"runtime": {"mtp_heads": True, "mtp_sidecar_repo": "org/mtp"}},
            "runtime.mtp_sidecar_repo",
        ),
        (
            {"runtime": {"assistant_model_repo": "org/assistant"}},
            "runtime.assistant_model_repo",
        ),
        (
            {
                "runtime": {
                    "served_spec_draft_repo": "org/draft",
                    "served_spec_draft_file": "draft.gguf",
                }
            },
            "runtime.served_spec_draft_repo",
        ),
        (
            {
                "runtime": {
                    "vllm_spec_method": "dflash",
                    "vllm_spec_draft_repo": "org/dflash",
                }
            },
            "runtime.vllm_spec_draft_repo",
        ),
    ],
)
def test_registry_rejects_unpinned_separate_companion_repository(
    section: dict[str, object], expected_field: str
) -> None:
    """Every companion source participates in signed artifact identity."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    cards = cast("list[dict[str, object]]", payload["cards"])
    card_payload = cast("dict[str, object]", cards[0]["card"])
    card_payload.update(section)
    catalog = RegistryCatalog.model_validate(payload, strict=False)

    with pytest.raises(ValueError, match=expected_field):
        registry_model_cards(catalog)


def test_registry_rejects_catalog_metadata_outside_card_identity() -> None:
    """Every published card has exactly one signed provenance record."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    payload["card_metadata"] = {}
    catalog = RegistryCatalog.model_validate(payload, strict=False)

    with pytest.raises(ValueError, match="metadata does not match"):
        registry_model_cards(catalog)


def test_offline_mode_disables_registry_network_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Air-gapped nodes use bundled cards without contacting public TUF."""
    monkeypatch.delenv("SKULK_TESTS", raising=False)
    monkeypatch.setattr(model_cards_module, "SKULK_MODEL_REGISTRY_ENABLED", True)
    monkeypatch.setattr(model_cards_module, "SKULK_OFFLINE", True)

    assert not model_cards_module._registry_enabled()


@pytest.mark.asyncio
async def test_air_gap_restart_loads_installed_card_without_registry_lkg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete installed bytes remain usable after registry cache expiry."""
    card = registry_model_cards(
        RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    )[0]
    artifact = tmp_path / card.model_id.normalize()
    artifact.mkdir()
    (artifact / "model-Q4_K_M.gguf").write_bytes(b"weights")
    (artifact / ".skulk-source-revision").write_text(f"{card.source_revision}\n")
    write_installed_card(
        artifact,
        build_installed_card_record(artifact, card),
    )

    async def _registry_unavailable() -> bool:
        return False

    async def _no_cards(_path: object, *, is_custom: bool) -> None:
        del is_custom

    original_cache = dict(model_cards_module._card_cache)
    original_installed = dict(model_cards_module._installed_card_cache)
    model_cards_module._card_cache.clear()
    monkeypatch.setattr(constants_module, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(constants_module, "SKULK_MODELS_PATH", None)
    monkeypatch.setattr(
        model_cards_module, "_load_cards_from_registry", _registry_unavailable
    )
    monkeypatch.setattr(model_cards_module, "_load_cards_from_dir", _no_cards)
    try:
        await model_cards_module._refresh_card_cache()
        assert model_cards_module.get_card(card.model_id) == card
        installed = model_cards_module.get_installed_card_record(card.model_id)
        assert installed is not None
        assert installed.installed_identity == card.registry_card_id
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)
        model_cards_module._installed_card_cache.clear()
        model_cards_module._installed_card_cache.update(original_installed)


def test_completed_stage_converges_installed_cache_without_registry_refresh(
    tmp_path: Path,
) -> None:
    """Offline model metadata changes immediately after durable staging."""

    card = registry_model_cards(
        RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    )[0]
    artifact = tmp_path / card.model_id.normalize()
    artifact.mkdir()
    (artifact / "model-Q4_K_M.gguf").write_bytes(b"weights")
    (artifact / ".skulk-source-revision").write_text(f"{card.source_revision}\n")
    record = build_installed_card_record(artifact, card)

    original_cache = dict(model_cards_module._card_cache)
    original_installed = dict(model_cards_module._installed_card_cache)
    original_installed_current = dict(
        model_cards_module._installed_current_registry_ids
    )
    model_cards_module._card_cache.clear()
    model_cards_module._installed_card_cache.clear()
    model_cards_module._installed_current_registry_ids.clear()
    try:
        model_cards_module.register_installed_card_record(record)

        assert model_cards_module.get_installed_card_record(card.model_id) == record
        assert model_cards_module.get_card(card.model_id) == card
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)
        model_cards_module._installed_card_cache.clear()
        model_cards_module._installed_card_cache.update(original_installed)
        model_cards_module._installed_current_registry_ids.clear()
        model_cards_module._installed_current_registry_ids.update(
            original_installed_current
        )


@pytest.mark.asyncio
async def test_installed_snapshot_preserves_registration_after_scan_started(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale background scan cannot erase a newly completed installation."""

    card = registry_model_cards(
        RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    )[0]
    artifact = tmp_path / card.model_id.normalize()
    artifact.mkdir()
    (artifact / "model-Q4_K_M.gguf").write_bytes(b"weights")
    record = build_installed_card_record(artifact, card)
    original_cache = dict(model_cards_module._card_cache)
    original_installed = dict(model_cards_module._installed_card_cache)
    original_installed_current = dict(
        model_cards_module._installed_current_registry_ids
    )
    scan_started = asyncio.Event()
    release_scan = threading.Event()
    loop = asyncio.get_running_loop()

    def stale_scan() -> list[InstalledCardRecord]:
        loop.call_soon_threadsafe(scan_started.set)
        release_scan.wait(timeout=5)
        return []

    monkeypatch.setattr(model_cards_module, "_discover_installed_cards", stale_scan)
    model_cards_module._card_cache.clear()
    model_cards_module._installed_card_cache.clear()
    model_cards_module._installed_current_registry_ids.clear()
    try:
        refresh = asyncio.create_task(model_cards_module._refresh_installed_cards())
        await scan_started.wait()
        model_cards_module.register_installed_card_record(record)
        release_scan.set()
        await refresh

        assert model_cards_module.get_installed_card_record(card.model_id) == record
        assert model_cards_module.get_card(card.model_id) == card
    finally:
        release_scan.set()
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)
        model_cards_module._installed_card_cache.clear()
        model_cards_module._installed_card_cache.update(original_installed)
        model_cards_module._installed_current_registry_ids.clear()
        model_cards_module._installed_current_registry_ids.update(
            original_installed_current
        )


def test_artifact_eviction_unregisters_installed_generation(tmp_path: Path) -> None:
    """Deleted local bytes stop being active installed truth immediately."""

    card = registry_model_cards(
        RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    )[0]
    artifact = tmp_path / card.model_id.normalize()
    artifact.mkdir()
    (artifact / "model-Q4_K_M.gguf").write_bytes(b"weights")
    record = build_installed_card_record(artifact, card)

    original_cache = dict(model_cards_module._card_cache)
    original_installed = dict(model_cards_module._installed_card_cache)
    original_installed_current = dict(
        model_cards_module._installed_current_registry_ids
    )
    original_dirty = model_cards_module._card_cache_dirty
    model_cards_module._card_cache.clear()
    model_cards_module._installed_card_cache.clear()
    model_cards_module._installed_current_registry_ids.clear()
    try:
        model_cards_module.register_installed_card_record(record)
        model_cards_module.unregister_installed_card_record(card.model_id)

        assert model_cards_module.get_installed_card_record(card.model_id) is None
        assert model_cards_module.get_card(card.model_id) is None
        assert model_cards_module._card_cache_dirty
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)
        model_cards_module._installed_card_cache.clear()
        model_cards_module._installed_card_cache.update(original_installed)
        model_cards_module._installed_current_registry_ids.clear()
        model_cards_module._installed_current_registry_ids.update(
            original_installed_current
        )
        model_cards_module._card_cache_dirty = original_dirty


def test_client_uses_hash_bound_last_known_good(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified catalog survives an outage but not local byte tampering."""
    payload_path = tmp_path / "downloaded.json"
    payload_path.write_bytes(_catalog_payload())
    embedded_root = tmp_path / "embedded-root.json"
    embedded_root.write_text("{}")
    cached_root = tmp_path / "cache/metadata/root.json"
    cached_root.parent.mkdir(parents=True)
    cached_root.write_text('{"attacker":"preseeded"}')
    observed_trusted_roots: list[bytes] = []

    class WorkingUpdater:
        def __init__(self, **kwargs: object) -> None:
            observed_trusted_roots.append(cast("bytes", kwargs["bootstrap"]))

        def refresh(self) -> None:
            pass

        def get_targetinfo(self, target_path: str) -> object | None:
            return object() if target_path == "v1/catalog.json" else None

        def download_target(self, _: object) -> str:
            return str(payload_path)

    monkeypatch.setattr(registry_module, "Updater", WorkingUpdater)
    client = TufRegistryClient(
        base_url="https://registry.example/",
        cache_dir=tmp_path / "cache",
        embedded_root=embedded_root,
        timeout_seconds=1,
        max_stale_days=30,
    )
    assert client.load_catalog(registry_model_cards).snapshot_id == "snapshot_1_test"
    assert observed_trusted_roots == [embedded_root.read_bytes()]

    malformed_payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    malformed_cards = cast("list[dict[str, object]]", malformed_payload["cards"])
    malformed_card = cast("dict[str, object]", malformed_cards[0]["card"])
    malformed_card["n_layers"] = "not-an-integer"
    payload_path.write_text(json.dumps(malformed_payload))
    assert client.load_catalog(registry_model_cards).snapshot_id == "snapshot_1_test"
    assert (tmp_path / "cache/last-known-good-catalog.json").read_bytes() == (
        _catalog_payload()
    )

    class FailingUpdater(WorkingUpdater):
        def refresh(self) -> None:
            raise OSError("offline")

    monkeypatch.setattr(registry_module, "Updater", FailingUpdater)
    assert client.load_catalog(registry_model_cards).snapshot_id == "snapshot_1_test"

    (tmp_path / "cache/last-known-good-catalog.json").write_bytes(b"tampered")
    with pytest.raises(RegistryUnavailableError):
        client.load_catalog()


def test_client_uses_hash_bound_last_known_good_advisories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verified warnings survive restart outages but not cache tampering."""

    payload_path = tmp_path / "advisories.json"
    payload_path.write_bytes(_advisories_payload())
    embedded_root = tmp_path / "embedded-root.json"
    embedded_root.write_text("{}")

    class WorkingUpdater:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def refresh(self) -> None:
            pass

        def get_targetinfo(self, target_path: str) -> object:
            assert target_path == "v1/advisories.json"
            return object()

        def download_target(self, _target: object) -> str:
            return str(payload_path)

    monkeypatch.setattr(registry_module, "Updater", WorkingUpdater)
    client = TufRegistryClient(
        base_url="https://registry.example/",
        cache_dir=tmp_path / "cache",
        embedded_root=embedded_root,
        timeout_seconds=1,
        max_stale_days=30,
    )
    assert client.load_advisories().advisories[0].advisory_id == "FLA-2026-0001"

    class FailingUpdater(WorkingUpdater):
        def refresh(self) -> None:
            raise OSError("offline")

    monkeypatch.setattr(registry_module, "Updater", FailingUpdater)
    assert client.load_advisories().advisories[0].severity == "critical"

    (tmp_path / "cache/last-known-good-advisories.json").write_bytes(b"tampered")
    with pytest.raises(RegistryUnavailableError):
        client.load_advisories()


def test_client_retains_hash_bound_engine_support_for_offline_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verified support survives offline indefinitely but cache tampering fails."""
    payload_path = tmp_path / "engine-support.json"
    payload_path.write_bytes(_engine_support_payload())
    embedded_root = tmp_path / "embedded-root.json"
    embedded_root.write_text("{}")

    class WorkingUpdater:
        def __init__(self, **kwargs: object) -> None:
            self.metadata_dir = Path(cast("str", kwargs["metadata_dir"]))

        def refresh(self) -> None:
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            (self.metadata_dir / "targets.json").write_text(
                json.dumps(
                    {
                        "signatures": [],
                        "signed": {
                            "_type": "targets",
                            "spec_version": "1.0.31",
                            "version": 7,
                            "expires": "2030-01-01T00:00:00Z",
                            "targets": {},
                        },
                    }
                )
            )

        def get_targetinfo(self, target_path: str) -> object:
            assert target_path == "v1/engine-support.json"
            return object()

        def download_target(self, _target: object) -> str:
            return str(payload_path)

    monkeypatch.setattr(registry_module, "Updater", WorkingUpdater)
    client = TufRegistryClient(
        base_url="https://registry.example/",
        cache_dir=tmp_path / "cache",
        embedded_root=embedded_root,
        timeout_seconds=1,
        max_stale_days=0,
    )
    assert client.load_engine_support().matrix_version == 7
    assert client.load_cached_engine_support().active_claims()[0].status == "supported"

    (tmp_path / "cache/last-known-good-engine-support.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        client.load_cached_engine_support()


def test_embedded_roots_match_release_resources() -> None:
    """Package and frozen-app trust anchors cannot drift independently."""
    package_root = registry_module.EMBEDDED_REGISTRY_ROOT.read_bytes()
    release_root = (
        Path(__file__).parents[5] / "resources/model_registry/root.json"
    ).read_bytes()
    assert package_root == release_root


@pytest.mark.asyncio
async def test_failed_refresh_removes_previous_registry_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired in-memory catalog cannot outlive the configured LKG bound."""
    registry_card = registry_model_cards(
        RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    )[0]
    bundled_card = registry_card.model_copy(
        update={
            "model_id": ModelId("org/bundled"),
            "source_repository": None,
            "registry_card_id": None,
            "registry_snapshot_id": None,
        }
    )

    class FailingClient:
        def load_catalog(self, _catalog_validator: object = None) -> RegistryCatalog:
            raise OSError("registry offline and LKG expired")

    original_cache = dict(model_cards_module._card_cache)
    model_cards_module._card_cache.clear()
    model_cards_module._card_cache[registry_card.model_id] = registry_card
    model_cards_module._card_cache[bundled_card.model_id] = bundled_card
    monkeypatch.setattr(model_cards_module, "_registry_enabled", lambda: True)
    monkeypatch.setattr(model_cards_module, "_registry_client", FailingClient())
    try:
        await model_cards_module._load_cards_from_registry()
        assert registry_card.model_id not in model_cards_module._card_cache
        assert model_cards_module._card_cache[bundled_card.model_id] == bundled_card
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)


@pytest.mark.asyncio
async def test_successful_refresh_excludes_unlisted_bundled_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signed snapshot can revoke a card that the distribution once bundled."""
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    registry_card = registry_model_cards(catalog)[0]
    bundled_card = registry_card.model_copy(
        update={
            "model_id": ModelId("org/revoked-bundled"),
            "source_repository": None,
            "registry_card_id": None,
            "registry_snapshot_id": None,
            "registry_provenance": None,
        }
    )
    custom_card = bundled_card.model_copy(
        update={"model_id": ModelId("org/custom"), "is_custom": True}
    )

    class WorkingClient:
        def load_catalog(self, _catalog_validator: object = None) -> RegistryCatalog:
            return catalog

    original_cache = dict(model_cards_module._card_cache)
    model_cards_module._card_cache.clear()
    model_cards_module._card_cache[bundled_card.model_id] = bundled_card
    model_cards_module._card_cache[custom_card.model_id] = custom_card
    monkeypatch.setattr(model_cards_module, "_registry_enabled", lambda: True)
    monkeypatch.setattr(model_cards_module, "_registry_client", WorkingClient())
    try:
        assert await model_cards_module._load_cards_from_registry()
        assert bundled_card.model_id not in model_cards_module._card_cache
        assert model_cards_module._card_cache[registry_card.model_id] == registry_card
        assert model_cards_module._card_cache[custom_card.model_id] == custom_card
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)


@pytest.mark.asyncio
async def test_current_custom_file_supersedes_retained_custom_sidecar(
    tmp_path: Path,
) -> None:
    """Operator edits remain authoritative after installed-card recovery."""

    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    base = registry_model_cards(catalog)[0].model_copy(
        update={
            "registry_card_id": None,
            "registry_snapshot_id": None,
            "registry_provenance": None,
            "is_custom": True,
        }
    )
    retained = base.model_copy(update={"hidden_size": 64})
    edited = base.model_copy(update={"hidden_size": 128})
    custom_directory = AsyncPath(tmp_path)
    await edited.save(custom_directory / "edited.toml")

    original_cache = dict(model_cards_module._card_cache)
    model_cards_module._card_cache.clear()
    model_cards_module._card_cache[retained.model_id] = retained
    try:
        await model_cards_module._load_cards_from_dir(
            custom_directory,
            is_custom=True,
        )

        selected = model_cards_module._card_cache[retained.model_id]
        assert selected.hidden_size == 128
        assert selected.is_custom
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)


@pytest.mark.asyncio
async def test_complete_catalog_is_available_without_image_ui_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store authority can resolve signed image cards on a non-image host."""
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    text_card = registry_model_cards(catalog)[0]
    image_card = text_card.model_copy(
        update={
            "model_id": ModelId("org/image"),
            "tasks": [model_cards_module.ModelTask.TextToImage],
        }
    )
    original_cache = dict(model_cards_module._card_cache)
    model_cards_module._card_cache.clear()
    model_cards_module._card_cache[text_card.model_id] = text_card
    model_cards_module._card_cache[image_card.model_id] = image_card
    monkeypatch.setattr(model_cards_module, "SKULK_ENABLE_IMAGE_MODELS", False)
    monkeypatch.setattr(model_cards_module, "_last_registry_refresh", 100.0)
    monkeypatch.setattr(model_cards_module.time, "monotonic", lambda: 101.0)
    try:
        assert await model_cards_module.get_all_model_cards() == [
            text_card,
            image_card,
        ]
        assert await model_cards_module.get_model_cards() == [text_card]
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)


@pytest.mark.asyncio
async def test_registry_refresh_helper_throttles_repeated_cache_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown model requests cannot refresh TUF more than once per interval."""
    refreshes = 0

    async def refresh() -> None:
        nonlocal refreshes
        refreshes += 1
        monkeypatch.setattr(model_cards_module, "_last_registry_refresh", 100.0)

    original_cache = dict(model_cards_module._card_cache)
    model_cards_module._card_cache.clear()
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    card = registry_model_cards(catalog)[0]
    model_cards_module._card_cache[card.model_id] = card
    monkeypatch.setattr(model_cards_module, "_registry_enabled", lambda: True)
    monkeypatch.setattr(model_cards_module, "_last_registry_refresh", 0.0)
    monkeypatch.setattr(model_cards_module.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(model_cards_module, "_refresh_card_cache", refresh)
    try:
        await model_cards_module._refresh_card_cache_if_due()
        await model_cards_module._refresh_card_cache_if_due()
        assert refreshes == 1
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)


@pytest.mark.asyncio
async def test_known_card_loads_check_registry_refresh_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known aliases still observe the throttled signed-registry deadline."""
    refresh = AsyncMock()
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    card = registry_model_cards(catalog)[0]

    monkeypatch.setitem(model_cards_module._card_cache, card.model_id, card)
    monkeypatch.setattr(
        model_cards_module,
        "_refresh_card_cache_if_due",
        refresh,
    )

    assert await ModelCard.load(card.model_id) == card
    assert await ModelCard.load_or_fetch_from_hf(card.model_id) == card
    assert refresh.await_count == 2


@pytest.mark.asyncio
async def test_registry_id_miss_forces_one_serialized_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store can bridge snapshot skew without allowing refresh storms."""
    refreshes = 0
    requested_id = f"card_{'z' * 52}"

    async def refresh() -> None:
        nonlocal refreshes
        refreshes += 1

    original_cache = dict(model_cards_module._card_cache)
    model_cards_module._card_cache.clear()
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    model_cards_module._card_cache.update(
        {card.model_id: card for card in registry_model_cards(catalog)}
    )
    monkeypatch.setattr(model_cards_module, "_registry_enabled", lambda: True)
    monkeypatch.setattr(model_cards_module, "_last_registry_refresh", 100.0)
    monkeypatch.setattr(model_cards_module, "_last_registry_miss_refresh", 0.0)
    monkeypatch.setattr(model_cards_module.time, "monotonic", lambda: 101.0)
    monkeypatch.setattr(model_cards_module, "_refresh_card_cache", refresh)
    try:
        assert (
            await model_cards_module.get_registry_card_by_id(
                requested_id,
                refresh_on_miss=True,
            )
            is None
        )
        assert (
            await model_cards_module.get_registry_card_by_id(
                requested_id,
                refresh_on_miss=True,
            )
            is None
        )
        assert refreshes == 1
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)


@pytest.mark.asyncio
async def test_current_registry_id_is_visible_behind_installed_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store authorization can resolve an update hidden by installed alias truth."""

    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    current = registry_model_cards(catalog)[0]
    installed = current.model_copy(update={"registry_card_id": f"card_{'z' * 52}"})
    original_cache = dict(model_cards_module._card_cache)
    original_current = dict(model_cards_module._registry_current_cards)
    model_cards_module._card_cache.clear()
    model_cards_module._registry_current_cards.clear()
    model_cards_module._card_cache[current.model_id] = installed
    model_cards_module._registry_current_cards[current.model_id] = current
    monkeypatch.setattr(model_cards_module, "_registry_enabled", lambda: False)
    try:
        assert (
            await model_cards_module.get_registry_card_by_id(
                str(current.registry_card_id)
            )
            == current
        )
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)
        model_cards_module._registry_current_cards.clear()
        model_cards_module._registry_current_cards.update(original_current)


async def test_installed_startup_selects_current_generation_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directory order cannot replace current installed truth with stale bytes."""

    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    current = registry_model_cards(catalog)[0]
    stale = current.model_copy(update={"registry_card_id": f"card_{'z' * 52}"})
    for directory_name, card in (("aaa-current", current), ("zzz-stale", stale)):
        artifact = tmp_path / directory_name
        artifact.mkdir()
        (artifact / "model-Q4_K_M.gguf").write_bytes(directory_name.encode())
        (artifact / ".skulk-source-revision").write_text(f"{card.source_revision}\n")
        write_installed_card(artifact, build_installed_card_record(artifact, card))

    original_cache = dict(model_cards_module._card_cache)
    original_installed = dict(model_cards_module._installed_card_cache)
    original_installed_current = dict(
        model_cards_module._installed_current_registry_ids
    )
    original_current = dict(model_cards_module._registry_current_cards)
    model_cards_module._card_cache.clear()
    model_cards_module._installed_card_cache.clear()
    model_cards_module._installed_current_registry_ids.clear()
    model_cards_module._registry_current_cards.clear()
    model_cards_module._registry_current_cards[current.model_id] = current
    monkeypatch.setattr(constants_module, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(constants_module, "SKULK_MODELS_PATH", None)
    try:
        await model_cards_module._refresh_installed_cards()

        assert model_cards_module.get_card(current.model_id) == current
        installed = model_cards_module.get_installed_card_record(current.model_id)
        assert installed is not None
        assert installed.installed_identity == current.registry_card_id
        assert (
            model_cards_module.get_current_registry_card_id(current.model_id)
            == current.registry_card_id
        )
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)
        model_cards_module._installed_card_cache.clear()
        model_cards_module._installed_card_cache.update(original_installed)
        model_cards_module._installed_current_registry_ids.clear()
        model_cards_module._installed_current_registry_ids.update(
            original_installed_current
        )
        model_cards_module._registry_current_cards.clear()
        model_cards_module._registry_current_cards.update(original_current)


def test_installed_custom_card_does_not_restore_deleted_catalog_authority(
    tmp_path: Path,
) -> None:
    """A retained custom sidecar cannot resurrect its deleted catalog card."""
    card = ModelCard(
        model_id=ModelId("org/custom-installed"),
        source_revision="b" * 40,
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        is_custom=True,
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.safetensors").write_bytes(b"weights")
    record = build_installed_card_record(artifact, card)
    original_cache = dict(model_cards_module._card_cache)
    original_installed = dict(model_cards_module._installed_card_cache)
    original_current = dict(model_cards_module._installed_current_registry_ids)
    model_cards_module._card_cache.clear()
    model_cards_module._installed_card_cache.clear()
    model_cards_module._installed_current_registry_ids.clear()
    model_cards_module._card_cache[card.model_id] = card
    try:
        model_cards_module._apply_installed_card_snapshot([record], 0)

        assert card.model_id not in model_cards_module._card_cache
        assert model_cards_module.get_installed_card_record(card.model_id) == record
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)
        model_cards_module._installed_card_cache.clear()
        model_cards_module._installed_card_cache.update(original_installed)
        model_cards_module._installed_current_registry_ids.clear()
        model_cards_module._installed_current_registry_ids.update(original_current)


def test_late_custom_install_registration_cannot_restore_deleted_card(
    tmp_path: Path,
) -> None:
    """A late staging callback cannot outlive custom-card deletion."""
    card = ModelCard(
        model_id=ModelId("org/custom-late-install"),
        source_revision="b" * 40,
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        is_custom=True,
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.safetensors").write_bytes(b"weights")
    record = build_installed_card_record(artifact, card)
    original_cache = dict(model_cards_module._card_cache)
    original_installed = dict(model_cards_module._installed_card_cache)
    original_current = dict(model_cards_module._installed_current_registry_ids)
    original_mutations = dict(model_cards_module._installed_card_mutation_versions)
    original_version = cast(
        "int", model_cards_module._installed_card_cache_version
    )
    model_cards_module._card_cache.clear()
    model_cards_module._installed_card_cache.clear()
    model_cards_module._installed_current_registry_ids.clear()
    model_cards_module._installed_card_mutation_versions.clear()
    try:
        model_cards_module._card_cache[card.model_id] = card
        model_cards_module._card_cache.pop(card.model_id)

        model_cards_module.register_installed_card_record(record)

        assert model_cards_module.get_card(card.model_id) is None
        assert model_cards_module.get_installed_card_record(card.model_id) == record
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)
        model_cards_module._installed_card_cache.clear()
        model_cards_module._installed_card_cache.update(original_installed)
        model_cards_module._installed_current_registry_ids.clear()
        model_cards_module._installed_current_registry_ids.update(original_current)
        model_cards_module._installed_card_mutation_versions.clear()
        model_cards_module._installed_card_mutation_versions.update(original_mutations)
        model_cards_module._installed_card_cache_version = original_version


@pytest.mark.asyncio
async def test_downloader_fetches_source_repository_under_alias_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Network origin uses artifact truth while local state keeps the alias."""
    observed: list[ModelId] = []

    async def fake_models_dir() -> Path:
        return tmp_path

    async def fake_file_list(
        model_id: ModelId, *_: object, **__: object
    ) -> list[object]:
        observed.append(model_id)
        return []

    async def ignore_progress(*_: object) -> None:
        pass

    monkeypatch.setattr(download_utils, "ensure_models_dir", fake_models_dir)
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", fake_file_list)
    card = ModelCard(
        model_id=ModelId("org/multi@q4"),
        source_repository=ModelId("org/multi"),
        source_revision="b" * 40,
        storage_size=Memory.from_bytes(1024),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=4,
        n_layers=4,
    )

    path, _ = await download_utils.download_shard(
        shard,
        ignore_progress,
        skip_download=True,
        skip_internet=True,
        allow_patterns=["config.json"],
    )

    assert observed == [ModelId("org/multi")]
    assert path.name == "org--multi@q4"


def _gguf_metadata_payload() -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "snapshot_id": "snapshot_1_test",
            "target_version": 7,
            "generated_at": "2026-09-08T00:00:00Z",
            "artifacts": {
                f"card_{'a' * 52}": {
                    "repository": "org/multi-gguf",
                    "revision": "b" * 40,
                    "selected_file": "model-Q4_K_M.gguf",
                    "header": {
                        "architecture": "qwen35",
                        "scalars": {
                            "block_count": 33,
                            "embedding_length": 4096,
                            "attention.head_count_kv": 4,
                            "attention.key_length": 256,
                            "attention.value_length": 256,
                            "ssm.conv_kernel": 4,
                            "ssm.inner_size": 4096,
                            "ssm.state_size": 128,
                            "ssm.group_count": 16,
                            "nextn_predict_layers": 1,
                        },
                        "has_recurrent_layer_override": False,
                        "metadata_sha256": "c" * 64,
                        "metadata_bytes": 10943341,
                    },
                }
            },
        }
    ).encode()


def test_signed_header_projects_geometry_without_changing_canonical_card() -> None:
    """Auxiliary artifact facts correct runtime dimensions and bind approvals."""
    original = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    metadata = RegistryGgufMetadata.model_validate_json(_gguf_metadata_payload())
    catalog = original.with_gguf_metadata(metadata)
    assert original.gguf_metadata is None
    assert catalog.model_dump_json() == original.model_dump_json()
    card = registry_model_cards(catalog)[0]
    assert card.registry_card_id == original.cards[0].card_id
    assert original.cards[0].card["n_layers"] == 4
    assert card.n_layers == 33
    assert card.hidden_size == 4096
    assert card.gguf_cache_geometry is not None
    assert (
        card.gguf_cache_geometry.recurrent_bytes(parallel_slots=16, rollback_depth=3)
        == 3372220416
    )
    assert card.registry_gguf_metadata is not None
    restored = ModelCard.model_validate_json(card.model_dump_json())
    assert restored == card
    assert model_cards_module.authorized_model_card_digest(
        card
    ) != model_cards_module.authorized_model_card_digest(
        registry_model_cards(original)[0]
    )


@pytest.mark.parametrize("field", ["repository", "revision", "selected_file"])
def test_header_metadata_rejects_cross_artifact_binding(field: str) -> None:
    """A same-name artifact or copied card ID cannot supply another file's geometry."""
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    metadata = RegistryGgufMetadata.model_validate_json(_gguf_metadata_payload())
    key = catalog.cards[0].card_id
    changed = metadata.artifacts[key].model_copy(
        update={field: "d" * 40 if field == "revision" else "other/file"}
    )
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        catalog.with_gguf_metadata(
            metadata.model_copy(update={"artifacts": {key: changed}})
        )


def test_header_metadata_rejects_cross_snapshot_and_numeric_coercion() -> None:
    """Even signed metadata must match the exact snapshot and strict scalar types."""
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    metadata = RegistryGgufMetadata.model_validate_json(_gguf_metadata_payload())
    with pytest.raises(ValueError, match="snapshot mismatch"):
        catalog.with_gguf_metadata(metadata.model_copy(update={"snapshot_id": "other"}))
    payload = _gguf_metadata_payload().replace(
        b'"block_count": 33', b'"block_count": true'
    )
    with pytest.raises(ValueError):
        RegistryGgufMetadata.model_validate_json(payload)


def test_client_recovers_snapshot_bound_header_cache_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network loss retains verified dimensions, while bad auxiliary bytes fail closed."""
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_bytes(_catalog_payload())
    header_path = tmp_path / "gguf-metadata.json"
    header_path.write_bytes(_gguf_metadata_payload())
    root = tmp_path / "root.json"
    root.write_text("{}")
    offline = False

    class MetadataUpdater:
        def __init__(self, **kwargs: object) -> None:
            self.metadata_dir = Path(cast("str", kwargs["metadata_dir"]))

        def refresh(self) -> None:
            if offline:
                raise OSError("offline")
            (self.metadata_dir / "targets.json").write_text('{"signed":{"version":7}}')

        def get_targetinfo(self, path: str) -> str:
            return str(header_path if path == "v1/gguf-metadata.json" else catalog_path)

        def download_target(self, target: str) -> str:
            return target

    monkeypatch.setattr(registry_module, "Updater", MetadataUpdater)
    client = TufRegistryClient(
        base_url="https://registry.example/",
        cache_dir=tmp_path / "cache",
        embedded_root=root,
        timeout_seconds=1,
        max_stale_days=30,
    )
    first = client.load_catalog(registry_model_cards)
    assert first.gguf_metadata is not None
    # A rejected newer target cannot displace previously verified facts.
    header_path.write_bytes(
        _gguf_metadata_payload().replace(b'"target_version": 7', b'"target_version": 8')
    )
    assert (
        client.load_catalog(registry_model_cards).gguf_metadata == first.gguf_metadata
    )
    offline = True
    recovered = client.load_catalog(registry_model_cards)
    assert registry_model_cards(recovered)[0] == registry_model_cards(first)[0]
    (tmp_path / "cache/last-known-good-gguf-metadata.json").write_text("tampered")
    with pytest.raises(RegistryUnavailableError):
        client.load_catalog(registry_model_cards)


def test_installed_same_card_keeps_current_signed_geometry(tmp_path: Path) -> None:
    """An older installed sidecar cannot hide newly verified same-artifact metadata."""
    original = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    metadata = RegistryGgufMetadata.model_validate_json(_gguf_metadata_payload())
    old_card = registry_model_cards(original)[0]
    current = registry_model_cards(original.with_gguf_metadata(metadata))[0]
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"weights")
    (tmp_path / ".skulk-source-revision").write_text(f"{old_card.source_revision}\n")
    record = build_installed_card_record(tmp_path, old_card)
    assert record.verification == "registry_verified"
    prior_cache = dict(model_cards_module._card_cache)
    prior_current = dict(model_cards_module._registry_current_cards)
    prior_installed = dict(model_cards_module._installed_card_cache)
    prior_current_ids = dict(model_cards_module._installed_current_registry_ids)
    try:
        model_cards_module._card_cache[current.model_id] = current
        model_cards_module._registry_current_cards[current.model_id] = current
        model_cards_module._apply_installed_card_snapshot([record], scan_version=10**9)
        assert model_cards_module._card_cache[current.model_id] == current
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(prior_cache)
        model_cards_module._registry_current_cards.clear()
        model_cards_module._registry_current_cards.update(prior_current)
        model_cards_module._installed_card_cache.clear()
        model_cards_module._installed_card_cache.update(prior_installed)
        model_cards_module._installed_current_registry_ids.clear()
        model_cards_module._installed_current_registry_ids.update(prior_current_ids)
