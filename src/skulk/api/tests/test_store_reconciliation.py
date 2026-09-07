# pyright: reportPrivateUsage=false
"""Generation-selection invariants for cache-to-store reconciliation."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import skulk.api.main as api_main
import skulk.store.artifact_inventory as artifact_inventory
from skulk.api.main import (
    _artifact_inventory_is_tombstoned,
    _combined_model_advisories,
    _complete_canonical_generations,
    _installed_artifact_roots,
    _inventory_installed_artifacts,
    _select_reconciliation_generations,
)
from skulk.api.types.api import ReconciliationStatus
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.models.registry import RegistryAdvisory
from skulk.shared.types.memory import Memory
from skulk.store.config import ModelStoreConfig, SkulkConfig
from skulk.store.installed_cards import (
    InstalledCardRecord,
    build_installed_card_record,
    write_installed_card,
)
from skulk.store.model_store_client import ModelStoreClient


def _installed_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("org/model"),
        storage_size=Memory.from_mb(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        source_revision="a" * 40,
        registry_card_id=f"card_{'a' * 52}",
        registry_snapshot_id="snapshot_1_test",
        registry_provenance="foxlight",
    )


def _write_artifact(
    root: Path, directory_name: str = "org--model"
) -> tuple[Path, InstalledCardRecord]:
    artifact = root / directory_name
    artifact.mkdir(parents=True)
    (artifact / "model.safetensors").write_bytes(b"weights")
    (artifact / ".skulk-source-revision").write_text(f"{'a' * 40}\n")
    record = build_installed_card_record(artifact, _installed_card())
    write_installed_card(artifact, record)
    return artifact, record


def _replica(
    *,
    model_id: str,
    verification: str,
    registry_card_id: str | None = None,
    owner_model_id: str | None = None,
    owner_card_id: str | None = None,
    artifact_role: str = "base",
) -> list[tuple[str, str, dict[str, object]]]:
    return [
        (
            "node-a",
            "http://node-a.invalid",
            {
                "modelId": model_id,
                "verificationState": verification,
                "registryCardId": registry_card_id,
                "ownerModelId": owner_model_id,
                "ownerCardId": owner_card_id,
                "artifactRole": artifact_role,
            },
        )
    ]


def test_reconciliation_selects_one_current_generation_per_artifact() -> None:
    current_id = "card_current"
    replicas = {
        ("aaa_stale", "1" * 64): _replica(
            model_id="org/base",
            verification="registry_verified",
            registry_card_id="card_stale",
        ),
        ("zzz_current", "2" * 64): _replica(
            model_id="org/base",
            verification="local_legacy",
            registry_card_id=current_id,
        ),
        ("aaa_old_companion", "3" * 64): _replica(
            model_id="org/sidecar",
            verification="registry_verified",
            owner_model_id="org/base",
            owner_card_id="card_stale",
            artifact_role="mtp_sidecar",
        ),
        ("zzz_current_companion", "4" * 64): _replica(
            model_id="org/sidecar",
            verification="local_legacy",
            owner_model_id="org/base",
            owner_card_id=current_id,
            artifact_role="mtp_sidecar",
        ),
    }

    selected = _select_reconciliation_generations(
        replicas,
        {"org/base": current_id},
    )

    assert set(selected) == {
        ("zzz_current", "2" * 64),
        ("zzz_current_companion", "4" * 64),
    }


def test_reconciliation_prefers_verified_generation_without_current_card() -> None:
    replicas = {
        ("aaa_local", "1" * 64): _replica(
            model_id="org/base",
            verification="local_legacy",
        ),
        ("zzz_verified", "2" * 64): _replica(
            model_id="org/base",
            verification="registry_verified",
        ),
    }

    selected = _select_reconciliation_generations(replicas, {})

    assert set(selected) == {("zzz_verified", "2" * 64)}


def test_reconciliation_suppresses_deleted_alias_and_owned_companions() -> None:
    """Stale node replicas cannot resurrect an operator-deleted model."""

    tombstones = frozenset({"org/base", "org/independent"})

    assert _artifact_inventory_is_tombstoned(
        {"modelId": "org/base", "artifactRole": "base"},
        tombstones,
    )
    assert _artifact_inventory_is_tombstoned(
        {
            "modelId": "org/sidecar",
            "ownerModelId": "org/base",
            "artifactRole": "mtp_sidecar",
        },
        tombstones,
    )
    assert _artifact_inventory_is_tombstoned(
        {"modelId": "org/independent", "artifactRole": "base"},
        tombstones,
    )
    assert not _artifact_inventory_is_tombstoned(
        {"modelId": "org/other", "artifactRole": "base"},
        tombstones,
    )


def test_advisories_cover_installed_and_current_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older active generation keeps its warning after a registry update."""

    installed_card = cast("ModelCard", object())
    current_card = cast("ModelCard", object())
    now = datetime.now(UTC)

    def advisory(advisory_id: str) -> RegistryAdvisory:
        return RegistryAdvisory(
            schema_version=1,
            advisory_id=advisory_id,
            severity="critical",
            title="Installed generation warning",
            description="The retained artifact needs operator attention.",
            active=True,
            created_at=now,
            updated_at=now,
            enforcement="warn",
        )

    installed_warning = advisory("FLA-2026-1001")
    shared_warning = advisory("FLA-2026-1002")
    current_warning = advisory("FLA-2026-1003")

    def advisories_for(card: ModelCard) -> tuple[RegistryAdvisory, ...]:
        if card is installed_card:
            return installed_warning, shared_warning
        return shared_warning, current_warning

    monkeypatch.setattr(api_main, "get_model_advisories", advisories_for)

    combined = _combined_model_advisories((installed_card, current_card))

    assert [item.advisory_id for item in combined] == [
        "FLA-2026-1001",
        "FLA-2026-1002",
        "FLA-2026-1003",
    ]


def test_inventory_includes_direct_and_read_only_model_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback downloads remain discoverable after the store reconnects."""

    staging_root = tmp_path / "staging"
    direct_root = tmp_path / "models"
    read_only_root = tmp_path / "archive"
    staging_root.mkdir()
    direct_root.mkdir()
    read_only_root.mkdir()
    artifact, _record = _write_artifact(direct_root)
    monkeypatch.setattr(
        artifact_inventory.shared_constants,
        "SKULK_MODELS_DIR",
        direct_root,
    )
    monkeypatch.setattr(
        artifact_inventory.shared_constants,
        "SKULK_MODELS_PATH",
        (read_only_root, direct_root),
    )

    roots = _installed_artifact_roots(staging_root)
    inventory = _inventory_installed_artifacts(roots, (_installed_card(),))

    assert roots == (staging_root, direct_root, read_only_root)
    assert [Path(item.directory) for item in inventory] == [artifact]
    assert inventory[0].manifest_complete


def test_canonical_generation_requires_adjacent_complete_manifest(
    tmp_path: Path,
) -> None:
    """A corrupt canonical generation stays eligible for peer recovery."""

    store_root = tmp_path / "store"
    store_root.mkdir()
    artifact, installed_record = _write_artifact(store_root)
    registry: list[dict[str, object]] = [
        {
            "model_id": "org/model",
            "store_path": artifact.name,
            "installed_card": installed_record.model_dump(mode="json"),
        }
    ]

    assert _complete_canonical_generations(store_root, registry) == {
        (installed_record.installed_identity, installed_record.manifest_sha256)
    }
    assert (
        installed_record.installed_identity,
        "f" * 64,
    ) not in _complete_canonical_generations(store_root, registry)

    (artifact / "model.safetensors").write_bytes(b"truncated")

    assert _complete_canonical_generations(store_root, registry) == set()


async def test_scheduled_startup_reconciliation_is_not_reported_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deliberate first-scan delay remains visibly non-converged."""

    api = object.__new__(api_main.API)
    api._store_client = cast(
        "ModelStoreClient",
        cast(object, SimpleNamespace(local_store_path=tmp_path)),
    )
    api._skulk_config = SkulkConfig(
        model_store=ModelStoreConfig(
            store_host="store-node",
            store_path=str(tmp_path),
        )
    )
    api._reconciliation_status = ReconciliationStatus()

    class StartupDelayObservedError(Exception):
        """Stop the lifetime loop after observing its first delay."""

    async def observe_startup_delay(delay_seconds: float) -> None:
        assert delay_seconds == 10
        assert api._reconciliation_status.state == "scanning"
        raise StartupDelayObservedError

    monkeypatch.setattr(api_main.anyio, "sleep", observe_startup_delay)

    with pytest.raises(StartupDelayObservedError):
        await api._store_reconciliation_loop()


@pytest.mark.parametrize("transition", ["startup-delay", "running-pass"])
@pytest.mark.parametrize(
    "replacement",
    ["missing-config", "missing-store", "store-disabled", "scan-disabled"],
)
async def test_reconciliation_survives_configuration_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transition: str,
    replacement: str,
) -> None:
    """Configuration withdrawal cannot escape the API lifetime task group."""
    from skulk.store.config import ReconciliationStoreConfig

    api = object.__new__(api_main.API)
    api._store_client = cast(
        "ModelStoreClient", cast(object, SimpleNamespace(local_store_path=tmp_path))
    )
    store = ModelStoreConfig(store_host="store-node", store_path=str(tmp_path))
    api._skulk_config = SkulkConfig(model_store=store)
    api._reconciliation_status = ReconciliationStatus()
    passes = 0
    delays: list[float] = []

    def replace_configuration() -> None:
        if replacement == "missing-config":
            api._skulk_config = None
        elif replacement == "missing-store":
            api._skulk_config = SkulkConfig()
        elif replacement == "store-disabled":
            api._skulk_config = SkulkConfig(
                model_store=ModelStoreConfig(
                    store_host="store-node", store_path=str(tmp_path), enabled=False
                )
            )
        else:
            api._skulk_config = SkulkConfig(
                model_store=ModelStoreConfig(
                    store_host="store-node",
                    store_path=str(tmp_path),
                    reconciliation=ReconciliationStoreConfig(enabled=False),
                )
            )

    class DormantLoopObservedError(Exception):
        """End the test once configuration withdrawal has made the loop dormant."""

    async def sleep(delay_seconds: float) -> None:
        delays.append(delay_seconds)
        assert len(delays) <= 3, "withdrawn configuration must stop automatic scans"
        if api._reconciliation_status.state == "idle":
            raise DormantLoopObservedError
        if len(delays) == 1 and transition == "startup-delay":
            replace_configuration()

    async def reconcile(_self: api_main.API) -> ReconciliationStatus:
        nonlocal passes
        passes += 1
        if transition == "running-pass":
            replace_configuration()
        return ReconciliationStatus(state="complete")

    monkeypatch.setattr(api_main.anyio, "sleep", sleep)
    monkeypatch.setattr(api_main.API, "_run_store_reconciliation", reconcile)
    with pytest.raises(DormantLoopObservedError):
        await api._store_reconciliation_loop()
    assert passes == (0 if transition == "startup-delay" else 1)
    assert delays == (
        [10, 10]
        if transition == "startup-delay"
        else [10, store.reconciliation.interval_seconds, 10]
    )
    assert api._reconciliation_status.state == "idle"


@pytest.mark.parametrize("initially_configured", [False, True])
async def test_reconciliation_resumes_after_local_store_configuration_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initially_configured: bool,
) -> None:
    """The sole lifetime task survives a missing or temporarily withdrawn store."""
    from skulk.store.config import ReconciliationStoreConfig

    api = object.__new__(api_main.API)
    client = cast(
        "ModelStoreClient", cast(object, SimpleNamespace(local_store_path=tmp_path))
    )
    store = ModelStoreConfig(
        store_host="store-node",
        store_path=str(tmp_path),
        reconciliation=ReconciliationStoreConfig(interval_seconds=1234),
    )
    api._store_client = client if initially_configured else None
    api._skulk_config = SkulkConfig(model_store=store) if initially_configured else None
    api._reconciliation_status = ReconciliationStatus()
    delays: list[float] = []
    passes = 0

    class RestoredPassObservedError(Exception):
        """End the test once a restored store has completed its first pass."""

    async def sleep(delay_seconds: float) -> None:
        delays.append(delay_seconds)
        if len(delays) == 1:
            api._store_client = None
            api._skulk_config = SkulkConfig()
        elif len(delays) == 2:
            assert api._reconciliation_status.state == "idle"
            api._store_client = client
            api._skulk_config = SkulkConfig(model_store=store)
        else:
            assert passes == 1
            raise RestoredPassObservedError

    async def reconcile(_self: api_main.API) -> ReconciliationStatus:
        nonlocal passes
        passes += 1
        return ReconciliationStatus(state="complete")

    monkeypatch.setattr(api_main.anyio, "sleep", sleep)
    monkeypatch.setattr(api_main.API, "_run_store_reconciliation", reconcile)
    with pytest.raises(RestoredPassObservedError):
        await api._store_reconciliation_loop()
    assert delays == [10, 10, 1234]
    assert passes == 1
