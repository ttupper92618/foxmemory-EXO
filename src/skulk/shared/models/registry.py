"""TUF-verified external model-card catalog with bounded offline fallback."""

import hashlib
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Self, cast

from filelock import FileLock
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from tuf.ngclient.updater import Updater
from tuf.ngclient.urllib3_fetcher import Urllib3Fetcher

from skulk.shared.models.registry_gguf_metadata import RegistryGgufMetadata

EMBEDDED_REGISTRY_ROOT = Path(__file__).with_name("model_registry_root.json")
"""Public TUF trust root shipped inside the Skulk Python package."""


class RegistryUnavailableError(RuntimeError):
    """Raised when neither the signed registry nor a safe local snapshot exists."""


class RegistryArtifactBundleFile(BaseModel):
    """One exact repository file required by a bundle-aware card."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    path: str = Field(min_length=1, max_length=4096)
    size_bytes: int = Field(ge=0)
    object_id: str | None = Field(
        default=None,
        pattern=r"^(?:sha256:[0-9a-f]{64}|git-sha1:[0-9a-f]{40})$",
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        """Require a canonical repository-relative POSIX path."""

        from pathlib import PurePosixPath

        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or value != path.as_posix()
            or "\\" in value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("bundle file path must be canonical and relative")
        return value


def _validate_bundle_path(value: str, *, label: str) -> None:
    """Validate one canonical repository-relative POSIX path."""

    from pathlib import PurePosixPath

    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be canonical and relative")


class RegistryArtifactBundleAlternate(BaseModel):
    """One proven equivalent repository location for a bundle payload."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    root: str | None = Field(default=None, min_length=1, max_length=2048)
    paths: tuple[str, ...] = Field(min_length=1)

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: str | None) -> str | None:
        """Require an alternate loader root to remain repository-relative."""

        if value is not None:
            _validate_bundle_path(value, label="alternate root")
        return value

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require canonical, unique alternate file paths."""

        for path in value:
            _validate_bundle_path(path, label="alternate path")
        if len(set(value)) != len(value):
            raise ValueError("alternate bundle paths must be unique")
        return value


class RegistryArtifactBundle(BaseModel):
    """Complete immutable repository file selection for one artifact."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    bundle_id: str = Field(pattern=r"^bundle_[a-z2-7]{52}$")
    root: str | None = Field(default=None, min_length=1, max_length=2048)
    files: tuple[RegistryArtifactBundleFile, ...] = Field(min_length=1)
    download_size: int = Field(gt=0)
    alternate_locations: tuple[RegistryArtifactBundleAlternate, ...] = ()

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: str | None) -> str | None:
        """Require a safe directory root when the bundle is directory-scoped."""

        if value is None:
            return None
        _validate_bundle_path(value, label="bundle root")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> "RegistryArtifactBundle":
        """Keep the signed size and optional loader root internally consistent."""

        paths = tuple(item.path for item in self.files)
        if len(set(paths)) != len(paths):
            raise ValueError("bundle files must not contain duplicate paths")
        if sum(item.size_bytes for item in self.files) != self.download_size:
            raise ValueError("bundle download_size must equal the file-size sum")
        if self.root is not None:
            prefix = f"{self.root}/"
            if any(path != self.root and not path.startswith(prefix) for path in paths):
                raise ValueError("every bundle file must be beneath the bundle root")
        for alternate in self.alternate_locations:
            if len(alternate.paths) != len(self.files):
                raise ValueError(
                    "alternate bundle locations must map every primary file"
                )
            if alternate.root is not None:
                prefix = f"{alternate.root}/"
                if any(
                    path != alternate.root and not path.startswith(prefix)
                    for path in alternate.paths
                ):
                    raise ValueError(
                        "alternate bundle files must remain beneath their root"
                    )
        return self


class RegistryArtifact(BaseModel):
    """Exact upstream artifact identity asserted by a registry card."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    repository: str = Field(min_length=3, max_length=512)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    selected_file: str | None = Field(default=None, max_length=2048)
    format: str = Field(min_length=1, max_length=80)
    quantization: str = Field(max_length=120)
    bundle: RegistryArtifactBundle | None = None


class RegistryCard(BaseModel):
    """One immutable registry envelope in a signed catalog target."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: Literal[1, 2]
    card_id: str = Field(pattern=r"^card_[a-z2-7]{52}$")
    alias: str = Field(min_length=1, max_length=512)
    model_ref: str = Field(min_length=1, max_length=512)
    artifact: RegistryArtifact
    card: dict[str, Any]

    @model_validator(mode="after")
    def validate_versioned_artifact(self) -> "RegistryCard":
        """Require the v2 envelope exactly when bundle identity is present."""

        if self.schema_version == 2 and self.artifact.bundle is None:
            raise ValueError("registry card schema v2 requires an artifact bundle")
        if self.schema_version == 1 and self.artifact.bundle is not None:
            raise ValueError("registry card schema v1 cannot carry an artifact bundle")
        return self

    @field_validator("alias")
    @classmethod
    def validate_alias(cls, alias: str) -> str:
        """Require a path-safe repository-style runtime identity.

        The alias becomes a staging-directory name after the single repository
        separator is normalized. Restricting the signed input here keeps dot
        segments and platform path separators away from destructive cache
        replacement operations.
        """
        if (
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._+@-]*",
                alias,
            )
            is None
        ):
            raise ValueError("registry alias must be a safe repository identifier")
        return alias


class RegistryCardMetadata(BaseModel):
    """Mutable catalog metadata that must not alter immutable card identity."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    provenance: Literal["foxlight", "agent", "community"]
    architecture: str | None = Field(default=None, min_length=1, max_length=300)
    capability_claims: tuple["RegistryCapabilityClaim", ...] = ()
    source_links: "RegistrySourceLinks | None" = None


class RegistrySourceLinks(BaseModel):
    """Signed navigation links derived from immutable artifact identity."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    repository_url: str = Field(
        min_length=8, max_length=2048, description="Upstream repository page."
    )
    revision_url: str = Field(
        min_length=8, max_length=2048, description="Pinned revision page."
    )
    artifact_url: str = Field(
        min_length=8, max_length=2048, description="Pinned selected-artifact page."
    )
    download_url: str | None = Field(
        default=None, max_length=2048, description="Optional pinned download URL."
    )


class RegistryCapabilityClaim(BaseModel):
    """Open signed model or artifact capability independent of runtime support."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    capability_id: str = Field(
        pattern=r"^[a-z0-9][a-z0-9._:-]{0,199}$",
        description="Open namespaced intrinsic capability identifier.",
    )
    scope: Literal["model", "artifact"] = Field(
        description="Whether the claim describes the model or selected artifact."
    )
    status: Literal[
        "claimed", "observed", "complete", "incomplete", "unknown"
    ] = Field(description="Evidence state without an engine-support implication.")
    source: Literal[
        "upstream_structured", "artifact_manifest", "agent_analysis"
    ] = Field(description="Evidence channel that produced the claim.")
    confidence: float = Field(
        ge=0, le=1, description="Source confidence from zero through one."
    )
    evidence_urls: tuple[str, ...] = Field(
        default=(), max_length=20, description="Bounded evidence references."
    )
    reviewer_model: str | None = Field(
        default=None, max_length=300, description="Agent reviewer identity, if any."
    )
    input_modalities: tuple[str, ...] = Field(
        default=(), max_length=20, description="Declared input modalities."
    )
    output_modalities: tuple[str, ...] = Field(
        default=(), max_length=20, description="Declared output modalities."
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Open capability-specific evidence details."
    )

    @field_validator(
        "evidence_urls", "input_modalities", "output_modalities", mode="before"
    )
    @classmethod
    def coerce_string_tuples(cls, value: object) -> object:
        """Coerce JSON and TOML arrays into immutable string tuples."""
        if isinstance(value, list):
            return tuple(cast("list[object]", value))
        return value


class RegistryCatalog(BaseModel):
    """Complete published model-card catalog verified as one TUF target."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: Literal[2]
    snapshot_id: str = Field(min_length=1, max_length=120)
    generated_at: datetime
    published_by: str = Field(min_length=1, max_length=320)
    note: str
    cards: tuple[RegistryCard, ...]
    card_metadata: dict[str, RegistryCardMetadata]
    _gguf_metadata: RegistryGgufMetadata | None = PrivateAttr(default=None)

    @property
    def gguf_metadata(self) -> RegistryGgufMetadata | None:
        """Return separately verified header facts, never an inline catalog field."""
        return self._gguf_metadata

    def with_gguf_metadata(self, metadata: RegistryGgufMetadata | None) -> Self:
        """Bind auxiliary evidence without modifying canonical catalog bytes."""
        if metadata is not None:
            if metadata.snapshot_id != self.snapshot_id:
                raise ValueError("GGUF metadata catalog snapshot mismatch")
            cards = {card.card_id: card for card in self.cards}
            for card_id, entry in metadata.artifacts.items():
                card = cards.get(card_id)
                if card is None or (
                    card.artifact.format != "gguf"
                    or entry.repository != card.artifact.repository
                    or entry.revision != card.artifact.revision
                    or entry.selected_file != card.artifact.selected_file
                ):
                    raise ValueError("GGUF metadata artifact identity mismatch")
        copied = self.model_copy()
        copied._gguf_metadata = metadata
        return copied


class RegistryAdvisory(BaseModel):
    """One signed warn-only security notice."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: Literal[1]
    advisory_id: str = Field(pattern=r"^FLA-[0-9]{4}-[0-9]{4,}$")
    severity: Literal["low", "moderate", "high", "critical"]
    title: str = Field(min_length=3, max_length=300)
    description: str = Field(min_length=3, max_length=4000)
    affected_card_ids: tuple[str, ...] = ()
    affected_model_aliases: tuple[str, ...] = ()
    reference_url: str | None = Field(default=None, max_length=2048)
    active: bool
    created_at: datetime
    updated_at: datetime
    enforcement: Literal["warn"]


class RegistryAdvisories(BaseModel):
    """Signed advisory target; enforcement is structurally warn-only."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: Literal[1]
    generated_at: datetime
    enforcement: Literal["warn"]
    advisories: tuple[RegistryAdvisory, ...]


class RegistryEngineSupportClaim(BaseModel):
    """One append-only signed engine/build compatibility decision."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    engine: str = Field(
        pattern=r"^[a-z0-9][a-z0-9._:-]{0,119}$",
        description="Open inference-engine identifier.",
    )
    engine_build: str = Field(
        min_length=1,
        max_length=300,
        description="Exact engine build qualified by this decision.",
    )
    architecture: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,299}$",
        description="Trusted open architecture identifier.",
    )
    artifact_format: str = Field(
        pattern=r"^[a-z0-9][a-z0-9._+-]{0,79}$",
        description="Exact selected artifact format.",
    )
    artifact_card_id: str | None = Field(
        default=None,
        pattern=r"^card_[a-z2-7]{52}$",
        description=(
            "Exact card qualified by load or feature evidence; null is permitted "
            "only for architecture-level upstream compatibility."
        ),
    )
    quantization: str | None = Field(
        default=None,
        max_length=120,
        description="Exact quantization or null as a format-wide wildcard.",
    )
    capability_id: str = Field(
        pattern=r"^[a-z0-9][a-z0-9._:-]{0,199}$",
        description="Intrinsic capability this engine decision covers.",
    )
    status: Literal["supported", "experimental", "unsupported"] = Field(
        description="Decision status; only supported can expand placement."
    )
    evidence_kind: Literal[
        "upstream_compatibility", "load_qualification", "feature_qualification"
    ] = Field(description="Kind of compatibility evidence reviewed.")
    evidence_trust: Literal[
        "self_attested",
        "third_party_reported",
        "reproducible",
        "independently_reproduced",
        "foxlight_observed",
    ] = Field(description="Trust tier assigned to the evidence.")
    source_url: str = Field(
        min_length=8, max_length=2048, description="Stable evidence location."
    )
    source_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Evidence bundle SHA-256 digest."
    )
    rationale: str = Field(
        min_length=3, max_length=4000, description="Auditable decision rationale."
    )
    hardware_classes: tuple[str, ...] = Field(
        default=(),
        max_length=50,
        description="Optional open hardware constraints; empty is neutral.",
    )
    supersedes_claim_id: str | None = Field(
        default=None,
        pattern=r"^support_[a-z2-7]{52}$",
        description="Prior active claim for the same exact key being replaced.",
    )
    claim_id: str = Field(
        pattern=r"^support_[a-z2-7]{52}$",
        description="Content-derived immutable support-claim identity.",
    )
    recorded_by: str = Field(
        min_length=1, max_length=320, description="Audited publishing actor."
    )
    created_at: datetime = Field(description="UTC claim publication time.")

    @model_validator(mode="after")
    def require_artifact_binding_for_qualification(self) -> Self:
        """Reject empirical qualification not bound to tested card bytes."""
        if (
            self.evidence_kind in {"load_qualification", "feature_qualification"}
            and self.artifact_card_id is None
        ):
            raise ValueError(
                "load and feature qualification require an exact artifact_card_id"
            )
        return self


class RegistryEngineSupport(BaseModel):
    """Versioned signed engine-support matrix target."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: Literal[1]
    matrix_version: int = Field(ge=1)
    generated_at: datetime
    claims: tuple[RegistryEngineSupportClaim, ...]

    @model_validator(mode="after")
    def validate_supersession_history(self) -> Self:
        """Reject ambiguous or cross-key replacement history."""
        claims_by_id = {claim.claim_id: claim for claim in self.claims}
        if len(claims_by_id) != len(self.claims):
            raise ValueError("engine-support claim ids must be unique")
        superseded_ids: set[str] = set()
        for claim in self.claims:
            replaced_id = claim.supersedes_claim_id
            if replaced_id is None:
                continue
            replaced = claims_by_id.get(replaced_id)
            if replaced is None:
                raise ValueError("superseded engine-support claim does not exist")
            if replaced_id in superseded_ids:
                raise ValueError("engine-support claim is superseded more than once")
            if self._support_key(claim) != self._support_key(replaced):
                raise ValueError("engine-support claim supersedes a different key")
            superseded_ids.add(replaced_id)
            seen = {claim.claim_id}
            cursor = replaced
            while cursor.supersedes_claim_id is not None:
                if cursor.claim_id in seen:
                    raise ValueError("engine-support supersession cycle detected")
                seen.add(cursor.claim_id)
                next_claim = claims_by_id.get(cursor.supersedes_claim_id)
                if next_claim is None:
                    raise ValueError(
                        "superseded engine-support claim does not exist"
                    )
                cursor = next_claim
        active_keys = [
            self._support_key(claim)
            for claim in self.claims
            if claim.claim_id not in superseded_ids
        ]
        if len(active_keys) != len(set(active_keys)):
            raise ValueError("multiple active engine-support claims share one key")
        return self

    @staticmethod
    def _support_key(
        claim: RegistryEngineSupportClaim,
    ) -> tuple[str, str, str, str, str | None, str | None, str, tuple[str, ...]]:
        """Return the fields defining one replaceable compatibility decision."""
        return (
            claim.engine,
            claim.engine_build,
            claim.architecture,
            claim.artifact_format,
            claim.quantization,
            claim.artifact_card_id,
            claim.capability_id,
            claim.hardware_classes,
        )

    def active_claims(self) -> tuple[RegistryEngineSupportClaim, ...]:
        """Return claims not explicitly replaced by a later history entry."""
        superseded = {
            claim.supersedes_claim_id
            for claim in self.claims
            if claim.supersedes_claim_id is not None
        }
        return tuple(claim for claim in self.claims if claim.claim_id not in superseded)


class _VerifiedCacheRecord(BaseModel):
    """Local proof that the last-known-good bytes passed TUF verification."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified_at: datetime
    snapshot_id: str


class _VerifiedAdvisoryCacheRecord(BaseModel):
    """Local proof that cached advisory bytes passed TUF verification."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified_at: datetime


class _VerifiedEngineSupportCacheRecord(BaseModel):
    """Local proof that cached engine-support bytes passed TUF verification."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified_at: datetime
    matrix_version: int = Field(ge=1)


class _TrustedTargetsSignedVersion(BaseModel):
    """Minimal version view over updater-verified targets metadata."""

    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")

    version: int = Field(ge=1)


class _TrustedTargetsMetadataVersion(BaseModel):
    """Minimal envelope for comparing a target payload with its signed role."""

    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")

    signed: _TrustedTargetsSignedVersion


class TufRegistryClient:
    """Load a signed catalog and retain a hash-bound last-known-good copy."""

    def __init__(
        self,
        *,
        base_url: str,
        cache_dir: Path,
        embedded_root: Path,
        timeout_seconds: int,
        max_stale_days: int,
    ) -> None:
        """Configure registry transport, trust anchor, and offline policy."""
        self._base_url = base_url.rstrip("/") + "/"
        self._cache_dir = cache_dir
        self._embedded_root = embedded_root
        self._timeout_seconds = timeout_seconds
        self._max_stale_days = max_stale_days
        self._metadata_dir = cache_dir / "metadata"
        self._targets_dir = cache_dir / "targets"
        self._last_known_good_path = cache_dir / "last-known-good-catalog.json"
        self._cache_record_path = cache_dir / "last-known-good.json"
        self._gguf_metadata_path = cache_dir / "last-known-good-gguf-metadata.json"
        self._gguf_cache_record_path = cache_dir / "last-known-good-gguf-record.json"
        self._last_known_good_advisories_path = (
            cache_dir / "last-known-good-advisories.json"
        )
        self._advisory_cache_record_path = (
            cache_dir / "last-known-good-advisories-metadata.json"
        )
        self._last_known_good_engine_support_path = (
            cache_dir / "last-known-good-engine-support.json"
        )
        self._engine_support_cache_record_path = (
            cache_dir / "last-known-good-engine-support-metadata.json"
        )
        self._lock = FileLock(str(cache_dir / "refresh.lock"))

    def load_catalog(
        self,
        catalog_validator: Callable[[RegistryCatalog], object] | None = None,
    ) -> RegistryCatalog:
        """Refresh and validate a signed catalog or return a bounded valid cache.

        Args:
            catalog_validator: Optional consumer-level validator applied before a
                newly verified target replaces the last-known-good catalog and
                whenever cached bytes are recovered.

        Returns:
            A TUF-verified catalog accepted by both the wire schema and the
            supplied runtime validator.

        Raises:
            RegistryUnavailableError: When neither the network target nor the
                bounded last-known-good catalog passes validation.

        Side effects:
            Persists newly verified target bytes only after complete validation.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            try:
                return self._refresh(catalog_validator)
            except Exception as error:  # noqa: BLE001 - security fallback boundary
                try:
                    return self._load_last_known_good(catalog_validator)
                except (OSError, ValueError):
                    raise RegistryUnavailableError(
                        "signed model registry is unavailable and no acceptable "
                        "last-known-good catalog exists"
                    ) from error

    def load_advisories(self) -> RegistryAdvisories:
        """Refresh signed advisories or recover their bounded verified cache."""

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            try:
                return self._refresh_advisories()
            except Exception as error:  # noqa: BLE001 - security fallback boundary
                try:
                    return self._load_last_known_good_advisories()
                except (OSError, ValueError):
                    raise RegistryUnavailableError(
                        "signed model advisories are unavailable and no acceptable "
                        "last-known-good advisory target exists"
                    ) from error

    def load_engine_support(self) -> RegistryEngineSupport:
        """Refresh the signed support matrix or recover its verified cache."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            try:
                return self._refresh_engine_support()
            except Exception as error:  # noqa: BLE001 - security fallback boundary
                try:
                    return self._load_last_known_good_engine_support()
                except (OSError, ValueError):
                    raise RegistryUnavailableError(
                        "signed engine support is unavailable and no acceptable "
                        "last-known-good matrix exists"
                    ) from error

    def load_cached_engine_support(self) -> RegistryEngineSupport:
        """Load hash-bound support data without network or freshness expiry."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            return self._load_last_known_good_engine_support(enforce_freshness=False)

    def _refresh_engine_support(self) -> RegistryEngineSupport:
        """Perform a TUF refresh and retain the verified support matrix."""
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        self._targets_dir.mkdir(parents=True, exist_ok=True)
        updater = Updater(
            metadata_dir=str(self._metadata_dir),
            metadata_base_url=f"{self._base_url}metadata/",
            target_dir=str(self._targets_dir),
            target_base_url=f"{self._base_url}targets/",
            fetcher=Urllib3Fetcher(
                socket_timeout=self._timeout_seconds,
                app_user_agent="Skulk model-registry client",
            ),
            bootstrap=self._embedded_root.read_bytes(),
        )
        updater.refresh()
        target = updater.get_targetinfo("v1/engine-support.json")
        if target is None:
            raise ValueError("signed repository has no v1/engine-support.json target")
        payload = Path(updater.download_target(target)).read_bytes()
        support = RegistryEngineSupport.model_validate_json(payload, strict=False)
        trusted_targets = _TrustedTargetsMetadataVersion.model_validate_json(
            (self._metadata_dir / "targets.json").read_bytes(), strict=False
        )
        if support.matrix_version != trusted_targets.signed.version:
            raise ValueError(
                "engine-support matrix version does not match signed targets metadata"
            )
        self._write_verified_engine_support_cache(payload, support)
        return support

    def _refresh_advisories(self) -> RegistryAdvisories:
        """Perform a TUF refresh and retain the verified advisory target."""

        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        self._targets_dir.mkdir(parents=True, exist_ok=True)
        updater = Updater(
            metadata_dir=str(self._metadata_dir),
            metadata_base_url=f"{self._base_url}metadata/",
            target_dir=str(self._targets_dir),
            target_base_url=f"{self._base_url}targets/",
            fetcher=Urllib3Fetcher(
                socket_timeout=self._timeout_seconds,
                app_user_agent="Skulk model-registry client",
            ),
            bootstrap=self._embedded_root.read_bytes(),
        )
        updater.refresh()
        target = updater.get_targetinfo("v1/advisories.json")
        if target is None:
            advisories = RegistryAdvisories(
                schema_version=1,
                generated_at=datetime.now(UTC),
                enforcement="warn",
                advisories=(),
            )
            payload = advisories.model_dump_json().encode()
        else:
            downloaded = Path(updater.download_target(target))
            payload = downloaded.read_bytes()
            advisories = RegistryAdvisories.model_validate_json(
                payload, strict=False
            )
        self._write_verified_advisory_cache(payload)
        return advisories

    def _refresh(
        self,
        catalog_validator: Callable[[RegistryCatalog], object] | None,
    ) -> RegistryCatalog:
        """Perform a normal python-tuf refresh and persist verified target bytes."""
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        self._targets_dir.mkdir(parents=True, exist_ok=True)
        updater = Updater(
            metadata_dir=str(self._metadata_dir),
            metadata_base_url=f"{self._base_url}metadata/",
            target_dir=str(self._targets_dir),
            target_base_url=f"{self._base_url}targets/",
            fetcher=Urllib3Fetcher(
                socket_timeout=self._timeout_seconds,
                app_user_agent="Skulk model-registry client",
            ),
            # Supplying an explicit bootstrap ignores an arbitrary cached
            # root.json. python-tuf then replays its verified local root_history
            # before consulting the network, preserving legitimate rotations.
            bootstrap=self._embedded_root.read_bytes(),
        )
        updater.refresh()
        target = updater.get_targetinfo("v1/catalog.json")
        if target is None:
            raise ValueError("signed repository has no v1/catalog.json target")
        downloaded = Path(updater.download_target(target))
        payload = downloaded.read_bytes()
        catalog = RegistryCatalog.model_validate_json(payload, strict=False)
        gguf_target = updater.get_targetinfo("v1/gguf-metadata.json")
        gguf_metadata: RegistryGgufMetadata | None = None
        if gguf_target is not None:
            gguf_metadata = RegistryGgufMetadata.model_validate_json(
                Path(updater.download_target(gguf_target)).read_bytes()
            )
            trusted_targets = _TrustedTargetsMetadataVersion.model_validate_json(
                (self._metadata_dir / "targets.json").read_bytes(), strict=False
            )
            if gguf_metadata.target_version != trusted_targets.signed.version:
                raise ValueError("GGUF metadata version does not match signed targets")
        catalog = catalog.with_gguf_metadata(gguf_metadata)
        if catalog_validator is not None:
            catalog_validator(catalog)
        self._write_verified_gguf_cache(catalog)
        self._write_verified_cache(payload, catalog)
        return catalog

    def _write_verified_gguf_cache(self, catalog: RegistryCatalog) -> None:
        """Retain a hash-bound auxiliary projection without changing old cache schemas."""
        payload = (
            catalog.gguf_metadata.model_dump_json().encode()
            if catalog.gguf_metadata is not None
            else b"null"
        )
        record = _VerifiedCacheRecord(
            sha256=hashlib.sha256(payload).hexdigest(),
            verified_at=datetime.now(UTC),
            snapshot_id=catalog.snapshot_id,
        )
        self._atomic_write(self._gguf_metadata_path, payload)
        self._atomic_write(
            self._gguf_cache_record_path, record.model_dump_json().encode()
        )

    def _attach_cached_gguf_metadata(self, catalog: RegistryCatalog) -> RegistryCatalog:
        """Recover only auxiliary facts tied to this exact cached catalog."""
        if (
            not self._gguf_cache_record_path.exists()
            and not self._gguf_metadata_path.exists()
        ):
            # A cache created by an older reader has no auxiliary evidence.
            return catalog
        record = _VerifiedCacheRecord.model_validate_json(
            self._gguf_cache_record_path.read_bytes(), strict=False
        )
        if record.snapshot_id != catalog.snapshot_id:
            raise ValueError("cached GGUF metadata snapshot mismatch")
        if datetime.now(UTC) - record.verified_at.astimezone(UTC) > timedelta(
            days=self._max_stale_days
        ):
            raise ValueError("cached GGUF metadata is too old")
        payload = self._gguf_metadata_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            raise ValueError("cached GGUF metadata hash mismatch")
        return catalog.with_gguf_metadata(
            None
            if payload == b"null"
            else RegistryGgufMetadata.model_validate_json(payload)
        )

    def _write_verified_cache(self, payload: bytes, catalog: RegistryCatalog) -> None:
        """Atomically retain bytes that the updater just verified."""
        record = _VerifiedCacheRecord(
            sha256=hashlib.sha256(payload).hexdigest(),
            verified_at=datetime.now(UTC),
            snapshot_id=catalog.snapshot_id,
        )
        self._atomic_write(self._last_known_good_path, payload)
        self._atomic_write(
            self._cache_record_path,
            record.model_dump_json().encode(),
        )

    def _write_verified_advisory_cache(self, payload: bytes) -> None:
        """Atomically retain advisory bytes that the updater just verified."""

        record = _VerifiedAdvisoryCacheRecord(
            sha256=hashlib.sha256(payload).hexdigest(),
            verified_at=datetime.now(UTC),
        )
        self._atomic_write(self._last_known_good_advisories_path, payload)
        self._atomic_write(
            self._advisory_cache_record_path,
            record.model_dump_json().encode(),
        )

    def _write_verified_engine_support_cache(
        self, payload: bytes, support: RegistryEngineSupport
    ) -> None:
        """Atomically retain matrix bytes that TUF just verified."""
        record = _VerifiedEngineSupportCacheRecord(
            sha256=hashlib.sha256(payload).hexdigest(),
            verified_at=datetime.now(UTC),
            matrix_version=support.matrix_version,
        )
        self._atomic_write(self._last_known_good_engine_support_path, payload)
        self._atomic_write(
            self._engine_support_cache_record_path,
            record.model_dump_json().encode(),
        )

    def _load_last_known_good(
        self,
        catalog_validator: Callable[[RegistryCatalog], object] | None,
    ) -> RegistryCatalog:
        """Load only hash-bound, previously verified, sufficiently fresh bytes."""
        if self._max_stale_days == 0:
            raise ValueError("last-known-good registry fallback is disabled")
        record = _VerifiedCacheRecord.model_validate_json(
            self._cache_record_path.read_bytes(), strict=False
        )
        now = datetime.now(UTC)
        verified_at = record.verified_at.astimezone(UTC)
        if now - verified_at > timedelta(days=self._max_stale_days):
            raise ValueError("last-known-good registry catalog is too old")
        payload = self._last_known_good_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            raise ValueError("last-known-good registry catalog hash mismatch")
        catalog = RegistryCatalog.model_validate_json(payload, strict=False)
        if catalog.snapshot_id != record.snapshot_id:
            raise ValueError("last-known-good snapshot identity mismatch")
        catalog = self._attach_cached_gguf_metadata(catalog)
        if catalog_validator is not None:
            catalog_validator(catalog)
        return catalog

    def _load_last_known_good_advisories(self) -> RegistryAdvisories:
        """Load hash-bound, previously verified, sufficiently fresh warnings."""

        if self._max_stale_days == 0:
            raise ValueError("last-known-good registry fallback is disabled")
        record = _VerifiedAdvisoryCacheRecord.model_validate_json(
            self._advisory_cache_record_path.read_bytes(), strict=False
        )
        now = datetime.now(UTC)
        verified_at = record.verified_at.astimezone(UTC)
        if now - verified_at > timedelta(days=self._max_stale_days):
            raise ValueError("last-known-good model advisories are too old")
        payload = self._last_known_good_advisories_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            raise ValueError("last-known-good model advisories hash mismatch")
        return RegistryAdvisories.model_validate_json(payload, strict=False)

    def _load_last_known_good_engine_support(
        self, *, enforce_freshness: bool = True
    ) -> RegistryEngineSupport:
        """Load only hash-bound, previously verified support-matrix bytes."""
        if enforce_freshness and self._max_stale_days == 0:
            raise ValueError("last-known-good engine support fallback is disabled")
        record = _VerifiedEngineSupportCacheRecord.model_validate_json(
            self._engine_support_cache_record_path.read_bytes(), strict=False
        )
        if enforce_freshness:
            now = datetime.now(UTC)
            verified_at = record.verified_at.astimezone(UTC)
            if now - verified_at > timedelta(days=self._max_stale_days):
                raise ValueError("last-known-good engine support is too old")
        payload = self._last_known_good_engine_support_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            raise ValueError("last-known-good engine support hash mismatch")
        support = RegistryEngineSupport.model_validate_json(payload, strict=False)
        if support.matrix_version != record.matrix_version:
            raise ValueError("last-known-good engine support version mismatch")
        return support

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        """Replace one local cache file without exposing partial contents."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
