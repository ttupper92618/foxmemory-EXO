import asyncio
import base64
import binascii
import contextlib
import copy
import hashlib
import hmac
import ipaddress
import json
import os
import platform
import random
import re
import shutil
import socket
import sqlite3
import time
import weakref
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
)
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    ClassVar,
    Final,
    Literal,
    Protocol,
    TypeVar,
    cast,
)
from uuid import uuid4

import anyio
import httpx
import hypercorn.asyncio as hypercorn_asyncio
import psutil
import yaml
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
    to_thread,
)
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from loguru import logger
from pydantic import UUID4, ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

import skulk.shared.types.tasks as task_types
from skulk.api.adapters.chat_completions import (
    chat_request_to_text_generation,
    collect_chat_response,
    fetch_image_url,
    generate_chat_stream,
    resolve_tool_choice,
)
from skulk.api.adapters.claude import (
    claude_request_to_text_generation,
    collect_claude_response,
    generate_claude_stream,
)
from skulk.api.adapters.ollama import (
    collect_ollama_chat_response,
    collect_ollama_generate_response,
    generate_ollama_chat_stream,
    generate_ollama_generate_stream,
    ollama_generate_request_to_text_generation,
    ollama_request_to_text_generation,
)
from skulk.api.adapters.responses import (
    collect_responses_response,
    generate_responses_stream,
    responses_request_to_text_generation,
)
from skulk.api.data_plane import DataPlaneObserver
from skulk.api.diagnostics_compatibility import (
    aggregate_diagnostics_version_status,
    compare_diagnostics_builds,
    parse_peer_node_diagnostics,
)
from skulk.api.field_telemetry import (
    FieldTelemetryCollector,
    hardware_class,
    prepare_telemetry_config_update,
    tap_generation_stream,
)
from skulk.api.keepalive import with_sse_keepalive
from skulk.api.model_search import (
    fetch_card_summary,
    list_gguf_quant_options,
    search_hugging_face_models,
)
from skulk.api.node_health import (
    compute_node_health,
    live_data_transports,
    live_skulk_build_mismatch,
)
from skulk.api.operator_auth import create_operator_auth_router
from skulk.api.operator_gateway import (
    OPERATOR_GATEWAY_AUTHORIZED_SCOPE_KEY,
    OperatorGatewayAuthorization,
)
from skulk.api.performance_envelope import (
    ClusterPerformanceEnvelopes,
    GenerationOutcome,
    NodePerformanceEnvelopes,
    PerformanceEnvelopeRegistry,
    PerformanceEnvelopeReport,
)
from skulk.api.provider_diagnostics import ProviderObserver
from skulk.api.realtime import (
    REALTIME_WEBSOCKET_MAX_MESSAGE_BYTES,
    ConversationMessage,
    RealtimeResponseConfig,
    RealtimeTranscriptionBridge,
)
from skulk.api.steward import (
    STEWARD_NOT_READY_MESSAGES,
    STEWARD_RETRY_AFTER_SECONDS,
    STEWARD_SYSTEM_PROMPT,
    STEWARD_VIRTUAL_MODEL_ID,
    StewardActionDecisionRequest,
    StewardActionDecisionResponse,
    StewardActionProposalView,
    StewardCanaryState,
    StewardChatMessage,
    StewardHarness,
    StewardStatusResponse,
    derive_steward_state,
    steward_action_proposal_view,
)
from skulk.api.types import (
    AddCustomModelParams,
    AddExactCustomModelCardParams,
    AdvancedImageParams,
    ArtifactExportRequest,
    ArtifactExportResponse,
    AudioCapabilitySection,
    AudioSpeechRequest,
    AudioTranscriptionCompletedEvent,
    AudioTranscriptionDeltaEvent,
    AudioTranscriptionErrorEvent,
    AudioTranscriptionResponseFormat,
    AudioTranscriptionStreamEvent,
    AudioTranscriptionUsageEvent,
    AudioVoice,
    AudioVoiceList,
    BenchChatCompletionRequest,
    BenchChatCompletionResponse,
    BenchImageGenerationResponse,
    BenchImageGenerationTaskParams,
    CachedArtifactLocation,
    CacheInventoryStatus,
    CancelCommandResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionMessageText,
    ChatCompletionRequest,
    CreateInstanceParams,
    CreateInstanceResponse,
    DeleteDownloadResponse,
    DeleteExactCustomModelCardParams,
    DeleteInstanceResponse,
    DeleteTracesRequest,
    DeleteTracesResponse,
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    ErrorInfo,
    ErrorResponse,
    ExtractPageToolRequest,
    ExtractPageToolResponse,
    FinishReason,
    GenerationStats,
    GgufQuantOptions,
    HuggingFaceCardSummary,
    HuggingFaceSearchResult,
    ImageData,
    ImageEditsTaskParams,
    ImageGenerationResponse,
    ImageGenerationStats,
    ImageGenerationTaskParams,
    ImageListItem,
    ImageListResponse,
    ImageSize,
    ModalitiesCapabilitySection,
    ModelList,
    ModelListModel,
    NodeStorageSummary,
    OpenUrlToolRequest,
    OpenUrlToolResponse,
    PlaceInstanceParams,
    PlacementPreview,
    PlacementPreviewResponse,
    PurgeStagingRequest,
    PurgeStagingResponse,
    ReasoningCapabilitySection,
    ReconciliationStatus,
    RemoteCodeApprovalView,
    ResolvedModelCapabilities,
    RuntimeCapabilitySection,
    StartDownloadParams,
    StartDownloadResponse,
    StoreDownloadRequest,
    StoreDownloadResponse,
    StoreRegistryResponse,
    ToolCall,
    ToolingCapabilitySection,
    TraceCategoryStats,
    TraceEventResponse,
    TraceListItem,
    TraceListResponse,
    TraceRankStats,
    TraceResponse,
    TraceSourceNode,
    TraceStatsResponse,
    TraceTaskKind,
    TracingStateResponse,
    UpdateTracingStateRequest,
    WebSearchToolRequest,
    WebSearchToolResponse,
    normalize_image_size,
)
from skulk.api.types.claude_api import (
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
)
from skulk.api.types.ollama_api import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaModelDetails,
    OllamaModelTag,
    OllamaPsModel,
    OllamaPsResponse,
    OllamaShowRequest,
    OllamaShowResponse,
    OllamaTagsResponse,
)
from skulk.api.types.openai_responses import (
    ResponsesRequest,
    ResponsesResponse,
)
from skulk.connectivity.remote_access import RemoteAccessInfo, build_remote_access_info
from skulk.connectivity.tailscale import TailscaleStatus, query_tailscale_status
from skulk.extensions import (
    DEFAULT_CALL_TIMEOUT_SECONDS,
    MAX_CALL_PAYLOAD_BYTES,
    MAX_CALL_TIMEOUT_SECONDS,
    REALTIME_STT_CAPABILITY_DESCRIPTOR,
    STT_CAPABILITY_DESCRIPTOR,
    TTS_CAPABILITY_DESCRIPTOR,
    BuiltinSpeechProvider,
    BuiltinVadProvider,
    CapabilityCall,
    CapabilityDescriptor,
    CapabilityError,
    CapabilityErrorCode,
    CapabilityInputStreamHandler,
    CapabilityResult,
    CapabilityStreamAdmissionHandler,
    CapabilityStreamCancel,
    CapabilityStreamError,
    CapabilityStreamErrorCode,
    CapabilityStreamFrame,
    CapabilityStreamHandler,
    CapabilityStreamInput,
    CapabilityStreamReceiver,
    CapabilityStreamSession,
    ExtensionContext,
    InlineMediaAttachment,
    LoadedExtensions,
    call_failure,
    descriptor_revision,
    resolve_skulk_version,
    snapshot_cluster,
    validate_against_schema,
)
from skulk.extensions.steward import StewardToolBinding
from skulk.master.image_store import ImageStore
from skulk.master.placement import (
    PlacementError,
    PlacementInfoPendingError,
    require_instance_model_card_identity,
    require_instance_model_code_approval,
)
from skulk.master.placement import place_instance as get_instance_placements
from skulk.master.placement_utils import (
    unified_memory_gpu_node_ids,
    usable_vram_by_node,
)
from skulk.operator.pairing import OperatorPairingService
from skulk.operator.relay import OperatorGatewayConnector, OperatorRelayConfiguration
from skulk.routing.provider_streams import ProviderStreamPacket
from skulk.routing.realtime_audio import RealtimeAudioPacket
from skulk.routing.speech_media import SpeechMediaPacket
from skulk.routing.trace_data import TraceDataPacket
from skulk.routing.vision_media import VisionMediaPacket
from skulk.shared.apply import apply
from skulk.shared.backends import engine_of
from skulk.shared.constants import (
    DASHBOARD_DIR,
    SKULK_CACHE_HOME,
    SKULK_ENABLE_IMAGE_MODELS,
    SKULK_EVENT_LOG_DIR,
    SKULK_IMAGE_CACHE_DIR,
    SKULK_IMAGE_TRANSPORT_DEBUG,
    SKULK_MAX_CHUNK_SIZE,
    SKULK_MODELS_DIR,
    SKULK_TRACING_CACHE_DIR,
    preferred_env_value,
)
from skulk.shared.election import ElectionMessage
from skulk.shared.experimental import experimental_mode_enabled
from skulk.shared.logging import InterceptLogger
from skulk.shared.models.capabilities import resolve_model_capability_profile
from skulk.shared.models.model_cards import (
    AudioCardKind,
    AudioResponseFormat,
    ModelCard,
    ModelId,
    ModelTask,
    add_to_card_cache,
    custom_card_mutation_applied,
    delete_custom_card,
    get_all_model_cards,
    get_bundled_card,
    get_card,
    get_current_registry_card,
    get_current_registry_card_id,
    get_custom_card_storage_collision,
    get_installed_card_record,
    get_model_advisories,
    get_model_cards,
    get_model_engine_support,
    preserve_generated_card_constraints,
    record_custom_card_mutation_applied,
    same_authorized_model_card,
)
from skulk.shared.models.registry import RegistryAdvisory
from skulk.shared.models.remote_code_approval import (
    approved_remote_code_identities,
    loopback_mutation_allowed,
    remote_code_execution_requires_approval,
    remote_code_is_automatically_trusted,
    remote_code_trust_identity,
    trusted_fabric_mutation_allowed,
)
from skulk.shared.tracing import (
    TraceEvent,
    compute_stats,
    export_trace,
    load_trace_file,
)
from skulk.shared.types.artifact_inventory import (
    ARTIFACT_INVENTORY_STALE_SECONDS,
    NodeArtifactInventory,
)
from skulk.shared.types.audio import (
    AudioTranscriptionTaskParams,
    RealtimeAudioInputFrame,
    RealtimeAudioTranscriptionTaskParams,
    SpeechSynthesisTaskParams,
)
from skulk.shared.types.chunks import (
    AudioChunk,
    DataChunk,
    EmbeddingChunk,
    ErrorChunk,
    GenerationChunk,
    ImageChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
    TranscriptionChunk,
)
from skulk.shared.types.commands import (
    AddCustomModelCard,
    AudioTranscription,
    Command,
    CreateInstance,
    DecideStewardAction,
    DeleteCustomModelCard,
    DeleteDownload,
    DeleteInstance,
    DownloadCommand,
    EvictStagedModel,
    FailInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    ProposeStewardAction,
    RealtimeAudioTranscription,
    SetModelTrustApproval,
    SetTracingEnabled,
    SpeechSynthesis,
    StartDownload,
    TaskCancelled,
    TaskFinished,
    TextEmbedding,
    TextGeneration,
)
from skulk.shared.types.common import CommandId, Id, NodeId, SystemId
from skulk.shared.types.diagnostics import (
    ClusterDiagnostics,
    ClusterNodeDiagnostics,
    ClusterTimeline,
    ClusterTimelineEntry,
    ClusterTimelineRunner,
    ClusterTimelineUnreachable,
    DataPlaneEgressDiagnostics,
    DataPlaneTransport,
    DiagnosticCaptureRequest,
    DiagnosticCaptureResponse,
    DiagnosticProcessSample,
    DiagnosticsProcess,
    DoctorCheckDiagnostics,
    InstancePlacementDiagnostics,
    MlxMemorySnapshot,
    NodeDiagnostics,
    NodeDiagnosticsVersionStatus,
    NodeResourceDiagnostics,
    NodeRuntimeDiagnostics,
    NodeTailscaleDiagnostics,
    PlacementRunnerDiagnostics,
    ProcessRole,
    RunnerSupervisorDiagnostics,
    RunnerTaskCancelRequest,
    RunnerTaskCancelResponse,
    RunnerTaskDiagnostics,
    TelemetryPlaneDiagnostics,
    VisionMediaIngressDiagnostics,
)
from skulk.shared.types.events import (
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    ModelTrustApprovalChanged,
    NodeTimedOut,
    RunnerStatusUpdated,
    StateSnapshotHydrated,
    TaskCreated,
    TaskDeleted,
    TaskFailed,
    TaskStatusUpdated,
    TraceEventData,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import (
    MemoryUsage,
    SystemPerformanceProfile,
    read_wired_memory_bytes,
)
from skulk.shared.types.state import State
from skulk.shared.types.steward_actions import (
    StewardActionProposal,
    StewardActionProposalId,
)
from skulk.shared.types.telemetry import (
    NODE_LIVENESS_TIMEOUT,
    NodeTelemetry,
    TelemetryView,
    record_membership_from_event,
)
from skulk.shared.types.text_generation import (
    InputMessage,
    TextGenerationTaskParams,
)
from skulk.shared.types.worker.downloads import (
    DownloadAttemptId,
    DownloadCompleted,
    DownloadOngoing,
    LiveDownloadProgress,
)
from skulk.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    LlamaRpcInstance,
    instance_meta_of,
)
from skulk.shared.types.worker.runners import RunnerId, RunnerReady, RunnerRunning
from skulk.shared.types.worker.shards import Sharding, ShardMetadata
from skulk.shared.version import get_skulk_version, get_skulk_version_label
from skulk.store.config import (
    SkulkConfig,
    TelemetryConfig,
    load_skulk_config,
    persist_model_trust_config,
    resolve_config_path,
    resolve_node_staging,
    update_skulk_config_atomic,
)
from skulk.tools.web_search import default_browser_tool_provider
from skulk.utils.banner import print_startup_banner
from skulk.utils.channels import Receiver, Sender, channel
from skulk.utils.disk_event_log import DiskEventLog
from skulk.utils.info_gatherer.net_profile import (
    REACHABILITY_ATTEMPTS,
    SWEEP_ATTEMPTS,
    SWEEP_TIMEOUT_SECONDS,
    check_reachable,
    first_reachable_ip,
)
from skulk.utils.power_sampler import PowerSampler
from skulk.utils.task_group import TaskGroup
from skulk.worker.engines.mlx.constants import (
    DEFAULT_KV_CACHE_BACKEND,
    VALID_KV_CACHE_BACKENDS,
)

if TYPE_CHECKING:
    from skulk.store.config import SkulkConfig
    from skulk.store.model_store_client import ModelStoreClient
from skulk.store.artifact_inventory import (
    installed_artifact_roots as _installed_artifact_roots,
)
from skulk.store.artifact_inventory import (
    inventory_installed_artifacts as _inventory_installed_artifacts,
)
from skulk.store.installed_cards import (
    InstalledCardRecord,
    read_installed_card,
    verify_installed_card,
)
from skulk.store.model_store import read_reconciliation_tombstones
from skulk.store.peer_exports import ArtifactExportManager

JsonObject = dict[str, object]
_DEFAULT_OPTIMIZER_CANDIDATE_BITS = [4, 8]
_EXACT_CARD_QUALIFICATION_TOKEN_ENV: Final = "SKULK_EXACT_CARD_QUALIFICATION_TOKEN"
_MINIMUM_QUALIFICATION_TOKEN_LENGTH: Final = 32
_EXACT_CARD_CONVERGENCE_TIMEOUT_SECONDS: Final = 15.0
_EXACT_CARD_CONVERGENCE_POLL_SECONDS: Final = 0.05
_MODEL_LIST_STORE_CACHE_TTL_SECONDS = 5.0
_MODEL_LIST_STORE_FETCH_TIMEOUT_SECONDS = 1.0

#: Chunk type flowing through the performance-envelope tap (duck-typed on
#: ``text`` / ``stats`` / ``finish_reason``, like the field-telemetry tap).
_EnvelopeChunk = TypeVar("_EnvelopeChunk")


class _HypercornServe(Protocol):
    """Typed surface for Hypercorn's asyncio serve entrypoint."""

    def __call__(
        self,
        app: ASGIFramework,
        config: Config,
        *,
        shutdown_trigger: Callable[[], Awaitable[object]] | None = None,
        mode: Literal["asgi", "wsgi"] | None = None,
    ) -> Awaitable[None]: ...


class _TelemetrySender(Protocol):
    """Minimal telemetry publication surface used by the API lifetime task."""

    async def send(self, item: NodeTelemetry) -> None:
        """Offer one latest-value telemetry reading without transport backpressure."""
        ...


serve = cast(_HypercornServe, hypercorn_asyncio.serve)

_API_EVENT_LOG_DIR = SKULK_EVENT_LOG_DIR / "api"

# Ring retention for the API event log. Unlike the master's log (compacted
# after every snapshot), the API log has NO compaction. Historically it
# recorded per-token ChunkGenerated events and grew without bound for the whole
# session (54 MB in 9 idle hours on every node; GBs/day under load; the file a
# node died writing in the 2026-06-06 launch smoke). As of #279 Phase 2 those
# output chunks ride the data plane, NOT the event log, so this log no longer
# carries the per-token firehose — but it still backs GET /events diagnostics
# and is uncompacted, so the ring retention stays as a backstop.
_API_EVENT_LOG_MAX_ACTIVE_BYTES = 256 * 1024 * 1024
_API_EVENT_LOG_COMPACT_KEEP_EVENTS = 20_000
_API_EVENT_LOG_CHECK_INTERVAL_APPENDS = 4096

# Data-plane stream idle backstop (#279 Phase 2b). Output chunks ride the
# best-effort DATA topic (no replay), so a dropped *final* chunk would leave a
# streaming response waiting forever. The timer is armed ONLY after the first
# real output token (it wraps the receive, not the yield, and measures the gap
# between tokens). Time-to-first-token is deliberately unbounded: a request can
# sit behind a long decode or a slow prefill (prefill emits progress chunks that
# are explicitly NOT treated as output and do not arm the timer). So this only
# fires on a genuine mid-stream stall, e.g. a dropped data-plane chunk.
_STREAM_IDLE_TIMEOUT_SECONDS = 120.0

# A completed realtime STT provider call must not release its client-visible
# terminal while authoritative state can still reject the next turn as busy.
# Normal control-plane propagation is sub-second; this bound turns a lost
# terminal/delete event into a typed provider failure instead of an indefinite
# resource hold.
_REALTIME_TASK_RELEASE_TIMEOUT_SECONDS = 10.0

# Trust mutations are synchronous operator decisions. Bound convergence so a
# missing master/event becomes an actionable 503 instead of a false success or
# an indefinitely hanging Settings request.
_MODEL_TRUST_DECISION_TIMEOUT_SECONDS = 10.0

# Largest per-command data-plane reorder window (#279 Phase 2b). The DATA topic
# is best-effort: a genuinely dropped chunk would otherwise stall the reorder
# buffer forever waiting for a sequence that never arrives. Once this many
# out-of-order chunks are held for one command, we give up on the missing
# sequence(s), skip ahead to the lowest buffered one, and drain — bounding memory
# and converting an undeliverable gap into (rare) minor loss rather than a hang.
# Normal mesh reordering spans only a few chunks, far below this.
_MAX_CHUNK_REORDER_BUFFER = 512

# Concurrency bound for extension capability calls served by this node
# (fabric-citizenship Phase 2b): one slow provider must not accumulate
# unbounded in-flight calls on the API node. Calls beyond the bound are
# rejected with the typed `overloaded` error rather than queued.
_MAX_CONCURRENT_CAPABILITY_CALLS = 8
# Provider streams hold their admission slot until a terminal frame, unlike a
# unary call whose slot is released with its result. Keep the same initial cap
# so an extension cannot create unbounded handler tasks or DATA queues.
_MAX_CONCURRENT_CAPABILITY_STREAMS = 8
_PROVIDER_STREAM_RECEIVE_BUFFER = 256
_REALTIME_STT_OUTPUT_BUFFER = 256
_AUDIO_CONTENT_TYPES: Final[dict[AudioResponseFormat, str]] = {
    AudioResponseFormat.Mp3: "audio/mpeg",
    AudioResponseFormat.Wav: "audio/wav",
    AudioResponseFormat.Flac: "audio/flac",
    AudioResponseFormat.Ogg: "audio/ogg",
    AudioResponseFormat.Opus: "audio/opus",
    AudioResponseFormat.Pcm: "audio/pcm",
}
_STREAMABLE_AUDIO_RESPONSE_FORMATS: Final[frozenset[AudioResponseFormat]] = frozenset(
    {AudioResponseFormat.Mp3, AudioResponseFormat.Pcm}
)
_MAX_AUDIO_UPLOAD_BYTES: Final[int] = 25 * 1024 * 1024
_SPEECH_MEDIA_CHUNK_BYTES: Final[int] = 1024 * 1024
_AUDIO_TRANSCRIPTION_FORMATS: Final[set[str]] = {
    "json",
    "text",
    "verbose_json",
    "srt",
    "vtt",
    "ndjson",
}
_AUDIO_UPLOAD_EXTENSIONS: Final[set[str]] = {
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}
_AUDIO_UPLOAD_CONTENT_TYPES: Final[set[str]] = {
    "application/octet-stream",
    "audio/flac",
    "audio/mp3",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/opus",
    "audio/wav",
    "audio/webm",
    "audio/x-m4a",
    "audio/x-wav",
}


class _BuiltinSttInputCancelledError(Exception):
    """Signal caller input cancellation without converting it to provider failure."""


def _normalize_upload_content_type(content_type: str | None) -> str | None:
    """Return the lowercase media type without parameters."""
    if content_type is None:
        return None
    normalized = content_type.split(";", 1)[0].strip().lower()
    return normalized or None


def _select_reconciliation_generations(
    replicas: dict[
        tuple[str, str],
        list[tuple[str, str, dict[str, object]]],
    ],
    current_registry_ids: dict[str, str | None],
) -> dict[
    tuple[str, str],
    list[tuple[str, str, dict[str, object]]],
]:
    """Select exactly one deterministic cache generation per artifact alias.

    Current signed card ownership wins when present. Otherwise reconciliation
    prefers revision-verified bytes and finally the lexical installed identity
    and manifest digest. Source-node preference is applied later within the
    selected generation so replica health cannot change generation truth.
    """

    generations_by_artifact: dict[str, list[tuple[str, str]]] = {}
    for generation, candidates in replicas.items():
        if not candidates:
            continue
        artifact_model_id = candidates[0][2].get("modelId")
        if isinstance(artifact_model_id, str):
            generations_by_artifact.setdefault(artifact_model_id, []).append(generation)

    def generation_rank(generation: tuple[str, str]) -> tuple[bool, bool, str, str]:
        identity, digest = generation
        candidates = replicas[generation]
        item = candidates[0][2]
        artifact_model_id = cast("str", item["modelId"])
        owner_model_id = item.get("ownerModelId")
        owner_alias = (
            owner_model_id if isinstance(owner_model_id, str) else artifact_model_id
        )
        current_registry_id = current_registry_ids.get(owner_alias)
        artifact_role = item.get("artifactRole")
        retained_registry_id = (
            item.get("registryCardId")
            if artifact_role == "base" or not isinstance(artifact_role, str)
            else item.get("ownerCardId")
        )
        is_current = (
            current_registry_id is not None
            and retained_registry_id == current_registry_id
        )
        is_registry_verified = any(
            candidate[2].get("verificationState") == "registry_verified"
            for candidate in candidates
        )
        return (not is_current, not is_registry_verified, identity, digest)

    selected: dict[
        tuple[str, str],
        list[tuple[str, str, dict[str, object]]],
    ] = {}
    for generations in generations_by_artifact.values():
        generation = min(generations, key=generation_rank)
        selected[generation] = replicas[generation]
    return selected


def _combined_model_advisories(
    model_cards: Iterable[ModelCard],
) -> tuple[RegistryAdvisory, ...]:
    """Return deduplicated warnings across installed and current generations."""

    advisories_by_id = {
        advisory.advisory_id: advisory
        for model_card in model_cards
        for advisory in get_model_advisories(model_card)
    }
    return tuple(advisories_by_id.values())


def _complete_canonical_generations(
    store_root: Path,
    registry: Iterable[dict[str, object]],
) -> set[tuple[str, str]]:
    """Return identity and manifest pairs complete in the canonical store."""

    resolved_root = store_root.expanduser().resolve()
    generations: set[tuple[str, str]] = set()
    for entry in registry:
        installed_raw = entry.get("installed_card")
        store_path = entry.get("store_path")
        if not isinstance(installed_raw, dict) or not isinstance(store_path, str):
            continue
        try:
            record = InstalledCardRecord.model_validate(installed_raw, strict=False)
            canonical_directory = (resolved_root / store_path).resolve()
            if not canonical_directory.is_relative_to(resolved_root):
                continue
            adjacent = read_installed_card(canonical_directory)
        except (OSError, ValueError):
            continue
        if adjacent == record and verify_installed_card(canonical_directory, record):
            generations.add((record.installed_identity, record.manifest_sha256))
    return generations


def _artifact_inventory_is_tombstoned(
    item: dict[str, object],
    tombstones: frozenset[str],
) -> bool:
    """Return whether an inventory item belongs to an operator-deleted alias.

    Companion artifacts inherit suppression from their owning base-card alias,
    while independently deleted artifact aliases are suppressed directly.

    Args:
        item: Camel-case node inventory entry.
        tombstones: Durable aliases excluded from automatic imports.

    Returns:
        ``True`` when reconciliation must retain but not import this replica.
    """

    model_id = item.get("modelId")
    owner_model_id = item.get("ownerModelId")
    return (isinstance(model_id, str) and model_id in tombstones) or (
        isinstance(owner_model_id, str) and owner_model_id in tombstones
    )


def _validate_audio_upload_metadata(file: StarletteUploadFile) -> None:
    """Reject uploads whose metadata is clearly not an audio container."""
    content_type = _normalize_upload_content_type(file.content_type)
    if file.filename is not None and len(file.filename) > 255:
        raise HTTPException(status_code=422, detail="Audio filename is too long")
    if content_type is not None and len(content_type) > 255:
        raise HTTPException(status_code=422, detail="Audio content type is too long")
    suffix = Path(file.filename or "").suffix.lower()
    if (
        content_type is not None
        and content_type not in _AUDIO_UPLOAD_CONTENT_TYPES
        and suffix not in _AUDIO_UPLOAD_EXTENSIONS
    ):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio upload content type: {content_type}",
        )


async def _read_audio_upload(file: StarletteUploadFile) -> bytes:
    """Read an uploaded audio file with Skulk's non-streaming size cap."""
    audio_bytes = await file.read(_MAX_AUDIO_UPLOAD_BYTES + 1)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio upload is empty")
    if len(audio_bytes) > _MAX_AUDIO_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "Audio upload exceeds Skulk's "
                f"{_MAX_AUDIO_UPLOAD_BYTES} byte transcription limit"
            ),
        )
    return audio_bytes


def _decode_audio_chunk_data(chunk: AudioChunk) -> bytes:
    """Decode one independently base64-encoded audio output chunk."""
    try:
        return base64.b64decode(chunk.data.encode("ascii"), validate=True)
    except binascii.Error as exc:
        raise HTTPException(
            status_code=500,
            detail="Speech runner returned invalid base64 audio",
        ) from exc


def _parse_timestamp_granularities(value: str | None) -> tuple[str, ...]:
    """Parse multipart timestamp granularity hints from JSON or comma syntax."""
    if value is None or not value.strip():
        return ()
    stripped = value.strip()
    try:
        parsed = cast(object, json.loads(stripped))
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        parsed_items = cast(list[object], parsed)
        if all(isinstance(item, str) for item in parsed_items):
            return tuple(cast(list[str], parsed_items))
    return tuple(item.strip() for item in stripped.split(",") if item.strip())


def _segment_float(
    segment: dict[str, str | int | float | bool | None], key: str
) -> float | None:
    value = segment.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _transcript_segments_or_fallback(
    text: str,
    segments: list[dict[str, str | int | float | bool | None]],
) -> list[dict[str, str | int | float | bool | None]]:
    if segments:
        return segments
    if not text:
        return []
    return [{"id": 0, "text": text, "start": 0.0, "end": 0.0}]


def _format_srt_timestamp(seconds: float) -> str:
    milliseconds_total = max(0, int(round(seconds * 1000)))
    milliseconds = milliseconds_total % 1000
    seconds_total = milliseconds_total // 1000
    second = seconds_total % 60
    minutes_total = seconds_total // 60
    minute = minutes_total % 60
    hour = minutes_total // 60
    return f"{hour:02}:{minute:02}:{second:02},{milliseconds:03}"


def _format_transcript_srt(
    text: str, segments: list[dict[str, str | int | float | bool | None]]
) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(
        _transcript_segments_or_fallback(text, segments), 1
    ):
        segment_text = segment.get("text")
        if not isinstance(segment_text, str):
            segment_text = ""
        start = _segment_float(segment, "start") or 0.0
        end = _segment_float(segment, "end")
        if end is None or end < start:
            end = start
        blocks.append(
            f"{index}\n"
            f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n"
            f"{segment_text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _format_transcript_vtt(
    text: str, segments: list[dict[str, str | int | float | bool | None]]
) -> str:
    body = _format_transcript_srt(text, segments)
    lines = body.splitlines()
    cleaned_lines = [line for line in lines if not line.isdigit()]
    return (
        "WEBVTT\n\n"
        + "\n".join(
            line.replace(",", ".") if " --> " in line else line
            for line in cleaned_lines
        )
        + ("\n" if cleaned_lines else "")
    )


def _transcription_chunk_payload(chunk: TranscriptionChunk) -> dict[str, object]:
    """Return the public JSON shape for one transcription chunk."""
    payload: dict[str, object] = {"text": chunk.text}
    if chunk.language is not None:
        payload["language"] = chunk.language
    if chunk.segments:
        payload["segments"] = chunk.segments
    if chunk.finish_reason is not None:
        payload["finish_reason"] = chunk.finish_reason
    return payload


def _build_audio_transcription_response(
    response_format: AudioTranscriptionResponseFormat,
    chunks: list[TranscriptionChunk],
) -> Response:
    """Format collected transcription chunks as an OpenAI-compatible response."""
    text = "".join(chunk.text for chunk in chunks).strip()
    language = next(
        (chunk.language for chunk in chunks if chunk.language is not None), None
    )
    segments: list[dict[str, str | int | float | bool | None]] = []
    for chunk in chunks:
        segments.extend(chunk.segments)

    if response_format == "text":
        return Response(content=text, media_type="text/plain; charset=utf-8")
    if response_format == "srt":
        return Response(
            content=_format_transcript_srt(text, segments),
            media_type="application/x-subrip; charset=utf-8",
        )
    if response_format == "vtt":
        return Response(
            content=_format_transcript_vtt(text, segments),
            media_type="text/vtt; charset=utf-8",
        )
    if response_format == "ndjson":
        content = "".join(
            json.dumps(_transcription_chunk_payload(chunk), separators=(",", ":"))
            + "\n"
            for chunk in chunks
        )
        return Response(content=content, media_type="application/x-ndjson")
    if response_format == "verbose_json":
        payload: dict[str, object] = {"text": text, "segments": segments}
        if language is not None:
            payload["language"] = language
        return JSONResponse(payload)
    return JSONResponse({"text": text})


def _encode_audio_transcription_sse(
    event: AudioTranscriptionStreamEvent,
) -> str:
    """Encode one typed transcription event as an SSE record."""

    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


# How long the reorder buffer waits for a missing sequence before giving up on
# it and releasing the chunks behind the gap (#279 Phase 2b). A genuine mesh
# reorder resolves in milliseconds; a sequence still missing after this was
# dropped on the best-effort topic. Without a time bound, a dropped sequence
# that the size cap never reaches (e.g. seq 0 lost on a short response, or a
# trailing gap with no later chunk to re-trigger the size check) would hold all
# the chunks behind it forever — and the stream's own idle backstop never arms
# because nothing has been yielded yet. A periodic sweep flushes such gaps.
_REORDER_GAP_FLUSH_SECONDS = 5.0
_VISION_MEDIA_PENDING_COMMANDS = 64
_VISION_MEDIA_PENDING_FRAMES = 64
_VISION_MEDIA_PENDING_COMMAND_BYTES = 32 * 1024 * 1024
_VISION_MEDIA_PENDING_TOTAL_BYTES = 512 * 1024 * 1024
_VISION_MEDIA_PENDING_TTL_SECONDS = 5 * 60.0
_VISION_MEDIA_ACK_TIMEOUT_SECONDS = 5 * 60.0
_VISION_MEDIA_RAW_IMAGE_BYTES = (_VISION_MEDIA_PENDING_COMMAND_BYTES // 4) * 3
_SPEECH_MEDIA_PENDING_COMMANDS = 64
_SPEECH_MEDIA_PENDING_TOTAL_BYTES = 512 * 1024 * 1024
_SPEECH_MEDIA_PENDING_TTL_SECONDS = 5 * 60.0
_TRACE_DATA_PENDING_TASKS = 256
_TRACE_DATA_PENDING_TTL_SECONDS = 5 * 60.0


@dataclass
class _ChunkReorderState:
    """Per-command reorder cursor for the data plane (#279 Phase 2b)."""

    next_seq: int = 0
    pending: dict[int, DataChunk] = field(default_factory=dict)
    # Monotonic time the current head-of-line gap was first observed (chunks
    # buffered above next_seq that couldn't drain). None when there is no gap.
    # `gap_at` is the next_seq the timer started waiting on, so later chunks
    # arriving behind the *same* gap don't refresh the timer (a moving stream
    # resets it; a stuck one ages to the flush window).
    gap_since: float | None = None
    gap_at: int | None = None


@dataclass
class _ProviderStreamReceiveState:
    """Caller-side queue and lifecycle state for one provider stream."""

    output_sender: Sender[CapabilityStreamFrame]
    receiver: CapabilityStreamReceiver
    call: CapabilityCall
    deadline_at: float
    input_stream: CapabilityStreamInput | None = None
    transport_failure: str | None = None
    cancel_provider: bool = False
    cancellation_scheduled: bool = False


@dataclass
class _ActiveProviderStream:
    """Provider-side lifecycle state for one admitted handler."""

    caller_node: str
    cancel_requested: anyio.Event
    descriptor: CapabilityDescriptor
    cancel_message: str = "caller cancelled the provider stream"
    cancel_scope: anyio.CancelScope | None = None
    input_sender: Sender[CapabilityStreamFrame] | None = None
    input_receiver: Receiver[CapabilityStreamFrame] | None = None
    input_lifecycle: CapabilityStreamReceiver | None = None
    input_failure: CapabilityStreamError | None = None
    reserved_instance_id: InstanceId | None = None


@dataclass(frozen=True, slots=True)
class _PendingVisionMedia:
    """Request-scoped image chunks awaiting authoritative task placement."""

    model: ModelId
    chunks: tuple[tuple[int, bytes], ...]
    image_count: int
    sha256: str
    created_at: float

    @property
    def byte_count(self) -> int:
        """Return serialized media bytes retained by this request."""

        return sum(len(data) for _, data in self.chunks)


@dataclass(frozen=True, slots=True)
class _PendingSpeechMedia:
    """Request-scoped batch audio awaiting authoritative task placement."""

    model: ModelId
    data: bytes
    filename: str | None
    content_type: str | None
    sha256: str
    created_at: float


@dataclass(slots=True)
class _PendingTraceData:
    """Bounded owner-side assembly for one multi-rank task trace."""

    expected_ranks: frozenset[int]
    traces_by_rank: dict[int, tuple[TraceEventData, ...]]
    updated_at: float


# Task statuses for which the runner has stopped producing — a mid-stream idle
# stall on such a command is a dropped FINAL chunk, not a stuck runner, so it
# must clean up via the normal TaskFinished path, not TaskCancelled (#298 review).
_TERMINAL_TASK_STATUSES: frozenset[task_types.TaskStatus] = frozenset(
    {
        task_types.TaskStatus.Complete,
        task_types.TaskStatus.Failed,
        task_types.TaskStatus.TimedOut,
        task_types.TaskStatus.Cancelled,
    }
)

# How long POST /place_instance waits for node memory info to finish
# gossiping before giving up with 503. Covers the window between cluster
# formation (identities present) and the first NodeGatheredInfo from every
# node — observed to resolve within a few seconds on a LAN cluster.
_PLACEMENT_INFO_WAIT_SECONDS = 15.0
ONBOARDING_COMPLETE_FILE = SKULK_CACHE_HOME / "onboarding_complete"

# A node with no live runners should sit near its idle wired baseline
# (~2GB on the M4 kites). Wired well above that with nothing running means
# memory leaked from an abnormal Metal termination (warmup-wedge SIGKILL,
# GPU-timeout abort) that only a reboot reclaims (Skulk#239). The threshold
# is a generous floor over the idle baseline; the signal is the leak, not a
# precise number.
_LEAKED_WIRED_THRESHOLD_BYTES = 5 * 1024 * 1024 * 1024


def _leaked_wired_warning(
    wired: "Memory | None",
    supervisor_runners: "Sequence[RunnerSupervisorDiagnostics]",
) -> str | None:
    """Flag likely-leaked wired memory: high wired with no live runners.

    Returns a human-readable warning, or None. macOS-only in practice (wired
    is None elsewhere). Mirrors the test-side preflight
    (tests/preflight_mem.sh): a poisoned node otherwise surfaces only as
    unexplained placement 400s and decode GPU-timeouts (Skulk#236), so this
    gives operators the actual cause — reboot to reclaim.
    """
    if wired is None:
        return None
    if any(runner.process_alive for runner in supervisor_runners):
        return None
    wired_bytes = wired.in_bytes
    if wired_bytes <= _LEAKED_WIRED_THRESHOLD_BYTES:
        return None
    return (
        f"Likely leaked wired memory: {wired_bytes / 2**30:.1f} GB wired with "
        "no live runners. An abnormal Metal termination (warmup wedge or GPU "
        "timeout) can leave wired memory the OS only reclaims on reboot, "
        "causing false 'insufficient memory' placements and decode timeouts. "
        "Reboot this node to reclaim it."
    )


def prune_old_trace_files(directory: Path, retention_days: int, now: float) -> int:
    """Delete saved trace files in *directory* whose mtime is older than the
    retention window.

    Pure function — extracted from :meth:`API._prune_old_traces` so unit tests
    can exercise the pruning logic without spinning up the API class. The
    janitor task is responsible for scheduling and config plumbing; this
    function only owns the "should this file go" decision.

    Args:
        directory: Path the janitor watches. Missing directories are treated
            as empty (return 0) — no error.
        retention_days: Files older than this many days are removed. Values
            <= 0 disable pruning entirely (return 0).
        now: Current Unix timestamp in seconds. Passed in so tests can pin the
            clock instead of relying on real time.

    Returns:
        The number of files that were successfully removed. Files that fail
        to delete (permission errors, races) are logged and counted out of
        the total — they are not raised, so a single corrupted entry can't
        crash the janitor loop.
    """
    if retention_days <= 0:
        return 0
    if not directory.exists():
        return 0
    cutoff = now - retention_days * 86400
    removed = 0
    for path in directory.glob("trace_*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError as err:
            logger.warning(f"Failed to prune trace {path}: {err}")
    return removed


def _text_image_hash_cache_enabled() -> bool:
    value = preferred_env_value(
        "SKULK_TEXT_IMAGE_HASH_CACHE",
        default="false",
    )
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def validate_renderable_text_generation(
    task_params: TextGenerationTaskParams,
) -> None:
    """Reject text-generation requests the runner cannot render.

    Raises ``HTTPException(400)`` so malformed input fails at the API instead
    of reaching the runner. An empty message set produced an empty prompt
    that crashed the runner inside ``apply_chat_template`` with an IndexError
    (#233); a non-positive ``max_tokens`` would request zero/negative output.
    Both are caught here, the single dispatch chokepoint shared by every wire
    format, so no adapter can route un-renderable work to a runner.
    """
    # Mirror the runner's renderability test EXACTLY (utils_mlx builds
    # formatted_messages): pre-formatted chat_template_messages or
    # instructions render as-is, but the runner drops input messages whose
    # content is falsy (`if not msg.content: continue`) before templating,
    # so an empty resulting list crashes apply_chat_template with an
    # IndexError. The chat adapter substitutes a single empty-string
    # message for an empty `messages` array, which the runner then filters
    # back to nothing — so "input is non-empty" is not a sufficient check.
    # Use the SAME truthiness test as the runner (not `.strip()`): a
    # whitespace-only message is truthy and the runner renders it, so the
    # guard must not reject inputs the runner can handle.
    has_renderable = (
        bool(task_params.chat_template_messages)
        or bool(task_params.instructions)
        or any(msg.content for msg in task_params.input)
    )
    if not has_renderable:
        raise HTTPException(
            status_code=400,
            detail=(
                "the request has no content to generate from "
                "(messages / input must contain a non-empty message)"
            ),
        )
    if task_params.max_output_tokens is not None and task_params.max_output_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail=("max_tokens (max_output_tokens) must be a positive integer"),
        )


API_TAGS_METADATA = [
    {
        "name": "Compatibility APIs",
        "description": "Endpoints that let existing SDKs and tools talk to Skulk using OpenAI, Claude, or Ollama-style request formats.",
    },
    {
        "name": "Models",
        "description": "Model discovery, search, listing, and custom model-card management.",
    },
    {
        "name": "Instances",
        "description": "Placement previews, launch flows, instance lookup, and lifecycle management for running models.",
    },
    {
        "name": "Downloads",
        "description": "Low-level node download control and staging-cache management.",
    },
    {
        "name": "Store",
        "description": "Shared model-store health, registry inspection, download workflows, deletion, and optimization.",
    },
    {
        "name": "Tools",
        "description": "Builtin tool endpoints that clients can execute and feed back into model conversations.",
    },
    {
        "name": "Config",
        "description": "Cluster configuration, safe filesystem browsing, and node identity helpers used by the dashboard.",
    },
    {
        "name": "State & Tracing",
        "description": "Cluster state, event log access, trace inspection/export, and onboarding helpers.",
    },
    {
        "name": "Diagnostics",
        "description": "Read-only local and cluster diagnostics for stuck runner, placement, resource, and process inspection.",
    },
    {
        "name": "Images",
        "description": "Image generation, editing, retrieval, and benchmarking endpoints.",
    },
    {
        "name": "Admin",
        "description": "Administrative operations such as node restart.",
    },
    {
        "name": "Authentication",
        "description": "Host-authorized device pairing and operator credential lifecycle.",
    },
]


def _format_to_content_type(image_format: Literal["png", "jpeg", "webp"] | None) -> str:
    return f"image/{image_format or 'png'}"


def _ensure_seed(params: AdvancedImageParams | None) -> AdvancedImageParams:
    """Ensure advanced params has a seed set for distributed consistency."""
    if params is None:
        return AdvancedImageParams(seed=random.randint(0, 2**32 - 1))
    if params.seed is None:
        return params.model_copy(update={"seed": random.randint(0, 2**32 - 1)})
    return params


def _log_image_transport(message: str) -> None:
    """Emit verbose image transport diagnostics only when explicitly enabled."""
    if SKULK_IMAGE_TRANSPORT_DEBUG:
        logger.info(message)
    else:
        logger.debug(message)


def _coerce_json_object(value: object) -> JsonObject:
    """Return a string-keyed object dict or an empty mapping for non-objects."""
    if not isinstance(value, dict):
        return {}
    raw_dict = cast(dict[object, object], value)
    return {str(key): item for key, item in raw_dict.items()}


async def _read_request_json_object(request: Request) -> JsonObject:
    """Parse one request body into a string-keyed JSON object."""
    payload = cast(object, await request.json())
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=422,
            detail="Request body must be a JSON object.",
        )
    return _coerce_json_object(cast(dict[object, object], payload))


def _load_yaml_object(path: Path) -> JsonObject:
    """Read one YAML document from disk as a string-keyed object."""
    with path.open() as handle:
        return _coerce_json_object(cast(object, yaml.safe_load(handle)))


def _coerce_float(value: object, *, default: float) -> float:
    """Normalize numeric request values to float with a conservative default."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_candidate_bits(value: object) -> list[int]:
    """Normalize OptiQ candidate bits to a clean integer list."""
    if not isinstance(value, list):
        return list(_DEFAULT_OPTIMIZER_CANDIDATE_BITS)
    raw_items = cast(list[object], value)
    normalized = [
        item
        for item in raw_items
        if isinstance(item, int) and not isinstance(item, bool)
    ]
    return normalized or list(_DEFAULT_OPTIMIZER_CANDIDATE_BITS)


def _create_fastapi_app() -> FastAPI:
    return FastAPI(
        title="Skulk API",
        summary="Distributed inference, placement, store, and compatibility APIs for Skulk.",
        description=(
            "Skulk exposes OpenAI-compatible inference APIs, cluster placement controls, "
            "model-store workflows, configuration endpoints, and debugging endpoints. "
            "Most text-generation requests require the target model to already be placed "
            "and running on the node or cluster."
        ),
        version=get_skulk_version(),
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_tags=API_TAGS_METADATA,
    )


def _json_request_body(schema: dict[str, object]) -> dict[str, object]:
    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": schema,
                }
            },
        }
    }


def _audio_speech_request_body() -> dict[str, object]:
    """Describe the JSON and multipart forms accepted by the speech route."""

    json_schema = cast(dict[str, object], AudioSpeechRequest.model_json_schema())
    multipart_schema = cast(dict[str, object], json.loads(json.dumps(json_schema)))
    properties = cast(dict[str, object], multipart_schema.get("properties", {}))
    properties["reference_audio"] = {
        "type": "string",
        "format": "binary",
        "description": "Bounded request-scoped reference audio upload.",
    }
    multipart_schema["properties"] = properties
    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": json_schema},
                "multipart/form-data": {"schema": multipart_schema},
            },
        }
    }


def _steward_canary_failure_command(instance_id: InstanceId) -> FailInstance:
    """Return safe terminal truth after repeated steward canary failures."""
    return FailInstance(
        instance_id=instance_id,
        error_code="runner_unresponsive",
        error_message=(
            "The fabric steward failed three consecutive health probes, so "
            "Skulk tore down the placement for automatic recovery."
        ),
    )


class API:
    def __init__(
        self,
        node_id: NodeId,
        *,
        port: int,
        event_receiver: Receiver[IndexedEvent],
        command_sender: Sender[ForwarderCommand],
        download_command_sender: Sender[ForwarderDownloadCommand],
        # This lets us pause the API if an election is running
        election_receiver: Receiver[ElectionMessage],
        skulk_config: "SkulkConfig | None" = None,
        store_client: "ModelStoreClient | None" = None,
        enable_event_log: bool = True,
        mount_dashboard: bool = True,
        telemetry_view: "TelemetryView | None" = None,
        telemetry_sender: _TelemetrySender | None = None,
        data_receiver: "Receiver[DataChunk] | None" = None,
        provider_stream_sender: "Sender[ProviderStreamPacket] | None" = None,
        provider_stream_receiver: "Receiver[ProviderStreamPacket] | None" = None,
        realtime_audio_sender: "Sender[RealtimeAudioInputFrame] | None" = None,
        realtime_audio_packet_sender: "Sender[RealtimeAudioPacket] | None" = None,
        realtime_audio_packet_receiver: "Receiver[RealtimeAudioPacket] | None" = None,
        speech_media_packet_sender: "Sender[SpeechMediaPacket] | None" = None,
        speech_media_packet_receiver: "Receiver[SpeechMediaPacket] | None" = None,
        trace_data_receiver: "Receiver[TraceDataPacket] | None" = None,
        vision_media_packet_sender: "Sender[VisionMediaPacket] | None" = None,
        vision_media_packet_receiver: "Receiver[VisionMediaPacket] | None" = None,
        data_plane_zenoh: bool = False,
        data_plane_egress_provider: (
            Callable[[], DataPlaneEgressDiagnostics] | None
        ) = None,
        vision_media_egress_provider: (
            Callable[[], DataPlaneEgressDiagnostics] | None
        ) = None,
        telemetry_plane_provider: (
            Callable[[], TelemetryPlaneDiagnostics] | None
        ) = None,
        extensions: LoadedExtensions | None = None,
        enable_builtin_providers: bool = False,
        operator_pairing_service: OperatorPairingService | None = None,
        apply_custom_card_mutations_locally: bool = False,
    ) -> None:
        self.state = State()
        self._apply_custom_card_mutations_locally = apply_custom_card_mutations_locally
        self._operator_pairing_service = operator_pairing_service
        self._operator_relay_configuration: OperatorRelayConfiguration | None = None
        if operator_pairing_service is not None:
            try:
                self._operator_relay_configuration = (
                    operator_pairing_service.relay_configuration()
                )
            except (OSError, RuntimeError, ValueError, sqlite3.Error):
                # Relay state is an optional ingress path. A damaged local
                # authority record must fail remote access closed without
                # turning that optional failure into cluster downtime.
                logger.warning(
                    "Operator relay configuration is unavailable; "
                    "continuing with local API access only"
                )
        # External extensions remain optional. Production nodes prepend
        # first-party provider facades that expose core services through the
        # same generic contracts without duplicating their runtimes.
        self._builtin_speech_provider_enabled = enable_builtin_providers
        if enable_builtin_providers:
            loaded_extensions = extensions or LoadedExtensions([])
            self._extensions: LoadedExtensions | None = (
                loaded_extensions.with_builtin_extensions(
                    (
                        BuiltinSpeechProvider(
                            admit_tts=self._admit_builtin_tts_stream,
                            stream_tts=self._stream_builtin_tts,
                            admit_stt=self._admit_builtin_stt_stream,
                            stream_stt=self._stream_builtin_stt,
                            admit_realtime_stt=(
                                self._admit_builtin_realtime_stt_stream
                            ),
                            stream_realtime_stt=self._stream_builtin_realtime_stt,
                        ),
                        BuiltinVadProvider(),
                    )
                )
            )
        else:
            self._extensions = extensions
        # (timestamp, result) of the last tailscale diagnostics probe; see
        # _tailscale_diagnostics for the TTL rationale.
        self._tailscale_diag_cache: (
            tuple[float, NodeTailscaleDiagnostics | None] | None
        ) = None
        self._extension_context = ExtensionContext(
            node_id=node_id,
            skulk_version=resolve_skulk_version(),
            embed_texts=self.embed_texts,
            # Telemetry-plane read access (fabric-citizenship Phase 1): snapshot
            # the live view at call time. `self._telemetry_view` is assigned just
            # below, so the closure reads it lazily, never at construction.
            read_cluster=lambda: snapshot_cluster(self._telemetry_view),
            # Telemetry-plane advertise access (fabric-citizenship Phase 1): record
            # the tag on the shared view's outbound set; the worker's info gatherer
            # gossips it on its next poll. Reads self._telemetry_view lazily too.
            advertise_capability=self._advertise_capability,
            withdraw_capability=self._withdraw_capability,
            # Capability discovery, heavy half (fabric-citizenship Phase 2a):
            # full descriptors on demand, local or via a reachable peer API.
            describe_node=self._describe_node_capabilities,
            # The generic call verb (Phase 2b): node-addressed, master never in
            # the hot path, typed results; local target is an in-process fast
            # path with the same guards.
            call_capability=self._call_capability,
            # Phase 3 streaming directions travel on the provider DATA topic.
            # The extension-facing callable remains transport-abstract.
            stream_capability=self._stream_capability,
        )
        # In-flight extension capability calls served by this node; bounded by
        # _MAX_CONCURRENT_CAPABILITY_CALLS (single-threaded loop => plain int).
        self._active_capability_calls = 0
        self._active_capability_streams: dict[str, _ActiveProviderStream] = {}
        self._provider_stream_sender = provider_stream_sender
        self._provider_stream_receiver = provider_stream_receiver
        self._realtime_audio_sender = realtime_audio_sender
        self._realtime_audio_packet_sender = realtime_audio_packet_sender
        self._realtime_audio_packet_receiver = realtime_audio_packet_receiver
        self._speech_media_packet_sender = speech_media_packet_sender
        self._speech_media_packet_receiver = speech_media_packet_receiver
        self._speech_media_commands: set[CommandId] = set()
        self._speech_media_targets: dict[CommandId, NodeId] = {}
        self._transcription_media_targets: dict[CommandId, tuple[NodeId, ...]] = {}
        self._pending_speech_media: dict[CommandId, _PendingSpeechMedia] = {}
        self._pending_speech_media_bytes = 0
        self._trace_data_receiver = trace_data_receiver
        self._pending_trace_data: dict[task_types.TaskId, _PendingTraceData] = {}
        self._vision_media_packet_sender = vision_media_packet_sender
        self._vision_media_packet_receiver = vision_media_packet_receiver
        self._pending_vision_media: dict[CommandId, _PendingVisionMedia] = {}
        self._pending_vision_media_bytes = 0
        self._active_vision_media_bytes: dict[CommandId, int] = {}
        self._active_vision_media_total_bytes = 0
        self._vision_media_commands: set[CommandId] = set()
        self._vision_media_targets: dict[CommandId, tuple[NodeId, ...]] = {}
        self._vision_media_pending_acks: dict[CommandId, set[NodeId]] = {}
        self._vision_media_ack_deadlines: dict[CommandId, float] = {}
        self._vision_media_models: dict[CommandId, ModelId] = {}
        self._vision_media_failures: dict[CommandId, ErrorChunk] = {}
        self._data_plane_zenoh = data_plane_zenoh
        self._provider_stream_receivers: dict[str, _ProviderStreamReceiveState] = {}
        # Data plane (#279 Phase 2): per-token output chunks arrive here direct
        # from the serving worker (DATA topic), not as ChunkGenerated events off
        # the master. Demuxed by command_id into the per-command stream queues.
        # None means no DATA topic wired (tests); streaming then has no source.
        self._data_receiver = data_receiver
        # Node telemetry off the event log (#279): placement previews read
        # node_resources from the live view, matching what the master enforces.
        self._telemetry_view = (
            telemetry_view if telemetry_view is not None else TelemetryView()
        )
        self._telemetry_sender = telemetry_sender
        self._artifact_inventory_expected_since: dict[NodeId, float] = {}
        self._artifact_inventory_nodes_seen: set[NodeId] = set()
        # Provider extensions (fabric-citizenship Phase 2a): auto-advertise
        # each served capability's id as its telemetry discovery tag. Startup
        # hooks run in run(), after construction succeeds and on the serving
        # loop; constructors must not launch unowned background work.
        if self._extensions is not None:
            for descriptor in self._extensions.capability_descriptors:
                self._telemetry_view.local_advertised_capabilities.add(descriptor.id)

        # Built-in speech availability is synchronous core state, not plugin
        # background startup. Preserve empty-capacity withdrawal at construction.
        self._sync_builtin_speech_capability()
        # Steward liveness history, shared between the canary loop that writes
        # it and the status endpoint that reads it (both event-loop tasks on
        # this node), so an outstanding failed probe surfaces as `degraded`
        # instead of hiding until the third failure tears the steward down.
        self._steward_canary = StewardCanaryState()
        self._event_log = DiskEventLog(_API_EVENT_LOG_DIR) if enable_event_log else None
        self._event_log_appends_since_retention_check = 0
        self._system_id = SystemId()
        self.command_sender = command_sender
        self.download_command_sender = download_command_sender
        self.event_receiver = event_receiver
        self.election_receiver = election_receiver
        self.node_id: NodeId = node_id
        self._master_node_id: NodeId = node_id
        self.last_completed_election: int = 0
        self.port = port
        self._skulk_config = skulk_config
        # Terminal failures that arrived before their command's stream queue
        # was registered (#223 follow-up): the low-level chunk stream
        # registers its queue lazily on first iteration, so a fast TaskFailed
        # (e.g. a pinned steward request whose instance vanished, on a
        # single-node cluster where the master round-trip is local) can beat
        # registration and _terminate_command_stream would silently no-op,
        # hanging the caller. Bounded FIFO; consumed at stream registration.
        self._pending_stream_failures: dict[CommandId, ErrorChunk] = {}
        self._store_client = store_client
        self._model_list_store_records_cache: dict[ModelId, InstalledCardRecord] = {}
        self._model_list_store_records_cached_at = 0.0
        self._model_list_store_records_lock = asyncio.Lock()
        self._artifact_exports = ArtifactExportManager()
        reconciliation_config = (
            skulk_config.model_store.reconciliation
            if skulk_config is not None and skulk_config.model_store is not None
            else None
        )
        self._reconciliation_status = ReconciliationStatus(
            inventory_only=(
                reconciliation_config.inventory_only
                if reconciliation_config is not None
                else True
            )
        )
        self._reconciliation_lock = asyncio.Lock()
        self._config_path = resolve_config_path()
        # Field telemetry (opt-in, consent-gated in skulk.yaml). The config
        # provider re-reads the file per check so dashboard consent changes
        # apply without a restart; the hardware provider snapshots the
        # telemetry plane. Node ids feed only in-process death diffing.
        self._telemetry_config_cache: "TelemetryConfig | None" = None
        self._telemetry_config_cached_until = 0.0
        self._field_telemetry = FieldTelemetryCollector(
            config_provider=self._current_telemetry_config,
            hardware_provider=self._telemetry_hardware_snapshot,
        )
        # Observe-only performance envelopes (adaptive concurrency, Phase 0): the
        # throughput/latency-vs-concurrency curve per (hardware x model x engine x
        # quant), fed one observation per completed generation. In-memory,
        # bounded, off State/event-log/telemetry-plane; exposed only via a
        # read-only diagnostics endpoint.
        self._performance_envelopes = PerformanceEnvelopeRegistry(
            now=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        # This API node's outstanding requests per model, the concurrency signal
        # captured at each request's admission.
        self._envelope_inflight: dict[str, int] = {}
        self._model_optimizer: "ModelOptimizer | None" = None
        self._runner_diagnostics_provider: (
            Callable[[], Sequence[RunnerSupervisorDiagnostics]] | None
        ) = None
        self._runner_cancel_provider: (
            Callable[[RunnerId, task_types.TaskId], Awaitable[RunnerTaskCancelResponse]]
            | None
        ) = None
        self._vision_media_ingress_provider: (
            Callable[[], VisionMediaIngressDiagnostics] | None
        ) = None
        self._sent_image_hashes: set[str] = set()
        # Initialize optimizer if store path is available
        if (
            skulk_config
            and skulk_config.model_store
            and skulk_config.model_store.enabled
        ):
            from skulk.store.model_optimizer import ModelOptimizer

            self._model_optimizer = ModelOptimizer(
                store_path=Path(skulk_config.model_store.store_path)
            )

        self.paused: bool = False
        self.paused_ev: anyio.Event = anyio.Event()

        self.app = _create_fastapi_app()

        async def log_requests_middleware(
            request: Request,
            call_next: Callable[[Request], Awaitable[StreamingResponse]],
        ) -> StreamingResponse:
            logger.debug(f"API request: {request.method} {request.url.path}")
            return await call_next(request)

        self.app.middleware("http")(log_requests_middleware)
        self._setup_exception_handlers()
        self._setup_cors()
        self._setup_routes()
        if operator_pairing_service is not None:
            self.app.include_router(
                create_operator_auth_router(operator_pairing_service)
            )

        # A headless/worker node may have no built dashboard assets
        # (DASHBOARD_DIR is None); serving the UI is then skipped so the node
        # still boots and serves the API (#333). The control/data planes and
        # all API routes are unaffected.
        if mount_dashboard and DASHBOARD_DIR is not None:
            dashboard_dir = DASHBOARD_DIR

            # The dashboard is a SPA that restores its active view from the
            # URL path (App.tsx reads location.pathname), so deep links and
            # browser refreshes hit these paths directly. StaticFiles only
            # serves index.html at "/" — without an explicit fallback,
            # refreshing on /chat lands on a bare {"detail":"Not Found"}.
            # Keep this list in sync with NavRoute in
            # dashboard-react/src/components/layout/HeaderNav.tsx.
            async def _spa_index() -> FileResponse:
                """Serve the dashboard SPA shell for client-routed paths."""
                return FileResponse(os.path.join(dashboard_dir, "index.html"))

            # Both slash forms: the StaticFiles mount at "/" swallows
            # unmatched paths before FastAPI's redirect-slashes logic can
            # normalize "/chat/" to "/chat".
            for _spa_route in (
                "/cluster",
                "/model-store",
                "/chat",
                "/steward",
                "/integrations",
                "/operator",
            ):
                self.app.get(_spa_route, include_in_schema=False)(_spa_index)
                self.app.get(f"{_spa_route}/", include_in_schema=False)(_spa_index)

            self.app.mount(
                "/",
                StaticFiles(
                    directory=dashboard_dir,
                    html=True,
                ),
                name="dashboard",
            )
        elif mount_dashboard:
            logger.info(
                "Dashboard assets not found (DASHBOARD_DIR is unset); serving the "
                "API without the dashboard UI. Build it with `cd dashboard-react "
                "&& npm run build` or set SKULK_DASHBOARD_DIR to serve the UI."
            )

        self._text_generation_queues: dict[
            CommandId,
            Sender[TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk],
        ] = {}
        self._image_generation_queues: dict[
            CommandId, Sender[ImageChunk | ErrorChunk]
        ] = {}
        self._embedding_queues: dict[
            CommandId, Sender[EmbeddingChunk | ErrorChunk]
        ] = {}
        self._audio_speech_queues: dict[CommandId, Sender[AudioChunk | ErrorChunk]] = {}
        self._audio_transcription_queues: dict[
            CommandId, Sender[TranscriptionChunk | ErrorChunk]
        ] = {}
        self._realtime_audio_transcription_commands: set[CommandId] = set()
        self._realtime_task_release_events: dict[CommandId, anyio.Event] = {}
        self._model_trust_decision_waiters: dict[
            tuple[str, bool], list[anyio.Event]
        ] = {}
        self._cancelled_command_ids: set[CommandId] = set()
        # Per-command data-plane reorder buffers (#279 Phase 2b). The DATA gossip
        # topic has no total order (unlike the master event idx it replaced), so
        # a multi-node producer's chunks can arrive out of order; we reorder by
        # the per-command sequence stamped on each DataChunk before dispatch.
        self._chunk_reorder: dict[CommandId, _ChunkReorderState] = {}
        # Per-command dedupe cursor for the buffer-off path (#279 Phase 3 / #311
        # review). Even with reordering disabled the data plane must drop
        # duplicate sequences: when the serving rank-0 worker and the owning API
        # are the same node, a chunk is delivered twice on Zenoh - once via the
        # DATA TopicRouter's local publish, once via the node's own
        # data/<node_id> Zenoh subscription (self-loopback). The reorder buffer
        # dropped these as below-cursor duplicates; without it, this cursor does
        # the same O(1) drop (sequence < cursor) while still dispatching in
        # arrival order. The transport delivers each command in order, so the
        # only out-of-cursor chunks are these duplicates.
        self._data_dedup_cursor: dict[CommandId, int] = {}
        # Data-plane ordering (#279 Phase 3): the reorder buffer is needed only
        # when the DATA transport can deliver a command's chunks out of order.
        # Gossipsub (the flag-off default) reorders, so the buffer stays on there
        # (the #301 fix). Zenoh delivers a command's chunks per-publisher FIFO, so
        # arrival order is generation order, so the buffer is skipped and chunks
        # dispatch as they arrive (validated: 20/20 buffer-off coherence on a 3-node
        # sampled-MTP matrix). SKULK_DATA_REORDER_BUFFER overrides the transport
        # default explicitly (`1`/`0`) for testing or belt-and-suspenders.
        _reorder_override = (
            os.environ.get("SKULK_DATA_REORDER_BUFFER", "").strip().lower()
        )
        if _reorder_override in ("1", "true", "yes", "on"):
            self._reorder_buffer_enabled: bool = True
        elif _reorder_override in ("0", "false", "no", "off"):
            self._reorder_buffer_enabled = False
        else:
            if _reorder_override:
                logger.warning(
                    f"Ignoring unrecognized SKULK_DATA_REORDER_BUFFER="
                    f"{_reorder_override!r} (expected 1/0/true/false/yes/no/on/"
                    f"off); following the transport default."
                )
            # Default by transport: gossipsub reorders (buffer on), Zenoh is
            # per-publisher FIFO (buffer off).
            self._reorder_buffer_enabled = not data_plane_zenoh
        if data_receiver is None:
            data_plane_transport: DataPlaneTransport = "disabled"
        elif data_plane_zenoh:
            data_plane_transport = "zenoh"
        else:
            data_plane_transport = "gossipsub"
        self._data_plane_observer = DataPlaneObserver(
            transport=data_plane_transport,
            reorder_buffer_enabled=self._reorder_buffer_enabled,
        )
        self._provider_observer = ProviderObserver()
        self._data_plane_egress_provider = data_plane_egress_provider
        self._vision_media_egress_provider = vision_media_egress_provider
        self._telemetry_plane_provider = telemetry_plane_provider
        self._image_store = ImageStore(SKULK_IMAGE_CACHE_DIR)
        self._tg: TaskGroup = TaskGroup()

    def set_runner_diagnostics_provider(
        self,
        provider: Callable[[], Sequence[RunnerSupervisorDiagnostics]] | None,
    ) -> None:
        """Attach a local worker diagnostics provider to this API instance."""

        self._runner_diagnostics_provider = provider

    def set_runner_cancel_provider(
        self,
        provider: Callable[
            [RunnerId, task_types.TaskId], Awaitable[RunnerTaskCancelResponse]
        ]
        | None,
    ) -> None:
        """Attach a local worker control provider for direct runner cancellation."""

        self._runner_cancel_provider = provider

    def set_vision_media_ingress_provider(
        self,
        provider: Callable[[], VisionMediaIngressDiagnostics] | None,
    ) -> None:
        """Attach local worker vision-ingress diagnostics to this API."""

        self._vision_media_ingress_provider = provider

    def _vision_media_ingress_diagnostics(self) -> VisionMediaIngressDiagnostics:
        """Combine local API admission pressure with worker ingress occupancy."""

        worker = (
            self._vision_media_ingress_provider()
            if self._vision_media_ingress_provider is not None
            else VisionMediaIngressDiagnostics.empty()
        )
        return worker.model_copy(
            update={
                "pending_api_commands": len(self._pending_vision_media),
                "pending_api_bytes": self._pending_vision_media_bytes,
                "active_api_commands": len(self._active_vision_media_bytes),
                "active_api_bytes": self._active_vision_media_total_bytes,
                "pending_worker_acknowledgements": sum(
                    len(targets) for targets in self._vision_media_pending_acks.values()
                ),
            }
        )

    def reset(
        self,
        result_clock: int,
        event_receiver: Receiver[IndexedEvent],
        master_node_id: NodeId,
    ):
        logger.info("Resetting API State")
        if self._event_log is not None:
            self._event_log.close()
            self._event_log = DiskEventLog(_API_EVENT_LOG_DIR)
            self._event_log_appends_since_retention_check = 0
        self.state = State()
        self._system_id = SystemId()
        self._fail_open_command_streams_for_session_reset()
        self._text_generation_queues = {}
        self._image_generation_queues = {}
        self._embedding_queues = {}
        self._audio_speech_queues = {}
        self._audio_transcription_queues = {}
        self._realtime_audio_transcription_commands = set()
        for release_event in self._realtime_task_release_events.values():
            release_event.set()
        self._realtime_task_release_events = {}
        for waiters in self._model_trust_decision_waiters.values():
            for waiter in waiters:
                waiter.set()
        self._model_trust_decision_waiters = {}
        self._speech_media_commands = set()
        self._speech_media_targets = {}
        self._transcription_media_targets = {}
        self._pending_speech_media = {}
        self._pending_speech_media_bytes = 0
        self._pending_trace_data = {}
        self._pending_vision_media = {}
        self._pending_vision_media_bytes = 0
        self._active_vision_media_bytes = {}
        self._active_vision_media_total_bytes = 0
        self._vision_media_commands = set()
        self._vision_media_targets = {}
        self._vision_media_pending_acks = {}
        self._vision_media_ack_deadlines = {}
        self._vision_media_models = {}
        self._vision_media_failures = {}
        self._cancelled_command_ids = set()
        self.unpause(result_clock, master_node_id=master_node_id)
        self.event_receiver.close()
        self.event_receiver = event_receiver
        self._tg.start_soon(self._apply_state)

    def _fail_open_command_streams_for_session_reset(self) -> None:
        """Terminate every open command stream when the cluster session changes.

        A new session (master failover) cannot carry the previous session's
        tasks, so in-flight requests are unrecoverable — and the old queue
        maps are about to be replaced, leaving their senders unreachable by
        chunk dispatch, cancellation, AND the orphaned-task sweep (whose
        TaskFailed can only reference the new session's state). Found by the
        #223 verification drill: killing the MASTER mid-generation still hung
        the client after the master-side sweep landed.

        Best-effort error chunk (the consumer is normally parked on the
        queue, so zero-buffer send_nowait delivers), then close — close alone
        still ends the stream rather than hanging it.
        """
        error_message = (
            "The cluster session changed (master failover) while this "
            "request was in flight; please retry"
        )
        for queue_map in (
            self._text_generation_queues,
            self._image_generation_queues,
            self._embedding_queues,
            self._audio_speech_queues,
            self._audio_transcription_queues,
        ):
            for sender in list(queue_map.values()):
                # The originating task is gone with the old session, so the
                # true model id is unknowable here.
                error_chunk = ErrorChunk(
                    model=ModelId("unknown"), error_message=error_message
                )
                with contextlib.suppress(
                    WouldBlock, BrokenResourceError, ClosedResourceError
                ):
                    sender.send_nowait(error_chunk)
                with contextlib.suppress(ClosedResourceError):
                    sender.close()

    def unpause(self, result_clock: int, master_node_id: NodeId | None = None):
        logger.info("Unpausing API")
        self.last_completed_election = result_clock
        if master_node_id is not None:
            self._master_node_id = master_node_id
        self.paused = False
        self.paused_ev.set()
        self.paused_ev = anyio.Event()

    def _setup_exception_handlers(self) -> None:
        self.app.exception_handler(HTTPException)(self.http_exception_handler)

    async def http_exception_handler(
        self, _: Request, exc: HTTPException
    ) -> JSONResponse:
        err = ErrorResponse(
            error=ErrorInfo(
                message=exc.detail,
                type=HTTPStatus(exc.status_code).phrase,
                code=exc.status_code,
            )
        )
        return JSONResponse(
            err.model_dump(),
            status_code=exc.status_code,
            headers=exc.headers,
        )

    def _setup_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[
                "X-Audio-Sample-Rate",
                "X-Audio-Channels",
                "X-Audio-Sample-Format",
                "X-Skulk-Placement-Failure",
            ],
        )

    def _setup_routes(self) -> None:
        self.app.get(
            "/node_id",
            tags=["State & Tracing"],
            summary="Get this API node's ID",
        )(lambda: self.node_id)
        self.app.post(
            "/instance",
            tags=["Instances"],
            summary="Create an instance from a fully specified placement",
            description=(
                "Create an instance from an already computed placement object "
                "when you want exact control instead of Skulk picking the "
                "placement. Every embedded shard card must identify the same "
                "model alias as the assignment and exactly match the effective "
                "card already present in the authorized local catalog."
            ),
        )(self.create_instance)
        self.app.post(
            "/place_instance",
            tags=["Instances"],
            summary="Quick-launch a model placement",
            responses={
                400: {
                    "description": "Current fleet facts admit no valid placement.",
                    "headers": {
                        "X-Skulk-Placement-Failure": {
                            "description": "Stable placement failure category.",
                            "schema": {"type": "string"},
                        }
                    },
                },
                503: {
                    "description": "Required placement telemetry is still arriving.",
                    "headers": {
                        "X-Skulk-Placement-Failure": {
                            "description": "Stable placement failure category.",
                            "schema": {"type": "string"},
                        }
                    },
                },
            },
            description=(
                "Place and launch a model with Skulk choosing a valid concrete placement "
                "from the requested sharding, instance metadata, and minimum-node constraints. "
                "The placement is validated against the current cluster state before the "
                "command is forwarded: an impossible placement returns 400 with the specific "
                "reason (no connected cycle, exclusions removed every candidate, a node cannot "
                "fit its shard with runtime headroom, ...). If node memory info is still being "
                "gathered (cluster just formed), the request waits up to 15 seconds for it "
                "before returning 503 — retry shortly in that case. Model cards may declare "
                "several open backend tags in preference order; the planner filters current "
                "trust, engine/build, topology, and capacity blockers before ranking and "
                "automatically falls through to the next launchable candidate. Placement "
                "accepts only models already present through signed publication, bundled "
                "distribution, or an authenticated add; it never discovers an unknown "
                "Hugging Face repository as a side effect. "
                "failures retain a readable error message and expose a stable category in the "
                "X-Skulk-Placement-Failure response header."
            ),
        )(self.place_instance)
        self.app.get(
            "/instance/placement",
            tags=["Instances"],
            summary="Compute a concrete placement for one requested combination",
            description="Return the exact instance shape Skulk would create for one requested model, sharding mode, instance metadata, and node-count combination.",
        )(self.get_placement)
        self.app.get(
            "/instance/previews",
            tags=["Instances"],
            summary="Preview valid placements for a model",
            description=(
                "Return candidate placements for a model before launch. This is the best first "
                "step when you want to see what Skulk can place on the current node or cluster. "
                "Pass `excluded_node_ids` (repeatable) to mirror the `excluded_nodes` field on "
                "POST /place_instance and preview against the post-exclusion topology. "
                "Besides the planner's ranked pick per placement shape, the response includes "
                "per-host single-node previews marked `alternative: true` for every other host "
                "that passes admission, so heterogeneous fleets expose the full set of valid "
                "hosts rather than only the ranking winner. Previews explain current planner "
                "choices; ordinary launch requests remain adaptive and do not reserve or replay "
                "one preview. Unavailable previews include a stable error_code."
            ),
        )(self.get_placement_previews)
        self.app.get(
            "/instance/{instance_id}",
            tags=["Instances"],
            summary="Get one running instance",
        )(self.get_instance)
        self.app.delete(
            "/instance/{instance_id}",
            tags=["Instances"],
            summary="Delete a running instance",
        )(self.delete_instance)
        self.app.get(
            "/models",
            tags=["Models"],
            summary="List known models",
            description=(
                "Return the effective model catalog: complete installed cards, "
                "the current signed registry snapshot, bundled fallback cards, "
                "and final operator-owned custom overrides. Entries expose active "
                "installed identity, current registry identity, update availability, "
                "verification state, and warn-only advisories."
            ),
        )(self.get_models)
        self.app.get(
            "/v1/models",
            tags=["Models"],
            summary="List known models",
            description=(
                "OpenAI-style listing of Skulk's effective model catalog rather "
                "than only running instances. Entries distinguish the active "
                "installed generation from current signed registry truth and "
                "report updates, verification, advisories, and repository-code "
                "authorization provenance."
            ),
        )(self.get_models)
        self.app.get(
            "/models/remote-code-approvals",
            tags=["Models"],
            summary="List legacy model-code approvals",
            description=(
                "Deprecated compatibility endpoint for approvals stored before "
                "model publication and explicit addition became the repository-"
                "code authorization boundaries. Returned identities are inert."
            ),
            deprecated=True,
        )(self.list_remote_code_approvals)
        self.app.post(
            "/models/remote-code-approvals/{card_id}",
            tags=["Models"],
            summary="Approve model repository code (deprecated)",
            description=(
                "Deprecated compatibility endpoint. Valid signed registry cards "
                "are authorized by publication and explicitly added models are "
                "authorized by the add action, so current cards reject redundant "
                "approval with HTTP 409."
            ),
            deprecated=True,
        )(self.approve_remote_code)
        self.app.delete(
            "/models/remote-code-approvals/{card_id}",
            tags=["Models"],
            summary="Remove a legacy model repository-code approval",
            description=(
                "Removes an inert approval retained from an older deployment. "
                "This does not revoke a published or explicitly added model."
            ),
            deprecated=True,
        )(self.revoke_remote_code)
        self.app.post(
            "/models/add",
            tags=["Models"],
            summary="Fetch and add a custom model card",
            description=(
                "Add a custom model card to Skulk's model catalog so it becomes "
                "searchable and launchable. An optional gguf_file selects one exact "
                "quant from a multi-quant GGUF repository. Mutable main is resolved "
                "once to an immutable commit, and the explicit add action authorizes "
                "repository code selected by that card. Success waits until the exact "
                "mutation is ordered and visible in the responding API's catalog. The "
                "mutation requires a direct loopback or trusted-fabric "
                "(private LAN / CGNAT) connection without proxy-forwarding "
                "headers, or an authenticated operator-gateway credential; "
                "browser requests must present an Origin on those trust classes "
                "or naming one of this node's own hostnames."
            ),
        )(self.add_custom_model)
        self.app.post(
            "/models/add-card",
            tags=["Models"],
            summary="Add one exact unsigned custom model card",
            description=(
                "Persist a complete exact model card supplied by an authenticated "
                "operator workflow without fetching or regenerating Hub metadata. "
                "Skulk forces custom-card semantics and removes registry trust "
                "claims; the explicit immutable-card add authorizes its selected "
                "repository code without a second approval."
            ),
        )(self.add_exact_custom_model_card)
        self.app.delete(
            "/models/custom/{model_id:path}",
            tags=["Models"],
            summary="Delete a custom model card",
            description="Remove a previously added custom model card from Skulk's local catalog.",
        )(self.delete_custom_model)
        self.app.get(
            "/models/search",
            tags=["Models"],
            summary="Search Hugging Face for models",
            description=(
                "Search for models to add or launch. Pass mlx_only=true to restrict results "
                "to mlx-community; omit or pass false to search all of Hugging Face. "
                "Exact GGUF filename queries also inspect bounded candidate repository "
                "manifests and return the matched repo-relative file path."
            ),
        )(self.search_models)
        self.app.get(
            "/models/card-summary",
            tags=["Models"],
            summary="Fetch a Hugging Face model card summary",
            description=(
                "Download a repository's model card README and return its first "
                "prose paragraphs with markup stripped, for the dashboard's "
                "model discovery popovers. The summary is empty when the card "
                "has no usable prose."
            ),
        )(self.get_model_card_summary)
        self.app.get(
            "/models/gguf-quants",
            tags=["Models"],
            summary="List a GGUF repository's quantizations",
            description=(
                "Enumerate the downloadable quantizations of a Hugging Face "
                "GGUF repository: one option per quant shard group with its "
                "loadable first shard, human label, exact total bytes, and "
                "shard count, smallest first. Companion artifacts such as "
                "speculative drafters and imatrix files are excluded."
            ),
        )(self.get_gguf_quant_options)
        self.app.post(
            "/v1/chat/completions",
            response_model=None,
            tags=["Compatibility APIs"],
            summary="OpenAI Chat Completions-compatible text generation",
            description=(
                "Generate text with an OpenAI Chat Completions-compatible payload. The requested "
                "model must already be placed, running, and declare TextGeneration; speech-only "
                "models are rejected before command dispatch. The reserved model id "
                "'skulk/steward' selects the intelligent-fabric steward and answers 404 when "
                "that mode is disabled, or 503 with the steward status payload while the "
                "steward is still being placed, staged, or loaded. Steward observation is "
                "available to ordinary clients, but proposal-creation tools are exposed only "
                "to requests with operator mutation authority."
            ),
        )(self.chat_completions_route)
        self.app.post(
            "/v1/embeddings",
            tags=["Compatibility APIs"],
            summary="Generate embeddings",
            description=(
                "Generate embeddings with an already-cataloged, placed model that "
                "declares TextEmbedding. Unknown or unplaced models return 404."
            ),
        )(self.embeddings)
        self.app.post(
            "/v1/audio/speech",
            response_model=None,
            tags=["Audio"],
            summary="Generate speech audio",
            description=(
                "OpenAI-compatible text-to-speech endpoint. The requested model "
                "must already be placed and running as a text-to-speech model. "
                "The stream=true path requires a mounted card that declares "
                "audio.supports_streaming=true."
            ),
            openapi_extra=_audio_speech_request_body(),
        )(self.audio_speech_http)
        self.app.post(
            "/v1/audio/transcriptions",
            response_model=None,
            tags=["Audio"],
            summary="Transcribe speech audio",
            description=(
                "OpenAI-compatible speech-to-text endpoint. The requested model "
                "must already be placed and running as a speech-to-text model. "
                "stream=true returns typed SSE events for cards declaring "
                "audio.supports_streaming=true; response_format=ndjson retains "
                "progressive NDJSON framing."
            ),
        )(self.audio_transcriptions)
        self.app.post(
            "/v1/audio/translations",
            response_model=None,
            tags=["Audio"],
            summary="Translate speech audio to English",
            description=(
                "OpenAI-compatible speech translation endpoint (translation "
                "target is English). The requested mounted model must "
                "explicitly declare translation support on its model card."
            ),
        )(self.audio_translations)
        self.app.get(
            "/v1/audio/voices",
            response_model=AudioVoiceList,
            tags=["Audio"],
            summary="List voices for a mounted speech model",
            description=(
                "Skulk extension that returns stable model-native and bundled-"
                "reference voice identifiers declared by a mounted text-to-speech "
                "model."
            ),
        )(self.audio_voices)
        self.app.websocket("/v1/realtime")(self.realtime_transcription)
        self.app.websocket("/v1/fabric/chains/speech")(self.fabric_speech_chain)
        self.app.post(
            "/bench/chat/completions",
            tags=["Compatibility APIs"],
            summary="Benchmark chat completions",
            description=(
                "Benchmark text generation for a placed model that declares "
                "TextGeneration. Speech-only models are rejected before dispatch."
            ),
        )(self.bench_chat_completions)
        self.app.post(
            "/v1/images/generations",
            response_model=None,
            tags=["Images"],
            summary="Generate images",
            description=(
                "Generate images with an already-cataloged, placed image model. "
                "Unknown or unplaced models return 404."
            ),
        )(self.image_generations)
        self.app.post(
            "/bench/images/generations",
            tags=["Images"],
            summary="Benchmark image generation",
        )(self.bench_image_generations)
        self.app.post(
            "/v1/images/edits",
            response_model=None,
            tags=["Images"],
            summary="Edit images",
        )(self.image_edits)
        self.app.post(
            "/bench/images/edits",
            tags=["Images"],
            summary="Benchmark image editing",
        )(self.bench_image_edits)
        self.app.get("/images", tags=["Images"], summary="List stored images")(
            self.list_images
        )
        self.app.get(
            "/images/{image_id}", tags=["Images"], summary="Fetch one stored image"
        )(self.get_image)
        self.app.post(
            "/v1/messages",
            response_model=None,
            tags=["Compatibility APIs"],
            summary="Anthropic Claude Messages-compatible endpoint",
            description=(
                "Claude Messages-compatible text generation endpoint. As with chat completions, "
                "the target model must already be placed, ready, and declare TextGeneration."
            ),
        )(self.claude_messages)
        self.app.post(
            "/v1/responses",
            response_model=None,
            tags=["Compatibility APIs"],
            summary="OpenAI Responses-compatible endpoint",
            description=(
                "OpenAI Responses-compatible endpoint for text generation and reasoning-style "
                "workflows backed by a placed Skulk model that declares TextGeneration."
            ),
        )(self.openai_responses)
        self.app.post(
            "/v1/cancel/{command_id}",
            tags=["Compatibility APIs"],
            summary="Cancel an active generation command",
            description="Request cancellation for an in-flight text, image, embedding, or speech command by its command ID.",
        )(self.cancel_command)
        self.app.post(
            "/v1/tools/web_search",
            tags=["Tools"],
            summary="Execute the generic web-search tool",
            description=(
                "Run the generic `web_search` tool and return structured search results that "
                "clients can feed back into a tool-calling conversation loop."
            ),
        )(self.web_search)
        self.app.post(
            "/v1/tools/open_url",
            tags=["Tools"],
            summary="Open one URL and inspect its metadata",
            description=(
                "Fetch one HTTP or HTTPS URL, follow redirects, and return structured "
                "metadata that clients can feed back into a tool-calling conversation loop."
            ),
        )(self.open_url)
        self.app.post(
            "/v1/tools/extract_page",
            tags=["Tools"],
            summary="Fetch one URL and extract readable page text",
            description=(
                "Fetch one HTTP or HTTPS URL and return bounded readable text extracted "
                "from the response body for tool-calling conversation loops."
            ),
        )(self.extract_page)

        # Ollama API
        self.app.head(
            "/ollama/", tags=["Compatibility APIs"], summary="Ollama version check"
        )(self.ollama_version)
        self.app.head(
            "/ollama/api/version",
            tags=["Compatibility APIs"],
            summary="Ollama version check",
        )(self.ollama_version)
        self.app.post(
            "/ollama/api/chat",
            response_model=None,
            tags=["Compatibility APIs"],
            summary="Ollama chat",
            description=(
                "Ollama-compatible chat endpoint backed by a placed model that "
                "declares TextGeneration."
            ),
            openapi_extra=_json_request_body(OllamaChatRequest.model_json_schema()),
        )(self.ollama_chat)
        self.app.post(
            "/ollama/api/api/chat",
            response_model=None,
            tags=["Compatibility APIs"],
            summary="Ollama chat alias",
        )(self.ollama_chat)
        self.app.post(
            "/ollama/api/v1/chat",
            response_model=None,
            tags=["Compatibility APIs"],
            summary="Ollama chat alias",
        )(self.ollama_chat)
        self.app.post(
            "/ollama/api/generate",
            response_model=None,
            tags=["Compatibility APIs"],
            summary="Ollama generate",
            description=(
                "Ollama-compatible prompt completion backed by a placed model "
                "that declares TextGeneration."
            ),
            openapi_extra=_json_request_body(OllamaGenerateRequest.model_json_schema()),
        )(self.ollama_generate)
        self.app.get(
            "/ollama/api/tags",
            tags=["Compatibility APIs"],
            summary="List Ollama models",
        )(self.ollama_tags)
        self.app.get(
            "/ollama/api/api/tags",
            tags=["Compatibility APIs"],
            summary="List Ollama models alias",
        )(self.ollama_tags)
        self.app.get(
            "/ollama/api/v1/tags",
            tags=["Compatibility APIs"],
            summary="List Ollama models alias",
        )(self.ollama_tags)
        self.app.post(
            "/ollama/api/show",
            tags=["Compatibility APIs"],
            summary="Show Ollama model details",
        )(self.ollama_show)
        self.app.get(
            "/ollama/api/ps",
            tags=["Compatibility APIs"],
            summary="List running Ollama models",
        )(self.ollama_ps)
        self.app.get(
            "/ollama/api/version",
            tags=["Compatibility APIs"],
            summary="Get Ollama API version",
        )(self.ollama_version)

        self.app.get(
            "/state",
            tags=["State & Tracing"],
            summary="Get cluster state",
            description="Return the current cluster state as seen by this API node, including topology, instances, node capabilities, and live telemetry (per-node memory and system profile).",
        )(self.get_cluster_state)
        self.app.get(
            "/events",
            tags=["State & Tracing"],
            summary="Get stored event log",
            description="Stream or return the API-side event log used for debugging state transitions and cluster behavior.",
        )(self.stream_events)
        self.app.post(
            "/download/start",
            tags=["Downloads"],
            summary="Start a node download",
            description=(
                "Start a low-level node download for an exact authorized catalog "
                "card. Requires a direct loopback or trusted-fabric (private "
                "LAN / CGNAT) connection without proxy-forwarding headers, or "
                "authenticated operator-gateway access; browser requests must "
                "present an Origin on those trust classes or naming one of "
                "this node's own hostnames."
            ),
        )(self.start_download)
        self.app.delete(
            "/download/{node_id}/{model_id:path}",
            tags=["Downloads"],
            summary="Delete a node download",
            description="Delete or cancel a download associated with a given node and model.",
        )(self.delete_download)
        self.app.get(
            "/v1/tracing",
            tags=["State & Tracing"],
            summary="Get cluster tracing state",
            description="Return whether runtime tracing is currently enabled for new requests across the cluster session.",
        )(self.get_tracing_state)
        self.app.get(
            "/v1/telemetry/preview",
            tags=["State & Tracing"],
            summary="Preview pending field telemetry",
            description=(
                "Return the current field-telemetry consent state and the exact"
                " pending sample batch that would be sent to the ingest service,"
                " so operators can inspect precisely what leaves the cluster."
                " Collection is opt-in and content-free (metrics, hardware"
                " classes, and error classes only)."
            ),
        )(self.get_telemetry_preview)
        self.app.put(
            "/v1/tracing",
            tags=["State & Tracing"],
            summary="Set cluster tracing state",
            description="Enable or disable runtime tracing for new requests across the cluster session.",
        )(self.update_tracing_state)
        self.app.get(
            "/v1/traces",
            tags=["State & Tracing"],
            summary="List saved traces",
            description="List saved trace files that can be inspected for debugging and performance analysis.",
        )(self.list_traces)
        self.app.get(
            "/v1/traces/cluster",
            tags=["State & Tracing"],
            summary="List cluster traces",
            description="List deduplicated traces discoverable from this node across reachable peer APIs.",
        )(self.list_cluster_traces)
        self.app.post(
            "/v1/traces/delete",
            tags=["State & Tracing"],
            summary="Delete saved traces",
            description="Delete one or more saved trace artifacts by task ID.",
        )(self.delete_traces)
        self.app.get(
            "/v1/traces/cluster/{task_id}",
            tags=["State & Tracing"],
            summary="Get one cluster trace",
            description="Return a trace from local storage or proxy it from a reachable peer node.",
        )(self.get_cluster_trace)
        self.app.get(
            "/v1/traces/cluster/{task_id}/stats",
            tags=["State & Tracing"],
            summary="Get one cluster trace summary",
            description="Return aggregated timing statistics for a trace available locally or on a reachable peer node.",
        )(self.get_cluster_trace_stats)
        self.app.get(
            "/v1/traces/cluster/{task_id}/raw",
            tags=["State & Tracing"],
            summary="Download raw cluster trace JSON",
            description="Download a raw Chrome trace artifact from local storage or a reachable peer node.",
            response_model=None,
        )(self.get_cluster_trace_raw)
        self.app.get(
            "/v1/capabilities",
            tags=["Extensions"],
            summary="List this node's extension-served capabilities",
            description=(
                "Return the self-describing capability descriptors served by "
                "this node's provider extensions (fabric-citizenship). Each "
                "descriptor carries the capability id, semantic version, "
                "human/LLM-readable description, JSON Schemas for input and "
                "output, the call's I/O mode, and a content revision digest. "
                "This is the heavy half of capability discovery; the light "
                "half is the capability tag gossiped on the telemetry plane "
                "(surfaced per node as nodeCapabilities in GET /state). "
                "Pass the node_id query parameter to describe a reachable "
                "peer instead of this node (an empty list when the peer is "
                "unreachable). An empty list otherwise means no provider "
                "extension is installed or all installed providers are unready. "
                "Providers with a readiness facet are filtered at discovery."
            ),
        )(self.list_node_capabilities)
        self.app.post(
            "/v1/capabilities/call",
            tags=["Extensions"],
            summary="Invoke a capability served by this node",
            description=(
                "Dispatch one unary capability call to this node's provider "
                "extensions (fabric-citizenship). The body is the typed call "
                "envelope (call id, capability id and exact version, the "
                "descriptor revision pinned at discovery, deadline, and the "
                "opaque payload). The payload is validated against the "
                "descriptor's input schema before the provider runs and the "
                "result against its output schema after. A syntactically valid "
                "envelope always gets HTTP 200 with a typed result: failures "
                "arrive as machine-readable error codes (not_found, "
                "version_mismatch, revision_mismatch, invalid_payload, "
                "invalid_result, payload_too_large, overloaded, timeout, "
                "provider_error) rather than transport errors. A body that "
                "does not parse as the envelope at all (malformed JSON, "
                "missing fields, out-of-range values) gets the standard 422, "
                "since there is no call id to correlate a typed result to. Callers normally use this through their extension "
                "context's call_capability rather than directly."
            ),
        )(self.serve_capability_call)
        self.app.post(
            "/v1/capabilities/stream",
            tags=["Extensions"],
            summary="Open a capability stream on this node",
            description=(
                "Validate and admit one server-, client-, or bidirectional "
                "provider stream. This control-sized request returns a typed "
                "opening result; active media directions travel separately on "
                "the provider DATA plane with ordered lifecycle headers, "
                "optional raw media, and explicit caller input half-close."
            ),
        )(self.serve_capability_stream)
        self.app.post(
            "/v1/capabilities/stream/cancel",
            tags=["Extensions"],
            summary="Cancel an admitted capability stream",
            description=(
                "Request explicit cancellation of one admitted provider "
                "stream. The provider emits one cancelled terminal frame when "
                "the stream is still active; repeated cancellation is idempotent."
            ),
        )(self.cancel_capability_stream)
        self.app.get(
            "/v1/diagnostics/node",
            tags=["Diagnostics"],
            summary="Get local node diagnostics",
            description=(
                "Return a read-only diagnostic bundle for this API node, including "
                "runtime identity, placement analysis, live runner-supervisor "
                "state, local resources, and relevant OS processes."
            ),
        )(self.get_node_diagnostics)
        self.app.get(
            "/v1/diagnostics/telemetry",
            tags=["Diagnostics"],
            summary="Get local telemetry-plane diagnostics",
            description=(
                "Return aggregate bounded-admission, coalescing, drop, queue, and "
                "publish metrics for this node's isolated telemetry transport."
            ),
        )(self.get_telemetry_plane_diagnostics)
        self.app.get(
            "/v1/diagnostics/performance-envelopes",
            tags=["Diagnostics"],
            summary="Get this node's performance envelopes",
            description=(
                "Return the observe-only throughput-and-latency-versus-concurrency "
                "curve this API node has measured for each (hardware class x model "
                "x engine x quantization) it has served, including a simple knee "
                "estimate. Adaptive-concurrency Phase 0: data only, no behavior "
                "change."
            ),
        )(self.get_performance_envelopes)
        self.app.get(
            "/v1/diagnostics/performance-envelopes/cluster",
            tags=["Diagnostics"],
            summary="Get performance envelopes across the cluster",
            description=(
                "Fan out to every reachable cluster member and return each one's "
                "performance-envelope report. Unreachable members appear as "
                "explicit failures rather than vanishing."
            ),
        )(self.get_cluster_performance_envelopes)
        self.app.get(
            "/v1/steward",
            tags=["Intelligent Fabric"],
            summary="Get intelligent-fabric cognition status",
            description=(
                "Report whether intelligent-fabric mode is enabled and whether "
                "a steward placement currently exists, with its model and "
                "instance id when present, plus a one-word lifecycle 'state' "
                "(disabled, downloading, starting, ready, degraded) derived "
                "from those fields and the liveness canary's history. The "
                "steward is the fabric-maintained resident assistant; see the "
                "steward chat endpoint."
            ),
        )(self.get_steward_status)
        self.app.get(
            "/v1/steward/proposals",
            tags=["Intelligent Fabric"],
            summary="List approval-gated steward action proposals",
            description=(
                "Return the bounded replicated proposal audit, newest first, with "
                "internal node, instance, command, and model-card payloads removed. "
                "The surface requires operator authority because it is paired with "
                "the approval workflow."
            ),
        )(self.list_steward_action_proposals)
        self.app.post(
            "/v1/steward/proposals/{proposal_id}/decision",
            tags=["Intelligent Fabric"],
            summary="Approve or reject one steward action proposal",
            description=(
                "Submit one authenticated, single-use decision to the elected master. "
                "Approval revalidates expiry, model truth, system-placement protection, "
                "and current cluster state before dispatching an existing typed command."
            ),
        )(self.decide_steward_action_proposal)
        self.app.post(
            "/v1/diagnostics/node/runners/{runner_id}/cancel",
            tags=["Diagnostics"],
            summary="Request direct local runner cancellation",
            description=(
                "Request cooperative cancellation for one live task on one local "
                "runner supervisor. This is a best-effort control path intended "
                "for diagnostics and may not stop a runner wedged in native code."
            ),
        )(self.cancel_local_runner_task)
        self.app.post(
            "/v1/diagnostics/node/capture",
            tags=["Diagnostics"],
            summary="Capture local node diagnostic bundle",
            description=(
                "Collect an on-demand diagnostic bundle for this node, optionally "
                "focused on one live runner or task. The bundle includes current "
                "node diagnostics, runner flight-recorder entries, the latest MLX "
                "memory snapshot, and best-effort macOS process samples."
            ),
        )(self.capture_local_node_diagnostics)
        self.app.get(
            "/v1/diagnostics/cluster",
            tags=["Diagnostics"],
            summary="Get cluster diagnostics",
            description=(
                "Fan out to reachable peer APIs and return read-only diagnostic "
                "bundles for the local node and peers. Unreachable peers are "
                "reported as partial failures instead of failing the whole request. "
                "Peer bundles tolerate additive diagnostic fields, and aggregate "
                "plus per-node versionStatus values expose mixed-build rollouts."
            ),
        )(self.get_cluster_diagnostics)
        self.app.get(
            "/v1/diagnostics/cluster/timeline",
            tags=["Diagnostics"],
            summary="Get cross-rank runner timeline",
            description=(
                "Stitch every reachable node's runner-supervisor diagnostics into "
                "one cross-rank chronological view. Returns a per-runner synopsis "
                "(rank, current phase, seconds-in-phase, MLX memory) and every "
                "flight-recorder entry across all ranks merged and sorted by "
                "wall-clock timestamp. The intended use is debugging distributed "
                "deadlocks where the rank-disagreement signature is invisible from "
                "any single node's local diagnostics."
            ),
        )(self.get_cluster_timeline)
        self.app.get(
            "/v1/diagnostics/cluster/{node_id}",
            tags=["Diagnostics"],
            summary="Get one cluster node diagnostic bundle",
            description=(
                "Return diagnostics for the requested node from local state or by "
                "proxying to a reachable peer API. Peer responses tolerate unknown "
                "additive diagnostic fields."
            ),
        )(self.get_cluster_node_diagnostics)
        self.app.post(
            "/v1/diagnostics/cluster/{node_id}/capture",
            tags=["Diagnostics"],
            summary="Capture one cluster node diagnostic bundle",
            description=(
                "Return an on-demand diagnostic capture bundle for the requested "
                "node, proxying to a reachable peer API when the target is remote. "
                "Peer responses tolerate unknown additive diagnostic fields."
            ),
        )(self.capture_cluster_node_diagnostics)
        self.app.post(
            "/v1/diagnostics/cluster/{node_id}/runners/{runner_id}/cancel",
            tags=["Diagnostics"],
            summary="Request direct runner cancellation on one cluster node",
            description=(
                "Proxy a cooperative live-task cancellation request to the "
                "requested node's local diagnostics control endpoint. This is "
                "best-effort and may not stop a runner wedged in native code."
            ),
        )(self.cancel_cluster_runner_task)
        self.app.get(
            "/v1/traces/{task_id}",
            tags=["State & Tracing"],
            summary="Get one saved trace",
            description="Return the structured trace events for one saved task trace.",
        )(self.get_trace)
        self.app.get(
            "/v1/traces/{task_id}/stats",
            tags=["State & Tracing"],
            summary="Get one trace summary",
            description="Return aggregated timing statistics for one saved trace.",
        )(self.get_trace_stats)
        self.app.get(
            "/v1/traces/{task_id}/raw",
            tags=["State & Tracing"],
            summary="Download raw trace JSON",
            description="Download the raw Chrome trace JSON artifact for one task.",
            response_model=None,
        )(self.get_trace_raw)
        self.app.get(
            "/onboarding",
            tags=["State & Tracing"],
            summary="Get onboarding completion status",
            description="Return whether the dashboard onboarding flow has been completed on this node.",
        )(self.get_onboarding)
        self.app.post(
            "/onboarding",
            tags=["State & Tracing"],
            summary="Mark onboarding complete",
            description="Mark the local dashboard onboarding flow as complete.",
        )(self.complete_onboarding)

        # Config & store endpoints
        self.app.get(
            "/config",
            tags=["Config"],
            summary="Get cluster config",
            description="Return the current cluster-wide config with sensitive fields such as the HF token removed.",
        )(self.get_config)
        self.app.put(
            "/config",
            tags=["Config"],
            summary="Update cluster config",
            description=(
                "Update cluster-wide config. Some changes apply to future launches immediately, "
                "while model-store location changes still require a restart. The deprecated "
                "model_trust compatibility field is preserved but cannot be replaced."
            ),
        )(self.update_config)
        self.app.get(
            "/store/health",
            tags=["Store"],
            summary="Get model-store health",
            description="Check whether the configured shared model store is enabled and reachable.",
        )(self.get_store_health)
        self.app.get(
            "/store/registry",
            tags=["Store"],
            summary="Get model-store registry",
            description=(
                "Return canonical store artifacts with their complete installed-card "
                "records, verification and companion ownership, current registry and "
                "advisory status, telemetry-derived fleet cache locations and coverage, "
                "and reconciliation state."
            ),
        )(self.get_store_registry)
        self.app.get(
            "/store/downloads",
            tags=["Store"],
            summary="List active store downloads",
            description=(
                "List downloads being managed by the shared model store: "
                "pending, in-progress, and failed. Failed entries stay listed "
                "with an actionable `error` explanation (for example how to "
                "authenticate for a gated Hugging Face repository) until a "
                "retry replaces them; cancelled downloads are not listed."
            ),
        )(self.get_store_downloads)
        self.app.post(
            "/store/models/{model_id:path}/download",
            tags=["Store"],
            summary="Request a store download",
            description=(
                "Ask the shared model store to download and register a base or "
                "companion artifact. Optional repository, revision, file, immutable "
                "card, ownership, and artifact-role fields bind the exact requested "
                "generation; omitted base fields inherit from the current model card."
            ),
        )(self.request_store_download)
        self.app.delete(
            "/store/models/{model_id:path}/download",
            tags=["Store"],
            summary="Cancel a store download",
            description=(
                "Cancel one pending or active shared-store model download. "
                "Partial files remain available for a later resumable request."
            ),
        )(self.cancel_store_download)
        self.app.get(
            "/store/models/{model_id:path}/download/status",
            tags=["Store"],
            summary="Get store download status",
            description="Return current status for a shared-store download request for one model.",
        )(self.get_store_download_status)
        self.app.delete(
            "/store/models/{model_id:path}",
            tags=["Store"],
            summary="Delete a model from the store",
            description="Delete a model and its shared-store artifacts from the configured model store.",
        )(self.delete_store_model)
        self.app.post(
            "/store/purge-staging",
            tags=["Downloads"],
            summary="Purge staging caches",
            description="Broadcast a staging-cache purge request to nodes, optionally scoped to one model ID.",
        )(self.purge_staging_caches)
        self.app.get(
            "/store/storage",
            tags=["Store"],
            summary="Per-node storage breakdown (local node)",
            description=(
                "Return the local node's artifact inventory across staging, direct "
                "download, and configured read-only roots. Every entry includes "
                "installed identity, verification and manifest state, companion "
                "ownership, location kind, size, last use, and live-runner use, plus "
                "event-log and filesystem capacity. Reconciliation queries each node "
                "directly; operator views use fabric telemetry."
            ),
        )(self.get_node_storage_summary)
        self.app.get(
            "/store/reconciliation",
            tags=["Store"],
            summary="Get model-store reconciliation status",
            description=(
                "Return fleet inventory progress, pending cache imports, and "
                "the most recent reconciliation failures."
            ),
        )(self.get_store_reconciliation)
        self.app.post(
            "/store/reconciliation/rescan",
            tags=["Store"],
            summary="Retry model-store reconciliation",
            description=(
                "Run an immediate node-local operator rescan. The endpoint is "
                "loopback-only; automatic periodic reconciliation remains enabled."
            ),
        )(self.rescan_store_reconciliation)
        self.app.post(
            "/store/internal/exports",
            tags=["Store Internal"],
            summary="Create a bounded artifact export capability",
            description=(
                "Internal fleet endpoint used by the authoritative store to bind "
                "one short-lived token to an exact local manifest and target node."
            ),
        )(self.create_artifact_export)
        self.app.get(
            "/store/internal/exports/{capability_token}/{relative_path:path}",
            tags=["Store Internal"],
            summary="Read a capability-bound artifact file",
            description=(
                "Internal range-capable file export. The bearer capability, target "
                "node header, path, byte ceiling, manifest, and expiry are validated."
            ),
        )(self.read_artifact_export)
        self.app.post(
            "/store/models/{model_id:path}/optimize",
            tags=["Store"],
            summary="Start model optimization",
            description=(
                "Start an optimization job for a model already present in the shared store. "
                "Use this for workflows such as OptiQ conversion or alternate artifact generation."
            ),
        )(self.optimize_model)
        self.app.post(
            "/admin/restart",
            tags=["Admin"],
            summary="Restart a node",
            description=(
                "Restart the Skulk process on this or a remote node. "
                "Pass node_install_id to resolve a stable installation identity "
                "to its current live runtime node, or node_id for a legacy "
                "session-scoped target. "
                "Active inference is interrupted, and the process is replaced; "
                "the node rejoins the cluster automatically on startup."
            ),
        )(self.restart_node)
        self.app.get(
            "/store/models/{model_id:path}/optimize/status",
            tags=["Store"],
            summary="Get model optimization status",
            description="Return the current status of an optimization job for one shared-store model.",
        )(self.get_optimize_status)
        self.app.get(
            "/filesystem/browse",
            tags=["Config"],
            summary="Browse filesystem roots for config selection",
            description="Browse a safe subset of filesystem roots so the dashboard can help users choose store and staging paths.",
        )(self.browse_filesystem)
        self.app.get(
            "/node/identity",
            tags=["Config"],
            summary="Get node identity and preferred IP",
            description="Return the node ID, hostname, and preferred LAN IP address that the dashboard uses during setup.",
        )(self.get_node_identity)
        self.app.get(
            "/v1/connectivity/tailscale",
            tags=["Connectivity"],
            summary="Get Tailscale status",
            description=(
                "Return whether tailscaled is running on this node and, if so, the node's "
                "Tailscale IP, hostname, DNS name, and tailnet. "
                "Pass the node_id query parameter to proxy the request to a specific cluster node."
            ),
        )(self.get_tailscale_status)
        self.app.get(
            "/v1/connectivity/remote-access",
            tags=["Connectivity"],
            summary="Get remote access info",
            description=(
                "Return the preferred URL, local LAN access, and Tailscale overlay access for "
                "this node. preferredUrl is the Tailscale URL when tailscaled is running, "
                "otherwise the LAN URL. operatorUrl appends /operator and is suitable for QR "
                "code generation so mobile users land directly on the operator panel."
            ),
        )(self.get_remote_access)

    @staticmethod
    async def _load_authorized_model_card(model_id: ModelId) -> ModelCard:
        """Return catalog truth without turning lookup into model authorization.

        Args:
            model_id: Exact selectable alias requested by an API caller.

        Returns:
            The signed, bundled, installed, or explicitly added model card.

        Raises:
            HTTPException: With status 404 when the alias is not authorized in
                the local catalog.
        """
        try:
            return await ModelCard.load(model_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def place_instance(self, payload: PlaceInstanceParams):
        command = PlaceInstance(
            model_card=await self._load_authorized_model_card(payload.model_id),
            sharding=payload.sharding,
            instance_meta=payload.instance_meta,
            min_nodes=payload.min_nodes,
            excluded_nodes=list(payload.excluded_nodes),
        )

        # Dry-run the placement against this node's replicated state before
        # forwarding the command. Without this, an impossible placement is
        # acknowledged with "Command received", fails silently on the master,
        # and the client only ever observes 404s on subsequent requests.
        # PlacementInfoPendingError covers the cluster-startup window where
        # cluster info is still gossiping (connection edges lag node
        # identities, then memory info lags the edges) — placement would
        # succeed seconds later, so wait it out rather than failing a
        # placement that raced cluster formation. The dry-run uses a copy
        # because place_instance normalizes single-node commands in place.
        #
        # While the API is paused (election/reset), self.state may be mid-
        # rebuild — validating against it could produce a false 400. Gate the
        # dry-run on the same unpause condition _send() uses.
        while self.paused:
            await self.paused_ev.wait()
        deadline = time.monotonic() + _PLACEMENT_INFO_WAIT_SECONDS
        while True:
            try:
                get_instance_placements(
                    copy.deepcopy(command),
                    topology=self.state.topology,
                    current_instances=self.state.instances,
                    node_memory=self._telemetry_view.node_memory,
                    node_network=self.state.node_network,
                    download_status=self._telemetry_view.effective_downloads(
                        self.state.downloads
                    ),
                    excluded_nodes=set(command.excluded_nodes),
                    node_resources=self._telemetry_view.node_resources,
                    node_vram=usable_vram_by_node(
                        self._telemetry_view.node_system,
                        self._telemetry_view.node_resources,
                        node_memory=self._telemetry_view.node_memory,
                    ),
                    unified_memory_gpu_nodes=unified_memory_gpu_node_ids(
                        self._telemetry_view.node_system,
                        self._telemetry_view.node_resources,
                        node_memory=self._telemetry_view.node_memory,
                    ),
                    approved_remote_code_identities=self._cluster_remote_code_approvals(),
                )
                break
            except PlacementInfoPendingError as exc:
                if time.monotonic() >= deadline:
                    raise HTTPException(
                        status_code=503,
                        detail=f"{exc} (waited {_PLACEMENT_INFO_WAIT_SECONDS:.0f}s)",
                        headers={"X-Skulk-Placement-Failure": exc.code},
                    ) from exc
                await anyio.sleep(1)
            except PlacementError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                    headers={"X-Skulk-Placement-Failure": exc.code},
                ) from exc
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                    headers={"X-Skulk-Placement-Failure": "no_valid_placement"},
                ) from exc

        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            instance_id=InstanceId(str(command.command_id)),
            model_card=command.model_card,
        )

    async def create_instance(
        self, payload: CreateInstanceParams
    ) -> CreateInstanceResponse:
        instance = payload.instance
        if instance.system_role is not None:
            # System placements (the intelligent-fabric steward) are minted
            # and maintained by the fabric's own invariant; accepting one
            # from a caller would let an arbitrary instance impersonate the
            # steward and suppress the configured placement.
            raise HTTPException(
                status_code=400,
                detail=(
                    "system_role placements are fabric-managed and cannot "
                    "be created through this endpoint"
                ),
            )
        model_card = await self._load_authorized_model_card(
            instance.shard_assignments.model_id
        )
        try:
            require_instance_model_card_identity(instance, model_card)
            require_instance_model_code_approval(
                instance,
                self._cluster_remote_code_approvals(),
            )
        except PlacementError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
                headers={"X-Skulk-Placement-Failure": exc.code},
            ) from exc
        required_memory = model_card.storage_size
        available_memory = self._calculate_total_available_memory()

        if required_memory > available_memory:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient memory to create instance. Required: {required_memory.in_gb:.1f}GB, Available: {available_memory.in_gb:.1f}GB",
            )

        command = CreateInstance(
            instance=instance,
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            instance_id=instance.instance_id,
            model_card=model_card,
        )

    async def get_placement(
        self,
        model_id: ModelId,
        sharding: Sharding = Sharding.Pipeline,
        instance_meta: InstanceMeta = InstanceMeta.MlxRing,
        min_nodes: int = 1,
    ) -> Instance:
        model_card = await self._load_authorized_model_card(model_id)

        try:
            placements = get_instance_placements(
                PlaceInstance(
                    model_card=model_card,
                    sharding=sharding,
                    instance_meta=instance_meta,
                    min_nodes=min_nodes,
                ),
                node_memory=self._telemetry_view.node_memory,
                node_network=self.state.node_network,
                topology=self.state.topology,
                current_instances=self.state.instances,
                download_status=self._telemetry_view.effective_downloads(
                    self.state.downloads
                ),
                node_resources=self._telemetry_view.node_resources,
                node_vram=usable_vram_by_node(
                    self._telemetry_view.node_system,
                    self._telemetry_view.node_resources,
                    node_memory=self._telemetry_view.node_memory,
                ),
                unified_memory_gpu_nodes=unified_memory_gpu_node_ids(
                    self._telemetry_view.node_system,
                    self._telemetry_view.node_resources,
                    node_memory=self._telemetry_view.node_memory,
                ),
                approved_remote_code_identities=self._cluster_remote_code_approvals(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        current_ids = set(self.state.instances.keys())
        new_ids = [
            instance_id for instance_id in placements if instance_id not in current_ids
        ]
        if len(new_ids) != 1:
            raise HTTPException(
                status_code=500,
                detail="Expected exactly one new instance from placement",
            )

        return placements[new_ids[0]]

    async def get_placement_previews(
        self,
        model_id: ModelId,
        node_ids: Annotated[list[NodeId] | None, Query()] = None,
        excluded_node_ids: Annotated[list[NodeId] | None, Query()] = None,
    ) -> PlacementPreviewResponse:
        seen: set[tuple[ModelId, Sharding, InstanceMeta, int]] = set()
        previews: list[PlacementPreview] = []
        required_nodes = set(node_ids) if node_ids else None
        excluded_nodes = set(excluded_node_ids) if excluded_node_ids else None

        if len(list(self.state.topology.list_nodes())) == 0:
            return PlacementPreviewResponse(previews=[])

        try:
            model_card = await self._load_authorized_model_card(model_id)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to load model card: {exc}"
            ) from exc
        trust_requirement = None
        matching_engine_support = get_model_engine_support(model_card)
        supported_claim_ids = [
            claim.claim_id
            for claim in matching_engine_support
            if claim.status == "supported"
        ]
        incomplete_capabilities = sorted(
            claim.capability_id
            for claim in model_card.registry_capability_claims
            if claim.status == "incomplete"
        )
        support_gap_detail = (
            "Artifact evidence marks required capability components incomplete: "
            + ", ".join(incomplete_capabilities)
            if incomplete_capabilities
            else (
                "Signed engine support matches this artifact, but no node advertises "
                "the exact supported engine build and hardware constraint."
                if supported_claim_ids
                else (
                    "Model or artifact capabilities are known, but no active signed "
                    "supported engine claim matches this architecture and artifact."
                    if model_card.registry_capability_claims
                    and not model_card.placement.compatible_backends
                    else None
                )
            )
        )

        def compatibility_for_instance(
            instance: Instance,
        ) -> tuple[Literal["card", "signed_engine_support"], list[str]]:
            """Identify whether immutable card or signed support admitted a backend."""
            resolved_backends = {
                shard.resolved_backend
                for shard in instance.shard_assignments.runner_to_shard.values()
                if shard.resolved_backend is not None
            }
            uses_matrix = bool(
                resolved_backends - model_card.placement.compatible_backends
            )
            applicable_claim_ids: set[str] = set()
            if uses_matrix:
                for (
                    node_id,
                    runner_id,
                ) in instance.shard_assignments.node_to_runner.items():
                    backend = instance.shard_assignments.runner_to_shard[
                        runner_id
                    ].resolved_backend
                    resources = self._telemetry_view.node_resources.get(node_id)
                    if backend is None or resources is None:
                        continue
                    engine = engine_of(backend)
                    build = resources.engine_builds.get(
                        backend,
                        resources.engine_builds.get(engine) if engine else None,
                    )
                    for claim in matching_engine_support:
                        hardware_matches = not claim.hardware_classes or bool(
                            set(claim.hardware_classes) & resources.hardware_classes
                        )
                        if (
                            claim.status == "supported"
                            and claim.engine in {backend, engine}
                            and claim.engine_build == build
                            and hardware_matches
                        ):
                            applicable_claim_ids.add(claim.claim_id)
            return (
                "signed_engine_support" if uses_matrix else "card",
                sorted(applicable_claim_ids),
            )

        placement_node_vram = usable_vram_by_node(
            self._telemetry_view.node_system,
            self._telemetry_view.node_resources,
            node_memory=self._telemetry_view.node_memory,
        )
        placement_unified_memory_gpu_nodes = unified_memory_gpu_node_ids(
            self._telemetry_view.node_system,
            self._telemetry_view.node_resources,
            node_memory=self._telemetry_view.node_memory,
        )
        instance_combinations: list[tuple[Sharding, InstanceMeta, int]] = []
        for sharding in (Sharding.Pipeline, Sharding.Tensor):
            for instance_meta in (InstanceMeta.MlxRing, InstanceMeta.MlxJaccl):
                instance_combinations.extend(
                    [
                        (sharding, instance_meta, i)
                        for i in range(
                            1, len(list(self.state.topology.list_nodes())) + 1
                        )
                    ]
                )
        # TODO: PDD
        # instance_combinations.append((Sharding.PrefillDecodeDisaggregation, InstanceMeta.MlxRing, 1))

        for sharding, instance_meta, min_nodes in instance_combinations:
            try:
                placements = get_instance_placements(
                    PlaceInstance(
                        model_card=model_card,
                        sharding=sharding,
                        instance_meta=instance_meta,
                        min_nodes=min_nodes,
                    ),
                    node_memory=self._telemetry_view.node_memory,
                    node_network=self.state.node_network,
                    topology=self.state.topology,
                    current_instances=self.state.instances,
                    required_nodes=required_nodes,
                    download_status=self._telemetry_view.effective_downloads(
                        self.state.downloads
                    ),
                    excluded_nodes=excluded_nodes,
                    node_resources=self._telemetry_view.node_resources,
                    node_vram=placement_node_vram,
                    unified_memory_gpu_nodes=placement_unified_memory_gpu_nodes,
                    approved_remote_code_identities=self._cluster_remote_code_approvals(),
                )
            except ValueError as exc:
                if (model_card.model_id, sharding, instance_meta, 0) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=model_card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=None,
                            error=str(exc),
                            error_code=(
                                exc.code
                                if isinstance(exc, PlacementError)
                                else "no_valid_placement"
                            ),
                            compatibility_detail=support_gap_detail,
                        )
                    )
                seen.add((model_card.model_id, sharding, instance_meta, 0))
                continue

            current_ids = set(self.state.instances.keys())
            new_instances = [
                instance
                for instance_id, instance in placements.items()
                if instance_id not in current_ids
            ]

            if len(new_instances) != 1:
                if (model_card.model_id, sharding, instance_meta, 0) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=model_card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=None,
                            error="Expected exactly one new instance from placement",
                            error_code="no_valid_placement",
                            compatibility_detail=support_gap_detail,
                        )
                    )
                seen.add((model_card.model_id, sharding, instance_meta, 0))
                continue

            instance = new_instances[0]
            shard_assignments = instance.shard_assignments
            placement_node_ids = list(shard_assignments.node_to_runner.keys())

            memory_delta_by_node: dict[str, int] = {}
            if placement_node_ids:
                total_bytes = model_card.storage_size.in_bytes
                per_node = total_bytes // len(placement_node_ids)
                remainder = total_bytes % len(placement_node_ids)
                for index, node_id in enumerate(sorted(placement_node_ids, key=str)):
                    extra = 1 if index < remainder else 0
                    memory_delta_by_node[str(node_id)] = per_node + extra

            # Report the MINTED instance's meta, not the requested one:
            # placement normalizes shapes (single-node cycles become rings
            # regardless of requested meta; an RPC-capable cycle mints a
            # LlamaRpcInstance from a ring request, #328). Consumers
            # (dashboard, test harness) match previews by instance_meta, so
            # the preview must be truthful about the shape a real placement
            # would produce.
            minted_meta = instance_meta_of(instance)
            compatibility_source, applicable_claim_ids = compatibility_for_instance(
                instance
            )
            if (
                model_card.model_id,
                sharding,
                minted_meta,
                len(placement_node_ids),
            ) not in seen:
                previews.append(
                    PlacementPreview(
                        model_id=model_card.model_id,
                        sharding=sharding,
                        instance_meta=minted_meta,
                        instance=instance,
                        memory_delta_by_node=memory_delta_by_node or None,
                        error=None,
                        compatibility_source=compatibility_source,
                        support_claim_ids=applicable_claim_ids,
                    )
                )
            seen.add(
                (
                    model_card.model_id,
                    sharding,
                    minted_meta,
                    len(placement_node_ids),
                )
            )

        # Per-host single-node alternatives (#557): the ranked pick above is
        # one preview per shape, so on a heterogeneous fleet the winner
        # (typically the largest free GPU) hides every other host that passes
        # admission, and the operator cannot choose cost/locality/keeping the
        # big GPU free. Re-run the planner once per remaining host as a
        # required single-node placement and surface the viable ones, marked
        # as alternatives. Skipped when the caller already constrained hosts
        # via node_ids. Hosts that fail admission stay silent: the ranked
        # pass already reported shape-level errors.
        if required_nodes is None:
            winner_hosts: set[NodeId] = set()
            # The (sharding, meta) shapes that actually single-node place for
            # this model, taken from the ranked previews. A model whose only
            # single-node shape is Tensor (e.g. DeepSeek-V3.1-8bit) would be
            # invisible if alternatives hardcoded Pipeline/MlxRing (#557).
            single_node_shapes: list[tuple[Sharding, InstanceMeta]] = []
            for preview in previews:
                if preview.instance is None:
                    continue
                nodes_of_preview = list(
                    preview.instance.shard_assignments.node_to_runner.keys()
                )
                if len(nodes_of_preview) != 1:
                    continue
                winner_hosts.add(nodes_of_preview[0])
                shape = (preview.sharding, preview.instance_meta)
                if shape not in single_node_shapes:
                    single_node_shapes.append(shape)
            # Hoisted: both depend only on telemetry snapshots and current
            # instances, not on the candidate.
            existing_instance_ids = set(self.state.instances.keys())
            for candidate in self.state.topology.list_nodes():
                if candidate in winner_hosts:
                    continue
                if excluded_nodes is not None and candidate in excluded_nodes:
                    continue
                for alt_sharding, alt_meta in single_node_shapes:
                    try:
                        alt_placements = get_instance_placements(
                            PlaceInstance(
                                model_card=model_card,
                                sharding=alt_sharding,
                                instance_meta=alt_meta,
                                min_nodes=1,
                            ),
                            node_memory=self._telemetry_view.node_memory,
                            node_network=self.state.node_network,
                            topology=self.state.topology,
                            current_instances=self.state.instances,
                            required_nodes={candidate},
                            download_status=self._telemetry_view.effective_downloads(
                                self.state.downloads
                            ),
                            excluded_nodes=excluded_nodes,
                            node_resources=self._telemetry_view.node_resources,
                            node_vram=placement_node_vram,
                            unified_memory_gpu_nodes=placement_unified_memory_gpu_nodes,
                            approved_remote_code_identities=self._cluster_remote_code_approvals(),
                        )
                    except ValueError:
                        continue
                    alt_instances = [
                        instance
                        for instance_id, instance in alt_placements.items()
                        if instance_id not in existing_instance_ids
                    ]
                    if len(alt_instances) != 1:
                        continue
                    alt_instance = alt_instances[0]
                    alt_nodes = list(
                        alt_instance.shard_assignments.node_to_runner.keys()
                    )
                    # The planner may satisfy required_nodes with a larger
                    # cycle; only a true single-host placement on the candidate
                    # is an alternative the operator can reason about.
                    if alt_nodes != [candidate]:
                        continue
                    alt_compatibility_source, alt_claim_ids = (
                        compatibility_for_instance(alt_instance)
                    )
                    previews.append(
                        PlacementPreview(
                            model_id=model_card.model_id,
                            sharding=alt_sharding,
                            instance_meta=instance_meta_of(alt_instance),
                            instance=alt_instance,
                            memory_delta_by_node={
                                str(candidate): model_card.storage_size.in_bytes
                            },
                            error=None,
                            alternative=True,
                            compatibility_source=alt_compatibility_source,
                            support_claim_ids=alt_claim_ids,
                        )
                    )
                    # One alternative per host is enough; stop at the first
                    # shape that places.
                    break

        return PlacementPreviewResponse(
            previews=[
                preview.model_copy(update={"trust_requirement": trust_requirement})
                for preview in previews
            ]
        )

    def get_instance(self, instance_id: InstanceId) -> Instance:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")
        return self.state.instances[instance_id]

    async def _steward_chat_completions(
        self,
        payload: ChatCompletionRequest,
        *,
        proposals_allowed: bool = True,
    ) -> StreamingResponse:
        """Serve the steward through the standard chat-completions surface.

        The steward's tool surface belongs to the server, so client tool
        definitions are rejected rather than silently ignored, and client
        system messages are dropped in favor of the steward's own prompt
        (documented in the API guide). History is the caller's user and
        assistant turns; multimodal content parts are flattened to their
        text. Both streaming and non-streaming ride the ordinary adapters
        over the harness's chunk stream.

        Refusals happen before the response begins, in this order: 400 for
        client tool definitions, 400 for a tool_choice forcing a function
        (the steward accepts no client tools, so a forced choice can never
        be honored), 404 when intelligent-fabric mode is off, 400 for a
        conversation with no question to answer, then 503 with the steward
        status payload while no steward is ready to answer.

        Raises:
            HTTPException: 400, 404, or 503 per the order above.

        Returns:
            The streaming (SSE) or collected (JSON) chat-completions
            response for one steward turn.
        """
        if payload.tools:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The steward's tool surface is server-side; client tool "
                    "definitions are not accepted for this model"
                ),
            )
        # The reserved id branches before the ordinary tool_choice
        # resolution, so the forced-name rejection has to be repeated here:
        # a caller forcing a function at a model that accepts no client
        # tools gets the same 400 the documented boundary gives, not a
        # steward answer that silently ignored the choice.
        resolve_tool_choice(payload.tools, payload.tool_choice)
        if not self._intelligent_fabric_enabled():
            raise HTTPException(
                status_code=404,
                detail=(
                    "Model not available: intelligent-fabric mode is "
                    "disabled. Enable intelligent_fabric in Settings to "
                    "use the steward."
                ),
            )
        history: list[StewardChatMessage] = []
        for message in payload.messages:
            if message.role not in ("user", "assistant"):
                continue
            content = message.content
            if isinstance(content, list):
                text = " ".join(
                    part.text
                    for part in content
                    if isinstance(part, ChatCompletionMessageText) and part.text
                )
            elif isinstance(content, ChatCompletionMessageText):
                text = content.text
            elif isinstance(content, str):
                text = content
            else:
                text = ""
            if text:
                history.append(StewardChatMessage(role=message.role, content=text))
        if not history or history[-1].role != "user":
            # A steward turn answers an operator; assistant-only history or
            # a trailing assistant message has no question to investigate.
            raise HTTPException(
                status_code=400,
                detail=(
                    "The conversation must end with a user message for the "
                    "steward to answer"
                ),
            )
        status = self._steward_status()
        if not status.ready:
            # Readiness preflight: a steward that is still placing, staging
            # weights, or loading cannot answer, and saying so as a 503 with
            # the status payload is a contract a client can act on (retry,
            # show progress). Reporting it as a 200 SSE error chunk instead
            # made "the fabric is still setting up" indistinguishable from
            # "the model failed mid-answer" to every OpenAI-compatible
            # client. The in-stream error path stays for the race where the
            # placement disappears between this check and dispatch.
            detail: dict[str, object] = status.model_dump(mode="json")
            detail["message"] = STEWARD_NOT_READY_MESSAGES[status.state]
            raise HTTPException(
                status_code=503,
                detail=detail,
                headers={"Retry-After": str(STEWARD_RETRY_AFTER_SECONDS)},
            )
        harness = StewardHarness(self, proposals_allowed=proposals_allowed)
        command_id = CommandId()
        history, system_prompt, task_params = await self._steward_extension_transform(
            history, stream=payload.stream
        )
        chunk_stream = harness.run_turn_chunks(history, system_prompt=system_prompt)
        if self._extensions is not None and self._extensions.has_chat_middleware:
            chunk_stream = self._extensions.tap_chat_stream(
                self._extension_context, task_params, chunk_stream
            )
        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_chat_stream(command_id, chunk_stream),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        # Returned as a stream even though the body is one complete object:
        # Starlette cancels a StreamingResponse body generator when the caller
        # disconnects, which propagates into the chunk stream and sends
        # TaskCancelled so the runner stops work nobody is waiting for. That
        # costs the ability to choose a status after the outcome is known, so
        # a post-commit failure is reported in the body instead (#872).
        return StreamingResponse(
            collect_chat_response(command_id, chunk_stream),
            media_type="application/json",
        )

    async def _steward_extension_transform(
        self, history: list[StewardChatMessage], *, stream: bool
    ) -> tuple[list[StewardChatMessage], str, TextGenerationTaskParams]:
        """Run chat middleware's request transform over one steward turn.

        The steward answers through a bespoke harness rather than the ordinary
        dispatch path, so ``chat_completions`` returns before its extension
        hook. Without this, an installed chat middleware silently sees every
        conversation on the node except the steward's own, which is the one
        conversation an ambient-memory or policy extension most needs.

        The turn is presented in the same canonical shape the ordinary path
        uses: the steward's system prompt as ``instructions`` and the operator
        history as ``input``. Those are also the only two channels read back,
        because they are the only ones the harness owns (sampling, tools, and
        the model are the steward's, not the caller's). A transform that
        leaves the turn without a trailing user message is discarded in full
        rather than obeyed, since the harness's contract is that a steward
        turn answers an operator question.

        Every middleware call inside is guarded by the loader, so a raising
        extension is logged and the steward answers unchanged.

        Args:
            history: The turn's user/assistant conversation.
            stream: Whether the caller requested SSE, mirrored into the params
                so observers see the real request shape.

        Returns:
            The history to run, the system prompt to run it with, and the
            params describing that same turn. The three always agree: on a
            rejected transform all three are the originals, so the response
            tap never describes a turn that was not the one served.
        """
        original = TextGenerationTaskParams(
            model=ModelId(STEWARD_VIRTUAL_MODEL_ID),
            input=[
                InputMessage(role=message.role, content=message.content)
                for message in history
            ],
            instructions=STEWARD_SYSTEM_PROMPT,
            stream=stream,
        )
        if self._extensions is None or not self._extensions.has_chat_middleware:
            return history, STEWARD_SYSTEM_PROMPT, original
        transformed = await self._extensions.transform_chat_request(
            self._extension_context, original
        )
        candidate = [
            StewardChatMessage(role=message.role, content=message.content)
            for message in transformed.input
            if message.role in ("user", "assistant") and message.content
        ]
        if not candidate or candidate[-1].role != "user":
            # Reject the whole transform, not just its history. Keeping the
            # transformed instructions or params here would run the turn on
            # one conversation while telling observers it was another, and
            # an ambient-memory or audit middleware would then file the
            # answer against the wrong conversation.
            logger.warning(
                "chat middleware left the steward turn without a trailing "
                "user message; discarding the transform"
            )
            return history, STEWARD_SYSTEM_PROMPT, original
        # Report the turn as it will actually run, not as the middleware
        # wrote it. Built from ``original`` rather than from the transform,
        # so accepting the two channels the harness honors cannot smuggle in
        # any of the ones it ignores: the steward's own sampling, tool set,
        # and response mode are what serve, whatever a middleware returned,
        # and an audit observer must not be told otherwise.
        prompt = transformed.instructions or STEWARD_SYSTEM_PROMPT
        effective = original.model_copy(
            update={
                "input": [
                    InputMessage(role=message.role, content=message.content)
                    for message in candidate
                ],
                "instructions": prompt,
            }
        )
        return candidate, prompt, effective

    async def _steward_canary_loop(self) -> None:
        """Deterministic degraded-but-alive detection for the steward.

        The lowest API-advertising node probes the steward generation path on
        a slow cadence; three consecutive failures tear the instance down
        (the master's invariant re-places it within a tick). Detection is
        code-checked, repair is the fabric's deterministic machinery: no
        model judges another model's health. Skips whenever the mode is
        off, this node is not the elected API, the runner is not idle-Ready, or a
        task is in flight (the worker's wedge detector owns the busy case).

        API election uses explicit ``NodeResources.api_available`` telemetry.
        Generation dispatch is already node-independent and pinned to the
        steward instance, so a worker launched with ``--no-api`` remains
        covered.
        """
        from skulk.api.steward import (
            CANARY_FAILURE_THRESHOLD,
            CANARY_INTERVAL_SECONDS,
            canary_probe_target,
        )

        canary = self._steward_canary
        while True:
            await anyio.sleep(CANARY_INTERVAL_SECONDS)
            try:
                if not self._intelligent_fabric_enabled():
                    canary.clear_failures()
                    continue
                advertised_api_nodes = {
                    peer_node_id
                    for peer_node_id, resources in self._telemetry_view.node_resources.items()
                    if resources.api_available
                }
                advertised_api_nodes.add(self.node_id)
                target = canary_probe_target(
                    self.state.instances,
                    self.state.runners,
                    self.state.tasks,
                    self.node_id,
                    advertised_api_nodes,
                )
                if target is None:
                    # A skipped probe (busy, loading, not the elected
                    # prober) is not evidence of health: keep the failure
                    # count so "3 consecutive failures" means three failed
                    # probes with no intervening success. Reset only when
                    # the steward instance itself is gone.
                    tracked = canary.instance_id
                    if tracked is not None and tracked not in self.state.instances:
                        canary.reset()
                    continue
                if target != canary.instance_id:
                    canary.track(target)
                harness = StewardHarness(self)
                located = harness.steward_instance()
                if located is None or located[0] != target:
                    continue
                if await harness.canary_probe(target, located[1]):
                    canary.clear_failures()
                    continue
                failures = canary.record_failure()
                logger.warning(
                    f"Steward canary probe failed ({failures}/"
                    f"{CANARY_FAILURE_THRESHOLD}) for instance {target}"
                )
                if failures >= CANARY_FAILURE_THRESHOLD:
                    logger.error(
                        f"Steward instance {target} failed "
                        f"{CANARY_FAILURE_THRESHOLD} consecutive canary "
                        "probes; tearing it down for re-placement"
                    )
                    await self._send(_steward_canary_failure_command(target))
                    canary.clear_failures()
            except Exception:
                # The canary must never take the API down; a broken probe
                # pass just waits for the next interval.
                logger.warning("Steward canary pass failed", exc_info=True)

    async def get_steward_status(self) -> "StewardStatusResponse":
        """Report intelligent-fabric mode and the current steward placement."""
        return self._steward_status()

    async def submit_steward_action_proposal(
        self, proposal: StewardActionProposal
    ) -> None:
        """Submit one inert steward proposal to the authoritative master.

        Args:
            proposal: Exact typed action, evidence, effect, and expiry produced
                by the resident steward harness.

        Side effects:
            Sends only :class:`ProposeStewardAction`; no proposed action is
            executed until a separate authenticated decision is ordered.
        """
        if not self._intelligent_fabric_enabled():
            raise ValueError("Intelligent-fabric mode is disabled")
        await self._send(ProposeStewardAction(proposal=proposal))

    async def load_authorized_steward_model_card(self, model_id: ModelId) -> ModelCard:
        """Resolve exact catalog truth for a steward placement proposal.

        Args:
            model_id: Exact selectable model alias proposed by the steward.

        Returns:
            The signed, bundled, installed, or explicitly added model card.

        Raises:
            HTTPException: When the alias is not authorized in the catalog.
        """
        return await self._load_authorized_model_card(model_id)

    async def list_steward_action_proposals(
        self, request: Request
    ) -> list[StewardActionProposalView]:
        """Return the bounded proposal audit to an authorized operator.

        Args:
            request: Incoming request proving operator authority.

        Returns:
            Newest-first replicated safe summaries without internal identities.

        Raises:
            HTTPException: If the caller lacks operator authority.
        """
        self._require_operator_mutation(request)
        return [
            steward_action_proposal_view(proposal)
            for proposal in sorted(
                self.state.steward_action_proposals.values(),
                key=lambda proposal: proposal.created_at,
                reverse=True,
            )
        ]

    def steward_active_download_attempt(
        self, node_id: NodeId, model_id: ModelId
    ) -> DownloadAttemptId | None:
        """Return the exact live download attempt for a model on one node.

        Args:
            node_id: Exact internal node selected from a unique friendly name.
            model_id: Exact model alias selected by the steward.

        Returns:
            The attempt identity for pending or ongoing effective download
            truth, or ``None`` when no safely identifiable attempt is active.
        """
        for progress in self._telemetry_view.effective_downloads(
            self.state.downloads
        ).get(node_id, ()):
            if (
                isinstance(progress, LiveDownloadProgress)
                and progress.shard_metadata.model_card.model_id == model_id
            ):
                return progress.attempt_id
        return None

    async def decide_steward_action_proposal(
        self,
        proposal_id: StewardActionProposalId,
        payload: StewardActionDecisionRequest,
        request: Request,
    ) -> StewardActionDecisionResponse:
        """Approve or reject one exact pending proposal through the master.

        Args:
            proposal_id: Stable proposal identity returned by the steward.
            payload: Explicit approve or reject decision.
            request: Incoming request proving operator authority.

        Returns:
            Acceptance of the decision command. Clients poll the proposal list
            for the authoritative master-ordered result.

        Raises:
            HTTPException: If authorization fails or the proposal is absent or
                already terminal in this node's replicated view.
        """
        self._require_operator_mutation(request)
        proposal = self.state.steward_action_proposals.get(proposal_id)
        if proposal is None:
            raise HTTPException(status_code=404, detail="Steward proposal not found")
        if proposal.status != "pending":
            raise HTTPException(
                status_code=409,
                detail=f"Steward proposal is already {proposal.status}",
            )
        actor = (
            "authenticated_operator_gateway"
            if request.scope.get(OPERATOR_GATEWAY_AUTHORIZED_SCOPE_KEY) is True
            else "trusted_fabric_operator"
        )
        command = DecideStewardAction(
            proposal_id=proposal_id,
            approved=payload.approved,
            decided_by=actor,
        )
        await self._send(command)
        return StewardActionDecisionResponse(
            proposal_id=str(proposal_id),
            command_id=str(command.command_id),
            message=(
                "Approval submitted to the elected master."
                if payload.approved
                else "Rejection submitted to the elected master."
            ),
        )

    def _steward_model_is_downloading(self, model_id: str) -> bool:
        """Whether any node holds a live download record for ``model_id``.

        Live means Pending or Ongoing in the effective view (telemetry's
        non-terminal records merged over the terminal ones in state), which
        is exactly the window in which the steward exists as a placement but
        its weights are not on disk yet.
        """
        for records in self._telemetry_view.effective_downloads(
            self.state.downloads
        ).values():
            for record in records:
                if not isinstance(record, LiveDownloadProgress):
                    continue
                if str(record.shard_metadata.model_card.model_id) == model_id:
                    return True
        return False

    def _steward_status(self) -> "StewardStatusResponse":
        """Build the steward status snapshot.

        Shared by ``GET /v1/steward`` and the chat-completions readiness
        preflight so a client polling the status endpoint and a client
        posting a turn can never disagree about whether the steward is
        servable.
        """
        located = StewardHarness(self).steward_instance()
        ready = False
        downloading = False
        canary_failures = 0
        if located is not None:
            instance = self.state.instances.get(located[0])
            if instance is not None:
                runner_ids = instance.shard_assignments.node_to_runner.values()
                # Running counts as ready: a steward mid-generation is
                # serving, not still loading.
                ready = bool(runner_ids) and all(
                    isinstance(
                        self.state.runners.get(runner_id),
                        (RunnerReady, RunnerRunning),
                    )
                    for runner_id in runner_ids
                )
            if not ready:
                downloading = self._steward_model_is_downloading(located[1])
            canary_failures = self._steward_canary.consecutive_failures_for(located[0])
        enabled = self._intelligent_fabric_enabled()
        desired_model = located[1] if located is not None else None
        transition: Literal["idle", "prestaging", "replacing", "repairing"] = (
            "repairing" if enabled and located is None else "idle"
        )
        transition_progress: float | None = None
        try:
            config = load_skulk_config()
        except Exception:
            config = None
        fabric = config.intelligent_fabric if config is not None else None
        if located is not None and fabric is not None:
            current_model = located[1]
            try:
                current_index = fabric.steward_models.index(current_model)
            except ValueError:
                current_index = len(fabric.steward_models)
            effective_downloads = self._telemetry_view.effective_downloads(
                self.state.downloads
            )
            for candidate in fabric.steward_models[:current_index]:
                candidate_records = [
                    record
                    for records in effective_downloads.values()
                    for record in records
                    if isinstance(record, LiveDownloadProgress)
                    and str(record.shard_metadata.model_card.model_id) == candidate
                ]
                if not candidate_records:
                    continue
                desired_model = candidate
                transition = "prestaging"
                downloaded = 0
                total = 0
                for record in candidate_records:
                    if isinstance(record, DownloadOngoing):
                        downloaded += record.download_progress.downloaded.in_bytes
                        total += record.download_progress.total.in_bytes
                    else:
                        downloaded += record.downloaded.in_bytes
                        total += record.total.in_bytes
                transition_progress = (
                    min(1.0, downloaded / total) if total > 0 else None
                )
                break
        return StewardStatusResponse(
            enabled=enabled,
            present=located is not None,
            ready=ready,
            steward_model=located[1] if located is not None else None,
            instance_id=str(located[0]) if located is not None else None,
            desired_model=desired_model,
            transition=transition,
            progress=transition_progress,
            state=derive_steward_state(
                enabled=enabled,
                present=located is not None,
                ready=ready,
                downloading=downloading,
                canary_failures=canary_failures,
            ),
        )

    def _intelligent_fabric_enabled(self) -> bool:
        """Whether intelligent-fabric mode is currently enabled.

        Reads the on-disk cluster config (kept current on every node by
        SyncConfig) so the answer tracks fleet-wide Settings changes made
        from any node, falling back to the startup-parsed config when the
        file is momentarily unreadable.
        """
        try:
            config = load_skulk_config()
        except Exception:
            config = self._skulk_config
        fabric = config.intelligent_fabric if config is not None else None
        return fabric is not None and fabric.enabled

    async def delete_instance(self, instance_id: InstanceId) -> DeleteInstanceResponse:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")
        instance = self.state.instances[instance_id]
        if instance.system_role == "steward" and self._intelligent_fabric_enabled():
            # The steward is a fabric-maintained system placement: while the
            # mode is on, the master would immediately re-place it anyway, so
            # an ordinary delete is refused loudly instead of producing a
            # confusing delete-then-reappear. Internal correctness paths
            # (worker give-up on a crashed instance, repair teardown) send
            # DeleteInstance directly on the command plane and are unaffected.
            raise HTTPException(
                status_code=409,
                detail=(
                    "This is the intelligent-fabric steward placement, which "
                    "the fabric maintains automatically. Disable intelligent "
                    "fabric in Settings to remove it."
                ),
            )

        command = DeleteInstance(
            instance_id=instance_id,
        )
        await self._send(command)
        return DeleteInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            instance_id=instance_id,
        )

    async def cancel_command(self, command_id: CommandId) -> CancelCommandResponse:
        """Cancel an active command by closing its stream and notifying workers."""
        sender = (
            self._text_generation_queues.get(command_id)
            or self._image_generation_queues.get(command_id)
            or self._embedding_queues.get(command_id)
            or self._audio_speech_queues.get(command_id)
            or self._audio_transcription_queues.get(command_id)
        )
        if sender is None:
            raise HTTPException(
                status_code=404,
                detail="Command not found or already completed",
            )

        await self._send(TaskCancelled(cancelled_command_id=command_id))
        # Suppress the final TaskFinished emitted by local stream cleanup so the
        # worker can observe the Cancelled task and deliver runner-local cancel
        # before event-sourced task deletion happens.
        self._cancelled_command_ids.add(command_id)
        sender.close()

        return CancelCommandResponse(
            message="Command cancelled.",
            command_id=command_id,
        )

    def _command_task_is_terminal(self, command_id: CommandId) -> bool:
        """Return True if the master already considers this command's task done.

        Used to disambiguate a mid-stream idle stall (#298 review): if the task
        has already reached a terminal status (Complete/Failed/TimedOut/Cancelled)
        the stall is just a dropped FINAL data-plane chunk — the runner finished.
        Sending TaskCancelled there is a no-op on a completed runner and leaves the
        master's task/command mapping leaked forever; we must let the normal
        TaskFinished path run instead. A non-terminal task means a genuinely stuck
        runner, which TaskCancelled correctly tears down.
        """

        target = str(command_id)
        for task in self.state.tasks.values():
            if self._task_command_id(task) == target:
                return task.task_status in _TERMINAL_TASK_STATUSES
        return False

    async def _finalize_command_stream(
        self,
        command_id: CommandId,
        queue_map: dict[CommandId, Sender[object]],
    ) -> None:
        """Clean up a local command stream and report natural completion.

        Local cancellations intentionally suppress TaskFinished. Otherwise the
        master can delete the task before workers synthesize CancelTask for the
        runner that is still executing it.
        """

        should_send_finished = command_id not in self._cancelled_command_ids
        vision_targets = self._vision_media_targets.get(command_id, ())
        vision_model = self._vision_media_models.get(command_id)
        if (
            not should_send_finished
            and vision_targets
            and vision_model is not None
            and self._vision_media_packet_sender is not None
        ):
            with anyio.move_on_after(2, shield=True):
                await self._cancel_vision_media_transport(
                    command_id,
                    vision_model,
                    vision_targets,
                )
        self._cancelled_command_ids.discard(command_id)
        self._realtime_audio_transcription_commands.discard(command_id)
        self._speech_media_commands.discard(command_id)
        self._speech_media_targets.pop(command_id, None)
        self._transcription_media_targets.pop(command_id, None)
        self._take_pending_speech_media(command_id)
        self._take_pending_vision_media(command_id)
        self._release_active_vision_media(command_id)
        self._vision_media_commands.discard(command_id)
        self._vision_media_targets.pop(command_id, None)
        self._vision_media_pending_acks.pop(command_id, None)
        self._vision_media_ack_deadlines.pop(command_id, None)
        self._vision_media_models.pop(command_id, None)
        self._vision_media_failures.pop(command_id, None)
        if should_send_finished:
            await self._send(TaskFinished(finished_command_id=command_id))
        queue_map.pop(command_id, None)
        # Drop the data-plane reorder buffer / dedupe cursor alongside the queue
        # so a late chunk arriving after finalize is dropped (no queue) instead
        # of leaking state.
        self._chunk_reorder.pop(command_id, None)
        self._data_dedup_cursor.pop(command_id, None)
        self._data_plane_observer.finalize(command_id)

    async def _token_chunk_stream(
        self, command_id: CommandId
    ) -> AsyncGenerator[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk, None
    ]:
        """Yield chunks for a given command until completion.

        This is the internal low-level stream used by all API adapters.
        """
        try:
            self._text_generation_queues[command_id], recv = channel[
                TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk
            ]()

            if pending_error := self._take_vision_media_failure(command_id):
                yield pending_error
                return

            if pending_failure := self._pending_stream_failures.pop(command_id, None):
                # The task already failed before this stream registered
                # (fast local TaskFailed, e.g. a pinned instance that
                # vanished); deliver the buffered terminal chunk.
                yield pending_failure
                return

            with recv as token_chunks:
                # Idle backstop (#279 Phase 2b): the DATA plane is best-effort, so
                # a dropped chunk could hang this receive forever. The idle timer
                # applies ONLY once streaming has started (after the first chunk)
                # and measures the gap BETWEEN chunks — wrapping just the receive,
                # not the yield. It deliberately does NOT bound time-to-first
                # chunk: a request can sit in the runner queue behind a long
                # decode, or in a slow prefill, for an unbounded time, and the
                # control plane (TaskFailed -> _terminate_command_stream) already
                # covers instance death. So this only fires on a mid-stream stall.
                started = False
                while True:
                    chunk: (
                        TokenChunk
                        | ErrorChunk
                        | ToolCallChunk
                        | PrefillProgressChunk
                        | None
                    ) = None
                    delay = _STREAM_IDLE_TIMEOUT_SECONDS if started else None
                    with anyio.move_on_after(delay) as scope:
                        try:
                            chunk = await token_chunks.receive()
                        except (EndOfStream, ClosedResourceError):
                            return  # producer closed normally
                    if scope.cancelled_caught:
                        self._data_plane_observer.record_idle_timeout()
                        # Mid-stream stall: the receive went idle for longer than
                        # the inter-token bound. Two distinct causes, disambiguated
                        # by the master's task status (#298 review):
                        #   - task already terminal -> the runner finished and only
                        #     the FINAL data-plane chunk was dropped. Sending
                        #     TaskCancelled here is a no-op on a completed runner and
                        #     leaks the master's task/command mapping forever, so let
                        #     the normal TaskFinished path (via _finalize) clean up.
                        #   - task not terminal -> a genuinely stuck runner; send
                        #     TaskCancelled so the master tears it down (otherwise we
                        #     orphan an in-flight runner task).
                        task_done = self._command_task_is_terminal(command_id)
                        logger.warning(
                            f"Token stream for command {command_id} idle for "
                            f">{_STREAM_IDLE_TIMEOUT_SECONDS:g}s mid-stream; "
                            + (
                                "task already terminal — finishing (a final "
                                "data-plane chunk was likely dropped)."
                                if task_done
                                else "task still active — cancelling (a data-plane "
                                "chunk may have been dropped)."
                            )
                        )
                        if not task_done:
                            self._cancelled_command_ids.add(command_id)
                            with anyio.CancelScope(shield=True):
                                await self.command_sender.send(
                                    ForwarderCommand(
                                        origin=self._system_id,
                                        command=TaskCancelled(
                                            cancelled_command_id=command_id
                                        ),
                                    )
                                )
                        yield ErrorChunk(
                            model=ModelId("unknown"),
                            error_message=(
                                "Stream stalled: no output for "
                                f"{_STREAM_IDLE_TIMEOUT_SECONDS:g}s; the response "
                                "may be incomplete."
                            ),
                        )
                        return
                    assert chunk is not None  # not timed out and not EndOfStream
                    yield chunk
                    if isinstance(chunk, PrefillProgressChunk):
                        # Prefill progress is NOT real output: keep the idle timer
                        # disarmed (delay stays None). Prefill of a huge prompt on
                        # a slow node can run far longer than the inter-token idle
                        # bound between progress updates, so arming here would
                        # falsely time out a still-prefilling request (#298 review).
                        continue
                    # First real output token: from here, gaps are inter-token and
                    # the idle timeout applies.
                    started = True
                    if chunk.finish_reason is not None:
                        break

        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            self._cancelled_command_ids.add(command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._finalize_command_stream(
                command_id,
                cast(
                    dict[CommandId, Sender[object]],
                    self._text_generation_queues,
                ),
            )

    def _tapped_text_stream(
        self,
        command_id: CommandId,
        model_id: ModelId,
        *,
        task_params: TextGenerationTaskParams | None = None,
        extension_tap: bool = True,
    ) -> AsyncGenerator[
        TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk, None
    ]:
        """Wrap the raw token stream with the standard observation taps.

        Every text-generation surface (chat completions, the Claude/Responses
        adapters, the Ollama endpoints, realtime assistant turns) routes through
        here so throughput/latency learning and field telemetry are not confined
        to ``/v1/chat/completions``. Applies, in order: field telemetry (a plain
        passthrough unless the operator opted in), the observe-only performance
        envelope (adaptive concurrency, Phase 0), and -- when ``task_params`` is
        given and chat middleware is loaded -- the extension chat-summary tap.
        All three are fully guarded and never alter token delivery.

        ``model_id`` must be the POST-transform model actually dispatched (chat
        middleware may reroute it), so observations attribute to the served
        model, not the caller's requested alias.

        ``extension_tap=False`` keeps the envelope and telemetry taps but
        withholds the extension chat-summary tap. It is for generations that
        are a step inside some larger turn rather than a turn of their own:
        the steward's investigation steps and its liveness probe run through
        this method, and an observer that saw each of them would record the
        steward's internal tool traffic and count one answer many times.
        Whoever owns the enclosing turn applies the single correct tap.
        """
        chunk_stream: AsyncGenerator[
            TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk, None
        ] = self._token_chunk_stream(command_id)
        chunk_stream = tap_generation_stream(
            self._field_telemetry,
            str(model_id),
            None,
            chunk_stream,
        )
        chunk_stream = self._tap_performance_envelope(model_id, chunk_stream)
        if (
            extension_tap
            and task_params is not None
            and self._extensions is not None
            and self._extensions.has_chat_middleware
        ):
            chunk_stream = self._extensions.tap_chat_stream(
                self._extension_context, task_params, chunk_stream
            )
        return chunk_stream

    async def _collect_text_generation_with_stats(
        self, command_id: CommandId, model_id: ModelId
    ) -> BenchChatCompletionResponse:
        sampler = PowerSampler(
            get_node_system=lambda: {
                node_id: profile
                for node_id, profile in self._telemetry_view.node_system.items()
                if node_id in self.state.last_seen
            }
        )
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        model: ModelId | None = None
        finish_reason: FinishReason | None = None

        stats: GenerationStats | None = None

        async with anyio.create_task_group() as tg:
            tg.start_soon(sampler.run)

            # Route the bench collector through the same taps as every other text
            # surface (#596): /bench/chat/completions is the concurrency load
            # generator this envelope is meant to learn from (the harness
            # `concurrent` suite is its offline mirror), so its completed
            # generations must feed the envelope and field telemetry, not bypass
            # them. No task_params here, so the extension chat tap is skipped.
            async for chunk in self._tapped_text_stream(command_id, model_id):
                if isinstance(chunk, PrefillProgressChunk):
                    continue

                if chunk.finish_reason == "error":
                    raise HTTPException(
                        status_code=500,
                        detail=chunk.error_message or "Internal server error",
                    )

                if model is None:
                    model = chunk.model

                if isinstance(chunk, TokenChunk):
                    text_parts.append(chunk.text)

                if isinstance(chunk, ToolCallChunk):
                    tool_calls.extend(
                        ToolCall(
                            id=str(uuid4()),
                            index=i,
                            function=tool,
                        )
                        for i, tool in enumerate(chunk.tool_calls)
                    )

                stats = chunk.stats or stats

                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason

            tg.cancel_scope.cancel()

        combined_text = "".join(text_parts)
        assert model is not None

        return BenchChatCompletionResponse(
            id=command_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=combined_text,
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            # The explicit benchmark surface exposes only non-identifying
            # batching truth; node and backend attribution remain private.
            generation_stats=(stats.redacted_for_benchmark_client() if stats else None),
            power_usage=sampler.result(),
        )

    async def _trigger_notify_user_to_download_model(self, model_id: ModelId) -> None:
        logger.warning(
            "TODO: we should send a notification to the user to download the model"
        )

    def _context_token_limit_for_model(self, model_id: ModelId) -> int | None:
        """Most permissive context-token ceiling across instances of a model.

        The API cannot tokenize (tokenizers live with the model files on the
        hosting nodes), so this limit supports only the exact pre-flight check
        on ``max_tokens`` alone; the runner enforces the precise
        prompt-inclusive limit per instance (#145). When several instances
        serve the same model the request may be routed to any of them, so the
        maximum avoids falsely rejecting a request the largest instance could
        still serve.
        """
        limits = [
            instance.context_token_limit
            for instance in self.state.instances.values()
            if instance.shard_assignments.model_id == model_id
            and instance.context_token_limit is not None
        ]
        return max(limits) if limits else None

    def _live_node_timestamps(self) -> dict[NodeId, datetime]:
        """Return replicated membership plus fresh telemetry-only participants.

        Replicated ``State.last_seen`` tracks worker control-plane activity and
        can omit local or remote ``--no-worker`` API nodes. Fresh telemetry is
        positive liveness evidence for those management participants. Bound it
        by the shared node timeout so a stopped telemetry-only node cannot
        remain in API health and resource views indefinitely.

        Returns:
            Live-node timestamps suitable for telemetry filtering and health
            derivation on this API process.
        """
        live = dict(self.state.last_seen)
        now = datetime.now(tz=timezone.utc)
        telemetry_nodes = set(self._telemetry_view.node_last_heartbeat) | set(
            self._telemetry_view.node_last_telemetry
        )
        for node_id in telemetry_nodes - live.keys():
            seen_at = self._telemetry_view.last_liveness_receipt(node_id)
            if seen_at is None:
                continue
            aware_seen_at = (
                seen_at
                if seen_at.tzinfo is not None
                else seen_at.replace(tzinfo=timezone.utc)
            )
            if now - aware_seen_at <= NODE_LIVENESS_TIMEOUT:
                live[node_id] = seen_at
        return live

    async def get_cluster_state(self) -> dict[str, object]:
        """Cluster state for ``GET /state``: event-sourced ``State`` plus live
        telemetry (per-node memory, system profile, and resources) merged back in.

        ``node_memory``, ``node_system``, and ``node_resources`` live on the
        telemetry plane, but ``/state`` is the dashboard's single data source.
        Merge the ``TelemetryView`` maps in under camelCase keys so the dashboard
        and external consumers can observe those facts. The merge is filtered to
        ``last_seen``-live nodes as defense-in-depth on top of the worker's
        ``NodeTimedOut`` prune, so a dead node never shows as a ghost.

        Declared ``async`` so FastAPI serves it ON the event loop rather than a
        worker thread: the telemetry subscriber and the ``NodeTimedOut`` prune
        mutate these dicts on the loop, and a threadpool read could iterate one
        mid-mutation and 500 with ``dictionary changed size during iteration``.
        There is no ``await`` here, so the comprehensions run atomically wrt
        those same-loop mutators.
        """
        live = self._live_node_timestamps()
        effective_downloads = self._telemetry_view.effective_downloads(
            self.state.downloads
        )
        payload = self.state.model_copy(
            update={"downloads": effective_downloads}
        ).model_dump(mode="json", by_alias=True)
        payload["nodeMemory"] = {
            str(node_id): usage.model_dump(mode="json", by_alias=True)
            for node_id, usage in self._telemetry_view.node_memory.items()
            if node_id in live
        }
        payload["nodeSystem"] = {
            str(node_id): profile.model_dump(mode="json", by_alias=True)
            for node_id, profile in self._telemetry_view.node_system.items()
            if node_id in live
        }
        payload["nodeResources"] = {
            str(node_id): resources.model_dump(mode="json", by_alias=True)
            for node_id, resources in self._telemetry_view.node_resources.items()
            if node_id in live
        }
        # Observational readings moved to telemetry in #279 slice 3; merge them
        # back under their original camelCase keys so the dashboard wire shape is
        # unchanged.
        payload["nodeIdentities"] = {
            str(node_id): identity.model_dump(mode="json", by_alias=True)
            for node_id, identity in self._telemetry_view.node_identities.items()
            if node_id in live
        }
        payload["nodeDisk"] = {
            str(node_id): disk.model_dump(mode="json", by_alias=True)
            for node_id, disk in self._telemetry_view.node_disk.items()
            if node_id in live
        }
        payload["nodeRdmaCtl"] = {
            str(node_id): status.model_dump(mode="json", by_alias=True)
            for node_id, status in self._telemetry_view.node_rdma_ctl.items()
            if node_id in live
        }
        # Extension-advertised capability tags (fabric-citizenship): the light
        # discovery layer, made operator-visible. Full descriptors are fetched
        # via GET /v1/capabilities, not carried here.
        payload["nodeCapabilities"] = {
            str(node_id): sorted(tags)
            for node_id, tags in self._telemetry_view.node_capabilities.items()
            if node_id in live and tags
        }
        # Derived per-node health (#388): explain a node's problems (and the fix)
        # in the topology so the master's silent recovery of a wedged/failed node
        # is legible. Read-only derivation from the same state/telemetry above; no
        # new gossip type. last_seen is tz-aware UTC, so use the same clock here.
        payload["nodeHealth"] = {
            node_id: summary.model_dump(mode="json", by_alias=True)
            for node_id, summary in compute_node_health(
                live_nodes=live,
                downloads=effective_downloads,
                node_disk=self._telemetry_view.node_disk,
                node_resources=self._telemetry_view.node_resources,
                node_identities=self._telemetry_view.node_identities,
                heartbeat_last_seen=self._telemetry_view.node_last_heartbeat,
                telemetry_last_seen=self._telemetry_view.node_last_telemetry,
                now=datetime.now(tz=timezone.utc),
            ).items()
        }
        return payload

    def _stage_audio_transcription_media(
        self,
        command_id: CommandId,
        task_params: AudioTranscriptionTaskParams,
        audio_bytes: bytes,
    ) -> None:
        """Retain bounded batch audio until the master selects its worker."""

        if self._speech_media_packet_sender is None:
            raise HTTPException(
                status_code=503,
                detail="Speech media transport is unavailable on this node",
            )
        pending = _PendingSpeechMedia(
            model=task_params.model,
            data=audio_bytes,
            filename=task_params.filename,
            content_type=task_params.content_type,
            sha256=hashlib.sha256(audio_bytes).hexdigest(),
            created_at=time.monotonic(),
        )
        if (
            len(self._pending_speech_media) >= _SPEECH_MEDIA_PENDING_COMMANDS
            or self._pending_speech_media_bytes + len(audio_bytes)
            > _SPEECH_MEDIA_PENDING_TOTAL_BYTES
        ):
            raise HTTPException(
                status_code=503,
                detail="Speech media admission capacity is exhausted",
            )
        self._pending_speech_media[command_id] = pending
        self._pending_speech_media_bytes += len(audio_bytes)
        self._speech_media_commands.add(command_id)

    def _take_pending_speech_media(
        self, command_id: CommandId
    ) -> _PendingSpeechMedia | None:
        """Release one pending batch upload from API admission accounting."""

        pending = self._pending_speech_media.pop(command_id, None)
        if pending is not None:
            self._pending_speech_media_bytes = max(
                0, self._pending_speech_media_bytes - len(pending.data)
            )
        return pending

    def _dispatch_pending_speech_media(self, task: task_types.Task) -> None:
        """Send batch STT audio after authoritative task placement replicates."""

        if not isinstance(task, task_types.AudioTranscription):
            return
        pending = self._take_pending_speech_media(task.command_id)
        if pending is None:
            return
        instance = self.state.instances.get(task.instance_id)
        if instance is None:
            self._tg.start_soon(
                self._fail_audio_transcription_media,
                task.command_id,
                "The selected model instance disappeared before audio upload",
            )
            return
        targets = tuple(sorted(instance.shard_assignments.node_to_runner, key=str))
        if not targets:
            self._tg.start_soon(
                self._fail_audio_transcription_media,
                task.command_id,
                "The selected model instance has no worker participants",
            )
            return
        self._transcription_media_targets[task.command_id] = targets
        self._tg.start_soon(
            self._send_audio_transcription_media,
            task.command_id,
            pending,
            targets,
        )

    async def _send_audio_transcription_media(
        self,
        command_id: CommandId,
        pending: _PendingSpeechMedia,
        targets: tuple[NodeId, ...],
    ) -> None:
        """Fan one bounded raw-audio stream to the task's selected workers."""

        sender = self._speech_media_packet_sender
        if sender is None:
            await self._fail_audio_transcription_media(
                command_id, "Speech media transport is unavailable"
            )
            return
        total_chunks = max(
            1,
            (len(pending.data) + _SPEECH_MEDIA_CHUNK_BYTES - 1)
            // _SPEECH_MEDIA_CHUNK_BYTES,
        )
        try:
            for target in targets:
                for sequence, offset in enumerate(
                    range(0, len(pending.data), _SPEECH_MEDIA_CHUNK_BYTES)
                ):
                    if command_id not in self._speech_media_commands:
                        return
                    await sender.send(
                        SpeechMediaPacket(
                            source_node=self.node_id,
                            target_node=target,
                            command_id=command_id,
                            sequence=sequence,
                            kind="chunk",
                            purpose="transcription_audio",
                            filename=pending.filename,
                            content_type=pending.content_type,
                            data=pending.data[
                                offset : offset + _SPEECH_MEDIA_CHUNK_BYTES
                            ],
                        )
                    )
                await sender.send(
                    SpeechMediaPacket(
                        source_node=self.node_id,
                        target_node=target,
                        command_id=command_id,
                        sequence=total_chunks,
                        kind="completed",
                        purpose="transcription_audio",
                        filename=pending.filename,
                        content_type=pending.content_type,
                        sha256=pending.sha256,
                    )
                )
        except anyio.get_cancelled_exc_class():
            raise
        except Exception as exception:  # noqa: BLE001 - transport boundary
            logger.opt(exception=exception).warning(
                f"Speech media upload failed for command {command_id}"
            )
            await self._fail_audio_transcription_media(
                command_id,
                "Speech media transport failed while uploading transcription input",
            )

    async def _fail_audio_transcription_media(
        self, command_id: CommandId, message: str
    ) -> None:
        """Fail one batch STT upload without disturbing unrelated streams."""

        if command_id not in self._speech_media_commands:
            return
        self._speech_media_commands.discard(command_id)
        self._take_pending_speech_media(command_id)
        await self._dispatch_generation_chunk(
            command_id,
            ErrorChunk(model=ModelId("unknown"), error_message=message),
        )
        await self._cancel_audio_transcription_command(command_id)
        self._cancelled_command_ids.discard(command_id)

    async def _sweep_pending_speech_media(self) -> None:
        """Fail batch uploads that never acquire authoritative placement."""

        while True:
            await anyio.sleep(5)
            cutoff = time.monotonic() - _SPEECH_MEDIA_PENDING_TTL_SECONDS
            expired = [
                command_id
                for command_id, pending in self._pending_speech_media.items()
                if pending.created_at <= cutoff
            ]
            for command_id in expired:
                await self._fail_audio_transcription_media(
                    command_id,
                    "Speech media timed out waiting for authoritative placement",
                )

    def _stage_vision_media(
        self,
        command_id: CommandId,
        model: ModelId,
        chunks: Sequence[tuple[int, str]],
        image_count: int,
    ) -> None:
        """Retain bounded image chunks until the master selects an instance."""

        if self._vision_media_packet_sender is None:
            raise HTTPException(
                status_code=503,
                detail="Vision media transport is unavailable on this node",
            )
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Image input must not be empty",
            )
        if len(chunks) > _VISION_MEDIA_PENDING_FRAMES:
            raise HTTPException(
                status_code=413,
                detail="Image input exceeds the per-request media frame limit",
            )
        encoded_chunks = tuple(
            (image_index, data.encode("ascii")) for image_index, data in chunks
        )
        pending = _PendingVisionMedia(
            model=model,
            chunks=encoded_chunks,
            image_count=image_count,
            sha256=hashlib.sha256(
                b"".join(data for _, data in encoded_chunks)
            ).hexdigest(),
            created_at=time.monotonic(),
        )
        if pending.byte_count > _VISION_MEDIA_PENDING_COMMAND_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image input exceeds the per-request media limit",
            )
        if (
            len(self._pending_vision_media) + len(self._active_vision_media_bytes)
            >= _VISION_MEDIA_PENDING_COMMANDS
            or self._pending_vision_media_bytes
            + pending.byte_count
            + self._active_vision_media_total_bytes
            > _VISION_MEDIA_PENDING_TOTAL_BYTES
        ):
            raise HTTPException(
                status_code=503,
                detail="Vision media admission capacity is exhausted",
            )
        self._pending_vision_media[command_id] = pending
        self._pending_vision_media_bytes += pending.byte_count
        self._vision_media_commands.add(command_id)
        self._vision_media_models[command_id] = model

    def _take_pending_vision_media(
        self, command_id: CommandId
    ) -> _PendingVisionMedia | None:
        """Release one pending upload from API admission accounting."""

        pending = self._pending_vision_media.pop(command_id, None)
        if pending is not None:
            self._pending_vision_media_bytes = max(
                0, self._pending_vision_media_bytes - pending.byte_count
            )
        return pending

    def _retain_active_vision_media(
        self, command_id: CommandId, byte_count: int
    ) -> None:
        """Keep transferred bytes charged until all selected ranks verify them."""

        previous = self._active_vision_media_bytes.get(command_id, 0)
        self._active_vision_media_bytes[command_id] = byte_count
        self._active_vision_media_total_bytes += byte_count - previous

    def _release_active_vision_media(self, command_id: CommandId) -> None:
        """Release source-side transfer accounting after terminal delivery."""

        released = self._active_vision_media_bytes.pop(command_id, 0)
        self._active_vision_media_total_bytes = max(
            0, self._active_vision_media_total_bytes - released
        )

    def _dispatch_pending_vision_media(self, task: task_types.Task) -> None:
        """Start direct upload after authoritative task placement is replicated."""

        if not isinstance(task, (task_types.TextGeneration, task_types.ImageEdits)):
            return
        pending = self._take_pending_vision_media(task.command_id)
        if pending is None:
            return
        self._retain_active_vision_media(task.command_id, pending.byte_count)
        instance = self.state.instances.get(task.instance_id)
        if instance is None:
            self._tg.start_soon(
                self._fail_vision_media_command,
                task.command_id,
                pending.model,
                "The selected model instance disappeared before image upload",
            )
            return
        # llama.cpp RPC donors only contribute memory; the stamped driver is
        # the sole process that executes inference and owns the projector.
        # MLX distributed vision still requires identical media on every rank.
        targets = (
            (instance.driver_node,)
            if isinstance(instance, LlamaRpcInstance)
            else tuple(sorted(instance.shard_assignments.node_to_runner, key=str))
        )
        if not targets:
            self._tg.start_soon(
                self._fail_vision_media_command,
                task.command_id,
                pending.model,
                "The selected model instance has no worker participants",
            )
            return
        self._vision_media_targets[task.command_id] = targets
        self._vision_media_pending_acks[task.command_id] = set(targets)
        self._vision_media_ack_deadlines[task.command_id] = (
            time.monotonic() + _VISION_MEDIA_ACK_TIMEOUT_SECONDS
        )
        self._tg.start_soon(
            self._send_vision_media_to_targets,
            task.command_id,
            pending,
            targets,
        )

    async def _send_vision_media_to_targets(
        self,
        command_id: CommandId,
        pending: _PendingVisionMedia,
        targets: tuple[NodeId, ...],
    ) -> None:
        """Fan one ordered image stream directly to every selected rank."""

        try:
            async with anyio.create_task_group() as task_group:
                for target in targets:
                    task_group.start_soon(
                        self._send_vision_media_to_target,
                        command_id,
                        pending,
                        target,
                    )
        except anyio.get_cancelled_exc_class():
            raise
        except Exception as exception:  # noqa: BLE001 - transport boundary
            logger.opt(exception=exception).warning(
                f"Vision media upload failed for command {command_id}"
            )
            await self._fail_vision_media_command(
                command_id,
                pending.model,
                "Vision media transport failed while uploading image input",
            )

    async def _send_vision_media_to_target(
        self,
        command_id: CommandId,
        pending: _PendingVisionMedia,
        target: NodeId,
    ) -> None:
        """Send one rank's explicit open, ordered chunks, and completion."""

        sender = self._vision_media_packet_sender
        if sender is None:
            raise RuntimeError("vision media transport is unavailable")
        total_chunks = len(pending.chunks)
        if command_id not in self._vision_media_commands:
            return
        await sender.send(
            VisionMediaPacket(
                source_node=self.node_id,
                target_node=target,
                command_id=command_id,
                model=pending.model,
                sequence=0,
                kind="opened",
                total_chunks=total_chunks,
                image_count=pending.image_count,
            )
        )
        for sequence, (image_index, data) in enumerate(pending.chunks, start=1):
            if command_id not in self._vision_media_commands:
                return
            await sender.send(
                VisionMediaPacket(
                    source_node=self.node_id,
                    target_node=target,
                    command_id=command_id,
                    model=pending.model,
                    sequence=sequence,
                    kind="chunk",
                    data=data,
                    image_index=image_index,
                    total_chunks=total_chunks,
                )
            )
        if command_id not in self._vision_media_commands:
            return
        await sender.send(
            VisionMediaPacket(
                source_node=self.node_id,
                target_node=target,
                command_id=command_id,
                model=pending.model,
                sequence=total_chunks + 1,
                kind="completed",
                total_chunks=total_chunks,
                image_count=pending.image_count,
                sha256=pending.sha256,
            )
        )

    async def _cancel_vision_media_transport(
        self,
        command_id: CommandId,
        model: ModelId,
        targets: tuple[NodeId, ...],
    ) -> None:
        """Send best-effort media cancellation to every selected worker."""

        sender = self._vision_media_packet_sender
        if sender is None:
            return
        for target in targets:
            with contextlib.suppress(
                BrokenResourceError, ClosedResourceError, WouldBlock
            ):
                await sender.send(
                    VisionMediaPacket(
                        source_node=self.node_id,
                        target_node=target,
                        command_id=command_id,
                        model=model,
                        sequence=0,
                        kind="cancelled",
                    )
                )

    async def _fail_vision_media_command(
        self,
        command_id: CommandId,
        model: ModelId,
        message: str,
    ) -> None:
        """Fail one upload without disturbing unrelated inference streams."""

        if (
            command_id not in self._vision_media_commands
            or command_id in self._vision_media_failures
        ):
            return
        error = ErrorChunk(model=model, error_message=message)
        self._vision_media_failures[command_id] = error
        self._vision_media_commands.discard(command_id)
        self._release_active_vision_media(command_id)
        self._vision_media_pending_acks.pop(command_id, None)
        self._vision_media_ack_deadlines.pop(command_id, None)
        await self._dispatch_generation_chunk(command_id, error)
        self._close_command_queue(command_id)
        self._cancelled_command_ids.add(command_id)
        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=TaskCancelled(cancelled_command_id=command_id),
            )
        )
        # Transport rejection is terminal to the caller. Let stream finalization
        # emit TaskFinished after cancellation so the master releases its mapping.
        self._cancelled_command_ids.discard(command_id)

    def _take_vision_media_failure(self, command_id: CommandId) -> ErrorChunk | None:
        """Consume a transport error that arrived before its response queue."""

        return self._vision_media_failures.pop(command_id, None)

    async def _sweep_pending_vision_media(self) -> None:
        """Fail uploads missing placement or worker verification acknowledgements."""

        while True:
            await anyio.sleep(5)
            await self._expire_stale_vision_media(time.monotonic())

    async def _expire_stale_vision_media(self, now: float) -> None:
        """Fail placement and acknowledgement deadlines at one monotonic instant."""

        cutoff = now - _VISION_MEDIA_PENDING_TTL_SECONDS
        expired = [
            (command_id, pending.model)
            for command_id, pending in self._pending_vision_media.items()
            if pending.created_at <= cutoff
        ]
        for command_id, model in expired:
            self._take_pending_vision_media(command_id)
            await self._fail_vision_media_command(
                command_id,
                model,
                "Vision media timed out waiting for authoritative placement",
            )
        unacknowledged = [
            command_id
            for command_id, deadline in self._vision_media_ack_deadlines.items()
            if deadline <= now
        ]
        for command_id in unacknowledged:
            model = self._vision_media_models.get(command_id)
            if model is not None:
                await self._fail_vision_media_command(
                    command_id,
                    model,
                    "Vision media timed out waiting for worker verification",
                )

    async def send_task_cancellation(self, command_id: CommandId) -> None:
        """Public cancellation seam for internal callers (the steward harness).

        Sends the same TaskCancelled command the HTTP cancel endpoint sends,
        suppressing the final TaskFinished so the worker can observe the
        cancelled task and stop the runner promptly.
        """
        await self._send(TaskCancelled(cancelled_command_id=command_id))
        self._cancelled_command_ids.add(command_id)

    async def dispatch_text_generation(
        self,
        task_params: TextGenerationTaskParams,
        target_instance_id: InstanceId | None = None,
    ) -> TextGeneration:
        """Public dispatch seam for internal callers (the steward harness).

        Sends a text-generation command through the same validated path as
        the HTTP chat adapters, optionally pinned to one instance.
        """
        return await self._send_text_generation_with_images(
            task_params, target_instance_id=target_instance_id
        )

    def text_generation_chunk_stream(
        self,
        command: TextGeneration,
        task_params: TextGenerationTaskParams,
        *,
        extension_tap: bool = True,
    ) -> AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ]:
        """Public tapped chunk stream for one dispatched command.

        Same observation taps as the HTTP adapters (envelopes, telemetry,
        extensions), so internal consumers are indistinguishable from
        external ones to the observability planes.

        Args:
            command: The dispatched generation to stream.
            task_params: The params that were dispatched.
            extension_tap: Pass ``False`` when this generation is one step
                inside a larger turn that applies its own extension tap; see
                :meth:`_tapped_text_stream`.
        """
        return self._tapped_text_stream(
            command.command_id,
            task_params.model,
            task_params=task_params,
            extension_tap=extension_tap,
        )

    async def running_model_card(self, model_id: ModelId) -> ModelCard | None:
        """Public card lookup preferring the card carried on a live instance."""
        return await self._get_running_model_card(model_id)

    async def _send_text_generation_with_images(
        self,
        task_params: TextGenerationTaskParams,
        target_instance_id: InstanceId | None = None,
    ) -> TextGeneration:
        # Single dispatch chokepoint for every text-generation wire format
        # (chat, claude, ollama, responses, bench) — reject un-renderable
        # requests here so malformed input is a clean 400 instead of an
        # IndexError that crashes the runner (#233; empty messages reached
        # apply_chat_template([]) -> list index out of range).
        validate_renderable_text_generation(task_params)
        # Context admission pre-flight (#145): the API has no tokenizer, so
        # only the prompt-independent certainty is checked here — an explicit
        # max_tokens that cannot fit even before counting a single prompt
        # token. The runner performs the exact prompt-inclusive check.
        context_token_limit = self._context_token_limit_for_model(task_params.model)
        if (
            context_token_limit is not None
            and task_params.max_output_tokens is not None
            and task_params.max_output_tokens >= context_token_limit
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"context_length_exceeded: max_tokens "
                    f"({task_params.max_output_tokens}) cannot fit within this "
                    f"model's usable context limit of {context_token_limit} "
                    f"tokens (the smaller of the model's advertised context "
                    f"length and what fits in memory next to the model weights "
                    f"on the hosting node(s))"
                ),
            )
        images = task_params.images
        if not images:
            command = TextGeneration(
                task_params=task_params,
                owner_node=self.node_id,
                target_instance_id=target_instance_id,
            )
            await self._send(command)
            return command

        hashes = [hashlib.sha256(img.encode("ascii")).hexdigest() for img in images]

        cached_hashes: dict[int, str] = {}
        new_images: list[tuple[int, str]] = []
        cache_enabled = _text_image_hash_cache_enabled()
        if cache_enabled:
            for idx, (img, h) in enumerate(zip(images, hashes, strict=True)):
                if h in self._sent_image_hashes:
                    cached_hashes[idx] = h
                else:
                    self._sent_image_hashes.add(h)
                    new_images.append((idx, img))
        else:
            # The worker cache is local, in-memory, and not placement-aware. Sending
            # hash-only references by default can crash or wedge nodes after restarts
            # or placement changes, so keep the optimization behind an explicit flag.
            new_images = list(enumerate(images))

        _log_image_transport(
            f"TextGeneration image transport: total={len(images)} "
            f"new={len(new_images)} cached={len(cached_hashes)} "
            f"hash_cache_enabled={cache_enabled}"
        )
        for idx, img in new_images:
            _log_image_transport(
                f"TextGeneration new image {idx}: b64_chars={len(img)} "
                f"b64_sha256={hashlib.sha256(img.encode('ascii')).hexdigest()[:12]}..."
            )
        for idx, h in cached_hashes.items():
            _log_image_transport(
                f"TextGeneration cached image {idx}: b64_sha256={h[:12]}..."
            )

        if not new_images:
            task_params = task_params.model_copy(
                update={"images": [], "image_hashes": cached_hashes}
            )
            command = TextGeneration(
                task_params=task_params,
                owner_node=self.node_id,
                target_instance_id=target_instance_id,
            )
            await self._send(command)
            return command

        all_chunks: list[tuple[int, str]] = []
        for img_idx, img_data in new_images:
            for i in range(0, len(img_data), SKULK_MAX_CHUNK_SIZE):
                all_chunks.append((img_idx, img_data[i : i + SKULK_MAX_CHUNK_SIZE]))

        task_params = task_params.model_copy(
            update={
                "images": [],
                "image_hashes": cached_hashes,
                "total_input_chunks": len(all_chunks),
                "image_count": len(new_images),
            }
        )
        command = TextGeneration(
            task_params=task_params,
            owner_node=self.node_id,
            target_instance_id=target_instance_id,
        )
        self._stage_vision_media(
            command.command_id,
            task_params.model,
            all_chunks,
            len(new_images),
        )
        try:
            await self._send(command)
        except BaseException:
            self._take_pending_vision_media(command.command_id)
            self._vision_media_commands.discard(command.command_id)
            self._vision_media_models.pop(command.command_id, None)
            raise
        return command

    async def chat_completions_route(
        self, payload: ChatCompletionRequest, request: Request
    ) -> StreamingResponse:
        """Serve chat completions with request-scoped steward proposal authority.

        Args:
            payload: OpenAI-compatible chat completion request.
            request: HTTP request used to determine operator mutation authority.

        Returns:
            A streaming or collected chat-completions response.
        """
        return await self.chat_completions(
            payload,
            steward_proposals_allowed=self._operator_mutation_allowed(request),
        )

    async def chat_completions(
        self,
        payload: ChatCompletionRequest,
        *,
        steward_proposals_allowed: bool = True,
    ) -> StreamingResponse:
        """OpenAI Chat Completions API - adapter."""
        if str(payload.model) == STEWARD_VIRTUAL_MODEL_ID:
            # The reserved steward id selects model-plus-harness: the
            # server-side investigation loop answers, with its tool trace
            # streamed as reasoning content. Checked before card resolution
            # so no repository of the same name can shadow it.
            return await self._steward_chat_completions(
                payload,
                proposals_allowed=steward_proposals_allowed,
            )
        resolved_model = await self._resolve_and_validate_text_model(payload.model)
        model_card = await self._get_running_model_card(resolved_model)
        task_params = await chat_request_to_text_generation(
            payload.model_copy(update={"model": resolved_model}),
            model_card=model_card,
        )

        # Extension hooks: request transform before dispatch, response tap
        # after. Both are inert when no extension provides chat middleware,
        # and every extension call inside is guarded (a raising extension is
        # logged and the request proceeds unchanged).
        if self._extensions is not None and self._extensions.has_chat_middleware:
            task_params = await self._extensions.transform_chat_request(
                self._extension_context, task_params
            )

        command = await self._send_text_generation_with_images(task_params)

        # All observation taps (field telemetry, performance envelope, extension
        # chat summary) applied through the shared helper. The POST-transform
        # model is passed so observations attribute to the dispatched model.
        chunk_stream = self._tapped_text_stream(
            command.command_id,
            task_params.model,
            task_params=task_params,
        )

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_chat_stream(
                        command.command_id,
                        chunk_stream,
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Streamed for the same reason as the ordinary chat route: it is
            # what makes a client disconnect cancel the generation (#872).
            return StreamingResponse(
                collect_chat_response(command.command_id, chunk_stream),
                media_type="application/json",
            )

    async def bench_chat_completions(
        self, payload: BenchChatCompletionRequest
    ) -> BenchChatCompletionResponse:
        resolved_model = await self._resolve_and_validate_text_model(payload.model)
        model_card = await self._get_running_model_card(resolved_model)
        task_params = await chat_request_to_text_generation(
            payload.model_copy(update={"model": resolved_model}),
            model_card=model_card,
        )
        task_params = task_params.model_copy(update={"stream": False, "bench": True})

        command = await self._send_text_generation_with_images(task_params)

        return await self._collect_text_generation_with_stats(
            command.command_id, task_params.model
        )

    async def _resolve_and_validate_text_model(self, model_id: ModelId) -> ModelId:
        """Validate a mounted model supports text generation.

        Raises HTTPException 404 if no instance is found for the model, or 400
        when the mounted model does not declare ``TextGeneration``.
        """
        if not any(
            instance.shard_assignments.model_id == model_id
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(model_id)
            raise HTTPException(
                status_code=404,
                detail=f"No instance found for model {model_id}",
            )
        model_card = await self._get_running_model_card(model_id)
        if ModelTask.TextGeneration not in model_card.tasks:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_card.model_id} is not a text-generation model",
            )
        return model_id

    async def _get_running_model_card(self, model_id: ModelId) -> ModelCard:
        """Return a model card for a running instance without requiring remote lookup.

        Text requests should prefer the in-memory shard metadata once a model is
        already running so request availability does not depend on model-card
        cache misses or Hugging Face fetches during normal inference.
        """
        for instance in self.state.instances.values():
            if instance.shard_assignments.model_id == model_id:
                card = self._model_card_for_instance(instance)
                if card is not None:
                    return card
        return await self._load_authorized_model_card(model_id)

    @staticmethod
    def _model_card_for_instance(instance: Instance) -> ModelCard | None:
        """Return the replicated model card carried by one mounted instance."""

        runner_to_shard = getattr(instance.shard_assignments, "runner_to_shard", None)
        if isinstance(runner_to_shard, dict):
            for shard in cast(dict[object, object], runner_to_shard).values():
                shard_model_card = cast(object, getattr(shard, "model_card", None))
                if isinstance(shard_model_card, ModelCard):
                    return shard_model_card
        fallback_card = cast(
            object,
            getattr(instance.shard_assignments, "model_card", None),
        )
        if isinstance(fallback_card, ModelCard):
            # Older tests and simplified in-memory stubs may attach the card
            # directly instead of through runner_to_shard.
            return fallback_card
        return None

    async def _validate_image_model(self, model: ModelId) -> ModelId:
        """Validate model exists and return resolved model ID.

        Raises HTTPException 404 if no instance is found for the model.
        """
        model_card = await self._load_authorized_model_card(model)
        resolved_model = model_card.model_id
        if not any(
            instance.shard_assignments.model_id == resolved_model
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(resolved_model)
            raise HTTPException(
                status_code=404, detail=f"No instance found for model {resolved_model}"
            )
        return resolved_model

    async def _validate_embedding_model(self, model_id: ModelId) -> ModelId:
        """Validate an embedding model exists and is the right type.

        Raises HTTPException 404 if no instance, 400 if not an embedding model.
        """
        from skulk.shared.models.model_cards import ModelTask

        model_card = await self._load_authorized_model_card(model_id)
        resolved = model_card.model_id
        if ModelTask.TextEmbedding not in model_card.tasks:
            raise HTTPException(
                status_code=400,
                detail=f"Model {resolved} is not an embedding model",
            )
        if not any(
            instance.shard_assignments.model_id == resolved
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(resolved)
            raise HTTPException(
                status_code=404,
                detail=f"No instance found for model {resolved}",
            )
        return resolved

    async def _validate_speech_synthesis_model(
        self,
        model_id: ModelId,
        response_format: AudioResponseFormat | None,
        *,
        stream: bool = False,
    ) -> tuple[ModelId, AudioResponseFormat]:
        """Validate a mounted TTS model and resolve the output audio format."""

        model_card = await self._get_running_model_card(model_id)
        resolved = model_card.model_id
        profile = resolve_model_capability_profile(resolved, model_card=model_card)
        if not profile.supports_speech_synthesis:
            raise HTTPException(
                status_code=400,
                detail=f"Model {resolved} is not a text-to-speech model",
            )
        if stream and (
            model_card.audio is None or model_card.audio.supports_streaming is not True
        ):
            raise HTTPException(
                status_code=400,
                detail=(f"Model {resolved} does not declare streaming speech support"),
            )
        if not any(
            instance.shard_assignments.model_id == resolved
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(resolved)
            raise HTTPException(
                status_code=404,
                detail=f"No instance found for model {resolved}",
            )
        resolved_response_format = (
            response_format
            or profile.default_audio_response_format
            or AudioResponseFormat.Mp3
        )
        if (
            profile.audio_response_formats
            and resolved_response_format not in profile.audio_response_formats
        ):
            supported = ", ".join(
                audio_format.value for audio_format in profile.audio_response_formats
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model {resolved} does not support audio response format "
                    f"{resolved_response_format.value}; supported formats: {supported}"
                ),
            )
        if (
            stream
            and resolved_response_format not in _STREAMABLE_AUDIO_RESPONSE_FORMATS
        ):
            supported = ", ".join(
                audio_format.value
                for audio_format in sorted(
                    _STREAMABLE_AUDIO_RESPONSE_FORMATS,
                    key=lambda audio_format: audio_format.value,
                )
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"`stream=true` supports only {supported} responses for now; "
                    f"requested {resolved_response_format.value}"
                ),
            )
        if stream and not self._streaming_tts_model_is_ready(
            resolved,
            resolved_response_format,
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"All mounted instances of streaming speech model {resolved} "
                    "must have a ready runner"
                ),
            )
        return resolved, resolved_response_format

    def _reference_tts_target(
        self,
        model_id: ModelId,
    ) -> tuple[InstanceId, NodeId]:
        """Select ready single-host TTS capacity supporting reference audio."""

        candidates: list[tuple[str, str, InstanceId, NodeId]] = []
        for instance_id, instance in self.state.instances.items():
            if instance.shard_assignments.model_id != model_id:
                continue
            if len(instance.shard_assignments.node_to_runner) != 1:
                continue
            card = self._model_card_for_instance(instance)
            if (
                card is None
                or card.audio is None
                or card.audio.supports_reference_audio is not True
            ):
                continue
            target_node, runner_id = next(
                iter(instance.shard_assignments.node_to_runner.items())
            )
            if not isinstance(
                self.state.runners.get(runner_id),
                (RunnerReady, RunnerRunning),
            ):
                continue
            candidates.append(
                (str(instance_id), str(target_node), instance_id, target_node)
            )
        if not candidates:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model {model_id} has no ready mounted instance that "
                    "supports reference audio"
                ),
            )
        _, _, instance_id, target_node = min(candidates)
        return instance_id, target_node

    def _streaming_tts_model_is_ready(
        self,
        model_id: ModelId | None = None,
        response_format: AudioResponseFormat = AudioResponseFormat.Mp3,
    ) -> bool:
        """Return whether every routable instance of one streaming model is ready."""

        eligible_by_model: dict[ModelId, list[Instance]] = {}
        for instance in self.state.instances.values():
            if model_id is not None and instance.shard_assignments.model_id != model_id:
                continue
            card = self._model_card_for_instance(instance)
            if card is None or card.audio is None:
                continue
            profile = resolve_model_capability_profile(
                card.model_id,
                model_card=card,
            )
            if (
                profile.supports_speech_synthesis
                and card.audio.supports_streaming is True
                and (
                    not profile.audio_response_formats
                    or response_format in profile.audio_response_formats
                )
            ):
                eligible_by_model.setdefault(card.model_id, []).append(instance)

        def instance_is_ready(instance: Instance) -> bool:
            runner_ids = tuple(instance.shard_assignments.node_to_runner.values())
            return len(runner_ids) == 1 and all(
                isinstance(
                    self.state.runners.get(runner_id),
                    (RunnerReady, RunnerRunning),
                )
                for runner_id in runner_ids
            )

        return any(
            instances and all(instance_is_ready(instance) for instance in instances)
            for instances in eligible_by_model.values()
        )

    def _has_mounted_streaming_tts_model(self) -> bool:
        """Return whether core serving currently exposes eligible TTS capacity."""

        return (
            self._builtin_speech_provider_enabled
            and self._streaming_tts_model_is_ready()
        )

    def _realtime_stt_instance(
        self,
        model_id: ModelId | None = None,
        *,
        call_id: str | None = None,
        require_idle: bool = False,
    ) -> tuple[InstanceId, ModelCard, NodeId] | None:
        """Return eligible cluster realtime STT capacity and its hosting node."""

        if not self._builtin_speech_provider_enabled or (
            self._realtime_audio_sender is None
            and self._realtime_audio_packet_sender is None
        ):
            return None
        candidates: list[tuple[bool, str, str, InstanceId, ModelCard, NodeId]] = []
        for instance_id, instance in self.state.instances.items():
            if len(instance.shard_assignments.node_to_runner) != 1:
                continue
            if require_idle and (
                any(
                    active.reserved_instance_id == instance_id
                    for active_call_id, active in self._active_capability_streams.items()
                    if active_call_id != call_id
                )
                or any(
                    task.instance_id == instance_id
                    # RunnerReady is authoritative for lifecycle completion.
                    # Download/load bookkeeping can remain non-terminal briefly
                    # on another API node after the runner becomes ready, but it
                    # does not occupy STT inference capacity. Only active STT
                    # workloads must block cross-owner admission here.
                    and isinstance(
                        task,
                        (
                            task_types.AudioTranscription,
                            task_types.RealtimeAudioTranscription,
                        ),
                    )
                    and task.task_status not in _TERMINAL_TASK_STATUSES
                    for task in self.state.tasks.values()
                )
            ):
                continue
            if model_id is not None and instance.shard_assignments.model_id != model_id:
                continue
            card = self._model_card_for_instance(instance)
            if card is None or card.audio is None:
                continue
            profile = resolve_model_capability_profile(
                card.model_id,
                model_card=card,
            )
            if (
                profile.supports_transcription
                and card.audio.supports_streaming is True
                and card.audio.supports_realtime is True
            ):
                for (
                    target_node,
                    runner_id,
                ) in instance.shard_assignments.node_to_runner.items():
                    if not isinstance(
                        self.state.runners.get(runner_id),
                        (RunnerReady, RunnerRunning),
                    ):
                        continue
                    is_remote = target_node != self.node_id
                    if is_remote and (
                        not self._data_plane_zenoh
                        or self._realtime_audio_packet_sender is None
                    ):
                        continue
                    if (
                        not is_remote
                        and self._realtime_audio_sender is None
                        and self._realtime_audio_packet_sender is None
                    ):
                        continue
                    candidates.append(
                        (
                            is_remote,
                            str(instance_id),
                            str(target_node),
                            instance_id,
                            card,
                            target_node,
                        )
                    )
        if not candidates:
            return None
        _, _, _, instance_id, card, target_node = min(candidates)
        return instance_id, card, target_node

    def _has_realtime_stt_model(self) -> bool:
        """Return whether this API can reach true realtime STT capacity."""

        return self._realtime_stt_instance() is not None

    def _mounted_stt_model_is_ready(self, model_id: ModelId | None = None) -> bool:
        """Return whether every routable requested STT instance is ready."""

        def instance_is_ready(instance: Instance) -> bool:
            if len(instance.shard_assignments.node_to_runner) != 1:
                return False
            card = self._model_card_for_instance(instance)
            if card is None:
                return False
            profile = resolve_model_capability_profile(card.model_id, model_card=card)
            if not profile.supports_transcription:
                return False
            placement_runners = tuple(
                instance.shard_assignments.node_to_runner.values()
            )
            return bool(placement_runners) and all(
                isinstance(
                    self.state.runners.get(runner_id), (RunnerReady, RunnerRunning)
                )
                for runner_id in placement_runners
            )

        if model_id is None:
            return any(
                instance_is_ready(instance)
                for instance in self.state.instances.values()
            )
        matching_instances = tuple(
            instance
            for instance in self.state.instances.values()
            if instance.shard_assignments.model_id == model_id
        )
        # AudioTranscription is not instance-pinned: the master may choose any
        # mounted instance of the model. Admission is truthful only if every
        # candidate it could choose is ready at this snapshot.
        return bool(matching_instances) and all(
            instance_is_ready(instance) for instance in matching_instances
        )

    def _has_mounted_stt_model(self, model_id: ModelId | None = None) -> bool:
        """Return whether the built-in provider can advertise ready STT."""

        return (
            self._builtin_speech_provider_enabled
            and self._mounted_stt_model_is_ready(model_id)
        )

    def _sync_builtin_speech_capability(self) -> None:
        """Advertise speech facades only while core capacity can serve them."""

        if not self._builtin_speech_provider_enabled:
            return
        if self._has_mounted_streaming_tts_model():
            self._advertise_capability(TTS_CAPABILITY_DESCRIPTOR.id)
        else:
            self._withdraw_capability(TTS_CAPABILITY_DESCRIPTOR.id)
        if self._has_mounted_stt_model():
            self._advertise_capability(STT_CAPABILITY_DESCRIPTOR.id)
        else:
            self._withdraw_capability(STT_CAPABILITY_DESCRIPTOR.id)
        if self._has_realtime_stt_model():
            self._advertise_capability(REALTIME_STT_CAPABILITY_DESCRIPTOR.id)
        else:
            self._withdraw_capability(REALTIME_STT_CAPABILITY_DESCRIPTOR.id)

    def refresh_config_dependent_capabilities(self) -> None:
        """Refresh capability advertisements after local config application."""

        self._sync_builtin_speech_capability()

    def set_model_store_runtime(
        self,
        skulk_config: "SkulkConfig | None",
        store_client: "ModelStoreClient | None",
    ) -> None:
        """Replace config and store client after cluster-config convergence.

        Args:
            skulk_config: The authoritative cluster configuration, or ``None``
                when the cluster has no configuration.
            store_client: Client for the authoritative model store, or ``None``
                when the model store is disabled.

        Side effects:
            Subsequent dashboard store requests use the replacement client and
            config-dependent provider advertisements are refreshed.
        """

        self._skulk_config = skulk_config
        if store_client is not self._store_client:
            # Last-known installation truth belongs to one authoritative store.
            # Carrying it across config convergence could manufacture installed
            # state when the replacement store is unavailable.
            self._model_list_store_records_cache = {}
            self._model_list_store_records_cached_at = 0.0
        self._store_client = store_client
        self.refresh_config_dependent_capabilities()

    async def _validate_audio_transcription_model(
        self,
        model_id: ModelId,
        *,
        stream: bool = False,
    ) -> ModelId:
        """Validate a mounted speech-to-text model exists and is servable."""

        model_card = await self._get_running_model_card(model_id)
        resolved = model_card.model_id
        profile = resolve_model_capability_profile(resolved, model_card=model_card)
        if not profile.supports_transcription:
            raise HTTPException(
                status_code=400,
                detail=f"Model {resolved} is not a speech-to-text model",
            )
        if stream and (
            model_card.audio is None or model_card.audio.supports_streaming is not True
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model {resolved} does not declare streaming transcription support"
                ),
            )
        if not any(
            instance.shard_assignments.model_id == resolved
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(resolved)
            raise HTTPException(
                status_code=404,
                detail=f"No instance found for model {resolved}",
            )
        if stream and not self._mounted_stt_model_is_ready(resolved):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"All mounted instances of streaming transcription model "
                    f"{resolved} must have a ready runner"
                ),
            )
        return resolved

    async def _validate_audio_translation_model(
        self,
        model_id: ModelId,
        source_language: str | None,
    ) -> ModelId:
        """Validate a mounted translation-capable model for `/v1/audio/translations`.

        Speech translation is a standard capability: the only gates are model
        truth (a mounted card that declares translation support) and instance
        availability, matching every other speech endpoint.
        """

        if not any(
            instance.shard_assignments.model_id == model_id
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(model_id)
            raise HTTPException(
                status_code=404,
                detail=f"No instance found for model {model_id}",
            )
        model_card = await self._get_running_model_card(model_id)
        resolved = model_card.model_id
        profile = resolve_model_capability_profile(resolved, model_card=model_card)
        if not profile.supports_speech_translation:
            raise HTTPException(
                status_code=400,
                detail=f"Model {resolved} does not support speech translation",
            )
        if model_card.family == "canary" and not source_language:
            raise HTTPException(
                status_code=400,
                detail="Canary translation requires a source `language` code",
            )
        return resolved

    def _advertise_capability(self, capability: str) -> None:
        """Advertise a capability tag on this node's telemetry plane.

        The telemetry-plane advertise surface extensions receive via their
        ``ExtensionContext`` (fabric-citizenship Phase 1). Records the tag on the
        shared ``TelemetryView`` outbound set; the worker's info gatherer gossips
        it last-write-wins on its next poll, so peers see it in their own
        ``read_cluster`` snapshots. Advertising is additive and idempotent.

        Empty or whitespace-only tags are ignored (a defensive guard: a tag is a
        discovery key, and a blank one would be meaningless gossip). A node that
        runs no worker never emits (it has no gatherer), so the tag is recorded
        but not gossiped there; the mainstream node runs both.

        Args:
            capability: The opaque capability tag to advertise (for example
                ``"memory"``).
        """
        tag = capability.strip()
        if not tag:
            logger.warning("Extension advertised an empty capability tag; ignoring it")
            return
        self._telemetry_view.local_advertised_capabilities.add(tag)

    def _withdraw_capability(self, capability: str) -> None:
        """Withdraw a previously advertised capability tag.

        The liveness counterpart of :meth:`_advertise_capability`
        (fabric-citizenship Phase 2a): a provider that stops offering a
        capability withdraws the tag so callers stop selecting this node for
        it. The worker's info gatherer publishes the shrunken set on its next
        poll (including a final empty reading when the last tag goes, so peers
        clear the entry instead of keeping the stale last-write-wins value).
        Withdrawing a tag that was never advertised is a no-op.

        Args:
            capability: The tag to withdraw (matched after whitespace trim).
        """
        self._telemetry_view.local_advertised_capabilities.discard(capability.strip())

    async def list_node_capabilities(
        self, node_id: str | None = None
    ) -> dict[str, object]:
        """Serve ``GET /v1/capabilities``: a node's capability descriptors.

        The describe surface of capability discovery (fabric-citizenship
        Phase 2a). Without ``node_id`` (or with this node's id), returns the
        descriptors served by this node's provider extensions; with a peer's
        ``node_id``, proxies the peer's describe surface (the same read-only
        fan-out pattern the trace cluster browsing uses), returning an empty
        list when the peer is unreachable. Each descriptor's content revision
        digest is included so callers can pin the exact shape they discovered.
        Extensions consume this through ``describe_node``.

        Args:
            node_id: Optional peer node to describe instead of this node.

        Returns:
            ``{"node_id": ..., "capabilities": [...], "revisions": {...}}``
            where ``revisions`` maps ``id@version`` to the revision digest.
        """
        # A blank or whitespace-padded node_id means "this node", not a
        # literal peer id that would always proxy to nothing.
        requested = node_id.strip() if node_id is not None else None
        if not requested or requested == str(self.node_id):
            descriptors = (
                self._extensions.capability_descriptors
                if self._extensions is not None
                else ()
            )
            described: NodeId = self.node_id
        else:
            described = NodeId(requested)
            descriptors = await self._describe_node_capabilities(described)
        return {
            "node_id": str(described),
            "capabilities": [
                descriptor.model_dump(mode="json") for descriptor in descriptors
            ],
            "revisions": {
                descriptor.qualified_id: descriptor_revision(descriptor)
                for descriptor in descriptors
            },
        }

    async def serve_capability_call(self, call: CapabilityCall) -> CapabilityResult:
        """Serve ``POST /v1/capabilities/call``: dispatch a call to a provider.

        The provider-side entry of the generic capability call (fabric-
        citizenship Phase 2b). The body is the typed call envelope; the
        response is always a :class:`CapabilityResult` (HTTP 200 even for
        typed failures, so callers switch on ``result.error.code`` rather than
        transport status). All isolation guards live in
        :meth:`_dispatch_capability_call`.
        """
        return await self._dispatch_capability_call(call)

    async def _dispatch_capability_call(self, call: CapabilityCall) -> CapabilityResult:
        """Run one capability call against this node's providers, guarded.

        Guard order (each failure is a typed error, never an exception):
        target-node addressing check; handler lookup (``not_found`` vs
        ``version_mismatch``); descriptor revision pin; then, INSIDE the
        provider concurrency bound (``overloaded``) and the caller's deadline
        (``timeout``): payload size cap, input-schema validation, and the
        handler itself (exceptions become ``provider_error``); finally result
        shape, size cap, and output-schema validation (``invalid_result``).
        Serialization and validation work counts against the bound and the
        deadline (#513); cheap guards stay outside so trivially-rejectable
        calls never consume a slot. The master is never involved and nothing
        here touches ``State``.
        """
        if call.target_node != str(self.node_id):
            # A misrouted or misaddressed envelope must not execute here: the
            # call is node-addressed, and accepting it would make logs and
            # auditing claim a different target than the one that served it.
            return call_failure(
                call.call_id,
                "invalid_payload",
                f"call addressed to {call.target_node}, served by {self.node_id}",
            )
        if self._extensions is None:
            return call_failure(
                call.call_id, "not_found", "this node loads no extensions"
            )
        qualified_id = f"{call.capability_id}@{call.version}"
        entry = self._extensions.call_handler(qualified_id)
        if entry is None:
            code: CapabilityErrorCode = (
                "version_mismatch"
                if call.capability_id in self._extensions.handled_capability_ids()
                else "not_found"
            )
            return call_failure(
                call.call_id, code, f"no callable capability {qualified_id!r}"
            )
        extension_name, handler, descriptor = entry
        current_revision = descriptor_revision(descriptor)
        if call.descriptor_revision != current_revision:
            return call_failure(
                call.call_id,
                "revision_mismatch",
                f"descriptor revision is {current_revision}, caller pinned "
                f"{call.descriptor_revision}; re-discover before calling",
            )
        # Bounded concurrency: one slow provider must not accumulate unbounded
        # in-flight calls on the API node, and (#513) the serialization and
        # schema-validation work below counts against the bound and the
        # deadline too, so a storm of large-but-invalid payloads cannot drive
        # unbounded concurrent validation outside the guard. The event loop is
        # single-threaded, so a plain counter is race-free here. Cheap guards
        # (addressing, handler lookup, revision pin) stay outside the bound so
        # trivially-rejectable calls never consume a slot.
        if self._active_capability_calls >= _MAX_CONCURRENT_CAPABILITY_CALLS:
            return call_failure(
                call.call_id,
                "overloaded",
                f"provider concurrency bound "
                f"({_MAX_CONCURRENT_CAPABILITY_CALLS}) reached",
            )
        self._active_capability_calls += 1
        try:
            with anyio.fail_after(call.timeout_seconds):
                try:
                    payload_bytes = len(
                        json.dumps(
                            call.payload, separators=(",", ":"), allow_nan=False
                        ).encode("utf-8")
                    )
                except (TypeError, ValueError, RecursionError, OverflowError) as exc:
                    # A local fast-path caller can hand a payload the
                    # endpoint's JSON parsing would never produce (bytes,
                    # sets, NaN); typed error, not an exception.
                    return call_failure(
                        call.call_id,
                        "invalid_payload",
                        f"payload is not JSON-serializable: {exc}",
                    )
                if payload_bytes > MAX_CALL_PAYLOAD_BYTES:
                    return call_failure(
                        call.call_id,
                        "payload_too_large",
                        f"payload is {payload_bytes} bytes "
                        f"(limit {MAX_CALL_PAYLOAD_BYTES})",
                    )
                schema_error = validate_against_schema(
                    call.payload, descriptor.input_schema, what="payload"
                )
                if schema_error is not None:
                    return call_failure(call.call_id, "invalid_payload", schema_error)
                try:
                    result_payload = await handler.handle_call(
                        self._extension_context, call
                    )
                except Exception as exc:  # noqa: BLE001 - a raising handler must not 500 the node
                    # logger.exception keeps the traceback: debugging a
                    # misbehaving extension from the message alone is much
                    # harder in production.
                    logger.exception(
                        f"capability handler '{extension_name}' for "
                        f"{qualified_id} raised: {exc}"
                    )
                    return call_failure(
                        call.call_id,
                        "provider_error",
                        f"{type(exc).__name__}: {exc}",
                    )
        except TimeoutError:
            return call_failure(
                call.call_id,
                "timeout",
                f"call did not finish within {call.timeout_seconds}s "
                f"(payload validation plus provider execution)",
            )
        finally:
            self._active_capability_calls -= 1
        if not isinstance(result_payload, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            # The handler protocol says dict, but a misbehaving plugin can
            # return anything; the strict result model would raise on it.
            return call_failure(
                call.call_id,
                "invalid_result",
                f"provider returned {type(result_payload).__name__}, not an object",
            )
        if any(not isinstance(key, str) for key in result_payload):  # pyright: ignore[reportUnnecessaryIsInstance]
            # json.dumps silently stringifies non-string keys (so the size
            # check passes), but the strict result model rejects them; catch
            # it here as a typed error instead of a 500.
            return call_failure(
                call.call_id,
                "invalid_result",
                "provider result has non-string keys",
            )
        try:
            result_bytes = len(
                json.dumps(
                    result_payload, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            )
        except (TypeError, ValueError, RecursionError, OverflowError) as exc:
            # allow_nan=False also rejects NaN/Infinity here: json.dumps would
            # otherwise accept them but the HTTP response renderer refuses
            # non-finite JSON, turning the call into a 500 downstream.
            return call_failure(
                call.call_id,
                "invalid_result",
                f"provider result is not JSON-serializable: {exc}",
            )
        if result_bytes > MAX_CALL_PAYLOAD_BYTES:
            return call_failure(
                call.call_id,
                "payload_too_large",
                f"result is {result_bytes} bytes (limit {MAX_CALL_PAYLOAD_BYTES})",
            )
        if descriptor.output_schema is not None:
            schema_error = validate_against_schema(
                result_payload, descriptor.output_schema, what="result"
            )
            if schema_error is not None:
                return call_failure(call.call_id, "invalid_result", schema_error)
        return CapabilityResult(call_id=call.call_id, ok=True, result=result_payload)

    async def serve_capability_stream(self, call: CapabilityCall) -> CapabilityResult:
        """Admit one streaming provider call on this node.

        The response covers only opening validation and admission. Once
        admitted, active lifecycle directions travel on ``PROVIDER_DATA``
        between caller and provider; no media bytes use this HTTP request.
        """

        return await self._dispatch_capability_stream(call)

    async def cancel_capability_stream(
        self, request: CapabilityStreamCancel
    ) -> CapabilityResult:
        """Cancel one admitted provider stream idempotently."""

        if request.target_node != str(self.node_id):
            return call_failure(
                request.call_id,
                "invalid_payload",
                f"cancellation addressed to {request.target_node}, served by "
                f"{self.node_id}",
            )
        active = self._active_capability_streams.get(request.call_id)
        if active is None:
            return CapabilityResult(
                call_id=request.call_id,
                ok=True,
                result={"cancelled": False},
            )
        if active.caller_node != request.caller_node:
            return call_failure(
                request.call_id,
                "invalid_payload",
                "only the caller that opened a provider stream may cancel it",
            )
        active.cancel_message = request.message
        self._provider_observer.record_cancellation_request(request.call_id)
        active.cancel_requested.set()
        if active.cancel_scope is not None:
            active.cancel_scope.cancel()
        return CapabilityResult(
            call_id=request.call_id,
            ok=True,
            result={"cancelled": True},
        )

    async def _dispatch_capability_stream(
        self, call: CapabilityCall
    ) -> CapabilityResult:
        """Validate, admit, and start one guarded provider stream."""

        started_at = anyio.current_time()
        if call.target_node != str(self.node_id):
            return call_failure(
                call.call_id,
                "invalid_payload",
                f"call addressed to {call.target_node}, served by {self.node_id}",
            )
        if self._extensions is None:
            return call_failure(
                call.call_id, "not_found", "this node loads no extensions"
            )
        qualified_id = f"{call.capability_id}@{call.version}"
        entry = self._extensions.stream_handler(qualified_id)
        if entry is None:
            code: CapabilityErrorCode = (
                "version_mismatch"
                if call.capability_id
                in self._extensions.handled_stream_capability_ids()
                else "not_found"
            )
            return call_failure(
                call.call_id,
                code,
                f"no streaming capability {qualified_id!r}",
            )
        extension_name, handler, descriptor = entry
        current_revision = descriptor_revision(descriptor)
        if call.descriptor_revision != current_revision:
            self._provider_observer.record_rejected(qualified_id)
            return call_failure(
                call.call_id,
                "revision_mismatch",
                f"descriptor revision is {current_revision}, caller pinned "
                f"{call.descriptor_revision}; re-discover before streaming",
            )
        if self._provider_stream_sender is None or not self._tg.is_running():
            self._provider_observer.record_rejected(qualified_id)
            return call_failure(
                call.call_id,
                "unreachable",
                "provider DATA transport is not running on this node",
            )
        if call.call_id in self._active_capability_streams:
            self._provider_observer.record_rejected(qualified_id)
            return call_failure(
                call.call_id,
                "invalid_payload",
                "call_id already names an active provider stream",
            )
        if len(self._active_capability_streams) >= _MAX_CONCURRENT_CAPABILITY_STREAMS:
            self._provider_observer.record_rejected(qualified_id, overloaded=True)
            return call_failure(
                call.call_id,
                "overloaded",
                f"provider stream concurrency bound "
                f"({_MAX_CONCURRENT_CAPABILITY_STREAMS}) reached",
            )

        try:
            with anyio.fail_after(call.timeout_seconds):
                try:
                    payload_bytes = len(
                        json.dumps(
                            call.payload,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    )
                except (
                    TypeError,
                    ValueError,
                    RecursionError,
                    OverflowError,
                ) as exc:
                    self._provider_observer.record_rejected(qualified_id)
                    return call_failure(
                        call.call_id,
                        "invalid_payload",
                        f"payload is not JSON-serializable: {exc}",
                    )
                if payload_bytes > MAX_CALL_PAYLOAD_BYTES:
                    self._provider_observer.record_rejected(qualified_id)
                    return call_failure(
                        call.call_id,
                        "payload_too_large",
                        f"payload is {payload_bytes} bytes "
                        f"(limit {MAX_CALL_PAYLOAD_BYTES})",
                    )
                schema_error = validate_against_schema(
                    call.payload, descriptor.input_schema, what="payload"
                )
                if schema_error is not None:
                    self._provider_observer.record_rejected(qualified_id)
                    return call_failure(call.call_id, "invalid_payload", schema_error)

                input_sender: Sender[CapabilityStreamFrame] | None = None
                input_receiver: Receiver[CapabilityStreamFrame] | None = None
                input_lifecycle: CapabilityStreamReceiver | None = None
                if descriptor.io_mode in ("client_streaming", "bidirectional"):
                    if self._provider_stream_receiver is None:
                        self._provider_observer.record_rejected(qualified_id)
                        return call_failure(
                            call.call_id,
                            "unreachable",
                            "provider DATA input transport is not running on this node",
                        )
                    input_sender, input_receiver = channel[CapabilityStreamFrame](
                        _PROVIDER_STREAM_RECEIVE_BUFFER
                    )
                    input_lifecycle = CapabilityStreamReceiver(
                        call_id=call.call_id,
                        direction="caller_to_provider",
                        gap_timeout_seconds=5.0,
                        idle_timeout_seconds=call.timeout_seconds,
                    )

                # Reserve before admission can yield so concurrent opens cannot
                # race past the stream bound. Rejections release this provisional
                # entry; successful dispatch hands ownership to the runner.
                active = _ActiveProviderStream(
                    caller_node=call.caller_node,
                    cancel_requested=anyio.Event(),
                    descriptor=descriptor,
                    input_sender=input_sender,
                    input_receiver=input_receiver,
                    input_lifecycle=input_lifecycle,
                )
                self._active_capability_streams[call.call_id] = active
                if isinstance(handler, CapabilityStreamAdmissionHandler):
                    try:
                        admission_error = await handler.admit_stream(
                            self._extension_context,
                            call,
                        )
                    except Exception as exc:  # noqa: BLE001 - provider boundary
                        logger.opt(exception=exc).warning(
                            f"capability stream admission for {qualified_id} raised"
                        )
                        self._active_capability_streams.pop(call.call_id, None)
                        self._close_active_provider_input(active)
                        self._provider_observer.record_rejected(qualified_id)
                        return call_failure(
                            call.call_id,
                            "provider_error",
                            f"stream admission raised {type(exc).__name__}: {exc}",
                        )
                    if admission_error is not None:
                        self._active_capability_streams.pop(call.call_id, None)
                        self._close_active_provider_input(active)
                        self._provider_observer.record_rejected(
                            qualified_id,
                            overloaded=admission_error.code == "overloaded",
                        )
                        if not isinstance(admission_error, CapabilityError):  # pyright: ignore[reportUnnecessaryIsInstance]
                            return call_failure(
                                call.call_id,
                                "provider_error",
                                "stream admission returned an invalid result",
                            )
                        return CapabilityResult(
                            call_id=call.call_id,
                            ok=False,
                            error=admission_error,
                        )
        except anyio.get_cancelled_exc_class():
            active = self._active_capability_streams.pop(call.call_id, None)
            if active is not None:
                self._close_active_provider_input(active)
            raise
        except TimeoutError:
            active = self._active_capability_streams.pop(call.call_id, None)
            if active is not None:
                self._close_active_provider_input(active)
            self._provider_observer.record_rejected(qualified_id)
            return call_failure(
                call.call_id,
                "timeout",
                "provider stream deadline expired during admission",
            )

        # Dynamic admission may yield while the provider is disabled or its
        # child dies. Recheck before scheduling executable stream work.
        if not self._extensions.capability_ready(qualified_id):
            active = self._active_capability_streams.pop(call.call_id, None)
            if active is not None:
                self._close_active_provider_input(active)
            self._provider_observer.record_rejected(qualified_id)
            return call_failure(call.call_id, "not_found", "capability is not ready")

        remaining = call.timeout_seconds - (anyio.current_time() - started_at)
        if remaining <= 0:
            active = self._active_capability_streams.pop(call.call_id, None)
            if active is not None:
                self._close_active_provider_input(active)
            self._provider_observer.record_rejected(qualified_id)
            return call_failure(
                call.call_id,
                "timeout",
                "provider stream deadline expired during admission",
            )
        admitted_call = call.model_copy(update={"timeout_seconds": remaining})
        try:
            self._tg.start_soon(
                self._run_capability_stream,
                admitted_call,
                extension_name,
                handler,
                descriptor,
                active,
            )
        except RuntimeError as exc:
            self._active_capability_streams.pop(call.call_id, None)
            self._close_active_provider_input(active)
            self._provider_observer.record_rejected(qualified_id)
            return call_failure(
                call.call_id,
                "unreachable",
                f"provider stream runtime could not start: {exc}",
            )
        self._provider_observer.record_admitted(call.call_id, qualified_id)
        return CapabilityResult(
            call_id=call.call_id,
            ok=True,
            result={"admitted": True, "io_mode": descriptor.io_mode},
        )

    async def _run_capability_stream(
        self,
        call: CapabilityCall,
        extension_name: str,
        handler: CapabilityStreamHandler | CapabilityInputStreamHandler,
        descriptor: CapabilityDescriptor,
        active: _ActiveProviderStream,
    ) -> None:
        """Execute one admitted handler and enforce its output contract."""

        next_sequence = 0
        terminal_sent = False

        async def emit(frame: CapabilityStreamFrame) -> None:
            if self._provider_stream_sender is None:
                raise ClosedResourceError
            await self._provider_stream_sender.send(
                ProviderStreamPacket(
                    owner_node=NodeId(call.caller_node),
                    frame=frame,
                )
            )
            self._provider_observer.record_output(call.call_id, frame)

        async def fail(code: CapabilityStreamErrorCode, message: str) -> None:
            nonlocal next_sequence, terminal_sent
            if terminal_sent:
                return
            await emit(
                CapabilityStreamFrame(
                    call_id=call.call_id,
                    direction="provider_to_caller",
                    sequence=next_sequence,
                    kind="failed",
                    synthetic=True,
                    error=CapabilityStreamError(
                        code=code,
                        message=message,
                    ),
                )
            )
            next_sequence += 1
            terminal_sent = True

        async def close_provider_output(
            stream: AsyncIterator[CapabilityStreamFrame],
        ) -> None:
            """Run iterator cleanup before exposing a synthetic failure."""

            close = cast(
                Callable[[], Awaitable[None]] | None,
                getattr(stream, "aclose", None),
            )
            if close is None:
                return
            try:
                await close()
            except anyio.get_cancelled_exc_class():
                raise
            except Exception as exc:  # noqa: BLE001 - provider boundary
                logger.opt(exception=exc).warning(
                    f"capability stream handler '{extension_name}' for "
                    f"{descriptor.qualified_id} raised during iterator cleanup"
                )

        try:
            with anyio.CancelScope() as cancel_scope:
                try:
                    with anyio.fail_after(call.timeout_seconds):
                        await emit(
                            CapabilityStreamFrame(
                                call_id=call.call_id,
                                direction="provider_to_caller",
                                sequence=0,
                                kind="started",
                            )
                        )
                        next_sequence = 1
                        # Cancellation cannot preempt the lifecycle opener. A
                        # request racing admission is recorded on the event and
                        # applied only after sequence zero is on the DATA plane.
                        active.cancel_scope = cancel_scope
                        if active.cancel_requested.is_set():
                            cancel_scope.cancel()
                        stream: AsyncIterator[CapabilityStreamFrame] | None = None
                        if not cancel_scope.cancel_called:
                            if descriptor.io_mode in (
                                "client_streaming",
                                "bidirectional",
                            ):
                                assert isinstance(handler, CapabilityInputStreamHandler)
                                assert active.input_receiver is not None
                                stream = handler.handle_input_stream(
                                    self._extension_context,
                                    call,
                                    self._consume_provider_input(active),
                                )
                            else:
                                assert isinstance(handler, CapabilityStreamHandler)
                                stream = handler.handle_stream(
                                    self._extension_context, call
                                )
                        if stream is None:
                            stream = self._empty_capability_stream()
                        async for frame in stream:
                            if active.input_failure is not None:
                                await close_provider_output(stream)
                                await fail(
                                    active.input_failure.code,
                                    active.input_failure.message,
                                )
                                break
                            if not isinstance(frame, CapabilityStreamFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
                                await close_provider_output(stream)
                                await fail(
                                    "invalid_frame",
                                    f"provider yielded {type(frame).__name__}, "
                                    "not CapabilityStreamFrame",
                                )
                                break
                            if (
                                frame.call_id != call.call_id
                                or frame.direction != "provider_to_caller"
                                or frame.sequence != next_sequence
                                or frame.kind == "started"
                            ):
                                await close_provider_output(stream)
                                await fail(
                                    "invalid_frame",
                                    "provider yielded a frame with invalid "
                                    "identity, direction, sequence, or lifecycle",
                                )
                                break
                            if frame.kind == "chunk" or (
                                frame.kind == "completed"
                                and (
                                    descriptor.output_schema is not None
                                    or frame.payload is not None
                                    or frame.media is not None
                                )
                            ):
                                payload = frame.payload or {}
                                try:
                                    payload_bytes = len(
                                        json.dumps(
                                            payload,
                                            separators=(",", ":"),
                                            allow_nan=False,
                                        ).encode("utf-8")
                                    )
                                except (
                                    TypeError,
                                    ValueError,
                                    RecursionError,
                                    OverflowError,
                                ) as exc:
                                    await close_provider_output(stream)
                                    await fail(
                                        "invalid_frame",
                                        f"provider chunk payload is not JSON: {exc}",
                                    )
                                    break
                                if payload_bytes > MAX_CALL_PAYLOAD_BYTES:
                                    await close_provider_output(stream)
                                    await fail(
                                        "invalid_frame",
                                        "provider chunk payload exceeds 1 MiB",
                                    )
                                    break
                                output_schema = (
                                    descriptor.output_chunk_schema
                                    if frame.kind == "chunk"
                                    else descriptor.output_schema
                                )
                                if output_schema is None:
                                    await close_provider_output(stream)
                                    await fail(
                                        "invalid_frame",
                                        "provider output carries data without a "
                                        "matching descriptor schema",
                                    )
                                    break
                                schema_error = validate_against_schema(
                                    payload,
                                    output_schema,
                                    what=(
                                        "output chunk"
                                        if frame.kind == "chunk"
                                        else "completed output"
                                    ),
                                )
                                if schema_error is not None:
                                    await close_provider_output(stream)
                                    await fail("invalid_frame", schema_error)
                                    break
                            if frame.is_terminal:
                                # A provider terminal is not observable until the
                                # handler has actually returned. Built-in speech
                                # handlers release core commands in ``finally``;
                                # stopping at the yielded terminal would leave
                                # that cleanup to nondeterministic async-generator
                                # finalization and could retain runner admission.
                                try:
                                    trailing_frame = await anext(stream)
                                except StopAsyncIteration:
                                    pass
                                else:
                                    await close_provider_output(stream)
                                    await fail(
                                        "invalid_frame",
                                        "provider yielded a frame after its terminal",
                                    )
                                    logger.warning(
                                        "provider stream handler "
                                        f"'{extension_name}' for "
                                        f"{descriptor.qualified_id} yielded "
                                        "after its terminal: "
                                        f"{type(trailing_frame).__name__}"
                                    )
                                    break
                                if active.input_failure is not None:
                                    await fail(
                                        active.input_failure.code,
                                        active.input_failure.message,
                                    )
                                    break
                            await emit(frame)
                            next_sequence += 1
                            if frame.is_terminal:
                                terminal_sent = True
                                break
                        if (
                            not terminal_sent
                            and not active.cancel_requested.is_set()
                            and active.input_failure is None
                        ):
                            await fail(
                                "provider_error",
                                "provider stream ended without a terminal frame",
                            )
                except TimeoutError:
                    with anyio.CancelScope(shield=True):
                        if active.input_failure is not None:
                            await fail(
                                active.input_failure.code,
                                active.input_failure.message,
                            )
                        else:
                            await fail(
                                "timeout",
                                "provider stream exceeded its single deadline budget",
                            )
                except anyio.get_cancelled_exc_class():
                    if not active.cancel_requested.is_set():
                        raise
            if active.input_failure is not None and not terminal_sent:
                with anyio.CancelScope(shield=True):
                    await fail(
                        active.input_failure.code,
                        active.input_failure.message,
                    )
            elif active.cancel_requested.is_set() and not terminal_sent:
                with anyio.CancelScope(shield=True):
                    await emit(
                        CapabilityStreamFrame(
                            call_id=call.call_id,
                            direction="provider_to_caller",
                            sequence=next_sequence,
                            kind="cancelled",
                            synthetic=True,
                            error=CapabilityStreamError(
                                code="cancelled",
                                message=active.cancel_message,
                            ),
                        )
                    )
                    terminal_sent = True
        except (BrokenResourceError, ClosedResourceError):
            logger.warning(f"provider DATA transport closed during {call.call_id}")
        except Exception as exc:  # noqa: BLE001 - plugin/transport must not escape
            logger.exception(
                f"capability stream handler '{extension_name}' for "
                f"{descriptor.qualified_id} raised: {exc}"
            )
            if not terminal_sent:
                with anyio.CancelScope(shield=True):
                    with contextlib.suppress(BrokenResourceError, ClosedResourceError):
                        if active.input_failure is not None:
                            await fail(
                                active.input_failure.code,
                                active.input_failure.message,
                            )
                        else:
                            await fail(
                                "provider_error",
                                f"{type(exc).__name__}: {exc}",
                            )
        finally:
            self._provider_observer.finalize(call.call_id)
            self._close_active_provider_input(active)
            current = self._active_capability_streams.get(call.call_id)
            if current is active:
                self._active_capability_streams.pop(call.call_id, None)

    async def _consume_provider_input(
        self,
        active: _ActiveProviderStream,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        """Yield one bounded caller input direction through its terminal."""

        assert active.input_receiver is not None
        assert active.input_lifecycle is not None
        last_sequence = -1
        terminal_yielded = False
        with active.input_receiver as frames:
            async for frame in frames:
                self._provider_observer.record_input_queue_depth(
                    active.input_lifecycle.call_id,
                    active.input_receiver.statistics().current_buffer_used,
                )
                last_sequence = frame.sequence
                terminal_yielded = frame.is_terminal
                yield frame
                if frame.is_terminal:
                    return
        if not terminal_yielded and active.input_failure is not None:
            yield CapabilityStreamFrame(
                call_id=active.input_lifecycle.call_id,
                direction="caller_to_provider",
                sequence=last_sequence + 1,
                kind="failed",
                synthetic=True,
                error=active.input_failure,
            )

    @staticmethod
    def _close_active_provider_input(active: _ActiveProviderStream) -> None:
        """Close both ends of a provider input queue idempotently."""

        if active.input_sender is not None:
            active.input_sender.close()
        if active.input_receiver is not None:
            active.input_receiver.close()

    async def _stream_capability(
        self,
        node_id: NodeId,
        capability_id: str,
        version: str,
        descriptor_revision: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> CapabilityStreamSession:
        """Open a provider stream and receive output over ``PROVIDER_DATA``."""

        call_id = str(uuid4())
        requested_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else DEFAULT_CALL_TIMEOUT_SECONDS
        )

        def failed(code: CapabilityErrorCode, message: str) -> CapabilityStreamSession:
            return CapabilityStreamSession(
                open_result=call_failure(call_id, code, message),
                frames=self._empty_capability_stream(),
            )

        if not isinstance(requested_timeout, (int, float)) or not (  # pyright: ignore[reportUnnecessaryIsInstance]
            0 < requested_timeout <= MAX_CALL_TIMEOUT_SECONDS
        ):
            return failed(
                "invalid_payload",
                f"timeout_seconds must be in (0, {MAX_CALL_TIMEOUT_SECONDS}]",
            )
        try:
            payload_bytes = len(
                json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(
                    "utf-8"
                )
            )
        except (TypeError, ValueError, RecursionError, OverflowError) as exc:
            return failed("invalid_payload", f"payload is not JSON-serializable: {exc}")
        if payload_bytes > MAX_CALL_PAYLOAD_BYTES:
            return failed(
                "payload_too_large",
                f"payload is {payload_bytes} bytes (limit {MAX_CALL_PAYLOAD_BYTES})",
            )
        try:
            call = CapabilityCall(
                call_id=call_id,
                capability_id=capability_id,
                version=version,
                descriptor_revision=descriptor_revision,
                caller_node=str(self.node_id),
                target_node=str(node_id),
                timeout_seconds=requested_timeout,
                payload=payload,
            )
        except ValidationError as exc:
            return failed("invalid_payload", f"invalid stream call envelope: {exc}")
        if self._provider_stream_receiver is None:
            return failed(
                "unreachable",
                "provider DATA receive transport is not running on this node",
            )

        output_sender, output_receiver = channel[CapabilityStreamFrame](
            _PROVIDER_STREAM_RECEIVE_BUFFER
        )
        started_at = anyio.current_time()
        deadline_at = started_at + requested_timeout
        receive_state = _ProviderStreamReceiveState(
            output_sender=output_sender,
            receiver=CapabilityStreamReceiver(
                call_id=call_id,
                direction="provider_to_caller",
                gap_timeout_seconds=5.0,
                idle_timeout_seconds=requested_timeout,
            ),
            call=call,
            deadline_at=deadline_at,
        )
        self._provider_stream_receivers[call_id] = receive_state

        if node_id == self.node_id:
            opened = await self._dispatch_capability_stream(call)
        else:
            base_url: str | None = None
            with anyio.move_on_after(requested_timeout) as lookup_scope:
                base_url = await self._peer_api_url_for(node_id)
            if lookup_scope.cancelled_caught:
                opened = call_failure(
                    call_id,
                    "timeout",
                    f"deadline exhausted resolving target {node_id}",
                )
            elif base_url is None:
                opened = call_failure(
                    call_id, "unreachable", f"node {node_id} is not reachable"
                )
            else:
                remaining = deadline_at - anyio.current_time()
                if remaining < 0.05:
                    opened = call_failure(
                        call_id,
                        "timeout",
                        f"target {node_id} resolved, but no stream budget remains",
                    )
                else:
                    remote_call = call.model_copy(update={"timeout_seconds": remaining})
                    try:
                        async with httpx.AsyncClient(
                            timeout=httpx.Timeout(
                                timeout=remaining + 5.0,
                                connect=2.0,
                            ),
                            verify=False,
                        ) as client:
                            response = await client.post(
                                f"{base_url}/v1/capabilities/stream",
                                json=remote_call.model_dump(mode="json"),
                            )
                            response.raise_for_status()
                            opened = CapabilityResult.model_validate(response.json())
                    except (httpx.HTTPError, ValueError, ValidationError) as exc:
                        opened = call_failure(
                            call_id,
                            "unreachable",
                            f"stream open to {node_id} failed: {exc}",
                        )

        if not opened.ok:
            self._provider_stream_receivers.pop(call_id, None)
            output_sender.close()
            output_receiver.close()
            return CapabilityStreamSession(
                open_result=opened,
                frames=self._empty_capability_stream(),
            )
        assert opened.result is not None
        io_mode = opened.result.get("io_mode")
        if io_mode not in (
            "server_streaming",
            "client_streaming",
            "bidirectional",
        ):
            await self._cancel_remote_capability_stream(call)
            self._provider_stream_receivers.pop(call_id, None)
            output_sender.close()
            output_receiver.close()
            return failed(
                "invalid_result",
                "provider stream admission returned no valid io_mode",
            )

        input_stream: CapabilityStreamInput | None = None
        if io_mode in ("client_streaming", "bidirectional"):
            if self._provider_stream_sender is None:
                await self._cancel_remote_capability_stream(call)
                self._provider_stream_receivers.pop(call_id, None)
                output_sender.close()
                output_receiver.close()
                return failed(
                    "unreachable",
                    "provider DATA send transport is not running on this node",
                )

            async def send_input_frame(frame: CapabilityStreamFrame) -> None:
                assert self._provider_stream_sender is not None
                await self._provider_stream_sender.send(
                    ProviderStreamPacket(owner_node=node_id, frame=frame)
                )

            input_stream = CapabilityStreamInput(
                call_id=call_id,
                deadline_at=deadline_at,
                send_frame=send_input_frame,
            )
            receive_state.input_stream = input_stream
            try:
                await input_stream.start()
            except (BrokenResourceError, ClosedResourceError, TimeoutError) as exc:
                await self._cancel_remote_capability_stream(call)
                self._provider_stream_receivers.pop(call_id, None)
                output_sender.close()
                output_receiver.close()
                return failed(
                    "unreachable",
                    f"provider input stream could not start: {exc}",
                )

        async def cancel_output() -> None:
            if receive_state.cancellation_scheduled:
                return
            receive_state.cancel_provider = True
            receive_state.cancellation_scheduled = True
            try:
                await self._cancel_remote_capability_stream(call)
            except Exception as exc:
                logger.opt(exception=exc).warning(
                    "Best-effort provider stream cancellation failed"
                )

        return CapabilityStreamSession(
            open_result=opened,
            frames=self._consume_capability_stream(
                call,
                receive_state,
                output_receiver,
            ),
            input=input_stream,
            cancel_output=cancel_output,
        )

    async def _consume_capability_stream(
        self,
        call: CapabilityCall,
        state: _ProviderStreamReceiveState,
        output_receiver: Receiver[CapabilityStreamFrame],
    ) -> AsyncIterator[CapabilityStreamFrame]:
        """Yield one caller stream and cancel its provider on early close."""

        terminal_yielded = False
        last_yielded_sequence = -1
        try:
            deadline_expired = False
            with output_receiver as frames:
                while True:
                    remaining = max(0.0, state.deadline_at - anyio.current_time())
                    frame: CapabilityStreamFrame | None = None
                    with anyio.move_on_after(remaining) as deadline_scope:
                        try:
                            frame = await frames.receive()
                        except (EndOfStream, ClosedResourceError):
                            break
                    if deadline_scope.cancelled_caught:
                        deadline_expired = True
                        break
                    assert frame is not None
                    last_yielded_sequence = frame.sequence
                    if frame.is_terminal:
                        terminal_yielded = True
                    # A cancel scope must never span an async-generator yield:
                    # finalization may resume the generator in a different task.
                    yield frame
                    if frame.is_terminal:
                        return
            if state.transport_failure is not None:
                state.cancel_provider = True
                terminal = CapabilityStreamFrame(
                    call_id=call.call_id,
                    direction="provider_to_caller",
                    sequence=last_yielded_sequence + 1,
                    kind="failed",
                    synthetic=True,
                    error=CapabilityStreamError(
                        code="transport_error",
                        message=state.transport_failure,
                    ),
                )
            elif deadline_expired:
                state.cancel_provider = True
                terminal = state.receiver.fail(
                    "timeout",
                    "provider stream exceeded its single deadline budget",
                )
            else:
                terminal = state.receiver.terminal or state.receiver.fail(
                    "transport_error",
                    "provider DATA stream closed without a terminal frame",
                )
                if terminal is not None:
                    state.cancel_provider = True
            if terminal is not None:
                terminal_yielded = True
                yield terminal
        finally:
            if state.input_stream is not None:
                self._schedule_provider_input_close(state.input_stream)
            self._provider_stream_receivers.pop(call.call_id, None)
            state.output_sender.close()
            output_receiver.close()
            if (
                not terminal_yielded or state.cancel_provider
            ) and not state.cancellation_scheduled:
                await self._cancel_remote_capability_stream(call)

    def _schedule_provider_stream_cancellation(
        self, state: _ProviderStreamReceiveState
    ) -> None:
        """Cancel a failed caller stream without waiting for its consumer."""

        state.cancel_provider = True
        if state.cancellation_scheduled:
            return
        state.cancellation_scheduled = True
        try:
            self._tg.start_soon(
                self._cancel_remote_capability_stream,
                state.call,
            )
        except RuntimeError:
            # API teardown may close the task group while DATA is draining.
            state.cancellation_scheduled = False

    def _schedule_provider_input_close(
        self, input_stream: CapabilityStreamInput
    ) -> None:
        """Close caller input immediately and retire its DATA stream safely."""

        input_stream.close_locally()
        try:
            self._tg.start_soon(input_stream.finish_local_close)
        except RuntimeError:
            # API teardown may close the task group while DATA is draining.
            return

    async def _cancel_remote_capability_stream(self, call: CapabilityCall) -> None:
        """Best-effort explicit cancellation for an early-closing caller."""

        request = CapabilityStreamCancel(
            call_id=call.call_id,
            caller_node=call.caller_node,
            target_node=call.target_node,
        )
        if call.target_node == str(self.node_id):
            await self.cancel_capability_stream(request)
            return
        with anyio.move_on_after(2.0):
            base_url = await self._peer_api_url_for(NodeId(call.target_node))
            if base_url is None:
                return
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout=2.0, connect=1.0),
                    verify=False,
                ) as client:
                    await client.post(
                        f"{base_url}/v1/capabilities/stream/cancel",
                        json=request.model_dump(mode="json"),
                    )
            except httpx.HTTPError:
                return

    async def _empty_capability_stream(
        self,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        """Return an empty iterator for a stream that failed before admission."""

        if False:
            yield CapabilityStreamFrame(
                call_id="unreachable",
                direction="provider_to_caller",
                sequence=0,
                kind="started",
            )

    async def _call_capability(
        self,
        node_id: NodeId,
        capability_id: str,
        version: str,
        descriptor_revision: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> CapabilityResult:
        """Invoke a capability on a provider node (the caller side of the verb).

        Extensions receive this via ``ExtensionContext.call_capability``. The
        local node is a fast path (in-process dispatch, same guards); a peer is
        reached directly over its API (node-addressed; the master is never in
        the hot path and calls are never event-sourced). Every failure mode
        returns a typed :class:`CapabilityResult` error; this never raises.
        Transport-abstract: the peer hop can move to a Zenoh queryable without
        changing this contract.

        Args:
            node_id: The provider node to call.
            capability_id: The capability's descriptor id.
            version: The exact negotiated semantic version.
            descriptor_revision: The revision digest discovered via
                describe, pinning the contract shape. Shadows the module-level
                digest helper inside this method (unused here) because the
                caller protocol fixes the parameter name.
            payload: The opaque call payload.
            timeout_seconds: Deadline for the call (default
                ``DEFAULT_CALL_TIMEOUT_SECONDS``).

        Returns:
            The typed result of the call.
        """
        call_id = str(uuid4())
        requested_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else DEFAULT_CALL_TIMEOUT_SECONDS
        )
        if not isinstance(requested_timeout, (int, float)) or not (  # pyright: ignore[reportUnnecessaryIsInstance]
            0 < requested_timeout <= MAX_CALL_TIMEOUT_SECONDS
        ):
            # Validate the timeout BEFORE it becomes the lookup budget: an
            # out-of-range value must fail fast as a typed error, not stall
            # the reachability probe or surface as a misleading unreachable.
            return call_failure(
                call_id,
                "invalid_payload",
                f"timeout_seconds must be in (0, {MAX_CALL_TIMEOUT_SECONDS}]; "
                f"got {requested_timeout}",
            )
        try:
            payload_bytes = len(
                json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(
                    "utf-8"
                )
            )
        except (TypeError, ValueError, RecursionError, OverflowError) as exc:
            return call_failure(
                call_id,
                "invalid_payload",
                f"payload is not JSON-serializable: {exc}",
            )
        if payload_bytes > MAX_CALL_PAYLOAD_BYTES:
            # Enforce the cap before the POST ships an oversized body across
            # the network just to be rejected on arrival.
            return call_failure(
                call_id,
                "payload_too_large",
                f"payload is {payload_bytes} bytes (limit {MAX_CALL_PAYLOAD_BYTES})",
            )

        # Validate the FULL envelope before any network work: a violation from
        # an untyped caller (non-string ids and the like) fails fast as a
        # typed error instead of after a wasted reachability probe.
        try:
            call = CapabilityCall(
                call_id=call_id,
                capability_id=capability_id,
                version=version,
                descriptor_revision=descriptor_revision,
                caller_node=str(self.node_id),
                target_node=str(node_id),
                timeout_seconds=requested_timeout,
                payload=payload,
            )
        except ValidationError as exc:
            return call_failure(
                call_id, "invalid_payload", f"invalid call envelope: {exc}"
            )

        if node_id == self.node_id:
            return await self._dispatch_capability_call(call)
        # One budget clock spans the WHOLE remote call (#513): target
        # resolution consumes from the same deadline the provider gets, so the
        # caller can never wait materially longer than it asked for (the old
        # shape allowed lookup + provider to each use the full budget).
        started_at = anyio.current_time()
        base_url: str | None = None
        with anyio.move_on_after(requested_timeout) as lookup_scope:
            base_url = await self._peer_api_url_for(node_id)
        if lookup_scope.cancelled_caught:
            # Deadline exhaustion while resolving is a timeout, not a verdict
            # that the node is unreachable; the caller can retry with a larger
            # budget where a true unreachable would not change.
            return call_failure(
                call_id,
                "timeout",
                f"deadline exhausted resolving target {node_id}",
            )
        if base_url is None:
            return call_failure(
                call_id, "unreachable", f"node {node_id} is not reachable"
            )
        remaining = requested_timeout - (anyio.current_time() - started_at)
        if remaining < 0.05:
            return call_failure(
                call_id,
                "timeout",
                f"target {node_id} resolved, but no budget remains for the call itself",
            )
        # Re-stamp the envelope with the remaining budget. model_copy skips
        # validation, which is safe here: remaining is bounded by the already
        # validated requested_timeout above and the 0.05s floor just checked.
        call = call.model_copy(update={"timeout_seconds": remaining})
        # The HTTP deadline extends slightly past the provider's remaining
        # budget so a typed timeout from the provider wins over a transport
        # timeout.
        http_timeout = httpx.Timeout(timeout=remaining + 5.0, connect=2.0)
        try:
            async with httpx.AsyncClient(timeout=http_timeout, verify=False) as client:
                response = await client.post(
                    f"{base_url}/v1/capabilities/call",
                    json=call.model_dump(mode="json"),
                )
                response.raise_for_status()
                return CapabilityResult.model_validate(response.json())
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            return call_failure(
                call.call_id, "unreachable", f"call to {node_id} failed: {exc}"
            )

    async def steward_extension_tools(
        self, *, proposals_allowed: bool
    ) -> tuple[StewardToolBinding, ...]:
        """Return eligible installed adapter tools for one steward model step."""
        if self._extensions is None:
            return ()
        return await self._extensions.steward_tools(
            self._extension_context, proposals_allowed=proposals_allowed
        )

    async def invoke_steward_extension_tool(
        self,
        binding: StewardToolBinding,
        arguments: dict[str, object],
        *,
        proposals_allowed: bool,
    ) -> str:
        """Invoke a read or inert proposal without exposing execution authority."""
        if self._extensions is None:
            return '{"error":"extension tool unavailable or refused"}'
        return await self._extensions.invoke_steward_tool(
            binding,
            self._extension_context,
            arguments,
            proposals_allowed=proposals_allowed,
        )

    async def _describe_node_capabilities(
        self, node_id: NodeId
    ) -> tuple[CapabilityDescriptor, ...]:
        """Fetch a node's full capability descriptors (the describe verb).

        The heavy half of capability discovery (fabric-citizenship Phase 2a):
        the telemetry tag says *that* a node offers a capability; this returns
        the self-describing descriptors that say *how to call it*. For the
        local node it reads the loaded extensions directly; for a peer it
        proxies ``GET /v1/capabilities`` over the reachable-peer-API path the
        trace cluster-browse already uses (an existing mechanism, not a new
        transport; the extension-facing contract stays transport-abstract so
        this can later ride the provider call plane instead).

        Returns an empty tuple when the node is unreachable, serves no
        capabilities, or returns an unparsable payload; never raises.

        Args:
            node_id: The node whose descriptors to fetch.

        Returns:
            The node's capability descriptors, or ``()``.
        """
        if node_id == self.node_id:
            if self._extensions is None:
                return ()
            return self._extensions.capability_descriptors
        peer_urls = await self._reachable_peer_api_urls()
        base_url = peer_urls.get(str(node_id))
        if base_url is None:
            return ()
        timeout = httpx.Timeout(timeout=10.0, connect=2.0)
        try:
            async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
                response = await client.get(f"{base_url}/v1/capabilities")
                response.raise_for_status()
                payload_raw = cast("object", response.json())
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                f"describe_node: peer {node_id} capabilities fetch failed: {exc}"
            )
            return ()
        # A peer can return any JSON shape; a non-object payload degrades to
        # "no capabilities" rather than violating the never-raises contract.
        if not isinstance(payload_raw, dict):
            return ()
        payload = cast("dict[str, object]", payload_raw)
        raw_descriptors = payload.get("capabilities")
        if not isinstance(raw_descriptors, list):
            return ()
        descriptors: list[CapabilityDescriptor] = []
        for entry in cast("list[object]", raw_descriptors):
            try:
                descriptors.append(CapabilityDescriptor.model_validate(entry))
            except ValidationError as exc:
                # A malformed entry degrades to "not offered" rather than
                # poisoning the whole describe (mixed-version peers are
                # unsupported, but a plugin can publish garbage on any version).
                logger.warning(
                    f"describe_node: peer {node_id} returned an invalid "
                    f"capability descriptor; skipping it: {exc}"
                )
        return tuple(descriptors)

    async def embed_texts(
        self,
        texts: list[str],
        model_id: ModelId | None = None,
        *,
        timeout_seconds: float = 30.0,
    ) -> list[list[float]] | None:
        """Embed texts through the cluster's embedding serving, in process.

        The programmatic equivalent of ``POST /v1/embeddings`` for in-process
        callers (extensions receive this via their ``ExtensionContext``).
        When ``model_id`` is omitted, the first running embedding instance is
        used. Returns one vector per input text, or ``None`` when no
        embedding instance is available, the task errors, or the call times
        out. Callers must treat ``None`` as "degrade gracefully": this method
        never raises for serving-availability reasons. Side effect: sends a
        ``TextEmbedding`` command through the cluster pipeline.
        """
        from skulk.shared.types.embedding import TextEmbeddingTaskParams

        resolved = (
            model_id if model_id is not None else self._find_running_embedding_model()
        )
        if resolved is None:
            return None
        command = TextEmbedding(
            owner_node=self.node_id,
            task_params=TextEmbeddingTaskParams(
                model=resolved,
                input_texts=texts,
                encoding_format="float",
            ),
        )
        command_id = command.command_id
        recv = self._open_stream_queue(self._embedding_queues, command_id)
        try:
            with anyio.fail_after(timeout_seconds):
                await self._send(command)
                with recv as chunks:
                    async for chunk in chunks:
                        if isinstance(chunk, ErrorChunk):
                            logger.warning(f"embed_texts failed: {chunk.error_message}")
                            return None
                        return [list(embedding) for embedding in chunk.embeddings]
            return None
        except TimeoutError:
            # The runner may still be computing this command. A bare finalize
            # would emit TaskFinished and mark a live task complete on the
            # master, so cancel it instead (mirrors the endpoint's
            # client-disconnect path); _cancelled_command_ids suppresses the
            # finished signal in the finally below.
            logger.warning(f"embed_texts timed out after {timeout_seconds}s")
            self._cancelled_command_ids.add(command_id)
            await self.command_sender.send(
                ForwarderCommand(
                    origin=self._system_id,
                    command=TaskCancelled(cancelled_command_id=command_id),
                )
            )
            return None
        finally:
            await self._finalize_command_stream(
                command_id,
                cast(
                    "dict[CommandId, Sender[object]]",
                    self._embedding_queues,
                ),
            )

    def _find_running_embedding_model(self) -> ModelId | None:
        """Return the model id of a running embedding instance, if any.

        Reads the in-memory shard metadata (like text-model resolution) so
        availability does not depend on model-card cache misses or remote
        fetches.
        """
        from skulk.shared.models.model_cards import ModelTask

        for instance in self.state.instances.values():
            assignments = instance.shard_assignments
            for shard in assignments.runner_to_shard.values():
                if ModelTask.TextEmbedding in shard.model_card.tasks:
                    return assignments.model_id
        return None

    def stream_events(self) -> StreamingResponse:
        def _generate_json_array(events: Iterable[Event]) -> Iterable[str]:
            yield "["
            first = True
            for event in events:
                if not first:
                    yield ","
                first = False
                yield event.model_dump_json()
            yield "]"

        return StreamingResponse(
            _generate_json_array(
                [] if self._event_log is None else self._event_log.read_all()
            ),
            media_type="application/json",
        )

    async def get_image(self, image_id: str) -> FileResponse:
        stored = self._image_store.get(Id(image_id))
        if stored is None:
            raise HTTPException(status_code=404, detail="Image not found or expired")
        return FileResponse(path=stored.file_path, media_type=stored.content_type)

    async def list_images(self, request: Request) -> ImageListResponse:
        """List all stored images."""
        stored_images = self._image_store.list_images()
        return ImageListResponse(
            data=[
                ImageListItem(
                    image_id=img.image_id,
                    url=self._build_image_url(request, img.image_id),
                    content_type=img.content_type,
                    expires_at=img.expires_at,
                )
                for img in stored_images
            ]
        )

    def _build_image_url(self, request: Request, image_id: Id) -> str:
        host = request.headers.get("host", f"localhost:{self.port}")
        scheme = "https" if request.url.scheme == "https" else "http"
        # The stored-image fetch route is GET /images/{image_id} (no /v1
        # prefix); emitting /v1/... here returned URLs that always 404ed,
        # found when the 1.5.0 docs review documented the url response mode.
        return f"{scheme}://{host}/images/{image_id}"

    async def image_generations(
        self, request: Request, payload: ImageGenerationTaskParams
    ) -> ImageGenerationResponse | StreamingResponse:
        """Handle image generation requests.

        When stream=True and partial_images > 0, returns a StreamingResponse
        with SSE-formatted events for partial and final images.
        """
        payload = payload.model_copy(
            update={
                "model": await self._validate_image_model(ModelId(payload.model)),
                "advanced_params": _ensure_seed(payload.advanced_params),
            }
        )

        command = ImageGeneration(
            owner_node=self.node_id,
            task_params=payload,
        )
        await self._send(command)

        # Check if streaming is requested
        if payload.stream and payload.partial_images and payload.partial_images > 0:
            return StreamingResponse(
                self._generate_image_stream(
                    request=request,
                    command_id=command.command_id,
                    num_images=payload.n or 1,
                    response_format=payload.response_format or "b64_json",
                ),
                media_type="text/event-stream",
            )

        # Non-streaming: collect all image chunks
        return await self._collect_image_generation(
            request=request,
            command_id=command.command_id,
            num_images=payload.n or 1,
            response_format=payload.response_format or "b64_json",
        )

    async def _generate_image_stream(
        self,
        request: Request,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> AsyncGenerator[str, None]:
        """Generate SSE stream of partial and final images."""
        # Track chunks: {(image_index, is_partial): {chunk_index: data}}
        image_chunks: dict[tuple[int, bool], dict[int, str]] = {}
        image_total_chunks: dict[tuple[int, bool], int] = {}
        image_metadata: dict[tuple[int, bool], tuple[int | None, int | None]] = {}
        images_complete = 0

        try:
            recv = self._open_stream_queue(self._image_generation_queues, command_id)

            if pending_error := self._take_vision_media_failure(command_id):
                error_response = ErrorResponse(
                    error=ErrorInfo(
                        message=pending_error.error_message
                        or "Vision media transport failed",
                        type="InternalServerError",
                        code=500,
                    )
                )
                yield f"data: {error_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            with recv as chunks:
                async for chunk in chunks:
                    if chunk.finish_reason == "error":
                        error_response = ErrorResponse(
                            error=ErrorInfo(
                                message=chunk.error_message or "Internal server error",
                                type="InternalServerError",
                                code=500,
                            )
                        )
                        yield f"data: {error_response.model_dump_json()}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    key = (chunk.image_index, chunk.is_partial)

                    if key not in image_chunks:
                        image_chunks[key] = {}
                        image_total_chunks[key] = chunk.total_chunks
                        image_metadata[key] = (
                            chunk.partial_index,
                            chunk.total_partials,
                        )

                    image_chunks[key][chunk.chunk_index] = chunk.data

                    # Check if this image is complete
                    if len(image_chunks[key]) == image_total_chunks[key]:
                        full_data = "".join(
                            image_chunks[key][i] for i in range(len(image_chunks[key]))
                        )

                        partial_idx, total_partials = image_metadata[key]

                        if chunk.is_partial:
                            # Yield partial image event (always use b64_json for partials)
                            event_data = {
                                "type": "partial",
                                "image_index": chunk.image_index,
                                "partial_index": partial_idx,
                                "total_partials": total_partials,
                                "format": str(chunk.format),
                                "data": {
                                    "b64_json": full_data
                                    if response_format == "b64_json"
                                    else None,
                                },
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"
                        else:
                            # Final image
                            if response_format == "url":
                                image_bytes = base64.b64decode(full_data)
                                content_type = _format_to_content_type(chunk.format)
                                stored = self._image_store.store(
                                    image_bytes, content_type
                                )
                                url = self._build_image_url(request, stored.image_id)
                                event_data = {
                                    "type": "final",
                                    "image_index": chunk.image_index,
                                    "format": str(chunk.format),
                                    "data": {"url": url},
                                }
                            else:
                                event_data = {
                                    "type": "final",
                                    "image_index": chunk.image_index,
                                    "format": str(chunk.format),
                                    "data": {"b64_json": full_data},
                                }
                            yield f"data: {json.dumps(event_data)}\n\n"
                            images_complete += 1

                            if images_complete >= num_images:
                                yield "data: [DONE]\n\n"
                                break

                        # Clean up completed image chunks
                        del image_chunks[key]
                        del image_total_chunks[key]
                        del image_metadata[key]

        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            self._cancelled_command_ids.add(command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._finalize_command_stream(
                command_id,
                cast(
                    dict[CommandId, Sender[object]],
                    self._image_generation_queues,
                ),
            )

    async def _collect_image_chunks(
        self,
        request: Request | None,
        command_id: CommandId,
        num_images: int,
        response_format: str,
        capture_stats: bool = False,
    ) -> tuple[list[ImageData], ImageGenerationStats | None]:
        """Collect image chunks and optionally capture stats."""
        # Track chunks per image: {image_index: {chunk_index: data}}
        # Only track non-partial (final) images
        image_chunks: dict[int, dict[int, str]] = {}
        image_total_chunks: dict[int, int] = {}
        image_formats: dict[int, Literal["png", "jpeg", "webp"] | None] = {}
        images_complete = 0
        stats: ImageGenerationStats | None = None

        try:
            recv = self._open_stream_queue(self._image_generation_queues, command_id)

            if pending_error := self._take_vision_media_failure(command_id):
                raise HTTPException(
                    status_code=500,
                    detail=pending_error.error_message
                    or "Vision media transport failed",
                )

            while images_complete < num_images:
                with recv as chunks:
                    async for chunk in chunks:
                        if chunk.finish_reason == "error":
                            raise HTTPException(
                                status_code=500,
                                detail=chunk.error_message or "Internal server error",
                            )

                        if chunk.is_partial:
                            continue

                        if chunk.image_index not in image_chunks:
                            image_chunks[chunk.image_index] = {}
                            image_total_chunks[chunk.image_index] = chunk.total_chunks
                            image_formats[chunk.image_index] = chunk.format

                        image_chunks[chunk.image_index][chunk.chunk_index] = chunk.data

                        if capture_stats and chunk.stats is not None:
                            stats = chunk.stats

                        if (
                            len(image_chunks[chunk.image_index])
                            == image_total_chunks[chunk.image_index]
                        ):
                            images_complete += 1

                        if images_complete >= num_images:
                            break

            images: list[ImageData] = []
            for image_idx in range(num_images):
                chunks_dict = image_chunks[image_idx]
                full_data = "".join(chunks_dict[i] for i in range(len(chunks_dict)))
                if response_format == "url" and request is not None:
                    image_bytes = base64.b64decode(full_data)
                    content_type = _format_to_content_type(image_formats.get(image_idx))
                    stored = self._image_store.store(image_bytes, content_type)
                    url = self._build_image_url(request, stored.image_id)
                    images.append(ImageData(b64_json=None, url=url))
                else:
                    images.append(
                        ImageData(
                            b64_json=full_data
                            if response_format == "b64_json"
                            else None,
                            url=None,
                        )
                    )

            return (images, stats if capture_stats else None)
        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            self._cancelled_command_ids.add(command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._finalize_command_stream(
                command_id,
                cast(
                    dict[CommandId, Sender[object]],
                    self._image_generation_queues,
                ),
            )

    async def _collect_image_generation(
        self,
        request: Request,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> ImageGenerationResponse:
        """Collect all image chunks (non-streaming) and return a single response."""
        images, _ = await self._collect_image_chunks(
            request, command_id, num_images, response_format, capture_stats=False
        )
        return ImageGenerationResponse(data=images)

    async def _collect_image_generation_with_stats(
        self,
        request: Request | None,
        command_id: CommandId,
        num_images: int,
        response_format: str,
    ) -> BenchImageGenerationResponse:
        sampler = PowerSampler(
            get_node_system=lambda: {
                node_id: profile
                for node_id, profile in self._telemetry_view.node_system.items()
                if node_id in self.state.last_seen
            }
        )
        images: list[ImageData] = []
        stats: ImageGenerationStats | None = None
        async with anyio.create_task_group() as tg:
            tg.start_soon(sampler.run)
            images, stats = await self._collect_image_chunks(
                request, command_id, num_images, response_format, capture_stats=True
            )
            tg.cancel_scope.cancel()
        return BenchImageGenerationResponse(
            data=images, generation_stats=stats, power_usage=sampler.result()
        )

    async def bench_image_generations(
        self, request: Request, payload: BenchImageGenerationTaskParams
    ) -> BenchImageGenerationResponse:
        payload = payload.model_copy(
            update={
                "model": await self._validate_image_model(ModelId(payload.model)),
                "stream": False,
                "partial_images": 0,
                "advanced_params": _ensure_seed(payload.advanced_params),
            }
        )

        command = ImageGeneration(
            owner_node=self.node_id,
            task_params=payload,
        )
        await self._send(command)

        return await self._collect_image_generation_with_stats(
            request=request,
            command_id=command.command_id,
            num_images=payload.n or 1,
            response_format=payload.response_format or "b64_json",
        )

    async def _send_image_edits_command(
        self,
        image: UploadFile,
        prompt: str,
        model: ModelId,
        n: int,
        size: ImageSize,
        response_format: Literal["url", "b64_json"],
        input_fidelity: Literal["low", "high"],
        stream: bool,
        partial_images: int,
        bench: bool,
        quality: Literal["high", "medium", "low"],
        output_format: Literal["png", "jpeg", "webp"],
        advanced_params: AdvancedImageParams | None,
    ) -> ImageEdits:
        """Prepare and send an image edits command with chunked image upload."""
        resolved_model = await self._validate_image_model(model)
        advanced_params = _ensure_seed(advanced_params)

        image_content = await image.read(_VISION_MEDIA_RAW_IMAGE_BYTES + 1)
        if len(image_content) > _VISION_MEDIA_RAW_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Input image exceeds the per-request media limit",
            )
        image_data = base64.b64encode(image_content).decode("utf-8")

        image_strength = 0.7 if input_fidelity == "high" else 0.3

        data_chunks = [
            image_data[i : i + SKULK_MAX_CHUNK_SIZE]
            for i in range(0, len(image_data), SKULK_MAX_CHUNK_SIZE)
        ]
        total_chunks = len(data_chunks)

        command = ImageEdits(
            owner_node=self.node_id,
            task_params=ImageEditsTaskParams(
                image_data="",
                total_input_chunks=total_chunks,
                prompt=prompt,
                model=resolved_model,
                n=n,
                size=size,
                response_format=response_format,
                image_strength=image_strength,
                stream=stream,
                partial_images=partial_images,
                bench=bench,
                quality=quality,
                output_format=output_format,
                advanced_params=advanced_params,
            ),
        )

        logger.info(
            f"Sending input image: {len(image_data)} bytes in {total_chunks} chunks"
        )
        self._stage_vision_media(
            command.command_id,
            resolved_model,
            [(0, chunk_data) for chunk_data in data_chunks],
            1,
        )
        try:
            await self._send(command)
        except BaseException:
            self._take_pending_vision_media(command.command_id)
            self._vision_media_commands.discard(command.command_id)
            self._vision_media_models.pop(command.command_id, None)
            raise
        return command

    async def image_edits(
        self,
        request: Request,
        image: UploadFile = File(...),  # noqa: B008
        prompt: str = Form(...),
        model: str = Form(...),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: Literal["url", "b64_json"] = Form("b64_json"),
        input_fidelity: Literal["low", "high"] = Form("low"),
        stream: str = Form("false"),
        partial_images: str = Form("0"),
        quality: Literal["high", "medium", "low"] = Form("medium"),
        output_format: Literal["png", "jpeg", "webp"] = Form("png"),
        advanced_params: str | None = Form(None),
    ) -> ImageGenerationResponse | StreamingResponse:
        """Handle image editing requests (img2img)."""
        # Parse string form values to proper types
        stream_bool = stream.lower() in ("true", "1", "yes")
        partial_images_int = int(partial_images) if partial_images.isdigit() else 0

        parsed_advanced_params: AdvancedImageParams | None = None
        if advanced_params:
            with contextlib.suppress(Exception):
                parsed_advanced_params = AdvancedImageParams.model_validate_json(
                    advanced_params
                )

        command = await self._send_image_edits_command(
            image=image,
            prompt=prompt,
            model=ModelId(model),
            n=n,
            size=normalize_image_size(size),
            response_format=response_format,
            input_fidelity=input_fidelity,
            stream=stream_bool,
            partial_images=partial_images_int,
            bench=False,
            quality=quality,
            output_format=output_format,
            advanced_params=parsed_advanced_params,
        )

        if stream_bool and partial_images_int > 0:
            return StreamingResponse(
                self._generate_image_stream(
                    request=request,
                    command_id=command.command_id,
                    num_images=n,
                    response_format=response_format,
                ),
                media_type="text/event-stream",
            )

        return await self._collect_image_generation(
            request=request,
            command_id=command.command_id,
            num_images=n,
            response_format=response_format,
        )

    async def bench_image_edits(
        self,
        request: Request,
        image: UploadFile = File(...),  # noqa: B008
        prompt: str = Form(...),
        model: str = Form(...),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: Literal["url", "b64_json"] = Form("b64_json"),
        input_fidelity: Literal["low", "high"] = Form("low"),
        quality: Literal["high", "medium", "low"] = Form("medium"),
        output_format: Literal["png", "jpeg", "webp"] = Form("png"),
        advanced_params: str | None = Form(None),
    ) -> BenchImageGenerationResponse:
        """Handle benchmark image editing requests with generation stats."""
        parsed_advanced_params: AdvancedImageParams | None = None
        if advanced_params:
            with contextlib.suppress(Exception):
                parsed_advanced_params = AdvancedImageParams.model_validate_json(
                    advanced_params
                )

        command = await self._send_image_edits_command(
            image=image,
            prompt=prompt,
            model=ModelId(model),
            n=n,
            size=normalize_image_size(size),
            response_format=response_format,
            input_fidelity=input_fidelity,
            stream=False,
            partial_images=0,
            bench=True,
            quality=quality,
            output_format=output_format,
            advanced_params=parsed_advanced_params,
        )

        return await self._collect_image_generation_with_stats(
            request=request,
            command_id=command.command_id,
            num_images=n,
            response_format=response_format,
        )

    async def claude_messages(
        self, payload: ClaudeMessagesRequest
    ) -> ClaudeMessagesResponse | StreamingResponse:
        """Claude Messages API - adapter."""
        resolved_model = await self._resolve_and_validate_text_model(payload.model)
        model_card = await self._get_running_model_card(resolved_model)
        task_params = await claude_request_to_text_generation(
            payload.model_copy(update={"model": resolved_model}),
            model_card=model_card,
        )
        if task_params.images:
            resolved_images: list[str] = []
            for img in task_params.images:
                if img.startswith(("http://", "https://")):
                    resolved_images.append(await fetch_image_url(img))
                else:
                    resolved_images.append(img)
            task_params = task_params.model_copy(update={"images": resolved_images})

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_claude_stream(
                        command.command_id,
                        payload.model,
                        self._tapped_text_stream(
                            command.command_id,
                            task_params.model,
                            task_params=task_params,
                        ),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_claude_response(
                    command.command_id,
                    payload.model,
                    self._tapped_text_stream(
                        command.command_id,
                        task_params.model,
                        task_params=task_params,
                    ),
                ),
                media_type="application/json",
            )

    async def openai_responses(
        self, payload: ResponsesRequest
    ) -> ResponsesResponse | StreamingResponse:
        """OpenAI Responses API."""
        resolved_model = await self._resolve_and_validate_text_model(payload.model)
        model_card = await self._get_running_model_card(resolved_model)
        task_params = await responses_request_to_text_generation(
            payload.model_copy(update={"model": resolved_model}),
            model_card=model_card,
        )

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                with_sse_keepalive(
                    generate_responses_stream(
                        command.command_id,
                        payload.model,
                        self._tapped_text_stream(
                            command.command_id,
                            task_params.model,
                            task_params=task_params,
                        ),
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            return StreamingResponse(
                collect_responses_response(
                    command.command_id,
                    payload.model,
                    self._tapped_text_stream(
                        command.command_id,
                        task_params.model,
                        task_params=task_params,
                    ),
                ),
                media_type="application/json",
            )

    async def web_search(self, payload: WebSearchToolRequest) -> WebSearchToolResponse:
        """Execute the generic web-search tool and return structured results."""
        provider = default_browser_tool_provider()
        try:
            results = await provider.search(payload.query, top_k=payload.top_k)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Web search failed: {exc}",
            ) from exc

        return WebSearchToolResponse(
            query=payload.query,
            results=results,
            provider=provider.provider_name,
        )

    async def open_url(self, payload: OpenUrlToolRequest) -> OpenUrlToolResponse:
        """Execute the generic URL-open tool and return structured metadata."""
        provider = default_browser_tool_provider()
        try:
            return await provider.open_url(payload.url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Open URL failed: {exc}",
            ) from exc

    async def extract_page(
        self, payload: ExtractPageToolRequest
    ) -> ExtractPageToolResponse:
        """Execute the generic page-extraction tool and return readable text."""
        provider = default_browser_tool_provider()
        try:
            return await provider.extract_page(payload.url, max_chars=payload.max_chars)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Extract page failed: {exc}",
            ) from exc

    async def _ollama_root(self) -> JSONResponse:
        """Respond to HEAD / from Ollama CLI connectivity checks."""
        return JSONResponse(content="Ollama is running")

    async def ollama_chat(
        self, request: Request
    ) -> OllamaChatResponse | StreamingResponse:
        """Ollama Chat API — accepts JSON regardless of Content-Type."""
        body = await request.body()
        payload = OllamaChatRequest.model_validate_json(body)
        resolved_model = await self._resolve_and_validate_text_model(payload.model)
        model_card = await self._get_running_model_card(resolved_model)
        task_params = ollama_request_to_text_generation(
            payload.model_copy(update={"model": resolved_model}),
            model_card=model_card,
        )

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                generate_ollama_chat_stream(
                    command.command_id,
                    self._tapped_text_stream(
                        command.command_id,
                        task_params.model,
                        task_params=task_params,
                    ),
                ),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_ollama_chat_response(
                    command.command_id,
                    self._tapped_text_stream(
                        command.command_id,
                        task_params.model,
                        task_params=task_params,
                    ),
                ),
                media_type="application/json",
            )

    async def ollama_generate(
        self, request: Request
    ) -> OllamaGenerateResponse | StreamingResponse:
        """Ollama Generate API — accepts JSON regardless of Content-Type."""
        body = await request.body()
        payload = OllamaGenerateRequest.model_validate_json(body)
        resolved_model = await self._resolve_and_validate_text_model(payload.model)
        model_card = await self._get_running_model_card(resolved_model)
        task_params = ollama_generate_request_to_text_generation(
            payload.model_copy(update={"model": resolved_model}),
            model_card=model_card,
        )

        command = await self._send_text_generation_with_images(task_params)

        if payload.stream:
            return StreamingResponse(
                generate_ollama_generate_stream(
                    command.command_id,
                    self._tapped_text_stream(
                        command.command_id,
                        task_params.model,
                        task_params=task_params,
                    ),
                ),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return StreamingResponse(
                collect_ollama_generate_response(
                    command.command_id,
                    self._tapped_text_stream(
                        command.command_id,
                        task_params.model,
                        task_params=task_params,
                    ),
                ),
                media_type="application/json",
            )

    async def ollama_tags(self) -> OllamaTagsResponse:
        """Returns list of models in Ollama tags format. We return the downloaded ones only."""

        def none_if_empty(value: str) -> str | None:
            return value or None

        downloaded_model_ids: set[str] = set()
        for node_downloads in self.state.downloads.values():
            for dl in node_downloads:
                if isinstance(dl, DownloadCompleted):
                    downloaded_model_ids.add(dl.shard_metadata.model_card.model_id)

        cards = [
            c for c in await get_model_cards() if c.model_id in downloaded_model_ids
        ]

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return OllamaTagsResponse(
            models=[
                OllamaModelTag(
                    name=str(card.model_id),
                    model=str(card.model_id),
                    modified_at=now,
                    size=card.storage_size.in_bytes,
                    digest="sha256:000000000000",
                    details=OllamaModelDetails(
                        family=none_if_empty(card.family),
                        quantization_level=none_if_empty(card.quantization),
                    ),
                )
                for card in cards
            ]
        )

    async def ollama_show(self, request: Request) -> OllamaShowResponse:
        """Returns model information in Ollama show format."""
        body = await request.body()
        payload = OllamaShowRequest.model_validate_json(body)
        model_name = payload.name or payload.model
        if not model_name:
            raise HTTPException(status_code=400, detail="name or model is required")
        try:
            card = await ModelCard.load(ModelId(model_name))
        except Exception as exc:
            raise HTTPException(
                status_code=404, detail=f"Model not found: {model_name}"
            ) from exc

        return OllamaShowResponse(
            modelfile=f"FROM {card.model_id}",
            template="{{ .Prompt }}",
            details=OllamaModelDetails(
                family=card.family or None,
                quantization_level=card.quantization or None,
            ),
        )

    async def ollama_ps(self) -> OllamaPsResponse:
        """Returns list of running models (active instances)."""
        models: list[OllamaPsModel] = []
        seen: set[str] = set()
        for instance in self.state.instances.values():
            model_id = str(instance.shard_assignments.model_id)
            if model_id in seen:
                continue
            seen.add(model_id)
            models.append(
                OllamaPsModel(
                    name=model_id,
                    model=model_id,
                    size=0,
                )
            )
        return OllamaPsResponse(models=models)

    async def ollama_version(self) -> dict[str, str]:
        """Returns version information for Ollama API compatibility."""
        return {"version": get_skulk_version_label()}

    def _calculate_total_available_memory(self) -> Memory:
        """Total available RAM across nodes still live in ``last_seen``.

        Feeds the ``POST /instance`` admission pre-check, so it must exclude
        timed-out nodes: counting a dead node's last RAM sample could admit a
        placement the live cluster cannot support. Filtered to ``last_seen``-live
        nodes as defense-in-depth on top of the worker's ``NodeTimedOut`` prune
        of the view, matching ``get_cluster_state`` and the power sampler.
        """
        total_available = Memory()
        for node_id, memory in self._telemetry_view.node_memory.items():
            if node_id in self.state.last_seen:
                total_available += memory.ram_available
        return total_available

    @staticmethod
    def _model_tags(card: "ModelCard") -> list[str]:
        """Derive display tags from model metadata."""
        tags: list[str] = []
        model_id_lower = str(card.model_id).lower()
        quant_lower = card.quantization.lower()
        # OptiQ mixed-precision models
        if "optiq" in model_id_lower or "optiq" in quant_lower:
            tags.append("optiq")
        # Thinking capability
        if "thinking" in card.capabilities:
            tags.append("thinking")
        # Vision-capable models
        if "vision" in card.capabilities:
            tags.append("vision")
        # Tensor parallel support
        if card.supports_tensor:
            tags.append("tensor")
        # Embedding models
        if "embedding" in card.capabilities:
            tags.append("embedding")
        # Speech models
        if (
            "tts" in card.capabilities
            or ModelTask.TextToSpeech in card.tasks
            or (
                card.audio is not None and card.audio.kind == AudioCardKind.TextToSpeech
            )
        ):
            tags.append("tts")
        if (
            "stt" in card.capabilities
            or ModelTask.SpeechToText in card.tasks
            or ModelTask.SpeechTranslation in card.tasks
            or (
                card.audio is not None and card.audio.kind == AudioCardKind.SpeechToText
            )
        ):
            tags.append("stt")
        return tags

    @staticmethod
    def _model_list_entry(
        card: "ModelCard",
        approved_remote_code_card_ids: frozenset[str] | None = None,
        installed_record: InstalledCardRecord | None = None,
    ) -> ModelListModel:
        """Build the public model-list representation for one model card.

        Args:
            card: Effective catalog card exposed by the API.
            approved_remote_code_card_ids: Deprecated approvals accepted for
                compatibility with older callers; current entries ignore them.
            installed_record: Authoritative central-store generation when one
                exists. When omitted, the node-local installed-card cache keeps
                air-gapped and store-unreachable operation self-describing.

        Returns:
            The public catalog projection for ``card``.
        """
        catalog_card = card
        local_installed_record = get_installed_card_record(card.model_id)
        if (
            local_installed_record is not None
            and local_installed_record.model_card.qualification_only
            and not catalog_card.qualification_only
        ):
            # A retained local qualification artifact remains usable for a
            # future signed-card refresh, but its temporary unsigned card is
            # not installed truth for a normal catalog entry sharing the alias.
            local_installed_record = None
        if catalog_card.is_custom:
            # A custom catalog entry is an operator-owned override. Prefer its
            # node-local custom generation, but do not hide the same custom
            # generation merely because it is canonical only in the store.
            # Non-custom records sharing the alias cannot establish installed
            # state for the custom card.
            installed_record = (
                local_installed_record
                if local_installed_record is not None
                and local_installed_record.model_card.is_custom
                else installed_record
                if installed_record is not None
                and installed_record.model_card.is_custom
                else None
            )
        else:
            installed_record = installed_record or local_installed_record
        if installed_record is not None:
            card = installed_record.model_card
        remote_code_approval_required = remote_code_execution_requires_approval(card)
        if remote_code_approval_required and approved_remote_code_card_ids is None:
            approved_remote_code_card_ids = approved_remote_code_identities()
        resolved_profile = resolve_model_capability_profile(
            card.model_id,
            model_card=card,
        )
        description = (
            "resolved_capabilities reflects the default tool-free request path; "
            "request-specific options such as tools may change prompt rendering "
            "and related resolved capability values."
        )
        current_registry_card = get_current_registry_card(catalog_card.model_id)
        current_registry_identity = get_current_registry_card_id(
            catalog_card.model_id
        ) or (
            current_registry_card.registry_card_id
            if current_registry_card is not None
            else None
        )
        advisories_by_id = {
            advisory.advisory_id: advisory
            for advisory in [
                *get_model_advisories(card),
                *(
                    get_model_advisories(current_registry_card)
                    if current_registry_card is not None
                    else ()
                ),
            ]
        }
        return ModelListModel(
            id=card.model_id,
            hugging_face_id=card.artifact_repository,
            name=card.model_id.short(),
            description=description,
            tags=API._model_tags(card),
            storage_size_megabytes=card.storage_size.in_mb,
            supports_tensor=card.supports_tensor,
            tasks=[task.value for task in card.tasks],
            is_custom=card.is_custom,
            family=card.family,
            quantization=card.quantization,
            base_model=card.base_model,
            artifact_repository=card.artifact_repository,
            artifact_file=card.gguf_file,
            registry_card_id=card.registry_card_id,
            registry_snapshot_id=card.registry_snapshot_id,
            registry_provenance=card.registry_provenance,
            registry_architecture=card.registry_architecture,
            capability_claims=list(card.registry_capability_claims),
            engine_support=list(get_model_engine_support(card)),
            installed=installed_record is not None,
            active_installed_identity=(
                installed_record.installed_identity
                if installed_record is not None
                else None
            ),
            installed_verification=(
                installed_record.verification if installed_record is not None else None
            ),
            current_registry_identity=current_registry_identity,
            update_available=(
                installed_record is not None
                and current_registry_identity is not None
                and current_registry_identity
                != installed_record.model_card.registry_card_id
            ),
            advisories=list(advisories_by_id.values()),
            catalog_source=(
                "custom"
                if card.is_custom
                else "registry"
                if card.registry_card_id is not None
                else "bundled"
            ),
            remote_code_approval_required=remote_code_approval_required,
            remote_code_trust_identity=(
                remote_code_trust_identity(card)
                if remote_code_approval_required
                else None
            ),
            remote_code_approved_for_cluster=(
                remote_code_approval_required
                and remote_code_trust_identity(card)
                in (approved_remote_code_card_ids or ())
            ),
            # Deprecated wire alias retained for older operator clients. Trust
            # is cluster-wide even though this historical field name is not.
            remote_code_approved_on_this_node=(
                remote_code_approval_required
                and remote_code_trust_identity(card)
                in (approved_remote_code_card_ids or ())
            ),
            remote_code_automatically_trusted=remote_code_is_automatically_trusted(
                card
            ),
            source_revision=card.source_revision,
            capabilities=card.capabilities,
            context_length=card.context_length,
            reasoning=ReasoningCapabilitySection.from_model_card(card),
            modalities=ModalitiesCapabilitySection.from_model_card(card),
            audio=AudioCapabilitySection.from_model_card(card),
            tooling=ToolingCapabilitySection.from_model_card(card),
            runtime=RuntimeCapabilitySection.from_model_card(card),
            resolved_capabilities=ResolvedModelCapabilities.from_profile(
                resolved_profile
            ),
        )

    @staticmethod
    def _store_installed_records(
        entries: Iterable[dict[str, object]],
    ) -> dict[ModelId, InstalledCardRecord]:
        """Validate central-store base records for the cluster model projection.

        Companion entries do not represent independently launchable model-list
        aliases. Malformed or internally mismatched records are ignored rather
        than allowing store-index corruption to manufacture installed state.

        Args:
            entries: Raw entries returned by the authoritative store server.

        Returns:
            Valid base installed-card records keyed by their exact model alias.
        """

        records: dict[ModelId, InstalledCardRecord] = {}
        for entry in entries:
            model_id_raw = entry.get("model_id")
            installed_raw = entry.get("installed_card")
            if not isinstance(model_id_raw, str) or not isinstance(installed_raw, dict):
                continue
            try:
                model_id = ModelId(model_id_raw)
                record = InstalledCardRecord.model_validate(
                    installed_raw,
                    strict=False,
                )
            except (ValidationError, ValueError):
                logger.warning(
                    "Ignoring malformed installed-card record from the model store "
                    "for alias {}",
                    model_id_raw,
                )
                continue
            if (
                record.artifact_role != "base"
                or record.artifact_model_id != model_id_raw
                or record.model_card.model_id != model_id
            ):
                logger.warning(
                    "Ignoring mismatched installed-card record from the model store "
                    "for alias {}",
                    model_id_raw,
                )
                continue
            if record.model_card.qualification_only:
                # Qualification keeps exact bytes in the central store, but the
                # temporary unsigned card is not installed catalog truth.  It
                # must not override a later signed card sharing the same alias.
                continue
            records[model_id] = record
        return records

    async def _cached_store_installed_records(
        self,
    ) -> dict[ModelId, InstalledCardRecord]:
        """Return responsive last-known installed truth from the central store.

        Model listing is a user-facing read path, so it cannot inherit the
        metadata client's 30-second timeout. A successful response, including
        an empty registry, replaces the short-lived cache. Failure or timeout
        retains the previous snapshot and lets node-local sidecars cover aliases
        that have never been observed centrally.

        Returns:
            Valid base installed-card records keyed by exact model alias.
        """

        if self._store_client is None:
            return {}
        now = time.monotonic()
        if (
            now - self._model_list_store_records_cached_at
            < _MODEL_LIST_STORE_CACHE_TTL_SECONDS
        ):
            return self._model_list_store_records_cache
        async with self._model_list_store_records_lock:
            now = time.monotonic()
            if (
                now - self._model_list_store_records_cached_at
                < _MODEL_LIST_STORE_CACHE_TTL_SECONDS
            ):
                return self._model_list_store_records_cache
            try:
                entries = await asyncio.wait_for(
                    self._store_client.fetch_registry(raise_on_error=True),
                    timeout=_MODEL_LIST_STORE_FETCH_TIMEOUT_SECONDS,
                )
            except Exception as error:
                logger.debug(
                    "Model-list store projection retained its last-known snapshot: {}",
                    error,
                )
                # Back off the user-facing path after failure as well as success;
                # otherwise every dashboard refresh would pay the timeout again.
                self._model_list_store_records_cached_at = time.monotonic()
                return self._model_list_store_records_cache
            self._model_list_store_records_cache = self._store_installed_records(
                entries
            )
            self._model_list_store_records_cached_at = time.monotonic()
            return self._model_list_store_records_cache

    @staticmethod
    def _model_list_card_visible(card: ModelCard) -> bool:
        """Apply the catalog's existing image-model visibility policy."""

        return SKULK_ENABLE_IMAGE_MODELS or not any(
            task in {ModelTask.TextToImage, ModelTask.ImageToImage}
            for task in card.tasks
        )

    async def get_models(self, status: str | None = Query(default=None)) -> ModelList:
        """Return available models with cluster-authoritative installed state.

        The central store is the canonical source for active installed
        generations. Node-local installed-card records remain the fallback when
        the store is absent, unreachable, or does not own a particular alias.

        Args:
            status: Optional legacy ``downloaded`` filter.

        Returns:
            Available catalog models and their effective installed state.
        """
        cards = await get_model_cards()
        store_installed_records = await self._cached_store_installed_records()
        cards_by_id = {card.model_id: card for card in cards}
        for model_id, record in store_installed_records.items():
            if model_id in cards_by_id or not self._model_list_card_visible(
                record.model_card
            ):
                continue
            cards.append(record.model_card)
            cards_by_id[model_id] = record.model_card

        if status == "downloaded":
            downloaded_model_ids: set[str] = set()
            for node_downloads in self.state.downloads.values():
                for dl in node_downloads:
                    if isinstance(dl, DownloadCompleted):
                        downloaded_model_ids.add(dl.shard_metadata.model_card.model_id)
            cards = [
                card
                for card in cards
                if card.model_id in downloaded_model_ids
                or card.model_id in store_installed_records
            ]

        approved_remote_code_card_ids = self._cluster_remote_code_approvals()
        entries = [
            self._model_list_entry(
                card,
                approved_remote_code_card_ids,
                store_installed_records.get(card.model_id),
            )
            for card in cards
        ]
        if self._intelligent_fabric_enabled():
            # The steward's virtual id is addressable like any chat model but
            # is fabric-managed: flagged so pickers can badge or separate it.
            entries.append(
                ModelListModel(
                    id=STEWARD_VIRTUAL_MODEL_ID,
                    name="Skulk",
                    description=(
                        "The intelligent distributed AI fabric's operator-facing "
                        "cognition. Ask Skulk about its health, models, and "
                        "diagnostics; it investigates through read-only tools "
                        "before answering and cannot change the cluster."
                    ),
                    tags=["system", "steward"],
                    tasks=["TextGeneration"],
                    system_role="steward",
                )
            )
        return ModelList(data=entries)

    async def list_remote_code_approvals(self) -> list[RemoteCodeApprovalView]:
        """List inert model-code approvals retained for API compatibility."""
        return [
            RemoteCodeApprovalView(
                card_id=card_id,
                approved_for_cluster=True,
                approved_on_this_node=True,
            )
            for card_id in sorted(self._cluster_remote_code_approvals())
        ]

    def _cluster_remote_code_approvals(self) -> frozenset[str]:
        """Read master-ordered cluster trust with a pre-bootstrap fallback."""
        if self.state.last_event_applied_idx >= 0:
            return frozenset(self.state.model_trust_approved_remote_code_identities)
        try:
            config = load_skulk_config(self._config_path)
        except Exception:
            config = self._skulk_config
        if config is None or config.model_trust is None:
            return frozenset()
        return frozenset(config.model_trust.approved_remote_code_identities)

    def _release_model_trust_decision_waiters(
        self,
        trust_identity: str,
        approved: bool,
    ) -> None:
        """Wake requests waiting for one exact indexed trust decision."""

        for waiter in self._model_trust_decision_waiters.pop(
            (trust_identity, approved),
            [],
        ):
            waiter.set()

    async def set_cluster_remote_code_approval(
        self, card_id: str, *, approved: bool
    ) -> None:
        """Submit one cluster-wide model trust decision to the elected master.

        Args:
            card_id: Exact immutable model-card trust identity.
            approved: Whether repository-code execution is allowed.

        Side effects:
            Sends a command that the elected master serializes into the indexed
            event log, then waits for this API to apply that indexed decision.
            Every node persists the resulting replicated State.
        """
        if (card_id in self._cluster_remote_code_approvals()) == approved:
            return

        waiter = anyio.Event()
        waiters = self._model_trust_decision_waiters.setdefault(
            (card_id, approved),
            [],
        )
        waiters.append(waiter)
        try:
            await self._send(
                SetModelTrustApproval(
                    trust_identity=card_id,
                    approved=approved,
                )
            )
            try:
                with anyio.fail_after(_MODEL_TRUST_DECISION_TIMEOUT_SECONDS):
                    await waiter.wait()
            except TimeoutError as exc:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "The elected master did not confirm the model trust "
                        "decision before the convergence deadline; retry."
                    ),
                ) from exc
            if (card_id in self._cluster_remote_code_approvals()) != approved:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "The cluster session changed before the model trust "
                        "decision converged; retry."
                    ),
                )
        finally:
            remaining = self._model_trust_decision_waiters.get((card_id, approved))
            if remaining is not None:
                with contextlib.suppress(ValueError):
                    remaining.remove(waiter)
                if not remaining:
                    self._model_trust_decision_waiters.pop((card_id, approved), None)

    async def approve_remote_code(
        self, card_id: str, request: Request
    ) -> RemoteCodeApprovalView:
        """Reject redundant approval for an already authorized model card.

        Args:
            card_id: Signed card ID or content-derived local card identity.
            request: Incoming request proving a local or authenticated operator.

        Returns:
            This endpoint has no successful response for current cards.

        Raises:
            HTTPException: If the caller is not an operator, the identifier is
                malformed or unknown, or the card is already authorized by its
                publication/addition boundary.

        Side effects:
            None for current cards. Legacy callers receive an explicit conflict
            instead of creating a meaningless new allow-list entry.
        """
        self._require_operator_mutation(request)
        if not re.fullmatch(r"(?:card|local)_[a-z2-7]{52}", card_id):
            raise HTTPException(status_code=422, detail="Invalid model trust identity")
        card = next(
            (
                candidate
                for candidate in await get_all_model_cards()
                if remote_code_trust_identity(candidate) == card_id
            ),
            None,
        )
        if card is None:
            raise HTTPException(status_code=404, detail="Model card not found")
        if not remote_code_execution_requires_approval(card):
            raise HTTPException(
                status_code=409,
                detail="Model card does not require explicit repository-code approval",
            )
        await self.set_cluster_remote_code_approval(card_id, approved=True)
        return RemoteCodeApprovalView(
            card_id=card_id,
            approved_for_cluster=True,
            approved_on_this_node=True,
        )

    async def revoke_remote_code(
        self, card_id: str, request: Request
    ) -> RemoteCodeApprovalView:
        """Remove an inert legacy approval from replicated cluster state.

        Args:
            card_id: Signed card ID or content-derived local card identity.
            request: Incoming request proving a local or authenticated operator.

        Returns:
            The cluster approval view showing the card as revoked.

        Raises:
            HTTPException: If the caller is not an operator or the identifier is
                malformed.

        Side effects:
            Submits cleanup of the legacy replicated value. Publication and
            explicit addition remain the active authorization boundaries.
        """
        self._require_operator_mutation(request)
        if not re.fullmatch(r"(?:card|local)_[a-z2-7]{52}", card_id):
            raise HTTPException(status_code=422, detail="Invalid model trust identity")
        await self.set_cluster_remote_code_approval(card_id, approved=False)
        return RemoteCodeApprovalView(
            card_id=card_id,
            approved_for_cluster=False,
            approved_on_this_node=False,
        )

    @staticmethod
    def _require_loopback_mutation(request: Request) -> None:
        """Reject an operator mutation not made directly through loopback."""
        client_host = request.client.host if request.client is not None else None
        forwarding_headers_present = any(
            raw_name.lower() == b"forwarded"
            or raw_name.lower().startswith(b"x-forwarded-")
            or raw_name.lower()
            in {b"x-real-ip", b"cf-connecting-ip", b"true-client-ip"}
            for raw_name, _raw_value in request.headers.raw
        )
        if not loopback_mutation_allowed(
            client_host,
            request.headers.get("origin"),
            forwarding_headers_present=forwarding_headers_present,
        ):
            raise HTTPException(
                status_code=403,
                detail="This operator mutation is available only through loopback",
            )

    _self_host_names: ClassVar[set[str]] = set()
    """Lowercased hostnames this node positively knows as its own.

    Seeded from the machine's hostname aliases on first use and extended with
    the Tailscale MagicDNS name once discovered. Consulted by the operator
    mutation guard so a dashboard opened by hostname passes, while a
    DNS-rebound attacker hostname (never one of ours) cannot.
    """

    @classmethod
    def _known_self_host_names(cls) -> set[str]:
        """Return the cached self-hostname set, seeding it on first use."""
        if not cls._self_host_names:
            from skulk.store.config import hostname_aliases

            cls._self_host_names = {
                alias.lower() for alias in hostname_aliases(socket.gethostname())
            }
            cls._self_host_names.add("localhost")
        return cls._self_host_names

    async def _prime_tailscale_self_host_name(self) -> None:
        """Record this node's MagicDNS name so hostname dashboards pass.

        Best-effort at startup: without it, a dashboard browsed via the
        MagicDNS URL falls back to the 403 (the fabric-IP and .local paths
        are unaffected), so failure here degrades rather than breaks.
        """
        try:
            from skulk.connectivity.tailscale import query_tailscale_status

            status = await query_tailscale_status()
        except Exception:  # noqa: BLE001 - absence of tailscale is normal
            return
        if status.dns_name:
            self._known_self_host_names().add(status.dns_name.lower())

    @classmethod
    def _require_operator_mutation(cls, request: Request) -> None:
        """Allow a mutation from the gateway, loopback, or a trusted-fabric peer.

        The dashboard is served on the LAN listener and browsed from other
        machines, so operator mutations accept a direct private-LAN or CGNAT
        socket peer: the cluster's standing trust posture, since such a peer
        can already join the mesh as a full member. Forwarded requests and
        public peers still require loopback or the authenticated gateway.
        """
        if request.scope.get(OPERATOR_GATEWAY_AUTHORIZED_SCOPE_KEY) is True:
            return
        client_host = request.client.host if request.client is not None else None
        forwarding_headers_present = any(
            raw_name.lower() == b"forwarded"
            or raw_name.lower().startswith(b"x-forwarded-")
            or raw_name.lower()
            in {b"x-real-ip", b"cf-connecting-ip", b"true-client-ip"}
            for raw_name, _raw_value in request.headers.raw
        )
        if trusted_fabric_mutation_allowed(
            client_host,
            request.headers.get("origin"),
            forwarding_headers_present=forwarding_headers_present,
            self_host_names=cls._known_self_host_names(),
        ):
            return
        raise HTTPException(
            status_code=403,
            detail=(
                "This operator mutation requires a direct loopback or "
                "trusted-fabric (private LAN / CGNAT) connection, or an "
                "authenticated operator-gateway credential"
            ),
        )

    @classmethod
    def _operator_mutation_allowed(cls, request: Request) -> bool:
        """Return whether a request may originate steward proposals."""
        try:
            cls._require_operator_mutation(request)
        except HTTPException:
            return False
        return True

    @classmethod
    def _require_exact_card_qualification_mutation(cls, request: Request) -> bool:
        """Authorize the narrow exact-card lifecycle used by registry qualification.

        The long-lived service credential deliberately grants less authority than
        an operator access token: only the exact-card installation and matching
        custom-card cleanup handlers call this guard. Loopback and the authenticated
        operator gateway retain their normal access.

        Returns:
            ``True`` only when the narrow service credential authorized the call;
            operator-gateway and loopback callers return ``False``.
        """
        if request.scope.get(OPERATOR_GATEWAY_AUTHORIZED_SCOPE_KEY) is True:
            return False
        configured_token = os.environ.get(_EXACT_CARD_QUALIFICATION_TOKEN_ENV, "")
        authorization = request.headers.get("authorization", "")
        scheme, separator, presented_token = authorization.partition(" ")
        credential_matches = (
            len(configured_token) >= _MINIMUM_QUALIFICATION_TOKEN_LENGTH
            and separator == " "
            and scheme.lower() == "bearer"
            and bool(presented_token.strip())
            and hmac.compare_digest(configured_token, presented_token.strip())
        )
        if credential_matches:
            return True
        cls._require_loopback_mutation(request)
        return False

    @staticmethod
    async def _describe_hf_fetch_failure(model_id: ModelId, exc: Exception) -> str:
        """Turn a Hub metadata-fetch failure into an operator-actionable detail.

        A gated or private repository surfaces here as an
        ``HfHubHTTPError`` whose raw text tells the operator to "log in",
        which is the wrong remediation for a Skulk node. Route 401/403
        through the download layer's explainer so the Add flow names the
        same concrete fixes as a failed download (configure a token, accept
        the model terms, match the accepting account to the token).

        Args:
            model_id: The repository the caller asked to add.
            exc: The exception raised while fetching Hub metadata.

        Returns:
            A human-readable failure description for the HTTP error detail.
        """
        # Imported lazily: the API module must not take a module-level
        # dependency on the download layer for one error path.
        from skulk.download.download_utils import build_auth_error_message

        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code in (401, 403):
            return await build_auth_error_message(status_code, model_id)
        return f"Failed to fetch model: {exc}"

    async def add_custom_model(
        self, payload: AddCustomModelParams, request: Request
    ) -> ModelListModel:
        """Fetch and persist a custom model card for the cluster.

        Args:
            payload: Hugging Face repository, optional quant file, and revision.
            request: Incoming request proving a local or authenticated operator.

        Returns:
            The generated custom model entry added to the cluster catalog.

        Raises:
            HTTPException: If the caller is not an operator or card generation fails.

        Side effects:
            Fetches Hub metadata, broadcasts a persistent custom-card mutation,
            and waits for that exact mutation to converge into the local catalog.
        """
        self._require_operator_mutation(request)
        # Load curated truth before generating the override. A generated card
        # is a metadata cache, not operator-authored placement policy, and must
        # retain architecture safety constraints from an exact bundled match.
        bundled_card = await get_bundled_card(payload.model_id)
        try:
            card = await ModelCard.fetch_from_hf(
                payload.model_id,
                gguf_file=payload.gguf_file,
                source_revision=payload.source_revision,
            )
            card = preserve_generated_card_constraints(card, bundled_card)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=await self._describe_hf_fetch_failure(payload.model_id, exc),
            ) from exc

        mutation = AddCustomModelCard(model_card=card)
        await self.command_sender.send(
            ForwarderCommand(origin=self._system_id, command=mutation)
        )
        await self._wait_for_exact_custom_card_convergence(card, mutation.command_id)

        return self._model_list_entry(card.model_copy(update={"is_custom": True}))

    async def add_exact_custom_model_card(
        self, payload: AddExactCustomModelCardParams, request: Request
    ) -> ModelListModel:
        """Persist an operator-supplied exact card under unsigned custom trust.

        Args:
            payload: Complete card whose immutable artifact contract is preserved.
            request: Incoming request proving a local or authenticated operator.

        Returns:
            The custom model entry added to the cluster catalog.

        Raises:
            HTTPException: If the caller is not an authorized operator.

        Side effects:
            Removes signed-registry trust claims and broadcasts a persistent
            custom-card mutation across the cluster.
        """
        qualification_service = self._require_exact_card_qualification_mutation(request)
        if payload.model_card.source_revision is None:
            raise HTTPException(
                status_code=422,
                detail="Exact-card qualification requires an immutable source revision",
            )
        try:
            payload.model_card.require_immutable_external_companions(
                context="exact-card qualification"
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        card = self._unsigned_exact_custom_card(
            payload.model_card,
            qualification_only=qualification_service,
        )
        collision = get_custom_card_storage_collision(card.model_id)
        if collision is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Exact-card alias collides with the persisted storage key "
                    f"owned by {collision.model_id}"
                ),
            )
        existing = get_card(card.model_id)
        if (
            qualification_service
            and existing is not None
            and not existing.qualification_only
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Qualification cannot replace an existing non-qualification "
                    "model card"
                ),
            )
        mutation = AddCustomModelCard(
            model_card=card,
            requires_qualification_ownership=qualification_service,
        )
        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=mutation,
            )
        )
        await self._wait_for_exact_custom_card_convergence(card, mutation.command_id)
        return self._model_list_entry(card)

    @staticmethod
    def _unsigned_exact_custom_card(
        card: ModelCard, *, qualification_only: bool
    ) -> ModelCard:
        """Normalize an exact supplied card to Skulk's unsigned custom trust.

        Args:
            card: Exact external card supplied by the operator workflow.
            qualification_only: Whether the narrow service owns its lifecycle.

        Returns:
            The complete custom card persisted and compared at master ordering.
        """
        return card.model_copy(
            update={
                "is_custom": True,
                "qualification_only": qualification_only,
                "registry_card_id": None,
                "registry_snapshot_id": None,
                "registry_provenance": None,
                "registry_architecture": None,
                "registry_artifact_format": None,
                "registry_capability_claims": (),
            }
        )

    @staticmethod
    async def _wait_for_exact_custom_card_convergence(
        card: ModelCard,
        mutation_command_id: CommandId,
        *,
        timeout_seconds: float = _EXACT_CARD_CONVERGENCE_TIMEOUT_SECONDS,
    ) -> None:
        """Wait until the exact ordered card is visible on this API node.

        Args:
            card: Unsigned exact card sent through the master ordering boundary.
            mutation_command_id: Exact command whose applied event must be observed.
            timeout_seconds: Maximum event round-trip time before failing.

        Raises:
            HTTPException: If a conflicting card wins authoritative ordering or
                the exact event does not converge before the deadline.
        """
        deadline = anyio.current_time() + timeout_seconds
        current: ModelCard | None = None
        while True:
            current = get_card(card.model_id)
            if custom_card_mutation_applied(mutation_command_id) and current == card:
                return
            if anyio.current_time() >= deadline:
                break
            await anyio.sleep(_EXACT_CARD_CONVERGENCE_POLL_SECONDS)
        if current is not None and current != card:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Exact-card mutation lost authoritative ordering to a "
                    "different model card"
                ),
            )
        raise HTTPException(
            status_code=504,
            detail="Exact-card mutation did not converge before the deadline",
        )

    async def delete_custom_model(
        self,
        model_id: ModelId,
        request: Request,
        expected: DeleteExactCustomModelCardParams | None = None,
    ) -> JSONResponse:
        """Delete an owned custom card and synchronize the exact deletion.

        Args:
            model_id: Alias whose custom card should be removed.
            request: Operator or narrow qualification-service authorization.
            expected: Exact candidate contract required for service cleanup.

        Returns:
            Confirmation after the service-owned deletion converges.

        Raises:
            HTTPException: If the caller does not own the current exact card.
        """
        qualification_service = self._require_exact_card_qualification_mutation(request)
        card = get_card(model_id)
        if card is None or not card.is_custom:
            raise HTTPException(status_code=404, detail="Custom model card not found")
        expected_card: ModelCard | None = None
        if qualification_service:
            if not card.qualification_only:
                raise HTTPException(
                    status_code=403,
                    detail=("Qualification can remove only its temporary custom cards"),
                )
            if expected is None:
                raise HTTPException(
                    status_code=422,
                    detail=("Qualification cleanup requires the exact candidate card"),
                )
            if expected.model_card.model_id != model_id:
                raise HTTPException(
                    status_code=422,
                    detail="Qualification cleanup card does not match the path alias",
                )
            expected_card = self._unsigned_exact_custom_card(
                expected.model_card,
                qualification_only=True,
            )
            if card != expected_card:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Qualification cleanup no longer owns the current exact card"
                    ),
                )

        mutation = DeleteCustomModelCard(
            model_id=model_id,
            requires_qualification_ownership=qualification_service,
            expected_qualification_card=expected_card,
        )
        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=mutation,
            )
        )
        if qualification_service:
            await self._wait_for_qualification_card_deletion(
                model_id,
                mutation.command_id,
                expected_card=expected_card,
            )

        return JSONResponse(
            {"message": "Model card deleted", "model_id": str(model_id)}
        )

    @staticmethod
    async def _wait_for_qualification_card_deletion(
        model_id: ModelId,
        mutation_command_id: CommandId,
        *,
        expected_card: ModelCard | None = None,
        timeout_seconds: float = _EXACT_CARD_CONVERGENCE_TIMEOUT_SECONDS,
    ) -> None:
        """Wait until this exact cleanup no longer exposes its temporary card.

        Args:
            model_id: Temporary model alias sent for cleanup.
            mutation_command_id: Exact delete command whose event must be applied.
            expected_card: Exact service-owned card authorized for removal.
            timeout_seconds: Maximum event round-trip time before failing.

        Raises:
            HTTPException: If a non-qualification card wins authoritative
                ordering or cleanup does not converge before the deadline.
        """
        deadline = anyio.current_time() + timeout_seconds
        current: ModelCard | None = None
        while True:
            current = get_card(model_id)
            mutation_applied = custom_card_mutation_applied(mutation_command_id)
            if mutation_applied and current != expected_card:
                return
            if (
                expected_card is not None
                and current is not None
                and current != expected_card
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Qualification-card cleanup lost authoritative ownership "
                        "to a different exact card"
                    ),
                )
            if anyio.current_time() >= deadline:
                break
            await anyio.sleep(_EXACT_CARD_CONVERGENCE_POLL_SECONDS)
        raise HTTPException(
            status_code=504,
            detail="Qualification-card cleanup did not converge before the deadline",
        )

    async def get_model_card_summary(self, model_id: str) -> HuggingFaceCardSummary:
        """Return the prose summary of one Hugging Face repository's model card.

        Args:
            model_id: Repository id in ``owner/name`` form.

        Returns:
            The extracted summary; the ``summary`` field is empty when the
            README is missing or carries no usable prose.
        """
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", model_id):
            raise HTTPException(status_code=422, detail="Invalid model id")
        try:
            summary = await to_thread.run_sync(fetch_card_summary, model_id)
        except Exception:
            summary = ""
        return HuggingFaceCardSummary(model_id=model_id, summary=summary)

    async def get_gguf_quant_options(self, model_id: str) -> GgufQuantOptions:
        """List one GGUF repository's downloadable quantizations.

        Args:
            model_id: Repository id in ``owner/name`` form.

        Returns:
            The quant inventory; empty options for a non-GGUF repository.
        """
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", model_id):
            raise HTTPException(status_code=422, detail="Invalid model id")
        try:
            options = await to_thread.run_sync(list_gguf_quant_options, model_id)
        except Exception:
            options = []
        return GgufQuantOptions(model_id=model_id, options=options)

    async def search_models(
        self,
        query: str = "",
        limit: int = Query(default=20, ge=1, le=200),
        mlx_only: bool = False,
        offset: int = Query(default=0, ge=0, le=2000),
        pipeline_tag: str | None = None,
    ) -> list[HuggingFaceSearchResult]:
        """Search Hugging Face repositories and exact GGUF filenames.

        An empty query returns trending repositories; ``offset`` skips leading
        results for "show more" paging, and ``pipeline_tag`` restricts results
        to one Hugging Face task.
        """
        return await to_thread.run_sync(
            search_hugging_face_models,
            query,
            limit,
            mlx_only,
            offset,
            pipeline_tag,
        )

    async def run(self):
        shutdown_ev = anyio.Event()

        try:
            if self._extensions is not None:
                self._extensions.run_startup_hooks(self._extension_context)
            async with self._tg as tg:
                logger.info("Starting API")
                tg.start_soon(self._apply_state)
                # Data plane (#279 Phase 2): per-command output chunks, off the
                # event log. Started once for the API's lifetime — the DATA
                # subscription is session-independent, so (unlike _apply_state)
                # it is NOT re-spawned by reset().
                tg.start_soon(self._apply_data)
                tg.start_soon(self._apply_provider_data)
                tg.start_soon(self._apply_realtime_audio_transport)
                tg.start_soon(self._apply_speech_media_transport)
                tg.start_soon(self._apply_trace_data)
                tg.start_soon(self._apply_vision_media_transport)
                tg.start_soon(self._sweep_pending_speech_media)
                tg.start_soon(self._sweep_pending_trace_data)
                tg.start_soon(self._sweep_pending_vision_media)
                tg.start_soon(self._sweep_provider_stream_receivers)
                # Steward canary (intelligent fabric): degraded-but-alive
                # detection on the hosting node; inert while the mode is off.
                tg.start_soon(self._steward_canary_loop)
                # Releases reorder gaps stuck on a chunk dropped by the
                # best-effort DATA topic (#279 Phase 2b); lifetime-scoped like
                # _apply_data. Only meaningful while the reorder buffer is on;
                # with it disabled (#279 Phase 3) no gaps are ever buffered.
                if self._reorder_buffer_enabled:
                    tg.start_soon(self._sweep_reorder_buffers)
                tg.start_soon(self._pause_on_new_election)
                tg.start_soon(self._cleanup_expired_images)
                tg.start_soon(self._prune_old_traces)
                # Opt-in field telemetry: consent-gated, fail-silent, bounded.
                tg.start_soon(self._field_telemetry.flush_loop)
                tg.start_soon(self._store_reconciliation_loop)
                print_startup_banner(self.port)
                tg.start_soon(self.run_api, shutdown_ev)
                tg.start_soon(self._prime_tailscale_self_host_name)
                if (
                    self._operator_pairing_service is not None
                    and self._operator_relay_configuration is not None
                ):
                    tg.start_soon(self.run_operator_remote_access, shutdown_ev)
                try:
                    await anyio.sleep_forever()
                finally:
                    with anyio.CancelScope(shield=True):
                        shutdown_ev.set()
        finally:
            if self._extensions is not None:
                await self._extensions.run_shutdown_hooks()
            if self._event_log is not None:
                self._event_log.close()
            self.command_sender.close()
            self.event_receiver.close()

    async def run_api(self, ev: anyio.Event):
        cfg = Config()
        cfg.bind = [f"0.0.0.0:{self.port}"]
        # nb: shared.logging needs updating if any of this changes
        cfg.accesslog = None
        cfg.errorlog = "-"
        cfg.logger_class = InterceptLogger
        cfg.websocket_max_message_size = REALTIME_WEBSOCKET_MAX_MESSAGE_BYTES
        cfg.websocket_ping_interval = 20.0
        with anyio.CancelScope(shield=True):
            await serve(
                cast(ASGIFramework, self.app),
                cfg,
                shutdown_trigger=ev.wait,
            )

    async def run_operator_api(self, ev: anyio.Event) -> None:
        """Serve the canonical API through a relay-only authenticated TLS bind.

        Args:
            ev: Shared API shutdown signal.

        Raises:
            RuntimeError: Relay configuration or pairing service is absent.
        """

        configuration = self._operator_relay_configuration
        service = self._operator_pairing_service
        if configuration is None or service is None:
            raise RuntimeError("operator relay API requires configured pairing")
        # Validate protected TLS material before opening a listener or carrier
        # lane. Hypercorn loads the same exact certificate and key paths below.
        configuration.server_ssl_context()
        cfg = Config()
        cfg.bind = [f"127.0.0.1:{configuration.operator_api_port}"]
        cfg.certfile = str(configuration.certificate_path)
        cfg.keyfile = str(configuration.private_key_path)
        cfg.accesslog = None
        cfg.errorlog = "-"
        cfg.logger_class = InterceptLogger
        cfg.websocket_max_message_size = REALTIME_WEBSOCKET_MAX_MESSAGE_BYTES
        cfg.websocket_ping_interval = 20.0
        protected_app = OperatorGatewayAuthorization(
            self.app,
            service,
        )
        with anyio.CancelScope(shield=True):
            await serve(
                cast(ASGIFramework, protected_app),
                cfg,
                shutdown_trigger=ev.wait,
            )

    async def run_operator_remote_access(self, ev: anyio.Event) -> None:
        """Supervise optional relay ingress without coupling it to the local API.

        Args:
            ev: Shared API shutdown signal.

        Side effects:
            Starts the relay-only TLS listener and outbound carrier lanes. Any
            startup or runtime failure disables remote access for this process
            while the ordinary local API continues serving.
        """

        configuration = self._operator_relay_configuration
        if configuration is None:
            return
        try:
            async with anyio.create_task_group() as operator_task_group:
                operator_gateway = OperatorGatewayConnector(configuration)
                operator_task_group.start_soon(self.run_operator_api, ev)
                operator_task_group.start_soon(operator_gateway.run)
                await ev.wait()
                operator_task_group.cancel_scope.cancel()
        except Exception:
            # Do not include exception text: relay paths and lower-level
            # transport errors can contain private deployment metadata.
            logger.warning(
                "Operator remote access is unavailable; the local API remains available"
            )

    def _maybe_compact_event_log(self) -> None:
        """Ring retention for the diagnostics event log.

        Every few thousand appends, check the active file size; past the
        budget, compact away everything but the most recent tail. The log
        only backs GET /events, so old diagnostic history is droppable even
        though payload events no longer enter it.
        """
        if self._event_log is None:
            return
        self._event_log_appends_since_retention_check += 1
        if (
            self._event_log_appends_since_retention_check
            < _API_EVENT_LOG_CHECK_INTERVAL_APPENDS
        ):
            return
        self._event_log_appends_since_retention_check = 0
        if self._event_log.active_size_bytes <= _API_EVENT_LOG_MAX_ACTIVE_BYTES:
            return
        keep_from_idx = len(self._event_log) - _API_EVENT_LOG_COMPACT_KEEP_EVENTS
        logger.info(
            "API event log exceeded "
            f"{_API_EVENT_LOG_MAX_ACTIVE_BYTES / 2**20:.0f} MiB; compacting "
            f"to the most recent {_API_EVENT_LOG_COMPACT_KEEP_EVENTS} events"
        )
        self._event_log.compact(keep_from_idx)

    async def _apply_state(self):
        with self.event_receiver as events:
            async for i_event in events:
                event = i_event.event
                previous_tasks = self.state.tasks
                if self._event_log is not None and not isinstance(
                    event, StateSnapshotHydrated
                ):
                    self._event_log.append(event)
                    self._maybe_compact_event_log()
                self.state = apply(self.state, i_event)

                if self._apply_custom_card_mutations_locally:
                    await self._apply_local_custom_card_mutation(event)

                if isinstance(
                    event, (ModelTrustApprovalChanged, StateSnapshotHydrated)
                ):
                    try:
                        self._skulk_config = persist_model_trust_config(
                            self._config_path,
                            self.state.model_trust_approved_remote_code_identities,
                        )
                    except (OSError, ValueError, ValidationError):
                        logger.exception(
                            "Failed to persist master-ordered model trust; "
                            "replicated State remains authoritative for placement"
                        )
                    if isinstance(event, ModelTrustApprovalChanged):
                        self._release_model_trust_decision_waiters(
                            event.trust_identity,
                            event.approved,
                        )

                released_task: task_types.Task | None = None
                if isinstance(event, TaskDeleted):
                    released_task = previous_tasks.get(event.task_id)
                elif isinstance(event, TaskFailed) or (
                    isinstance(event, TaskStatusUpdated)
                    and event.task_status in _TERMINAL_TASK_STATUSES
                ):
                    released_task = self.state.tasks.get(event.task_id)
                if isinstance(released_task, task_types.RealtimeAudioTranscription):
                    self._mark_realtime_task_released(released_task.command_id)

                if isinstance(
                    event,
                    (
                        InstanceCreated,
                        InstanceDeleted,
                        NodeTimedOut,
                        RunnerStatusUpdated,
                        StateSnapshotHydrated,
                    ),
                ):
                    self._sync_builtin_speech_capability()

                # Prune telemetry for timed-out nodes from the API applier too,
                # so a --no-worker node still tracks live membership (#279).
                record_membership_from_event(self._telemetry_view, event)

                if isinstance(event, TaskCreated):
                    self._dispatch_pending_speech_media(event.task)
                    self._dispatch_pending_vision_media(event.task)

                # Output chunks no longer travel the event log — they arrive on
                # the data plane (#279 Phase 2), demuxed by _apply_data.
                if isinstance(event, TaskFailed):
                    await self._terminate_command_stream(
                        event.task_id,
                        event.error_message
                        or "The instance executing this request was lost",
                    )
                if (
                    isinstance(event, TaskStatusUpdated)
                    and event.task_status == task_types.TaskStatus.Cancelled
                ):
                    # Operator-initiated instance deletion cancels in-flight
                    # tasks (get_transition_events); without a terminal chunk
                    # those requests would hang exactly like node-death ones.
                    # Client-initiated cancels already closed their queue, so
                    # this is a no-op for them.
                    await self._terminate_command_stream(
                        event.task_id,
                        "The request was cancelled because its instance was deleted",
                    )

    @staticmethod
    async def _apply_local_custom_card_mutation(event: Event) -> None:
        """Persist one indexed card mutation when this API has no local worker.

        Args:
            event: Indexed event already accepted by the API state applier.

        Side effects:
            Updates the node-local custom-card file and cache, then records the
            originating command acknowledgement only after persistence succeeds.
        """
        if isinstance(event, CustomModelCardAdded):
            try:
                await event.model_card.save_to_custom_dir()
                add_to_card_cache(event.model_card)
                if event.mutation_command_id is not None:
                    record_custom_card_mutation_applied(event.mutation_command_id)
            except Exception:
                logger.exception(
                    "Failed to save custom model card in API-only process "
                    f"(model_id={event.model_card.model_id})"
                )
        elif isinstance(event, CustomModelCardDeleted):
            try:
                await delete_custom_card(event.model_id)
                if event.mutation_command_id is not None:
                    record_custom_card_mutation_applied(event.mutation_command_id)
            except Exception:
                logger.exception(
                    "Failed to delete custom model card in API-only process "
                    f"(model_id={event.model_id})"
                )

    async def _apply_data(self) -> None:
        """Consume the data plane (#279 Phase 2): per-command output chunks.

        Output chunks stream here directly from the serving worker (DATA topic)
        rather than as ``ChunkGenerated`` events off the master, so they never
        touch the ordered event log or the master hop. Demux by ``command_id``
        into the per-command stream queues exactly as the event path did. No-op
        on a node with no DATA topic wired (tests).
        """
        if self._data_receiver is None:
            return
        if not self._reorder_buffer_enabled:
            logger.info(
                "Data-plane reorder buffer DISABLED; dispatching output chunks in "
                "arrival order (with O(1) sequence dedupe). Only safe on an "
                "ordered transport - Zenoh delivers per-publisher FIFO; gossipsub "
                "reorders, so forcing this off there (SKULK_DATA_REORDER_BUFFER=0) "
                "can garble multi-node output. #279 Phase 3."
            )
        with self._data_receiver as data_chunks:
            async for data in data_chunks:
                self._data_plane_observer.record_received()
                if self._reorder_buffer_enabled:
                    await self._reorder_and_dispatch(data)
                elif self._command_has_queue(data.command_id):
                    # Buffer off: dispatch in arrival order (the transport orders
                    # per command), but still dedupe by sequence so a same-node
                    # Zenoh self-loopback duplicate is dropped (#311 review).
                    # Late chunks for a finalized stream are already excluded by
                    # _command_has_queue above.
                    cursor = self._data_dedup_cursor.get(data.command_id, 0)
                    if data.sequence < cursor:
                        self._data_plane_observer.record_duplicate()
                        continue  # duplicate (local publish + Zenoh loopback)
                    if data.sequence > cursor:
                        self._data_plane_observer.record_out_of_order()
                        self._data_plane_observer.record_skipped_sequences(
                            data.sequence - cursor
                        )
                        await self._fail_data_stream_transport(
                            data.command_id,
                            f"DATA sequence gap: expected {cursor}, received "
                            f"{data.sequence}",
                        )
                        continue
                    self._data_dedup_cursor[data.command_id] = data.sequence + 1
                    await self._dispatch_data_frame(data)
                else:
                    self._data_plane_observer.record_late()

    async def _apply_provider_data(self) -> None:
        """Demultiplex both provider DATA directions into isolated call queues."""

        if self._provider_stream_receiver is None:
            return
        with self._provider_stream_receiver as packets:
            async for packet in packets:
                if packet.owner_node != self.node_id:
                    # The gossipsub fallback broadcasts provider DATA; only the
                    # addressed owner may consume it. Zenoh filters by key.
                    continue
                call_id = packet.frame.call_id
                if packet.frame.direction == "caller_to_provider":
                    self._apply_provider_input_frame(packet.frame)
                    continue
                state = self._provider_stream_receivers.get(call_id)
                if state is None:
                    continue
                batch = state.receiver.accept(
                    packet.frame,
                    observed_at=anyio.current_time(),
                )
                queue_failed = False
                for frame in batch.ready:
                    try:
                        state.output_sender.send_nowait(frame)
                    except WouldBlock:
                        state.transport_failure = (
                            "caller provider stream queue exceeded its bound"
                        )
                        state.cancel_provider = True
                        queue_failed = True
                        break
                    except (BrokenResourceError, ClosedResourceError):
                        state.transport_failure = (
                            "caller provider stream queue closed before delivery"
                        )
                        state.cancel_provider = True
                        queue_failed = True
                        break
                if batch.synthesized_terminal is not None:
                    self._schedule_provider_stream_cancellation(state)
                    with contextlib.suppress(
                        WouldBlock, BrokenResourceError, ClosedResourceError
                    ):
                        state.output_sender.send_nowait(batch.synthesized_terminal)
                    queue_failed = True
                if queue_failed or state.receiver.terminal is not None:
                    if queue_failed:
                        self._schedule_provider_stream_cancellation(state)
                    self._provider_stream_receivers.pop(call_id, None)
                    if state.input_stream is not None:
                        self._schedule_provider_input_close(state.input_stream)
                    state.output_sender.close()

    async def _apply_realtime_audio_transport(self) -> None:
        """Fail source commands when remote PCM transport rejects a stream."""

        if self._realtime_audio_packet_receiver is None:
            return
        with self._realtime_audio_packet_receiver as packets:
            async for packet in packets:
                if (
                    packet.target_node != self.node_id
                    or packet.kind != "transport_failed"
                    or packet.command_id
                    not in self._realtime_audio_transcription_commands
                ):
                    continue
                await self._dispatch_generation_chunk(
                    packet.command_id,
                    ErrorChunk(
                        model=ModelId("unknown"),
                        error_message=(
                            packet.error_message or "realtime audio transport failed"
                        ),
                    ),
                )
                await self._cancel_audio_transcription_command(packet.command_id)

    async def _apply_speech_media_transport(self) -> None:
        """Fail source commands when ephemeral media transport rejects input."""

        if self._speech_media_packet_receiver is None:
            return
        with self._speech_media_packet_receiver as packets:
            async for packet in packets:
                if (
                    packet.target_node != self.node_id
                    or packet.kind != "transport_failed"
                    or packet.command_id not in self._speech_media_commands
                ):
                    continue
                message = packet.error_message or "speech media transport failed"
                await self._dispatch_generation_chunk(
                    packet.command_id,
                    ErrorChunk(model=ModelId("unknown"), error_message=message),
                )
                if packet.purpose == "transcription_audio":
                    await self._cancel_audio_transcription_command(packet.command_id)
                else:
                    await self._cancel_audio_speech_command(packet.command_id)
                # Unlike a client disconnect, a transport rejection is already
                # terminal from the caller's perspective. Allow finalization to
                # notify the master after cancellation stops the runner, or the
                # command-to-task mapping remains orphaned indefinitely.
                self._cancelled_command_ids.discard(packet.command_id)

    async def _apply_trace_data(self) -> None:
        """Merge node-addressed runner traces without involving the master."""

        if self._trace_data_receiver is None:
            return
        with self._trace_data_receiver as packets:
            async for packet in packets:
                if packet.owner_node != self.node_id:
                    continue
                expected_ranks = frozenset(packet.expected_ranks)
                if packet.rank not in expected_ranks:
                    logger.warning(
                        f"Ignoring trace rank {packet.rank} outside its expected set"
                    )
                    continue
                pending = self._pending_trace_data.get(packet.task_id)
                if pending is None:
                    if len(self._pending_trace_data) >= _TRACE_DATA_PENDING_TASKS:
                        oldest_task_id = min(
                            self._pending_trace_data,
                            key=lambda task_id: self._pending_trace_data[
                                task_id
                            ].updated_at,
                        )
                        self._pending_trace_data.pop(oldest_task_id, None)
                        logger.warning(
                            "Evicted the oldest incomplete trace assembly at the "
                            "owner-side capacity limit"
                        )
                    pending = _PendingTraceData(
                        expected_ranks=expected_ranks,
                        traces_by_rank={},
                        updated_at=time.monotonic(),
                    )
                    self._pending_trace_data[packet.task_id] = pending
                elif pending.expected_ranks != expected_ranks:
                    logger.warning(
                        f"Ignoring inconsistent expected ranks for trace {packet.task_id}"
                    )
                    continue
                pending.traces_by_rank.setdefault(packet.rank, packet.traces)
                pending.updated_at = time.monotonic()
                if set(pending.traces_by_rank) >= pending.expected_ranks:
                    traces = [
                        trace
                        for rank in sorted(pending.expected_ranks)
                        for trace in pending.traces_by_rank[rank]
                    ]
                    self._pending_trace_data.pop(packet.task_id, None)
                    self._save_trace(packet.task_id, traces)

    async def _sweep_pending_trace_data(self) -> None:
        """Bound incomplete trace assemblies whose ranks never all report."""

        while True:
            await anyio.sleep(30)
            cutoff = time.monotonic() - _TRACE_DATA_PENDING_TTL_SECONDS
            for task_id in [
                task_id
                for task_id, pending in self._pending_trace_data.items()
                if pending.updated_at <= cutoff
            ]:
                self._pending_trace_data.pop(task_id, None)
                logger.warning(f"Expired incomplete trace assembly for {task_id}")

    async def _apply_vision_media_transport(self) -> None:
        """Apply worker verification or failure to the source image request."""

        if self._vision_media_packet_receiver is None:
            return
        with self._vision_media_packet_receiver as packets:
            async for packet in packets:
                if (
                    packet.target_node != self.node_id
                    or packet.command_id not in self._vision_media_commands
                ):
                    continue
                targets = self._vision_media_targets.get(packet.command_id, ())
                model = self._vision_media_models.get(packet.command_id)
                if packet.source_node not in targets or packet.model != model:
                    logger.warning(
                        "Ignoring vision media terminal from a node that does not "
                        f"own command {packet.command_id}"
                    )
                    continue
                if packet.kind == "accepted":
                    pending = self._vision_media_pending_acks.get(packet.command_id)
                    if pending is None:
                        continue
                    pending.discard(packet.source_node)
                    if not pending:
                        self._vision_media_pending_acks.pop(packet.command_id, None)
                        self._vision_media_ack_deadlines.pop(packet.command_id, None)
                        self._release_active_vision_media(packet.command_id)
                    continue
                if packet.kind == "transport_failed":
                    await self._fail_vision_media_command(
                        packet.command_id,
                        packet.model,
                        packet.error_message or "vision media transport failed",
                    )

    def _apply_provider_input_frame(self, frame: CapabilityStreamFrame) -> None:
        """Validate and queue one caller frame for its active provider handler."""

        active = self._active_capability_streams.get(frame.call_id)
        if (
            active is None
            or active.input_lifecycle is None
            or active.input_sender is None
        ):
            return
        batch = active.input_lifecycle.accept(
            frame,
            observed_at=anyio.current_time(),
        )
        if batch.synthesized_terminal is not None:
            error = batch.synthesized_terminal.error
            self._fail_active_provider_input(
                active,
                error
                or CapabilityStreamError(
                    code="transport_error",
                    message="provider input lifecycle rejected a frame",
                ),
            )
            return

        for ready in batch.ready:
            if ready.kind == "chunk":
                payload = ready.payload or {}
                try:
                    payload_bytes = len(
                        json.dumps(
                            payload,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    )
                except (
                    TypeError,
                    ValueError,
                    RecursionError,
                    OverflowError,
                ) as exc:
                    self._fail_active_provider_input(
                        active,
                        CapabilityStreamError(
                            code="invalid_frame",
                            message=f"provider input chunk is not JSON: {exc}",
                        ),
                    )
                    return
                if payload_bytes > MAX_CALL_PAYLOAD_BYTES:
                    self._fail_active_provider_input(
                        active,
                        CapabilityStreamError(
                            code="invalid_frame",
                            message="provider input chunk payload exceeds 1 MiB",
                        ),
                    )
                    return
                assert active.descriptor.input_chunk_schema is not None
                schema_error = validate_against_schema(
                    payload,
                    active.descriptor.input_chunk_schema,
                    what="input chunk",
                )
                if schema_error is not None:
                    self._fail_active_provider_input(
                        active,
                        CapabilityStreamError(
                            code="invalid_frame",
                            message=schema_error,
                        ),
                    )
                    return
            elif ready.kind == "completed" and (
                ready.payload is not None or ready.media is not None
            ):
                self._fail_active_provider_input(
                    active,
                    CapabilityStreamError(
                        code="invalid_frame",
                        message="provider input half-close carries no data",
                    ),
                )
                return

            try:
                active.input_sender.send_nowait(ready)
            except WouldBlock:
                self._fail_active_provider_input(
                    active,
                    CapabilityStreamError(
                        code="transport_error",
                        message="provider input queue exceeded its bound",
                    ),
                )
                return
            except (BrokenResourceError, ClosedResourceError):
                self._fail_active_provider_input(
                    active,
                    CapabilityStreamError(
                        code="transport_error",
                        message="provider input queue closed before delivery",
                    ),
                )
                return

            self._provider_observer.record_input(
                frame.call_id,
                ready,
                queue_depth=active.input_sender.statistics().current_buffer_used,
            )

            if ready.kind == "completed":
                active.input_sender.close()
            elif ready.kind == "cancelled":
                active.cancel_message = (
                    ready.error.message
                    if ready.error is not None
                    else "caller cancelled the provider input stream"
                )
                active.cancel_requested.set()
                active.input_sender.close()
                if active.cancel_scope is not None:
                    active.cancel_scope.cancel()
            elif ready.kind == "failed":
                self._fail_active_provider_input(
                    active,
                    ready.error
                    or CapabilityStreamError(
                        code="transport_error",
                        message="caller input direction failed",
                    ),
                )

    @staticmethod
    def _fail_active_provider_input(
        active: _ActiveProviderStream,
        error: CapabilityStreamError,
    ) -> None:
        """Fail one input direction and wake its provider without affecting peers."""

        if active.input_failure is None:
            active.input_failure = error
        if active.input_sender is not None:
            active.input_sender.close()

    async def _sweep_provider_stream_receivers(self) -> None:
        """Expire provider sequence gaps without waiting for the call deadline."""

        while True:
            await anyio.sleep(0.5)
            observed_at = anyio.current_time()
            for call_id, state in list(self._provider_stream_receivers.items()):
                batch = state.receiver.expire(observed_at=observed_at)
                terminal = batch.synthesized_terminal
                if terminal is None:
                    continue
                self._schedule_provider_stream_cancellation(state)
                with contextlib.suppress(
                    WouldBlock, BrokenResourceError, ClosedResourceError
                ):
                    state.output_sender.send_nowait(terminal)
                self._provider_stream_receivers.pop(call_id, None)
                if state.input_stream is not None:
                    self._schedule_provider_input_close(state.input_stream)
                state.output_sender.close()
            for active in list(self._active_capability_streams.values()):
                if active.input_lifecycle is None:
                    continue
                batch = active.input_lifecycle.expire(observed_at=observed_at)
                terminal = batch.synthesized_terminal
                if terminal is None:
                    continue
                self._fail_active_provider_input(
                    active,
                    terminal.error
                    or CapabilityStreamError(
                        code="transport_error",
                        message="provider input stream expired",
                    ),
                )

    def _command_has_queue(self, command_id: CommandId) -> bool:
        return (
            command_id in self._text_generation_queues
            or command_id in self._image_generation_queues
            or command_id in self._embedding_queues
            or command_id in self._audio_speech_queues
            or command_id in self._audio_transcription_queues
        )

    async def _reorder_and_dispatch(self, frame: DataChunk) -> None:
        """Reorder one command's data-plane frames by sequence, then dispatch.

        The DATA gossip topic has no total order (it replaced the master event
        ``idx`` that did), so a multi-node producer's per-token chunks can arrive
        out of order at the owning API node, which silently transposed generation
        output. We hold each command's chunks in a small per-command buffer and
        release them strictly in ``sequence`` order. Late duplicates (below the
        cursor) are dropped; if the buffer exceeds ``_MAX_CHUNK_REORDER_BUFFER``
        (a genuinely dropped sequence on the best-effort topic) we fail the
        affected stream rather than return incomplete output. State is only
        created while the command has a live
        stream queue, so a chunk arriving after the stream finalized is dropped
        without leaking a buffer (the queue and buffer are cleared together).
        """
        command_id = frame.command_id
        if not self._command_has_queue(command_id):
            self._data_plane_observer.record_late()
            return  # client gone / stream finalized: drop late chunk, no buffer
        state = self._chunk_reorder.setdefault(command_id, _ChunkReorderState())
        if frame.sequence < state.next_seq or frame.sequence in state.pending:
            self._data_plane_observer.record_duplicate()
            return  # already delivered (duplicate / late re-send)
        if frame.sequence > state.next_seq:
            self._data_plane_observer.record_out_of_order()
        state.pending[frame.sequence] = frame
        await self._drain_in_order(command_id, state)
        # Emergency size cap: if the buffer runs away behind a dropped sequence,
        # fail the affected stream rather than returning incomplete output.
        if len(state.pending) > _MAX_CHUNK_REORDER_BUFFER:
            skipped_to = min(state.pending)
            logger.warning(
                f"Data-plane reorder buffer for command {command_id} exceeded "
                f"{_MAX_CHUNK_REORDER_BUFFER}; a chunk was likely dropped on the "
                f"best-effort DATA topic. Failing at seq {state.next_seq} "
                f"before buffered seq {skipped_to}."
            )
            self._data_plane_observer.record_skipped_sequences(
                skipped_to - state.next_seq
            )
            await self._fail_data_stream_transport(
                command_id,
                f"DATA reorder window exceeded waiting for sequence {state.next_seq}",
            )
            self._chunk_reorder.pop(command_id, None)
            return
        self._mark_reorder_gap(state)

    @staticmethod
    def _mark_reorder_gap(state: "_ChunkReorderState") -> None:
        """Stamp the head-of-line gap's age, preserving it across new chunks.

        The timer must measure how long *this* gap (the one at ``next_seq``) has
        been stuck, not refresh on every chunk that arrives behind it — otherwise
        a long stream with a dropped sequence never ages to the flush window. So
        only (re)start the clock when there is no timer yet or the gap moved to a
        new ``next_seq`` (the stream made progress).
        """
        if state.pending:
            if state.gap_since is None or state.gap_at != state.next_seq:
                state.gap_since = time.monotonic()
                state.gap_at = state.next_seq
        else:
            state.gap_since = None
            state.gap_at = None

    async def _drain_in_order(
        self, command_id: CommandId, state: "_ChunkReorderState"
    ) -> None:
        """Dispatch contiguous buffered chunks starting at ``next_seq``."""
        while state.next_seq in state.pending:
            ready = state.pending.pop(state.next_seq)
            state.next_seq += 1
            await self._dispatch_data_frame(ready)

    async def _sweep_reorder_buffers(self) -> None:
        """Fail reorder gaps stuck waiting for a sequence dropped on the topic.

        A genuine mesh reorder resolves in milliseconds; a gap older than
        ``_REORDER_GAP_FLUSH_SECONDS`` means the missing sequence was dropped on
        the best-effort DATA topic. The affected command receives an explicit
        transport error and cancellation instead of hanging or returning a
        partial response. This covers the case the size cap misses when no later
        chunk arrives to trigger it (#279 Phase 2b review).
        """
        while True:
            await anyio.sleep(1.0)
            await self._flush_stale_reorder_gaps(time.monotonic())

    async def _flush_stale_reorder_gaps(self, now: float) -> None:
        """One sweep pass: fail reorder gaps older than the flush window."""
        for command_id, state in list(self._chunk_reorder.items()):
            if (
                state.pending
                and state.gap_since is not None
                and now - state.gap_since > _REORDER_GAP_FLUSH_SECONDS
            ):
                skipped_to = min(state.pending)
                logger.warning(
                    f"Data-plane reorder gap for command {command_id} unfilled "
                    f"for >{_REORDER_GAP_FLUSH_SECONDS:g}s (seq "
                    f"{state.next_seq}..{skipped_to - 1} dropped on the "
                    "best-effort DATA topic); failing the affected stream."
                )
                self._data_plane_observer.record_skipped_sequences(
                    skipped_to - state.next_seq
                )
                await self._fail_data_stream_transport(
                    command_id,
                    f"DATA sequence gap at {state.next_seq} did not resolve "
                    f"within {_REORDER_GAP_FLUSH_SECONDS:g}s",
                )
                self._chunk_reorder.pop(command_id, None)

    async def _fail_data_stream_transport(
        self,
        command_id: CommandId,
        detail: str,
    ) -> None:
        """Synthesize one terminal error for an unrecoverable DATA delivery gap."""

        if not self._command_has_queue(command_id):
            return
        self._data_plane_observer.record_transport_failure(command_id)
        logger.warning(f"Failing command {command_id} after {detail}")
        if not self._command_task_is_terminal(command_id):
            self._cancelled_command_ids.add(command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(
                        origin=self._system_id,
                        command=TaskCancelled(cancelled_command_id=command_id),
                    )
                )
        await self._dispatch_generation_chunk(
            command_id,
            ErrorChunk(
                model=ModelId("unknown"),
                error_message=f"Data-plane transport failure: {detail}",
            ),
        )
        self._close_command_queue(command_id)

    async def _dispatch_data_frame(self, frame: DataChunk) -> None:
        """Observe and route one ordered lifecycle frame for a live command."""

        self._data_plane_observer.record_dispatched(frame)
        if frame.chunk is not None:
            await self._dispatch_generation_chunk(frame.command_id, frame.chunk)
        if frame.is_terminal:
            self._close_command_queue(frame.command_id)

    def _close_command_queue(self, command_id: CommandId) -> None:
        """Close the endpoint queue after an explicit terminal lifecycle frame."""

        for queue_map in (
            self._text_generation_queues,
            self._image_generation_queues,
            self._embedding_queues,
            self._audio_speech_queues,
            self._audio_transcription_queues,
        ):
            if queue := queue_map.get(command_id):
                queue.close()

    async def _dispatch_generation_chunk(
        self, command_id: CommandId, chunk: GenerationChunk
    ) -> None:
        """Route one output chunk to its per-command stream queue by type.

        Shared by the data plane (#279 Phase 2). ``await send`` preserves the
        backpressure the event path had (a slow client throttles its producer);
        a closed receiver (client gone) drops the queue so a late chunk can't
        wedge the dispatcher.
        """
        if queue := self._image_generation_queues.get(command_id, None):
            # ErrorChunk too: a runner failure on an ImageGeneration/ImageEdits
            # command emits ErrorChunk (RunnerSupervisor._check_runner), which
            # must reach the image client instead of crashing the data loop on a
            # too-narrow assert. The queue is typed ImageChunk | ErrorChunk.
            assert isinstance(chunk, (ImageChunk, ErrorChunk))
            try:
                await queue.send(chunk)
            except (BrokenResourceError, ClosedResourceError):
                self._image_generation_queues.pop(command_id, None)
        if queue := self._text_generation_queues.get(command_id, None):
            if not isinstance(
                chunk, (TokenChunk, ErrorChunk, ToolCallChunk, PrefillProgressChunk)
            ):
                logger.warning(
                    "Dropping unsupported output chunk "
                    f"{type(chunk).__name__} for text command {command_id}"
                )
                return
            try:
                await queue.send(chunk)
            except (BrokenResourceError, ClosedResourceError):
                self._text_generation_queues.pop(command_id, None)
        if queue := self._embedding_queues.get(command_id, None):
            assert isinstance(chunk, (EmbeddingChunk, ErrorChunk))
            try:
                await queue.send(chunk)
            except (BrokenResourceError, ClosedResourceError):
                self._embedding_queues.pop(command_id, None)
        if queue := self._audio_speech_queues.get(command_id, None):
            assert isinstance(chunk, (AudioChunk, ErrorChunk))
            try:
                await queue.send(chunk)
            except (BrokenResourceError, ClosedResourceError):
                self._audio_speech_queues.pop(command_id, None)
        if queue := self._audio_transcription_queues.get(command_id, None):
            assert isinstance(chunk, (TranscriptionChunk, ErrorChunk))
            if command_id in self._realtime_audio_transcription_commands:
                try:
                    queue.send_nowait(chunk)
                except WouldBlock:
                    logger.warning(
                        "Realtime STT output exceeded its bounded API queue; "
                        f"cancelling command {command_id}"
                    )
                    queue.close()
                    self._audio_transcription_queues.pop(command_id, None)
                    await self._cancel_audio_transcription_command(command_id)
                except (BrokenResourceError, ClosedResourceError):
                    self._audio_transcription_queues.pop(command_id, None)
                return
            try:
                await queue.send(chunk)
            except (BrokenResourceError, ClosedResourceError):
                self._audio_transcription_queues.pop(command_id, None)

    def _open_stream_queue[ChunkT](
        self,
        queue_map: dict[CommandId, Sender[ChunkT | ErrorChunk]],
        command_id: CommandId,
    ) -> Receiver[ChunkT | ErrorChunk]:
        """Register a fresh per-command stream queue, draining any buffered
        terminal failure.

        A fast local TaskFailed can beat the lazily-registered stream queue,
        leaving its terminal chunk in the pending-failure buffer; every
        non-text stream family opens its queue through here so that chunk is
        delivered through the fresh queue instead of the request hanging
        (the text stream drains the same buffer inside its generator).
        """
        sender, receiver = channel[ChunkT | ErrorChunk]()
        queue_map[command_id] = sender
        if pending_failure := self._pending_stream_failures.pop(command_id, None):
            sender.send_nowait(pending_failure)
        return receiver

    async def _terminate_command_stream(
        self, task_id: task_types.TaskId, error_message: str
    ) -> None:
        """Deliver a terminal ErrorChunk for a task the master declared dead.

        Without this, a request whose instance died mid-generation never
        receives a terminal chunk and the HTTP connection hangs until the
        client's own timeout (#223). The chunk flows through the same
        per-command queue as runner output, so streaming responses emit an
        error chunk and close, and non-streaming handlers raise their
        existing finish_reason == "error" path.
        """
        task = self.state.tasks.get(task_id)
        if not isinstance(
            task,
            (
                task_types.TextGeneration,
                task_types.ImageGeneration,
                task_types.ImageEdits,
                task_types.TextEmbedding,
                task_types.SpeechSynthesis,
                task_types.AudioTranscription,
                task_types.RealtimeAudioTranscription,
            ),
        ):
            return
        # Image task params carry the wire-form model string; the others are
        # already ModelId.
        error_chunk = ErrorChunk(
            model=ModelId(task.task_params.model),
            error_message=error_message,
        )
        if isinstance(task, task_types.RealtimeAudioTranscription):
            queue = self._audio_transcription_queues.get(task.command_id)
            if queue is not None:
                try:
                    queue.send_nowait(error_chunk)
                except WouldBlock:
                    queue.close()
                    self._audio_transcription_queues.pop(task.command_id, None)
                    await self._cancel_audio_transcription_command(task.command_id)
                except (BrokenResourceError, ClosedResourceError):
                    self._audio_transcription_queues.pop(task.command_id, None)
            return
        delivered = False
        for queue_map in (
            self._text_generation_queues,
            self._image_generation_queues,
            self._embedding_queues,
            self._audio_speech_queues,
            self._audio_transcription_queues,
        ):
            if queue := queue_map.get(task.command_id):
                delivered = True
                try:
                    await queue.send(error_chunk)
                except (BrokenResourceError, ClosedResourceError):
                    queue_map.pop(task.command_id, None)
        owner = getattr(task, "owner_node", None)
        if not delivered and (owner is None or owner == self.node_id):
            # Every task family that reaches this point streams through one
            # of the queue maps above (Realtime returned earlier). Only the
            # task's owning API can ever register the command's queue, so
            # other nodes skip buffering entries no consumer will drain
            # (owner None = legacy gossip fan-out, where any API may serve).
            # No stream queue exists yet: buffer the terminal chunk for the
            # lazily-registered stream to consume at startup, instead of
            # dropping it and hanging the request. Every stream family
            # drains this buffer at its queue-registration site.
            while len(self._pending_stream_failures) >= 256:
                oldest = next(iter(self._pending_stream_failures))
                del self._pending_stream_failures[oldest]
            self._pending_stream_failures[task.command_id] = error_chunk

    def _save_trace(
        self, task_id: task_types.TaskId, trace_data: Sequence[TraceEventData]
    ) -> None:
        """Persist one completely assembled task trace on its owning API node."""

        traces = [
            TraceEvent(
                name=t.name,
                start_us=t.start_us,
                duration_us=t.duration_us,
                rank=t.rank,
                category=t.category,
                node_id=t.node_id,
                model_id=t.model_id,
                task_kind=t.task_kind,
                tags=tuple(t.tags),
                attrs=t.attrs,
            )
            for t in trace_data
        ]
        output_path = SKULK_TRACING_CACHE_DIR / f"trace_{task_id}.json"
        export_trace(traces, output_path)
        logger.debug(f"Saved assembled trace to {output_path}")

    async def _pause_on_new_election(self):
        with self.election_receiver as ems:
            async for message in ems:
                if message.clock > self.last_completed_election:
                    self.paused = True

    async def _cleanup_expired_images(self):
        """Periodically clean up expired images from the store."""
        cleanup_interval_seconds = 300  # 5 minutes
        while True:
            await anyio.sleep(cleanup_interval_seconds)
            removed = self._image_store.cleanup_expired()
            if removed > 0:
                logger.debug(f"Cleaned up {removed} expired images")

    async def _prune_old_traces(self):
        """Drop saved trace files older than the configured retention.

        Saved Chrome-trace JSON files accumulate under
        ``SKULK_TRACING_CACHE_DIR`` indefinitely otherwise. The janitor runs
        once shortly after startup and then hourly. Retention is read fresh
        each tick so SyncConfig updates take effect without an API restart.

        Setting ``tracing.retention_days`` to ``0`` (or any non-positive
        value) disables pruning.

        Pruning logic itself lives in :func:`prune_old_trace_files` so it can
        be unit-tested without instantiating the API class.
        """
        startup_delay_seconds = 60
        prune_interval_seconds = 3600
        await anyio.sleep(startup_delay_seconds)
        while True:
            try:
                retention_days = (
                    self._skulk_config.tracing.retention_days
                    if self._skulk_config is not None
                    and self._skulk_config.tracing is not None
                    else 3
                )
                removed = prune_old_trace_files(
                    SKULK_TRACING_CACHE_DIR, retention_days, time.time()
                )
                if removed > 0:
                    logger.info(
                        f"Trace janitor pruned {removed} trace(s) older than {retention_days} day(s)"
                    )
            except Exception as err:
                # Janitor failures must never crash the API loop; log and
                # try again on the next tick.
                logger.warning(f"Trace janitor error: {err}")
            await anyio.sleep(prune_interval_seconds)

    async def _send(self, command: Command):
        while self.paused:
            await self.paused_ev.wait()
        await self.command_sender.send(
            ForwarderCommand(origin=self._system_id, command=command)
        )

    async def _send_download(self, command: DownloadCommand):
        await self.download_command_sender.send(
            ForwarderDownloadCommand(origin=self._system_id, command=command)
        )

    async def start_download(
        self, payload: StartDownloadParams, request: Request
    ) -> StartDownloadResponse:
        """Start one exact catalog-authorized shard download.

        Args:
            payload: Target node and shard metadata selected for download.
            request: Incoming request proving local or authenticated operator access.

        Returns:
            The command identity after dispatch to the download plane.

        Raises:
            HTTPException: If the caller lacks operator authority, the model is
                unknown, or the embedded card differs from catalog truth.
        """
        self._require_operator_mutation(request)
        authorized_card = await self._load_authorized_model_card(
            payload.shard_metadata.model_card.model_id
        )
        if not same_authorized_model_card(
            payload.shard_metadata.model_card, authorized_card
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Download shard embeds model-card content that does not match "
                    "the authorized catalog. Refresh model truth and retry."
                ),
            )
        command = StartDownload(
            target_node_id=payload.target_node_id,
            shard_metadata=payload.shard_metadata,
        )
        await self._send_download(command)
        return StartDownloadResponse(command_id=command.command_id)

    async def delete_download(
        self, node_id: NodeId, model_id: ModelId
    ) -> DeleteDownloadResponse:
        command = DeleteDownload(
            target_node_id=node_id,
            model_id=ModelId(model_id),
        )
        await self._send_download(command)
        return DeleteDownloadResponse(command_id=command.command_id)

    async def purge_staging_caches(
        self, payload: PurgeStagingRequest
    ) -> PurgeStagingResponse:
        from skulk.shared.types.commands import PurgeStagingCache

        command = PurgeStagingCache(
            model_id=ModelId(payload.model_id) if payload.model_id else None,
        )
        await self._send_download(command)
        model_suffix = f" for model {payload.model_id}" if payload.model_id else ""
        return PurgeStagingResponse(
            command_id=command.command_id,
            message=f"Purge staging command broadcast to all nodes{model_suffix}",
        )

    def _store_models_in_use(self) -> frozenset[str]:
        """Repo-form model IDs THIS node's live shards depend on, incl. companions.

        Node-scoped on purpose: the storage view describes this node's
        staging directory, and an idle staged copy here is an eviction
        candidate even when some other node runs the same model.
        """
        in_use: set[str] = set()
        for instance in self.state.instances.values():
            if self.node_id not in instance.shard_assignments.node_to_runner:
                continue
            for shard in instance.shard_assignments.runner_to_shard.values():
                card = shard.model_card
                in_use.add(str(card.model_id))
                if card.vision and card.vision.weights_repo:
                    in_use.add(card.vision.weights_repo)
                if card.runtime is not None:
                    if card.runtime.mtp_sidecar_repo:
                        in_use.add(card.runtime.mtp_sidecar_repo)
                    if card.runtime.assistant_model_repo:
                        in_use.add(card.runtime.assistant_model_repo)
        return frozenset(in_use)

    async def get_node_storage_summary(self) -> NodeStorageSummary:
        """Compute the local node's storage breakdown.

        Directory walks run in a worker thread — staged models are few large
        files, but the event loop must not block on filesystem traversal.
        """
        staging_root: Path | None = None
        if (
            self._skulk_config is not None
            and self._skulk_config.model_store is not None
            and self._skulk_config.model_store.enabled
        ):
            staging = resolve_node_staging(
                self._skulk_config.model_store, str(self.node_id)
            )
            if staging.enabled:
                staging_root = Path(staging.node_cache_path).expanduser()

        in_use = self._store_models_in_use()
        cards = await get_all_model_cards()

        def _collect() -> NodeStorageSummary:
            staged = _inventory_installed_artifacts(
                _installed_artifact_roots(staging_root),
                cards,
                in_use,
                self._store_client.local_store_path
                if self._store_client is not None
                else None,
            )
            event_log_bytes = 0
            for file_path in SKULK_EVENT_LOG_DIR.rglob("*"):
                # Best-effort: archives rotate and files vanish mid-walk; a
                # transient stat failure must not 500 the whole summary.
                with contextlib.suppress(OSError):
                    if file_path.is_file():
                        event_log_bytes += file_path.stat().st_size
            # Report the volume that staging actually lives on (it may be a
            # different disk from the models dir — e.g. an external store
            # volume); fall back to the models dir when staging is off.
            disk_probe_path = (
                staging_root if staging_root is not None else SKULK_MODELS_DIR
            )
            try:
                disk = shutil.disk_usage(disk_probe_path)
                disk_total_bytes, disk_free_bytes = disk.total, disk.free
            except OSError:
                # Fresh node where the directory doesn't exist yet — the
                # summary is best-effort, report zeros rather than 500.
                disk_total_bytes, disk_free_bytes = 0, 0
            return NodeStorageSummary(
                node_id=self.node_id,
                staging_root=str(staging_root) if staging_root is not None else None,
                staged_models=staged,
                staged_total_bytes=sum(info.size_bytes for info in staged),
                event_log_bytes=event_log_bytes,
                disk_total_bytes=disk_total_bytes,
                disk_free_bytes=disk_free_bytes,
            )

        return await to_thread.run_sync(_collect)

    @staticmethod
    def _get_trace_path(task_id: str) -> Path:
        trace_path = SKULK_TRACING_CACHE_DIR / f"trace_{task_id}.json"
        if not trace_path.resolve().is_relative_to(SKULK_TRACING_CACHE_DIR.resolve()):
            raise HTTPException(status_code=400, detail=f"Invalid task ID: {task_id}")
        return trace_path

    def _friendly_name_for_trace_node(self, node_id: str) -> str | None:
        for known_node_id, identity in self._telemetry_view.node_identities.items():
            if str(known_node_id) != node_id:
                continue
            if identity.friendly_name and identity.friendly_name != "Unknown":
                return identity.friendly_name
            return None
        return None

    def _trace_source_node(self, node_id: str) -> TraceSourceNode:
        return TraceSourceNode(
            node_id=node_id,
            friendly_name=self._friendly_name_for_trace_node(node_id),
        )

    def _trace_source_nodes(
        self, trace_events: list[TraceEvent]
    ) -> list[TraceSourceNode]:
        node_ids = [event.node_id for event in trace_events if event.node_id]
        if not node_ids:
            node_ids = [str(self.node_id)]

        unique_node_ids: list[str] = []
        for node_id in node_ids:
            assert node_id is not None
            if node_id not in unique_node_ids:
                unique_node_ids.append(node_id)

        return [self._trace_source_node(node_id) for node_id in unique_node_ids]

    def _build_trace_response(
        self, task_id: str, trace_events: list[TraceEvent]
    ) -> TraceResponse:
        return TraceResponse(
            task_id=task_id,
            traces=[
                TraceEventResponse(
                    name=event.name,
                    start_us=event.start_us,
                    duration_us=event.duration_us,
                    rank=event.rank,
                    category=event.category,
                    node_id=event.node_id,
                    model_id=event.model_id,
                    task_kind=event.task_kind,
                    tags=list(event.tags),
                    attrs=event.attrs,
                )
                for event in trace_events
            ],
            source_nodes=self._trace_source_nodes(trace_events),
        )

    def _build_trace_stats_response(
        self, task_id: str, trace_events: list[TraceEvent]
    ) -> TraceStatsResponse:
        stats = compute_stats(trace_events)
        return TraceStatsResponse(
            task_id=task_id,
            total_wall_time_us=stats.total_wall_time_us,
            by_category={
                category: TraceCategoryStats(
                    total_us=cat_stats.total_us,
                    count=cat_stats.count,
                    min_us=cat_stats.min_us,
                    max_us=cat_stats.max_us,
                    avg_us=cat_stats.avg_us,
                )
                for category, cat_stats in stats.by_category.items()
            },
            by_rank={
                rank: TraceRankStats(
                    by_category={
                        category: TraceCategoryStats(
                            total_us=cat_stats.total_us,
                            count=cat_stats.count,
                            min_us=cat_stats.min_us,
                            max_us=cat_stats.max_us,
                            avg_us=cat_stats.avg_us,
                        )
                        for category, cat_stats in rank_stats.items()
                    }
                )
                for rank, rank_stats in stats.by_rank.items()
            },
            source_nodes=self._trace_source_nodes(trace_events),
        )

    def _build_trace_list_item(self, task_id: str, trace_path: Path) -> TraceListItem:
        stat = trace_path.stat()
        created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        trace_events = load_trace_file(trace_path)

        model_id = next(
            (event.model_id for event in trace_events if event.model_id), None
        )
        task_kind = cast(
            TraceTaskKind | None,
            next((event.task_kind for event in trace_events if event.task_kind), None),
        )
        categories = sorted(
            {event.category for event in trace_events if event.category}
        )
        tags = sorted({tag for event in trace_events for tag in event.tags})
        has_tool_activity = any("tool_call" in event.tags for event in trace_events)

        return TraceListItem(
            task_id=task_id,
            created_at=created_at,
            file_size=stat.st_size,
            model_id=model_id,
            task_kind=task_kind,
            categories=categories,
            tags=tags,
            has_tool_activity=has_tool_activity,
            source_nodes=self._trace_source_nodes(trace_events),
        )

    @staticmethod
    def _merge_trace_list_item(
        existing: TraceListItem, incoming: TraceListItem
    ) -> TraceListItem:
        source_nodes_by_id = {
            source.node_id: source
            for source in [*existing.source_nodes, *incoming.source_nodes]
        }
        return existing.model_copy(
            update={
                "created_at": max(existing.created_at, incoming.created_at),
                "file_size": max(existing.file_size, incoming.file_size),
                "model_id": existing.model_id or incoming.model_id,
                "task_kind": existing.task_kind or incoming.task_kind,
                "categories": sorted({*existing.categories, *incoming.categories}),
                "tags": sorted({*existing.tags, *incoming.tags}),
                "has_tool_activity": existing.has_tool_activity
                or incoming.has_tool_activity,
                "source_nodes": list(source_nodes_by_id.values()),
            }
        )

    async def _peer_api_url_for(self, node_id: NodeId) -> str | None:
        """Resolve one peer's API base URL, stopping at the first hit.

        The capability-call hot path must not wait for every peer probe to
        finish (a stale peer's failed probe would delay an unrelated call).
        Probe only the requested node and cancel its remaining interface probes
        as soon as one verified address answers.
        """

        ip_address = await first_reachable_ip(
            self.state.topology,
            self.node_id,
            self.state.node_network,
            node_id,
        )
        if ip_address is None:
            return None
        host = f"[{ip_address}]" if ":" in ip_address else ip_address
        return f"http://{host}:52415"

    async def _reachable_peer_api_urls(self, fail_fast: bool = False) -> dict[str, str]:
        """Return reachable peer API base URLs keyed by node ID.

        ``fail_fast`` selects the probe budget: the interactive observability
        sweep passes ``True`` so one dead advertised address costs ~2s rather
        than the patient retry budget (#558); targeted proxy paths (runner
        cancellation, per-node diagnostics) keep the patient default, where
        tolerating a slow link matters more than latency.
        """

        reachable_by_node: dict[str, str] = {}
        async for ip_address, node_id in check_reachable(
            self.state.topology,
            self.node_id,
            self.state.node_network,
            attempts=SWEEP_ATTEMPTS if fail_fast else REACHABILITY_ATTEMPTS,
            timeout_seconds=SWEEP_TIMEOUT_SECONDS if fail_fast else 5.0,
        ):
            normalized_node_id = str(node_id)
            if normalized_node_id in reachable_by_node:
                continue
            host = f"[{ip_address}]" if ":" in ip_address else ip_address
            reachable_by_node[normalized_node_id] = f"http://{host}:52415"
        return reachable_by_node

    async def _reachable_peer_trace_urls(self) -> list[str]:
        """Return reachable peer API base URLs for trace proxying."""

        return list((await self._reachable_peer_api_urls()).values())

    @staticmethod
    def _status_kind(status: object | None) -> str | None:
        """Return the concrete status model name for diagnostics."""

        if status is None:
            return None
        return status.__class__.__name__

    def _friendly_name_for_node(self, node_id: NodeId) -> str | None:
        """Return a known friendly node name for diagnostics."""

        identity = self._telemetry_view.node_identities.get(node_id)
        if identity is None:
            return None
        if identity.friendly_name and identity.friendly_name != "Unknown":
            return identity.friendly_name
        return None

    @staticmethod
    def _task_kind(task: task_types.Task) -> str:
        """Return a stable task kind name from a tagged task model."""

        return task.__class__.__name__

    @staticmethod
    def _task_command_id(task: task_types.Task) -> str | None:
        """Return a user command ID for task types that carry one."""

        if isinstance(
            task,
            (
                task_types.TextGeneration,
                task_types.ImageGeneration,
                task_types.ImageEdits,
                task_types.TextEmbedding,
                task_types.SpeechSynthesis,
                task_types.AudioTranscription,
                task_types.RealtimeAudioTranscription,
            ),
        ):
            return str(task.command_id)
        return None

    @staticmethod
    def _task_model_id(
        task: task_types.Task, default_model_id: str | None
    ) -> str | None:
        """Return the model associated with a task when available."""

        if isinstance(
            task,
            (
                task_types.TextGeneration,
                task_types.ImageGeneration,
                task_types.ImageEdits,
                task_types.TextEmbedding,
                task_types.SpeechSynthesis,
                task_types.AudioTranscription,
                task_types.RealtimeAudioTranscription,
            ),
        ):
            return str(task.task_params.model)
        if isinstance(task, task_types.DownloadModel):
            return str(task.shard_metadata.model_card.model_id)
        if isinstance(task, task_types.CreateRunner):
            return str(task.bound_instance.bound_shard.model_card.model_id)
        return default_model_id

    def _task_diagnostics(
        self,
        task: task_types.Task,
        *,
        runner_id: str | None,
        default_model_id: str | None,
    ) -> RunnerTaskDiagnostics:
        """Build a compact event-sourced task diagnostics record."""

        return RunnerTaskDiagnostics(
            task_id=str(task.task_id),
            task_kind=self._task_kind(task),
            task_status=str(task.task_status.value),
            instance_id=str(task.instance_id),
            command_id=self._task_command_id(task),
            runner_id=runner_id,
            model_id=self._task_model_id(task, default_model_id),
        )

    def _placement_diagnostics(self) -> list[InstancePlacementDiagnostics]:
        """Build placement diagnostics from event-sourced state."""

        master_node_id = self._master_node_id
        placements: list[InstancePlacementDiagnostics] = []

        for instance_id, instance in self.state.instances.items():
            shard_assignments = instance.shard_assignments
            placement_node_ids = list(shard_assignments.node_to_runner.keys())
            placement_node_id_strings = [str(node_id) for node_id in placement_node_ids]
            master_is_placement_node = (
                master_node_id in shard_assignments.node_to_runner
            )
            local_node_is_placement_node = (
                self.node_id in shard_assignments.node_to_runner
            )
            warnings: list[str] = []

            runners: list[PlacementRunnerDiagnostics] = []
            for node_id, runner_id in shard_assignments.node_to_runner.items():
                shard = shard_assignments.runner_to_shard[runner_id]
                status = self.state.runners.get(runner_id)
                task_records = [
                    self._task_diagnostics(
                        task,
                        runner_id=str(runner_id),
                        default_model_id=str(shard_assignments.model_id),
                    )
                    for task in self.state.tasks.values()
                    if task.instance_id == instance_id
                ]
                if (
                    status is not None
                    and status.__class__.__name__ == "RunnerWarmingUp"
                    and not master_is_placement_node
                ):
                    warnings.append(
                        f"Runner {runner_id} is warming up while master is outside the placement."
                    )
                runners.append(
                    PlacementRunnerDiagnostics(
                        runner_id=str(runner_id),
                        node_id=str(node_id),
                        friendly_name=self._friendly_name_for_node(node_id),
                        status_kind=self._status_kind(status),
                        device_rank=shard.device_rank,
                        world_size=shard.world_size,
                        start_layer=shard.start_layer,
                        end_layer=shard.end_layer,
                        n_layers=shard.n_layers,
                        is_local=node_id == self.node_id,
                        is_master=node_id == master_node_id,
                        tasks=task_records,
                    )
                )

            placements.append(
                InstancePlacementDiagnostics(
                    instance_id=str(instance_id),
                    model_id=str(shard_assignments.model_id),
                    master_node_id=str(master_node_id),
                    master_is_placement_node=master_is_placement_node,
                    local_node_is_placement_node=local_node_is_placement_node,
                    placement_node_ids=placement_node_id_strings,
                    runners=sorted(runners, key=lambda runner: runner.device_rank),
                    warnings=sorted(set(warnings)),
                )
            )

        return placements

    @staticmethod
    def _short_diagnostics_id(value: str) -> str:
        """Return a compact identifier for human-facing diagnostics warnings."""

        return f"{value[:8]}…{value[-4:]}" if len(value) > 12 else value

    def _augment_runner_divergence_warnings(
        self,
        placements: Sequence[InstancePlacementDiagnostics],
        supervisor_runners: Sequence[RunnerSupervisorDiagnostics],
    ) -> list[str]:
        """Add warnings when live supervisor state diverges from cluster state."""

        task_by_id = {str(task_id): task for task_id, task in self.state.tasks.items()}
        placement_by_instance = {
            placement.instance_id: placement for placement in placements
        }
        top_level_warnings: set[str] = set()
        placement_warnings_by_instance: dict[str, set[str]] = {
            placement.instance_id: set() for placement in placements
        }

        for runner in supervisor_runners:
            placement = placement_by_instance.get(runner.instance_id)
            placement_task_ids: set[str] = set()
            instance_task_ids: set[str] = set()
            if placement is not None:
                instance_task_ids = {
                    task.task_id
                    for placement_runner in placement.runners
                    for task in placement_runner.tasks
                }
                for placement_runner in placement.runners:
                    if placement_runner.runner_id == runner.runner_id:
                        placement_task_ids = {
                            task.task_id for task in placement_runner.tasks
                        }
                        break

            if (
                runner.process_alive
                and runner.in_progress_tasks
                and not instance_task_ids
            ):
                message = (
                    "Runner "
                    f"{self._short_diagnostics_id(runner.runner_id)} is still alive with "
                    f"{len(runner.in_progress_tasks)} in-progress task(s), but event-sourced "
                    f"state shows no active tasks for instance "
                    f"{self._short_diagnostics_id(runner.instance_id)}."
                )
                top_level_warnings.add(message)
                placement_warnings_by_instance.setdefault(
                    runner.instance_id, set()
                ).add(message)

            for task in runner.in_progress_tasks:
                matching_state_task = task_by_id.get(task.task_id)
                if matching_state_task is None:
                    message = (
                        "Runner "
                        f"{self._short_diagnostics_id(runner.runner_id)} still reports "
                        f"{task.task_kind}:{self._short_diagnostics_id(task.task_id)} in progress, "
                        "but cluster state no longer tracks that task."
                    )
                elif matching_state_task.task_status.value not in {
                    "Pending",
                    "Running",
                }:
                    message = (
                        "Runner "
                        f"{self._short_diagnostics_id(runner.runner_id)} still reports "
                        f"{task.task_kind}:{self._short_diagnostics_id(task.task_id)} in progress "
                        f"while cluster state says {matching_state_task.task_status.value}."
                    )
                elif placement is not None and task.task_id not in placement_task_ids:
                    message = (
                        "Runner "
                        f"{self._short_diagnostics_id(runner.runner_id)} still reports "
                        f"{task.task_kind}:{self._short_diagnostics_id(task.task_id)} in progress, "
                        "but placement diagnostics show no active task on this runner."
                    )
                else:
                    continue

                top_level_warnings.add(message)
                placement_warnings_by_instance.setdefault(
                    runner.instance_id, set()
                ).add(message)

        for placement in placements:
            extra_warnings = placement_warnings_by_instance.get(
                placement.instance_id, set()
            )
            if extra_warnings:
                placement.warnings = sorted(set(placement.warnings) | extra_warnings)

        return sorted(top_level_warnings)

    def _collect_runner_supervisor_diagnostics(
        self,
    ) -> list[RunnerSupervisorDiagnostics]:
        """Collect live runner-supervisor diagnostics if the worker provided them."""

        if self._runner_diagnostics_provider is None:
            return []
        try:
            return list(self._runner_diagnostics_provider())
        except Exception as exc:
            logger.opt(exception=exc).warning(
                "Failed to collect runner supervisor diagnostics"
            )
            return []

    @staticmethod
    def _process_role(
        process: psutil.Process,
        command: str,
        *,
        current_pid: int,
        runner_pids: set[int],
    ) -> ProcessRole:
        """Infer a Skulk-specific role for a process."""

        executable = ""
        with contextlib.suppress(psutil.Error, OSError):
            cmdline = process.cmdline()
            if cmdline:
                executable = Path(cmdline[0]).name.lower()

        command_lower = command.lower()
        if process.pid == current_pid:
            return "skulk"
        if process.pid in runner_pids:
            return "runner"
        if executable == "vector" or " vector --config " in f" {command_lower} ":
            return "vector"
        if "resource_tracker" in command_lower:
            return "python"
        if "skulk" in command_lower:
            return "skulk"
        if "spawn_main" in command_lower:
            return "runner"
        if "python" in executable:
            return "python"
        return "other"

    @staticmethod
    def _process_command(process: psutil.Process) -> str:
        """Return a safe joined process command line."""

        try:
            command = process.cmdline()
        except (psutil.Error, OSError):
            return ""
        return " ".join(command)

    def _process_diagnostics(
        self,
        process: psutil.Process,
        *,
        child_pids: set[int],
        runner_pids: set[int],
    ) -> DiagnosticsProcess | None:
        """Build diagnostics for one OS process, skipping vanished processes."""

        try:
            command = self._process_command(process)
            rss = None
            with contextlib.suppress(psutil.Error, OSError):
                rss = Memory.from_bytes(process.memory_info().rss)
            elapsed_seconds = None
            with contextlib.suppress(psutil.Error, OSError):
                elapsed_seconds = max(0.0, time.time() - process.create_time())
            status = None
            with contextlib.suppress(psutil.Error, OSError):
                status = process.status()
            cpu_percent = None
            with contextlib.suppress(psutil.Error, OSError):
                cpu_percent = process.cpu_percent(interval=None)
            memory_percent = None
            with contextlib.suppress(psutil.Error, OSError):
                memory_percent = process.memory_percent()
            parent_pid = None
            with contextlib.suppress(psutil.Error, OSError):
                parent_pid = process.ppid()

            return DiagnosticsProcess(
                pid=process.pid,
                parent_pid=parent_pid,
                role=self._process_role(
                    process,
                    command,
                    current_pid=os.getpid(),
                    runner_pids=runner_pids,
                ),
                command=command,
                status=status,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                rss=rss,
                elapsed_seconds=elapsed_seconds,
                is_child_of_skulk=process.pid in child_pids,
            )
        except psutil.NoSuchProcess:
            return None

    def _collect_process_diagnostics(
        self,
        supervisor_runners: Sequence[RunnerSupervisorDiagnostics],
    ) -> list[DiagnosticsProcess]:
        """Collect relevant Skulk, runner, and Vector process diagnostics."""

        current = psutil.Process(os.getpid())
        try:
            children = current.children(recursive=True)
        except (psutil.Error, OSError):
            children = []

        runner_pids = {
            runner.pid for runner in supervisor_runners if runner.pid is not None
        }
        child_pids = {current.pid, *(child.pid for child in children)}
        process_by_pid: dict[int, psutil.Process] = {
            current.pid: current,
            **{child.pid: child for child in children},
        }

        for process in psutil.process_iter():
            if process.pid in process_by_pid:
                continue
            command = self._process_command(process)
            if "vector" not in command.lower():
                continue
            process_by_pid[process.pid] = process

        diagnostics: list[DiagnosticsProcess] = []
        for process in sorted(process_by_pid.values(), key=lambda proc: proc.pid):
            process_diagnostics = self._process_diagnostics(
                process,
                child_pids=child_pids,
                runner_pids=runner_pids,
            )
            if process_diagnostics is not None:
                diagnostics.append(process_diagnostics)
        return diagnostics

    def _runtime_diagnostics(self) -> NodeRuntimeDiagnostics:
        """Build local runtime diagnostics from API process and state."""

        identity = self._telemetry_view.node_identities.get(self.node_id)
        logging_config = (
            self._skulk_config.logging if self._skulk_config is not None else None
        )
        master_node_id = self._master_node_id
        return NodeRuntimeDiagnostics(
            node_id=str(self.node_id),
            hostname=socket.gethostname(),
            friendly_name=self._friendly_name_for_node(self.node_id),
            is_master=master_node_id == self.node_id,
            master_node_id=str(master_node_id),
            cwd=str(Path.cwd()),
            config_path=str(self._config_path),
            config_file_exists=self._config_path.exists(),
            skulk_version=get_skulk_version(),
            skulk_commit=identity.skulk_commit if identity is not None else "Unknown",
            libp2p_namespace=preferred_env_value(
                "SKULK_LIBP2P_NAMESPACE",
            ),
            python_unbuffered=os.environ.get("PYTHONUNBUFFERED")
            in {"1", "true", "True"},
            tracing_enabled=self.state.tracing_enabled,
            structured_logging_configured=bool(
                logging_config is not None
                and logging_config.enabled
                and logging_config.ingest_url
            ),
            logging_ingest_url=logging_config.ingest_url
            if logging_config is not None and logging_config.ingest_url
            else None,
        )

    def _resource_diagnostics(self) -> NodeResourceDiagnostics:
        """Build local resource diagnostics from gathered state and psutil."""

        current_memory = None
        current_wired = None
        with contextlib.suppress(Exception):
            current_memory = MemoryUsage.from_psutil(override_memory=None)
        with contextlib.suppress(Exception):
            wired_bytes = read_wired_memory_bytes()
            current_wired = (
                Memory.from_bytes(wired_bytes) if wired_bytes is not None else None
            )

        return NodeResourceDiagnostics(
            gathered_memory=self._telemetry_view.node_memory.get(self.node_id),
            current_memory=current_memory,
            current_wired=current_wired,
            disk=self._telemetry_view.node_disk.get(self.node_id),
            system=self._telemetry_view.node_system.get(self.node_id),
            network=self.state.node_network.get(self.node_id),
        )

    # Tailnet identity changes rarely; cache the CLI probe so repeated
    # diagnostics fetches do not each spawn a tailscale subprocess.
    _TAILSCALE_DIAG_TTL_SECONDS: Final = 60.0

    async def _tailscale_diagnostics(self) -> NodeTailscaleDiagnostics | None:
        """This node's tailnet state for the diagnostics bundle, TTL-cached.

        ``query_tailscale_status`` is itself best-effort (a missing CLI,
        timeout, or unparsable output reads as ``running=False``), so the
        result is almost always a value; ``None`` marks only the unexpected
        exception path, logged at debug and never failing diagnostics.
        """

        now = time.monotonic()
        cached = self._tailscale_diag_cache
        if cached is not None and now - cached[0] < self._TAILSCALE_DIAG_TTL_SECONDS:
            return cached[1]
        result: NodeTailscaleDiagnostics | None
        try:
            ts_status = await query_tailscale_status()
            result = NodeTailscaleDiagnostics(
                running=ts_status.running,
                self_ip=ts_status.self_ip,
                hostname=ts_status.hostname,
                dns_name=ts_status.dns_name,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must not fail on a probe
            logger.debug(f"tailscale diagnostics probe failed: {exc!r}")
            result = None
        self._tailscale_diag_cache = (now, result)
        return result

    async def get_node_diagnostics(self) -> NodeDiagnostics:
        """Return local read-only diagnostics for this Skulk node."""

        from skulk.doctor.checks import run_checks
        from skulk.facts import current_node_facts

        # This node's OWN tailnet identity rides the bundle so the per-node
        # dashboard view shows the selected node's Tailscale state (the
        # standalone connectivity endpoint reports whichever node served the
        # HTTP request, which is wrong when browsing another node).
        tailscale = await self._tailscale_diagnostics()

        supervisor_runners = self._collect_runner_supervisor_diagnostics()
        placements = self._placement_diagnostics()
        resources = self._resource_diagnostics()
        warnings = {
            warning for placement in placements for warning in placement.warnings
        }
        warnings.update(
            self._augment_runner_divergence_warnings(placements, supervisor_runners)
        )
        leak = _leaked_wired_warning(resources.current_wired, supervisor_runners)
        if leak is not None:
            warnings.add(leak)
        fleet_data_transports = live_data_transports(
            live_nodes=self._live_node_timestamps(),
            node_resources=self._telemetry_view.node_resources,
        )
        if len(fleet_data_transports) > 1:
            warnings.add(
                "Fleet DATA transport mismatch: live nodes advertise "
                f"{', '.join(sorted(fleet_data_transports))}. Cross-transport "
                "inference output cannot be delivered."
            )
        if live_skulk_build_mismatch(
            live_nodes=self._live_node_timestamps(),
            node_identities=self._telemetry_view.node_identities,
        ):
            warnings.add(
                "Fleet Skulk version mismatch: live nodes report different "
                "package versions or source commits. Complete the deployment "
                "before starting new inference work."
            )
        return NodeDiagnostics(
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            runtime=self._runtime_diagnostics(),
            identity=self._telemetry_view.node_identities.get(self.node_id),
            resources=resources,
            processes=self._collect_process_diagnostics(supervisor_runners),
            supervisor_runners=supervisor_runners,
            placements=placements,
            data_plane=self._data_plane_observer.snapshot(
                self._data_plane_egress_provider()
                if self._data_plane_egress_provider is not None
                else None
            ),
            vision_media_egress=(
                self._vision_media_egress_provider()
                if self._vision_media_egress_provider is not None
                else DataPlaneEgressDiagnostics.empty()
            ),
            vision_media_ingress=self._vision_media_ingress_diagnostics(),
            provider=self._provider_observer.snapshot(
                active_unary_calls=self._active_capability_calls,
                stream_slots_in_use=len(self._active_capability_streams),
                unary_concurrency_limit=_MAX_CONCURRENT_CAPABILITY_CALLS,
                stream_concurrency_limit=_MAX_CONCURRENT_CAPABILITY_STREAMS,
            ),
            doctor=[
                DoctorCheckDiagnostics.model_validate(
                    result.model_dump(mode="json", by_alias=False)
                )
                for result in run_checks(current_node_facts())[:64]
            ],
            warnings=sorted(warnings),
            tailscale=tailscale,
        )

    async def get_telemetry_plane_diagnostics(self) -> TelemetryPlaneDiagnostics:
        """Return aggregate local telemetry admission and egress diagnostics."""

        if self._telemetry_plane_provider is None:
            return TelemetryPlaneDiagnostics.empty()
        return self._telemetry_plane_provider()

    async def get_performance_envelopes(self) -> PerformanceEnvelopeReport:
        """Return this node's observe-only performance-envelope report."""

        return self._performance_envelopes.snapshot()

    async def get_cluster_performance_envelopes(self) -> ClusterPerformanceEnvelopes:
        """Return performance envelopes gathered from every reachable member.

        The local report is always first; each reachable peer contributes its
        own, and a member with no reachable API route appears as an explicit
        failure so the fleet view never silently omits a node.
        """

        local = self._performance_envelopes.snapshot()
        nodes = [
            NodePerformanceEnvelopes(
                node_id=str(self.node_id), url=None, ok=True, report=local
            )
        ]
        peer_urls = await self._reachable_peer_api_urls(fail_fast=True)
        timeout = httpx.Timeout(timeout=10.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            for node_id, base_url in peer_urls.items():
                try:
                    response = await client.get(
                        f"{base_url}/v1/diagnostics/performance-envelopes"
                    )
                    response.raise_for_status()
                    report = PerformanceEnvelopeReport.model_validate(response.json())
                    nodes.append(
                        NodePerformanceEnvelopes(
                            node_id=node_id, url=base_url, ok=True, report=report
                        )
                    )
                except (httpx.HTTPError, ValueError) as exc:
                    nodes.append(
                        NodePerformanceEnvelopes(
                            node_id=node_id,
                            url=base_url,
                            ok=False,
                            error=f"{exc.__class__.__name__}: {exc}",
                        )
                    )
        # A topology member with no reachable API route must appear as an
        # explicit failure rather than vanish, matching get_cluster_diagnostics
        # and this endpoint's documented contract (an overlay-joined node whose
        # advertised addresses its peers cannot route otherwise has no
        # observability presence at all).
        reported = {entry.node_id for entry in nodes}
        for topology_node_id in self.state.topology.list_nodes():
            normalized = str(topology_node_id)
            if normalized in reported:
                continue
            nodes.append(
                NodePerformanceEnvelopes(
                    node_id=normalized,
                    url=None,
                    ok=False,
                    error=(
                        "no reachable API route among the node's advertised addresses"
                    ),
                )
            )
        return ClusterPerformanceEnvelopes(
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            nodes=nodes,
        )

    @staticmethod
    def _proxy_error_detail(response: httpx.Response) -> str:
        """Extract one user-facing error message from a proxied API response."""

        with contextlib.suppress(ValueError):
            payload = _coerce_json_object(cast(object, response.json()))
            detail = payload.get("detail")
            if isinstance(detail, str):
                return detail
            error = _coerce_json_object(payload.get("error"))
            message = error.get("message")
            if isinstance(message, str):
                return message
        return f"Peer request failed with status {response.status_code}"

    async def cancel_local_runner_task(
        self,
        runner_id: RunnerId,
        request: RunnerTaskCancelRequest,
    ) -> RunnerTaskCancelResponse:
        """Request cooperative cancellation for one live task on this node."""

        if self._runner_cancel_provider is None:
            raise HTTPException(
                status_code=503,
                detail="Local worker runner controls are not available on this node.",
            )

        try:
            return await self._runner_cancel_provider(runner_id, request.task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def cancel_cluster_runner_task(
        self,
        node_id: str,
        runner_id: RunnerId,
        request: RunnerTaskCancelRequest,
    ) -> RunnerTaskCancelResponse:
        """Proxy one direct runner cancellation request to the requested node."""

        if node_id == str(self.node_id):
            return await self.cancel_local_runner_task(runner_id, request)

        peer_urls = await self._reachable_peer_api_urls()
        base_url = peer_urls.get(node_id)
        if base_url is None:
            raise HTTPException(
                status_code=404,
                detail=f"Node diagnostics endpoint not reachable: {node_id}",
            )

        timeout = httpx.Timeout(timeout=10.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            try:
                response = await client.post(
                    f"{base_url}/v1/diagnostics/node/runners/{runner_id}/cancel",
                    json=request.model_dump(mode="json", by_alias=True),
                )
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to proxy runner cancellation to {node_id}: {exc}",
                ) from exc

        if response.status_code >= 400:
            raise HTTPException(
                status_code=response.status_code,
                detail=self._proxy_error_detail(response),
            )
        return RunnerTaskCancelResponse.model_validate(response.json())

    @staticmethod
    def _truncate_capture_output(value: bytes | str | None) -> str | None:
        """Decode and cap heavyweight diagnostic command output."""

        if value is None:
            return None
        text = value.decode(errors="replace") if isinstance(value, bytes) else value
        max_chars = 64_000
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... <truncated>"

    async def _run_process_sample_command(
        self,
        *,
        name: str,
        command: list[str],
        timeout_seconds: float,
    ) -> DiagnosticProcessSample:
        """Run one local process-sampling command with a strict timeout."""

        started = time.monotonic()
        try:
            completed = None
            with anyio.move_on_after(timeout_seconds) as scope:
                completed = await anyio.run_process(command, check=False)
            duration = time.monotonic() - started
            if scope.cancel_called or completed is None:
                return DiagnosticProcessSample(
                    name=name,
                    command=command,
                    ok=False,
                    duration_seconds=duration,
                    error=f"Timed out after {timeout_seconds:.1f}s",
                )
            return DiagnosticProcessSample(
                name=name,
                command=command,
                ok=completed.returncode == 0,
                exit_code=completed.returncode,
                duration_seconds=duration,
                stdout=self._truncate_capture_output(completed.stdout),
                stderr=self._truncate_capture_output(completed.stderr),
                error=None if completed.returncode == 0 else "Command exited non-zero",
            )
        except FileNotFoundError as exc:
            missing_file = str(cast(object, exc.filename))
            return DiagnosticProcessSample(
                name=name,
                command=command,
                ok=False,
                duration_seconds=time.monotonic() - started,
                error=f"Command not found: {missing_file}",
            )
        except Exception as exc:
            return DiagnosticProcessSample(
                name=name,
                command=command,
                ok=False,
                duration_seconds=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _collect_process_samples(
        self,
        pid: int,
        sample_duration_seconds: float,
    ) -> list[DiagnosticProcessSample]:
        """Collect best-effort heavyweight process samples for one PID."""

        if platform.system() != "Darwin":
            return [
                DiagnosticProcessSample(
                    name="process_samples",
                    command=[],
                    ok=False,
                    duration_seconds=0.0,
                    error="Process sampling is currently implemented for macOS only.",
                )
            ]

        requested_duration = max(1, int(round(sample_duration_seconds)))
        commands = [
            (
                "sample",
                ["sample", str(pid), str(requested_duration)],
                float(requested_duration + 5),
            ),
            ("vmmap", ["vmmap", "-summary", str(pid)], 8.0),
            ("footprint", ["footprint", "-p", str(pid)], 8.0),
        ]
        samples: list[DiagnosticProcessSample] = []
        for name, command, timeout in commands:
            executable = command[0]
            if shutil.which(executable) is None:
                samples.append(
                    DiagnosticProcessSample(
                        name=name,
                        command=command,
                        ok=False,
                        duration_seconds=0.0,
                        error=f"{executable} not found on PATH",
                    )
                )
                continue
            samples.append(
                await self._run_process_sample_command(
                    name=name,
                    command=command,
                    timeout_seconds=timeout,
                )
            )
        return samples

    @staticmethod
    def _match_capture_runner(
        diagnostics: NodeDiagnostics,
        request: DiagnosticCaptureRequest,
    ) -> RunnerSupervisorDiagnostics | None:
        """Find the requested runner or task inside one node diagnostics bundle."""

        runner_id = str(request.runner_id) if request.runner_id is not None else None
        task_id = str(request.task_id) if request.task_id is not None else None
        if runner_id is None and task_id is None:
            return None

        for runner in diagnostics.supervisor_runners:
            if runner_id is not None and runner.runner_id != runner_id:
                continue
            if task_id is not None:
                task_ids = {task.task_id for task in runner.in_progress_tasks}
                task_ids.update(runner.pending_task_ids)
                task_ids.update(runner.cancelled_task_ids)
                if task_id not in task_ids and runner.active_task_id != task_id:
                    continue
            return runner
        return None

    async def capture_local_node_diagnostics(
        self,
        request: DiagnosticCaptureRequest,
    ) -> DiagnosticCaptureResponse:
        """Collect an on-demand local diagnostics bundle for stuck-runner triage."""

        diagnostics = await self.get_node_diagnostics()
        runner = self._match_capture_runner(diagnostics, request)
        if (
            request.runner_id is not None or request.task_id is not None
        ) and runner is None:
            raise HTTPException(
                status_code=404,
                detail="No local runner matched the requested runnerId/taskId.",
            )

        warnings: list[str] = []
        process_samples: list[DiagnosticProcessSample] = []
        mlx_memory: MlxMemorySnapshot | None = None
        if runner is not None:
            mlx_memory = runner.last_mlx_memory
            if request.include_process_samples:
                if runner.pid is None:
                    warnings.append("Matched runner has no known subprocess PID.")
                elif not runner.process_alive:
                    warnings.append("Matched runner process is no longer alive.")
                else:
                    process_samples = await self._collect_process_samples(
                        runner.pid,
                        request.sample_duration_seconds,
                    )
            if mlx_memory is None:
                warnings.append(
                    "Matched runner has not reported an MLX memory snapshot yet."
                )

        return DiagnosticCaptureResponse(
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            node_id=self.node_id,
            node_diagnostics=diagnostics,
            runner=runner,
            flight_recorder=list(runner.flight_recorder) if runner is not None else [],
            mlx_memory=mlx_memory,
            process_samples=process_samples,
            warnings=warnings,
        )

    async def capture_cluster_node_diagnostics(
        self,
        node_id: str,
        request: DiagnosticCaptureRequest,
    ) -> DiagnosticCaptureResponse:
        """Return an on-demand diagnostics capture for one local or peer node."""

        if node_id == str(self.node_id):
            return await self.capture_local_node_diagnostics(request)

        peer_urls = await self._reachable_peer_api_urls()
        base_url = peer_urls.get(node_id)
        if base_url is None:
            raise HTTPException(
                status_code=404,
                detail=f"Node diagnostics endpoint not reachable: {node_id}",
            )

        timeout = httpx.Timeout(timeout=30.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            try:
                response = await client.post(
                    f"{base_url}/v1/diagnostics/node/capture",
                    json=request.model_dump(mode="json", by_alias=True),
                )
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to proxy diagnostics capture to {node_id}: {exc}",
                ) from exc

        if response.status_code >= 400:
            raise HTTPException(
                status_code=response.status_code,
                detail=self._proxy_error_detail(response),
            )
        return DiagnosticCaptureResponse.model_validate(response.json(), extra="ignore")

    async def get_cluster_diagnostics(self) -> ClusterDiagnostics:
        """Return read-only diagnostics for every topology member.

        Reachable peers carry their collected bundle; topology members with no
        reachable API route appear as explicit ``ok=false`` entries so an
        overlay-joined node keeps an observability presence (#558).
        """

        local_diagnostics = await self.get_node_diagnostics()
        nodes = [
            ClusterNodeDiagnostics(
                node_id=str(self.node_id),
                url=None,
                ok=True,
                version_status="current",
                diagnostics=local_diagnostics,
            )
        ]

        peer_urls = await self._reachable_peer_api_urls(fail_fast=True)
        timeout = httpx.Timeout(timeout=10.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            for node_id, base_url in peer_urls.items():
                try:
                    response = await client.get(f"{base_url}/v1/diagnostics/node")
                    response.raise_for_status()
                    diagnostics = parse_peer_node_diagnostics(
                        cast(object, response.json())
                    )
                    nodes.append(
                        ClusterNodeDiagnostics(
                            node_id=node_id,
                            url=base_url,
                            ok=True,
                            version_status=compare_diagnostics_builds(
                                local_diagnostics.runtime,
                                diagnostics.runtime,
                            ),
                            diagnostics=diagnostics,
                        )
                    )
                except (httpx.HTTPError, ValueError) as exc:
                    nodes.append(
                        ClusterNodeDiagnostics(
                            node_id=node_id,
                            url=base_url,
                            ok=False,
                            error=f"{exc.__class__.__name__}: {exc}",
                        )
                    )

        # Every advertised route participates in the comparison, including a
        # failed collection whose build is therefore unknown. Topology members
        # with no route remain visible below but cannot be queried by this API
        # owner and do not make the reachable-build result uncertain.
        routed_version_statuses: list[NodeDiagnosticsVersionStatus] = [
            entry.version_status for entry in nodes
        ]

        # A topology member with no reachable API route must appear as an
        # explicit failure, not vanish: an overlay-joined node (advertising
        # only addresses its peers cannot route) otherwise has no
        # observability presence at all and the gap is invisible (#558).
        reported = {entry.node_id for entry in nodes}
        for topology_node_id in self.state.topology.list_nodes():
            normalized = str(topology_node_id)
            if normalized in reported:
                continue
            nodes.append(
                ClusterNodeDiagnostics(
                    node_id=normalized,
                    url=None,
                    ok=False,
                    error=(
                        "no reachable API route among the node's advertised addresses"
                    ),
                )
            )

        return ClusterDiagnostics(
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            local_node_id=str(self.node_id),
            master_node_id=str(self._master_node_id),
            version_status=aggregate_diagnostics_version_status(
                routed_version_statuses
            ),
            nodes=nodes,
        )

    async def get_cluster_timeline(self) -> ClusterTimeline:
        """Return a cross-rank chronological view of runner activity.

        Stitches every reachable node's ``RunnerSupervisorDiagnostics`` into
        one structure: a per-runner synopsis sorted by ``(model_id, rank)``,
        and every flight-recorder entry across all ranks merged and sorted by
        ``at`` so distributed-deadlock signatures (e.g. "rank 0 entered
        pipeline_last_eval_output 27 minutes ago while rank 1 is still in
        pipeline_first_recv_like") are visible at a glance.

        Reuses ``get_cluster_diagnostics`` for fan-out so peer discovery,
        timeouts, and unreachable-peer handling stay in one place.
        """
        cluster = await self.get_cluster_diagnostics()

        runners: list[ClusterTimelineRunner] = []
        timeline_entries: list[ClusterTimelineEntry] = []
        unreachable: list[ClusterTimelineUnreachable] = []

        for node in cluster.nodes:
            if not node.ok or node.diagnostics is None:
                unreachable.append(
                    ClusterTimelineUnreachable(
                        node_id=node.node_id,
                        url=node.url,
                        error=node.error or "Node diagnostics returned no payload.",
                    )
                )
                continue
            for runner in node.diagnostics.supervisor_runners:
                runners.append(
                    ClusterTimelineRunner(
                        node_id=runner.node_id,
                        runner_id=runner.runner_id,
                        instance_id=runner.instance_id,
                        model_id=runner.model_id,
                        device_rank=runner.device_rank,
                        world_size=runner.world_size,
                        pid=runner.pid,
                        process_alive=runner.process_alive,
                        status_kind=runner.status_kind,
                        phase=runner.phase,
                        phase_detail=runner.phase_detail,
                        seconds_in_phase=runner.seconds_in_phase,
                        last_progress_at=runner.last_progress_at,
                        active_task_id=runner.active_task_id,
                        active_command_id=runner.active_command_id,
                        last_mlx_memory=runner.last_mlx_memory,
                    )
                )
                for entry in runner.flight_recorder:
                    # Lift node/rank identity onto each entry so the merged
                    # timeline reads top-to-bottom as a distributed log.
                    timeline_entries.append(
                        ClusterTimelineEntry(
                            at=entry.at,
                            node_id=runner.node_id,
                            runner_id=runner.runner_id,
                            device_rank=runner.device_rank,
                            world_size=runner.world_size,
                            phase=entry.phase,
                            event=entry.event,
                            detail=entry.detail,
                            attrs=dict(entry.attrs),
                            task_id=entry.task_id,
                            command_id=entry.command_id,
                            mlx_memory=entry.mlx_memory,
                        )
                    )

        # Stable sort by (model_id, rank) — multi-model deployments stay grouped.
        runners.sort(key=lambda r: (r.model_id, r.device_rank, r.runner_id))
        # Chronological merge across ranks. Ties broken by (node_id, rank) so the
        # ordering is deterministic when two ranks emit at the same `at` string.
        timeline_entries.sort(key=lambda e: (e.at, e.node_id, e.device_rank))

        return ClusterTimeline(
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            local_node_id=cluster.local_node_id,
            master_node_id=cluster.master_node_id,
            runners=runners,
            timeline=timeline_entries,
            unreachable_nodes=unreachable,
        )

    async def get_cluster_node_diagnostics(self, node_id: str) -> NodeDiagnostics:
        """Return diagnostics for one local or reachable peer node."""

        if node_id == str(self.node_id):
            return await self.get_node_diagnostics()

        peer_urls = await self._reachable_peer_api_urls()
        base_url = peer_urls.get(node_id)
        if base_url is None:
            raise HTTPException(
                status_code=404,
                detail=f"Node diagnostics endpoint not reachable: {node_id}",
            )

        timeout = httpx.Timeout(timeout=10.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            response = await client.get(f"{base_url}/v1/diagnostics/node")
            if response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Node diagnostics not found: {node_id}",
                )
            response.raise_for_status()
            return parse_peer_node_diagnostics(cast(object, response.json()))

    async def get_tracing_state(self) -> TracingStateResponse:
        return TracingStateResponse(enabled=self.state.tracing_enabled)

    async def update_tracing_state(
        self, payload: UpdateTracingStateRequest
    ) -> TracingStateResponse:
        await self._send(SetTracingEnabled(enabled=payload.enabled))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self.state.tracing_enabled == payload.enabled:
                break
            await anyio.sleep(0.05)
        return TracingStateResponse(enabled=self.state.tracing_enabled)

    async def list_traces(self) -> TraceListResponse:
        traces: list[TraceListItem] = []

        for trace_file in sorted(
            SKULK_TRACING_CACHE_DIR.glob("trace_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            task_id = trace_file.stem.removeprefix("trace_")
            try:
                traces.append(self._build_trace_list_item(task_id, trace_file))
            except OSError as exc:
                logger.opt(exception=exc).warning(
                    f"Failed to inspect trace file {trace_file}"
                )

        return TraceListResponse(traces=traces)

    async def list_cluster_traces(self) -> TraceListResponse:
        deduped: dict[str, TraceListItem] = {
            trace.task_id: trace for trace in (await self.list_traces()).traces
        }
        peer_urls = await self._reachable_peer_trace_urls()
        if not peer_urls:
            return TraceListResponse(
                traces=sorted(
                    deduped.values(),
                    key=lambda trace: trace.created_at,
                    reverse=True,
                )
            )

        timeout = httpx.Timeout(timeout=5.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            for base_url in peer_urls:
                try:
                    response = await client.get(f"{base_url}/v1/traces")
                    if response.status_code != 200:
                        continue
                    peer_traces = TraceListResponse.model_validate(response.json())
                except (httpx.HTTPError, ValueError) as exc:
                    logger.opt(exception=exc).debug(
                        f"Skipping peer trace index from {base_url}"
                    )
                    continue

                for trace in peer_traces.traces:
                    if trace.task_id in deduped:
                        deduped[trace.task_id] = self._merge_trace_list_item(
                            deduped[trace.task_id], trace
                        )
                    else:
                        deduped[trace.task_id] = trace

        return TraceListResponse(
            traces=sorted(
                deduped.values(),
                key=lambda trace: trace.created_at,
                reverse=True,
            )
        )

    async def get_trace(self, task_id: str) -> TraceResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        trace_events = load_trace_file(trace_path)
        return self._build_trace_response(task_id, trace_events)

    async def get_trace_stats(self, task_id: str) -> TraceStatsResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        trace_events = load_trace_file(trace_path)
        return self._build_trace_stats_response(task_id, trace_events)

    async def get_trace_raw(self, task_id: str) -> FileResponse:
        trace_path = self._get_trace_path(task_id)

        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

        return FileResponse(
            path=trace_path,
            media_type="application/json",
            filename=f"trace_{task_id}.json",
        )

    async def get_cluster_trace(self, task_id: str) -> TraceResponse:
        try:
            return await self.get_trace(task_id)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise

        peer_urls = await self._reachable_peer_trace_urls()
        timeout = httpx.Timeout(timeout=10.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            for base_url in peer_urls:
                try:
                    response = await client.get(f"{base_url}/v1/traces/{task_id}")
                except httpx.HTTPError as exc:
                    logger.opt(exception=exc).debug(
                        f"Failed to proxy trace {task_id} from {base_url}"
                    )
                    continue
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                return TraceResponse.model_validate(response.json())

        raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

    async def get_cluster_trace_stats(self, task_id: str) -> TraceStatsResponse:
        try:
            return await self.get_trace_stats(task_id)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise

        peer_urls = await self._reachable_peer_trace_urls()
        timeout = httpx.Timeout(timeout=10.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            for base_url in peer_urls:
                try:
                    response = await client.get(f"{base_url}/v1/traces/{task_id}/stats")
                except httpx.HTTPError as exc:
                    logger.opt(exception=exc).debug(
                        f"Failed to proxy trace stats {task_id} from {base_url}"
                    )
                    continue
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                return TraceStatsResponse.model_validate(response.json())

        raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

    async def get_cluster_trace_raw(
        self, task_id: str
    ) -> FileResponse | StreamingResponse:
        try:
            return await self.get_trace_raw(task_id)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise

        peer_urls = await self._reachable_peer_trace_urls()
        timeout = httpx.Timeout(timeout=30.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            for base_url in peer_urls:
                try:
                    response = await client.get(f"{base_url}/v1/traces/{task_id}/raw")
                except httpx.HTTPError as exc:
                    logger.opt(exception=exc).debug(
                        f"Failed to proxy raw trace {task_id} from {base_url}"
                    )
                    continue
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                content_type = cast(
                    str,
                    response.headers.get("content-type", "application/json"),
                )
                content_disposition = cast(
                    str,
                    response.headers.get(
                        "content-disposition",
                        f'attachment; filename="trace_{task_id}.json"',
                    ),
                )
                return StreamingResponse(
                    iter([response.content]),
                    media_type=content_type,
                    headers={"content-disposition": content_disposition},
                )

        raise HTTPException(status_code=404, detail=f"Trace not found: {task_id}")

    async def delete_traces(self, request: DeleteTracesRequest) -> DeleteTracesResponse:
        deleted: list[str] = []
        not_found: list[str] = []
        for task_id in request.task_ids:
            trace_path = self._get_trace_path(task_id)
            if trace_path.exists():
                trace_path.unlink()
                deleted.append(task_id)
            else:
                not_found.append(task_id)
        return DeleteTracesResponse(deleted=deleted, not_found=not_found)

    async def get_onboarding(self) -> JSONResponse:
        return JSONResponse({"completed": ONBOARDING_COMPLETE_FILE.exists()})

    async def complete_onboarding(self) -> JSONResponse:
        ONBOARDING_COMPLETE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ONBOARDING_COMPLETE_FILE.write_text("true")
        return JSONResponse({"completed": True})

    # ------------------------------------------------------------------
    # Config & Store endpoints
    # ------------------------------------------------------------------

    def _effective_kv_cache_backend(self) -> str:
        """Return the effective KV backend from the env (or empty when unset)."""
        configured_backend = preferred_env_value(
            "SKULK_KV_CACHE_BACKEND",
            default="",
        )
        if not configured_backend:
            return DEFAULT_KV_CACHE_BACKEND

        if configured_backend not in VALID_KV_CACHE_BACKENDS:
            return DEFAULT_KV_CACHE_BACKEND
        return configured_backend

    def _current_telemetry_config(self) -> "TelemetryConfig | None":
        """Read the current telemetry consent from skulk.yaml (never raises).

        Cached for a few seconds: the enabled-check runs per generation, and
        a YAML parse on that hot path would be wasted work. Dashboard consent
        changes still apply within the TTL.
        """
        now = time.monotonic()
        if now < self._telemetry_config_cached_until:
            return self._telemetry_config_cache
        try:
            config = load_skulk_config(self._config_path)
            self._telemetry_config_cache = (
                config.telemetry if config is not None else None
            )
        except Exception:  # noqa: BLE001 - consent read must never break the API
            self._telemetry_config_cache = None
        self._telemetry_config_cached_until = now + 5.0
        return self._telemetry_config_cache

    def _telemetry_hardware_snapshot(
        self,
    ) -> dict[str, tuple[SystemPerformanceProfile | None, int | None]]:
        """Per-node (system profile, ram bytes) for hardware classification."""
        snapshot: dict[str, tuple[SystemPerformanceProfile | None, int | None]] = {}
        for node_id in self.state.last_seen:
            memory = self._telemetry_view.node_memory.get(node_id)
            snapshot[str(node_id)] = (
                self._telemetry_view.node_system.get(node_id),
                memory.ram_total.in_bytes if memory is not None else None,
            )
        return snapshot

    def _hardware_class_for_node(self, node_id: NodeId) -> str | None:
        """Return the envelope hardware class for one node, or ``None``.

        Requires BOTH the node's system profile and memory reading before
        answering. They populate independently (on Linux GPU nodes
        ``LinuxGpuMetrics`` and ``MemoryUsage`` arrive separately), so answering
        with one missing yields an incomplete class (a missing memory tier, or an
        "unknown" accelerator) that later samples would correct -- splitting the
        very envelope this feature is meant to learn. The accelerator within the
        profile stays optional (a CPU-only node reports none).
        """
        profile = self._telemetry_view.node_system.get(node_id)
        memory = self._telemetry_view.node_memory.get(node_id)
        if profile is None or memory is None:
            return None
        accelerator = profile.accelerator
        return hardware_class(
            accelerator.vendor if accelerator is not None else None,
            accelerator.name if accelerator is not None else None,
            memory.ram_total.in_bytes,
        )

    def _model_quantization(self, model_id: ModelId) -> str | None:
        """Return the card quantization for any instance serving ``model_id``.

        Quantization is model truth: identical across replicas, so any serving
        shard answers. Used by the runner-reported envelope path, where the
        serving node/backend/concurrency come from the terminal stats but the
        quantization still lives on the card. Returns ``None`` when the model is
        not currently served here.
        """
        for instance in self.state.instances.values():
            if instance.shard_assignments.model_id != model_id:
                continue
            for shard in instance.shard_assignments.runner_to_shard.values():
                return shard.model_card.quantization
        return None

    def _resolve_envelope_context(
        self, model_id: ModelId
    ) -> tuple[str, str, str] | None:
        """Resolve ``(hardware_class, backend, quantization)`` for a served model.

        The fallback path for in-process serial engines (MLX, in-process
        llama.cpp) whose runner does not report a serving node. Uses the rank-0
        shard's node for the hardware class (a multi-node instance spans hardware;
        rank 0 is the deterministic representative) and that shard's resolved
        backend and card quantization. Returns ``None`` when no instance or node
        telemetry is available, so an observation is skipped rather than recorded
        against a wrong key. Records only when EXACTLY one instance serves the
        model: with replicas, attributing to an arbitrary one could mis-key
        hardware/backend/quantization. The served engines (llama_server, vLLM)
        instead stamp their own node/backend/concurrency onto the terminal stats,
        so their observations are per-instance and replica-safe without this path.
        """
        matching = [
            candidate
            for candidate in self.state.instances.values()
            if candidate.shard_assignments.model_id == model_id
        ]
        if len(matching) != 1:
            # Zero: not served here. More than one: replicas, and an arbitrary
            # pick could corrupt the envelope key. Skip rather than mis-attribute.
            return None
        instance = matching[0]
        rank_zero: tuple[RunnerId, ShardMetadata] | None = None
        for runner_id, shard in instance.shard_assignments.runner_to_shard.items():
            if rank_zero is None or shard.device_rank < rank_zero[1].device_rank:
                rank_zero = (runner_id, shard)
        if rank_zero is None:
            return None
        rank_zero_runner, rank_zero_shard = rank_zero
        backend = rank_zero_shard.resolved_backend
        if backend is None:
            # The master has not stamped the resolved backend yet (node resources
            # had not gossiped at placement). Skip rather than collapse different
            # actual serving backends under one empty-backend key.
            return None
        quantization = rank_zero_shard.model_card.quantization
        node_id = next(
            (
                node
                for node, runner in instance.shard_assignments.node_to_runner.items()
                if runner == rank_zero_runner
            ),
            None,
        )
        if node_id is None:
            return None
        node_hardware = self._hardware_class_for_node(node_id)
        if node_hardware is None:
            # Skip until the serving node's hardware is fully known (both system
            # and memory readings), rather than record an incomplete key.
            return None
        return (node_hardware, backend, quantization)

    def _envelope_record_args(
        self,
        *,
        model_id: ModelId,
        fallback_context: tuple[str, str, str] | None,
        fallback_concurrency: int,
        serving_node: str | None,
        serving_backend: str | None,
        in_flight_at_admission: int | None,
        serving_batches: bool | None,
    ) -> tuple[str, str, str, int, bool] | None:
        """Resolve the envelope record args, preferring runner-reported context.

        Returns ``(hardware_class, backend, quantization, concurrency, batches)``
        or ``None`` when the serving context cannot be keyed (skip rather than
        mis-attribute).

        When the terminal stats carry a ``serving_node`` (the served engines
        stamp it), the observation is per-instance: hardware comes from that
        node's telemetry, the backend and batching mode from the runner, the
        quantization from the model card, and the concurrency from the runner's
        in-flight-at-admission count. This un-blinds replicas and is correct when
        several API nodes drive one instance, because the count is the serving
        process's own, not any single API node's offered load.

        Absent a runner-reported node (in-process serial engines: MLX, in-process
        llama.cpp), it falls back to the dispatch-time API context and this API
        node's offered in-flight count -- correct for serial engines, which serve
        one request at a time so the API's outstanding count is the load the
        instance actually sees.
        """
        if serving_node is not None and serving_backend is not None:
            hardware = self._hardware_class_for_node(NodeId(serving_node))
            if hardware is None:
                return None
            quantization = self._model_quantization(model_id)
            if quantization is None:
                return None
            # Runner reports its in-flight at admission; fall back to this API
            # node's offered count if the field is absent (older stamp path).
            concurrency = (
                in_flight_at_admission
                if in_flight_at_admission is not None
                else fallback_concurrency
            )
            # The runner declares whether it batches concurrent requests
            # (continuous batching); default to False (serial) when unstamped so
            # aggregate throughput is never falsely scaled upward.
            batches = serving_batches if serving_batches is not None else False
            return (hardware, serving_backend, quantization, concurrency, batches)
        if fallback_context is None:
            return None
        hardware, backend, quantization = fallback_context
        # The fallback path is for the in-process serial engines (MLX, in-process
        # llama.cpp): they never stamp and genuinely serve one request at a time,
        # so batches=False is correct. A SERVED backend (llama_server, vLLM)
        # reaching here means the stamp was MISSING -- most commonly a request
        # that errored before terminal stats (an ErrorChunk carries no stats). We
        # cannot know a served engine's batching mode without the stamp, and
        # recording batches=False for it would flip the per-key batching
        # classification (the registry stores it once per key), corrupting the
        # knee for previously-stamped successful batching samples until the next
        # success re-stamps it. Skip rather than corrupt.
        if engine_of(backend) in ("llama_server", "vllm"):
            return None
        return (hardware, backend, quantization, fallback_concurrency, False)

    def _tap_performance_envelope(
        self, model_id: ModelId, stream: AsyncIterator[_EnvelopeChunk]
    ) -> AsyncGenerator[_EnvelopeChunk, None]:
        """Record one performance-envelope observation per completed generation.

        Observe-only and fully guarded: any failure here disturbs neither the
        stream nor the response. The in-flight counter is incremented EAGERLY at
        dispatch so a request admitted during the window before this response's
        body starts iterating still sees it (a lazy, in-generator increment would
        under-count concurrent bursts and corrupt the knee). The matching
        decrement runs at most once, from the generator's ``finally`` on the
        common path AND from a ``weakref.finalize`` safety net: closing an async
        generator that never started does not run its ``finally``, so without the
        finalizer a client that disconnects before the body iterates would strand
        the counter forever. An aborted stream (no terminal chunk) is not
        recorded, matching the field-telemetry tap.
        """
        key = str(model_id)
        # Eager increment: the concurrency this and later requests observe must
        # include this one from the moment it is dispatched.
        concurrency = self._envelope_inflight.get(key, 0) + 1
        self._envelope_inflight[key] = concurrency
        started = time.monotonic()
        released = False
        # Resolve the serving context ONCE at dispatch and reuse it when
        # recording, so a mid-stream placement change (a same-model replica added
        # or removed while this response is still open) cannot drop the sample or
        # attribute it to a different instance's hardware/backend. Guarded: a
        # resolution failure must never break the response.
        try:
            envelope_context = self._resolve_envelope_context(model_id)
        except Exception:  # noqa: BLE001 - observability must not propagate
            envelope_context = None

        def _release() -> None:
            # Idempotent single decrement, called from the generator finally
            # (common path) or the finalizer (never-started path), whichever runs.
            nonlocal released
            if released:
                return
            released = True
            remaining = self._envelope_inflight.get(key, 1) - 1
            if remaining > 0:
                self._envelope_inflight[key] = remaining
            else:
                # Drop the key at zero so a long-lived API node serving many
                # distinct models does not accumulate idle entries.
                self._envelope_inflight.pop(key, None)

        async def _recording() -> AsyncGenerator[_EnvelopeChunk, None]:
            ttft_seconds: float | None = None
            decode_tps: float | None = None
            outcome: GenerationOutcome = "cancelled"
            # Runner-reported serving context, latched from the terminal stats.
            # The served engines (llama_server, vLLM) stamp the actual serving
            # node, backend, in-flight-at-admission, and batching mode onto their
            # final GenerationStats, so their observations are per-instance and
            # survive replicas and multiple API nodes. Absent (in-process serial
            # engines) the tap falls back to the dispatch-time API context below.
            serving_node: str | None = None
            serving_backend: str | None = None
            in_flight_at_admission: int | None = None
            serving_batches: bool | None = None
            try:
                async for chunk in stream:
                    if ttft_seconds is None and getattr(chunk, "text", None):
                        ttft_seconds = time.monotonic() - started
                    stats = cast("object | None", getattr(chunk, "stats", None))
                    if stats is not None:
                        tps = cast(
                            "float | None", getattr(stats, "generation_tps", None)
                        )
                        if tps is not None:
                            decode_tps = float(tps)
                        node = cast("str | None", getattr(stats, "serving_node", None))
                        if node is not None:
                            serving_node = node
                            serving_backend = cast(
                                "str | None", getattr(stats, "serving_backend", None)
                            )
                            in_flight_at_admission = cast(
                                "int | None",
                                getattr(stats, "in_flight_at_admission", None),
                            )
                            serving_batches = cast(
                                "bool | None", getattr(stats, "serving_batches", None)
                            )
                    finish = cast("str | None", getattr(chunk, "finish_reason", None))
                    if finish == "error":
                        outcome = "error"
                    elif finish is not None and outcome != "error":
                        # Latch an observed error: a later terminal chunk must not
                        # reclassify an errored generation as success (matches the
                        # field-telemetry tap).
                        outcome = "success"
                    yield chunk
            finally:
                try:
                    _release()
                    if outcome != "cancelled":
                        recorded = self._envelope_record_args(
                            model_id=model_id,
                            fallback_context=envelope_context,
                            fallback_concurrency=concurrency,
                            serving_node=serving_node,
                            serving_backend=serving_backend,
                            in_flight_at_admission=in_flight_at_admission,
                            serving_batches=serving_batches,
                        )
                        if recorded is not None:
                            hardware, backend, quantization, obs_conc, batches = (
                                recorded
                            )
                            self._performance_envelopes.record(
                                hardware_class=hardware,
                                model_id=key,
                                backend=backend,
                                quantization=quantization,
                                concurrency=obs_conc,
                                ttft_seconds=ttft_seconds,
                                decode_tps=decode_tps,
                                outcome=outcome,
                                batches=batches,
                            )
                except Exception as exc:  # noqa: BLE001 - must not propagate
                    logger.debug(f"performance-envelope record failed: {exc}")

        recording = _recording()
        # Safety net: if the response is discarded before the body ever iterates
        # (client disconnect after dispatch, before the first SSE payload), the
        # generator's finally never runs, so decrement on GC instead. Idempotent
        # with the finally, so the common path decrements exactly once. Guarded so
        # observability never breaks the response.
        try:
            weakref.finalize(recording, _release)
        except Exception as exc:  # noqa: BLE001 - observability must not propagate
            logger.debug(f"performance-envelope finalizer setup failed: {exc}")
        return recording

    async def get_telemetry_preview(self) -> JSONResponse:
        """Return consent state and the exact pending telemetry batch."""
        return JSONResponse(self._field_telemetry.preview())

    async def get_config(self) -> JSONResponse:
        if not self._config_path.exists():
            return JSONResponse(
                {
                    "config": {},
                    "configPath": str(self._config_path),
                    "fileExists": False,
                    "effective": {
                        "kv_cache_backend": self._effective_kv_cache_backend(),
                        # No config file, so no file token; a token can still come
                        # from the environment. Keep the effective shape identical
                        # to the file-present branch so clients never branch on it.
                        "has_hf_token": "HF_TOKEN" in os.environ,
                        "experimental_mode_enabled": experimental_mode_enabled(),
                    },
                }
            )
        raw = _load_yaml_object(self._config_path)
        # Remove sensitive fields — tokens/passwords managed separately
        safe_raw = dict(raw)
        has_hf_token = bool(safe_raw.pop("hf_token", None))
        return JSONResponse(
            {
                "config": safe_raw,
                "configPath": str(self._config_path),
                "fileExists": True,
                "effective": {
                    "kv_cache_backend": self._effective_kv_cache_backend(),
                    "has_hf_token": has_hf_token or "HF_TOKEN" in os.environ,
                    "experimental_mode_enabled": experimental_mode_enabled(),
                },
            }
        )

    async def update_config(self, request: Request) -> JSONResponse:
        body = await _read_request_json_object(request)
        if "config" in body:
            raw_config = body["config"]
            if not isinstance(raw_config, dict):
                raise HTTPException(
                    status_code=422,
                    detail="'config' field must be a JSON object.",
                )
            config_data = _coerce_json_object(cast(dict[object, object], raw_config))
        else:
            config_data = dict(body)
        if "model_trust" in config_data:
            self._require_operator_mutation(request)
            raise HTTPException(
                status_code=409,
                detail=(
                    "model_trust is a deprecated compatibility field and cannot "
                    "be changed through the current API"
                ),
            )
        requested_config = dict(config_data)

        def merge_local_fields(existing: dict[str, object]) -> dict[str, object]:
            """Preserve local-only fields inside the atomic config transaction."""

            updated = dict(requested_config)
            for field_name in (
                "hf_token",
                "logging",
                "experiments",
                "model_trust",
            ):
                if field_name not in updated and field_name in existing:
                    updated[field_name] = existing[field_name]
            # Normalize telemetry against the same locked snapshot that will be
            # replaced, so neither trust nor consent metadata can be stale.
            prepare_telemetry_config_update(updated, existing or None)
            SkulkConfig.model_validate(updated)
            return updated

        try:
            config_data = update_skulk_config_atomic(
                self._config_path,
                merge_local_fields,
            )
            parsed_config = SkulkConfig.model_validate(config_data)
        except (ValueError, yaml.YAMLError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        self._skulk_config = parsed_config
        # A consent change must apply immediately, not after the TTL.
        self._telemetry_config_cached_until = 0.0
        self._sync_builtin_speech_capability()
        # Broadcast to all nodes via gossipsub. hf_token rides along so a
        # token entered in any node's Settings converges onto the node that
        # actually fetches (the store host, or a worker doing an
        # allow_hf_fallback direct download). The fabric is PSK-encrypted and
        # trusted by doctrine; GET /config still never returns the token. A
        # blank token is dropped so it can never clobber a real one on peers.
        import copy

        from skulk.shared.types.commands import SyncConfig
        from skulk.store.config import normalized_hf_token

        broadcast_data = copy.deepcopy(config_data)
        if normalized_hf_token(broadcast_data.get("hf_token")) is None:
            broadcast_data.pop("hf_token", None)
        broadcast_data.pop("model_trust", None)
        broadcast_yaml = yaml.safe_dump(
            broadcast_data, default_flow_style=False, sort_keys=False
        )
        await self._send_download(SyncConfig(config_yaml=broadcast_yaml))
        # Apply inference config to env var immediately so next model launch uses it.
        # Don't overwrite if user provided the env var at launch.
        inference = _coerce_json_object(config_data.get("inference"))
        if "kv_cache_backend" in inference and not os.environ.get(
            "_SKULK_KV_BACKEND_USER_SET"
        ):
            os.environ["SKULK_KV_CACHE_BACKEND"] = str(inference["kv_cache_backend"])
        # Apply HF token immediately: replaces a config-derived value so
        # rotation converges, never an operator-supplied launch value, and
        # whitespace never lands in the environment.
        from skulk.store.config import promote_hf_token

        _ = promote_hf_token(config_data.get("hf_token"), source="settings update")
        # Apply logging config immediately
        logging_cfg_update = _coerce_json_object(config_data.get("logging"))
        if logging_cfg_update:
            from skulk.shared.logging import (
                external_log_pipe_enabled,
                set_structured_stdout,
            )

            # External-shipper mode is install-level (env var set by the
            # service wrapper) and overrides runtime sync. Operators turn
            # it off by removing the env var and restarting Skulk.
            #
            # In internal-subprocess mode (env var off), preserve the
            # legacy contract that runtime sync requires *both* `enabled`
            # and a non-empty `ingest_url` — clearing the URL at runtime
            # must still disable shipping, not silently leave the
            # already-running Vector subprocess shipping to the prior URL.
            ingest_url_str = str(logging_cfg_update.get("ingest_url", ""))
            log_on = external_log_pipe_enabled() or (
                bool(logging_cfg_update.get("enabled", False)) and bool(ingest_url_str)
            )
            set_structured_stdout(log_on, ingest_url=ingest_url_str)
        # model_store changes still require restart; inference-only changes don't
        has_store_changes = "model_store" in config_data
        return JSONResponse(
            {
                "success": True,
                "message": "Config saved and synced to cluster."
                + (
                    " KV cache backend takes effect on next model launch."
                    if inference
                    else ""
                )
                + (
                    " Restart required for model store changes."
                    if has_store_changes
                    else ""
                ),
                "requiresRestart": has_store_changes,
            }
        )

    def _model_in_use_on_node(self, model_id: str, node_id: NodeId) -> bool:
        """Return whether current runtime truth places one model on one node."""

        return any(
            str(instance.shard_assignments.model_id) == model_id
            and node_id in instance.shard_assignments.node_to_runner
            for instance in self.state.instances.values()
        )

    def _cache_inventory_projection(
        self,
    ) -> tuple[
        CacheInventoryStatus,
        dict[str, list[CachedArtifactLocation]],
        tuple[NodeId, ...],
    ]:
        """Project per-node telemetry and its freshness into cache locations."""

        expected_nodes = set(self.state.topology.list_nodes())
        expected_nodes.add(self.node_id)
        # Membership pruning makes a later rejoin a new convergence window;
        # retaining historical "seen" state would mislabel it degraded before
        # the rejoined node has had one publication interval to answer.
        self._artifact_inventory_nodes_seen.intersection_update(expected_nodes)
        now = datetime.now(tz=timezone.utc)
        now_monotonic = time.monotonic()
        self._artifact_inventory_expected_since = {
            node_id: self._artifact_inventory_expected_since.get(node_id, now_monotonic)
            for node_id in expected_nodes
        }
        usable: dict[NodeId, NodeArtifactInventory] = {}
        fresh: dict[NodeId, NodeArtifactInventory] = {}
        for node_id in sorted(expected_nodes, key=str):
            reading = self._telemetry_view.node_artifact_inventories.get(node_id)
            received_at = self._telemetry_view.node_artifact_inventory_received_at.get(
                node_id
            )
            if reading is None or received_at is None:
                continue
            usable[node_id] = reading
            self._artifact_inventory_nodes_seen.add(node_id)
            normalized_receipt = (
                received_at
                if received_at.tzinfo is not None
                else received_at.replace(tzinfo=timezone.utc)
            )
            if (
                now - normalized_receipt
            ).total_seconds() <= ARTIFACT_INVENTORY_STALE_SECONDS:
                fresh[node_id] = reading

        observed_nodes = len(usable)
        expected_count = len(expected_nodes)
        any_truncated = any(reading.truncated for reading in usable.values())
        missing_nodes = expected_nodes - usable.keys()
        has_stale_reading = usable.keys() != fresh.keys()
        first_reading_pending = any(
            node_id not in self._artifact_inventory_nodes_seen
            and now_monotonic
            - self._artifact_inventory_expected_since.get(node_id, now_monotonic)
            < ARTIFACT_INVENTORY_STALE_SECONDS
            for node_id in missing_nodes
        )
        if self._telemetry_sender is None:
            state: Literal["syncing", "current", "degraded", "unavailable"] = (
                "unavailable"
            )
        elif len(fresh) == expected_count and not any_truncated:
            state = "current"
        elif observed_nodes == 0 and first_reading_pending:
            state = "syncing"
        elif observed_nodes == 0:
            state = "unavailable"
        elif any_truncated or has_stale_reading:
            state = "degraded"
        elif first_reading_pending:
            state = "syncing"
        else:
            state = "degraded"

        locations: dict[str, list[CachedArtifactLocation]] = {}
        for node_id, reading in usable.items():
            for artifact in reading.artifacts:
                locations.setdefault(artifact.installed_identity, []).append(
                    CachedArtifactLocation(
                        node_id=str(node_id),
                        complete=artifact.manifest_complete,
                        installed_identity=artifact.installed_identity,
                        bytes=artifact.size_bytes,
                        last_use_epoch_seconds=artifact.last_used_epoch_seconds,
                        in_use=artifact.in_use,
                        location_kind="node_cache",
                    )
                )
        store_hosts = tuple(
            sorted(
                (node_id for node_id, reading in usable.items() if reading.store_host),
                key=str,
            )
        )
        return (
            CacheInventoryStatus(
                state=state,
                observed_nodes=observed_nodes,
                expected_nodes=expected_count,
                store_nodes=[str(node_id) for node_id in store_hosts],
            ),
            locations,
            store_hosts,
        )

    async def get_store_health(self) -> JSONResponse:
        if self._store_client is None:
            raise HTTPException(status_code=503, detail="Store not configured")
        health = await self._store_client.health_check()
        if health is None:
            raise HTTPException(status_code=503, detail="Store unreachable")
        return JSONResponse(
            {
                "storePath": health.store_path,
                "freeBytes": health.free_bytes,
                "totalBytes": health.total_bytes,
                "usedBytes": health.used_bytes,
            }
        )

    async def get_store_registry(self) -> StoreRegistryResponse:
        """Return canonical artifacts enriched with fleet and registry status."""

        if self._store_client is None:
            raise HTTPException(status_code=503, detail="Store not configured")
        entries = await self._store_client.fetch_registry()
        cache_inventory, cached_locations, store_hosts = (
            self._cache_inventory_projection()
        )
        cards_by_id = {str(card.model_id): card for card in await get_all_model_cards()}
        enriched: list[dict[str, object]] = []
        for raw_entry in entries:
            entry = dict(raw_entry)
            installed = entry.get("installed_card")
            identity = (
                cast("dict[str, object]", installed).get("installed_identity")
                if isinstance(installed, dict)
                else None
            )
            locations_by_node = {
                location.node_id: location
                for location in (
                    cached_locations.get(identity, [])
                    if isinstance(identity, str)
                    else []
                )
            }
            model_id = entry.get("model_id")
            total_bytes = entry.get("total_bytes")
            if (
                isinstance(identity, str)
                and isinstance(model_id, str)
                and isinstance(total_bytes, int)
            ):
                for store_host in store_hosts:
                    locations_by_node[str(store_host)] = CachedArtifactLocation(
                        node_id=str(store_host),
                        complete=True,
                        installed_identity=identity,
                        bytes=total_bytes,
                        last_use_epoch_seconds=0,
                        in_use=self._model_in_use_on_node(model_id, store_host),
                        location_kind="store_local",
                    )
            entry["cached_on_nodes"] = [
                location.model_dump(mode="json", by_alias=False)
                for _, location in sorted(locations_by_node.items())
            ]
            installed_dict = (
                cast("dict[str, object]", installed)
                if isinstance(installed, dict)
                else None
            )
            owner_model_id = (
                installed_dict.get("owner_model_id")
                if installed_dict is not None
                else None
            )
            model_alias = (
                owner_model_id
                if isinstance(owner_model_id, str)
                else entry.get("model_id")
            )
            current_card = (
                get_current_registry_card(ModelId(model_alias))
                if isinstance(model_alias, str)
                else None
            )
            if current_card is None and isinstance(model_alias, str):
                current_card = cards_by_id.get(model_alias)
            current_identity = (
                get_current_registry_card_id(ModelId(model_alias))
                if isinstance(model_alias, str)
                else None
            )
            if current_identity is None and current_card is not None:
                current_identity = current_card.registry_card_id
            installed_registry_identity: object = None
            if installed_dict is not None:
                model_card = installed_dict.get("model_card")
                if isinstance(model_card, dict):
                    installed_registry_identity = cast(
                        "dict[str, object]", model_card
                    ).get("registry_card_id")
            update_available = (
                current_identity is not None
                and installed_registry_identity is not None
                and current_identity != installed_registry_identity
            )
            entry["current_registry_identity"] = current_identity
            entry["installed_not_current"] = (
                current_identity is None or update_available
            )
            entry["update_available"] = update_available
            advisory_cards: list[ModelCard] = []
            if installed_dict is not None:
                installed_model_card = installed_dict.get("model_card")
                if isinstance(installed_model_card, dict):
                    advisory_cards.append(
                        ModelCard.model_validate(installed_model_card, strict=False)
                    )
            if current_card is not None:
                advisory_cards.append(current_card)
            entry["advisories"] = [
                advisory.model_dump(mode="json")
                for advisory in _combined_model_advisories(advisory_cards)
            ]
            entry["reconciliation_state"] = self._reconciliation_status.state
            entry["last_verified_at"] = self._reconciliation_status.last_verified_at
            enriched.append(entry)
        return StoreRegistryResponse.model_validate(
            {
                "entries": enriched,
                "cache_inventory": cache_inventory.model_dump(
                    mode="json", by_alias=False
                ),
            },
            strict=False,
        )

    def _configured_staging_root(self) -> Path | None:
        """Return this node's configured cache root when staging is enabled."""

        if (
            self._skulk_config is None
            or self._skulk_config.model_store is None
            or not self._skulk_config.model_store.enabled
        ):
            return None
        staging = resolve_node_staging(
            self._skulk_config.model_store,
            str(self.node_id),
        )
        return Path(staging.node_cache_path).expanduser() if staging.enabled else None

    async def create_artifact_export(
        self,
        payload: ArtifactExportRequest,
        request: Request,
    ) -> ArtifactExportResponse:
        """Create a target-bound capability for one complete local artifact."""

        self._require_artifact_export_target(request, payload.target_node_id)
        staging_root = self._configured_staging_root()
        cards = await get_all_model_cards()
        staged = await to_thread.run_sync(
            _inventory_installed_artifacts,
            _installed_artifact_roots(staging_root),
            cards,
        )
        selected = next(
            (
                item
                for item in staged
                if item.installed_identity == payload.installed_identity
                and item.manifest_sha256 == payload.manifest_sha256
                and item.manifest_complete
            ),
            None,
        )
        if selected is None:
            raise HTTPException(status_code=404, detail="Installed artifact not found")
        try:
            grant = await to_thread.run_sync(
                lambda: self._artifact_exports.issue(
                    Path(selected.directory),
                    manifest_sha256=payload.manifest_sha256,
                    target_node_id=payload.target_node_id,
                )
            )
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return ArtifactExportResponse(
            capability_token=grant.token,
            expires_at_epoch_seconds=grant.expires_at,
            byte_ceiling=grant.byte_ceiling,
            record=grant.record,
        )

    def _require_artifact_export_target(
        self,
        request: Request,
        target_node_id: str,
    ) -> None:
        """Require the caller socket to belong to the claimed store node."""

        client = request.client
        if client is None:
            raise HTTPException(status_code=403, detail="Missing caller identity")
        try:
            client_address = ipaddress.ip_address(client.host.split("%", 1)[0])
        except ValueError as error:
            raise HTTPException(
                status_code=403, detail="Invalid caller address"
            ) from error
        allowed_addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
        if target_node_id == str(self.node_id):
            allowed_addresses.update(
                {ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("::1")}
            )
        network = self.state.node_network.get(NodeId(target_node_id))
        if network is not None:
            for interface in network.interfaces:
                try:
                    allowed_addresses.add(
                        ipaddress.ip_address(interface.ip_address.split("%", 1)[0])
                    )
                except ValueError:
                    continue
        if client_address not in allowed_addresses:
            raise HTTPException(
                status_code=403,
                detail="Caller address does not belong to the target store node",
            )

    async def read_artifact_export(
        self,
        capability_token: str,
        relative_path: str,
        request: Request,
    ) -> StreamingResponse:
        """Serve one range-capable file authorized by an export capability."""

        target_node_id = request.headers.get("x-skulk-store-node")
        if target_node_id is None:
            raise HTTPException(status_code=403, detail="Missing target node binding")
        self._require_artifact_export_target(request, target_node_id)
        range_header = request.headers.get("range")
        try:
            authorized = self._artifact_exports.resolve(
                capability_token,
                target_node_id=target_node_id,
                relative_path=relative_path,
                range_header=range_header,
            )
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        async def stream_authorized_range() -> AsyncIterator[bytes]:
            remaining = authorized.byte_count
            with authorized.path.open("rb") as source:
                source.seek(authorized.start_offset)
                while remaining:
                    chunk = await to_thread.run_sync(
                        source.read,
                        min(1024 * 1024, remaining),
                    )
                    if not chunk:
                        raise RuntimeError("artifact export ended before its range")
                    remaining -= len(chunk)
                    yield chunk
                    # StreamingResponse resumes the generator only after the
                    # preceding ASGI send completes. Charging here preserves
                    # the unused ceiling when a connection drops mid-transfer.
                    self._artifact_exports.consume(capability_token, len(chunk))

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(authorized.byte_count),
        }
        status_code = 200
        if range_header is not None:
            status_code = 206
            headers["Content-Range"] = (
                f"bytes {authorized.start_offset}-{authorized.end_offset}/"
                f"{authorized.path.stat().st_size}"
            )
        return StreamingResponse(
            stream_authorized_range(),
            status_code=status_code,
            headers=headers,
            media_type="application/octet-stream",
        )

    async def get_store_reconciliation(self) -> ReconciliationStatus:
        """Return the latest automatic reconciliation status."""

        return self._reconciliation_status

    async def rescan_store_reconciliation(
        self,
        request: Request,
    ) -> ReconciliationStatus:
        """Run an immediate loopback-only fleet reconciliation pass."""

        self._require_loopback_mutation(request)
        return await self._run_store_reconciliation()

    async def _run_store_reconciliation(self) -> ReconciliationStatus:
        """Inventory reachable node caches and import each missing identity once."""

        if self._store_client is None or self._store_client.local_store_path is None:
            self._reconciliation_status = ReconciliationStatus(
                state="failed",
                inventory_only=True,
                failures=("This node is not the authoritative store host",),
            )
            return self._reconciliation_status
        async with self._reconciliation_lock:
            config = self._skulk_config
            assert config is not None and config.model_store is not None
            inventory_only = config.model_store.reconciliation.inventory_only
            self._reconciliation_status = ReconciliationStatus(
                state="scanning",
                inventory_only=inventory_only,
            )
            sources: dict[str, tuple[str, list[dict[str, object]]]] = {}
            local_summary = await self.get_node_storage_summary()
            sources[str(self.node_id)] = (
                f"http://127.0.0.1:{self.port}",
                [
                    cast(
                        "dict[str, object]", item.model_dump(mode="json", by_alias=True)
                    )
                    for item in local_summary.staged_models
                ],
            )
            peer_urls = await self._reachable_peer_api_urls(fail_fast=True)
            failures: list[str] = []
            timeout = httpx.Timeout(timeout=30.0, connect=3.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                for node_id, base_url in sorted(peer_urls.items()):
                    try:
                        response = await client.get(f"{base_url}/store/storage")
                        response.raise_for_status()
                        payload = cast("object", json.loads(response.text))
                        if not isinstance(payload, dict):
                            raise ValueError("storage response is not an object")
                        staged_raw = cast("dict[str, object]", payload).get(
                            "stagedModels", []
                        )
                        if not isinstance(staged_raw, list):
                            raise ValueError("stagedModels is not an array")
                        staged_items: list[dict[str, object]] = []
                        for raw_item in cast("list[object]", staged_raw):
                            if isinstance(raw_item, dict):
                                staged_items.append(cast("dict[str, object]", raw_item))
                        sources[node_id] = (base_url, staged_items)
                    except (httpx.HTTPError, ValueError) as error:
                        failures.append(f"Could not inventory node {node_id}: {error}")
                registry = await self._store_client.fetch_registry(
                    recover_installed_cards=True
                )
                existing_generations = await to_thread.run_sync(
                    _complete_canonical_generations,
                    self._store_client.local_store_path,
                    registry,
                )
                tombstone_metadata_valid = True
                reconciliation_tombstones: frozenset[str]
                try:
                    reconciliation_tombstones = await to_thread.run_sync(
                        read_reconciliation_tombstones,
                        self._store_client.local_store_path,
                    )
                except (OSError, ValueError) as error:
                    tombstone_metadata_valid = False
                    failures.append(
                        f"Could not read reconciliation deletion tombstones: {error}"
                    )
                    reconciliation_tombstones = frozenset()
                replicas: dict[
                    tuple[str, str], list[tuple[str, str, dict[str, object]]]
                ] = {}
                for node_id, (base_url, items) in sources.items():
                    for item in items:
                        identity = item.get("installedIdentity")
                        digest = item.get("manifestSha256")
                        if (
                            isinstance(identity, str)
                            and isinstance(digest, str)
                            and item.get("manifestComplete") is True
                        ):
                            if (
                                not tombstone_metadata_valid
                                or _artifact_inventory_is_tombstoned(
                                    item, reconciliation_tombstones
                                )
                            ):
                                continue
                            replicas.setdefault((identity, digest), []).append(
                                (node_id, base_url, item)
                            )
                current_registry_ids: dict[str, str | None] = {}
                for candidates in replicas.values():
                    if not candidates:
                        continue
                    item = candidates[0][2]
                    artifact_model_id = item.get("modelId")
                    owner_model_id = item.get("ownerModelId")
                    owner_alias = (
                        owner_model_id
                        if isinstance(owner_model_id, str)
                        else artifact_model_id
                    )
                    if not isinstance(owner_alias, str):
                        continue
                    current_card = get_current_registry_card(ModelId(owner_alias))
                    current_registry_ids[owner_alias] = (
                        current_card.registry_card_id
                        if current_card is not None
                        else None
                    )
                selected_replicas = _select_reconciliation_generations(
                    replicas,
                    current_registry_ids,
                )
                pending = sorted(
                    identity
                    for identity, digest in selected_replicas
                    if (identity, digest) not in existing_generations
                )
                imported = 0
                if not inventory_only:
                    self._reconciliation_status = ReconciliationStatus(
                        state="importing",
                        inventory_only=False,
                        scanned_nodes=len(sources),
                        discovered_artifacts=len(selected_replicas),
                        pending_imports=tuple(pending),
                        failures=tuple(failures),
                    )
                    for identity, digest in sorted(selected_replicas):
                        if (identity, digest) in existing_generations:
                            continue
                        candidates = sorted(
                            selected_replicas[(identity, digest)],
                            key=lambda candidate: (
                                candidate[0] != str(self.node_id),
                                candidate[2].get("verificationState")
                                != "registry_verified",
                                candidate[0],
                            ),
                        )
                        imported_this_identity = False
                        for source_node_id, base_url, _item in candidates:
                            try:
                                export_response = await client.post(
                                    f"{base_url}/store/internal/exports",
                                    json={
                                        "installed_identity": identity,
                                        "manifest_sha256": digest,
                                        "target_node_id": str(self.node_id),
                                    },
                                )
                                export_response.raise_for_status()
                                grant = cast("object", json.loads(export_response.text))
                                if not isinstance(grant, dict):
                                    raise ValueError("export grant is not an object")
                                grant_dict = cast("dict[str, object]", grant)
                                token = grant_dict.get("capability_token")
                                record = grant_dict.get("record")
                                if not isinstance(token, str) or not isinstance(
                                    record, dict
                                ):
                                    raise ValueError("export grant is incomplete")
                                await self._store_client.request_peer_import(
                                    record=cast("dict[str, object]", record),
                                    source_base_url=base_url,
                                    capability_token=token,
                                    target_node_id=str(self.node_id),
                                )
                                imported += 1
                                imported_this_identity = True
                                existing_generations.add((identity, digest))
                                break
                            except (httpx.HTTPError, RuntimeError, ValueError) as error:
                                failures.append(
                                    f"Import {identity} from node {source_node_id} failed: {error}"
                                )
                        if not imported_this_identity:
                            continue
            now = datetime.now(tz=timezone.utc).isoformat()
            remaining = tuple(
                identity
                for identity, digest in selected_replicas
                if identity in pending
                and (identity, digest) not in existing_generations
            )
            state: Literal["complete", "failed"] = (
                "failed"
                if failures and (remaining or not tombstone_metadata_valid)
                else "complete"
            )
            self._reconciliation_status = ReconciliationStatus(
                state=state,
                inventory_only=inventory_only,
                scanned_nodes=len(sources),
                discovered_artifacts=len(selected_replicas),
                imported_artifacts=imported,
                pending_imports=remaining if not inventory_only else tuple(pending),
                failures=tuple(failures),
                last_verified_at=now,
            )
            return self._reconciliation_status

    async def _store_reconciliation_loop(self) -> None:
        """Run startup and periodic reconciliation on the store host."""

        if (
            self._store_client is None
            or self._store_client.local_store_path is None
            or self._skulk_config is None
            or self._skulk_config.model_store is None
            or not self._skulk_config.model_store.reconciliation.enabled
        ):
            return
        # The API becomes reachable after this lifetime task is scheduled but
        # before its intentional convergence delay expires. Represent that
        # scheduled first pass as active so dashboard clients do not mistake
        # the initial ``idle`` value for terminal convergence.
        self._reconciliation_status = ReconciliationStatus(
            state="scanning",
            inventory_only=(
                self._skulk_config.model_store.reconciliation.inventory_only
            ),
        )
        await anyio.sleep(10)
        while True:
            try:
                await self._run_store_reconciliation()
            except Exception as error:  # noqa: BLE001 - lifetime service boundary
                logger.exception(f"Model-store reconciliation failed: {error}")
            await anyio.sleep(
                self._skulk_config.model_store.reconciliation.interval_seconds
            )

    _ALLOWED_BROWSE_ROOTS = ["/Volumes", "/home", "/mnt", "/tmp", "/opt"]

    async def browse_filesystem(self, path: str = "/Volumes") -> JSONResponse:
        resolved = Path(path).resolve()
        if not any(
            resolved.is_relative_to(root) for root in self._ALLOWED_BROWSE_ROOTS
        ):
            raise HTTPException(status_code=400, detail="Path outside allowed roots")
        if not resolved.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")
        try:
            dirs = sorted(
                [
                    {"name": d.name, "path": str(d)}
                    for d in resolved.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ],
                key=lambda d: d["name"],
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="Permission denied") from exc
        return JSONResponse({"path": str(resolved), "directories": dirs})

    async def get_node_identity(self) -> JSONResponse:
        hostname = socket.gethostname()
        ip = hostname
        network = self.state.node_network.get(self.node_id)
        if network:
            # Prefer IPv4 LAN addresses — skip loopback, IPv6, and link-local
            for iface in network.interfaces:
                addr = iface.ip_address
                if (
                    addr
                    and not addr.startswith("127.")
                    and ":" not in addr  # skip IPv6
                    and not addr.startswith("169.254.")  # skip link-local
                ):
                    ip = addr
                    break
        return JSONResponse(
            {
                "nodeId": str(self.node_id),
                "hostname": hostname,
                "ipAddress": ip,
            }
        )

    async def get_tailscale_status(
        self, node_id: Annotated[str | None, Query()] = None
    ) -> TailscaleStatus:
        """Return this node's (or a peer's) Tailscale connectivity state.

        Queries tailscaled via ``tailscale status --json`` for the local node.
        When ``node_id`` is provided and refers to a different cluster node,
        the request is proxied to that node's ``/v1/connectivity/tailscale``
        endpoint.  Returns a ``TailscaleStatus`` with ``running=False`` when
        tailscale is not installed or tailscaled is not running.

        Args:
            node_id: Optional cluster node ID to proxy the request to.
                Omit or pass the local node's ID to query this node directly.
        """

        if node_id is not None and node_id != str(self.node_id):
            peer_urls = await self._reachable_peer_api_urls()
            base_url = peer_urls.get(node_id)
            if base_url is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Node not reachable: {node_id}",
                )
            timeout = httpx.Timeout(timeout=10.0, connect=2.0)
            async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
                try:
                    response = await client.get(f"{base_url}/v1/connectivity/tailscale")
                except httpx.HTTPError as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Failed to proxy Tailscale status to {node_id}: {exc}",
                    ) from exc
            if response.status_code >= 400:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=self._proxy_error_detail(response),
                )
            return TailscaleStatus.model_validate(response.json())

        return await query_tailscale_status()

    async def get_remote_access(self) -> RemoteAccessInfo:
        """Return aggregated remote access information for the local node.

        Combines local LAN access and Tailscale overlay access into a single
        snapshot.  The ``preferredUrl`` field gives the best address to reach
        this node — Tailscale URL when tailscaled is running, otherwise the
        LAN URL.  ``operatorUrl`` appends ``/operator`` and is suitable for
        QR code generation.
        """

        info = await build_remote_access_info(
            self.node_id,
            self.state.node_network,
            self.port,
        )
        # Any MagicDNS name this endpoint advertises must also be accepted by
        # the operator-mutation guard: a dashboard opened via the advertised
        # URL sends that name in Origin. Registering it here covers tailscaled
        # starting after Skulk did, when the startup priming saw nothing.
        dns_name = info.tailscale.dns_name
        if dns_name:
            self._known_self_host_names().add(dns_name.lower())
        return info

    async def get_store_downloads(self) -> JSONResponse:
        if self._store_client is None:
            raise HTTPException(status_code=503, detail="Store not configured")
        downloads = await self._store_client.list_active_downloads()
        return JSONResponse({"downloads": downloads})

    async def request_store_download(
        self,
        model_id: str,
        payload: StoreDownloadRequest | None = None,
    ) -> StoreDownloadResponse:
        """Request a store download with optional base or companion pins."""
        if self._store_client is None:
            raise HTTPException(status_code=503, detail="Store not configured")
        requested_model_id = ModelId(model_id)
        artifact_role = payload.artifact_role if payload is not None else "base"
        card = (
            get_current_registry_card(requested_model_id)
            or get_card(requested_model_id)
            if artifact_role == "base"
            else None
        )
        if card is None and artifact_role == "base":
            await get_model_cards()
            card = get_current_registry_card(requested_model_id) or get_card(
                requested_model_id
            )
        gguf_file = payload.gguf_file if payload is not None else None
        extra_gguf_files = payload.extra_gguf_files if payload is not None else []
        source_revision = payload.source_revision if payload is not None else None
        source_repository = payload.source_repository if payload is not None else None
        registry_card_id = payload.registry_card_id if payload is not None else None
        artifact_bundle_id = payload.artifact_bundle_id if payload is not None else None
        owner_model_id = payload.owner_model_id if payload is not None else None
        owner_registry_card_id = (
            payload.owner_registry_card_id if payload is not None else None
        )
        requested_card_id = payload.registry_card_id if payload is not None else None
        if requested_card_id is not None and artifact_role == "base":
            current_card = get_current_registry_card(requested_model_id)
            installed_card = get_card(requested_model_id)
            card = next(
                (
                    candidate
                    for candidate in (current_card, installed_card)
                    if candidate is not None
                    and candidate.registry_card_id == requested_card_id
                    and candidate.model_id == requested_model_id
                ),
                None,
            )
            if card is None:
                raise HTTPException(
                    status_code=409,
                    detail="Requested immutable card is not available for this alias",
                )
        if card is not None:
            card_bundle = card.artifact_bundle
            if artifact_bundle_id is not None and (
                card_bundle is None or card_bundle.bundle_id != artifact_bundle_id
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Requested immutable artifact bundle is not available "
                        "for this alias"
                    ),
                )
            gguf_file = gguf_file or card.gguf_file
            source_revision = source_revision or card.source_revision
            source_repository = source_repository or str(card.artifact_repository)
            registry_card_id = card.registry_card_id
        result = await self._store_client.request_store_download(
            model_id,
            gguf_file=gguf_file,
            extra_gguf_files=extra_gguf_files,
            source_revision=source_revision,
            source_repository=source_repository,
            registry_card_id=registry_card_id,
            artifact_bundle_id=artifact_bundle_id,
            owner_model_id=owner_model_id,
            owner_registry_card_id=owner_registry_card_id,
            artifact_role=artifact_role,
        )
        return StoreDownloadResponse.model_validate(result, strict=False)

    async def get_store_download_status(self, model_id: str) -> JSONResponse:
        if self._store_client is None:
            raise HTTPException(status_code=503, detail="Store not configured")
        result = await self._store_client.get_store_download_status(model_id)
        return JSONResponse(result)

    async def cancel_store_download(self, model_id: str) -> JSONResponse:
        """Cancel one pending or active canonical-store download.

        Args:
            model_id: HuggingFace-style model identifier from the route.

        Returns:
            A confirmation naming the cancelled model and terminal state.

        Raises:
            HTTPException: If the store is unavailable or no cancellable
                transfer exists.

        The store retains partial files so a later download can resume.
        """

        if self._store_client is None:
            raise HTTPException(status_code=503, detail="Store not configured")
        cancelled = await self._store_client.cancel_store_download(model_id)
        if not cancelled:
            raise HTTPException(
                status_code=409,
                detail=f"No active store download for {model_id}",
            )
        return JSONResponse(
            {
                "modelId": model_id,
                "status": "cancelled",
                "cancelled": True,
            }
        )

    async def delete_store_model(self, model_id: str) -> JSONResponse:
        if self._store_client is None:
            raise HTTPException(status_code=503, detail="Store not configured")
        deleted = await self._store_client.delete_store_model(model_id)
        if not deleted:
            raise HTTPException(
                status_code=404, detail=f"Model {model_id} not in store"
            )
        # The store-delete above only removed the store host's canonical copy.
        # Workers cache their own staged shards independently, so broadcast a
        # fleet-wide eviction to free that disk too (#427). The canonical delete
        # has already succeeded, so a failure to dispatch the (best-effort)
        # eviction must not turn this into a 500: that would report the delete as
        # failed while it actually happened, and a retry would 404 and never
        # broadcast the eviction at all. Log and still report success; the
        # staging budget pass and POST /store/purge-staging are the backstops.
        try:
            await self.command_sender.send(
                ForwarderCommand(
                    origin=self._system_id,
                    command=EvictStagedModel(model_id=ModelId(model_id)),
                )
            )
        except Exception:
            logger.exception(
                "Store-delete succeeded but the staged-eviction broadcast failed "
                f"(model_id={model_id}); staged copies will be reclaimed by the "
                "staging budget pass or a manual purge"
            )
        return JSONResponse({"modelId": model_id, "deleted": True})

    async def optimize_model(self, model_id: str, request: Request) -> JSONResponse:
        """Start an OptiQ mixed-precision optimization for a model."""
        if self._model_optimizer is None:
            raise HTTPException(
                status_code=503,
                detail="Model optimizer not available (store not configured)",
            )
        try:
            body = await _read_request_json_object(request)
        except Exception:
            body = {}
        target_bpw = _coerce_float(body.get("target_bpw", 4.5), default=4.5)
        candidate_bits = _coerce_candidate_bits(
            body.get("candidate_bits", _DEFAULT_OPTIMIZER_CANDIDATE_BITS)
        )
        try:
            await self._model_optimizer.optimize(
                model_id=model_id,
                target_bpw=target_bpw,
                candidate_bits=candidate_bits,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(
            {
                "modelId": model_id,
                "status": "started",
                "targetBpw": target_bpw,
            }
        )

    async def get_optimize_status(self, model_id: str) -> JSONResponse:
        """Get optimization status for a model."""
        if self._model_optimizer is None:
            raise HTTPException(status_code=503, detail="Model optimizer not available")
        job = self._model_optimizer.get_status(model_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No optimization job found")
        return JSONResponse(
            {
                "modelId": job.model_id,
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
                "resultPath": job.result_path,
                "achievedBpw": job.achieved_bpw,
                "estimatedSizeMb": job.estimated_size_mb,
                "error": job.error,
            }
        )

    async def embeddings(self, request: EmbeddingRequest) -> JSONResponse:
        """OpenAI-compatible embeddings endpoint — routes through cluster pipeline."""
        from skulk.shared.types.embedding import TextEmbeddingTaskParams

        texts = [request.input] if isinstance(request.input, str) else request.input
        if not texts:
            raise HTTPException(status_code=400, detail="input must not be empty")

        if request.dimensions is not None:
            raise HTTPException(
                status_code=400,
                detail="The `dimensions` parameter is not supported; embeddings are returned at the model's native dimensionality.",
            )

        model_id = await self._validate_embedding_model(ModelId(request.model))
        command = TextEmbedding(
            owner_node=self.node_id,
            task_params=TextEmbeddingTaskParams(
                model=model_id,
                input_texts=texts,
                encoding_format=request.encoding_format,
            ),
        )
        command_id = command.command_id

        try:
            recv = self._open_stream_queue(self._embedding_queues, command_id)

            await self._send(command)

            with recv as chunks:
                async for chunk in chunks:
                    if isinstance(chunk, ErrorChunk):
                        raise HTTPException(
                            status_code=500,
                            detail=f"Embedding failed: {chunk.error_message}",
                        )
                    import base64
                    import struct

                    embeddings_data: list[list[float] | str] = []
                    for emb in chunk.embeddings:
                        if request.encoding_format == "base64":
                            b64 = base64.b64encode(
                                struct.pack(f"<{len(emb)}f", *emb)
                            ).decode("ascii")
                            embeddings_data.append(b64)
                        else:
                            embeddings_data.append(emb)

                    response = EmbeddingResponse(
                        data=[
                            EmbeddingObject(index=idx, embedding=emb_data)
                            for idx, emb_data in enumerate(embeddings_data)
                        ],
                        model=model_id,
                        usage=EmbeddingUsage(
                            prompt_tokens=chunk.token_count,
                            total_tokens=chunk.token_count,
                        ),
                    )
                    return JSONResponse(response.model_dump())

            raise HTTPException(
                status_code=500, detail="No embedding response received"
            )

        except anyio.get_cancelled_exc_class():
            cancel_command = TaskCancelled(cancelled_command_id=command_id)
            self._cancelled_command_ids.add(command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=cancel_command)
                )
            raise
        finally:
            await self._finalize_command_stream(
                command_id,
                cast(
                    dict[CommandId, Sender[object]],
                    self._embedding_queues,
                ),
            )

    @staticmethod
    def _speech_synthesis_task_params(
        request: AudioSpeechRequest,
        model_id: ModelId,
        response_format: AudioResponseFormat,
        *,
        reference_voice_profile: str | None = None,
        reference_audio_filename: str | None = None,
        reference_audio_content_type: str | None = None,
        reference_audio_sha256: str | None = None,
    ) -> SpeechSynthesisTaskParams:
        """Translate a validated API/facade request into the core task contract."""

        return SpeechSynthesisTaskParams(
            model=model_id,
            input_text=request.input,
            response_format=response_format,
            voice=request.voice,
            speed=request.speed,
            instruct=request.instruct,
            lang_code=request.lang_code,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            max_tokens=request.max_tokens,
            seed=request.seed,
            stream=request.stream,
            streaming_interval=request.streaming_interval,
            reference_text=request.reference_text,
            reference_voice_profile=reference_voice_profile,
            reference_audio_present=reference_audio_sha256 is not None,
            reference_audio_filename=reference_audio_filename,
            reference_audio_content_type=reference_audio_content_type,
            reference_audio_sha256=reference_audio_sha256,
        )

    async def _apply_default_speech_voice(
        self,
        request: AudioSpeechRequest,
        model_id: ModelId,
    ) -> AudioSpeechRequest:
        """Validate explicit voices and apply the card default when omitted."""

        matching_instances = tuple(
            instance
            for instance in self.state.instances.values()
            if instance.shard_assignments.model_id == model_id
        )
        # Normal callers reach this only after mounted-model validation. Keep
        # the helper inert for isolated adapters/tests that intentionally
        # replace that validator without constructing replicated state.
        if not matching_instances:
            return request
        card = next(
            (
                candidate
                for instance in matching_instances
                if (candidate := self._model_card_for_instance(instance)) is not None
            ),
            None,
        )
        if card is None:
            card = await self._get_running_model_card(model_id)
        if card.audio is None:
            return request
        if request.voice is not None:
            if card.audio.supports_voice_listing is not True:
                reference_guidance = (
                    "; omit `voice` and provide `reference_audio` instead"
                    if card.audio.supports_reference_audio is True
                    else "; omit `voice`"
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Model {model_id} does not expose a voice catalog"
                        f"{reference_guidance}"
                    ),
                )
            if request.voice not in card.audio.voices:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Voice {request.voice!r} is not available for model "
                        f"{model_id}; use GET /v1/audio/voices"
                    ),
                )
            return request
        if card.audio.default_voice is None:
            return request
        return request.model_copy(update={"voice": card.audio.default_voice})

    async def _bundled_reference_profile_for_voice(
        self,
        model_id: ModelId,
        voice: str | None,
    ) -> str | None:
        """Return the bundled profile backing a validated voice selection.

        Args:
            model_id: Mounted TTS model selected for synthesis.
            voice: Explicit or card-default voice identifier.

        Returns:
            The bundled profile identifier, or ``None`` for model-native voices.
        """

        if voice is None:
            return None
        matching_instances = tuple(
            instance
            for instance in self.state.instances.values()
            if instance.shard_assignments.model_id == model_id
        )
        # Request validation normally guarantees mounted state. Keep this
        # lookup inert for isolated adapters/tests that replace validation.
        if not matching_instances:
            return None
        card = next(
            (
                candidate
                for instance in matching_instances
                if (candidate := self._model_card_for_instance(instance)) is not None
            ),
            None,
        )
        if card is None:
            card = await self._get_running_model_card(model_id)
        if card.audio is None:
            return None
        for catalog_voice in card.audio.voice_catalog:
            if catalog_voice.id == voice:
                return catalog_voice.reference_profile
        return None

    async def _start_speech_synthesis(
        self,
        task_params: SpeechSynthesisTaskParams,
        *,
        target_instance_id: InstanceId | None = None,
        target_node: NodeId | None = None,
        reference_audio: bytes | None = None,
    ) -> tuple[CommandId, Receiver[AudioChunk | ErrorChunk]]:
        """Submit one core TTS command and register its DATA receive queue."""

        command = SpeechSynthesis(
            owner_node=self.node_id,
            task_params=task_params,
            target_instance_id=target_instance_id,
        )
        command_id = command.command_id
        recv = self._open_stream_queue(self._audio_speech_queues, command_id)
        if reference_audio is not None:
            self._speech_media_commands.add(command_id)
            assert target_node is not None
            self._speech_media_targets[command_id] = target_node
        try:
            await self._send(command)
            if reference_audio is not None:
                if target_node is None or self._speech_media_packet_sender is None:
                    raise HTTPException(
                        status_code=503,
                        detail="Ephemeral speech media transport is unavailable",
                    )
                for sequence, offset in enumerate(
                    range(0, len(reference_audio), _SPEECH_MEDIA_CHUNK_BYTES)
                ):
                    await self._speech_media_packet_sender.send(
                        SpeechMediaPacket(
                            source_node=self.node_id,
                            target_node=target_node,
                            command_id=command_id,
                            sequence=sequence,
                            kind="chunk",
                            filename=task_params.reference_audio_filename,
                            content_type=task_params.reference_audio_content_type,
                            data=reference_audio[
                                offset : offset + _SPEECH_MEDIA_CHUNK_BYTES
                            ],
                        )
                    )
                await self._speech_media_packet_sender.send(
                    SpeechMediaPacket(
                        source_node=self.node_id,
                        target_node=target_node,
                        command_id=command_id,
                        sequence=(len(reference_audio) + _SPEECH_MEDIA_CHUNK_BYTES - 1)
                        // _SPEECH_MEDIA_CHUNK_BYTES,
                        kind="completed",
                        filename=task_params.reference_audio_filename,
                        content_type=task_params.reference_audio_content_type,
                        sha256=task_params.reference_audio_sha256,
                    )
                )
        except Exception:
            if command_id not in self._cancelled_command_ids:
                with anyio.CancelScope(shield=True):
                    await self._cancel_audio_speech_command(command_id)
            if reference_audio is not None and target_node is not None:
                with contextlib.suppress(Exception):
                    assert self._speech_media_packet_sender is not None
                    await self._speech_media_packet_sender.send(
                        SpeechMediaPacket(
                            source_node=self.node_id,
                            target_node=target_node,
                            command_id=command_id,
                            sequence=0,
                            kind="cancelled",
                        )
                    )
            await self._finalize_command_stream(
                command_id,
                cast(
                    dict[CommandId, Sender[object]],
                    self._audio_speech_queues,
                ),
            )
            raise
        return command_id, recv

    async def _prepare_builtin_tts_task(
        self,
        call: CapabilityCall,
    ) -> SpeechSynthesisTaskParams:
        """Validate one generic TTS payload against live core speech state."""

        request_payload = dict(call.payload)
        text = request_payload.pop("text", None)
        if not isinstance(text, str) or not text:
            raise HTTPException(
                status_code=422,
                detail="TTS provider payload requires non-empty text",
            )
        request_payload["input"] = text
        request_payload["stream"] = True
        request_payload.setdefault("response_format", AudioResponseFormat.Mp3.value)
        request = AudioSpeechRequest.model_validate(request_payload)
        model_id, response_format = await self._validate_speech_synthesis_model(
            ModelId(request.model),
            request.response_format,
            stream=True,
        )
        request = await self._apply_default_speech_voice(request, model_id)
        reference_voice_profile = await self._bundled_reference_profile_for_voice(
            model_id,
            request.voice,
        )
        if response_format != AudioResponseFormat.Mp3:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The TTS provider facade currently streams only mp3 audio; "
                    f"resolved {response_format.value}"
                ),
            )
        return self._speech_synthesis_task_params(
            request,
            model_id,
            response_format,
            reference_voice_profile=reference_voice_profile,
        )

    async def _admit_builtin_tts_stream(
        self,
        call: CapabilityCall,
    ) -> CapabilityError | None:
        """Reject an unavailable or invalid mounted TTS request before start."""

        if not self._has_mounted_streaming_tts_model():
            return CapabilityError(
                code="not_found",
                message="no streaming TTS capacity is currently advertised",
            )
        requested_model = call.payload.get("model")
        if not isinstance(requested_model, str) or not any(
            instance.shard_assignments.model_id == ModelId(requested_model)
            for instance in self.state.instances.values()
        ):
            return CapabilityError(
                code="not_found",
                message=f"no mounted TTS instance for model {requested_model!r}",
            )
        try:
            await self._prepare_builtin_tts_task(call)
        except ValidationError as exc:
            return CapabilityError(code="invalid_payload", message=str(exc))
        except HTTPException as exc:
            if exc.status_code == 404:
                code: CapabilityErrorCode = "not_found"
            elif exc.status_code in (400, 422):
                code = "invalid_payload"
            else:
                code = "provider_error"
            return CapabilityError(code=code, message=str(exc.detail))
        return None

    async def _stream_builtin_tts(
        self,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        """Stream core ``AudioChunk`` output as generic binary media frames."""

        try:
            task_params = await self._prepare_builtin_tts_task(call)
        except (ValidationError, HTTPException) as exc:
            raise RuntimeError(
                f"TTS availability changed after admission: {exc}"
            ) from exc
        command_id, recv = await self._start_speech_synthesis(task_params)
        sequence = 1
        received_audio = False
        terminal_received = False
        try:
            with recv as chunks:
                while True:
                    chunk: AudioChunk | ErrorChunk | None = None
                    with anyio.move_on_after(_STREAM_IDLE_TIMEOUT_SECONDS) as scope:
                        try:
                            chunk = await chunks.receive()
                        except (EndOfStream, ClosedResourceError) as exc:
                            raise RuntimeError(
                                "Core TTS DATA stream closed without a terminal chunk"
                            ) from exc
                    if scope.cancelled_caught:
                        self._data_plane_observer.record_idle_timeout()
                        raise RuntimeError(
                            "Core TTS DATA stream exceeded its idle deadline"
                        )
                    assert chunk is not None
                    if isinstance(chunk, ErrorChunk):
                        raise RuntimeError(
                            f"Core speech synthesis failed: {chunk.error_message}"
                        )
                    audio_bytes = _decode_audio_chunk_data(chunk)
                    if audio_bytes:
                        received_audio = True
                        payload: dict[str, object] = {
                            "model": str(chunk.model),
                            "format": chunk.format.value,
                            "chunk_index": chunk.chunk_index,
                            "is_partial": chunk.is_partial,
                        }
                        if chunk.sample_rate is not None:
                            payload["sample_rate"] = chunk.sample_rate
                        yield CapabilityStreamFrame(
                            call_id=call.call_id,
                            direction="provider_to_caller",
                            sequence=sequence,
                            kind="chunk",
                            payload=payload,
                            media=InlineMediaAttachment(
                                data=audio_bytes,
                                media_type=_AUDIO_CONTENT_TYPES[chunk.format],
                                codec=chunk.format.value,
                                sample_rate=chunk.sample_rate,
                            ),
                        )
                        sequence += 1
                    if chunk.finish_reason is not None:
                        if not received_audio:
                            raise RuntimeError("Core TTS produced no audio")
                        terminal_received = True
                        yield CapabilityStreamFrame(
                            call_id=call.call_id,
                            direction="provider_to_caller",
                            sequence=sequence,
                            kind="completed",
                        )
                        return
        finally:
            if not terminal_received and not self._command_task_is_terminal(command_id):
                await self._cancel_audio_speech_command(command_id)
            await self._finalize_command_stream(
                command_id,
                cast(
                    dict[CommandId, Sender[object]],
                    self._audio_speech_queues,
                ),
            )

    async def _prepare_builtin_stt_task(
        self,
        call: CapabilityCall,
    ) -> AudioTranscriptionTaskParams:
        """Validate batch STT metadata against mounted core speech state."""

        payload = dict(call.payload)
        model = payload.pop("model", None)
        if not isinstance(model, str) or not model:
            raise HTTPException(
                status_code=422,
                detail="STT provider payload requires a non-empty model",
            )
        model_id = await self._validate_audio_transcription_model(ModelId(model))
        filename = payload.get("filename")
        content_type = _normalize_upload_content_type(
            cast(str | None, payload.get("content_type"))
        )
        suffix = Path(filename if isinstance(filename, str) else "").suffix.lower()
        if (
            content_type is not None
            and content_type not in _AUDIO_UPLOAD_CONTENT_TYPES
            and suffix not in _AUDIO_UPLOAD_EXTENSIONS
        ):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported audio input content type: {content_type}",
            )
        payload["content_type"] = content_type
        return AudioTranscriptionTaskParams.model_validate(
            {
                **payload,
                "model": model_id,
                "audio_sha256": "0" * 64,
            }
        )

    async def _admit_builtin_stt_stream(
        self,
        call: CapabilityCall,
    ) -> CapabilityError | None:
        """Reject batch STT before start unless mounted capacity is available."""

        if not self._has_mounted_stt_model():
            return CapabilityError(
                code="not_found",
                message="no batch STT capacity is currently advertised",
            )
        try:
            task_params = await self._prepare_builtin_stt_task(call)
        except ValidationError as exc:
            return CapabilityError(code="invalid_payload", message=str(exc))
        except HTTPException as exc:
            if exc.status_code == 404:
                code: CapabilityErrorCode = "not_found"
            elif exc.status_code in (400, 415, 422):
                code = "invalid_payload"
            else:
                code = "provider_error"
            return CapabilityError(code=code, message=str(exc.detail))
        if not self._has_mounted_stt_model(task_params.model):
            return CapabilityError(
                code="overloaded",
                message=(
                    f"no ready batch STT runner for requested model {task_params.model}"
                ),
            )
        return None

    async def _collect_builtin_stt_audio(
        self,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> tuple[bytes, str]:
        """Collect one bounded encoded clip from ordered provider media frames."""

        audio_parts: list[bytes] = []
        total_bytes = 0
        completed = False
        media_type: str | None = None
        async for frame in input_frames:
            if frame.kind == "started":
                continue
            if frame.kind == "chunk":
                if not isinstance(frame.media, InlineMediaAttachment):
                    raise ValueError(
                        "batch STT requires inline binary media attachments; "
                        "managed blob resolution is not available yet"
                    )
                frame_media_type = _normalize_upload_content_type(
                    frame.media.media_type
                )
                if frame_media_type not in _AUDIO_UPLOAD_CONTENT_TYPES:
                    raise ValueError(
                        f"unsupported batch STT media type: {frame_media_type}"
                    )
                if media_type is None:
                    media_type = frame_media_type
                elif media_type != frame_media_type:
                    raise ValueError(
                        "batch STT media frames must use one consistent media type"
                    )
                total_bytes += len(frame.media.data)
                if total_bytes > _MAX_AUDIO_UPLOAD_BYTES:
                    raise ValueError(
                        f"batch STT audio exceeds {_MAX_AUDIO_UPLOAD_BYTES} bytes"
                    )
                audio_parts.append(frame.media.data)
                continue
            if frame.kind == "completed":
                completed = True
                break
            if frame.kind == "cancelled":
                detail = (
                    frame.error.message
                    if frame.error is not None
                    else "caller cancelled batch STT input"
                )
                raise _BuiltinSttInputCancelledError(detail)
            detail = frame.error.message if frame.error is not None else frame.kind
            raise RuntimeError(f"batch STT input terminated: {detail}")
        if not completed:
            raise RuntimeError("batch STT input ended without half-close")
        audio_bytes = b"".join(audio_parts)
        if not audio_bytes:
            raise ValueError("batch STT audio input is empty")
        assert media_type is not None
        return audio_bytes, media_type

    async def _stream_builtin_stt(
        self,
        call: CapabilityCall,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> AsyncIterator[CapabilityStreamFrame]:
        """Transcribe one bounded provider media input after input half-close."""

        try:
            task_params = await self._prepare_builtin_stt_task(call)
        except (ValidationError, HTTPException) as exc:
            raise RuntimeError(
                f"batch STT availability changed after admission: {exc}"
            ) from exc
        try:
            audio_bytes, media_type = await self._collect_builtin_stt_audio(
                input_frames
            )
        except _BuiltinSttInputCancelledError as exc:
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=1,
                kind="cancelled",
                error=CapabilityStreamError(code="cancelled", message=str(exc)),
            )
            return
        except ValueError as exc:
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=1,
                kind="failed",
                error=CapabilityStreamError(
                    code="invalid_frame",
                    message=str(exc),
                ),
            )
            return
        if (
            task_params.content_type is not None
            and task_params.content_type != media_type
        ):
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=1,
                kind="failed",
                error=CapabilityStreamError(
                    code="invalid_frame",
                    message=(
                        "batch STT attachment media type does not match opening "
                        "content_type"
                    ),
                ),
            )
            return
        if not self._has_mounted_stt_model(task_params.model):
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=1,
                kind="failed",
                error=CapabilityStreamError(
                    code="unreachable",
                    message=(
                        "batch STT capacity changed while receiving input; "
                        "retry after the requested model is ready"
                    ),
                ),
            )
            return
        task_params = task_params.model_copy(update={"content_type": media_type})
        chunks = await self._execute_audio_transcription(task_params, audio_bytes)
        text = "".join(chunk.text for chunk in chunks).strip()
        language = next(
            (chunk.language for chunk in chunks if chunk.language is not None),
            None,
        )
        segments: list[dict[str, str | int | float | bool | None]] = []
        for chunk in chunks:
            segments.extend(chunk.segments)
        payload: dict[str, object] = {
            "model": str(task_params.model),
            "text": text,
        }
        if language is not None:
            payload["language"] = language
        if segments:
            payload["segments"] = segments
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=1,
            kind="completed",
            payload=payload,
        )

    async def _prepare_builtin_realtime_stt_task(
        self,
        call: CapabilityCall,
    ) -> tuple[RealtimeAudioTranscriptionTaskParams, InstanceId, NodeId]:
        """Validate one realtime STT call against mounted cluster capacity."""

        payload = dict(call.payload)
        sample_rate = payload.pop("sample_rate", None)
        params = RealtimeAudioTranscriptionTaskParams.model_validate(
            {**payload, "input_sample_rate": sample_rate}
        )
        if self._realtime_stt_instance(params.model) is None:
            raise HTTPException(
                status_code=404,
                detail=(f"No reachable realtime STT instance for model {params.model}"),
            )
        selected = self._realtime_stt_instance(
            params.model,
            call_id=call.call_id,
            require_idle=True,
        )
        if selected is None:
            raise HTTPException(
                status_code=409,
                detail="All realtime STT runners for this model are already busy",
            )
        instance_id, _, target_node = selected
        return params, instance_id, target_node

    async def _admit_builtin_realtime_stt_stream(
        self,
        call: CapabilityCall,
    ) -> CapabilityError | None:
        """Reject realtime STT before start unless reachable capacity is truthful."""

        if not self._has_realtime_stt_model():
            return CapabilityError(
                code="not_found",
                message="no reachable realtime STT capacity is currently advertised",
            )
        try:
            _, instance_id, _ = await self._prepare_builtin_realtime_stt_task(call)
        except ValidationError as exc:
            return CapabilityError(code="invalid_payload", message=str(exc))
        except HTTPException as exc:
            if exc.status_code == 404:
                code: CapabilityErrorCode = "not_found"
            elif exc.status_code == 409:
                code = "overloaded"
            elif exc.status_code in (400, 422):
                code = "invalid_payload"
            else:
                code = "provider_error"
            return CapabilityError(code=code, message=str(exc.detail))
        active = self._active_capability_streams.get(call.call_id)
        if active is None:
            return CapabilityError(
                code="provider_error",
                message="realtime STT admission lost its stream reservation",
            )
        active.reserved_instance_id = instance_id
        return None

    async def _start_realtime_audio_transcription(
        self,
        params: RealtimeAudioTranscriptionTaskParams,
        instance_id: InstanceId,
    ) -> tuple[
        CommandId,
        Sender[TranscriptionChunk | ErrorChunk],
        Receiver[TranscriptionChunk | ErrorChunk],
    ]:
        """Submit one pinned realtime STT command and register its output."""

        command = RealtimeAudioTranscription(
            owner_node=self.node_id,
            target_instance_id=instance_id,
            task_params=params,
        )
        command_id = command.command_id
        output_sender, output_receiver = channel[TranscriptionChunk | ErrorChunk](
            _REALTIME_STT_OUTPUT_BUFFER
        )
        self._audio_transcription_queues[command_id] = output_sender
        self._realtime_audio_transcription_commands.add(command_id)
        self._realtime_task_release_events[command_id] = anyio.Event()
        try:
            await self._send(command)
        except Exception:
            self._realtime_task_release_events.pop(command_id, None)
            await self._finalize_command_stream(
                command_id,
                cast(
                    dict[CommandId, Sender[object]],
                    self._audio_transcription_queues,
                ),
            )
            raise
        return command_id, output_sender, output_receiver

    def _mark_realtime_task_released(self, command_id: CommandId) -> None:
        """Wake a provider waiting for authoritative realtime task release."""

        release_event = self._realtime_task_release_events.get(command_id)
        if release_event is not None:
            release_event.set()

    async def _cancel_audio_transcription_command(self, command_id: CommandId) -> None:
        """Cancel one core STT command while preserving cleanup ordering."""

        if command_id in self._cancelled_command_ids:
            return
        self._cancelled_command_ids.add(command_id)
        sender = self._speech_media_packet_sender
        if sender is not None:
            for target_node in self._transcription_media_targets.get(command_id, ()):
                with contextlib.suppress(Exception):
                    await sender.send(
                        SpeechMediaPacket(
                            source_node=self.node_id,
                            target_node=target_node,
                            command_id=command_id,
                            sequence=0,
                            kind="cancelled",
                            purpose="transcription_audio",
                        )
                    )
        await self.command_sender.send(
            ForwarderCommand(
                origin=self._system_id,
                command=TaskCancelled(cancelled_command_id=command_id),
            )
        )

    async def _send_realtime_audio_input(
        self,
        target_node: NodeId,
        frame: RealtimeAudioInputFrame,
    ) -> None:
        """Send one PCM frame through the local fast path or remote DATA path."""

        if target_node == self.node_id and self._realtime_audio_sender is not None:
            try:
                await self._realtime_audio_sender.send(frame)
            except (BrokenResourceError, ClosedResourceError):
                # Worker recreation closes the original receive end while the
                # API survives. Fall through to the node-local packet topic,
                # whose receiver is rewired on every Worker construction.
                self._realtime_audio_sender = None
            else:
                return
        if self._realtime_audio_packet_sender is not None:
            await self._realtime_audio_packet_sender.send(
                RealtimeAudioPacket(
                    source_node=self.node_id,
                    target_node=target_node,
                    command_id=frame.command_id,
                    sequence=frame.sequence,
                    kind=frame.kind,
                    data=frame.data,
                )
            )
            return
        raise RuntimeError("realtime audio worker ingress is unavailable")

    async def _pump_builtin_realtime_stt_input(
        self,
        *,
        command_id: CommandId,
        params: RealtimeAudioTranscriptionTaskParams,
        target_node: NodeId,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> None:
        """Translate provider frames into local or remote worker PCM ingress."""

        terminal_sent = False
        async for frame in input_frames:
            if frame.kind == "started":
                continue
            if frame.kind == "chunk":
                media = frame.media
                payload = frame.payload or {}
                if not isinstance(media, InlineMediaAttachment):
                    raise ValueError(
                        "realtime STT requires inline PCM media attachments"
                    )
                if (
                    media.codec != "pcm_s16le"
                    or media.channels != 1
                    or media.sample_rate != params.input_sample_rate
                    or payload.get("format") != "pcm_s16le"
                    or payload.get("channels") != 1
                    or payload.get("sample_rate") != params.input_sample_rate
                ):
                    raise ValueError(
                        "realtime STT frame metadata must match the negotiated "
                        "mono pcm_s16le sample rate"
                    )
                await self._send_realtime_audio_input(
                    target_node,
                    RealtimeAudioInputFrame(
                        command_id=command_id,
                        sequence=frame.sequence,
                        kind="chunk",
                        data=media.data,
                    ),
                )
                continue
            if frame.kind == "failed":
                detail = frame.error.message if frame.error is not None else "unknown"
                raise RuntimeError(f"caller input stream failed: {detail}")
            kind = "completed" if frame.kind == "completed" else "cancelled"
            await self._send_realtime_audio_input(
                target_node,
                RealtimeAudioInputFrame(
                    command_id=command_id,
                    sequence=frame.sequence,
                    kind=kind,
                ),
            )
            terminal_sent = True
            return
        if not terminal_sent:
            raise RuntimeError("realtime STT input ended without a terminal frame")

    async def _stream_builtin_realtime_stt(
        self,
        call: CapabilityCall,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> AsyncIterator[CapabilityStreamFrame]:
        """Bridge provider PCM input to a true incremental core STT session."""

        try:
            (
                params,
                instance_id,
                target_node,
            ) = await self._prepare_builtin_realtime_stt_task(call)
        except (ValidationError, HTTPException) as exc:
            raise RuntimeError(
                f"realtime STT availability changed after admission: {exc}"
            ) from exc
        (
            command_id,
            output_sender,
            output_receiver,
        ) = await self._start_realtime_audio_transcription(params, instance_id)
        input_cancel_scope = anyio.CancelScope()
        input_done = anyio.Event()

        async def pump_input() -> None:
            with input_cancel_scope:
                try:
                    await self._pump_builtin_realtime_stt_input(
                        command_id=command_id,
                        params=params,
                        target_node=target_node,
                        input_frames=input_frames,
                    )
                except anyio.get_cancelled_exc_class():
                    raise
                except Exception as exc:
                    with contextlib.suppress(BrokenResourceError, ClosedResourceError):
                        await output_sender.send(
                            ErrorChunk(
                                model=params.model,
                                error_message=f"Realtime STT input failed: {exc}",
                            )
                        )
                    with anyio.CancelScope(shield=True):
                        await self._cancel_audio_transcription_command(command_id)
                finally:
                    input_done.set()

        try:
            self._tg.start_soon(pump_input)
        except RuntimeError as exc:
            await self._cancel_audio_transcription_command(command_id)
            raise RuntimeError("realtime STT input pump could not start") from exc

        sequence = 1
        terminal_received = False
        try:
            with output_receiver as chunks:
                while True:
                    chunk: TranscriptionChunk | ErrorChunk | None = None
                    with anyio.move_on_after(_STREAM_IDLE_TIMEOUT_SECONDS) as scope:
                        try:
                            chunk = await chunks.receive()
                        except (EndOfStream, ClosedResourceError) as exc:
                            raise RuntimeError(
                                "Core realtime STT DATA stream closed without "
                                "a terminal chunk"
                            ) from exc
                    if scope.cancelled_caught:
                        raise RuntimeError(
                            "Core realtime STT DATA stream exceeded its idle deadline"
                        )
                    assert chunk is not None
                    if isinstance(chunk, ErrorChunk):
                        terminal_received = True
                        raise RuntimeError(
                            f"Core realtime transcription failed: {chunk.error_message}"
                        )
                    if chunk.finish_reason is not None:
                        terminal_received = True
                        yield CapabilityStreamFrame(
                            call_id=call.call_id,
                            direction="provider_to_caller",
                            sequence=sequence,
                            kind="completed",
                            payload={
                                "model": str(chunk.model),
                                "text": chunk.text,
                                "is_partial": False,
                            },
                        )
                        return
                    yield CapabilityStreamFrame(
                        call_id=call.call_id,
                        direction="provider_to_caller",
                        sequence=sequence,
                        kind="chunk",
                        payload={
                            "model": str(chunk.model),
                            "text": chunk.text,
                            "is_partial": True,
                        },
                    )
                    sequence += 1
        finally:
            input_cancel_scope.cancel()
            try:
                with anyio.CancelScope(shield=True):
                    with anyio.move_on_after(1.0):
                        await input_done.wait()
                    if not terminal_received and not self._command_task_is_terminal(
                        command_id
                    ):
                        await self._cancel_audio_transcription_command(command_id)
                    await self._finalize_command_stream(
                        command_id,
                        cast(
                            dict[CommandId, Sender[object]],
                            self._audio_transcription_queues,
                        ),
                    )
                    release_event = self._realtime_task_release_events.get(command_id)
                    if terminal_received and release_event is not None:
                        with anyio.move_on_after(
                            _REALTIME_TASK_RELEASE_TIMEOUT_SECONDS
                        ) as release_scope:
                            await release_event.wait()
                        if release_scope.cancelled_caught:
                            raise RuntimeError(
                                "Core realtime STT task did not reach authoritative "
                                "terminal state"
                            )
            finally:
                self._realtime_task_release_events.pop(command_id, None)

    async def _open_realtime_transcription_session(
        self,
        model: str,
        sample_rate: int,
    ) -> CapabilityStreamSession:
        """Open the built-in realtime STT provider for one WebSocket owner.

        Args:
            model: Mounted model selected by the compatibility client.
            sample_rate: Negotiated PCM16 sample rate.

        Returns:
            Generic Fabric provider session owned by this API node.
        """

        return await self._stream_capability(
            self.node_id,
            REALTIME_STT_CAPABILITY_DESCRIPTOR.id,
            REALTIME_STT_CAPABILITY_DESCRIPTOR.version,
            descriptor_revision(REALTIME_STT_CAPABILITY_DESCRIPTOR),
            {"model": model, "sample_rate": sample_rate},
            timeout_seconds=MAX_CALL_TIMEOUT_SECONDS,
        )

    async def _generate_realtime_assistant(
        self,
        model: str,
        messages: tuple[ConversationMessage, ...],
        max_output_tokens: int,
        enable_thinking: bool,
    ) -> AsyncIterator[str]:
        """Stream one token-bounded assistant response for a realtime turn."""

        resolved_model = await self._resolve_and_validate_text_model(ModelId(model))
        model_card = await self._get_running_model_card(resolved_model)
        request = ChatCompletionRequest(
            model=resolved_model,
            messages=[
                ChatCompletionMessage(role=role, content=content)
                for role, content in messages
            ],
            stream=True,
            max_tokens=max_output_tokens,
            enable_thinking=enable_thinking,
        )
        task_params = await chat_request_to_text_generation(
            request,
            model_card=model_card,
        )
        if self._extensions is not None and self._extensions.has_chat_middleware:
            task_params = await self._extensions.transform_chat_request(
                self._extension_context,
                task_params,
            )
        command = await self._send_text_generation_with_images(task_params)
        chunk_stream = self._tapped_text_stream(
            command.command_id,
            task_params.model,
            task_params=task_params,
        )
        async for chunk in chunk_stream:
            if isinstance(chunk, PrefillProgressChunk):
                continue
            if isinstance(chunk, ErrorChunk):
                raise RuntimeError(
                    chunk.error_message or "assistant model generation failed"
                )
            if isinstance(chunk, ToolCallChunk):
                raise RuntimeError(
                    "realtime automatic responses do not support tool calls"
                )
            if not chunk.is_thinking and chunk.text:
                yield chunk.text

    async def _open_realtime_speech_session(
        self,
        model: str,
        text: str,
        voice: str | None,
    ) -> CapabilityStreamSession:
        """Open one mounted TTS provider stream for a realtime response."""

        model_id, response_format = await self._validate_speech_synthesis_model(
            ModelId(model),
            AudioResponseFormat.Mp3,
            stream=True,
        )
        request = await self._apply_default_speech_voice(
            AudioSpeechRequest(
                model=str(model_id),
                input=text,
                voice=voice,
                response_format=response_format,
                stream=True,
            ),
            model_id,
        )
        payload: dict[str, object] = {
            "model": str(model_id),
            "text": text,
            "response_format": "mp3",
        }
        if request.voice is not None:
            payload["voice"] = request.voice
        return await self._stream_capability(
            self.node_id,
            TTS_CAPABILITY_DESCRIPTOR.id,
            TTS_CAPABILITY_DESCRIPTOR.version,
            descriptor_revision(TTS_CAPABILITY_DESCRIPTOR),
            payload,
            timeout_seconds=MAX_CALL_TIMEOUT_SECONDS,
        )

    async def _validate_realtime_response_config(
        self,
        config: RealtimeResponseConfig,
    ) -> None:
        """Validate mounted chat and TTS participants before accepting a session."""

        await self._resolve_and_validate_text_model(ModelId(config.model))
        if config.tts_model is None:
            return
        model_id, response_format = await self._validate_speech_synthesis_model(
            ModelId(config.tts_model),
            AudioResponseFormat.Mp3,
            stream=True,
        )
        await self._apply_default_speech_voice(
            AudioSpeechRequest(
                model=str(model_id),
                input="realtime participant readiness probe",
                voice=config.voice,
                response_format=response_format,
                stream=True,
            ),
            model_id,
        )

    async def realtime_transcription(
        self,
        websocket: WebSocket,
        model: Annotated[str, Query(min_length=1, max_length=512)],
    ) -> None:
        """Serve one OpenAI-compatible realtime transcription WebSocket.

        Args:
            websocket: Client WebSocket owned by this API node.
            model: Mounted realtime STT model selected in the URL query.

        Side effects:
            Opens one stable ``stt.realtime`` provider session and
            cancels it when the socket disconnects or reaches a terminal event.
        """

        await RealtimeTranscriptionBridge(
            websocket=websocket,
            model=model,
            open_session=self._open_realtime_transcription_session,
            generate_assistant=self._generate_realtime_assistant,
            open_speech_session=self._open_realtime_speech_session,
            validate_response_config=self._validate_realtime_response_config,
        ).serve()

    async def fabric_speech_chain(
        self,
        websocket: WebSocket,
        stt_model: Annotated[str, Query(min_length=1, max_length=512)],
    ) -> None:
        """Serve one typed Fabric speech composition WebSocket.

        Args:
            websocket: Client WebSocket owned by this API node.
            stt_model: Mounted realtime STT participant selected for the chain.

        Side effects:
            Composes the selected STT participant with optional mounted chat and
            TTS participants declared by ``session.update``. Media remains on
            bounded provider transports and disconnect cancels active work.
        """

        await RealtimeTranscriptionBridge(
            websocket=websocket,
            model=stt_model,
            open_session=self._open_realtime_transcription_session,
            generate_assistant=self._generate_realtime_assistant,
            open_speech_session=self._open_realtime_speech_session,
            validate_response_config=self._validate_realtime_response_config,
        ).serve()

    async def audio_speech_http(self, http_request: Request) -> Response:
        """Parse JSON or multipart speech requests into one strict core path."""

        content_type = _normalize_upload_content_type(
            http_request.headers.get("content-type")
        )
        if content_type == "application/json":
            try:
                request = AudioSpeechRequest.model_validate(await http_request.json())
            except (json.JSONDecodeError, ValidationError) as exc:
                detail = (
                    json.dumps(exc.errors(include_context=False))
                    if isinstance(exc, ValidationError)
                    else "Malformed JSON request body"
                )
                raise HTTPException(status_code=422, detail=detail) from exc
            return await self.audio_speech(request)
        if content_type != "multipart/form-data":
            raise HTTPException(
                status_code=415,
                detail="Speech requests require application/json or multipart/form-data",
            )
        form = await http_request.form()
        reference_value = form.get("reference_audio")
        if reference_value is not None and not isinstance(
            reference_value, StarletteUploadFile
        ):
            raise HTTPException(
                status_code=422,
                detail="Multipart `reference_audio` must be a file upload",
            )
        reference_audio = (
            reference_value
            if isinstance(reference_value, StarletteUploadFile)
            else None
        )
        payload: dict[str, object] = {
            key: value
            for key, value in form.multi_items()
            if key != "reference_audio" and isinstance(value, str)
        }
        try:
            for key in (
                "speed",
                "streaming_interval",
                "temperature",
                "top_p",
                "repetition_penalty",
            ):
                if key in payload:
                    payload[key] = float(cast(str, payload[key]))
            for key in ("top_k", "max_tokens", "seed"):
                if key in payload:
                    payload[key] = int(cast(str, payload[key]))
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="Multipart numeric fields must contain valid numbers",
            ) from exc
        if "stream" in payload:
            stream_value = cast(str, payload["stream"]).strip().lower()
            if stream_value not in {"true", "false"}:
                raise HTTPException(
                    status_code=422,
                    detail="Multipart `stream` must be true or false",
                )
            payload["stream"] = stream_value == "true"
        try:
            request = AudioSpeechRequest.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=json.dumps(exc.errors(include_context=False)),
            ) from exc
        return await self.audio_speech(
            request,
            reference_audio_file=reference_audio,
        )

    async def audio_speech(
        self,
        request: AudioSpeechRequest,
        *,
        reference_audio_file: StarletteUploadFile | None = None,
    ) -> Response:
        """OpenAI-compatible text-to-speech endpoint."""

        if request.streaming_interval is not None and not request.stream:
            raise HTTPException(
                status_code=400,
                detail=("`streaming_interval` is only supported with `stream=true`"),
            )
        if request.reference_audio is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "`reference_audio` managed uploads are not supported yet; "
                    "server-local paths are never accepted"
                ),
            )
        if request.reference_text is not None and reference_audio_file is None:
            raise HTTPException(
                status_code=400,
                detail="`reference_text` is only supported with reference-audio TTS flows",
            )

        requested_response_format = (
            AudioResponseFormat.Mp3
            if request.stream and request.response_format is None
            else request.response_format
        )
        model_id, response_format = await self._validate_speech_synthesis_model(
            ModelId(request.model), requested_response_format, stream=request.stream
        )
        if reference_audio_file is not None and request.voice is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "`voice` cannot be combined with an uploaded `reference_audio`; "
                    "omit `voice` to use the request-scoped reference"
                ),
            )
        reference_voice_profile: str | None = None
        if reference_audio_file is None:
            request = await self._apply_default_speech_voice(request, model_id)
            reference_voice_profile = await self._bundled_reference_profile_for_voice(
                model_id,
                request.voice,
            )
        if request.stream and response_format not in _STREAMABLE_AUDIO_RESPONSE_FORMATS:
            supported = ", ".join(
                audio_format.value
                for audio_format in sorted(
                    _STREAMABLE_AUDIO_RESPONSE_FORMATS,
                    key=lambda audio_format: audio_format.value,
                )
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"`stream=true` supports only {supported} responses for now; "
                    f"requested {response_format.value}"
                ),
            )
        reference_audio: bytes | None = None
        reference_filename: str | None = None
        reference_content_type: str | None = None
        reference_sha256: str | None = None
        target_instance_id: InstanceId | None = None
        target_node: NodeId | None = None
        if reference_audio_file is not None:
            if not self._data_plane_zenoh or self._speech_media_packet_sender is None:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Reference audio requires the node-addressed Zenoh data plane; "
                        "broadcast fallback is not permitted for private media"
                    ),
                )
            target_instance_id, target_node = self._reference_tts_target(model_id)
            _validate_audio_upload_metadata(reference_audio_file)
            try:
                reference_audio = await _read_audio_upload(reference_audio_file)
            finally:
                await reference_audio_file.close()
            reference_filename = reference_audio_file.filename
            reference_content_type = _normalize_upload_content_type(
                reference_audio_file.content_type
            )
            reference_sha256 = hashlib.sha256(reference_audio).hexdigest()
        command_id, recv = await self._start_speech_synthesis(
            self._speech_synthesis_task_params(
                request,
                model_id,
                response_format,
                reference_voice_profile=reference_voice_profile,
                reference_audio_filename=reference_filename,
                reference_audio_content_type=reference_content_type,
                reference_audio_sha256=reference_sha256,
            ),
            target_instance_id=target_instance_id,
            target_node=target_node,
            reference_audio=reference_audio,
        )

        if request.stream:
            try:
                first_chunk = await self._receive_initial_audio_speech_chunk(
                    command_id, recv
                )
                if first_chunk.format != response_format:
                    await self._cancel_audio_speech_command(command_id)
                    raise HTTPException(
                        status_code=500,
                        detail="Speech synthesis returned an unexpected audio format",
                    )
                if (
                    response_format == AudioResponseFormat.Pcm
                    and first_chunk.sample_rate is None
                ):
                    await self._cancel_audio_speech_command(command_id)
                    raise HTTPException(
                        status_code=500,
                        detail="Raw PCM speech response did not declare a sample rate",
                    )
            except anyio.get_cancelled_exc_class():
                await self._cancel_audio_speech_command(command_id)
                await self._finalize_command_stream(
                    command_id,
                    cast(
                        dict[CommandId, Sender[object]],
                        self._audio_speech_queues,
                    ),
                )
                raise
            except HTTPException:
                await self._finalize_command_stream(
                    command_id,
                    cast(
                        dict[CommandId, Sender[object]],
                        self._audio_speech_queues,
                    ),
                )
                raise
            return StreamingResponse(
                self._stream_audio_speech_chunks(command_id, recv, first_chunk),
                media_type=_AUDIO_CONTENT_TYPES[response_format],
                headers={
                    "Content-Disposition": (
                        f"attachment; filename=speech.{response_format.value}"
                    ),
                    **(
                        {
                            "X-Audio-Sample-Rate": str(first_chunk.sample_rate),
                            "X-Audio-Channels": "1",
                            "X-Audio-Sample-Format": "s16le",
                        }
                        if response_format == AudioResponseFormat.Pcm
                        and first_chunk.sample_rate is not None
                        else {}
                    ),
                },
            )

        try:
            (
                audio_bytes,
                response_format,
                sample_rate,
            ) = await self._collect_audio_speech_chunks(command_id, recv)
            if response_format == AudioResponseFormat.Pcm and sample_rate is None:
                raise HTTPException(
                    status_code=500,
                    detail="Raw PCM speech response did not declare a sample rate",
                )
            return Response(
                content=audio_bytes,
                media_type=_AUDIO_CONTENT_TYPES[response_format],
                headers=(
                    {
                        "X-Audio-Sample-Rate": str(sample_rate),
                        "X-Audio-Channels": "1",
                        "X-Audio-Sample-Format": "s16le",
                    }
                    if response_format == AudioResponseFormat.Pcm
                    else {}
                ),
            )

        except anyio.get_cancelled_exc_class():
            await self._cancel_audio_speech_command(command_id)
            raise
        finally:
            await self._finalize_command_stream(
                command_id,
                cast(
                    dict[CommandId, Sender[object]],
                    self._audio_speech_queues,
                ),
            )

    async def _collect_audio_speech_chunks(
        self,
        command_id: CommandId,
        recv: Receiver[AudioChunk | ErrorChunk],
    ) -> tuple[bytes, AudioResponseFormat, int | None]:
        """Collect a non-streaming TTS response with a terminal-task backstop."""
        audio_parts: list[bytes] = []
        response_format: AudioResponseFormat | None = None
        sample_rate: int | None = None
        with recv as chunks:
            while True:
                chunk: AudioChunk | ErrorChunk | None = None
                with anyio.move_on_after(_STREAM_IDLE_TIMEOUT_SECONDS) as scope:
                    try:
                        chunk = await chunks.receive()
                    except (EndOfStream, ClosedResourceError) as exc:
                        if not self._command_task_is_terminal(command_id):
                            await self._cancel_audio_speech_command(command_id)
                        detail = (
                            "Speech synthesis stream closed before receiving "
                            "a terminal chunk"
                            if audio_parts
                            else "No speech audio response received"
                        )
                        raise HTTPException(status_code=500, detail=detail) from exc
                if scope.cancelled_caught:
                    if self._command_task_is_terminal(command_id):
                        detail = (
                            "Speech synthesis completed, but the final response "
                            "chunk was not received"
                            if audio_parts
                            else (
                                "Speech synthesis completed, but no audio response "
                                "was received"
                            )
                        )
                        raise HTTPException(
                            status_code=500,
                            detail=detail,
                        )
                    continue
                assert chunk is not None
                if isinstance(chunk, ErrorChunk):
                    raise HTTPException(
                        status_code=500,
                        detail=f"Speech synthesis failed: {chunk.error_message}",
                    )
                if response_format is not None and response_format != chunk.format:
                    await self._cancel_audio_speech_command(command_id)
                    raise HTTPException(
                        status_code=500,
                        detail="Speech synthesis changed format mid-response",
                    )
                audio_parts.append(_decode_audio_chunk_data(chunk))
                response_format = chunk.format
                if chunk.sample_rate is not None:
                    if sample_rate is not None and sample_rate != chunk.sample_rate:
                        await self._cancel_audio_speech_command(command_id)
                        raise HTTPException(
                            status_code=500,
                            detail="Speech synthesis changed sample rate mid-response",
                        )
                    sample_rate = chunk.sample_rate
                if chunk.finish_reason is not None:
                    break
        if not audio_parts:
            raise HTTPException(
                status_code=500, detail="No speech audio response received"
            )
        return b"".join(audio_parts), response_format, sample_rate

    async def _cancel_audio_speech_command(self, command_id: CommandId) -> None:
        """Cancel a speech synthesis command that cannot finish cleanly."""

        self._cancelled_command_ids.add(command_id)
        with anyio.CancelScope(shield=True):
            target_node = self._speech_media_targets.get(command_id)
            if target_node is not None and self._speech_media_packet_sender is not None:
                with contextlib.suppress(Exception):
                    await self._speech_media_packet_sender.send(
                        SpeechMediaPacket(
                            source_node=self.node_id,
                            target_node=target_node,
                            command_id=command_id,
                            sequence=0,
                            kind="cancelled",
                        )
                    )
            await self.command_sender.send(
                ForwarderCommand(
                    origin=self._system_id,
                    command=TaskCancelled(cancelled_command_id=command_id),
                )
            )

    async def _receive_initial_audio_speech_chunk(
        self,
        command_id: CommandId,
        recv: Receiver[AudioChunk | ErrorChunk],
    ) -> AudioChunk:
        """Receive the first TTS stream chunk before response headers commit."""
        while True:
            chunk: AudioChunk | ErrorChunk | None = None
            with anyio.move_on_after(_STREAM_IDLE_TIMEOUT_SECONDS) as scope:
                try:
                    chunk = await recv.receive()
                except (EndOfStream, ClosedResourceError) as exc:
                    raise HTTPException(
                        status_code=500, detail="No speech audio response received"
                    ) from exc
            if scope.cancelled_caught:
                if self._command_task_is_terminal(command_id):
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Speech synthesis completed, but no audio response "
                            "was received"
                        ),
                    )
                continue
            assert chunk is not None
            if isinstance(chunk, ErrorChunk):
                raise HTTPException(
                    status_code=500,
                    detail=f"Speech synthesis failed: {chunk.error_message}",
                )
            chunk_data = _decode_audio_chunk_data(chunk)
            if chunk.finish_reason is not None and len(chunk_data) == 0:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Speech synthesis completed, but no audio response was received"
                    ),
                )
            return chunk

    async def _stream_audio_speech_chunks(
        self,
        command_id: CommandId,
        recv: Receiver[AudioChunk | ErrorChunk],
        first_chunk: AudioChunk | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Yield TTS audio bytes as chunks arrive from the data plane."""
        received_audio = False
        expected_format = first_chunk.format if first_chunk is not None else None
        expected_pcm_sample_rate = (
            first_chunk.sample_rate
            if first_chunk is not None and first_chunk.format == AudioResponseFormat.Pcm
            else None
        )
        try:
            if first_chunk is not None:
                first_chunk_data = _decode_audio_chunk_data(first_chunk)
                if first_chunk_data:
                    received_audio = True
                    yield first_chunk_data
                if first_chunk.finish_reason is not None:
                    if not received_audio:
                        raise RuntimeError("No speech audio response received")
                    return
            with recv as chunks:
                while True:
                    chunk: AudioChunk | ErrorChunk | None = None
                    delay = _STREAM_IDLE_TIMEOUT_SECONDS if received_audio else None
                    with anyio.move_on_after(delay) as scope:
                        try:
                            chunk = await chunks.receive()
                        except (EndOfStream, ClosedResourceError) as exc:
                            if self._command_task_is_terminal(command_id):
                                if received_audio:
                                    return
                                raise RuntimeError(
                                    "No speech audio response received"
                                ) from exc
                            await self._cancel_audio_speech_command(command_id)
                            raise RuntimeError(
                                "Speech stream closed before receiving a terminal chunk"
                            ) from exc
                    if scope.cancelled_caught:
                        self._data_plane_observer.record_idle_timeout()
                        task_done = self._command_task_is_terminal(command_id)
                        logger.warning(
                            f"Speech stream for command {command_id} idle for "
                            f">{_STREAM_IDLE_TIMEOUT_SECONDS:g}s mid-stream; "
                            + (
                                "task already terminal — finishing (a final "
                                "data-plane chunk was likely dropped)."
                                if task_done
                                else "task still active — cancelling (a data-plane "
                                "chunk may have been dropped)."
                            )
                        )
                        if not task_done:
                            await self._cancel_audio_speech_command(command_id)
                        else:
                            return
                        raise RuntimeError(
                            "Speech stream stalled before receiving a terminal chunk"
                        )
                    assert chunk is not None
                    if isinstance(chunk, ErrorChunk):
                        raise RuntimeError(
                            f"Speech synthesis failed: {chunk.error_message}"
                        )
                    if expected_format is None:
                        expected_format = chunk.format
                    elif chunk.format != expected_format:
                        await self._cancel_audio_speech_command(command_id)
                        raise RuntimeError("Speech synthesis changed format mid-stream")
                    if (
                        expected_pcm_sample_rate is not None
                        and chunk.sample_rate != expected_pcm_sample_rate
                    ):
                        await self._cancel_audio_speech_command(command_id)
                        raise RuntimeError(
                            "Speech synthesis changed sample rate mid-stream"
                        )
                    chunk_data = _decode_audio_chunk_data(chunk)
                    if chunk_data:
                        received_audio = True
                        yield chunk_data
                    if chunk.finish_reason is not None:
                        if not received_audio:
                            raise RuntimeError("No speech audio response received")
                        return
            if not received_audio:
                raise RuntimeError("No speech audio response received")
        except anyio.get_cancelled_exc_class():
            command = TaskCancelled(cancelled_command_id=command_id)
            self._cancelled_command_ids.add(command_id)
            with anyio.CancelScope(shield=True):
                await self.command_sender.send(
                    ForwarderCommand(origin=self._system_id, command=command)
                )
            raise
        finally:
            await self._finalize_command_stream(
                command_id,
                cast(
                    dict[CommandId, Sender[object]],
                    self._audio_speech_queues,
                ),
            )

    async def audio_transcriptions(
        self,
        file: Annotated[
            UploadFile,
            File(
                description="Audio file to transcribe.",
            ),
        ],
        model: Annotated[
            str,
            Form(
                description="Mounted speech-to-text model id to serve.",
            ),
        ],
        language: Annotated[
            str | None,
            Form(
                description="Optional input language hint.",
            ),
        ] = None,
        prompt: Annotated[
            str | None,
            Form(
                description="Optional transcription prompt or context.",
            ),
        ] = None,
        response_format: Annotated[
            AudioTranscriptionResponseFormat,
            Form(
                description="Transcription response format.",
            ),
        ] = "json",
        temperature: Annotated[
            float | None,
            Form(
                description="Optional model-specific sampling temperature.",
            ),
        ] = None,
        stream: Annotated[
            bool,
            Form(
                description=(
                    "Return typed SSE transcript events as model deltas arrive. "
                    "With response_format=ndjson, retain progressive NDJSON framing."
                ),
            ),
        ] = False,
        max_tokens: Annotated[
            int | None,
            Form(
                description="Optional model-specific maximum token budget.",
            ),
        ] = None,
        chunk_duration: Annotated[
            float | None,
            Form(
                description="Optional model-specific audio chunk duration.",
            ),
        ] = None,
        frame_threshold: Annotated[
            int | None,
            Form(
                description="Optional model-specific frame threshold.",
            ),
        ] = None,
        context: Annotated[
            str | None,
            Form(
                description="Optional model-specific context text.",
            ),
        ] = None,
        prefill_step_size: Annotated[
            int | None,
            Form(
                description="Optional model-specific prefill step size.",
            ),
        ] = None,
        text: Annotated[
            str | None,
            Form(
                description="Optional model-specific text prefix.",
            ),
        ] = None,
        word_timestamps: Annotated[
            bool,
            Form(
                description="Whether to request word timestamp metadata when supported.",
            ),
        ] = False,
        timestamp_granularities: Annotated[
            str | None,
            Form(
                description="Comma-separated or JSON list of timestamp granularities.",
            ),
        ] = None,
    ) -> Response:
        """OpenAI-compatible speech-to-text endpoint."""

        if not model.strip():
            raise HTTPException(status_code=400, detail="`model` must not be empty")
        if response_format not in _AUDIO_TRANSCRIPTION_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported transcription response_format: {response_format}",
            )

        _validate_audio_upload_metadata(file)
        audio_bytes = await _read_audio_upload(file)
        model_id = (
            await self._validate_audio_transcription_model(ModelId(model), stream=True)
            if stream
            else await self._validate_audio_transcription_model(ModelId(model))
        )
        normalized_content_type = _normalize_upload_content_type(file.content_type)
        task_params = AudioTranscriptionTaskParams(
            model=model_id,
            filename=file.filename,
            content_type=normalized_content_type,
            audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
            language=language,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            chunk_duration=chunk_duration,
            frame_threshold=frame_threshold,
            context=context,
            prefill_step_size=prefill_step_size,
            text=text,
            word_timestamps=word_timestamps,
            timestamp_granularities=_parse_timestamp_granularities(
                timestamp_granularities
            ),
            stream=stream,
        )
        if stream:
            command_id, recv = await self._start_audio_transcription(
                task_params, audio_bytes
            )
            ndjson = response_format == "ndjson"
            body = self._stream_audio_transcription(
                command_id,
                recv,
                model_id=model_id,
                input_bytes=len(audio_bytes),
                ndjson=ndjson,
            )
            return StreamingResponse(
                body if ndjson else with_sse_keepalive(body),
                media_type=("application/x-ndjson" if ndjson else "text/event-stream"),
                headers=(
                    {}
                    if ndjson
                    else {
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    }
                ),
            )
        transcript_chunks = await self._execute_audio_transcription(
            task_params,
            audio_bytes,
        )
        return _build_audio_transcription_response(response_format, transcript_chunks)

    async def audio_translations(
        self,
        file: Annotated[
            UploadFile,
            File(description="Audio file to translate to English."),
        ],
        model: Annotated[
            str,
            Form(description="Mounted translation-capable speech model id."),
        ],
        language: Annotated[
            str | None,
            Form(description="Optional model-specific source language code."),
        ] = None,
        prompt: Annotated[
            str | None,
            Form(description="Optional translation prompt or context."),
        ] = None,
        response_format: Annotated[
            AudioTranscriptionResponseFormat,
            Form(description="Translation response format."),
        ] = "json",
        temperature: Annotated[
            float | None,
            Form(description="Optional model-specific sampling temperature."),
        ] = None,
    ) -> Response:
        """Translate uploaded speech into English with a mounted STT model."""

        if not model.strip():
            raise HTTPException(status_code=400, detail="`model` must not be empty")
        if response_format not in _AUDIO_TRANSCRIPTION_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported translation response_format: {response_format}",
            )
        model_id = await self._validate_audio_translation_model(
            ModelId(model), language
        )
        _validate_audio_upload_metadata(file)
        audio_bytes = await _read_audio_upload(file)
        transcript_chunks = await self._execute_audio_transcription(
            AudioTranscriptionTaskParams(
                model=model_id,
                filename=file.filename,
                content_type=_normalize_upload_content_type(file.content_type),
                audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
                language=language,
                prompt=prompt,
                temperature=temperature,
                translate_to_english=True,
            ),
            audio_bytes,
        )
        return _build_audio_transcription_response(response_format, transcript_chunks)

    async def audio_voices(
        self,
        model: Annotated[
            str,
            Query(description="Mounted text-to-speech model id."),
        ],
    ) -> AudioVoiceList:
        """Return stable built-in and bundled-reference voices for a TTS model."""

        if not model.strip():
            raise HTTPException(status_code=400, detail="`model` must not be empty")
        requested_model = ModelId(model)
        if not any(
            instance.shard_assignments.model_id == requested_model
            for instance in self.state.instances.values()
        ):
            raise HTTPException(
                status_code=404,
                detail=f"No instance found for model {requested_model}",
            )
        model_card = await self._get_running_model_card(requested_model)
        resolved = model_card.model_id
        profile = resolve_model_capability_profile(resolved, model_card=model_card)
        if not profile.supports_speech_synthesis:
            raise HTTPException(
                status_code=400,
                detail=f"Model {resolved} is not a text-to-speech model",
            )
        if (
            model_card.audio is None
            or model_card.audio.supports_voice_listing is not True
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Model {resolved} does not support voice listing",
            )
        catalog_by_id = {voice.id: voice for voice in model_card.audio.voice_catalog}
        return AudioVoiceList(
            data=tuple(
                AudioVoice(
                    id=voice,
                    name=catalog_by_id[voice].name if voice in catalog_by_id else voice,
                    model=str(resolved),
                    preferred_languages=(
                        catalog_by_id[voice].preferred_languages
                        if voice in catalog_by_id
                        else ()
                    ),
                    kind=(
                        "reference"
                        if voice in catalog_by_id
                        and catalog_by_id[voice].reference_profile is not None
                        else "builtin"
                    ),
                )
                for voice in model_card.audio.voices
            )
        )

    async def _execute_audio_transcription(
        self,
        task_params: AudioTranscriptionTaskParams,
        audio_bytes: bytes,
    ) -> list[TranscriptionChunk]:
        """Run the shared bounded batch STT command path for REST and Fabric."""

        command_id, recv = await self._start_audio_transcription(
            task_params, audio_bytes
        )
        try:
            transcript_chunks = await self._collect_transcription_chunks(
                command_id, recv
            )
            if not transcript_chunks:
                raise HTTPException(
                    status_code=500,
                    detail="No speech transcription response received",
                )
            return transcript_chunks
        except anyio.get_cancelled_exc_class():
            with anyio.CancelScope(shield=True):
                await self._cancel_audio_transcription_command(command_id)
            raise
        finally:
            await self._finalize_command_stream(
                command_id,
                cast(
                    dict[CommandId, Sender[object]],
                    self._audio_transcription_queues,
                ),
            )

    async def _start_audio_transcription(
        self,
        task_params: AudioTranscriptionTaskParams,
        audio_bytes: bytes,
    ) -> tuple[CommandId, Receiver[TranscriptionChunk | ErrorChunk]]:
        """Submit one bounded batch-file STT command before response admission."""

        audio_sha256 = hashlib.sha256(audio_bytes).hexdigest()
        total_chunks = max(
            1,
            (len(audio_bytes) + _SPEECH_MEDIA_CHUNK_BYTES - 1)
            // _SPEECH_MEDIA_CHUNK_BYTES,
        )
        params = task_params.model_copy(
            update={
                "total_input_chunks": total_chunks,
                "audio_sha256": audio_sha256,
            }
        )
        command = AudioTranscription(owner_node=self.node_id, task_params=params)
        command_id = command.command_id
        try:
            recv = self._open_stream_queue(self._audio_transcription_queues, command_id)
            self._stage_audio_transcription_media(command_id, params, audio_bytes)
            await self._send(command)
        except BaseException:
            self._speech_media_commands.discard(command_id)
            self._take_pending_speech_media(command_id)
            await self._finalize_command_stream(
                command_id,
                cast(
                    dict[CommandId, Sender[object]],
                    self._audio_transcription_queues,
                ),
            )
            raise
        return command_id, recv

    async def _stream_audio_transcription(
        self,
        command_id: CommandId,
        recv: Receiver[TranscriptionChunk | ErrorChunk],
        *,
        model_id: ModelId,
        input_bytes: int,
        ndjson: bool,
    ) -> AsyncIterator[str]:
        """Yield progressive STT output and own cancellation through cleanup."""

        text_parts: list[str] = []
        language: str | None = None
        segments: list[dict[str, str | int | float | bool | None]] = []
        sequence = 0
        terminal = False
        try:
            with recv as chunks:
                while True:
                    chunk: TranscriptionChunk | ErrorChunk | None = None
                    with anyio.move_on_after(_STREAM_IDLE_TIMEOUT_SECONDS) as scope:
                        try:
                            chunk = await chunks.receive()
                        except (EndOfStream, ClosedResourceError):
                            break
                    if scope.cancelled_caught:
                        if self._command_task_is_terminal(command_id):
                            error_event = AudioTranscriptionErrorEvent(
                                model=str(model_id),
                                sequence=sequence,
                                code="missing_terminal_chunk",
                                message=(
                                    "Speech transcription completed without a final "
                                    "response chunk"
                                ),
                            )
                            yield (
                                json.dumps(
                                    {"error": error_event.model_dump(mode="json")},
                                    separators=(",", ":"),
                                )
                                + "\n"
                                if ndjson
                                else _encode_audio_transcription_sse(error_event)
                            )
                            terminal = True
                            return
                        continue
                    assert chunk is not None
                    if isinstance(chunk, ErrorChunk):
                        error_event = AudioTranscriptionErrorEvent(
                            model=str(model_id),
                            sequence=sequence,
                            code="transcription_failed",
                            message=chunk.error_message,
                        )
                        yield (
                            json.dumps(
                                {"error": error_event.model_dump(mode="json")},
                                separators=(",", ":"),
                            )
                            + "\n"
                            if ndjson
                            else _encode_audio_transcription_sse(error_event)
                        )
                        terminal = True
                        return
                    if chunk.text:
                        text_parts.append(chunk.text)
                    if chunk.language is not None:
                        language = chunk.language
                    segments.extend(chunk.segments)
                    if ndjson:
                        yield (
                            json.dumps(
                                _transcription_chunk_payload(chunk),
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    elif chunk.text:
                        yield _encode_audio_transcription_sse(
                            AudioTranscriptionDeltaEvent(
                                model=str(model_id),
                                sequence=sequence,
                                delta=chunk.text,
                                language=chunk.language,
                                segment_index=chunk.segment_index,
                            )
                        )
                        sequence += 1
                    if chunk.finish_reason is None:
                        continue
                    if not ndjson:
                        complete_text = "".join(text_parts)
                        yield _encode_audio_transcription_sse(
                            AudioTranscriptionCompletedEvent(
                                model=str(model_id),
                                sequence=sequence,
                                text=complete_text,
                                language=language,
                                segments=tuple(segments),
                            )
                        )
                        sequence += 1
                        yield _encode_audio_transcription_sse(
                            AudioTranscriptionUsageEvent(
                                model=str(model_id),
                                sequence=sequence,
                                input_bytes=input_bytes,
                                output_characters=len(complete_text),
                            )
                        )
                    terminal = True
                    return
        finally:
            if not terminal and not self._command_task_is_terminal(command_id):
                with anyio.CancelScope(shield=True):
                    with anyio.move_on_after(1.0):
                        await self._cancel_audio_transcription_command(command_id)
            with anyio.CancelScope(shield=True):
                await self._finalize_command_stream(
                    command_id,
                    cast(
                        dict[CommandId, Sender[object]],
                        self._audio_transcription_queues,
                    ),
                )

    async def _collect_transcription_chunks(
        self,
        command_id: CommandId,
        recv: Receiver[TranscriptionChunk | ErrorChunk],
    ) -> list[TranscriptionChunk]:
        """Collect a non-streaming STT response with a terminal-task backstop."""
        transcript_chunks: list[TranscriptionChunk] = []
        with recv as chunks:
            while True:
                chunk: TranscriptionChunk | ErrorChunk | None = None
                with anyio.move_on_after(_STREAM_IDLE_TIMEOUT_SECONDS) as scope:
                    try:
                        chunk = await chunks.receive()
                    except (EndOfStream, ClosedResourceError):
                        break
                if scope.cancelled_caught:
                    if self._command_task_is_terminal(command_id):
                        raise HTTPException(
                            status_code=500,
                            detail=(
                                "Speech transcription completed, but the final "
                                "response chunk was not received"
                            ),
                        )
                    continue
                assert chunk is not None
                if isinstance(chunk, ErrorChunk):
                    raise HTTPException(
                        status_code=500,
                        detail=f"Speech transcription failed: {chunk.error_message}",
                    )
                transcript_chunks.append(chunk)
                if chunk.finish_reason is not None:
                    break
        return transcript_chunks

    async def restart_node(
        self,
        node_id: Annotated[
            NodeId | None,
            Query(
                description=(
                    "Legacy session-scoped runtime node ID. Prefer node_install_id "
                    "for operator actions."
                )
            ),
        ] = None,
        node_install_id: Annotated[
            UUID4 | None,
            Query(
                description=(
                    "Stable per-installation node identity resolved against current "
                    "live cluster truth."
                )
            ),
        ] = None,
    ) -> JSONResponse:
        """Restart the Skulk process on this or a remote node.

        Args:
            node_id: Legacy session-scoped runtime target. Omit for this API node.
            node_install_id: Stable installation identity resolved to one live
                runtime node before command dispatch.

        Returns:
            Accepted local or remote restart status and the resolved runtime node.

        Raises:
            HTTPException: Both target forms are supplied, or the stable target
                is missing or ambiguous in current live cluster truth.
        """
        if node_id is not None and node_install_id is not None:
            raise HTTPException(
                status_code=400,
                detail="provide either node_id or node_install_id, not both",
            )
        target = (
            self._runtime_node_for_installation(node_install_id)
            if node_install_id is not None
            else node_id or self.node_id
        )
        response_identity = (
            {"node_install_id": str(node_install_id)}
            if node_install_id is not None
            else {}
        )

        if target == self.node_id:
            from skulk.utils.restart import schedule_restart

            logger.info(
                "Node restart requested via API — scheduling process replacement"
            )
            scheduled = schedule_restart()
            if not scheduled:
                return JSONResponse(
                    {
                        "status": "restart_already_pending",
                        "node_id": str(self.node_id),
                        **response_identity,
                    },
                    status_code=409,
                )
            return JSONResponse(
                {
                    "status": "restarting",
                    "node_id": str(self.node_id),
                    **response_identity,
                }
            )

        # Remote restart — send command via download commands channel
        from skulk.shared.types.commands import RestartNode

        logger.info(f"Remote node restart requested for {target}")
        await self._send_download(RestartNode(target_node_id=target))
        return JSONResponse(
            {
                "status": "restart_sent",
                "node_id": str(target),
                **response_identity,
            }
        )

    def _runtime_node_for_installation(self, node_install_id: UUID4) -> NodeId:
        """Resolve one stable installation identity to its live runtime node.

        Args:
            node_install_id: Persistent per-host identity reported in node telemetry.

        Returns:
            The current session-scoped runtime node ID.

        Raises:
            HTTPException: No live node or more than one live node reports the
                requested installation identity.
        """
        live_nodes = self._live_node_timestamps()
        matches = [
            runtime_node_id
            for runtime_node_id, identity in self._telemetry_view.node_identities.items()
            if runtime_node_id in live_nodes
            and identity.node_install_id == node_install_id
        ]
        if not matches:
            raise HTTPException(
                status_code=404,
                detail="stable node identity is not present in current live cluster truth",
            )
        if len(matches) > 1:
            raise HTTPException(
                status_code=409,
                detail="stable node identity is ambiguous in current live cluster truth",
            )
        return matches[0]
