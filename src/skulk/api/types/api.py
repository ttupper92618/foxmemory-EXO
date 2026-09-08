import time
from collections.abc import Generator
from typing import Annotated, Any, Literal, final, get_args
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from skulk.shared.models.capabilities import ResolvedCapabilityProfile
from skulk.shared.models.model_cards import AudioResponseFormat, ModelCard, ModelId
from skulk.shared.models.registry import (
    RegistryAdvisory,
    RegistryCapabilityClaim,
    RegistryEngineSupportClaim,
)
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.text_generation import ReasoningEffort
from skulk.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from skulk.shared.types.worker.shards import Sharding, ShardMetadata
from skulk.store.installed_cards import InstalledArtifactRole, InstalledCardRecord
from skulk.store.staging_eviction import StagedModelInfo
from skulk.utils.pydantic_ext import CamelCaseModel

FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call", "error"
]
AudioTranscriptionResponseFormat = Literal[
    "json", "text", "verbose_json", "srt", "vtt", "ndjson"
]


class AudioTranscriptionDeltaEvent(BaseModel):
    """One progressive text delta from streaming speech transcription."""

    model_config = ConfigDict(frozen=True, strict=True)

    type: Literal["transcription.delta"] = "transcription.delta"
    model: str = Field(description="Mounted STT model serving the request.")
    sequence: int = Field(ge=0, description="Zero-based stream event sequence.")
    delta: str = Field(description="New transcript text not emitted previously.")
    language: str | None = Field(
        default=None, description="Detected or requested language when available."
    )
    segment_index: int | None = Field(
        default=None, description="Model-provided segment index when available."
    )


class AudioTranscriptionCompletedEvent(BaseModel):
    """Terminal successful transcript event for streaming STT."""

    model_config = ConfigDict(frozen=True, strict=True)

    type: Literal["transcription.completed"] = "transcription.completed"
    model: str = Field(description="Mounted STT model serving the request.")
    sequence: int = Field(ge=0, description="Zero-based stream event sequence.")
    text: str = Field(description="Complete transcript assembled from emitted deltas.")
    language: str | None = Field(
        default=None, description="Detected or requested language when available."
    )
    segments: tuple[dict[str, str | int | float | bool | None], ...] = Field(
        default=(), description="Normalized model-provided segment metadata."
    )


class AudioTranscriptionUsageEvent(BaseModel):
    """Terminal request-size usage event for streaming STT."""

    model_config = ConfigDict(frozen=True, strict=True)

    type: Literal["transcription.usage"] = "transcription.usage"
    model: str = Field(description="Mounted STT model serving the request.")
    sequence: int = Field(ge=0, description="Zero-based stream event sequence.")
    input_bytes: int = Field(ge=0, description="Accepted encoded upload byte count.")
    output_characters: int = Field(
        ge=0, description="Number of transcript characters emitted."
    )


class AudioTranscriptionErrorEvent(BaseModel):
    """Terminal typed failure event after streaming response admission."""

    model_config = ConfigDict(frozen=True, strict=True)

    type: Literal["transcription.error"] = "transcription.error"
    model: str = Field(description="Mounted STT model serving the request.")
    sequence: int = Field(ge=0, description="Zero-based stream event sequence.")
    code: str = Field(description="Stable machine-readable stream failure code.")
    message: str = Field(description="Human-readable failure detail.")


AudioTranscriptionStreamEvent = (
    AudioTranscriptionDeltaEvent
    | AudioTranscriptionCompletedEvent
    | AudioTranscriptionUsageEvent
    | AudioTranscriptionErrorEvent
)


class ErrorInfo(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: int


class ErrorResponse(BaseModel):
    error: ErrorInfo


@final
class ModelRequirements(BaseModel):
    """Advisory whole-model requirements from effective catalog metadata.

    A result reserves no resources and grants no execution authority. Joined
    node compatibility, current artifact identity and placement admission must
    still be checked before loading a runner.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    model_id: str = Field(description="Exact selectable model alias.")
    card_digest: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="SHA-256 of canonical card JSON excluding registry_snapshot_id.",
    )
    registry_card_id: str | None = Field(
        description="Signed card identity, when present."
    )
    context_tokens: int = Field(
        ge=1, description="Requested per-sequence context budget."
    )
    context_limit: int | None = Field(
        description="Advertised card context limit; null when unknown."
    )
    storage_bytes: int = Field(
        ge=0,
        description="Card weight bytes plus a declared GGUF vision projector; excludes runtime image, staging and caches.",
    )
    estimated_memory_bytes: int | None = Field(
        ge=0,
        description="Whole-model core footprint at the requested context; null when text KV geometry is unavailable or the task is not text generation.",
    )
    discrete_gpu_memory_fraction: float = Field(
        gt=0,
        le=1,
        description="Core fraction of discrete VRAM usable for the estimated footprint; do not sum GPU memory without supported sharding.",
    )
    unified_memory_fraction: float = Field(
        gt=0,
        le=1,
        description="Core Apple unified-memory working-set fraction of total RAM.",
    )
    compatible_backends: tuple[str, ...] = Field(
        description="Declared backend tags; live platform and runner constraints still apply."
    )
    engine_support: tuple[RegistryEngineSupportClaim, ...] = Field(
        description="Matching signed engine claims; status, exact build and hardware constraints remain authoritative."
    )
    incomplete_capabilities: tuple[str, ...] = Field(
        description="Artifact capabilities explicitly marked incomplete; these block admission."
    )
    estimate_only: Literal[True] = Field(
        default=True,
        description="Estimates are neither measured fit nor a placement or spending authorization.",
    )


class ModelListModel(BaseModel):
    """Public model-catalog entry returned by the models endpoints."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "skulk"
    # openwebui fields
    hugging_face_id: str = Field(default="")
    name: str = Field(default="")
    description: str = Field(default="")
    context_length: int = Field(default=0)
    tags: list[str] = Field(default=[])
    storage_size_megabytes: int = Field(default=0)
    supports_tensor: bool = Field(default=False)
    tasks: list[str] = Field(default=[])
    is_custom: bool = Field(default=False)
    system_role: Literal["steward"] | None = Field(
        default=None,
        description=(
            "Set for fabric-managed system entries (\"steward\" = the "
            "intelligent-fabric resident, addressable as a chat model but "
            "not user-placeable). Model pickers should badge or separate "
            "these rather than listing them as ordinary models."
        ),
    )
    family: str = Field(default="")
    quantization: str = Field(default="")
    base_model: str = Field(default="")
    artifact_repository: str = Field(
        default="",
        description=(
            "Upstream repository containing the active installed artifact when one "
            "exists, otherwise the effective catalog artifact; distinct from id when "
            "multiple registry cards select files from one repository."
        ),
    )
    artifact_file: str | None = Field(
        default=None,
        description="Exact selected repository file for file-addressed artifacts.",
    )
    registry_card_id: str | None = Field(
        default=None,
        pattern=r"^card_[a-z2-7]{52}$",
        description=(
            "Immutable signed identity of the active installed card when one exists, "
            "otherwise the effective catalog card; null for local cards."
        ),
    )
    registry_snapshot_id: str | None = Field(
        default=None,
        description="Signed snapshot that supplied this card, or null for local cards.",
    )
    registry_provenance: Literal["foxlight", "agent", "community"] | None = Field(
        default=None,
        description=(
            "Audited signed-registry origin, or null for bundled and custom cards."
        ),
    )
    registry_architecture: str | None = Field(
        default=None,
        description="Trusted open architecture identity from signed metadata.",
    )
    capability_claims: list[RegistryCapabilityClaim] = Field(
        default_factory=list,
        description="Open signed model/artifact capabilities independent of engines.",
    )
    engine_support: list[RegistryEngineSupportClaim] = Field(
        default_factory=list,
        description="Active signed engine/build claims matching this exact artifact.",
    )
    installed: bool = Field(
        default=False,
        description=(
            "Whether the authoritative cluster store, or the local node when the "
            "store has no record, has an active complete installed generation."
        ),
    )
    active_installed_identity: str | None = Field(
        default=None,
        description=(
            "Durable identity of the active cluster-store generation, falling back "
            "to the node-local generation when necessary."
        ),
    )
    installed_verification: (
        Literal["registry_verified", "local_legacy", "custom", "unresolved"] | None
    ) = Field(
        default=None,
        description="Evidence level binding the active card to local artifact bytes.",
    )
    current_registry_identity: str | None = Field(
        default=None,
        description="Current signed card identity for this alias, when available.",
    )
    update_available: bool = Field(
        default=False,
        description="Whether registry truth names a newer generation than the active installation.",
    )
    advisories: list[RegistryAdvisory] = Field(
        default_factory=list,
        description="Active signed warnings affecting this installed or current card.",
    )
    catalog_source: Literal["registry", "bundled", "custom"] = Field(
        default="bundled",
        description="Trust and precedence source for this catalog entry.",
    )
    remote_code_approval_required: bool = Field(
        default=False,
        deprecated=True,
        description=(
            "Deprecated compatibility field. Current cards are authorized by "
            "signed publication or explicit addition and return false."
        ),
    )
    remote_code_trust_identity: str | None = Field(
        default=None,
        pattern=r"^(?:card|local)_[a-z2-7]{52}$",
        deprecated=True,
        description=(
            "Deprecated compatibility identity for the retired secondary "
            "repository-code approval ceremony."
        ),
    )
    remote_code_approved_for_cluster: bool = Field(
        default=False,
        deprecated=True,
        description=(
            "Deprecated compatibility state for a legacy cluster approval."
        ),
    )
    remote_code_approved_on_this_node: bool = Field(
        default=False,
        deprecated=True,
        description=(
            "Deprecated compatibility alias for remote_code_approved_for_cluster; "
            "model trust is no longer node-local."
        ),
    )
    remote_code_automatically_trusted: bool = Field(
        default=False,
        description=(
            "Whether repository code is authorized by the card's signed "
            "publication, explicit addition, or bundled distribution boundary."
        ),
    )
    source_revision: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{40}$",
        description=(
            "Immutable Hugging Face source commit used for this model's "
            "qualified artifacts, or null when the card follows mutable main."
        ),
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="Coarse catalog capability labels such as text, vision, thinking, or embedding.",
    )
    reasoning: "ReasoningCapabilitySection | None" = Field(
        default=None,
        description="Optional declarative reasoning controls from the model card.",
    )
    modalities: "ModalitiesCapabilitySection | None" = Field(
        default=None,
        description="Optional declarative modality support details from the model card.",
    )
    audio: "AudioCapabilitySection | None" = Field(
        default=None,
        description="Optional declarative speech-serving metadata from the model card.",
    )
    tooling: "ToolingCapabilitySection | None" = Field(
        default=None,
        description="Optional declarative tool-calling metadata from the model card.",
    )
    runtime: "RuntimeCapabilitySection | None" = Field(
        default=None,
        description="Optional declarative runtime integration hints from the model card.",
    )
    resolved_capabilities: "ResolvedModelCapabilities | None" = Field(
        default=None,
        description=(
            "Normalized runtime capabilities resolved from the model card and "
            "model-family defaults for the default tool-free request path. "
            "Request-specific options such as tools may change some resolved values."
        ),
    )


class RemoteCodeApprovalView(BaseModel):
    """Cluster approval state for one immutable model-card identity."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    card_id: str = Field(
        pattern=r"^(?:card|local)_[a-z2-7]{52}$",
        description=(
            "Signed registry card ID or content-derived local identity for an "
            "unsigned/custom card."
        ),
    )
    approved_for_cluster: bool = Field(
        description=(
            "Whether cluster Settings permit the card to download or execute "
            "repository code on every node."
        )
    )
    approved_on_this_node: bool = Field(
        deprecated=True,
        description=(
            "Deprecated compatibility alias for approved_for_cluster; trust "
            "decisions are synchronized across the cluster."
        ),
    )


class ResolvedModelCapabilities(BaseModel):
    """Normalized runtime behavior that UI and API consumers can safely inspect."""

    family: str = Field(
        default="",
        description="Resolved model family used for runtime behavior decisions.",
    )
    supports_thinking: bool = Field(
        default=False,
        description="Whether the runtime expects the model to expose a reasoning or thinking mode.",
    )
    supports_thinking_toggle: bool = Field(
        default=False,
        description="Whether thinking can be explicitly enabled or disabled for requests.",
    )
    supports_thinking_budget: bool = Field(
        default=False,
        description="Whether the runtime expects the model to accept a thinking or reasoning budget control.",
    )
    default_reasoning_effort: ReasoningEffort = Field(
        default="medium",
        description="Reasoning effort used when thinking is enabled without an explicit effort override.",
    )
    disabled_reasoning_effort: ReasoningEffort = Field(
        default="none",
        description="Reasoning effort used when thinking is explicitly disabled.",
    )
    thinking_format: str = Field(
        default="none",
        description="Resolved reasoning marker format expected from this model family.",
    )
    supports_image_input: bool = Field(
        default=False,
        description="Whether the runtime should treat the model as accepting image inputs.",
    )
    supports_audio_input: bool = Field(
        default=False,
        description="Whether the runtime should treat the model as accepting audio inputs.",
    )
    supports_speech_synthesis: bool = Field(
        default=False,
        description="Whether the runtime should treat the model as a text-to-speech model.",
    )
    supports_transcription: bool = Field(
        default=False,
        description="Whether the runtime should treat the model as a speech-to-text model.",
    )
    supports_speech_translation: bool = Field(
        default=False,
        description="Whether the runtime should treat the model as supporting speech translation.",
    )
    supports_audio_output: bool = Field(
        default=False,
        description="Whether the runtime should expect this model to produce audio output.",
    )
    supports_realtime_audio: bool = Field(
        default=False,
        description="Whether the runtime should expect this model to expose realtime audio sessions.",
    )
    default_audio_response_format: str | None = Field(
        default=None,
        description="Default encoded audio response format for speech synthesis, when declared.",
    )
    audio_response_formats: list[str] = Field(
        default_factory=list,
        description="Encoded audio response formats the model can produce.",
    )
    supports_tool_calling: bool = Field(
        default=False,
        description="Whether the runtime expects the model to support structured tool calling.",
    )
    builtin_tools: list[str] = Field(
        default_factory=list,
        description="Builtin platform tool contracts that Skulk may expose to this model family.",
    )
    tool_call_format: str = Field(
        default="generic",
        description="Resolved tool-call output format family used for parsing.",
    )
    prompt_renderer: str = Field(
        default="tokenizer",
        description="Resolved prompt renderer strategy used to prepare requests for this model.",
    )
    output_parser: str = Field(
        default="generic",
        description="Resolved output parser strategy used to interpret model responses.",
    )
    supports_native_multimodal: bool = Field(
        default=False,
        description="Whether the runtime can use a native multimodal execution path for the model.",
    )

    @classmethod
    def from_profile(
        cls, profile: ResolvedCapabilityProfile
    ) -> "ResolvedModelCapabilities":
        return cls(
            family=profile.family,
            supports_thinking=profile.supports_thinking,
            supports_thinking_toggle=profile.supports_thinking_toggle,
            supports_thinking_budget=profile.supports_thinking_budget,
            default_reasoning_effort=profile.default_reasoning_effort,
            disabled_reasoning_effort=profile.disabled_reasoning_effort,
            thinking_format=profile.thinking_format.value,
            supports_image_input=profile.supports_image_input,
            supports_audio_input=profile.supports_audio_input,
            supports_speech_synthesis=profile.supports_speech_synthesis,
            supports_transcription=profile.supports_transcription,
            supports_speech_translation=profile.supports_speech_translation,
            supports_audio_output=profile.supports_audio_output,
            supports_realtime_audio=profile.supports_realtime_audio,
            default_audio_response_format=(
                profile.default_audio_response_format.value
                if profile.default_audio_response_format is not None
                else None
            ),
            audio_response_formats=[
                response_format.value
                for response_format in profile.audio_response_formats
            ],
            supports_tool_calling=profile.supports_tool_calling,
            builtin_tools=[tool.value for tool in profile.builtin_tools],
            tool_call_format=profile.tool_call_format.value,
            prompt_renderer=profile.prompt_renderer.value,
            output_parser=profile.output_parser.value,
            supports_native_multimodal=profile.supports_native_multimodal,
        )


class ReasoningCapabilitySection(BaseModel):
    """Snake-case reasoning metadata exposed by the models API."""

    supports_toggle: bool | None = None
    supports_budget: bool | None = None
    format: str | None = None
    default_effort: ReasoningEffort | None = None
    disabled_effort: ReasoningEffort | None = None

    @classmethod
    def from_model_card(
        cls, model_card: ModelCard
    ) -> "ReasoningCapabilitySection | None":
        config = model_card.reasoning
        if config is None:
            return None
        return cls(
            supports_toggle=config.supports_toggle,
            supports_budget=config.supports_budget,
            format=config.format.value if config.format is not None else None,
            default_effort=config.default_effort,
            disabled_effort=config.disabled_effort,
        )


class ModalitiesCapabilitySection(BaseModel):
    """Snake-case modality metadata exposed by the models API."""

    supports_audio_input: bool | None = None
    supports_native_multimodal: bool | None = None

    @classmethod
    def from_model_card(
        cls, model_card: ModelCard
    ) -> "ModalitiesCapabilitySection | None":
        config = model_card.modalities
        if config is None:
            return None
        return cls(
            supports_audio_input=config.supports_audio_input,
            supports_native_multimodal=config.supports_native_multimodal,
        )


class AudioCapabilitySection(BaseModel):
    """Snake-case speech metadata exposed by the models API."""

    kind: str | None = Field(
        default=None,
        description="Speech serving kind declared by the card: tts or stt.",
    )
    default_response_format: str | None = Field(
        default=None,
        description="Default encoded audio response format for TTS requests.",
    )
    response_formats: list[str] = Field(
        default_factory=list,
        description="Encoded audio response formats declared for TTS requests.",
    )
    supports_streaming: bool | None = Field(
        default=None,
        description=(
            "Whether the card declares streaming speech support after "
            "runtime validation."
        ),
    )
    supports_realtime: bool | None = Field(
        default=None,
        description="Whether the model declares realtime audio session support.",
    )
    supports_voice_listing: bool | None = Field(
        default=None,
        description="Whether the model declares voice-listing support.",
    )
    default_voice: str | None = Field(
        default=None,
        description="Stable voice used when a request omits an explicit voice.",
    )
    voices: list[str] = Field(
        default_factory=list,
        description="Stable built-in or bundled-reference voice identifiers.",
    )
    supports_reference_audio: bool | None = Field(
        default=None,
        description="Whether the model accepts managed reference audio.",
    )
    supports_translation: bool | None = Field(
        default=None,
        description="Whether the model declares speech translation support.",
    )
    sample_rates: list[int] = Field(
        default_factory=list,
        description="Declared input or output sample rates in hertz.",
    )

    @classmethod
    def from_model_card(cls, model_card: ModelCard) -> "AudioCapabilitySection | None":
        config = model_card.audio
        if config is None:
            return None
        return cls(
            kind=config.kind.value if config.kind is not None else None,
            default_response_format=(
                config.default_response_format.value
                if config.default_response_format is not None
                else None
            ),
            response_formats=[item.value for item in config.response_formats],
            supports_streaming=config.supports_streaming,
            supports_realtime=config.supports_realtime,
            supports_voice_listing=config.supports_voice_listing,
            default_voice=config.default_voice,
            voices=list(config.voices),
            supports_reference_audio=config.supports_reference_audio,
            supports_translation=config.supports_translation,
            sample_rates=list(config.sample_rates),
        )


class ToolingCapabilitySection(BaseModel):
    """Snake-case tool-calling metadata exposed by the models API."""

    supports_tool_calling: bool | None = None
    builtin_tools: list[str] | None = None
    tool_call_format: str | None = None

    @classmethod
    def from_model_card(
        cls, model_card: ModelCard
    ) -> "ToolingCapabilitySection | None":
        config = model_card.tooling
        if config is None:
            return None
        return cls(
            supports_tool_calling=config.supports_tool_calling,
            builtin_tools=(
                [tool.value for tool in config.builtin_tools]
                if config.builtin_tools is not None
                else None
            ),
            tool_call_format=(
                config.tool_call_format.value
                if config.tool_call_format is not None
                else None
            ),
        )


class RuntimeCapabilitySection(BaseModel):
    """Snake-case runtime metadata exposed by the models API."""

    prompt_renderer: str | None = None
    output_parser: str | None = None
    mtp_sidecar_repo: str | None = Field(
        default=None,
        description=(
            "Repo of this model's MTP sidecar (prediction heads), when it "
            "declares one. The sidecar is a companion loaded alongside the base "
            "model, not an independently placeable model. Lets clients mark the "
            "sidecar repo as a companion rather than a launchable entry."
        ),
    )
    mtp_sidecar_revision: str | None = Field(
        default=None,
        description="Immutable commit of the MTP sidecar repository.",
    )
    assistant_model_repo: str | None = Field(
        default=None,
        description=(
            "Repo of this model's speculative-decoding assistant (drafter), when "
            "it declares one. A companion loaded with the base model, not "
            "independently placeable."
        ),
    )
    assistant_model_revision: str | None = Field(
        default=None,
        description="Immutable commit of the assistant-model repository.",
    )
    served_spec_draft_repo: str | None = Field(
        default=None,
        description=(
            "Repo of this model's served-engine draft GGUF, when it declares a "
            "separate one. A companion loaded with the base model, not "
            "independently placeable."
        ),
    )
    served_spec_draft_revision: str | None = Field(
        default=None,
        description="Immutable commit of the served-engine draft repository.",
    )
    vllm_spec_draft_repo: str | None = Field(
        default=None,
        description=(
            "Repo of this model's vLLM speculative-decoding drafter, when it "
            "declares one. A companion loaded with the base model, not "
            "independently placeable."
        ),
    )
    vllm_spec_draft_revision: str | None = Field(
        default=None,
        description="Immutable commit of the vLLM drafter repository.",
    )

    @classmethod
    def from_model_card(
        cls, model_card: ModelCard
    ) -> "RuntimeCapabilitySection | None":
        config = model_card.runtime
        if config is None:
            return None
        return cls(
            prompt_renderer=(
                config.prompt_renderer.value
                if config.prompt_renderer is not None
                else None
            ),
            output_parser=(
                config.output_parser.value if config.output_parser is not None else None
            ),
            mtp_sidecar_repo=config.mtp_sidecar_repo,
            mtp_sidecar_revision=config.mtp_sidecar_revision,
            assistant_model_repo=config.assistant_model_repo,
            assistant_model_revision=config.assistant_model_revision,
            served_spec_draft_repo=config.served_spec_draft_repo,
            served_spec_draft_revision=config.served_spec_draft_revision,
            vllm_spec_draft_repo=config.vllm_spec_draft_repo,
            vllm_spec_draft_revision=config.vllm_spec_draft_revision,
        )


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelListModel]


class ChatCompletionMessageText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatCompletionMessageImageUrl(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: dict[str, str]  # {"url": "data:image/png;base64,..."}


ChatCompletionContentPart = ChatCompletionMessageText | ChatCompletionMessageImageUrl


class ToolCallItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    index: int | None = None
    type: Literal["function"] = "function"
    function: ToolCallItem


class WebSearchToolRequest(BaseModel):
    """Request body for the generic web-search tool endpoint."""

    query: str = Field(
        min_length=1,
        description="Natural-language search query to execute.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of search results to return.",
    )


class WebSearchResult(BaseModel):
    """One structured result returned by the web-search tool."""

    title: str = Field(description="Human-readable page title.")
    url: str = Field(description="Canonical result URL.")
    snippet: str = Field(description="Short result snippet suitable for tool context.")


class WebSearchToolResponse(BaseModel):
    """Structured response returned by the web-search tool endpoint."""

    query: str = Field(description="Original search query.")
    results: list[WebSearchResult] = Field(
        default_factory=list,
        description="Structured search results ordered by provider relevance.",
    )
    provider: str = Field(
        description="Backend provider implementation that produced the results."
    )


class OpenUrlToolRequest(BaseModel):
    """Request body for the generic URL-open tool endpoint."""

    url: str = Field(
        min_length=1,
        description="HTTP or HTTPS URL to inspect.",
    )


class OpenUrlToolResponse(BaseModel):
    """Structured response returned by the generic URL-open tool endpoint."""

    url: str = Field(description="Original URL requested by the caller.")
    final_url: str = Field(description="Final URL after redirects were followed.")
    title: str | None = Field(
        default=None,
        description="Best-effort page title when one could be determined.",
    )
    status_code: int = Field(
        description="HTTP response status code observed for the final response."
    )
    content_type: str | None = Field(
        default=None,
        description="Normalized response Content-Type when the server provided one.",
    )
    provider: str = Field(
        description="Backend provider implementation that produced the result."
    )


class ExtractPageToolRequest(BaseModel):
    """Request body for the generic page-extraction tool endpoint."""

    url: str = Field(
        min_length=1,
        description="HTTP or HTTPS URL to fetch and extract readable text from.",
    )
    max_chars: int = Field(
        default=12000,
        ge=500,
        le=50000,
        description="Maximum number of characters of extracted text to return.",
    )


class ExtractPageToolResponse(BaseModel):
    """Structured response returned by the generic page-extraction tool endpoint."""

    url: str = Field(description="Original URL requested by the caller.")
    final_url: str = Field(description="Final URL after redirects were followed.")
    title: str | None = Field(
        default=None,
        description="Best-effort page title when one could be determined.",
    )
    text: str = Field(
        description="Readable extracted text content from the fetched page."
    )
    truncated: bool = Field(
        description="Whether the extracted text was truncated to satisfy the max_chars limit."
    )
    provider: str = Field(
        description="Backend provider implementation that produced the result."
    )


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool", "function"]
    content: (
        str | ChatCompletionContentPart | list[ChatCompletionContentPart] | None
    ) = None
    reasoning_content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None


class BenchChatCompletionMessage(ChatCompletionMessage):
    pass


class TopLogprobItem(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobsContentItem(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprobItem]


class Logprobs(BaseModel):
    content: list[LogprobsContentItem] | None = None


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0
    audio_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails
    completion_tokens_details: CompletionTokensDetails


class StreamingChoiceResponse(BaseModel):
    index: int
    delta: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None


class ChatCompletionResponse(BaseModel):
    """One complete, non-streaming chat completion.

    The `object` discriminator is the first field strict OpenAI clients check,
    so streaming must not reuse this model: see `ChatCompletionChunkResponse`.
    """

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage | None = None
    service_tier: str | None = None


class ChatCompletionChunkResponse(BaseModel):
    """One frame of a streaming chat completion.

    Identical to `ChatCompletionResponse` except for the two things the OpenAI
    streaming format requires to differ: the `object` discriminator is
    `chat.completion.chunk`, and every choice carries a `delta` rather than a
    complete `message`.

    These were one model until the external-API compatibility suite caught the
    streaming path emitting the non-streaming discriminator. Lenient clients
    read `choices[0].delta` and never noticed; strict ones, including the
    Vercel AI SDK's openai-compatible provider, reject the stream outright.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamingChoiceResponse]
    usage: Usage | None = None
    service_tier: str | None = None


class GenerationStats(BaseModel):
    prompt_tps: float
    generation_tps: float
    prompt_tokens: int
    generation_tokens: int
    peak_memory_usage: Memory
    # Runner-reported ground truth for the performance-envelope tap (#596). A
    # served runner stamps these from its own state so the envelope is attributed
    # to the serving instance with the true in-flight concurrency, immune to which
    # API node dispatched the request. None for engines that do not report them
    # (in-process MLX / llama.cpp), where the API falls back to its own
    # outstanding-request count.
    serving_node: str | None = None
    """Node id of the node whose runner produced this generation (for the API to
    resolve the hardware class from telemetry)."""
    serving_backend: str | None = None
    """The resolved engine+backend tag the runner actually ran (e.g. vllm-cuda);
    can differ per node on a heterogeneous cluster, so it must come from the
    serving instance, not the model."""
    in_flight_at_admission: int | None = None
    """Requests this instance's runner was serving when this one started (>= 1)."""
    serving_batches: bool | None = None
    """Whether the serving engine decodes concurrent requests together (its
    configured max concurrency > 1), so aggregate throughput scales with
    concurrency. Distinguishes a parallel llama-server from a serial one."""

    def redacted_for_client(self) -> "GenerationStats":
        """Return a copy with all internal runner-attribution fields cleared.

        Ordinary generation APIs must not expose live runner topology or
        admission state. Client-facing serializers call this method after the
        API's internal performance-envelope tap has consumed the full runner
        attribution.
        """
        return self.model_copy(
            update={
                "serving_node": None,
                "serving_backend": None,
                "in_flight_at_admission": None,
                "serving_batches": None,
            }
        )

    def redacted_for_benchmark_client(self) -> "GenerationStats":
        """Return qualification statistics without identifying attribution.

        The explicit benchmark API retains task-local batching mode and
        admission width so a black-box qualification client can prove that
        configured concurrency was exercised. Node identity and backend
        selection remain private cluster topology.
        """
        return self.model_copy(
            update={
                "serving_node": None,
                "serving_backend": None,
            }
        )


class ImageGenerationStats(BaseModel):
    seconds_per_step: float
    total_generation_time: float

    num_inference_steps: int
    num_images: int

    image_width: int
    image_height: int

    peak_memory_usage: Memory


class NodePowerStats(BaseModel, frozen=True):
    node_id: NodeId
    samples: int
    avg_sys_power: float


class PowerUsage(BaseModel, frozen=True):
    elapsed_seconds: float
    nodes: list[NodePowerStats]
    total_avg_sys_power_watts: float
    total_energy_joules: float


class BenchChatCompletionResponse(ChatCompletionResponse):
    generation_stats: GenerationStats | None = None
    power_usage: PowerUsage | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "messages": [{"role": "user", "content": "Hello from Skulk"}],
                "stream": False,
                "temperature": 0.7,
            }
        }
    )

    model: ModelId
    frequency_penalty: float | None = None
    messages: list[ChatCompletionMessage]
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tools: list[dict[str, Any]] | None = None
    reasoning_effort: ReasoningEffort | None = None
    enable_thinking: bool | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    repetition_context_size: int | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None


class BenchChatCompletionRequest(ChatCompletionRequest):
    pass


class AddCustomModelParams(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "bartowski/my-custom-model-GGUF",
                "gguf_file": "my-custom-model-Q4_K_M.gguf",
                "source_revision": "0123456789abcdef0123456789abcdef01234567",
            }
        }
    )

    model_id: ModelId
    gguf_file: str | None = Field(
        default=None,
        description=(
            "Exact repo-relative GGUF file identifying the quant to select when "
            "adding a multi-quant repository. Split weights are normalized to "
            "their first shard for backend loading. Omit for non-GGUF "
            "repositories or default selection."
        ),
    )
    source_revision: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{40}$",
        description=(
            "Immutable Hugging Face commit to inspect and persist on the custom "
            "card. When omitted, Skulk resolves main once and persists the "
            "returned immutable commit."
        ),
    )


class AddExactCustomModelCardParams(BaseModel):
    """Operator-supplied exact card retained as unsigned local model truth."""

    model_config = ConfigDict(frozen=True, strict=True)

    model_card: ModelCard = Field(
        description=(
            "Complete exact model card to install temporarily or permanently as "
            "an unsigned custom card. Skulk removes any claimed registry trust "
            "metadata before persisting it."
        )
    )


class DeleteExactCustomModelCardParams(BaseModel):
    """Exact temporary card a qualification service intends to remove."""

    model_config = ConfigDict(frozen=True, strict=True)

    model_card: ModelCard = Field(
        description=(
            "Complete candidate card originally supplied to /models/add-card. "
            "Skulk applies the same unsigned-card normalization and deletes only "
            "when that exact temporary card still owns the alias."
        )
    )


class HuggingFaceSearchResult(BaseModel):
    id: str
    author: str = ""
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""
    tags: list[str] = Field(default_factory=list)
    matched_file: str | None = Field(
        default=None,
        description=(
            "Exact repo-relative GGUF path matched by a filename search, or null "
            "for ordinary repository search results."
        ),
    )
    pipeline_tag: str | None = Field(
        default=None,
        description="Hugging Face task tag (for example text-generation), when declared.",
    )
    library_name: str | None = Field(
        default=None,
        description="Framework the repository targets (transformers, diffusers, mlx, gguf).",
    )
    gated: bool = Field(
        default=False,
        description=(
            "True when downloading requires accepting the repository's license "
            "on Hugging Face and presenting an access token."
        ),
    )
    license: str | None = Field(
        default=None,
        description="License identifier from the model card, when declared.",
    )
    param_count: int | None = Field(
        default=None,
        description=(
            "Total parameter count reported by the repository's safetensors or "
            "GGUF metadata, when available."
        ),
    )
    total_file_size: int | None = Field(
        default=None,
        description="Exact total artifact bytes reported by GGUF metadata, when available.",
    )
    context_length: int | None = Field(
        default=None,
        description="Model context window reported by GGUF metadata, when available.",
    )
    base_model_repo: str | None = Field(
        default=None,
        description="Parent repository this model derives from, when tagged.",
    )
    base_model_relation: str | None = Field(
        default=None,
        description="How this model derives from its parent: finetune, quantized, merge, or adapter.",
    )
    arxiv_ids: list[str] = Field(
        default_factory=list,
        description="arXiv paper identifiers tagged on the repository.",
    )
    languages: list[str] = Field(
        default_factory=list,
        description="ISO 639-1 language tags declared on the repository.",
    )
    architecture: str | None = Field(
        default=None,
        description="Model architecture from repository config or GGUF metadata.",
    )


class GgufQuantOption(BaseModel):
    """One downloadable quantization of a GGUF repository."""

    model_config = ConfigDict(frozen=True, strict=True)

    gguf_file: str = Field(
        description="Repo-relative first shard of the quant's group; pin this to download it."
    )
    label: str = Field(description="Human quant label, e.g. Q4_K_M or UD-Q2_K_XL.")
    total_bytes: int = Field(
        description="Exact total bytes of the quant's shard group."
    )
    shard_count: int = Field(description="Number of GGUF shards in the group.")


class GgufQuantOptions(BaseModel):
    """Quantization inventory for one GGUF repository."""

    model_config = ConfigDict(frozen=True, strict=True)

    model_id: str
    options: list[GgufQuantOption] = Field(default_factory=list)


class HuggingFaceCardSummary(BaseModel):
    """Prose summary extracted from a Hugging Face model card README."""

    model_config = ConfigDict(frozen=True, strict=True)

    model_id: str
    summary: str = Field(
        description="First prose paragraphs of the model card, markup stripped; empty when the card has no usable prose.",
    )


class StoreDownloadRequest(BaseModel):
    """Optional artifact selection for a shared-store model download."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    gguf_file: str | None = Field(
        default=None,
        min_length=1,
        max_length=2048,
        description=(
            "Exact repo-relative GGUF file whose shard group the store should "
            "download. Omit to use the repository's default quant selection."
        ),
    )
    extra_gguf_files: list[
        Annotated[str, Field(min_length=1, max_length=2048)]
    ] = Field(
        default_factory=list,
        description=(
            "Same-repository companion GGUF paths to fetch with the selected "
            "base quant."
        ),
    )
    source_revision: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{40}$",
        description=(
            "Immutable Hugging Face commit to download. Omit to resolve the "
            "repository's mutable main branch."
        ),
    )
    registry_card_id: str | None = Field(
        default=None,
        pattern=r"^card_[a-z2-7]{52}$",
        description=(
            "Optional immutable signed card identity. Omit to select the current "
            "card for the model alias."
        ),
    )
    artifact_bundle_id: str | None = Field(
        default=None,
        pattern=r"^bundle_[a-z2-7]{52}$",
        description=(
            "Optional immutable v2 artifact-bundle identity. When supplied, "
            "the selected local card and canonical store must both match it."
        ),
    )
    source_repository: str | None = Field(
        default=None,
        min_length=3,
        max_length=512,
        pattern=r"^[^/]+/.+$",
        description=(
            "Upstream owner/repository containing the bytes when model_id is "
            "a distinct store alias."
        ),
    )
    owner_model_id: str | None = Field(
        default=None,
        min_length=3,
        max_length=512,
        pattern=r"^[^/]+/.+$",
        description="Owning base-model alias for a companion artifact.",
    )
    owner_registry_card_id: str | None = Field(
        default=None,
        pattern=r"^card_[a-z2-7]{52}$",
        description="Immutable signed identity of the owning base card.",
    )
    artifact_role: InstalledArtifactRole = Field(
        default="base",
        description="Base or declared companion role retained in installed truth.",
    )


class StoreDownloadResponse(CamelCaseModel):
    """Current state returned after requesting a canonical-store download."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    model_id: str | None = Field(
        default=None,
        description="Store artifact alias accepted for download, when available.",
    )
    source_revision: str | None = Field(
        default=None,
        description="Immutable source commit selected for this transfer, or null for mutable main.",
    )
    status: str = Field(
        description="Current store transfer state, or error when the store rejected the request."
    )
    progress: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Completed transfer fraction when the store supplied one.",
    )
    error: str | None = Field(
        default=None,
        description="Operator-readable store error when status is error.",
    )


class CachedArtifactLocation(BaseModel):
    """One node where an exact store artifact generation is available."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    node_id: str = Field(description="Fabric node retaining this exact generation.")
    complete: bool = Field(description="Whether the node reported a complete manifest.")
    installed_identity: str = Field(description="Durable installed generation identity.")
    bytes: int = Field(ge=0, description="Artifact bytes retained on the node.")
    last_use_epoch_seconds: float = Field(
        ge=0,
        description="Unix time of the node's most recent artifact use.",
    )
    in_use: bool = Field(description="Whether a live runner currently depends on the replica.")
    location_kind: Literal["store_local", "node_cache"] = Field(
        default="node_cache",
        description="Whether the bytes are canonical-store-local or a node cache."
    )


class CacheInventoryStatus(BaseModel):
    """Freshness and coverage of telemetry-derived fleet artifact availability."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    state: Literal["syncing", "current", "degraded", "unavailable"] = Field(
        description="Whether current node availability is complete enough to rely on."
    )
    observed_nodes: int = Field(
        ge=0,
        description=(
            "Live nodes with a usable artifact-inventory reading, including stale "
            "readings retained as partial truth."
        ),
    )
    expected_nodes: int = Field(
        ge=0,
        description="Nodes currently expected from live cluster topology.",
    )
    store_nodes: list[str] = Field(
        default_factory=list,
        description=(
            "Live nodes currently advertising the authoritative canonical-store role."
        ),
    )


class StoreRegistryEntry(BaseModel):
    """Canonical store entry enriched with fleet cache and registry status."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    model_id: str = Field(description="Canonical store artifact alias.")
    store_path: str = Field(description="Artifact directory relative to the store root.")
    files: list[str] = Field(description="Registered files relative to the artifact directory.")
    downloaded_at: str = Field(description="ISO 8601 UTC registration time.")
    total_bytes: int = Field(ge=0, description="Registered artifact byte total.")
    source_revision: str | None = Field(
        default=None,
        description="Immutable Hugging Face source commit, or null for mutable main.",
    )
    source_repository: str | None = Field(
        default=None,
        description="Upstream byte repository when different from the store alias.",
    )
    repo_has_projector: bool | None = Field(
        default=None,
        description="Whether the registered repository contains a multimodal projector.",
    )
    installed_card: InstalledCardRecord | None = Field(
        default=None,
        description="Complete durable card, artifact identity, ownership, and manifest.",
    )

    cached_on_nodes: list[CachedArtifactLocation] = Field(
        default_factory=list,
        description="Complete replicas of this installed identity reported across the fleet.",
    )
    current_registry_identity: str | None = Field(
        default=None,
        description="Current signed card identity for the owning model alias.",
    )
    installed_not_current: bool = Field(
        description="Whether the installed generation is absent from or superseded by the current registry."
    )
    update_available: bool = Field(
        description="Whether a different signed generation is available for installation."
    )
    advisories: list[RegistryAdvisory] = Field(
        default_factory=list,
        description="Active signed warn-only advisories affecting the installed or current card.",
    )
    reconciliation_state: Literal[
        "idle", "scanning", "importing", "complete", "failed"
    ] = Field(description="Current fleet reconciliation state.")
    last_verified_at: str | None = Field(
        default=None,
        description="ISO 8601 UTC completion time of the latest reconciliation pass.",
    )


class StoreRegistryResponse(BaseModel):
    """Public model-store registry and fleet cache-placement view."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    entries: list[StoreRegistryEntry] = Field(
        default_factory=list,
        description="Canonical artifacts known to the authoritative model store.",
    )
    cache_inventory: CacheInventoryStatus = Field(
        description="Telemetry freshness and coverage for cached_on_nodes projections."
    )


class ArtifactExportRequest(BaseModel):
    """Request a short-lived capability for one exact staged artifact."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    installed_identity: str = Field(description="Installed generation to export.")
    manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Exact manifest digest the store intends to import.",
    )
    target_node_id: str = Field(description="Store node allowed to redeem the token.")


class ArtifactExportResponse(BaseModel):
    """Issued artifact export capability and immutable manifest."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    capability_token: str = Field(
        description=(
            "Opaque, single-purpose bearer capability used to redeem files from "
            "this exact artifact export before expiry."
        )
    )
    expires_at_epoch_seconds: float = Field(
        description="Unix epoch time after which the capability cannot be redeemed."
    )
    byte_ceiling: int = Field(
        ge=0,
        description=(
            "Maximum cumulative artifact bytes the capability permits the target "
            "node to read."
        ),
    )
    record: InstalledCardRecord = Field(
        description=(
            "Immutable installed-card record whose manifest and generation the "
            "capability exports."
        )
    )


class ReconciliationStatus(BaseModel):
    """Fleet cache-to-store reconciliation progress."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    state: Literal["idle", "scanning", "importing", "complete", "failed"] = Field(
        default="idle",
        description=(
            "Current pass state: idle before scheduling, scanning inventories, "
            "importing selected artifacts, complete after convergence, or failed "
            "when one or more required operations did not converge."
        ),
    )
    inventory_only: bool = Field(
        default=True,
        description=(
            "Whether the pass reports eligible artifacts without importing them."
        ),
    )
    scanned_nodes: int = Field(
        default=0,
        ge=0,
        description="Number of reachable node inventories included in the pass.",
    )
    discovered_artifacts: int = Field(
        default=0,
        ge=0,
        description=(
            "Distinct complete installed generations selected after replica "
            "deduplication."
        ),
    )
    imported_artifacts: int = Field(
        default=0,
        ge=0,
        description=(
            "Selected generations newly committed to the canonical store by this pass."
        ),
    )
    pending_imports: tuple[str, ...] = Field(
        default=(),
        description=(
            "Installed identities still absent from the canonical store, or all "
            "eligible identities during an inventory-only pass."
        ),
    )
    failures: tuple[str, ...] = Field(
        default=(),
        description="Operator-readable inventory or import failures from this pass.",
    )
    last_verified_at: str | None = Field(
        default=None,
        description=(
            "ISO 8601 UTC completion time of the latest finished reconciliation pass."
        ),
    )


class PlaceInstanceParams(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "sharding": "Pipeline",
                "instance_meta": "MlxRing",
                "min_nodes": 1,
                "excluded_nodes": [],
            }
        }
    )

    model_id: ModelId
    sharding: Sharding = Sharding.Pipeline
    instance_meta: InstanceMeta = InstanceMeta.MlxRing
    min_nodes: int = 1
    excluded_nodes: list[NodeId] = Field(
        default_factory=list,
        description=(
            "Optional. Node IDs the master should treat as if absent when "
            "scoring candidate cycles for this placement. Empty list = "
            "consider all nodes. Already-running instances on the listed "
            "nodes are not affected — exclusion is per-placement, not "
            "cluster-wide."
        ),
    )


class CreateInstanceParams(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "instance": {
                    "MlxRingInstance": {
                        "instanceId": "00000000-0000-0000-0000-000000000000",
                        "shardAssignments": {
                            "modelId": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                            "runnerToShard": {
                                "runner-1": {
                                    "PipelineShardMetadata": {
                                        "modelCard": {
                                            "modelId": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                                            "storageSize": {"inBytes": 2147483648},
                                            "nLayers": 32,
                                            "hiddenSize": 2048,
                                            "supportsTensor": False,
                                            "tasks": ["TextGeneration"],
                                        },
                                        "deviceRank": 0,
                                        "worldSize": 1,
                                        "startLayer": 0,
                                        "endLayer": 32,
                                        "nLayers": 32,
                                    }
                                }
                            },
                            "nodeToRunner": {"node-1": "runner-1"},
                        },
                        "hostsByNode": {"node-1": []},
                        "ephemeralPort": 52416,
                    }
                }
            }
        }
    )

    instance: Instance


class PlacementPreview(BaseModel):
    model_id: ModelId
    sharding: Sharding
    instance_meta: InstanceMeta
    instance: Instance | None = None
    # Keys are NodeId strings, values are additional bytes that would be used on that node
    memory_delta_by_node: dict[str, int] | None = None
    error: str | None = None
    error_code: Literal[
        "no_valid_placement",
        "placement_info_pending",
        "model_code_approval_required",
        "model_card_identity_mismatch",
    ] | None = Field(
        default=None,
        description=(
            "Stable placement failure category, or null for a launchable preview. "
            "model_code_approval_required is retained only for older nodes; "
            "current authorization policy does not emit it. Backend and "
            "hardware identifiers remain open strings elsewhere."
        ),
    )
    trust_requirement: str | None = Field(
        default=None,
        deprecated=True,
        description=(
            "Deprecated compatibility detail from the retired secondary "
            "model-approval ceremony; current previews return null."
        ),
    )
    compatibility_source: Literal["card", "signed_engine_support"] | None = Field(
        default=None,
        description="Truth source that admitted the selected backend, or null on error.",
    )
    support_claim_ids: list[str] = Field(
        default_factory=list,
        description="Active signed support claims applicable to this placement.",
    )
    compatibility_detail: str | None = Field(
        default=None,
        description="Operator-readable model, artifact, engine/build, or platform gap.",
    )
    alternative: bool = Field(
        default=False,
        description=(
            "True for a per-host alternative to the planner's ranked pick: a "
            "single-node placement on a host that passes admission but lost "
            "the ranking. On heterogeneous fleets the ranked winner (often "
            "the largest GPU) would otherwise hide every other valid host."
        ),
    )


class PlacementPreviewResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "previews": [
                    {
                        "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                        "sharding": "Pipeline",
                        "instance_meta": "MlxRing",
                        "instance": None,
                        "memory_delta_by_node": None,
                        "error": None,
                    }
                ]
            }
        }
    )

    previews: list[PlacementPreview]


class DeleteInstanceTaskParams(BaseModel):
    instance_id: str


class CreateInstanceResponse(BaseModel):
    message: str
    command_id: CommandId
    instance_id: InstanceId = Field(
        description="Exact placement identity created by the accepted command."
    )
    model_card: ModelCard


class DeleteInstanceResponse(BaseModel):
    message: str
    command_id: CommandId
    instance_id: InstanceId


class CancelCommandResponse(BaseModel):
    message: str
    command_id: CommandId


class AudioSpeechRequest(BaseModel):
    """OpenAI-compatible text-to-speech request payload."""

    model_config = ConfigDict(extra="forbid", strict=True)

    model: str = Field(description="Mounted text-to-speech model id to serve.")
    input: str = Field(
        min_length=1,
        description="Text to synthesize into speech.",
    )
    voice: str | None = Field(
        default=None,
        description=(
            "Model-specific native voice or bundled-reference profile when "
            "declared by the mounted card."
        ),
    )
    speed: float | None = Field(
        default=None,
        gt=0,
        description="Optional model-specific speaking speed multiplier.",
    )
    response_format: AudioResponseFormat | None = Field(
        default=None,
        description=(
            "Audio format to return, including raw PCM when supported. When "
            "omitted, Skulk uses the mounted model card default when declared "
            "and otherwise falls back to mp3."
        ),
    )
    stream: bool = Field(
        default=False,
        description=(
            "Whether to stream MP3 or raw PCM bytes as they are produced by the "
            "mounted text-to-speech model. The selected response_format must be "
            "declared by the model card, and the card must declare "
            "audio.supports_streaming=true."
        ),
    )
    streaming_interval: float | None = Field(
        default=None,
        gt=0,
        description="Requested streaming chunk interval in seconds for stream=true requests.",
    )
    instruct: str | None = Field(
        default=None,
        description="Optional model-specific style or instruction text.",
    )
    lang_code: str | None = Field(
        default=None,
        description="Optional language code passed to models that accept it.",
    )
    temperature: float | None = Field(
        default=None,
        description="Optional model-specific sampling temperature.",
    )
    top_p: float | None = Field(
        default=None,
        description="Optional nucleus sampling parameter.",
    )
    top_k: int | None = Field(
        default=None,
        ge=0,
        description="Optional top-k sampling parameter.",
    )
    repetition_penalty: float | None = Field(
        default=None,
        gt=0,
        description="Optional model-specific repetition penalty.",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional model-specific maximum generation token budget. When omitted, "
            "the speech runner supplies 4096 only to generators that explicitly "
            "declare this control."
        ),
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**32 - 1,
        description=(
            "Optional deterministic sampling seed applied immediately before "
            "this speech generation."
        ),
    )
    reference_audio: str | None = Field(
        default=None,
        description=(
            "Managed reference-audio id for voice conditioning. Arbitrary server "
            "filesystem paths are not accepted."
        ),
    )
    reference_text: str | None = Field(
        default=None,
        description="Transcript for the managed reference audio when supported.",
    )

    @field_validator("response_format", mode="before")
    @classmethod
    def _validate_response_format(
        cls, value: str | AudioResponseFormat | None
    ) -> AudioResponseFormat | None:
        if value is None:
            return None
        if isinstance(value, AudioResponseFormat):
            return value
        return AudioResponseFormat(value)


class AudioVoice(BaseModel, frozen=True):
    """One stable voice identifier exposed by a mounted TTS model."""

    id: str = Field(description="Model-specific voice identifier.")
    name: str = Field(description="Human-readable voice name.")
    model: str = Field(description="Mounted text-to-speech model id.")
    preferred_languages: tuple[str, ...] = Field(
        default=(),
        description=(
            "Ordered BCP 47 language tags for which the model card recommends "
            "this voice."
        ),
    )
    kind: Literal["builtin", "reference"] = Field(
        default="builtin",
        description=(
            "Voice source: a model-native preset or a bundled reference profile."
        ),
    )


class AudioVoiceList(BaseModel, frozen=True):
    """Voice catalog for one mounted text-to-speech model."""

    object: Literal["list"] = "list"
    data: tuple[AudioVoice, ...] = ()


ImageSize = Literal[
    "auto",
    "512x512",
    "768x768",
    "1024x768",
    "768x1024",
    "1024x1024",
    "1024x1536",
    "1536x1024",
]


def normalize_image_size(v: object) -> ImageSize:
    """Shared validator for ImageSize fields: maps None → "auto" and rejects invalid values."""
    if v is None:
        return "auto"
    if v not in get_args(ImageSize):
        raise ValueError(f"Invalid size: {v!r}. Must be one of {get_args(ImageSize)}")
    return v  # pyright: ignore[reportReturnType]


class AdvancedImageParams(BaseModel):
    seed: Annotated[int, Field(ge=0)] | None = None
    num_inference_steps: Annotated[int, Field(ge=1, le=100)] | None = None
    guidance: Annotated[float, Field(ge=1.0, le=20.0)] | None = None
    negative_prompt: str | None = None
    num_sync_steps: Annotated[int, Field(ge=1, le=100)] | None = None


class ImageGenerationTaskParams(BaseModel):
    prompt: str
    background: str | None = None
    model: str
    moderation: str | None = None
    n: int | None = 1
    output_compression: int | None = None
    output_format: Literal["png", "jpeg", "webp"] = "png"
    partial_images: int | None = 0
    quality: Literal["high", "medium", "low"] | None = "medium"
    response_format: Literal["url", "b64_json"] | None = "b64_json"
    size: ImageSize = "auto"
    stream: bool | None = False
    style: str | None = "vivid"
    user: str | None = None
    advanced_params: AdvancedImageParams | None = None
    # Internal flag for benchmark mode - set by API, preserved through serialization
    bench: bool = False

    @field_validator("size", mode="before")
    @classmethod
    def normalize_size(cls, v: object) -> ImageSize:
        return normalize_image_size(v)


class BenchImageGenerationTaskParams(ImageGenerationTaskParams):
    bench: bool = True


class ImageEditsTaskParams(BaseModel):
    """Internal task params for image-editing requests."""

    image_data: str = ""  # Base64-encoded image (empty when using chunked transfer)
    total_input_chunks: int = 0
    prompt: str
    model: str
    n: int | None = 1
    quality: Literal["high", "medium", "low"] | None = "medium"
    output_format: Literal["png", "jpeg", "webp"] = "png"
    response_format: Literal["url", "b64_json"] | None = "b64_json"
    size: ImageSize = "auto"
    image_strength: float | None = 0.7
    stream: bool = False
    partial_images: int | None = 0
    advanced_params: AdvancedImageParams | None = None
    bench: bool = False

    @field_validator("size", mode="before")
    @classmethod
    def normalize_size(cls, v: object) -> ImageSize:
        return normalize_image_size(v)

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "image_data":
                yield name, f"<{len(self.image_data)} chars>"
            elif name is not None:
                yield name, value


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "b64_json" and self.b64_json is not None:
                yield name, f"<{len(self.b64_json)} chars>"
            elif name is not None:
                yield name, value


class ImageGenerationResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ImageData]


# ── Embeddings ──────────────────────────────────────────


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float] | str


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: EmbeddingUsage


class BenchImageGenerationResponse(ImageGenerationResponse):
    generation_stats: ImageGenerationStats | None = None
    power_usage: PowerUsage | None = None


class ImageListItem(BaseModel, frozen=True):
    image_id: str
    url: str
    content_type: str
    expires_at: float


class ImageListResponse(BaseModel, frozen=True):
    data: list[ImageListItem]


class StartDownloadParams(CamelCaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "targetNodeId": "12D3KooWExampleNodeId",
                "shardMetadata": {
                    "TensorShardMetadata": {
                        "modelCard": {
                            "modelId": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                            "storageSize": {"inBytes": 2147483648},
                            "nLayers": 32,
                            "hiddenSize": 2048,
                            "supportsTensor": True,
                            "tasks": ["TextGeneration"],
                        },
                        "deviceRank": 0,
                        "worldSize": 1,
                        "startLayer": 0,
                        "endLayer": 32,
                        "nLayers": 32,
                    }
                },
            }
        }
    )

    target_node_id: NodeId
    shard_metadata: ShardMetadata


class StartDownloadResponse(CamelCaseModel):
    command_id: CommandId


class DeleteDownloadResponse(CamelCaseModel):
    command_id: CommandId


class PurgeStagingRequest(CamelCaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"modelId": "mlx-community/Llama-3.2-1B-Instruct-4bit"}
        }
    )

    model_id: str | None = None


class PurgeStagingResponse(CamelCaseModel):
    command_id: CommandId
    message: str


class NodeStorageSummary(CamelCaseModel):
    """Per-node storage breakdown for the local node.

    Sizes are computed on demand by walking the relevant directories;
    staged models include last-use times and whether a live instance
    (or its companion repos) currently depends on them.
    """

    node_id: NodeId
    staging_root: str | None
    """Resolved staging directory for this node, or None when the model
    store / staging is not configured."""

    staged_models: list[StagedModelInfo]
    staged_total_bytes: int
    event_log_bytes: int
    """Total size of this node's event-log directory (active + archives)."""

    disk_total_bytes: int
    disk_free_bytes: int
    """Capacity and free space of the volume holding the staging directory
    (falls back to the models directory when staging is not configured)."""


TraceTaskKind = Literal["image", "text", "embedding", "speech"]


class TraceSourceNode(CamelCaseModel):
    node_id: str
    friendly_name: str | None = None


class TraceEventResponse(CamelCaseModel):
    name: str
    start_us: int
    duration_us: int
    rank: int
    category: str
    node_id: str | None = None
    model_id: str | None = None
    task_kind: TraceTaskKind | None = None
    tags: list[str] = Field(default_factory=list)
    attrs: dict[str, str | int | float | bool | list[str]] = Field(default_factory=dict)


class TraceResponse(CamelCaseModel):
    task_id: str
    traces: list[TraceEventResponse]
    source_nodes: list[TraceSourceNode] = Field(default_factory=list)


class TraceCategoryStats(CamelCaseModel):
    total_us: int
    count: int
    min_us: int
    max_us: int
    avg_us: float


class TraceRankStats(CamelCaseModel):
    by_category: dict[str, TraceCategoryStats]


class TraceStatsResponse(CamelCaseModel):
    task_id: str
    total_wall_time_us: int
    by_category: dict[str, TraceCategoryStats]
    by_rank: dict[int, TraceRankStats]
    source_nodes: list[TraceSourceNode] = Field(default_factory=list)


class TraceListItem(CamelCaseModel):
    task_id: str
    created_at: str
    file_size: int
    model_id: str | None = None
    task_kind: TraceTaskKind | None = None
    categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    has_tool_activity: bool = False
    source_nodes: list[TraceSourceNode] = Field(default_factory=list)


class TraceListResponse(CamelCaseModel):
    traces: list[TraceListItem]


class TracingStateResponse(CamelCaseModel):
    enabled: bool


class UpdateTracingStateRequest(CamelCaseModel):
    enabled: bool


class DeleteTracesRequest(CamelCaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"taskIds": ["chatcmpl-123", "chatcmpl-456"]}}
    )

    task_ids: list[str]


class DeleteTracesResponse(CamelCaseModel):
    deleted: list[str]
    not_found: list[str]
