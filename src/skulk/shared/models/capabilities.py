"""Runtime capability resolution derived from model cards.

This module keeps model cards as the persisted declarative source of truth while
providing a normalized runtime profile that inference code can consume without
sprinkling optional-field checks throughout the hot path.
"""

from typing import TYPE_CHECKING, Literal

from skulk.shared.models.model_cards import (
    AudioCardKind,
    AudioResponseFormat,
    BuiltinToolType,
    ModelCard,
    ModelId,
    ModelTask,
    OutputParserType,
    PromptRendererType,
    ReasoningFormat,
    ToolCallFormat,
    get_card,
)
from skulk.shared.types.text_generation import ReasoningEffort, TextGenerationTaskParams
from skulk.utils.pydantic_ext import FrozenModel

if TYPE_CHECKING:
    from mlx_lm.tokenizer_utils import TokenizerWrapper


class ResolvedCapabilityProfile(FrozenModel):
    """Normalized runtime behavior derived from a model card and runtime facts."""

    family: str = "generic"
    supports_thinking: bool = False
    supports_thinking_toggle: bool = False
    supports_thinking_budget: bool = False
    default_reasoning_effort: ReasoningEffort = "medium"
    disabled_reasoning_effort: ReasoningEffort = "none"
    thinking_format: ReasoningFormat = ReasoningFormat.None_
    supports_image_input: bool = False
    supports_audio_input: bool = False
    supports_speech_synthesis: bool = False
    supports_transcription: bool = False
    supports_speech_translation: bool = False
    supports_audio_output: bool = False
    supports_realtime_audio: bool = False
    default_audio_response_format: AudioResponseFormat | None = None
    audio_response_formats: tuple[AudioResponseFormat, ...] = ()
    supports_tool_calling: bool = False
    builtin_tools: tuple[BuiltinToolType, ...] = ()
    tool_call_format: ToolCallFormat = ToolCallFormat.Generic
    prompt_renderer: PromptRendererType = PromptRendererType.Tokenizer
    output_parser: OutputParserType = OutputParserType.Generic
    supports_native_multimodal: bool = False


def _infer_family(model_id: ModelId, model_card: ModelCard | None) -> str:
    if model_card is not None and model_card.family:
        return model_card.family
    short_model_id = model_id.short()
    family = short_model_id.split("-", 1)[0]
    return family or "generic"


def _is_gemma4_id(model_id: object) -> bool:
    """Return whether a model id string contains a Gemma 4 marker.

    Underscore-tolerant so cards using ``gemma_4`` and ``gemma-4`` both match.
    Single source of truth for id-string-based detection used by both the
    capability resolver (which has only an id at resolution time) and the
    full ``is_gemma4_family`` predicate (which composes id + card-declared
    hints).
    """
    normalized = str(model_id).lower().replace("_", "-")
    return "gemma-4" in normalized or "gemma4" in normalized


def _is_gemma4_family(profile_family: str, normalized_model_id: str) -> bool:
    """Limited Gemma 4 check used inside ``resolve_model_capability_profile``.

    Cannot consult ``card.runtime`` / ``card.tooling`` / ``card.vision``
    because those fields are being computed at this stage of resolution.
    For post-resolution use cases (the runner and the planner), prefer the
    public ``is_gemma4_family(card)`` which incorporates those signals.
    """
    return _is_gemma4_id(normalized_model_id) or profile_family in {
        "gemma4",
        "gemma-4",
    }


def is_gemma4_family(
    card: ModelCard | None = None,
    model_id: ModelId | None = None,
) -> bool:
    """Return whether the model is in the Gemma 4 family.

    At least one of ``card`` or ``model_id`` should be provided. Cards carry
    declarative hints (``vision.model_type``, ``runtime.prompt_renderer``,
    ``runtime.output_parser``, ``tooling.tool_call_format``) that allow
    detection even when the model id doesn't visibly contain ``gemma-4``,
    so a card is preferred when available.

    This is the post-resolution consolidation of detection logic that was
    previously duplicated across ``runner.py`` and ``plan.py``. See
    ``_is_gemma4_family`` for the resolver-time variant.
    """
    target_id = (card.model_id if card is not None else None) or model_id
    if target_id is not None and _is_gemma4_id(target_id):
        return True
    if card is None:
        return False
    if card.family in {"gemma4", "gemma-4"}:
        return True
    if card.vision is not None and card.vision.model_type == "gemma4":
        return True
    if card.runtime is not None and (
        card.runtime.prompt_renderer == PromptRendererType.Gemma4
        or card.runtime.output_parser == OutputParserType.Gemma4
    ):
        return True
    return (
        card.tooling is not None
        and card.tooling.tool_call_format == ToolCallFormat.Gemma4
    )


def _is_deepseek_v32_family(profile_family: str, normalized_model_id: str) -> bool:
    return (
        "deepseek-v3.2" in normalized_model_id
        or profile_family in {"deepseek-v3.2", "deepseek_v32"}
    )


def _is_gpt_oss_family(profile_family: str, normalized_model_id: str) -> bool:
    return profile_family == "gpt-oss" or any(
        marker in normalized_model_id for marker in ("gpt-oss", "gpt_oss")
    )


def _is_muse_glimmer_family(profile_family: str, normalized_model_id: str) -> bool:
    """Muse Glimmer (Meta, 2026-08): channel reasoning plus ATEM tool calls.

    Matched on the card family (the registry compiles ``muse-glimmer`` from the
    upstream ``model_type``; bundled and custom cards may spell it with an
    underscore) or on the model id, so an auto-imported quant resolves to the
    family contract before any card declares it. The contract is fixed by the
    chat template Meta ships with every artifact: reasoning is always on (the
    template opens the ``to=self`` channel unconditionally, so there is no
    toggle), strength is a template kwarg rather than an on/off switch, and
    tool calls use the ATEM markup.
    """
    family = profile_family.lower().replace("_", "-")
    return family == "muse-glimmer" or any(
        marker in normalized_model_id
        for marker in ("muse-glimmer", "muse_glimmer", "museglimmer")
    )


def uses_muse_glimmer_protocol(profile: ResolvedCapabilityProfile) -> bool:
    """Whether a resolved profile speaks Muse Glimmer's channel/ATEM protocol."""
    return profile.output_parser == OutputParserType.MuseGlimmer


#: Muse Glimmer's ``reasoning_strength`` template levels, in ascending order.
MuseGlimmerReasoningStrength = Literal["low", "medium", "high", "xhigh"]

_MUSE_GLIMMER_STRENGTH_BY_EFFORT: dict[ReasoningEffort, MuseGlimmerReasoningStrength] = {
    # Reasoning cannot be switched off (the template opens the channel
    # unconditionally), so the disabling efforts map to the lightest level the
    # model was trained on rather than being dropped.
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}


def muse_glimmer_reasoning_strength(
    reasoning_effort: ReasoningEffort | None,
) -> MuseGlimmerReasoningStrength | None:
    """Map an OpenAI-style reasoning effort onto Muse Glimmer's strength level.

    Returns ``None`` when no effort was requested, leaving the template's own
    default (``high``) in force.
    """
    if reasoning_effort is None:
        return None
    return _MUSE_GLIMMER_STRENGTH_BY_EFFORT[reasoning_effort]


def muse_glimmer_template_kwargs(
    profile: ResolvedCapabilityProfile,
    reasoning_effort: ReasoningEffort | None,
) -> dict[str, str]:
    """Chat-template kwargs that carry the request's effort to a Muse Glimmer model.

    Muse Glimmer reads ``reasoning_strength`` (low / medium / high / xhigh),
    not the ``enable_thinking`` toggle or the harmony ``reasoning_effort`` the
    other families use, so every engine's prompt path (the MLX template call,
    the llama-server and vLLM ``chat_template_kwargs`` request field) merges
    this in. Empty for every other family and when no effort was requested.
    """
    if not uses_muse_glimmer_protocol(profile):
        return {}
    strength = muse_glimmer_reasoning_strength(reasoning_effort)
    if strength is None:
        return {}
    return {"reasoning_strength": strength}


def _is_qwen3_thinking_family(profile_family: str, normalized_model_id: str) -> bool:
    """Qwen3 and its point releases (Qwen3.5, Qwen3.6) are hybrid thinking models.

    They reason behind a token-delimited ``<think>`` toggle. Built-in cards
    declare ``thinking``/``thinking_toggle`` explicitly, but an auto-imported
    card (no built-in entry, e.g. a fresh quant) arrives with empty capabilities
    and would otherwise resolve to no-thinking, so the model reasons
    unconditionally and returns empty content under a normal request (#384).
    Matched on the normalized id or the inferred family token. The ``Coder``
    variants are instruct-only (no thinking mode) and are excluded; Qwen2.x and
    earlier never contain ``qwen3`` so they fall through to generic handling.
    """
    if "coder" in normalized_model_id:
        return False
    return "qwen3" in normalized_model_id or profile_family.lower().startswith("qwen3")


def resolve_model_capability_profile(
    model_id: ModelId,
    *,
    model_card: ModelCard | None = None,
    tokenizer: "TokenizerWrapper | None" = None,
    task_params: TextGenerationTaskParams | None = None,
) -> ResolvedCapabilityProfile:
    """Resolve runtime capabilities for one request.

    The resolver is intentionally conservative: if a card does not declare an
    advanced capability, we fall back to generic runtime behavior and only
    preserve the broad support we can infer from the existing card fields.
    """

    card = model_card or get_card(model_id)
    normalized_model_id = model_id.normalize().lower()
    profile_family = _infer_family(model_id, card)

    supports_image_input = bool(
        card is not None
        and (
            "vision" in card.capabilities
            or card.vision is not None
            or (
                card.modalities is not None
                and card.modalities.supports_native_multimodal is True
            )
        )
    )
    supports_speech_synthesis = bool(
        card is not None
        and (
            ModelTask.TextToSpeech in card.tasks
            or "tts" in card.capabilities
            or (
                card.audio is not None
                and card.audio.kind == AudioCardKind.TextToSpeech
            )
        )
    )
    supports_transcription = bool(
        card is not None
        and (
            ModelTask.SpeechToText in card.tasks
            or ModelTask.SpeechTranslation in card.tasks
            or "stt" in card.capabilities
            or (
                card.audio is not None
                and card.audio.kind == AudioCardKind.SpeechToText
            )
        )
    )
    supports_speech_translation = bool(
        card is not None
        and (
            ModelTask.SpeechTranslation in card.tasks
            or (
                card.audio is not None
                and card.audio.supports_translation is True
            )
        )
    )
    if supports_speech_translation:
        supports_transcription = True
    audio_response_formats: tuple[AudioResponseFormat, ...] = ()
    default_audio_response_format: AudioResponseFormat | None = None
    supports_realtime_audio = False
    if card is not None and card.audio is not None:
        audio_response_formats = card.audio.response_formats
        default_audio_response_format = card.audio.default_response_format
        if default_audio_response_format is None and audio_response_formats:
            default_audio_response_format = audio_response_formats[0]
        supports_realtime_audio = card.audio.supports_realtime is True
    supports_reference_audio = bool(
        card is not None
        and card.audio is not None
        and card.audio.supports_reference_audio is True
    )
    supports_thinking = bool(
        card is not None
        and ("thinking" in card.capabilities or card.reasoning is not None)
    )
    supports_thinking_toggle = bool(
        card is not None and "thinking_toggle" in card.capabilities
    )
    supports_tool_calling = bool(
        (tokenizer is not None and getattr(tokenizer, "has_tool_calling", False))
        or (
            card is not None
            and (
                (card.tooling is not None and card.tooling.supports_tool_calling is True)
                or (
                    card.tooling is not None
                    and card.tooling.tool_call_format is not None
                    and card.tooling.tool_call_format != ToolCallFormat.Generic
                )
            )
        )
    )
    thinking_format = (
        ReasoningFormat.TokenDelimited
        if tokenizer is not None and getattr(tokenizer, "has_thinking", False)
        else ReasoningFormat.None_
    )

    profile = ResolvedCapabilityProfile(
        family=profile_family,
        supports_thinking=supports_thinking,
        supports_thinking_toggle=supports_thinking_toggle,
        supports_image_input=supports_image_input,
        supports_audio_input=(
            supports_transcription
            or supports_speech_translation
            or supports_reference_audio
        ),
        supports_speech_synthesis=supports_speech_synthesis,
        supports_transcription=supports_transcription,
        supports_speech_translation=supports_speech_translation,
        supports_audio_output=supports_speech_synthesis,
        supports_realtime_audio=supports_realtime_audio,
        default_audio_response_format=default_audio_response_format,
        audio_response_formats=audio_response_formats,
        supports_tool_calling=supports_tool_calling,
        thinking_format=thinking_format,
        supports_native_multimodal=False,
    )

    # Family-specific defaults preserve current behavior until cards opt in to
    # richer declarations. Explicit card fields override these defaults below.
    if _is_gemma4_family(profile.family, normalized_model_id):
        prompt_renderer = PromptRendererType.Tokenizer
        if task_params is None or (
            task_params.chat_template_messages is not None and not task_params.tools
        ):
            prompt_renderer = PromptRendererType.Gemma4

        profile = profile.model_copy(
            update={
                "supports_thinking": True,
                "supports_thinking_toggle": True,
                "supports_tool_calling": True,
                "thinking_format": ReasoningFormat.ChannelDelimited,
                "prompt_renderer": prompt_renderer,
                "output_parser": OutputParserType.Gemma4,
                "tool_call_format": ToolCallFormat.Gemma4,
                "supports_native_multimodal": supports_image_input,
            }
        )
    elif _is_deepseek_v32_family(profile.family, normalized_model_id):
        profile = profile.model_copy(
            update={
                "supports_thinking": True,
                "supports_thinking_toggle": True,
                "supports_tool_calling": True,
                "prompt_renderer": PromptRendererType.Dsml,
                "output_parser": OutputParserType.DeepseekV32,
                "tool_call_format": ToolCallFormat.Dsml,
            }
        )
    elif _is_gpt_oss_family(profile.family, normalized_model_id):
        profile = profile.model_copy(
            update={
                "supports_tool_calling": True,
                "output_parser": OutputParserType.GptOss,
                "tool_call_format": ToolCallFormat.GptOss,
            }
        )
    elif _is_muse_glimmer_family(profile.family, normalized_model_id):
        profile = profile.model_copy(
            update={
                "supports_thinking": True,
                # The template opens the reasoning channel unconditionally;
                # requests steer how much with reasoning_strength instead.
                "supports_thinking_toggle": False,
                "thinking_format": ReasoningFormat.ChannelDelimited,
                "default_reasoning_effort": "high",
                "disabled_reasoning_effort": "low",
                "supports_tool_calling": True,
                "output_parser": OutputParserType.MuseGlimmer,
                "tool_call_format": ToolCallFormat.Atem,
                "supports_native_multimodal": supports_image_input,
            }
        )
    elif _is_qwen3_thinking_family(profile.family, normalized_model_id) and (
        card is None or not card.capabilities or "thinking" in card.capabilities
    ):
        # Qwen3/3.5/3.6 reason behind a token-delimited <think> toggle. Default
        # the thinking contract on so an auto-imported card (empty capabilities;
        # fetch_from_hf never fills the field) is still treated as
        # toggle-capable, instead of resolving to no-thinking and reasoning
        # unconditionally into empty content (#384). A card that DOES declare
        # capabilities without "thinking" is an explicit statement that this
        # variant does not think (the Instruct-2507 / Next-Instruct releases),
        # and the family default must not override card truth: `supports_
        # thinking` has no card-level off switch below (the [reasoning]
        # overrides cover toggle/format/effort only), so this gate is the only
        # thing keeping instruct-only Qwen3 variants from advertising a
        # thinking contract they cannot honor. Only the thinking fields are
        # set; tool/prompt/parser behavior stays generic. Explicit
        # card.reasoning still overrides below.
        profile = profile.model_copy(
            update={
                "supports_thinking": True,
                "supports_thinking_toggle": True,
                "thinking_format": ReasoningFormat.TokenDelimited,
            }
        )

    if card is not None and card.reasoning is not None:
        updates: dict[str, object] = {}
        if card.reasoning.supports_toggle is not None:
            updates["supports_thinking_toggle"] = card.reasoning.supports_toggle
        if card.reasoning.supports_budget is not None:
            updates["supports_thinking_budget"] = card.reasoning.supports_budget
        if card.reasoning.format is not None:
            updates["thinking_format"] = card.reasoning.format
        if card.reasoning.default_effort is not None:
            updates["default_reasoning_effort"] = card.reasoning.default_effort
        if card.reasoning.disabled_effort is not None:
            updates["disabled_reasoning_effort"] = card.reasoning.disabled_effort
        if updates:
            profile = profile.model_copy(update=updates)

    if card is not None and card.modalities is not None:
        updates = {}
        if card.modalities.supports_audio_input is not None:
            updates["supports_audio_input"] = (
                card.modalities.supports_audio_input or profile.supports_audio_input
            )
        if card.modalities.supports_native_multimodal is not None:
            updates["supports_native_multimodal"] = (
                card.modalities.supports_native_multimodal
            )
        if updates:
            profile = profile.model_copy(update=updates)

    if card is not None and card.tooling is not None:
        updates = {}
        if card.tooling.supports_tool_calling is not None:
            updates["supports_tool_calling"] = card.tooling.supports_tool_calling
        if card.tooling.builtin_tools is not None:
            updates["builtin_tools"] = tuple(card.tooling.builtin_tools)
        if card.tooling.tool_call_format is not None:
            updates["tool_call_format"] = card.tooling.tool_call_format
        if updates:
            profile = profile.model_copy(update=updates)

    if card is not None and card.runtime is not None:
        updates = {}
        if card.runtime.prompt_renderer is not None:
            updates["prompt_renderer"] = card.runtime.prompt_renderer
        if card.runtime.output_parser is not None:
            updates["output_parser"] = card.runtime.output_parser
        if updates:
            profile = profile.model_copy(update=updates)

    # Preserve the existing fallback for tool-enabled Gemma 4 requests until we
    # land the full prompt grammar. Cards can declare Gemma 4 behavior, but the
    # runtime must still stay conservative here.
    if (
        task_params is not None
        and task_params.tools
        and profile.prompt_renderer == PromptRendererType.Gemma4
    ):
        profile = profile.model_copy(
            update={"prompt_renderer": PromptRendererType.Tokenizer}
        )

    return profile
