import pytest
from pydantic import ValidationError

from skulk.shared.models.capabilities import (
    ResolvedCapabilityProfile,
    family_predates_in_process_llama_cpp,
    muse_glimmer_template_kwargs,
    resolve_model_capability_profile,
    uses_muse_glimmer_protocol,
)
from skulk.shared.models.model_cards import (
    AudioCardConfig,
    AudioCardKind,
    AudioResponseFormat,
    AudioVoiceConfig,
    BuiltinToolType,
    ModalitiesCardConfig,
    ModelCard,
    ModelId,
    ModelTask,
    OutputParserType,
    PromptRendererType,
    ReasoningCardConfig,
    ReasoningFormat,
    RuntimeCapabilityCardConfig,
    ToolCallFormat,
    ToolingCardConfig,
    card_serves_speech,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.text_generation import (
    InputMessage,
    TextGenerationTaskParams,
    resolve_reasoning_params,
)


def _base_model_card(model_id: str) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="gemma",
        capabilities=["text", "vision", "thinking"],
    )


def test_resolve_model_capability_profile_uses_extended_model_card_fields() -> None:
    card = _base_model_card("example/gemma-test").model_copy(
        update={
            "reasoning": ReasoningCardConfig(
                supports_toggle=True,
                supports_budget=True,
                format=ReasoningFormat.ChannelDelimited,
                default_effort="high",
                disabled_effort="none",
            ),
            "modalities": ModalitiesCardConfig(
                supports_audio_input=True,
                supports_native_multimodal=True,
            ),
            "tooling": ToolingCardConfig(
                supports_tool_calling=True,
                tool_call_format=ToolCallFormat.Gemma4,
            ),
            "runtime": RuntimeCapabilityCardConfig(
                prompt_renderer=PromptRendererType.Gemma4,
                output_parser=OutputParserType.Gemma4,
            ),
        }
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking_toggle is True
    assert profile.supports_thinking_budget is True
    assert profile.thinking_format == ReasoningFormat.ChannelDelimited
    assert profile.default_reasoning_effort == "high"
    assert profile.supports_audio_input is True
    assert profile.tool_call_format == ToolCallFormat.Gemma4
    assert profile.prompt_renderer == PromptRendererType.Gemma4
    assert profile.output_parser == OutputParserType.Gemma4


def test_resolve_model_capability_profile_exposes_tts_audio_metadata() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/kokoro-test"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextToSpeech],
        family="kokoro",
        capabilities=["tts"],
        audio=AudioCardConfig(
            kind=AudioCardKind.TextToSpeech,
            default_response_format=AudioResponseFormat.Mp3,
            response_formats=(AudioResponseFormat.Mp3, AudioResponseFormat.Wav),
            supports_streaming=True,
            supports_realtime=True,
            supports_voice_listing=True,
            voices=("alloy",),
            sample_rates=(24000,),
        ),
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_speech_synthesis is True
    assert profile.supports_transcription is False
    assert profile.supports_speech_translation is False
    assert profile.supports_audio_output is True
    assert profile.supports_audio_input is False
    assert profile.supports_realtime_audio is True
    assert profile.default_audio_response_format == AudioResponseFormat.Mp3
    assert profile.audio_response_formats == (
        AudioResponseFormat.Mp3,
        AudioResponseFormat.Wav,
    )


def test_resolve_model_capability_profile_treats_reference_audio_as_input() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/kokoro-voice-clone-test"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextToSpeech],
        family="kokoro",
        capabilities=["tts"],
        modalities=ModalitiesCardConfig(supports_audio_input=False),
        audio=AudioCardConfig(
            kind=AudioCardKind.TextToSpeech,
            supports_reference_audio=True,
        ),
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_speech_synthesis is True
    assert profile.supports_audio_input is True
    assert profile.supports_audio_output is True


def test_resolve_model_capability_profile_exposes_stt_and_translation() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/whisper-test"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.SpeechToText],
        family="whisper",
        capabilities=["stt"],
        modalities=ModalitiesCardConfig(supports_audio_input=False),
        audio=AudioCardConfig(
            kind=AudioCardKind.SpeechToText,
            supports_translation=True,
            supports_realtime=False,
            sample_rates=(16000,),
        ),
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_speech_synthesis is False
    assert profile.supports_transcription is True
    assert profile.supports_speech_translation is True
    assert profile.supports_audio_input is True
    assert profile.supports_audio_output is False
    assert profile.supports_realtime_audio is False


def test_audio_card_config_requires_stt_kind_for_translation() -> None:
    with pytest.raises(ValidationError, match="supports_translation"):
        AudioCardConfig(supports_translation=True)

    with pytest.raises(ValidationError, match="supports_translation"):
        AudioCardConfig(
            kind=AudioCardKind.TextToSpeech,
            supports_translation=True,
        )


def test_audio_card_config_validates_static_voice_catalog() -> None:
    config = AudioCardConfig(
        kind=AudioCardKind.TextToSpeech,
        supports_voice_listing=True,
        voices=("alloy", "coral"),
        voice_catalog=(
            AudioVoiceConfig(
                id="alloy",
                name="Alloy",
                preferred_languages=("en-US",),
            ),
            AudioVoiceConfig(id="coral", name="Coral"),
        ),
    )

    assert config.voices == ("alloy", "coral")
    assert config.voice_catalog[0].preferred_languages == ("en-us",)

    with pytest.raises(ValidationError, match="supports_voice_listing"):
        AudioCardConfig(kind=AudioCardKind.TextToSpeech, voices=("alloy",))

    with pytest.raises(ValidationError, match="requires voices"):
        AudioCardConfig(
            kind=AudioCardKind.TextToSpeech,
            supports_voice_listing=True,
        )

    with pytest.raises(ValidationError, match="default_voice"):
        AudioCardConfig(
            kind=AudioCardKind.TextToSpeech,
            supports_voice_listing=True,
            voices=("alloy",),
            default_voice="missing",
        )

    with pytest.raises(ValidationError, match="duplicates"):
        AudioCardConfig(
            kind=AudioCardKind.TextToSpeech,
            supports_voice_listing=True,
            voices=("alloy", "alloy"),
        )

    with pytest.raises(ValidationError, match="must match voices exactly"):
        AudioCardConfig(
            kind=AudioCardKind.TextToSpeech,
            supports_voice_listing=True,
            voices=("alloy",),
            voice_catalog=(AudioVoiceConfig(id="coral", name="Coral"),),
        )


def test_resolve_model_capability_profile_treats_translation_as_transcription() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/whisper-translate-test"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.SpeechTranslation],
        family="whisper",
        capabilities=[],
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_transcription is True
    assert profile.supports_speech_translation is True
    assert profile.supports_audio_input is True
    assert profile.supports_audio_output is False


def test_resolve_model_capability_profile_keeps_legacy_translation_coherent() -> None:
    legacy_audio = AudioCardConfig.model_construct(
        kind=None,
        default_response_format=None,
        response_formats=(),
        supports_streaming=None,
        supports_realtime=None,
        supports_voice_listing=None,
        supports_reference_audio=None,
        supports_translation=True,
        sample_rates=(),
    )
    card = _base_model_card("mlx-community/legacy-translation-test").model_copy(
        update={
            "capabilities": [],
            "modalities": ModalitiesCardConfig(supports_audio_input=False),
            "audio": legacy_audio,
        }
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_speech_translation is True
    assert profile.supports_transcription is True
    assert profile.supports_audio_input is True


def test_card_serves_speech_treats_legacy_capability_tags_as_speech() -> None:
    text_card = _base_model_card("example/text-test").model_copy(
        update={"capabilities": ["text"]}
    )
    tts_card = _base_model_card("example/tts-test").model_copy(
        update={"capabilities": ["tts"]}
    )
    stt_card = _base_model_card("example/stt-test").model_copy(
        update={"capabilities": ["stt"]}
    )

    assert card_serves_speech(text_card) is False
    assert card_serves_speech(tts_card) is True
    assert card_serves_speech(stt_card) is True


def test_resolve_model_capability_profile_keeps_gemma4_tool_fallback() -> None:
    card = _base_model_card("mlx-community/gemma-4-26b-a4b-it-4bit").model_copy(
        update={
            "runtime": RuntimeCapabilityCardConfig(
                prompt_renderer=PromptRendererType.Gemma4,
                output_parser=OutputParserType.Gemma4,
            )
        }
    )
    task_params = TextGenerationTaskParams(
        model=card.model_id,
        input=[InputMessage(role="user", content="hello")],
        chat_template_messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Lookup weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    profile = resolve_model_capability_profile(
        card.model_id,
        model_card=card,
        task_params=task_params,
    )

    assert profile.supports_tool_calling is True
    assert profile.prompt_renderer == PromptRendererType.Tokenizer
    assert profile.output_parser == OutputParserType.Gemma4


def test_resolve_reasoning_params_uses_profile_defaults() -> None:
    profile = ResolvedCapabilityProfile(
        supports_thinking=True,
        supports_thinking_toggle=True,
        default_reasoning_effort="high",
        disabled_reasoning_effort="none",
    )

    assert resolve_reasoning_params(None, True, profile) == ("high", True)
    assert resolve_reasoning_params(None, False, profile) == ("none", False)
    assert resolve_reasoning_params(None, None, profile) == ("none", False)


def test_resolve_reasoning_params_treats_none_as_disabled_even_for_custom_profiles() -> None:
    profile = ResolvedCapabilityProfile(
        supports_thinking=True,
        supports_thinking_toggle=True,
        default_reasoning_effort="high",
        disabled_reasoning_effort="minimal",
    )

    assert resolve_reasoning_params("none", None, profile) == ("minimal", False)
    assert resolve_reasoning_params("none", True, profile) == ("minimal", False)


def test_resolve_reasoning_params_ignores_toggle_inputs_for_non_toggleable_profiles() -> None:
    profile = ResolvedCapabilityProfile(
        supports_thinking=True,
        supports_thinking_toggle=False,
        default_reasoning_effort="high",
        disabled_reasoning_effort="minimal",
    )

    assert resolve_reasoning_params(None, False, profile) == (None, None)
    assert resolve_reasoning_params("minimal", None, profile) == ("minimal", None)
    assert resolve_reasoning_params("high", False, profile) == ("high", None)


def test_resolve_model_capability_profile_uses_safe_generic_fallback() -> None:
    card = ModelCard(
        model_id=ModelId("example/plain-text-model"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="example",
        capabilities=["text"],
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.family == "example"
    assert profile.supports_thinking is False
    assert profile.supports_thinking_toggle is False
    assert profile.supports_image_input is False
    assert profile.supports_tool_calling is False
    assert profile.prompt_renderer == PromptRendererType.Tokenizer
    assert profile.output_parser == OutputParserType.Generic


def test_resolve_model_capability_profile_infers_family_from_short_model_id() -> None:
    profile = resolve_model_capability_profile(
        ModelId("mlx-community/gemma-4-custom"),
        model_card=None,
    )

    assert profile.family == "gemma"


def test_resolve_model_capability_profile_honors_coarse_thinking_toggle_capability() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/Qwen3.5-122B-A10B-4bit"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        capabilities=["text", "thinking", "thinking_toggle"],
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking is True
    assert profile.supports_thinking_toggle is True
    assert profile.prompt_renderer == PromptRendererType.Tokenizer
    assert profile.output_parser == OutputParserType.Generic


def test_resolve_model_capability_profile_honors_nemotron_reasoning_metadata() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="nemotron",
        capabilities=["text", "thinking", "thinking_toggle"],
        reasoning=ReasoningCardConfig(
            supports_toggle=True,
            format=ReasoningFormat.TokenDelimited,
            default_effort="medium",
            disabled_effort="none",
        ),
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking is True
    assert profile.supports_thinking_toggle is True
    assert profile.thinking_format == ReasoningFormat.TokenDelimited
    assert profile.default_reasoning_effort == "medium"
    assert profile.disabled_reasoning_effort == "none"


def test_resolve_model_capability_profile_honors_qwen35_reasoning_metadata() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/Qwen3.5-9B-4bit"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        capabilities=["text", "thinking", "thinking_toggle"],
        reasoning=ReasoningCardConfig(
            supports_toggle=True,
            format=ReasoningFormat.TokenDelimited,
            default_effort="medium",
            disabled_effort="none",
        ),
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking is True
    assert profile.supports_thinking_toggle is True
    assert profile.thinking_format == ReasoningFormat.TokenDelimited
    assert profile.default_reasoning_effort == "medium"
    assert profile.disabled_reasoning_effort == "none"


def test_resolve_model_capability_profile_honors_deepseek_v32_metadata() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/DeepSeek-V3.2-4bit"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="deepseek",
        capabilities=["text", "thinking", "thinking_toggle"],
        reasoning=ReasoningCardConfig(
            supports_toggle=True,
            format=ReasoningFormat.TokenDelimited,
            default_effort="medium",
            disabled_effort="none",
        ),
        tooling=ToolingCardConfig(
            supports_tool_calling=True,
            tool_call_format=ToolCallFormat.Dsml,
        ),
        runtime=RuntimeCapabilityCardConfig(
            prompt_renderer=PromptRendererType.Dsml,
            output_parser=OutputParserType.DeepseekV32,
        ),
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking is True
    assert profile.supports_thinking_toggle is True
    assert profile.thinking_format == ReasoningFormat.TokenDelimited
    assert profile.supports_tool_calling is True
    assert profile.prompt_renderer == PromptRendererType.Dsml
    assert profile.output_parser == OutputParserType.DeepseekV32
    assert profile.tool_call_format == ToolCallFormat.Dsml


def test_resolve_model_capability_profile_uses_declared_tooling_without_model_id_match() -> None:
    card = ModelCard(
        model_id=ModelId("custom/open-model"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="custom",
        capabilities=["text"],
        tooling=ToolingCardConfig(
            supports_tool_calling=True,
            tool_call_format=ToolCallFormat.GptOss,
        ),
        runtime=RuntimeCapabilityCardConfig(
            output_parser=OutputParserType.GptOss,
        ),
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_tool_calling is True
    assert profile.tool_call_format == ToolCallFormat.GptOss
    assert profile.output_parser == OutputParserType.GptOss


def test_resolve_model_capability_profile_uses_gpt_oss_family_defaults() -> None:
    card = ModelCard(
        model_id=ModelId("custom/oss-20b"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="gpt-oss",
        capabilities=["text", "thinking"],
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.family == "gpt-oss"
    assert profile.supports_thinking is True
    assert profile.supports_tool_calling is True
    assert profile.tool_call_format == ToolCallFormat.GptOss
    assert profile.output_parser == OutputParserType.GptOss


def test_resolve_model_capability_profile_exposes_builtin_tools() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/gpt-oss-20b-MXFP4-Q8"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="gpt-oss",
        capabilities=["text", "thinking"],
        tooling=ToolingCardConfig(
            supports_tool_calling=True,
            builtin_tools=[
                BuiltinToolType.WebSearch,
                BuiltinToolType.OpenUrl,
                BuiltinToolType.ExtractPage,
            ],
            tool_call_format=ToolCallFormat.GptOss,
        ),
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.builtin_tools == (
        BuiltinToolType.WebSearch,
        BuiltinToolType.OpenUrl,
        BuiltinToolType.ExtractPage,
    )


def test_resolve_model_capability_profile_uses_deepseek_v32_family_defaults() -> None:
    card = ModelCard(
        model_id=ModelId("custom/deepseek-compatible"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="deepseek-v3.2",
        capabilities=["text", "thinking"],
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking is True
    assert profile.supports_thinking_toggle is True
    assert profile.supports_tool_calling is True
    assert profile.prompt_renderer == PromptRendererType.Dsml
    assert profile.output_parser == OutputParserType.DeepseekV32
    assert profile.tool_call_format == ToolCallFormat.Dsml


def test_resolve_model_capability_profile_keeps_native_multimodal_conservative_by_default() -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/vision-model"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="vision",
        capabilities=["text", "vision"],
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_image_input is True
    assert profile.supports_native_multimodal is False


# ---------------------------------------------------------------------------
# is_gemma4_family — consolidated detection predicate
# ---------------------------------------------------------------------------


def _bare_card(model_id: str, *, family: str = "") -> ModelCard:
    """Build a minimal card with no gemma4 hints declared."""
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family=family,
        capabilities=["text"],
    )


def test_is_gemma4_family_matches_card_with_gemma_4_in_id() -> None:
    """Gemma 4 cards from resources/ have ``gemma-4`` in the id and must match."""
    from skulk.shared.models.capabilities import is_gemma4_family

    card = _bare_card("mlx-community/gemma-4-26b-a4b-it-4bit", family="gemma")
    assert is_gemma4_family(card) is True


def test_is_gemma4_family_matches_id_with_underscore_normalization() -> None:
    """``gemma_4`` and ``gemma-4`` are both valid Gemma 4 markers."""
    from skulk.shared.models.capabilities import is_gemma4_family

    assert is_gemma4_family(_bare_card("mlx-community/gemma_4_e4b_it")) is True
    assert is_gemma4_family(_bare_card("mlx-community/gemma4-26b")) is True


def test_is_gemma4_family_matches_card_family_field_explicitly() -> None:
    """If a card doesn't have ``gemma-4`` in the id, family field still detects."""
    from skulk.shared.models.capabilities import is_gemma4_family

    card = _bare_card("mlx-community/some-model", family="gemma4")
    assert is_gemma4_family(card) is True


def test_is_gemma4_family_matches_via_vision_model_type() -> None:
    """Vision-declared gemma4 model_type wins even without id/family hints."""
    from skulk.shared.models.capabilities import is_gemma4_family
    from skulk.shared.models.model_cards import VisionCardConfig

    card = _bare_card("mlx-community/relabelled-model")
    card = card.model_copy(
        update={
            "vision": VisionCardConfig(
                image_token_id=258880,
                model_type="gemma4",
                boi_token_id=255999,
                eoi_token_id=258882,
            )
        }
    )
    assert is_gemma4_family(card) is True


def test_is_gemma4_family_matches_via_runtime_prompt_renderer() -> None:
    """Cards declaring runtime.prompt_renderer = Gemma4 are gemma4."""
    from skulk.shared.models.capabilities import is_gemma4_family

    card = _bare_card("mlx-community/relabelled-model")
    card = card.model_copy(
        update={
            "runtime": RuntimeCapabilityCardConfig(prompt_renderer=PromptRendererType.Gemma4)
        }
    )
    assert is_gemma4_family(card) is True


def test_is_gemma4_family_matches_via_runtime_output_parser() -> None:
    from skulk.shared.models.capabilities import is_gemma4_family

    card = _bare_card("mlx-community/relabelled-model")
    card = card.model_copy(
        update={
            "runtime": RuntimeCapabilityCardConfig(output_parser=OutputParserType.Gemma4)
        }
    )
    assert is_gemma4_family(card) is True


def test_is_gemma4_family_matches_via_tooling_format() -> None:
    """Cards declaring tooling.tool_call_format = Gemma4 are gemma4."""
    from skulk.shared.models.capabilities import is_gemma4_family

    card = _bare_card("mlx-community/relabelled-model")
    card = card.model_copy(
        update={"tooling": ToolingCardConfig(tool_call_format=ToolCallFormat.Gemma4)}
    )
    assert is_gemma4_family(card) is True


def test_is_gemma4_family_rejects_non_gemma_models() -> None:
    """Plain non-Gemma cards must not match."""
    from skulk.shared.models.capabilities import is_gemma4_family

    for model_id in (
        "mlx-community/Qwen2.5-7B-Instruct",
        "mlx-community/Llama-3.3-70B",
        "mlx-community/DeepSeek-V3.2",
        "mlx-community/Gemma-3n-E4B-it",
        "mlx-community/Gemma-2-9b-it",
    ):
        card = _bare_card(model_id)
        assert is_gemma4_family(card) is False, (
            f"non-gemma4 card {model_id} should not be detected as gemma4"
        )


def test_is_gemma4_family_handles_none_card_with_id_fallback() -> None:
    """Detection should still work via id alone when no card is supplied."""
    from skulk.shared.models.capabilities import is_gemma4_family

    assert (
        is_gemma4_family(model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"))
        is True
    )
    assert (
        is_gemma4_family(model_id=ModelId("mlx-community/Llama-3.3-70B"))
        is False
    )


def test_is_gemma4_family_handles_no_inputs_at_all() -> None:
    """Defensive: both arguments None returns False without raising."""
    from skulk.shared.models.capabilities import is_gemma4_family

    assert is_gemma4_family() is False
    assert is_gemma4_family(card=None, model_id=None) is False


def test_is_gemma4_family_resource_cards_all_match() -> None:
    """Every Gemma 4 model card shipped under resources/ must be detected."""
    import tomllib
    from pathlib import Path

    from skulk.shared.models.capabilities import is_gemma4_family

    cards_dir = (
        Path(__file__).resolve().parents[4]
        / "resources"
        / "inference_model_cards"
    )
    gemma4_paths = list(cards_dir.glob("*gemma-4*.toml")) + list(
        cards_dir.glob("*gemma_4*.toml")
    )
    assert gemma4_paths, (
        "expected gemma-4 cards under resources/inference_model_cards/"
    )
    for path in gemma4_paths:
        with path.open("rb") as fp:
            data = tomllib.load(fp)
        card = ModelCard.model_validate(data)
        assert is_gemma4_family(card) is True, f"{path.name} should be gemma4"


def test_is_gemma4_family_resource_cards_other_families_dont_match() -> None:
    """Sanity: non-Gemma-4 cards under resources/ must not detect as gemma4."""
    import tomllib
    from pathlib import Path

    from skulk.shared.models.capabilities import is_gemma4_family

    cards_dir = (
        Path(__file__).resolve().parents[4]
        / "resources"
        / "inference_model_cards"
    )
    non_gemma4_paths = [
        path
        for path in cards_dir.glob("*.toml")
        if "gemma-4" not in path.name
        and "gemma_4" not in path.name
        and "gemma4" not in path.name
    ]
    if not non_gemma4_paths:
        return  # No control set; skip silently rather than fail
    for path in non_gemma4_paths:
        with path.open("rb") as fp:
            data = tomllib.load(fp)
        card = ModelCard.model_validate(data)
        # Some non-gemma-4 cards may legitimately set tooling.tool_call_format
        # to something other than Gemma4 — those must not be detected as gemma4.
        assert is_gemma4_family(card) is False, (
            f"{path.name} should not match the gemma4 predicate"
        )


def _autoimport_card(model_id: str, family: str) -> ModelCard:
    """An auto-imported card: no built-in entry, so empty capabilities."""
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family=family,
        capabilities=[],
    )


def test_qwen3_autoimport_resolves_thinking_toggle() -> None:
    # The #384 case: an auto-imported Qwen3.6 quant with empty capabilities and a
    # raw family token must still resolve as a toggle-capable thinking model, or
    # it reasons unconditionally and returns empty content.
    card = _autoimport_card("mlx-community/Qwen3.6-35B-A3B-nvfp4", "Qwen3.6")

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking is True
    assert profile.supports_thinking_toggle is True
    assert profile.thinking_format == ReasoningFormat.TokenDelimited


def test_qwen3_point_releases_match_by_id() -> None:
    for model_id in (
        "mlx-community/Qwen3-8B-4bit",
        "mlx-community/Qwen3.5-9B-4bit",
        "mlx-community/Qwen3.6-27B-4bit",
    ):
        card = _autoimport_card(model_id, "qwen")
        profile = resolve_model_capability_profile(card.model_id, model_card=card)
        assert profile.supports_thinking_toggle is True, model_id


def test_qwen3_coder_excluded_from_thinking_default() -> None:
    # Coder variants are instruct-only; the family default must not flip them to
    # thinking when their card does not declare it.
    card = _autoimport_card("mlx-community/Qwen3-Coder-30B-A3B-Instruct", "qwen")

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking is False
    assert profile.supports_thinking_toggle is False


def test_qwen2_not_matched_by_qwen3_default() -> None:
    card = _autoimport_card("mlx-community/Qwen2.5-7B-Instruct", "qwen")

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking is False
    assert profile.supports_thinking_toggle is False


def test_qwen3_explicit_card_reasoning_still_overrides_family_default() -> None:
    # A card that explicitly disables the toggle must win over the family default.
    card = ModelCard(
        model_id=ModelId("mlx-community/Qwen3.6-Custom"),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        capabilities=[],
        reasoning=ReasoningCardConfig(supports_toggle=False),
    )

    profile = resolve_model_capability_profile(card.model_id, model_card=card)

    assert profile.supports_thinking_toggle is False


def _qwen3_card(model_id: str, capabilities: list[str]) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        capabilities=capabilities,
    )


def test_qwen3_instruct_card_declaring_no_thinking_is_respected() -> None:
    # An instruct-only Qwen3 release explicitly carded with capabilities that
    # omit "thinking" must not have the family default force a thinking
    # contract it cannot honor (the audit found five bundled Instruct-2507 /
    # Next-Instruct cards mis-resolved this way). `supports_thinking` has no
    # card-level off switch, so the family-default gate is the only guard.
    card = _qwen3_card("mlx-community/Qwen3-4B-Instruct-2507-4bit", ["text"])
    profile = resolve_model_capability_profile(card.model_id, model_card=card)
    assert profile.supports_thinking is False
    assert profile.supports_thinking_toggle is False


def test_qwen3_auto_imported_card_still_defaults_to_thinking() -> None:
    # The #384 behavior must survive: an auto-imported card arrives with EMPTY
    # capabilities (fetch_from_hf never fills the field), and the family
    # default keeps it toggle-capable so a hybrid Qwen3 does not reason
    # unconditionally into empty content.
    card = _qwen3_card("someone/Qwen3.5-9B-fresh-quant", [])
    profile = resolve_model_capability_profile(card.model_id, model_card=card)
    assert profile.supports_thinking is True
    assert profile.supports_thinking_toggle is True
    assert profile.thinking_format == ReasoningFormat.TokenDelimited


def test_qwen3_thinking_card_keeps_thinking() -> None:
    card = _qwen3_card(
        "mlx-community/Qwen3.5-9B-4bit", ["text", "thinking", "thinking_toggle"]
    )
    profile = resolve_model_capability_profile(card.model_id, model_card=card)
    assert profile.supports_thinking is True
    assert profile.supports_thinking_toggle is True


def test_qwen3_coder_stays_excluded_from_thinking_default() -> None:
    card = _qwen3_card("mlx-community/Qwen3-Coder-30B-4bit", [])
    profile = resolve_model_capability_profile(card.model_id, model_card=card)
    assert profile.supports_thinking is False


def _muse_card(model_id: str, **overrides: object) -> ModelCard:
    card = ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(100),
        n_layers=52,
        hidden_size=6656,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="muse-glimmer",
        capabilities=["text", "vision"],
    )
    return card.model_copy(update=overrides) if overrides else card


def test_muse_glimmer_family_default_contract() -> None:
    # The registry compiles the family from the upstream model_type and emits
    # no tooling/runtime/reasoning sections; the family default must still
    # resolve the wire contract Meta's chat template fixes.
    card = _muse_card("unsloth/Muse-Glimmer-30B-GGUF")
    profile = resolve_model_capability_profile(card.model_id, model_card=card)
    assert profile.supports_thinking is True
    assert profile.supports_thinking_toggle is False
    assert profile.thinking_format == ReasoningFormat.ChannelDelimited
    assert profile.default_reasoning_effort == "high"
    assert profile.disabled_reasoning_effort == "low"
    assert profile.supports_tool_calling is True
    assert profile.tool_call_format == ToolCallFormat.Atem
    assert profile.output_parser == OutputParserType.MuseGlimmer
    assert profile.supports_image_input is True
    assert profile.supports_native_multimodal is True
    assert uses_muse_glimmer_protocol(profile)


def test_muse_glimmer_family_matches_on_model_id_alone() -> None:
    card = _muse_card("mlx-community/Muse-Glimmer-30B-4bit").model_copy(
        update={"family": ""}
    )
    profile = resolve_model_capability_profile(card.model_id, model_card=card)
    assert profile.output_parser == OutputParserType.MuseGlimmer
    assert profile.tool_call_format == ToolCallFormat.Atem


def test_muse_glimmer_explicit_card_fields_still_win() -> None:
    card = _muse_card(
        "unsloth/Muse-Glimmer-30B-GGUF",
        tooling=ToolingCardConfig(supports_tool_calling=False),
        reasoning=ReasoningCardConfig(default_effort="medium"),
    )
    profile = resolve_model_capability_profile(card.model_id, model_card=card)
    assert profile.supports_tool_calling is False
    assert profile.default_reasoning_effort == "medium"
    assert profile.output_parser == OutputParserType.MuseGlimmer


def test_muse_glimmer_no_toggle_means_thinking_cannot_be_disabled() -> None:
    card = _muse_card("unsloth/Muse-Glimmer-30B-GGUF")
    profile = resolve_model_capability_profile(card.model_id, model_card=card)
    # A request that asks to disable thinking on a no-toggle model keeps its
    # explicit strength hint and drops the toggle, as the Phase 2 contract says.
    assert resolve_reasoning_params("low", False, profile) == ("low", None)
    assert resolve_reasoning_params(None, False, profile) == (None, None)


def test_muse_glimmer_template_kwargs_map_effort_to_strength() -> None:
    card = _muse_card("unsloth/Muse-Glimmer-30B-GGUF")
    profile = resolve_model_capability_profile(card.model_id, model_card=card)
    assert muse_glimmer_template_kwargs(profile, None) == {}
    assert muse_glimmer_template_kwargs(profile, "xhigh") == {
        "reasoning_strength": "xhigh"
    }
    assert muse_glimmer_template_kwargs(profile, "minimal") == {
        "reasoning_strength": "low"
    }
    assert muse_glimmer_template_kwargs(profile, "none") == {
        "reasoning_strength": "low"
    }
    generic = resolve_model_capability_profile(
        ModelId("example/other"), model_card=_base_model_card("example/other")
    )
    assert muse_glimmer_template_kwargs(generic, "high") == {}


def test_runtime_card_accepts_vllm_reasoning_parser() -> None:
    runtime = RuntimeCapabilityCardConfig(
        vllm_tool_call_parser="muse_glimmer",
        vllm_reasoning_parser="muse_glimmer",
    )
    assert runtime.vllm_reasoning_parser == "muse_glimmer"


def test_muse_glimmer_predates_the_in_process_llama_cpp_binding() -> None:
    card = _muse_card("unsloth/Muse-Glimmer-30B-GGUF")
    profile = resolve_model_capability_profile(card.model_id, model_card=card)
    assert family_predates_in_process_llama_cpp(profile)
    generic = resolve_model_capability_profile(
        ModelId("example/other"), model_card=_base_model_card("example/other")
    )
    assert not family_predates_in_process_llama_cpp(generic)
