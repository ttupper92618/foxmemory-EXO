"""Tests for thinking-control forwarding to llama-server (#428/#420)."""

from __future__ import annotations

from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory
from skulk.shared.types.text_generation import (
    InputMessage,
    TextGenerationTaskParams,
)
from skulk.worker.runner.llama_server.runner import reasoning_request_overrides


def _params(**kwargs: object) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("m"),
        input=[InputMessage(role="user", content="hi")],
        **kwargs,  # type: ignore[arg-type]
    )


def test_enable_thinking_false_forwards_chat_template_kwargs() -> None:
    # The throughput-cell case: enable_thinking=False must reach llama-server, or
    # the model reasons through the whole budget and returns empty content (#428).
    overrides = reasoning_request_overrides(_params(enable_thinking=False))
    assert overrides["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in overrides


def test_enable_thinking_true_forwards_toggle() -> None:
    overrides = reasoning_request_overrides(_params(enable_thinking=True))
    assert overrides["chat_template_kwargs"] == {"enable_thinking": True}


def test_reasoning_effort_forwarded_but_none_dropped() -> None:
    assert reasoning_request_overrides(_params(reasoning_effort="high")) == {
        "reasoning_effort": "high"
    }
    # "none" is not a valid server effort; disabling is expressed via
    # enable_thinking, so it must not be forwarded as reasoning_effort.
    assert reasoning_request_overrides(_params(reasoning_effort="none")) == {}


def test_no_controls_yields_no_overrides() -> None:
    # Neither set -> let the model's own default behavior stand.
    assert reasoning_request_overrides(_params()) == {}


def test_both_controls_combine() -> None:
    overrides = reasoning_request_overrides(
        _params(enable_thinking=True, reasoning_effort="medium")
    )
    assert overrides == {
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_effort": "medium",
    }


def _muse_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("unsloth/Muse-Glimmer-30B-GGUF"),
        storage_size=Memory.from_mb(100),
        n_layers=52,
        hidden_size=6656,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="muse-glimmer",
        capabilities=["text", "vision"],
    )


def test_muse_glimmer_effort_becomes_reasoning_strength() -> None:
    # Muse Glimmer's template reads reasoning_strength, not enable_thinking or
    # the harmony reasoning_effort field; both generic levers are dropped so a
    # toggle the template cannot honor is never sent as a false promise.
    overrides = reasoning_request_overrides(
        _params(reasoning_effort="xhigh", enable_thinking=True), _muse_card()
    )
    assert overrides == {"chat_template_kwargs": {"reasoning_strength": "xhigh"}}


def test_muse_glimmer_disable_request_maps_to_low() -> None:
    overrides = reasoning_request_overrides(
        _params(reasoning_effort="none"), _muse_card()
    )
    assert overrides == {"chat_template_kwargs": {"reasoning_strength": "low"}}


def test_muse_glimmer_without_effort_leaves_template_default() -> None:
    assert reasoning_request_overrides(_params(), _muse_card()) == {}


def test_other_families_keep_generic_levers_with_a_card() -> None:
    card = _muse_card().model_copy(
        update={"model_id": ModelId("Qwen/Qwen3.6-27B-FP8"), "family": "qwen"}
    )
    overrides = reasoning_request_overrides(
        _params(enable_thinking=False, reasoning_effort="low"), card
    )
    assert overrides == {
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "low",
    }
