# pyright: reportPrivateUsage=false
"""Tests for the served-runner SSE delta parser and spec-flag mapping.

The runner's HTTP/subprocess plumbing is exercised live on a GPU node; this
covers the pure parsing surface (the subtle part) without a server.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from skulk.shared.types.common import ModelId
from skulk.worker.runner.llama_server.runner import (
    _SPEC_TYPE_FLAG,
    _draft_model_args,
    _gpu_layers_for_backend,
    model_declares_reasoning,
    _parse_sse_line,
    _projector_server_args,
)


def _runtime(repo: str | None = None, file: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(served_spec_draft_repo=repo, served_spec_draft_file=file)


def test_draft_args_none_when_no_draft_and_not_required() -> None:
    # draft_mtp without a draft = baked-in heads (Qwen); ngram needs no model.
    assert _draft_model_args(_runtime(), "draft_mtp") == []
    assert _draft_model_args(_runtime(), "ngram") == []
    assert _draft_model_args(None, "draft_mtp") == []


def test_draft_args_required_modes_raise_without_draft() -> None:
    for mode in ("draft_simple", "draft_eagle3"):
        with pytest.raises(RuntimeError, match="requires a draft model"):
            _draft_model_args(_runtime(), mode)


def test_draft_args_ngram_ignores_spurious_draft_repo() -> None:
    # ngram-cache uses no --model-draft; a draft repo on an ngram card is
    # spurious and must NOT drop --spec-type ngram-cache (returns [], not None).
    assert (
        _draft_model_args(_runtime(repo="org/draft-GGUF", file="draft.gguf"), "ngram") == []
    )


def test_draft_args_repo_without_file_degrades() -> None:
    # A card that names a draft repo but no file cannot pass --model-draft;
    # degrade to plain decode (None) rather than crash the runner (#574).
    assert _draft_model_args(_runtime(repo="org/draft-GGUF"), "draft_mtp") is None


def test_draft_args_resolves_model_draft_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # With a draft repo+file present on disk, returns --model-draft <path>.
    (tmp_path / "draft.gguf").write_bytes(b"GGUF")
    import skulk.download.download_utils as du

    def _fake_build_model_path(
        _model_id: object, _source_revision: str | None = None
    ) -> Path:
        return tmp_path

    monkeypatch.setattr(du, "build_model_path", _fake_build_model_path)
    args = _draft_model_args(_runtime(repo="org/draft-GGUF", file="draft.gguf"), "draft_mtp")
    assert args == ["--model-draft", str(tmp_path / "draft.gguf")]


def test_same_repo_draft_inherits_pinned_base_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bundled draft lookup must accept the base's qualified cache directory."""

    revision = "1" * 40
    (tmp_path / "draft.gguf").write_bytes(b"GGUF")
    observed: list[tuple[object, str | None]] = []
    import skulk.download.download_utils as du

    def _fake_build_model_path(
        model_id: object, source_revision: str | None = None
    ) -> Path:
        observed.append((model_id, source_revision))
        return tmp_path

    monkeypatch.setattr(du, "build_model_path", _fake_build_model_path)

    args = _draft_model_args(
        _runtime(repo="org/bundle", file="draft.gguf"),
        "draft_mtp",
        base_model_id=ModelId("org/bundle"),
        source_revision=revision,
    )

    assert args == ["--model-draft", str(tmp_path / "draft.gguf")]
    assert observed == [(ModelId("org/bundle"), revision)]


def test_draft_args_missing_file_on_disk_degrades(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Draft dir resolves but the declared file is not present: degrade to plain
    # decode (None), not crash (#574).
    import skulk.download.download_utils as du

    def _fake_build_model_path(
        _model_id: object, _source_revision: str | None = None
    ) -> Path:
        return tmp_path

    monkeypatch.setattr(du, "build_model_path", _fake_build_model_path)
    assert (
        _draft_model_args(_runtime(repo="org/draft-GGUF", file="missing.gguf"), "draft_simple")
        is None
    )


def test_draft_args_draft_dir_absent_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    # The gemma-4-31B scenario: the cross-repo draft was never staged (its
    # best-effort co-fetch failed and was swallowed, #574), so build_model_path
    # raises FileNotFoundError. Degrade to plain decode rather than crash the
    # runner at launch.
    import skulk.download.download_utils as du

    def _raise(_model_id: object, _source_revision: str | None = None) -> Path:
        raise FileNotFoundError("Model org/draft-GGUF not found on disk")

    monkeypatch.setattr(du, "build_model_path", _raise)
    assert (
        _draft_model_args(_runtime(repo="org/draft-GGUF", file="draft.gguf"), "draft_mtp")
        is None
    )


def _card(reasoning: object = None, capabilities: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(reasoning=reasoning, capabilities=capabilities or [])


def test_reasoning_card_section_declares_reasoning() -> None:
    assert model_declares_reasoning(_card(reasoning=SimpleNamespace())) is True


def test_thinking_capability_declares_reasoning() -> None:
    assert model_declares_reasoning(_card(capabilities=["text", "thinking"])) is True


def test_plain_text_card_declares_no_reasoning() -> None:
    # Gemma 4 served card: reasoning=None, capabilities=["text"] -> served with
    # --reasoning-format none so output stays in message.content.
    assert model_declares_reasoning(_card(capabilities=["text"])) is False
    assert model_declares_reasoning(_card()) is False


def test_parse_content_delta() -> None:
    d = _parse_sse_line('data: {"choices":[{"delta":{"content":"hi"}}]}')
    assert d is not None
    assert d.content == "hi"
    assert d.reasoning == ""
    assert d.finish is None
    assert d.done is False


def test_parse_reasoning_delta_is_separate() -> None:
    # reasoning_content rides its own field so the runner can flag is_thinking.
    d = _parse_sse_line('data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}')
    assert d is not None
    assert d.reasoning == "hmm"
    assert d.content == ""


def test_parse_finish_length_is_preserved() -> None:
    # A max_tokens truncation must surface as "length", not be masked as "stop".
    d = _parse_sse_line(
        'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"length"}]}'
    )
    assert d is not None
    assert d.content == "x"
    assert d.finish == "length"


def test_parse_finish_stop_and_eos_map_to_stop() -> None:
    for reason in ("stop", "eos_token"):
        d = _parse_sse_line(
            f'data: {{"choices":[{{"delta":{{}},"finish_reason":"{reason}"}}]}}'
        )
        assert d is not None
        assert d.finish == "stop"


def test_parse_done_sentinel() -> None:
    d = _parse_sse_line("data: [DONE]")
    assert d is not None
    assert d.done is True


def test_parse_skips_non_data_and_blank_lines() -> None:
    assert _parse_sse_line("") is None
    assert _parse_sse_line(": keep-alive comment") is None
    assert _parse_sse_line("event: message") is None


def test_parse_skips_malformed_json_and_choiceless() -> None:
    # A stray/garbled line must not break the stream (returns None to skip).
    assert _parse_sse_line("data: {not json") is None
    assert _parse_sse_line('data: {"choices":[]}') is None
    assert _parse_sse_line('data: {"id":"x"}') is None


def test_spec_type_flag_maps_to_llama_server_flags() -> None:
    # The card's served_spec_type underscores become the llama-server hyphen flags.
    assert _SPEC_TYPE_FLAG["draft_mtp"] == "draft-mtp"
    assert _SPEC_TYPE_FLAG["draft_eagle3"] == "draft-eagle3"
    assert _SPEC_TYPE_FLAG["draft_simple"] == "draft-simple"
    assert _SPEC_TYPE_FLAG["draft_dflash"] == "draft-dflash"
    # ngram is the special case: it maps to ngram-cache, not "ngram".
    assert _SPEC_TYPE_FLAG["ngram"] == "ngram-cache"


@pytest.mark.parametrize(
    ("resolved", "expected"),
    [
        ("llama_server-vulkan", "99"),  # GPU compute tag -> full offload
        ("llama_server-rocm", "99"),
        ("llama_server-cuda", "99"),
        ("llama_server-cpu", "0"),  # CPU tag was RAM-admitted -> no GPU offload
        ("llama_server", "0"),  # bare tag is NOT GPU-offload (RAM-admitted)
        (None, "99"),  # no resolution (manual/fallback) -> default GPU offload
    ],
)
def test_gpu_layers_match_vram_admission(resolved: str | None, expected: str) -> None:
    # The runner's -ngl decision must mirror placement_utils._has_gpu_offload_backend
    # so a RAM-admitted placement never grabs an unbudgeted GPU.
    assert _gpu_layers_for_backend(resolved) == expected


def test_projector_args_disable_offload_only_for_cpu() -> None:
    """Accelerators keep default projector offload while CPU opts out explicitly."""

    projector = Path("/models/mmproj-F16.gguf")
    assert _projector_server_args(projector, "llama_server-cuda") == [
        "--mmproj",
        str(projector),
    ]
    assert _projector_server_args(projector, "llama_server-cpu") == [
        "--mmproj",
        str(projector),
        "--no-mmproj-offload",
    ]
    assert _projector_server_args(None, "llama_server-cuda") == []


def test_parse_sse_line_extracts_final_chunk_timings() -> None:
    # llama-server attaches its native timings to the final streamed chunk
    # when the request set timings_per_token; they become GenerationStats.
    line = (
        'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], '
        '"timings": {"prompt_n": 50, "prompt_ms": 250.0, '
        '"predicted_n": 128, "predicted_ms": 4000.0}}'
    )
    delta = _parse_sse_line(line)
    assert delta is not None
    assert delta.finish == "stop"
    assert delta.timings == {
        "prompt_n": 50,
        "prompt_ms": 250.0,
        "predicted_n": 128,
        "predicted_ms": 4000.0,
    }


def test_parse_sse_line_timings_absent_or_malformed_is_none() -> None:
    plain = _parse_sse_line('data: {"choices": [{"delta": {"content": "hi"}}]}')
    assert plain is not None
    assert plain.timings is None
    wrong_shape = _parse_sse_line(
        'data: {"choices": [{"delta": {}}], "timings": [1, 2]}'
    )
    assert wrong_shape is not None
    assert wrong_shape.timings is None


def test_parse_sse_line_skips_non_dict_choice_and_delta_shapes() -> None:
    # A malformed payload is skipped (or treated as empty), never raised: one
    # stray line must not break the whole stream.
    assert _parse_sse_line('data: {"choices": ["garbage"]}') is None
    listy_delta = _parse_sse_line('data: {"choices": [{"delta": []}]}')
    assert listy_delta is not None
    assert listy_delta.content == ""
    assert listy_delta.reasoning == ""


def test_parse_sse_line_skips_non_list_choices() -> None:
    assert _parse_sse_line('data: {"choices": {"0": {}}}') is None
    assert _parse_sse_line('data: [1, 2, 3]') is None
