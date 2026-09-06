# pyright: reportPrivateUsage=false
"""Tests for the backend capability tag vocabulary.

Probing and derivation behavior (which tags a node advertises given what it
observed and declared) lives in ``skulk.facts`` and is tested with synthetic
facts in ``src/skulk/facts/tests/``; this module covers the tag vocabulary,
resolution, and the thin ``probe_node_backends`` delegation.
"""

import sys

import pytest

import skulk.facts
from skulk.facts.derive import BackendDerivation
from skulk.shared.backends import (
    engine_of,
    engine_supports_multi_node,
    make_backend_tag,
    platform_compatible_backends,
    probe_node_backends,
    resolve_node_backend,
    resolve_node_engine,
)


def test_make_backend_tag_is_compound() -> None:
    assert make_backend_tag("mlx", "metal") == "mlx-metal"
    assert make_backend_tag("mlx_audio", "metal") == "mlx_audio-metal"
    assert make_backend_tag("llama_cpp", "vulkan") == "llama_cpp-vulkan"


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("mlx", "mlx"),
        ("mlx-metal", "mlx"),
        ("mlx_audio", "mlx_audio"),
        ("mlx_audio-metal", "mlx_audio"),
        ("llama_cpp", "llama_cpp"),
        ("llama_cpp-vulkan", "llama_cpp"),
        ("llama_cpp-rocm", "llama_cpp"),
        ("cuda", None),  # bare compute, no engine
        ("vllm", "vllm"),
        ("vllm-cuda", "vllm"),
        ("bogus-cuda", None),  # unknown engine
        ("", None),
    ],
)
def test_engine_of(tag: str, expected: str | None) -> None:
    assert engine_of(tag) == expected


def test_probe_includes_mlx_on_darwin() -> None:
    tags = probe_node_backends()
    if sys.platform == "darwin":
        # Bare engine tag kept for back-compat with original {"mlx"} cards.
        assert "mlx" in tags
        assert "mlx-metal" in tags
    else:
        assert "mlx" not in tags


def test_probe_node_backends_delegates_to_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The public probe entry point is a thin delegate over the process-wide
    # facts snapshot; consumers must see exactly the derived tags.
    monkeypatch.setattr(
        skulk.facts,
        "current_backend_derivation",
        lambda: BackendDerivation(backends=frozenset({"llama_server-cuda"})),
    )
    assert probe_node_backends() == frozenset({"llama_server-cuda"})


def test_resolve_node_engine_existing_mlx_cards_unchanged() -> None:
    # An original {"mlx"} card on a Mac node ({"mlx","mlx-metal"}) -> mlx.
    engine = resolve_node_engine(
        frozenset({"mlx"}), (), frozenset({"mlx", "mlx-metal"})
    )
    assert engine == "mlx"


def test_resolve_node_engine_picks_llama_cpp() -> None:
    engine = resolve_node_engine(
        frozenset({"llama_cpp-vulkan", "llama_cpp-rocm", "llama_cpp-cpu"}),
        ("llama_cpp-vulkan", "llama_cpp-rocm"),
        frozenset({"llama_cpp", "llama_cpp-vulkan"}),
    )
    assert engine == "llama_cpp"


def test_resolve_node_engine_none_when_no_intersection() -> None:
    # Node advertises only mlx but the card requires llama_cpp -> no match
    # (placement would have excluded this node; caller falls back to default).
    engine = resolve_node_engine(
        frozenset({"llama_cpp-vulkan"}), (), frozenset({"mlx", "mlx-metal"})
    )
    assert engine is None


def test_resolve_node_backend_returns_preferred_tag() -> None:
    # The winning tag (not just the engine) honors backend_preference order.
    tag = resolve_node_backend(
        frozenset({"llama_cpp-vulkan", "llama_cpp-rocm", "llama_cpp-cpu"}),
        ("llama_cpp-rocm", "llama_cpp-vulkan"),
        frozenset({"llama_cpp-vulkan", "llama_cpp-rocm"}),
    )
    assert tag == "llama_cpp-rocm"


def test_resolve_node_backend_falls_back_to_sorted_when_no_preference() -> None:
    # With no preference match, the intersection is ordered deterministically,
    # and a CPU compute tag never beats a GPU tag on alphabetical accident:
    # GPU serving dominates CPU for every model class shipped, so the platform
    # default puts -cpu last (a card can still prefer CPU explicitly).
    tag = resolve_node_backend(
        frozenset({"llama_cpp-vulkan", "llama_cpp-cpu"}),
        (),
        frozenset({"llama_cpp-vulkan", "llama_cpp-cpu"}),
    )
    assert tag == "llama_cpp-vulkan"
    # The served-engine shape from the card sweep: no llama_server preference,
    # node advertises GPU + CPU server builds -> GPU wins, never -ngl 0.
    tag = resolve_node_backend(
        frozenset({"llama_server-vulkan", "llama_server-cpu"}),
        ("llama_cpp-vulkan",),
        frozenset({"llama_server-vulkan", "llama_server-cpu"}),
    )
    assert tag == "llama_server-vulkan"
    # An explicit CPU preference still wins: the fallback order is a platform
    # default, not an override of card policy.
    tag = resolve_node_backend(
        frozenset({"llama_cpp-vulkan", "llama_cpp-cpu"}),
        ("llama_cpp-cpu",),
        frozenset({"llama_cpp-vulkan", "llama_cpp-cpu"}),
    )
    assert tag == "llama_cpp-cpu"


def test_platform_compatible_backends_gates_vision_off_served() -> None:
    # MODEL truth vs PLATFORM truth: a vision card keeps every declared tag on
    # engines whose runner can load its projector (in-process llama.cpp, MLX)
    # and loses the served llama_server tags until that runner passes mmproj.
    declared = frozenset(
        {
            "mlx",
            "llama_cpp-vulkan",
            "llama_cpp-cpu",
            "llama_server-vulkan",
            "llama_server-cpu",
        }
    )
    assert platform_compatible_backends(declared, card_serves_vision=True) == (
        frozenset({"mlx", "llama_cpp-vulkan", "llama_cpp-cpu"})
    )
    # A text-only card is untouched: no platform gate applies.
    assert (
        platform_compatible_backends(declared, card_serves_vision=False) == declared
    )
    assert platform_compatible_backends(
        declared,
        card_serves_vision=True,
        card_has_pinned_projector=True,
    ) == declared


def test_vision_card_preferring_llama_server_still_resolves_to_llama_cpp() -> None:
    # A vision GGUF card ships with a served-first backend_preference (MODEL
    # truth: the model's artifacts run on llama_server too). On a node that
    # advertises BOTH engines, the platform filter must still strip the served
    # tags before resolution so the card lands on the in-process llama.cpp runner
    # that can load its mmproj projector. This guards the card sweep that made
    # every GGUF card prefer llama_server: the vision exception is owned entirely
    # by code, so the served preference on a vision card is a safe no-op.
    declared = frozenset(
        {
            "llama_cpp-vulkan",
            "llama_cpp-cpu",
            "llama_server-vulkan",
            "llama_server-cpu",
        }
    )
    served_first = (
        "llama_server-vulkan",
        "llama_server-cpu",
        "llama_cpp-vulkan",
        "llama_cpp-cpu",
    )
    node_backends = frozenset({"llama_server-vulkan", "llama_cpp-vulkan"})
    platform_backends = platform_compatible_backends(
        declared, card_serves_vision=True
    )
    assert "llama_server-vulkan" not in platform_backends
    tag = resolve_node_backend(platform_backends, served_first, node_backends)
    assert tag == "llama_cpp-vulkan"
    # A text card on the same node honors the served preference (no gate applies).
    text_backends = platform_compatible_backends(declared, card_serves_vision=False)
    assert (
        resolve_node_backend(text_backends, served_first, node_backends)
        == "llama_server-vulkan"
    )


def test_platform_compatible_backends_gates_speech_to_mlx_audio() -> None:
    # Speech cards stay on the dedicated speech engine until another runner owns
    # the TTS/STT contract.
    declared = frozenset(
        {
            "mlx",
            "mlx-metal",
            "mlx_audio",
            "mlx_audio-metal",
            "llama_cpp-vulkan",
            "future_speech_engine",
        }
    )
    assert platform_compatible_backends(
        declared,
        card_serves_vision=False,
        card_serves_speech=True,
    ) == frozenset(
        {"mlx_audio", "mlx_audio-metal", "future_speech_engine"}
    )


def test_platform_compatible_backends_requires_vllm_tool_parser() -> None:
    """Tool-using cards cannot land on a vLLM server without its parser pair."""
    declared = frozenset({"mlx", "vllm-cuda", "vllm-rocm"})

    assert platform_compatible_backends(
        declared,
        card_serves_vision=False,
        card_supports_tool_calling=True,
    ) == frozenset({"mlx"})
    assert platform_compatible_backends(
        declared,
        card_serves_vision=False,
        card_supports_tool_calling=True,
        card_vllm_tool_call_parser="qwen3_xml",
    ) == declared
    assert platform_compatible_backends(
        declared,
        card_serves_vision=False,
        card_supports_tool_calling=False,
    ) == declared


def test_resolve_node_backend_none_when_no_intersection() -> None:
    assert (
        resolve_node_backend(
            frozenset({"llama_cpp-vulkan"}), (), frozenset({"mlx", "mlx-metal"})
        )
        is None
    )


def test_resolve_node_engine_matches_backend_engine() -> None:
    # resolve_node_engine is exactly engine_of(resolve_node_backend(...)).
    compatible = frozenset({"llama_cpp-vulkan", "mlx"})
    preference = ("llama_cpp-vulkan",)
    node = frozenset({"llama_cpp-vulkan", "mlx"})
    tag = resolve_node_backend(compatible, preference, node)
    assert tag == "llama_cpp-vulkan"
    assert resolve_node_engine(compatible, preference, node) == engine_of(tag)


def test_engine_supports_multi_node() -> None:
    # MLX shards across nodes (ring/jaccl); llama.cpp is single-node until its
    # RPC runner lands (#328). This is the placement single-node guard's hinge.
    assert engine_supports_multi_node("mlx") is True
    assert engine_supports_multi_node("mlx_audio") is False
    assert engine_supports_multi_node("llama_cpp") is False


def test_platform_compatible_backends_gates_new_families_off_in_process_llama_cpp() -> None:
    """A family the pinned binding predates keeps only its served llama.cpp lanes."""
    declared = frozenset(
        {
            "llama_cpp-vulkan",
            "llama_cpp-cuda",
            "llama_cpp-cpu",
            "llama_server-vulkan",
            "llama_server-cuda",
            "mlx",
        }
    )
    assert platform_compatible_backends(
        declared,
        card_serves_vision=False,
        card_family_predates_in_process_binding=True,
    ) == frozenset({"llama_server-vulkan", "llama_server-cuda", "mlx"})
    assert (
        platform_compatible_backends(declared, card_serves_vision=False) == declared
    )
