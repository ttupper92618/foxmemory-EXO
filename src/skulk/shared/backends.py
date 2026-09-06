"""Backend capability tags: the shared vocabulary for engine + compute routing.

A *backend tag* names how a model is actually executed on a node, in the form
``<engine>-<compute>`` (for example ``mlx-metal``, ``llama_cpp-vulkan``,
``llama_cpp-rocm``). Two axes are deliberately folded into one self-describing
string:

- **engine** -- which inference runtime loads and runs the model (``mlx`` vs
  ``llama_cpp``). This is what selects the worker runner class.
- **compute** -- which compute backend that runtime drives on a given node
  (Apple ``metal``; for llama.cpp ``vulkan`` / ``rocm`` / ``cuda`` / ``cpu``).
  The *same* model file runs identically across compute backends, but their
  performance differs per model, so a card may prefer one (see
  ``PlacementCardConfig.backend_preference``).

Nodes advertise the set of tags they can actually serve (probed +
operator-declared) in ``NodeResources.backends``; cards declare the set they are
*allowed* on (``compatible_backends``, a hard placement filter) and an ordered
*preference* among them (soft, with graceful fallback). Keeping the filter and
the preference separate is what lets a model say "fastest on Vulkan, but ROCm is
fine" and still place on a ROCm-only node.

For backward compatibility a node also advertises the bare engine tag alongside
each compound tag (a Mac advertises both ``mlx`` and ``mlx-metal``), so cards
written against the original ``{"mlx"}`` vocabulary keep matching unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Literal

from loguru import logger


def _is_executable_file(path: str) -> bool:
    """Whether ``path`` names an existing executable file."""
    return os.path.isfile(path) and os.access(path, os.X_OK)

EngineType = Literal["mlx", "mlx_audio", "llama_cpp", "llama_server", "vllm"]
"""Inference runtime that loads and runs a model; selects the worker runner.

``llama_server`` is a *served-backend* engine: instead of loading the model
in-process (like ``mlx`` and ``llama_cpp``), the worker launches an external
``llama-server`` subprocess and proxies its OpenAI HTTP API. It is the only way
to reach llama.cpp's native multi-token-prediction speculative decoding
(``--spec-type draft-mtp``), whose orchestration lives in the server application
rather than ``libllama`` / the Python binding. It coexists with the in-process
``llama_cpp`` engine and is selected per model by the card's ``compatible_backends``.

``vllm`` is a second served-backend engine (same managed-subprocess-plus-proxy
shape as ``llama_server``): the worker launches an external ``vllm serve`` process
and proxies its OpenAI HTTP API. It is the GPU-serving fast path -- vLLM's
continuous batching and paged attention hold latency flat and grow aggregate
throughput under concurrent load where the single-stream engines collapse (a
benchmark on an A100 measured llama.cpp's time-to-first-token hitting ~31s at
64-way concurrency vs vLLM's ~0.5s). It coexists with ``llama_cpp`` /
``llama_server``: the planner picks by hardware and expected concurrency, since
vLLM's win is concurrency, not single-stream (on GPUs without native FP4 the
in-process engines can beat it for one request). GPU-only in scope: ``cuda``
(NVIDIA) and ``rocm`` (AMD CDNA).

``mlx_audio`` is the single-node speech engine backed by the upstream
``mlx-audio`` package. It is kept separate from ``mlx`` because TTS/STT model
loading, generation, and future realtime session contracts are not the same as
the text/vision MLX runner.
"""

ComputeBackend = Literal["metal", "vulkan", "rocm", "cuda", "cpu"]
"""Compute backend a runtime drives on a node."""

# Explicit typed tuples (rather than typing.get_args, which erases to Any) so the
# values stay narrowed to their Literal types where they are consumed.
_ENGINES: Final[tuple[EngineType, ...]] = (
    "mlx",
    "mlx_audio",
    "llama_cpp",
    "llama_server",
    "vllm",
)
_COMPUTE_BACKENDS: Final[tuple[ComputeBackend, ...]] = (
    "metal",
    "vulkan",
    "rocm",
    "cuda",
    "cpu",
)

_TAG_SEPARATOR: Final = "-"

# Operator declaration of which llama.cpp compute backends a node was built with,
# as a comma-separated list (e.g. "vulkan" or "vulkan,rocm"). The compiled build
# -- not what libraries happen to be installed -- determines what llama.cpp can
# actually use, and the Python binding does not cleanly expose that, so we treat
# this as authoritative operator policy (mirroring SKULK_NODE_PARTICIPATION).
LLAMA_CPP_BACKENDS_ENV: Final = "SKULK_LLAMA_CPP_BACKENDS"

# Path to the external ``llama-server`` binary the served-backend engine launches.
# Set this on a node that should serve models via the ``llama_server`` engine
# (e.g. for native MTP). Absent => the node does not advertise ``llama_server``,
# so it is never a placement candidate for served-engine cards. The binary must
# be a build recent enough to expose ``--spec-type`` (>= b9196 for ``draft-mtp``).
LLAMA_SERVER_BIN_ENV: Final = "SKULK_LLAMA_SERVER_BIN"

# Compute backends the ``llama-server`` build was compiled with (comma-separated,
# e.g. "vulkan" or "vulkan,rocm"), same vocabulary as ``SKULK_LLAMA_CPP_BACKENDS``.
# When unset, the served engine falls back to the node's llama.cpp backend
# declaration (the GPU is the same regardless of which engine drives it), then to
# ``cpu``.
LLAMA_SERVER_BACKENDS_ENV: Final = "SKULK_LLAMA_SERVER_BACKENDS"

# Path to the ``vllm`` CLI the vLLM served-backend engine launches (``vllm serve``).
# Set this on a GPU node that should serve models via the ``vllm`` engine. Absent
# => the node does not advertise ``vllm`` and is never a placement candidate for
# vLLM cards. Mirrors ``SKULK_LLAMA_SERVER_BIN``.
VLLM_BIN_ENV: Final = "SKULK_VLLM_BIN"

# Compute backends the vLLM install targets (comma-separated). vLLM is GPU-only in
# our scope, so only ``cuda`` / ``rocm`` are honored (``metal`` / ``vulkan`` /
# ``cpu`` are ignored). When unset, falls back to the node's
# ``SKULK_LLAMA_SERVER_BACKENDS`` then ``SKULK_LLAMA_CPP_BACKENDS`` declaration
# (the GPU is the same whichever engine drives it). Unlike the llama_server probe
# there is no ``cpu`` fallback: a vLLM node with no GPU backend is not useful.
VLLM_BACKENDS_ENV: Final = "SKULK_VLLM_BACKENDS"

# vLLM compute backends we support advertising. GPU-only: NVIDIA CUDA and AMD
# CDNA ROCm. vLLM's Vulkan/Metal/CPU paths are out of scope for placement.
_VLLM_COMPUTE_BACKENDS: Final[tuple[ComputeBackend, ...]] = ("cuda", "rocm")

# Path to the ``ggml-rpc-server`` binary an RPC memory-donor runner launches
# (#328, multi-node GGUF pooling). Optional: when unset, the donor looks for
# ``ggml-rpc-server`` next to the node's ``SKULK_LLAMA_SERVER_BIN`` (the two are
# built together by ``cmake --build build --target ggml-rpc-server llama-server``
# with ``-DGGML_RPC=ON``; note the target was renamed upstream from
# ``rpc-server``).
RPC_SERVER_BIN_ENV: Final = "SKULK_RPC_SERVER_BIN"


def rpc_server_binary() -> str | None:
    """Resolve the ``ggml-rpc-server`` binary path for an RPC donor runner.

    Prefers the explicit ``SKULK_RPC_SERVER_BIN`` override; otherwise looks for
    a ``ggml-rpc-server`` sibling of ``SKULK_LLAMA_SERVER_BIN`` (they are built
    from the same llama.cpp tree). Returns ``None`` when neither yields an
    executable file, in which case the donor runner fails loudly at spawn.
    """
    explicit = os.environ.get(RPC_SERVER_BIN_ENV, "").strip()
    if explicit:
        if _is_executable_file(explicit):
            return explicit
        # #462: without this, a donor spawn failure on a node with a typo'd
        # override reads exactly like the env var was never set.
        logger.error(
            f"{RPC_SERVER_BIN_ENV} is set to {explicit!r} but that path is not "
            "an executable file; RPC donor spawns on this node will fail until "
            "it is fixed or unset."
        )
        return None
    server = os.environ.get(LLAMA_SERVER_BIN_ENV, "").strip()
    if not server:
        return None
    sibling = str(Path(server).resolve().parent / "ggml-rpc-server")
    return sibling if _is_executable_file(sibling) else None


def make_backend_tag(engine: EngineType, compute: ComputeBackend) -> str:
    """Return the compound ``<engine>-<compute>`` tag for an engine + compute pair."""
    return f"{engine}{_TAG_SEPARATOR}{compute}"


def engine_of(tag: str) -> EngineType | None:
    """Return the engine a backend tag selects, or ``None`` if it names no known engine.

    Accepts both compound tags (``llama_cpp-vulkan``) and bare engine tags
    (``llama_cpp``). Returns ``None`` for unrecognized strings so callers can
    skip tags they do not understand rather than crash on forward-compat input.
    """
    for engine in _ENGINES:
        if tag == engine or tag.startswith(f"{engine}{_TAG_SEPARATOR}"):
            return engine
    return None


# Engines that can serve a model sharded across multiple nodes. MLX has the
# multi-node ring / jaccl path. The served ``llama_server`` engine pools memory
# across nodes via llama.cpp's RPC backend (#328): one driver node runs
# ``llama-server --rpc donor:port,...`` and each donor runs ``ggml-rpc-server``;
# llama.cpp splits weights/KV across the devices itself, so Skulk computes no
# GGUF layer math. The in-process ``llama_cpp`` engine stays single-node: the
# Python binding cannot drive the RPC backend, and its runner asserts
# ``world_size == 1``. This is the single place that capability lives.
_MULTI_NODE_ENGINES: Final[frozenset[EngineType]] = frozenset({"mlx", "llama_server"})


def engine_supports_multi_node(engine: EngineType) -> bool:
    """Whether an engine can serve a model sharded across more than one node.

    Placement uses this to pin a model to a single-node cycle when none of its
    compatible engines can shard across nodes (otherwise the placement would
    download and then crash at runner startup with ``world_size != 1``). MLX
    (ring/jaccl) and the served ``llama_server`` engine (RPC driver + donors,
    #328) are multi-node capable; the in-process ``llama_cpp`` engine is
    single-node (binding gap).
    """
    return engine in _MULTI_NODE_ENGINES


# Modalities each engine's RUNNER can currently exploit on THIS platform.
# This is PLATFORM truth, deliberately separate from the model card: a card's
# [vision] section declares what the MODEL can do (its projector artifact
# exists and is grounded); this table declares which of our runner
# implementations can actually serve it. The served ``llama_server`` engine is
# admitted conditionally below only when a card pins one exact projector; legacy
# cards stay on the in-process compatibility path. Keeping the limitation here
# rather than deleting model capability from cards preserves model/platform
# truth and makes newly compiled cards light up without a broad card sweep.
_VISION_SERVING_ENGINES: Final[frozenset[EngineType]] = frozenset(
    {"mlx", "llama_cpp"}
)
_SPEECH_SERVING_ENGINES: Final[frozenset[EngineType]] = frozenset({"mlx_audio"})

# Engines whose runner binding cannot LOAD a family the served sibling serves
# fine. The in-process ``llama_cpp`` engine runs whatever llama.cpp build the
# pinned llama-cpp-python binding vendors, and that build trails the served
# engine's pin by months; a family that landed upstream in between (Muse
# Glimmer, merged 2026-08-10, first shipped in b10353, while llama-cpp-python
# 0.3.30 vendors a 2026-06-16 build) loads through ``llama_server`` only. The
# call sites resolve the family from the capability profile, so registry,
# bundled, and custom cards all hit the same gate; the table of affected
# families is the resolver's, this is the platform consequence. Drop the gate
# for a family once the binding advances past its first supporting build.
_FAMILY_GATED_ENGINES: Final[frozenset[EngineType]] = frozenset({"llama_cpp"})


def platform_compatible_backends(
    compatible_backends: frozenset[str],
    *,
    card_serves_vision: bool,
    card_serves_speech: bool = False,
    card_has_pinned_projector: bool = False,
    card_supports_tool_calling: bool = False,
    card_vllm_tool_call_parser: str | None = None,
    card_family_predates_in_process_binding: bool = False,
) -> frozenset[str]:
    """Filter a card's declared backends down to what this platform can serve.

    The card's ``compatible_backends`` is MODEL truth (which engines the model's
    artifacts run on); this helper subtracts current PLATFORM limitations (which
    of our runners can exploit the card's declared capabilities) so the two are
    never conflated on the card itself. Today the platform gates are vision
    and speech: a card with a ``[vision]`` section is kept off engines whose
    runner cannot load its projector, and a TTS/STT card is kept off non-speech
    engines until the ``mlx_audio`` runner owns that contract. Placement and the
    worker's engine resolution both apply this filter, keeping master and
    worker in agreement.

    Args:
        compatible_backends: the card's declared backend tags.
        card_serves_vision: whether the card declares a vision capability.
        card_serves_speech: whether the card declares a speech capability.
        card_has_pinned_projector: whether a vision card identifies one exact
            immutable projector that the served runner can validate and load.
        card_supports_tool_calling: whether the model exposes tool calling.
        card_vllm_tool_call_parser: exact vLLM parser pinned by the card. vLLM
            tool requests fail closed when this is absent, so those cards are
            not platform-compatible with vLLM for resident tool use.
        card_family_predates_in_process_binding: whether the card resolves to
            a family the in-process llama.cpp binding cannot load yet (see
            ``_FAMILY_GATED_ENGINES``); those cards keep only their served
            llama.cpp backends.

    Returns:
        The subset of tags whose engine can serve everything the card declares.
    """
    filtered = compatible_backends
    if card_serves_vision:
        vision_engines = (
            _VISION_SERVING_ENGINES | frozenset({"llama_server"})
            if card_has_pinned_projector
            else _VISION_SERVING_ENGINES
        )
        filtered = frozenset(
            tag
            for tag in filtered
            if (engine := engine_of(tag)) is None or engine in vision_engines
        )
    if card_serves_speech:
        filtered = frozenset(
            tag
            for tag in filtered
            if (engine := engine_of(tag)) is None or engine in _SPEECH_SERVING_ENGINES
        )
    if card_supports_tool_calling and card_vllm_tool_call_parser is None:
        filtered = frozenset(tag for tag in filtered if engine_of(tag) != "vllm")
    if card_family_predates_in_process_binding:
        filtered = frozenset(
            tag for tag in filtered if engine_of(tag) not in _FAMILY_GATED_ENGINES
        )
    return filtered


def resolve_node_backend(
    compatible_backends: frozenset[str],
    backend_preference: tuple[str, ...],
    node_backends: frozenset[str],
) -> str | None:
    """Resolve the winning backend TAG a node should use to serve a model.

    Intersects the card's ``compatible_backends`` (hard filter) with the node's
    advertised ``node_backends``, orders the result by ``backend_preference``
    (preferred tags first, then the rest deterministically), and returns the top
    tag (e.g. ``"llama_cpp-vulkan"``). Returns ``None`` when the node advertises
    none of the model's compatible backends. This is the single point that turns
    the backend-tag vocabulary into a concrete choice; both the master (to stamp
    ``resolved_backend`` on a shard at placement, #330) and the worker (to pick a
    runner) go through it so the two cannot disagree.
    """
    intersection = compatible_backends & node_backends
    if not intersection:
        return None
    ordered = [tag for tag in backend_preference if tag in intersection]
    # Fallback for tags outside the card's preference list: deterministic, but
    # never let a CPU compute tag beat a GPU tag on alphabetical accident --
    # GPU serving dominates CPU for every model class we ship, and a card
    # without an explicit llama_server preference would otherwise resolve
    # ``llama_server-cpu`` over ``llama_server-vulkan`` and run ``-ngl 0`` on a
    # GPU-admitted node. Platform default, not card policy.
    ordered += [
        tag
        for tag in sorted(
            intersection, key=lambda tag: (tag.endswith(f"{_TAG_SEPARATOR}cpu"), tag)
        )
        if tag not in backend_preference
    ]
    return ordered[0]


def resolve_node_engine(
    compatible_backends: frozenset[str],
    backend_preference: tuple[str, ...],
    node_backends: frozenset[str],
) -> EngineType | None:
    """Resolve which engine a node should use to serve a model.

    Thin wrapper over :func:`resolve_node_backend` that maps the winning tag to
    its engine. Returns ``None`` when the node advertises none of the model's
    compatible backends (which placement should already have ruled out, so the
    caller treats it as "fall back to the default engine").
    """
    tag = resolve_node_backend(compatible_backends, backend_preference, node_backends)
    return engine_of(tag) if tag is not None else None


def probe_node_backends() -> frozenset[str]:
    """Return the backend tags this node can actually serve.

    Backed by the Node Facts subsystem (#614): one probe pass per process
    gathers what the node observes about its hardware, software, and declared
    configuration (``skulk.facts.probe``), and a pure derivation turns that
    into advertised tags (``skulk.facts.derive``). Detection creates
    capability, ``SKULK_*_BACKENDS`` declarations override it, and every
    disagreement between the two is a loud
    :class:`~skulk.shared.types.node_facts.CapabilityConflict` (logged at
    startup and advertised on ``NodeResources`` into ``nodeHealth``) -- never a
    silent CPU default (#609, #612, #462).

    macOS nodes advertise ``{"mlx", "mlx-metal"}`` (plus ``mlx_audio`` tags
    when importable); any node with an importable ``llama_cpp`` adds its
    llama.cpp tags; a node with ``SKULK_LLAMA_SERVER_BIN`` set adds served
    tags derived from its declaration, the binary's own ``--list-devices``
    report, or observed GPU hardware, in that order; a GPU node with
    ``SKULK_VLLM_BIN`` set adds its ``vllm`` tags. A bare Linux node with none
    advertises an empty set and is therefore not a placement candidate.

    The result is cached for the process lifetime (the same contract as any
    import-time capability): installing a dependency or changing an env var
    requires a restart to notice, and every consumer reads the same snapshot.
    """
    # Function-level import: skulk.facts imports this module for the tag
    # vocabulary, so importing it at module level would be a cycle.
    from skulk.facts import current_backend_derivation

    return current_backend_derivation().backends
