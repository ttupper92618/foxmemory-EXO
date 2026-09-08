"""Shared memory-footprint estimation for placement and the worker OOM guard.

Single source of truth so the master's placement fit-check
(``master/placement_utils.py``) and the worker's local pre-spawn guard
(``worker/main.py``) agree on what a shard will cost to load and serve.
Disagreement would let placement admit a shard the worker then refuses, or let
the worker abort on a load placement believed safe — both produce the
GLM-4.7-Flash failure class (oversized load -> uncaught Metal OOM -> SIGABRT ->
wired GPU memory leaked until reboot, 2026-06-08).

The estimate is intentionally a planning approximation, not a measurement:
weights are known exactly, but KV cache is reserved for an assumed
``KV_CONTEXT_BUDGET_TOKENS`` (models advertise a max context far larger than
typical serving use) and runtime overhead is a multiplicative factor measured
on real loads.
"""

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Final

from skulk.shared.models.llama_server_settings import (
    LLAMA_SERVER_DEFAULT_DRAFT_DEPTH,
    LlamaServerSettings,
)
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.common import NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.runners import ShardAssignments
from skulk.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    RpcDonorShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)

MEMORY_OVERHEAD_FACTOR: float = 1.30
"""Multiplier on a shard's *weight* bytes covering runtime overhead that scales
with weight size but is not the KV cache (estimated separately): activation
workspace, the MLX buffer cache, and the Python/MLX runtime. Measured on
Qwen3.5-9B-MLX-4bit at ~6.4 GB resident for 5.2 GB of weights (1.23x) during
warmup with negligible KV; 1.30 adds margin. Raised from a historical 1.05
after the 16 GB GLM-4.7-Flash incident (2026-06-08)."""

MEMORY_OVERHEAD_FLOOR: Memory = Memory.from_mb(256)
"""Flat per-shard overhead (Python interpreter, MLX runtime, IPC buffers) that
the multiplicative factor under-counts for small shards."""

GPU_WORKING_SET_FRACTION: float = 0.75
"""Fraction of a node's *total* RAM usable as the Metal GPU working set on
Apple Silicon. ``mx.device_info()["max_recommended_working_set_size"]`` measured
11.84 GB on a 16 GB M-series box (0.74); 0.75 tracks it. Both the master
placement check and the worker's local guard derive the ceiling from
``ram_total`` via ``gpu_working_set_ceiling`` (placement cannot gossip the exact
``max_recommended_working_set_size`` under ``extra=forbid``, and keeping the
worker on the same heuristic makes the two checks agree); the worker's advantage
is using *current local* ``ram_available`` rather than the gossiped value."""

GPU_VRAM_WORKING_SET_FRACTION: float = 0.90
"""Fraction of a node's *discrete GPU VRAM* usable for weights + KV on a
GPU-offload node (AMD/NVIDIA running llama.cpp/vLLM/CUDA), as opposed to
``GPU_WORKING_SET_FRACTION`` for Apple unified memory. Discrete VRAM is a
dedicated pool the engine allocates from, not shared with the OS and app
working set, so only driver/runtime/fragmentation overhead is unavailable
(hence a higher fraction than Apple's 0.75). 0.90 matches the de-facto GPU
default (e.g. vLLM ``gpu_memory_utilization``). Used only when a node reports
discrete VRAM telemetry; Apple unified-memory nodes keep the 0.75 path."""

UMA_GPU_OS_HEADROOM: Memory = Memory.from_gb(16)
"""System RAM to leave for the OS on a unified-memory GPU node (AMD APU, e.g.
Strix Halo) when counting host memory the GPU can map via GTT toward the usable
pool. On such a node the GPU addresses the BIOS VRAM carve-out *plus* system RAM
through GTT, so a model larger than the carve-out runs there; this reserve keeps
the OS + worker + download staging from being squeezed out of the shared pool.
16 GB is generous for a headless inference node; the worker's local pre-spawn
guard backstops it with current free memory."""

LLAMA_CPP_MEMORY_OVERHEAD_FACTOR: float = 1.10
"""``MEMORY_OVERHEAD_FACTOR`` for the llama.cpp (GGUF) engine. The 1.30 default
covers MLX's buffer cache + Python/MLX runtime, which the C++ llama.cpp runtime
does not carry; its resident footprint is ~weights + KV + a modest compute
buffer, so 1.10 is the right margin. Over-applying 1.30 to a GGUF wrongly refuses
models that fit (e.g. a 58.5 GB gpt-oss-120B looked like 76 GB)."""

KV_CONTEXT_BUDGET_TOKENS: int = 8192
"""Per-sequence context length reserved for KV cache during the fit check.
Reserving a model's advertised max (e.g. GLM-4.7-Flash: 131072) would over-
refuse by tens of GB. Planning assumption only; exposing it as an operator/UI
knob is tracked follow-up work."""

KV_HEAD_DIM_FALLBACK: int = 128
"""Attention head dimension assumed when a model card omits it (cards do not
persist ``head_dim``). 128 dominates current MLX families (Llama/Qwen/GLM)."""

KV_DTYPE_BYTES: int = 2
"""Bytes per KV-cache element. MLX keeps the KV cache in fp16 even for 4-bit
weights unless quantized-KV is explicitly enabled, which Skulk does not."""

LLAMA_CPP_FULL_SWA_CACHE: Final = False
"""Whether the in-process llama.cpp runner expands sliding-window attention.

This is part of the memory-admission contract, not a performance preference.
The generic GGUF estimate conservatively charges every layer for the full
context using the card's scalar KV geometry. That safely overestimates a
bounded sliding-window cache, but it can underestimate ``swa_full=True`` for
architectures whose sliding layers use wider K/V tensors than their global
layers. Keep the runner tied to this constant so a dependency default cannot
silently invalidate the context window Skulk admitted.
"""


def memory_overhead_factor(model_card: ModelCard) -> float:
    """Engine-appropriate weight-overhead multiplier for a model.

    Returns ``LLAMA_CPP_MEMORY_OVERHEAD_FACTOR`` for a GGUF model (the card
    carries a ``gguf_file``, so it runs on the llama.cpp/C++ engine) and the
    MLX-tuned ``MEMORY_OVERHEAD_FACTOR`` otherwise. GGUF is lighter because the
    C++ runtime carries no MLX buffer cache and no Python/MLX interpreter
    overhead; its resident footprint is essentially weights + KV + a modest
    compute buffer. Applying the 1.30 MLX factor to a GGUF over-refuses models
    that fit (a 58.5 GB gpt-oss-120B looked like ~76 GB).
    """
    if model_card.gguf_file:
        return LLAMA_CPP_MEMORY_OVERHEAD_FACTOR
    return MEMORY_OVERHEAD_FACTOR


def estimate_kv_cache_bytes(
    model_card: ModelCard, n_layers: int, context_tokens: int
) -> Memory:
    """Estimate KV-cache bytes for ``n_layers`` layers at ``context_tokens``.

    A GGUF cache geometry uses its actual attention layers and widths, scaled
    by the requested layer fraction. It does not fold recurrent state into KV.
    Without that geometry the legacy approximation below remains in use.

    The cache holds a key and a value vector per token, per layer, each sized
    ``num_key_value_heads * head_dim``::

        bytes = 2 (K+V) * n_layers * context_tokens
                * num_key_value_heads * head_dim * KV_DTYPE_BYTES

    Returns zero when the card lacks ``num_key_value_heads`` or an argument is
    non-positive — the weight-overhead factor must absorb the slack then.
    ``head_dim`` falls back to ``KV_HEAD_DIM_FALLBACK`` (cards omit it).
    """
    if context_tokens <= 0 or n_layers <= 0:
        return Memory()
    kv_bytes = (
        per_token_kv_bytes(model_card)
        * n_layers
        * context_tokens
        // model_card.n_layers
    )
    return Memory.from_bytes(kv_bytes)


def estimate_shard_footprint(
    model_card: ModelCard,
    shard_fraction: float,
    context_budget: int = KV_CONTEXT_BUDGET_TOKENS,
    *,
    resolved_backend: str | None = None,
    llama_server_settings: LlamaServerSettings | None = None,
) -> Memory:
    """Estimate resident memory for a shard holding ``shard_fraction`` of a model.

    ``weights_share * overhead + kv_share + recurrent_share + overhead_floor``
    where weights and KV both scale by ``shard_fraction``. That single fraction
    works for every sharding because both quantities are linear in it:

    * Pipeline: ``shard_fraction = layers_held / n_layers`` (a node holds a
      contiguous layer range; weights and KV scale with the layer count).
    * Tensor: ``shard_fraction = 1 / world_size`` (a node holds all layers but
      ``1/world_size`` of each weight matrix and of the KV heads).

    ``shard_fraction == 1.0`` gives the whole-model footprint (single node).
    ``resolved_backend`` and ``llama_server_settings`` select the actual engine's
    slot and speculation costs; omitted settings use the shipped server defaults
    for advisory planning. Persisted placements supply their stamped settings.
    """
    if shard_fraction <= 0.0:
        return Memory()
    weights_share = model_card.storage_size * shard_fraction
    full_kv = Memory.from_bytes(
        per_token_kv_bytes(
            model_card,
            resolved_backend=resolved_backend,
            llama_server_settings=llama_server_settings,
        )
        * max(0, context_budget)
    )
    kv_share = full_kv * shard_fraction
    footprint = (
        weights_share * memory_overhead_factor(model_card)
        + kv_share
        + estimate_recurrent_cache_bytes(
            model_card,
            resolved_backend=resolved_backend,
            llama_server_settings=llama_server_settings,
        )
        * shard_fraction
        + MEMORY_OVERHEAD_FLOOR
    )
    # A single-node GGUF vision runner owns the complete projector in addition
    # to the base weights. RPC donors bypass this shard guard; the RPC driver
    # reservation is handled explicitly by placement because llama.cpp chooses
    # the pooled weight split itself.
    if (
        shard_fraction >= 1.0
        and model_card.gguf_file is not None
        and model_card.vision is not None
        and model_card.vision.projector_size is not None
    ):
        footprint += Memory.from_bytes(model_card.vision.projector_size)
    return footprint


def gpu_working_set_ceiling(ram_total: Memory) -> Memory:
    """Metal GPU working-set ceiling derived from total RAM (placement path)."""
    return ram_total * GPU_WORKING_SET_FRACTION


# The served engines that size a fixed context WINDOW at load from
# ``serving_n_ctx``: in-process llama.cpp and llama-server allocate ``n_ctx`` of KV
# up front (OOM-kill on overflow); vLLM passes it as ``vllm serve --max-model-len``
# (the max sequence length the loaded server accepts). All three need the served-context
# clamps and the worker fit-guard sized to the real window. MLX is NOT here: it
# grows KV lazily per request, so it must not be force-clamped (that would regress
# MLX context).
_UPFRONT_WINDOW_ENGINE_PREFIXES: Final = ("llama_cpp", "llama_server", "vllm")

#: Ceiling on the context window stamped (and therefore served and
#: admitted-against) for vLLM-resolved placements, until vLLM-aware admission
#: models engine-start cost. vLLM pre-allocates and CUDA-graph-captures its
#: FULL ``--max-model-len`` at startup: a 262k-context card turned a
#: ~3-minute bring-up into ~90 minutes on an A100-80GB even though the window
#: fit in memory. Applied at the stamp so placement admission and the served
#: window can never disagree (PR #649 review, both reviewers); the runner
#: min()s against the same constant as defense in depth.
VLLM_MAX_MODEL_LEN: Final = 32768


def shard_preallocates_kv_upfront(shard: ShardMetadata) -> bool:
    """Whether the runner for THIS shard commits a fixed context window at load.

    True for the served engines that size their load-time context from
    ``serving_n_ctx`` (llama.cpp, llama-server, vLLM); false for MLX (lazy KV).
    Those engines must have the served-context clamps and the worker fit-guard
    sized to the real window, or a stamped 32k/64k context could exceed what loads.

    Keys off the placement-RESOLVED backend (``shard.resolved_backend``, the engine
    this shard actually runs on) when available, which is the precise per-placement
    signal: a hybrid card that also allows MLX but resolves to a served engine on a
    non-MLX node is correctly treated as committing a window, and the same card
    resolving to ``mlx`` is not -- no MLX regression in the common (resolved) case.

    When the backend is UNRESOLVED (an exact ``CreateInstance`` payload, or a
    placement made before ``NodeResources`` gossiped, where the worker later falls
    back to its own local backend probe) it is CONSERVATIVE: any gguf pin OR any
    served-engine compatible-backend tag makes it window-committing, even for a card
    that ALSO lists MLX. We cannot prove such a shard will run MLX, and on a
    non-Darwin node the local probe won't advertise MLX and may dispatch it to a
    served engine -- so a load-time OOM / spawn failure from an unclamped window is
    the worse outcome than a conservatively floored context in that transient, rare
    window. MLX-only cards (no served tag) still grow KV lazily and are not clamped.
    """
    resolved = shard.resolved_backend
    if resolved is not None:
        return resolved.startswith(_UPFRONT_WINDOW_ENGINE_PREFIXES)
    card = shard.model_card
    if card.gguf_file:
        return True
    return any(
        tag.startswith(_UPFRONT_WINDOW_ENGINE_PREFIXES)
        for tag in card.placement.compatible_backends
    )


def backend_offloads_to_vram(resolved_backend: str | None) -> bool:
    """Whether a resolved backend allocates weights + KV from DISCRETE GPU VRAM.

    A GPU compute tag (``llama_cpp-cuda`` / ``-rocm``, ``llama_server-cuda`` /
    ``-rocm``, ``vllm-cuda`` / ``-rocm``) offloads to VRAM. A ``-cpu`` tag OR a
    bare engine tag (no compute suffix) does not offload to the GPU and allocates
    from system RAM, so it does NOT. ``None`` (unresolved) is treated as not-VRAM:
    we cannot confirm VRAM offload, so the caller stays conservative.
    """
    if resolved_backend is None:
        return False
    return resolved_backend.startswith(
        ("llama_cpp-", "llama_server-", "vllm-")
    ) and not resolved_backend.endswith("-cpu")


def _served_speculative_mode(
    model_card: ModelCard,
    resolved_backend: str | None,
    settings: LlamaServerSettings,
) -> str | None:
    if (
        not settings.speculation_enabled
        or (
            resolved_backend is not None
            and not resolved_backend.startswith("llama_server")
        )
        or model_card.runtime is None
    ):
        return None
    return model_card.runtime.served_spec_type


def estimate_recurrent_cache_bytes(
    model_card: ModelCard,
    *,
    resolved_backend: str | None = None,
    llama_server_settings: LlamaServerSettings | None = None,
) -> Memory:
    """Return fixed FP32 recurrent state for the configured llama.cpp instance.

    Unknown geometry retains the legacy approximation; it must not be described
    as a proven zero-cost recurrent model. Non-llama engines use their own memory
    contract. Unresolved planning uses shipped llama-server settings.
    """
    geometry = model_card.gguf_cache_geometry
    if geometry is None or (
        resolved_backend is not None
        and not resolved_backend.startswith(("llama_cpp", "llama_server"))
    ):
        return Memory()
    settings = llama_server_settings or LlamaServerSettings()
    mode = _served_speculative_mode(model_card, resolved_backend, settings)
    depth = (
        (model_card.runtime.served_spec_n_max or LLAMA_SERVER_DEFAULT_DRAFT_DEPTH)
        if mode in ("draft_mtp", "draft_eagle3", "draft_dflash")
        and model_card.runtime is not None
        else 0
    )
    slots = (
        1
        if resolved_backend is not None and resolved_backend.startswith("llama_cpp")
        else settings.effective_slots(
            speculative_vision=model_card.vision is not None
            and mode not in (None, "none")
        )
    )
    return Memory.from_bytes(
        geometry.recurrent_bytes(parallel_slots=slots, rollback_depth=depth)
    )


def per_token_kv_bytes(
    model_card: ModelCard,
    *,
    resolved_backend: str | None = None,
    llama_server_settings: LlamaServerSettings | None = None,
) -> int:
    """Whole-model KV-cache bytes consumed by ONE token of context.

    Uses artifact attention geometry when present, including embedded MTP only
    for a speculative served instance. Otherwise covers all layers at fp16
    (``KV_DTYPE_BYTES``); a node holding
    ``shard_fraction`` of the model pays ``per_token_kv_bytes * shard_fraction``
    per token. Returns 0 when the card lacks ``num_key_value_heads`` —
    callers must treat 0 as "KV cost unknown, cannot enforce a memory ceiling".
    """
    geometry = model_card.gguf_cache_geometry
    if geometry is not None and (
        resolved_backend is None
        or resolved_backend.startswith(("llama_cpp", "llama_server"))
    ):
        mode = _served_speculative_mode(
            model_card, resolved_backend, llama_server_settings or LlamaServerSettings()
        )
        return geometry.attention_bytes_per_token(embedded_mtp=mode == "draft_mtp")
    kv_heads = model_card.num_key_value_heads
    if kv_heads is None or model_card.n_layers <= 0:
        return 0
    return 2 * model_card.n_layers * kv_heads * KV_HEAD_DIM_FALLBACK * KV_DTYPE_BYTES


def shard_fraction_of_model(shard: ShardMetadata) -> float | None:
    """Fraction of the whole model's weights AND KV held by one shard.

    Mirrors the placement-side accounting in ``estimate_shard_footprint``:
    pipeline shards hold a contiguous layer range, tensor shards hold all
    layers but ``1/world_size`` of each matrix and of the KV heads. CFG shards
    (image models) have no text-generation KV admission story, so ``None``.
    """
    match shard:
        case TensorShardMetadata():
            return 1.0 / shard.world_size if shard.world_size > 0 else None
        case PipelineShardMetadata():
            if shard.n_layers <= 0:
                return None
            return (shard.end_layer - shard.start_layer) / shard.n_layers
        case CfgShardMetadata():
            return None
        case RpcDonorShardMetadata():
            # An RPC memory donor (#328) holds no Skulk-assigned share:
            # llama.cpp splits the pooled model across the devices itself, so
            # per-node fraction accounting does not apply. None makes the
            # context-ceiling stamp fall back to the card's advertised limit
            # for RPC instances.
            return None


def instance_context_token_limit(
    shard_assignments: ShardAssignments,
    node_ram_totals: Mapping[NodeId, Memory],
    node_vram: Mapping[NodeId, Memory] | None = None,
    unified_memory_gpu_nodes: AbstractSet[NodeId] | None = None,
    fixed_memory_by_node: Mapping[NodeId, Memory] | None = None,
) -> int | None:
    """Deterministic context-token ceiling for one placed instance.

    For each hosting node: tokens that fit in the GPU working set after the
    node's weight share and overhead, at the shard's per-token KV cost. The
    instance ceiling is the minimum across nodes (the smallest rank OOMs
    first and takes the whole ring down), then min'd with the card's
    advertised ``context_length`` (0 means unadvertised).

    Determinism is load-bearing: on multi-rank instances every rank admits or
    rejects a request independently, and divergent verdicts deadlock the
    collectives. All inputs here are static (``ram_total`` and a node's discrete
    VRAM total never change for a node), and placement only happens after the
    master has indexed memory for every hosting node, so every worker computes
    the identical value. Time-varying ``ram_available`` must NOT be used here.

    ``node_vram`` (a node's usable GPU memory, see ``usable_vram_by_node``) is
    the working-set ceiling for a GPU-offload node, mirroring the memory-fit
    admission: without it a model whose weights exceed ``0.75 * ram_total`` but
    fit in GPU memory would get a negative KV budget and a 0-token ceiling
    (every request rejected), even though placement admitted it against that
    pool. ``unified_memory_gpu_nodes`` distinguishes APUs whose usable pool
    includes host RAM from true discrete VRAM. Fixed-window served engines may
    lift above the conservative context floor only on the latter: on an APU,
    llama.cpp's load-time GPU allocation also consumes host pages and a
    combined-pool steady-state fit does not bound that transient allocation.

    Returns ``None`` when no ceiling is enforceable (unknown KV cost or
    missing node memory, falling back to the card limit when that exists).
    """
    node_vram = node_vram or {}
    unified_memory_gpu_nodes = unified_memory_gpu_nodes or frozenset()
    fixed_memory_by_node = fixed_memory_by_node or {}
    model_card: ModelCard | None = None
    memory_limit: int | None = None
    for shard in shard_assignments.runner_to_shard.values():
        model_card = shard.model_card
        break
    if model_card is None:
        return None
    # Whether this placement is served by a window-committing engine (llama.cpp /
    # llama-server / vLLM, per shard_preallocates_kv_upfront), keyed off the resolved
    # backend per shard. Gates the load-time OOM/spawn-failure clamps below. All
    # shards of an instance share the engine, so any window-committing shard makes
    # the instance window-committing.
    served_preallocates = any(
        shard_preallocates_kv_upfront(shard)
        for shard in shard_assignments.runner_to_shard.values()
    )

    if per_token_kv_bytes(model_card) > 0:
        node_to_runner = shard_assignments.node_to_runner
        for node_id, runner_id in node_to_runner.items():
            shard = shard_assignments.runner_to_shard[runner_id]
            whole_model_token_bytes = per_token_kv_bytes(
                model_card,
                resolved_backend=shard.resolved_backend,
                llama_server_settings=shard.llama_server_settings,
            )
            fraction = shard_fraction_of_model(shard)
            ram_total = node_ram_totals.get(node_id)
            if (
                fraction is None
                or fraction <= 0.0
                or ram_total is None
                or whole_model_token_bytes <= 0
            ):
                memory_limit = None
                break
            working_set = node_vram.get(node_id) or gpu_working_set_ceiling(ram_total)
            kv_budget = (
                working_set
                - model_card.storage_size
                * fraction
                * memory_overhead_factor(model_card)
                - MEMORY_OVERHEAD_FLOOR
                - estimate_recurrent_cache_bytes(
                    model_card,
                    resolved_backend=shard.resolved_backend,
                    llama_server_settings=shard.llama_server_settings,
                )
                * fraction
                - fixed_memory_by_node.get(node_id, Memory())
            )
            node_tokens = max(
                0, int(kv_budget.in_bytes / (whole_model_token_bytes * fraction))
            )
            memory_limit = (
                node_tokens if memory_limit is None else min(memory_limit, node_tokens)
            )

    # A window-committing served engine commits this context at load (serving_n_ctx),
    # so it must fit the memory actually available then. On a discrete-VRAM node the
    # fit above is derived from VRAM -- the same pool placement admits against -- so
    # the lift is safe (and validated on GPU hardware). On a node WITHOUT discrete
    # VRAM the fit is derived from static ``ram_total`` (for cross-rank determinism),
    # but the load-time window competes with live ``ram_available`` (which placement
    # admits against and which can be far lower under memory pressure), so a
    # ram_total-sized window could OOM the node on load. Keep such placements at the
    # budget floor; the memory-fit lift applies to discrete-VRAM (GPU) nodes.
    # ``node_vram`` membership is static per node, so this stays deterministic.
    if served_preallocates and memory_limit is not None:
        # Keep the lift only where the served window lands in DISCRETE VRAM (the
        # pool placement admitted against): every hosting shard must both resolve
        # to a GPU-offload backend (``-cuda`` / ``-rocm``; a ``-cpu`` / bare tag
        # does not offload to the GPU and commits the window in SYSTEM RAM, which
        # competes with live ram_available) AND run on a node reporting discrete
        # VRAM. A unified-memory APU remains in ``node_vram`` for placement because
        # its GPU can use the combined carve-out + GTT pool, but it is explicitly
        # excluded here: llama.cpp's load-time amdgpu allocations consume host
        # pages too, so a steady-state combined-pool fit can still globally OOM the
        # node while committing a large fixed KV window. Anything else -- UMA,
        # CPU-resolved on a GPU node, a non-VRAM node, or an unresolved backend --
        # clamps to the floor to avoid a load-time OOM.
        lift_in_vram = all(
            node_vram.get(node_id) is not None
            and node_id not in unified_memory_gpu_nodes
            and backend_offloads_to_vram(
                shard_assignments.runner_to_shard[runner_id].resolved_backend
            )
            for node_id, runner_id in shard_assignments.node_to_runner.items()
        )
        if not lift_in_vram:
            memory_limit = min(memory_limit, KV_CONTEXT_BUDGET_TOKENS)

    card_limit = model_card.context_length if model_card.context_length > 0 else None
    if memory_limit is None:
        limit = card_limit
    elif card_limit is None:
        limit = memory_limit
    else:
        limit = min(memory_limit, card_limit)

    # vLLM startup-cost cap (see VLLM_MAX_MODEL_LEN): applies when the
    # placement resolves to the vllm engine (every shard's resolved_backend),
    # falling back to the card declaring ONLY vllm backends when resolution
    # has not happened yet. Deterministic: resolved backends and card
    # backends are both static placement inputs.
    resolved = [
        shard.resolved_backend
        for shard in shard_assignments.runner_to_shard.values()
    ]
    if any(r is not None for r in resolved):
        placement_is_vllm = all(
            r is not None and r.startswith("vllm") for r in resolved
        )
    else:
        declared = model_card.placement.compatible_backends
        placement_is_vllm = bool(declared) and all(
            tag.startswith("vllm") for tag in declared
        )
    if placement_is_vllm:
        limit = (
            VLLM_MAX_MODEL_LEN if limit is None else min(limit, VLLM_MAX_MODEL_LEN)
        )

    # This ceiling is now the value the runner actually serves: the window-committing
    # served engines (llama_cpp / llama_server / vllm, the runners that call
    # ``serving_n_ctx``) commit their load-time context window at
    # ``serving_n_ctx(context_token_limit)``, i.e. this memory-fit window, not a
    # fixed 8192. The ceiling and the runner
    # window moved off the shared constant together, as the previous fixed-clamp
    # comment anticipated: placement's per-node fit is derived from the same working
    # set as this ceiling, so a node admitted at the KV_CONTEXT_BUDGET_TOKENS floor
    # can hold the KV for this larger window (see filter_cycles_by_memory and
    # serving_n_ctx). The old fixed 8192 clamp made served models unusable for
    # real-context work (a codebase does not fit in 8192 tokens). On an admitted
    # node the memory fit is >= the floor, so the window is too -- EXCEPT when the
    # model's own advertised max context is smaller, in which case it serves that
    # smaller max (a 4k-context model serves 4k, not 8k).
    #
    # SAFETY (served engine, load-time OOM): a window-committing served engine
    # (llama.cpp / llama-server / vLLM, per shard_preallocates_kv_upfront) commits a
    # fixed context window at load, so it MUST be a real memory-fit value. When the
    # fit is uncomputable (the card has no ``num_key_value_heads`` so
    # ``per_token_kv_bytes`` is 0, node memory is missing, or an RPC-donor / CFG
    # shard has no ``shard_fraction``), ``memory_limit`` is None and the branch above
    # would fall back to the card's advertised max -- which for a 128k model would
    # commit a fictitious window and OOM / fail to load. In that uncomputable case
    # only, clamp the ceiling back to the budget floor (the old safety the fixed
    # clamp used to provide). This does not affect MLX, which grows its KV cache
    # lazily per request (no up-front window) and keeps the full memory/card fit.
    #
    # KV dtype (#584): the fit assumes fp16 KV (KV_DTYPE_BYTES); enabling KV-cache
    # quantization must feed the estimate the quantized bytes-per-token or this
    # ceiling would be sized against the wrong footprint.
    if served_preallocates and memory_limit is None:
        limit = (
            KV_CONTEXT_BUDGET_TOKENS
            if limit is None
            else min(limit, KV_CONTEXT_BUDGET_TOKENS)
        )
    return limit
