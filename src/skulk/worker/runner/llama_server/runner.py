# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Served-backend text-generation runner: launches and proxies ``llama-server``.

Unlike the in-process ``llama_cpp`` runner (which loads the GGUF via
``llama-cpp-python`` and calls ``create_chat_completion``), this runner launches
an external ``llama-server`` subprocess pointed at the staged GGUF and proxies its
OpenAI-compatible HTTP API. That is the only way to reach llama.cpp's native
multi-token-prediction speculative decoding (``--spec-type draft-mtp``): the MTP
orchestration lives in the server application (``tools/server``), not in the
``libllama`` C API or the Python binding, so it cannot be driven in-process.

This is the first *served* engine. Its shape (managed inference server + OpenAI
proxy) is deliberately generic so vLLM and other OpenAI-compatible servers can
become additional served backends without new runner architecture.

Single-node or the driver of a homogeneous llama.cpp RPC placement (no Skulk
ring / ConnectToGroup / warmup). Linux-oriented: the subprocess is reaped on
parent death via ``PR_SET_PDEATHSIG`` so a runner crash never orphans a server
holding GPU memory. Per-request cancellation aborts the proxied HTTP connection
(which stops server-side generation); ``SIGTERM`` is for instance teardown of the
whole server, not a single request. The server emits structured
``reasoning_content`` and ``tool_calls`` itself, so the in-process text parsers
(harmony / think / tool) are not used here.
"""

import contextlib
import ctypes
import json
import os
import random
import signal
import socket
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, cast

import httpx

from skulk.api.types import GenerationStats
from skulk.shared.backends import LLAMA_SERVER_BIN_ENV
from skulk.shared.constants import MAX_OUTPUT_TOKENS
from skulk.shared.models.model_cards import ModelCard, OutputParserType
from skulk.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
from skulk.shared.types.common import CommandId, ModelId
from skulk.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.tasks import (
    LoadModel,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from skulk.shared.types.worker.instances import BoundInstance, LlamaRpcInstance
from skulk.shared.types.worker.runners import (
    RunnerIdle,
    RunnerLoading,
    RunnerReady,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
)
from skulk.utils.channels import MpReceiver, MpSender
from skulk.worker.runner.bootstrap import logger
from skulk.worker.runner.diagnostics import record_runner_phase, runner_phase
from skulk.worker.runner.generation_stats import (
    StreamStatsClock,
    blocking_call_stats,
    served_stream_stats,
    stats_from_llama_server_timings,
    subprocess_peak_memory,
)
from skulk.worker.runner.llama_cpp.runner import (
    generation_kwargs,
    map_finish_reason,
    messages_for_llama,
    select_gguf_file,
    serving_n_ctx,
    tool_calls_from_message,
    wants_logprobs,
)
from skulk.worker.runner.llama_server.channel_text_parser import (
    GemmaChannelTextParser,
)
from skulk.worker.runner.llm_inference.reasoning_controls import (
    muse_glimmer_strength_kwargs,
)
from skulk.worker.runner.llm_inference.scaffolding_scrub import (
    StreamingScaffoldingScrub,
)
from skulk.worker.runner.served_concurrency import ServedConcurrentDispatch


def _effective_server_parallel(card: Any) -> int:
    """Return the safe served slot count for this card generation.

    llama.cpp can serve multimodal requests with native MTP, but concurrent
    multimodal qualification is intentionally outside the current evidence
    envelope. Keep that exact combination serial while preserving configured
    batching for text-only and non-speculative vision models.
    """

    runtime = getattr(card, "runtime", None)
    served_spec_type = getattr(runtime, "served_spec_type", None) if runtime else None
    if (
        getattr(card, "vision", None) is not None
        and served_spec_type not in (None, "none")
        and not _force_no_spec()
    ):
        return 1
    return _llama_server_parallel()

# Card ``served_spec_type`` value -> the ``llama-server --spec-type`` token.
# ``draft_mtp`` usually uses the model's own built-in MTP heads; a separate
# draft is optional (Gemma 4 supplies its assistant as one).
_SPEC_TYPE_FLAG: Final[dict[str, str]] = {
    "draft_mtp": "draft-mtp",
    "draft_eagle3": "draft-eagle3",
    "draft_simple": "draft-simple",
    "draft_dflash": "draft-dflash",
    "ngram": "ngram-cache",
}

# Served spec modes that REQUIRE a separate ``--model-draft`` GGUF (DFlash's
# block-parallel drafter always ships separately). For ``draft_mtp`` a draft is
# optional (Qwen/DeepSeek/GLM bake the heads into the base GGUF; Gemma 4
# instead supplies its assistant as one), and ``ngram`` needs no model at all.
_DRAFT_MODEL_REQUIRED: Final[frozenset[str]] = frozenset(
    {"draft_simple", "draft_eagle3", "draft_dflash"}
)


def _force_no_spec() -> bool:
    """True if this node is configured to serve without speculative decoding.

    ``SKULK_LLAMA_SERVER_FORCE_NO_SPEC`` makes the served runner ignore a card's
    ``served_spec_type`` and launch ``llama-server`` without any ``--spec-type``
    flags, so the same GGUF serves as a plain-decode baseline. Intended for an
    MTP on-vs-off throughput comparison and for debugging a misbehaving spec
    pairing; unset in normal operation.
    """
    return os.environ.get("SKULK_LLAMA_SERVER_FORCE_NO_SPEC", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


_LLAMA_SERVER_PARALLEL_ENV: Final = "SKULK_LLAMA_SERVER_PARALLEL"
# The served engine exists to batch concurrent requests; shipping it serial made
# fresh installs behave materially worse than the fleet used to qualify them.
# Sixteen is the exercised fleet setting. A unified KV buffer keeps every slot's
# advertised context window truthful without allocating N private caches; users
# dominated by near-window prompts can still opt back to serial explicitly.
_DEFAULT_LLAMA_SERVER_PARALLEL: Final = 16
# An omitted OpenAI ``max_tokens`` value otherwise lets llama-server consume the
# remainder of the shared KV pool. Bound it to the same normal-generation width
# as Skulk's MLX path so aggregate admission has a finite reservation and one
# unconstrained request cannot silently monopolize every concurrent slot.
_LLAMA_SERVER_MAX_OUTPUT_TOKENS: Final = MAX_OUTPUT_TOKENS


def _llama_server_parallel() -> int:
    """Return the operator-declared concurrent-generation slot count.

    The declared value is honored EXACTLY (#689). It used to be capped to
    ``floor(n_ctx / 8192)`` because llama.cpp sliced the one
    ``-c`` window into fixed per-slot shares, so a high count silently shrank
    every request's real window to ``n_ctx / N`` while Skulk's API kept admitting
    against the full one. That slicing is not a llama.cpp law, it is a
    consequence of how the server is launched: ``llama_context`` sets
    ``n_ctx_seq = n_ctx`` under a unified KV buffer and only falls back to
    ``n_ctx / n_seq_max`` without one (``src/llama-context.cpp``). The runner now
    passes ``--kv-unified`` whenever it asks for more than one slot, so each slot
    sees the whole window, the stamped ``context_token_limit`` stays the truth,
    and there is nothing left for a cap to protect.

    Under a unified buffer the slots draw from ONE pool of ``n_ctx`` tokens
    rather than N private slices. The runner therefore counts each exact
    rendered input and admits prompt-plus-output reservations in FIFO order
    only while their aggregate fits the pool. The shipped 16-slot default is a
    real concurrency ceiling without making aggregate exhaustion possible;
    setting the override to ``1`` remains an explicit serial-isolation option.

    Returns:
        The declared slot count, or ``16`` when unset. An unparseable or
        below-one value also yields ``16``, but is warned about first: an
        operator who declared something meant it, so a rejected declaration is
        never applied silently. An absent declaration is not a mistake and
        stays quiet.
    """
    raw = os.environ.get(_LLAMA_SERVER_PARALLEL_ENV, "").strip()
    if not raw:
        return _DEFAULT_LLAMA_SERVER_PARALLEL
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            f"{_LLAMA_SERVER_PARALLEL_ENV}={raw!r} is not an integer; "
            f"using {_DEFAULT_LLAMA_SERVER_PARALLEL}"
        )
        return _DEFAULT_LLAMA_SERVER_PARALLEL
    if value < 1:
        logger.warning(
            f"{_LLAMA_SERVER_PARALLEL_ENV}={value} is below 1; "
            f"using {_DEFAULT_LLAMA_SERVER_PARALLEL}"
        )
        return _DEFAULT_LLAMA_SERVER_PARALLEL
    return value


def _request_context_reservation(
    input_tokens: int,
    requested_max_output_tokens: int | None,
    context_pool_tokens: int,
) -> tuple[int, int]:
    """Return the bounded output limit and shared-KV reservation for a request.

    ``llama-server``'s unified KV cache keeps each slot's per-request window
    truthful, but all slots consume cells from one fixed pool. The runner counts
    the exact rendered input through llama-server's token-count endpoint and
    reserves ``input + max_output`` before starting generation. This helper
    normalizes that reservation:

    * an omitted output limit becomes the shipped 4096-token bound;
    * an over-window request reserves the whole pool and then runs alone so the
      server can return its normal context error;
    * a valid bounded request reserves no more than the pool.

    Args:
        input_tokens: Exact prompt token count reported by llama-server.
        requested_max_output_tokens: Caller-supplied output bound, when present.
        context_pool_tokens: Total unified KV cells available to all slots.

    Returns:
        ``(effective_max_output_tokens, reservation_tokens)``.
    """
    if input_tokens < 0:
        raise ValueError("input token count cannot be negative")
    if context_pool_tokens < 1:
        raise ValueError("context pool must contain at least one token")
    available_output = max(1, context_pool_tokens - input_tokens)
    effective_max_output = (
        requested_max_output_tokens
        if requested_max_output_tokens is not None
        else min(_LLAMA_SERVER_MAX_OUTPUT_TOKENS, available_output)
    )
    reservation = min(
        context_pool_tokens,
        input_tokens + max(1, effective_max_output),
    )
    return effective_max_output, reservation


def _slot_server_args(max_concurrency: int) -> list[str]:
    """llama-server flags that govern slot count and how slots share the window.

    Returns ``--parallel N``, plus ``--kv-unified`` above one slot (#689).

    The unified flag is what makes concurrency honest. llama.cpp sets a slot's
    context to the whole ``-c`` window under a unified KV buffer and to
    ``n_ctx / n_seq_max`` without one (``src/llama-context.cpp``: ``n_ctx_seq``),
    and the server reads exactly that for each slot
    (``tools/server/server-context.cpp``: ``llama_n_ctx_seq``). So without it,
    asking for N slots silently reduces every request's real window to a
    fraction of the ``context_token_limit`` that placement stamped and the API
    admits against, and a long prompt truncates mid-generation instead of being
    refused. It costs no extra memory: the cache allocates
    ``n_ctx_seq * n_stream`` cells and ``n_stream`` collapses to 1 when unified,
    so the same buffer is shared rather than sliced.

    Passed only above one slot, so an explicit serial override keeps the
    validated single-slot command line used by draft-mtp speculation and the
    RPC driver.

    Args:
        max_concurrency: Effective slot count from the shipped default or an
            operator override.

    Returns:
        The flags to splice into the llama-server command line.
    """
    args = ["--parallel", str(max_concurrency)]
    if max_concurrency > 1:
        args.append("--kv-unified")
    return args


def _draft_model_args(
    runtime: Any,
    spec_type: str,
    *,
    base_model_id: ModelId | None = None,
    source_revision: str | None = None,
) -> list[str] | None:
    """Resolve the ``--model-draft`` args for a served spec mode.

    When the card declares a draft GGUF (``served_spec_draft_repo`` +
    ``served_spec_draft_file``), resolve its on-disk path and return
    ``["--model-draft", path]`` (Gemma 4 draft_mtp, draft_simple, draft_eagle3).

    Returns ``None`` when the card DECLARES a draft but it cannot be provided
    (missing ``served_spec_draft_file``, or the draft GGUF is not on disk): the
    draft is best-effort at download (a failed cross-repo co-fetch is swallowed,
    #574), so a declared-but-absent draft must degrade to serving WITHOUT
    speculation rather than crashing the runner. The caller drops ``--spec-type``
    entirely in that case, since a draft-backed mode without its draft is not a
    valid llama-server invocation.

    Returns ``[]`` when no draft applies (``draft_mtp`` with built-in heads,
    ``ngram``); the caller still passes ``--spec-type`` for those. Modes in
    ``_DRAFT_MODEL_REQUIRED`` with no draft declared at all are a card
    misconfiguration and still raise loudly. Pure except for the on-disk path
    resolution, so the validation branches are unit-testable. A draft sharing
    the base repository inherits the base card's immutable source revision;
    separate-repository drafts use their card-declared immutable revision.

    Args:
        runtime: Resolved runtime capability section from the base model card.
        spec_type: Served speculative-decoding mode.
        base_model_id: Base repository identifier, used to identify a bundled
            same-repository draft.
        source_revision: Immutable base repository revision, when pinned.

    Returns:
        ``--model-draft`` arguments, an empty list for draft-free modes, or
        ``None`` when a declared best-effort draft is unavailable.
    """
    draft_repo = getattr(runtime, "served_spec_draft_repo", None) if runtime else None
    draft_file = getattr(runtime, "served_spec_draft_file", None) if runtime else None
    # ngram-cache speculation uses no `--model-draft`, so a draft repo on an
    # ngram card is spurious: ignore it and keep `--spec-type ngram-cache`
    # rather than dropping speculation for a draft that mode never consults.
    if draft_repo and spec_type != "ngram":
        if not draft_file:
            logger.warning(
                f"Card declares served_spec_draft_repo {draft_repo!r} without "
                "served_spec_draft_file; serving without speculation."
            )
            return None
        from skulk.download.download_utils import build_model_path

        draft_revision = (
            source_revision
            if base_model_id is not None and draft_repo == str(base_model_id)
            else getattr(runtime, "served_spec_draft_revision", None)
        )
        try:
            draft_dir = build_model_path(ModelId(draft_repo), draft_revision)
        except FileNotFoundError:
            logger.warning(
                f"Served {spec_type} draft repo {draft_repo!r} is not on disk; "
                "serving without speculation (the draft is a best-effort "
                "companion, so its absence must not fail the model)."
            )
            return None
        draft_path = (draft_dir / draft_file).resolve()
        if not draft_path.is_file() or not draft_path.is_relative_to(
            draft_dir.resolve()
        ):
            logger.warning(
                f"Served draft GGUF {draft_file!r} not found under {draft_dir}; "
                "serving without speculation."
            )
            return None
        return ["--model-draft", str(draft_path)]
    if spec_type in _DRAFT_MODEL_REQUIRED:
        raise RuntimeError(
            f"served_spec_type={spec_type!r} requires a draft model; set "
            "served_spec_draft_repo + served_spec_draft_file on the card"
        )
    return []


def _model_declares_reasoning(card: Any) -> bool:
    """Whether the card advertises a reasoning/thinking capability.

    Drives ``--reasoning-format``: a reasoning model keeps llama-server's default
    (``auto``) so thoughts land in ``message.reasoning_content`` (which the runner
    flags as ``is_thinking``); a non-reasoning model is served with
    ``--reasoning-format none`` so all output stays in ``message.content``.
    Without that, llama-server's ``auto`` can extract a plain model's prose into
    ``reasoning_content`` (observed with Gemma 4 served via ``--jinja``), leaving
    ``message.content`` empty for the client. Detection mirrors the capability
    spine: an explicit ``reasoning`` card section or a ``thinking`` capability.
    """
    if getattr(card, "reasoning", None) is not None:
        return True
    return "thinking" in (getattr(card, "capabilities", None) or [])


def reasoning_request_overrides(
    task_params: Any, card: ModelCard | None = None
) -> dict[str, Any]:
    """Map Skulk's thinking controls onto llama-server request fields.

    ``generation_kwargs`` carries sampling params but NOT thinking control, so
    without this the served runner never tells llama-server to suppress reasoning.
    A reasoning model then thinks on every request regardless of
    ``enable_thinking=False``, and on a bounded ``max_tokens`` it can spend the
    whole budget thinking and return EMPTY content (#428/#420).

    llama-server exposes two levers, forwarded here:

    - ``chat_template_kwargs`` -> the model's jinja chat template. ``enable_thinking``
      is the canonical Qwen3 / Gemma toggle; a template that doesn't understand it
      simply ignores it, so forwarding is safe across families.
    - ``reasoning_effort`` -> OpenAI-style effort for harmony models (gpt-oss).
      ``"none"`` is not a valid server value; disabling is expressed via
      ``enable_thinking=False`` instead, so it is dropped here.
    - Muse Glimmer (resolved from ``card``) reads neither: its template steers
      always-on reasoning with a ``reasoning_strength`` template kwarg, so the
      effort is translated onto that and the other two levers are omitted.
    """
    overrides: dict[str, Any] = {}
    effort = getattr(task_params, "reasoning_effort", None)
    strength_kwargs = muse_glimmer_strength_kwargs(card, effort)
    if strength_kwargs:
        overrides["chat_template_kwargs"] = strength_kwargs
        return overrides
    enable_thinking = getattr(task_params, "enable_thinking", None)
    if enable_thinking is not None:
        overrides["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if effort is not None and effort != "none":
        overrides["reasoning_effort"] = effort
    return overrides



# How long to wait for the server to finish loading the model and report healthy.
# A large GGUF on a GPU node can take a while to map + warm up.
_HEALTH_DEADLINE_S: Final = 600.0
# Between tasks the runner wakes at this cadence to verify its llama-server
# subprocess is still alive (dead server between requests must crash the
# runner, not wedge it Ready).
_LIVENESS_POLL_S: Final = 2.0


class _StreamDelta(NamedTuple):
    """One parsed SSE delta from the proxied ``/v1/chat/completions`` stream."""

    reasoning: str
    content: str
    finish: Literal["stop", "length", "content_filter"] | None
    done: bool  # the terminal ``data: [DONE]`` sentinel
    # llama-server's native phase measurements, present on the final chunk
    # when the request asked for them (timings_per_token); source of the
    # engine-exact GenerationStats (#532).
    timings: dict[str, object] | None = None


class _ContextWaiter(NamedTuple):
    """One FIFO admission request against llama-server's shared KV pool."""

    task_id: TaskId
    reservation_tokens: int


def _gpu_layers_for_backend(resolved_backend: str | None) -> str:
    """The ``-ngl`` (n-gpu-layers) value to pass llama-server for a backend tag.

    Mirrors the master's VRAM admission exactly (placement_utils
    ``_has_gpu_offload_backend``): offload every layer (``"99"``) only for a
    recognized GPU compute tag (``llama_server-<gpu>``). A ``-cpu`` tag OR a bare
    ``llama_server`` tag was admitted against system RAM, not VRAM, so use
    ``"0"`` rather than grabbing a GPU that was not budgeted (or may not exist). A
    missing resolution (a manual / fallback launch off the placement path)
    defaults to full GPU offload, the common served case.
    """
    if resolved_backend is None:
        return "99"
    if resolved_backend.startswith("llama_server-") and not resolved_backend.endswith(
        "-cpu"
    ):
        return "99"
    return "0"


def _projector_server_args(
    projector_path: Path | None,
    resolved_backend: str | None,
) -> list[str]:
    """Return served multimodal flags for one authenticated projector.

    CPU-resolved service keeps the projector in host memory explicitly;
    accelerator backends retain llama.cpp's default projector offload.
    """

    if projector_path is None:
        return []
    args = ["--mmproj", str(projector_path)]
    if _gpu_layers_for_backend(resolved_backend) == "0":
        args.append("--no-mmproj-offload")
    return args


def _parse_sse_line(line: str) -> _StreamDelta | None:
    """Parse one SSE line into a ``_StreamDelta``, or ``None`` to skip it.

    Handles the OpenAI streaming shape llama-server emits: ``data: {json}`` lines
    whose first choice carries a ``delta`` (``content`` and/or ``reasoning_content``)
    and an optional ``finish_reason``, plus the terminal ``data: [DONE]``. Returns
    ``None`` for non-``data:`` lines, ``[DONE]`` is reported via ``done=True``, and
    malformed JSON or a choice-less payload is skipped (``None``) so a stray line
    never breaks the stream. Pure (no I/O) so the parse is unit-testable.
    """
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if data == "[DONE]":
        return _StreamDelta("", "", None, done=True)
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(chunk, dict):
        return None
    choices = chunk.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], dict)
    ):
        return None
    choice = choices[0]
    raw_delta = choice.get("delta")
    # Non-dict shapes are skipped (the docstring's malformed-payload promise),
    # not raised: one stray line must never break the whole stream.
    delta = raw_delta if isinstance(raw_delta, dict) else {}
    raw_timings = chunk.get("timings")
    return _StreamDelta(
        reasoning=delta.get("reasoning_content") or "",
        content=delta.get("content") or "",
        finish=map_finish_reason(choice.get("finish_reason")),
        done=False,
        timings=raw_timings if isinstance(raw_timings, dict) else None,
    )


def _set_pdeathsig() -> None:
    """Ask the kernel to SIGKILL this child when its parent (the runner) dies.

    Runs in the forked child before ``exec`` (``preexec_fn``). Linux-only; a
    best-effort guard so a runner-process crash never leaves an orphaned
    ``llama-server`` holding GPU memory. Any failure is swallowed (the explicit
    teardown path still applies on graceful shutdown).
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGKILL, 0, 0, 0)
    except Exception:  # noqa: BLE001 - best-effort; non-Linux or no libc
        pass


class Runner(ServedConcurrentDispatch):
    """Served-backend runner that proxies an external ``llama-server``.

    Lifecycle mirrors the in-process llama.cpp runner: it skips the ring
    (``ConnectToGroup`` / ``StartWarmup``), spawns the server on ``LoadModel``,
    and serves ``TextGeneration`` by streaming the server's SSE output back as
    ``ChunkGenerated`` events.
    """

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
        context_token_limit: int | None = None,
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.bound_instance = bound_instance
        self.context_token_limit = context_token_limit
        self.instance, self.runner_id, self.shard_metadata = (
            bound_instance.instance,
            bound_instance.bound_runner_id,
            bound_instance.bound_shard,
        )
        # Multi-node is legal only as the RPC driver of a LlamaRpcInstance
        # (#328): rank 0 runs llama-server with --rpc against the stamped donor
        # endpoints, and llama.cpp does the cross-device split itself. Any
        # other multi-node shape reaching this runner is a placement bug.
        self._rpc_donor_endpoints: dict[str, str] = {}
        if isinstance(self.instance, LlamaRpcInstance):
            if self.shard_metadata.device_rank != 0:
                raise RuntimeError(
                    "llama-server runner on a LlamaRpcInstance must be the "
                    f"driver (rank 0), got rank {self.shard_metadata.device_rank}"
                )
            self._rpc_donor_endpoints = {
                str(node_id): endpoint
                for node_id, endpoint in self.instance.donor_endpoints.items()
            }
        elif self.shard_metadata.world_size != 1:
            raise RuntimeError(
                "llama-server runner requires single-node placement, got "
                f"world_size={self.shard_metadata.world_size}"
            )
        self.setup_start_time = time.time()
        self.cancelled_tasks: set[TaskId] = set()
        self.seen: set[TaskId] = set()
        self.server_proc: subprocess.Popen[bytes] | None = None
        self.server_log: Any = None
        self.server_log_path: Path | None = None
        self.base_url: str | None = None
        # Set at load: the card declares a channel output parser (Gemma 4), so the
        # served runner reparses ``<|channel>`` markers out of the content stream
        # itself (llama-server can't), splitting reasoning from the answer.
        self._uses_channel_parser: bool = False
        self.current_status: RunnerStatus = RunnerIdle()
        self._serving_context_tokens = serving_n_ctx(
            self.context_token_limit, logits_all=False
        )
        self._context_budget_condition = threading.Condition()
        self._reserved_context_tokens = 0
        self._context_active_requests = 0
        self._context_waiters: deque[_ContextWaiter] = deque()
        # The declared slot count is honored exactly; --kv-unified (added in
        # _spawn_server) keeps every slot's context at the full stamped window,
        # while the weighted gate below prevents their aggregate reservations
        # from exhausting the one shared pool.
        effective_parallel = _effective_server_parallel(self.shard_metadata.model_card)
        self._init_concurrent_dispatch(effective_parallel, "llama-gen")
        if (
            effective_parallel == 1
            and self.shard_metadata.model_card.vision is not None
            and self.shard_metadata.model_card.runtime is not None
            and self.shard_metadata.model_card.runtime.served_spec_type
            not in (None, "none")
        ):
            logger.warning(
                "served vision plus speculative decoding is running serially "
                "because concurrent multimodal serving is not yet qualified"
            )
        if self._max_concurrency > 1:
            # The shipped default uses the shared buffer, so normal startup
            # records the trade without presenting the supported default as an
            # operator error. Explicit non-default concurrency remains visible
            # at warning level because it changes the exercised admission width.
            raw_parallel = os.environ.get(_LLAMA_SERVER_PARALLEL_ENV, "").strip()
            log_slot_contract = (
                logger.warning
                if raw_parallel and raw_parallel != str(_DEFAULT_LLAMA_SERVER_PARALLEL)
                else logger.info
            )
            log_slot_contract(
                f"serving up to {self._max_concurrency} concurrent generations "
                f"from one shared {self._serving_context_tokens}-token KV pool "
                f"(set {_LLAMA_SERVER_PARALLEL_ENV}=1 for serial service); "
                "aggregate token reservations queue before the pool can be exhausted"
            )
        logger.info("llama-server runner created")
        self.update_status(RunnerIdle())

    # --- runner-contract plumbing (mirrors the llama.cpp runner) ---------------

    def update_status(self, status: RunnerStatus) -> None:
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def send_task_status(self, task: Task, status: TaskStatus) -> None:
        self.event_sender.send(
            TaskStatusUpdated(task_id=task.task_id, task_status=status)
        )

    def acknowledge_task(self, task: Task) -> None:
        record_runner_phase(
            "task_submission",
            event="task_acknowledged",
            detail=task.__class__.__name__,
            task_id=task.task_id,
        )
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    def main(self) -> None:
        # Concurrent dispatch: keeps up to SKULK_LLAMA_SERVER_PARALLEL generations
        # in flight so llama-server's continuous batching engages (the shared loop
        # streams each request on its own thread). LoadModel / Shutdown run inline
        # on the dispatch thread. Logic lives in ServedConcurrentDispatch.
        self.run_dispatch_loop()

    def _ensure_server_alive(self) -> None:
        """Raise if the spawned llama-server exited behind our back.

        Raising kills the runner process; the supervisor observes the crash and
        the peer-failure cascade tears down the whole instance (donors included
        for pooled placements) instead of leaving a wedged Ready runner.
        """
        proc = self.server_proc
        if proc is None or isinstance(
            self.current_status, (RunnerShuttingDown, RunnerShutdown)
        ):
            return
        if proc.poll() is not None:
            record_runner_phase(
                "error",
                event="server_exited",
                detail=f"llama-server exited unexpectedly (code {proc.returncode})",
            )
            raise RuntimeError(
                f"llama-server exited unexpectedly (code {proc.returncode}); "
                f"log tail:\n{self._server_log_tail()}"
            )

    def handle_task(self, task: Task) -> None:
        # TextGeneration and Shutdown are handled directly by the concurrent
        # dispatch loop (ServedConcurrentDispatch); this serves the inline
        # lifecycle path (LoadModel).
        match task:
            case LoadModel() if isinstance(self.current_status, RunnerIdle):
                self._load_model(task)
            case _:
                raise RuntimeError(
                    f"llama-server runner received unsupported task "
                    f"{task.__class__.__name__} in status "
                    f"{self.current_status.__class__.__name__}"
                )

    # --- model load: spawn + health-check the server --------------------------

    def _load_model(self, task: Task) -> None:
        self.update_status(RunnerLoading())
        self.acknowledge_task(task)

        from skulk.download.download_utils import (
            build_model_path,
            resolve_artifact_file,
        )

        card = self.shard_metadata.model_card
        model_id = card.model_id
        model_dir = build_model_path(
            ModelId(model_id),
            card.source_revision,
            card.artifact_bundle.root if card.artifact_bundle is not None else None,
        )
        # Load the file the card pinned (the selected quant); fall back to scanning
        # so download / sizing / loading stay in agreement. Reject an absolute or
        # ``..`` path that escapes the model dir.
        pinned = card.gguf_file
        gguf_path: Path | None = None
        if pinned:
            try:
                gguf_path = resolve_artifact_file(
                    model_dir,
                    card.artifact_bundle.root
                    if card.artifact_bundle is not None
                    else None,
                    pinned,
                )
            except (FileNotFoundError, ValueError):
                logger.warning(
                    f"card gguf_file {pinned!r} is missing or outside the model "
                    f"dir; scanning {model_dir} instead"
                )
        if gguf_path is None:
            gguf_path = select_gguf_file(model_dir)
        projector_path = self._resolve_projector(model_dir)

        # When the card declares a channel output parser we strip reasoning
        # markers ourselves, so llama-server must hand back raw text
        # (--reasoning-format none) regardless of whether the model "reasons":
        # its own reasoning parsers don't understand Gemma 4's <|channel> tokens.
        self._uses_channel_parser = (
            card.runtime is not None
            and card.runtime.output_parser == OutputParserType.Gemma4
        )
        reasoning_format_none = self._uses_channel_parser or not (
            _model_declares_reasoning(card)
        )
        n_ctx = self._serving_context_tokens
        try:
            with runner_phase(
                "load_model",
                detail="spawn_llama_server",
                task_id=task.task_id,
                attrs={
                    "gguf_file": gguf_path.name,
                    "n_ctx": n_ctx,
                    "parallel": self._max_concurrency,
                    # Every slot sees the whole window: above one slot the
                    # server runs with --kv-unified, and at one slot there is
                    # nothing to divide.
                    "slot_context_tokens": n_ctx,
                },
            ):
                self._spawn_server(
                    gguf_path,
                    n_ctx,
                    card.runtime,
                    reasoning_format_none,
                    projector_path,
                )
                self._await_health()
        except Exception:
            self._teardown_server()
            raise
        self.current_status = RunnerReady()
        record_runner_phase("idle", event="runner_ready", task_id=task.task_id)
        logger.info(
            f"llama-server runner ready in {time.time() - self.setup_start_time:.1f}s "
            f"(url={self.base_url})"
        )

    def _spawn_server(
        self,
        gguf_path: Path,
        n_ctx: int,
        runtime: Any,
        reasoning_format_none: bool,
        projector_path: Path | None = None,
    ) -> None:
        binary = os.environ.get(LLAMA_SERVER_BIN_ENV, "").strip()
        if not binary:
            raise RuntimeError(
                f"{LLAMA_SERVER_BIN_ENV} is not set; cannot launch llama-server"
            )
        # Validate the binary up front (same check the probe uses) so a
        # misconfigured path fails with a clear, actionable error rather than a
        # bare FileNotFoundError/PermissionError from Popen later.
        if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            raise RuntimeError(
                f"{LLAMA_SERVER_BIN_ENV}={binary!r} is not an executable file; "
                "point it at a llama-server binary built >= b9196 (for draft-mtp)"
            )
        port = self._pick_port()
        n_gpu_layers = _gpu_layers_for_backend(self.shard_metadata.resolved_backend)
        cmd: list[str] = [
            binary,
            "-m",
            str(gguf_path),
            "-ngl",
            n_gpu_layers,
            "-c",
            str(n_ctx),
            *_slot_server_args(self._max_concurrency),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            # --jinja enables the GGUF's chat template path, which is what makes
            # tool calling and reasoning-content extraction work server-side.
            "--jinja",
        ]
        # RPC driver (#328): dial the placement-stamped donor endpoints so
        # llama.cpp pools their GPU memory with the local device. Sorted for a
        # deterministic command line.
        if self._rpc_donor_endpoints:
            cmd += ["--rpc", ",".join(sorted(self._rpc_donor_endpoints.values()))]
        cmd += _projector_server_args(
            projector_path,
            self.shard_metadata.resolved_backend,
        )
        # --reasoning-format none hands back raw text in message.content. We use
        # it for (a) plain non-reasoning models (otherwise llama-server's default
        # `auto` extracts their prose into reasoning_content, leaving content
        # empty) and (b) models we parse ourselves (Gemma 4's <|channel> markers,
        # which llama-server's parsers mishandle). A reasoning model llama-server
        # *can* parse keeps the default so its thoughts land in reasoning_content.
        if reasoning_format_none:
            cmd += ["--reasoning-format", "none"]
        spec_type = getattr(runtime, "served_spec_type", None) if runtime else None
        # Operator/benchmark override: force plain decode for a served model whose
        # card asks for speculation. Serving the SAME GGUF with the spec flags
        # omitted is the apples-to-apples "MTP off" baseline for an on-vs-off
        # throughput comparison (identical weights, speculation disabled), and a
        # debug lever when a spec pairing misbehaves. Node-level, read at launch.
        if spec_type and spec_type != "none" and _force_no_spec():
            logger.info(
                f"SKULK_LLAMA_SERVER_FORCE_NO_SPEC set; serving {spec_type!r} model "
                "with speculative decoding disabled (plain decode)"
            )
            spec_type = None
        if spec_type and spec_type != "none":
            flag = _SPEC_TYPE_FLAG.get(spec_type)
            if flag is None:
                logger.warning(
                    f"unknown served_spec_type {spec_type!r}; serving without "
                    "speculative decoding"
                )
            else:
                # Resolve the draft FIRST: a card that declares a draft it
                # cannot provide (the draft is a best-effort companion, so a
                # failed cross-repo co-fetch is swallowed and the model is still
                # marked complete, #574) degrades to plain decode rather than
                # crashing. --spec-type is dropped too, since a draft-backed mode
                # without its draft is not a valid llama-server invocation.
                # None => the declared draft is unavailable; _draft_model_args
                # has already logged the specific reason, so drop the spec
                # silently (serve plain decode). Otherwise pass the spec + draft.
                card = self.shard_metadata.model_card
                draft_args = _draft_model_args(
                    runtime,
                    spec_type,
                    base_model_id=card.model_id,
                    source_revision=card.source_revision,
                )
                if draft_args is not None:
                    cmd += ["--spec-type", flag]
                    n_max = getattr(runtime, "served_spec_n_max", None)
                    if n_max is not None:
                        cmd += ["--spec-draft-n-max", str(n_max)]
                    cmd += draft_args

        self.server_log_path = (
            Path(tempfile.gettempdir()) / f"skulk-llama-server-{self.runner_id}.log"
        )
        self.server_log = open(self.server_log_path, "wb")  # noqa: SIM115
        # Modern llama.cpp links libllama.so / libggml*.so from the binary's own
        # directory (rpath $ORIGIN). Add that dir to LD_LIBRARY_PATH too so the
        # shared libs resolve regardless of the runner's working directory.
        env = os.environ.copy()
        bin_dir = str(Path(binary).resolve().parent)
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing}" if existing else bin_dir
        logger.info("launching llama-server: " + " ".join(cmd))
        self.server_proc = subprocess.Popen(  # noqa: S603 - args are built here, not user input
            cmd,
            stdout=self.server_log,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=_set_pdeathsig,  # noqa: PLW1509 - Linux reap-on-parent-death
        )
        self.base_url = f"http://127.0.0.1:{port}"

    def _resolve_projector(self, model_dir: Path) -> Path | None:
        """Resolve and authenticate the card-pinned served-vision projector.

        Legacy vision cards intentionally return ``None`` and remain gated to
        the in-process runner. A new served card must prove the exact projector
        path, byte size, and installed-manifest digest before llama-server is
        allowed to read it.
        """

        card = self.shard_metadata.model_card
        vision = card.vision
        if vision is None or not vision.has_pinned_projector:
            return None
        assert vision.projector_file is not None
        assert vision.projector_size is not None
        from skulk.download.download_utils import (
            artifact_install_directory,
            resolve_artifact_file,
        )
        from skulk.store.installed_cards import (
            read_installed_card_with_fallback,
            verify_installed_file,
        )

        artifact_root = (
            card.artifact_bundle.root if card.artifact_bundle is not None else None
        )
        install_directory = artifact_install_directory(model_dir, artifact_root)

        try:
            record = read_installed_card_with_fallback(install_directory)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"served vision projector metadata is unreadable for "
                f"{card.model_id}: {error}"
            ) from error
        if record is None or not verify_installed_file(
            install_directory,
            record,
            vision.projector_file,
            expected_size=vision.projector_size,
        ):
            raise RuntimeError(
                f"served vision projector {vision.projector_file!r} is missing, "
                "stale, incorrectly sized, or corrupt; re-stage the exact signed "
                "model generation"
            )
        return resolve_artifact_file(
            model_dir, artifact_root, vision.projector_file
        )

    def _pick_port(self) -> int:
        """Pick a free ephemeral port for the server, avoiding the API port."""
        for _ in range(30):
            port = random.randint(49153, 65535)
            if port == 52415:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    continue
            return port
        raise RuntimeError("could not find a free port for llama-server")

    def _await_health(self) -> None:
        assert self.server_proc is not None and self.base_url is not None
        deadline = time.time() + _HEALTH_DEADLINE_S
        with httpx.Client(timeout=5.0) as client:
            while time.time() < deadline:
                if self.server_proc.poll() is not None:
                    raise RuntimeError(
                        "llama-server exited during startup (code "
                        f"{self.server_proc.returncode}); log tail:\n"
                        f"{self._server_log_tail()}"
                    )
                try:
                    resp = client.get(f"{self.base_url}/health")
                    if (
                        resp.status_code == 200
                        and (resp.json() or {}).get("status") == "ok"
                    ):
                        return
                except Exception:  # noqa: BLE001 - not up yet; keep polling
                    pass
                time.sleep(2)
        raise RuntimeError(
            f"llama-server did not become healthy within {_HEALTH_DEADLINE_S:.0f}s; "
            f"log tail:\n{self._server_log_tail()}"
        )

    def _server_log_tail(self, lines: int = 30) -> str:
        if self.server_log_path is None or not self.server_log_path.exists():
            return "(no log)"
        try:
            text = self.server_log_path.read_text(errors="replace")
        except OSError:
            return "(log unreadable)"
        return "\n".join(text.splitlines()[-lines:])

    def _teardown_server(self) -> None:
        proc = self.server_proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
            self.server_proc = None
        if self.server_log is not None:
            with contextlib.suppress(Exception):
                self.server_log.close()
            self.server_log = None

    # --- generation: proxy the server's OpenAI streaming API ------------------

    def _count_input_tokens(self, body: dict[str, Any]) -> int | None:
        """Ask llama-server to count the exact rendered chat input.

        The endpoint applies the same model chat template and tokenizer as the
        subsequent completion, including tools and thinking controls. Returning
        ``None`` is a fail-safe signal: callers reserve the entire shared pool
        and run alone rather than guessing low and risking server termination.

        Args:
            body: OpenAI chat-completion body before stream-only fields.

        Returns:
            Exact input-token count, or ``None`` if the server cannot provide it.
        """
        assert self.base_url is not None
        try:
            response = httpx.post(
                f"{self.base_url}/v1/chat/completions/input_tokens",
                json=body,
                timeout=30.0,
            )
            response.raise_for_status()
            raw_count = response.json().get("input_tokens")
            if isinstance(raw_count, int) and not isinstance(raw_count, bool):
                return max(0, raw_count)
            raise ValueError("token-count response omitted integer input_tokens")
        except Exception as exc:  # noqa: BLE001 - safe full-pool fallback
            logger.opt(exception=exc).warning(
                "llama-server input-token count unavailable; reserving the "
                "whole shared KV pool for this request"
            )
            return None

    def _acquire_context_budget(self, task_id: TaskId, reservation: int) -> bool:
        """Wait in FIFO order until ``reservation`` fits the shared KV pool.

        Args:
            task_id: Generation waiting for admission.
            reservation: Exact prompt plus bounded maximum output tokens.

        Returns:
            ``True`` after reserving the budget, or ``False`` if cancelled while
            queued.
        """
        reservation = min(max(1, reservation), self._serving_context_tokens)
        waiter = _ContextWaiter(task_id, reservation)
        admitted = False
        with self._context_budget_condition:
            self._context_waiters.append(waiter)
            self._context_budget_condition.notify_all()
        try:
            while True:
                if self._is_cancelled(task_id):
                    return False
                with self._context_budget_condition:
                    is_next = (
                        bool(self._context_waiters)
                        and self._context_waiters[0] == waiter
                    )
                    if (
                        is_next
                        and self._reserved_context_tokens + reservation
                        <= self._serving_context_tokens
                    ):
                        self._context_waiters.popleft()
                        self._reserved_context_tokens += reservation
                        self._context_active_requests += 1
                        active_requests = self._context_active_requests
                        admitted = True
                        self._context_budget_condition.notify_all()
                        break
                    self._context_budget_condition.wait(timeout=_LIVENESS_POLL_S)
                self._ensure_server_alive()
        finally:
            if not admitted:
                with self._context_budget_condition:
                    with contextlib.suppress(ValueError):
                        self._context_waiters.remove(waiter)
                    self._context_budget_condition.notify_all()
        # Refine the generic submitted-task stamp: only requests past this gate
        # actively compete inside llama-server.
        self._set_admission_concurrency(task_id, active_requests)
        return True

    def _release_context_budget(self, reservation: int) -> None:
        """Release one request's shared-KV reservation and wake queued work."""
        reservation = min(max(1, reservation), self._serving_context_tokens)
        with self._context_budget_condition:
            self._reserved_context_tokens = max(
                0, self._reserved_context_tokens - reservation
            )
            self._context_active_requests = max(0, self._context_active_requests - 1)
            self._context_budget_condition.notify_all()

    def _generate(self, task: Task) -> None:
        # Runs on a pool worker thread. Runner status (Running/Ready) is owned by
        # the dispatch loop's in-flight counter, and the task was already
        # acknowledged at acceptance in the loop (before backpressure), so neither
        # is touched here.
        assert isinstance(task, TextGeneration)
        assert self.base_url is not None

        model_id = self.shard_metadata.model_card.model_id
        command_id = task.command_id
        body: dict[str, Any] = generation_kwargs(task.task_params)
        body["messages"] = messages_for_llama(task.task_params)
        # Forward thinking-control (enable_thinking / reasoning_effort) to
        # llama-server. Without this a reasoning model thinks on every request and
        # can return empty content under a bounded budget (#428/#420).
        body.update(
            reasoning_request_overrides(
                task.task_params, self.shard_metadata.model_card
            )
        )
        # Tool definitions change the rendered prompt, so include them before the
        # exact token-count request. _generate_with_tools sets the same fields
        # again before the real completion for local clarity.
        if task.task_params.tools:
            body["tools"] = task.task_params.tools
            tool_choice = getattr(task.task_params, "tool_choice", None)
            if tool_choice is not None:
                body["tool_choice"] = tool_choice

        input_tokens = self._count_input_tokens(body)
        if input_tokens is None:
            reservation = self._serving_context_tokens
            if "max_tokens" not in body:
                body["max_tokens"] = _LLAMA_SERVER_MAX_OUTPUT_TOKENS
        else:
            effective_max_output, reservation = _request_context_reservation(
                input_tokens,
                task.task_params.max_output_tokens,
                self._serving_context_tokens,
            )
            body["max_tokens"] = effective_max_output
        if not self._acquire_context_budget(task.task_id, reservation):
            return

        record_runner_phase(
            "task_submission",
            event="submit_text_generation",
            task_id=task.task_id,
            command_id=str(command_id),
            attrs={"tools": bool(task.task_params.tools)},
        )
        try:
            try:
                # Per-token logprobs are not wired over the SSE proxy yet. Fail loud
                # rather than return a successful response with logprobs silently
                # missing, matching the in-process runner's #385 no-silent-empty
                # contract (the raise surfaces as an ErrorChunk below).
                if wants_logprobs(
                    task.task_params.logprobs, task.task_params.top_logprobs
                ):
                    body.pop("logprobs", None)
                    body.pop("top_logprobs", None)
                    raise RuntimeError(
                        "Per-token logprobs are not supported on the served "
                        "(llama_server) engine: the OpenAI SSE proxy does not "
                        "surface them. Retry without logprobs/top_logprobs."
                    )
                record_runner_phase(
                    "decode_stream",
                    event="request_started",
                    task_id=task.task_id,
                    command_id=str(command_id),
                )
                if task.task_params.tools:
                    self._generate_with_tools(task, body, model_id, command_id)
                else:
                    self._generate_streaming(task, body, model_id, command_id)
            except Exception as exc:  # noqa: BLE001 - surface as an ErrorChunk
                record_runner_phase(
                    "error",
                    event="generation_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    task_id=task.task_id,
                    command_id=str(command_id),
                )
                logger.opt(exception=exc).warning("llama-server generation failed")
                self.event_sender.send(
                    ChunkGenerated(
                        command_id=command_id,
                        chunk=ErrorChunk(model=model_id, error_message=str(exc)),
                    )
                )
            else:
                # Read the shared cancel set through the lock-guarded helper:
                # generations run concurrently, so an unlocked membership read
                # here races the pool workers mutating cancelled_tasks.
                was_cancelled = self._was_cancelled(task.task_id)
                record_runner_phase(
                    "cancel_observed" if was_cancelled else "completion",
                    event="generation_finished",
                    task_id=task.task_id,
                    command_id=str(command_id),
                )
        finally:
            self._release_context_budget(reservation)
        # Status is NOT flipped here: the dispatch loop returns the runner to Ready
        # only when the LAST in-flight generation drains, so a peer generation still
        # streaming keeps the runner Running.

    def _generate_streaming(
        self,
        task: TextGeneration,
        body: dict[str, Any],
        model_id: ModelId,
        command_id: CommandId,
    ) -> None:
        body["stream"] = True
        # Ask llama-server to attach its native timings object to the final
        # streamed chunk: engine-side prompt_n/prompt_ms/predicted_n/
        # predicted_ms beat any proxy-side wall clock (#532).
        body["timings_per_token"] = True
        assert self.base_url is not None
        clock = StreamStatsClock()
        last_timings: dict[str, object] | None = None
        # In-flight captured at THIS task's admission on the dispatch loop, for
        # the performance-envelope tap (#596): the runner's own count is the true
        # per-instance concurrency, and reading it from the admission capture
        # (not live here on the worker thread) avoids a burst collapsing every
        # sample into one concurrency bucket.
        admission_in_flight = self._admission_concurrency(task.task_id)

        def final_stats() -> GenerationStats:
            # Engine timings win single-stream; under --parallel batching the
            # per-slot eval rates are not wall rates, so the proxy clock
            # provides the rates and the engine keeps the token counts (#611).
            # Peak memory always comes from the server child, never this proxy.
            base = served_stream_stats(
                clock, last_timings, batching=self._max_concurrency > 1
            ).model_copy(update={"peak_memory_usage": self._server_peak_memory()})
            return self.stamp_runner_stats(base, admission_in_flight)

        emitted_finish = False
        # Gemma 4 emits its reasoning as literal <|channel> markers in content;
        # reparse them here (llama-server can't) into reasoning/content chunks.
        parser = GemmaChannelTextParser() if self._uses_channel_parser else None
        # With no tools offered the server's own tool parser never runs, so a
        # model that writes a call anyway would leak its dialect markers to the
        # caller as content (#889). Scrub them here, the same invariant the
        # MLX path enforces in parse_tool_calls with emit_calls=False.
        scrub = (
            StreamingScaffoldingScrub() if not task.task_params.tools else None
        )

        def _content_pieces(text: str) -> str:
            return scrub.feed(text) if scrub is not None else text
        # No read timeout: generation can pause between tokens on a busy GPU. The
        # connection is closed (aborting server generation) when we break out.
        timeout = httpx.Timeout(connect=15.0, read=None, write=30.0, pool=None)
        with (
            httpx.Client(timeout=timeout) as client,
            client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=body
            ) as resp,
        ):
            resp.raise_for_status()
            for line in resp.iter_lines():
                if self._is_cancelled(task.task_id):
                    logger.info(f"llama-server generation cancelled: {task.task_id}")
                    break
                delta = _parse_sse_line(line)
                if delta is None:
                    continue
                if delta.done:
                    break
                if delta.timings is not None:
                    last_timings = delta.timings
                if delta.reasoning or delta.content:
                    # One SSE delta per generated token piece.
                    clock.mark_piece()
                if delta.reasoning:
                    self._send_token(
                        command_id, model_id, delta.reasoning, is_thinking=True
                    )
                if delta.content:
                    if parser is not None:
                        for text, is_thinking in parser.feed(delta.content):
                            emit = text if is_thinking else _content_pieces(text)
                            if emit or is_thinking:
                                self._send_token(
                                    command_id, model_id, emit, is_thinking=is_thinking
                                )
                    else:
                        emit = _content_pieces(delta.content)
                        if emit:
                            self._send_token(command_id, model_id, emit)
                if delta.finish is not None:
                    if parser is not None:
                        for text, is_thinking in parser.flush():
                            emit = text if is_thinking else _content_pieces(text)
                            if emit or is_thinking:
                                self._send_token(
                                    command_id, model_id, emit, is_thinking=is_thinking
                                )
                    if scrub is not None:
                        tail = scrub.flush()
                        if tail:
                            self._send_token(command_id, model_id, tail)
                    self._send_token(
                        command_id,
                        model_id,
                        "",
                        finish_reason=delta.finish,
                        stats=final_stats(),
                    )
                    emitted_finish = True
        # Guarantee a terminal chunk so the consumer's stream closes even if the
        # server ended without an explicit finish_reason; drain any held tail.
        if not emitted_finish and not self._is_cancelled(task.task_id):
            if parser is not None:
                for text, is_thinking in parser.flush():
                    emit = text if is_thinking else _content_pieces(text)
                    if emit or is_thinking:
                        self._send_token(
                            command_id, model_id, emit, is_thinking=is_thinking
                        )
            if scrub is not None:
                tail = scrub.flush()
                if tail:
                    self._send_token(command_id, model_id, tail)
            self._send_token(
                command_id, model_id, "", finish_reason="stop", stats=final_stats()
            )

    def _generate_with_tools(
        self,
        task: TextGeneration,
        body: dict[str, Any],
        model_id: ModelId,
        command_id: CommandId,
    ) -> None:
        # Tool calls are requested non-streamed (the caller wants the assembled
        # call): llama-server parses the model's native tool-call format and
        # returns structured ``tool_calls`` via --jinja, so no text parsing here.
        body["stream"] = False
        body["tools"] = task.task_params.tools
        tool_choice = getattr(task.task_params, "tool_choice", None)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        assert self.base_url is not None
        if self._is_cancelled(task.task_id):
            return
        timeout = httpx.Timeout(connect=15.0, read=None, write=30.0, pool=None)
        request_started = time.perf_counter()
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{self.base_url}/v1/chat/completions", json=body)
            resp.raise_for_status()
            result = resp.json()
        request_seconds = time.perf_counter() - request_started
        # A cancel that arrived while the (non-streamed) request was in flight:
        # drain it (the streaming path checks every chunk; this blocking path has
        # no mid-flight checkpoint) and skip emission so main() marks the task
        # Cancelled, not Complete, and no tool call is surfaced for it.
        if self._is_cancelled(task.task_id):
            logger.info(f"llama-server tool generation cancelled: {task.task_id}")
            return
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        # Engine-side timings when the server includes them, usage-derived
        # effective rates otherwise (#532).
        raw_timings = result.get("timings")
        # Same wall-honesty rule as the streaming path (#611): engine timings
        # only when serving single-stream; under batching the usage-derived
        # whole-request wall rates are the honest ones, with the prompt RATE
        # over the engine's processed count so a slot-cache hit's cached
        # prefix cannot inflate it.
        processed_prompt: int | None = None
        if isinstance(raw_timings, dict):
            raw_prompt_n = cast("dict[str, object]", raw_timings).get("prompt_n")
            if isinstance(raw_prompt_n, (int, float)) and not isinstance(
                raw_prompt_n, bool
            ):
                processed_prompt = int(raw_prompt_n)
        stats = (
            stats_from_llama_server_timings(raw_timings)
            if isinstance(raw_timings, dict) and self._max_concurrency <= 1
            else None
        ) or blocking_call_stats(
            result.get("usage"), request_seconds, processed_prompt
        )
        if stats is not None:
            # The model lives in the server child; never report proxy RSS.
            stats = stats.model_copy(
                update={"peak_memory_usage": self._server_peak_memory()}
            )
            # Stamp runner attribution (#596) so tool-call generations feed the
            # per-instance performance envelope exactly like the streaming path;
            # otherwise tool workloads (the agentic served audience) would fall
            # back to the API's ambiguous offered-count attribution.
            stats = self.stamp_runner_stats(
                stats, self._admission_concurrency(task.task_id)
            )
        tool_calls = tool_calls_from_message(message)
        if tool_calls:
            self.event_sender.send(
                ChunkGenerated(
                    command_id=command_id,
                    chunk=ToolCallChunk(
                        model=model_id, tool_calls=tool_calls, usage=None, stats=stats
                    ),
                )
            )
            return
        # The model answered in prose: emit its reasoning + content, then close.
        # Preserve the server's finish_reason (e.g. "length" when the answer hit
        # max_tokens) rather than hard-coding "stop", so a truncated prose answer
        # still signals truncation to the client.
        reasoning = message.get("reasoning_content") or ""
        content = message.get("content") or ""
        if reasoning:
            self._send_token(command_id, model_id, reasoning, is_thinking=True)
        if content:
            if self._uses_channel_parser:
                # Reparse Gemma 4's <|channel> markers out of the prose answer.
                parser = GemmaChannelTextParser()
                for text, is_thinking in parser.feed(content) + parser.flush():
                    self._send_token(
                        command_id, model_id, text, is_thinking=is_thinking
                    )
            else:
                self._send_token(command_id, model_id, content)
        finish = map_finish_reason(choice.get("finish_reason")) or "stop"
        self._send_token(command_id, model_id, "", finish_reason=finish, stats=stats)

    def _server_peak_memory(self) -> Memory:
        """Peak RSS of the llama-server child, or zero when unmeasurable.

        The weights and KV cache live in the external server process, so the
        proxy's own RSS would misattribute memory in telemetry (#536 review).
        Zero means "unmeasured" (non-Linux, or the child already exited);
        a fabricated proxy-side figure would be worse than none.
        """
        proc = self.server_proc
        if proc is None:
            return Memory.from_bytes(0)
        return subprocess_peak_memory(proc.pid) or Memory.from_bytes(0)

    def _send_token(
        self,
        command_id: CommandId,
        model_id: ModelId,
        text: str,
        *,
        is_thinking: bool = False,
        finish_reason: Any = None,
        stats: GenerationStats | None = None,
    ) -> None:
        """Emit one TokenChunk; skip empty non-terminal chunks."""
        if not text and finish_reason is None:
            return
        self.event_sender.send(
            ChunkGenerated(
                command_id=command_id,
                chunk=TokenChunk(
                    model=model_id,
                    text=text,
                    token_id=-1,  # the OpenAI proxy stream does not expose ids
                    usage=None,
                    finish_reason=finish_reason,
                    is_thinking=is_thinking,
                    stats=stats,
                ),
            )
        )
