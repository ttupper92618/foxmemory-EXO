# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Served-backend text-generation runner: launches and proxies ``vllm serve``.

The second *served* engine (after ``llama_server``), reusing that engine's
generic shape -- a managed inference-server subprocess plus an OpenAI HTTP proxy
-- with the vLLM CLI instead of ``llama-server``. vLLM is the GPU-serving fast
path: its continuous batching and paged attention hold latency flat and grow
aggregate throughput under concurrent load, where the single-stream engines
(``llama_cpp`` / ``llama_server``) collapse. It coexists with those engines and
is selected per model by the card's ``compatible_backends`` on a node that set
``SKULK_VLLM_BIN``.

Single-node only in this first slice (no ring / warmup / RPC), mirroring the
in-process runners. Linux-oriented: the subprocess is reaped on parent death via
``PR_SET_PDEATHSIG`` so a runner crash never orphans a ``vllm serve`` process
holding GPU memory. Per-request cancellation aborts the proxied HTTP connection
(stopping server-side generation); ``SIGTERM`` is for whole-server teardown.

Scope of this slice: streamed chat completions only. Tool calling and per-token
logprobs are rejected loudly rather than silently mismeasured (the OpenAI SSE
proxy does not surface logprobs, and tool-call round-tripping is a follow-up).

Reasoning is best-effort in this slice. Thinking control (``enable_thinking`` /
``reasoning_effort``) is forwarded so the model thinks, and both ``reasoning_content``
and ``reasoning`` SSE deltas are parsed into ``is_thinking`` chunks. But vLLM only
SPLITS reasoning from content when the server is launched with a family-specific
``--reasoning-parser`` (e.g. ``qwen3`` / ``deepseek_r1`` / ``openai_gptoss``); this
slice does not yet map the card to that flag, so on a reasoning model the thinking
text arrives inline in ``content`` (raw markers) rather than as a separated
reasoning stream. Threading the card's reasoning family into ``--reasoning-parser``
is a follow-up (alongside tool calling and logprobs).

Concurrent dispatch: unlike the in-process runners (which serialize one task at a
time), ``main()`` keeps up to N ``TextGeneration`` requests in flight at once,
each streaming its own HTTP request to the one shared ``vllm serve`` on its own
thread. That is what actually lets vLLM's continuous batching + paged attention
activate: the server sees concurrent in-flight requests and batches their decode
steps, holding latency flat and growing aggregate throughput under load (the
whole reason this engine exists). N is bounded by a thread pool sized from
``SKULK_VLLM_MAX_CONCURRENT_REQUESTS`` (vLLM itself caps at ``--max-num-seqs``, so
this is a client-side admission bound, not the batch width). Runner status is
``RunnerRunning`` while any generation is in flight and ``RunnerReady`` when the
last one drains; ``MpSender`` event sends and the diagnostic emitter are already
thread-safe, and each ``DataChunk`` carries ``command_id`` + ``sequence`` so the
API demultiplexes interleaved token streams. The lifecycle tasks (``LoadModel``,
``Shutdown``) run inline on the dispatch thread; shutdown cancels every in-flight
generation and drains the pool before tearing down the server.
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
import time
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, cast

import httpx

from skulk.api.types import GenerationStats
from skulk.download.download_utils import build_model_path
from skulk.shared.backends import VLLM_BIN_ENV
from skulk.shared.models.memory_estimate import VLLM_MAX_MODEL_LEN
from skulk.shared.models.model_cards import ModelCard
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
from skulk.shared.types.worker.instances import BoundInstance
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
    subprocess_peak_memory,
)
from skulk.worker.runner.llama_cpp.runner import (
    map_finish_reason,
    messages_for_llama,
    serving_n_ctx,
    tool_calls_from_message,
    wants_logprobs,
)
from skulk.worker.runner.llm_inference.reasoning_controls import (
    muse_glimmer_strength_kwargs,
)
from skulk.worker.runner.llm_inference.scaffolding_scrub import (
    StreamingScaffoldingScrub,
)
from skulk.worker.runner.served_concurrency import ServedConcurrentDispatch
from skulk.worker.runner.vllm.orphan_sweep import sweep_orphaned_vllm_engines

# vLLM startup can be slow: weight load + torch.compile + CUDA-graph capture on a
# large model runs to minutes, and 0.28.0's torch-2.13 AOT compile chain pushed a
# COLD-cache first start past 600s (observed live: a 27B FP8 model on A100-80GB
# finished compiling at ~10:05 and was killed by the old 600s deadline moments
# before health). Warm compile caches come up far faster; the ceiling must fit
# the cold case because every fresh node hits it. A crashed server is caught
# separately by process-exit detection, so a long ceiling does not delay real
# failure reporting.
_HEALTH_DEADLINE_S: Final = 1800.0

_PORT_COLLISION_ATTEMPTS: Final = 3
"""Startup attempts allowed when the chosen server port is taken at bind time."""

_ADDRESS_IN_USE_MARKER: Final = "Address already in use"
"""Marker identifying a lost port race in a failed server's log tail."""

# Fraction of GPU VRAM vLLM may use for weights + KV cache. Operator-tunable via
# env; vLLM's own default is 0.90. Placement admits against the same usable-VRAM
# figure, so this stays a node-local serving knob for now (a card-level override
# is a follow-up when vLLM-aware admission lands).
_GPU_MEMORY_UTILIZATION_ENV: Final = "SKULK_VLLM_GPU_MEMORY_UTILIZATION"
_DEFAULT_GPU_MEMORY_UTILIZATION: Final = 0.90

# Upper bound on concurrent in-flight generations the runner streams to the one
# ``vllm serve`` at once. This is a client-side admission bound (the thread-pool
# width), NOT vLLM's batch width -- the server batches up to its own
# ``--max-num-seqs`` (a version- and hardware-band-dependent default, 256 at
# minimum on the validated matrix). Kept below every such default so queued
# requests wait in the runner's bounded pool rather than piling unbounded
# threads against the server.
_MAX_CONCURRENT_REQUESTS_ENV: Final = "SKULK_VLLM_MAX_CONCURRENT_REQUESTS"
_DEFAULT_MAX_CONCURRENT_REQUESTS: Final = 32

# Deep-speculation scheduler budget. vLLM reserves draft slots per sequence
# out of --max-num-batched-tokens, so a deep drafter needs
# batched >= seqs * (depth - 1) or engine init fails with a negative
# max_num_scheduled_tokens ("set to -1536" observed live with the Laguna
# depth-15 card on vLLM 0.25.1, whose effective defaults were 2048/256).
# Both sides of that arithmetic are version- and hardware-band-dependent
# (0.28.0 raises serve-time defaults as high as 16384/1024 on large GPUs),
# so at deep depths the runner pins BOTH flags explicitly rather than
# raising one against an assumed default: max-num-seqs is pinned to the
# 0.25.1-validated 256 and the batched budget derives from it. Depth 9 is
# where the historical default budget went non-positive (2048 - 256 * 8);
# the sizing kicks in at 8 because depth 8 leaves a degenerate 256-token
# budget. 8192 is the fresh-box-validated floor (Laguna depth-15 card,
# A100-80GB).
_SPEC_PINNED_BATCH_BASE_TOKENS: Final = 2048
_SPEC_PINNED_MAX_NUM_SEQS: Final = 256
_SPEC_DEPTH_NEEDING_BATCH_SIZING: Final = 8
_SPEC_BATCHED_TOKENS_FLOOR: Final = 8192


def _max_concurrent_requests() -> int:
    """The concurrent in-flight generation cap, from env or the default.

    An unparseable or below-1 value falls back to the default rather than
    disabling concurrency (0) or crashing the pool at construction.
    """
    raw = os.environ.get(_MAX_CONCURRENT_REQUESTS_ENV, "").strip()
    if not raw:
        return _DEFAULT_MAX_CONCURRENT_REQUESTS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            f"{_MAX_CONCURRENT_REQUESTS_ENV}={raw!r} is not an integer; "
            f"using {_DEFAULT_MAX_CONCURRENT_REQUESTS}"
        )
        return _DEFAULT_MAX_CONCURRENT_REQUESTS
    if value < 1:
        logger.warning(
            f"{_MAX_CONCURRENT_REQUESTS_ENV}={value} is below 1; "
            f"using {_DEFAULT_MAX_CONCURRENT_REQUESTS}"
        )
        return _DEFAULT_MAX_CONCURRENT_REQUESTS
    return value


class _StreamDelta(NamedTuple):
    """One parsed SSE delta from the proxied ``/v1/chat/completions`` stream."""

    reasoning: str
    content: str
    finish: Literal["stop", "length", "content_filter"] | None
    done: bool  # the terminal ``data: [DONE]`` sentinel
    usage: dict[str, Any] | None = None  # the include_usage final-chunk counts


def _gpu_memory_utilization() -> float:
    """The ``--gpu-memory-utilization`` fraction, from env or the 0.90 default.

    An unparseable or out-of-range (0, 1] value falls back to the default rather
    than passing vLLM a nonsense fraction that would fail the server at spawn.
    """
    raw = os.environ.get(_GPU_MEMORY_UTILIZATION_ENV, "").strip()
    if not raw:
        return _DEFAULT_GPU_MEMORY_UTILIZATION
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            f"{_GPU_MEMORY_UTILIZATION_ENV}={raw!r} is not a number; "
            f"using {_DEFAULT_GPU_MEMORY_UTILIZATION}"
        )
        return _DEFAULT_GPU_MEMORY_UTILIZATION
    if not 0.0 < value <= 1.0:
        logger.warning(
            f"{_GPU_MEMORY_UTILIZATION_ENV}={value} is outside (0, 1]; "
            f"using {_DEFAULT_GPU_MEMORY_UTILIZATION}"
        )
        return _DEFAULT_GPU_MEMORY_UTILIZATION
    return value


def build_vllm_serve_args(
    binary: str,
    model_dir: Path,
    served_model_name: str,
    host: str,
    port: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
    spec_method: str | None = None,
    spec_num_tokens: int | None = None,
    spec_draft_repo: str | None = None,
    spec_draft_revision: str | None = None,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
) -> list[str]:
    """Build the ``vllm serve`` command line. Pure, so it is unit-testable.

    vLLM auto-detects the platform (CUDA vs ROCm) from its own install, so unlike
    llama-server there is no per-compute-backend flag to set -- the node advertised
    ``vllm-cuda`` / ``vllm-rocm`` only because the matching vLLM build is present.
    ``--served-model-name`` pins the model id callers address (the Skulk model id),
    decoupled from the on-disk directory path. ``--trust-remote-code`` is added when
    the card permits it (the ModelCard default; required by custom-code HF repos)
    since vLLM's flag defaults off and those models would otherwise fail at startup.
    """
    args = [
        binary,
        "serve",
        str(model_dir),
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        f"{gpu_memory_utilization:.2f}",
        "--tensor-parallel-size",
        "1",
        # Off by default in vLLM; without it the include_usage final chunk
        # carries no prompt_tokens_details, so the cache-honest prompt rate
        # (#631) would silently never subtract cached prefix tokens.
        "--enable-prompt-tokens-details",
    ]
    if trust_remote_code:
        args.append("--trust-remote-code")
    if tool_call_parser is not None:
        # vLLM refuses --tool-call-parser without --enable-auto-tool-choice;
        # the pair enables server-side parsing of the model's native
        # tool-call format into structured OpenAI tool_calls, mirroring the
        # llama_server runner's --jinja role.
        args.extend(
            ["--enable-auto-tool-choice", "--tool-call-parser", tool_call_parser]
        )
    if reasoning_parser is not None:
        # Card-pinned reasoning parser (runtime.vllm_reasoning_parser): vLLM
        # only splits reasoning_content from content when one is configured;
        # without it a reasoning model's thinking streams inline as answer
        # text. Explicit only, like the tool parser.
        args.extend(["--reasoning-parser", reasoning_parser])
    if spec_method is not None:
        # Card-declared speculative decoding (runtime.vllm_spec_method /
        # vllm_spec_num_tokens / vllm_spec_draft_repo): method "mtp" engages
        # the checkpoint's own native prediction heads (vLLM resolves the
        # matching drafter architecture, no separate draft model); measured
        # on Qwen3.6-27B-FP8: 2.01x single-stream decode at
        # num_speculative_tokens=2. Draft-model methods ("dflash") name a
        # separate speculator repo in the config's "model" key, which vLLM
        # resolves from its own HF cache at engine start (the card validator
        # guarantees the method/draft-repo pairing is consistent).
        speculative: dict[str, Any] = {"method": spec_method}
        if spec_num_tokens is not None:
            speculative["num_speculative_tokens"] = spec_num_tokens
        if spec_draft_repo is not None:
            speculative["model"] = spec_draft_repo
        if spec_draft_revision is not None:
            speculative["revision"] = spec_draft_revision
        args.extend(["--speculative-config", json.dumps(speculative)])
        if (
            spec_num_tokens is not None
            and spec_num_tokens >= _SPEC_DEPTH_NEEDING_BATCH_SIZING
        ):
            # Deep block-parallel drafters need the scheduler budget sized
            # to the depth, and BOTH flags pinned so the arithmetic cannot
            # be broken by a vLLM default change underneath us (see the
            # constant block above). Shallow MTP depths keep vLLM's
            # defaults untouched (the shape the #649 cards validated
            # under).
            batched = max(
                _SPEC_BATCHED_TOKENS_FLOOR,
                _SPEC_PINNED_BATCH_BASE_TOKENS
                + _SPEC_PINNED_MAX_NUM_SEQS * (spec_num_tokens - 1),
            )
            args.extend(
                [
                    "--max-num-batched-tokens",
                    str(batched),
                    "--max-num-seqs",
                    str(_SPEC_PINNED_MAX_NUM_SEQS),
                ]
            )
    return args


def tool_call_finish_surfaces(raw_finish: object) -> bool:
    """Whether a non-streamed response's parsed tool calls may reach the caller.

    "stop" is a COMPLETE generation and must surface its calls: with a named
    ``tool_choice``, vLLM follows OpenAI semantics and reports the forced call
    under finish_reason "stop" rather than "tool_calls" (observed live on
    0.28.0; excluding it returned an empty stop chunk to the caller). Only
    length/content_filter cut a call short with incomplete arguments, so those
    finishes keep the calls unsurfaced and fall through to the prose path.

    Args:
        raw_finish: The server's raw ``finish_reason`` for the choice.

    Returns:
        ``True`` when parsed tool calls are complete and safe to surface.
    """
    return raw_finish in (None, "stop", "tool_calls")


def resolve_vllm_reasoning_parser(card: ModelCard) -> str | None:
    """The vLLM ``--reasoning-parser`` name a card pins, or ``None``.

    Explicit ``runtime.vllm_reasoning_parser`` only, the same doctrine as
    :func:`resolve_vllm_tool_call_parser`: there is no family fallback, so an
    unpinned card launches without a reasoning parser and its thinking, if
    any, arrives inline.
    """
    runtime = card.runtime
    if runtime is not None and runtime.vllm_reasoning_parser is not None:
        return runtime.vllm_reasoning_parser
    return None


def resolve_vllm_tool_call_parser(card: ModelCard) -> str | None:
    """The vLLM tool parser this card should launch with, or None.

    Explicit ``runtime.vllm_tool_call_parser`` only: there is deliberately
    no family fallback, because one Skulk family string can span tool-call
    generations with different wire formats (Qwen2.5 emits Hermes JSON
    while Qwen3.6 emits the XML function format), and a wrong parser fails
    at request time with opaque server errors rather than at card review.
    A card without the field launches without the parser pair and tool
    requests are rejected loudly at request time (the #385 no-silent-empty
    contract).
    """
    runtime = card.runtime
    if runtime is not None and runtime.vllm_tool_call_parser is not None:
        return runtime.vllm_tool_call_parser
    return None


def vllm_generation_kwargs(task_params: Any) -> dict[str, Any]:
    """Translate Skulk sampling params into vLLM ``/v1/chat/completions`` fields.

    Distinct from the llama.cpp mapper (``generation_kwargs``): vLLM's OpenAI server
    uses OpenAI/HF parameter names, so the repetition control is ``repetition_penalty``
    (llama.cpp's ``repeat_penalty`` would be silently ignored by vLLM). ``top_k`` /
    ``min_p`` are vLLM sampling extensions passed through by name. Pure, so the
    mapping is unit-testable. Thinking control is layered separately by
    :func:`vllm_reasoning_overrides`; logprobs are rejected before this is called.
    """
    kwargs: dict[str, Any] = {}
    if task_params.max_output_tokens is not None:
        kwargs["max_tokens"] = task_params.max_output_tokens
    if task_params.temperature is not None:
        kwargs["temperature"] = task_params.temperature
    if task_params.top_p is not None:
        kwargs["top_p"] = task_params.top_p
    if task_params.top_k is not None:
        kwargs["top_k"] = task_params.top_k
    if task_params.min_p is not None:
        kwargs["min_p"] = task_params.min_p
    if task_params.repetition_penalty is not None:
        kwargs["repetition_penalty"] = task_params.repetition_penalty
    if task_params.stop is not None:
        kwargs["stop"] = task_params.stop
    if task_params.seed is not None:
        kwargs["seed"] = task_params.seed
    return kwargs


def vllm_reasoning_overrides(
    task_params: Any, card: ModelCard | None = None
) -> dict[str, Any]:
    """Map Skulk's thinking controls onto vLLM request fields.

    vLLM's OpenAI server exposes the same two levers as llama-server:
    ``chat_template_kwargs`` (the model's jinja template reads ``enable_thinking``,
    the Qwen3 / Gemma toggle; a template that ignores it is harmless) and
    ``reasoning_effort`` (OpenAI-style effort for gpt-oss). Without this the sampling
    body carries no thinking control, so ``enable_thinking=False`` would be silently
    ignored and a reasoning model would think on every request. ``"none"`` effort is
    not a valid server value (disabling goes through ``enable_thinking=False``), so
    it is dropped. A Muse Glimmer card (resolved from ``card``) reads neither
    lever: its template takes a ``reasoning_strength`` kwarg, which the effort
    is translated onto instead.
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


def _usage_count(usage: dict[str, Any] | None, key: str) -> int | None:
    """A non-negative integer token count from a usage object, or ``None``.

    Bool-guarded like every timings extraction: JSON ``true`` is an ``int``
    to ``isinstance`` and must not read as a count of one.
    """
    if usage is None:
        return None
    value = usage.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def parse_openai_sse_line(line: str) -> _StreamDelta | None:
    """Parse one OpenAI SSE line into a ``_StreamDelta``, or ``None`` to skip it.

    Handles the standard streaming shape vLLM emits: ``data: {json}`` lines whose
    first choice carries a ``delta`` (``content`` and/or ``reasoning_content`` when
    a reasoning parser is configured) plus an optional ``finish_reason``, and the
    terminal ``data: [DONE]``. Returns ``None`` for non-``data:`` lines; ``[DONE]``
    is reported via ``done=True``; malformed JSON or a choice-less payload is
    skipped (``None``) so a stray line never breaks the stream. Pure (no I/O).
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
    raw_usage = chunk.get("usage")
    usage = raw_usage if isinstance(raw_usage, dict) else None
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        # The stream_options include_usage final chunk carries exactly
        # ``"choices": []`` plus the engine-exact token counts (#631); only
        # that shape is accepted, so a malformed payload that happens to have
        # a usage-shaped field stays skipped like any other stray line.
        if usage is not None and isinstance(choices, list) and not choices:
            return _StreamDelta("", "", None, done=False, usage=usage)
        return None
    choice = choices[0]
    raw_delta = choice.get("delta")
    delta = raw_delta if isinstance(raw_delta, dict) else {}
    # Preserve OpenAI's `content_filter` finish reason, which vLLM can emit but the
    # shared llama.cpp `map_finish_reason` collapses to `stop` (llama.cpp never
    # emits it); otherwise a filtered response is misreported as a normal stop.
    raw_finish = choice.get("finish_reason")
    finish = (
        "content_filter"
        if raw_finish == "content_filter"
        else map_finish_reason(raw_finish)
    )
    return _StreamDelta(
        # vLLM has streamed reasoning under `reasoning_content` (DeepSeek-style)
        # and, in newer versions, `reasoning`; accept both so a reasoning model's
        # thinking stream is not silently dropped.
        reasoning=delta.get("reasoning_content") or delta.get("reasoning") or "",
        content=delta.get("content") or "",
        finish=finish,
        done=False,
        usage=usage,
    )


def _set_pdeathsig() -> None:
    """Ask the kernel to SIGKILL this child when its parent (the runner) dies.

    Runs in the forked child before ``exec`` (``preexec_fn``). Linux-only,
    best-effort: a runner-process crash must never leave an orphaned ``vllm serve``
    holding GPU memory. Any failure is swallowed (the explicit teardown path still
    applies on graceful shutdown).
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGKILL, 0, 0, 0)
    except Exception:  # noqa: BLE001 - best-effort; non-Linux or no libc
        pass


class Runner(ServedConcurrentDispatch):
    """Single-node served-backend runner that proxies an external ``vllm serve``.

    Lifecycle mirrors the ``llama_server`` runner: it skips the ring
    (``ConnectToGroup`` / ``StartWarmup``), spawns the server on ``LoadModel``, and
    serves ``TextGeneration`` by streaming the server's SSE output back as
    ``ChunkGenerated`` events. The concurrent-dispatch machinery (bounded pool,
    backpressure, in-flight status, cancellation) is inherited from
    ``ServedConcurrentDispatch``; this class provides the vLLM-specific hooks
    (``_generate``, server spawn/liveness/teardown, ``handle_task``).
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
        # vLLM is single-node in this slice: vLLM's own tensor/pipeline parallelism
        # is a later track, so any multi-node placement reaching here is a bug.
        if self.shard_metadata.world_size != 1:
            raise RuntimeError(
                "vllm runner requires single-node placement, got "
                f"world_size={self.shard_metadata.world_size}"
            )
        self.setup_start_time = time.time()
        self.cancelled_tasks: set[TaskId] = set()
        self.seen: set[TaskId] = set()
        # Concurrent-dispatch state lives in ServedConcurrentDispatch (shared with
        # the llama_server runner): the bounded pool, the SUBMITTED-job backpressure
        # semaphore, the lock-guarded in-flight/status accounting, and the dispatch
        # loop. This runner supplies only the vLLM-specific hooks below.
        self._init_concurrent_dispatch(_max_concurrent_requests(), "vllm-gen")
        self.server_proc: subprocess.Popen[bytes] | None = None
        self.server_log: Any = None
        self.server_log_path: Path | None = None
        self.base_url: str | None = None
        # Resolved at spawn; None means the server was launched without the
        # tool-parser pair and tool requests must be rejected loudly.
        self._tool_call_parser: str | None = None
        self.current_status: RunnerStatus = RunnerIdle()
        logger.info("vllm runner created")
        self.update_status(RunnerIdle())

    # --- runner-contract plumbing (mirrors the llama_server runner) ------------

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
        # The concurrent dispatch loop (bounded pool, SUBMITTED-job backpressure,
        # lock-guarded in-flight status, cancellation drain, Ready-after-Complete
        # ordering, and the shutdown drain) lives in ServedConcurrentDispatch.
        self.run_dispatch_loop()

    def _ensure_server_alive(self) -> None:
        """Raise if the spawned ``vllm serve`` exited behind our back.

        Raising kills the runner process; the supervisor observes the crash and
        the peer-failure cascade tears the instance down instead of leaving a
        wedged Ready runner.
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
                detail=f"vllm serve exited unexpectedly (code {proc.returncode})",
            )
            raise RuntimeError(
                f"vllm serve exited unexpectedly (code {proc.returncode}); "
                f"log tail:\n{self._server_log_tail()}"
            )

    def handle_task(self, task: Task) -> None:
        # TextGeneration and Shutdown are handled directly by the concurrent
        # dispatch loop in main(); this serves the inline lifecycle path (LoadModel).
        match task:
            case LoadModel() if isinstance(self.current_status, RunnerIdle):
                self._load_model(task)
            case _:
                raise RuntimeError(
                    f"vllm runner received unsupported task "
                    f"{task.__class__.__name__} in status "
                    f"{self.current_status.__class__.__name__}"
                )

    # --- model load: spawn + health-check the server --------------------------

    def _load_model(self, task: Task) -> None:
        self.update_status(RunnerLoading())
        self.acknowledge_task(task)

        card = self.shard_metadata.model_card
        model_id = card.model_id
        model_dir = build_model_path(
            ModelId(model_id),
            card.source_revision,
            card.artifact_bundle.root if card.artifact_bundle is not None else None,
        )
        # Placement stamps the vllm startup-cost cap into context_token_limit
        # (VLLM_MAX_MODEL_LEN at the stamp, so admission and the served
        # window agree); the min() here is defense in depth for instances
        # stamped by a pre-cap master during a rolling window.
        stamped_n_ctx = serving_n_ctx(self.context_token_limit, logits_all=False)
        n_ctx = min(stamped_n_ctx, VLLM_MAX_MODEL_LEN)
        if n_ctx < stamped_n_ctx:
            logger.warning(
                f"vllm max-model-len defensively capped at {n_ctx} (stamped "
                f"{stamped_n_ctx}): the instance predates the stamp-side cap"
            )
        try:
            with runner_phase(
                "load_model",
                detail="spawn_vllm_serve",
                task_id=task.task_id,
                attrs={"model_dir": model_dir.name, "n_ctx": n_ctx},
            ):
                self._spawn_server_with_port_retry(model_dir, str(model_id), n_ctx)
        except Exception:
            self._teardown_server()
            raise
        self.current_status = RunnerReady()
        record_runner_phase("idle", event="runner_ready", task_id=task.task_id)
        logger.info(
            f"vllm runner ready in {time.time() - self.setup_start_time:.1f}s "
            f"(url={self.base_url})"
        )

    def _spawn_server_with_port_retry(
        self, model_dir: Path, served_model_name: str, n_ctx: int
    ) -> None:
        """Start the server, retrying a lost port race with a fresh port.

        ``_pick_port`` proves a port free by binding and closing it, but the
        server binds it seconds later, after Python and vLLM start up. The
        probe range is the kernel's ephemeral range, so in that window any
        outbound connection on a busy node (data plane, store transfers,
        telemetry) can be assigned the same port and the server's own bind
        fails with EADDRINUSE. Losing that race is transient and retryable;
        every other startup failure is not, and is re-raised on the spot so
        real faults still fail fast.
        """
        for attempt in range(1, _PORT_COLLISION_ATTEMPTS + 1):
            self._spawn_server(model_dir, served_model_name, n_ctx)
            try:
                self._await_health()
            except RuntimeError as exc:
                lost_race = (
                    _ADDRESS_IN_USE_MARKER in str(exc)
                    and attempt < _PORT_COLLISION_ATTEMPTS
                )
                if not lost_race:
                    raise
                logger.warning(
                    "vllm serve lost a port race on startup "
                    f"(attempt {attempt}/{_PORT_COLLISION_ATTEMPTS}); "
                    "retrying with a fresh port"
                )
                # Reclaim the failed process and its log before rebinding, so a
                # retry cannot leak the previous attempt's handles.
                self._teardown_server()
                continue
            return

    def _spawn_server(
        self, model_dir: Path, served_model_name: str, n_ctx: int
    ) -> None:
        binary = os.environ.get(VLLM_BIN_ENV, "").strip()
        if not binary or not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
            # Validate up front (like the llama_server runner) so a misconfigured or
            # vanished-since-probe binary surfaces as a clear runner error rather
            # than a bare FileNotFoundError/PermissionError out of subprocess.Popen.
            raise RuntimeError(
                f"{VLLM_BIN_ENV}={binary!r} is not an existing executable; the vllm "
                "runner cannot spawn a server. This node should not have been a "
                "placement candidate for a vLLM card."
            )
        host = "127.0.0.1"
        port = self._pick_port()
        self.base_url = f"http://{host}:{port}"
        card_runtime = self.shard_metadata.model_card.runtime
        self._tool_call_parser = resolve_vllm_tool_call_parser(
            self.shard_metadata.model_card
        )
        args = build_vllm_serve_args(
            binary,
            model_dir,
            served_model_name,
            host,
            port,
            n_ctx,
            _gpu_memory_utilization(),
            self.shard_metadata.model_card.trust_remote_code,
            spec_method=(
                card_runtime.vllm_spec_method if card_runtime is not None else None
            ),
            spec_num_tokens=(
                card_runtime.vllm_spec_num_tokens
                if card_runtime is not None
                else None
            ),
            spec_draft_repo=(
                card_runtime.vllm_spec_draft_repo
                if card_runtime is not None
                else None
            ),
            spec_draft_revision=(
                card_runtime.vllm_spec_draft_revision
                if card_runtime is not None
                else None
            ),
            tool_call_parser=self._tool_call_parser,
            reasoning_parser=resolve_vllm_reasoning_parser(
                self.shard_metadata.model_card
            ),
        )
        # Deterministic log path keyed by runner_id (matching llama_server), so
        # postmortem debugging is easy and restarts truncate rather than pile up
        # random temp files.
        self.server_log_path = (
            Path(tempfile.gettempdir()) / f"skulk-vllm-serve-{self.runner_id}.log"
        )
        self.server_log = open(self.server_log_path, "wb")  # noqa: SIM115
        logger.info(f"spawning vllm serve: {' '.join(args)} (log={self.server_log_path})")
        # The vllm CLI lives in its own venv; putting that venv's bin dir at
        # the front of the child's PATH lets tools installed alongside it
        # resolve (vLLM >= 0.24 shells out to `ninja` for FlashInfer JIT
        # sampling kernels and dies at engine init when it is absent).
        server_env = dict(os.environ)
        server_env["PATH"] = (
            f"{Path(binary).parent}{os.pathsep}{server_env.get('PATH', '')}"
        )
        # start_new_session puts vllm serve at the head of its own process
        # group, so teardown can signal the WHOLE group: vLLM spawns its
        # EngineCore as a grandchild, and terminating only the direct child
        # left the engine core alive holding the full GPU allocation when the
        # parent died mid-teardown (#653). PDEATHSIG still covers the direct
        # child on runner death; the worker-startup orphan sweep covers the
        # crash shapes neither reaches.
        self.server_proc = subprocess.Popen(
            args,
            stdout=self.server_log,
            stderr=subprocess.STDOUT,
            env=server_env,
            preexec_fn=_set_pdeathsig if os.name == "posix" else None,
            start_new_session=True,
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
        raise RuntimeError("could not find a free port for vllm serve")

    def _await_health(self) -> None:
        assert self.server_proc is not None and self.base_url is not None
        deadline = time.time() + _HEALTH_DEADLINE_S
        with httpx.Client(timeout=5.0) as client:
            while time.time() < deadline:
                if self.server_proc.poll() is not None:
                    raise RuntimeError(
                        "vllm serve exited during startup (code "
                        f"{self.server_proc.returncode}); log tail:\n"
                        f"{self._server_log_tail()}"
                    )
                try:
                    # vLLM's /health returns 200 with an empty body once the engine
                    # is up (no JSON status field, unlike llama-server).
                    if client.get(f"{self.base_url}/health").status_code == 200:
                        return
                except Exception:  # noqa: BLE001 - not up yet; keep polling
                    pass
                time.sleep(2)
        raise RuntimeError(
            f"vllm serve did not become healthy within {_HEALTH_DEADLINE_S:.0f}s; "
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
                    # Signal the process GROUP (the server was started with
                    # start_new_session, so its pgid is its pid): vLLM's
                    # EngineCore is a grandchild and a plain terminate() left
                    # it orphaned with the GPU allocation when teardown raced
                    # a node restart (#653). Fall back to the single-process
                    # path if group signalling is unavailable.
                    self._signal_server_group(proc, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._signal_server_group(proc, signal.SIGKILL)
                        proc.wait(timeout=5)
                # The leader being gone (whether it exited just now or died
                # BEFORE teardown ran: a startup crash observed by
                # _await_health, an unexpected server exit) does not prove
                # its descendants did: an EngineCore that outlived the
                # leader is already init-reparented and holding VRAM. A
                # blanket killpg here would race pgid reuse once the group
                # is empty (PR #656 review), so the mop is the orphan sweep
                # on EVERY teardown path: it kills only per-pid re-verified,
                # marker-matched engine cores.
                sweep_orphaned_vllm_engines()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
            self.server_proc = None
        if self.server_log is not None:
            with contextlib.suppress(Exception):
                self.server_log.close()
            self.server_log = None

    @staticmethod
    def _signal_server_group(proc: "subprocess.Popen[bytes]", sig: int) -> None:
        """Send ``sig`` to the server's process group, or just the server.

        The group covers vLLM's EngineCore grandchild (#653); the
        single-process fallback keeps teardown working if the group is
        already gone or the platform lacks ``killpg``.
        """
        try:
            os.killpg(proc.pid, sig)
        except (OSError, AttributeError):
            proc.send_signal(sig)

    # --- generation: proxy the server's OpenAI streaming API ------------------

    def _generate(self, task: Task) -> None:
        # Runs on a pool worker thread. Runner status (Running/Ready) is owned by
        # the dispatch loop's in-flight counter, not this per-request path, so a
        # finishing generation never flips the runner to Ready while others run.
        # The task was already acknowledged at acceptance in the dispatch loop
        # (before backpressure), so it is not re-acknowledged here.
        assert isinstance(task, TextGeneration)
        assert self.base_url is not None

        model_id = self.shard_metadata.model_card.model_id
        command_id = task.command_id
        body: dict[str, Any] = vllm_generation_kwargs(task.task_params)
        # vLLM's OpenAI server requires `model` in the request body (unlike
        # llama-server, which serves one model and ignores it); it must match the
        # server's --served-model-name, which the runner sets to the Skulk model id.
        body["model"] = str(model_id)
        body["messages"] = messages_for_llama(task.task_params)
        # Forward thinking control (enable_thinking / reasoning_effort) to vLLM;
        # without it a reasoning model thinks on every request regardless of the
        # request's toggle.
        body.update(
            vllm_reasoning_overrides(task.task_params, self.shard_metadata.model_card)
        )

        record_runner_phase(
            "task_submission",
            event="submit_text_generation",
            task_id=task.task_id,
            command_id=str(command_id),
            attrs={"tools": bool(task.task_params.tools)},
        )
        try:
            # Per-token logprobs remain out of scope for this slice: the SSE
            # proxy does not surface them. Fail loud rather than silently
            # drop them (the #385 no-silent-empty contract).
            if task.task_params.tools and self._tool_call_parser is None:
                raise RuntimeError(
                    "This model's card declares no vLLM tool-call parser "
                    "(runtime.vllm_tool_call_parser or a family default), so "
                    "the server was launched without tool support. Retry "
                    "without tools or serve a tool-capable card."
                )
            if wants_logprobs(
                task.task_params.logprobs, task.task_params.top_logprobs
            ):
                body.pop("logprobs", None)
                body.pop("top_logprobs", None)
                raise RuntimeError(
                    "Per-token logprobs are not supported on the vllm engine: the "
                    "OpenAI SSE proxy does not surface them. Retry without "
                    "logprobs/top_logprobs."
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
            logger.opt(exception=exc).warning("vllm generation failed")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=command_id,
                    chunk=ErrorChunk(model=model_id, error_message=str(exc)),
                )
            )
        else:
            # Read the shared cancel set through the lock-guarded helper: generations
            # run concurrently, so an unlocked membership read here races the pool
            # workers mutating cancelled_tasks.
            was_cancelled = self._was_cancelled(task.task_id)
            record_runner_phase(
                "cancel_observed" if was_cancelled else "completion",
                event="generation_finished",
                task_id=task.task_id,
                command_id=str(command_id),
            )
        # Status is NOT flipped here: the dispatch loop returns the runner to
        # Ready only when the LAST in-flight generation drains (see
        # _note_generation_finished), so a peer generation still streaming keeps
        # the runner Running.

    def _generate_with_tools(
        self,
        task: TextGeneration,
        body: dict[str, Any],
        model_id: ModelId,
        command_id: CommandId,
    ) -> None:
        """Non-streamed tool round trip, mirroring the llama_server runner.

        vLLM parses the model's native tool-call format server-side
        (--enable-auto-tool-choice + --tool-call-parser at launch) and
        returns assembled ``tool_calls``; the caller wants the whole call,
        and the API's streaming adapter emits tool calls as one delta
        anyway, so nothing is lost by skipping SSE here.
        """
        body["stream"] = False
        body["tools"] = task.task_params.tools
        if task.task_params.tool_choice is not None:
            body["tool_choice"] = task.task_params.tool_choice
        assert self.base_url is not None
        if self._is_cancelled(task.task_id):
            return
        admission_in_flight = self._admission_concurrency(task.task_id)
        timeout = httpx.Timeout(connect=15.0, read=None, write=30.0, pool=None)
        request_started = time.perf_counter()
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{self.base_url}/v1/chat/completions", json=body)
            resp.raise_for_status()
            result = cast("dict[str, Any]", resp.json())
        request_seconds = time.perf_counter() - request_started
        # A cancel that landed while the blocking POST was in flight: drain it
        # (the streaming path checks per chunk; this path has no mid-flight
        # checkpoint) so the task ends Cancelled and no tool call surfaces.
        if self._is_cancelled(task.task_id):
            logger.info(f"vllm tool generation cancelled: {task.task_id}")
            return
        choice = cast("dict[str, Any]", (result.get("choices") or [{}])[0])
        message = cast("dict[str, Any]", choice.get("message") or {})
        # vLLM responses carry OpenAI usage but no llama-server-style engine
        # timings, so usage-derived whole-request wall rates are the honest
        # stats here; cached-prefix subtraction does not apply to the
        # blocking path (no prompt_tokens_details outside include_usage).
        stats = blocking_call_stats(result.get("usage"), request_seconds, None)
        if stats is not None:
            stats = stats.model_copy(
                update={"peak_memory_usage": self._server_peak_memory()}
            )
            # Runner attribution (#596): tool-call generations feed the
            # performance envelope exactly like the streaming path.
            stats = self.stamp_runner_stats(stats, admission_in_flight)
        raw_finish = choice.get("finish_reason")
        tool_calls = tool_calls_from_message(message)
        if tool_calls and tool_call_finish_surfaces(raw_finish):
            self.event_sender.send(
                ChunkGenerated(
                    command_id=command_id,
                    chunk=ToolCallChunk(
                        model=model_id,
                        tool_calls=tool_calls,
                        usage=None,
                        stats=stats,
                    ),
                )
            )
            return
        # Prose answer, or a tool attempt cut short by length/content_filter:
        # a truncated tool call has incomplete arguments and must NOT surface
        # as an executable call, so those fall through here and the terminal
        # chunk carries the server's real finish reason. content_filter is
        # preserved exactly like the streaming parser does; everything else
        # maps through the shared finish mapping.
        reasoning = str(message.get("reasoning_content") or "")
        content = str(message.get("content") or "")
        if reasoning:
            self._send_token(command_id, model_id, reasoning, is_thinking=True)
        if content:
            self._send_token(command_id, model_id, content)
        finish = (
            "content_filter"
            if raw_finish == "content_filter"
            else map_finish_reason(raw_finish)
        ) or "stop"
        self._send_token(command_id, model_id, "", finish_reason=finish, stats=stats)

    def _generate_streaming(
        self,
        task: TextGeneration,
        body: dict[str, Any],
        model_id: ModelId,
        command_id: CommandId,
    ) -> None:
        body["stream"] = True
        # The final pre-[DONE] chunk then carries engine-exact token counts
        # (empty choices + usage), the only place the proxy can learn the
        # prompt size (#631).
        body["stream_options"] = {"include_usage": True}
        assert self.base_url is not None
        clock = StreamStatsClock()
        # In-flight captured at THIS task's admission on the dispatch loop, for
        # the performance-envelope tap (#596): the runner's own count is the true
        # per-instance concurrency (vLLM's continuous batching decodes these
        # together). Read from the admission capture, not live here on the worker
        # thread, so a burst does not collapse every sample into one bucket.
        admission_in_flight = self._admission_concurrency(task.task_id)

        last_usage: dict[str, Any] | None = None

        def final_stats() -> GenerationStats:
            # Peak memory always comes from the server child (weights + KV live
            # there), never this proxy. Token counts come from the
            # include_usage final chunk when it arrived (engine-exact); a
            # stream that ended without one falls back to prompt 0
            # ("unmeasured") and the piece count. The prompt RATE covers only
            # newly processed tokens so a prefix-cache hit cannot inflate it,
            # while the reported prompt COUNT stays the request's true size
            # (the same cache-hit rule as the llama_server path, #611/#631).
            prompt_total = _usage_count(last_usage, "prompt_tokens") or 0
            # Explicit None check: an engine-exact completion count of 0 is a
            # real measurement and must not fall back to the piece count.
            usage_generation = _usage_count(last_usage, "completion_tokens")
            generation = (
                usage_generation if usage_generation is not None else clock.pieces
            )
            processed_prompt = prompt_total
            if last_usage is not None:
                details = last_usage.get("prompt_tokens_details")
                if isinstance(details, dict):
                    cached = details.get("cached_tokens")
                    if (
                        isinstance(cached, int)
                        and not isinstance(cached, bool)
                        and 0 < cached <= prompt_total
                    ):
                        processed_prompt = prompt_total - cached
            base = clock.stats(
                prompt_tokens=processed_prompt, generation_tokens=generation
            ).model_copy(
                update={
                    "prompt_tokens": prompt_total,
                    "peak_memory_usage": self._server_peak_memory(),
                }
            )
            return self.stamp_runner_stats(base, admission_in_flight)

        finish_reason: Literal["stop", "length", "content_filter"] | None = None
        # With no tools offered the server's tool parser never runs, so a model
        # writing a call anyway would leak dialect markers as content (#889).
        # Same invariant the MLX path enforces with emit_calls=False.
        scrub = (
            StreamingScaffoldingScrub() if not task.task_params.tools else None
        )
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
                    logger.info(f"vllm generation cancelled: {task.task_id}")
                    break
                delta = parse_openai_sse_line(line)
                if delta is None:
                    continue
                if delta.done:
                    break
                if delta.usage is not None:
                    last_usage = delta.usage
                if delta.reasoning or delta.content:
                    # One SSE delta per generated token piece.
                    clock.mark_piece()
                if delta.reasoning:
                    self._send_token(
                        command_id, model_id, delta.reasoning, is_thinking=True
                    )
                if delta.content:
                    emit = (
                        scrub.feed(delta.content)
                        if scrub is not None
                        else delta.content
                    )
                    if emit:
                        self._send_token(command_id, model_id, emit)
                if delta.finish is not None:
                    # The terminal chunk is deferred past the loop: the
                    # include_usage counts arrive AFTER the finish_reason
                    # chunk, and stats must include them (#631).
                    finish_reason = delta.finish
        # One guaranteed terminal chunk (with final stats) once the stream is
        # fully drained, whether or not the server sent an explicit
        # finish_reason.
        if not self._is_cancelled(task.task_id):
            if scrub is not None:
                tail = scrub.flush()
                if tail:
                    self._send_token(command_id, model_id, tail)
            self._send_token(
                command_id,
                model_id,
                "",
                finish_reason=finish_reason or "stop",
                stats=final_stats(),
            )

    def _server_peak_memory(self) -> Memory:
        """Peak RSS of the ``vllm serve`` child, or zero when unmeasurable.

        The weights and KV cache live in the external server process, so the
        proxy's own RSS would misattribute memory in telemetry. Zero means
        "unmeasured" (non-Linux, or the child already exited).
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
