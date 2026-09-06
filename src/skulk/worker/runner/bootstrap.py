import contextlib
import gc
import os
import resource
import signal
import threading
import time
from collections.abc import Callable, Iterator

import loguru

from skulk.shared.constants import preferred_env_value
from skulk.shared.models.capabilities import (
    family_predates_in_process_llama_cpp,
    resolve_model_capability_profile,
)
from skulk.shared.models.model_cards import (
    RuntimeCapabilityCardConfig,
    card_serves_speech,
)
from skulk.shared.models.remote_code_approval import require_remote_code_approval
from skulk.shared.types.audio import RealtimeAudioInputFrame
from skulk.shared.types.diagnostics import (
    RunnerDiagnosticContext,
    RunnerDiagnosticUpdate,
)
from skulk.shared.types.events import Event, RunnerStatusUpdated
from skulk.shared.types.tasks import Task, TaskId
from skulk.shared.types.worker.instances import BoundInstance
from skulk.shared.types.worker.runners import RunnerFailed
from skulk.shared.types.worker.shards import RpcDonorShardMetadata
from skulk.utils.channels import ClosedResourceError, MpReceiver, MpSender
from skulk.worker.runner.diagnostics import (
    configure_runner_diagnostics,
    record_runner_phase,
)

logger: "loguru.Logger" = loguru.logger

# Set by signal handler so the layer-by-layer loading loop can bail early.
shutdown_requested: bool = False


FAST_SYNCH_CLUSTER_DEFAULT: bool = False
"""Cluster-wide default for ``MLX_METAL_FAST_SYNCH`` when neither the operator
nor the model card has expressed a preference.

OFF as of 2026-06-10. The old True default ("backwards compatibility") had no
measured upside and a catastrophic failure mode for uncarded models:

- Vanilla dense decode is unaffected by the flag (measured 2026-06-06,
  Qwen3.5-9B-4bit on M4: 20.8 tok/s off vs 20.7 on).
- Speculative-decoding loops collapse 46x under the flag (same measurement;
  the spec-card rule below already forced those off).
- Hybrid-SSM models WEDGE at warmup under the flag: gpt-oss hit the 300s
  warmup deadline (#236, card-pinned off 2026-06-07) and NemotronH-9B did
  the same (#259, 2026-06-10 — off, warmup completes in seconds and decode
  runs 19+ tok/s; on, the deadline kill additionally leaks ~5GB of wired
  GPU memory and degrades the node until reboot).

Every model that benefits measurably from FAST_SYNCH can pin
``runtime.metal_fast_synch = true`` on its card; the per-model card pin and
the operator override both outrank this default."""


def _card_declares_speculative_decoding(
    card_runtime: RuntimeCapabilityCardConfig | None,
) -> bool:
    """True when the card declares any speculative-decoding mechanism.

    Covers both the Qwen3/DeepSeek embedded-head convention
    (``mtp_heads`` / ``mtp_sidecar_repo``) and the Gemma 4 companion
    assistant convention (``assistant_model_repo``).
    """
    if card_runtime is None:
        return False
    return bool(
        card_runtime.mtp_heads
        or card_runtime.mtp_sidecar_repo
        or card_runtime.assistant_model_repo
    )


def resolve_metal_fast_synch(card_runtime: RuntimeCapabilityCardConfig | None) -> bool:
    """Resolve the effective ``MLX_METAL_FAST_SYNCH`` setting for this runner.

    Priority order, highest to lowest:

    1. **Operator override.** ``SKULK_FAST_SYNCH`` (or legacy ``SKULK_FAST_SYNCH``)
       set to ``"on"`` or ``"off"``. Set by the CLI ``--fast-synch`` /
       ``--no-fast-synch`` flags or directly in the runner environment.
       This is the escape hatch when a card preference is wrong in the field.
    2. **Model card.** ``runtime.metal_fast_synch`` on the bound shard's model
       card. Pinned per-model when the model is known to deadlock or
       known to benefit measurably from FAST_SYNCH.
    3. **Speculative-decoding default.** Cards that declare an MTP sidecar,
       embedded MTP heads, or a companion assistant model default to
       ``False``. FAST_SYNCH is catastrophically incompatible with the
       speculative decoding loop: measured 2026-06-06 on Qwen3.5-9B-4bit
       (M4, mlx 0.31.2), the same MTP loop runs 27.7 tok/s with
       ``MLX_METAL_FAST_SYNCH=0`` and 0.6 tok/s with ``=1`` — a 46x
       collapse — while vanilla decode is unaffected (20.8 vs 20.7 tok/s).
       The per-round pattern of small evals across streams that
       speculative decoding requires is exactly the shape FAST_SYNCH's
       completion-signal path pathologizes.
    4. **Cluster default.** ``FAST_SYNCH_CLUSTER_DEFAULT`` (False — see its
       docstring for the 2026-06-10 rationale and measurements).

    Returns ``True`` when ``MLX_METAL_FAST_SYNCH`` should be ``"1"``.
    """
    override = preferred_env_value("SKULK_FAST_SYNCH")
    if override is not None:
        normalized = override.strip().lower()
        if normalized == "on":
            return True
        if normalized == "off":
            return False
        # Anything else (empty string, "auto", garbage) falls through to the
        # next layer rather than silently picking a value. The CLI surface
        # only ever sets "on" or "off", so reaching this branch means
        # someone set the env var by hand to an unknown value.
    if card_runtime is not None and card_runtime.metal_fast_synch is not None:
        return card_runtime.metal_fast_synch
    if _card_declares_speculative_decoding(card_runtime):
        return False
    return FAST_SYNCH_CLUSTER_DEFAULT


WARMUP_DEADLINE_SECONDS_DEFAULT: float = 300.0
"""How long a runner may spend in warmup before it is declared wedged.

Healthy warmups finish in seconds (kernel compile + a 1-token generation);
five minutes is far beyond any legitimate cold start while still bounding
the failure. The canonical wedge (launch smoke, 2026-06-05): a Metal fault
leaves ``mx.eval`` parked in ``IOSurfaceSharedEvent`` forever at 0% CPU —
uninterruptible from Python — and the runner sits in ``RunnerWarmingUp``
silently blocking ALL task dispatch on the node. Override with
``SKULK_WARMUP_DEADLINE_SECONDS``.
"""

WEDGE_EXIT_CODE: int = 86
"""Exit code the deadline watchdog uses for a suspected GPU wedge.

Distinct from generic failures so the supervisor can mark the death as a
wedge (see ``WEDGE_FAILURE_MARKER``) and the worker can refuse to retry:
exiting while the main thread is parked inside a faulted Metal eval does
NOT reliably reclaim wired GPU memory (measured live 2026-06-09 on an M4 —
each wedge-exit left ~5GB wired behind, recoverable only by reboot), so
every relaunch of a wedging model leaks another shard's worth of wired
memory until the node dies.
"""

WEDGE_FAILURE_MARKER: str = "gpu-wedge-deadline"
"""Substring the supervisor embeds in ``RunnerFailed.error_message`` for
wedge-class deaths.

The worker matches on this marker to give the instance up immediately
instead of relaunching. A string marker (rather than a new RunnerStatus
field) keeps the gossiped status type wire-compatible with older nodes
during rolling upgrades.
"""


GROUP_CONNECT_DEADLINE_SECONDS_DEFAULT: float = 120.0
"""Deadline for distributed group formation (``mx.distributed.init``).

The ring backend with ``strict=True`` blocks FOREVER when any of its four
neighbor sockets fails the post-TCP rank-identity handshake — observed live
(#265): a 4-node ring sat with all links ESTABLISHED while every rank looped
connect retries, and the cluster spent 30+ minutes in a probe-timeout/
CancelTask loop with no recovery. Healthy group formation completes in
seconds even over slow paths; 120s is generous. On expiry the runner exits
via the wedge path, the worker gives the instance up on the first failure
(#260), and a fresh placement mints a NEW ephemeral port — which also clears
the stale-socket handshake collisions that same-port retries can hit.
Override with SKULK_GROUP_CONNECT_DEADLINE_SECONDS.
"""


def resolve_group_connect_deadline_seconds() -> float:
    """Resolve the group-connect deadline, honoring the operator override."""
    raw = preferred_env_value(
        "SKULK_GROUP_CONNECT_DEADLINE_SECONDS"
    )
    if raw is not None:
        try:
            parsed = float(raw)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
        logger.warning(
            "Ignoring invalid group-connect deadline override "
            f"(SKULK_GROUP_CONNECT_DEADLINE_SECONDS) value {raw!r}; using "
            f"default {GROUP_CONNECT_DEADLINE_SECONDS_DEFAULT:.0f}s"
        )
    return GROUP_CONNECT_DEADLINE_SECONDS_DEFAULT


GROUP_CONNECT_STALL_DIAGNOSIS: str = (
    "the distributed group never formed (ring init blocks forever when a "
    "neighbor socket fails the rank handshake — stale peer from a prior "
    "attempt, unreachable advertised address, or a dropped link). "
    "Terminating this runner so the instance fails cleanly instead of "
    "looping request timeouts; a fresh placement gets a new ring port."
)
"""Diagnosis for a group-connect stall — a NETWORK condition, not a GPU
wedge, so the watchdog's default Metal guidance would mislead operators."""


def resolve_warmup_deadline_seconds() -> float:
    """Resolve the warmup deadline, honoring the operator env override."""
    raw = preferred_env_value(
        "SKULK_WARMUP_DEADLINE_SECONDS"
    )
    if raw is not None:
        try:
            parsed = float(raw)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
        logger.warning(
            "Ignoring invalid warmup-deadline override "
            f"(SKULK_WARMUP_DEADLINE_SECONDS) "
            f"value {raw!r}; using default "
            f"{WARMUP_DEADLINE_SECONDS_DEFAULT:.0f}s"
        )
    return WARMUP_DEADLINE_SECONDS_DEFAULT


def deadline_message(
    description: str, seconds: float, diagnosis: str | None = None
) -> str:
    """Build the CRITICAL line the deadline watchdog logs before exiting.

    Pure so the operator-facing text is testable (the default timeout action
    itself ends in ``os._exit``). ``diagnosis`` defaults to the GPU-wedge
    guidance; non-Metal call sites (group connect, #265) supply their own so
    the log doesn't send operators chasing the wrong subsystem.
    """
    explanation = (
        diagnosis
        if diagnosis is not None
        else (
            "the GPU may be wedged (a faulted Metal eval blocks forever at "
            "0% CPU). Terminating this runner so the node can keep "
            "dispatching. If runners keep dying here, test the GPU with a "
            "small matmul; if that hangs too, the machine needs a reboot to "
            "reset the Metal device queue."
        )
    )
    return f"{description} exceeded its {seconds:.0f}s deadline — {explanation}"


@contextlib.contextmanager
def deadline_watchdog(
    seconds: float,
    description: str,
    on_timeout: Callable[[], None] | None = None,
    diagnosis: str | None = None,
) -> Iterator[None]:
    """Hard deadline for a block that may wedge uninterruptibly.

    A wedged Metal eval blocks inside the driver at 0% CPU and cannot be
    interrupted by signals or async timeouts — the only reliable escape is
    process exit (the supervisor observes the death and reports
    ``RunnerFailed``; Metal memory is reclaimed on exit). A daemon thread
    waits out the deadline and, if the block has not finished, logs a
    CRITICAL diagnosis and terminates the process via ``os._exit``.

    Args:
        seconds: Deadline for the wrapped block.
        description: Human-readable name of the operation (appears in the
            CRITICAL log line).
        on_timeout: Test seam — replaces the default log-and-``os._exit``
            action when provided.
        diagnosis: Operator-facing explanation appended to the CRITICAL log
            line. Defaults to the GPU-wedge guidance; non-Metal call sites
            (group connect, #265) supply their own so the log doesn't send
            operators chasing the wrong subsystem.
    """
    finished = threading.Event()

    def _default_timeout_action() -> None:
        logger.critical(deadline_message(description, seconds, diagnosis))
        # Deliberately NO _release_metal_resources() here: mx.clear_cache()
        # would touch the very Metal device this watchdog assumes is wedged
        # and could block before the exit.
        #
        # WARNING: unlike every other crash class (verified 2026-06-05),
        # exiting out of a faulted Metal eval does NOT reliably reclaim wired
        # GPU memory — measured live 2026-06-09: each wedge-exit left ~5GB
        # wired behind, recoverable only by reboot. The distinct exit code
        # lets the supervisor mark this as a wedge so the worker gives the
        # instance up instead of relaunching into another leak.
        os._exit(WEDGE_EXIT_CODE)

    action = on_timeout if on_timeout is not None else _default_timeout_action

    def _watch() -> None:
        if not finished.wait(seconds):
            action()

    watcher = threading.Thread(
        target=_watch, name="runner-deadline-watchdog", daemon=True
    )
    watcher.start()
    try:
        yield
    finally:
        finished.set()


def _release_metal_resources() -> None:
    """Best-effort release of Metal/MLX resources before process exit.

    Clears the MLX buffer cache and runs garbage collection so that Metal
    wired memory is returned to the OS instead of leaking when the runner
    subprocess is terminated mid-load.
    """
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
    gc.collect()


def _install_parent_death_watchdog(
    initial_ppid: int, poll_seconds: float = 1.0
) -> None:
    """Self-terminate the runner when the agent supervisor dies.

    The runner is a ``mp.Process(daemon=True)`` child of the agent. ``daemon=True``
    only kills the child on a *clean* Python interpreter exit in the parent;
    when the agent receives SIGKILL the child is reparented to launchd (pid 1
    on macOS) and continues holding 4-5 GB of unified GPU memory until killed
    explicitly. Without this watchdog, killing the agent leaves wired Metal
    memory allocated, which is the failure mode that today forces a node
    restart.

    A dedicated daemon thread polls ``os.getppid()`` once per second; if it
    changes from the original supervisor pid we run a best-effort
    ``mx.clear_cache()`` and ``os._exit(1)``. We use ``os._exit`` rather than
    ``sys.exit`` so we bypass any Python-level finally/atexit hooks that could
    block on broken multiprocessing pipes.
    """

    def watchdog() -> None:
        while True:
            time.sleep(poll_seconds)
            try:
                current_ppid = os.getppid()
            except OSError:
                continue
            if current_ppid != initial_ppid:
                logger.warning(
                    f"Runner parent died (ppid {initial_ppid} -> {current_ppid}); "
                    "self-terminating to release Metal/GPU memory."
                )
                _release_metal_resources()
                # Bypass interpreter shutdown to avoid hanging on broken
                # multiprocessing pipes back to the now-dead supervisor.
                os._exit(1)

    thread = threading.Thread(
        target=watchdog,
        name="runner-parent-death-watchdog",
        daemon=True,
    )
    thread.start()


def _metal_cleanup_signal_handler(signum: int, _frame: object) -> None:
    """Handle SIGTERM/SIGINT via cooperative cancellation.

    Sets ``shutdown_requested`` so the per-layer load loop in
    ``utils_mlx.py`` can bail early with ``InterruptedError``.  Then
    raises ``InterruptedError`` directly so that code paths *outside*
    the load loop (inference, idle) also terminate promptly.

    ``InterruptedError`` is a subclass of ``Exception``, so it flows
    through the ``except Exception`` block in ``entrypoint()`` which
    reports ``RunnerFailed`` to the supervisor.  Metal cleanup happens
    in the ``finally`` block of ``entrypoint()``.
    """
    global shutdown_requested
    shutdown_requested = True
    logger.info(f"Runner received signal {signum}, requesting cooperative shutdown")
    # Raise instead of sys.exit() so the normal exception path in
    # entrypoint() can report RunnerFailed to the supervisor.
    raise InterruptedError(f"Runner interrupted by signal {signum}")


def _resolve_text_engine(bound_instance: BoundInstance) -> str | None:
    """Resolve which engine serves this (non-image, non-embedding) text model here.

    Prefers the backend the master already resolved for this node and persisted
    on the shard (``resolved_backend``, #330): dispatch is then deterministic
    from replicated state and cannot disagree with the placement decision. Only
    when that is absent (a master that did not record one, or a manual launch off
    the normal placement path) does it fall back to a node-local probe of the
    card's ``compatible_backends`` intersected with this node's advertised
    backends, ordered by ``backend_preference``. Returns the engine, or ``None``
    to fall through to the default MLX runner.
    """
    from skulk.facts import (
        current_backend_derivation,
        current_node_facts,
        engine_build_inventory,
        hardware_class_inventory,
    )
    from skulk.shared.backends import (
        engine_of,
        platform_compatible_backends,
        probe_node_backends,
        resolve_node_engine,
    )
    from skulk.shared.models.model_cards import (
        load_cached_registry_engine_support,
        registry_supported_backends_for_node,
    )
    shard = bound_instance.bound_shard
    require_remote_code_approval(shard.model_card)
    if shard.resolved_backend is not None:
        return engine_of(shard.resolved_backend)

    placement = shard.model_card.placement
    derivation = current_backend_derivation()
    facts = current_node_facts()
    engine_builds = safe_engine_build_inventory(
        lambda: engine_build_inventory(derivation.backends, facts)
    )
    if shard.model_card.registry_card_id is not None:
        load_cached_registry_engine_support()
    compatible_backends = placement.compatible_backends | (
        registry_supported_backends_for_node(
            shard.model_card,
            node_backends=derivation.backends,
            engine_builds=engine_builds,
            hardware_classes=hardware_class_inventory(facts),
        )
    )
    # Same platform-capability filter the master applies at placement: the
    # card declares MODEL truth, and engines whose runner cannot serve one of
    # the card's declared capabilities (e.g. vision without served mmproj
    # support) are subtracted in code so the fallback probe cannot pick one.
    profile = resolve_model_capability_profile(
        shard.model_card.model_id, model_card=shard.model_card
    )
    return resolve_node_engine(
        platform_compatible_backends(
            compatible_backends,
            card_serves_vision=shard.model_card.vision is not None,
            card_serves_speech=card_serves_speech(shard.model_card),
            card_has_pinned_projector=(
                shard.model_card.vision is not None
                and shard.model_card.vision.has_pinned_projector
            ),
            card_supports_tool_calling=profile.supports_tool_calling,
            card_vllm_tool_call_parser=(
                shard.model_card.runtime.vllm_tool_call_parser
                if shard.model_card.runtime is not None
                else None
            ),
            card_family_predates_in_process_binding=(
                family_predates_in_process_llama_cpp(shard.model_card)
            ),
        ),
        placement.backend_preference,
        probe_node_backends(),
    )


def safe_engine_build_inventory(
    inventory_provider: Callable[[], dict[str, str]],
) -> dict[str, str]:
    """Fail signed matrix fallback closed without breaking legacy placement."""
    try:
        return inventory_provider()
    except ValueError as error:
        logger.error(
            "Invalid local engine-build inventory; disabling signed engine-support "
            f"fallback while preserving legacy card compatibility: {error}"
        )
        return {}


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    diagnostic_sender: MpSender[RunnerDiagnosticUpdate],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    realtime_audio_receiver: MpReceiver[RealtimeAudioInputFrame],
    _logger: "loguru.Logger",
    context_token_limit: int | None = None,
) -> None:
    global logger
    logger = _logger

    # Install signal handlers so that SIGTERM/SIGINT from the supervisor
    # triggers Metal cleanup instead of an abrupt death that leaks wired RAM.
    signal.signal(signal.SIGTERM, _metal_cleanup_signal_handler)
    signal.signal(signal.SIGINT, _metal_cleanup_signal_handler)

    # Backstop for SIGKILL of the supervisor: signal handlers above only fire
    # for graceful agent shutdown. If the agent is SIGKILLed we get reparented
    # to launchd, never receive a signal, and orphan ~5 GB of wired Metal
    # memory. The watchdog detects reparenting and self-exits cleanly.
    _install_parent_death_watchdog(initial_ppid=os.getppid())

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    shard = bound_instance.bound_shard
    fast_synch_enabled = resolve_metal_fast_synch(shard.model_card.runtime)
    os.environ["MLX_METAL_FAST_SYNCH"] = "1" if fast_synch_enabled else "0"
    logger.info(
        f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']} "
        f"(model={shard.model_card.model_id})"
    )
    configure_runner_diagnostics(
        diagnostic_sender,
        RunnerDiagnosticContext(
            node_id=str(bound_instance.bound_node_id),
            runner_id=str(bound_instance.bound_runner_id),
            pid=os.getpid(),
            instance_id=str(bound_instance.instance.instance_id),
            model_id=str(shard.model_card.model_id),
            rank=shard.device_rank,
            world_size=shard.world_size,
            start_layer=shard.start_layer,
            end_layer=shard.end_layer,
            n_layers=shard.n_layers,
        ),
    )
    record_runner_phase("created", event="process_started", include_memory=True)

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        # This guard belongs before runner-type dispatch: image and embedding
        # runners do not pass through text-engine resolution. Keep it inside the
        # failure-reporting boundary so a denied card becomes an actionable
        # RunnerFailed state rather than an unreported process-exit retry loop.
        require_remote_code_approval(shard.model_card)
        if bound_instance.is_image_model:
            from skulk.worker.runner.image_models.runner import Runner as ImageRunner

            runner = ImageRunner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()
        elif bound_instance.is_embedding_model:
            from skulk.worker.runner.embeddings.runner import Runner as EmbeddingRunner

            runner = EmbeddingRunner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()
        elif card_serves_speech(bound_instance.bound_shard.model_card):
            resolved_text_engine = _resolve_text_engine(bound_instance)
            if resolved_text_engine != "mlx_audio":
                raise RuntimeError(
                    "Speech model cards require an mlx_audio-capable serving "
                    "node, but this node did not resolve the mlx_audio backend "
                    f"for {bound_instance.bound_shard.model_card.model_id}."
                )
            from skulk.worker.runner.speech.runner import Runner as SpeechRunner

            runner = SpeechRunner(
                bound_instance,
                event_sender,
                task_receiver,
                cancel_receiver,
                realtime_audio_receiver,
            )
            runner.main()
        elif isinstance(bound_instance.bound_shard, RpcDonorShardMetadata):
            # RPC memory donor of a multi-node GGUF placement (#328): serves
            # ggml-rpc-server on the placement-stamped endpoint and lends GPU
            # memory to the driver's llama-server --rpc. Dispatched on the
            # SHARD TYPE, not resolved_backend: the donor's tag also says
            # llama_server-*, but its job is not to run a server app.
            from skulk.worker.runner.rpc_donor.runner import Runner as RpcDonorRunner

            runner = RpcDonorRunner(
                bound_instance,
                event_sender,
                task_receiver,
                cancel_receiver,
                context_token_limit=context_token_limit,
            )
            runner.main()
        else:
            resolved_text_engine = _resolve_text_engine(bound_instance)
            if resolved_text_engine == "llama_cpp":
                # Heterogeneous (non-MLX) text generation via in-process llama.cpp.
                # Selected when the model card's compatible backends resolve to the
                # llama_cpp engine on this node (e.g. a GGUF model on a GPU node that
                # advertises llama_cpp-vulkan / llama_cpp-rocm). Single-node only.
                from skulk.worker.runner.llama_cpp.runner import (
                    Runner as LlamaCppRunner,
                )

                runner = LlamaCppRunner(
                    bound_instance,
                    event_sender,
                    task_receiver,
                    cancel_receiver,
                    context_token_limit=context_token_limit,
                )
                runner.main()
            elif resolved_text_engine == "llama_server":
                # Served-backend text generation: the worker launches an external
                # llama-server subprocess and proxies its OpenAI API. Selected when
                # the card's compatible backends resolve to the llama_server engine
                # on this node (SKULK_LLAMA_SERVER_BIN set). The only path to native
                # MTP (--spec-type draft-mtp). Single-node only.
                from skulk.worker.runner.llama_server.runner import (
                    Runner as LlamaServerRunner,
                )

                runner = LlamaServerRunner(
                    bound_instance,
                    event_sender,
                    task_receiver,
                    cancel_receiver,
                    context_token_limit=context_token_limit,
                )
                runner.main()
            elif resolved_text_engine == "vllm":
                # Served-backend text generation via vLLM: the worker launches an
                # external `vllm serve` subprocess and proxies its OpenAI API.
                # Selected when the card's compatible backends resolve to the vllm
                # engine on this node (SKULK_VLLM_BIN set). The GPU-serving fast
                # path (continuous batching); single-node only in this slice.
                from skulk.worker.runner.vllm.runner import Runner as VllmRunner

                runner = VllmRunner(
                    bound_instance,
                    event_sender,
                    task_receiver,
                    cancel_receiver,
                    context_token_limit=context_token_limit,
                )
                runner.main()
            else:
                from skulk.worker.engines.mlx.patches import apply_mlx_patches
                from skulk.worker.runner.llm_inference.runner import Runner

                apply_mlx_patches()

                runner = Runner(
                    bound_instance,
                    event_sender,
                    task_receiver,
                    cancel_receiver,
                    context_token_limit=context_token_limit,
                )
                runner.main()

    except ClosedResourceError:
        # The parent intentionally closes IPC during bounded teardown. Actual
        # runner failures take the exception path below and retain warning-level
        # diagnostics, while a closed parent channel is normal shutdown noise.
        logger.info("Runner communication closed during shutdown")
    except Exception as e:
        record_runner_phase(
            "error",
            event="runner_crashed",
            detail=f"{type(e).__name__}: {e}",
            include_memory=True,
        )
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=bound_instance.bound_runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
    finally:
        # Safety net: release Metal resources on any exit path, even if the
        # signal handler didn't fire (e.g. parent was SIGKILL'd and we got
        # a broken pipe, or an unexpected exception during load).
        record_runner_phase(
            "shutdown_cleanup",
            event="metal_cleanup_begin",
            include_memory=True,
        )
        _release_metal_resources()
        record_runner_phase(
            "shutdown_cleanup",
            event="metal_cleanup_complete",
            include_memory=True,
        )
        try:
            event_sender.close()
            task_receiver.close()
            diagnostic_sender.close()
        finally:
            event_sender.join()
            task_receiver.join()
            # Diagnostics are best-effort and must never delay Metal cleanup.
            # Closing is enough; do not wait for a full diagnostic queue to drain.
            logger.info("bye from the runner")
