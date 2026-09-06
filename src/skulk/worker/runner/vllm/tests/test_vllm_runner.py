# pyright: reportPrivateUsage=false, reportAny=false
"""Unit tests for the pure helpers of the vLLM served-backend runner.

The live subprocess + streaming path is validated on GPU hardware; these cover
the pure, engine-specific logic: the ``vllm serve`` argument builder, the OpenAI
SSE parser, and the GPU-memory-utilization knob.
"""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from skulk.shared.types.common import CommandId
from skulk.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from skulk.shared.types.text_generation import TextGenerationTaskParams
from skulk.shared.types.worker.runners import RunnerReady, RunnerRunning
from skulk.worker.runner.vllm.runner import (
    _DEFAULT_GPU_MEMORY_UTILIZATION,
    _DEFAULT_MAX_CONCURRENT_REQUESTS,
    _GPU_MEMORY_UTILIZATION_ENV,
    _MAX_CONCURRENT_REQUESTS_ENV,
    _gpu_memory_utilization,
    _max_concurrent_requests,
    build_vllm_serve_args,
    parse_openai_sse_line,
    vllm_generation_kwargs,
    vllm_reasoning_overrides,
)
from skulk.worker.runner.vllm.runner import (
    Runner as VllmRunner,
)


def _params(**overrides: object) -> SimpleNamespace:
    base: dict[str, object] = dict(
        max_output_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        stop=None,
        seed=None,
        enable_thinking=None,
        reasoning_effort=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_vllm_generation_kwargs_uses_vllm_parameter_names() -> None:
    kwargs = vllm_generation_kwargs(
        _params(
            max_output_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.1,
            stop=["</s>"],
            seed=7,
        )
    )
    assert kwargs["max_tokens"] == 256
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["top_k"] == 40
    assert kwargs["min_p"] == 0.05
    # vLLM's name, not llama.cpp's repeat_penalty (which vLLM would ignore).
    assert kwargs["repetition_penalty"] == 1.1
    assert "repeat_penalty" not in kwargs
    assert kwargs["stop"] == ["</s>"]
    assert kwargs["seed"] == 7


def test_vllm_generation_kwargs_omits_unset() -> None:
    assert vllm_generation_kwargs(_params()) == {}


def test_vllm_reasoning_overrides_maps_thinking_controls() -> None:
    assert vllm_reasoning_overrides(_params(enable_thinking=False)) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert vllm_reasoning_overrides(_params(reasoning_effort="high")) == {
        "reasoning_effort": "high"
    }
    # "none" effort is not a valid server value; disabling goes via enable_thinking.
    assert vllm_reasoning_overrides(_params(reasoning_effort="none")) == {}
    assert vllm_reasoning_overrides(_params()) == {}


def _serve_args(**overrides: object) -> list[str]:
    kwargs: dict[str, object] = dict(
        binary="/opt/vllm/bin/vllm",
        model_dir=Path("/models/org--repo"),
        served_model_name="org/repo",
        host="127.0.0.1",
        port=51234,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
    )
    kwargs.update(overrides)
    return build_vllm_serve_args(**kwargs)  # type: ignore[arg-type]


def test_build_vllm_serve_args_shape() -> None:
    args = _serve_args()
    assert args[0] == "/opt/vllm/bin/vllm"
    assert args[1] == "serve"
    assert args[2] == "/models/org--repo"
    # served-model-name decouples the addressed id from the on-disk path.
    assert args[args.index("--served-model-name") + 1] == "org/repo"
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == "51234"
    assert args[args.index("--max-model-len") + 1] == "8192"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.90"
    # single-node in this slice.
    assert args[args.index("--tensor-parallel-size") + 1] == "1"
    # Required for prompt_tokens_details in the include_usage final chunk;
    # without it the cache-honest prompt rate (#631) never sees cached counts.
    assert "--enable-prompt-tokens-details" in args


def test_build_vllm_serve_args_reasoning_parser() -> None:
    # A card-pinned reasoning parser reaches the server; without one vLLM
    # streams a reasoning model's thinking inline as answer text.
    args = _serve_args(reasoning_parser="muse_glimmer")
    assert args[args.index("--reasoning-parser") + 1] == "muse_glimmer"
    assert "--reasoning-parser" not in _serve_args()


def test_build_vllm_serve_args_trust_remote_code() -> None:
    assert "--trust-remote-code" not in _serve_args(trust_remote_code=False)
    assert "--trust-remote-code" in _serve_args(trust_remote_code=True)


def test_parse_sse_content_delta() -> None:
    line = 'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}'
    delta = parse_openai_sse_line(line)
    assert delta is not None
    assert delta.content == "hello"
    assert delta.reasoning == ""
    assert delta.finish is None
    assert delta.done is False


def test_parse_sse_reasoning_delta() -> None:
    line = 'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}'
    delta = parse_openai_sse_line(line)
    assert delta is not None
    assert delta.reasoning == "think"
    assert delta.content == ""


def test_parse_sse_finish_reason_mapped() -> None:
    line = 'data: {"choices":[{"delta":{"content":""},"finish_reason":"length"}]}'
    delta = parse_openai_sse_line(line)
    assert delta is not None
    assert delta.finish == "length"


def test_parse_sse_preserves_content_filter() -> None:
    # vLLM can emit content_filter; it must not be collapsed to a normal stop.
    line = 'data: {"choices":[{"delta":{"content":""},"finish_reason":"content_filter"}]}'
    delta = parse_openai_sse_line(line)
    assert delta is not None
    assert delta.finish == "content_filter"


def test_parse_sse_done_sentinel() -> None:
    delta = parse_openai_sse_line("data: [DONE]")
    assert delta is not None
    assert delta.done is True


@pytest.mark.parametrize(
    "line",
    [
        "event: ping",  # non-data line
        "data: {not json}",  # malformed json
        'data: {"choices":[]}',  # choice-less payload without usage
        # usage may only ride a well-formed chunk: choices exactly [] (the
        # include_usage final chunk) or a dict-bearing choices list.
        'data: {"usage":{"prompt_tokens":5}}',  # no choices key at all
        'data: {"choices":"x","usage":{"prompt_tokens":5}}',  # malformed choices
        'data: {"choices":[42],"usage":{"prompt_tokens":5}}',  # non-dict choice
        "",  # blank
    ],
)
def test_parse_sse_skips_non_deltas(line: str) -> None:
    assert parse_openai_sse_line(line) is None


def test_parse_sse_usage_final_chunk() -> None:
    # The stream_options include_usage final chunk: empty choices, engine-exact
    # counts (#631). Must parse into a usage-bearing delta, not be skipped.
    line = (
        'data: {"choices":[],"usage":{"prompt_tokens":1000,'
        '"completion_tokens":40,"prompt_tokens_details":{"cached_tokens":990}}}'
    )
    delta = parse_openai_sse_line(line)
    assert delta is not None
    assert delta.usage == {
        "prompt_tokens": 1000,
        "completion_tokens": 40,
        "prompt_tokens_details": {"cached_tokens": 990},
    }
    assert delta.content == "" and delta.finish is None and not delta.done


def test_parse_sse_usage_on_choice_chunk() -> None:
    # Some servers attach usage on the last choice-bearing chunk instead of a
    # separate one; it must ride along with the delta.
    line = (
        'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":7,"completion_tokens":2}}'
    )
    delta = parse_openai_sse_line(line)
    assert delta is not None
    assert delta.content == "hi"
    assert delta.finish == "stop"
    assert delta.usage == {"prompt_tokens": 7, "completion_tokens": 2}


def test_usage_count_bool_and_shape_guards() -> None:
    from skulk.worker.runner.vllm.runner import _usage_count

    assert _usage_count({"prompt_tokens": 1000}, "prompt_tokens") == 1000
    assert _usage_count({"prompt_tokens": True}, "prompt_tokens") is None
    assert _usage_count({"prompt_tokens": -1}, "prompt_tokens") is None
    assert _usage_count({"prompt_tokens": "10"}, "prompt_tokens") is None
    assert _usage_count(None, "prompt_tokens") is None


def test_gpu_memory_utilization_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_GPU_MEMORY_UTILIZATION_ENV, raising=False)
    assert _gpu_memory_utilization() == _DEFAULT_GPU_MEMORY_UTILIZATION


def test_gpu_memory_utilization_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_GPU_MEMORY_UTILIZATION_ENV, "0.75")
    assert _gpu_memory_utilization() == 0.75


@pytest.mark.parametrize("bad", ["nonsense", "0", "1.5", "-0.2"])
def test_gpu_memory_utilization_rejects_bad_values(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    # Unparseable or out-of-(0,1] values fall back to the default rather than
    # passing vLLM a fraction that would fail the server at spawn.
    monkeypatch.setenv(_GPU_MEMORY_UTILIZATION_ENV, bad)
    assert _gpu_memory_utilization() == _DEFAULT_GPU_MEMORY_UTILIZATION


def test_max_concurrent_requests_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_MAX_CONCURRENT_REQUESTS_ENV, raising=False)
    assert _max_concurrent_requests() == _DEFAULT_MAX_CONCURRENT_REQUESTS


def test_max_concurrent_requests_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_MAX_CONCURRENT_REQUESTS_ENV, "8")
    assert _max_concurrent_requests() == 8


@pytest.mark.parametrize("bad", ["nonsense", "0", "-3", "1.5"])
def test_max_concurrent_requests_rejects_bad_values(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    # Unparseable or below-1 values fall back to the default rather than
    # disabling concurrency or crashing the pool at construction.
    monkeypatch.setenv(_MAX_CONCURRENT_REQUESTS_ENV, bad)
    assert _max_concurrent_requests() == _DEFAULT_MAX_CONCURRENT_REQUESTS


# --- concurrent dispatch --------------------------------------------------
#
# These exercise the runner's dispatch orchestration (in-flight counting, status
# transitions, terminal-status classification, stale-cancel recovery) with a fake
# ``_generate`` -- no server, no GPU. The live streaming path is validated on
# hardware. The runner is built with ``__new__`` so only the dispatch state is
# set up; ``update_status`` / ``send_task_status`` are stubbed to record calls
# (and keep ``current_status`` in sync) rather than construct wire events.


def _fake_task() -> Any:
    """A duck-typed stand-in for a TextGeneration (only ids are read here)."""
    return SimpleNamespace(task_id=TaskId(), command_id=CommandId())


def _gen_params() -> "TextGenerationTaskParams":
    """A minimal valid TextGenerationTaskParams for main-loop tests."""
    from skulk.shared.types.common import ModelId

    return TextGenerationTaskParams(model=ModelId("m"), input=[])


def _bare_runner(max_concurrency: int = 4) -> Any:
    runner: Any = VllmRunner.__new__(VllmRunner)
    # The concurrent-dispatch state (locks, in-flight counter, permit semaphore,
    # thread-name prefix) is owned by ServedConcurrentDispatch; initialize it the
    # same way the real __init__ does so these tests drive the shared loop.
    runner._init_concurrent_dispatch(max_concurrency, "vllm-gen")
    runner.cancelled_tasks = set()
    runner.seen = set()
    runner.current_status = RunnerReady()
    runner.status_updates = []
    runner.task_statuses = []

    def _record_status(status: Any) -> None:
        runner.status_updates.append(status)
        runner.current_status = status

    def _record_task_status(task: Any, status: Any) -> None:
        runner.task_statuses.append((task.task_id, status))

    runner.update_status = _record_status
    runner.send_task_status = _record_task_status
    return runner


def test_dispatch_runs_generations_concurrently() -> None:
    runner = _bare_runner(max_concurrency=4)
    started = threading.Semaphore(0)
    release = threading.Event()

    def fake_generate(_task: Any) -> None:
        started.release()  # signal this generation has started
        release.wait(5)  # hold it open until the test releases all at once

    runner._generate = fake_generate
    tasks = [_fake_task() for _ in range(3)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        for task in tasks:
            # Must return immediately: dispatch is non-blocking so the loop can
            # keep receiving while prior generations stream.
            runner._dispatch_generation(task, pool)
        # All three run at once (proving they are not serialized).
        for _ in range(3):
            assert started.acquire(timeout=5)
        assert runner._inflight_count() == 3
        assert isinstance(runner.current_status, RunnerRunning)
        release.set()
        pool.shutdown(wait=True)

    # Done-callbacks (terminal status + inflight decrement) run on the worker
    # threads as their futures resolve; all have run by the time shutdown returns.
    assert runner._inflight_count() == 0
    assert isinstance(runner.current_status, RunnerReady)
    running = [s for _, s in runner.task_statuses if s is TaskStatus.Running]
    complete = [s for _, s in runner.task_statuses if s is TaskStatus.Complete]
    assert len(running) == 3
    assert len(complete) == 3


def test_finish_generation_marks_cancelled_and_returns_ready() -> None:
    runner = _bare_runner()
    runner._inflight = 1
    runner.current_status = RunnerRunning()
    task = _fake_task()
    runner.cancelled_tasks.add(task.task_id)
    future: Future[None] = Future()
    future.set_result(None)

    runner._finish_generation(task, future)

    assert (task.task_id, TaskStatus.Cancelled) in runner.task_statuses
    assert runner._inflight == 0
    assert isinstance(runner.current_status, RunnerReady)


def test_finish_generation_cancel_all_marks_cancelled() -> None:
    runner = _bare_runner()
    runner._inflight = 1
    runner.current_status = RunnerRunning()
    runner.cancelled_tasks.add(CANCEL_ALL_TASKS)
    task = _fake_task()
    future: Future[None] = Future()
    future.set_result(None)

    runner._finish_generation(task, future)

    assert (task.task_id, TaskStatus.Cancelled) in runner.task_statuses


def test_idle_admission_clears_stale_cancel_all_before_dispatch() -> None:
    # A lingering cluster-wide cancel must not kill a fresh request admitted when
    # nothing else is in flight.
    runner = _bare_runner()
    runner.cancelled_tasks.add(CANCEL_ALL_TASKS)
    release = threading.Event()

    def fake_generate(_task: Any) -> None:
        release.wait(5)

    runner._generate = fake_generate
    task = _fake_task()

    with ThreadPoolExecutor(max_workers=1) as pool:
        runner._clear_stale_cancel_all_if_idle()
        runner._dispatch_generation(task, pool)
        assert CANCEL_ALL_TASKS not in runner.cancelled_tasks
        release.set()
        pool.shutdown(wait=True)

    assert (task.task_id, TaskStatus.Complete) in runner.task_statuses


def test_note_generation_status_transitions() -> None:
    runner = _bare_runner()
    assert isinstance(runner.current_status, RunnerReady)

    runner._note_generation_started()
    assert runner._inflight == 1
    assert isinstance(runner.current_status, RunnerRunning)

    # A second concurrent generation does not re-flip status.
    runner._note_generation_started()
    assert runner._inflight == 2
    assert isinstance(runner.current_status, RunnerRunning)

    # Only the LAST one to drain returns the runner to Ready.
    runner._note_generation_finished()
    assert isinstance(runner.current_status, RunnerRunning)
    runner._note_generation_finished()
    assert runner._inflight == 0
    assert isinstance(runner.current_status, RunnerReady)


def test_inflight_never_goes_negative() -> None:
    # Defensive: an extra finish (double done-callback) must not drive the count
    # below zero or spuriously toggle status.
    runner = _bare_runner()
    runner._note_generation_finished()
    assert runner._inflight == 0
    assert isinstance(runner.current_status, RunnerReady)


def test_main_broadcasts_ready_after_load_model() -> None:
    # Regression: the concurrent main() must re-broadcast the runner status after
    # a lifecycle task, because _load_model sets current_status = RunnerReady() by
    # DIRECT ASSIGNMENT (no event). Without the broadcast the runner loads but
    # never announces Ready, so the worker never dispatches a generation to it.
    # (Caught live: a probe saw statuses stop at [Idle, Loading] though the server
    # loaded fine.)
    #
    # ORDER is also load-bearing and asserted here: RunnerSupervisor._forward_events
    # asserts the runner is in an active state (Loading/Running/...) when a terminal
    # task status arrives, so the LoadModel Complete must precede the RunnerReady
    # broadcast (Loading -> Complete -> Ready), not follow it. This drives the real
    # main() loop over genuine mp channels.
    from skulk.shared.types.events import (
        Event,
        RunnerStatusUpdated,
        TaskStatusUpdated,
    )
    from skulk.shared.types.tasks import LoadModel, Shutdown, Task, TaskStatus
    from skulk.shared.types.worker.instances import InstanceId
    from skulk.shared.types.worker.runners import RunnerId, RunnerIdle
    from skulk.utils.channels import mp_channel

    runner: Any = VllmRunner.__new__(VllmRunner)
    runner._init_concurrent_dispatch(2, "vllm-gen")
    runner.cancelled_tasks = set()
    runner.seen = set()
    runner.runner_id = RunnerId("vllm-test")
    runner.current_status = RunnerIdle()

    evt_s, evt_r = mp_channel[Event]()
    task_s, task_r = mp_channel[Task]()
    _cancel_s, cancel_r = mp_channel[TaskId]()
    runner.event_sender = evt_s
    runner.task_receiver = task_r
    runner.cancel_receiver = cancel_r

    # Fakes for the lifecycle path: no real vllm serve. _load_model mimics the
    # real one's direct assignment (the exact behavior that made the bug latent).
    def fake_load(_task: Any) -> None:
        runner.current_status = RunnerReady()

    runner._load_model = fake_load
    runner._teardown_server = lambda: None
    runner._ensure_server_alive = lambda: None

    thread = threading.Thread(target=runner.main, daemon=True)
    thread.start()
    try:
        iid = InstanceId("probe-inst")
        task_s.send(LoadModel(instance_id=iid))
        load_complete_at: int | None = None
        ready_at: int | None = None
        seq = 0
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and ready_at is None:
            try:
                event = evt_r.receive_timeout(0.5)
            except Exception:
                continue
            seq += 1
            if (
                isinstance(event, TaskStatusUpdated)
                and event.task_status is TaskStatus.Complete
                and load_complete_at is None
            ):
                load_complete_at = seq
            if isinstance(event, RunnerStatusUpdated) and isinstance(
                event.runner_status, RunnerReady
            ):
                ready_at = seq
        assert ready_at is not None, "runner never broadcast RunnerReady after LoadModel"
        assert load_complete_at is not None, "LoadModel never reported Complete"
        # Ready must come AFTER the terminal Complete (the supervisor assertion).
        assert load_complete_at < ready_at, (
            "RunnerReady was broadcast before the LoadModel Complete; the supervisor "
            "asserts an active runner state on terminal task status"
        )
    finally:
        task_s.send(
            Shutdown(instance_id=InstanceId("probe-inst"), runner_id=runner.runner_id)
        )
        thread.join(timeout=5)


def test_main_backpressure_caps_submitted_generations() -> None:
    # The dispatch loop must not submit an unbounded backlog into the pool's queue:
    # with max_concurrency=2 and 4 generations forwarded at once, only 2 run
    # concurrently; the loop blocks on _dispatch_permits before dispatching the
    # 3rd/4th until a slot frees (backpressure). Drives the real main() loop.
    from skulk.shared.types.events import Event
    from skulk.shared.types.tasks import LoadModel, Shutdown, Task
    from skulk.shared.types.worker.instances import InstanceId
    from skulk.shared.types.worker.runners import RunnerId, RunnerIdle
    from skulk.utils.channels import mp_channel

    runner: Any = VllmRunner.__new__(VllmRunner)
    runner._init_concurrent_dispatch(2, "vllm-gen")
    runner.cancelled_tasks = set()
    runner.seen = set()
    runner.runner_id = RunnerId("vllm-bp")
    runner.current_status = RunnerIdle()

    evt_s, _evt_r = mp_channel[Event]()
    task_s, task_r = mp_channel[Task]()
    _cancel_s, cancel_r = mp_channel[TaskId]()
    runner.event_sender = evt_s
    runner.task_receiver = task_r
    runner.cancel_receiver = cancel_r

    started = threading.Semaphore(0)
    release = threading.Event()
    peak_lock = threading.Lock()
    peak_inflight = 0

    def fake_load(_task: Any) -> None:
        runner.current_status = RunnerReady()

    def fake_generate(_task: Any) -> None:
        nonlocal peak_inflight
        started.release()
        with peak_lock:
            peak_inflight = max(peak_inflight, runner._inflight_count())
        release.wait(5)

    runner._load_model = fake_load
    runner._generate = fake_generate
    runner._teardown_server = lambda: None
    runner._ensure_server_alive = lambda: None

    thread = threading.Thread(target=runner.main, daemon=True)
    thread.start()
    try:
        iid = InstanceId("bp-inst")
        task_s.send(LoadModel(instance_id=iid))
        # Forward 4 generations at once.
        for _ in range(4):
            task_s.send(
                TextGeneration(command_id=CommandId(), task_params=_gen_params(), instance_id=iid)
            )
        # Exactly 2 should start; a 3rd must NOT start while the first 2 are held.
        assert started.acquire(timeout=5)
        assert started.acquire(timeout=5)
        assert not started.acquire(timeout=1), "a 3rd generation ran despite the 2-permit cap"
        assert runner._inflight_count() == 2
        # Release: the remaining 2 now run as slots free.
        release.set()
        assert started.acquire(timeout=5)
        assert started.acquire(timeout=5)
        assert peak_inflight == 2, f"in-flight exceeded the cap: peak={peak_inflight}"
    finally:
        release.set()
        task_s.send(Shutdown(instance_id=InstanceId("bp-inst"), runner_id=runner.runner_id))
        thread.join(timeout=5)


def test_build_vllm_serve_args_speculative_config() -> None:
    # Card-declared MTP speculation maps to vLLM's --speculative-config JSON
    # (probe-validated shape: {"method": "mtp", "num_speculative_tokens": 2}).
    args = _serve_args(spec_method="mtp", spec_num_tokens=2)
    payload = args[args.index("--speculative-config") + 1]
    import json as _json

    assert _json.loads(payload) == {"method": "mtp", "num_speculative_tokens": 2}
    # Method without an explicit depth uses vLLM's default: no key emitted.
    args = _serve_args(spec_method="mtp")
    payload = args[args.index("--speculative-config") + 1]
    assert _json.loads(payload) == {"method": "mtp"}
    # No method: the flag must be absent entirely.
    assert "--speculative-config" not in _serve_args()
    # Draft-model methods (dflash) name the separate speculator repo in the
    # config's "model" key (vendor-published shape for the Laguna cards).
    args = _serve_args(
        spec_method="dflash",
        spec_num_tokens=15,
        spec_draft_repo="poolside/Laguna-XS-2.1-DFlash-FP8",
        spec_draft_revision="a" * 40,
    )
    payload = args[args.index("--speculative-config") + 1]
    assert _json.loads(payload) == {
        "method": "dflash",
        "num_speculative_tokens": 15,
        "model": "poolside/Laguna-XS-2.1-DFlash-FP8",
        "revision": "a" * 40,
    }
    # Deep depths must raise the scheduler's batched-token budget AND pin
    # the sequence cap: the budget constraint is
    # batched >= seqs * (depth - 1), and both sides are vLLM-version- and
    # hardware-band-dependent defaults if left unpinned (0.25.1's effective
    # 2048/256 failed engine init at depth 15 with
    # "max_num_scheduled_tokens is set to -1536"; 0.28.0 defaults seqs as
    # high as 1024, which would sink the raised budget again). 8192 is the
    # fresh-box-validated floor for the Laguna depth-15 card.
    assert args[args.index("--max-num-batched-tokens") + 1] == "8192"
    assert args[args.index("--max-num-seqs") + 1] == "256"
    # Shallow MTP depths keep vLLM's default scheduler sizing (the exact
    # shape the #649 cards validated under): neither flag emitted.
    shallow = _serve_args(spec_method="mtp", spec_num_tokens=2)
    assert "--max-num-batched-tokens" not in shallow
    assert "--max-num-seqs" not in shallow
    # Depths past the validated floor scale linearly rather than re-hitting
    # the same wall (2048 + 256 * 30 = 9728 at depth 31).
    deep = _serve_args(
        spec_method="dflash",
        spec_num_tokens=31,
        spec_draft_repo="poolside/Laguna-XS-2.1-DFlash-FP8",
    )
    assert deep[deep.index("--max-num-batched-tokens") + 1] == "9728"
    assert deep[deep.index("--max-num-seqs") + 1] == "256"


def test_vllm_max_model_len_constant_shared_with_placement() -> None:
    # The cap lives at the placement stamp (memory_estimate.VLLM_MAX_MODEL_LEN)
    # and the runner min()s against the SAME constant as defense in depth, so
    # admission and the served window cannot disagree (PR #649 review).
    from skulk.shared.models.memory_estimate import VLLM_MAX_MODEL_LEN

    assert VLLM_MAX_MODEL_LEN == 32768


def test_tool_call_finish_surfaces_forced_choice_stop() -> None:
    # With a named tool_choice, vLLM reports the forced call under
    # finish_reason "stop" (OpenAI semantics); gating on "tool_calls" alone
    # returned an empty stop chunk to the caller (observed live on 0.28.0).
    from skulk.worker.runner.vllm.runner import tool_call_finish_surfaces

    assert tool_call_finish_surfaces("stop")
    assert tool_call_finish_surfaces("tool_calls")
    assert tool_call_finish_surfaces(None)
    # A call cut short has incomplete arguments and must not surface.
    assert not tool_call_finish_surfaces("length")
    assert not tool_call_finish_surfaces("content_filter")


def _retry_runner(health_outcomes: list[Exception | None]) -> tuple[Any, list[int], list[int]]:
    """Build a runner whose spawn/health pair is scripted, tracking both calls."""
    runner: Any = VllmRunner.__new__(VllmRunner)
    spawns: list[int] = []
    teardowns: list[int] = []
    remaining = list(health_outcomes)

    def fake_spawn(_model_dir: Path, _served: str, _n_ctx: int) -> None:
        spawns.append(len(spawns) + 1)

    def fake_health() -> None:
        outcome = remaining.pop(0)
        if outcome is not None:
            raise outcome

    def fake_teardown() -> None:
        teardowns.append(len(teardowns) + 1)

    runner._spawn_server = fake_spawn
    runner._await_health = fake_health
    runner._teardown_server = fake_teardown
    return runner, spawns, teardowns


def test_lost_port_race_retries_with_a_fresh_port() -> None:
    """A bind-time EADDRINUSE is transient and must not fail the placement.

    _pick_port proves a port free, then vllm serve binds it seconds later; in
    that window the kernel can hand the same ephemeral port to an outbound
    connection on a busy node. Retrying picks a new port.
    """
    collision = RuntimeError(
        "vllm serve exited during startup (code 1); log tail:\n"
        "OSError: [Errno 98] Address already in use"
    )
    runner, spawns, teardowns = _retry_runner([collision, None])

    runner._spawn_server_with_port_retry(Path("/models/m"), "org/m", 8192)

    assert len(spawns) == 2
    # The failed attempt is reclaimed before rebinding so no handles leak.
    assert len(teardowns) == 1


def test_port_race_retries_are_bounded() -> None:
    collision = RuntimeError(
        "vllm serve exited during startup (code 1); log tail:\n"
        "OSError: [Errno 98] Address already in use"
    )
    runner, spawns, _ = _retry_runner([collision, collision, collision])

    with pytest.raises(RuntimeError, match="Address already in use"):
        runner._spawn_server_with_port_retry(Path("/models/m"), "org/m", 8192)

    assert len(spawns) == 3


def test_other_startup_failures_are_not_retried() -> None:
    """A real fault must fail fast rather than burn attempts.

    A CUDA OOM, a missing weight shard, or a bad flag will fail identically on
    a fresh port, so retrying only delays the operator's error by minutes.
    """
    fault = RuntimeError(
        "vllm serve exited during startup (code 1); log tail:\n"
        "torch.AcceleratorError: CUDA error: out of memory"
    )
    runner, spawns, _ = _retry_runner([fault])

    with pytest.raises(RuntimeError, match="out of memory"):
        runner._spawn_server_with_port_retry(Path("/models/m"), "org/m", 8192)

    assert len(spawns) == 1
