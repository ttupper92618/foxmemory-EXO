"""The steward pins thinking off on every generation it dispatches.

Bench measurement, not taste: ranking every candidate brain with thinking
disabled and then re-running the two finalists with it enabled made both
WORSE on the trust axes and better on nothing. The harness therefore sends
``enable_thinking=False`` on both the investigation turns and the liveness
canary, and these tests assert it survives the capability boundary as a real
disabled-thinking task rather than being dropped on the way to the runner.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from skulk.api.steward import (
    STEWARD_THINKING_ENABLED,
    StewardChatMessage,
    StewardHarness,
)
from skulk.extensions.steward import StewardToolBinding
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    ReasoningCardConfig,
)
from skulk.shared.types.chunks import TokenChunk
from skulk.shared.types.common import CommandId
from skulk.shared.types.memory import Memory
from skulk.shared.types.text_generation import TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from skulk.api.main import API

_STEWARD_MODEL = "org/steward-brain"


def _thinking_toggle_card() -> ModelCard:
    """A card shaped like the steward brains: thinking present, toggleable."""
    return ModelCard(
        model_id=ModelId(_STEWARD_MODEL),
        storage_size=Memory.from_gb(20),
        n_layers=40,
        hidden_size=2048,
        num_key_value_heads=2,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        capabilities=["text", "thinking", "thinking_toggle"],
        reasoning=ReasoningCardConfig(
            supports_toggle=True,
            default_effort="medium",
            disabled_effort="none",
        ),
    )


class _CapturingApi:
    """Records the task params the harness dispatches."""

    def __init__(self) -> None:
        self.dispatched: list[TextGenerationTaskParams] = []
        self.extension_taps: list[bool] = []

    async def running_model_card(self, model_id: ModelId) -> ModelCard:
        assert str(model_id) == _STEWARD_MODEL
        return _thinking_toggle_card()

    async def steward_extension_tools(
        self, *, proposals_allowed: bool
    ) -> tuple[StewardToolBinding, ...]:
        """Keep this reasoning fixture independent of installed adapter tools."""
        return ()

    async def dispatch_text_generation(
        self,
        task_params: TextGenerationTaskParams,
        target_instance_id: InstanceId | None = None,
    ) -> object:
        self.dispatched.append(task_params)
        return SimpleNamespace(command_id=CommandId())

    def text_generation_chunk_stream(
        self,
        command: object,
        task_params: TextGenerationTaskParams,
        *,
        extension_tap: bool = True,
    ) -> "AsyncGenerator[TokenChunk, None]":
        self.extension_taps.append(extension_tap)

        async def _stream() -> "AsyncGenerator[TokenChunk, None]":
            yield TokenChunk(
                model=ModelId(_STEWARD_MODEL),
                text="All healthy.",
                token_id=-1,
                usage=None,
                finish_reason="stop",
            )

        return _stream()


async def test_investigation_turns_dispatch_with_thinking_disabled() -> None:
    api = _CapturingApi()
    harness = StewardHarness(cast("API", cast(object, api)))
    harness.steward_instance = lambda: (InstanceId(), _STEWARD_MODEL)

    async for _chunk in harness.run_turn_chunks(
        [StewardChatMessage(role="user", content="is the fleet ok?")]
    ):
        pass

    assert api.dispatched, "the harness dispatched no generation"
    for task_params in api.dispatched:
        assert task_params.enable_thinking is False
        # The capability boundary must also normalize the effort, or a
        # served engine would still be handed a thinking budget.
        assert task_params.reasoning_effort == "none"


async def test_canary_probe_dispatches_with_thinking_disabled() -> None:
    api = _CapturingApi()
    harness = StewardHarness(cast("API", cast(object, api)))

    assert await harness.canary_probe(InstanceId(), _STEWARD_MODEL) is True
    assert len(api.dispatched) == 1
    assert api.dispatched[0].enable_thinking is False


async def test_inner_generations_withhold_the_extension_tap() -> None:
    """Investigation steps and the canary are not turns of their own.

    Both run through ``API.text_generation_chunk_stream``, which applies the
    extension chat-summary tap by default. Left on, an ambient-memory or
    audit observer would fire once per investigation step (recording the
    steward's internal tool traffic as conversation) plus once for the turn
    the caller taps, and once per liveness probe. The single correct tap is
    applied by ``API._steward_chat_completions`` around the whole turn.
    """
    api = _CapturingApi()
    harness = StewardHarness(cast("API", cast(object, api)))
    harness.steward_instance = lambda: (InstanceId(), _STEWARD_MODEL)

    async for _chunk in harness.run_turn_chunks(
        [StewardChatMessage(role="user", content="is the fleet ok?")]
    ):
        pass
    assert api.extension_taps and not any(api.extension_taps)

    canary = _CapturingApi()
    probe = StewardHarness(cast("API", cast(object, canary)))
    assert await probe.canary_probe(InstanceId(), _STEWARD_MODEL) is True
    assert canary.extension_taps == [False]


def test_thinking_stays_off_until_a_measurement_says_otherwise() -> None:
    """Pin the verdict: flipping this constant must be a deliberate change."""
    assert STEWARD_THINKING_ENABLED is False
