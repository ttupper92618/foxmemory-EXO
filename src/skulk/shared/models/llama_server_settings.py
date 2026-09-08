"""Serving settings shared by node advertisement, admission and process launch."""

from collections.abc import Mapping
from typing import Final, final

from pydantic import Field

from skulk.utils.pydantic_ext import FrozenModel

LLAMA_SERVER_DEFAULT_PARALLEL: Final = 16
LLAMA_SERVER_DEFAULT_DRAFT_DEPTH: Final = 3


@final
class LlamaServerSettings(FrozenModel):
    """Node settings that change llama-server's persistent cache allocation."""

    parallel_slots: int = Field(
        default=LLAMA_SERVER_DEFAULT_PARALLEL,
        gt=0,
        description="Operator-selected concurrent slots before model-specific limits.",
    )
    speculation_enabled: bool = Field(
        default=True,
        description="Whether the node permits the model card's speculative mode.",
    )

    def effective_slots(self, *, speculative_vision: bool) -> int:
        """Apply the existing serial limit only to speculative vision serving."""
        return (
            1
            if speculative_vision and self.speculation_enabled
            else self.parallel_slots
        )


def resolve_llama_server_settings(
    environment: Mapping[str, str],
) -> LlamaServerSettings:
    """Resolve existing environment controls without changing their defaults.

    Invalid slot declarations retain the runner's historical default. The runner
    remains responsible for warning the operator about such declarations.
    """
    try:
        slots = int(environment.get("SKULK_LLAMA_SERVER_PARALLEL", "").strip())
    except ValueError:
        slots = LLAMA_SERVER_DEFAULT_PARALLEL
    if slots < 1:
        slots = LLAMA_SERVER_DEFAULT_PARALLEL
    disabled = environment.get("SKULK_LLAMA_SERVER_FORCE_NO_SPEC", "").strip().lower()
    return LlamaServerSettings(
        parallel_slots=slots,
        speculation_enabled=disabled not in ("1", "true", "yes", "on"),
    )
