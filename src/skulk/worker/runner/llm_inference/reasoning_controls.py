"""Card-aware translation of Skulk's thinking controls for the served engines.

The served runners (``llama_server``, ``vllm``) forward ``enable_thinking``
and ``reasoning_effort`` to their servers as generic levers. Families whose
templates read something else need a translation step that knows the card;
this module holds those translations so both runners share one behavior.
"""

from __future__ import annotations

from typing import Final, cast

from skulk.shared.models.capabilities import (
    muse_glimmer_template_kwargs,
    resolve_model_capability_profile,
    uses_muse_glimmer_protocol,
)
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.text_generation import ReasoningEffort
from skulk.worker.runner.bootstrap import logger

_REASONING_EFFORTS: Final[frozenset[str]] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)


def muse_glimmer_strength_kwargs(
    card: ModelCard | None, effort: object
) -> dict[str, str]:
    """``reasoning_strength`` template kwargs when ``card`` is a Muse Glimmer model.

    Empty for every other family, for a missing card, and when no effort was
    requested (the template's own default, ``high``, then applies). A card the
    capability resolver cannot read is treated as not-Muse rather than raised,
    so an odd custom card degrades to the generic levers instead of failing
    the request.
    """
    if card is None:
        return {}
    try:
        profile = resolve_model_capability_profile(card.model_id, model_card=card)
    except Exception as exc:  # noqa: BLE001 - an unreadable card is just not Muse
        logger.opt(exception=exc).warning(
            f"capability resolution failed for {card.model_id}; "
            "forwarding generic thinking controls"
        )
        return {}
    if not uses_muse_glimmer_protocol(profile):
        return {}
    if not isinstance(effort, str) or effort not in _REASONING_EFFORTS:
        return {}
    return muse_glimmer_template_kwargs(profile, cast("ReasoningEffort", effort))
