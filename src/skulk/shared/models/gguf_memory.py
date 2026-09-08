"""Artifact-derived cache geometry for llama.cpp memory admission.

Geometry describes tensors, not a measured total or a model-name heuristic.
Engine configuration supplies slot count and recurrent rollback depth separately.
The Qwen3.5 mapping follows llama.cpp b10753's qwen35 loader, hparams and
recurrent-memory implementation; other recurrent architectures need their own
verified mapping before they can use this estimate.
"""

from collections.abc import Mapping
from typing import Self, final

from pydantic import Field, model_validator

from skulk.utils.pydantic_ext import FrozenModel


@final
class GgufCacheGeometry(FrozenModel):
    """Fixed recurrent state and per-token attention dimensions of one artifact."""

    attention_layers: int = Field(ge=0, description="Target full-attention layers.")
    recurrent_layers: int = Field(ge=0, description="Target recurrent layers.")
    nextn_layers: int = Field(ge=0, description="Embedded MTP attention layers.")
    key_width: int = Field(gt=0, description="Key elements per token per layer.")
    value_width: int = Field(gt=0, description="Value elements per token per layer.")
    convolution_width: int = Field(
        ge=0, description="Convolution-state elements per recurrent layer and row."
    )
    recurrent_width: int = Field(
        ge=0, description="Recurrent-state elements per recurrent layer and row."
    )

    @model_validator(mode="after")
    def validate_recurrent_state(self) -> Self:
        """Reject incomplete geometry instead of treating an unknown cost as zero."""
        if self.attention_layers + self.recurrent_layers <= 0:
            raise ValueError("cache geometry requires target layers")
        if self.recurrent_layers > 0 and self.recurrent_width <= 0:
            raise ValueError("recurrent layers require their state width")
        if self.recurrent_layers == 0 and (
            self.convolution_width or self.recurrent_width
        ):
            raise ValueError("recurrent state requires recurrent layers")
        return self

    def recurrent_bytes(self, *, parallel_slots: int, rollback_depth: int) -> int:
        """Return FP32 recurrent buffers, including each slot's rollback rows.

        llama.cpp allocates both state tensors as FP32, independent of weight
        quantization or attention-cache dtype. Rollback copies multiply rows;
        they do not duplicate the model's weights or all attention caches.
        """
        if parallel_slots < 1 or rollback_depth < 0:
            raise ValueError("invalid recurrent cache configuration")
        return (
            (self.convolution_width + self.recurrent_width)
            * self.recurrent_layers
            * 4
            * parallel_slots
            * (1 + rollback_depth)
        )

    def attention_bytes_per_token(self, *, embedded_mtp: bool) -> int:
        """Return FP16 K/V bytes for target and, when enabled, embedded MTP.

        The MTP context shares target weights but owns only the NextN layers'
        attention cache. Its full context window is charged once, not per slot.
        """
        layers = self.attention_layers + (self.nextn_layers if embedded_mtp else 0)
        return layers * (self.key_width + self.value_width) * 2


def qwen35_cache_geometry(
    architecture: str,
    metadata: Mapping[str, int],
    *,
    has_recurrent_layer_override: bool = False,
) -> GgufCacheGeometry | None:
    """Resolve scalar Qwen3.5 GGUF dimensions, or return unknown for other layouts.

    Keys are architecture-relative GGUF names. An explicit recurrent-layer
    vector supersedes the interval in llama.cpp; until that vector is decoded,
    its presence must never be mistaken for the regular interval layout.
    Missing dimensions remain unknown and malformed complete dimensions raise.
    """
    if architecture != "qwen35" or has_recurrent_layer_override:
        return None
    required = (
        "block_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "ssm.conv_kernel",
        "ssm.inner_size",
        "ssm.state_size",
        "ssm.group_count",
    )
    if any(key not in metadata for key in required):
        return None
    if any(metadata[key] <= 0 for key in required):
        raise ValueError("GGUF cache dimensions must be positive")
    nextn = metadata.get("nextn_predict_layers", 0)
    layers = metadata["block_count"] - nextn
    interval = metadata.get("full_attention_interval", 4)
    if nextn < 0 or layers <= 0 or interval <= 0:
        raise ValueError("invalid GGUF attention layer layout")
    attention_layers = layers // interval
    recurrent_layers = layers - attention_layers
    inner = metadata["ssm.inner_size"]
    state = metadata["ssm.state_size"]
    groups = metadata["ssm.group_count"]
    heads = metadata["attention.head_count_kv"]
    return GgufCacheGeometry(
        attention_layers=attention_layers,
        recurrent_layers=recurrent_layers,
        nextn_layers=nextn,
        key_width=heads * metadata["attention.key_length"],
        value_width=heads * metadata["attention.value_length"],
        convolution_width=(metadata["ssm.conv_kernel"] - 1)
        * (inner + 2 * groups * state)
        if recurrent_layers
        else 0,
        recurrent_width=state * inner if recurrent_layers else 0,
    )
