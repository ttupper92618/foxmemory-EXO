"""Strict wire contract for independently signed GGUF header metadata."""

from datetime import datetime
from typing import Annotated, Final, Literal, final

from pydantic import BaseModel, ConfigDict, Field, field_validator

GGUF_SCALAR_FIELDS: Final = frozenset(
    {
        "block_count",
        "embedding_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "full_attention_interval",
        "nextn_predict_layers",
        "ssm.conv_kernel",
        "ssm.inner_size",
        "ssm.state_size",
        "ssm.group_count",
        "ssm.time_step_rank",
    }
)


@final
class RegistryGgufHeaderMetadata(BaseModel):
    """
    Bounded facts read from one exact artifact's complete GGUF metadata area.

    The digest covers the file prefix through the last metadata value, including
    the GGUF preamble. It is evidence identity, not a full-artifact checksum.
    Consumers own architecture-specific interpretation and engine allocation.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    architecture: str = Field(min_length=1, max_length=128)
    scalars: dict[str, int] = Field(
        max_length=len(GGUF_SCALAR_FIELDS),
        description="Architecture-relative integer GGUF fields, without defaults.",
    )
    has_recurrent_layer_override: bool = Field(
        description="Whether an explicit attention.recurrent_layers field exists.",
    )
    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_bytes: int = Field(ge=24, le=16 * 1024 * 1024)

    @field_validator("scalars")
    @classmethod
    def validate_scalars(cls, values: dict[str, int]) -> dict[str, int]:
        """Keep only the bounded, unsigned integer vocabulary this version reads."""
        if not set(values).issubset(GGUF_SCALAR_FIELDS):
            raise ValueError("unknown GGUF scalar field")
        if any(value < 0 or value > 2**64 - 1 for value in values.values()):
            raise ValueError("GGUF scalar lies outside unsigned 64-bit range")
        return values


@final
class RegistryGgufArtifactMetadata(BaseModel):
    """Header evidence bound to one immutable repository file."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    repository: str = Field(min_length=3, max_length=512)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    selected_file: str = Field(min_length=1, max_length=4096)
    header: RegistryGgufHeaderMetadata


@final
class RegistryGgufMetadata(BaseModel):
    """One verified auxiliary target bound to its catalog and signed role version."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: Literal[1]
    snapshot_id: str = Field(min_length=1, max_length=120)
    target_version: int = Field(ge=1)
    generated_at: datetime
    artifacts: dict[
        Annotated[str, Field(pattern=r"^card_[a-z2-7]{52}$")],
        RegistryGgufArtifactMetadata,
    ]
