"""Cache accounting from artifact geometry and configured engine dimensions."""

import pytest
from pydantic import ValidationError

from skulk.shared.models.gguf_memory import qwen35_cache_geometry


def metadata() -> dict[str, int]:
    """Return the intrinsic dimensions of the pinned Qwen3.5 9B artifact."""
    return {
        "block_count": 33,
        "embedding_length": 4096,
        "attention.head_count_kv": 4,
        "attention.key_length": 256,
        "attention.value_length": 256,
        "context_length": 262144,
        "ssm.conv_kernel": 4,
        "ssm.inner_size": 4096,
        "ssm.state_size": 128,
        "ssm.group_count": 16,
        "full_attention_interval": 4,
        "nextn_predict_layers": 1,
    }


def test_recurrent_allocation_matches_engine_tensor_dimensions() -> None:
    """Reserve both FP32 state buffers and all 64 retained sequence rows."""
    geometry = qwen35_cache_geometry("qwen35", metadata())
    assert geometry is not None
    assert geometry.attention_layers == 8
    assert geometry.recurrent_layers == 24
    assert geometry.nextn_layers == 1
    assert geometry.convolution_width == 24576
    assert geometry.recurrent_width == 524288
    assert geometry.recurrent_bytes(parallel_slots=16, rollback_depth=3) == 3372220416
    assert geometry.recurrent_bytes(parallel_slots=1, rollback_depth=0) == 52690944
    assert geometry.attention_bytes_per_token(embedded_mtp=False) == 32768
    assert geometry.attention_bytes_per_token(embedded_mtp=True) == 36864


def test_geometry_uses_dimensions_without_model_size_or_quantization_guess() -> None:
    """Different layer/width/slot/depth choices change their own tensor factors."""
    values = metadata() | {
        "block_count": 17,
        "ssm.inner_size": 2048,
        "ssm.group_count": 8,
        "attention.value_length": 128,
    }
    geometry = qwen35_cache_geometry("qwen35", values)
    assert geometry is not None
    assert geometry.recurrent_bytes(parallel_slots=8, rollback_depth=2) == 316145664
    assert geometry.attention_bytes_per_token(embedded_mtp=True) == 15360


@pytest.mark.parametrize("architecture", ["qwen35moe", "qwen3next", "mamba", "llama"])
def test_unmapped_architectures_do_not_inherit_qwen35_formula(
    architecture: str,
) -> None:
    """Similar names and partial fields cannot prove the same engine allocation."""
    assert qwen35_cache_geometry(architecture, metadata()) is None


def test_incomplete_or_overridden_layout_stays_unknown() -> None:
    """Unknown dimensions and explicit layer vectors cannot become zero-cost state."""
    values = metadata()
    del values["ssm.state_size"]
    assert qwen35_cache_geometry("qwen35", values) is None
    assert (
        qwen35_cache_geometry("qwen35", metadata(), has_recurrent_layer_override=True)
        is None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("nextn_predict_layers", 33),
        ("nextn_predict_layers", -1),
        ("ssm.conv_kernel", 0),
        ("full_attention_interval", 0),
    ],
)
def test_malformed_complete_geometry_is_rejected(field: str, value: int) -> None:
    """Complete but invalid geometry is not accepted as a usable memory bound."""
    with pytest.raises(ValueError):
        qwen35_cache_geometry("qwen35", metadata() | {field: value})


def test_geometry_requires_strict_numeric_fields() -> None:
    """Serialized intrinsic geometry cannot coerce strings into dimensions."""
    geometry = qwen35_cache_geometry("qwen35", metadata())
    assert geometry is not None
    with pytest.raises(ValidationError):
        type(geometry).model_validate(geometry.model_dump() | {"key_width": "1024"})
