# pyright: reportPrivateUsage=false
"""Tests for GGUF-repo detection and llama.cpp card creation (slice 3a)."""

import struct
from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from skulk.shared.models import model_cards
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    _gguf_shard_base,
    gguf_weight_siblings,
    select_requested_gguf,
)
from skulk.shared.types.memory import Memory

# --- GGUF binary header builders (for #327 header-parse tests) --------------
#
# Minimal encoders for the GGUF metadata block: magic, version, tensor count,
# kv count, then typed key/value pairs. Mirrors the spec well enough to exercise
# read_gguf_structural_fields without a real multi-GB weights file.

_GGUF_T_FLOAT32 = 6
_GGUF_T_UINT32 = 4
_GGUF_T_STRING = 8
_GGUF_T_ARRAY = 9


def _g_str(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _kv_string(key: str, value: str) -> bytes:
    return _g_str(key) + struct.pack("<I", _GGUF_T_STRING) + _g_str(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _g_str(key) + struct.pack("<I", _GGUF_T_UINT32) + struct.pack("<I", value)


def _kv_string_array(key: str, values: list[str]) -> bytes:
    body = struct.pack("<I", _GGUF_T_STRING) + struct.pack("<Q", len(values))
    for value in values:
        body += _g_str(value)
    return _g_str(key) + struct.pack("<I", _GGUF_T_ARRAY) + body


def _kv_f32_array(key: str, values: list[float]) -> bytes:
    body = struct.pack("<I", _GGUF_T_FLOAT32) + struct.pack("<Q", len(values))
    for value in values:
        body += struct.pack("<f", value)
    return _g_str(key) + struct.pack("<I", _GGUF_T_ARRAY) + body


def _build_gguf(kvs: list[bytes], *, version: int = 3, tensor_count: int = 0) -> bytes:
    return (
        model_cards._GGUF_MAGIC
        + struct.pack("<I", version)
        + struct.pack("<Q", tensor_count)
        + struct.pack("<Q", len(kvs))
        + b"".join(kvs)
    )


def _mem_fetch(blob: bytes) -> "Callable[[int, int], Awaitable[bytes]]":
    async def _fetch(offset: int, length: int) -> bytes:
        return blob[offset : offset + length]

    return _fetch


def _fake_model_info(filenames: list[str]):
    """A stand-in for huggingface_hub.model_info with files_metadata=True."""

    def _factory(
        _model_id: object,
        *,
        revision: str | None = None,
        files_metadata: bool = False,
    ) -> object:
        del revision, files_metadata
        siblings = [SimpleNamespace(rfilename=name, size=100) for name in filenames]
        return SimpleNamespace(
            siblings=siblings,
            safetensors=None,
            sha="d" * 40,
        )

    return _factory


def _mock_dense_header(monkeypatch: pytest.MonkeyPatch, layers: int = 32) -> None:
    """Serve an exact GGUF header independently of the repository config."""
    from skulk.download import download_utils

    blob = _build_gguf(
        [
            _kv_string("general.architecture", "llama"),
            _kv_u32("llama.block_count", layers),
            _kv_u32("llama.embedding_length", 4096),
            _kv_u32("llama.attention.head_count_kv", 8),
            _kv_u32("llama.context_length", 8192),
        ]
    )

    async def read_range(
        _model_id: object, _revision: str, _path: str, start: int, length: int
    ) -> bytes:
        return blob[start : start + length]

    monkeypatch.setattr(download_utils, "range_read", read_range)


def test_gguf_weight_siblings_filters_gguf_and_mmproj(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_cards,
        "model_info",
        _fake_model_info(
            ["model.gguf", "mmproj-model.gguf", "config.json", "README.md"]
        ),
    )
    siblings = gguf_weight_siblings(ModelId("some/gguf-repo"))
    names = {name for name, _ in siblings}
    assert names == {"model.gguf"}  # .gguf only, mmproj excluded


def test_non_gguf_repo_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_cards,
        "model_info",
        _fake_model_info(["model.safetensors", "config.json"]),
    )
    assert gguf_weight_siblings(ModelId("some/mlx-repo")) == []


def test_model_card_source_revision_requires_full_commit() -> None:
    with pytest.raises(ValidationError):
        ModelCard(
            model_id=ModelId("some/model"),
            storage_size=Memory.from_bytes(1),
            n_layers=1,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            source_revision="main",
        )


def test_shard_base_detection() -> None:
    assert _gguf_shard_base("model.gguf") is None
    assert _gguf_shard_base("model-00001-of-00003.gguf") == "model"
    assert (
        _gguf_shard_base("Qwen2.5-7B-Instruct-Q4_K_M-00001-of-00003.gguf")
        == "Qwen2.5-7B-Instruct-Q4_K_M"
    )


async def test_fetch_gguf_card_stamps_both_llama_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_dense_header(monkeypatch)
    monkeypatch.setattr(model_cards, "model_info", _fake_model_info(["model-q4.gguf"]))

    async def _fake_config(
        _model_id: object,
        _revision: str | None = None,
    ) -> object:
        return SimpleNamespace(
            layer_count=32,
            hidden_size=4096,
            num_key_value_heads=8,
            max_position_embeddings=8192,
        )

    monkeypatch.setattr(model_cards, "fetch_config_data", _fake_config)

    card = await ModelCard.fetch_from_hf(ModelId("some/gguf-repo"))
    assert card.source_revision == "d" * 40
    # Both llama.cpp engines, served first (mirrors the bundled GGUF cards,
    # #607): only llama_server pools multiple nodes via RPC, and a
    # llama_cpp-only card would be silently ineligible for every multi-node
    # GGUF placement.
    assert card.placement.compatible_backends == frozenset(
        {
            "llama_server-vulkan",
            "llama_server-rocm",
            "llama_server-cuda",
            "llama_server-cpu",
            "llama_cpp-vulkan",
            "llama_cpp-rocm",
            "llama_cpp-cuda",
            "llama_cpp-cpu",
        }
    )
    assert card.placement.backend_preference[0] == "llama_server-vulkan"
    assert card.supports_tensor is False  # no tensor parallelism either way
    assert card.n_layers == 32 and card.hidden_size == 4096
    assert card.storage_size.in_bytes == 100  # the single selected gguf


async def test_fetch_gguf_card_honors_requested_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_dense_header(monkeypatch)
    requested = "model-IQ3_XXS.gguf"
    monkeypatch.setattr(
        model_cards,
        "model_info",
        _fake_model_info(["model-Q4_K_M.gguf", requested]),
    )

    async def _fake_config(
        _model_id: object,
        _revision: str | None = None,
    ) -> object:
        return SimpleNamespace(
            layer_count=32,
            hidden_size=4096,
            num_key_value_heads=8,
            max_position_embeddings=8192,
        )

    monkeypatch.setattr(model_cards, "fetch_config_data", _fake_config)

    card = await ModelCard.fetch_from_hf(
        ModelId("some/multi-quant-repo"),
        gguf_file=requested,
    )

    assert card.gguf_file == requested
    assert card.quantization == "IQ3"


def test_requested_gguf_must_exist_in_repository() -> None:
    with pytest.raises(ValueError, match="was not found"):
        select_requested_gguf(
            "model-IQ3_XXS.gguf",
            [("model-Q4_K_M.gguf", 100)],
        )


def test_requested_sharded_gguf_uses_first_shard_as_entrypoint() -> None:
    files = [
        ("weights/model-IQ3_XXS-00001-of-00002.gguf", 100),
        ("weights/model-IQ3_XXS-00002-of-00002.gguf", 120),
        ("weights/model-Q4_K_M.gguf", 200),
    ]

    assert (
        select_requested_gguf(
            "weights/model-IQ3_XXS-00002-of-00002.gguf",
            files,
        )
        == "weights/model-IQ3_XXS-00001-of-00002.gguf"
    )


async def test_fetch_gguf_card_reads_header_when_no_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A bare GGUF repo (no config.json) has its structural fields read from the
    # GGUF binary header instead (#327), rather than failing or fabricating them.
    monkeypatch.setattr(model_cards, "model_info", _fake_model_info(["model-q4.gguf"]))

    async def _raises(
        _model_id: object,
        _revision: str | None = None,
    ) -> object:
        raise FileNotFoundError("no config.json in this bare GGUF repo")

    monkeypatch.setattr(model_cards, "fetch_config_data", _raises)

    blob = _build_gguf(
        [
            _kv_string("general.architecture", "llama"),
            _kv_u32("llama.block_count", 24),
            _kv_u32("llama.embedding_length", 3072),
            _kv_u32("llama.attention.head_count_kv", 6),
            _kv_u32("llama.context_length", 16384),
        ]
    )

    from skulk.download import download_utils

    async def _range(
        _model_id: object, _revision: str, _path: str, start: int, length: int
    ) -> bytes:
        return blob[start : start + length]

    monkeypatch.setattr(download_utils, "range_read", _range)

    card = await ModelCard.fetch_from_hf(ModelId("bare/gguf-repo"))
    assert card.n_layers == 24 and card.hidden_size == 3072
    assert card.num_key_value_heads == 6 and card.context_length == 16384
    assert card.gguf_file == "model-q4.gguf"


async def test_fetch_gguf_card_reads_header_when_config_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A config.json that is present but missing num_hidden_layers fails ConfigData
    # validation; that is "config present but unusable", so the header fallback
    # must still kick in rather than letting the ValidationError abort the build.
    monkeypatch.setattr(model_cards, "model_info", _fake_model_info(["model-q4.gguf"]))

    async def _bad_config(
        _model_id: object,
        _revision: str | None = None,
    ) -> object:
        # Raises a genuine pydantic ValidationError (layer_count is required).
        model_cards.ConfigData.model_validate({"hidden_size": 1})
        raise AssertionError("unreachable")

    monkeypatch.setattr(model_cards, "fetch_config_data", _bad_config)

    blob = _build_gguf(
        [
            _kv_string("general.architecture", "qwen2"),
            _kv_u32("qwen2.block_count", 12),
            _kv_u32("qwen2.embedding_length", 1536),
            _kv_u32("qwen2.attention.head_count_kv", 2),
            _kv_u32("qwen2.context_length", 8192),
        ]
    )

    from skulk.download import download_utils

    async def _range(
        _model_id: object, _revision: str, _path: str, start: int, length: int
    ) -> bytes:
        return blob[start : start + length]

    monkeypatch.setattr(download_utils, "range_read", _range)

    card = await ModelCard.fetch_from_hf(ModelId("partial-config/gguf-repo"))
    assert card.n_layers == 12 and card.hidden_size == 1536


async def test_read_gguf_stops_at_tokenizer_when_kv_heads_absent() -> None:
    # head_count_kv is best-effort: when it is absent, the reader must stop at the
    # tokenizer section once the required fields are known, NOT scan the whole KV
    # block. The blob is truncated right after the first tokenizer key, so any
    # attempt to read past it raises; a correct early stop returns cleanly (#327).
    import struct as _struct

    kvs = b"".join(
        [
            _kv_string("general.architecture", "llama"),
            _kv_u32("llama.block_count", 8),
            _kv_u32("llama.embedding_length", 512),
            _kv_u32("llama.context_length", 2048),
        ]
    )
    # Claim more keys than are fully present; the next key is a tokenizer key with
    # no value bytes after it (truncated), which would fail if actually read.
    header = (
        model_cards._GGUF_MAGIC
        + _struct.pack("<I", 3)
        + _struct.pack("<Q", 0)
        + _struct.pack("<Q", 5)
    )
    truncated = header + kvs + _g_str("tokenizer.ggml.model")

    fields = await model_cards.read_gguf_structural_fields(_mem_fetch(truncated))
    assert fields.n_layers == 8 and fields.hidden_size == 512
    assert fields.num_key_value_heads is None and fields.context_length == 2048


async def test_read_gguf_structural_fields_basic() -> None:
    blob = _build_gguf(
        [
            _kv_string("general.architecture", "llama"),
            _kv_u32("llama.block_count", 32),
            _kv_u32("llama.embedding_length", 4096),
            _kv_u32("llama.attention.head_count_kv", 8),
            _kv_u32("llama.context_length", 8192),
        ]
    )
    fields = await model_cards.read_gguf_structural_fields(_mem_fetch(blob))
    assert fields == model_cards.GgufStructuralFields(32, 4096, 8, 8192)


async def test_read_gguf_skips_arrays_before_structural_keys() -> None:
    # A multi-thousand-entry tokenizer array sitting before the structural keys
    # must be skipped, not parsed into the metadata, and not block the read.
    blob = _build_gguf(
        [
            _kv_string("general.architecture", "qwen2"),
            _kv_string_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(5000)]),
            _kv_f32_array("tokenizer.ggml.scores", [0.1] * 5000),
            _kv_u32("qwen2.block_count", 28),
            _kv_u32("qwen2.embedding_length", 3584),
            _kv_u32("qwen2.attention.head_count_kv", 4),
            _kv_u32("qwen2.context_length", 32768),
        ]
    )
    fields = await model_cards.read_gguf_structural_fields(_mem_fetch(blob))
    assert fields.n_layers == 28 and fields.hidden_size == 3584
    assert fields.num_key_value_heads == 4 and fields.context_length == 32768


async def test_read_gguf_missing_kv_heads_is_none() -> None:
    blob = _build_gguf(
        [
            _kv_string("general.architecture", "llama"),
            _kv_u32("llama.block_count", 16),
            _kv_u32("llama.embedding_length", 2048),
            _kv_u32("llama.context_length", 4096),
        ]
    )
    fields = await model_cards.read_gguf_structural_fields(_mem_fetch(blob))
    assert fields.num_key_value_heads is None
    assert fields.n_layers == 16 and fields.context_length == 4096


async def test_read_gguf_bad_magic_raises() -> None:
    with pytest.raises(ValueError, match="not a GGUF"):
        await model_cards.read_gguf_structural_fields(_mem_fetch(b"XXXX" + b"\x00" * 64))


async def test_read_gguf_unsupported_version_raises() -> None:
    blob = (
        model_cards._GGUF_MAGIC
        + struct.pack("<I", 1)  # v1: 32-bit lengths, obsolete
        + struct.pack("<Q", 0)
        + struct.pack("<Q", 0)
    )
    with pytest.raises(ValueError, match="version"):
        await model_cards.read_gguf_structural_fields(_mem_fetch(blob))


async def test_read_gguf_missing_architecture_raises() -> None:
    blob = _build_gguf([_kv_u32("llama.block_count", 8)])
    with pytest.raises(ValueError, match="architecture"):
        await model_cards.read_gguf_structural_fields(_mem_fetch(blob))


async def test_read_gguf_windowed_fetch() -> None:
    # A transport that returns only a few bytes per call must be reassembled,
    # not mistaken for EOF after the first short read.
    blob = _build_gguf(
        [
            _kv_string("general.architecture", "llama"),
            _kv_u32("llama.block_count", 10),
            _kv_u32("llama.embedding_length", 100),
            _kv_u32("llama.attention.head_count_kv", 2),
            _kv_u32("llama.context_length", 2048),
        ]
    )
    calls = 0

    async def _chunked(offset: int, length: int) -> bytes:
        nonlocal calls
        calls += 1
        return blob[offset : offset + min(length, 7)]

    fields = await model_cards.read_gguf_structural_fields(_chunked)
    assert fields.n_layers == 10 and fields.hidden_size == 100
    assert calls > 1  # required multiple fetches to assemble the header


def test_select_preferred_gguf_prefers_quant_over_bf16() -> None:
    from skulk.shared.models.model_cards import (
        gguf_allow_patterns,
        gguf_shard_group_size,
        select_preferred_gguf,
    )

    files = [
        ("M-BF16.gguf", 2_000),
        ("M-Q4_K_M.gguf", 800),
        ("M-Q8_0.gguf", 1_300),
    ]
    sel = select_preferred_gguf(files)
    assert sel == "M-Q4_K_M.gguf"  # quant beats BF16; Q4_K_M is top preference
    assert gguf_shard_group_size(sel, files).in_bytes == 800
    # Single-file LM quant plus the always-included projector glob (#346).
    assert gguf_allow_patterns(sel) == ["M-Q4_K_M.gguf", "*mmproj*.gguf"]


def test_select_preferred_gguf_sharded_group() -> None:
    from skulk.shared.models.model_cards import (
        gguf_allow_patterns,
        gguf_shard_group_size,
        select_preferred_gguf,
    )

    files = [
        ("big-Q4_K_M-00001-of-00002.gguf", 500),
        ("big-Q4_K_M-00002-of-00002.gguf", 600),
        ("big-BF16.gguf", 4_000),
    ]
    sel = select_preferred_gguf(files)
    assert sel == "big-Q4_K_M-00001-of-00002.gguf"
    assert gguf_shard_group_size(sel, files).in_bytes == 1_100  # both shards
    # Sharded LM group glob plus the always-included projector glob (#346).
    assert gguf_allow_patterns(sel) == ["big-Q4_K_M-*-of-*.gguf", "*mmproj*.gguf"]


async def test_gguf_card_pins_selected_quant(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_dense_header(monkeypatch, layers=16)
    monkeypatch.setattr(
        model_cards,
        "model_info",
        _fake_model_info(["model-BF16.gguf", "model-Q4_K_M.gguf"]),
    )

    async def _cfg(
        _m: object,
        _revision: str | None = None,
    ) -> object:
        return SimpleNamespace(
            layer_count=16,
            hidden_size=2048,
            num_key_value_heads=8,
            max_position_embeddings=8192,
        )

    monkeypatch.setattr(model_cards, "fetch_config_data", _cfg)
    card = await ModelCard.fetch_from_hf(ModelId("some/gguf-repo"))
    assert card.gguf_file == "model-Q4_K_M.gguf"
    assert card.quantization == "Q4_K_M"


def test_served_spec_n_max_must_be_positive() -> None:
    """A non-positive served_spec_n_max fails at card validation (it would produce
    an undefined --spec-draft-n-max at the server), while a positive value and
    None (server default) are accepted."""
    from pydantic import ValidationError

    from skulk.shared.models.model_cards import RuntimeCapabilityCardConfig

    assert RuntimeCapabilityCardConfig(served_spec_n_max=3).served_spec_n_max == 3
    assert RuntimeCapabilityCardConfig(served_spec_n_max=None).served_spec_n_max is None
    for bad in (0, -1):
        with pytest.raises(ValidationError):
            RuntimeCapabilityCardConfig(served_spec_n_max=bad)


def test_default_gguf_selection_never_picks_companion_artifacts() -> None:
    """A repo default must not stage a drafter as the model (the dspark trap)."""
    from skulk.shared.models.model_cards import select_preferred_gguf

    files = [
        ("dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf", 10),
        ("UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf", 50),
    ]
    assert "dspark" not in select_preferred_gguf(files)
    # A drafter-only repo (a published draft companion) still resolves.
    assert select_preferred_gguf([("gemma-mtp-draft-Q8_0.gguf", 1)])


async def test_header_keeps_hybrid_fields_after_basic_structural_dimensions() -> None:
    """The real header ordering must not trigger the old four-field early exit."""
    from skulk.shared.models.tests.test_gguf_memory import metadata

    blob = _build_gguf(
        [_kv_string("general.architecture", "qwen35")]
        + [_kv_u32("qwen35." + key, value) for key, value in metadata().items()]
        + [_kv_string("tokenizer.ggml.model", "unused")]
    )
    fields = await model_cards.read_gguf_structural_fields(_mem_fetch(blob))
    assert fields.cache_geometry == model_cards.qwen35_cache_geometry(
        "qwen35", metadata()
    )
