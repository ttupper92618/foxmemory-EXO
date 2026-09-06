import contextlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

if TYPE_CHECKING:
    from skulk.worker.engines.mlx.vision import VisionProcessor

# Monkey-patch for transformers 5.x compatibility
# Kimi's tokenization_kimi.py imports bytes_to_unicode from the old location
# which was moved in transformers 5.0.0rc2
try:
    import transformers.models.gpt2.tokenization_gpt2 as gpt2_tokenization
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    if not hasattr(gpt2_tokenization, "bytes_to_unicode"):
        gpt2_tokenization.bytes_to_unicode = bytes_to_unicode  # type: ignore[attr-defined]
except ImportError:
    pass  # transformers < 5.0 or bytes_to_unicode not available

from mlx_lm.models.cache import KVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3Model
from mlx_lm.tokenizer_utils import TokenizerWrapper

from skulk.shared.constants import preferred_env_value
from skulk.shared.models.capabilities import (
    muse_glimmer_template_kwargs,
    resolve_model_capability_profile,
)
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    PromptRendererType,
    ToolCallFormat,
    get_card,
    multi_node_speculation_disabled,
)
from skulk.worker.engines.mlx.constants import TRUST_REMOTE_CODE

try:
    from mlx_lm.tokenizer_utils import load_tokenizer
except ImportError:
    from mlx_lm.tokenizer_utils import load as load_tokenizer

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model as _mlx_lm_load_model
from pydantic import RootModel
from safetensors import safe_open

from skulk.download.download_utils import (
    build_companion_model_path,
    build_model_path,
    build_sidecar_path,
    companion_artifact_location,
)
from skulk.shared.types.common import Host
from skulk.shared.types.memory import Memory
from skulk.shared.types.mlx import Model
from skulk.shared.types.tasks import TaskId, TextGeneration
from skulk.shared.types.text_generation import TextGenerationTaskParams
from skulk.shared.types.worker.instances import (
    BoundInstance,
    LlamaRpcInstance,
    MlxJacclInstance,
    MlxRingInstance,
)
from skulk.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    RpcDonorShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from skulk.worker.engines.mlx.auto_parallel import (
    LayerLoadedCallback,
    TimeoutCallback,
    eval_with_timeout,
    get_inner_model,
    get_layers,
    pipeline_auto_parallel,
    pipeline_eval_timeout_seconds,
    pipeline_timeout_callback,
    tensor_auto_parallel,
)
from skulk.worker.engines.mlx.drafters.gemma4_assistant import load_assistant_model
from skulk.worker.engines.mlx.gemma4_prompt import render_gemma4_prompt
from skulk.worker.runner.bootstrap import logger
from skulk.worker.runner.diagnostics import remember_wired_limit_bytes

Group = mx.distributed.Group


def _object_dict(value: object) -> dict[str, object]:
    """Return a string-keyed object mapping for dynamic Python objects."""
    return cast(dict[str, object], value)


class _Gemma4PatchEmbedder(Protocol):
    """Typed callable surface for Gemma 4 patch embedding."""

    def __call__(
        self,
        pixel_values: mx.array,
        patch_positions: mx.array,
        padding_positions: mx.array,
    ) -> mx.array: ...


class _Gemma4Encoder(Protocol):
    """Typed callable surface for Gemma 4 vision encoding."""

    def __call__(
        self,
        inputs_embeds: mx.array,
        patch_positions: mx.array,
        attn_mask: mx.array,
    ) -> mx.array: ...


class _Gemma4Pooler(Protocol):
    """Typed callable surface for Gemma 4 pooled patch selection."""

    def __call__(
        self,
        hidden_states: mx.array,
        patch_positions: mx.array,
        padding_positions: mx.array,
        *,
        output_length: int,
    ) -> tuple[mx.array, mx.array]: ...


class _Gemma4Config(Protocol):
    """Subset of Gemma 4 config used by the dynamic native-vision wrapper."""

    standardize: bool


class _Gemma4NativeVisionTower(Protocol):
    """Typed view of the Gemma 4 vision tower fields used by Skulk."""

    patch_size: int
    pooling_kernel_size: int
    max_patches: int
    patch_embedder: _Gemma4PatchEmbedder
    encoder: _Gemma4Encoder
    pooler: _Gemma4Pooler
    config: _Gemma4Config | None
    std_bias: mx.array
    std_scale: mx.array


class _HasLogits(Protocol):
    """Minimal model output that exposes logits."""

    logits: mx.array


class _VlmModelLoader(Protocol):
    """Typed callable surface for mlx-vlm's model loader."""

    def __call__(self, model_path: Path, **kwargs: object) -> nn.Module: ...


class _MlxLmLoadModel(Protocol):
    """Typed callable surface for mlx-lm's model loader."""

    def __call__(
        self,
        model_path: Path,
        *,
        trust_remote_code: bool = ...,
        **kwargs: object,
    ) -> tuple[nn.Module, object]: ...


class _SafeTensorMetadataHandle(Protocol):
    """Typed metadata surface used from safetensors' dynamic extension type."""

    def metadata(self) -> dict[str, str] | None: ...


_QWEN_CONVERTED_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)
_VLM_LOAD_PATCH_LOCK = threading.Lock()
def _has_mlx_safetensors(model_path: Path) -> bool:
    """Return whether a local model bundle contains converted MLX weights."""
    for weight_path in sorted(model_path.glob("*.safetensors")):
        try:
            with safe_open(str(weight_path), framework="numpy") as raw_handle:
                handle = cast(
                    _SafeTensorMetadataHandle,
                    cast(object, raw_handle),
                )
                metadata = handle.metadata() or {}
        except (OSError, ValueError):
            continue
        if metadata.get("format") == "mlx":
            return True
    return False


def _guard_converted_qwen_norm_sanitizer(
    sanitizer: Callable[
        [object, dict[str, mx.array]],
        dict[str, mx.array],
    ],
    sanitize_key: Callable[[str], str],
) -> Callable[[object, dict[str, mx.array]], dict[str, mx.array]]:
    """Backport mlx-vlm's converted-Qwen RMSNorm guard from v0.6.5.

    mlx-vlm 0.6.4 adds one to every Qwen3.5-family RMSNorm weight during
    loading. That conversion is correct for source Hugging Face weights but
    corrupts MLX checkpoints whose ``language_model.*`` keys were already
    converted. Preserve those exact arrays while leaving all other upstream
    sanitization intact.
    """

    def guarded(
        model: object,
        weights: dict[str, mx.array],
    ) -> dict[str, mx.array]:
        converted_norm_weights = {
            # mlx-vlm 0.6.4 uses ``value += 1``; MLX mutates that array, so
            # retaining the original object would retain the corruption too.
            sanitize_key(key): mx.array(value)
            for key, value in weights.items()
            if key.startswith("language_model.")
            and value.ndim == 1
            and any(key.endswith(suffix) for suffix in _QWEN_CONVERTED_NORM_SUFFIXES)
        }
        sanitized = sanitizer(model, weights)
        sanitized.update(converted_norm_weights)
        return sanitized

    return guarded


def _install_converted_qwen_norm_guard(
    model_path: Path,
) -> tuple[type[object], object] | None:
    """Install the v0.6.5 Qwen sanitizer fix for one v0.6.4 model load."""
    if not _has_mlx_safetensors(model_path):
        return None
    try:
        raw_config = cast(
            object,
            json.loads((model_path / "config.json").read_text()),
        )
    except (json.JSONDecodeError, OSError):
        return None
    config = _object_dict(raw_config)
    model_type = config.get("model_type")
    if model_type not in {"qwen3_5", "qwen3_5_moe"}:
        return None

    model_module = import_module(f"mlx_vlm.models.{model_type}")
    model_class = cast(type[object], vars(model_module)["Model"])
    model_class_dict = cast(Mapping[str, object], vars(model_class))
    original = model_class_dict["sanitize"]
    sanitize_key_module = import_module("mlx_vlm.models.qwen3_5.qwen3_5")
    sanitize_key = cast(
        Callable[[str], str],
        vars(sanitize_key_module)["sanitize_key"],
    )
    guarded = _guard_converted_qwen_norm_sanitizer(
        cast(
            Callable[[object, dict[str, mx.array]], dict[str, mx.array]],
            original,
        ),
        sanitize_key,
    )
    type.__setattr__(model_class, "sanitize", guarded)
    logger.info(
        "Applying the mlx-vlm 0.6.5 converted-Qwen norm sanitizer guard"
    )
    return model_class, original


def _apply_nemotron_reasoning_controls(
    formatted_messages: list[dict[str, Any]],
    *,
    enable_thinking: bool | None,
    family: str,
) -> list[dict[str, Any]]:
    """Translate explicit thinking toggles into Nemotron chat-template controls.

    Nemotron v2 models expect `/think` or `/no_think` to appear in the rendered
    chat messages. Passing ``enable_thinking`` as a template kwarg is not enough
    because their tokenizer template keys off literal control strings in system
    or user turns.

    We only inject a control message when the caller explicitly asked for
    thinking on/off and the conversation does not already contain a turn-level
    Nemotron control string.
    """
    if family != "nemotron" or enable_thinking is None:
        return formatted_messages

    control_text = "/think" if enable_thinking else "/no_think"
    existing_controls = {"/think", "/no_think"}
    for msg in formatted_messages:
        if msg.get("role") not in {"system", "user", "developer"}:
            continue
        content = msg.get("content")
        if isinstance(content, str) and any(
            marker in content for marker in existing_controls
        ):
            return formatted_messages

    updated_messages = [dict(msg) for msg in formatted_messages]
    if updated_messages and updated_messages[0].get("role") == "system":
        existing_content = updated_messages[0].get("content")
        if isinstance(existing_content, str) and existing_content:
            updated_messages[0] = {
                **updated_messages[0],
                "content": f"{control_text}\n{existing_content}",
            }
        else:
            updated_messages[0] = {**updated_messages[0], "content": control_text}
        return updated_messages

    return [{"role": "system", "content": control_text}, *updated_messages]


def _request_shape_debug_enabled() -> bool:
    """Return whether request-shape tracing is enabled for prompt debugging."""
    value = preferred_env_value(
        "SKULK_TRACE_REQUEST_SHAPES",
    )
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def log_request_shape(
    label: str,
    task_params: TextGenerationTaskParams,
    prompt: str,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit an exact request-shape trace for prompt/rendering comparisons.

    This is intentionally opt-in because it logs full rendered prompts and the
    complete internal request payload. We use it to compare synthetic warmup
    requests against real traffic when a model wedges only during startup.
    """
    if not _request_shape_debug_enabled():
        return

    payload: dict[str, Any] = {
        "label": label,
        "task_params": task_params.model_dump(mode="json"),
        "prompt_chars": len(prompt),
        "prompt_lines": prompt.count("\n") + 1,
    }
    if extra:
        payload.update(extra)

    logger.info(
        f"[request-shape] {json.dumps(payload, ensure_ascii=True, sort_keys=True)}"
    )
    logger.info(f"[request-shape] prompt label={label}\n{prompt}")


def _gemma4_output_length_for_pixel_values(
    pixel_values: mx.array,
    patch_size: int,
    pooling_kernel_size: int,
) -> int:
    """Compute Gemma 4's pooled visual token count for a preprocessed image batch.

    Gemma 4's processor resizes images so the patch grid is divisible by the
    pooling kernel size. The number of pooled visual tokens is therefore the
    number of patches divided by ``pooling_kernel_size ** 2``.
    """
    if pixel_values.ndim != 4:
        raise ValueError(
            "Gemma 4 vision expects pixel values shaped [batch, channels, height, width]"
        )

    _, _, height, width = pixel_values.shape
    patches_per_image = (height // patch_size) * (width // patch_size)
    pooled_tokens, remainder = divmod(
        patches_per_image, pooling_kernel_size * pooling_kernel_size
    )
    if remainder != 0:
        raise ValueError(
            "Gemma 4 preprocessing produced a patch grid that cannot be pooled "
            f"cleanly: {patches_per_image=} is not divisible by "
            f"{pooling_kernel_size ** 2=}"
        )
    return pooled_tokens


def _gemma4_patch_positions_and_padding(
    pixel_values: mx.array,
    patch_size: int,
    max_patches: int,
) -> tuple[mx.array, mx.array]:
    """Build Gemma 4 patch positions with padding that matches the real sequence length.

    MLX-VLM's Gemma 4 implementation pads patch positions up to ``max_patches`` when
    the processed image is small enough, but some higher token budgets produce more
    real patches than that default cap. In that case we must stop padding entirely
    so the patch positions, attention mask, and hidden-state sequence all agree on
    the same length.
    """
    if pixel_values.ndim != 4:
        raise ValueError(
            "Gemma 4 vision expects pixel values shaped [batch, channels, height, width]"
        )

    batch_size, _, height, width = pixel_values.shape
    patch_height = height // patch_size
    patch_width = width // patch_size
    num_real_patches = patch_height * patch_width

    grid_x = mx.arange(patch_width, dtype=mx.int32)
    grid_y = mx.arange(patch_height, dtype=mx.int32)
    mesh_x, mesh_y = mx.meshgrid(grid_x, grid_y, indexing="xy")
    real_positions = mx.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], axis=-1)
    real_positions = mx.broadcast_to(
        mx.expand_dims(real_positions, axis=0),
        (batch_size, num_real_patches, 2),
    )

    if num_real_patches >= max_patches:
        padding_positions = mx.zeros((batch_size, num_real_patches), dtype=mx.bool_)
        return real_positions, padding_positions

    num_padding = max_patches - num_real_patches
    pad_positions = mx.full((batch_size, num_padding, 2), -1, dtype=mx.int32)
    patch_positions = mx.concatenate([real_positions, pad_positions], axis=1)
    padding_positions = mx.concatenate(
        [
            mx.zeros((batch_size, num_real_patches), dtype=mx.bool_),
            mx.ones((batch_size, num_padding), dtype=mx.bool_),
        ],
        axis=1,
    )
    return patch_positions, padding_positions


class _Gemma4DynamicVisionTower(nn.Module):
    """Wrap Gemma 4's vision tower so pooling matches the processor's token count.

    The current MLX-VLM Gemma 4 encoder always pools to
    ``config.default_output_length`` (280), even when the processor resized the
    image to produce a different number of soft tokens. Wide images can therefore
    lose a large fraction of their patches during pooling and produce unrelated
    captions. This wrapper keeps the existing encoder path intact but derives the
    pooling length from the actual preprocessed image size.
    """

    _inner: _Gemma4NativeVisionTower

    def __init__(self, inner: _Gemma4NativeVisionTower) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)

    def _encode_one(self, pixel_values: mx.array) -> mx.array:
        if pixel_values.ndim != 4:
            raise ValueError(
                "Gemma 4 vision tower expects batched pixel values for native vision"
            )

        patch_size = int(self._inner.patch_size)
        pooling_kernel_size = int(self._inner.pooling_kernel_size)
        output_length = _gemma4_output_length_for_pixel_values(
            pixel_values,
            patch_size=patch_size,
            pooling_kernel_size=pooling_kernel_size,
        )
        logger.info(
            "Gemma 4 native vision pooling: "
            f"image_shape={tuple(pixel_values.shape)} "
            f"patch_size={patch_size} "
            f"pooling_kernel_size={pooling_kernel_size} "
            f"default_output_length={getattr(self._inner, 'default_output_length', 'unknown')} "
            f"dynamic_output_length={output_length}"
        )

        batch_size, _, height, width = pixel_values.shape
        num_real_patches = (height // patch_size) * (width // patch_size)
        patch_positions, padding_positions = _gemma4_patch_positions_and_padding(
            pixel_values,
            patch_size=patch_size,
            max_patches=int(self._inner.max_patches),
        )
        sequence_length = patch_positions.shape[1]

        inputs_embeds = self._inner.patch_embedder(
            pixel_values,
            patch_positions[:, :num_real_patches],
            padding_positions[:, :num_real_patches],
        )

        num_padding = sequence_length - num_real_patches
        if num_padding > 0:
            pad_embeds = mx.zeros(
                (batch_size, num_padding, inputs_embeds.shape[-1]),
                dtype=inputs_embeds.dtype,
            )
            inputs_embeds = mx.concatenate([inputs_embeds, pad_embeds], axis=1)

        valid_mask = ~padding_positions
        attn_mask = mx.expand_dims(valid_mask, 1) * mx.expand_dims(valid_mask, 2)
        neg_inf = mx.array(float("-inf"), dtype=inputs_embeds.dtype)
        attn_mask = mx.where(
            attn_mask, mx.array(0.0, dtype=inputs_embeds.dtype), neg_inf
        )
        attn_mask = mx.expand_dims(attn_mask, 1)

        hidden_states = self._inner.encoder(inputs_embeds, patch_positions, attn_mask)
        pooled, pool_mask = self._inner.pooler(
            hidden_states,
            patch_positions,
            padding_positions,
            output_length=output_length,
        )

        pooled_mask = pool_mask if pool_mask.shape[1] == output_length else ~pool_mask
        all_real: list[mx.array] = []
        for batch_idx in range(batch_size):
            n_valid = int(pooled_mask[batch_idx].astype(mx.int32).sum().item())
            all_real.append(pooled[batch_idx, :n_valid])

        hidden_states = mx.concatenate(all_real, axis=0)[None]

        config = self._inner.config
        if config is not None and config.standardize:
            hidden_states = (
                hidden_states - self._inner.std_bias
            ) * self._inner.std_scale

        return hidden_states

    def __call__(self, pixel_values: mx.array | list[mx.array]) -> mx.array:
        if isinstance(pixel_values, list):
            features: list[mx.array] = []
            for pixel_value in pixel_values:
                batched = pixel_value[None] if pixel_value.ndim == 3 else pixel_value
                encoded = self._encode_one(batched)
                features.append(encoded[0] if encoded.ndim == 3 else encoded)
            return mx.concatenate(features, axis=0)[None]
        return self._encode_one(_batch_gemma4_pixel_values(pixel_values))

    def __getattr__(self, name: str) -> object:
        if name == "_inner":
            return cast(object, object.__getattribute__(self, name))
        return cast(object, getattr(self._inner, name))


def _batch_gemma4_pixel_values(pixel_values: mx.array) -> mx.array:
    """Add the image batch dimension omitted by single-image processors."""

    return pixel_values[None] if pixel_values.ndim == 3 else pixel_values


def _patch_gemma4_native_vision(model: nn.Module) -> nn.Module:
    """Patch Gemma 4's native vision tower to pool using the real image grid size."""
    if getattr(getattr(model, "config", None), "model_type", None) != "gemma4":
        return model

    vision_tower = getattr(model, "vision_tower", None)
    if vision_tower is None or isinstance(vision_tower, _Gemma4DynamicVisionTower):
        return model

    logger.info(
        "Patching Gemma 4 vision tower to use dynamic pooling lengths for native vision"
    )
    model.vision_tower = _Gemma4DynamicVisionTower(
        cast(_Gemma4NativeVisionTower, vision_tower)
    )
    return model


class _VlmModelWrapper(nn.Module):
    """Wrapper that unwraps LanguageModelOutput → mx.array for mlx-lm compat.

    mlx-vlm models return LanguageModelOutput from __call__, but mlx-lm's
    generation pipeline (stream_generate, generate_step) expects a plain
    mx.array.  Python resolves __call__ on the *class*, not the instance,
    so a simple instance attribute patch doesn't work — we need a real
    subclass override.
    """

    _inner: nn.Module

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        # Set by the vision pipeline before prefill for native-vision models
        # like Gemma 4, then cleared once prefill has populated the KV cache.
        object.__setattr__(self, "_pixel_values", None)
        object.__setattr__(self, "_image_grid_thw", None)

    def set_pixel_values(self, pixel_values: mx.array | list[mx.array] | None) -> None:
        """Cache native vision pixels without family-specific grid metadata."""
        object.__setattr__(self, "_pixel_values", pixel_values)
        object.__setattr__(self, "_image_grid_thw", None)

    def set_vision_inputs(
        self,
        pixel_values: mx.array | list[mx.array] | None,
        image_grid_thw: mx.array | None,
    ) -> None:
        """Cache native vision tensors for the next prefill call only."""
        object.__setattr__(self, "_pixel_values", pixel_values)
        object.__setattr__(self, "_image_grid_thw", image_grid_thw)

    def make_cache(self) -> list[object]:
        """Create the native language model's cache, including hybrid SSM state."""
        inner = cast(object, self._inner)
        language_model = getattr(inner, "language_model", None)
        cache_owner = language_model if language_model is not None else inner
        if hasattr(cache_owner, "make_cache"):
            factory = cast(
                Callable[[], list[object]],
                object.__getattribute__(cache_owner, "make_cache"),
            )
            return factory()
        layers = cast(Model, inner).layers
        return [KVCache() for _ in layers]

    def tensor_parallel_target(self) -> nn.Module:
        """Return the nested language model whose weights tensor sharding owns.

        The wrapper itself owns generation compatibility and the native VLM
        owns the vision tower, but tensor parallelism shards only the language
        trunk. Keeping those roles separate also ensures the distributed
        generation synchronization patch continues to observe plain logits
        rather than an upstream ``LanguageModelOutput``.
        """
        inner = cast(object, self._inner)
        language_model = getattr(inner, "language_model", None)
        if not isinstance(language_model, nn.Module):
            raise ValueError(
                "Native vision model does not expose a tensor-shardable "
                "language model"
            )
        return language_model

    def __call__(self, *args: object, **kwargs: object) -> mx.array:
        pixel_values = cast(
            mx.array | list[mx.array] | None,
            _object_dict(cast(object, object.__getattribute__(self, "__dict__"))).get(
                "_pixel_values"
            ),
        )
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
            image_grid_thw = cast(
                mx.array | None,
                _object_dict(
                    cast(object, object.__getattribute__(self, "__dict__"))
                ).get("_image_grid_thw"),
            )
            if image_grid_thw is not None:
                kwargs["image_grid_thw"] = image_grid_thw
            # Pixels belong only to the prompt forward that contains their
            # image-token span. mlx-lm's prefill helper performs an additional
            # one-token forward before returning, so retaining them here would
            # inject image features into a decode token with no image markers.
            object.__setattr__(self, "_pixel_values", None)
            object.__setattr__(self, "_image_grid_thw", None)
        inner = cast(object, self._inner)
        language_model = getattr(inner, "language_model", None)
        call_target = (
            inner
            if pixel_values is not None or language_model is None
            else language_model
        )
        call = cast(Callable[..., object], call_target)
        result = call(*args, **kwargs)
        if hasattr(result, "logits"):
            return cast(_HasLogits, result).logits
        return cast(mx.array, result)

    def __getattr__(self, name: str) -> object:
        # Delegate attribute access to the inner model so layer iteration,
        # parameter access, etc. all work transparently.
        if name in {"_pixel_values", "_image_grid_thw"}:
            return _object_dict(
                cast(object, object.__getattribute__(self, "__dict__"))
            ).get(name)
        if name == "_inner":
            return cast(object, object.__getattribute__(self, name))
        return cast(object, getattr(self._inner, name))


def _load_vlm_model(model_path: Path, **kwargs: object) -> tuple[nn.Module, object]:
    """Load and wrap one native multimodal model through MLX-VLM."""
    mlx_vlm_utils = import_module("mlx_vlm.utils")
    mlx_vlm_utils_dict = cast(dict[str, object], vars(mlx_vlm_utils))
    mlx_vlm_load_model = cast(
        _VlmModelLoader,
        mlx_vlm_utils_dict["load_model"],
    )
    with _VLM_LOAD_PATCH_LOCK:
        patch = _install_converted_qwen_norm_guard(model_path)
        try:
            model = mlx_vlm_load_model(model_path, **kwargs)
        finally:
            if patch is not None:
                model_class, original = patch
                type.__setattr__(model_class, "sanitize", original)
    model = _patch_gemma4_native_vision(model)
    return _VlmModelWrapper(model), None


def load_model(
    model_path: Path,
    *,
    prefer_vlm: bool = False,
    trust_remote_code: bool = False,
    **kwargs: object,
) -> tuple[nn.Module, object]:
    """Load a text or native multimodal model.

    Args:
        model_path: Local model bundle.
        prefer_vlm: Load through MLX-VLM even when MLX-LM recognizes the model
            type. MLX-LM intentionally strips vision towers from Qwen VLM
            checkpoints, so vision placements must opt into the native model to
            preserve their image-grid-aware embeddings and RoPE.
        trust_remote_code: Allow mlx-lm to execute a custom architecture file
            named by the config's ``model_file`` key (CVE-2026-5843 gate).
            Callers pass the card's own ``trust_remote_code``, so only a card
            authorized as executable can reach that path; the mlx-vlm loaders
            manage repository code internally and do not take this flag.
        **kwargs: Loader options forwarded to the selected upstream loader.

    Returns:
        The loaded model and the optional upstream tokenizer placeholder.
    """
    if prefer_vlm:
        logger.info("Loading vision model through native mlx-vlm path")
        try:
            return _load_vlm_model(model_path, **kwargs)
        except ImportError as exc:
            raise ValueError("Native vision model loading requires mlx-vlm") from exc

    mlx_lm_load_model = cast(_MlxLmLoadModel, _mlx_lm_load_model)
    try:
        return mlx_lm_load_model(
            model_path, trust_remote_code=trust_remote_code, **kwargs
        )
    except ValueError as exc:
        if "not supported" not in str(exc):
            raise
        logger.info(f"Model type not supported by mlx-lm, trying mlx-vlm: {exc}")
        try:
            return _load_vlm_model(model_path, **kwargs)
        except ImportError:
            raise ValueError(
                f"{exc}. Install mlx-vlm for vision model support: pip install -U mlx-vlm"
            ) from exc


def _prefers_native_vlm(model_card: ModelCard) -> bool:
    """Return whether a card's primary weights require the native VLM loader.

    MLX-LM recognizes some multimodal checkpoints as text models and strips
    their vision towers. When the card points vision at the same checkpoint as
    its language weights, every placement shape must load through MLX-VLM so
    family-specific image-grid positions and embeddings remain available.
    """
    vision_config = model_card.vision
    return vision_config is not None and vision_config.weights_repo == str(
        model_card.model_id
    )


def sidecar_load_eligible(
    bound_shard: ShardMetadata,
    *,
    single_node: bool,
    speculation_blocked: bool,
) -> bool:
    """Whether this rank may load MTP sidecar weights for its placement.

    Single-node placements always draft locally. Multi-node PIPELINE
    placements load on the decider (last) rank only — drafts and accept
    decisions are broadcast (#254), so other ranks never draft and must not
    pay sidecar memory. Multi-rank TENSOR placements load on EVERY rank
    (#263): draft logits go through the TP-sharded lm_head, so a
    "decider-local" draft issues all-rank collectives that idle receivers
    never join — the decider GPU-times-out inside its first draft round and
    SIGABRTs in the Metal completion block. Rank-symmetric drafting (every
    rank drafts, lockstep implicit — the same envelope assistants use on
    TP, and the measured +31% TP configuration) keeps the collective
    schedule identical instead; it relies on bit-identical per-rank logits,
    which TP placements already require in practice. A card that blocks
    multi-node speculation (``speculative_multi_node=false``) loads no
    sidecar on any multi-node rank.
    """
    if speculation_blocked:
        return False
    if single_node:
        return True
    if isinstance(bound_shard, TensorShardMetadata):
        return True
    return (
        isinstance(bound_shard, PipelineShardMetadata)
        and bound_shard.device_rank == bound_shard.world_size - 1
    )


def get_weights_size(model_shard_meta: ShardMetadata) -> Memory:
    return Memory.from_float_kb(
        (model_shard_meta.end_layer - model_shard_meta.start_layer)
        / model_shard_meta.n_layers
        * model_shard_meta.model_card.storage_size.in_kb
        / (
            1
            if isinstance(model_shard_meta, PipelineShardMetadata)
            else model_shard_meta.world_size
        )
    )


class ModelLoadingTimeoutError(Exception):
    pass


class HostList(RootModel[list[str]]):
    @classmethod
    def from_hosts(cls, hosts: list[Host]) -> "HostList":
        return cls(root=[str(host) for host in hosts])


def mlx_distributed_init(
    bound_instance: BoundInstance,
) -> Group:
    """
    Initialize MLX distributed.
    """
    rank = bound_instance.bound_shard.device_rank
    logger.info(f"Starting initialization for rank {rank}")

    with tempfile.TemporaryDirectory() as tmpdir:
        coordination_file = str(
            Path(tmpdir) / f"hosts_{bound_instance.instance.instance_id}_{rank}.json"
        )
        # TODO: singleton instances
        match bound_instance.instance:
            case MlxRingInstance(hosts_by_node=hosts_by_node, ephemeral_port=_):
                hosts_for_node = hosts_by_node[bound_instance.bound_node_id]
                hosts_json = HostList.from_hosts(hosts_for_node).model_dump_json()

                with open(coordination_file, "w") as f:
                    _ = f.write(hosts_json)

                logger.info(
                    f"rank {rank} hostfile: {coordination_file} hosts: {hosts_json}"
                )

                os.environ["MLX_HOSTFILE"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_RING_VERBOSE"] = "1"
                group = mx.distributed.init(backend="ring", strict=True)

            case MlxJacclInstance(
                jaccl_devices=jaccl_devices, jaccl_coordinators=jaccl_coordinators
            ):
                assert all(
                    jaccl_devices[i][i] is None for i in range(len(jaccl_devices))
                )
                # Use RDMA connectivity matrix
                jaccl_devices_json = json.dumps(jaccl_devices)

                with open(coordination_file, "w") as f:
                    _ = f.write(jaccl_devices_json)

                jaccl_coordinator = jaccl_coordinators[bound_instance.bound_node_id]

                logger.info(
                    f"rank {rank} MLX_IBV_DEVICES: {coordination_file} with devices: {jaccl_devices_json}"
                )
                logger.info(f"rank {rank} MLX_JACCL_COORDINATOR: {jaccl_coordinator}")
                os.environ["MLX_IBV_DEVICES"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_JACCL_COORDINATOR"] = jaccl_coordinator
                group = mx.distributed.init(backend="jaccl", strict=True)

            case LlamaRpcInstance():
                # Multi-node llama.cpp (#328) never reaches the MLX runner:
                # its driver/donor runners are dispatched off the shard type in
                # bootstrap. Fail loud rather than wedging in a ring init that
                # cannot complete.
                raise RuntimeError(
                    "LlamaRpcInstance cannot initialize an MLX distributed "
                    "group; it is served by the llama_server driver and RPC "
                    "donor runners"
                )

        logger.info(f"Rank {rank} mlx distributed initialization complete")

        return group


def initialize_mlx(
    bound_instance: BoundInstance,
) -> Group:
    # should we unseed it?
    # TODO: pass in seed from params
    mx.random.seed(42)

    assert len(bound_instance.instance.shard_assignments.node_to_runner) > 1, (
        "Tried to initialize mlx for a single node instance"
    )
    return mlx_distributed_init(bound_instance)


def load_mlx_items(
    bound_instance: BoundInstance,
    group: Group | None,
    on_timeout: TimeoutCallback | None,
    on_layer_loaded: LayerLoadedCallback | None,
) -> "tuple[Model, TokenizerWrapper, VisionProcessor | None, dict[str, mx.array] | None, object | None]":
    if group is None:
        logger.info(f"Single device used for {bound_instance.instance}")
        card = bound_instance.bound_shard.model_card
        model_path = build_model_path(
            card.model_id,
            card.source_revision,
            card.artifact_bundle.root if card.artifact_bundle is not None else None,
        )
        start_time = time.perf_counter()
        model, _ = load_model(
            model_path,
            lazy=True,
            strict=False,
            prefer_vlm=_prefers_native_vlm(card),
            trust_remote_code=card.trust_remote_code,
        )
        # Eval layers one by one for progress reporting
        try:
            import skulk.worker.runner.bootstrap as bootstrap

            inner = get_inner_model(model)
            layers = get_layers(inner)
            total = len(layers)
            for i, layer in enumerate(layers):
                if bootstrap.shutdown_requested:
                    logger.info("Aborting model load — shutdown requested")
                    raise InterruptedError("Model load interrupted by shutdown signal")
                mx.eval(layer)  # type: ignore
                if on_layer_loaded is not None:
                    on_layer_loaded(i, total)
        except ValueError as e:
            logger.opt(exception=e).debug(
                "Model architecture doesn't support layer-by-layer progress tracking",
            )
        mx.eval(model)
        end_time = time.perf_counter()
        logger.info(f"Time taken to load model: {(end_time - start_time):.2f}s")
        tokenizer = get_tokenizer(model_path, bound_instance.bound_shard)

    else:
        logger.info("Starting distributed init")
        start_time = time.perf_counter()
        model, tokenizer = shard_and_load(
            bound_instance.bound_shard,
            group=group,
            on_timeout=on_timeout,
            on_layer_loaded=on_layer_loaded,
        )
        end_time = time.perf_counter()
        logger.info(
            f"Time taken to shard and load model: {(end_time - start_time):.2f}s"
        )

    set_wired_limit_for_model(get_weights_size(bound_instance.bound_shard))

    mx.clear_cache()

    vision_config = bound_instance.bound_shard.model_card.vision

    if vision_config is not None:
        from skulk.worker.engines.mlx.vision import VisionProcessor

        vision_processor: VisionProcessor | None = VisionProcessor(
            vision_config,
            bound_instance.bound_shard.model_card.model_id,
            bound_instance.bound_shard.model_card.source_revision,
            bound_instance.bound_shard.model_card.artifact_repository,
            (
                bound_instance.bound_shard.model_card.artifact_bundle.root
                if bound_instance.bound_shard.model_card.artifact_bundle is not None
                else None
            ),
        )
    else:
        vision_processor = None

    mtp_weights: dict[str, mx.array] | None = None
    runtime = bound_instance.bound_shard.model_card.runtime
    # Multi-node speculation drafts on ONE rank (the last) and fans both the
    # draft tokens and the accept decision out via fixed-shape collectives
    # (#254): rank-symmetric drafting relied on every rank reproducing
    # bit-identical decisions, which heterogeneous chips (M5 vs M4 GEMM
    # kernels, NAX reduced precision) violate — desynced ranks abandon the
    # pipeline collectives and SIGABRT in the Metal completion block. Only
    # the decider rank pays drafter memory; other ranks receive decisions.
    single_node = group is None or group.size() <= 1
    # Cards can forbid speculation on multi-node placements outright
    # (measured slower than plain distributed decode — e.g. sharded MoE);
    # skip the drafter weights entirely so no rank pays the memory.
    speculation_blocked_for_placement = multi_node_speculation_disabled(
        runtime, 1 if group is None else group.size()
    )
    if speculation_blocked_for_placement and runtime is not None:
        logger.info(
            "Speculative decoding disabled for this placement by the model "
            f"card (speculative_multi_node=false, world_size={group.size() if group else 1}); "
            "running plain distributed decode (single-node placements of "
            "this model keep speculation)."
        )
    # Assistants on pipeline placements load on the LAST rank only (#201
    # Track 2b): the assistant cross-attends the target's last
    # full-attention and sliding KV layers, which live in the final slice —
    # other ranks join the per-round draft exchange without paying
    # assistant memory.
    bound_shard = bound_instance.bound_shard
    is_last_pipeline_rank = (
        isinstance(bound_shard, PipelineShardMetadata)
        and bound_shard.device_rank == bound_shard.world_size - 1
    )
    assistant_placement_ok = (
        single_node
        or isinstance(bound_shard, TensorShardMetadata)
        or is_last_pipeline_rank
    ) and not speculation_blocked_for_placement
    # Sidecar load envelope: pipeline loads on the decider rank only (#254);
    # tensor loads on every rank for rank-symmetric drafting (#263) — full
    # rationale on sidecar_load_eligible.
    sidecar_placement_ok = sidecar_load_eligible(
        bound_shard,
        single_node=single_node,
        speculation_blocked=speculation_blocked_for_placement,
    )
    if (
        runtime
        and runtime.mtp_sidecar_repo
        and runtime.mtp_heads
        and sidecar_placement_ok
    ):
        # Sidecar repos carry only mtp.safetensors (no config.json), so they
        # must be resolved with the sidecar resolver — build_model_path's
        # model-completeness check rejects their directories.
        sidecar_model_id, sidecar_revision = companion_artifact_location(
            bound_instance.bound_shard.model_card,
            runtime.mtp_sidecar_repo,
            runtime.mtp_sidecar_revision,
        )
        mtp_safetensors = build_sidecar_path(
            sidecar_model_id,
            "mtp.safetensors",
            sidecar_revision,
        )
        if mtp_safetensors is not None:
            mtp_weights = cast("dict[str, mx.array]", mx.load(str(mtp_safetensors)))
            logger.info(f"MTP sidecar weights loaded from {mtp_safetensors}")
        else:
            logger.warning(
                f"MTP sidecar repo {runtime.mtp_sidecar_repo!r} not downloaded; "
                "running without MTP"
            )
    elif (
        runtime
        and runtime.mtp_sidecar_repo
        and runtime.mtp_heads
        and not speculation_blocked_for_placement
    ):
        logger.info(
            "MTP sidecar declared; this pipeline rank "
            f"({bound_shard.device_rank}/{bound_shard.world_size}) skips the "
            "sidecar load — drafting runs on the decider (last) rank only and "
            "drafts + accept decisions arrive via the per-round exchange (#254)"
        )
    assistant_model: object | None = None
    if runtime and runtime.assistant_model_repo and assistant_placement_ok:
        # Gemma 4 assistant drafter (gemma4-mtp Phase C). Same placement
        # envelope as MTP sidecars (#200/#201).
        assistant_model_id, assistant_revision = companion_artifact_location(
            bound_instance.bound_shard.model_card,
            runtime.assistant_model_repo,
            runtime.assistant_model_revision,
        )
        assistant_dir = build_companion_model_path(
            assistant_model_id,
            assistant_revision,
        )
        if assistant_dir is not None:
            assistant_model = load_assistant_model(assistant_dir)
        else:
            logger.warning(
                f"Assistant model repo {runtime.assistant_model_repo!r} not "
                "downloaded; assistant drafting disabled (MTP sidecar "
                "speculation, if configured, is unaffected)"
            )
    elif runtime and runtime.assistant_model_repo:
        logger.info(
            "Assistant model declared; this pipeline rank "
            f"({bound_shard.device_rank}/{bound_shard.world_size}) skips the "
            "assistant load — drafting runs on the last rank only and drafts "
            "arrive via the per-round exchange (#201 Track 2b)"
        )

    return cast(Model, model), tokenizer, vision_processor, mtp_weights, assistant_model


def shard_and_load(
    shard_metadata: ShardMetadata,
    group: Group,
    on_timeout: TimeoutCallback | None,
    on_layer_loaded: LayerLoadedCallback | None,
) -> tuple[nn.Module, TokenizerWrapper]:
    model_path = build_model_path(
        shard_metadata.model_card.model_id,
        shard_metadata.model_card.source_revision,
        (
            shard_metadata.model_card.artifact_bundle.root
            if shard_metadata.model_card.artifact_bundle is not None
            else None
        ),
    )

    model, _ = load_model(
        model_path,
        lazy=True,
        strict=False,
        prefer_vlm=_prefers_native_vlm(shard_metadata.model_card),
        trust_remote_code=shard_metadata.model_card.trust_remote_code,
    )
    logger.debug(model)
    if hasattr(model, "model") and isinstance(model.model, DeepseekV3Model):  # type: ignore
        pass
        # TODO: See if we should quantize the model.
        # def is_attention_layer(path: str) -> bool:
        #     path = path.lower()

        #     return "self_attn" in path and "layernorm" not in path

        # def quant_predicate(path: str, module: nn.Module):
        #     if not isinstance(module, nn.Linear):
        #         return False

        #     return is_attention_layer(path)
        # model, config = quantize_model(
        #        model, config, group_size=KV_GROUP_SIZE, bits=ATTENTION_KV_BITS, quant_predicate=quant_predicate, mode=QUANTIZE_MODEL_MODE
        #    )

    assert isinstance(model, nn.Module)

    tokenizer = get_tokenizer(model_path, shard_metadata)

    logger.info(f"Group size: {group.size()}, group rank: {group.rank()}")

    # Estimate timeout based on model size (5x default for large queued workloads)
    base_timeout = float(os.environ.get("SKULK_MODEL_LOAD_TIMEOUT", "300"))
    model_size = get_weights_size(shard_metadata)
    timeout_seconds = base_timeout + model_size.in_gb
    logger.info(
        f"Evaluating model parameters with timeout of {timeout_seconds:.0f}s "
        f"(model size: {model_size.in_gb:.1f}GB)"
    )

    match shard_metadata:
        case TensorShardMetadata():
            logger.info(f"loading model from {model_path} with tensor parallelism")
            generation_model = model
            sharding_model = (
                model.tensor_parallel_target()
                if isinstance(model, _VlmModelWrapper)
                else model
            )
            model = tensor_auto_parallel(
                sharding_model,
                group,
                timeout_seconds,
                on_timeout,
                on_layer_loaded,
                generation_model=generation_model,
            )
        case PipelineShardMetadata():
            logger.info(f"loading model from {model_path} with pipeline parallelism")
            model = pipeline_auto_parallel(
                model, group, shard_metadata, on_layer_loaded=on_layer_loaded
            )
            eval_with_timeout(model.parameters(), timeout_seconds, on_timeout)
        case CfgShardMetadata():
            raise ValueError(
                "CfgShardMetadata is not supported for text model loading - "
                "this metadata type is only for image generation models"
            )
        case RpcDonorShardMetadata():
            raise ValueError(
                "RpcDonorShardMetadata never loads a model - an RPC donor "
                "(#328) serves ggml-rpc-server and holds no layers"
            )

    # TODO: Do we need this?
    mx.eval(model)

    logger.debug("SHARDED")
    logger.debug(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model, tokenizer


def get_tokenizer(model_path: Path, shard_metadata: ShardMetadata) -> TokenizerWrapper:
    """Load tokenizer for a model shard. Delegates to load_tokenizer_for_model_id."""
    return load_tokenizer_for_model_id(
        shard_metadata.model_card.model_id,
        model_path,
        model_card=shard_metadata.model_card,
        trust_remote_code=shard_metadata.model_card.trust_remote_code,
    )


def get_eos_token_ids_for_model(model_id: ModelId) -> list[int] | None:
    """
    Get the EOS token IDs for a model based on its ID.

    Some models require explicit EOS token configuration that isn't in their
    tokenizer config. This function returns the known EOS token IDs for such models.

    Args:
        model_id: The HuggingFace model ID

    Returns:
        List of EOS token IDs, or None if the model uses standard tokenizer config
    """
    model_id_lower = model_id.lower()
    if "kimi-k2" in model_id_lower:
        return [163586]
    elif "glm-5" in model_id_lower or "glm-4.7" in model_id_lower:
        # For GLM-5 and GLM-4.7
        # 154820: <|endoftext|>, 154827: <|user|>, 154829: <|observation|>
        return [154820, 154827, 154829]
    elif "glm" in model_id_lower:
        # For GLM-4.5 and older
        return [151336, 151329, 151338]
    elif "gpt-oss" in model_id_lower:
        return [200002, 200012]
    elif "qwen3.5" in model_id_lower or "qwen-3.5" in model_id_lower:
        # For Qwen3.5: 248046 (<|im_end|>), 248044 (<|endoftext|>)
        return [248046, 248044]
    elif "moonlight" in model_id_lower:
        # Moonlight's tokenizer eos_token is [EOS] (163585), but its chat
        # template ends assistant turns with <|im_end|> (163586). Without the
        # latter in the eos set, <|im_end|> is generated as an ordinary token,
        # leaks into the output, and generation runs past the turn boundary
        # into hallucinated follow-on turns (Skulk #304). Moonlight shares the
        # Kimi tokenizer family, so 163586 is the same <|im_end|> id used for
        # kimi-k2 above. Keep [EOS] too so the model's own stop token still works.
        return [163585, 163586]
    return None


def load_tokenizer_for_model_id(
    model_id: ModelId,
    model_path: Path,
    *,
    model_card: ModelCard | None = None,
    trust_remote_code: bool = TRUST_REMOTE_CODE,
) -> TokenizerWrapper:
    """
    Load tokenizer for a model given its ID and local path.

    This is the core tokenizer loading logic, handling special cases for different
    model families (Kimi, GLM, etc.) and transformers 5.x compatibility.

    Args:
        model_id: The HuggingFace model ID (e.g., "moonshotai/Kimi-K2-Instruct")
        model_path: Local path where the model/tokenizer files are stored

    Returns:
        TokenizerWrapper instance configured for the model
    """
    model_id_lower = model_id.lower()
    eos_token_ids = get_eos_token_ids_for_model(model_id)
    model_card = model_card or get_card(model_id)
    capability_profile = resolve_model_capability_profile(
        model_id,
        model_card=model_card,
    )

    # Kimi uses a custom TikTokenTokenizer that transformers 5.x can't load via AutoTokenizer
    if "kimi-k2" in model_id_lower:
        import importlib.util
        import types

        sys.path.insert(0, str(model_path))

        # Load tool_declaration_ts first (tokenization_kimi imports it with relative import)
        tool_decl_path = model_path / "tool_declaration_ts.py"
        if tool_decl_path.exists():
            spec = importlib.util.spec_from_file_location(
                "tool_declaration_ts", tool_decl_path
            )
            if spec and spec.loader:
                tool_decl_module = importlib.util.module_from_spec(spec)
                sys.modules["tool_declaration_ts"] = tool_decl_module
                spec.loader.exec_module(tool_decl_module)

        # Load tokenization_kimi with patched source (convert relative to absolute import)
        tok_path = model_path / "tokenization_kimi.py"
        source = tok_path.read_text()
        source = source.replace("from .tool_declaration_ts", "from tool_declaration_ts")
        spec = importlib.util.spec_from_file_location("tokenization_kimi", tok_path)
        if spec:
            tok_module = types.ModuleType("tokenization_kimi")
            tok_module.__file__ = str(tok_path)
            sys.modules["tokenization_kimi"] = tok_module
            exec(compile(source, tok_path, "exec"), tok_module.__dict__)  # noqa: S102
            TikTokenTokenizer = tok_module.TikTokenTokenizer  # type: ignore[attr-defined]  # noqa: N806
        else:
            from tokenization_kimi import TikTokenTokenizer  # type: ignore[import-not-found]  # noqa: I001

        hf_tokenizer: Any = TikTokenTokenizer.from_pretrained(model_path)  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

        # Patch encode to use internal tiktoken model directly
        # transformers 5.x has a bug in the encode->pad path for slow tokenizers
        def _patched_encode(text: str, **_kwargs: object) -> list[int]:
            # Pass allowed_special="all" to handle special tokens like <|im_user|>
            return list(hf_tokenizer.model.encode(text, allowed_special="all"))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

        hf_tokenizer.encode = _patched_encode
        return TokenizerWrapper(
            hf_tokenizer,
            eos_token_ids=eos_token_ids,
            tool_call_start="<|tool_calls_section_begin|>",
            tool_call_end="<|tool_calls_section_end|>",
            tool_parser=_parse_kimi_tool_calls,
        )

    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config_extra={"trust_remote_code": trust_remote_code},
        eos_token_ids=eos_token_ids,
    )

    if "gemma-3" in model_id_lower or "gemma-4" in model_id_lower:
        gemma_eos_id = 1
        gemma_end_of_turn_id = 106
        if tokenizer.eos_token_ids is not None:
            if gemma_end_of_turn_id not in tokenizer.eos_token_ids:
                tokenizer.eos_token_ids = list(tokenizer.eos_token_ids) + [
                    gemma_end_of_turn_id
                ]
        else:
            tokenizer.eos_token_ids = [gemma_eos_id, gemma_end_of_turn_id]

    # Llama 3.1+ ends a tool-calling turn with <|eom_id|> ("end of message",
    # handing off to a tool) and a user-facing turn with <|eot_id|> ("end of
    # turn"). Only <|eot_id|> reaches us from tokenizer_config, because
    # generation_config carries no eos_token_id for these repos, so without
    # this the model runs straight past the end of its own tool call: the
    # scaffolding detokenizes into visible content and a second call begins.
    # Upstream (Meta's reference, vLLM and llama.cpp) all stop on both.
    # Detected by vocabulary rather than by the template mentioning the token:
    # Llama 3.2's template never writes <|eom_id|> or <|python_tag|> literally,
    # it only routes tool results through the "ipython" role, so a template
    # substring check silently misses the family this exists for.
    llama_eom_id = _token_id_or_none(tokenizer, "<|eom_id|>")
    if llama_eom_id is not None:
        # <|eom_id|> is "end of message, handing off to a tool". Llama declares
        # only <|eot_id|> as its stop token, so without this the model runs
        # straight past the end of its tool call and generates the next turn's
        # header, and the caller sees control tokens in the answer text.
        existing = list(tokenizer.eos_token_ids or [])
        if llama_eom_id not in existing:
            tokenizer.eos_token_ids = existing + [llama_eom_id]
        if not getattr(tokenizer, "tool_parser", None):
            # Llama writes the call as a bare object with no opening marker, so
            # the block opens on "{" and is closed by the end of the message
            # rather than by a closing marker. The whole-block dialect parser
            # reads both that form and the <|python_tag|> variant.
            object.__setattr__(tokenizer, "_tool_call_start", "{")
            object.__setattr__(tokenizer, "_tool_call_end", "<|eom_id|>")
            from skulk.worker.runner.llm_inference.tool_parsers import (
                UNMARKED_TOOL_DIALECT,
            )

            object.__setattr__(tokenizer, "_tool_parser", UNMARKED_TOOL_DIALECT)

    if capability_profile.tool_call_format == ToolCallFormat.Gemma4:
        # mlx-lm exposes tool-call markers through read-only properties on
        # TokenizerWrapper. Configure the internal fields directly so Gemma 4
        # tool parsing works across mlx-lm versions without mutating the
        # wrapped HF tokenizer.
        object.__setattr__(tokenizer, "_tool_call_start", "<|tool_call>")
        object.__setattr__(tokenizer, "_tool_call_end", "<tool_call|>")
        object.__setattr__(tokenizer, "_tool_parser", _parse_gemma4_tool_calls)

    if capability_profile.tool_call_format == ToolCallFormat.Generic and (
        "[TOOL_CALLS]" in (getattr(tokenizer, "chat_template", None) or "")
    ):
        # Mistral writes `[TOOL_CALLS] [{...}]` and closes the call by ending
        # the message, not with a closing marker, so the end marker is an
        # impossible sentinel that never appears in detokenized text, and the
        # streaming parser's end-of-generation path parses the still-open
        # block when the message finishes, the same mechanism that closes
        # Llama's unmarked dialect.
        #
        # This wiring deliberately REPLACES a parser mlx-lm already assigned:
        # the pinned mlx-lm auto-installs its own Mistral parser for any
        # template containing [TOOL_CALLS], and that parser reads the
        # `NAME[ARGS]{...}` form, rejecting the JSON-array form the 2410-era
        # instruct models actually emit. The override tries the array form
        # first and delegates to the displaced parser on a miss, so models
        # writing the upstream form keep working.
        displaced = cast(
            "Callable[[str], list[dict[str, Any]]] | None",
            getattr(tokenizer, "tool_parser", None),
        )

        def _mistral_with_fallback(text: str) -> list[dict[str, Any]]:
            try:
                return _parse_mistral_tool_calls(text)
            except ValueError:
                if displaced is None:
                    raise
                return displaced(text)

        object.__setattr__(tokenizer, "_tool_call_start", "[TOOL_CALLS]")
        # The end marker is an impossible sentinel, not the EOS literal: the
        # streaming scanner closes a block at the FIRST occurrence of the end
        # marker in accumulated text, and a model can emit a literal "</s>"
        # inside arguments or prose, which would truncate a valid call. The
        # sentinel can never occur in detokenized text, so the block closes
        # only at end of generation.
        object.__setattr__(
            tokenizer, "_tool_call_end", _MISTRAL_IMPOSSIBLE_END
        )
        object.__setattr__(tokenizer, "_tool_parser", _mistral_with_fallback)

    if (
        capability_profile.tool_call_format == ToolCallFormat.Generic
        and not getattr(tokenizer, "tool_parser", None)
        and "<tool_call>" in (getattr(tokenizer, "chat_template", None) or "")
    ):
        # #728: tokenizers that reach this path without mlx-lm's inferred
        # tool parser (notably the mlx-vlm native-vision loaders, which is
        # every natively-multimodal family) silently pass tool-call markup
        # through as text. When the chat template speaks the <tool_call>
        # dialect, wire the repo's shared text parser (Qwen3 XML plus
        # Hermes JSON, the same one the llama_cpp engine uses) through the
        # standard marker mechanism. Template truth decides, so models
        # whose templates use other dialects are untouched.
        object.__setattr__(tokenizer, "_tool_call_start", "<tool_call>")
        object.__setattr__(tokenizer, "_tool_call_end", "</tool_call>")
        object.__setattr__(tokenizer, "_tool_parser", _parse_generic_text_tool_calls)

    return tokenizer


# Never occurs in detokenized text; see the wiring comment above.
_MISTRAL_IMPOSSIBLE_END = "\x00skulk:mistral:end-of-message\x00"


def _mistral_tool_call_id(raw: str) -> str:
    """Map any tool-call id onto Mistral's required nine-alphanumeric form.

    Mistral's chat template hard-fails rendering ("Tool call IDs should be
    alphanumeric strings with length 9!") on any other shape, and Skulk mints
    UUID-style ids, so a tool-result round trip could never render: the
    template raised and killed the runner, observed live at the first
    round-trip request. A digest rather than a truncation: ids differing only
    past the ninth character (parallel calls with sequential caller-minted
    ids) must not collide, and the mapping must be deterministic ACROSS
    requests, because callers echo ids back in later turns, which rules out
    request-local bijections.
    """
    import hashlib

    return hashlib.sha256(raw.encode()).hexdigest()[:9]


def _normalize_mistral_tool_call_ids(messages: list[dict[str, Any]]) -> None:
    """Rewrite tool-call ids in history messages for a Mistral template."""
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:  # pyright: ignore[reportUnknownVariableType]
                if isinstance(tool_call, dict) and isinstance(
                    tool_call.get("id"), str  # pyright: ignore[reportUnknownMemberType]
                ):
                    tool_call["id"] = _mistral_tool_call_id(
                        cast("str", tool_call["id"])
                    )
        tool_call_id = msg.get("tool_call_id")
        if isinstance(tool_call_id, str):
            msg["tool_call_id"] = _mistral_tool_call_id(tool_call_id)


def _normalize_tool_calls(msg_dict: dict[str, Any]) -> None:
    """Normalize tool_calls in a message dict.

    OpenAI format has tool_calls[].function.arguments as a JSON string,
    but some chat templates (e.g., GLM) expect it as a dict.
    """
    tool_calls = msg_dict.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        return

    for tc in tool_calls:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(tc, dict):
            continue
        func = tc.get("function")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if not isinstance(func, dict):
            continue
        args = func.get("arguments")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if isinstance(args, str):
            with contextlib.suppress(json.JSONDecodeError):
                func["arguments"] = json.loads(args)


def _collect_nested_property_names(schema: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    properties: dict[str, Any] = schema.get("properties", {})  # type: ignore[reportAny]
    for prop_spec in properties.values():  # pyright: ignore[reportAny]
        if not isinstance(prop_spec, dict):
            continue
        if prop_spec.get("type") == "array":  # type: ignore[reportAny]
            items: dict[str, Any] | None = prop_spec.get("items")  # type: ignore[reportAny]
            if isinstance(items, dict) and items.get("type") == "object":  # type: ignore[reportAny]
                inner_props: dict[str, Any] = items.get("properties", {})  # type: ignore[reportAny]
                for k in inner_props:  # pyright: ignore[reportUnknownVariableType]
                    names.add(str(k))  # pyright: ignore[reportUnknownArgumentType]
                names.update(_collect_nested_property_names(items))  # pyright: ignore[reportUnknownArgumentType]
    return names


def _schemas_lost_in_prompt(prompt: str, tools: list[dict[str, Any]]) -> bool:
    """Return True if nested property names from any tool schema are absent."""
    for tool in tools:
        fn: dict[str, Any] = tool.get("function", {})  # type: ignore
        params: dict[str, Any] = fn.get("parameters", {})  # type: ignore
        nested = _collect_nested_property_names(params)
        if nested and not all(name in prompt for name in nested):
            return True
    return False


_LOSSY_TEMPLATE_PATTERN = re.compile(
    r"""inner_type\s*==\s*["']object \| object["']\s*or\s*inner_type\|length\s*>\s*\d+""",
)


def _patch_lossy_chat_template(template: str) -> str | None:
    """Patch chat templates that collapse nested object schemas to ``any[]``.

    Some templates (e.g., GPT-OSS) have a guard like::

        inner_type == "object | object" or inner_type|length > 50

    The length check silently drops complex array-of-object schemas.
    We remove the length guard, keeping only the object-union check.
    Returns the patched template, or *None* if no patch was needed.
    """
    patched, n = _LOSSY_TEMPLATE_PATTERN.subn(
        lambda m: m.group(0).split(" or ")[0],  # keep only the object-union check
        template,
    )
    return patched if n > 0 else None


def _log_rendered_prompt_shape(
    renderer: PromptRendererType,
    prompt: str,
    message_count: int,
) -> None:
    """Log prompt construction metadata without retaining user content."""
    logger.info(
        "Rendered model prompt "
        f"(renderer={renderer.value}, messages={message_count}, chars={len(prompt)})"
    )


def apply_chat_template(
    tokenizer: TokenizerWrapper,
    task_params: TextGenerationTaskParams,
    model_card: ModelCard | None = None,
    suppress_empty_gemma4_thought_channel: bool = False,
) -> str:
    """Convert TextGenerationTaskParams to a chat template prompt.

    Converts the internal format (input + instructions) to a messages list
    that can be processed by the tokenizer's chat template.

    When chat_template_messages is available (from Chat Completions API),
    uses those directly to preserve tool_calls, thinking, and other fields.
    """
    formatted_messages: list[dict[str, Any]] = []
    if task_params.chat_template_messages is not None:
        # Use pre-formatted messages that preserve tool_calls, thinking, etc.
        formatted_messages = list(task_params.chat_template_messages)
    else:
        # Add system message (instructions) if present
        if task_params.instructions:
            formatted_messages.append(
                {"role": "system", "content": task_params.instructions}
            )

        # Convert input to messages
        for msg in task_params.input:
            if not msg.content:
                logger.warning("Received message with empty content, skipping")
                continue
            formatted_messages.append({"role": msg.role, "content": msg.content})

    # For assistant prefilling, append content after templating to avoid a closing turn token.
    partial_assistant_content: str | None = None
    if formatted_messages and formatted_messages[-1].get("role") == "assistant":
        partial_assistant_content = cast(str, formatted_messages[-1].get("content", ""))
        formatted_messages = formatted_messages[:-1]

    capability_profile = resolve_model_capability_profile(
        task_params.model,
        model_card=model_card,
        tokenizer=tokenizer,
        task_params=task_params,
    )

    formatted_messages = _apply_nemotron_reasoning_controls(
        formatted_messages,
        enable_thinking=task_params.enable_thinking,
        family=capability_profile.family,
    )

    if capability_profile.prompt_renderer == PromptRendererType.Dsml:
        from skulk.worker.engines.mlx.dsml_encoding import encode_messages

        prompt = encode_messages(
            messages=formatted_messages,
            # Only use chat mode if enable thinking is explicitly False.
            thinking_mode="chat"
            if task_params.enable_thinking is False
            else "thinking",
            tools=task_params.tools,
        )
        if partial_assistant_content:
            prompt += partial_assistant_content
        _log_rendered_prompt_shape(
            capability_profile.prompt_renderer,
            prompt,
            len(formatted_messages),
        )
        return prompt

    if capability_profile.prompt_renderer == PromptRendererType.Gemma4:
        prompt = render_gemma4_prompt(
            formatted_messages,
            add_generation_prompt=True,
            enable_thinking=task_params.enable_thinking,
            suppress_empty_thought_channel=suppress_empty_gemma4_thought_channel,
        )
        if partial_assistant_content:
            prompt += partial_assistant_content
        _log_rendered_prompt_shape(
            capability_profile.prompt_renderer,
            prompt,
            len(formatted_messages),
        )
        return prompt

    for msg in formatted_messages:
        _normalize_tool_calls(msg)

    if "[TOOL_CALLS]" in (getattr(tokenizer, "chat_template", None) or ""):
        _normalize_mistral_tool_call_ids(formatted_messages)

    extra_kwargs: dict[str, Any] = {}
    if task_params.enable_thinking is not None:
        # Qwen3 and GLM use "enable_thinking"; DeepSeek uses "thinking".
        # Jinja ignores unknown variables, so passing both is safe.
        extra_kwargs["enable_thinking"] = task_params.enable_thinking
        extra_kwargs["thinking"] = task_params.enable_thinking
    if task_params.reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = task_params.reasoning_effort
    # Muse Glimmer steers its always-on reasoning with a strength level rather
    # than a toggle or the harmony effort field.
    extra_kwargs.update(
        muse_glimmer_template_kwargs(capability_profile, task_params.reasoning_effort)
    )

    patched_template: str | None = None
    if task_params.tools:
        original_template: str | None = getattr(tokenizer, "chat_template", None)
        if isinstance(original_template, str):
            patched_template = _patch_lossy_chat_template(original_template)
            if patched_template is not None:
                logger.info(
                    "Patched lossy chat template (removed inner_type length guard)"
                )

    prompt: str = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=task_params.tools,
        **({"chat_template": patched_template} if patched_template is not None else {}),
        **extra_kwargs,
    )

    if task_params.tools and _schemas_lost_in_prompt(prompt, task_params.tools):
        logger.warning("Chat template lost nested tool schemas even after patching")

    if partial_assistant_content:
        prompt += partial_assistant_content

    _log_rendered_prompt_shape(
        capability_profile.prompt_renderer,
        prompt,
        len(formatted_messages),
    )

    return prompt


def system_prompt_token_count(
    task_params: TextGenerationTaskParams,
    tokenizer: TokenizerWrapper,
) -> int:
    """Approximate token count of the system prompt portion of the input."""
    parts: list[str] = []
    if task_params.chat_template_messages is not None:
        for msg in task_params.chat_template_messages:
            if msg.get("role") in ("system", "developer"):
                content = msg.get("content", "")  # type: ignore
                if isinstance(content, str):
                    parts.append(content)
    else:
        if task_params.instructions:
            parts.append(task_params.instructions)
        for msg in task_params.input:
            if msg.role in ("system", "developer"):
                parts.append(msg.content)
    if len(parts) == 0:
        return 0
    return len(tokenizer.encode(" ".join(parts), add_special_tokens=False))


def detect_thinking_prompt_suffix(prompt: str, tokenizer: TokenizerWrapper) -> bool:
    """
    Detect if prompt ends with a thinking opening tag that should be
    prepended to the output stream.
    """
    think_token = tokenizer.think_start

    return think_token is not None and prompt.rstrip().endswith(think_token)


def fix_unmatched_think_end_tokens(
    tokens: mx.array, tokenizer: TokenizerWrapper
) -> mx.array:
    if not tokenizer.has_thinking:
        return tokens
    try:
        maybe_start_id: int | None = tokenizer.think_start_id
        maybe_end_id: int | None = tokenizer.think_end_id
    except ValueError:
        # mlx-lm raises when a tokenizer's think marker is not a single
        # token (observed: gemma-4-12B-it's unified tokenizer, launch E2E
        # smoke 2026-06-05). The unmatched-end fix only makes sense for
        # single-token markers — skip it rather than crash the runner
        # during warmup of an otherwise healthy model.
        logger.info(
            "Thinking markers are multi-token for this tokenizer; "
            "skipping unmatched think-end token fix"
        )
        return tokens
    if maybe_start_id is None or maybe_end_id is None:
        return tokens
    think_start_id: int = maybe_start_id
    think_end_id: int = maybe_end_id
    token_list: list[int] = cast(list[int], tokens.tolist())
    result: list[int] = []
    depth = 0
    for token in token_list:
        if token == think_start_id:
            depth += 1
        elif token == think_end_id:
            if depth == 0:
                result.append(think_start_id)
            else:
                depth -= 1
        result.append(token)
    return mx.array(result)


class NullKVCache(KVCache):
    """
    A KVCache that pretends to exist but holds zero tokens.
    It satisfies .state/.meta_state and never allocates real keys/values.
    """

    def __init__(self, dtype: mx.Dtype = mx.float16):
        super().__init__()
        # zero-length K/V so shapes/dtypes are defined but empty
        self.keys = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.values = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.offset = 0

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        # matches what mx.save_safetensors / mx.eval expect
        assert self.keys is not None and self.values is not None
        return self.keys, self.values

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        raise NotImplementedError("We should not be setting a NullKVCache.")


def mlx_force_oom(size: int = 200000) -> None:
    """
    Force an Out-Of-Memory (OOM) error in MLX by performing large tensor operations.
    """
    mx.set_default_device(mx.gpu)
    a = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    b = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    d = mx.matmul(a, c)
    e = mx.matmul(b, c)
    f = mx.sigmoid(d + e)
    mx.eval(f)


def set_wired_limit_for_model(model_size: Memory):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        return

    max_rec_size = Memory.from_bytes(
        int(mx.device_info()["max_recommended_working_set_size"])
    )
    if model_size > 0.9 * max_rec_size:
        logger.warning(
            f"Generating with a model that requires {model_size.in_float_mb:.1f} MB "
            f"which is close to the maximum recommended size of {max_rec_size.in_float_mb:.1f} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    mx.set_wired_limit(max_rec_size.in_bytes)
    remember_wired_limit_bytes(max_rec_size.in_bytes)
    logger.info(f"Wired limit set to {max_rec_size}.")


def mlx_cleanup(
    model: Model | None, tokenizer: TokenizerWrapper | None, group: Group | None
) -> None:
    del model, tokenizer, group
    mx.clear_cache()
    import gc

    gc.collect()


def mx_any(bool_: bool, group: Group | None) -> bool:
    if group is None:
        return bool_
    num_true = mx.distributed.all_sum(
        mx.array(bool_), group=group, stream=mx.default_stream(mx.Device(mx.cpu))
    )
    mx.eval(num_true)
    return num_true.item() > 0


def mx_barrier(group: Group | None):
    if group is None:
        return
    # Wrap the all_sum eval in eval_with_timeout so a desynced ring-collective
    # state cannot wedge the runner indefinitely. Healthy barriers complete in
    # milliseconds; the slowest legitimate observation is the post-prefill
    # decode barrier at ~1s. Bound by SKULK_PIPELINE_EVAL_TIMEOUT_SECONDS so
    # all distributed evals share the same recovery floor.
    eval_with_timeout(
        mx.distributed.all_sum(
            mx.array(1.0), group=group, stream=mx.default_stream(mx.Device(mx.cpu))
        ),
        timeout_seconds=pipeline_eval_timeout_seconds(),
        on_timeout=pipeline_timeout_callback(
            "mx_barrier", {"group_size": group.size()}, is_prefill=False
        ),
    )


def _token_id_or_none(tokenizer: object, token: str) -> int | None:
    """Return a special token's id, or None when the tokenizer lacks it.

    Vocabularies differ across quantizations and conversions, so a missing
    token is normal and must not raise.
    """

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    try:
        token_id = cast("object", convert(token))
    except Exception:  # noqa: BLE001 - tokenizer implementations vary
        return None
    if not isinstance(token_id, int):
        return None
    unknown = getattr(tokenizer, "unk_token_id", None)
    if token_id < 0 or (unknown is not None and token_id == unknown):
        return None
    return token_id


def _parse_generic_text_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse generic-format tool calls (Qwen3 XML or Hermes JSON) from text.

    Receives the inner text between the ``<tool_call>`` markers (the runner's
    marker mechanism strips them) and delegates to the shared text parser so
    the MLX lane recognizes exactly the formats the llama_cpp engine does.
    Argument values arrive as JSON strings, the shape ``ToolCallItem``
    validation expects.
    """
    from skulk.worker.runner.llm_inference.tool_text_parser import (
        parse_tool_calls_from_text,
    )

    items = parse_tool_calls_from_text(f"<tool_call>{text}</tool_call>")
    if not items:
        # Raising routes the runner to its malformed-tool-call fallback
        # (raw text with finish_reason="error"), matching the behavior of
        # every other parser instead of fabricating an empty success.
        raise ValueError("no recognized tool calls in block")
    return [{"name": item.name, "arguments": item.arguments} for item in items]


def _parse_mistral_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse Mistral ``[TOOL_CALLS]`` arrays from model output.

    Receives the text after the ``[TOOL_CALLS]`` marker (the runner's marker
    mechanism strips it) and delegates to the shared text parser's Mistral
    branch, so the MLX lane recognizes exactly the format the llama_cpp
    engine's recovery path does. Argument values arrive as JSON strings, the
    shape ``ToolCallItem`` validation expects.
    """
    from skulk.worker.runner.llm_inference.tool_text_parser import (
        parse_tool_calls_from_text,
    )

    items = parse_tool_calls_from_text(f"[TOOL_CALLS]{text}")
    if not items:
        # Raising routes the runner to its malformed-tool-call fallback (raw
        # text with finish_reason="error"), matching every other parser
        # instead of fabricating an empty success.
        raise ValueError("no recognized tool calls in block")
    return [{"name": item.name, "arguments": item.arguments} for item in items]


def _parse_gemma4_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse Gemma 4 tool calls from model output.

    Delegates to the shared text parser's Gemma 4 dialect
    (``tool_text_parser.gemma4_calls``), so the MLX lane and the llama_cpp
    recovery path read exactly the same format, and converts the JSON-string
    arguments back to the dict shape this engine's marker mechanism expects.
    """
    from skulk.worker.runner.llm_inference.tool_text_parser import gemma4_calls

    results: list[dict[str, Any]] = [
        {"name": item.name, "arguments": cast(dict[str, object], json.loads(item.arguments))}
        for item in gemma4_calls(text)
    ]
    if not results:
        raise ValueError(
            f"No Gemma 4 tool calls found in generated text ({len(text)} chars)"
        )
    return results


def _parse_kimi_tool_calls(text: str):
    import regex as re

    # kimi has a fixed function naming scheme, with a json formatted arg
    #   functions.multiply:0<|tool_call_argument_begin|>{"a": 2, "b": 3}
    _func_name_regex = re.compile(
        r"^\s*((?:functions\.)?(.+?):\d+)\s*<\|tool_call_argument_begin\|>", re.DOTALL
    )
    _func_arg_regex = re.compile(r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL)
    _tool_call_split_regex = re.compile(
        r"<\|tool_call_begin\|>(.*?)<\|tool_call_end\|>", re.DOTALL
    )

    def _parse_single_tool(text: str) -> dict[str, Any]:
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError("No tool call found.")
        tool_call_id = func_name_match.group(1)  # e.g. "functions.get_weather:0"
        func_name = func_name_match.group(2)  # e.g. "get_weather"

        func_args_match = _func_arg_regex.search(text)
        if func_args_match is None:
            raise ValueError("No tool call arguments found.")
        func_args = func_args_match.group(1)
        arg_dct = json.loads(func_args)  # pyright: ignore[reportAny]

        return dict(id=tool_call_id, name=func_name, arguments=arg_dct)  # pyright: ignore[reportAny]

    tool_matches = _tool_call_split_regex.findall(text)
    if tool_matches:
        return [_parse_single_tool(match) for match in tool_matches]  # pyright: ignore[reportAny]
    else:
        return [_parse_single_tool(text)]


def mx_all_gather_tasks(
    tasks: list[TextGeneration],
    group: mx.distributed.Group | None,
    *,
    preserve_rank_zero_order: bool = False,
) -> tuple[list[TextGeneration], list[TextGeneration]]:
    """Agree on tasks present on every rank.

    Args:
        tasks: Locally observed candidate tasks.
        group: MLX distributed group, or ``None`` for one rank.
        preserve_rank_zero_order: Use rank zero's arrival order as the canonical
            order for agreed tasks. The default retains the historical
            task-ID-sorted behavior.

    Returns:
        Agreed local task objects in canonical order and candidates not yet
        observed on every rank.
    """
    def encode_task_id(task_id: TaskId) -> list[int]:
        utf8_task_id = task_id.encode()
        return [
            int.from_bytes(utf8_task_id[i : i + 1]) for i in range(len(utf8_task_id))
        ]

    def decode_task_id(encoded_task_id: list[int]) -> TaskId:
        return TaskId(
            bytes.decode(b"".join((x).to_bytes(length=1) for x in encoded_task_id))
        )

    uuid_byte_length = 36

    n_tasks = len(tasks)
    all_counts = cast(
        list[int],
        mx.distributed.all_gather(mx.array([n_tasks]), group=group).tolist(),
    )
    max_tasks = max(all_counts)
    world_size: int = 1 if group is None else group.size()

    if max_tasks == 0:
        return [], []

    padded = [encode_task_id(task.task_id) for task in tasks] + [
        [0] * uuid_byte_length
    ] * (max_tasks - n_tasks)

    assert all(len(encoded_task_id) == uuid_byte_length for encoded_task_id in padded)

    gathered = cast(
        list[list[list[int]]],
        mx.distributed.all_gather(mx.array(padded), group=group)
        .reshape(world_size, max_tasks, -1)
        .tolist(),
    )
    all_task_ids: list[list[TaskId]] = [
        [decode_task_id(encoded_task_id) for encoded_task_id in rank_tasks[:count]]
        for rank_tasks, count in zip(gathered, all_counts, strict=True)
    ]

    agreed_ids = set[TaskId].intersection(*(set(tids) for tids in all_task_ids))

    local_tasks = {task.task_id: task for task in tasks}
    canonical_ids = (
        [task_id for task_id in all_task_ids[0] if task_id in agreed_ids]
        if preserve_rank_zero_order
        else sorted(agreed_ids)
    )
    agreed = [local_tasks[tid] for tid in canonical_ids]
    different = [task for task in tasks if task.task_id not in agreed_ids]
    return agreed, different
