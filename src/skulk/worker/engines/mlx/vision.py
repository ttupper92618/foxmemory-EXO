"""Vision encoding pipeline for multimodal (VLM) inference.

Converts images from base64 into vision embeddings that replace image-token
placeholders in the prompt. The pipeline is:

1. **Decode** base64 images to PIL
2. **Preprocess** via the model's HuggingFace image processor
3. **Encode** through the vision tower (+ optional projector) to get features
4. **Expand** image-token placeholders in the prompt to match feature count
5. **Embed** by replacing placeholder token embeddings with vision features

Supports models with bundled vision weights (e.g. Qwen3-VL) and models
with separate vision repos. Results are cached by image content hash to
avoid re-encoding identical images across turns.
"""

import base64
import contextlib
import hashlib
import importlib
import inspect
import io
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast, final

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.tokenizer_utils import TokenizerWrapper
from PIL import Image
from safetensors import safe_open
from transformers import AutoImageProcessor

if TYPE_CHECKING:
    from mlx_vlm.prompt_utils import get_message_json
    from mlx_vlm.utils import load_image_processor
else:
    # mlx-vlm ships only on darwin: its mlx floor (>=0.31.2) is unsatisfiable
    # against the Linux mlx[cpu]==0.30.6 pin. Text-only inference must still
    # import this module, so the dependency is optional at runtime — the two
    # vision entry points below raise only when vision is actually used.
    try:
        from mlx_vlm.prompt_utils import get_message_json
        from mlx_vlm.utils import load_image_processor
    except ImportError:

        def _raise_mlx_vlm_unavailable(*_args, **_kwargs):
            raise RuntimeError(
                "mlx-vlm is not installed (it is a darwin-only dependency); "
                "vision/VLM features are unavailable on this platform"
            )

        get_message_json = _raise_mlx_vlm_unavailable
        load_image_processor = _raise_mlx_vlm_unavailable

from skulk.download.download_utils import build_model_path
from skulk.shared.constants import SKULK_IMAGE_TRANSPORT_DEBUG
from skulk.shared.models.capabilities import muse_glimmer_reasoning_strength
from skulk.shared.models.model_cards import VisionCardConfig
from skulk.shared.tracing import TraceAttrValue
from skulk.shared.types.common import ModelId
from skulk.shared.types.mlx import Model
from skulk.shared.types.tasks import TaskId
from skulk.shared.types.text_generation import ReasoningEffort
from skulk.worker.engines.mlx.cache import encode_prompt
from skulk.worker.engines.mlx.gemma4_prompt import render_gemma4_prompt
from skulk.worker.engines.mlx.utils_mlx import fix_unmatched_think_end_tokens
from skulk.worker.runner.bootstrap import logger
from skulk.worker.runner.diagnostics import record_runner_phase, runner_phase

JsonDict = dict[str, object]
MxArrayInput = (
    mx.array
    | np.ndarray[Any, Any]
    | list[object]
    | tuple[object, ...]
    | int
    | float
    | bool
)


def _object_dict(value: object) -> JsonDict:
    """Return a string-keyed object mapping for dynamic config or processor data.

    Accepts any mapping rather than only ``dict``. Hugging Face image
    processors return ``BatchFeature``, which derives from ``UserDict`` and so
    fails an ``isinstance(value, dict)`` test despite behaving like a mapping.
    Narrowing to ``dict`` turned every processor result into an empty mapping,
    and the caller's ``raw_dict["pixel_values"]`` then raised ``KeyError`` on
    every image the model was sent.
    """
    if isinstance(value, Mapping):
        return {
            str(key): item
            for key, item in cast(Mapping[object, object], value).items()
        }
    return {}


def _module_attr(module: object, name: str) -> object:
    """Read one attribute from a dynamic module through its dict."""
    module_dict = cast(dict[str, object], getattr(module, "__dict__", {}))
    return module_dict[name]


def _int_value(value: object, default: int) -> int:
    """Return ``value`` as an int when possible, otherwise ``default``."""
    return int(value) if isinstance(value, (int, float, str)) else default


class _ImageProcessorProtocol(Protocol):
    """Minimal processor surface used by the vision pipeline."""

    max_soft_tokens: int | None

    def preprocess(
        self,
        messages: list[JsonDict],
        *,
        return_tensors: str,
        **kwargs: object,
    ) -> JsonDict: ...

    def __call__(
        self,
        *,
        images: list[Image.Image],
        return_tensors: str,
        **kwargs: object,
    ) -> object: ...


class _ProcessorContainer(Protocol):
    """Container returned by mlx-vlm processor factories."""

    image_processor: _ImageProcessorProtocol | None


class _FromPretrainedProcessor(Protocol):
    """Processor class surface that supports ``from_pretrained``."""

    @classmethod
    def from_pretrained(cls, repo: str, **kwargs: object) -> _ProcessorContainer: ...


class _ImageProcessorFactory(Protocol):
    """Dynamic Transformers image processor factory surface."""

    def from_pretrained(self, repo: str, **kwargs: object) -> object: ...


def _filter_config(cls: type, d: JsonDict) -> JsonDict:
    valid = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    return {k: v for k, v in d.items() if k in valid}


def _load_mlx_vlm_image_processor_from_pretrained(
    proc_mod: object, repo: str, revision: str | None = None
) -> _ImageProcessorProtocol | None:
    """Load an MLX-VLM processor via ``from_pretrained`` and extract its image processor.

    Gemma 4's standalone ``Gemma4ImageProcessor`` defaults to a low visual token
    budget unless it is initialized from ``processor_config.json``. When
    ``transformers.AutoImageProcessor`` does not recognize the model type, we need
    to follow MLX-VLM's processor loading path so model-specific preprocessing
    settings are preserved.
    """
    for attr_name in dir(proc_mod):
        obj = cast(object, getattr(proc_mod, attr_name))
        if (
            isinstance(obj, type)
            and attr_name.endswith("Processor")
            and "ImageProcessor" not in attr_name
            and "VideoProcessor" not in attr_name
            and hasattr(obj, "from_pretrained")
        ):
            try:
                factory = cast(_FromPretrainedProcessor, obj)
                from_pretrained = cast(
                    Callable[..., _ProcessorContainer],
                    factory.from_pretrained,
                )
                parameters = inspect.signature(from_pretrained).parameters.values()
                supports_trust_remote_code = any(
                    parameter.name == "trust_remote_code"
                    or parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
                loader_options: dict[str, object] = {}
                if supports_trust_remote_code:
                    loader_options["trust_remote_code"] = True
                if revision is not None:
                    loader_options["revision"] = revision
                processor = from_pretrained(repo, **loader_options)
            except Exception as exc:
                logger.info(
                    f"mlx_vlm {attr_name}.from_pretrained failed for {repo}: {exc}"
                )
                continue
            image_processor = processor.image_processor
            if image_processor is not None:
                logger.info(
                    f"Using mlx_vlm {attr_name}.from_pretrained image processor"
                )
                return image_processor
    return None


def _mlx_vlm_processor_modules(model_type: str) -> list[object]:
    """Return available MLX-VLM processor modules for one model family.

    MLX-VLM is not consistent about where it exports the combined processor:
    Qwen 3.5 exposes ``Qwen3VLProcessor`` from the family package while older
    families often keep it in ``processing_<model_type>``. Trying both paths
    keeps image preprocessing on the native MLX stack instead of routing those
    families through the installed Transformers/torchvision fallback.
    """
    module_names = (
        f"mlx_vlm.models.{model_type}",
        f"mlx_vlm.models.{model_type}.processing_{model_type}",
    )
    modules: list[object] = []
    for module_name in module_names:
        try:
            modules.append(importlib.import_module(module_name))
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
    return modules


def _instantiate_mlx_vlm_image_processor(
    processor_modules: list[object],
) -> _ImageProcessorProtocol | None:
    """Instantiate a family image processor when no combined loader succeeds."""
    for proc_mod in processor_modules:
        for attr_name in dir(proc_mod):
            obj = cast(object, getattr(proc_mod, attr_name))
            if isinstance(obj, type) and "ImageProcessor" in attr_name:
                try:
                    processor = cast(Callable[[], object], obj)()
                except TypeError as exc:
                    logger.info(
                        f"mlx_vlm {obj.__name__} requires constructor arguments: {exc}"
                    )
                    continue
                logger.info(f"Using mlx_vlm {obj.__name__} as image processor")
                return cast(_ImageProcessorProtocol, processor)
    return None


def _load_gemma3n_pil_image_processor(
    repo: str, revision: str | None = None
) -> _ImageProcessorProtocol:
    """Load Gemma 3n's configured SigLIP processor without torchvision.

    Transformers 5 exposes ``AutoImageProcessor`` as a torchvision-gated dummy
    when torchvision is absent, even when callers request its portable PIL
    backend. Importing the concrete PIL implementation preserves every value in
    ``preprocessor_config.json`` and avoids invoking torchvision on this path.
    The macOS runtime still installs torchvision because other supported
    processor families require Transformers' gated fallback.
    """
    from transformers.models.siglip.image_processing_pil_siglip import (
        SiglipImageProcessorPil,
    )

    factory = cast(_ImageProcessorFactory, cast(object, SiglipImageProcessorPil))
    return cast(
        _ImageProcessorProtocol,
        factory.from_pretrained(
            repo,
            **({"revision": revision} if revision is not None else {}),
        ),
    )


def _gemma4_native_processor_kwargs(
    chat_template_messages: list[JsonDict], processor: _ImageProcessorProtocol
) -> JsonDict:
    """Select processor kwargs for Gemma 4 native vision preprocessing.

    By default we keep the processor's built-in token budget so Gemma 4
    matches the reference Hugging Face and Ollama preprocessing path. A
    higher budget remains available through explicit environment overrides
    for debugging or local experiments.
    """
    has_image = False
    for message in chat_template_messages:
        content = message.get("content")
        if isinstance(content, list):
            for raw_part in cast(list[object], content):
                part = _object_dict(raw_part)
                if str(part.get("type", "")) == "image":
                    has_image = True
                    break
        if has_image:
            break

    configured_budget: int | None = None
    for env_var_name in (
        "SKULK_GEMMA4_MAX_SOFT_TOKENS",
        "SKULK_GEMMA4_IMAGE_ONLY_MAX_SOFT_TOKENS",
    ):
        override_value = os.environ.get(env_var_name)
        if override_value is None:
            continue
        with contextlib.suppress(ValueError):
            parsed_override = int(override_value)
            if parsed_override > 0:
                configured_budget = parsed_override
                break

    current_budget = getattr(processor, "max_soft_tokens", None)
    if (
        configured_budget is None
        or not has_image
        or not isinstance(current_budget, int)
        or current_budget >= configured_budget
    ):
        return {}

    logger.info(
        "Gemma 4 prompt with images detected; raising max_soft_tokens "
        f"from {current_budget} to {configured_budget} for better visual detail"
    )
    return {"max_soft_tokens": configured_budget}


_video_processor_patched = False
_video_processor_mapping_entry: object | None = None


def _patch_video_processor() -> None:
    """Patch so we don't crash horribly when torch vision isn't installed"""
    # TODO: Update if we add torch vision.
    global _video_processor_mapping_entry, _video_processor_patched
    if _video_processor_patched:
        return
    try:
        from transformers.processing_utils import MODALITY_TO_AUTOPROCESSOR_MAPPING

        mapping = cast(
            dict[str, object],
            MODALITY_TO_AUTOPROCESSOR_MAPPING._MAPPING_NAMES,  # type: ignore
        )
        _video_processor_mapping_entry = mapping.get("video_processor")
        mapping.pop("video_processor", None)
    except (ImportError, AttributeError):
        pass
    _video_processor_patched = True


def _restore_video_processor() -> None:
    """Restore the Transformers video slot removed by the compatibility patch."""
    global _video_processor_patched
    if not _video_processor_patched:
        return
    try:
        from transformers.processing_utils import MODALITY_TO_AUTOPROCESSOR_MAPPING

        mapping = cast(
            dict[str, object],
            MODALITY_TO_AUTOPROCESSOR_MAPPING._MAPPING_NAMES,  # type: ignore
        )
        if _video_processor_mapping_entry is not None:
            mapping["video_processor"] = _video_processor_mapping_entry
    except (ImportError, AttributeError):
        pass
    _video_processor_patched = False


def _configure_video_processor_compatibility(model_type: str) -> None:
    """Select the reversible no-torchvision mapping state for one model family."""
    if model_type == "gemma4":
        # Gemma 4's MLX-VLM processor ships a NumPy-only video processor and
        # requires this constructor slot even when callers only process images.
        _restore_video_processor()
    else:
        _patch_video_processor()


def decode_base64_image(b64_data: str) -> Image.Image:
    """Decode a raw base64 string into an RGB PIL Image."""
    raw = base64.b64decode(b64_data)
    with Image.open(io.BytesIO(raw)) as img:
        return img.convert("RGB")


@dataclass(frozen=True)
class _DecodedImageDebug:
    b64_sha256: str
    raw_sha256: str
    rgb_sha256: str
    raw_bytes: int
    width: int
    height: int
    original_mode: str


def _decode_base64_image_with_debug(
    b64_data: str,
) -> tuple[Image.Image, _DecodedImageDebug]:
    raw = base64.b64decode(b64_data)
    with Image.open(io.BytesIO(raw)) as img:
        original_mode = img.mode
        rgb_img = img.convert("RGB")
    debug = _DecodedImageDebug(
        b64_sha256=hashlib.sha256(b64_data.encode("ascii")).hexdigest(),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        rgb_sha256=hashlib.sha256(rgb_img.tobytes()).hexdigest(),
        raw_bytes=len(raw),
        width=rgb_img.width,
        height=rgb_img.height,
        original_mode=original_mode,
    )
    return rgb_img, debug


def _vision_transport_debug_enabled() -> bool:
    """Return whether expensive per-image debug hashing/logging is enabled."""
    return SKULK_IMAGE_TRANSPORT_DEBUG or bool(
        os.environ.get("SKULK_VISION_DEBUG_SAVE_DIR")
    )


def _vision_prompt_debug_enabled() -> bool:
    """Return whether full prompt-shape logging is enabled for vision prompts."""
    for env_var_name in ("SKULK_TRACE_REQUEST_SHAPES", "SKULK_TRACE_REQUEST_SHAPES"):
        value = os.environ.get(env_var_name)
        if value is not None:
            return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return False


def _maybe_dump_debug_image(
    image: Image.Image,
    debug: _DecodedImageDebug,
    image_index: int,
) -> None:
    """Persist the decoded image when explicit vision debugging is enabled.

    This makes it easy to distinguish "the model saw the wrong image" from
    "the model misread the right image" without reintroducing giant base64
    blobs into the logs.
    """
    configured_dir = os.environ.get("SKULK_VISION_DEBUG_SAVE_DIR")
    if not configured_dir:
        return

    output_dir = Path(configured_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{debug.raw_sha256[:16]}-image{image_index}.png"
    image.save(output_path, format="PNG")
    logger.info(
        f"Saved decoded vision image {image_index} to {output_path} "
        f"(raw_sha256={debug.raw_sha256[:12]}... rgb_sha256={debug.rgb_sha256[:12]}...)"
    )


def _format_vlm_messages(
    messages: list[JsonDict],
    model_type: str,
    *,
    task_id: TaskId | str | None = None,
) -> list[JsonDict]:
    def _count_image_parts(message: JsonDict) -> int:
        content = message.get("content")
        if not isinstance(content, list):
            return 0
        return sum(
            1
            for raw_part in cast(list[object], content)
            if str(_object_dict(raw_part).get("type", "")) in {"image", "image_url"}
        )

    formatted: list[JsonDict] = []
    image_count = sum(_count_image_parts(msg) for msg in messages)
    label_images = model_type in {"gemma3n", "gemma4"} and image_count > 1
    record_runner_phase(
        "vision_preprocess",
        event="format_vlm_messages",
        attrs={
            "model_type": model_type,
            "message_count": len(messages),
            "image_count": image_count,
            "gemma_image_label_count": image_count if label_images else 0,
        },
        task_id=task_id,
    )
    image_number = 1
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content")
        if not isinstance(content, list):
            formatted.append(msg)
            continue
        parts: list[JsonDict] = []
        for raw_part in cast(list[object], content):
            parts.append(_object_dict(raw_part))

        # Preserve the original multimodal ordering for Gemma family prompts.
        # Rebuilding the message from "all text + image count" destroys
        # interleaving such as text -> image -> text, which Gemma 4 explicitly
        # supports and which its chat template can represent directly.
        if model_type in {"gemma3n", "gemma4"}:
            normalized_parts: list[JsonDict] = []
            for part in parts:
                part_type = str(part.get("type", ""))
                if part_type == "text":
                    normalized_parts.append(
                        {"type": "text", "text": str(part.get("text", ""))}
                    )
                elif part_type in {"image", "image_url"}:
                    image_part: JsonDict = {"type": "image"}
                    if label_images:
                        # Gemma 4 has been observed to answer from an earlier
                        # image when multiple bare image placeholders are present
                        # in chat history. Stable chronological labels make
                        # "the latest image" unambiguous without changing the
                        # image tensor order.
                        image_part["label"] = f"Image {image_number}"
                    image_number += 1
                    normalized_parts.append(image_part)
            formatted.append({"role": role, "content": normalized_parts})
            continue

        text_parts = [str(p.get("text", "")) for p in parts if p.get("type") == "text"]
        n_images = sum(1 for p in parts if p.get("type") in ("image", "image_url"))
        result = cast(
            JsonDict,
            get_message_json(
                model_type, " ".join(text_parts), role, num_images=n_images
            ),
        )
        formatted.append(result)
    return formatted


@dataclass(frozen=True)
class _VisionPromptDebug:
    """Prompt rendering facts used to compare Skulk against processor templates."""

    raw_prompt_chars: int
    raw_image_placeholder_positions: list[int]
    expanded_prompt_chars: int
    tokens_per_image: list[int]
    image_token: str

    def attrs(self) -> dict[str, TraceAttrValue]:
        """Return JSON-safe tracing attributes for the rendered vision prompt."""
        return {
            "raw_prompt_chars": self.raw_prompt_chars,
            "raw_image_placeholder_count": len(self.raw_image_placeholder_positions),
            "raw_image_placeholder_offsets": [
                str(position) for position in self.raw_image_placeholder_positions
            ],
            "expanded_prompt_chars": self.expanded_prompt_chars,
            "tokens_per_image": [
                str(token_count) for token_count in self.tokens_per_image
            ],
            "image_token": self.image_token,
        }


@dataclass(frozen=True)
class _VisionPromptBuild:
    """Rendered prompt plus the unexpanded prompt facts that produced it."""

    prompt: str
    raw_prompt: str
    debug: _VisionPromptDebug


def _prompt_placeholder_positions(prompt: str, image_token: str) -> list[int]:
    """Return character offsets for image placeholders in the unexpanded prompt."""
    positions: list[int] = []
    offset = 0
    while True:
        position = prompt.find(image_token, offset)
        if position == -1:
            return positions
        positions.append(position)
        offset = position + len(image_token)


def _log_vision_prompt_shape(
    label: str,
    raw_prompt: str,
    expanded_prompt: str,
    attrs: dict[str, TraceAttrValue],
) -> None:
    """Log full raw and expanded vision prompts when request-shape tracing is on."""
    if not _vision_prompt_debug_enabled():
        return

    payload: dict[str, TraceAttrValue] = {"label": label, **attrs}
    logger.info(
        f"[request-shape] {json.dumps(payload, ensure_ascii=True, sort_keys=True)}"
    )
    logger.info(f"[request-shape] vision raw prompt label={label}\n{raw_prompt}")
    logger.info(
        f"[request-shape] vision expanded prompt label={label}\n{expanded_prompt}"
    )


def _diagnostic_attrs(attrs: dict[str, TraceAttrValue]) -> dict[str, object]:
    """Adapt trace-safe attrs to the runner diagnostic helper's invariant dict."""
    return cast(dict[str, object], attrs)



#: ``vision.model_type`` spellings the registry and cards use for Muse Glimmer.
MUSE_GLIMMER_VISION_MODEL_TYPES: frozenset[str] = frozenset(
    {"muse_glimmer", "muse-glimmer"}
)

def _build_vision_prompt_with_debug(
    tokenizer: TokenizerWrapper,
    chat_template_messages: list[JsonDict],
    n_tokens_per_image: list[int],
    image_token: str,
    model_type: str | None = None,
    boi_token_id: int | None = None,
    eoi_token_id: int | None = None,
    enable_thinking: bool | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> _VisionPromptBuild:
    """Build the expanded prompt and retain the raw placeholder layout.

    For models that use BOI/EOI framing (Gemma 3n/4), each expanded image
    sequence is wrapped as ``BOI + (IMAGE × N) + EOI`` matching the format
    the model was trained on."""
    logger.info(
        "Building vision prompt "
        f"(messages={len(chat_template_messages)}, images={len(n_tokens_per_image)})"
    )
    uses_gemma4_reference_prompt = model_type == "gemma4"
    if uses_gemma4_reference_prompt:
        prompt = render_gemma4_prompt(
            chat_template_messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    else:
        extra_kwargs: dict[str, Any] = {}
        if enable_thinking is not None:
            # Qwen3 and GLM use "enable_thinking"; DeepSeek uses "thinking".
            # Unknown Jinja variables are ignored by templates that do not need them.
            extra_kwargs["enable_thinking"] = enable_thinking
            extra_kwargs["thinking"] = enable_thinking
        if reasoning_effort is not None:
            extra_kwargs["reasoning_effort"] = reasoning_effort
            if model_type in MUSE_GLIMMER_VISION_MODEL_TYPES:
                # Muse Glimmer's template reads a strength level, not effort.
                strength = muse_glimmer_reasoning_strength(reasoning_effort)
                if strength is not None:
                    extra_kwargs["reasoning_strength"] = strength
        prompt = tokenizer.apply_chat_template(
            chat_template_messages,
            tokenize=False,
            add_generation_prompt=True,
            **extra_kwargs,
        )
    raw_prompt = prompt
    raw_placeholder_positions = _prompt_placeholder_positions(raw_prompt, image_token)

    # Decode BOI/EOI token IDs to their string form so we can insert them
    # into the prompt text before tokenization.
    boi_str = tokenizer.decode([boi_token_id]) if boi_token_id is not None else ""
    eoi_str = tokenizer.decode([eoi_token_id]) if eoi_token_id is not None else ""

    # Walk the prompt and expand each single image_token placeholder into
    # N copies where N = the number of vision features for that image,
    # wrapped in BOI/EOI markers when configured.
    image_idx = 0
    result: list[str] = []
    i = 0
    pad_len = len(image_token)
    while i < len(raw_prompt):
        if raw_prompt[i : i + pad_len] == image_token:
            n = (
                n_tokens_per_image[image_idx]
                if image_idx < len(n_tokens_per_image)
                else 1
            )
            result.append(f"{boi_str}{image_token * n}{eoi_str}")
            image_idx += 1
            i += pad_len
        else:
            result.append(raw_prompt[i])
            i += 1

    expanded_prompt = "".join(result)
    debug = _VisionPromptDebug(
        raw_prompt_chars=len(raw_prompt),
        raw_image_placeholder_positions=raw_placeholder_positions,
        expanded_prompt_chars=len(expanded_prompt),
        tokens_per_image=n_tokens_per_image,
        image_token=image_token,
    )
    _log_vision_prompt_shape(
        "vision",
        raw_prompt,
        expanded_prompt,
        debug.attrs(),
    )
    return _VisionPromptBuild(
        prompt=expanded_prompt, raw_prompt=raw_prompt, debug=debug
    )


def build_vision_prompt(
    tokenizer: TokenizerWrapper,
    chat_template_messages: list[JsonDict],
    n_tokens_per_image: list[int],
    image_token: str,
    model_type: str | None = None,
    boi_token_id: int | None = None,
    eoi_token_id: int | None = None,
    enable_thinking: bool | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> str:
    """Build a full prompt string with image placeholders expanded to features.

    Args:
        tokenizer: Tokenizer whose chat template renders the textual turns.
        chat_template_messages: OpenAI-style chat messages that may include
            image parts.
        n_tokens_per_image: Number of placeholder tokens each image expands to.
        image_token: Placeholder token text used by the model's vision template.
        model_type: Optional model family override for custom prompt renderers.
        boi_token_id: Optional beginning-of-image token for framed templates.
        eoi_token_id: Optional end-of-image token for framed templates.
        enable_thinking: Explicit thinking toggle forwarded to templates that
            support it.
        reasoning_effort: Optional reasoning effort forwarded to templates that
            support it.

    Returns:
        The rendered prompt with each image placeholder expanded to match the
        encoded vision feature count.
    """
    return _build_vision_prompt_with_debug(
        tokenizer,
        chat_template_messages,
        n_tokens_per_image,
        image_token,
        model_type=model_type,
        boi_token_id=boi_token_id,
        eoi_token_id=eoi_token_id,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    ).prompt


def _pixel_value_shapes(pixel_values: mx.array | list[mx.array]) -> list[str]:
    """Return compact tensor shape strings for diagnostics."""

    if isinstance(pixel_values, list):
        return [str(tuple(value.shape)) for value in pixel_values]
    return [str(tuple(pixel_values.shape))]


def _image_b64_hashes(images: list[str]) -> list[str]:
    """Return stable hashes of the base64 image payloads in request order."""
    return [hashlib.sha256(image.encode("ascii")).hexdigest() for image in images]


def _media_region_ranges(media_regions: list["MediaRegion"]) -> list[str]:
    """Return compact token ranges for prompt media regions."""
    return [f"{region.start_pos}:{region.end_pos}" for region in media_regions]


def _media_region_hashes(media_regions: list["MediaRegion"]) -> list[str]:
    """Return compact content hashes for prompt media regions."""
    return [region.content_hash[:12] for region in media_regions]


@dataclass
class MediaRegion:
    """A contiguous span of image-token positions in the prompt, tagged with
    a content hash so the KV prefix cache can detect when images change."""

    content_hash: str
    start_pos: int
    end_pos: int


@dataclass
class VisionResult:
    """Output of the vision pipeline: the expanded prompt, its token IDs,
    the vision embeddings to splice in, and the media regions for caching."""

    prompt: str
    prompt_tokens: mx.array
    embeddings: mx.array
    media_regions: list[MediaRegion]
    pixel_values: mx.array | list[mx.array] | None = None
    image_grid_thw: mx.array | None = None
    debug_info: "VisionDebugInfo | None" = None


@dataclass(frozen=True)
class VisionDebugInfo:
    """Compact multimodal alignment facts for traces and runner diagnostics."""

    model_type: str
    native_vision: bool
    image_b64_hashes: list[str]
    prompt_debug: _VisionPromptDebug
    prompt_tokens: int
    image_token_count: int
    media_region_ranges: list[str]
    media_region_hashes: list[str]
    pixel_value_shapes: list[str]
    vision_feature_shape: str | None = None
    image_grid_shape: str | None = None

    def attrs(self) -> dict[str, TraceAttrValue]:
        """Return bounded JSON-safe attributes for tracing and diagnostics."""
        attrs: dict[str, TraceAttrValue] = {
            "vision_model_type": self.model_type,
            "native_vision": self.native_vision,
            "image_count": len(self.image_b64_hashes),
            "image_b64_hashes": [h[:12] for h in self.image_b64_hashes],
            "prompt_tokens": self.prompt_tokens,
            "image_token_count": self.image_token_count,
            "media_region_ranges": self.media_region_ranges,
            "media_region_hashes": self.media_region_hashes,
            "pixel_value_shapes": self.pixel_value_shapes,
            **self.prompt_debug.attrs(),
        }
        if self.vision_feature_shape is not None:
            attrs["vision_feature_shape"] = self.vision_feature_shape
        if self.image_grid_shape is not None:
            attrs["image_grid_shape"] = self.image_grid_shape
        return attrs


_QUANTIZATION_SUFFIXES = (".biases", ".scales")


def _load_projector_weights(
    projector: nn.Module, weights: dict[str, mx.array], config: JsonDict
) -> None:
    """Load projector weights, quantizing the module first if needed.

    Quantized models store Linear weights in a compressed format with
    accompanying ``.scales`` and ``.biases`` tensors. We detect this by
    checking for those keys and apply ``nn.quantize`` to the projector
    before loading so the shapes match."""
    has_quantized = any(k.endswith(_QUANTIZATION_SUFFIXES) for k in weights)
    if has_quantized:
        quant_cfg = _object_dict(config.get("quantization", {}))
        bits = _int_value(quant_cfg.get("bits", 4), 4)
        group_size = _int_value(quant_cfg.get("group_size", 64), 64)
        logger.info(
            f"Quantizing projector module ({bits}-bit, group_size={group_size}) "
            "to match checkpoint format"
        )
        cast(Callable[..., object], nn.quantize)(
            projector,
            bits=bits,
            group_size=group_size,
        )
    projector.load_weights(list(weights.items()))


def _instantiate_projector(
    cls: type,
    model_config: object,
    vision_config: object,
    text_config: object,
) -> nn.Module:
    """Try multiple calling conventions for projector/embedder classes.

    Different model families use different constructor signatures:
    - Qwen/Kimi: ``Projector(model_config)``
    - Gemma 3n: ``Embedder(vision_config, text_config=text_config)``
    - Gemma 4: ``Embedder(embedding_dim, text_hidden_size, eps)``
    """
    params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    # Try single ModelConfig arg first (Qwen/Kimi pattern)
    if "config" in params or len(params) == 1:
        try:
            return cast(nn.Module, cls(model_config))
        except TypeError:
            pass
    # Try (multimodal_config, text_config) pattern (Gemma 3n)
    if "multimodal_config" in params or "text_config" in params:
        try:
            return cast(nn.Module, cls(vision_config, text_config=text_config))
        except TypeError:
            pass
    # Try raw dims pattern (Gemma 4: embedding_dim, text_hidden_size, eps)
    if "embedding_dim" in params:
        kwargs: JsonDict = {
            "embedding_dim": getattr(vision_config, "hidden_size", 768),
            "text_hidden_size": getattr(text_config, "hidden_size", 2048),
        }
        if "eps" in params:
            kwargs["eps"] = getattr(vision_config, "rms_norm_eps", 1e-6)
        return cast(nn.Module, cls(**cast(dict[str, Any], kwargs)))
    # Fallback: pass through _filter_config
    return cast(
        nn.Module,
        cls(**cast(dict[str, Any], _filter_config(cls, vars(model_config)))),
    )


class VisionEncoder:
    """Lazy-loaded vision tower + projector that encodes PIL images into
    feature tensors. Supports both bundled weights (loaded from the main
    model repo) and separate vision weight repos."""

    def __init__(
        self,
        config: VisionCardConfig,
        model_id: ModelId,
        source_revision: str | None = None,
        source_repository: ModelId | None = None,
        artifact_root: str | None = None,
    ):
        self._config = config
        self._source_revision = source_revision
        self._source_repository = source_repository or model_id
        self._main_model_path = (
            build_model_path(model_id, source_revision)
            if artifact_root is None
            else build_model_path(
                model_id, source_revision, artifact_root=artifact_root
            )
        )
        if config.weights_repo == str(self._source_repository):
            self._model_path = self._main_model_path
        else:
            self._model_path = build_model_path(
                ModelId(config.weights_repo), config.weights_revision
            )
        self._vision_tower: nn.Module | None = None
        self._projector: nn.Module | None = None
        self._processor: _ImageProcessorProtocol | None = None
        self._spatial_merge_size: int = 2
        self._merge_kernel_size: list[int] | None = None
        self._needs_nhwc: bool = False
        self._processor_loaded = False
        self._loaded = False

    @property
    def processor(self) -> _ImageProcessorProtocol | None:
        """Return the loaded image processor, if available."""
        return self._processor

    def _load_config_json(self) -> JsonDict:
        for candidate in (self._main_model_path, self._model_path):
            path = candidate / "config.json"
            if path.exists():
                with open(path) as f:
                    return _object_dict(cast(object, json.load(f)))
        return {}

    def _import_mlx_vlm(self, *submodules: str) -> object | tuple[object, ...]:
        mt = self._config.model_type
        results: list[object] = []
        for sub in submodules:
            name = f"mlx_vlm.models.{mt}.{sub}"
            results.append(importlib.import_module(name))
        return results[0] if len(results) == 1 else tuple(results)

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_weights()
        self._loaded = True

    def ensure_processor_loaded(self) -> None:
        """Load only the image processor needed for preprocessing images.

        Native multimodal models like Gemma 4 already carry their vision tower and
        projector inside the main loaded model. In that case we only need the
        processor contract from mlx-vlm/HF, not a second copy of the vision weights.
        """
        if self._processor_loaded:
            return
        self._load_processor()
        self._processor_loaded = True

    def _load_weights(self) -> None:
        _configure_video_processor_compatibility(self._config.model_type)
        logger.info(f"Loading vision weights from {self._model_path}")
        config = self._load_config_json()
        if not config:
            raise FileNotFoundError(f"config.json not found in {self._model_path}")

        vision_cfg = _object_dict(config.get("vision_config", {}))

        config_mod, vision_mod = cast(
            tuple[object, object],
            self._import_mlx_vlm("config", "vision"),
        )
        vision_config_cls = cast(type, _module_attr(config_mod, "VisionConfig"))
        vision_model_cls = cast(type, _module_attr(vision_mod, "VisionModel"))

        vision_config = cast(
            object,
            vision_config_cls(
                **cast(dict[str, Any], _filter_config(vision_config_cls, vision_cfg))
            ),
        )
        self._spatial_merge_size = _int_value(
            cast(object, getattr(vision_config, "spatial_merge_size", 2)),
            2,
        )
        self._vision_tower = cast(nn.Module, vision_model_cls(vision_config))
        model_mod: object | None = None
        with contextlib.suppress(ImportError):
            model_mod = self._import_mlx_vlm(self._config.model_type)

        projector_cls = None
        # Match common projector class naming conventions across model
        # families: *Projector (Qwen, Kimi), *Embedder (Gemma 3n/4).
        _projector_patterns = ("Projector", "Embedder")
        if model_mod is not None:
            for attr_name in dir(model_mod):
                obj = cast(object, getattr(model_mod, attr_name))
                if (
                    isinstance(obj, type)
                    and issubclass(obj, nn.Module)
                    and any(p in attr_name for p in _projector_patterns)
                ):
                    projector_cls = obj
                    break

        if projector_cls is not None:
            text_config_cls = cast(type, _module_attr(config_mod, "TextConfig"))
            model_config_cls = cast(type, _module_attr(config_mod, "ModelConfig"))
            text_config = cast(
                object,
                text_config_cls(
                    **cast(
                        dict[str, Any],
                        _filter_config(
                            text_config_cls,
                            _object_dict(config.get("text_config", {})),
                        ),
                    ),
                ),
            )
            extra = {
                k: v
                for k, v in config.items()
                if k not in ("text_config", "vision_config")
            }
            extra.setdefault("model_type", self._config.model_type)
            model_config = cast(
                object,
                model_config_cls(
                    text_config=text_config,
                    vision_config=vision_config,
                    **cast(dict[str, Any], _filter_config(model_config_cls, extra)),
                ),
            )
            self._projector = _instantiate_projector(
                projector_cls, model_config, vision_config, text_config
            )

        processor_repo = self._config.processor_repo
        if processor_repo:
            self._load_weights_from_separate_repo(config)
        else:
            self._load_weights_from_model_repo(config)

        self._load_processor(config=config, vision_cfg=vision_cfg)

    def _load_processor(
        self,
        *,
        config: JsonDict | None = None,
        vision_cfg: JsonDict | None = None,
    ) -> None:
        """Load the image processor without forcing vision-weight initialization."""
        _configure_video_processor_compatibility(self._config.model_type)
        config = config or self._load_config_json()
        if not config:
            raise FileNotFoundError(f"config.json not found in {self._model_path}")
        vision_cfg = vision_cfg or _object_dict(config.get("vision_config", {}))

        processor_repo = self._config.processor_repo
        repo = processor_repo or str(self._model_path)
        processor_revision = (
            self._source_revision
            if processor_repo == str(self._source_repository)
            else self._config.processor_revision
        )
        loader_options: dict[str, object] = (
            {"revision": processor_revision}
            if processor_revision is not None
            else {}
        )
        image_proc: _ImageProcessorProtocol | None = None
        load_failures: list[str] = []
        try:
            image_proc = cast(
                _ImageProcessorProtocol | None,
                load_image_processor(repo, **loader_options),
            )
        except (ImportError, OSError, ValueError) as exc:
            load_failures.append(f"mlx_vlm.utils.load_image_processor: {exc}")
            logger.info(f"mlx_vlm image processor loader failed for {repo}: {exc}")

        if image_proc is None and self._config.model_type == "gemma3n":
            try:
                image_proc = _load_gemma3n_pil_image_processor(
                    repo, processor_revision
                )
                logger.info("Using Transformers PIL SigLIP image processor")
            except (ImportError, OSError, TypeError, ValueError) as exc:
                load_failures.append(f"Transformers PIL SigLIP processor: {exc}")
                logger.info(
                    f"Transformers PIL SigLIP processor failed for {repo}: {exc}"
                )

        processor_modules = _mlx_vlm_processor_modules(self._config.model_type)
        if image_proc is None:
            for proc_mod in processor_modules:
                image_proc = _load_mlx_vlm_image_processor_from_pretrained(
                    proc_mod,
                    repo,
                    processor_revision,
                )
                if image_proc is not None:
                    break

        if image_proc is None:
            try:
                auto_from_pretrained = cast(
                    Callable[..., object],
                    AutoImageProcessor.from_pretrained,
                )
                image_proc = cast(
                    _ImageProcessorProtocol,
                    auto_from_pretrained(
                        repo,
                        trust_remote_code=True,
                        **loader_options,
                    ),
                )
            except (ImportError, OSError, ValueError) as exc:
                load_failures.append(f"transformers.AutoImageProcessor: {exc}")
                logger.info(
                    f"transformers image processor loader failed for {repo}: {exc}"
                )

        if image_proc is None:
            image_proc = _instantiate_mlx_vlm_image_processor(processor_modules)
        if image_proc is None:
            failure_detail = "; ".join(load_failures) or "no processor class found"
            raise RuntimeError(
                f"Unable to load an image processor for {self._config.model_type} "
                f"from {repo}: {failure_detail}"
            )
        self._processor = image_proc
        if processor_repo:
            self._merge_kernel_size = cast(
                list[int],
                vision_cfg.get("merge_kernel_size", [2, 2]),
            )
            self._needs_nhwc = True
        max_soft_tokens = getattr(self._processor, "max_soft_tokens", None)
        if isinstance(max_soft_tokens, int):
            logger.info(f"Image processor max_soft_tokens={max_soft_tokens}")
        logger.info(f"HF image processor loaded from {repo}")
        self._processor_loaded = True

    def _load_weights_from_separate_repo(self, config: dict[str, Any]) -> None:
        safetensors_files = list(self._model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {self._model_path}")

        weights: dict[str, mx.array] = {}
        for sf_path in safetensors_files:
            with safe_open(str(sf_path), framework="pt") as f:
                keys = f.keys()
                for key in keys:
                    tensor = f.get_tensor(key)  # type: ignore
                    np_tensor = tensor.float().numpy()  # type: ignore
                    weights[key] = mx.array(np_tensor, dtype=mx.bfloat16)  # type: ignore

        # Partition weights into vision tower vs projector, stripping prefixes
        # and remapping key names to match mlx_vlm's expected parameter layout.
        vision_weights: dict[str, mx.array] = {}
        projector_weights: dict[str, mx.array] = {}
        for key, val in weights.items():
            if key.startswith("vision_tower."):
                short_key = key[len("vision_tower.") :]
                if short_key.startswith("encoder."):
                    short_key = short_key[len("encoder.") :]
                m = re.match(r"^(blocks\.\d+)\.(wqkv|wo)\.(weight|bias)$", short_key)
                if m:
                    short_key = f"{m.group(1)}.attn.{m.group(2)}.{m.group(3)}"
                if short_key == "patch_embed.proj.weight" and val.ndim == 4:
                    val = val.transpose(0, 2, 3, 1)
                vision_weights[short_key] = val
            elif key.startswith(("mm_projector.", "multi_modal_projector.")):
                if key.startswith("multi_modal_projector."):
                    short_key = key[len("multi_modal_projector.") :]
                    if short_key.startswith("mm_projector."):
                        short_key = short_key[len("mm_projector.") :]
                else:
                    short_key = key[len("mm_projector.") :]
                short_key = short_key.replace("proj.0.", "linear_1.").replace(
                    "proj.2.", "linear_2."
                )
                projector_weights[short_key] = val

        assert self._vision_tower is not None
        self._vision_tower.load_weights(list(vision_weights.items()))
        mx.eval(self._vision_tower.parameters())

        if self._projector is not None and projector_weights:
            _load_projector_weights(self._projector, projector_weights, config)
            mx.eval(self._projector.parameters())

        n_vision = sum(v.size for _, v in vision_weights.items())
        n_proj = sum(v.size for _, v in projector_weights.items())
        logger.info(
            f"Vision encoder loaded: {n_vision / 1e6:.1f}M params"
            + (f", projector: {n_proj / 1e6:.1f}M params" if n_proj else "")
        )

    def _load_weights_from_model_repo(self, config: dict[str, Any]) -> None:
        safetensors_files = sorted(self._model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {self._model_path}")

        vision_prefixes = ["vision_tower.", "model.visual."]
        # Gemma 3n/4 store their projector (MultimodalEmbedder) under
        # "embed_vision." in the same safetensors files as vision weights.
        projector_prefixes = ["embed_vision."]
        vision_weights: dict[str, mx.array] = {}
        projector_weights: dict[str, mx.array] = {}
        found_raw_prefix = False
        for sf_path in safetensors_files:
            file_weights: dict[str, mx.array] = mx.load(str(sf_path))  # type: ignore
            for key, val in file_weights.items():
                for prefix in vision_prefixes:
                    if key.startswith(prefix):
                        short_key = key[len(prefix) :]
                        vision_weights[short_key] = val
                        if prefix == "model.visual.":
                            found_raw_prefix = True
                        break
                else:
                    for prefix in projector_prefixes:
                        if key.startswith(prefix):
                            short_key = key[len(prefix) :]
                            projector_weights[short_key] = val
                            break

        if not vision_weights:
            raise ValueError(
                f"No vision weights found with prefixes {vision_prefixes} in {self._model_path}. "
                "Ensure the model repo contains bundled vision weights."
            )

        assert self._vision_tower is not None
        if found_raw_prefix and hasattr(self._vision_tower, "sanitize"):
            vision_weights = self._vision_tower.sanitize(vision_weights)  # type: ignore

        self._vision_tower.load_weights(list(vision_weights.items()))  # type: ignore
        mx.eval(self._vision_tower.parameters())

        if self._projector is not None and projector_weights:
            _load_projector_weights(self._projector, projector_weights, config)
            mx.eval(self._projector.parameters())

        n_vision = sum(v.size for _, v in vision_weights.items())  # type: ignore
        n_proj = sum(v.size for _, v in projector_weights.items())
        logger.info(
            f"Vision encoder loaded: {n_vision / 1e6:.1f}M params"
            + (f", projector: {n_proj / 1e6:.1f}M params" if n_proj else "")
        )

    def preprocess_images(
        self,
        pil_images: list[Image.Image],
        *,
        processor_kwargs: JsonDict | None = None,
    ) -> tuple[mx.array | list[mx.array], mx.array | None, list[int]]:
        """Public interface to image preprocessing (pixel_values + token counts)."""
        self.ensure_processor_loaded()
        with runner_phase(
            "vision_preprocess",
            detail="preprocess_images",
            attrs={
                "image_count": len(pil_images),
                "processor_kwargs": list((processor_kwargs or {}).keys()),
            },
            include_memory=True,
        ):
            return self._preprocess_images(
                pil_images, processor_kwargs=processor_kwargs
            )

    def encode_images(self, images: list[str]) -> tuple[mx.array, list[int]]:
        """Encode base64 images into feature tensors and per-image token counts."""
        self.ensure_loaded()
        assert self._vision_tower is not None
        assert self._processor is not None

        with runner_phase(
            "vision_preprocess",
            detail="base64_decode",
            attrs={"image_count": len(images)},
        ):
            pil_images = [decode_base64_image(b64) for b64 in images]
        for idx, img in enumerate(pil_images):
            logger.info(f"Image {idx}: {img.width}x{img.height} mode={img.mode}")
            record_runner_phase(
                "vision_preprocess",
                event="image_decoded",
                attrs={"image_index": idx, "width": img.width, "height": img.height},
            )

        pixel_values, grid_thw, n_tokens_per_image = self._preprocess_images(pil_images)
        record_runner_phase(
            "vision_preprocess",
            event="image_preprocessed",
            attrs={
                "image_count": len(pil_images),
                "pixel_value_shapes": _pixel_value_shapes(pixel_values),
                "tokens_per_image": [
                    str(token_count) for token_count in n_tokens_per_image
                ],
            },
            include_memory=True,
        )
        hidden_states = self._run_vision_tower(pixel_values, grid_thw)

        if self._projector is not None:
            image_features: mx.array = self._projector(hidden_states)
        else:
            image_features = hidden_states

        return image_features, n_tokens_per_image

    def _preprocess_images(
        self,
        pil_images: list[Image.Image],
        *,
        processor_kwargs: JsonDict | None = None,
    ) -> tuple[mx.array | list[mx.array], mx.array | None, list[int]]:
        """Run the image processor and return pixel values, optional grid, and token counts."""
        assert self._processor is not None
        processor_kwargs = processor_kwargs or {}

        if self._config.processor_repo:
            processed = self._processor.preprocess(
                [{"type": "image", "image": img} for img in pil_images],
                return_tensors="np",
                **processor_kwargs,
            )
            pixel_values = mx.array(processed["pixel_values"])  # type: ignore
            grid_thw = mx.array(processed["grid_thws"])  # type: ignore
            assert self._merge_kernel_size is not None
            merge_length = int(np.prod(self._merge_kernel_size))
            n_tokens_per_image = [
                int(mx.prod(grid_thw[i]).item()) // merge_length
                for i in range(grid_thw.shape[0])
            ]
            return pixel_values, grid_thw, n_tokens_per_image

        raw = self._processor(
            images=pil_images, return_tensors="np", **processor_kwargs
        )

        # Gemma 4's processor returns (data_dict, num_soft_tokens_per_image).
        if isinstance(raw, tuple):
            raw_tuple = cast(tuple[object, object], raw)
            data_dict = _object_dict(raw_tuple[0])
            soft_tokens = [
                _int_value(token, 0) for token in cast(list[object], raw_tuple[1])
            ]
            pv_raw = data_dict["pixel_values"]
            if isinstance(pv_raw, list):
                # Variable-sized images — keep as list for per-image encoding.
                pixel_values_list = [
                    mx.array(cast(MxArrayInput, v)) for v in cast(list[object], pv_raw)
                ]
                return pixel_values_list, None, soft_tokens
            pixel_values = mx.array(cast(MxArrayInput, pv_raw))
            # Gemma 4's single-image processor may omit the batch dimension.
            # Normalize it here so prefix-cache and pipeline-prefill slicing
            # count images rather than treating the RGB channels as images.
            if pixel_values.ndim == 3:
                pixel_values = pixel_values[None]
            return pixel_values, None, soft_tokens

        # Standard HuggingFace processor (Qwen, SiglipImageProcessor, etc.)
        raw_dict = _object_dict(raw)
        pixel_values = mx.array(cast(MxArrayInput, raw_dict["pixel_values"]))

        # Processors that return grid info (Qwen-VL family).
        if "image_grid_thw" in raw_dict:
            grid_thw = mx.array(cast(MxArrayInput, raw_dict["image_grid_thw"]))
            merge_unit = self._spatial_merge_size**2
            n_tokens_per_image = [
                int(
                    grid_thw[i, 0].item()
                    * grid_thw[i, 1].item()
                    * grid_thw[i, 2].item()
                )
                // merge_unit
                for i in range(grid_thw.shape[0])
            ]
            return pixel_values, grid_thw, n_tokens_per_image

        # Gemma 3n / simple processors: no grid info, use config's
        # vision_soft_tokens_per_image or the vision tower's default_output_length.
        n_per_image = self._get_soft_tokens_per_image()
        n_tokens_per_image = [n_per_image] * len(pil_images)
        return pixel_values, None, n_tokens_per_image

    def _get_soft_tokens_per_image(self) -> int:
        """Determine the number of soft tokens per image from model config.

        Caches the result to avoid re-reading config.json on every call."""
        cached = cast(object | None, getattr(self, "_soft_tokens_cache", None))
        if cached is not None:
            return _int_value(cached, 256)

        result = 256  # sensible default for SiglipImageProcessor-based models
        config = self._load_config_json()
        if config:
            # Top-level vision_soft_tokens_per_image (gemma3n, gemma4).
            vst = config.get("vision_soft_tokens_per_image")
            if vst is not None:
                result = _int_value(vst, result)
            else:
                # VisionConfig.default_output_length (gemma4).
                vc = _object_dict(config.get("vision_config", {}))
                dol = vc.get("default_output_length")
                if dol is not None:
                    result = _int_value(dol, result)

        self._soft_tokens_cache = result
        return result

    def _run_vision_tower(
        self, pixel_values: mx.array | list[mx.array], grid_thw: mx.array | None
    ) -> mx.array:
        """Run the vision tower, dispatching by input format."""
        assert self._vision_tower is not None

        if self._needs_nhwc:
            assert grid_thw is not None
            grid_hw = grid_thw[:, 1:] if grid_thw.shape[-1] == 3 else grid_thw
            return self._vision_tower(
                pixel_values.transpose(0, 2, 3, 1),  # type: ignore
                output_hidden_states=True,
                grid_thw=grid_hw,
            )

        if isinstance(pixel_values, list):
            # Variable-sized images (Gemma 4): encode each independently and
            # concatenate features.
            features: list[mx.array] = []
            for pv in pixel_values:
                if pv.ndim == 3:
                    pv = pv[None]  # add batch dim
                out: mx.array = self._vision_tower(pv)
                out = out[0] if isinstance(out, tuple) else out
                if out.ndim == 3:
                    out = out.reshape(-1, out.shape[-1])  # flatten batch
                features.append(out)
            return mx.concatenate(features, axis=0)

        if grid_thw is not None:
            result = self._vision_tower(pixel_values, grid_thw)
        else:
            # No grid info (Gemma 3n/4, SiglipImageProcessor) — vision tower
            # takes just pixel_values.
            result = self._vision_tower(pixel_values)
        out = result[0] if isinstance(result, tuple) else result
        # Some vision towers (Gemma 4) preserve the batch dimension.
        # Flatten to (total_tokens, hidden_dim) for create_vision_embeddings.
        if out.ndim == 3:
            out = out.reshape(-1, out.shape[-1])
        return out


def has_native_vision(model: nn.Module) -> bool:
    """Return whether the loaded model retains its native vision tower.

    Gemma exposes ``embed_vision`` while Qwen exposes
    ``get_input_embeddings``. Both require the raw processor outputs so their
    family-specific fusion and multimodal position handling remain intact.
    """
    inner: object = getattr(model, "_inner", model)
    return hasattr(inner, "vision_tower") and (
        hasattr(inner, "embed_vision") or hasattr(inner, "get_input_embeddings")
    )


def get_inner_model(model: nn.Module) -> Any:  # type: ignore
    """Traverse the model tree to find the inner transformer with ``embed_tokens``."""
    for candidate in (
        getattr(model, "model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ):
        if candidate is not None and hasattr(candidate, "embed_tokens"):  # type: ignore
            return candidate  # type: ignore

    raise ValueError(
        f"Could not find inner transformer (embed_tokens) in {type(model).__name__}. "
        "Add a new pattern to _get_inner_model() for this architecture."
    )


def create_vision_embeddings(
    model: Model,
    prompt_tokens: mx.array,
    image_features: mx.array,
    image_token_id: int,
) -> mx.array:
    """Replace image-token placeholder embeddings with vision features.

    Uses cumsum indexing to map each image-token position to the corresponding
    row in ``image_features``, then splices them into the text embeddings."""
    inner = get_inner_model(model)  # type: ignore
    embed_tokens = inner.embed_tokens  # type: ignore

    input_embeddings: mx.array = embed_tokens(prompt_tokens[None])  # type: ignore

    is_image: mx.array = mx.equal(prompt_tokens, image_token_id)
    n_placeholders = int(mx.sum(is_image).item())

    if n_placeholders > 0:
        if n_placeholders != image_features.shape[0]:
            logger.warning(
                f"Placeholder count ({n_placeholders}) != image features "
                f"({image_features.shape[0]}). Using min of both."
            )
            n = min(n_placeholders, image_features.shape[0])
            image_features = image_features[:n]

        # Gemma 4's text trunk multiplies every value returned by
        # ``embed_tokens`` by sqrt(hidden_size). Its native multimodal model
        # scatters projected image features *after* applying that text-only
        # scale. Distributed placement uses this precomputed-embedding path,
        # so compensate before ``patch_embed_tokens`` returns the mixed tensor;
        # the trunk then restores image features to their native magnitude.
        embed_scale = getattr(cast(object, inner), "embed_scale", None)
        if isinstance(embed_scale, (int, float)) and embed_scale != 0:
            image_features = image_features / embed_scale

        # Map each image-token position to its feature row via cumulative sum:
        # cumsum over the boolean mask gives 1-based indices; subtract 1 for 0-based.
        # Clip so non-image positions (which get -1) don't go out of bounds.
        image_indices = mx.cumsum(is_image.astype(mx.int32)) - 1
        image_indices = mx.clip(image_indices, 0, image_features.shape[0] - 1)

        # Gather vision features at image positions, keep text embeddings elsewhere
        gathered = image_features[image_indices].astype(input_embeddings.dtype)
        result = mx.where(is_image[:, None], gathered, input_embeddings[0])
        input_embeddings = result[None]

    return input_embeddings


def _find_media_regions(
    prompt_tokens: mx.array,
    images: list[str],
    image_token_id: int,
    boi_token_id: int | None = None,
    eoi_token_id: int | None = None,
) -> list[MediaRegion]:
    """Find contiguous image-token runs and expand to include BOI/EOI markers."""
    tokens_np = np.array(prompt_tokens)
    is_pad = tokens_np == image_token_id  # type: ignore

    regions: list[MediaRegion] = []
    in_run = False
    run_start = 0
    for pos, pad in enumerate(is_pad):  # type: ignore
        if pad and not in_run:
            run_start = pos
            in_run = True
        elif not pad and in_run:
            regions.append(
                MediaRegion(content_hash="", start_pos=run_start, end_pos=pos)
            )
            in_run = False
    if in_run:
        regions.append(
            MediaRegion(content_hash="", start_pos=run_start, end_pos=len(tokens_np))
        )

    # Expand region boundaries to include surrounding BOI/EOI tokens so
    # the KV prefix cache invalidates the full image span.
    for region in regions:
        if (
            boi_token_id is not None
            and region.start_pos > 0
            and tokens_np[region.start_pos - 1] == boi_token_id
        ):
            region.start_pos -= 1
        if (
            eoi_token_id is not None
            and region.end_pos < len(tokens_np)
            and tokens_np[region.end_pos] == eoi_token_id
        ):
            region.end_pos += 1

    for i, region in enumerate(regions):
        if i < len(images):
            img = decode_base64_image(images[i])
            region.content_hash = hashlib.sha256(img.tobytes()).hexdigest()
        else:
            logger.warning(f"Media region {i} has no corresponding image")

    region_summaries = [
        {
            "index": i,
            "start": region.start_pos,
            "end": region.end_pos,
            "length": region.end_pos - region.start_pos,
            "content_hash": region.content_hash[:12],
        }
        for i, region in enumerate(regions)
    ]
    logger.info(f"Vision media regions: {region_summaries}")

    return regions


class VisionProcessor:
    """
    Pipeline for vision models:
    1. Encode images into features (or grab from cache)
    2. Replace image placeholders with the features
    3. Build vision prompt
    4. Provide media regions for prefix caching
    """

    def __init__(
        self,
        config: VisionCardConfig,
        model_id: ModelId,
        source_revision: str | None = None,
        source_repository: ModelId | None = None,
        artifact_root: str | None = None,
    ):
        self.vision_config = config
        self._encoder = VisionEncoder(
            config,
            model_id,
            source_revision,
            source_repository,
            artifact_root,
        )
        self._feature_cache: dict[str, tuple[mx.array, list[int]]] = {}
        self._feature_cache_max = 32

    def load(self) -> None:
        self._encoder.ensure_loaded()

    def _image_cache_key(self, images: list[str]) -> str:
        h = hashlib.sha256()
        for img in images:
            pil = decode_base64_image(img)
            h.update(pil.tobytes())
        return h.hexdigest()

    def process(
        self,
        images: list[str],
        chat_template_messages: list[dict[str, Any]],
        tokenizer: TokenizerWrapper,
        model: Model,
        *,
        task_id: TaskId | str | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> VisionResult:
        """Convert images and chat messages into model-ready vision inputs.

        Args:
            images: Base64-encoded image payloads from the request.
            chat_template_messages: Chat messages containing the image parts.
            tokenizer: Tokenizer used to render and encode the multimodal prompt.
            model: Loaded MLX model receiving the resulting prompt inputs.
            task_id: Optional task identifier for diagnostics.
            enable_thinking: Explicit thinking toggle forwarded to the prompt
                renderer.
            reasoning_effort: Optional reasoning effort forwarded to the prompt
                renderer.

        Returns:
            Vision preprocessing output containing prompt tokens, media regions,
            and either fused embeddings or native image tensors.
        """
        logger.info(f"Vision pipeline: {len(images)} image(s)")
        record_runner_phase(
            "vision_preprocess",
            event="vision_process_start",
            attrs={"image_count": len(images)},
            task_id=task_id,
            include_memory=True,
        )

        native = has_native_vision(model)
        if native:
            return self._process_native(
                images,
                chat_template_messages,
                tokenizer,
                model,
                task_id=task_id,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )

        cache_key = self._image_cache_key(images)
        cached = self._feature_cache.pop(cache_key, None)
        if cached is not None:
            self._feature_cache[cache_key] = cached
            image_features, n_tokens_per_image = cached
        else:
            image_features, n_tokens_per_image = self._encoder.encode_images(images)
            self._feature_cache[cache_key] = (image_features, n_tokens_per_image)
            while len(self._feature_cache) > self._feature_cache_max:
                del self._feature_cache[next(iter(self._feature_cache))]
        logger.info(
            f"Vision features: {image_features.shape} "
            f"({image_features.shape[0]} tokens, per-image: {n_tokens_per_image})"
        )

        image_token = self.vision_config.image_token
        if image_token is None:
            image_token = tokenizer.decode([self.vision_config.required_image_token_id])

        formatted_messages = _format_vlm_messages(
            chat_template_messages, self.vision_config.model_type, task_id=task_id
        )

        prompt_build = _build_vision_prompt_with_debug(
            tokenizer,
            formatted_messages,
            n_tokens_per_image,
            image_token,
            model_type=self.vision_config.model_type,
            boi_token_id=self.vision_config.boi_token_id,
            eoi_token_id=self.vision_config.eoi_token_id,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        prompt = prompt_build.prompt
        record_runner_phase(
            "vision_preprocess",
            event="vision_prompt_rendered",
            attrs=_diagnostic_attrs(prompt_build.debug.attrs()),
            task_id=task_id,
        )

        logger.info(
            f"Expanded prompt has {prompt.count(image_token)} image_token occurrences, total len={len(prompt)}"
        )

        prompt_tokens: mx.array = encode_prompt(tokenizer, prompt)
        prompt_tokens = fix_unmatched_think_end_tokens(prompt_tokens, tokenizer)
        n_image_tokens = int(
            mx.sum(
                mx.equal(prompt_tokens, self.vision_config.required_image_token_id)
            ).item()
        )
        record_runner_phase(
            "vision_preprocess",
            event="image_token_expansion",
            attrs={
                "prompt_tokens": len(prompt_tokens),
                "image_token_count": n_image_tokens,
                "image_count": len(images),
            },
            task_id=task_id,
        )
        logger.info(
            f"Encoded prompt: {len(prompt_tokens)} tokens, {n_image_tokens} image pad tokens"
        )

        embeddings = create_vision_embeddings(
            model,
            prompt_tokens,
            image_features,
            self.vision_config.required_image_token_id,
        )
        mx.eval(embeddings)

        media_regions = _find_media_regions(
            prompt_tokens,
            images,
            self.vision_config.required_image_token_id,
            boi_token_id=self.vision_config.boi_token_id,
            eoi_token_id=self.vision_config.eoi_token_id,
        )
        record_runner_phase(
            "vision_preprocess",
            event="media_regions_ready",
            attrs={
                "media_region_count": len(media_regions),
                "media_region_lengths": [
                    str(region.end_pos - region.start_pos) for region in media_regions
                ],
            },
            task_id=task_id,
        )
        debug_info = VisionDebugInfo(
            model_type=self.vision_config.model_type,
            native_vision=False,
            image_b64_hashes=_image_b64_hashes(images),
            prompt_debug=prompt_build.debug,
            prompt_tokens=len(prompt_tokens),
            image_token_count=n_image_tokens,
            media_region_ranges=_media_region_ranges(media_regions),
            media_region_hashes=_media_region_hashes(media_regions),
            pixel_value_shapes=[],
            vision_feature_shape=str(tuple(image_features.shape)),
        )
        record_runner_phase(
            "vision_preprocess",
            event="vision_alignment_debug",
            attrs=_diagnostic_attrs(debug_info.attrs()),
            task_id=task_id,
        )

        return VisionResult(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            embeddings=embeddings,
            media_regions=media_regions,
            debug_info=debug_info,
        )

    def _process_native(
        self,
        images: list[str],
        chat_template_messages: list[dict[str, Any]],
        tokenizer: TokenizerWrapper,
        model: Model,
        *,
        task_id: TaskId | str | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> VisionResult:
        """Process images for models with native vision support (e.g. Gemma 4).

        Instead of pre-computing fused embeddings, this preprocesses images into
        ``pixel_values`` tensors. During prefill, the wrapper injects those raw
        tensors into the model's native ``__call__`` so Gemma 4 follows the same
        path as mlx-vlm: vision tower -> projector -> masked_scatter."""
        logger.info("Using native vision pipeline (model has built-in vision tower)")
        self._encoder.ensure_processor_loaded()
        processor = self._encoder.processor
        assert processor is not None

        debug_enabled = _vision_transport_debug_enabled()
        if debug_enabled:
            decoded_images = [_decode_base64_image_with_debug(b64) for b64 in images]
            pil_images = [image for image, _debug in decoded_images]
            for idx, (img, debug) in enumerate(decoded_images):
                logger.info(
                    f"Image {idx}: {img.width}x{img.height} mode={img.mode} "
                    f"original_mode={debug.original_mode} raw_bytes={debug.raw_bytes} "
                    f"b64_sha256={debug.b64_sha256[:12]}... "
                    f"raw_sha256={debug.raw_sha256[:12]}... "
                    f"rgb_sha256={debug.rgb_sha256[:12]}..."
                )
                _maybe_dump_debug_image(img, debug, idx)
                record_runner_phase(
                    "vision_preprocess",
                    event="native_image_decoded",
                    attrs={
                        "image_index": idx,
                        "width": img.width,
                        "height": img.height,
                        "raw_bytes": debug.raw_bytes,
                        "raw_sha256": debug.raw_sha256[:12],
                        "rgb_sha256": debug.rgb_sha256[:12],
                    },
                    task_id=task_id,
                )
        else:
            with runner_phase(
                "vision_preprocess",
                detail="native_base64_decode",
                attrs={"image_count": len(images)},
                task_id=task_id,
            ):
                pil_images = [decode_base64_image(b64) for b64 in images]
            for idx, img in enumerate(pil_images):
                logger.debug(f"Image {idx}: {img.width}x{img.height} mode={img.mode}")
                record_runner_phase(
                    "vision_preprocess",
                    event="native_image_decoded",
                    attrs={
                        "image_index": idx,
                        "width": img.width,
                        "height": img.height,
                    },
                    task_id=task_id,
                )

        processor_kwargs = (
            _gemma4_native_processor_kwargs(chat_template_messages, processor)
            if self.vision_config.model_type == "gemma4"
            else {}
        )
        logger.info(f"Native vision processor kwargs: {processor_kwargs}")
        pixel_values, grid_thw, n_tokens_per_image = self._encoder.preprocess_images(
            pil_images,
            processor_kwargs=processor_kwargs,
        )
        record_runner_phase(
            "vision_preprocess",
            event="native_image_preprocessed",
            attrs={
                "image_count": len(images),
                "processor_kwargs": list(processor_kwargs.keys()),
                "pixel_value_shapes": _pixel_value_shapes(pixel_values),
                "tokens_per_image": [
                    str(token_count) for token_count in n_tokens_per_image
                ],
            },
            task_id=task_id,
            include_memory=True,
        )
        logger.info(
            f"Native vision: preprocessed {len(images)} image(s), "
            f"tokens per image: {n_tokens_per_image}"
        )

        image_token = self.vision_config.image_token
        if image_token is None:
            image_token = tokenizer.decode([self.vision_config.required_image_token_id])

        formatted_messages = _format_vlm_messages(
            chat_template_messages, self.vision_config.model_type, task_id=task_id
        )
        prompt_build = _build_vision_prompt_with_debug(
            tokenizer,
            formatted_messages,
            n_tokens_per_image,
            image_token,
            model_type=self.vision_config.model_type,
            boi_token_id=self.vision_config.boi_token_id,
            eoi_token_id=self.vision_config.eoi_token_id,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        prompt = prompt_build.prompt
        record_runner_phase(
            "vision_preprocess",
            event="native_vision_prompt_rendered",
            attrs=_diagnostic_attrs(prompt_build.debug.attrs()),
            task_id=task_id,
        )

        prompt_tokens: mx.array = encode_prompt(tokenizer, prompt)
        prompt_tokens = fix_unmatched_think_end_tokens(prompt_tokens, tokenizer)
        n_image_tokens = int(
            mx.sum(
                mx.equal(prompt_tokens, self.vision_config.required_image_token_id)
            ).item()
        )
        record_runner_phase(
            "vision_preprocess",
            event="native_image_token_expansion",
            attrs={
                "prompt_tokens": len(prompt_tokens),
                "image_token_count": n_image_tokens,
                "gemma_image_label_count": len(images)
                if self.vision_config.model_type == "gemma4" and len(images) > 1
                else 0,
            },
            task_id=task_id,
        )
        logger.info(
            f"Encoded prompt: {len(prompt_tokens)} tokens, {n_image_tokens} image pad tokens"
        )

        media_regions = _find_media_regions(
            prompt_tokens,
            images,
            self.vision_config.required_image_token_id,
            boi_token_id=self.vision_config.boi_token_id,
            eoi_token_id=self.vision_config.eoi_token_id,
        )
        record_runner_phase(
            "vision_preprocess",
            event="native_media_regions_ready",
            attrs={
                "media_region_count": len(media_regions),
                "media_region_lengths": [
                    str(region.end_pos - region.start_pos) for region in media_regions
                ],
            },
            task_id=task_id,
        )
        debug_info = VisionDebugInfo(
            model_type=self.vision_config.model_type,
            native_vision=True,
            image_b64_hashes=_image_b64_hashes(images),
            prompt_debug=prompt_build.debug,
            prompt_tokens=len(prompt_tokens),
            image_token_count=n_image_tokens,
            media_region_ranges=_media_region_ranges(media_regions),
            media_region_hashes=_media_region_hashes(media_regions),
            pixel_value_shapes=_pixel_value_shapes(pixel_values),
            image_grid_shape=str(tuple(grid_thw.shape))
            if grid_thw is not None
            else None,
        )
        record_runner_phase(
            "vision_preprocess",
            event="native_vision_alignment_debug",
            attrs=_diagnostic_attrs(debug_info.attrs()),
            task_id=task_id,
            include_memory=True,
        )

        # Native vision models fuse image features inside __call__ during prefill,
        # so there are no precomputed embeddings to splice.
        empty_embeddings = mx.zeros((1, 0, 1))

        return VisionResult(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            embeddings=empty_embeddings,
            media_regions=media_regions,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            debug_info=debug_info,
        )


@final
class VisionPreprocessingError(Exception):
    """A request whose images could not be turned into model input.

    Answering a vision request without its image is the worst outcome this
    engine can produce. The model still replies, fluently and in the requested
    format, but every visual claim in that reply is invented, and nothing
    downstream can tell it apart from a real answer. A caller who asked what is
    in a picture is better served by an error than by a confident fiction, so
    every path that loses the image raises this instead of continuing.

    The runner terminates just the affected task on this error and stays alive,
    the same treatment context-admission rejections get: the condition is a
    property of the request and the node's install, not a corrupted engine.
    """


def prepare_vision(
    images: list[str] | None,
    chat_template_messages: list[dict[str, Any]] | None,
    vision_processor: VisionProcessor,
    tokenizer: TokenizerWrapper,
    model: Model,
    *,
    task_id: TaskId | str | None = None,
    enable_thinking: bool | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> VisionResult | None:
    """Encode images and build the vision-augmented prompt.

    Args:
        images: Base64-encoded image payloads, or ``None`` for text-only tasks.
        chat_template_messages: Chat messages preserving the request's image
            placement.
        vision_processor: Processor configured for the selected vision model.
        tokenizer: Tokenizer used by the model's chat template.
        model: Loaded MLX model receiving the vision-preprocessed inputs.
        task_id: Optional task identifier for diagnostics.
        enable_thinking: Explicit thinking toggle forwarded to vision prompt
            rendering.
        reasoning_effort: Optional reasoning effort forwarded to vision prompt
            rendering.

    Returns:
        ``None`` for requests without images, otherwise a ``VisionResult`` ready
        for generation.

    Raises:
        VisionPreprocessingError: Raised when a request contains images but the
            prompt cannot be built without dropping or hallucinating them.

    Returns ``None`` only when the request carried no images. A request that
    does carry images either gets them spliced into the prompt or raises
    ``VisionPreprocessingError``; it never quietly degrades to text.
    """
    if not images:
        return None
    if chat_template_messages is None:
        raise VisionPreprocessingError(
            f"cannot place {len(images)} image(s) in the prompt: the request "
            "arrived without chat_template_messages, so there is no message "
            "structure to attach them to"
        )

    return vision_processor.process(
        images=images,
        chat_template_messages=chat_template_messages,
        tokenizer=tokenizer,
        model=model,
        task_id=task_id,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
