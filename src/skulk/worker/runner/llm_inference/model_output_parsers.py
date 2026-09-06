import json
from collections.abc import Generator, Mapping
from functools import cache
from typing import Any, cast

from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    HarmonyError,  # pyright: ignore[reportUnknownVariableType]
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from skulk.api.types import ToolCallItem
from skulk.shared.constants import preferred_env_value
from skulk.shared.models.capabilities import (
    resolve_model_capability_profile,
    uses_muse_glimmer_protocol,
)
from skulk.shared.models.model_cards import (
    ModelCard,
    OutputParserType,
    ReasoningFormat,
)
from skulk.shared.tracing import record_trace_marker
from skulk.shared.types.common import ModelId
from skulk.shared.types.mlx import Model
from skulk.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from skulk.worker.engines.mlx.utils_mlx import (
    detect_thinking_prompt_suffix,
)
from skulk.worker.runner.bootstrap import logger
from skulk.worker.runner.llm_inference.muse_glimmer_text_parser import (
    CONTROL_MARKERS,
    MuseGlimmerTextParser,
    TextEmission,
    ToolCallEmission,
)
from skulk.worker.runner.llm_inference.tool_parsers import (
    ToolParser,
    coerce_tool_calls_to_schema,
    declared_tool_calls,
    find_close_marker,
)

_GEMMA4_THINK_START = "<|channel>thought\n"
_GEMMA4_THINK_END = "<channel|>"
_DEFAULT_TOKEN_THINK_START = "<think>"
_DEFAULT_TOKEN_THINK_END = "</think>"
ParserChunk = GenerationResponse | ToolCallResponse | None


def _thinking_stream_debug_enabled() -> bool:
    """Return whether opt-in thinking stream tracing is enabled."""
    value = preferred_env_value(
        "SKULK_TRACE_THINKING_STREAM",
    )
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _trace_generation_stream(
    label: str,
    model_id: ModelId,
    responses: Generator[ParserChunk],
) -> Generator[ParserChunk]:
    """Log parser-stage generation chunks when thinking stream tracing is enabled."""
    if not _thinking_stream_debug_enabled():
        yield from responses
        return

    for response in responses:
        if response is None:
            logger.info(f"[thinking-stream] stage={label} model={model_id} chunk=None")
            yield None
            continue

        if isinstance(response, ToolCallResponse):
            logger.info(
                f"[thinking-stream] stage={label} model={model_id} "
                f"tool_calls={len(response.tool_calls)}"
            )
            yield response
            continue

        logger.info(
            f"[thinking-stream] stage={label} model={model_id} "
            f"text={response.text!r} token={response.token} "
            f"is_thinking={response.is_thinking} finish_reason={response.finish_reason!r}"
        )
        yield response


@cache
def get_gpt_oss_encoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def apply_all_parsers(
    receiver: Generator[GenerationResponse | None],
    prompt: str,
    tool_parser: ToolParser | None,
    tokenizer: TokenizerWrapper,
    model_type: type[Model],
    model_id: ModelId,
    tools: list[dict[str, Any]] | None,
    model_card: ModelCard | None = None,
    trace_task_id: str | None = None,
    trace_rank: int = 0,
) -> Generator[ParserChunk]:
    mlx_generator = receiver
    mlx_generator = _trace_generation_stream("raw", model_id, mlx_generator)
    capability_profile = resolve_model_capability_profile(
        model_id,
        model_card=model_card,
        tokenizer=tokenizer,
    )

    if uses_muse_glimmer_protocol(capability_profile):
        # Muse Glimmer's channels carry reasoning, content, AND tool calls in
        # one grammar, so a single parser owns the whole split; like gpt-oss
        # it never passes the marker path, so the offered-tools rule is
        # applied downstream.
        mlx_generator = reject_unoffered_tool_calls(
            parse_muse_glimmer(
                mlx_generator, _muse_glimmer_marker_ids(tokenizer), tools
            ),
            tools,
        )
        return _trace_generation_stream("post-all-parsers", model_id, mlx_generator)

    if capability_profile.thinking_format == ReasoningFormat.ChannelDelimited:
        mlx_generator = parse_gemma4_thinking_channels(mlx_generator)
    elif capability_profile.thinking_format == ReasoningFormat.TokenDelimited:
        think_start, think_end = _resolve_token_delimited_markers(tokenizer)
        mlx_generator = parse_thinking_models(
            mlx_generator,
            think_start,
            think_end,
            starts_in_thinking=_detect_thinking_prompt_suffix(
                prompt,
                tokenizer,
                fallback_think_start=think_start,
            ),
        )
        mlx_generator = _trace_generation_stream("post-thinking-parser", model_id, mlx_generator)

    if capability_profile.output_parser == OutputParserType.GptOss or issubclass(
        model_type, GptOssModel
    ):
        # These two parse their own calls out of the token stream, so unlike
        # the marker path they need the offered-tools rule applied downstream.
        mlx_generator = reject_unoffered_tool_calls(
            parse_gpt_oss(mlx_generator), tools
        )
    elif capability_profile.output_parser == OutputParserType.DeepseekV32 or issubclass(
        model_type, DeepseekV32Model
    ):
        mlx_generator = reject_unoffered_tool_calls(
            parse_deepseek_v32(mlx_generator), tools
        )
    elif tool_parser:
        # Always scan, even with no tools offered. The parser is wired from the
        # tokenizer and cannot see the request, so `emit_calls` carries that:
        # with no tools nothing may be returned as a call, which is what makes
        # tool_choice "none" hold, but the block is still recognized so its
        # markers are stripped rather than delivered to the caller.
        mlx_generator = parse_tool_calls(
            mlx_generator,
            tool_parser,
            tools,
            emit_calls=bool(tools),
            trace_task_id=trace_task_id,
            trace_rank=trace_rank,
        )

    mlx_generator = _trace_generation_stream("post-all-parsers", model_id, mlx_generator)
    return mlx_generator


def _resolve_token_delimited_markers(
    tokenizer: TokenizerWrapper,
) -> tuple[str, str]:
    """Resolve token-delimited thinking markers from tokenizer metadata or fallbacks."""
    think_start = tokenizer.think_start or _DEFAULT_TOKEN_THINK_START
    think_end = tokenizer.think_end or _DEFAULT_TOKEN_THINK_END
    return think_start, think_end


def _detect_thinking_prompt_suffix(
    prompt: str,
    tokenizer: TokenizerWrapper,
    *,
    fallback_think_start: str | None = None,
) -> bool:
    """Detect whether the prompt already ends in an opening thinking marker."""
    if detect_thinking_prompt_suffix(prompt, tokenizer):
        return True
    return (
        fallback_think_start is not None
        and prompt.rstrip().endswith(fallback_think_start)
    )


def parse_gemma4_thinking_channels(
    responses: Generator[ParserChunk],
) -> Generator[ParserChunk]:
    """Route Gemma 4 channel-delimited reasoning via ``is_thinking``.

    Gemma 4 does not expose ``TokenizerWrapper.has_thinking`` metadata, but its
    tokenizer config defines assistant reasoning as a ``<|channel>thought``
    block terminated by ``<channel|>``. We strip those channel markers from the
    visible stream and mark the enclosed text as thinking so API adapters can
    route it to reasoning fields instead of assistant content.
    """

    buffer = ""
    is_thinking = False

    def _emit_text(
        template: GenerationResponse,
        text: str,
        *,
        thinking: bool,
    ) -> GenerationResponse | None:
        if not text:
            return None
        return template.model_copy(
            update={"text": text, "is_thinking": thinking, "finish_reason": None}
        )

    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, ToolCallResponse):
            yield response
            continue

        buffer += response.text

        if response.finish_reason is None:
            while True:
                if not is_thinking:
                    start_index = buffer.find(_GEMMA4_THINK_START)
                    if start_index != -1:
                        emitted = _emit_text(
                            response,
                            buffer[:start_index],
                            thinking=False,
                        )
                        if emitted is not None:
                            yield emitted
                        buffer = buffer[start_index + len(_GEMMA4_THINK_START) :]
                        is_thinking = True
                        continue

                    safe_length = len(buffer) - (len(_GEMMA4_THINK_START) - 1)
                    if safe_length > 0:
                        emitted = _emit_text(
                            response,
                            buffer[:safe_length],
                            thinking=False,
                        )
                        if emitted is not None:
                            yield emitted
                        buffer = buffer[safe_length:]
                    break

                end_index = buffer.find(_GEMMA4_THINK_END)
                if end_index != -1:
                    emitted = _emit_text(
                        response,
                        buffer[:end_index],
                        thinking=True,
                    )
                    if emitted is not None:
                        yield emitted
                    buffer = buffer[end_index + len(_GEMMA4_THINK_END) :]
                    is_thinking = False
                    continue

                safe_length = len(buffer) - (len(_GEMMA4_THINK_END) - 1)
                if safe_length > 0:
                    emitted = _emit_text(
                        response,
                        buffer[:safe_length],
                        thinking=True,
                    )
                    if emitted is not None:
                        yield emitted
                    buffer = buffer[safe_length:]
                break
            continue

        while buffer:
            if not is_thinking:
                start_index = buffer.find(_GEMMA4_THINK_START)
                if start_index == -1:
                    emitted = _emit_text(response, buffer, thinking=False)
                    if emitted is not None:
                        yield emitted
                    buffer = ""
                    break

                emitted = _emit_text(response, buffer[:start_index], thinking=False)
                if emitted is not None:
                    yield emitted
                buffer = buffer[start_index + len(_GEMMA4_THINK_START) :]
                is_thinking = True
                continue

            end_index = buffer.find(_GEMMA4_THINK_END)
            if end_index == -1:
                emitted = _emit_text(response, buffer, thinking=True)
                if emitted is not None:
                    yield emitted
                buffer = ""
                break

            emitted = _emit_text(response, buffer[:end_index], thinking=True)
            if emitted is not None:
                yield emitted
            buffer = buffer[end_index + len(_GEMMA4_THINK_END) :]
            is_thinking = False

        # Always emit a terminal chunk with the finish reason so SSE clients close cleanly.
        yield response.model_copy(
            update={"text": "", "is_thinking": False, "finish_reason": response.finish_reason}
        )


def parse_gpt_oss(
    responses: Generator[ParserChunk],
) -> Generator[ParserChunk]:
    encoding = get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    thinking = False
    current_tool_name: str | None = None
    tool_arg_parts: list[str] = []

    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, ToolCallResponse):
            yield response
            continue
        try:
            stream.process(response.token)
        except HarmonyError:
            logger.error("Encountered critical Harmony Error, returning early")
            return

        delta = stream.last_content_delta
        ch = stream.current_channel
        recipient = stream.current_recipient

        # Keep parser-state diagnostics useful without retaining generated text.
        logger.debug(
            f"parse_gpt_oss token={response.token} "
            f"text_chars={len(response.text)} "
            f"has_recipient={recipient is not None} channel={ch!r} "
            f"delta_chars={len(delta or '')} state={stream.state} "
            f"has_current_tool={current_tool_name is not None}"
        )

        if recipient != current_tool_name:
            if current_tool_name is not None:
                prefix = "functions."
                if current_tool_name.startswith(prefix):
                    current_tool_name = current_tool_name[len(prefix) :]
                logger.info(
                    "parse_gpt_oss yielding tool call "
                    f"(name_chars={len(current_tool_name)})"
                )
                yield ToolCallResponse(
                    tool_calls=[
                        ToolCallItem(
                            name=current_tool_name,
                            arguments="".join(tool_arg_parts).strip(),
                        )
                    ],
                    usage=response.usage,
                )
                tool_arg_parts = []
            current_tool_name = recipient

        # If inside a tool call, accumulate arguments
        if current_tool_name is not None:
            if delta:
                tool_arg_parts.append(delta)
            if response.finish_reason is not None:
                yield response.model_copy(update={"text": "".join(tool_arg_parts)})
                tool_arg_parts = []
            continue

        if ch == "analysis" and not thinking:
            thinking = True

        if ch != "analysis" and thinking:
            thinking = False

        if delta:
            yield response.model_copy(update={"text": delta, "is_thinking": thinking})

        if response.finish_reason is not None:
            yield response


def _muse_glimmer_marker_ids(tokenizer: TokenizerWrapper) -> dict[int, str]:
    """Map Muse Glimmer's control-token ids to their marker text.

    The channel markers are tokenizer special tokens. The streaming
    detokenizers normally render them as literal text, which is what the
    parser reads; a detokenizer that drops special tokens would instead hand
    back an empty delta for the token, so the id map lets the stream be
    reconstructed either way. Unknown markers (a vocabulary without one) are
    simply absent.
    """
    marker_by_id: dict[int, str] = {}
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return marker_by_id
    for marker in CONTROL_MARKERS:
        try:
            token_id = cast(object, convert(marker))
        except Exception:  # noqa: BLE001 - a vocabulary probe, best effort
            continue
        if isinstance(token_id, int) and token_id >= 0:
            unknown_id = cast(object, getattr(tokenizer, "unk_token_id", None))
            if token_id != unknown_id:
                marker_by_id[token_id] = marker
    return marker_by_id


def parse_muse_glimmer(
    responses: Generator[ParserChunk],
    marker_by_id: Mapping[int, str],
    tools: list[dict[str, Any]] | None = None,
) -> Generator[ParserChunk]:
    """Route Muse Glimmer channels into reasoning, content, and tool calls.

    Feeds the detokenized stream through :class:`MuseGlimmerTextParser`: the
    ``to=self`` channel becomes ``is_thinking`` chunks, ``to=user`` becomes
    content, and a tool-addressed channel is held until it closes and then
    delivered as one :class:`ToolCallResponse` per channel (the message-level
    coalescing the OpenAI shape wants happens in the API adapter, which stops
    at the first terminal chunk, so calls are emitted as they close and the
    terminal chunk follows them). Control markers never reach the caller.
    ``marker_by_id`` (see :func:`_muse_glimmer_marker_ids`) reconstructs a
    control marker whose delta arrived empty. Argument values are retyped
    against the offered ``tools`` schemas (the ATEM reader keeps scalars as
    strings by design), the same coercion the text dialect path applies.
    """
    parser = MuseGlimmerTextParser()

    def _deliver(
        template: GenerationResponse, emissions: list[TextEmission | ToolCallEmission]
    ) -> Generator[ParserChunk]:
        for emission in emissions:
            if isinstance(emission, ToolCallEmission):
                calls = (
                    coerce_tool_calls_to_schema(emission.calls, tools)
                    if tools
                    else emission.calls
                )
                yield ToolCallResponse(
                    tool_calls=calls,
                    usage=template.usage,
                    stats=template.stats,
                )
                continue
            if emission.text:
                yield template.model_copy(
                    update={
                        "text": emission.text,
                        "is_thinking": emission.is_thinking,
                        "finish_reason": None,
                    }
                )

    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, ToolCallResponse):
            yield response
            continue
        text = response.text
        if not text and response.token in marker_by_id:
            text = marker_by_id[response.token]
        yield from _deliver(response, parser.feed(text))
        if response.finish_reason is not None:
            yield from _deliver(response, parser.flush())
            # Always emit a terminal chunk so SSE clients close cleanly.
            yield response.model_copy(
                update={
                    "text": "",
                    "is_thinking": False,
                    "finish_reason": response.finish_reason,
                }
            )


def parse_deepseek_v32(
    responses: Generator[ParserChunk],
) -> Generator[ParserChunk]:
    """Parse DeepSeek V3.2 DSML tool calls from the generation stream.

    Uses accumulated-text matching (not per-token marker checks) because
    DSML markers like <｜DSML｜function_calls> may span multiple tokens.
    Also handles <think>...</think> blocks for thinking mode.
    """
    from skulk.worker.engines.mlx.dsml_encoding import (
        THINKING_END,
        THINKING_START,
        TOOL_CALLS_END,
        TOOL_CALLS_START,
        parse_dsml_output,
    )

    accumulated = ""
    in_tool_call = False
    thinking = False
    # Tokens buffered while we detect the start of a DSML block
    pending_buffer: list[GenerationResponse] = []
    # Text accumulated during a tool call block
    tool_call_text = ""

    def _try_parse_tool_call(
        text: str, response: GenerationResponse
    ) -> ToolCallResponse | GenerationResponse:
        parsed = parse_dsml_output(text)
        if parsed is not None:
            return ToolCallResponse(
                tool_calls=parsed, usage=response.usage, stats=response.stats
            )
        logger.warning(
            f"DSML tool call parsing failed (generated_chars={len(text)})"
        )
        return response.model_copy(update={"text": text})

    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, ToolCallResponse):
            yield response
            continue

        if response.finish_reason is not None:
            yield from pending_buffer
            pending_buffer.clear()
            if in_tool_call:
                tool_call_text += response.text
                yield (
                    _try_parse_tool_call(tool_call_text, response)
                    if TOOL_CALLS_END in tool_call_text
                    else response.model_copy(update={"text": tool_call_text})
                )
            elif TOOL_CALLS_START in response.text and TOOL_CALLS_END in response.text:
                dsml_start = response.text.index(TOOL_CALLS_START)
                before = response.text[:dsml_start]
                if before:
                    yield response.model_copy(update={"text": before})
                yield _try_parse_tool_call(response.text[dsml_start:], response)
            else:
                yield response
            break

        # ── Handle thinking tags ──
        if not thinking and THINKING_START in response.text:
            thinking = True
            # Yield any text before the <think> tag
            before = response.text[: response.text.index(THINKING_START)]
            if before:
                yield response.model_copy(update={"text": before})
            continue

        if thinking and THINKING_END in response.text:
            thinking = False
            # Yield any text after the </think> tag
            after = response.text[
                response.text.index(THINKING_END) + len(THINKING_END) :
            ]
            if after:
                yield response.model_copy(update={"text": after, "is_thinking": False})
            continue

        if thinking:
            yield response.model_copy(update={"is_thinking": True})
            continue

        # ── Handle tool call accumulation ──
        if in_tool_call:
            tool_call_text += response.text
            if TOOL_CALLS_END in tool_call_text:
                yield _try_parse_tool_call(tool_call_text, response)
                in_tool_call = False
                tool_call_text = ""
            continue

        # ── Detect start of tool call block ──
        accumulated += response.text

        if TOOL_CALLS_START in accumulated:
            # The start marker might be split across pending_buffer + current token
            start_idx = accumulated.index(TOOL_CALLS_START)
            # Yield any pending tokens that are purely before the marker
            pre_text = accumulated[:start_idx]
            if pre_text:
                # Flush pending buffer tokens that contributed text before the marker
                for buf_resp in pending_buffer:
                    if not pre_text:
                        break
                    chunk = buf_resp.text
                    if len(chunk) <= len(pre_text):
                        yield buf_resp
                        pre_text = pre_text[len(chunk) :]
                    else:
                        yield buf_resp.model_copy(update={"text": pre_text})
                        pre_text = ""
            pending_buffer = []
            tool_call_text = accumulated[start_idx:]
            accumulated = ""

            # Check if the end marker is already present (entire tool call in one token)
            if TOOL_CALLS_END in tool_call_text:
                yield _try_parse_tool_call(tool_call_text, response)
                tool_call_text = ""
            else:
                in_tool_call = True
            continue

        # Check if accumulated text might be the start of a DSML marker
        # Buffer tokens if we see a partial match at the end
        if _could_be_dsml_prefix(accumulated):
            pending_buffer.append(response)
            continue

        # No partial match — flush all pending tokens and the current one
        yield from pending_buffer
        pending_buffer.clear()
        accumulated = ""
        yield response

    # Flush any remaining pending buffer at generator end
    yield from pending_buffer


def _could_be_dsml_prefix(text: str) -> bool:
    """Check if the end of text could be the start of a DSML function_calls marker.

    We look for suffixes of text that are prefixes of the TOOL_CALLS_START pattern.
    This allows us to buffer tokens until we can determine if a tool call is starting.
    """
    from skulk.worker.engines.mlx.dsml_encoding import TOOL_CALLS_START

    # Only check the last portion of text that could overlap with the marker
    max_check = len(TOOL_CALLS_START)
    tail = text[-max_check:] if len(text) > max_check else text

    # Check if any suffix of tail is a prefix of TOOL_CALLS_START
    for i in range(len(tail)):
        suffix = tail[i:]
        if TOOL_CALLS_START.startswith(suffix):
            return True
    return False


def parse_thinking_models(
    responses: Generator[ParserChunk],
    think_start: str | None,
    think_end: str | None,
    starts_in_thinking: bool = True,
) -> Generator[ParserChunk]:
    """Route thinking tokens via is_thinking flag.

    Swallows think tag tokens, sets ``is_thinking`` on all others, and buffers
    partial marker fragments so split or fused ``<think>`` tags do not leak into
    visible output.

    Always yields a terminal chunk with ``finish_reason`` so the stream closes
    cleanly even when the model ends inside a thinking block.
    """
    if think_start is None or think_end is None:
        for response in responses:
            yield response
        return

    buffer = ""
    is_thinking = starts_in_thinking

    def _emit_text(
        template: GenerationResponse,
        text: str,
        *,
        thinking: bool,
    ) -> GenerationResponse | None:
        if not text:
            return None
        return template.model_copy(
            update={"text": text, "is_thinking": thinking, "finish_reason": None}
        )

    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, ToolCallResponse):
            yield response
            continue

        buffer += response.text

        if response.finish_reason is None:
            while True:
                if not is_thinking:
                    start_index = buffer.find(think_start)
                    if start_index != -1:
                        emitted = _emit_text(
                            response,
                            buffer[:start_index],
                            thinking=False,
                        )
                        if emitted is not None:
                            yield emitted
                        buffer = buffer[start_index + len(think_start) :]
                        is_thinking = True
                        continue

                    safe_length = len(buffer) - (len(think_start) - 1)
                    if safe_length > 0:
                        emitted = _emit_text(
                            response,
                            buffer[:safe_length],
                            thinking=False,
                        )
                        if emitted is not None:
                            yield emitted
                        buffer = buffer[safe_length:]
                    break

                end_index = buffer.find(think_end)
                if end_index != -1:
                    emitted = _emit_text(
                        response,
                        buffer[:end_index],
                        thinking=True,
                    )
                    if emitted is not None:
                        yield emitted
                    buffer = buffer[end_index + len(think_end) :]
                    is_thinking = False
                    continue

                safe_length = len(buffer) - (len(think_end) - 1)
                if safe_length > 0:
                    emitted = _emit_text(
                        response,
                        buffer[:safe_length],
                        thinking=True,
                    )
                    if emitted is not None:
                        yield emitted
                    buffer = buffer[safe_length:]
                break
            continue

        while buffer:
            if not is_thinking:
                start_index = buffer.find(think_start)
                if start_index == -1:
                    emitted = _emit_text(response, buffer, thinking=False)
                    if emitted is not None:
                        yield emitted
                    buffer = ""
                    break

                emitted = _emit_text(response, buffer[:start_index], thinking=False)
                if emitted is not None:
                    yield emitted
                buffer = buffer[start_index + len(think_start) :]
                is_thinking = True
                continue

            end_index = buffer.find(think_end)
            if end_index == -1:
                emitted = _emit_text(response, buffer, thinking=True)
                if emitted is not None:
                    yield emitted
                buffer = ""
                break

            emitted = _emit_text(response, buffer[:end_index], thinking=True)
            if emitted is not None:
                yield emitted
            buffer = buffer[end_index + len(think_end) :]
            is_thinking = False

        yield response.model_copy(
            update={"text": "", "is_thinking": False, "finish_reason": response.finish_reason}
        )


def reject_unoffered_tool_calls(
    responses: Generator[ParserChunk], tools: list[dict[str, Any]] | None
) -> Generator[ParserChunk]:
    """Keep a family parser from returning a tool the caller never offered.

    gpt-oss and DeepSeek V3.2 parse their calls from the token stream
    themselves, so they never pass through the offered-tools filter the marker
    path applies. Observed live on gpt-oss: a request sending
    ``tool_choice: "none"``, which removes the tools, still came back with a
    call, and its name carried the harmony namespace prefix as well.

    The rejected call is delivered as content, which is what every other path
    here does with a block naming no offered tool, so the caller sees what the
    model did rather than an empty answer.
    """

    template: GenerationResponse | None = None
    pending_text = ""
    rejected: ToolCallResponse | None = None
    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, ToolCallResponse):
            kept = declared_tool_calls(response.tool_calls, tools) if tools else []
            if kept:
                yield response.model_copy(update={"tool_calls": kept})
                continue
            # Held rather than emitted with a finish reason of its own: these
            # streams usually carry a terminal chunk after the call, and adding
            # a second terminal would end the stream at the consumer before the
            # real one arrives. If none follows, it is released at the end.
            rendered = [
                json.dumps({"name": call.name, "arguments": call.arguments})
                for call in response.tool_calls
            ]
            # Separated, so several rejected calls in a row do not run together
            # into text a caller cannot read back.
            pending_text = "\n".join(
                part for part in [pending_text, *rendered] if part
            )
            rejected = response
            continue
        template = response
        if pending_text:
            yield response.model_copy(
                update={
                    "text": pending_text + response.text,
                    "token": 0,
                    "is_thinking": False,
                }
            )
            pending_text = ""
            continue
        yield response
    if pending_text:
        # The rejected response's accounting is the message's accounting, so it
        # is carried rather than replaced with a fabricated empty one.
        base = template or GenerationResponse(text="", token=0, usage=None)
        yield base.model_copy(
            update={
                "text": pending_text,
                "token": 0,
                "is_thinking": False,
                "finish_reason": "stop",
                "usage": rejected.usage if rejected is not None else base.usage,
                "stats": rejected.stats if rejected is not None else base.stats,
            }
        )


def _block_as_content(text: str, tool_parser: ToolParser) -> str:
    """Strip a dialect's markers from a block being delivered as content.

    A block that named no offered tool, or that did not parse but reads as an
    answer, is handed back to the caller as content. Handing it back verbatim
    puts the dialect's control tokens in their answer text, which is the leak
    this whole path exists to prevent: `<|python_tag|>` reached a caller that
    way, found by the harness's tool-contract suite.

    The error path deliberately does NOT use this. There the raw block is the
    evidence of what was malformed, and the response is already flagged as an
    error rather than offered as an answer.
    """

    stripped = text
    for marker in (*tool_parser.start_markers, tool_parser.end_parsing):
        if marker and marker != "{":
            stripped = stripped.replace(marker, "")
    return stripped


def _block_start_index(
    text: str, tool_parser: ToolParser, *, at_message_start: bool
) -> tuple[int, bool] | None:
    """Where a tool-call block begins in ``text``, or ``None``.

    Distinctive markers open a block wherever they appear, because models
    routinely write a sentence before calling ("I'll check that." then the
    call). The unmarked dialect's opening marker is ``{``, which appears in
    ordinary prose and JSON answers, so it opens a block only at the start of
    the message, which is the only place the families using it write a call.

    Returns the index plus whether the anchored primary marker is what opens
    there: that opening is provisional (the tentative classification in
    :func:`parse_tool_calls` decides whether it is a call or a JSON answer),
    while a distinctive marker commits the block immediately.
    """

    earliest: int | None = None
    for marker in tool_parser.extra_start_parsing:
        found = text.find(marker)
        if found != -1 and (earliest is None or found < earliest):
            earliest = found

    anchored_open = False
    if not tool_parser.anchored:
        found = text.find(tool_parser.start_parsing)
        if found != -1 and (earliest is None or found < earliest):
            earliest = found
    elif at_message_start:
        stripped = text.lstrip()
        if stripped.startswith(tool_parser.start_parsing):
            found = len(text) - len(stripped)
            if earliest is None or found < earliest:
                earliest = found
                anchored_open = True
    if earliest is None:
        return None
    return earliest, anchored_open


_ANCHORED_CALL_KEYS = frozenset({"name", "arguments", "parameters"})
# Keys some Llama variants add around the call signature without changing
# what the object is; they neither make the object a call nor rule it out.
_ANCHORED_NEUTRAL_KEYS = frozenset({"type", "id"})


def _skip_json_value(text: str, index: int) -> int | None:
    """Position just past the JSON value starting at ``index``, or ``None``.

    ``None`` means the value is still incomplete in the buffered prefix. A
    scalar running to the end of the buffer is also incomplete, because more
    of it may still arrive.
    """

    length = len(text)
    char = text[index]
    if char == '"':
        escaped = False
        for position in range(index + 1, length):
            current = text[position]
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif current == '"':
                return position + 1
        return None
    if char in "{[":
        depth = 0
        in_string = False
        escaped = False
        for position in range(index, length):
            current = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current in "{[":
                depth += 1
            elif current in "}]":
                depth -= 1
                if depth == 0:
                    return position + 1
        return None
    for position in range(index, length):
        if text[position] in ",}]" or text[position].isspace():
            return position
    return None


def _classify_anchored_prefix(text: str) -> str:
    """Decide whether a message-opening ``{`` prefix is a call or an answer.

    Returns ``"call"``, ``"content"``, or ``"undecided"``. The unmarked
    dialect's block opens on ``{``, which a model asked for JSON also emits,
    and its closing token is a generation stop that never arrives as text, so
    without this the whole message was held until the terminal chunk and a
    plain JSON answer lost incremental streaming entirely. The prefix is a
    call once a top-level ``"name"`` string and an ``"arguments"`` or
    ``"parameters"`` value are both distinguishable, and content the moment
    it can no longer be one: a top-level key outside the call shape, a
    non-string name, an argument value that is neither an object nor the
    string-encoded form the OpenAI wire shape allows, malformed JSON, or
    the object closing without the signature. The hold is therefore bounded
    by the first decisive key rather than by the message.
    """

    length = len(text)
    index = 0
    while index < length and text[index].isspace():
        index += 1
    if index >= length:
        return "undecided"
    if text[index] != "{":
        return "content"
    index += 1
    has_name = False
    has_args = False
    while True:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            return "undecided"
        char = text[index]
        if char == "}":
            return "call" if has_name and has_args else "content"
        if char != '"':
            return "content"
        key_end = _skip_json_value(text, index)
        if key_end is None:
            return "undecided"
        key = text[index + 1 : key_end - 1]
        if (
            key not in _ANCHORED_CALL_KEYS
            and key not in _ANCHORED_NEUTRAL_KEYS
        ):
            return "content"
        index = key_end
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            return "undecided"
        if text[index] != ":":
            return "content"
        index += 1
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            return "undecided"
        value_start = text[index]
        if key == "name":
            if value_start != '"':
                return "content"
            has_name = True
        elif key in _ANCHORED_CALL_KEYS:
            # Arguments must be an object, or the JSON-encoded string form
            # the OpenAI wire shape allows; anything else is not a call.
            if value_start not in '{"':
                return "content"
            has_args = True
        if has_name and has_args:
            return "call"
        value_end = _skip_json_value(text, index)
        if value_end is None:
            return "undecided"
        index = value_end
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            return "undecided"
        if text[index] == ",":
            index += 1
            continue
        if text[index] == "}":
            return "call" if has_name and has_args else "content"
        return "content"


def _partial_marker_suffix_length(text: str, markers: tuple[str, ...]) -> int:
    """Length of the trailing run of ``text`` that could still become a marker.

    Held back rather than emitted, so a marker split across chunks is still
    recognized. Bounded by the longest marker, so this is a few characters of
    latency at most and never an unbounded buffer.
    """

    longest = max(len(marker) for marker in markers) - 1
    for length in range(min(longest, len(text)), 0, -1):
        tail = text[-length:]
        if any(marker.startswith(tail) for marker in markers):
            return length
    return 0


def _scan_remaining_blocks(
    text: str,
    tool_parser: ToolParser,
    tools: list[dict[str, Any]] | None,
    *,
    emit_calls: bool,
) -> tuple[list[ToolCallItem], str, str | None]:
    """Parse every remaining block in a complete text.

    Used once generation has ended, where there is no further chunk to drive
    the streaming scan and the rest of the message is already in hand. Returns
    the calls found, the text that was not part of any block, and the raw
    malformed block if one was met, so a message that puts a call the caller
    cannot run before one they can still delivers the second call and the
    surrounding prose.

    A block met here follows the same rules as one met by the streaming scan:
    ``emit_calls`` of ``False`` keeps every call out of the result no matter
    how the block parses, a block delivered as content has its dialect markers
    stripped rather than being handed back verbatim, and a closed marked block
    that does not parse is the same failure it is mid-stream. The malformed
    block ends the scan and is returned raw, as evidence, because whether a
    message errors must not depend on where its chunks were split.
    """

    calls: list[ToolCallItem] = []
    leftover: list[str] = []
    remaining = text
    while remaining:
        found = _block_start_index(remaining, tool_parser, at_message_start=False)
        if found is None:
            leftover.append(remaining)
            break
        start, _ = found
        end = find_close_marker(
            remaining, tool_parser.end_parsing, tool_parser.close_scan, start=start
        )
        if end == -1:
            leftover.append(remaining)
            break
        end_of_block = end + len(tool_parser.end_parsing)
        block = remaining[start:end_of_block]
        parsed = tool_parser.parse(block.strip(), tools=tools)
        if parsed is None and not tool_parser.unparsed_is_text:
            leftover.append(remaining[:start])
            return calls, "".join(leftover), block
        kept = (
            declared_tool_calls(parsed, tools)
            if parsed is not None and emit_calls
            else []
        )
        leftover.append(remaining[:start])
        if kept:
            calls.extend(kept)
        else:
            # Not a call the caller may run, so it is content like any other
            # rejected block, markers stripped the same way.
            leftover.append(_block_as_content(block, tool_parser))
        remaining = remaining[end_of_block:]
    return calls, "".join(leftover), None


def parse_tool_calls(
    responses: Generator[ParserChunk],
    tool_parser: ToolParser,
    tools: list[dict[str, Any]] | None,
    *,
    emit_calls: bool = True,
    trace_task_id: str | None = None,
    trace_rank: int = 0,
) -> Generator[ParserChunk]:
    """Recover tool calls from the generated stream, one response per message.

    The calls of every block in a message are coalesced into a single
    ``ToolCallResponse``. That is the OpenAI shape, where one assistant message
    carries a ``tool_calls`` array, and it is what makes a model's parallel
    calls survive: several families write each call in its own block, and the
    consumer of this stream stops at the first chunk carrying a finish reason,
    so a response per block would deliver the first call and drop the rest.
    """

    in_tool_call = False
    # Held until the message ends rather than emitted per block, so several
    # blocks arrive as one response. The stream does not end when generation
    # does (the source keeps idling), so the terminal chunk is the signal.
    accumulated_calls: list[ToolCallItem] = []
    last_response: GenerationResponse | None = None
    tool_call_text_parts: list[str] = []
    # A chunk is whatever the streaming detokenizer could resolve this step, not
    # a token: an opening marker that is one token id still arrives split across
    # chunks ("<tool", "_", "c", "all>"). Testing each chunk on its own misses
    # the marker for most models, so text is scanned across chunk boundaries by
    # carrying forward only the trailing run that could still become a marker.
    # That run is shorter than the longest marker, so ordinary answers stream
    # with at most a few characters of latency and nothing is ever held for a
    # message that turns out not to contain a call.
    held_text = ""
    at_message_start = True
    # An anchored open (the unmarked dialect's message-opening "{") is
    # provisional: the block is held only until the buffered prefix is
    # distinguishably a call or distinguishably not one, so a plain JSON
    # answer streams incrementally instead of losing all output to a hold
    # that could only resolve at the terminal chunk.
    tentative = False

    def _scan_outside_block(
        scanned: str, response: GenerationResponse
    ) -> Generator[ParserChunk]:
        """Scan text while no block is open; may open one (possibly tentative).

        Also the re-entry point for a tentative block that turned out to be
        content: the released text goes back through this same scan, so a
        distinctive marker later in it still opens a real block and the rest
        streams out under the usual partial-marker holding.
        """

        nonlocal in_tool_call, tentative, held_text, at_message_start
        nonlocal accumulated_calls
        found = _block_start_index(
            scanned, tool_parser, at_message_start=at_message_start
        )
        if found is not None:
            start, anchored_open = found
            preamble = scanned[:start]
            if preamble:
                yield response.model_copy(
                    update={
                        "text": preamble,
                        "token": 0,
                        "finish_reason": None,
                    }
                )
            in_tool_call = True
            tentative = anchored_open
            held_text = ""
            at_message_start = False
            tool_call_text_parts.append(scanned[start:])
            return
        keep = _partial_marker_suffix_length(scanned, tool_parser.start_markers)
        if response.finish_reason is not None:
            # Nothing more is coming, so a partial marker is just text.
            keep = 0
        emitted = scanned[: len(scanned) - keep]
        held_text = scanned[len(scanned) - keep :]
        if emitted.strip():
            at_message_start = False
        if response.finish_reason is not None and accumulated_calls:
            # A call was found earlier in this message and the tool
            # response has to be the terminal chunk, so this trailing
            # text is released without the finish reason.
            if emitted:
                yield response.model_copy(
                    update={
                        "text": emitted,
                        "token": 0,
                        "finish_reason": None,
                    }
                )
            yield ToolCallResponse(
                tool_calls=accumulated_calls,
                usage=response.usage,
                stats=response.stats,
            )
            accumulated_calls = []
            return
        if emitted == response.text and not held_text:
            yield response
        elif emitted or response.finish_reason is not None:
            yield response.model_copy(update={"text": emitted, "token": 0})

    def _finish_message(
        response: GenerationResponse,
    ) -> Generator[ParserChunk]:
        """Close out a message once a block has been dealt with.

        Every exit from the close site routes through here so the same three
        rules hold whatever the block turned out to be: the rest of the message
        is parsed rather than emitted whole (no further chunk will arrive to
        drive the streaming scan), the calls found across the whole message are
        delivered as one response, and exactly one chunk carries the finish
        reason, since the consumer stops at the first one that does.
        """

        nonlocal held_text, accumulated_calls
        if response.finish_reason is None:
            return
        terminal_sent = False
        if held_text:
            more_calls, leftover, malformed = _scan_remaining_blocks(
                held_text, tool_parser, tools, emit_calls=emit_calls
            )
            held_text = ""
            if malformed is not None:
                # The same failure the streaming close path reports: a closed
                # marked block that does not parse errors the message, calls
                # and all, so the outcome does not depend on where the chunks
                # were split. The raw block is the evidence, unstripped.
                logger.warning(
                    "Tool-call parsing failed in terminal suffix "
                    f"(generated_chars={len(malformed)})"
                )
                if trace_task_id is not None:
                    record_trace_marker(
                        "tool_call_parse_error",
                        trace_rank,
                        category="tooling",
                        task_id=trace_task_id,
                        tags=["tool_call", "error"],
                        attrs={"raw_length": len(malformed)},
                    )
                accumulated_calls = []
                if leftover:
                    yield response.model_copy(
                        update={
                            "text": leftover,
                            "token": 0,
                            "finish_reason": None,
                        }
                    )
                yield response.model_copy(
                    update={"text": malformed, "token": 0, "finish_reason": "error"}
                )
                return
            accumulated_calls.extend(more_calls)
            if leftover:
                carries_finish = not accumulated_calls
                yield response.model_copy(
                    update={
                        "text": leftover,
                        "token": 0,
                        "finish_reason": response.finish_reason
                        if carries_finish
                        else None,
                    }
                )
                terminal_sent = carries_finish
        if accumulated_calls:
            yield ToolCallResponse(
                tool_calls=accumulated_calls,
                usage=response.usage,
                stats=response.stats,
            )
            accumulated_calls = []
            return
        if not terminal_sent:
            # The block's own content went out without the finish reason, so
            # something still has to end the stream.
            yield response.model_copy(update={"text": "", "token": 0})

    for response in responses:
        if response is None:
            yield None
            continue
        if isinstance(response, ToolCallResponse):
            yield response
            continue

        # Reasoning is never part of a tool-call block: this parser runs
        # downstream of the thinking parser, and a call a model only
        # contemplated inside its reasoning must not be executed. Passing those
        # chunks straight through also keeps them out of the opening decision,
        # so a thinking model that reasons before calling still has its marker
        # examined when the visible answer begins.
        if response.is_thinking:
            yield response
            continue

        last_response = response
        if in_tool_call:
            tool_call_text_parts.append(response.text)
        else:
            yield from _scan_outside_block(held_text + response.text, response)
            if not in_tool_call:
                continue
        if tentative:
            verdict = _classify_anchored_prefix("".join(tool_call_text_parts))
            if verdict == "call":
                tentative = False
            elif verdict == "content":
                released = "".join(tool_call_text_parts)
                tentative = False
                in_tool_call = False
                tool_call_text_parts = []
                yield from _scan_outside_block(released, response)
                if not in_tool_call:
                    continue
                # A distinctive marker inside the released text opened a
                # real block; it cannot be tentative again (the anchored
                # marker only opens at message start), so fall through to
                # the closing scan.
            elif response.finish_reason is None:
                # Still consistent with both readings: keep holding. The
                # hold is bounded by the first decisive key, not by the
                # message.
                continue
            # Undecided on the terminal chunk falls through: the
            # end-of-generation parse decides what the block was, exactly
            # as it did before the tentative open existed.
        # The closing marker splits across chunks for the same reason the
        # opening one does, so it is located in the accumulated block rather
        # than tested against one chunk. Locating rather than matching the end
        # also matters because a model may keep writing after the call ("...
        # </tool_call> Done."): everything past the marker is ordinary text and
        # goes back to the opening scan, where a second call in the same
        # message is still found. The quote-aware modes keep a closing marker
        # inside a quoted argument value from truncating the block.
        block_so_far = "".join(tool_call_text_parts)
        end_index = find_close_marker(
            block_so_far, tool_parser.end_parsing, tool_parser.close_scan
        )
        if end_index != -1:
            end_of_block = end_index + len(tool_parser.end_parsing)
            combined = block_so_far[:end_of_block]
            held_text = block_so_far[end_of_block:]
            tool_call_text_parts = [combined]
            parsed = tool_parser.parse(combined.strip(), tools=tools)
            if parsed is not None:
                # With no tools offered nothing may be called, but the block
                # still has to be recognized: skipping the scan entirely left
                # the markers in the answer, which is what a caller saw.
                kept = declared_tool_calls(parsed, tools) if emit_calls else []
                if not kept:
                    logger.info(
                        "Block named no offered tool, emitting it as content "
                        f"(parsed_calls={len(parsed)})"
                    )
                    in_tool_call = False
                    tool_call_text_parts = []
                    # The remainder stays with the scan rather than being
                    # emitted here, so a further call in the trailing text is
                    # still found.
                    yield response.model_copy(
                        update={
                            "text": _block_as_content(combined, tool_parser),
                            "token": 0,
                            "finish_reason": None,
                        }
                    )
                    yield from _finish_message(response)
                    continue
                parsed = kept
            logger.info(
                "Parsed generated tool-call block "
                f"(chunks={len(tool_call_text_parts)}, "
                f"generated_chars={len(combined)}, "
                f"parsed_calls={len(parsed) if parsed is not None else 0})"
            )
            in_tool_call = False
            tool_call_text_parts = []

            if parsed is None and tool_parser.unparsed_is_text:
                logger.info(
                    "Unmarked block did not parse as a tool call, "
                    f"emitting it as content (generated_chars={len(combined)})"
                )
                yield response.model_copy(
                    update={
                        "text": _block_as_content(combined, tool_parser),
                        "token": 0,
                        "finish_reason": None,
                    }
                )
                yield from _finish_message(response)
                continue

            if parsed is None:
                logger.warning(
                    "Tool-call parsing failed "
                    f"(generated_chars={len(combined)})"
                )
                if trace_task_id is not None:
                    record_trace_marker(
                        "tool_call_parse_error",
                        trace_rank,
                        category="tooling",
                        task_id=trace_task_id,
                        tags=["tool_call", "error"],
                        attrs={"raw_length": len(combined)},
                    )
                # The error chunk is the stream's one terminal: calls held
                # from earlier blocks in this message are dropped rather than
                # released after it, where the consumer would never look.
                accumulated_calls = []
                yield response.model_copy(
                    update={"text": combined, "token": 0, "finish_reason": "error"}
                )
                break

            if trace_task_id is not None:
                record_trace_marker(
                    "tool_call_parsed",
                    trace_rank,
                    category="tooling",
                    task_id=trace_task_id,
                    tags=["tool_call"],
                    attrs={"tool_call_count": len(parsed)},
                )
            accumulated_calls.extend(parsed)
            yield from _finish_message(response)
            continue

        if response.finish_reason is not None:
            # Generation ended while inside a tool-call block. That is not
            # always truncation: several families close a call by ending the
            # message rather than by emitting a closing marker. Llama 3.1+ is
            # the clearest case, where <|eom_id|> means "end of message,
            # handing off to a tool", so the block is complete and the closing
            # marker never arrives. Try to parse before declaring it garbage;
            # only a block that genuinely does not parse falls through to the
            # error path, which is what truncation actually looks like.
            combined = "".join(tool_call_text_parts)
            # Truncation is the one case where an unclosed block must not be
            # read as a call. A marker dialect's inner parser only strips the
            # closing marker if it is there, so a call cut off at max_tokens
            # would otherwise parse and be handed to the caller to execute.
            # The split parse also reports visible text the model wrote after
            # its call, which only the dialect itself can find here: there is
            # no closing marker in the text to split at.
            parsed, trailing_text = (
                (None, "")
                if response.finish_reason == "length"
                else tool_parser.parse_split(combined.strip(), tools=tools)
            )
            if parsed is not None and (
                not emit_calls or not declared_tool_calls(parsed, tools)
            ):
                logger.info(
                    "Block named no offered tool, emitting it as content "
                    f"(parsed_calls={len(parsed)})"
                )
                yield response.model_copy(
                    update={
                        "text": _block_as_content(combined, tool_parser),
                        "token": 0,
                        "finish_reason": None,
                    }
                )
                yield from _finish_message(response)
                break
            if parsed and emit_calls:
                parsed = declared_tool_calls(parsed, tools)
                # Trailing prose after the call rejoins the message-finishing
                # scan, the same path text after a closing marker takes, so it
                # reaches the caller as content and the tool response stays
                # the terminal chunk.
                held_text = trailing_text
                logger.info(
                    "Parsed tool-call block closed by end of generation "
                    f"(generated_chars={len(combined)}, parsed_calls={len(parsed)})"
                )
                if trace_task_id is not None:
                    record_trace_marker(
                        "tool_call_parsed",
                        trace_rank,
                        category="tooling",
                        task_id=trace_task_id,
                        tags=["tool_call"],
                        attrs={"tool_call_count": len(parsed)},
                    )
                accumulated_calls.extend(parsed)
                yield from _finish_message(response)
                break
            if tool_parser.unparsed_is_text:
                logger.info(
                    "Unmarked block ended without parsing as a tool call, "
                    f"emitting it as content (generated_chars={len(combined)})"
                )
                yield response.model_copy(
                    update={
                        "text": _block_as_content(combined, tool_parser),
                        "token": 0,
                        "finish_reason": None,
                    }
                )
                yield from _finish_message(response)
                break
            logger.info(
                "tool call parsing interrupted, yield partial tool call as text"
            )
            if accumulated_calls:
                # An earlier block in this message did produce calls, so they
                # are delivered rather than lost to the truncated one. The
                # finish reason is withheld here or the consumer stops on this
                # chunk and never sees them.
                yield response.model_copy(
                    update={
                        "text": _block_as_content(combined, tool_parser),
                        "token": 0,
                        "finish_reason": None,
                    }
                )
                yield from _finish_message(response)
                break
            yield response.model_copy(
                update={
                    "text": combined,
                    "token": 0,
                    "finish_reason": "error",
                }
            )

    if accumulated_calls and last_response is not None:
        # A finite source can end without ever carrying a finish reason, so the
        # calls held for coalescing are released here rather than dropped.
        yield ToolCallResponse(
            tool_calls=accumulated_calls,
            usage=last_response.usage,
            stats=last_response.stats,
        )
