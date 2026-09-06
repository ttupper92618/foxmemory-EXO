"""``parse_muse_glimmer``: the MLX-side generator over the channel parser."""

from __future__ import annotations

import json
from collections.abc import Generator

from skulk.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from skulk.worker.runner.llm_inference.model_output_parsers import (
    ParserChunk,
    parse_muse_glimmer,
)

REASONING = " to=self<|message|>Thinking.<|eom|>"
CALL = (
    "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n"
    '<atem:invoke name="get_weather">\n'
    '<atem:parameter name="city">Denver</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls><|eom|>"
)
ANSWER = "<|start|>assistant to=user<|message|>Sunny.<|eot|>"


def _chunks(pieces: list[tuple[str, int]]) -> Generator[ParserChunk]:
    last = len(pieces) - 1
    for index, (text, token) in enumerate(pieces):
        yield GenerationResponse(
            text=text,
            token=token,
            usage=None,
            finish_reason="stop" if index == last else None,
        )


def _run(
    pieces: list[tuple[str, int]],
    marker_by_id: dict[int, str] | None = None,
    tools: list[dict[str, object]] | None = None,
) -> list[ParserChunk]:
    return list(parse_muse_glimmer(_chunks(pieces), marker_by_id or {}, tools))


def _texts(chunks: list[ParserChunk], *, thinking: bool) -> str:
    return "".join(
        chunk.text
        for chunk in chunks
        if isinstance(chunk, GenerationResponse) and chunk.is_thinking is thinking
    )


def _calls(chunks: list[ParserChunk]) -> list[list[tuple[str, object]]]:
    return [
        [(call.name, json.loads(call.arguments)) for call in chunk.tool_calls]
        for chunk in chunks
        if isinstance(chunk, ToolCallResponse)
    ]


def _last(chunks: list[ParserChunk]) -> GenerationResponse:
    tail = chunks[-1]
    assert isinstance(tail, GenerationResponse)
    return tail


def test_channels_route_to_thinking_content_and_calls() -> None:
    chunks = _run([(char, 1) for char in REASONING + CALL + ANSWER])
    assert _texts(chunks, thinking=True) == "Thinking."
    assert _texts(chunks, thinking=False) == "Sunny."
    assert _calls(chunks) == [[("get_weather", {"city": "Denver"})]]
    assert _last(chunks).finish_reason == "stop"
    assert _last(chunks).text == ""


def test_marker_ids_are_reconstructed_when_detokenizer_drops_them() -> None:
    ids = {200006: "<|start|>", 200008: "<|message|>", 200007: "<|eom|>", 200009: "<|eot|>"}
    pieces: list[tuple[str, int]] = [
        (" to=self", 1),
        ("", 200008),
        ("Thinking.", 2),
        ("", 200007),
        ("", 200006),
        ("assistant to=user", 3),
        ("", 200008),
        ("Sunny.", 4),
        ("", 200009),
    ]
    chunks = _run(pieces, ids)
    assert _texts(chunks, thinking=True) == "Thinking."
    assert _texts(chunks, thinking=False) == "Sunny."
    assert _last(chunks).finish_reason == "stop"


def test_terminal_chunk_always_closes_the_stream() -> None:
    chunks = _run([(" to=user<|message|>Hi", 1)])
    assert _texts(chunks, thinking=False) == "Hi"
    assert _last(chunks).finish_reason == "stop"


def test_arguments_are_retyped_against_the_offered_schema() -> None:
    # The ATEM reader keeps scalars as strings; the offered tool's schema says
    # count is an integer, so the MLX path must coerce like the text path does.
    call = (
        "<|start|>assistant to=repeat<|message|><atem:function_calls>\n"
        '<atem:invoke name="repeat">\n'
        '<atem:parameter name="count">3</atem:parameter>\n'
        '<atem:parameter name="word">hi</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    )
    tools: list[dict[str, object]] = [
        {
            "type": "function",
            "function": {
                "name": "repeat",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "word": {"type": "string"},
                    },
                },
            },
        }
    ]
    chunks = _run([(REASONING + call, 1)], tools=tools)
    assert _calls(chunks) == [[("repeat", {"count": 3, "word": "hi"})]]
    # Without a schema in hand the value stays as written.
    assert _calls(_run([(REASONING + call, 1)])) == [
        [("repeat", {"count": "3", "word": "hi"})]
    ]


def test_parallel_channels_arrive_as_one_tool_call_response() -> None:
    # The API stream stops at the first terminal tool chunk, so two channels
    # must be coalesced into one response or the second call is lost.
    second = (
        "<|start|>assistant to=get_time<|message|><atem:function_calls>\n"
        '<atem:invoke name="get_time">\n'
        '<atem:parameter name="zone">America/Denver</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eot|>"
    )
    chunks = _run([(char, 1) for char in REASONING + CALL + second])
    assert _calls(chunks) == [
        [("get_weather", {"city": "Denver"}), ("get_time", {"zone": "America/Denver"})]
    ]
    # The single call response precedes the terminal chunk.
    kinds = [type(c).__name__ for c in chunks if c is not None]
    assert kinds.index("ToolCallResponse") == len(kinds) - 2
