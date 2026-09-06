"""Muse Glimmer channel/ATEM streaming parser.

The wire format interleaves control markers with text, so the parser's bugs
are split bugs: a marker divided across two deltas, a header that arrives one
character at a time, a tool body cut off before its closing marker. Every
message here is therefore replayed at every single split point and character
by character, and the streamed result must equal the whole-message result.
"""

from __future__ import annotations

import json

import pytest

from skulk.worker.runner.llm_inference.muse_glimmer_text_parser import (
    MuseGlimmerTextParser,
    TextEmission,
    ToolCallEmission,
)

REASONING = " to=self<|message|>The user wants 17 times 23.<|eom|>"
ANSWER = "<|start|>assistant to=user<|message|>391<|eot|>"
ONE_CALL = (
    "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n"
    '<atem:invoke name="get_weather">\n'
    '<atem:parameter name="city">Denver</atem:parameter>\n'
    '<atem:parameter name="units">{"scale": "c", "precision": 1}</atem:parameter>\n'
    '<atem:parameter name="verbose">true</atem:parameter>\n'
    "</atem:invoke>\n"
    "</atem:function_calls><|eom|>"
)
SECOND_CALL = (
    "<|start|>assistant to=get_time<|message|><atem:function_calls>\n"
    '<atem:invoke name="get_time">\n'
    '<atem:parameter name="zone">America/Denver</atem:parameter>\n'
    "</atem:invoke>\n"
    "</atem:function_calls><|eot|>"
)


def _stream(text: str, *, split: int | None = None) -> list[object]:
    parser = MuseGlimmerTextParser()
    emissions: list[object] = []
    if split is None:
        for char in text:
            emissions.extend(parser.feed(char))
    else:
        emissions.extend(parser.feed(text[:split]))
        emissions.extend(parser.feed(text[split:]))
    emissions.extend(parser.flush())
    return emissions


def _whole(text: str) -> list[object]:
    parser = MuseGlimmerTextParser()
    emissions: list[object] = list(parser.feed(text))
    emissions.extend(parser.flush())
    return emissions


def _collapse(emissions: list[object]) -> list[object]:
    """Merge adjacent text emissions so chunking does not affect equality."""
    collapsed: list[object] = []
    for emission in emissions:
        if (
            isinstance(emission, TextEmission)
            and collapsed
            and isinstance(collapsed[-1], TextEmission)
            and collapsed[-1].is_thinking == emission.is_thinking
        ):
            previous = collapsed.pop()
            assert isinstance(previous, TextEmission)
            collapsed.append(
                TextEmission(previous.text + emission.text, emission.is_thinking)
            )
        elif isinstance(emission, ToolCallEmission):
            collapsed.append(
                [(call.name, json.loads(call.arguments)) for call in emission.calls]
            )
        else:
            collapsed.append(emission)
    return collapsed


class TestRouting:
    def test_reasoning_then_answer(self) -> None:
        assert _collapse(_whole(REASONING + ANSWER)) == [
            TextEmission("The user wants 17 times 23.", True),
            TextEmission("391", False),
        ]

    def test_answer_only_channel(self) -> None:
        assert _collapse(_whole(" to=user<|message|>Hello.<|eot|>")) == [
            TextEmission("Hello.", False)
        ]

    def test_unaddressed_channel_is_content(self) -> None:
        assert _collapse(_whole("<|message|>Hello.<|eot|>")) == [
            TextEmission("Hello.", False)
        ]

    def test_tool_channel_yields_typed_arguments(self) -> None:
        collapsed = _collapse(_whole(REASONING + ONE_CALL))
        assert collapsed == [
            TextEmission("The user wants 17 times 23.", True),
            [
                (
                    "get_weather",
                    {
                        "city": "Denver",
                        "units": {"scale": "c", "precision": 1},
                        "verbose": True,
                    },
                )
            ],
        ]

    def test_parallel_calls_are_separate_channels_in_order(self) -> None:
        collapsed = _collapse(_whole(REASONING + ONE_CALL + SECOND_CALL))
        assert collapsed[1] == [("get_weather", {"city": "Denver", "units": {"scale": "c", "precision": 1}, "verbose": True})]
        assert collapsed[2] == [("get_time", {"zone": "America/Denver"})]

    def test_no_protocol_text_passes_through(self) -> None:
        # A template that skips the channel grammar must not have its answer
        # swallowed while the parser hunts for a header.
        text = "Plain prose with no markers at all, longer than a header would be."
        assert _collapse(_whole(text)) == [TextEmission(text, False)]

    def test_short_plain_text_is_released_at_flush(self) -> None:
        assert _collapse(_whole("ok")) == [TextEmission("ok", False)]

    def test_truncated_tool_body_is_delivered_as_stripped_content(self) -> None:
        truncated = (
            "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n"
            '<atem:invoke name="get_weather">\n<atem:parameter name="city">Den'
        )
        collapsed = _collapse(_whole(REASONING + truncated))
        assert collapsed[0] == TextEmission("The user wants 17 times 23.", True)
        assert len(collapsed) == 2
        tail = collapsed[1]
        assert isinstance(tail, TextEmission)
        assert not tail.is_thinking
        assert "<atem:" not in tail.text
        assert "Den" in tail.text

    def test_protocol_free_text_never_leaks_markers(self) -> None:
        collapsed = _collapse(
            _whole("Prose that skips the protocol but leaks a <|eom|> marker<|eot|>")
        )
        assert collapsed == [
            TextEmission("Prose that skips the protocol but leaks a  marker", False)
        ]

    def test_end_of_text_terminates_like_eot(self) -> None:
        assert _collapse(_whole(" to=user<|message|>Bye.<|end_of_text|>")) == [
            TextEmission("Bye.", False)
        ]

    def test_stream_ending_inside_header_emits_nothing(self) -> None:
        assert _whole(" to=sel") == []

    def test_markers_never_reach_output(self) -> None:
        for emission in _whole(REASONING + ONE_CALL + SECOND_CALL):
            if isinstance(emission, TextEmission):
                assert "<|" not in emission.text


MESSAGES = [
    REASONING + ANSWER,
    REASONING + ONE_CALL + SECOND_CALL,
    REASONING + ONE_CALL + "<|start|>assistant to=user<|message|>Done.<|eot|>",
    "Plain prose with no markers at all, longer than a header would be.",
    "Prose that skips the protocol but leaks a <|eom|> marker<|eot|>",
]


@pytest.mark.parametrize("message", MESSAGES)
def test_every_split_point_matches_the_whole_message(message: str) -> None:
    expected = _collapse(_whole(message))
    for split in range(1, len(message)):
        assert _collapse(_stream(message, split=split)) == expected, split


@pytest.mark.parametrize("message", MESSAGES)
def test_character_stream_matches_the_whole_message(message: str) -> None:
    assert _collapse(_stream(message)) == _collapse(_whole(message))
