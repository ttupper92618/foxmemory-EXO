"""Invariant sweep over the no-tools scaffolding scrub (#889).

The scrub sits on streamed text, so the bugs it can have are split bugs: a
marker divided across two chunks that a per-chunk replace would miss, or a
held-back partial marker that never gets released. The invariants, checked
across every single split point of each message plus character-by-character:

1. No scaffolding marker survives to the output, wherever the split fell.
2. Everything that is not a marker is preserved exactly: the streamed result
   equals the blocking ``strip_scaffolding`` of the whole message.
3. A message with no markers passes through byte-identical.
4. ``flush`` releases a trailing partial marker as ordinary text, so a stream
   that ends mid-lookalike does not truncate the answer.
"""

from __future__ import annotations

from skulk.worker.runner.llm_inference.scaffolding_scrub import (
    SCAFFOLDING_MARKERS,
    StreamingScaffoldingScrub,
    strip_scaffolding,
)

GEMMA_LEAK = '<|tool_call>_call:get_weather{location: "Denver"}<tool_call|>'
ATEM_LEAK = (
    '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
    '<atem:parameter name="city">Denver</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)

MESSAGES: list[str] = [
    "The weather is fine.",
    GEMMA_LEAK,
    f"I'll check. {GEMMA_LEAK} Done.",
    ATEM_LEAK,
    f"Checking. {ATEM_LEAK} Done.",
    '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>',
    "<|python_tag|>print('hello')",
    '[TOOL_CALLS] [{"name": "get_weather"}]',
    "<｜tool▁calls▁begin｜>x<｜tool▁calls▁end｜>",
    (
        '<｜DSML｜function_calls><｜DSML｜invoke name="get_weather">'
        '<｜DSML｜parameter name="location">Denver</｜DSML｜parameter>'
        "</｜DSML｜invoke></｜DSML｜function_calls>"
    ),
    "Braces {like this} and <angles> are fine.",
    "A lone < at the end",
    "Ends with a partial marker <tool_ca",
    "<tool_call><tool_call>doubled</tool_call></tool_call>",
]


def splits(message: str) -> list[list[str]]:
    """Every single split point, plus whole and character-by-character."""

    variants: list[list[str]] = [[message]]
    variants.extend(
        [message[:index], message[index:]] for index in range(1, len(message))
    )
    variants.append(list(message))
    return variants


def run_stream(pieces: list[str]) -> str:
    scrub = StreamingScaffoldingScrub()
    out = [scrub.feed(piece) for piece in pieces]
    out.append(scrub.flush())
    return "".join(out)


class TestStreamingScrubInvariants:
    def test_every_split_of_every_message(self) -> None:
        for message in MESSAGES:
            expected = strip_scaffolding(message)
            for pieces in splits(message):
                result = run_stream(pieces)
                where = f"{message!r} split as {pieces!r}"
                assert result == expected, f"stream differs from blocking for {where}"
                for marker in SCAFFOLDING_MARKERS:
                    assert marker not in result, f"{marker!r} leaked for {where}"

    def test_marker_free_text_is_identity(self) -> None:
        for message in MESSAGES:
            if any(marker in message for marker in SCAFFOLDING_MARKERS):
                continue
            for pieces in splits(message):
                assert run_stream(pieces) == message

    def test_trailing_partial_marker_is_released_on_flush(self) -> None:
        scrub = StreamingScaffoldingScrub()
        emitted = scrub.feed("answer <tool_ca")
        assert emitted == "answer "
        assert scrub.flush() == "<tool_ca"


class TestBlockingStrip:
    def test_strips_the_observed_live_leak(self) -> None:
        assert (
            strip_scaffolding(GEMMA_LEAK) == '_call:get_weather{location: "Denver"}'
        )

    def test_every_marker_is_stripped(self) -> None:
        for marker in SCAFFOLDING_MARKERS:
            assert strip_scaffolding(f"a{marker}b") == "ab"

    def test_dsml_namespace_never_reaches_the_caller(self) -> None:
        """DSML control tokens are stripped; attribute debris may remain."""
        leak = (
            '<｜DSML｜function_calls><｜DSML｜invoke name="get_weather">'
            "</｜DSML｜invoke></｜DSML｜function_calls>"
        )
        result = strip_scaffolding(leak)
        assert "｜DSML｜" not in result
        assert result == 'invoke name="get_weather">'
