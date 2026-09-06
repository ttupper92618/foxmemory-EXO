"""Strip tool-dialect scaffolding from no-tools response streams (#889).

The invariant, shared with the in-process MLX parser: a request that offered
no tools never surfaces tool-call control markup as assistant content, however
the stream was chunked. The MLX path enforces this inside ``parse_tool_calls``
(the scan always runs and recognized blocks are delivered as marker-stripped
content). The served engines have no equivalent seam: with no tools in the
request the server's own parser does not run, so a model that writes a call
anyway leaks its dialect markers to the caller verbatim. Observed live on
`llama_server` with a gemma4 card and ``tool_choice: "none"``:
``<|tool_call>_call:get_weather{...}<tool_call|>`` arrived as content.

This module is that seam. It deliberately strips the marker vocabulary rather
than recognizing whole blocks: on a no-tools request nothing may be executed,
so the body of whatever the model wrote is ordinary content either way, and
marker-stripping is the exact transformation the MLX path's
``_block_as_content`` applies to a recognized block. Stripping unconditionally
also covers a malformed or truncated block, which block recognition would
hand back verbatim.

Used by the served runners (`llama_server`, `vllm`) and the in-process
`llama_cpp` runner on their no-tools paths only. A request that offered tools
keeps the engine's own behavior: there the server-parsed calls are the
product, and a block the server did not parse is evidence the caller may need
to see.
"""

from __future__ import annotations

from typing import Final, final

# Every dialect's block scaffolding, as it leaks into plain text. Grouped by
# the family that writes it; each entry either leaked from a live model during
# tool validation or is the write-side marker of a dialect Skulk parses
# (`tool_text_parser`). Sorted longest-first at strip time so no marker can
# shadow a longer one that contains it.
SCAFFOLDING_MARKERS: Final[tuple[str, ...]] = (
    # Generic marker dialect (Qwen3 XML / Hermes JSON carriers).
    "<tool_call>",
    "</tool_call>",
    # Llama 3.1+ built-in / tool handoff.
    "<|python_tag|>",
    # Mistral call arrays.
    "[TOOL_CALLS]",
    # Gemma 4 (`call:NAME{...}` blocks); response markers included because the
    # family writes both sides with the same scaffolding.
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    # Muse Glimmer ATEM markup (`<atem:function_calls>` blocks). The invoke
    # and parameter OPENING tags carry a name attribute, so like DSML the
    # namespace prefix is stripped on its own and the attribute debris is the
    # accepted trade for a bounded fixed-string vocabulary.
    "<atem:function_calls>",
    "</atem:function_calls>",
    "</atem:invoke>",
    "</atem:parameter>",
    "<atem:",
    # DeepSeek V3/R1 block containers (the pre-DSML wire format).
    "<｜tool▁calls▁begin｜>",
    "<｜tool▁calls▁end｜>",
    "<｜tool▁call▁begin｜>",
    "<｜tool▁call▁end｜>",
    "<｜tool▁sep｜>",
    # DeepSeek V3.2 DSML scaffolding (`dsml_encoding.py`). The containers and
    # closing tags are fixed strings; the invoke/parameter OPENING tags carry
    # attributes (`<｜DSML｜invoke name="...">`), which a fixed-string marker
    # cannot cover, so the namespace prefixes are stripped on their own. That
    # leaves attribute debris (`invoke name="...">`) as plain text, which is
    # the accepted trade: the control namespace never reaches the caller, and
    # a bounded fixed-string vocabulary is what keeps the streaming hold-back
    # bounded.
    "<｜DSML｜function_calls>",
    "</｜DSML｜function_calls>",
    "</｜DSML｜invoke>",
    "</｜DSML｜parameter>",
    "<｜DSML｜",
    "</｜DSML｜",
)

_MARKERS_LONGEST_FIRST: Final = tuple(
    sorted(SCAFFOLDING_MARKERS, key=len, reverse=True)
)
_LONGEST_MARKER: Final = max(len(marker) for marker in SCAFFOLDING_MARKERS)


def strip_scaffolding(text: str) -> str:
    """Remove every complete scaffolding marker from ``text``.

    For blocking (non-streamed) response paths, where the whole message is in
    hand. Streaming paths must use :class:`StreamingScaffoldingScrub` instead,
    because a marker split across two stream chunks is invisible to a
    per-chunk replace.
    """

    for marker in _MARKERS_LONGEST_FIRST:
        text = text.replace(marker, "")
    return text


def _partial_marker_suffix_length(text: str) -> int:
    """Length of the trailing run of ``text`` that could still become a marker.

    The same hold-back rule as the MLX streaming parser: bounded by the
    longest marker, so at most a few characters of latency and never an
    unbounded buffer.
    """

    for length in range(min(_LONGEST_MARKER - 1, len(text)), 0, -1):
        tail = text[-length:]
        if any(marker.startswith(tail) for marker in SCAFFOLDING_MARKERS):
            return length
    return 0


@final
class StreamingScaffoldingScrub:
    """Marker-strip a text stream, holding back partial markers across chunks.

    ``feed`` returns the text safe to emit now; ``flush`` returns whatever was
    held once the stream ends (a trailing partial marker is then just text).
    The interface mirrors ``GemmaChannelTextParser`` so the served runners
    compose both the same way.
    """

    def __init__(self) -> None:
        self._held = ""

    def feed(self, text: str) -> str:
        """Scrub ``text`` (plus any held tail) and return the emittable part.

        The hold-back is computed on the RAW combined text, before stripping:
        some markers are strict prefixes of longer ones (the bare DSML
        namespace inside its container tags), and stripping a completed short
        marker eagerly would make the result depend on where the stream was
        split, because the longer marker could never match across the
        boundary its prefix was consumed at.
        """

        combined = self._held + text
        keep = _partial_marker_suffix_length(combined)
        self._held = combined[len(combined) - keep :] if keep else ""
        emittable = combined[: len(combined) - keep] if keep else combined
        return strip_scaffolding(emittable)

    def flush(self) -> str:
        """Return the held tail at end of stream; nothing more is coming.

        Stripped once more on the way out: the tail may be a complete short
        marker held because a longer marker extends it. A genuine partial
        lookalike survives the strip and is returned as ordinary text.
        """

        held = self._held
        self._held = ""
        return strip_scaffolding(held)
