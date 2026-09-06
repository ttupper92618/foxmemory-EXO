"""Streaming parser for Muse Glimmer's channel and ATEM wire format.

Muse Glimmer (Meta, August 2026) writes every assistant turn as a sequence of
*channels*. The generation prompt ends at ``<|start|>assistant``, so the model's
first tokens complete that header and open the body::

     to=self<|message|>...reasoning...<|eom|>
    <|start|>assistant to=user<|message|>...the answer...<|eot|>

A tool call is a channel addressed to the tool, carrying Meta's ATEM markup::

    <|start|>assistant to=get_weather<|message|><atem:function_calls>
    <atem:invoke name="get_weather">
    <atem:parameter name="city">Denver</atem:parameter>
    </atem:invoke>
    </atem:function_calls><|eom|>

Parallel calls are successive channels separated by ``<|eom|>``; ``<|eot|>``
ends the turn. The recipient decides the routing: ``self`` is reasoning,
``user`` (or no recipient) is content, and anything else is a tool call whose
body is parsed with the shared ATEM dialect reader.

The parser is pure Python and engine-agnostic: the MLX engine feeds it
detokenized deltas (``model_output_parsers.parse_muse_glimmer``), and an
in-process llama.cpp path can feed it string deltas the same way once its
binding grows the architecture. The served engines never need it: llama-server
(b10353 and later) and vLLM 0.28 parse this format natively.

Streaming discipline mirrors the other text parsers here: a marker split
across two deltas is held back rather than emitted as literal text, the
hold-back is bounded by the longest marker, and a tool body is delivered whole
because a call is only a call once its closing marker has arrived. Text that
never resolves into a channel header (a model or template that skips the
protocol) passes through as ordinary content instead of being swallowed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, final

from skulk.api.types import ToolCallItem
from skulk.worker.runner.llm_inference.scaffolding_scrub import strip_scaffolding
from skulk.worker.runner.llm_inference.tool_text_parser import atem_calls

#: Opens a channel header; the header runs to ``<|message|>``.
START: Final = "<|start|>"
#: Ends the channel header and opens the body.
MESSAGE: Final = "<|message|>"
#: End of *message*: the turn continues with another channel.
EOM: Final = "<|eom|>"
#: End of turn.
EOT: Final = "<|eot|>"
#: The tokenizer's end-of-text, a stop token that ends the turn as well.
END_OF_TEXT: Final = "<|end_of_text|>"

#: Every control marker the parser understands, longest first for matching.
CONTROL_MARKERS: Final[tuple[str, ...]] = (END_OF_TEXT, MESSAGE, START, EOM, EOT)
_BODY_TERMINATORS: Final[tuple[str, ...]] = (EOM, EOT, START, END_OF_TEXT)
_LONGEST_MARKER: Final = max(len(marker) for marker in CONTROL_MARKERS)

# A header is the tail of ``<|start|>assistant to=<recipient>`` (the prompt
# already carries the ``<|start|>assistant`` prefix, so the stream usually
# begins with ``" to=self"``). Anything that stops looking like this grammar is
# not a header, and the text is content.
_HEADER_GRAMMAR: Final = re.compile(r"^\s*(?:assistant)?(?:\s+to=(?P<recipient>\S*))?\s*$")
# Every proper prefix of a header, so a header arriving one character at a
# time (" t", " to", " to=") keeps the parser waiting instead of reading the
# fragment as content.
_HEADER_PREFIX: Final = re.compile(
    r"^\s*(?:a|as|ass|assi|assis|assist|assista|assistan|assistant)?"
    r"(?:\s+(?:t|to|to=\S*))?\s*$"
)
_HEADER_LIMIT: Final = 96
_REASONING_RECIPIENT: Final = "self"
_USER_RECIPIENT: Final = "user"


@dataclass(frozen=True, slots=True)
class TextEmission:
    """A run of visible text, flagged when it belongs to the reasoning channel."""

    text: str
    is_thinking: bool


@dataclass(frozen=True, slots=True)
class ToolCallEmission:
    """Tool calls recovered from one completed tool channel."""

    calls: list[ToolCallItem]


Emission = TextEmission | ToolCallEmission


def _marker_prefix_length(text: str) -> int:
    """Length of the trailing run of ``text`` that could still become a marker."""
    for length in range(min(_LONGEST_MARKER - 1, len(text)), 0, -1):
        tail = text[-length:]
        if any(marker.startswith(tail) for marker in CONTROL_MARKERS):
            return length
    return 0


def _earliest(text: str, markers: tuple[str, ...]) -> tuple[int, str | None]:
    """Position and identity of the earliest marker in ``text``, or ``(-1, None)``."""
    best_index = -1
    best_marker: str | None = None
    for marker in markers:
        index = text.find(marker)
        if index != -1 and (best_index == -1 or index < best_index):
            best_index, best_marker = index, marker
    return best_index, best_marker


@final
class MuseGlimmerTextParser:
    """Incrementally route Muse Glimmer channels into reasoning, content, and calls.

    Feed raw string deltas with :meth:`feed`; it returns the emissions that are
    safe to deliver now. Call :meth:`flush` once the stream ends to drain the
    tail: an unterminated tool body is parsed if its ATEM block is complete and
    delivered as marker-stripped content otherwise, and a partial marker is
    released as literal text.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        # ``None`` while reading a header (the stream starts inside the
        # ``<|start|>assistant`` header the prompt opened); otherwise the
        # current channel's recipient, ``""`` meaning none was named.
        self._recipient: str | None = None
        self._in_header: bool = True
        # Set once a body was opened without a recognizable header, so the
        # rest of the stream is ordinary content rather than header hunting.
        self._protocol_abandoned: bool = False
        self._tool_body: str = ""

    @property
    def in_tool_channel(self) -> bool:
        """Whether the parser is currently inside a tool-addressed channel."""
        return (
            not self._in_header
            and self._recipient not in (None, "", _USER_RECIPIENT, _REASONING_RECIPIENT)
        )

    def feed(self, text: str) -> list[Emission]:
        """Consume a raw delta and return the emissions ready for delivery."""
        if text:
            self._buffer += text
        emissions: list[Emission] = []
        self._drain(emissions, final=False)
        return emissions

    def flush(self) -> list[Emission]:
        """Drain everything once the stream has ended."""
        emissions: list[Emission] = []
        self._drain(emissions, final=True)
        return emissions

    # -- internals ---------------------------------------------------------

    def _drain(self, emissions: list[Emission], *, final: bool) -> None:
        while self._buffer:
            if self._in_header:
                if not self._drain_header(emissions, final=final):
                    break
                continue
            if not self._drain_body(emissions, final=final):
                break
        if final:
            self._finish(emissions)

    def _drain_header(self, emissions: list[Emission], *, final: bool) -> bool:
        """Resolve the channel header at the front of the buffer.

        Returns ``True`` when progress was made and draining should continue.
        """
        message_at = self._buffer.find(MESSAGE)
        start_at = self._buffer.find(START)
        if start_at != -1 and (message_at == -1 or start_at < message_at):
            # A fresh ``<|start|>`` restarts the header; whatever preceded it
            # was a header fragment the model abandoned, never content.
            self._buffer = self._buffer[start_at + len(START) :]
            return True
        if message_at != -1:
            header = self._buffer[:message_at]
            self._buffer = self._buffer[message_at + len(MESSAGE) :]
            match = _HEADER_GRAMMAR.match(header)
            recipient = match.group("recipient") if match is not None else None
            self._open_body(recipient or "")
            return True
        # No header terminator yet. Keep waiting only while the text still
        # looks like a header; otherwise the model skipped the protocol and the
        # buffered text is content.
        candidate = self._buffer
        held = _marker_prefix_length(candidate)
        visible = candidate[: len(candidate) - held] if held else candidate
        if _HEADER_PREFIX.match(visible) is None or len(visible) > _HEADER_LIMIT:
            self._protocol_abandoned = True
            self._open_body(_USER_RECIPIENT)
            return True
        if final:
            # Stream ended inside a header: a bare recipient name is not output.
            self._buffer = ""
        return False

    def _open_body(self, recipient: str) -> None:
        self._recipient = recipient
        self._in_header = False
        self._tool_body = ""

    def _drain_body(self, emissions: list[Emission], *, final: bool) -> bool:
        """Deliver the body at the front of the buffer up to its terminator."""
        if self._protocol_abandoned:
            # Everything is content from here on; a stray control marker is
            # stripped rather than delivered or re-read as a channel boundary.
            index, marker = _earliest(self._buffer, CONTROL_MARKERS)
            if index != -1:
                assert marker is not None
                self._deliver_body_text(self._buffer[:index], emissions)
                self._buffer = self._buffer[index + len(marker) :]
                return True
        index, marker = _earliest(
            self._buffer, () if self._protocol_abandoned else _BODY_TERMINATORS
        )
        if index == -1:
            held = 0 if final else _marker_prefix_length(self._buffer)
            emit_until = len(self._buffer) - held
            if emit_until > 0:
                self._deliver_body_text(self._buffer[:emit_until], emissions)
                self._buffer = self._buffer[emit_until:]
            return False
        assert marker is not None
        self._deliver_body_text(self._buffer[:index], emissions)
        self._buffer = self._buffer[index + len(marker) :]
        self._close_channel(emissions, complete=True)
        if marker == START:
            self._in_header = True
        elif marker == EOM:
            # The next channel follows as ``<|start|>assistant to=...``; a
            # model that omits the ``<|start|>`` still gets a header parse.
            self._in_header = True
        else:
            # ``<|eot|>`` / end of text: the turn is over. Anything after it is
            # content, which a conforming model never produces.
            self._recipient = _USER_RECIPIENT
            self._in_header = False
        return True

    def _deliver_body_text(self, text: str, emissions: list[Emission]) -> None:
        if not text:
            return
        if self.in_tool_channel:
            self._tool_body += text
            return
        emissions.append(
            TextEmission(text, is_thinking=self._recipient == _REASONING_RECIPIENT)
        )

    def _close_channel(self, emissions: list[Emission], *, complete: bool) -> None:
        if not self.in_tool_channel:
            return
        body = self._tool_body
        self._tool_body = ""
        calls = atem_calls(body)
        if calls:
            emissions.append(ToolCallEmission(calls))
            return
        # A tool channel with no parseable ATEM block (a truncated or malformed
        # call) is delivered as content with the scaffolding removed, so the
        # caller sees what the model wrote without receiving control markup.
        visible = strip_scaffolding(body).strip()
        if visible:
            emissions.append(TextEmission(visible, is_thinking=False))
        del complete

    def _finish(self, emissions: list[Emission]) -> None:
        if self._in_header:
            self._buffer = ""
            return
        if self._buffer:
            self._deliver_body_text(self._buffer, emissions)
            self._buffer = ""
        self._close_channel(emissions, complete=False)
