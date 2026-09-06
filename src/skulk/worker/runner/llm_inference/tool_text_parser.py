# pyright: reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""MLX-free recovery of a reasoning model's tool call from llama.cpp output.

``llama_cpp``'s ``create_chat_completion(tools=...)`` only populates structured
``tool_calls`` for models whose native tool-call format its bundled chat handlers
recognize. A reasoning model emits its tool call in its own format that
llama-cpp-python leaves unparsed, so the call falls through into the message
``content`` as raw text instead of a structured ``tool_calls`` (#416). The three
formats seen on the llama.cpp engine:

- **gpt-oss harmony**: a ``commentary`` channel whose header carries
  ``to=functions.NAME`` and whose ``<|message|>`` body is the JSON arguments,
  e.g. ``...<|channel|>commentary to=functions.get_weather <|constrain|>json``
  ``<|message|>{"city":"Paris"}``.
- **Qwen3 XML**: ``<tool_call><function=NAME><parameter=KEY>VALUE</parameter>``
  ``...</function></tool_call>``.
- **Hermes / older Qwen JSON**: ``<tool_call>{"name":..,"arguments":{..}}``
  ``</tool_call>``.

This module reparses those from the string so the runner can emit a proper
``ToolCallChunk``, mirroring what the MLX engine does at the token level. It is
pure-Python (no MLX) because it runs on non-Mac GPU nodes (e.g. AMD).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, cast

from skulk.api.types import ToolCallItem

if TYPE_CHECKING:
    from skulk.shared.models.model_cards import ToolCallFormat
from skulk.worker.runner.bootstrap import logger
from skulk.worker.runner.llm_inference.tool_parsers import (
    coerce_tool_calls_to_schema,
    declared_tool_calls,
)

# gpt-oss harmony tool call: the recipient `to=functions.NAME` and a `commentary`
# channel together, then a `<|message|>` body holding the JSON arguments (up to
# the next control marker or end of text). gpt-oss emits the recipient in EITHER
# order relative to the channel marker, both documented by the repo's
# FORMAT_A/FORMAT_B fixtures, so match both:
#   B (channel first):   <|channel|>commentary ... to=functions.NAME ... <|message|>
#   A (recipient first): to=functions.NAME<|channel|>commentary ... <|message|>
# Both tie the recipient to the commentary channel header (before `<|message|>`),
# so a bare `to=functions.` written as prose in the analysis (reasoning) channel
# body is NOT matched. `_HARMONY_MESSAGE_TAIL` is the shared args + terminator.
_HARMONY_MESSAGE_TAIL = (
    r"(.*?)(?=<\|call\|>|<\|end\|>|<\|return\|>|<\|start\|>|<\|channel\|>|$)"
)
_HARMONY_CALL_RES = (
    re.compile(
        r"<\|channel\|>commentary[^<]*?to=functions\.([A-Za-z0-9_.\-]+)"
        r".*?<\|message\|>" + _HARMONY_MESSAGE_TAIL,
        re.DOTALL,
    ),
    re.compile(
        r"to=functions\.([A-Za-z0-9_.\-]+)\s*<\|channel\|>commentary"
        r".*?<\|message\|>" + _HARMONY_MESSAGE_TAIL,
        re.DOTALL,
    ),
)
# A `<tool_call>...</tool_call>` block (JSON or Qwen3 XML inside), embedded in
# prose/reasoning. There may be several.
_TOOLCALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>", re.DOTALL
)
# Llama 3.1+ marks a tool call with <|python_tag|> and ends the turn with
# <|eom_id|> (end of MESSAGE, handing off to a tool) rather than <|eot_id|>
# (end of TURN, handing back to the user). The body is one or more JSON
# objects using "parameters" rather than "arguments"; several calls are
# separated by ";".
_PYTHON_TAG_RE = re.compile(
    r"<\|python_tag\|>(.*?)(?=<\|eom_id\|>|<\|eot_id\|>|<\|start_header_id\|>|$)",
    re.DOTALL,
)
# Mistral emits a JSON array behind a [TOOL_CALLS] marker.
_MISTRAL_RE = re.compile(r"\[TOOL_CALLS\]\s*(\[.*)", re.DOTALL)
# GLM puts the function name on its own line inside <tool_call>, then names
# arguments in <arg_key>/<arg_value> pairs rather than as JSON.
_GLM_ARG_RE = re.compile(
    r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
    re.DOTALL,
)


def _first_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first balanced ``{...}`` JSON object in ``text``, or None."""
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001 - fall through to a bracket scan
        pass
    start = stripped.find("{")
    if start == -1:
        return None
    # Brace scan to find the end of the first object. Track string state so a
    # brace inside a string value (e.g. {"pattern": "{a}"}) does not throw off
    # the depth count.
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(stripped[start : index + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:  # noqa: BLE001 - malformed JSON, give up
                    return None
    return None


def _call_from_json_object(obj: object) -> ToolCallItem | None:
    """Build a call from the ``{"name": ..., "arguments"/"parameters": ...}`` shape.

    Shared by every JSON-carrying dialect (Hermes, Llama, Mistral), which
    differ only in the markup around this object. Llama uses ``parameters``
    where Hermes uses ``arguments``; both are accepted.

    ``ToolCallItem.arguments`` must decode to a JSON object downstream (schema
    coercion, the Claude adapter's dict input). A dict is re-serialized; the
    OpenAI shape where ``arguments`` is already a JSON-encoded string is kept
    as-is when it decodes to an object; any other shape (list, scalar, or a
    string that is not a JSON object) is malformed and falls back to ``{}``
    rather than being invented.
    """

    if not isinstance(obj, dict):
        return None
    payload = cast("dict[str, Any]", obj)
    if not isinstance(payload.get("name"), str):
        return None
    args = payload.get("arguments", payload.get("parameters", {}))
    if isinstance(args, dict):
        args_str = json.dumps(args)
    elif isinstance(args, str) and _first_json_object(args) is not None:
        args_str = args
    else:
        args_str = "{}"
    return ToolCallItem(name=str(payload["name"]), arguments=args_str)


def _successive_json_objects(text: str) -> list[dict[str, Any]]:
    """Every top-level balanced JSON object in ``text``, in order.

    Llama chains several ``<|python_tag|>`` calls in one body separated by
    ``;``, but a semicolon is also ordinary data inside a quoted argument
    (shell commands, SQL, prose), so splitting on the separator divided the
    JSON string itself and lost the call. The objects are instead read as
    successive balanced spans, string-aware, and whatever separates them is
    skipped rather than interpreted. A balanced span that is not valid JSON
    is skipped whole rather than rescanned from inside, so its interior
    braces cannot mint an object; an unterminated span is truncated
    generation, and nothing after it can be complete either.
    """

    objects: list[dict[str, Any]] = []
    position = 0
    while True:
        start = text.find("{", position)
        if start == -1:
            return objects
        depth = 0
        in_string = False
        escaped = False
        end = -1
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end == -1:
            return objects
        try:
            decoded = json.loads(text[start : end + 1])
        except ValueError:
            decoded = None
        if isinstance(decoded, dict):
            objects.append(cast("dict[str, Any]", decoded))
        position = end + 1


def _python_tag_calls(text: str) -> list[ToolCallItem]:
    """Parse Llama 3.1+ ``<|python_tag|>`` calls."""

    calls: list[ToolCallItem] = []
    for match in _PYTHON_TAG_RE.finditer(text):
        for obj in _successive_json_objects(match.group(1)):
            call = _call_from_json_object(obj)
            if call is not None:
                calls.append(call)
    return calls


def _mistral_calls(text: str) -> tuple[list[ToolCallItem], str]:
    """Parse Mistral ``[TOOL_CALLS] [...]`` arrays.

    Returns the calls plus the visible text around the array: the model may
    write a preamble before the marker and keep writing after the array
    (``[TOOL_CALLS] [...] I will check that.``), and both are the assistant's
    content rather than markup. The remainder is empty when nothing parsed,
    so a message that is not a call is left whole for the caller.
    """

    match = _MISTRAL_RE.search(text)
    if match is None:
        return [], ""
    decoder = json.JSONDecoder()
    body = match.group(1).strip()
    try:
        array, consumed = decoder.raw_decode(body)
    except ValueError:
        return [], ""
    if not isinstance(array, list):
        return [], ""
    calls: list[ToolCallItem] = []
    for entry in array:
        call = _call_from_json_object(entry)
        if call is not None:
            calls.append(call)
    if not calls:
        return [], ""
    remainder = (text[: match.start()] + " " + body[consumed:]).strip()
    return calls, remainder


def _bare_json_call(text: str) -> tuple[list[ToolCallItem], str]:
    """Parse an unmarked call that opens the message.

    Llama omits ``<|python_tag|>`` in some templates and simply emits the call
    object. The message must *begin* with that object, so prose containing JSON
    is never read as a call, but the model may keep writing after it: a call
    followed by a closing remark is still a call, and the remark is returned
    as the remainder so it reaches the caller as content rather than being
    swallowed with the markup.

    Two things keep this from mistaking a JSON answer for a call. The object
    must carry a ``name`` alongside an ``arguments`` or ``parameters`` value,
    which an answer rarely has; and the caller's tools are checked afterwards,
    so an object naming nothing the caller offered is dropped and delivered as
    content.
    """

    stripped = text.strip()
    if not stripped.startswith("{"):
        return [], ""
    decoder = json.JSONDecoder()
    try:
        obj, consumed = decoder.raw_decode(stripped)
    except ValueError:
        return [], ""
    if not isinstance(obj, dict):
        return [], ""
    payload = cast("dict[str, Any]", obj)
    if "name" not in payload:
        return [], ""
    if not isinstance(payload.get("arguments", payload.get("parameters")), (dict, str)):
        return [], ""
    call = _call_from_json_object(payload)
    if call is None:
        return [], ""
    return [call], stripped[consumed:].strip()


def _harmony_tool_calls(text: str) -> list[ToolCallItem]:
    calls: list[ToolCallItem] = []
    seen: set[tuple[int, str]] = set()
    for pattern in _HARMONY_CALL_RES:
        for match in pattern.finditer(text):
            # A given call matches only one ordering, but guard against a region
            # being claimed twice by keying on its start offset + name.
            key = (match.start(), match.group(1))
            if key in seen:
                continue
            seen.add(key)
            name = match.group(1)
            body = match.group(2)
            obj = _first_json_object(body)
            if obj is not None:
                calls.append(ToolCallItem(name=name, arguments=json.dumps(obj)))
            elif not body.strip():
                # A genuine no-argument call (empty body) is valid; only then is
                # {} correct. A non-empty body that did not parse is a
                # truncated/garbled call, so skip it rather than fabricate args.
                calls.append(ToolCallItem(name=name, arguments="{}"))
    return calls


def _toolcall_block_calls(text: str) -> list[ToolCallItem]:
    calls: list[ToolCallItem] = []
    for block in _TOOLCALL_BLOCK_RE.finditer(text):
        inner = block.group(1).strip()
        # Disambiguate by a real Qwen3 XML function tag FIRST. Matching the full
        # <function=NAME>...</function> (not just the substring "<function=")
        # avoids two failure modes: a JSON-scan first would misread an
        # object-valued XML parameter containing a "name" field as the Hermes
        # form, and a substring check would misclassify a Hermes JSON call whose
        # argument value merely contains the literal "<function=" as XML.
        xml_functions = list(_FUNCTION_RE.finditer(inner))
        if xml_functions:
            for function in xml_functions:
                name = function.group(1)
                params = {
                    key: value.strip()
                    for key, value in _PARAMETER_RE.findall(function.group(2))
                }
                calls.append(ToolCallItem(name=name, arguments=json.dumps(params)))
            continue
        # Hermes / older Qwen JSON form: {"name": ..., "arguments": {...}}.
        # GLM names arguments in <arg_key>/<arg_value> pairs with the function
        # name on the first line, so there is no JSON object to find. Checked
        # before the JSON scan because a value may itself contain JSON.
        arg_pairs = _GLM_ARG_RE.findall(inner)
        if arg_pairs:
            name = inner.split("<arg_key>", 1)[0].strip().splitlines()
            if name and name[-1].strip():
                params = {key: value for key, value in arg_pairs}
                calls.append(
                    ToolCallItem(
                        name=name[-1].strip(), arguments=json.dumps(params)
                    )
                )
                continue
        call = _call_from_json_object(_first_json_object(inner))
        if call is not None:
            calls.append(call)
    return calls


def _gemma_blocks(text: str) -> list[str]:
    """Extract complete Gemma 4 marker-delimited blocks, quote-aware.

    A quoted argument may contain the closing marker itself (a tool result or
    user text echoed into a string), so a regex split would end the outer
    block at the quoted closer and mint the quoted remainder as a separate
    executable call. The scan therefore skips ``<|"|>`` spans while looking
    for the closer, exactly as the argument-body scan does. A block without a
    real closer is truncated generation and yields nothing. The recovery
    dispatch parses only these blocks; the MLX marker mechanism bounds blocks
    by tokenizer markers before its family parser sees the text.
    """
    blocks: list[str] = []
    position = 0
    while True:
        opener = text.find("<|tool_call>", position)
        if opener == -1:
            return blocks
        index = opener + len("<|tool_call>")
        closer = -1
        while index < len(text):
            if text.startswith('<|"|>', index):
                closing_quote = text.find('<|"|>', index + 5)
                if closing_quote == -1:
                    return blocks
                index = closing_quote + 5
                continue
            if text.startswith("<tool_call|>", index):
                closer = index
                break
            index += 1
        if closer == -1:
            return blocks
        blocks.append(text[opener + len("<|tool_call>") : closer])
        position = closer + len("<tool_call|>")


def gemma4_calls(text: str) -> list[ToolCallItem]:
    """Parse Gemma 4 ``call:NAME{...}`` blocks (the ``<|tool_call>`` dialect).

    Gemma 4 emits ``call:FUNCTION{key1:value1,key2:<|"|>string<|"|>}`` where
    ``<|"|>`` delimits string values; bare values keep their JSON type. Uses
    the three-phase approach from ollama PR #15306: extract quoted strings
    into placeholders, quote bare keys, then restore the strings through
    ``json.dumps`` for correct escaping. Shared by the MLX family parser and
    this module's text recovery, so both engines read the same dialect.
    """
    _call_start_re = re.compile(r"call:([\w.-]+)\{")
    _gemma_quote_re = re.compile(r'(?s)<\|"\|>(.*?)<\|"\|>')
    _bare_key_re = re.compile(r"([,{])\s*([\w.-]+):")

    def _balanced_args(start: int) -> tuple[str, int] | None:
        """Return the argument body from ``start`` (past the opening brace).

        Walks the text tracking brace depth so nested objects survive, and
        skips ``<|"|>``-delimited string spans entirely so a brace inside a
        quoted value cannot close the call. Returns ``None`` for an
        unterminated body (truncated generation), which is not a call.
        """
        depth = 1
        index = start
        while index < len(text):
            if text.startswith('<|"|>', index):
                closing = text.find('<|"|>', index + 5)
                if closing == -1:
                    return None
                index = closing + 5
                continue
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index], index
            index += 1
        return None

    def _args_to_json(raw_args: str) -> str:
        extracted: list[str] = []

        def _replace_quoted(match: "re.Match[str]") -> str:
            extracted.append(match.group(1))
            return f"\x00{len(extracted) - 1}\x00"

        skeleton = _gemma_quote_re.sub(_replace_quoted, raw_args)
        skeleton = "{" + skeleton
        skeleton = _bare_key_re.sub(r'\1"\2":', skeleton)
        skeleton = skeleton[1:]
        for index, value in enumerate(extracted):
            skeleton = skeleton.replace(f"\x00{index}\x00", json.dumps(value))
        return skeleton

    calls: list[ToolCallItem] = []
    position = 0
    while True:
        match = _call_start_re.search(text, position)
        if match is None:
            break
        # The opener search is quote-aware too: a quoted span at the block's
        # top level (before any real call) may carry call-shaped content,
        # and matching inside it would mint that content as executable.
        quote = text.find('<|"|>', position)
        if quote != -1 and quote < match.start():
            closing_quote = text.find('<|"|>', quote + 5)
            if closing_quote == -1:
                break
            position = closing_quote + 5
            continue
        balanced = _balanced_args(match.end())
        if balanced is None:
            # Unterminated body (truncated generation): nothing after it can
            # be a complete call either.
            break
        raw_args, body_end = balanced
        # Resume AFTER the consumed body: call-shaped text inside a quoted
        # argument (a tool result or user text echoed into a string) must
        # never be recovered as a second executable call.
        position = body_end + 1
        args_json = "{" + _args_to_json(raw_args) + "}"
        try:
            json.loads(args_json)
        except json.JSONDecodeError:
            # Drop the call rather than fabricating empty arguments: a
            # side-effecting offered tool invoked with silently discarded
            # required arguments is worse than no call, and the harmony
            # branch already treats a malformed non-empty body this way.
            logger.warning(
                "Dropping unparseable Gemma 4 tool call "
                f"(argument_chars={len(args_json)})"
            )
            continue
        calls.append(ToolCallItem(name=match.group(1), arguments=args_json))
    return calls


_ATEM_BLOCK = re.compile(
    r"<atem:function_calls>(?P<body>.*?)</atem:function_calls>", re.DOTALL
)
_ATEM_INVOKE = re.compile(
    r'<atem:invoke\s+name="(?P<name>[^"]+)"\s*>(?P<body>.*?)</atem:invoke>',
    re.DOTALL,
)
_ATEM_PARAMETER = re.compile(
    r'<atem:parameter\s+name="(?P<name>[^"]+)"\s*>(?P<value>.*?)</atem:parameter>',
    re.DOTALL,
)


def _atem_value(raw: str) -> Any:
    """Type one ATEM parameter value the way the template wrote it.

    The template renders lists and objects as JSON, booleans as ``true`` /
    ``false``, ``None`` as ``null``, and everything else verbatim ("spaces for
    string values are not stripped"). Only those JSON shapes are decoded here;
    numbers stay strings so a string-typed parameter that happens to hold
    digits is not retyped, and the shared schema coercion turns them into
    numbers where the tool's schema says so.
    """
    stripped = raw.strip()
    if stripped.startswith(("{", "[")):
        try:
            decoded = cast("object", json.loads(stripped))
        except ValueError:
            return raw
        return decoded if isinstance(decoded, (dict, list)) else raw
    if stripped == "true":
        return True
    if stripped == "false":
        return False
    if stripped == "null":
        return None
    return raw


def atem_calls(text: str) -> list[ToolCallItem]:
    """Parse Muse Glimmer's ATEM tool-call markup out of ``text``.

    Reads every complete ``<atem:function_calls>`` block (Meta's protocol,
    parsed with regular expressions by design: the template tells the model
    the output "is not expected to be valid XML"), each ``<atem:invoke
    name="...">`` inside it, and its ``<atem:parameter name="...">`` values.
    Blocks without a closing marker are not calls: the message is still being
    written, or was cut short, and a truncated call must not be executed.
    Parallel calls in one block, or across blocks, are returned in order.
    """
    calls: list[ToolCallItem] = []
    for block in _ATEM_BLOCK.finditer(text):
        for invoke in _ATEM_INVOKE.finditer(block.group("body")):
            arguments: dict[str, Any] = {}
            for parameter in _ATEM_PARAMETER.finditer(invoke.group("body")):
                arguments[parameter.group("name")] = _atem_value(
                    parameter.group("value")
                )
            calls.append(
                ToolCallItem(
                    name=invoke.group("name"),
                    arguments=json.dumps(arguments),
                )
            )
    return calls


def parse_tool_calls_from_text(
    text: str,
    tools: list[dict[str, Any]] | None = None,
    tool_call_format: "ToolCallFormat | None" = None,
) -> list[ToolCallItem] | None:
    """Recover tool calls a reasoning model emitted as text (llama.cpp engine).

    The whole-block form of :func:`parse_tool_calls_with_remainder`: same
    dispatch and same result, with any trailing visible text discarded.
    Callers that can deliver content alongside calls should use the
    remainder-preserving variant instead.
    """

    calls, _ = parse_tool_calls_with_remainder(
        text, tools, tool_call_format=tool_call_format
    )
    return calls


def parse_tool_calls_with_remainder(
    text: str,
    tools: list[dict[str, Any]] | None = None,
    tool_call_format: "ToolCallFormat | None" = None,
) -> tuple[list[ToolCallItem] | None, str]:
    """Recover tool calls a reasoning model emitted as text (llama.cpp engine).

    Detects the dialect from the markers present and parses the calls, then
    coerces argument types to the tool schema. Recognized dialects:

    - harmony ``to=functions.`` channels (gpt-oss)
    - ``<tool_call>`` blocks carrying Hermes JSON, Qwen3 XML, or GLM
      ``<arg_key>``/``<arg_value>`` pairs
    - Gemma 4 ``<|tool_call>call:NAME{...}<tool_call|>`` blocks
    - Llama ``<|python_tag|>`` calls, which use ``parameters`` rather than
      ``arguments`` and may chain several with ``;``
    - Mistral ``[TOOL_CALLS]`` arrays
    - Muse Glimmer ATEM ``<atem:function_calls>`` blocks
    - an unmarked call object opening the message, which the model may keep
      writing after

    When ``tools`` is given, calls naming a tool the caller did not offer are
    dropped, because a model reaching for one of its own built-ins has not
    called anything the caller can run.

    Returns the calls plus the visible text around the call markup for the
    dialects that know where their markup ends (the unmarked object and the
    Mistral array); other dialects report no remainder. The remainder is
    meaningful only when calls were returned: with no call (or every call
    dropped by the offered-tools rule) the result is ``(None, "")`` and the
    caller falls back to emitting the whole content, exactly as before.
    """
    if not text:
        return None, ""
    # When the caller knows the model's resolved format, dialect selection is
    # card truth rather than text inference: a Gemma-format model parses only
    # its own dialect, and any OTHER format can never have Gemma or harmony
    # shapes minted from echoed prose (a foreign-dialect block quoted in a
    # preamble is content, not a call). Text inference remains only within
    # the genuinely ambiguous Generic family, whose templates legitimately
    # vary, and for callers with no profile. This dispatch runs BEFORE the
    # leading-object branch below, so a specialized-format model echoing a
    # bare JSON object cannot have the unmarked dialect minted against card
    # truth; the leading-object branch itself only serves formats that
    # actually speak the unmarked dialect.
    if tool_call_format is not None:
        from skulk.shared.models.model_cards import ToolCallFormat as _Format

        if tool_call_format == _Format.Gemma4:
            gemma_calls: list[ToolCallItem] = []
            for block in _gemma_blocks(text):
                gemma_calls.extend(gemma4_calls(block))
            return _finish(gemma_calls, tools), ""
        if tool_call_format == _Format.GptOss:
            return _finish(_harmony_tool_calls(text), tools), ""
        if tool_call_format == _Format.Atem:
            return _finish(atem_calls(text), tools), ""
        if tool_call_format != _Format.Generic:
            # A specialized format with no text dialect here (DSML parses at
            # the token level elsewhere) gets NO text inference at all:
            # foreign markers or a bare object in its prose are content.
            return None, ""

    # A message that OPENS with a valid JSON object is the unmarked dialect,
    # selected first and exclusively: the outermost structure is JSON, so a
    # dialect marker inside one of its string values (a tool result or user
    # text echoed into an argument) is content, and letting the marker scan
    # below run on such a message would mint an executable call from that
    # string. A leading brace that does not decode as an object is just
    # prose and falls through to marker selection.
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            json.JSONDecoder().raw_decode(stripped)
        except ValueError:
            pass
        else:
            leading_calls, leading_remainder = _bare_json_call(text)
            return _finish_with_remainder(leading_calls, leading_remainder, tools)
    # The dialect is SELECTED by the earliest recognized marker in the text
    # and parsed exclusively. Marker presence anywhere is not enough: a
    # quoted argument may legitimately carry text shaped like ANY other
    # dialect (a tool result or user text echoed into a string), in either
    # direction, and letting a later branch rescan the same message would
    # mint an executable call from that quoted content. The outermost
    # structure decides; each dialect's own quote and JSON handling keeps
    # interior marker-shaped text as string content. A selected dialect that
    # parses nothing (prose mentioning a marker, a truncated block) yields
    # no call rather than a fallback scan, for the same reason.
    markers: list[tuple[int, str]] = []
    # Reaching the marker scan with a non-null format means Generic: the
    # specialized formats returned above. Generic families never write the
    # gemma or harmony shapes, so those cannot be minted from echoed prose;
    # distinguishing WITHIN the Generic family (a Hermes model echoing a
    # Mistral array) needs template truth this seam does not have and is
    # tracked as a follow-up.
    excluded_kinds = (
        {"gemma4", "harmony", "atem"} if tool_call_format is not None else set()
    )
    for marker, kind in (
        # Harmony is selected by its outer channel carrier, not only by the
        # commentary header: a gpt-oss response begins with <|channel|>
        # (analysis first), and to=functions. appears later, so a
        # contemplated block inside analysis would otherwise sit earlier in
        # the text and win the selection for a different dialect.
        ("<|channel|>", "harmony"),
        ("to=functions.", "harmony"),
        ("<|tool_call>", "gemma4"),
        ("<atem:function_calls>", "atem"),
        ("<tool_call>", "generic"),
        ("<|python_tag|>", "python_tag"),
        ("[TOOL_CALLS]", "mistral"),
    ):
        if kind in excluded_kinds:
            continue
        position = text.find(marker)
        if position != -1:
            markers.append((position, kind))
    calls: list[ToolCallItem] = []
    if markers:
        # Uniformly exclusive: the selected dialect's result is final, and
        # the unmarked-object fallback below never runs for a marker-bearing
        # message, so no dialect's quoted or contemplated interior can be
        # rescanned by anything.
        _, dialect = min(markers)
        if dialect == "harmony":
            calls = _harmony_tool_calls(text)
        elif dialect == "gemma4":
            for block in _gemma_blocks(text):
                calls.extend(gemma4_calls(block))
        elif dialect == "atem":
            calls = atem_calls(text)
        elif dialect == "generic":
            calls = _toolcall_block_calls(text)
        elif dialect == "python_tag":
            calls = _python_tag_calls(text)
        else:
            mistral_calls, mistral_remainder = _mistral_calls(text)
            return _finish_with_remainder(mistral_calls, mistral_remainder, tools)
        return _finish(calls, tools), ""
    # Last resort, and deliberately narrow: the message must begin with the
    # call object. Unmarked dialects are otherwise indistinguishable from a
    # model answering in JSON, so anything looser invents tool calls from
    # prose. The object must also carry a name alongside arguments, and the
    # caller's tools are checked afterwards.
    bare_calls, bare_remainder = _bare_json_call(text)
    return _finish_with_remainder(bare_calls, bare_remainder, tools)


def _finish(
    calls: list[ToolCallItem], tools: list[dict[str, Any]] | None
) -> list[ToolCallItem] | None:
    """Apply the shared offered-tools filter and schema coercion to a result.

    A model may reach for one of its own built-ins: Llama answers some plain
    questions with a call to ``print``, and gpt-oss has ``python`` and
    ``browser``. Those name nothing the caller can run, so a block left with
    no offered tool reads as prose and the caller gets the text instead.
    """
    if not calls:
        return None
    if tools is not None:
        calls = declared_tool_calls(calls, tools)
        if not calls:
            return None
        calls = coerce_tool_calls_to_schema(calls, tools)
    return calls


def _finish_with_remainder(
    calls: list[ToolCallItem],
    remainder: str,
    tools: list[dict[str, Any]] | None,
) -> tuple[list[ToolCallItem] | None, str]:
    """Apply :func:`_finish`, keeping the remainder only when calls survive.

    When the offered-tools rule drops every call the whole message is content,
    so the caller must fall back to the full original text rather than a
    remainder that excludes the rejected markup.
    """

    finished = _finish(calls, tools)
    if finished is None:
        return None, ""
    return finished, remainder
