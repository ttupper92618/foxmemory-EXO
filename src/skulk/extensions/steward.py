"""Trusted extension tools for bounded steward reads and inert proposals.

This facet supplies no operator credentials or execution approval. Effectful
capabilities remain behind their own approval authority; a proposal hook must
only retain an inert request. Installing Python extensions is a trust decision,
not a sandbox boundary.
"""

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, final, runtime_checkable

from jsonschema import Draft202012Validator
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from skulk.extensions.types import ExtensionContext
from skulk.extensions.validation import validate_against_schema


@final
class StewardTool(BaseModel):
    """One installed adapter's bounded read or inert proposal contract."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: str = Field(
        pattern=r"^extension_[a-z][a-z0-9_]{0,48}$",
        description="Reserved extension namespace; cannot shadow built-in tools.",
    )
    description: str = Field(min_length=1, max_length=1024)
    mode: Literal["read", "proposal"] = Field(
        description="Proposals never execute effects or mint their own approval."
    )
    input_schema: dict[str, JsonValue] = Field(
        description="Self-contained JSON Schema for model-supplied arguments."
    )

    @property
    def revision(self) -> str:
        """Bind invocation to the exact contract exposed to this model turn."""
        raw = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()


@runtime_checkable
class StewardToolProvider(Protocol):
    """Optional trusted extension facet; all methods must cooperate with cancellation."""

    async def steward_tools(self, context: ExtensionContext) -> Sequence[StewardTool]:
        """Discover currently eligible tools without performing external effects."""
        ...

    async def handle_steward_tool(
        self,
        context: ExtensionContext,
        tool: StewardTool,
        arguments: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        """Perform a bounded read or store an inert proposal; never self-approve."""
        ...


@final
@dataclass(frozen=True)
class StewardToolBinding:
    """Request-local ownership of one validated tool and its exact adapter."""

    tool: StewardTool
    provider: StewardToolProvider


async def collect_steward_tools(
    extensions: Sequence[object],
    context: ExtensionContext,
    *,
    proposals_allowed: bool,
) -> tuple[StewardToolBinding, ...]:
    """Bound discovery, omit conflicting names, and filter proposal authority.

    A failed adapter contributes no tools. Names claimed by multiple adapters
    are removed rather than giving installation order authority over dispatch.
    """

    async def discover(extension: StewardToolProvider) -> tuple[StewardTool, ...]:
        try:
            async with asyncio.timeout(2):
                supplied = await extension.steward_tools(context)
            if len(supplied) > 16:
                raise ValueError("too many extension tools")
            validated: list[StewardTool] = []
            for item in supplied:
                # Copy mutable schema dictionaries at the request boundary.
                raw = item.model_dump_json()
                if len(raw.encode()) > 8192:
                    raise ValueError("extension tool exceeds size limit")
                tool = StewardTool.model_validate_json(raw)
                if tool.input_schema.get("type") != "object":
                    raise ValueError("tool input must describe an object")
                Draft202012Validator.check_schema(tool.input_schema)
                validated.append(tool)
            return tuple(validated)
        except Exception as error:
            # Adapter errors may contain payloads or credentials; log only type.
            logger.warning(
                "steward extension discovery failed: {}", type(error).__name__
            )
            return ()

    providers = [
        extension
        for extension in extensions
        if isinstance(extension, StewardToolProvider)
    ]
    if len(providers) > 32:
        return ()
    discovered = await asyncio.gather(*(discover(provider) for provider in providers))
    bindings: dict[str, StewardToolBinding] = {}
    conflicts: set[str] = set()
    for provider, supplied in zip(providers, discovered, strict=True):
        for tool in supplied:
            if tool.mode == "proposal" and not proposals_allowed:
                continue
            if tool.name in bindings or tool.name in conflicts:
                bindings.pop(tool.name, None)
                conflicts.add(tool.name)
                continue
            bindings[tool.name] = StewardToolBinding(tool, provider)
    # Bound the total prompt footprint even when many trusted adapters install.
    if (
        len(bindings) > 32
        or sum(len(item.tool.model_dump_json().encode()) for item in bindings.values())
        > 65536
    ):
        return ()
    return tuple(bindings.values())


async def invoke_steward_tool(
    binding: StewardToolBinding,
    context: ExtensionContext,
    arguments: dict[str, object],
    *,
    proposals_allowed: bool,
) -> str:
    """Recheck current eligibility and contract before invoking the exact owner.

    Failures are sanitized. No approval credential is supplied, and proposal
    tools cannot be invoked by a read-only steward session even via a forged call.
    """
    try:
        if binding.tool.mode == "proposal" and not proposals_allowed:
            raise ValueError("proposal authority required")
        raw = json.dumps(arguments, allow_nan=False)
        if len(raw.encode()) > 8192:
            raise ValueError("tool arguments exceed size limit")
        from pydantic import TypeAdapter

        payload = TypeAdapter(dict[str, JsonValue]).validate_json(raw)
        async with asyncio.timeout(5):
            current = await collect_steward_tools(
                [binding.provider], context, proposals_allowed=proposals_allowed
            )
            if not any(item.tool.revision == binding.tool.revision for item in current):
                raise ValueError("tool contract changed or withdrawn")
            if (
                validate_against_schema(
                    payload, dict(binding.tool.input_schema), what="arguments"
                )
                is not None
            ):
                raise ValueError("tool arguments refused")
            result = await binding.provider.handle_steward_tool(
                context, binding.tool, payload
            )
            encoded = json.dumps(result, allow_nan=False)
            if len(encoded.encode()) > 16384:
                raise ValueError("tool result exceeds size limit")
            return encoded
    except Exception as error:
        logger.warning("steward extension invocation failed: {}", type(error).__name__)
        return '{"error":"extension tool unavailable or refused"}'
