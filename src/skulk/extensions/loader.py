"""Discovery, version gating, and guarded dispatch for Skulk extensions.

Extensions are installed Python packages exposing a zero-argument factory in
the ``skulk.extensions`` entry-point group. They are discovered once at node
startup. A failing or version-incompatible extension is skipped with a loud
error; it can never prevent the fabric from serving.
"""

import asyncio
import os
from collections.abc import AsyncGenerator, Callable, Iterable, Sequence
from importlib.metadata import EntryPoint, PackageNotFoundError, entry_points, version
from typing import cast, final

import anyio
from loguru import logger
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from skulk.extensions.capabilities import CapabilityDescriptor
from skulk.extensions.types import (
    CapabilityCallHandler,
    CapabilityInputStreamHandler,
    CapabilityProvider,
    CapabilityReadiness,
    CapabilityStreamHandler,
    ChatMiddleware,
    ChatResponseSummary,
    ExtensionContext,
    SkulkExtension,
    SupportsExtensionShutdown,
    SupportsExtensionStartup,
)
from skulk.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from skulk.shared.types.text_generation import TextGenerationTaskParams

ENTRY_POINT_GROUP = "skulk.extensions"
_EXTENSION_SHUTDOWN_TIMEOUT_SECONDS = 30.0

# Chunk union yielded by the API's per-command chat stream; the tap must be
# transparent to every variant.
ChatStreamChunk = TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk

# Keep strong references to scheduled observer tasks so they are not
# garbage-collected mid-flight; each task discards itself on completion.
_observer_tasks: set[asyncio.Task[None]] = set()


def resolve_skulk_version() -> str:
    """Return the installed Skulk version, or ``0.0.0`` if undetectable."""
    try:
        return version("skulk")
    except PackageNotFoundError:
        return "0.0.0"


@final
class LoadedExtensions:
    """The set of successfully loaded extensions plus guarded dispatch.

    All extension calls route through this class so the "never degrade
    inference" invariant lives in exactly one place rather than at every
    call site.
    """

    def __init__(self, extensions: Sequence[SkulkExtension]) -> None:
        """Wrap validated extensions (see :func:`load_extensions`).

        Every extension attribute access here is guarded too: this runs at
        node startup, so a plugin whose ``name`` property or
        ``chat_middleware()`` raises must be skipped loudly, never allowed to
        crash the process (the "loader never raises" contract).
        """
        self._extension_instances: list[SkulkExtension] = []
        self._capability_providers: dict[str, SkulkExtension] = {}
        self._started = False
        self._stopping = False
        self._names: list[str] = []
        self._chat_middlewares: list[tuple[str, ChatMiddleware]] = []
        self._startup_hooks: list[tuple[str, SupportsExtensionStartup]] = []
        self._capability_descriptors: list[CapabilityDescriptor] = []
        # qualified_id -> (extension name, handler, descriptor) for providers
        # that also serve unary calls (fabric-citizenship Phase 2b).
        self._call_handlers: dict[
            str, tuple[str, CapabilityCallHandler, CapabilityDescriptor]
        ] = {}
        self._stream_handlers: dict[
            str,
            tuple[
                str,
                CapabilityStreamHandler | CapabilityInputStreamHandler,
                CapabilityDescriptor,
            ],
        ] = {}
        seen_qualified_ids: set[str] = set()
        for extension in extensions:
            try:
                name = str(extension.name)
            except Exception as exc:  # noqa: BLE001 - plugin must not crash startup
                logger.error(f"extension name property raised; skipping: {exc}")
                continue
            self._extension_instances.append(extension)
            try:
                middleware = extension.chat_middleware()
            except Exception as exc:  # noqa: BLE001 - plugin must not crash startup
                logger.error(
                    f"extension '{name}' chat_middleware() raised; loading it "
                    f"without chat hooks: {exc}"
                )
                middleware = None
            self._names.append(name)
            if middleware is not None:
                self._chat_middlewares.append((name, middleware))
            # Provider facet (fabric-citizenship): collect this extension's
            # capability descriptors. Structural check + guarded call so a
            # non-provider extension or a raising capabilities() can never
            # break loading.
            if isinstance(extension, CapabilityProvider):
                try:
                    descriptors = list(extension.capabilities())
                except Exception as exc:  # noqa: BLE001 - plugin must not crash startup
                    logger.error(
                        f"extension '{name}' capabilities() raised; loading it "
                        f"without capabilities: {exc}"
                    )
                    descriptors = []
                for descriptor in descriptors:
                    if not isinstance(descriptor, CapabilityDescriptor):  # pyright: ignore[reportUnnecessaryIsInstance]
                        logger.error(
                            f"extension '{name}' returned a non-descriptor "
                            f"capability entry; skipping it"
                        )
                        continue
                    # One provider per id@version per node: a duplicate would
                    # make describe/call ambiguous locally.
                    if descriptor.qualified_id in seen_qualified_ids:
                        logger.error(
                            f"extension '{name}' capability "
                            f"'{descriptor.qualified_id}' is already provided "
                            f"by another extension on this node; skipping it"
                        )
                        continue
                    seen_qualified_ids.add(descriptor.qualified_id)
                    self._capability_descriptors.append(descriptor)
                    self._capability_providers[descriptor.qualified_id] = extension
                    # Call facet (Phase 2b): a provider that also implements
                    # handle_call serves this descriptor's unary calls. A
                    # descriptor without a handler is discovery-only (valid:
                    # for example a streaming-only capability before Phase 3).
                    # Only unary descriptors register: routing a streaming
                    # capability through the unary surface would serve it with
                    # the wrong payload/result contract.
                    if descriptor.io_mode == "unary" and isinstance(
                        extension, CapabilityCallHandler
                    ):
                        self._call_handlers[descriptor.qualified_id] = (
                            name,
                            extension,
                            descriptor,
                        )
                    if descriptor.io_mode == "server_streaming" and isinstance(
                        extension, CapabilityStreamHandler
                    ):
                        self._stream_handlers[descriptor.qualified_id] = (
                            name,
                            extension,
                            descriptor,
                        )
                    if descriptor.io_mode in (
                        "client_streaming",
                        "bidirectional",
                    ) and isinstance(extension, CapabilityInputStreamHandler):
                        self._stream_handlers[descriptor.qualified_id] = (
                            name,
                            extension,
                            descriptor,
                        )
            # Startup-hook facet: a pure provider has no chat hook through
            # which to reach the context, so registration must not depend on
            # a chat request arriving.
            if isinstance(extension, SupportsExtensionStartup):
                self._startup_hooks.append((name, extension))

    def with_builtin_extensions(
        self, extensions: Sequence[SkulkExtension]
    ) -> "LoadedExtensions":
        """Return a registry with first-party providers taking precedence.

        Built-ins are prepended so their reserved ``id@version`` contracts win
        deterministically over an external extension attempting to publish the
        same qualified capability. Existing extension facets are copied from
        this registry rather than discovered again because middleware and
        capability factories may be stateful.
        """

        combined = LoadedExtensions(extensions)
        combined._extension_instances.extend(self._extension_instances)
        combined._names.extend(self._names)
        combined._chat_middlewares.extend(self._chat_middlewares)
        combined._startup_hooks.extend(self._startup_hooks)

        seen_qualified_ids = {
            descriptor.qualified_id for descriptor in combined._capability_descriptors
        }
        for descriptor in self._capability_descriptors:
            qualified_id = descriptor.qualified_id
            if qualified_id in seen_qualified_ids:
                logger.error(
                    f"external capability '{qualified_id}' is reserved by a "
                    "first-party provider on this node; skipping it"
                )
                continue
            seen_qualified_ids.add(qualified_id)
            combined._capability_descriptors.append(descriptor)
            combined._capability_providers[qualified_id] = self._capability_providers[
                qualified_id
            ]
            call_handler = self._call_handlers.get(qualified_id)
            if call_handler is not None:
                combined._call_handlers[qualified_id] = call_handler
            stream_handler = self._stream_handlers.get(qualified_id)
            if stream_handler is not None:
                combined._stream_handlers[qualified_id] = stream_handler
        return combined

    @property
    def names(self) -> list[str]:
        """Names of all loaded extensions."""
        return list(self._names)

    @property
    def has_chat_middleware(self) -> bool:
        """Whether any loaded extension provides chat middleware."""
        return bool(self._chat_middlewares)

    @property
    def capability_descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        """All capability descriptors served by loaded provider extensions."""
        return tuple(
            descriptor
            for descriptor in self._capability_descriptors
            if self.capability_ready(descriptor.qualified_id)
        )

    def call_handler(
        self, qualified_id: str
    ) -> tuple[str, CapabilityCallHandler, CapabilityDescriptor] | None:
        """Look up the call handler serving ``id@version`` on this node.

        Returns ``(extension name, handler, descriptor)``, or ``None`` when no
        loaded provider serves unary calls for that capability (unknown id,
        wrong version, or a discovery-only descriptor without a handler).
        """
        if not self.capability_ready(qualified_id):
            return None
        return self._call_handlers.get(qualified_id)

    def handled_capability_ids(self) -> frozenset[str]:
        """Capability ids (bare, unversioned) with at least one call handler."""
        return frozenset(
            entry[2].id
            for entry in self._call_handlers.values()
            if self.capability_ready(entry[2].qualified_id)
        )

    def stream_handler(
        self, qualified_id: str
    ) -> (
        tuple[
            str,
            CapabilityStreamHandler | CapabilityInputStreamHandler,
            CapabilityDescriptor,
        ]
        | None
    ):
        """Look up the streaming handler serving ``id@version`` on this node."""

        if not self.capability_ready(qualified_id):
            return None
        return self._stream_handlers.get(qualified_id)

    def handled_stream_capability_ids(self) -> frozenset[str]:
        """Bare capability ids with an executable streaming handler."""

        return frozenset(
            entry[2].id
            for entry in self._stream_handlers.values()
            if self.capability_ready(entry[2].qualified_id)
        )

    def capability_ready(self, qualified_id: str) -> bool:
        """Read cached provider readiness; unknown, stopping and broken fail closed."""
        provider = self._capability_providers.get(qualified_id)
        if provider is None or self._stopping:
            return False
        try:
            return (
                provider.capability_ready(qualified_id) is True
                if isinstance(provider, CapabilityReadiness)
                else True
            )
        except Exception:
            # Cached readiness is a plugin boundary: failures cannot expose an
            # executable contract or leak a provider's private exception payload.
            return False

    async def run_shutdown_hooks(self) -> None:
        """Withdraw discovery and clean owned resources once, within thirty seconds.

        Hooks are independent and concurrent, so one failing or slow extension
        cannot prevent another from flushing state. Cooperative cancellation is
        required of in-process extensions, just as for ordinary API handlers.
        """
        if self._stopping:
            return
        self._stopping = True

        async def stop(extension: SupportsExtensionShutdown) -> None:
            try:
                await extension.on_stop()
            except Exception as error:
                logger.error(f"Extension shutdown failed: {type(error).__name__}")

        with anyio.move_on_after(_EXTENSION_SHUTDOWN_TIMEOUT_SECONDS, shield=True) as scope:
            async with anyio.create_task_group() as tasks:
                for extension in self._extension_instances:
                    if isinstance(extension, SupportsExtensionShutdown):
                        tasks.start_soon(stop, extension)
        if scope.cancel_called:
            logger.warning("Extension shutdown exceeded the thirty-second budget")

    def run_startup_hooks(self, context: ExtensionContext) -> None:
        """Run every extension's ``on_start`` hook, each one guarded.

        Called once when the node's extension surface comes up (the API wires
        the context and invokes this). A raising hook is logged loudly and the
        extension continues without its startup work; startup of the node is
        never blocked by a broken plugin.
        """
        if self._started or self._stopping:
            return
        self._started = True
        for name, extension in self._startup_hooks:
            try:
                extension.on_start(context)
            except Exception as exc:  # noqa: BLE001 - extension must not break startup
                logger.error(
                    f"extension '{name}' on_start failed; continuing without "
                    f"its startup work: {exc}"
                )

    async def transform_chat_request(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
    ) -> TextGenerationTaskParams:
        """Run every middleware's request transform, each one guarded.

        A raising middleware is logged and skipped; the request continues
        with the most recent successfully transformed params.
        """
        params = task_params
        for name, middleware in self._chat_middlewares:
            try:
                params = await middleware.transform_chat_request(context, params)
            except Exception as exc:  # noqa: BLE001 - extension must not break serving
                logger.error(
                    f"extension '{name}' transform_chat_request failed; "
                    f"continuing without it: {exc}"
                )
        return params

    def tap_chat_stream(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
        stream: AsyncGenerator[ChatStreamChunk, None],
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        """Wrap a chat chunk stream so observers see a completion summary.

        Skulk owns the accumulation loop; extensions only ever receive an
        immutable :class:`ChatResponseSummary` in a background task, so a
        buggy observer cannot corrupt, reorder, or stall token delivery. When
        no middleware is loaded the stream is returned untouched.
        """
        if not self._chat_middlewares:
            return stream
        return self._tap(context, task_params, stream)

    async def _tap(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
        stream: AsyncGenerator[ChatStreamChunk, None],
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        finish_reason: str | None = None
        had_error = False
        scheduled = False

        def schedule() -> None:
            summary = ChatResponseSummary(
                text="".join(text_parts),
                thinking_text="".join(thinking_parts),
                finish_reason=finish_reason,
                had_error=had_error,
            )
            for name, middleware in self._chat_middlewares:
                task = asyncio.get_running_loop().create_task(
                    self._observe(name, middleware, context, task_params, summary)
                )
                _observer_tasks.add(task)
                task.add_done_callback(_observer_tasks.discard)

        try:
            async for chunk in stream:
                match chunk:
                    case TokenChunk():
                        if chunk.is_thinking:
                            thinking_parts.append(chunk.text)
                        else:
                            text_parts.append(chunk.text)
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                    case ToolCallChunk():
                        finish_reason = chunk.finish_reason
                    case ErrorChunk():
                        finish_reason = chunk.finish_reason
                        had_error = True
                    case PrefillProgressChunk():
                        pass
                # Schedule at the terminal chunk rather than after the loop:
                # consumers close this generator immediately after the final
                # chunk, so code after the loop may never run on the happy
                # path (the finally below covers abnormal endings).
                if finish_reason is not None and not scheduled:
                    scheduled = True
                    schedule()
                yield chunk
        finally:
            # Client disconnects and upstream errors end the stream without a
            # terminal chunk; observers still get the partial summary.
            if not scheduled:
                scheduled = True
                schedule()

    @staticmethod
    async def _observe(
        name: str,
        middleware: ChatMiddleware,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
        summary: ChatResponseSummary,
    ) -> None:
        try:
            await middleware.observe_chat_response(context, task_params, summary)
        except Exception as exc:  # noqa: BLE001 - extension must not break serving
            logger.error(f"extension '{name}' observe_chat_response failed: {exc}")


def _validate_extension(
    candidate: object, skulk_version: str, source: str
) -> SkulkExtension | None:
    """Duck-type and version-gate one loaded entry-point object.

    Metadata is read exactly once under a guard and the captured values are
    used for gating and every log line: ``hasattr`` would only suppress
    ``AttributeError``, and a plugin property that raises anything else must
    not escape the "loader never raises" contract.
    """
    # Unchecked protocol cast; every attribute read below stays inside the
    # guard because a plugin property may raise anything, not just
    # AttributeError.
    extension = cast("SkulkExtension", candidate)
    try:
        name = str(extension.name)
        requires = str(extension.skulk_requires)
        middleware_factory = extension.chat_middleware
        if not callable(middleware_factory):
            logger.error(
                f"extension from {source}: chat_middleware is not callable; skipping"
            )
            return None
    except Exception as exc:  # noqa: BLE001 - plugin must not crash startup
        logger.error(
            f"extension from {source} has unreadable metadata; skipping: {exc}"
        )
        return None
    try:
        specifier = SpecifierSet(requires)
    except InvalidSpecifier:
        logger.error(
            f"extension '{name}' has invalid skulk_requires {requires!r}; skipping"
        )
        return None
    try:
        running = Version(skulk_version)
    except InvalidVersion:
        logger.warning(
            f"cannot parse running skulk version {skulk_version!r}; "
            f"loading extension '{name}' without a version gate"
        )
        return extension
    if running not in specifier:
        # Same anti-pattern as a mixed-version cluster: refuse loudly rather
        # than run a plugin against a fabric it was not built for.
        logger.error(
            f"extension '{name}' requires skulk {requires!r} but this node "
            f"runs {skulk_version}; refusing to load it. Upgrade the "
            f"extension and the fleet together."
        )
        return None
    return extension


def load_extensions(
    *,
    candidates: Iterable[EntryPoint] | None = None,
    skulk_version: str | None = None,
) -> LoadedExtensions:
    """Discover, validate, and version-gate all installed extensions.

    Called once at node startup. Every failure mode (bad entry point, raising
    factory, missing protocol attributes, version mismatch) is logged loudly
    and skipped; this function never raises. The ``SKULK_EXTENSIONS_DISABLE=1``
    environment variable is a node-local kill switch that skips discovery
    entirely.

    Args:
        candidates: Entry points to load (defaults to the installed
            ``skulk.extensions`` group); injectable for tests.
        skulk_version: Version to gate against (defaults to the installed
            Skulk version); injectable for tests.
    """
    if os.environ.get("SKULK_EXTENSIONS_DISABLE", "").strip() == "1":
        logger.warning("SKULK_EXTENSIONS_DISABLE=1: skipping extension discovery")
        return LoadedExtensions([])

    if candidates is None:
        candidates = entry_points(group=ENTRY_POINT_GROUP)
    running_version = (
        skulk_version if skulk_version is not None else resolve_skulk_version()
    )

    loaded: list[SkulkExtension] = []
    for entry_point in candidates:
        source = f"entry point '{entry_point.name}' ({entry_point.value})"
        try:
            factory = cast("Callable[[], object]", entry_point.load())
            candidate: object = factory()
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not stop the node
            logger.error(f"failed to load {source}: {exc}")
            continue
        extension = _validate_extension(candidate, running_version, source)
        if extension is not None:
            loaded.append(extension)

    result = LoadedExtensions(loaded)
    if result.names:
        # Log via the guarded .names, never by re-reading plugin properties.
        logger.info(
            f"loaded {len(result.names)} extension(s): {', '.join(result.names)}"
        )
    return result
