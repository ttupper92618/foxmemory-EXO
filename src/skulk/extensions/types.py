"""Public types for Skulk extensions (plugins).

Extensions are separately installed Python packages that Skulk discovers at
startup through the ``skulk.extensions`` entry-point group and loads as
first-class citizens of the fabric. The contract is deliberately small and
general: an extension can transform a chat request before it is dispatched to
the cluster and observe the completed response after it streams back. Nothing
in this module is specific to any particular extension.

Design invariants (enforced by the call sites in :mod:`skulk.api.main` and
:mod:`skulk.extensions.loader`):

- **Extensions must never degrade inference.** Every extension call is
  guarded; a raising extension is logged loudly and the request proceeds as
  if the extension did not exist.
- **Extensions never own the response stream.** Skulk accumulates the
  response and hands extensions an immutable summary, so a buggy extension
  cannot corrupt or stall token delivery.
- **No extension installed = Skulk unchanged.** The hooks are inert when the
  loaded-extension list is empty.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol, final, runtime_checkable

from pydantic import BaseModel

from skulk.extensions.calls import CapabilityCall, CapabilityError, CapabilityResult
from skulk.extensions.capabilities import CapabilityDescriptor
from skulk.extensions.streams import CapabilityStreamFrame, CapabilityStreamSession
from skulk.extensions.telemetry import ClusterNodeView
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.text_generation import TextGenerationTaskParams


class EmbedTexts(Protocol):
    """Async callable that embeds texts via the cluster's embedding serving.

    Returns one embedding vector per input text, or ``None`` when no
    embedding instance is available or the call fails or times out. Callers
    must treat ``None`` as "degrade gracefully", never as an error to raise.
    """

    async def __call__(
        self, texts: list[str], model_id: ModelId | None = None
    ) -> list[list[float]] | None: ...


class ReadClusterTelemetry(Protocol):
    """Synchronous callable returning a snapshot of the telemetry plane.

    Each call returns an immutable, per-node projection of what this node
    currently sees on the telemetry plane (peers, their backends, participation
    role, accelerator vendor, version, and liveness). It is a read: an extension
    can observe the cluster it belongs to but never mutate telemetry. Cheap and
    side-effect free (an in-memory snapshot, no network I/O), so it is safe to
    call from an inline hook.
    """

    def __call__(self) -> tuple[ClusterNodeView, ...]: ...


class AdvertiseCapability(Protocol):
    """Synchronous callable that advertises a capability on the telemetry plane.

    The extension calls this to publish an opaque capability tag (for example
    ``"memory"``) that peers then see in their
    :attr:`ClusterNodeView.capabilities`. It is the *advertise* half of
    telemetry-plane access: how a plugin announces what it offers, the same way
    native nodes advertise their backends.

    Advertising is additive and idempotent: re-advertising the same tag is a
    no-op, and the tag keeps being gossiped (last-write-wins) until the node
    leaves the cluster. Cheap and side-effect free beyond recording the tag (it
    only mutates a local set; the gossip happens on the node's normal telemetry
    poll), so it is safe to call from an inline hook (typically once at startup).
    """

    def __call__(self, capability: str) -> None: ...


class WithdrawCapability(Protocol):
    """Synchronous callable that withdraws a previously advertised capability.

    The provider-liveness counterpart of :class:`AdvertiseCapability`: a plugin
    that stops offering a capability (shutting down its backend, entering
    maintenance) withdraws the tag so callers stop selecting it. Withdrawing a
    tag that was never advertised is a no-op. Peers see the withdrawal on the
    node's next telemetry poll.
    """

    def __call__(self, capability: str) -> None: ...


class DescribeNode(Protocol):
    """Async callable returning a node's full capability descriptors.

    The heavy half of discovery: the telemetry tag says *that* a node offers a
    capability; ``describe_node`` fetches the self-describing
    :class:`~skulk.extensions.capabilities.CapabilityDescriptor` list that says
    *how to call it* (schemas, I/O mode, version). Returns an empty tuple when
    the node is unreachable or serves no capabilities; callers must degrade
    gracefully, never raise. Descriptors travel on demand so telemetry stays
    cheap.
    """

    async def __call__(self, node_id: NodeId) -> tuple[CapabilityDescriptor, ...]: ...


@runtime_checkable
class CapabilityProvider(Protocol):
    """Optional extension facet: the extension serves capabilities.

    An extension that implements ``capabilities()`` is a **provider**: its
    descriptors are collected at load time, exposed to the cluster via the
    node's describe surface (``GET /v1/capabilities`` and peers'
    ``describe_node``), and their ids are auto-advertised on the telemetry
    plane. Detected structurally (``isinstance`` against this runtime
    protocol), so a plain chat-middleware extension is unaffected.
    """

    def capabilities(self) -> Sequence[CapabilityDescriptor]:
        """Return the capability descriptors this extension serves."""
        ...


@runtime_checkable
class CapabilityReadiness(Protocol):
    """Optional immediate readiness check for each installed capability.

    Discovery and new unary/stream admission consult this facet. Return a cached
    boolean without blocking or I/O; false or a raised exception hides the
    descriptor and refuses new calls. Already admitted calls keep their ordinary
    deadline and cancellation contract. Providers without this facet stay ready.
    """

    def capability_ready(self, qualified_id: str) -> bool:
        """Return readiness for the exact ``id@version`` without side effects."""
        ...


@runtime_checkable
class CapabilityCallHandler(Protocol):
    """Optional provider facet: the extension serves unary capability calls.

    A provider implementing ``handle_call`` becomes callable: the node's call
    surface routes an inbound :class:`~skulk.extensions.calls.CapabilityCall`
    whose ``capability_id@version`` matches one of the provider's descriptors
    to this method, AFTER the fabric has validated the payload against the
    descriptor's ``input_schema`` and BEFORE the result is validated against
    its ``output_schema``. The handler returns the raw result payload dict.

    Isolation contract (enforced by the dispatch in :mod:`skulk.api.main`):
    the call runs under the caller's deadline, inside the node's provider
    concurrency bound, with payload size caps on both directions. A raising
    handler becomes a typed ``provider_error`` for the caller and never
    degrades the node. The handler is ``async``; CPU-heavy or blocking work
    inside it must be moved off the event loop by the extension (for example
    via a worker thread), or it will stall the API node it runs on.
    """

    async def handle_call(
        self, context: "ExtensionContext", call: CapabilityCall
    ) -> dict[str, object]:
        """Serve one unary capability call, returning the result payload."""
        ...


@runtime_checkable
class CapabilityStreamHandler(Protocol):
    """Optional provider facet for server or bidirectional output streams."""

    def handle_stream(
        self,
        context: "ExtensionContext",
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        """Yield ordered chunks and one terminal for an admitted provider call.

        Skulk emits ``started`` at sequence zero after admission. The handler
        therefore begins at sequence one, yields zero or more ``chunk`` frames,
        and finishes with exactly one ``completed``, ``failed``, or
        ``cancelled`` frame, then returns. Skulk withholds the terminal until
        the iterator reaches exhaustion so handler ``finally`` cleanup is
        complete before callers can begin dependent work. It validates
        identity, sequence, chunk schema, and terminal ownership before
        publishing any handler frame. Malformed output closes a closable
        iterator before Skulk publishes its synthetic failure terminal.
        """

        ...


@runtime_checkable
class CapabilityInputStreamHandler(Protocol):
    """Optional provider facet for client-streaming or bidirectional calls."""

    def handle_input_stream(
        self,
        context: "ExtensionContext",
        call: CapabilityCall,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> AsyncIterator[CapabilityStreamFrame]:
        """Consume ordered caller frames and yield the provider output direction.

        ``input_frames`` begins with ``started`` and ends with exactly one
        ``completed``, ``failed``, or ``cancelled`` frame. Caller ``completed``
        is a half-close, not cancellation. Provider output follows the same
        sequence-one-through-terminal contract as
        :class:`CapabilityStreamHandler` because Skulk owns output ``started``;
        its terminal is likewise withheld until the output iterator returns.
        """

        ...


@runtime_checkable
class CapabilityStreamAdmissionHandler(Protocol):
    """Optional dynamic validation before a provider stream is admitted."""

    async def admit_stream(
        self,
        context: "ExtensionContext",
        call: CapabilityCall,
    ) -> CapabilityError | None:
        """Return a typed opening error, or ``None`` to emit ``started``.

        The fabric already validates the static descriptor schema. This hook
        covers dynamic requirements such as a mounted model or a healthy
        external service. It runs inside the stream's concurrency and deadline
        budget, before Skulk creates the active stream lifecycle.
        """

        ...


class StreamCapability(Protocol):
    """Async callable opening any executable streaming provider capability."""

    async def __call__(
        self,
        node_id: NodeId,
        capability_id: str,
        version: str,
        descriptor_revision: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> CapabilityStreamSession:
        """Open a stream and return output plus an input sink when negotiated."""
        ...


class CallCapability(Protocol):
    """Async callable that invokes a capability on a provider node.

    The caller side of the generic call verb (fabric-citizenship Phase 2b):
    address a node (discovered via ``read_cluster``/``describe_node``), name
    the exact ``capability_id@version`` and the descriptor revision you
    discovered, and pass the payload. Always returns a
    :class:`~skulk.extensions.calls.CapabilityResult`; every failure mode
    (unreachable node, drifted descriptor, invalid payload, provider error,
    timeout) arrives as a typed error on the result, never as an exception.
    Transport-abstract: the underlying hop may change without affecting
    callers.
    """

    async def __call__(
        self,
        node_id: NodeId,
        capability_id: str,
        version: str,
        descriptor_revision: str,
        payload: dict[str, object],
        *,
        timeout_seconds: float | None = None,
    ) -> CapabilityResult: ...


@runtime_checkable
class SupportsExtensionStartup(Protocol):
    """Optional extension facet: a startup hook.

    ``on_start`` runs once, when the node's extension surface comes up, with
    the same :class:`ExtensionContext` the request hooks receive. It exists
    because a pure provider has no chat hook through which to reach the
    context; without it, advertising or any startup registration would depend
    on a chat request arriving. Guarded like every extension call (a raising
    ``on_start`` is logged and the extension continues without its startup
    work). Must be fast and non-blocking: it runs during node startup, so
    heavy initialization belongs in background work the extension owns.
    """

    def on_start(self, context: "ExtensionContext") -> None:
        """Run startup registration with the live extension context."""
        ...


@runtime_checkable
class SupportsExtensionShutdown(Protocol):
    """Optional asynchronous cleanup of extension-owned background resources.

    The API withdraws provider discovery before invoking cleanup on its original
    event loop. Hooks run concurrently under a thirty-second cancellation-shielded
    budget. Implementations must cooperate with cancellation and keep blocking
    work off the event loop; this trusted-code hook is not a process sandbox.
    """

    async def on_stop(self) -> None:
        """Flush bounded state and stop owned tasks/processes before returning."""
        ...


@final
@dataclass(frozen=True)
class ExtensionContext:
    """Capabilities Skulk hands to an extension at each hook invocation.

    Attributes:
        node_id: The identity of the node running this API process.
        skulk_version: The installed Skulk version (PEP 440 string).
        embed_texts: Programmatic access to the cluster's embedding serving
            (the in-process equivalent of ``POST /v1/embeddings``).
        read_cluster: The telemetry-plane read surface: returns an immutable
            per-node snapshot of the cluster (fabric-citizenship Phase 1). How a
            plugin discovers its peers and their health.
        advertise_capability: The telemetry-plane advertise surface: publishes
            an opaque capability tag this node offers so peers discover it via
            their own ``read_cluster`` snapshots.
        withdraw_capability: The advertise surface's liveness counterpart:
            stops advertising a tag so callers stop selecting this node for it.
        describe_node: The heavy half of discovery: fetches a node's full
            capability descriptors (schemas, I/O modes, versions) on demand.
        call_capability: The generic call verb: invokes a capability on a
            provider node with a typed result (fabric-citizenship Phase 2b).
        stream_capability: Open a streaming capability whose active input and
            output directions travel on provider DATA (Phase 3).
    """

    node_id: NodeId
    skulk_version: str
    embed_texts: EmbedTexts
    read_cluster: ReadClusterTelemetry
    advertise_capability: AdvertiseCapability
    withdraw_capability: WithdrawCapability
    describe_node: DescribeNode
    call_capability: CallCapability
    stream_capability: StreamCapability


@final
class ChatResponseSummary(BaseModel, frozen=True):
    """Immutable summary of a completed chat generation, for observers.

    Attributes:
        text: The concatenated non-thinking output text.
        thinking_text: The concatenated reasoning/thinking text, if any.
        finish_reason: The terminal finish reason (``stop``, ``length``,
            ``tool_calls``, ``error``, ...), or ``None`` when the stream
            ended without a terminal chunk (for example a client disconnect).
        had_error: Whether the stream terminated with an error chunk.
    """

    text: str
    thinking_text: str
    finish_reason: str | None
    had_error: bool


class ChatMiddleware(Protocol):
    """Per-request chat hooks an extension can provide.

    Both methods are optional in spirit; subclass :class:`BaseChatMiddleware`
    to implement only one of them.
    """

    async def transform_chat_request(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
    ) -> TextGenerationTaskParams:
        """Return (possibly) modified task params before cluster dispatch.

        Runs on the API node after the OpenAI adapter and before the command
        is sent to the master. Raising is safe: the request proceeds with the
        params returned by the previous middleware (or the originals).
        """
        ...

    async def observe_chat_response(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
        summary: ChatResponseSummary,
    ) -> None:
        """Observe a completed generation.

        Scheduled as a background task once the response finishes (or the
        stream ends); never blocks token delivery. Raising is safe and only
        logs.
        """
        ...


class BaseChatMiddleware:
    """No-op :class:`ChatMiddleware` base; override what you need."""

    async def transform_chat_request(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
    ) -> TextGenerationTaskParams:
        """Return the params unchanged (identity transform)."""
        return task_params

    async def observe_chat_response(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
        summary: ChatResponseSummary,
    ) -> None:
        """Ignore the response (no-op observer)."""
        return None


class SkulkExtension(Protocol):
    """The object a ``skulk.extensions`` entry point must produce.

    The entry point value is a zero-argument callable (class or factory)
    returning an instance satisfying this protocol.
    """

    @property
    def name(self) -> str:
        """Short unique extension name, used in logs."""
        ...

    @property
    def skulk_requires(self) -> str:
        """PEP 440 version specifier for compatible Skulk versions.

        Example: ``==1.3.1`` or ``>=1.3.1,<1.4``. An extension whose
        specifier does not match the running Skulk is refused at load time
        with a loud error (mixed plugin/fabric versions are the same
        anti-pattern as mixed-version clusters).
        """
        ...

    def chat_middleware(self) -> ChatMiddleware | None:
        """Return this extension's chat middleware, or ``None`` for none."""
        ...
