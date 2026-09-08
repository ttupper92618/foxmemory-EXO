import os
import re
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, Self, cast, final

import psutil
from pydantic import UUID4, BaseModel, Field, field_serializer, field_validator

from skulk.shared.models.llama_server_settings import (
    LlamaServerSettings,
    resolve_llama_server_settings,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.node_facts import CapabilityConflict
from skulk.shared.types.thunderbolt import ThunderboltIdentifier
from skulk.utils.pydantic_ext import CamelCaseModel


class MemoryUsage(CamelCaseModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory

    @classmethod
    def from_bytes(
        cls, *, ram_total: int, ram_available: int, swap_total: int, swap_available: int
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
        )

    @classmethod
    def from_psutil(cls, *, override_memory: int | None) -> Self:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        return cls.from_bytes(
            ram_total=vm.total,
            ram_available=vm.available if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
        )

    @classmethod
    def from_local_gpu_wireable(cls) -> Self:
        """Local snapshot with ``ram_available`` as the GPU-wireable figure.

        ``total − wired − anonymous − compressor`` from a vm_stat snapshot —
        the same metric the telemetry path gossips for placement admission, so
        the master's check and the worker's local pre-spawn guard agree on
        what "available" means. psutil's ``available`` (free + inactive)
        counts reclaimable file cache as used and is kept only as the fallback
        when vm_stat fails.
        """
        categories = read_mach_memory_categories()
        return cls.from_psutil(
            override_memory=None
            if categories is None
            else gpu_wireable_memory_bytes(
                int(psutil.virtual_memory().total), categories
            )
        )


@final
class MachMemoryCategories(BaseModel, frozen=True, strict=True):
    """One consistent snapshot of macOS Mach page-category counters.

    Sourced from ``vm_stat`` (one ``host_statistics64`` snapshot), which is the
    only stock interface exposing all three of wired, anonymous, and compressor
    occupancy together — psutil lacks compressor occupancy and the ``vm.*``
    sysctls lack wired/compressor, and mixing sources tears the snapshot.
    """

    wired_bytes: int
    """Unpageable memory (kernel + GPU-wired). Never reclaimable."""

    anonymous_bytes: int
    """Resident anonymous (non-file-backed) pages — process heaps. Disjoint
    from wired in Mach accounting; reclaimable only via compression/swap."""

    compressor_bytes: int
    """Physical pages holding compressed memory. Resident until decompressed
    or swapped; counting them avoids overstating availability on a box
    already under memory pressure."""


_VM_STAT_PAGE_SIZE_PATTERN = re.compile(r"page size of (\d+) bytes")
_VM_STAT_COUNTER_PATTERN = re.compile(r"^(.+?):\s+(\d+)\.?\s*$", re.MULTILINE)


def parse_vm_stat_output(text: str) -> MachMemoryCategories | None:
    """Parse ``vm_stat`` output into a :class:`MachMemoryCategories` snapshot.

    Pure function (the subprocess lives at the caller). Returns ``None`` when
    the expected header or counters are missing, so a changed/foreign format
    degrades to "no snapshot" rather than a wrong number.
    """
    page_size_match = _VM_STAT_PAGE_SIZE_PATTERN.search(text)
    if page_size_match is None:
        return None
    page_size = int(page_size_match.group(1))
    counters = {
        match.group(1).strip(): int(match.group(2))
        for match in _VM_STAT_COUNTER_PATTERN.finditer(text)
    }
    try:
        return MachMemoryCategories(
            wired_bytes=counters["Pages wired down"] * page_size,
            anonymous_bytes=counters["Anonymous pages"] * page_size,
            compressor_bytes=counters["Pages occupied by compressor"] * page_size,
        )
    except KeyError:
        return None


def gpu_wireable_memory_bytes(
    ram_total_bytes: int, categories: MachMemoryCategories
) -> int:
    """Memory the GPU could wire without fighting resident working sets.

    ``total − wired − anonymous − compressor``: everything else (free,
    file-backed cache, purgeable) is reclaimed by macOS the moment Metal wires
    pages. The naive ``free + inactive + speculative`` figure that mactop (and
    macmon before it) reports counts reclaimable file cache as *used* — after a
    model download, ~weights-sized cache deflates it by the model's full size
    and placement refuses fits that run comfortably (observed on a 24 GB node:
    11.6 GB of just-downloaded weights in cache dropped "available" to 12 GB
    while 14.6 GB was genuinely wireable). Deliberately does NOT credit
    compression of idle anonymous memory — that would re-introduce the
    oversized-placement OOM class that the 1.30 overhead factor guards against.
    """
    return max(
        0,
        ram_total_bytes
        - categories.wired_bytes
        - categories.anonymous_bytes
        - categories.compressor_bytes,
    )


def read_mach_memory_categories() -> MachMemoryCategories | None:
    """One synchronous ``vm_stat`` snapshot, or ``None`` on any failure.

    For rare, latency-tolerant call sites (the worker's pre-spawn fit guard);
    the telemetry loop has its own anyio-based reader so the 1 Hz sample
    cadence never blocks the event loop.
    """
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return parse_vm_stat_output(result.stdout.decode("utf-8", errors="replace"))


def read_wired_memory_bytes() -> int | None:
    """OS-level wired (unpageable) memory in use, or None where unavailable.

    Kept OFF the gossiped ``MemoryUsage`` (which rides ``NodeGatheredInfo``
    events under ``extra=forbid`` — adding a field there breaks old nodes in
    a mixed-version rollout). Read locally on the diagnostics path only, where
    it powers leaked-wired detection (Skulk#239). psutil exposes ``wired`` on
    macOS only.
    """
    wired: object = getattr(psutil.virtual_memory(), "wired", None)
    return int(wired) if isinstance(wired, (int, float)) else None


class DiskUsage(CamelCaseModel):
    """Disk space usage for the models directory."""

    total: Memory
    available: Memory

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Get disk usage stats for the partition containing path."""
        total, _used, free = shutil.disk_usage(path)
        return cls(
            total=Memory.from_bytes(total),
            available=Memory.from_bytes(free),
        )


AcceleratorVendor = Literal["apple", "amd", "nvidia", "intel", "cpu", "unknown"]


class AcceleratorMetrics(CamelCaseModel):
    """One accelerator's live readings, normalized across collectors.

    The collector-agnostic GPU/accelerator expression: any platform's collector
    (mactop on Apple Silicon, rocm-smi/sysfs on AMD, nvidia-smi on CUDA) fills
    the same shape, so the planner and dashboard reason about a heterogeneous
    fleet uniformly. A field a given collector cannot measure stays ``None``
    (never a fake zero), so a reader can tell "0%" apart from "not reported".
    Units are fixed here so collectors normalize at their boundary:
    ``utilization_ratio`` is a 0..1 fraction, power is watts, temperature is
    degrees Celsius.
    """

    vendor: AcceleratorVendor = "unknown"
    name: str = "Unknown"
    utilization_ratio: float | None = None
    vram_total_bytes: int | None = None
    vram_used_bytes: int | None = None
    gtt_total_bytes: int | None = None
    """GPU-mappable host (GTT) memory, for unified-memory APUs (e.g. AMD Strix
    Halo). On such a node the GPU addresses system RAM beyond the BIOS VRAM
    carve-out through GTT, so the usable GPU pool is far larger than
    ``vram_total_bytes`` (placement uses this to admit big models on a UMA node).
    ``None`` on discrete GPUs / collectors that do not report it."""
    power_watts: float | None = None
    temperature_celsius: float | None = None
    clock_mhz: int | None = None
    compute_capability: str | None = None
    """Discrete-GPU compute capability as ``"<major>.<minor>"`` (NVIDIA SM level),
    e.g. ``"8.0"`` (A100 Ampere), ``"9.0"`` (H100 Hopper), ``"10.0"`` (B100/B200
    Blackwell), ``"12.0"`` (RTX 50 Blackwell). The engine/quant/placement decision
    keys on this, not on vendor: the same model+engine performs oppositely across
    generations (a benchmark showed vLLM's MXFP4 path losing single-stream on
    Ampere, which has no native FP4, but winning on Blackwell, which does).
    ``None`` on collectors that do not report it (AMD sysfs, Apple)."""
    native_fp4: bool | None = None
    """Whether the GPU accelerates FP4 natively (Blackwell sm100+, i.e. SM level
    (major, minor) >= (10, 0)). Derived at the collector boundary from the parsed
    compute capability (a numeric tuple compare, not a string compare). ``None``
    when unmeasured."""
    native_fp8: bool | None = None
    """Whether the GPU accelerates FP8 natively (Ada sm89 / Hopper sm90 and later).
    Derived at the collector boundary from the compute capability. ``None`` when
    unmeasured."""


class SystemPerformanceProfile(CamelCaseModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0
    # Collector-agnostic accelerator readings (None when unreported, e.g. a
    # management or CPU-only node, or a collector that cannot measure them).
    # The scalars above stay for back-compat with existing Mac-only readers;
    # cross-vendor readers use this block.
    accelerator: AcceleratorMetrics | None = None


InterfaceType = Literal["wifi", "ethernet", "maybe_ethernet", "thunderbolt", "unknown"]


class NetworkInterfaceInfo(CamelCaseModel):
    name: str
    ip_address: str
    interface_type: InterfaceType = "unknown"


class NodeIdentity(CamelCaseModel):
    """Static and slow-changing node identification data."""

    node_install_id: UUID4 | None = Field(
        default=None,
        description=(
            "Stable per-installation operator identity, independent of the current "
            "runtime libp2p node ID."
        ),
    )
    model_id: str = "Unknown"
    chip_id: str = "Unknown"
    friendly_name: str = "Unknown"
    os_version: str = "Unknown"
    os_build_version: str = "Unknown"
    skulk_version: str = "Unknown"
    skulk_commit: str = "Unknown"


NodeParticipation = Literal["full", "management", "ffn_only"]
"""How deeply a node participates in inference (Axis 1 of the heterogeneous-
participation model, #149/#286):

- ``full``: attention + FFN; an ordinary inference rank (today's default).
- ``management``: control plane only; sees the whole cluster and serves the
  API/dashboard, but the planner never assigns it an inference shard. The
  declared form of the ``excluded_nodes`` workaround (e.g. a remote node on a
  high-latency link).
- ``ffn_only``: reserved for LARQL slice placement (FFN/expert but not
  attention); not yet honored by the planner.
"""

NodeDataTransport = Literal["gossipsub", "zenoh"]
"""Resolved transport used for node-addressed DATA-plane traffic."""


class NodeResources(CamelCaseModel):
    """Inference capability, policy, and transport facts a node advertises.

    Mixes probed capability (``backends``) with operator-declared policy
    (``participation``) and the startup-resolved ``data_transport``; all ride the
    same node-info telemetry path. The planner reads capability, policy, and
    positive data-plane isolation evidence to hard-filter placement candidates,
    while cluster health also projects transport faults for operators. Defaults
    describe a normal Apple-Silicon
    full-participation node so pre-upgrade telemetry stays non-breaking.
    """

    backends: frozenset[str] = frozenset({"mlx"})
    engine_builds: dict[str, str] = Field(
        default_factory=dict,
        description="Exact installed build identities keyed by engine and backend tag.",
    )
    llama_server_settings: LlamaServerSettings | None = Field(
        default=None,
        description="Observed serving controls needed to budget per-slot recurrent state.",
    )
    hardware_classes: frozenset[str] = Field(
        default_factory=frozenset,
        description="Open observed hardware identifiers for support constraints.",
    )
    participation: NodeParticipation = "full"
    api_available: bool = True
    """Whether this process exposes the Skulk HTTP/WebSocket API. Defaults to
    true so mixed-version telemetry preserves the pre-existing all-nodes API
    assumption; nodes launched with ``--no-api`` advertise false explicitly."""
    data_transport: NodeDataTransport = "gossipsub"
    zenoh_connected_peers: int | None = None
    """Live Zenoh peer transports on this node's data-plane session, sampled at
    each advertisement. ``None`` when DATA rides gossipsub or during the
    post-startup grace window before isolation is trustworthy. Zero after that
    window, while other live nodes advertise Zenoh, means this node's data
    plane is isolated (its remote streams will fail even though the control
    plane is healthy); cluster health raises ``zenoh_isolated`` from it."""
    capability_conflicts: tuple[CapabilityConflict, ...] = ()
    """Loud observation-vs-declaration disagreements from backend derivation
    (#614): each entry names a way this node's serving capability is degraded
    or conflicted (silent CPU serving, degraded GPU detection, an unusable
    engine binary override), with its remediation. Cluster health maps these
    one-to-one onto ``nodeHealth`` reasons so the dashboard shows them."""

    @field_validator("backends", mode="before")
    @classmethod
    def _coerce_backends(cls, v: object) -> object:
        # Strict mode rejects a list where a frozenset is declared, but the
        # wire path (model_dump(mode="json") -> array -> model_validate) and
        # any list-shaped input arrive as a list. Coerce iterables to a
        # frozenset before strict validation so node_resources actually
        # populates over gossip (without this the feature is inert).
        if isinstance(v, (list, tuple, set, frozenset)):
            return frozenset(cast("Iterable[str]", v))
        return v

    @field_validator("hardware_classes", mode="before")
    @classmethod
    def _coerce_hardware_classes(cls, v: object) -> object:
        """Coerce JSON arrays into immutable hardware-class inventory."""
        if isinstance(v, (list, tuple, set, frozenset)):
            return frozenset(cast("Iterable[str]", v))
        return v

    @field_validator("capability_conflicts", mode="before")
    @classmethod
    def _coerce_capability_conflicts(cls, v: object) -> object:
        # Same wire-shape coercion as backends: JSON arrays arrive as lists,
        # and strict mode rejects a list where a tuple is declared.
        if isinstance(v, list):
            return tuple(cast("Iterable[object]", v))
        return v

    @field_serializer("backends")
    def _serialize_backends(self, value: frozenset[str]) -> list[str]:
        # Emit a sorted list in both json and python dump modes so JSON wire
        # encoding and TOML serialization (tomlkit cannot encode a frozenset)
        # both succeed and round-trip deterministically.
        return sorted(value)

    @field_serializer("hardware_classes")
    def _serialize_hardware_classes(self, value: frozenset[str]) -> list[str]:
        """Emit stable JSON for open hardware-class identifiers."""
        return sorted(value)

    @classmethod
    async def gather(
        cls,
        *,
        api_available: bool = True,
        data_transport: NodeDataTransport = "gossipsub",
        zenoh_connected_peers: int | None = None,
    ) -> "NodeResources":
        """Probe backends and read node policy plus the resolved DATA transport.

        Args:
            api_available: Whether this process exposes the API surface.
            data_transport: Transport already resolved during node startup. Passing
                the resolved value avoids reinterpreting environment configuration
                independently from the router that actually owns DATA delivery.
            zenoh_connected_peers: Live Zenoh peer transports sampled from the
                router that owns the session (``None`` when DATA rides gossipsub
                or while isolation is not yet trustworthy after startup).

        Returns:
            The capability and policy facts this node advertises to the fleet.
        """
        # Function-level import: the facts package imports shared type modules,
        # so a module-level import here would risk a cycle as facts grows.
        from skulk.facts import (
            current_backend_derivation,
            current_node_facts,
            engine_build_inventory,
            hardware_class_inventory,
        )

        derivation = current_backend_derivation()
        facts = current_node_facts()
        try:
            engine_builds = engine_build_inventory(derivation.backends, facts)
        except ValueError as error:
            from loguru import logger

            logger.error(f"engine build inventory is invalid: {error}")
            engine_builds = {}
        declared = os.environ.get("SKULK_NODE_PARTICIPATION", "full").strip().lower()
        participation: NodeParticipation = (
            declared if declared in ("full", "management", "ffn_only") else "full"
        )
        return cls(
            backends=derivation.backends,
            engine_builds=engine_builds,
            llama_server_settings=(
                resolve_llama_server_settings(os.environ)
                if any(
                    backend.startswith("llama_server")
                    for backend in derivation.backends
                )
                else None
            ),
            hardware_classes=hardware_class_inventory(facts),
            participation=participation,
            api_available=api_available,
            data_transport=data_transport,
            zenoh_connected_peers=zenoh_connected_peers,
            capability_conflicts=derivation.conflicts,
        )


class NodeNetworkInfo(CamelCaseModel):
    """Network interface information for a node."""

    interfaces: Sequence[NetworkInterfaceInfo] = []


class NodeThunderboltInfo(CamelCaseModel):
    """Thunderbolt interface identifiers for a node."""

    interfaces: Sequence[ThunderboltIdentifier] = []


class NodeRdmaCtlStatus(CamelCaseModel):
    """Whether RDMA is enabled on this node (via rdma_ctl)."""

    enabled: bool
    interfaces_present: bool = True


class ThunderboltBridgeStatus(CamelCaseModel):
    """Whether the Thunderbolt Bridge network service is enabled on this node."""

    enabled: bool
    exists: bool
    service_name: str | None = None
