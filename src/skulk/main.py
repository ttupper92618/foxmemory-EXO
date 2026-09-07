import argparse
import hashlib
import ipaddress
import multiprocessing as mp
import os
import resource
import signal
import socket
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Self, cast

import anyio
import psutil
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
    to_thread,
)
from loguru import logger
from pydantic import PositiveInt

import skulk.routing.topics as topics
from skulk.api.main import API
from skulk.connectivity.local_network import (
    check_local_network_access,
    local_network_denied_message,
)
from skulk.connectivity.tailscale import query_tailscale_status
from skulk.download.coordinator import DownloadCoordinator
from skulk.download.impl_shard_downloader import skulk_shard_downloader
from skulk.extensions import load_extensions
from skulk.master.main import Master
from skulk.operator.pairing import OperatorPairingService
from skulk.routing.event_router import EventRouter
from skulk.routing.router import Router, TelemetrySender, get_node_id_keypair
from skulk.routing.zenoh_status import ZenohPeerSampler
from skulk.shared.constants import SKULK_LOG
from skulk.shared.election import Election, ElectionResult
from skulk.shared.logging import (
    external_log_pipe_enabled,
    logger_cleanup,
    logger_setup,
)
from skulk.shared.models.model_cards import (
    get_all_model_cards,
    get_current_registry_cards,
    register_installed_card_record,
)
from skulk.shared.session_carryover import seed_state_for_new_session
from skulk.shared.types.artifact_inventory import (
    ARTIFACT_INVENTORY_DEBOUNCE_SECONDS,
    ARTIFACT_INVENTORY_ENTRY_LIMIT,
    ARTIFACT_INVENTORY_REFRESH_SECONDS,
    NodeArtifactAvailability,
    NodeArtifactInventory,
)
from skulk.shared.types.audio import RealtimeAudioInputFrame
from skulk.shared.types.commands import ForwarderDownloadCommand, SyncConfig
from skulk.shared.types.common import NodeId, SessionId, SystemId
from skulk.shared.types.events import (
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    ModelTrustApprovalChanged,
    NodeDownloadProgress,
    RunnerStatusUpdated,
    StagedModelEvicted,
    StateSnapshotHydrated,
)
from skulk.shared.types.profiling import NodeDataTransport, NodeResources
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.shared.types.telemetry import NodeTelemetry, TelemetryView
from skulk.startup_recovery import preflight_api_port
from skulk.store.artifact_inventory import (
    installed_artifact_roots,
    inventory_installed_artifacts,
)
from skulk.store.config import (
    SkulkConfig,
    load_skulk_config,
    node_matches_store_host,
    persist_model_trust_config,
    resolve_config_path,
    resolve_node_staging,
)
from skulk.store.installed_cards import VerifiedDetachedInstalledCardCache
from skulk.store.model_store import ModelStore
from skulk.store.model_store_client import ModelStoreClient, ModelStoreDownloader
from skulk.store.model_store_server import ModelStoreServer
from skulk.utils.channels import Receiver, Sender, channel
from skulk.utils.info_gatherer.info_gatherer import NodeCapabilities
from skulk.utils.pydantic_ext import CamelCaseModel
from skulk.utils.task_group import TaskGroup
from skulk.worker.main import Worker


def _derive_zenoh_namespace(raw: str) -> str:
    """Map a libp2p namespace to a Zenoh key-expr namespace segment (#308).

    This is the Zenoh data-plane isolation boundary, so distinct libp2p
    namespaces must not collide on the same Zenoh namespace, or peers on
    different libp2p namespaces could read each other's ``data``. We SHA-256-hash
    unconditionally rather than a verbatim/hash split: a char-replacement
    sanitizer collapses ``prod/main`` and ``prod_main`` (#312 review P1), and a
    verbatim-when-safe split still lets a fleet named literally
    ``ns<sha256(victim)>`` collide with the victim's hashed namespace (#312 review
    P2). A SHA-256 hex digest is collision-resistant (no two distinct namespaces
    collide in practice) and always a valid key-expr segment; the ``ns`` prefix
    keeps it from starting with a digit. The trade-off is a non-human-readable
    namespace, which is fine for an internal key prefix. Note: neither this
    derived namespace nor the raw libp2p token is ever logged (with no TLS the
    namespace is itself the isolation value); startup logs only a non-routing
    fingerprint of it (see ``_namespace_fingerprint``).
    """
    return "ns" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Keep in sync with rust/networking/src/swarm.rs: NETWORK_VERSION and the
# OVERRIDE_VERSION_ENV_VAR name used to build the libp2p private-network key
# (a lockstep test in tests/test_zenoh_namespace_lockstep.py parses the Rust
# source and fails on drift).
_LIBP2P_NETWORK_VERSION = "v0.0.2"
_LIBP2P_NAMESPACE_ENV_VAR = "SKULK_LIBP2P_NAMESPACE"
_NODE_RESOURCES_POLL_INTERVAL_SECONDS = 2.0
_CLUSTER_CONFIG_SYNC_ATTEMPTS = 30
_CLUSTER_CONFIG_SYNC_RESPONSE_TIMEOUT_SECONDS = 1.0
_CLUSTER_CONFIG_SYNC_RETRY_INTERVAL_SECONDS = 0.2
# Cadence of the local zenoh-isolation check, and the floor between repeated
# operator warnings while the condition persists. The check is a cheap local
# session introspection; the warning floor keeps a permanently isolated node
# from flooding its log.
_ZENOH_ISOLATION_CHECK_INTERVAL_SECONDS = 30.0
_ZENOH_ISOLATION_WARNING_INTERVAL_SECONDS = 300.0


async def _publish_management_node_resources(
    node_id: NodeId,
    api_available: bool,
    data_transport: NodeDataTransport,
    telemetry_sender: TelemetrySender | Sender[NodeTelemetry],
    zenoh_peer_sampler: "ZenohPeerSampler | None" = None,
    poll_interval: float = _NODE_RESOURCES_POLL_INTERVAL_SECONDS,
    capabilities_provider: Callable[[], frozenset[str]] | None = None,
) -> None:
    """Advertise resource truth for a node started without a worker.

    A management/API-only node still owns DATA receivers and must participate in
    fleet transport consistency checks. It advertises no inference backends and
    effective management-only participation so the same reading cannot make it
    eligible for placement.

    Args:
        node_id: Stable identity attached to the telemetry reading.
        api_available: Whether this management process exposes the API surface.
        data_transport: DATA transport already resolved during node startup.
        telemetry_sender: Existing latest-value telemetry admission handle.
        zenoh_peer_sampler: Live data-plane connectivity sampler; a
            management node owns DATA receivers, so its isolation matters to
            the fleet exactly like a worker's. ``None`` advertises unknown.
        poll_interval: Seconds between repeated advertisements for late joiners
            and fallback liveness. The default matches the worker heartbeat
            cadence and stays below the node-health warning threshold.
        capabilities_provider: Cached extension tags, including withdrawals.
            No provider means an empty reading; capability service never grants
            this host inference placement.

    Side effects:
        Publishes one immediate and then periodic ``NodeResources`` reading until
        the owning task is cancelled or telemetry admission closes.
    """
    while True:
        try:
            resources = NodeResources(
                backends=frozenset(),
                participation="management",
                api_available=api_available,
                data_transport=data_transport,
                zenoh_connected_peers=(
                    await zenoh_peer_sampler.advertised_count()
                    if zenoh_peer_sampler is not None
                    else None
                ),
            )
            await telemetry_sender.send(NodeTelemetry(node_id=node_id, info=resources))
            await telemetry_sender.send(
                NodeTelemetry(
                    node_id=node_id,
                    info=NodeCapabilities(
                        capabilities=(
                            capabilities_provider()
                            if capabilities_provider is not None
                            else frozenset()
                        )
                    ),
                )
            )
        except (ClosedResourceError, BrokenResourceError):
            return
        except Exception as error:
            logger.warning(
                "Management-node resource advertisement failed: "
                f"{type(error).__name__}: {error}"
            )
        await anyio.sleep(poll_interval)


def _libp2p_namespace_token(environ: Mapping[str, str]) -> str:
    """Return the exact token libp2p isolates on, for the Zenoh namespace (#312).

    The Zenoh namespace MUST derive from the identical token that builds the
    libp2p private-network key in ``swarm.rs`` (``PNET_PRESHARED_KEY``); otherwise
    two nodes in the same libp2p cluster can land in different Zenoh namespaces
    and silently drop all cross-node generation output. Since #659, ``swarm.rs``
    ALWAYS feeds ``NETWORK_VERSION`` into the key and layers
    ``SKULK_LIBP2P_NAMESPACE`` on top when the var is *present* (Rust
    ``env::var`` returns ``Ok`` even for an empty value). We mirror that
    precisely: the token is the version alone when the var is unset, and the
    version concatenated with the namespace when it is present, so a version
    bump re-keys BOTH transports together on every deployment shape.
    """
    override = environ.get(_LIBP2P_NAMESPACE_ENV_VAR)
    if override is not None:
        # NUL-delimited to keep (version, namespace) pairs injective in the
        # token, mirroring the Rust key derivation's delimiter (#659 review).
        return _LIBP2P_NETWORK_VERSION + "\0" + override
    return _LIBP2P_NETWORK_VERSION


def _namespace_fingerprint(namespace: str) -> str:
    """Return a short non-routing fingerprint of a Zenoh namespace (#312 review).

    With no transport auth/TLS the namespace prefix is itself the isolation
    value: a peer that learns it can subscribe to the fully prefixed key and read
    ``data``. So startup logging emits this fingerprint instead of the namespace.
    It is a truncated second hash, so it cannot be used to subscribe and cannot be
    reversed to the namespace, yet it is stable per namespace, which is all an
    operator needs to confirm two nodes resolved to the same isolation segment.
    """
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:12]


_DEFAULT_ZENOH_PORT: Final = 7447
_PRIVATE_LAN_IPV4_NETWORKS: Final[tuple[ipaddress.IPv4Network, ...]] = (
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
)
_CGNAT_IPV4_NETWORK: Final = ipaddress.IPv4Network("100.64.0.0/10")


def _is_trusted_fabric_ipv4(address: str) -> bool:
    """Return whether an IPv4 address belongs to an auto-bind-safe fabric.

    Automatic Zenoh listeners may use conventional private LAN or CGNAT
    overlay addresses. Public addresses require an explicit operator-supplied
    listener because the default Zenoh session has no transport authentication
    or TLS.
    """
    try:
        ip = ipaddress.IPv4Address(address)
    except ipaddress.AddressValueError:
        return False
    return (
        any(ip in network for network in _PRIVATE_LAN_IPV4_NETWORKS)
        or ip in _CGNAT_IPV4_NETWORK
    )


def _resolve_zenoh_listen(env_value: str) -> str:
    """Return the configured Zenoh listener or a safe zero-config default.

    Fresh installs use Zenoh just like the qualification fleet, but they must
    not silently expose an unauthenticated listener on every interface. When no
    override is present, bind the best private-LAN or CGNAT fabric address
    selected by the model-store interface policy. An offline or public-only
    single-node install falls back to loopback and remains functional; binding
    a public address requires an explicit override.
    """
    listen = env_value.strip()
    if listen:
        return listen
    candidate = _routable_local_ipv4()
    host = (
        candidate if candidate and _is_trusted_fabric_ipv4(candidate) else "127.0.0.1"
    )
    return f"tcp/{host}:{_DEFAULT_ZENOH_PORT}"


def _resolve_zenoh_enabled(data_plane_env: str, listen_env: str) -> bool:
    """Resolve whether the Zenoh DATA plane is enabled.

    The shipping default is Zenoh, including on a zero-config fresh install:

    - Explicit truthy (``1``/``true``/``yes``/``on``) -> enabled.
    - Explicit falsy (``0``/``false``/``no``/``off``) -> disabled (gossipsub).
    - Unset/blank -> enabled. The listener is selected safely by
      :func:`_resolve_zenoh_listen`.
    - Any other non-empty value -> ``ValueError``. An unrecognized value
      (a typo, or a boolean spelling we don't accept) must NOT silently fall
      through to the default; refuse to guess the transport.

    ``listen_env`` remains in the internal signature for callers and tests
    written against the former soft-default resolver. Listener presence no
    longer controls transport selection.
    """
    del listen_env
    value = data_plane_env.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    if value:
        raise ValueError(
            f"SKULK_ZENOH_DATA_PLANE={data_plane_env!r} is not a recognized "
            "boolean. Use 1/true/yes/on or 0/false/no/off, or leave it unset "
            "to use the Zenoh default. Refusing to guess the DATA transport."
        )
    return True


def _add_model_search_path(path: Path) -> None:
    """Ensure the given model path is visible to the current process and children."""

    expanded = path.expanduser()
    existing_path = os.environ.get(
        "SKULK_MODELS_PATH", os.environ.get("SKULK_MODELS_PATH", "")
    )
    paths = [p for p in existing_path.split(":") if p]
    path_str = str(expanded)
    if path_str not in paths:
        paths.append(path_str)
    joined = ":".join(paths)
    os.environ["SKULK_MODELS_PATH"] = joined
    os.environ["SKULK_MODELS_PATH"] = joined  # legacy compat

    from skulk.shared.constants import add_model_search_path

    add_model_search_path(expanded)


_VIRTUAL_IFACE_PREFIXES: Final = (
    "docker",
    "br-",
    "virbr",
    "vmnet",
    "vboxnet",
    "veth",
    "cni",
    "flannel",
    "kube",
)


def _is_virtual_iface(name: str) -> bool:
    """Whether an interface name looks like a Docker/VM/container bridge.

    These carry RFC1918 addresses (e.g. Docker's ``172.17.0.1``) that are not
    reachable from peers on the real LAN, so they must not be advertised as the
    store host. VPN tunnels (Tailscale ``utun``/``tailscale0``) are deliberately
    NOT excluded: they are a valid fallback path and are already ranked below the
    LAN address.
    """
    lowered = name.lower()
    return lowered.startswith(_VIRTUAL_IFACE_PREFIXES)


def _routable_local_ipv4() -> str | None:
    """Return this node's best peer-reachable IPv4 address.

    Virtual, loopback, link-local, and unspecified interfaces are excluded.
    Prefer a conventional private LAN, then Tailscale's CGNAT range, then any
    remaining routable IPv4. The policy is shared by zero-config Zenoh binding
    and model-store advertisement so both planes choose the same kind of
    peer-reachable interface.
    """
    routable: list[str] = []
    for iface_name, addresses in psutil.net_if_addrs().items():
        if _is_virtual_iface(iface_name):
            continue
        for address in addresses:
            if address.family != socket.AF_INET:
                continue
            try:
                ip = ipaddress.ip_address(address.address)
            except ValueError:
                continue
            if ip.is_loopback or ip.is_link_local or ip.is_unspecified:
                continue
            routable.append(address.address)

    def _rank(address: str) -> int:
        ip = ipaddress.IPv4Address(address)
        if any(ip in network for network in _PRIVATE_LAN_IPV4_NETWORKS):
            return 0
        if ip in _CGNAT_IPV4_NETWORK:
            return 1
        return 2

    routable.sort(key=lambda address: (_rank(address), address))
    return routable[0] if routable else None


def _routable_store_advertise_host(
    configured: str | None, hostname_fallback: str
) -> str:
    """Pick an address other nodes can actually reach the model store host at.

    The store host broadcasts this as ``store_http_host`` so workers build the
    download URL ``http://<host>:<port>``. A bare hostname (``kite3.local``) is
    fragile on a Thunderbolt-meshed fleet: mDNS can resolve it to the host's
    link-local TB address (``169.254.x``), which a peer lacking a direct TB link
    cannot route to, so its downloads fail while the LAN path works fine.

    An operator-supplied **routable IP literal** is honored as-is. Anything else
    (a hostname, or a loopback/link-local literal) is replaced with this host's
    own best routable IPv4: a private LAN address (RFC1918) is preferred over any
    other routable address, and loopback / link-local / unspecified addresses are
    skipped. Falls back to the hostname only when no routable IPv4 is found.
    """
    if configured:
        try:
            literal = ipaddress.ip_address(configured)
        except ValueError:
            literal = None  # a hostname, not an IP -> recompute below
        # Honor only a routable IPv4 literal: the store URL is built as
        # http://{host}:{port} with no IPv6-bracket handling, so an IPv6 literal
        # would produce an invalid URL -> treat it like a hostname and recompute.
        if (
            literal is not None
            and literal.version == 4
            and not (
                literal.is_loopback or literal.is_link_local or literal.is_unspecified
            )
        ):
            return configured

    return _routable_local_ipv4() or hostname_fallback


def _configure_model_store_runtime(
    node_id: NodeId,
    skulk_config: SkulkConfig | None,
) -> tuple[ModelStoreClient | None, ModelStoreServer | None]:
    """Build store client/server wiring from the current config."""

    if (
        skulk_config is None
        or skulk_config.model_store is None
        or not skulk_config.model_store.enabled
    ):
        return None, None

    ms = skulk_config.model_store
    is_store_host = node_matches_store_host(
        ms.store_host,
        str(node_id),
        hostname=socket.gethostname(),
    )

    local_store_path: Path | None = Path(ms.store_path) if is_store_host else None
    store_client = ModelStoreClient(
        store_host=ms.store_http_host or ms.store_host,
        store_port=ms.store_port,
        local_store_path=local_store_path,
    )

    store_server: ModelStoreServer | None = None
    if is_store_host:
        model_store = ModelStore(Path(ms.store_path))
        store_server = ModelStoreServer(model_store, port=ms.store_port)
        logger.info(
            f"ModelStore: this node is the store host — "
            f"store at {ms.store_path}, server on port {ms.store_port}"
        )

    staging_cfg = resolve_node_staging(ms, str(node_id))
    staging_path = Path(staging_cfg.node_cache_path)
    _add_model_search_path(staging_path)
    logger.info(
        f"ModelStore: added staging path {staging_path.expanduser()} to SKULK_MODELS_PATH"
    )

    if is_store_host:
        store_root = Path(ms.store_path)
        _add_model_search_path(store_root)
        logger.info(
            f"ModelStore: store host — added store root {store_root.expanduser()} to SKULK_MODELS_PATH (skip staging)"
        )

    return store_client, store_server


def _state_sync_store_http_host(
    node_id: NodeId,
    skulk_config: SkulkConfig | None,
) -> str | None:
    """Return this store host's routable address for cluster bootstrap."""

    if (
        skulk_config is None
        or skulk_config.model_store is None
        or not skulk_config.model_store.enabled
    ):
        return None
    model_store = skulk_config.model_store
    local_hostname = socket.gethostname()
    if not node_matches_store_host(
        model_store.store_host,
        str(node_id),
        hostname=local_hostname,
    ):
        return None
    return _routable_store_advertise_host(
        model_store.store_http_host,
        local_hostname,
    )


def merge_cluster_config_bootstrap(
    config_yaml: str,
    config_path: Path,
) -> dict[str, object]:
    """Merge an authoritative bootstrap config into the local file.

    A payload carrying an ``hf_token`` is adopted, persisted mode ``0o600``,
    and promoted into ``HF_TOKEN`` when that variable is unset, so a freshly
    joined node's downloads authenticate without a restart. Absent-or-blank
    incoming tokens never erase a locally configured one, and node-local
    deprecated ``model_trust`` compatibility state is preserved.

    Args:
        config_yaml: The master's serialized bootstrap configuration.
        config_path: Destination ``skulk.yaml``.

    Returns:
        The merged configuration mapping as persisted.
    """

    import yaml as yaml_module

    from skulk.store.config import update_skulk_config_atomic

    decoded: object = cast(object, yaml_module.safe_load(config_yaml))
    if not isinstance(decoded, dict):
        # A malformed payload must degrade to "keep local config" in full:
        # merging an empty mapping here would wipe every local field except
        # the explicitly preserved ones. The identity update returns the
        # existing config untouched (and still stamps the 0600 mode). The
        # trusted fabric makes this a bug signal, not an attack surface, so
        # a warning is the right volume.
        if decoded is not None:
            logger.warning("Ignoring non-mapping cluster bootstrap config payload")
        return update_skulk_config_atomic(config_path, lambda existing: existing)
    received = cast("dict[str, object]", decoded)

    from skulk.store.config import normalized_hf_token, promote_hf_token

    def preserve_local_fields(
        existing: dict[str, object],
    ) -> dict[str, object]:
        updated = dict(received)
        if normalized_hf_token(updated.get("hf_token")) is None and existing.get(
            "hf_token"
        ):
            updated["hf_token"] = existing["hf_token"]
        if "model_trust" not in updated and "model_trust" in existing:
            updated["model_trust"] = existing["model_trust"]
        return updated

    merged = update_skulk_config_atomic(config_path, preserve_local_fields)
    _ = promote_hf_token(merged.get("hf_token"), source="cluster config bootstrap")
    return merged


@dataclass
class Node:
    router: Router
    event_router: EventRouter
    download_coordinator: DownloadCoordinator | None
    worker: Worker | None
    election: Election  # Every node participates in election, as we do want a node to become master even if it isn't a master candidate if no master candidates are present.
    election_result_receiver: Receiver[ElectionResult]
    master: Master | None
    api: API | None

    node_id: NodeId
    offline: bool
    skulk_config: SkulkConfig | None
    store_client: ModelStoreClient | None
    store_server: ModelStoreServer | None
    # Live node telemetry off the event log (#279). Node-owned so it survives
    # master re-election; the subscriber feeds it, the master/API read it.
    telemetry_view: TelemetryView
    telemetry_receiver: Receiver[NodeTelemetry]
    # A node-level event tap drives cache rescans even when this process was
    # intentionally launched without an HTTP API.
    artifact_inventory_event_receiver: Receiver[IndexedEvent] | None = None
    data_plane_zenoh: bool = False
    # Samples the router's live Zenoh peer-transport count for NodeResources
    # advertisement and the local isolation warning. None only in tests that
    # construct Node without create().
    zenoh_peer_sampler: ZenohPeerSampler | None = None
    _tg: TaskGroup = field(init=False, default_factory=TaskGroup)
    _artifact_inventory_trigger_sender: Sender[None] = field(init=False)
    _artifact_inventory_trigger_receiver: Receiver[None] = field(init=False)
    _artifact_inventory_detached_cache: VerifiedDetachedInstalledCardCache = field(
        init=False,
        default_factory=VerifiedDetachedInstalledCardCache,
    )

    def __post_init__(self) -> None:
        """Create the bounded coalescing trigger used by artifact rescans."""

        (
            self._artifact_inventory_trigger_sender,
            self._artifact_inventory_trigger_receiver,
        ) = channel[None](1)

    @classmethod
    async def create(cls, args: "Args") -> Self:
        keypair = get_node_id_keypair()
        node_id = NodeId(keypair.to_node_id())
        session_id = SessionId(master_node_id=node_id, election_clock=0)
        # Zenoh data plane (#279 follow-on). Node-addressed model output,
        # provider media, and realtime PCM ingress ride a Zenoh peer session;
        # all other planes stay on libp2p. Endpoint overrides are per-node, so
        # they come from the environment rather than gossip-synced config.
        # Zenoh is the shipping default, including for a fresh zero-config node.
        # SKULK_ZENOH_DATA_PLANE can force it off; listen/connect overrides
        # remain available for routed or locked-down deployments. See
        # _resolve_zenoh_enabled.
        _zenoh_data_plane_env = os.environ.get("SKULK_ZENOH_DATA_PLANE", "")
        _zenoh_listen_env = os.environ.get("SKULK_ZENOH_LISTEN", "")
        _zenoh_on = _resolve_zenoh_enabled(_zenoh_data_plane_env, _zenoh_listen_env)
        _zenoh_connect = [
            endpoint.strip()
            for endpoint in os.environ.get("SKULK_ZENOH_CONNECT", "").split(",")
            if endpoint.strip()
        ]
        _zenoh_listen_endpoints: list[str] | None = None
        _zenoh_namespace: str | None = None
        if _zenoh_on:
            _zenoh_listen = _resolve_zenoh_listen(_zenoh_listen_env)
            _zenoh_listen_endpoints = [_zenoh_listen]
            # Namespace isolation (#308): Zenoh transparently prefixes all keys
            # with this segment, so a peer on a different namespace cannot read
            # this fleet's `data`. Derive it from the EXACT token libp2p isolates
            # on (_libp2p_namespace_token mirrors swarm.rs), via a
            # collision-resistant SHA-256 hash (see _derive_zenoh_namespace). If
            # the source diverged from libp2p (legacy env, different default),
            # two nodes in one libp2p cluster could land in different Zenoh
            # namespaces and silently drop all cross-node output (#312 review).
            # We never log the raw token or the derived namespace: the raw token
            # seeds libp2p's private-network PSK (swarm.rs PNET_PRESHARED_KEY), and
            # because the plane has no transport auth/TLS the derived namespace IS
            # the only isolation value (a peer that learns it can subscribe to the
            # prefixed key and read `data`). Log only a non-routing fingerprint and
            # whether an override was set, so operators can still confirm two nodes
            # share a namespace without exposing it (#312 review).
            _ns_raw = _libp2p_namespace_token(os.environ)
            _zenoh_namespace = _derive_zenoh_namespace(_ns_raw)
            _ns_override_set = _LIBP2P_NAMESPACE_ENV_VAR in os.environ
            if "0.0.0.0" in _zenoh_listen:
                logger.warning(
                    f"SKULK_ZENOH_LISTEN={_zenoh_listen} binds all interfaces; "
                    f"prefer a specific private IP on a shared network (#308)."
                )
            logger.warning(
                f"Zenoh DATA plane ENABLED: model and provider media "
                f"use Zenoh on {_zenoh_listen}, namespace"
                f"-isolated (fingerprint {_namespace_fingerprint(_zenoh_namespace)}; "
                f"{_LIBP2P_NAMESPACE_ENV_VAR} "
                f"{'set' if _ns_override_set else 'unset, using default'}). There "
                f"is still NO transport auth/TLS, so on an untrusted network "
                f"enable Zenoh TLS or keep it firewalled (#308)."
            )
        router = Router.create(
            keypair,
            bootstrap_peers=args.bootstrap_peers,
            listen_port=args.libp2p_port,
            zenoh_listen_endpoints=_zenoh_listen_endpoints,
            zenoh_connect_endpoints=_zenoh_connect,
            node_id=str(node_id),
            zenoh_namespace=_zenoh_namespace,
            zenoh_multicast_scouting=not _zenoh_connect,
        )
        # Data-plane connectivity ground truth (zenoh isolation visibility):
        # sampled at every NodeResources advertisement and by the local
        # isolation monitor. Created unconditionally; it reports None (unknown)
        # when DATA rides gossipsub.
        zenoh_peer_sampler = ZenohPeerSampler(router.zenoh_connected_peer_count)
        await router.register_topic(topics.GLOBAL_EVENTS)
        await router.register_topic(topics.LOCAL_EVENTS)
        await router.register_topic(topics.COMMANDS)
        await router.register_topic(topics.ELECTION_MESSAGES)
        await router.register_topic(topics.AUTHORITY_MESSAGES)
        await router.register_topic(topics.CONNECTION_MESSAGES)
        await router.register_topic(topics.DOWNLOAD_COMMANDS)
        await router.register_topic(topics.STATE_SYNC_MESSAGES)
        await router.register_topic(topics.TELEMETRY)
        await router.register_topic(topics.DATA)
        await router.register_topic(topics.PROVIDER_DATA)
        await router.register_topic(topics.REALTIME_AUDIO)
        await router.register_topic(topics.SPEECH_MEDIA)
        await router.register_topic(topics.TRACE_DATA)
        await router.register_topic(topics.VISION_MEDIA)
        telemetry_view = TelemetryView()
        realtime_audio_sender, realtime_audio_receiver = channel[
            RealtimeAudioInputFrame
        ](64)
        event_router = EventRouter(
            node_id,
            session_id,
            command_sender=router.sender(topics.COMMANDS),
            state_sync_sender=router.sender(topics.STATE_SYNC_MESSAGES),
            state_sync_receiver=router.receiver_with_origin(topics.STATE_SYNC_MESSAGES),
            external_outbound=router.sender(topics.LOCAL_EVENTS),
            external_inbound=router.receiver(topics.GLOBAL_EVENTS),
        )

        logger.info(f"Starting node {node_id}")

        # Load skulk.yaml (returns None if absent, for zero-config compatibility:
        # when skulk.yaml is missing, all store references stay None and the
        # node behaves identically to the zero-config default).
        skulk_config = load_skulk_config()

        # Track whether user provided the KV backend env var at launch —
        # if so, config syncs must not overwrite it.
        _user_set_kv_backend = "SKULK_KV_CACHE_BACKEND" in os.environ
        os.environ["_SKULK_KV_BACKEND_USER_SET"] = "1" if _user_set_kv_backend else ""

        # Apply inference config to env var so runner subprocesses inherit it.
        # Env var takes precedence if user set it at launch.
        if (
            skulk_config is not None
            and skulk_config.inference is not None
            and not _user_set_kv_backend
        ):
            os.environ["SKULK_KV_CACHE_BACKEND"] = (
                skulk_config.inference.kv_cache_backend
            )
            logger.info(
                f"Inference config: kv_cache_backend={skulk_config.inference.kv_cache_backend}"
            )

        # Track whether the operator supplied HF_TOKEN at launch (directly or
        # via the service env file). If so, config syncs must never replace
        # it; a value merely promoted from skulk.yaml below may be replaced
        # by a newer fleet token so rotation converges without restarts.
        # An inherited marker is trusted rather than recomputed: an in-place
        # restart (os.execv) carries the previous process's environment, so a
        # config-promoted HF_TOKEN would otherwise look operator-supplied
        # after every /admin/restart and block rotation forever (#922 review).
        from skulk.store.config import (
            promote_hf_token,
            stamp_hf_token_provenance,
        )

        stamp_hf_token_provenance()
        if skulk_config is not None:
            _ = promote_hf_token(skulk_config.hf_token, source="local config")

        store_client, store_server = _configure_model_store_runtime(
            node_id, skulk_config
        )

        # Create DownloadCoordinator (unless --no-downloads)
        if not args.no_downloads:
            base_downloader = skulk_shard_downloader(offline=args.offline)
            if (
                skulk_config is not None
                and skulk_config.model_store is not None
                and skulk_config.model_store.enabled
                and store_client is not None
            ):
                ms = skulk_config.model_store
                staging_cfg = resolve_node_staging(ms, str(node_id))
                shard_downloader = ModelStoreDownloader(
                    inner=base_downloader,
                    store_client=store_client,
                    staging_config=staging_cfg,
                    allow_hf_fallback=ms.download.allow_hf_fallback,
                    installed_card_callback=register_installed_card_record,
                )
            else:
                shard_downloader = base_downloader

            coordinator_staging_path = (
                Path(
                    resolve_node_staging(
                        skulk_config.model_store, str(node_id)
                    ).node_cache_path
                )
                if skulk_config is not None
                and skulk_config.model_store is not None
                and skulk_config.model_store.enabled
                else None
            )
            download_coordinator = DownloadCoordinator(
                node_id,
                shard_downloader,
                event_sender=event_router.sender(),
                telemetry_sender=router.telemetry_sender(),
                download_command_receiver=router.receiver(topics.DOWNLOAD_COMMANDS),
                offline=args.offline,
                staging_cache_path=coordinator_staging_path,
            )
        else:
            download_coordinator = None

        if args.spawn_api:
            api = API(
                node_id,
                port=args.api_port,
                event_receiver=event_router.receiver(),
                command_sender=router.sender(topics.COMMANDS),
                download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
                election_receiver=router.receiver(topics.ELECTION_MESSAGES),
                skulk_config=skulk_config,
                store_client=store_client,
                telemetry_view=telemetry_view,
                telemetry_sender=router.telemetry_sender(),
                data_receiver=router.receiver(topics.DATA),
                provider_stream_sender=router.sender(topics.PROVIDER_DATA),
                provider_stream_receiver=router.receiver(topics.PROVIDER_DATA),
                realtime_audio_packet_sender=router.sender(topics.REALTIME_AUDIO),
                realtime_audio_packet_receiver=router.receiver(topics.REALTIME_AUDIO),
                speech_media_packet_sender=router.sender(topics.SPEECH_MEDIA),
                speech_media_packet_receiver=router.receiver(topics.SPEECH_MEDIA),
                trace_data_receiver=router.receiver(topics.TRACE_DATA),
                vision_media_packet_sender=router.sender(topics.VISION_MEDIA),
                vision_media_packet_receiver=router.receiver(topics.VISION_MEDIA),
                realtime_audio_sender=(
                    None if args.no_worker else realtime_audio_sender
                ),
                data_plane_zenoh=_zenoh_on,
                data_plane_egress_provider=router.data_plane_egress_diagnostics,
                vision_media_egress_provider=(router.vision_media_egress_diagnostics),
                telemetry_plane_provider=router.telemetry_plane_diagnostics,
                # Installed plugins (skulk.extensions entry points), discovered
                # once per process. First-party provider facades are registered
                # by the API and delegate to the existing core runtimes.
                extensions=load_extensions(),
                enable_builtin_providers=True,
                operator_pairing_service=OperatorPairingService.from_default_paths(),
                apply_custom_card_mutations_locally=args.no_worker,
            )
        else:
            api = None

        if download_coordinator is not None and api is not None:
            download_coordinator.config_applied_callback = (
                api.refresh_config_dependent_capabilities
            )

        if not args.no_worker:
            worker_store_client: ModelStoreClient | None = store_client
            if (
                skulk_config is not None
                and skulk_config.model_store is not None
                and skulk_config.model_store.enabled
            ):
                worker_staging_cfg = resolve_node_staging(
                    skulk_config.model_store, str(node_id)
                )
            else:
                worker_staging_cfg = None
            worker = Worker(
                node_id,
                event_receiver=event_router.receiver(),
                event_sender=event_router.sender(),
                command_sender=router.sender(topics.COMMANDS),
                download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
                telemetry_sender=router.telemetry_sender(),
                telemetry_view=telemetry_view,
                api_available=args.spawn_api,
                data_transport="zenoh" if _zenoh_on else "gossipsub",
                zenoh_peer_sampler=zenoh_peer_sampler,
                data_sender=router.sender(topics.DATA),
                trace_data_sender=router.sender(topics.TRACE_DATA),
                realtime_audio_receiver=realtime_audio_receiver,
                realtime_audio_packet_receiver=router.receiver(topics.REALTIME_AUDIO),
                speech_media_packet_receiver=router.receiver(topics.SPEECH_MEDIA),
                vision_media_packet_sender=router.sender(topics.VISION_MEDIA),
                vision_media_packet_receiver=router.receiver(topics.VISION_MEDIA),
                connection_message_receiver=router.receiver(topics.CONNECTION_MESSAGES),
                session_connection_snapshot=router.current_session_connections,
                store_client=worker_store_client,
                staging_config=worker_staging_cfg,
            )
            if download_coordinator is not None and isinstance(
                download_coordinator.shard_downloader,
                ModelStoreDownloader,
            ):
                download_coordinator.shard_downloader.set_staging_capacity_callback(
                    worker.prepare_staging_transfer
                )
            if api is not None:
                api.set_runner_diagnostics_provider(worker.collect_runner_diagnostics)
                api.set_runner_cancel_provider(worker.cancel_runner_task)
                api.set_vision_media_ingress_provider(
                    worker.collect_vision_media_ingress_diagnostics
                )
        else:
            worker = None

        # We start every node with a master
        master = Master(
            node_id,
            session_id,
            event_sender=event_router.sender(),
            global_event_sender=router.sender(topics.GLOBAL_EVENTS),
            local_event_receiver=router.receiver(topics.LOCAL_EVENTS),
            command_receiver=router.receiver(topics.COMMANDS),
            state_sync_receiver=router.receiver(topics.STATE_SYNC_MESSAGES),
            state_sync_sender=router.sender(topics.STATE_SYNC_MESSAGES),
            download_command_sender=router.sender(topics.DOWNLOAD_COMMANDS),
            telemetry_view=telemetry_view,
            state_sync_store_http_host=_state_sync_store_http_host(
                node_id,
                skulk_config,
            ),
            initial_model_trust_identities=(
                tuple(skulk_config.model_trust.approved_remote_code_identities)
                if skulk_config is not None and skulk_config.model_trust is not None
                else ()
            ),
        )

        er_send, er_recv = channel[ElectionResult]()
        election = Election(
            node_id,
            # If someone manages to assemble 1 MILLION devices into a Skulk cluster then. well done. good job champ.
            seniority=1_000_000 if args.force_master else 0,
            # nb: this DOES feedback right now. i have thoughts on how to address this,
            # but ultimately it seems not worth the complexity
            election_message_sender=router.sender(topics.ELECTION_MESSAGES),
            election_message_receiver=router.receiver(topics.ELECTION_MESSAGES),
            connection_message_receiver=router.receiver(topics.CONNECTION_MESSAGES),
            command_receiver=router.receiver(topics.COMMANDS),
            election_result_sender=er_send,
        )

        return cls(
            router,
            event_router,
            download_coordinator,
            worker,
            election,
            er_recv,
            master,
            api,
            node_id,
            args.offline,
            skulk_config,
            store_client,
            store_server,
            telemetry_view,
            router.receiver(topics.TELEMETRY),
            event_router.receiver(),
            _zenoh_on,
            zenoh_peer_sampler,
        )

    async def run(self):
        # Command ownership and persistence collision checks are synchronous.
        # Load durable card truth before any API or master task can accept a
        # mutation, including on nodes that do not host the model store.
        await get_all_model_cards()
        if self.store_server is not None:
            await self.store_server.refresh_recovered_generations(
                get_current_registry_cards(),
            )
        async with self._tg as tg:
            signal.signal(signal.SIGINT, lambda _, __: self.shutdown())
            signal.signal(signal.SIGTERM, lambda _, __: self.shutdown())
            tg.start_soon(self.router.run)
            tg.start_soon(self.event_router.run)
            tg.start_soon(self.election.run)
            tg.start_soon(self._run_telemetry)
            tg.start_soon(self._artifact_inventory_loop)
            if self.artifact_inventory_event_receiver is not None:
                tg.start_soon(self._observe_artifact_inventory_events)
            if self.worker is None:
                tg.start_soon(
                    _publish_management_node_resources,
                    self.node_id,
                    self.api is not None,
                    "zenoh" if self.data_plane_zenoh else "gossipsub",
                    self.router.telemetry_sender(),
                    self.zenoh_peer_sampler,
                    _NODE_RESOURCES_POLL_INTERVAL_SECONDS,
                    lambda: frozenset(
                        self.telemetry_view.local_advertised_capabilities
                    ),
                )
            tg.start_soon(self._monitor_zenoh_isolation)
            if self.store_server:
                tg.start_soon(self.store_server.start)
            if self.download_coordinator:
                tg.start_soon(self.download_coordinator.run)
            if self.worker:
                tg.start_soon(self.worker.run)
            if self.master:
                tg.start_soon(self.master.run)
            if self.api:
                tg.start_soon(self.api.run)
            tg.start_soon(self._elect_loop)

    def _mark_artifact_inventory_dirty(self) -> None:
        """Schedule one debounced local artifact rescan without blocking events."""

        try:
            self._artifact_inventory_trigger_sender.send_nowait(None)
        except (WouldBlock, ClosedResourceError):
            return

    async def _observe_artifact_inventory_events(self) -> None:
        """Apply node-level replicated side effects and local rescan hints.

        Dedicated model-store hosts may intentionally run without either a
        worker or an API. Those roles normally persist the master-ordered trust
        set, so this node-level subscriber owns that side effect only when both
        are absent. This keeps repository-code authorization cluster-scoped for
        every supported node role without duplicating writes on ordinary nodes.
        """

        receiver = self.artifact_inventory_event_receiver
        if receiver is None:
            return
        trust_approvals = set(
            self.skulk_config.model_trust.approved_remote_code_identities
            if self.skulk_config is not None
            and self.skulk_config.model_trust is not None
            else ()
        )
        with receiver as events:
            async for indexed_event in events:
                event = indexed_event.event
                if self.worker is None and self.api is None:
                    if isinstance(event, StateSnapshotHydrated):
                        trust_approvals = set(
                            event.state.model_trust_approved_remote_code_identities
                        )
                    elif isinstance(event, ModelTrustApprovalChanged):
                        if event.approved:
                            trust_approvals.add(event.trust_identity)
                        else:
                            trust_approvals.discard(event.trust_identity)
                    if isinstance(
                        event, (ModelTrustApprovalChanged, StateSnapshotHydrated)
                    ):
                        try:
                            self.skulk_config = persist_model_trust_config(
                                resolve_config_path(), trust_approvals
                            )
                        except (OSError, ValueError):
                            logger.exception(
                                "Store-only node failed to persist master-ordered "
                                "model trust; repository-code downloads remain "
                                "fail-closed"
                            )
                if isinstance(
                    event,
                    (
                        InstanceCreated,
                        InstanceDeleted,
                        NodeDownloadProgress,
                        RunnerStatusUpdated,
                        StagedModelEvicted,
                        StateSnapshotHydrated,
                    ),
                ):
                    self._mark_artifact_inventory_dirty()

    def _current_artifact_inventory_state(self) -> State:
        """Return this node's freshest replicated runtime view."""

        if self.worker is not None:
            return self.worker.state
        if self.api is not None:
            return self.api.state
        if self.master is not None:
            return self.master.state
        return State()

    def _artifact_models_in_use(self) -> frozenset[str]:
        """Return model and companion repositories used by local live shards."""

        in_use: set[str] = set()
        for instance in self._current_artifact_inventory_state().instances.values():
            if self.node_id not in instance.shard_assignments.node_to_runner:
                continue
            for shard in instance.shard_assignments.runner_to_shard.values():
                card = shard.model_card
                in_use.add(str(card.model_id))
                if card.vision and card.vision.weights_repo:
                    in_use.add(card.vision.weights_repo)
                if card.runtime is not None:
                    if card.runtime.mtp_sidecar_repo:
                        in_use.add(card.runtime.mtp_sidecar_repo)
                    if card.runtime.assistant_model_repo:
                        in_use.add(card.runtime.assistant_model_repo)
        return frozenset(in_use)

    def _configured_artifact_cache_root(self) -> Path | None:
        """Return this node's configured cache root when staging is enabled."""

        config = self.skulk_config
        if (
            config is None
            or config.model_store is None
            or not config.model_store.enabled
        ):
            return None
        staging = resolve_node_staging(config.model_store, str(self.node_id))
        return Path(staging.node_cache_path).expanduser() if staging.enabled else None

    async def _artifact_inventory_loop(self) -> None:
        """Publish startup, change-triggered, and periodic availability readings."""

        while True:
            try:
                await self._publish_artifact_inventory()
            except Exception as error:  # noqa: BLE001 - lifetime service boundary
                logger.exception(f"Artifact-inventory telemetry scan failed: {error}")
            triggered = False
            with anyio.move_on_after(ARTIFACT_INVENTORY_REFRESH_SECONDS):
                try:
                    await self._artifact_inventory_trigger_receiver.receive()
                except EndOfStream:
                    return
                triggered = True
            if not triggered:
                continue
            await anyio.sleep(ARTIFACT_INVENTORY_DEBOUNCE_SECONDS)
            while True:
                try:
                    self._artifact_inventory_trigger_receiver.receive_nowait()
                except WouldBlock:
                    break
                except EndOfStream:
                    return

    async def _publish_artifact_inventory(self) -> None:
        """Scan compact node-cache truth and offer one telemetry snapshot."""

        config = self.skulk_config
        store_enabled = (
            config is not None
            and config.model_store is not None
            and config.model_store.enabled
        )
        canonical_root = (
            self.store_client.local_store_path
            if self.store_client is not None
            else None
        )
        artifacts: list[NodeArtifactAvailability] = []
        truncated = False
        if store_enabled:
            canonical_resolved = (
                canonical_root.expanduser().resolve()
                if canonical_root is not None
                else None
            )
            roots = tuple(
                root
                for root in installed_artifact_roots(
                    self._configured_artifact_cache_root()
                )
                if canonical_resolved is None
                or not root.expanduser().resolve().is_relative_to(canonical_resolved)
            )
            cards = await get_all_model_cards()
            discovered = await to_thread.run_sync(
                inventory_installed_artifacts,
                roots,
                cards,
                self._artifact_models_in_use(),
                None,
                self._artifact_inventory_detached_cache,
            )
            cache_items = [
                item
                for item in discovered
                if item.installed_identity is not None
                and (
                    canonical_resolved is None
                    or not Path(item.directory)
                    .resolve()
                    .is_relative_to(canonical_resolved)
                )
            ]
            cache_items.sort(
                key=lambda item: (
                    not item.in_use,
                    not item.manifest_complete,
                    -item.last_used_epoch_seconds,
                    item.model_id,
                    item.installed_identity or "",
                )
            )
            truncated = len(cache_items) > ARTIFACT_INVENTORY_ENTRY_LIMIT
            artifacts = [
                NodeArtifactAvailability(
                    model_id=item.model_id,
                    installed_identity=cast(str, item.installed_identity),
                    size_bytes=item.size_bytes,
                    last_used_epoch_seconds=item.last_used_epoch_seconds,
                    in_use=item.in_use,
                    manifest_complete=item.manifest_complete,
                )
                for item in cache_items[:ARTIFACT_INVENTORY_ENTRY_LIMIT]
            ]
        await self.router.telemetry_sender().send(
            NodeTelemetry(
                node_id=self.node_id,
                info=NodeArtifactInventory(
                    artifacts=artifacts,
                    store_host=canonical_root is not None,
                    truncated=truncated,
                ),
            )
        )

    async def _run_telemetry(self) -> None:
        """Maintain the node-owned TelemetryView from the telemetry plane (#279).

        Runs for the node's lifetime, independent of master election, so the
        view of every node's resources persists across a master flip. Each
        message coalesces last-write-wins; there is no ordering or persistence.
        """
        with self.telemetry_receiver as messages:
            async for message in messages:
                self.telemetry_view.apply(message)

    async def _monitor_zenoh_isolation(self) -> None:
        """Warn loudly while this node's Zenoh data plane has zero peers.

        The libp2p control plane keeps working when the Zenoh mesh never
        forms (the canonical shape: a zero-config remote member that
        multicast scouting cannot reach), so without this monitor the only
        symptom is remote streams dying one at a time. Warns only when the
        sampler reports a trustworthy 0 (post-grace) AND at least one other
        node advertises Zenoh on telemetry, i.e. there is genuinely a mesh
        this node should have joined. Cluster health raises the matching
        ``zenoh_isolated`` reason from the advertised count; this local
        warning is for the operator tailing THIS node's log.
        """
        if not self.data_plane_zenoh or self.zenoh_peer_sampler is None:
            return
        last_warning = 0.0
        while True:
            await anyio.sleep(_ZENOH_ISOLATION_CHECK_INTERVAL_SECONDS)
            count = await self.zenoh_peer_sampler.advertised_count()
            if count != 0:
                continue
            zenoh_peers = [
                node_id
                for node_id, resources in self.telemetry_view.node_resources.items()
                if node_id != self.node_id and resources.data_transport == "zenoh"
            ]
            if not zenoh_peers:
                continue
            now = time.monotonic()
            if now - last_warning < _ZENOH_ISOLATION_WARNING_INTERVAL_SECONDS:
                continue
            last_warning = now
            logger.warning(
                f"Zenoh data plane ISOLATED: this node has 0 Zenoh peer "
                f"transports while {len(zenoh_peers)} other node(s) advertise "
                f"the Zenoh data plane. Remote model/provider streams to and "
                f"from this node WILL fail even though cluster membership "
                f"looks healthy. If this node cannot reach peers via local "
                f"multicast (e.g. it joined over a routed/overlay network), "
                f"set SKULK_ZENOH_CONNECT to a reachable peer's Zenoh "
                f"endpoint (tcp/<peer-ip>:7447) and ensure SKULK_ZENOH_LISTEN "
                f"binds an address peers can dial."
            )

    def shutdown(self):
        # if this is our second call to shutdown, just sys.exit
        if self._tg.cancel_called():
            import sys

            sys.exit(1)
        self._tg.cancel_tasks()

    async def _request_cluster_config(self, session_id: SessionId) -> str | None:
        """Request the authoritative cluster config from the current master."""

        requester = SystemId()
        state_sync_sender = self.router.sender(topics.STATE_SYNC_MESSAGES)
        state_sync_receiver = self.router.receiver_with_origin(
            topics.STATE_SYNC_MESSAGES
        )
        with state_sync_receiver as messages:
            for attempt in range(_CLUSTER_CONFIG_SYNC_ATTEMPTS):
                await state_sync_sender.send(
                    StateSyncMessage(
                        kind="request",
                        requester=requester,
                        session_id=session_id,
                    )
                )
                with anyio.move_on_after(_CLUSTER_CONFIG_SYNC_RESPONSE_TIMEOUT_SECONDS):
                    async for origin, message in messages:
                        if message.kind != "response":
                            continue
                        if message.requester != requester:
                            continue
                        if message.session_id != session_id:
                            continue
                        if origin != str(session_id.master_node_id):
                            continue
                        return message.config_yaml
                if attempt < _CLUSTER_CONFIG_SYNC_ATTEMPTS - 1:
                    await anyio.sleep(_CLUSTER_CONFIG_SYNC_RETRY_INTERVAL_SECONDS)
        logger.warning(
            "Authoritative cluster config was unavailable after "
            f"{_CLUSTER_CONFIG_SYNC_ATTEMPTS} bootstrap attempts; retaining "
            "the local config"
        )
        return None

    def _apply_cluster_config_yaml(self, config_yaml: str) -> None:
        """Persist cluster config locally and rebuild derived runtime wiring.

        Merges rather than overwrites: an authoritative payload that carries
        no ``hf_token`` must not erase one configured locally (the previous
        raw ``write_text`` did exactly that on every bootstrap). An incoming
        token is adopted and promoted to ``HF_TOKEN`` when unset, mirroring
        the ordinary config-sync receive path, so downloads on a freshly
        joined node authenticate without a restart.
        """

        config_path = resolve_config_path()
        merge_cluster_config_bootstrap(config_yaml, config_path)
        self.skulk_config = load_skulk_config(config_path)

    async def _apply_authoritative_cluster_config(self, config_yaml: str) -> None:
        """Converge config plus every live model-store consumer atomically."""

        previous_store_server = self.store_server
        self._apply_cluster_config_yaml(config_yaml)
        new_store_client, new_store_server = _configure_model_store_runtime(
            self.node_id,
            self.skulk_config,
        )
        self.store_client = new_store_client
        if new_store_server is None:
            if previous_store_server is not None:
                await previous_store_server.stop()
            self.store_server = None
        else:
            self.store_server = (
                previous_store_server
                if previous_store_server is not None
                else new_store_server
            )
            await get_all_model_cards()
            await self.store_server.refresh_recovered_generations(
                get_current_registry_cards(),
            )
        if self.api is not None:
            self.api.set_model_store_runtime(
                self.skulk_config,
                self.store_client,
            )
        self._mark_artifact_inventory_dirty()

    async def _broadcast_config_if_store_host(self) -> None:
        """If this node is the store host, broadcast a valid config to all nodes.

        Resolves ``store_http_host`` to a routable IPv4 (see
        ``_routable_store_advertise_host``) so worker nodes receive an address
        they can actually reach over HTTP, rather than a hostname that may
        mDNS-resolve to an unreachable link-local address, ``127.0.0.1``, or
        None. An operator-supplied routable IPv4 literal is honored as-is;
        otherwise this host's best routable LAN address is used.

        The resolved address is broadcast over the cluster config-sync path,
        which every node (including this store host, via local delivery) applies
        and persists. We therefore do not separately write the local config file
        here: a second write would only be clobbered by the host applying its
        own broadcast.
        """
        # Reload persisted truth first: config sync updates the file (and the
        # environment) but not this startup snapshot, so serializing
        # self.skulk_config as-is after a Settings token rotation would
        # rebroadcast the stale token and overwrite the rotated one
        # fleet-wide (#922 review).
        refreshed_config = load_skulk_config()
        if refreshed_config is not None:
            self.skulk_config = refreshed_config
        if self.skulk_config is None or self.skulk_config.model_store is None:
            return
        ms = self.skulk_config.model_store
        if not ms.enabled:
            return
        local_hostname = socket.gethostname()
        is_store_host = node_matches_store_host(
            ms.store_host,
            str(self.node_id),
            hostname=local_hostname,
        )
        if not is_store_host:
            return

        # Advertise a routable IP, not a hostname. A bare hostname (e.g.
        # ``kite3.local``) can mDNS-resolve on a Thunderbolt-meshed fleet to the
        # store host's link-local TB address (169.254.x), which peers without a
        # direct TB link cannot route to, so their store downloads fail even
        # though they can reach the host fine over the LAN.
        reachable_host = _routable_store_advertise_host(
            ms.store_http_host, local_hostname
        )

        import copy

        import yaml

        # Broadcast the resolved reachable host to the cluster. The store
        # host's hf_token (if any) rides along so a fleet formed from one
        # configured node converges on that token; a blank one is dropped so
        # it can never clobber a real token on peers. The store host applies
        # its own broadcast via local delivery and persists it through the
        # normal config-sync path, so there is no separate local write here
        # (it would only be clobbered by that same broadcast).
        from skulk.store.config import normalized_hf_token

        broadcast_dict = copy.deepcopy(self.skulk_config.model_dump())
        broadcast_dict["model_store"]["store_http_host"] = reachable_host
        if normalized_hf_token(broadcast_dict.get("hf_token")) is None:
            broadcast_dict.pop("hf_token", None)
        broadcast_dict.pop("model_trust", None)
        broadcast_yaml = yaml.safe_dump(
            broadcast_dict, default_flow_style=False, sort_keys=False
        )

        await self.router.sender(topics.DOWNLOAD_COMMANDS).send(
            ForwarderDownloadCommand(
                origin=SystemId(),
                command=SyncConfig(config_yaml=broadcast_yaml),
            )
        )
        logger.info(
            f"ModelStore: broadcast config to cluster (store_http_host={reachable_host})"
        )

    async def _elect_loop(self):
        with self.election_result_receiver as results:
            async for result in results:
                # This function continues to have a lot of very specific entangled logic
                # At least it's somewhat contained

                # I don't like this duplication, but it's manageable for now.
                # TODO: This function needs refactoring generally

                # Ok:
                # On new master:
                # - Elect master locally if necessary
                # - Shutdown and re-create the worker
                # - Shut down and re-create the API

                start_replacement_event_router = False
                start_replacement_download_coordinator = False
                previous_store_server = self.store_server
                if result.is_new_master:
                    await anyio.sleep(0)
                    self.event_router.shutdown()
                    self.event_router = EventRouter(
                        self.node_id,
                        result.session_id,
                        command_sender=self.router.sender(topics.COMMANDS),
                        state_sync_sender=self.router.sender(
                            topics.STATE_SYNC_MESSAGES
                        ),
                        state_sync_receiver=self.router.receiver_with_origin(
                            topics.STATE_SYNC_MESSAGES
                        ),
                        external_inbound=self.router.receiver(topics.GLOBAL_EVENTS),
                        external_outbound=self.router.sender(topics.LOCAL_EVENTS),
                    )
                    self.artifact_inventory_event_receiver = (
                        self.event_router.receiver()
                    )
                    # Wait to bootstrap the replacement event router until the
                    # replacement worker/API receivers are attached. Otherwise,
                    # a fast snapshot hydrate can be emitted before those
                    # consumers exist, and the next live event will arrive out
                    # of sequence against blank local state.
                    start_replacement_event_router = True
                    if previous_store_server is None and self.store_server is not None:
                        self._tg.start_soon(self.store_server.start)

                if (
                    result.session_id.master_node_id == self.node_id
                    and self.master is not None
                ):
                    logger.info("Node elected Master")
                elif (
                    result.session_id.master_node_id == self.node_id
                    and self.master is None
                ):
                    logger.info("Node elected Master - promoting self")
                    # Seed the new session from this node's freshest replicated
                    # view (captured before local roles are torn down and
                    # re-created). API-only management nodes must use API State:
                    # their startup config can lag a master-ordered trust event,
                    # so falling back to it could resurrect a revocation after
                    # promotion. apply() replaces State wholesale (immutable
                    # convention), making either reference a consistent snapshot.
                    prior_state = (
                        self.worker.state
                        if self.worker is not None
                        else self.api.state
                        if self.api is not None
                        else None
                    )
                    self.master = Master(
                        self.node_id,
                        result.session_id,
                        initial_state=(
                            seed_state_for_new_session(prior_state)
                            if prior_state is not None
                            else None
                        ),
                        event_sender=self.event_router.sender(),
                        global_event_sender=self.router.sender(topics.GLOBAL_EVENTS),
                        local_event_receiver=self.router.receiver(topics.LOCAL_EVENTS),
                        command_receiver=self.router.receiver(topics.COMMANDS),
                        state_sync_receiver=self.router.receiver(
                            topics.STATE_SYNC_MESSAGES
                        ),
                        state_sync_sender=self.router.sender(
                            topics.STATE_SYNC_MESSAGES
                        ),
                        download_command_sender=self.router.sender(
                            topics.DOWNLOAD_COMMANDS
                        ),
                        telemetry_view=self.telemetry_view,
                        state_sync_store_http_host=_state_sync_store_http_host(
                            self.node_id,
                            self.skulk_config,
                        ),
                        initial_model_trust_identities=(
                            tuple(
                                self.skulk_config.model_trust.approved_remote_code_identities
                            )
                            if self.skulk_config is not None
                            and self.skulk_config.model_trust is not None
                            else ()
                        ),
                    )
                    self._tg.start_soon(self.master.run)
                elif (
                    result.session_id.master_node_id != self.node_id
                    and self.master is not None
                ):
                    logger.info(
                        f"Node {result.session_id.master_node_id} elected master - demoting self"
                    )
                    await self.master.shutdown()
                    self.master = None
                else:
                    logger.info(
                        f"Node {result.session_id.master_node_id} elected master"
                    )
                if (
                    result.is_new_master
                    and result.session_id.master_node_id != self.node_id
                ):
                    authoritative_config_yaml = await self._request_cluster_config(
                        result.session_id
                    )
                    if authoritative_config_yaml is not None:
                        await self._apply_authoritative_cluster_config(
                            authoritative_config_yaml
                        )
                if result.is_new_master:
                    if self.download_coordinator:
                        await self.download_coordinator.shutdown()
                        base_dl = skulk_shard_downloader(offline=self.offline)
                        ms = (
                            self.skulk_config.model_store
                            if self.skulk_config is not None
                            else None
                        )
                        if (
                            ms is not None
                            and ms.enabled
                            and self.store_client is not None
                        ):
                            elect_staging = resolve_node_staging(ms, str(self.node_id))
                            elect_downloader = ModelStoreDownloader(
                                inner=base_dl,
                                store_client=self.store_client,
                                staging_config=elect_staging,
                                allow_hf_fallback=ms.download.allow_hf_fallback,
                                installed_card_callback=(
                                    register_installed_card_record
                                ),
                            )
                        else:
                            elect_downloader = base_dl
                        elect_staging_path = (
                            Path(
                                resolve_node_staging(
                                    ms, str(self.node_id)
                                ).node_cache_path
                            )
                            if ms is not None and ms.enabled
                            else None
                        )
                        self.download_coordinator = DownloadCoordinator(
                            self.node_id,
                            elect_downloader,
                            event_sender=self.event_router.sender(),
                            telemetry_sender=self.router.telemetry_sender(),
                            download_command_receiver=self.router.receiver(
                                topics.DOWNLOAD_COMMANDS
                            ),
                            offline=self.offline,
                            staging_cache_path=elect_staging_path,
                            config_applied_callback=(
                                self.api.refresh_config_dependent_capabilities
                                if self.api is not None
                                else None
                            ),
                        )
                        # Do not start receiving StartDownload commands until
                        # the replacement worker below has attached the
                        # staging-capacity callback. Worker shutdown yields,
                        # so starting here creates a window where store-backed
                        # transfers bypass disk admission entirely.
                        start_replacement_download_coordinator = True
                    if self.worker:
                        await self.worker.shutdown()
                        ms2 = (
                            self.skulk_config.model_store
                            if self.skulk_config is not None
                            else None
                        )
                        elect_staging2 = (
                            resolve_node_staging(ms2, str(self.node_id))
                            if ms2 is not None and ms2.enabled
                            else None
                        )
                        # TODO: add profiling etc to resource monitor
                        self.worker = Worker(
                            self.node_id,
                            event_receiver=self.event_router.receiver(),
                            event_sender=self.event_router.sender(),
                            command_sender=self.router.sender(topics.COMMANDS),
                            download_command_sender=self.router.sender(
                                topics.DOWNLOAD_COMMANDS
                            ),
                            store_client=self.store_client,
                            staging_config=elect_staging2,
                            # Must match Node.create's Worker wiring: without this
                            # the recreated worker stops publishing NodeResources
                            # telemetry, and after a master restart (fresh
                            # telemetry_view) the node never reappears in
                            # node_resources, so placement silently treats a
                            # management/edge node as eligible (#279 review).
                            telemetry_sender=self.router.telemetry_sender(),
                            telemetry_view=self.telemetry_view,
                            api_available=self.api is not None,
                            data_transport=(
                                "zenoh" if self.data_plane_zenoh else "gossipsub"
                            ),
                            zenoh_peer_sampler=self.zenoh_peer_sampler,
                            # Must ALSO match Node.create's wiring: without this
                            # the recreated worker has no data sender, so every
                            # generation output chunk falls back to the event
                            # plane — which the API no longer routes (#279 Phase
                            # 2a) — and every completion stream hangs forever.
                            data_sender=self.router.sender(topics.DATA),
                            trace_data_sender=self.router.sender(topics.TRACE_DATA),
                            realtime_audio_packet_receiver=self.router.receiver(
                                topics.REALTIME_AUDIO
                            ),
                            speech_media_packet_receiver=self.router.receiver(
                                topics.SPEECH_MEDIA
                            ),
                            connection_message_receiver=self.router.receiver(
                                topics.CONNECTION_MESSAGES
                            ),
                            session_connection_snapshot=(
                                self.router.current_session_connections
                            ),
                            vision_media_packet_sender=self.router.sender(
                                topics.VISION_MEDIA
                            ),
                            vision_media_packet_receiver=self.router.receiver(
                                topics.VISION_MEDIA
                            ),
                        )
                        if self.download_coordinator is not None and isinstance(
                            self.download_coordinator.shard_downloader,
                            ModelStoreDownloader,
                        ):
                            self.download_coordinator.shard_downloader.set_staging_capacity_callback(
                                self.worker.prepare_staging_transfer
                            )
                        self._tg.start_soon(self.worker.run)
                        if self.api is not None:
                            self.api.set_runner_diagnostics_provider(
                                self.worker.collect_runner_diagnostics
                            )
                            self.api.set_runner_cancel_provider(
                                self.worker.cancel_runner_task
                            )
                            self.api.set_vision_media_ingress_provider(
                                self.worker.collect_vision_media_ingress_diagnostics
                            )
                    if start_replacement_download_coordinator:
                        assert self.download_coordinator is not None
                        self._tg.start_soon(self.download_coordinator.run)
                    if self.api:
                        self.api.reset(
                            result.won_clock,
                            self.event_router.receiver(),
                            result.session_id.master_node_id,
                        )
                    if start_replacement_event_router:
                        self._tg.start_soon(self._observe_artifact_inventory_events)
                        self._tg.start_soon(self.event_router.run)
                    # Broadcast config to cluster so worker nodes get the right store address
                    await self._broadcast_config_if_store_host()
                else:
                    if self.api:
                        self.api.unpause(
                            result.won_clock,
                            master_node_id=result.session_id.master_node_id,
                        )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "operator":
        # Operator commands are local administration and never launch a node.
        from skulk.operator.cli import main as operator_main

        sys.exit(operator_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        # `skulk doctor` is a standalone audit, not a node launch: dispatch
        # before Args.parse() so the node argument parser never sees it.
        from skulk.doctor.cli import main as doctor_main

        sys.exit(doctor_main(sys.argv[2:]))
    args = Args.parse()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, 65535), hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))

    mp.set_start_method("spawn", force=True)

    # Load config early so the logging section is available before anything
    # else runs.  The full config is loaded again inside Node.create() for
    # the model store and inference sections.  If the file is malformed we
    # fall back gracefully — logging will start without the JSON sink and
    # the validation error is logged once the logger is up.
    _log_cfg = None
    _early_config: SkulkConfig | None = None
    try:
        _early_config = load_skulk_config()
        _log_cfg = _early_config.logging if _early_config else None
    except Exception:
        pass  # Logged after logger_setup below

    # External-shipper mode (SKULK_LOGGING_EXTERNAL=1, set by the
    # launchd / systemd wrapper when an external Vector agent is
    # installed) implies "structured logging on at boot" without
    # requiring an `enabled: true` in skulk.yaml — the env var is the
    # operator's signal that they have a shipper hooked up. The
    # dashboard / config sync still controls the sink at runtime via
    # set_structured_stdout, so an operator can disable shipping live.
    _structured = external_log_pipe_enabled() or bool(_log_cfg and _log_cfg.enabled)
    logger_setup(
        SKULK_LOG,
        args.verbosity,
        structured_stdout=_structured,
        ingest_url=_log_cfg.ingest_url if _log_cfg else "",
    )
    logger.info("Starting Skulk")
    # The libp2p namespace token seeds the private-network PSK (swarm.rs) and, when
    # the Zenoh data plane is on, the no-TLS Zenoh namespace too; logging its value
    # would let anyone with log access compute the namespace and subscribe to
    # `data` (#312 review). Log only whether it is set and a non-routing
    # fingerprint, which is enough to confirm two nodes share a namespace.
    _libp2p_ns = os.environ.get(_LIBP2P_NAMESPACE_ENV_VAR)
    if _libp2p_ns is not None:
        logger.info(
            f"{_LIBP2P_NAMESPACE_ENV_VAR} set (fingerprint "
            f"{_namespace_fingerprint(_libp2p_ns)})"
        )
    else:
        logger.info(f"{_LIBP2P_NAMESPACE_ENV_VAR} unset, using default")

    # Engine auto-provisioning (#614 Phase 3): before any serving decision,
    # ensure a Linux node without an explicit llama-server override has the
    # pinned managed build (fetch + checksum-verify on first run), then
    # re-derive capability facts so the advertised backends include it. macOS
    # and opted-out nodes return immediately. --offline nodes never reach for
    # the network (the air-gapped contract covers engine artifacts exactly
    # like model downloads) but still wire an already-provisioned managed
    # install from disk, so an offline restart keeps its served capability.
    # Management-only launches skip entirely (whether via --no-worker or the
    # declared SKULK_NODE_PARTICIPATION=management): they are never placement
    # candidates and must stay side-effect free.
    declared_participation = (
        os.environ.get("SKULK_NODE_PARTICIPATION", "").strip().lower()
    )
    if not args.no_worker and declared_participation != "management":
        from skulk.facts import current_node_facts, refresh_node_facts
        from skulk.provisioning import ensure_llama_server

        if (
            ensure_llama_server(current_node_facts(), allow_download=not args.offline)
            is not None
        ):
            refresh_node_facts()

    if args.spawn_api:
        preflight_api_port(args.api_port)

    # Tailscale: if configured, query tailscaled and merge bootstrap peers.
    _ts_config = (
        _early_config.connectivity.tailscale
        if _early_config and _early_config.connectivity
        else None
    )
    if _ts_config and _ts_config.enabled:
        import asyncio as _asyncio

        _ts_status = _asyncio.run(query_tailscale_status())
        if _ts_status.running:
            logger.info(
                f"Tailscale: running | IP {_ts_status.self_ip} | {_ts_status.dns_name}"
            )
        else:
            logger.warning(
                "Tailscale connectivity configured but tailscaled is not running"
            )
        # Auto-discover tailnet peers as bootstrap addresses so a Tailscale
        # cluster needs no hand-maintained IP list — just `enabled: true`. Each
        # peer is dialed on this node's libp2p port; non-Skulk tailnet peers
        # fail the private-network handshake and are harmlessly ignored.
        #
        # Auto-discovery needs a fixed, known port to build the peer multiaddrs.
        # With --libp2p-port 0 (OS-assigned) the listen port differs per node and
        # is unknown to peers, so an auto-built /tcp/0 address could never be
        # dialed. Skip auto-discovery in that case and tell the operator how to
        # make it work, rather than silently producing dead /tcp/0 peers.
        if args.libp2p_port == 0:
            _auto_peers: list[str] = []
            if _ts_status.peer_ips:
                logger.warning(
                    "Tailscale auto-discovery is enabled but --libp2p-port is 0 "
                    "(OS-assigned); auto-discovered peers need a fixed port and were "
                    "skipped. Set --libp2p-port / SKULK_LIBP2P_PORT (default 52416) or "
                    "list connectivity.tailscale.bootstrap_peers explicitly."
                )
        else:
            _auto_peers = [
                f"/ip4/{ip}/tcp/{args.libp2p_port}" for ip in _ts_status.peer_ips
            ]
        # Merge auto-discovered + config-listed peers, de-duplicating against
        # CLI/existing peers and each other while preserving order.
        _seen = set(args.bootstrap_peers)
        _extra: list[str] = []
        for _peer in _auto_peers + list(_ts_config.bootstrap_peers):
            if _peer not in _seen:
                _seen.add(_peer)
                _extra.append(_peer)
        if _extra:
            args = args.model_copy(
                update={"bootstrap_peers": args.bootstrap_peers + _extra}
            )
            logger.info(
                f"Tailscale: added {len(_extra)} bootstrap peer(s) "
                f"({len(_auto_peers)} auto-discovered, "
                f"{len(_ts_config.bootstrap_peers)} from config)"
            )

    # macOS Local Network Privacy: a denied process silently fails to reach LAN
    # / Thunderbolt peers (EHOSTUNREACH), so cluster discovery never forms.
    # Detect it early and tell the operator how to grant access.
    if check_local_network_access() == "blocked":
        logger.warning(local_network_denied_message())

    if args.offline:
        logger.info("Running in OFFLINE mode — no internet checks, local models only")

    if args.bootstrap_peers:
        logger.info(f"Bootstrap peers: {args.bootstrap_peers}")

    if args.no_batch:
        os.environ["SKULK_NO_BATCH"] = "1"
        os.environ["SKULK_NO_BATCH"] = "1"  # legacy compat
        logger.info("Continuous batching disabled (--no-batch)")

    # Set FAST_SYNCH override env var for runner subprocesses
    if args.fast_synch is True:
        os.environ["SKULK_FAST_SYNCH"] = "on"
        os.environ["SKULK_FAST_SYNCH"] = "on"  # legacy compat
        logger.info("FAST_SYNCH forced ON")
    elif args.fast_synch is False:
        os.environ["SKULK_FAST_SYNCH"] = "off"
        os.environ["SKULK_FAST_SYNCH"] = "off"  # legacy compat
        logger.info("FAST_SYNCH forced OFF")

    try:
        anyio.run(run_node, args)
    except BaseException as exception:
        logger.opt(exception=exception).critical(
            "Skulk terminated due to unhandled exception"
        )
        raise
    finally:
        logger.info("Skulk shutdown complete")
        logger_cleanup()


async def run_node(args: "Args") -> None:
    """Create and serve a node on one event loop until shutdown.

    Loop-bound extension tasks and transport resources must outlive construction;
    closing a temporary creation loop cancels them before serving can begin.
    """
    node = await Node.create(args)
    await node.run()


class Args(CamelCaseModel):
    verbosity: int = 0
    force_master: bool = False
    spawn_api: bool = False
    api_port: PositiveInt = 52415
    tb_only: bool = False
    no_worker: bool = False
    no_downloads: bool = False
    offline: bool = os.getenv("SKULK_OFFLINE", "false").lower() == "true"
    no_batch: bool = False
    fast_synch: bool | None = None  # None = auto, True = force on, False = force off
    bootstrap_peers: list[str] = []
    libp2p_port: int

    @classmethod
    def parse(cls) -> Self:
        parser = argparse.ArgumentParser(prog="skulk")
        default_verbosity = 0
        parser.add_argument(
            "-q",
            "--quiet",
            action="store_const",
            const=-1,
            dest="verbosity",
            default=default_verbosity,
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            dest="verbosity",
            default=default_verbosity,
        )
        parser.add_argument(
            "-m",
            "--force-master",
            action="store_true",
            dest="force_master",
        )
        parser.add_argument(
            "--no-api",
            action="store_false",
            dest="spawn_api",
        )
        parser.add_argument(
            "--api-port",
            type=int,
            dest="api_port",
            default=52415,
        )
        parser.add_argument(
            "--no-worker",
            action="store_true",
        )
        parser.add_argument(
            "--no-downloads",
            action="store_true",
            help="Disable the download coordinator (node won't download models)",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            default=os.getenv("SKULK_OFFLINE", "false").lower() == "true",
            help="Run in offline/air-gapped mode: skip internet checks, use only pre-staged local models",
        )
        parser.add_argument(
            "--no-batch",
            action="store_true",
            help="Disable continuous batching, use sequential generation",
        )
        parser.add_argument(
            "--bootstrap-peers",
            type=lambda s: [p for p in s.split(",") if p],
            default=os.getenv("SKULK_BOOTSTRAP_PEERS", "").split(",")
            if os.getenv("SKULK_BOOTSTRAP_PEERS")
            else [],
            dest="bootstrap_peers",
            help="Comma-separated libp2p multiaddrs to dial on startup (env: SKULK_BOOTSTRAP_PEERS)",
        )
        parser.add_argument(
            "--libp2p-port",
            type=int,
            # Default to a fixed, well-known port rather than an OS-assigned one
            # so that bootstrap-peer multiaddrs (Tailscale, cross-subnet) have a
            # predictable port to dial — a user can write
            # /ip4/<peer>/tcp/52416 without first inspecting each node's random
            # port. mDNS discovery advertises the real port either way, so this
            # is harmless on a single local network. Pass 0 for OS-assigned.
            default=int(os.getenv("SKULK_LIBP2P_PORT", "52416")),
            dest="libp2p_port",
            help="Fixed TCP port for libp2p to listen on (default 52416; 0 = OS-assigned; env: SKULK_LIBP2P_PORT).",
        )
        fast_synch_group = parser.add_mutually_exclusive_group()
        fast_synch_group.add_argument(
            "--fast-synch",
            action="store_true",
            dest="fast_synch",
            default=None,
            help="Force MLX FAST_SYNCH on (for JACCL backend)",
        )
        fast_synch_group.add_argument(
            "--no-fast-synch",
            action="store_false",
            dest="fast_synch",
            help="Force MLX FAST_SYNCH off",
        )

        args = parser.parse_args()
        return cls(**vars(args))  # pyright: ignore[reportAny] - We are intentionally validating here, we can't do it statically
