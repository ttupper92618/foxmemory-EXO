"""Pip-installable CUDA llama-server for Skulk's served GGUF engine.

The wheel carries the Foxlight-built ``llama-server`` (and ``ggml-rpc-server``)
compiled from the pinned upstream llama.cpp release with CUDA enabled; the
CUDA runtime resolves from NVIDIA's official PyPI wheels declared as ordinary
dependencies. :func:`binary_path` is what Skulk's engine provisioning wires
into ``SKULK_LLAMA_SERVER_BIN``; the ``llama-server-cuda`` console script is
the same thing for humans.

The shim exists because ``llama-server`` is an executable, not a library:
its CUDA libraries must be on the loader path at exec time, and pip installs
them into the ``nvidia`` namespace packages rather than a system directory.
"""

from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_BIN_DIR = _PACKAGE_DIR / "bin"

#: NVIDIA namespace packages whose lib/ dirs the binary needs at exec time.
_NVIDIA_LIB_PACKAGES = ("nvidia.cuda_runtime", "nvidia.cublas", "nvidia.nccl")


def binary_path() -> Path:
    """Absolute path of the bundled ``llama-server`` binary.

    Raises:
        FileNotFoundError: When the wheel payload is missing (a source
            checkout rather than a built wheel).
    """
    binary = _BIN_DIR / "llama-server"
    if not binary.is_file():
        raise FileNotFoundError(
            f"no llama-server payload at {binary}; this is a source checkout, "
            "not a built wheel"
        )
    return binary


def rpc_server_path() -> Path | None:
    """Absolute path of the bundled ``ggml-rpc-server``, or ``None``."""
    rpc = _BIN_DIR / "ggml-rpc-server"
    return rpc if rpc.is_file() else None


def nvidia_library_dirs() -> list[Path]:
    """Library directories of the installed NVIDIA runtime wheels."""
    dirs: list[Path] = []
    for package in _NVIDIA_LIB_PACKAGES:
        spec = find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            continue
        for location in spec.submodule_search_locations:
            lib_dir = Path(location) / "lib"
            if lib_dir.is_dir():
                dirs.append(lib_dir)
    return dirs


def launch_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for exec'ing the binary: NVIDIA lib dirs prepended.

    Args:
        base: Environment to extend (defaults to ``os.environ``).

    Returns:
        A copy with ``LD_LIBRARY_PATH`` prefixed by the NVIDIA wheel lib
        directories and the wheel's own ``bin`` dir (the ggml shared
        libraries sit next to the binary and resolve via ``$ORIGIN``, so the
        explicit entry is belt and suspenders).
    """
    env = dict(os.environ if base is None else base)
    prefix = [str(d) for d in nvidia_library_dirs()] + [str(_BIN_DIR)]
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(prefix + ([existing] if existing else []))
    return env


def main() -> int:
    """Console entry point: exec llama-server with the CUDA libs wired.

    Forwards all arguments verbatim, so ``llama-server-cuda --list-devices``
    behaves exactly like the underlying binary (which is what lets Skulk's
    facts probe validate the wheel like any other engine binary).
    """
    binary = binary_path()
    os.execve(str(binary), [str(binary), *sys.argv[1:]], launch_environment())
    return 1  # pragma: no cover - execve does not return on success


def rpc_main() -> int:
    """Console entry point for the bundled ``ggml-rpc-server`` (RPC donor).

    Same loader wiring as :func:`main`, so a multi-node GGUF donor on a
    wheel-provisioned node runs the CUDA RPC server without any manual
    library-path setup.
    """
    rpc = rpc_server_path()
    if rpc is None:
        raise FileNotFoundError(
            "no ggml-rpc-server payload in this wheel; this is a source "
            "checkout, not a built wheel"
        )
    os.execve(str(rpc), [str(rpc), *sys.argv[1:]], launch_environment())
    return 1  # pragma: no cover - execve does not return on success
