# pyright: reportPrivateUsage=false
"""Provisioning tests: variant selection, verification, wiring, overrides."""

import hashlib
import importlib.metadata
import io
import os
import tarfile
from pathlib import Path
from typing import cast

import pytest

import skulk.provisioning.llama_server as provisioning
from skulk.facts.testing import AMD_STRIX, NVIDIA_A40, make_facts, ok_bin
from skulk.provisioning.llama_server import (
    ensure_llama_server,
    provision_llama_server,
    select_variant,
    select_variant_chain,
)
from skulk.provisioning.manifest import (
    LLAMA_SERVER_ARTIFACTS,
    LLAMA_SERVER_PIN,
    EngineArtifact,
)
from skulk.shared.backends import LLAMA_SERVER_BIN_ENV


def test_select_variant_chain_by_gpu_vendor() -> None:
    # NVIDIA prefers the Foxlight CUDA build (container clouds have no usable
    # Vulkan ICD) with Vulkan as the bare-metal-friendly fallback; AMD uses
    # the fleet-proven RADV Vulkan path.
    assert select_variant_chain(make_facts(gpus=(NVIDIA_A40,))) == ("cuda", "vulkan")
    assert select_variant_chain(make_facts(gpus=(AMD_STRIX,))) == ("vulkan",)
    assert select_variant(make_facts(gpus=(NVIDIA_A40,))) == "cuda"
    assert select_variant(make_facts(gpus=(AMD_STRIX,))) == "vulkan"


def test_manifest_matches_pinned_linux_release_shape() -> None:
    """Only Linux assets that exist for the pinned upstream tag are exposed."""
    assert set(LLAMA_SERVER_ARTIFACTS) == {
        ("x86_64", "cpu"),
        ("x86_64", "vulkan"),
        ("aarch64", "cpu"),
        ("aarch64", "vulkan"),
    }
    for artifact in LLAMA_SERVER_ARTIFACTS.values():
        assert LLAMA_SERVER_PIN in artifact.asset_name
        assert len(artifact.sha256) == 64
        assert artifact.url().endswith(artifact.asset_name)


def test_select_variant_cpu_without_gpu() -> None:
    assert select_variant(make_facts()) == "cpu"


def test_select_variant_none_on_darwin() -> None:
    assert select_variant(make_facts(platform="darwin")) is None


def _fake_archive(binary_relpath: str = "build/bin/llama-server") -> bytes:
    """A gzipped tarball containing a fake llama-server binary."""
    payload = b"#!/bin/sh\necho fake llama-server\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo(binary_relpath)
        info.size = len(payload)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _pin_fake_artifact(
    monkeypatch: pytest.MonkeyPatch, archive: bytes, *, corrupt_checksum: bool = False
) -> None:
    """Point the manifest at a synthetic artifact and stub the download."""
    sha256 = hashlib.sha256(archive).hexdigest()
    if corrupt_checksum:
        sha256 = "0" * 64
    artifact = EngineArtifact(asset_name="fake.tar.gz", sha256=sha256)
    import platform as platform_module

    monkeypatch.setitem(
        LLAMA_SERVER_ARTIFACTS, (platform_module.machine(), "vulkan"), artifact
    )

    def _fake_download(art: EngineArtifact, destination: Path) -> None:
        destination.write_bytes(archive)

    monkeypatch.setattr(provisioning, "_download", _fake_download)


def _isolate_engines_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    engines = tmp_path / "engines"
    monkeypatch.setattr(provisioning, "SKULK_ENGINES_DIR", engines)
    return engines


# The real installer, captured before the autouse stub below replaces the
# module attribute: tests that cover try_install_cuda_wheel ITSELF call this
# reference (its internals still resolve subprocess/shutil/_cuda_wheel_usable
# through the module, so per-test monkeypatches apply), while everything
# routed through ensure_llama_server gets the safe stub (PR #665 review).
_REAL_TRY_INSTALL_CUDA_WHEEL = provisioning.try_install_cuda_wheel


@pytest.fixture(autouse=True)
def _never_install_wheels(  # pyright: ignore[reportUnusedFunction] - autouse
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No test here may attempt a real wheel install (#661 review, P1).

    On a machine with uv on PATH and no CUDA wheel, the on-demand install
    branch in ensure_llama_server would otherwise run a REAL multi-GB
    `uv pip install` against the live index from inside the older NVIDIA
    tarball-path tests. Tests that exercise the install stub it themselves,
    overriding this default; tests of the installer itself use
    ``_REAL_TRY_INSTALL_CUDA_WHEEL`` with its effects patched.
    """

    def _no_install(installed_facts: object) -> bool:
        return False

    # Existing synthetic NVIDIA facts model the established x86_64 wheel lane.
    # Tests for the narrower arm64 kernel contract override this explicitly.
    monkeypatch.setattr(provisioning.platform_module, "machine", lambda: "x86_64")
    monkeypatch.setattr(provisioning, "try_install_cuda_wheel", _no_install)


def test_provision_downloads_verifies_and_installs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engines = _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive())
    binary = provision_llama_server("vulkan")
    assert binary.is_file()
    assert os.access(binary, os.X_OK)
    assert binary.is_relative_to(engines / "llama-server" / LLAMA_SERVER_PIN)


def test_provision_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive())
    first = provision_llama_server("vulkan")

    def _must_not_download(art: EngineArtifact, destination: Path) -> None:
        raise AssertionError("re-downloaded an already-provisioned build")

    monkeypatch.setattr(provisioning, "_download", _must_not_download)
    assert provision_llama_server("vulkan") == first


def test_provision_rejects_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive(), corrupt_checksum=True)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        provision_llama_server("vulkan")


def test_provision_rejects_archive_without_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive(binary_relpath="build/bin/other"))
    with pytest.raises(RuntimeError, match="no llama-server binary"):
        provision_llama_server("vulkan")


def test_ensure_honors_explicit_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An operator's SKULK_LLAMA_SERVER_BIN wins; provisioning never runs.
    _isolate_engines_dir(monkeypatch, tmp_path)
    facts = make_facts(
        gpus=(NVIDIA_A40,), llama_server_bin=ok_bin(LLAMA_SERVER_BIN_ENV)
    )
    assert ensure_llama_server(facts) is None


def test_ensure_honors_opt_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_engines_dir(monkeypatch, tmp_path)
    monkeypatch.setenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, "1")
    assert ensure_llama_server(make_facts(gpus=(NVIDIA_A40,))) is None


def test_ensure_provisions_and_exports_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The end-to-end just-works wiring: no override, Linux GPU node ->
    # download, verify, and export SKULK_LLAMA_SERVER_BIN for this process.
    _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive())
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    binary = ensure_llama_server(make_facts(gpus=(NVIDIA_A40,)))
    assert binary is not None
    assert os.environ[LLAMA_SERVER_BIN_ENV] == str(binary)


def test_ensure_swallows_download_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A node must start without network: provisioning failure degrades to
    # "no served engine" with a warning, never a crash.
    _isolate_engines_dir(monkeypatch, tmp_path)

    def _fail(art: EngineArtifact, destination: Path) -> None:
        raise OSError("network down")

    monkeypatch.setattr(provisioning, "_download", _fail)
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    assert ensure_llama_server(make_facts(gpus=(NVIDIA_A40,))) is None


def test_ensure_offline_wires_existing_managed_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An --offline restart must keep its served capability: the managed build
    # already on disk wires without any network touch (PR #615 review).
    _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive())
    provisioned = provision_llama_server("vulkan")

    def _must_not_download(art: EngineArtifact, destination: Path) -> None:
        raise AssertionError("offline ensure touched the network")

    monkeypatch.setattr(provisioning, "_download", _must_not_download)
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    wired = ensure_llama_server(make_facts(gpus=(NVIDIA_A40,)), allow_download=False)
    assert wired == provisioned
    assert os.environ[LLAMA_SERVER_BIN_ENV] == str(provisioned)


def test_ensure_offline_without_install_is_absent_not_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_engines_dir(monkeypatch, tmp_path)

    def _must_not_download(art: EngineArtifact, destination: Path) -> None:
        raise AssertionError("offline ensure touched the network")

    monkeypatch.setattr(provisioning, "_download", _must_not_download)
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    assert (
        ensure_llama_server(make_facts(gpus=(NVIDIA_A40,)), allow_download=False)
        is None
    )


def test_ensure_offline_prefers_capable_variant_over_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # With a stale cpu install AND a vulkan install on disk, an offline GPU
    # node must wire the vulkan build, not the alphabetically-first cpu one
    # (PR #615 review).
    engines = _isolate_engines_dir(monkeypatch, tmp_path)
    for variant in ("cpu", "vulkan"):
        vdir = engines / "llama-server" / LLAMA_SERVER_PIN / variant
        vdir.mkdir(parents=True)
        binary = vdir / "llama-server"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    wired = ensure_llama_server(make_facts(gpus=(NVIDIA_A40,)), allow_download=False)
    assert wired is not None
    assert "vulkan" in str(wired)


def test_interrupted_extraction_leaves_no_partial_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An extraction failure must not leave a partial variant dir that would
    # satisfy the fast path unverified on the next start (PR #615 review).
    engines = _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive())

    def _boom(archive: Path, destination: Path) -> None:
        (destination / "partial-file").write_text("half")
        raise OSError("disk full mid-extract")

    monkeypatch.setattr(provisioning, "_safe_extract", _boom)
    with pytest.raises(OSError):
        provision_llama_server("vulkan")
    target = engines / "llama-server" / LLAMA_SERVER_PIN / "vulkan"
    assert not target.exists()
    # Recovery: a later attempt with a working extractor installs cleanly.
    monkeypatch.undo()
    _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive())
    assert provision_llama_server("vulkan").is_file()


def _fake_wheel(tmp_path: Path, name: str) -> Path:
    shim = tmp_path / name
    shim.write_text("#!/bin/sh\n")
    shim.chmod(0o755)
    return shim


def test_wheel_outranks_tarball_provisioning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A pip-installed engine wheel is the preferred managed source: no
    # download may happen when one is present, and its RPC donor shim is
    # exported alongside.
    _isolate_engines_dir(monkeypatch, tmp_path)
    shim = _fake_wheel(tmp_path, "llama-server-cuda")
    rpc = _fake_wheel(tmp_path, "ggml-rpc-server-cuda")

    def _fake_lookup(
        vendor: str, facts: object
    ) -> tuple[Path, Path | None] | None:
        return (shim, rpc) if vendor == "nvidia" else None

    monkeypatch.setattr(provisioning, "wheel_llama_server", _fake_lookup)

    def _must_not_download(art: EngineArtifact, destination: Path) -> None:
        raise AssertionError("downloaded despite an installed engine wheel")

    monkeypatch.setattr(provisioning, "_download", _must_not_download)
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    monkeypatch.delenv(provisioning.RPC_SERVER_BIN_ENV, raising=False)
    wired = ensure_llama_server(make_facts(gpus=(NVIDIA_A40,)))
    assert wired == shim
    assert os.environ[LLAMA_SERVER_BIN_ENV] == str(shim)
    assert os.environ[provisioning.RPC_SERVER_BIN_ENV] == str(rpc)


def test_vulkan_wheel_wired_on_amd_nodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # AMD nodes use the Vulkan wheel when installed; the tarball path is the
    # fallback, not the preference.
    _isolate_engines_dir(monkeypatch, tmp_path)
    shim = _fake_wheel(tmp_path, "llama-server-vulkan")

    def _fake_lookup(
        vendor: str, facts: object
    ) -> tuple[Path, Path | None] | None:
        return (shim, None) if vendor == "amd" else None

    monkeypatch.setattr(provisioning, "wheel_llama_server", _fake_lookup)
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    wired = ensure_llama_server(make_facts(gpus=(AMD_STRIX,)))
    assert wired == shim


def test_tarball_fallback_when_no_wheel_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_engines_dir(monkeypatch, tmp_path)

    def _no_wheel(vendor: str, facts: object) -> tuple[Path, Path | None] | None:
        return None

    monkeypatch.setattr(provisioning, "wheel_llama_server", _no_wheel)
    _pin_fake_artifact(monkeypatch, _fake_archive())
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    wired = ensure_llama_server(make_facts(gpus=(AMD_STRIX,)))
    assert wired is not None
    assert "vulkan" in str(wired)


def test_dormant_reports_wheel_without_wiring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The read-only diagnostics twin (#628): reports the shim startup would
    # adopt, but never exports environment.
    _isolate_engines_dir(monkeypatch, tmp_path)
    shim = _fake_wheel(tmp_path, "llama-server-cuda")

    def _fake_lookup(
        vendor: str, facts: object
    ) -> tuple[Path, Path | None] | None:
        return (shim, None) if vendor == "nvidia" else None

    monkeypatch.setattr(provisioning, "wheel_llama_server", _fake_lookup)
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    assert provisioning.dormant_llama_server(make_facts(gpus=(NVIDIA_A40,))) == shim
    assert LLAMA_SERVER_BIN_ENV not in os.environ


def test_dormant_matches_ensure_offline_adoption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Drift guard: the diagnostic answer must be the binary offline startup
    # actually wires. Hermetic against the developer/CI environment: a real
    # installed engine wheel would outrank the provisioned tarball for BOTH
    # functions, silently weakening the comparison (PR #634 review).
    _isolate_engines_dir(monkeypatch, tmp_path)

    def _no_wheel(vendor: str, facts: object) -> tuple[Path, Path | None] | None:
        return None

    monkeypatch.setattr(provisioning, "wheel_llama_server", _no_wheel)
    _pin_fake_artifact(monkeypatch, _fake_archive())
    provisioned = provision_llama_server("vulkan")
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    facts = make_facts(gpus=(NVIDIA_A40,))
    dormant = provisioning.dormant_llama_server(facts)
    assert dormant == provisioned
    assert LLAMA_SERVER_BIN_ENV not in os.environ
    wired = ensure_llama_server(facts, allow_download=False)
    assert wired == dormant


def test_dormant_honors_override_and_opt_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_engines_dir(monkeypatch, tmp_path)
    shim = _fake_wheel(tmp_path, "llama-server-cuda")

    def _cuda_wheel(
        vendor: str, facts: object
    ) -> tuple[Path, Path | None] | None:
        return (shim, None)

    monkeypatch.setattr(provisioning, "wheel_llama_server", _cuda_wheel)
    # Explicit override wins: nothing dormant to report.
    override_facts = make_facts(
        gpus=(NVIDIA_A40,), llama_server_bin=ok_bin(LLAMA_SERVER_BIN_ENV)
    )
    assert provisioning.dormant_llama_server(override_facts) is None
    # Opt-out disables adoption, so the diagnostic must not promise it.
    monkeypatch.setenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, "1")
    assert provisioning.dormant_llama_server(make_facts(gpus=(NVIDIA_A40,))) is None


def test_dormant_none_when_nothing_on_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_engines_dir(monkeypatch, tmp_path)

    def _no_wheel(vendor: str, facts: object) -> tuple[Path, Path | None] | None:
        return None

    monkeypatch.setattr(provisioning, "wheel_llama_server", _no_wheel)
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    assert provisioning.dormant_llama_server(make_facts(gpus=(NVIDIA_A40,))) is None


def test_cuda_capability_gate() -> None:
    # The CUDA wheel is compiled for SM >= 8.0: a T4 (7.5) or a
    # capability-unknown NVML-degraded GPU must not select it (it would fail
    # only at model load), while an A40 (8.6) passes (PR #615 review).
    from skulk.facts.testing import NVIDIA_PRESENCE_ONLY
    from skulk.shared.types.node_facts import GpuDeviceFact

    t4 = GpuDeviceFact(
        vendor="nvidia",
        name="Tesla T4",
        detection_source="nvml",
        vram_total_bytes=16 * 2**30,
        compute_capability="7.5",
    )
    assert provisioning._cuda_capability_ok(make_facts(gpus=(NVIDIA_A40,))) is True
    assert provisioning._cuda_capability_ok(make_facts(gpus=(t4,))) is False
    assert (
        provisioning._cuda_capability_ok(make_facts(gpus=(NVIDIA_PRESENCE_ONLY,)))
        is False
    )


def test_cuda_capability_gate_on_arm64(monkeypatch: pytest.MonkeyPatch) -> None:
    """The arm64 package tag must not admit a GPU without a compiled kernel."""
    from skulk.shared.types.node_facts import GpuDeviceFact

    monkeypatch.setattr(provisioning.platform_module, "machine", lambda: "aarch64")
    gb10 = GpuDeviceFact(
        vendor="nvidia",
        name="NVIDIA GB10",
        detection_source="nvml",
        vram_total_bytes=128 * 2**30,
        compute_capability="12.1",
    )
    gh200 = GpuDeviceFact(
        vendor="nvidia",
        name="NVIDIA GH200",
        detection_source="nvml",
        vram_total_bytes=96 * 2**30,
        compute_capability="9.0",
    )

    assert provisioning._cuda_capability_ok(make_facts(gpus=(gb10,))) is True
    assert provisioning._cuda_capability_ok(make_facts(gpus=(gh200,))) is False


def test_cuda_wheel_install_gates_on_capability_and_uv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wheel install never runs for silicon the wheel cannot serve.

    A pre-Ampere GPU (or NVML-degraded unknown capability) must not pull a
    multi-GB wheel that would fail at model load; a missing uv degrades with
    the manual remediation logged instead of crashing provisioning.
    """
    def _unexpected_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("uv must not run for gated-out facts")

    monkeypatch.setattr(provisioning.subprocess, "run", _unexpected_run)
    old_gpu = NVIDIA_A40.model_copy(update={"compute_capability": "7.5"})
    assert not _REAL_TRY_INSTALL_CUDA_WHEEL(make_facts(gpus=(old_gpu,)))

    # Capability fine, but no uv on PATH: degrade, do not crash.
    def _no_uv(_name: str) -> None:
        return None

    monkeypatch.setattr(provisioning.shutil, "which", _no_uv)
    assert not _REAL_TRY_INSTALL_CUDA_WHEEL(make_facts(gpus=(NVIDIA_A40,)))


def test_cuda_wheel_install_sanitizes_index_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env-level index overrides cannot redirect the engine wheel install.

    UV_INDEX (and pip equivalents) outrank CLI flags under uv's first-index
    strategy, so a host-level mirror could serve a stale or non-Foxlight
    build despite the explicit pins; the install must run with those
    stripped (PR #665 review).
    """
    seen_env: dict[str, str] = {}
    seen_argv: list[str] = []

    class _Completed:
        returncode = 1
        stderr = "stop here"

    def _capture_run(*args: object, **kwargs: object) -> "_Completed":
        env = kwargs.get("env")
        assert isinstance(env, dict)
        seen_env.update(cast(dict[str, str], env))
        argv = args[0]
        assert isinstance(argv, list)
        seen_argv.extend(cast(list[str], argv))
        return _Completed()

    def _uv_on_path(_name: str) -> str:
        return "/usr/bin/uv"

    monkeypatch.setenv("UV_INDEX", "https://mirror.corp.example/simple")
    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://mirror.corp.example/simple")
    monkeypatch.setenv("UV_CONFIG_FILE", "/etc/uv/corp.toml")
    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.corp.example/simple")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://mirror.corp.example/x")
    monkeypatch.setattr(provisioning.shutil, "which", _uv_on_path)
    monkeypatch.setattr(provisioning.subprocess, "run", _capture_run)

    assert not _REAL_TRY_INSTALL_CUDA_WHEEL(make_facts(gpus=(NVIDIA_A40,)))
    assert seen_env, "install subprocess never ran"
    for forbidden in (
        "UV_INDEX",
        "UV_DEFAULT_INDEX",
        "UV_CONFIG_FILE",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
    ):
        assert forbidden not in seen_env
    # Config-file discovery (project pyproject.toml/uv.toml) is the other
    # index side channel; the install must opt out of it entirely.
    assert "--no-config" in seen_argv
    pin = provisioning.LLAMA_SERVER_PIN.removeprefix("b")
    minimum = provisioning.LLAMA_SERVER_CUDA_MIN_REVISION
    assert f"skulk-llama-server-cuda==0.{pin}.*,>=0.{pin}.{minimum}" in seen_argv


def test_cuda_wheel_install_degrades_when_uv_cannot_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A which()-found uv that fails to exec degrades, never crashes.

    A stale path, stripped exec bit, or exec-format error surfaces from
    subprocess.run as OSError; letting it propagate would break
    ensure_llama_server's degrade-never-crash startup contract
    (PR #665 review).
    """

    def _uv_on_path(_name: str) -> str:
        return "/usr/bin/uv"

    def _cannot_exec(*args: object, **kwargs: object) -> None:
        raise OSError(8, "Exec format error")

    monkeypatch.setattr(provisioning.shutil, "which", _uv_on_path)
    monkeypatch.setattr(provisioning.subprocess, "run", _cannot_exec)
    assert not _REAL_TRY_INSTALL_CUDA_WHEEL(make_facts(gpus=(NVIDIA_A40,)))


def test_ensure_installs_cuda_wheel_on_demand(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bare-install NVIDIA node completes the CUDA lane on demand (#661).

    The tarball chain has no CUDA artifact by design, so without this an
    NVIDIA container degrades through Vulkan (no ICD) to a CPU-tagged
    engine. ensure_llama_server must attempt the wheel install once and wire
    the shim it produces.
    """
    _isolate_engines_dir(monkeypatch, tmp_path)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    facts = make_facts(gpus=(NVIDIA_A40,))

    shim = tmp_path / "llama-server-cuda"
    shim.write_text("#!/bin/sh\n")
    shim.chmod(0o755)
    installed = {"done": False}

    def _fake_install(installed_facts: object) -> bool:
        installed["done"] = True
        return True

    def _wheel_after_install(
        vendor: str, wheel_facts: object
    ) -> tuple[Path, Path | None] | None:
        return (shim, None) if installed["done"] else None

    def _cuda_usable_tracks_install() -> bool:
        # Stubbed so the test is hermetic: on a host that actually has the
        # pin-matched CUDA wheel installed (the documented NVIDIA install
        # path), the real probe would return True and skip the on-demand
        # install this test exists to exercise (PR #665 review).
        return installed["done"]

    monkeypatch.setattr(provisioning, "try_install_cuda_wheel", _fake_install)
    monkeypatch.setattr(provisioning, "wheel_llama_server", _wheel_after_install)
    monkeypatch.setattr(
        provisioning, "_cuda_wheel_usable", _cuda_usable_tracks_install
    )

    assert ensure_llama_server(facts) == shim
    assert installed["done"]
    assert os.environ.get(LLAMA_SERVER_BIN_ENV) == str(shim)


def test_ensure_skips_wheel_install_when_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--offline nodes never attempt the network wheel install."""
    _isolate_engines_dir(monkeypatch, tmp_path)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)

    def _unexpected_install(installed_facts: object) -> bool:
        raise AssertionError("offline must not attempt a wheel install")

    monkeypatch.setattr(provisioning, "try_install_cuda_wheel", _unexpected_install)
    def _no_wheel(vendor: str, wheel_facts: object) -> None:
        return None

    monkeypatch.setattr(provisioning, "wheel_llama_server", _no_wheel)

    assert ensure_llama_server(make_facts(gpus=(NVIDIA_A40,)), allow_download=False) is None


def test_installed_vulkan_wheel_does_not_shadow_cuda_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An NVIDIA node with only the Vulkan wheel still completes the CUDA lane.

    The Vulkan shim resolves happily but cannot drive the GPU in a
    compute-only container; the on-demand install keys on the CUDA wheel
    itself, not on any-wheel-resolved (PR #665 review).
    """
    _isolate_engines_dir(monkeypatch, tmp_path)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    facts = make_facts(gpus=(NVIDIA_A40,))

    vulkan_shim = tmp_path / "llama-server-vulkan"
    cuda_shim = tmp_path / "llama-server-cuda"
    for shim in (vulkan_shim, cuda_shim):
        shim.write_text("#!/bin/sh\n")
        shim.chmod(0o755)
    installed = {"done": False}

    def _fake_cuda_usable() -> bool:
        return installed["done"]

    def _fake_install(installed_facts: object) -> bool:
        installed["done"] = True
        return True

    def _resolver(vendor: str, wheel_facts: object) -> tuple[Path, None]:
        return ((cuda_shim, None) if installed["done"] else (vulkan_shim, None))

    monkeypatch.setattr(provisioning, "_cuda_wheel_usable", _fake_cuda_usable)
    monkeypatch.setattr(provisioning, "try_install_cuda_wheel", _fake_install)
    monkeypatch.setattr(provisioning, "wheel_llama_server", _resolver)

    assert ensure_llama_server(facts) == cuda_shim
    assert installed["done"]


def test_install_success_requires_usable_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-exit install that lands no usable wheel is not success.

    uv resolving a different interpreter (or a broken wheel) must degrade to
    the Vulkan/tarball chain rather than letting the caller believe the CUDA
    lane is live (#661 review round).
    """

    class _Completed:
        returncode = 0
        stderr = ""

    def _fake_run(*args: object, **kwargs: object) -> "_Completed":
        return _Completed()

    def _fake_which(_name: str) -> str:
        return "/usr/bin/uv"

    def _still_unusable() -> bool:
        return False

    monkeypatch.setattr(provisioning.subprocess, "run", _fake_run)
    monkeypatch.setattr(provisioning.shutil, "which", _fake_which)
    monkeypatch.setattr(provisioning, "_cuda_wheel_usable", _still_unusable)
    assert not _REAL_TRY_INSTALL_CUDA_WHEEL(make_facts(gpus=(NVIDIA_A40,)))


@pytest.mark.parametrize(
    ("distribution", "installed", "accepted"),
    [
        ("skulk-llama-server-cuda", "0.10753.0", False),
        ("skulk-llama-server-cuda", "0.10753.1", True),
        ("skulk-llama-server-cuda", "0.10753.2", True),
        ("skulk-llama-server-cuda", "0.10753.1rc1", False),
        ("skulk-llama-server-cuda", "0.10754.1", False),
        ("skulk-llama-server-cuda", "malformed", False),
        ("skulk-llama-server-vulkan", "0.10753.0", True),
    ],
)
def test_engine_selection_rejects_broken_cuda_revision(
    monkeypatch: pytest.MonkeyPatch, distribution: str, installed: str, accepted: bool
) -> None:
    """An already installed old wheel must not bypass the corrected package floor."""
    def version(name: str) -> str:
        return installed

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert provisioning._wheel_version_matches_pin(distribution) is accepted
