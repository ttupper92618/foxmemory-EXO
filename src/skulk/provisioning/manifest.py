"""Pinned engine artifact manifest (#614 Phase 3).

Skulk manages engine binaries the way it manages models: pinned known-good
versions, checksums recorded in-repo, fetched on demand and verified before
use. A new user never builds llama.cpp.

The pin is a specific upstream llama.cpp release tag whose official prebuilt
Linux artifacts we resolve by platform, architecture, and compute variant.
The pinned release publishes Linux CPU, Vulkan, and ROCm builds (its CUDA
prebuilts remain Windows-only). Skulk consumes only CPU and Vulkan: the
Vulkan build drives both AMD (RADV, fleet-proven) and NVIDIA GPUs through
their Vulkan ICDs, so it is the GPU default, and the upstream Linux ROCm
archive stays unconsumed until a ROCm lane is qualified on real hardware. macOS
provisions nothing (in-process MLX owns that platform), and vLLM remains the
first-class CUDA serving path.

Pinning beats "latest" for supply-chain and behavior stability both: the
checksums below make substitution loud, and an upstream behavior change (the
kind that broke pooled HTTP clients in newer builds) arrives only when the pin
is deliberately advanced and re-validated.
"""

from __future__ import annotations

from typing import Final, Literal, final

from pydantic import ConfigDict

from skulk.utils.pydantic_ext import CamelCaseModel

LLAMA_SERVER_PIN: Final = "b10753"
"""The pinned upstream llama.cpp release tag for managed llama-server builds.

Includes the RPC tensor-memset protocol required by current DeepSeek V4
multi-node execution, Qwen 3.8 text and native long-context support, recurrent
state rollback, and served reasoning-effort plumbing. Advancing from b10434
additionally picks up Kimi-K3 text, native MTP speculative decoding
(``--spec-type draft-mtp`` second generation), a gemma4-assistant model fix,
and MTP/next-n loader ordering fixes that matter to the served draft path.
The RPC protocol changed in the b10434 window, so ``llama-server`` and every
``ggml-rpc-server`` donor must always advance together. Advance deliberately,
re-recording checksums and re-running the fresh-box gauntlet.
"""

LLAMA_SERVER_CUDA_MIN_REVISION: Final = 1
"""Minimum CUDA packaging revision for the current engine pin.

Revision 1 fixes missing NCCL and build-host CPU instructions. Reset deliberately
when advancing the engine pin, after validating the new wheel's packaging.
"""

EngineVariant = Literal["cpu", "vulkan", "rocm", "cuda"]
"""Compute variant of a managed llama-server build.

``cuda`` has no upstream Linux prebuilt and is delivered by the Foxlight wheel.
Upstream does publish a Linux ROCm x86_64 archive as of this pin, but Skulk
does not consume it: AMD nodes stay on the fleet-qualified Vulkan wheel or
archive until a ROCm lane is qualified on real Strix hardware. NVIDIA nodes fall through from the
CUDA wheel to Vulkan when necessary (bare-metal drivers ship a working Vulkan
ICD; compute-only container drivers generally do not)."""


@final
class EngineArtifact(CamelCaseModel):
    """One downloadable pinned engine build with its integrity checksum."""

    model_config = ConfigDict(frozen=True)

    asset_name: str
    """Release asset filename."""

    sha256: str
    """Hex SHA-256 of the archive; verification failure aborts provisioning."""

    url_override: str | None = None
    """Full download URL for artifacts not hosted on the upstream release
    (e.g. the Foxlight-built Linux CUDA build, which upstream does not
    publish). ``None`` resolves against the upstream llama.cpp release."""

    def url(self) -> str:
        """The download URL for this artifact."""
        if self.url_override is not None:
            return self.url_override
        return (
            "https://github.com/ggml-org/llama.cpp/releases/download/"
            f"{LLAMA_SERVER_PIN}/{self.asset_name}"
        )


# (machine, variant) -> artifact, for sys.platform == "linux". Checksums are
# the upstream release asset digests, recorded 2026-09-02 (verified by
# streaming each asset and comparing against the release API digest).
LLAMA_SERVER_ARTIFACTS: Final[dict[tuple[str, EngineVariant], EngineArtifact]] = {
    ("x86_64", "cpu"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-x64.tar.gz",
        sha256="a25f023c1c68bafb315ada095fa7780e286d5867783e5eebd7dfc1e36eb1a856",
    ),
    ("x86_64", "vulkan"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-vulkan-x64.tar.gz",
        sha256="30362addb83f0d1275a608c2cc9521d2b2d9a3596704aacebaf1294f94aa91e3",
    ),
    ("aarch64", "cpu"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-arm64.tar.gz",
        sha256="4224302b9bdb52b3fdfbf2439c320d6f51e3a3afc48500d4c1a5f9a76623d2ae",
    ),
    ("aarch64", "vulkan"): EngineArtifact(
        asset_name=f"llama-{LLAMA_SERVER_PIN}-bin-ubuntu-vulkan-arm64.tar.gz",
        sha256="cba2f4a533c77a0bc5e0bcc13d4ac1129f941ba784be915196acbb403c1b2ffa",
    ),
}
