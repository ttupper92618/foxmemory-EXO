# skulk-llama-server-cuda

The pip-installable CUDA `llama-server` build for [Skulk](https://github.com/Foxlight-Foundation/Skulk)'s served GGUF engine on Linux x86_64 and aarch64.

The wheel carries the Foxlight-built `llama-server` and `ggml-rpc-server` binaries, compiled from the pinned upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) release with CUDA enabled (upstream publishes no Linux CUDA prebuilt). The CUDA runtime is not rehosted here: it resolves from NVIDIA's official PyPI wheels (`nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`, `nvidia-nccl-cu12`), which install as ordinary dependencies. NCCL is required to load the pinned CUDA binary even for single-device inference. The `llama-server-cuda` entry point puts those libraries on the loader path and execs the real binary, forwarding all arguments.

```bash
uv pip install --extra-index-url https://wheels.foxlight.ai/simple/ \
  skulk-llama-server-cuda
llama-server-cuda --list-devices
```

The index flag is required: this wheel is published only to the Foxlight index (its payload exceeds PyPI's per-file limit), while its NVIDIA runtime dependencies still resolve from PyPI.

Skulk's engine provisioning discovers the installed wheel automatically and wires it as the node's served engine; no configuration is needed. A machine additionally needs the NVIDIA driver (anything where `nvidia-smi` works), which only NVIDIA can ship.

The CPU backend is built with `GGML_NATIVE=OFF`, so installation does not
inherit CPU instruction requirements from the CI build host.

The x86_64 wheel carries kernels for the established Ampere-through-Hopper fleet. The aarch64 wheel targets compute capability 12.1 with CUDA 12.9 for Grace Blackwell systems such as GB10. The platform-specific filenames share one package version so ordinary Python package resolution selects the correct payload.

Version scheme: `0.<llama.cpp build>.<packaging revision>`; `0.10068.0` is the first packaging of upstream `b10068`. Built and published by the `engine-wheel` workflow to Foxlight's package index with build-provenance attestations.

For a CUDA-only packaging revision, dispatch with `publish=true` and
`publish_variant=cuda`. This publishes both CUDA architectures without replacing
the unchanged Vulkan version with rebuilt bytes.

How this wheel fits into Skulk's install and provisioning flow is documented in the [Build & Runtime Paths guide](https://foxlight-foundation.github.io/Skulk/build-and-runtime/).

The bundled `llama-server` and `ggml-rpc-server` binaries derive from llama.cpp (MIT); its license text ships in the wheel under `skulk_llama_server_cuda/licenses/`.
