#!/usr/bin/env bash
# Skulk one-command installer (#614 Phase 4).
#
# From a fresh macOS or Linux machine to a working node:
#
#   curl -fsSL https://raw.githubusercontent.com/Foxlight-Foundation/Skulk/main/install.sh | bash
#
# The installer is deliberately thin: it fetches prerequisites (uv, rustup, a
# C toolchain), clones the repo, syncs the environment, builds the dashboard
# with either system Node.js or Skulk's bundled runtime, and then hands off to
# `skulk doctor --fix`, which
# owns all environment intelligence (GPU detection, engine provisioning,
# remediation). Anything the doctor cannot fix is printed with its exact
# consequence and remediation.
#
# Flags / environment:
#   --dir <path>       install location            (default: ~/skulk, or SKULK_INSTALL_DIR)
#   --ref <git-ref>    branch or tag to install    (default: main, or SKULK_INSTALL_REF)
#   --headless         intentionally skip the dashboard build
#   --with-vllm        NVIDIA Linux only: create a dedicated vLLM venv with
#                      Skulk's validated dependency matrix (several GB of
#                      wheels; the concurrency-serving fast path on CUDA)
#
# Re-running is safe: every step is idempotent.

set -euo pipefail

INSTALL_DIR="${SKULK_INSTALL_DIR:-$HOME/skulk}"
INSTALL_REF="${SKULK_INSTALL_REF:-main}"
HEADLESS=0
WITH_VLLM=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir) INSTALL_DIR="$2"; shift 2 ;;
        --ref) INSTALL_REF="$2"; shift 2 ;;
        --headless) HEADLESS=1; shift ;;
        --with-vllm) WITH_VLLM=1; shift ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

log()  { printf '\033[1;36m[skulk-install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[skulk-install]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[skulk-install]\033[0m %s\n' "$*" >&2; exit 1; }

OS="$(uname -s)"
case "$OS" in
    Darwin|Linux) ;;
    *) die "unsupported platform: $OS (Skulk supports macOS and Linux)" ;;
esac

# --- prerequisites ---------------------------------------------------------

need_apt() {
    # Return 0 when we can install system packages non-interactively.
    [[ "$OS" == "Linux" ]] && command -v apt-get >/dev/null 2>&1 \
        && { [[ "$(id -u)" == "0" ]] || command -v sudo >/dev/null 2>&1; }
}

apt_install() {
    if [[ "$(id -u)" == "0" ]]; then
        apt-get update -qq && apt-get install -y -qq "$@"
    else
        sudo apt-get update -qq && sudo apt-get install -y -qq "$@"
    fi
}

if ! command -v git >/dev/null 2>&1; then
    if need_apt; then
        log "installing git"
        apt_install git
    else
        die "git is required; install it and re-run (macOS: xcode-select --install)"
    fi
fi

if ! command -v curl >/dev/null 2>&1; then
    if need_apt; then
        log "installing curl"
        apt_install curl ca-certificates
    else
        die "curl is required; install it and re-run"
    fi
fi

if ! command -v cc >/dev/null 2>&1; then
    # The Rust networking bindings compile from source; that needs a linker.
    if need_apt; then
        log "installing C toolchain (build-essential) for the Rust bindings"
        apt_install build-essential
    elif [[ "$OS" == "Darwin" ]]; then
        die "no C compiler found; run: xcode-select --install, then re-run"
    else
        die "no C compiler found; install your distro's build tools and re-run"
    fi
fi

# A GPU Linux node serves GGUF through the managed Vulkan llama-server build,
# which needs the Vulkan loader; minimal CUDA/ROCm container images often lack
# it while the driver's ICD is present.
if [[ "$OS" == "Linux" ]] && command -v ldconfig >/dev/null 2>&1 \
    && ! ldconfig -p 2>/dev/null | grep -q libvulkan.so.1; then
    if { command -v nvidia-smi >/dev/null 2>&1 || [[ -d /sys/class/drm ]]; } && need_apt; then
        log "installing Vulkan loader (libvulkan1) for GPU serving"
        apt_install libvulkan1 || warn "libvulkan1 install failed; GPU GGUF serving may be unavailable (skulk doctor will report it)"
    fi
fi

if ! command -v cargo >/dev/null 2>&1 && [[ ! -x "$HOME/.cargo/bin/cargo" ]]; then
    log "installing Rust (rustup) for the skulk networking bindings"
    curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
fi
# The rustup installer puts cargo here; make it visible to uv's build backend.
export PATH="$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
    log "installing uv"
    curl -fsSL https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || die "uv installation failed; see https://docs.astral.sh/uv/"

# --- fetch -----------------------------------------------------------------

if [[ -d "$INSTALL_DIR/.git" ]]; then
    log "updating existing checkout at $INSTALL_DIR (ref: $INSTALL_REF)"
    git -C "$INSTALL_DIR" fetch origin "$INSTALL_REF"
    # A tag or remote-only ref may not be checkout-able by name after a bare
    # fetch; FETCH_HEAD always is, keeping re-runs idempotent for any ref.
    git -C "$INSTALL_DIR" checkout "$INSTALL_REF" 2>/dev/null         || git -C "$INSTALL_DIR" checkout --detach FETCH_HEAD
    git -C "$INSTALL_DIR" pull --ff-only origin "$INSTALL_REF" 2>/dev/null || true
else
    log "cloning Skulk into $INSTALL_DIR (ref: $INSTALL_REF)"
    if [[ "$INSTALL_REF" =~ ^[0-9a-fA-F]{40}$ ]]; then
        # `git clone --branch` rejects commit object IDs even though `--ref`
        # promises a git ref. Release qualification pins the exact candidate
        # commit so a moving dev branch cannot change between approval and
        # installation; fetch the object directly and detach at FETCH_HEAD.
        git init "$INSTALL_DIR"
        git -C "$INSTALL_DIR" remote add origin \
            https://github.com/Foxlight-Foundation/Skulk.git
        git -C "$INSTALL_DIR" fetch --depth 1 origin "$INSTALL_REF"
        git -C "$INSTALL_DIR" checkout --detach FETCH_HEAD
    else
        git clone --branch "$INSTALL_REF" \
            https://github.com/Foxlight-Foundation/Skulk.git "$INSTALL_DIR"
    fi
fi
cd "$INSTALL_DIR"
RESOLVED_COMMIT="$(git rev-parse HEAD)"
log "resolved ref $INSTALL_REF to commit $RESOLVED_COMMIT"

# --- filesystem sanity -------------------------------------------------------

# Network filesystems break uv's installer mid-sync ('Stale file handle',
# reproduced on RunPod's /workspace MooseFS volume, #627) and are slow homes
# for a venv regardless. Refuse loudly with the fix instead of failing
# cryptically minutes later; SKULK_INSTALL_ALLOW_NETWORK_FS=1 overrides for
# network mounts known to behave.
if [[ "$OS" == "Linux" ]] && command -v findmnt >/dev/null 2>&1; then
    # $PWD, not $INSTALL_DIR: after the cd above, a relative --dir would
    # resolve against the new working directory and silently skip the guard
    # (PR #640 review).
    FS_TYPE="$(findmnt -n -o FSTYPE --target "$PWD" 2>/dev/null || true)"
    case "$FS_TYPE" in
        nfs|nfs4|cifs|smb3|9p|lustre|ceph|glusterfs|fuse.*|moosefs|mfs)
            if [[ "${SKULK_INSTALL_ALLOW_NETWORK_FS:-}" == "1" ]]; then
                warn "installing on a network filesystem ($FS_TYPE) because SKULK_INSTALL_ALLOW_NETWORK_FS=1; uv sync may fail with stale file handles"
            else
                die "install directory $INSTALL_DIR sits on a network filesystem ($FS_TYPE), which breaks uv's installer (stale file handles) and slows the node. Re-run with --dir on a local disk (container/cloud boxes: e.g. --dir \$HOME/skulk on the container disk, not /workspace), or set SKULK_INSTALL_ALLOW_NETWORK_FS=1 to proceed anyway."
            fi
            ;;
    esac
fi

# --- environment -----------------------------------------------------------

# An existing-but-old uv rejects this project's configuration; upgrade it to
# the repo's declared minimum before syncing so re-runs stay idempotent on
# hosts that already had uv.
UV_MIN="$(grep -oE 'required-version = ">=([0-9.]+)"' pyproject.toml | grep -oE '[0-9.]+' | head -1 || true)"
if [[ -n "$UV_MIN" ]]; then
    UV_HAVE="$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
    if [[ -n "$UV_HAVE" ]] && [[ "$(printf '%s\n%s\n' "$UV_MIN" "$UV_HAVE" | sort -V | head -1)" != "$UV_MIN" ]]; then
        log "upgrading uv ($UV_HAVE -> >=$UV_MIN)"
        uv self update >/dev/null 2>&1 || curl -fsSL https://astral.sh/uv/install.sh | sh
        UV_HAVE="$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
        if [[ -n "$UV_HAVE" ]] && [[ "$(printf '%s\n%s\n' "$UV_MIN" "$UV_HAVE" | sort -V | head -1)" != "$UV_MIN" ]]; then
            die "uv $UV_HAVE is below the required >=$UV_MIN and could not be upgraded (package-manager installs need a manual upgrade)"
        fi
    fi
fi

log "syncing the Python environment (first run compiles the Rust bindings; this can take a few minutes)"
uv sync

# --- engine wheels (Linux GPU) ----------------------------------------------

# The wheel version derives from the CHECKED-OUT ref's engine pin, so
# installing --ref dev after a pin advance pulls the matching wheel instead
# of a hardcoded one the runtime would then ignore as a pin mismatch.
ENGINE_BUILD="$(grep -oE 'LLAMA_SERVER_PIN: Final = "b[0-9]+"' src/skulk/provisioning/manifest.py | grep -oE '[0-9]+' || true)"

# Reject broken packaging revisions even when the engine build itself matches.
CUDA_MIN_REVISION="$(grep -oE 'LLAMA_SERVER_CUDA_MIN_REVISION: Final = [0-9]+' src/skulk/provisioning/manifest.py | grep -oE '[0-9]+' || true)"

# The Foxlight wheel index is the source of truth for engine wheels (the
# CUDA wheel exceeds PyPI's per-file limit); wheels carry sigstore build
# provenance (gh attestation verify <wheel> --owner Foxlight-Foundation).
FOXLIGHT_WHEEL_INDEX="https://wheels.foxlight.ai/simple/"
# UV SEMANTICS, verified live: uv consults --extra-index-url indexes BEFORE
# the --index-url default (the opposite of pip), and its default
# first-index-wins strategy is the dependency-confusion defense. So the
# Foxlight index is passed as the extra index (consulted first, wins for the
# packages it carries) while PyPI stays the default for the NVIDIA runtime
# dependencies. Making Foxlight the --index-url instead DEMOTES it under uv:
# resolution then finds the empty PyPI project first and fails.
# The default index is pinned to PyPI explicitly: a host exporting
# UV_INDEX_URL / UV_DEFAULT_INDEX would otherwise silently replace the
# fallback that supplies the NVIDIA runtime dependencies.
ENGINE_INDEX_FLAGS=(--extra-index-url "$FOXLIGHT_WHEEL_INDEX" --index-url "https://pypi.org/simple/")

if [[ "$OS" == "Linux" ]] && [[ -z "$ENGINE_BUILD" || -z "$CUDA_MIN_REVISION" ]]; then
    warn "could not read the engine pin from the checkout; skipping engine wheel install (skulk doctor will report the outcome)"
elif [[ "$OS" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
    log "installing the CUDA llama-server engine wheel (engine build b$ENGINE_BUILD)"
    if ! uv pip install "${ENGINE_INDEX_FLAGS[@]}" "skulk-llama-server-cuda==0.${ENGINE_BUILD}.*,>=0.${ENGINE_BUILD}.${CUDA_MIN_REVISION}"; then
        warn "the CUDA engine wheel is unavailable (index not yet live, no network, or unsupported platform);"
        warn "trying the Vulkan engine wheel (NVIDIA GPUs run the Vulkan build on bare metal)"
        # Mirrors runtime preference: cuda wheel, then vulkan wheel, then the
        # managed tarball path that skulk itself provisions.
        uv pip install "${ENGINE_INDEX_FLAGS[@]}" "skulk-llama-server-vulkan==0.${ENGINE_BUILD}.*" \
            || warn "vulkan wheel also unavailable; falling back to the managed tarball build (skulk doctor will report the outcome)"
    fi
elif [[ "$OS" == "Linux" ]] \
    && compgen -G "/sys/class/drm/card*/device/gpu_busy_percent" > /dev/null 2>&1; then
    log "installing the Vulkan llama-server engine wheel (engine build b$ENGINE_BUILD)"
    if ! uv pip install "${ENGINE_INDEX_FLAGS[@]}" "skulk-llama-server-vulkan==0.${ENGINE_BUILD}.*"; then
        warn "skulk-llama-server-vulkan unavailable (index not yet live or no network);"
        warn "falling back to the managed tarball build; skulk doctor will report the outcome"
    fi
fi

# --- dashboard -------------------------------------------------------------

run_bundled_npm() {
    uv run --project "$INSTALL_DIR" python \
        "$INSTALL_DIR/scripts/run_bundled_npm.py" "$@"
}

if [[ "$HEADLESS" == "1" ]]; then
    log "skipping dashboard build (--headless); the API serves without the web UI"
elif run_bundled_npm --version; then
    log "building the dashboard with Skulk's bundled Node.js runtime"
    (
        cd dashboard-react
        run_bundled_npm install --no-fund --no-audit
        run_bundled_npm run build
    )
elif command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    warn "Skulk's bundled Node.js runtime is unavailable; falling back to the system toolchain"
    (cd dashboard-react && npm install --no-fund --no-audit && npm run build)
else
    die "dashboard build requires Node.js, but neither system node/npm nor Skulk's bundled runtime is usable; re-run with --headless only if this node is intentionally API-only"
fi

# --- vLLM (optional, NVIDIA Linux) ----------------------------------------

if [[ "$WITH_VLLM" == "1" ]]; then
    if [[ "$OS" != "Linux" ]] || ! command -v nvidia-smi >/dev/null 2>&1 \
        || ! nvidia-smi -L 2>/dev/null | grep -q GPU; then
        warn "--with-vllm requested but no NVIDIA GPU is visible on this Linux node; skipping"
    else
        VLLM_ENV="$HOME/.skulk/vllm-env"
        log "installing vLLM into $VLLM_ENV (validated matrix; several GB)"
        # --allow-existing keeps re-runs idempotent (a bare `uv venv` refuses
        # to reuse an existing environment and would fail the second install).
        uv venv "$VLLM_ENV" --python 3.12 --allow-existing
        # vLLM lives in its own venv and Skulk drives its CLI as an external
        # served engine (SKULK_VLLM_BIN). Validated matrix notes:
        # - vLLM's PyPI default wheel links CUDA 13 runtime libraries
        #   (libcudart.so.13), which cannot import on the CUDA 12.x drivers
        #   common on GPU clouds; the cu129 VARIANT wheel from wheels.vllm.ai
        #   is the one that runs there (probe-validated on 0.25.1; 0.28.0
        #   keeps publishing the cu129 variant). 0.25.1 remains the floor
        #   for the DFlash speculator architectures (Laguna cards); 0.28.0
        #   adds DFlash2 drafter checkpoints (selected by the checkpoint,
        #   same "dflash" method string, V2 model runner auto-engaged).
        # - ninja must be resolvable by the vllm server process (FlashInfer
        #   JIT sampling kernels shell out to it); installing it into the
        #   venv suffices because the runner prepends the venv bin dir to the
        #   server's PATH.
        # - torch backend must be cu129 (0.28.0 pairs with torch 2.13): the
        #   cu128 torch index historically failed resolution outright
        #   (torchcodec too old, fresh-box validated on 0.25.1). DFlash
        #   cards additionally JIT their speculator kernels through nvrtc
        #   and need a CUDA >= 12.8 toolchain on the node (12.4 headers
        #   lack __nv_fp8_e8m0).
        # Default index pinned explicitly (mirroring ENGINE_INDEX_FLAGS
        # above): a host exporting UV_INDEX_URL/UV_DEFAULT_INDEX would
        # otherwise redirect dependency resolution to its own mirror.
        uv pip install --python "$VLLM_ENV/bin/python" \
            "vllm==0.28.0+cu129" ninja \
            --extra-index-url "https://wheels.vllm.ai/0.28.0/cu129/" \
            --index-url "https://pypi.org/simple/" \
            --torch-backend=cu129
        mkdir -p "$HOME/.skulk"
        if ! grep -q "SKULK_VLLM_BIN" "$HOME/.skulk/skulk.env" 2>/dev/null; then
            echo "SKULK_VLLM_BIN=$VLLM_ENV/bin/vllm" >> "$HOME/.skulk/skulk.env"
        fi
        export SKULK_VLLM_BIN="$VLLM_ENV/bin/vllm"
        log "vLLM installed; SKULK_VLLM_BIN recorded in ~/.skulk/skulk.env"
        log "The service wrappers (deployment/install) source that file; an"
        log "interactive shell does not, so launch interactively with:"
        log "    SKULK_VLLM_BIN=$VLLM_ENV/bin/vllm uv run skulk"
    fi
fi

# --- model store (single-node default) -------------------------------------

# The documented model flow is store-first (download once into the store,
# stage to workers), but the store API returns 503 "Store not configured" on
# a node without a skulk.yaml (#629). A fresh box therefore starts with this
# host serving ~/.skulk/model-store. When multiple fresh nodes discover one
# another, followers adopt the elected master's store address and stop their
# temporary local store servers, so the cluster converges on one source of
# truth without installer-time inventory. An existing skulk.yaml is never
# touched, so re-runs and operator configs win.
if [[ ! -f skulk.yaml ]]; then
    STORE_PATH="$HOME/.skulk/model-store"
    mkdir -p "$STORE_PATH"
    cat > skulk.yaml <<EOF
# Generated by install.sh: bootstrap model-store defaults so the store-first
# download flow works immediately. Safe to edit or delete. If other fresh
# nodes join, Skulk converges them on the elected master's store. Set an
# explicit shared store_host on every node to override that bootstrap choice
# (see the Model Store page in the docs).
model_store:
  store_host: "$(hostname -s)"
  # Kept outside operating-system dynamic client-port ranges so an unrelated
  # outbound connection cannot claim the listener before Skulk starts.
  store_port: 12415
  # Loopback keeps the single-node client working even when the short
  # hostname is not locally resolvable; on the store host, skulk replaces a
  # loopback literal with its best routable IPv4 before broadcasting to
  # peers, so this stays correct if the node later joins a cluster.
  store_http_host: "127.0.0.1"
  store_path: "$STORE_PATH"
EOF
    log "wrote skulk.yaml: bootstrap model store at $STORE_PATH"
else
    log "existing skulk.yaml found; model store configuration left untouched"
fi

# --- doctor ----------------------------------------------------------------

log "auditing the node (skulk doctor --fix)"
set +e
uv run skulk doctor --fix
DOCTOR_EXIT=$?
set -e

echo
case "$DOCTOR_EXIT" in
    0) log "node is fully healthy." ;;
    2) warn "node works but has DEGRADED findings above; each lists its consequence and fix." ;;
    *) warn "node has FAILED findings above; serving will not work correctly until they are fixed." ;;
esac

log "install complete. Start a node with:"
log "    cd $INSTALL_DIR && uv run skulk"
log "Dashboard (when built): http://localhost:52415"
log "Run as a service: deployment/install/install-systemd.sh (Linux) or install-launchd.sh (macOS)"

# DEGRADED (exit 2) still means a working node; only FAIL blocks the install.
if [[ "$DOCTOR_EXIT" != "0" && "$DOCTOR_EXIT" != "2" ]]; then
    exit 1
fi
exit 0
