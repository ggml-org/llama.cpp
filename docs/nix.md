# Nix Build System Documentation

This document describes the Nix flake infrastructure for building llama.cpp with various GPU backends.

## Overview

The llama.cpp Nix build system uses [flake-parts](https://github.com/hercules-ci/flake-parts) to provide a modular, multi-platform build configuration. It supports:

- **CPU**: Default build with BLAS acceleration
- **CUDA**: NVIDIA GPU acceleration
- **ROCm**: AMD GPU acceleration (including gfx906/MI50/MI60 optimizations)
- **Vulkan**: Cross-platform GPU via Vulkan
- **Metal**: Apple Silicon acceleration

## Quick Start

```bash
# Build the default (CPU) package
nix build .#default

# Build with ROCm support (AMD GPUs — all architectures)
nix build .#rocm

# Build for a specific AMD GPU (faster compilation)
nix build .#rocm-gfx906   # MI50, MI60
nix build .#rocm-gfx1100  # Radeon RX 7900 XTX/XT

# Build with CUDA support (NVIDIA GPUs)
nix build .#cuda

# Build with Vulkan support
nix build .#vulkan

# Run llama-cli directly
nix run .#llama-cli -- --help

# Enter a development shell
nix develop .#rocm
```

## File Structure

```
.
├── flake.nix                      # Main flake entry point
├── flake.lock                     # Pinned dependency versions
└── .devops/nix/
    ├── package.nix                # Main llama-cpp derivation (orchestration)
    ├── backends.nix               # Hardware backend definitions (BLAS, CUDA, ROCm, etc.)
    ├── scope.nix                  # Package scope (makeScope)
    ├── nixpkgs-instances.nix      # CUDA/ROCm nixpkgs configurations
    ├── apps.nix                   # Runnable applications
    ├── devshells.nix              # Development shells
    ├── jetson-support.nix         # NVIDIA Jetson support
    ├── docker.nix                 # Docker image builds
    ├── sif.nix                    # Singularity image builds
    ├── package-gguf-py.nix        # Python gguf package
    └── python-scripts.nix         # Python conversion scripts
```

## Architecture

### backends.nix — Modular Hardware Backends

Each hardware backend (BLAS, CUDA, ROCm, Metal, Vulkan, MPI) is defined as a self-contained function in `backends.nix`. Each backend takes its specific dependencies and returns a structured config:

```nix
{
  suffix = "ROCm";          # Used for pname suffix and description
  buildInputs = [ ... ];     # Libraries needed at build/runtime
  nativeBuildInputs = [ ... ]; # Tools needed only at build time
  cmakeFlags = [ ... ];       # Backend-specific CMake flags
  env = { ... };              # Environment variables for the build
}
```

`package.nix` imports these definitions and uses `lib.foldl'` to merge all active backends into a single config. This means:

- **Adding a new backend** = add one entry to `backends.nix`
- **package.nix** stays focused on orchestration (source filtering, shared cmake flags, meta)
- **Darwin platform deps** (Accelerate, CoreVideo, CoreGraphics) remain in package.nix as they're platform deps, not a backend

### nixpkgs Multi-Instance Rationale

The flake creates separate nixpkgs instances for CUDA (`pkgsCuda`) and ROCm (`pkgsRocm`) in `nixpkgs-instances.nix`. This is architecturally necessary because:

- `config.cudaSupport = true` and `config.rocmSupport = true` propagate through the **entire** dependency tree
- For example, `openmpi`, `ucc`, and `ucx` all need to be built with CUDA/ROCm support when the top-level package uses them
- Overlays cannot achieve this — they modify individual packages, not the config that flows into every `callPackage` invocation
- This is a fundamental nixpkgs constraint, not an oversight

See [1000 instances of nixpkgs](https://zimbatm.com/notes/1000-instances-of-nixpkgs) for background.

### effectiveStdenv Pattern

`package.nix` shadows the auto-injected `stdenv` with a `throw` to force use of `effectiveStdenv`. This is a valid nixpkgs pattern: `callPackage` auto-injects `stdenv`, but CUDA builds require `backendStdenv` for consistent libstdc++. The throw catches accidental uses at evaluation time rather than producing cryptic build failures.

### Sandbox Escape (__noChroot)

The `__noChroot = true` attribute is a last-resort Nix sandbox escape, used only when `precompileMetalShaders` is enabled on Darwin. The Metal compiler (`xcrun`) lives outside the Nix sandbox at a variable system path. This escape grants read access to Apple's toolchain for deterministic shader compilation. See [PR #6118](https://github.com/ggml-org/llama.cpp/pull/6118) for discussion.

### Version Management

The flake uses `self.dirtyShortRev or self.shortRev or "0.0.0"` for version strings. This gives meaningful versions for dev builds (based on the git revision) while falling back to `"0.0.0"` for builds outside a git repo. Rebuilds only trigger when the git tree actually changes.

## Flake Outputs

### Packages

| Package | Description |
|---------|-------------|
| `default` | CPU build with BLAS |
| `rocm` | AMD ROCm/HIP build — all GPU architectures (x86_64-linux only) |
| `rocm-gfx906` | ROCm build for MI50, MI60 |
| `rocm-gfx908` | ROCm build for MI100 |
| `rocm-gfx90a` | ROCm build for MI210, MI250, MI250X |
| `rocm-gfx942` | ROCm build for MI300A, MI300X, MI325X |
| `rocm-gfx1030` | ROCm build for Radeon PRO W6800, Radeon PRO V620 |
| `rocm-gfx1100` | ROCm build for Radeon RX 7900 XTX, Radeon RX 7900 XT |
| `rocm-gfx1101` | ROCm build for Radeon RX 7800 XT, Radeon RX 7700 XT, Radeon PRO W7700 |
| `rocm-gfx1200` | ROCm build for Radeon RX 9060 XT |
| `rocm-gfx1201` | ROCm build for Radeon RX 9070 XT, Radeon RX 9070 |
| `cuda` | NVIDIA CUDA build (Linux only) |
| `vulkan` | Vulkan backend |
| `mpi-cpu` | CPU build with MPI for distributed inference (Linux only) |
| `mpi-cuda` | CUDA build with MPI for distributed inference (Linux only) |
| `windows` | Cross-compiled Windows build |
| `python-scripts` | Python conversion utilities |
| `docker-rocm` | Docker image — all ROCm GPU architectures |
| `docker-rocm-gfx906` | Docker image for MI50, MI60 |
| `docker-rocm-gfx908` | Docker image for MI100 |
| `docker-rocm-gfx90a` | Docker image for MI210, MI250, MI250X |
| `docker-rocm-gfx942` | Docker image for MI300A, MI300X, MI325X |
| `docker-rocm-gfx1030` | Docker image for Radeon PRO W6800, Radeon PRO V620 |
| `docker-rocm-gfx1100` | Docker image for Radeon RX 7900 XTX, Radeon RX 7900 XT |
| `docker-rocm-gfx1101` | Docker image for Radeon RX 7800 XT, Radeon RX 7700 XT, Radeon PRO W7700 |
| `docker-rocm-gfx1200` | Docker image for Radeon RX 9060 XT |
| `docker-rocm-gfx1201` | Docker image for Radeon RX 9070 XT, Radeon RX 9070 |

### Docker Images

Build a GPU-specific Docker image and load it into Docker:

```bash
# Build the container (reuses cached llama-cpp from the nix store)
nix build .#docker-rocm-gfx906

# Load into Docker
docker load < ./result

# Run llama-server inside the container
docker run --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -p 8080:8080 \
  llama-cpp-rocm:latest \
  llama-server -hf Qwen/Qwen3-0.6B-GGUF -ngl 99 --host 0.0.0.0 --port 8080
```

The images are built with `dockerTools.buildLayeredImage`, so the llama-cpp derivation is shared as a layer — if you've already built the package, the Docker image build only takes seconds.

### Apps

Run directly with `nix run`:

```bash
nix run .#llama-cli
nix run .#llama-server
nix run .#llama-quantize
nix run .#llama-embedding
```

### Development Shells

Each package has a corresponding dev shell, plus an `-extra` variant that includes python conversion scripts and tiktoken:

```bash
nix develop .#default      # CPU development
nix develop .#rocm         # ROCm development
nix develop .#cuda         # CUDA development

# -extra variants include python-scripts and tiktoken
nix develop .#rocm-extra   # ROCm + python tools
nix develop .#cuda-extra   # CUDA + python tools
```

### NixOS Modules

The flake provides a NixOS module for running llama-server as a systemd service:

```nix
# In your NixOS configuration (flake-based):
{
  inputs.llama-cpp-gfx906.url = "github:your-fork/llama.cpp-gfx906";

  outputs = { nixpkgs, llama-cpp-gfx906, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      modules = [
        llama-cpp-gfx906.nixosModules.default
        {
          services.llama-server = {
            enable = true;
            model = "/models/my-model.gguf";
            host = "0.0.0.0";
            port = 8000;
            gpuLayers = 99;
            enableMetrics = true;
          };
        }
      ];
    };
  };
}
```

The module includes systemd hardening (`DynamicUser`, `ProtectSystem`, `NoNewPrivileges`) and grants ROCm GPU access via `/dev/kfd` and `/dev/dri` device rules.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable` | bool | `false` | Enable the llama-server service |
| `package` | package | `pkgs.llama-cpp` | The llama.cpp package to use |
| `model` | path | (required) | Path to GGUF model file |
| `host` | str | `"127.0.0.1"` | Listen address |
| `port` | port | `8000` | Listen port |
| `gpuLayers` | int | `99` | Layers to offload to GPU |
| `contextSize` | null or int | `null` | Context size (model default when null) |
| `parallel` | null or int | `null` | Parallel sequences to decode |
| `apiKey` | null or str | `null` | API key (mutually exclusive with `apiKeyFile`) |
| `apiKeyFile` | null or path | `null` | Path to API key file |
| `enableMetrics` | bool | `false` | Enable Prometheus metrics |
| `extraArgs` | list of str | `[]` | Extra CLI arguments |
| `environment` | attrs of str | `{}` | Environment variables |

## Configuration Details

### package.nix Parameters

The main package accepts these configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `useBlas` | auto | Enable BLAS acceleration |
| `useCuda` | `config.cudaSupport` | Enable CUDA backend |
| `useRocm` | `config.rocmSupport` | Enable ROCm/HIP backend |
| `useVulkan` | `false` | Enable Vulkan backend |
| `useMetalKit` | auto (Darwin) | Enable Metal backend |
| `useMpi` | `false` | Enable MPI for distributed inference |
| `useRpc` | `false` | Enable RPC backend |
| `enableStatic` | platform-dependent | Build static binaries |
| `precompileMetalShaders` | `false` | Precompile Metal shaders (requires sandbox escape) |
| `rocmGpuTargets` | from rocmPackages | Target GPU architectures |

### ROCm/gfx906 Optimizations

This fork includes specific optimizations for AMD MI50/MI60 (gfx906) GPUs, defined in `backends.nix` under the `rocm` backend.

**Universal flags** (applied to all ROCm builds):

```nix
(cmakeBool "GGML_HIP_GRAPHS" true)         # HIP kernel batching
(cmakeBool "GGML_HIP_EXPORT_METRICS" true) # Performance metrics
(cmakeBool "GGML_CUDA_FA" true)            # Flash Attention
(cmakeBool "GGML_CUDA_FA_ALL_QUANTS" true) # FA for all quant types
```

**gfx906-only workarounds** (applied only when `rocmGpuTargets` contains `gfx906`):

```nix
(cmakeBool "GGML_HIP_NO_VMM" true)         # Required for MI50 — disable VMM
(cmakeBool "GGML_CUDA_NO_PEER_COPY" true)  # Disable peer-to-peer GPU copies
```

This means `rocm-gfx1100`, `rocm-gfx942`, and other non-gfx906 builds no longer carry the MI50 workaround flags.

### Running on MI50 (gfx906)

Build the gfx906-specific package and run inference, downloading a model directly from HuggingFace:

```bash
# Build for MI50
nix build .#rocm-gfx906

# Run a quick completion test (downloads the model on first run)
HSA_OVERRIDE_GFX_VERSION=9.0.6 \
ROCR_VISIBLE_DEVICES=1 \
  ./result/bin/llama-completion \
    -hf Qwen/Qwen3-0.6B-GGUF \
    -ngl 99 \
    -p "The capital of France is" \
    -n 32
```

Or start the server for OpenAI-compatible API access:

```bash
HSA_OVERRIDE_GFX_VERSION=9.0.6 \
ROCR_VISIBLE_DEVICES=1 \
  ./result/bin/llama-server \
    -hf Qwen/Qwen3-0.6B-GGUF \
    -ngl 99 \
    -fa on \
    --host 0.0.0.0 \
    --port 8080
```

**Environment variables:**

| Variable | Purpose |
|----------|---------|
| `HSA_OVERRIDE_GFX_VERSION=9.0.6` | Required for MI50 — tells the ROCm runtime the exact GFX ISA version |
| `ROCR_VISIBLE_DEVICES=N` | Select GPU by index (use `rocminfo` to list devices) |
| `HIP_VISIBLE_DEVICES=GPU-xxxx` | Select GPU by UUID (use `rocminfo` to find UUIDs) |

**Typical performance on MI50 (32 GB, Qwen3-0.6B Q8_0):**

```
prompt eval: ~900 tokens/sec
generation:  ~200 tokens/sec
VRAM usage:  ~5.4 GiB (model 604 MiB + KV cache 4.5 GiB at 40k context)
```

The `-hf` flag downloads and caches models to `~/.cache/llama.cpp/`. HTTPS support is provided by OpenSSL via the bundled cpp-httplib library.

**Recommended models for MI50 (32 GB VRAM):**

[Qwen3-Coder 30B](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) — MoE model (30B total, 3B active), strong at coding and agentic tasks:

```bash
HSA_OVERRIDE_GFX_VERSION=9.0.6 \
ROCR_VISIBLE_DEVICES=1 \
  ./result/bin/llama-server \
    -hf unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF \
    -ngl 99 \
    -fa on \
    --host 0.0.0.0 \
    --port 8080 \
    --jinja
```

| Quant | Size | Fits 32 GB? |
|-------|------|-------------|
| Q4_K_M | ~18.6 GB | Yes |
| Q5_K_M | ~21.7 GB | Yes |
| Q6_K | ~25.1 GB | Tight |
| Q8_0 | ~32.5 GB | No |

[GPT-OSS 20B](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) — OpenAI's open-weight MoE model with native MXFP4 quantization (~12 GB regardless of quant level):

```bash
HSA_OVERRIDE_GFX_VERSION=9.0.6 \
ROCR_VISIBLE_DEVICES=1 \
  ./result/bin/llama-server \
    -hf ggml-org/gpt-oss-20b-GGUF \
    -ngl 99 \
    -fa on \
    --host 0.0.0.0 \
    --port 8080 \
    --jinja
```

Both models use Mixture of Experts (MoE) — only a fraction of parameters are active per token, so inference speed is much better than the total parameter count suggests. See the [llama.cpp GPT-OSS guide](https://github.com/ggml-org/llama.cpp/discussions/15396) for more details.

### Using the Overlay

For integration into your own flake:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    llama-cpp-gfx906.url = "github:your-fork/llama.cpp-gfx906";
  };

  outputs = { nixpkgs, llama-cpp-gfx906, ... }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        overlays = [ llama-cpp-gfx906.overlays.default ];
        config.rocmSupport = true;
      };
    in {
      packages.x86_64-linux.default = pkgs.llama-cpp;
    };
}
```

## Building for Specific GPU Architectures

### AMD ROCm

The flake provides named per-GPU targets for faster compilation. Instead of building for all GPU architectures at once (`nix build .#rocm`), pick the target matching your card:

```bash
nix build .#rocm-gfx906   # MI50, MI60
nix build .#rocm-gfx908   # MI100
nix build .#rocm-gfx90a   # MI210, MI250, MI250X
nix build .#rocm-gfx942   # MI300A, MI300X, MI325X
nix build .#rocm-gfx1030  # Radeon PRO W6800, Radeon PRO V620
nix build .#rocm-gfx1100  # Radeon RX 7900 XTX, Radeon RX 7900 XT
nix build .#rocm-gfx1101  # Radeon RX 7800 XT, Radeon RX 7700 XT, Radeon PRO W7700
nix build .#rocm-gfx1200  # Radeon RX 9060 XT
nix build .#rocm-gfx1201  # Radeon RX 9070 XT, Radeon RX 9070
```

For custom multi-architecture builds, you can still use `.override` in your own flake:

```nix
llama-cpp-gfx906.packages.x86_64-linux.rocm.override {
  rocmGpuTargets = "gfx906;gfx90a";
}
```

## Environment Variables

For runtime GPU selection with ROCm builds:

```bash
# Use specific GPU by UUID
HIP_VISIBLE_DEVICES=GPU-xxxxx ./result/bin/llama-cli ...

# Use specific GPU by index
ROCR_VISIBLE_DEVICES=0 ./result/bin/llama-cli ...

# Override GFX version (compatibility mode)
HSA_OVERRIDE_GFX_VERSION=9.0.6 ./result/bin/llama-cli ...
```

---

## Potential Improvements

### 1. Binary Cache Integration

Add cachix or other binary cache for pre-built ROCm packages:

```nix
# In flake.nix, uncomment and configure:
nixConfig = {
  extra-substituters = [
    "https://llama-cpp-gfx906.cachix.org"
  ];
  extra-trusted-public-keys = [
    "llama-cpp-gfx906.cachix.org-1:..."
  ];
};
```

### 2. OpenMP Support

Add OpenMP for better CPU parallelization:

```nix
# In package.nix:
useOpenMP ? true,

# In backends.nix, add an openmp backend:
openmp = { llvmPackages }: {
  suffix = "OpenMP";
  buildInputs = [ llvmPackages.openmp ];
  nativeBuildInputs = [ ];
  cmakeFlags = [ ];
  env = { };
};
```

### 3. Test Suite Integration

Add automated tests to the flake:

```nix
# In flake.nix perSystem:
checks = {
  inherit (config.packages) default rocm;

  # Add actual test runs
  test-inference = pkgs.runCommand "test-inference" {
    buildInputs = [ config.packages.default ];
  } ''
    llama-cli --version > $out
  '';
};
```

---

## References

- [Nix Flakes](https://nixos.wiki/wiki/Flakes)
- [flake-parts documentation](https://flake.parts/)
- [ROCm on NixOS](https://nixos.wiki/wiki/AMD_GPU)
- [llama.cpp upstream](https://github.com/ggml-org/llama.cpp)
- [gfx906 optimization details](../README.md)
