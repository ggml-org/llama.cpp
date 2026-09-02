# llama.cpp RDNA2 / RDNA3

This fork keeps the RDNA2/gfx1030 and RDNA3/gfx1100 paths separate. Use only
the section matching the GPU you are building for.
Unsupported models, shapes, quantizations, and topologies retain the normal
llama.cpp fallbacks.

## At a glance

| GPU | Build helper | Runtime profile |
|---|---|---|
| Native RDNA3/gfx1100 | `scripts/build-rdna3-portable.sh` | `GGML_HIP_RDNA3_AUTO=1` with the RCCL-enabled build |
| RDNA2/gfx1030/V620 | `scripts/build-rdna2-portable.sh` | `HSA_OVERRIDE_GFX_VERSION=10.3.0` plus the RDNA2 profile |

`HSA_OVERRIDE_GFX_VERSION=10.3.0` is **never** used on native gfx1100.

## RDNA3: native gfx1100

The RDNA3 profile is qualified for two or more matching AMD Radeon RX 7900 XT /
`gfx1100` cards. The launcher selects all matching cards by default. For Auto to
activate, every visible physical GPU must match and every pair must support
bidirectional peer access. The current machine has two cards, so runtime
measurements for a 3+ card topology remain to be collected on suitable hardware.

### Build

The portable helper discovers ROCm and clang, detects one gfx11 target, builds
the server and speculative sidecars, and disables embedded/prebuilt UI assets:

```bash
./scripts/build-rdna3-portable.sh
```

Defaults are `GGML_HIP_RCCL=ON`, `BUILD_SIDECARS=ON`, and `BUILD_TESTS=OFF`.
If discovery is ambiguous, provide the installation and target explicitly:

```bash
ROCM_PATH=/opt/rocm/core-10.0 \
TARGET_ARCH=gfx1100 \
./scripts/build-rdna3-portable.sh --jobs 2
```

The lower-level `scripts/build-rdna-unified.sh` helper remains available for
maintainer-controlled builds; set `GGML_HIP_RCCL=ON` when using it for RDNA3
Auto.

### Model and sidecar assets

The supplied launcher requires the verified Qwen3.8-27B Q4 model, projector,
and sidecar assets. Set `MODEL_DIR` if they are not under the default model
location. The pinned model revision is:

```text
04a41723de3622e56bb499676ebaaacaa430f345
Qwen3.8-27B-Q4_0-AutoRound-Code.gguf  6f02e53c762a4a29a795a2346704c07f35c8a8ae7b74967aa1c0fda6bf047100
mmproj-model.gguf                     9da757136cb044abdf552334c56f2dcb63839799ea54c705ba4bcee807abdad2
```

With `SPEC_SIDECAR=1`, the server detects the embedded MTP block and prepares
the required 40,960-row sidecar assets natively on first start. Later starts
reuse the validated cache under `${LLAMA_CACHE:-~/.cache/llama.cpp}/spec-sidecar`.
No Python preparation step is required; use `--spec-sidecar-cache /path` to
choose a different cache root. The launcher uses this automatic path unless an
existing explicit bundle is supplied with `--bundle`.

### Launch: all available RDNA3 options

This enables the available RDNA3 options: RCCL/direct-P2P auto policy, sidecar
MTP plus ngram drafting (launcher default), experimental chunked GDN, and the
validated but default-off Add+RMSNorm fusion.

```bash
GGML_HIP_RDNA3_AUTO=1 \
./scripts/run-qwen38-rdna-unified.sh \
  --build-dir build-gfx1100-portable \
  --profile experimental \
  --gfx1100-add-rms-fusion
```

For the conservative production profile, leave the two experimental options
out. The sidecar and ngram path remains enabled by default:

```bash
GGML_HIP_RDNA3_AUTO=1 \
./scripts/run-qwen38-rdna-unified.sh \
  --build-dir build-gfx1100-portable \
  --profile safe
```

The launcher dynamically verifies the RX 7900 XT identity tuple, uses all
matching GPUs by default, and generates the tensor split. Set
`REQUIRE_GPUS=N` only when an exact number of matching GPUs is wanted. At very
large context sizes it keeps the vision projector on the CPU to avoid GPU VRAM
exhaustion.

### RDNA3 optimization status

| Optimization | Status and activation | Scope / notes |
|---|---|---|
| RCCL AllReduce and direct P2P | **Automatic** with `GGML_HIP_RDNA3_AUTO=1` and an RCCL build | Defaults only unset `GGML_CUDA_ALLREDUCE=nccl`, `GGML_CUDA_P2P=1`, and `NCCL_P2P_DISABLE=0`; RCCL level, algorithm, protocol, and channel tuning stay on Auto. The preliminary TP2 host-snapshot candidate remains explicit via `GGML_HIP_P2P_ALLREDUCE=1` after gfx1100 A/B testing found it slower than RCCL. |
| gfx11 MMQ / WMMA | **Automatic** in a gfx11 build | Qualified Q4 prompt kernels and compatible F16 flash attention use the compiled gfx11 paths; no runtime switch is needed. |
| gfx1100 flash-attention launch shapes | **Automatic** | Selected by architecture and shape under the safe profile. |
| Q8_0 MMVQ VDR=4 | **Automatic** for native gfx1100 | Passed backend correctness and shape A/B tests on both RX 7900 XT cards; this is not end-to-end Q8 GGUF validation. |
| MTP sidecar + ngram drafting | **Automatic** in the launcher | First start prepares and caches the embedded MTP assets natively; exact 262K native/in-process MTP does not fit, so the HIP sidecar is the validated path. |
| TP output-head sharding | Explicit `GGML_TP_SHARDED_OUTPUT=1` | Supported Qwen35/Qwen35MoE models can shard the primary output head along vocabulary rows and avoid the full-logit output AllReduce for CPU/sidecar sampling. If `--backend-sampling` is enabled at model load, the primary head safely retains hidden-axis/full-logit output so target backend sampling works; vocabulary-axis sharding and target backend sampling are not combined yet. Sidecar-local sampling remains available. Unset/`auto` retain hidden-axis/full-logit output. |
| Sidecar n-gram verification cap | **Automatic** for MTP-sidecar stacks | Caps n-gram proposals at the configured MTP width (the launcher uses 3) to avoid oversized target verification bursts; explicit `speculative.n_max` remains authoritative. |
| Chunked GDN prefill | `--profile experimental` | Default-off in `safe`; keep experimental until workload-specific validation is complete. |
| Add+RMSNorm+MUL fusion | `--gfx1100-add-rms-fusion` | Exact output parity; prompt-heavy throughput improved historically, while decode was effectively neutral. Default-off. |

The following are intentionally not RDNA3 defaults: the retired MMVQ wide-load
and native-dot8/Q8-cache trials, and the block-08 Q8_1/SSM fusion pending exact
parity testing. Do not set `HSA_OVERRIDE_GFX_VERSION` on gfx1100.

### RDNA3 runtime rules

- Use the single `GGML_HIP_RDNA3_AUTO=1` runtime opt-in; manual NCCL/RCCL
  variables are not required.
- Explicit values always win. For example, `NCCL_P2P_DISABLE=1` remains a
  request to disable RCCL P2P.
- The auto profile activates only when all visible physical devices are identical
  native RX 7900 XT/`gfx1100` cards, there are at least two, every pair has
  bidirectional peer access, and `GGML_HIP_RCCL` is compiled in. Mixed, virtual,
  or partial-peer topologies stay on the safe generic behavior.
- Do not force `NCCL_P2P_LEVEL=PXB`, `NCCL_ALGO`, or `NCCL_PROTO` on this topology;
  RCCL Auto selected the tested direct transport.
- `GGML_TP_SHARDED_OUTPUT=1` is an explicit Qwen35/Qwen35MoE output policy: it
  selects vocabulary-axis primary output for CPU/sidecar sampling. When
  `--backend-sampling` is enabled at model load, the loader instead selects the
  hidden-axis/full-logit primary policy so target backend sampling remains
  usable; the two output layouts are not combined yet. Sidecar-local sampling
  is unaffected. Unset/`auto` preserve the normal hidden-axis/full-logit policy.
- With the launcher’s stacked MTP+K4V configuration, sidecar n-gram proposals
  are capped at the configured `--spec-draft-n-max` width (the launcher uses 3)
  to avoid oversized target verification bursts. The same fixed cap applies to
  the other sidecar n-gram stacks without changing their command-line settings.
  Explicit request `speculative.n_max` remains authoritative and intentionally
  bypasses this internal sidecar cap.

## RDNA2: gfx1030 / V620

Use this section only for native RDNA2/gfx1030 systems, especially the validated
four-V620 topology.

### Build and launch

```bash
./scripts/build-rdna2-portable.sh
```

For the tested V620 native profile:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
GGML_HIP_RDNA2_AUTO=1 \
HSA_NO_SCRATCH_RECLAIM=1 \
GGML_HIP_SAFE_STATE_IO=1 \
./build/bin/llama-server \
  -m /path/to/main.gguf \
  -ngl all \
  --split-mode tensor \
  --tensor-split 1,1,1,1 \
  --flash-attn on \
  --host 0.0.0.0 \
  --port 8080
```

The RDNA2 auto control is a legacy broad profile: it is enabled by default
unless disabled with `GGML_HIP_RDNA2_AUTO=0`, while
`HSA_OVERRIDE_GFX_VERSION=10.3.0` selects the tested gfx1030 kernel profile.
Explicit `GGML_HIP_GFX1030_*` variables remain per-feature overrides.

### RDNA2 optimization status

| Optimization | Status and activation | Scope / notes |
|---|---|---|
| Native RDNA2 kernel profile | `HSA_OVERRIDE_GFX_VERSION=10.3.0` | Q4_0 DOT8 MMVQ, native tiled flash attention, and chunked GDN with architecture/shape fallbacks. |
| Routed MMQ and Q4_K/Q6_K MMVQ | **Automatic** | Validated expert-width and conservative six-row dispatch policies. |
| MTP/DFlash rows2 width-eight paths | **Automatic** for eligible shapes | Q4_K/Q6_K/MXFP4 paths; `GGML_HIP_GFX1030_MMVQ_W8_ROWS2` is an override only. |
| MXFP4/NVFP4 native arithmetic | **Automatic** for qualified shapes | Unsupported widths retain normal kernels. |
| Muse Q8_0 MMVQ | **Automatic** for the validated shape | Restricted to its qualified `K=6656, N=128` case. |
| Q8_1 activation reuse and fusion | **Automatic** on RDNA2 | Includes the graph-owned cache and eligible routed projection staging. It is structurally disabled on RDNA3. |
| GDN sibling projection fusion | **Automatic** when eligible | Applies to validated Qwen MoE loader/model conditions, not dense Qwen3.8-27B. |
| V620 topology/P2P/RCCL policy | **Automatic** on the qualified topology | `GGML_HIP_GFX1030_P2P_ALLREDUCE=auto-expanded` controls the certified TP4 host-snapshot experiment. Qualified TP2 uses the shared two-rank host-snapshot candidate under the RDNA2 Auto policy. |
| TP output-head sharding | Explicit `GGML_TP_SHARDED_OUTPUT=1` | Uses vocabulary-axis output for CPU/sidecar sampling; an explicit backend-sampling request retains hidden-axis/full logits instead. |
| Automatic MTP/DFlash assets | `SPEC_SIDECAR=1` | Detects supported runtime GGUFs, creates a validated first-start cache, and reuses it on warm starts. |

## Shared rules

- The launcher and build helpers use architecture-specific paths. Build gfx1030
  and gfx1100 separately; do not reuse a HIP build directory across targets.
- The launcher matches complete GPU identity tuples and never hard-codes PCI
  addresses or ordinals.
- The server requires `SOURCE.txt` and `SHA256SUMS` evidence beside the verified
  model. Do not bypass those integrity checks.
- After an ROCm illegal-memory fault, reset the affected GPUs or reboot before
  trusting subsequent measurements.

For general llama.cpp documentation and releases, see the
[upstream llama.cpp repository](https://github.com/ggml-org/llama.cpp).
