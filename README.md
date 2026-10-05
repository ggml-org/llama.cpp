# ggml-llama.cpp (Raudbjorn fork)

> Single-maintainer fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) focused on the TurboQuant+ codec stack and SYCL/Vulkan deployment on Intel Arc A770. Backends shipped: CPU, BLAS, SYCL, Vulkan, OpenVINO. CUDA, ROCm/HIP, Metal, OpenCL, WebGPU, CANN, MUSA, Hexagon, RPC, zDNN, ZenDNN, ET and VirtGPU are not built or supported here.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Maintained by Raudbjorn](https://img.shields.io/badge/maintainer-Raudbjorn-blueviolet.svg)](https://github.com/Raudbjorn)

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) integrating the **TurboQuant+** codec stack -- Walsh-Hadamard rotated polar quantization and layer-aware V compression policies. The codec design, calibration, and validation papers live at [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus); this repository is the llama.cpp runtime integration.

### Lineage -- why the `+`

This project extends that foundation -- adding the asymmetric K/V policy (V is free, K is everything), layer-aware Boundary V protection, the `TQ3_1S` / `TQ4_1S` weight quantization formats, the `turbo2` / `turbo3` / `turbo4` tier variants, the SYCL/Vulkan kernel coverage, and a body of model-family-specific quality and operational fixes. The trailing `+` denotes ongoing extension work; the original TurboQuant codec itself was the Google ICLR 2026 contribution.

This fork is additive within its scope: every shipped backend (CPU, BLAS, SYCL, Vulkan, OpenVINO) continues to work as in upstream, and all TurboQuant+ types (weights, KV cache) are opt-in via the standard `--cache-type-k` / `--cache-type-v` and `llama-quantize` interfaces. Backends not in the shipped set (CUDA, HIP/ROCm, Metal, OpenCL, CANN, MUSA, WebGPU, RPC, Hexagon, zDNN, ZenDNN, ET, VirtGPU) are not built here; pull upstream for those.

## Maintenance

This is a single-maintainer fork. No production deployments are tracked here -- if you deploy this fork somewhere public, please open an issue so the README can be updated.

| | |
|---|---|
| Maintainer | Sveinbjorn Geirsson (Raudbjorn) |
| Default branch | `master` |
| Backends shipped | CPU, BLAS, SYCL, Vulkan, OpenVINO |
| Upstream tracking | continuous sync from `ggml-org/llama.cpp` master |
| Upstream PR status | not yet upstreamed; running as a long-lived fork |


---

## What this fork adds

### Quantization types

| Type | Domain | Approx. bits | Notes | Paper |
|---|---|---|---|---|
| `TQ3_1S` | weights | 4.0 | smaller VRAM than `q8_0`; CPU `vec_dot` and Vulkan kernels; SYCL single-token matvec only (see backend table) | [weight-compression-tq4](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/weight-compression-tq4.md) |
| `TQ4_1S` | weights | 5.0 | smaller VRAM than `q8_0`; CPU `vec_dot` and Vulkan kernels; SYCL single-token matvec only (see backend table) | [weight-compression-tq4](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/weight-compression-tq4.md) |
| `turbo2` | KV cache | 2.125 | aggressive; pair with Boundary V | [block-size-experiment](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/block-size-experiment.md) |
| `turbo3` | KV cache | 3.125 | 5.12x analytic compression vs f16; +5.11% PPL vs q8_0 with turbo3 K and V on Llama-3.1-8B-Instruct Q4_K_M at ctx 512 on this fork (see [turbo3 gate note](docs/research/turbo/turbo3-quality-gate-llama31-8b-2026-09.md)) | [attn-rotation-and-ppl-artifact](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/attn-rotation-and-ppl-artifact.md) |
| `turbo4` | KV cache | 4.25 | rehabilitated to beat `q4_0` on fidelity | [turbo4-resurrection](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/turbo4-resurrection.md) |

Bits per value include the stored scales (block sizes in `ggml/src/ggml-common.h`, table in [docs/turboquant/KV-cache-quantization.md](docs/turboquant/KV-cache-quantization.md)). The KV types use Walsh-Hadamard rotation followed by polar codebook quantization on 128-element blocks; `TQ3_1S` / `TQ4_1S` use WHT-rotated Lloyd-Max codebooks on 32-element blocks. Why this works where MSE-driven codecs fail: [why-mse-fails-for-kv-quantization](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/why-mse-fails-for-kv-quantization.md).

### Compression policies

- **Auto-asymmetric K/V compression** -- V tolerates aggressive compression while K does not; when the same turbo type is requested for K and V on a high-GQA or Qwen-family model, K is rewritten to `q8_0` (see [Automatic behavior](#automatic-behavior)). [asymmetric-kv-compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md)
- **Boundary V (experimental, layer-aware)** -- auto-enabled for `turbo2-V`. Protects layers where aggressive V quantization degrades quality, leaves the rest at full aggression. [layer-aware-v-compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/layer-aware-v-compression.md), [moe-v-compression-frontier](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/moe-v-compression-frontier.md)
- **Sparse V dequantization** is described in the paper corpus ([sparse-v-dequant](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/sparse-v-dequant.md)) but is **not implemented in this tree**: the SYCL and Vulkan flash-attention kernels dequantize every attended V position. The `fattn-sparse` files serve DeepSeek/MiniMax sparse-attention graphs, not turbo V.
- **InnerQ per-channel equalization** has CPU and SYCL hooks (`ggml/src/ggml-innerq.c`, `ggml/src/ggml-sycl/innerq.cpp`) but is off by default (`LLAMA_ENABLE_INNERQ=1` opts in) and is not runtime-proven end to end on the A770 ([skipped-cases ledger](docs/research/sycl/2026-07-09-a770-skipped-cases-ledger.md)).

### Backend coverage

| Backend | Quant kernels | Flash Attention | Notes |
|---|---|---|---|
| **SYCL** (Intel Arc / oneAPI) | turbo `SET_ROWS` / `CPY` / dequant converters + WHT custom op (the turbo `mmvq` kernels are not selected by the mat-mul router); `TQ3_1S` / `TQ4_1S` single-token dequant matvec only (MoE `MUL_MAT_ID` on these types has no SYCL kernel and aborts, [ornith research section 8](docs/research/sycl/ornith-a770-perf-research-2026-09-27.md)); fused single-token MoE `mul_mat_id`; `q8_0` KV "quants-first" layout (default on) | `q8_0` / `f16` KV at mainline parity with VEC, TILE and oneMKL prefill routes; turbo K or V takes the VEC route by default (head dim must be a multiple of 128), with an opt-in XMX (DPAS) route (`GGML_SYCL_FA_XMX=1`, same turbo type on K and V, head dim 128 or 256). In the CPU-vs-SYCL correctness harness (section [5], run with `LLAMA_TEST_TURBO_FA=1`) turbo3/turbo4 pass and turbo2 is XFAIL (2-bit precision below the cosine floor) | A770 (DG2) is the canonical target; builds with `GGML_SYCL_F16=ON` or `OFF` |
| **Vulkan** | `TQ3_1S` / `TQ4_1S` weights (mat-mul, mat-vec, MoE `mul_mat_id`), `SET_ROWS` / `GET_ROWS` / `CPY` for `turbo2`/`turbo3`/`turbo4` | scalar and coopmat1 flash attention accept `turbo2`/`turbo3`/`turbo4` K and V (dequant fused into the shader); not on the coopmat2 path | Compute-shader path |


OpenVINO is shipped in-tree but is not exercised by the TurboQuant+ probes in this fork -- build with `-DGGML_OPENVINO=ON` only if you need the upstream OpenVINO backend.

### Model-family support

- **Gemma 4** -- large head-dim (`dk=512`) FA kernels, MoE token routing, op-concurrency handling
- **Large MoE** -- SYCL fused top-k MoE routing for power-of-two expert counts up to 512 (`ggml/src/ggml-sycl/topk-moe.cpp`)
- **Hybrid architectures (GDN, Mamba)** -- speculative decoding cherry-picked from upstream feature branches; SYCL kernels for `GATED_DELTA_NET`, `SSM_CONV`, `SSM_SCAN`
- **Hybrid MoE (qwen35moe / Ornith-1.5-35B-A3B)** -- runs, but see [Performance notes](#performance-notes-sycl-on-arc-a770) for where the time goes on SYCL
- All existing llama.cpp model families remain fully supported

### SYCL performance work carried by this fork

Default-on items were promoted after paired measurement on the A770 (dated evidence in
`docs/research/`). Opt-in items are listed so they can be found; they carry no throughput claim
beyond what their own documentation states.

Default on:

- `q8_0` KV "quants-first" layout for 128-wide heads (`GGML_SYCL_Q8_KV_QUANTS_FIRST=0` opts out); measured +8 to +21% decode at depth 4096-16384 ([P5.11](docs/research/sycl/sycl-a770-p5-performance-campaign-2026-07-19.md#p511---scale-separated-q8_0-kv-rows) paired campaign; depths 0 and 2048 were not covered)
- oneMKL GEMM prefill route for flash attention (`GGML_SYCL_ENABLE_MKL_FA=0` disables)
- Fused single-token MoE `mul_mat_id` matvec for the legacy (Q4_0 ... Q8_0), K-quant (Q2_K ... Q6_K), MXFP4 / NVFP4 and IQ weight types (`ggml_sycl_mul_mat_vec_q_id_supports_type`, `mmvq.cpp`)
- Per-kernel device-code split (`GGML_SYCL_DEVICE_CODE_SPLIT`) and pre-grown FA scratch buffers

Opt in:

- SYCL-Graph record-once / replay for stable-shape decode (`GGML_SYCL_ENABLE_GRAPH=1`); measured byte-identical to eager, gain depends on how many CPU/GPU splits break the graph
- MoE expert cache in spare VRAM (`--moe-cache`); experimental, can regress decode when combined with `--fit`, read [docs/backend/MOE-CACHE.md](docs/backend/MOE-CACHE.md) before enabling
- Lookahead upload of host-resident MoE experts on a private SYCL copy queue (`--prefetch-experts-slots N`, N >= 2); Level Zero v1 adapter only, and on DG2 with the `xe` driver it needs `UR_L0_USE_COPY_ENGINE=1`, which re-enables the blitter failure described below ([docs/backend/SYCL.md](docs/backend/SYCL.md))
- 256-GRF flash-attention tile kernels for multi-row (prefill) launches: build with `-DGGML_SYCL_FA_LARGE_GRF=ON`, run with `GGML_SYCL_FA_LARGE_GRF=1`. Measured +2.4% (Ornith IQ2_M, d=256) to +10.0% (Llama-3.1-8B, d=128) pp512 at depth 0, flat at depth 8192 and on decode ([2026-09-30 campaign](docs/research/sycl/sycl-fa-large-grf-2026-09-30.md)). Per-kernel, unlike the global large-GRF mode listed under dead ends

### Operational fixes carried by this fork

- CPU `vec_dot` for turbo / TQ types dequantizes through fixed 256-element stack chunks: no per-call `malloc`, no row-length cap (`ggml/src/ggml-cpu/ggml-cpu.c`)
- DG2 on the `xe` kernel driver: ggml-sycl keeps USM copies off the blitter by default (sets `UR_L0_USE_COPY_ENGINE=0`, or the v2 adapter equivalent), because blitter copies hang or reset the `bcs` engine there; `GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0` restores the adapter defaults (`ggml/src/ggml-sycl/xe-kmd.cpp`, Known Issues in [docs/backend/SYCL.md](docs/backend/SYCL.md))
- Arch Linux packaging for the SYCL build ([packaging/arch/PKGBUILD](packaging/arch/PKGBUILD)) with a systemd unit

---

## Quick start

Build with the backends you need. TurboQuant+ types work on the CPU backend; SYCL and Vulkan add GPU kernels for them.

```bash
# CPU only
cmake -B build && cmake --build build -j

# SYCL (Intel Arc / oneAPI) -- canonical target
# source /opt/intel/oneapi/setvars.sh first (oneMKL headers); see docs/backend/SYCL.md
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON -DGGML_NATIVE=OFF \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER= && cmake --build build -j

# Vulkan
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j

# SYCL + Vulkan (A770 with Vulkan fallback)
cmake -B build -DGGML_SYCL=ON -DGGML_VULKAN=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx && cmake --build build -j
```

KV-cache types are selected per-side via the standard `--cache-type-k` / `--cache-type-v` flags.

> **Start light, then compress.** Some model families -- small models, certain MoE configurations, quant-sensitive instruction-tuned variants -- are more delicate than others. Pick a light asymmetric configuration first, verify output quality (eyeball + PPL on a hold-out set) on your specific model, then ratchet up V aggression if you have memory headroom to gain. **Do not start at maximum compression** and work backwards.

The core finding from the asymmetric-kv-compression paper -- [**Asymmetric K/V Cache Compression: Why V is Free and K is Everything**](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md) -- drives all the configs below: **V tolerates aggressive compression, K does not**. Always keep K at higher precision than V; never start symmetric. That paper documents the specific failure modes you'll hit if you ignore this and compress K aggressively (PPL blow-up on certain model families, attention-rotation interaction with low-bit K, etc.) -- read it before considering step 6.

Higher turbo number = more bits per element = less aggressive compression. The V-side compression ladder is `turbo4` (lightest) -> `turbo3` -> `turbo2` (heaviest). On the K side, prefer `f16` or `q8_0`; never lead with a turbo K.

Recommendations, ordered from most conservative to most aggressive:

| Step | `--cache-type-k` | `--cache-type-v` | When | Notes |
|---|---|---|---|---|
| **1. Safest start** | `f16` | `turbo4` | First contact with any new model | K untouched, V at the lightest turbo tier. If output isn't faithful at this step, the model is unusually quant-sensitive -- stop and investigate before escalating. |
| **2. Conservative** | `q8_0` | `turbo4` | Verified safe at step 1, want a memory win without much risk | Light on both sides. Typically near-indistinguishable from `f16`/`f16` outputs. |
| **3. Recommended default** | `q8_0` | `turbo3` | Most dense models when KV memory is the constraint | The "asymmetric turbo" sweet spot from the [asymmetric-kv-compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md) paper. Near-lossless K, 5.12x compressed V; total KV 2.75x smaller than `f16`/`f16` (analytic, from block sizes). Measured +0.54% PPL vs `q8_0`/`q8_0` on Llama-3.1-8B-Instruct Q4_K_M at ctx 512 ([turbo3 gate note](docs/research/turbo/turbo3-quality-gate-llama31-8b-2026-09.md)). On SYCL this mixed pair runs flash attention on the VEC route (the opt-in XMX route needs the same turbo type on K and V), which costs prefill throughput at long context (see [Performance notes](#performance-notes-sycl-on-arc-a770)). |
| **4. Aggressive V** | `q8_0` | `turbo2` | Memory-bound long context, after validating quality at step 3 | Boundary V auto-engages on models with at least 8 layers: the first 2 and last 2 layers keep V at `q8_0`, the rest use `turbo2`. No in-tree PPL measurement of this pair yet; validate on your model. |
| **5. MoE-aware aggressive** | `q8_0` | `turbo2` | Large non-MLA MoE models (Qwen3 MoE, Mixtral-style) | Same flags and the same layer-based Boundary V; nothing in it is expert-aware. MLA models (DeepSeek V2/V3 with MLA metadata) reject different K and V cache types at context creation. See [moe-v-compression-frontier](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/moe-v-compression-frontier.md). |
| **6. Discouraged: symmetric K compression** | any `turbo*` | any `turbo*` | Only with model-specific quality validation in hand | Compressing K is where models break. The asymmetric paper documents the failure modes. Not a starting point. On GQA >= 6 or Qwen-family models a same-type request has K rewritten to `q8_0` anyway (see [Automatic behavior](#automatic-behavior)). |

Example invocations:

```bash
# Step 1 -- safest start (first contact with a new model)
llama-cli -m model.gguf --cache-type-k f16 --cache-type-v turbo4 -p "..."

# Step 3 -- recommended default (asymmetric turbo)
llama-cli -m model.gguf --cache-type-k q8_0 --cache-type-v turbo3 -p "..."

# Step 4 -- aggressive V at long context
llama-cli -m model.gguf --cache-type-k q8_0 --cache-type-v turbo2 -c 131072 -p "..."
```

If output quality drops between steps, walk back to the previous step. The compression frontier is per-model -- there is no global "best" setting.

### Weight quantization (offline)

Weight quantization is selected at conversion time via `llama-quantize`:

```bash
# TQ4_1S -- 5.0 bpw
llama-quantize model.f16.gguf model.tq4_1s.gguf TQ4_1S

# TQ3_1S -- 4.0 bpw
llama-quantize model.f16.gguf model.tq3_1s.gguf TQ3_1S
```

Vulkan has mat-mul, mat-vec and MoE kernels for both types; SYCL has only a single-token matvec, and MoE models on these types abort there (see the backend table). No in-tree PPL measurement of either type exists yet.

### Automatic behavior

The following activate based on the selected types -- no flags required:

- **Auto-asymmetric K/V** -- when K and V request the same turbo type on a model with GQA ratio >= 6 or from the Qwen family, K is rewritten to `q8_0` and V keeps the turbo type; MLA and DeepSeek4 models are exempt, and `TURBO_AUTO_ASYMMETRIC=0` opts out (`src/llama-kv-cache.cpp`). Measured motivation: Qwen2.5 7:1 GQA with symmetric turbo K went from PPL 7.4 to 2887; Mistral 4:1 was fine. Kernels downstream must accept `K=q8_0, V=turbo*`.
- **Boundary V (layer-aware)** -- `TURBO_LAYER_ADAPTIVE` mode 7 auto-enables for `turbo2-V` on models with at least 8 layers (`TURBO_LAYER_ADAPTIVE=0` opts out); shallower models stay uniform, and it's inert for non-turbo types.
- **Flash Attention** -- a turbo K or V type turns flash attention on even when it was disabled (a Grok model with flash attention disabled is rejected instead; `src/llama-context.cpp`). It works with turbo KV at runtime on SYCL (VEC route by default, opt-in XMX via `GGML_SYCL_FA_XMX=1`) and Vulkan (scalar / coopmat1). `LLAMA_TEST_TURBO_FA=1` is an opt-in for the CPU-vs-SYCL correctness harness, not a runtime switch.

If this fork or any of its quantization types is used in your work, please cite the corresponding paper from the [TurboQuant+ paper corpus](https://github.com/TheTom/turboquant_plus/tree/main/docs/papers).

## Performance notes (SYCL on Arc A770)

High-level reading of the measurements in `docs/research/`. Numbers are from this fork on an A770; treat anything not
labelled as a paired campaign as order-of-magnitude.

- **Turbo KV is a capacity feature, not a speed feature.** It buys more context or a bigger model in the same VRAM. Do
  not expect `f16` / `q8_0` decode parity, and on SYCL turbo K or V takes the VEC kernel by default (no TILE,
  no oneMKL prefill route); `GGML_SYCL_FA_XMX=1` can route the same turbo type on K and V at head size 128 or 256 to
  the opt-in XMX kernel. The production Ornith-1.5-35B-A3B service runs `q8_0`/`q8_0`
  ([ornith research](docs/research/sycl/ornith-a770-perf-research-2026-09-27.md), section 1). Reach for turbo when memory, not throughput, is the limit.
- **Decode on this GPU is launch-bound, not purely bandwidth-bound.** A hybrid MoE step (Ornith: 40 layers, ~1500
  kernels) reads about 1.9 GiB of weights per token (routed experts plus the non-expert weights every layer reads
  regardless of routing -- attention, SSM, shared-expert, and output; the embedding table is a per-token row lookup,
  not a full read, so it's excluded), which the A770 and DDR5 could serve in ~22 ms; the observed token takes ~48 ms.
  The remaining gap is per-kernel submission cost (~21 us per small kernel in the runtime's default immediate-command-list
  mode, ~7 us batched) plus host-device synchronisation at every CPU/GPU split. Two levers target the submission cost:
  SYCL-Graph replay (`GGML_SYCL_ENABLE_GRAPH=1`) and batched submission (`UR_L0_USE_IMMEDIATE_COMMANDLISTS=0
  UR_L0_BATCH_SIZE=64`). Graph replay was already on in the ~48 ms measurement; graphs on versus off differed by about
  2% there, and whether Ornith's MoE segments replay at all has not been observed. Batched submission gave +32% tg128 on
  dense Mistral-7B in the 2026-08-13 paired probe and has not been measured on Ornith. Whether the two gains add is
  unmeasured until a paired run crosses both settings. Neither is on by default in the fork or the runtime.
- **MoE prefill is bounded by the per-expert loop.** SYCL `mul_mat_id` is fused only for single-token decode; any batch
  larger than one runs one small GEMM per *touched* expert (the loop skips experts nothing routed to). The exception
  is the IQ weight types, which take one grouped XMX GEMM launch while each expert's row slice stays narrow
  (`GGML_SYCL_XMX_GATHER_TYPES`; measured for IQ4_NL / IQ3_S only, no gain at 256 experts x 512 tokens, ornith research
  section 8); Ornith's Q4_K / Q6_K experts take the loop. A large
  prefill ubatch is likely to touch every expert at least once, so at ub=512-2048 that's close to the 256 x 3
  launches per layer upper bound; a small speculative-decoding verify batch of `n` tokens touches at most `8*n`
  (8 experts/token), far fewer. Per-matrix cost is nearly flat from 512 to 2048 tokens, so a larger `--ubatch-size`
  amortises the large-batch case almost linearly, but the *launch count* still scales with the small batch's own
  `8*n`, which is why draft-based speculation only wins at very high acceptance (n-gram on code) on this backend. A
  grouped kernel over the expert-sorted rows for the K-quants is the open code item.
- **CPU/GPU expert placement.** `--fit` on SYCL sees free VRAM equal to total VRAM on this driver, so it assumes sole
  tenancy and only `--fit-target` protects you. CPU-resident experts land in pinned host memory (not repacked); each
  CPU-resident expert layer adds two synchronous split boundaries per token. For a 16 GiB card and a Q4_K_M 35B-A3B
  model that is roughly 17 of 40 expert layers on the CPU.
- **Dead ends already measured** (do not re-run without a driver or compiler change): SLM centroid-LUT dequant in FA,
  global large-GRF mode, GPU oneDNN prefill, alternate MMVQ geometry, tensor-core WHT, `joint_matrix` XMX at sub-group
  16 (IGC internal error; SG 8 is 4-7x slower than VEC, so XMX ships off).

Where the evidence lives: [ornith-a770-perf-research-2026-09-27](docs/research/sycl/ornith-a770-perf-research-2026-09-27.md),
[round2-decode-probes-2026-08-13](docs/research/sycl/round2-decode-probes-2026-08-13.md),
[sycl-a770-p5-performance-campaign-2026-07-19](docs/research/sycl/sycl-a770-p5-performance-campaign-2026-07-19.md),
[standard-sycl-baseline-2026-07-11](docs/research/sycl/standard-sycl-baseline-2026-07-11.md),
[turbo-fa-research-artifact](docs/research/turbo/turbo-fa-research-artifact.md).

## License

MIT, same as upstream llama.cpp.

---

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/features/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)

----


## Quick start

A few options to get `llama.cpp` installed on your machine:

- Build from source by cloning this repository - check out [our build guide](docs/build/build.md) and, for SYCL, [docs/backend/SYCL.md](docs/backend/SYCL.md)
- Arch Linux: [packaging/arch/PKGBUILD](packaging/arch/PKGBUILD) builds the SYCL variant and installs a systemd unit
- Docker: this fork ships no Dockerfiles (`.devops/` is not in the tree) and publishes no images; see [docs/build/docker.md](docs/build/docker.md)
- Upstream's pre-built binaries and https://llama.app do not contain the TurboQuant+ types or the SYCL work in this fork

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- x86_64 (AVX / AVX2 / AVX512 / AMX), ARM NEON, and RISC-V (RVV) SIMD paths on CPU
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization, plus the TurboQuant+ family (`turbo2`/`turbo3`/`turbo4`/`TQ3_1S`/`TQ4_1S`)
- Vulkan and SYCL backend support (this fork's GPU targets)
- BLAS support (OpenBLAS, oneMKL, AOCL, BLIS) for CPU matrix multiplication
- CPU+GPU hybrid inference to partially accelerate models larger than total VRAM

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends in this fork

Only the backends below are built and tested. CUDA, HIP/ROCm, Metal, OpenCL, CANN, MUSA, WebGPU, RPC, Hexagon, ET, ZenDNN, IBM zDNN, and VirtGPU are present in upstream `ggml-org/llama.cpp` but **not shipped here** -- see the upstream README for those.

| Backend | Target devices | Notes |
| --- | --- | --- |
| [CPU](docs/build/build.md#cpu-build) | x86_64 / ARM / RISC-V | Default backend; SIMD-accelerated |
| [BLAS](docs/build/build.md#blas-build) | All | OpenBLAS, oneMKL, AOCL, BLIS (`GGML_BLAS_VENDOR=FLAME`), Accelerate |
| [SYCL](docs/backend/SYCL.md) | Intel GPU (Arc / iGPU / Data Center GPU Max) | Canonical target is the Arc A770; oneAPI/DPC++ toolchain required |
| [Vulkan](docs/build/build.md#vulkan) | GPU (cross-vendor) | Compute-shader path; works on Intel/AMD/NVIDIA |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs | Shipped in-tree; not exercised by TurboQuant+ probes |

## Documentation

Full index: [docs/README.md](docs/README.md). Research corpus: [docs/research/README.md](docs/research/README.md).

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build/build.md)
- [Build on Android](docs/build/android.md)
- [Multi-GPU usage](docs/user/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/build/xcframework.md) (upstream; `build-xcframework.sh` targets the Metal backend, which this fork does not ship)
- [Completions](docs/user/completions.md)
- [Models](docs/user/models.md)
- [Release process](docs/development/release.md)

## Contributing

- Open PRs against `master` of `Raudbjorn/ggml-llama.cpp`; do not send fork-specific changes to `ggml-org/llama.cpp`
- Read [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) first: ASCII-only code and comments, reuse existing infrastructure, `Assisted-by:` commit trailer
- Performance claims need paired measurements on the A770 (see `scripts/README.md`), and every PR body ends with a "Not claimed" section stating what was not tested
- Report a suspected vulnerability through [the fork's private vulnerability-reporting form](https://github.com/Raudbjorn/ggml-llama.cpp/security/advisories/new), not a public issue. Include the affected commit, configuration, reproduction steps and expected impact. Report vulnerabilities that affect only upstream llama.cpp to upstream.

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
