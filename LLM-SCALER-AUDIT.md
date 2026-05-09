# llm-scaler audit

Audit of Intel's `llm-scaler` from the perspective of porting compute-density
wins into `ggml-sycl`. Companion to `vllm-xpu-kernels/AUDIT.md`.

Repo: `git@github.com:intel/llm-scaler.git`
Tip at audit time: `2b14e1a llm-scaler-vllm-b8.2.1 and llm-scaler-omni-b7`

## TL;DR

`llm-scaler` is **substantially more relevant than `vllm-xpu-kernels`** for
closing the Q4_K watt gap on Battlemage. Two reasons:

1. It contains **GGUF-aware kernels**: ESIMD dequant for Q4_0 / Q8_0 / **Q4_K** / Q6_K, plus fused INT4-dequant + GEMM via oneDNN. This is the exact data path `ggml-sycl` currently splits into two passes.
2. It contains a **MoE INT4 kernel specifically tuned for Qwen3.5-35B-A3B** (the model we benchmarked at 98W) consuming **GGML Q4_0 layout directly**. Not a general-purpose primitive — a kernel built for this workload.

## Top-level shape

```
llm-scaler/
├── vllm/                          # llm-scaler-vllm: Intel's vLLM fork
│   ├── custom-esimd-kernels-vllm/ # ESIMD kernels for inference
│   ├── tpp/                       # docs/licensing only
│   └── patches/                   # vLLM upstream patches
└── omni/                          # llm-scaler-omni: image gen / ComfyUI side
    └── omni_xpu_kernel/           # ESIMD + oneDNN kernels for ComfyUI-GGUF
```

Two distinct ESIMD kernel sets with **different scope**:

- `vllm/custom-esimd-kernels-vllm/` — text inference, MoE-heavy.
- `omni/omni_xpu_kernel/` — built for ComfyUI-GGUF (image gen), but the dequant kernels apply equally to LLM inference because **the GGUF format is the same**.

**Hardware target:** Battlemage explicitly. Build sets
`TORCH_XPU_ARCH_LIST=bmg`. Some kernels also have PVC paths.

## Implementation style: ESIMD vs CUTLASS

This is the major architectural difference vs `vllm-xpu-kernels`:

- **ESIMD** (Explicit SIMD): SYCL extension where you write directly in terms of SIMD lanes and explicit register allocation. Compiles closer to GPU ISA. You call `xmx::dpas<...>` directly instead of letting CUTE schedule for you. `llm-scaler` is **almost entirely ESIMD**.
- **CUTLASS-via-sycl-tla**: Template metaprogramming over tile shapes; familiar to anyone from the NVIDIA ecosystem. `vllm-xpu-kernels` uses this for its newer paths.
- **oneDNN**: Both repos use this for cases where its primitives are competitive (notably W4A16 INT4 GEMM with fused dequant).

For Battlemage specifically, ESIMD tends to win on small/decode-shape GEMMs where register pressure and DPAS scheduling matter more than tile algorithm. CUTLASS tends to be competitive on large prefill GEMMs.

`llm-scaler/omni/omni_xpu_kernel/lgrf_uni/` even has **LGRF (Large GRF mode) variants** — Battlemage can configure 256 registers/lane instead of 128, trading occupancy for register pressure. ESIMD lets you write code that requires this mode; high-level abstractions can't.

## What's in `vllm/custom-esimd-kernels-vllm/`

### MoE INT4 (the killer kernel)

| File | LoC | What it is |
|---|---|---|
| `csrc/moe_batch/moe_int4.sycl` | **3660** | **Full MoE INT4 inference path for Qwen3.5-35B-A3B.** Reads GGML Q4_0 layout directly: 8 INT4 nibbles packed per int32, group_size=128 fp16 scales. Includes router GEMV, expert GEMMs (wide and narrow variants), gather/scatter. |
| `csrc/moe_batch/moe.sycl` | 1451 | FP16 MoE counterpart. |
| `csrc/moe_batch/int4_nmajor_gemm.h` | 325 | DPAS-based INT4 N-major grouped GEMM core. Per-group scales, signed INT4 nibbles via `(nibble_unsigned - 8)`. Grid dispatch: `range<2>(num_experts, N / N_TILE)`, `N_TILE=16` for up-projection, `32` for down-projection, `M_TILE=32` chunk loop. |
| `csrc/moe_batch/moe_topk.h` | 102 | TopK routing. |
| `csrc/moe_prefill/moe_prefill_int4.sycl` | 1016 | Prefill-shape MoE INT4 (K-major weight layout, larger M). |

**Note on layout:** This kernel uses GGML **Q4_0** (32-element blocks, 4-byte fp16 scale, 16 bytes data → 18 bytes/block). The user's MoE benchmark used **Q4_K_M** (much more complex 256-element super-blocks). Direct reuse requires either re-quantizing to Q4_0 or writing a Q4_K-aware variant. The DPAS dispatch logic and tile sizing transfer; the unpack/dequant inner loop needs adaptation.

### General ESIMD kernels (`csrc/xpu/`)

| File | LoC | Notes |
|---|---|---|
| `esimd_kernel.sycl` | 422 | Generic kernel collection (header forwards). |
| `esimd_kernel_gemm.sycl` | 45 | Stub/dispatcher. |
| `esimd_kernel_lgrf.sycl` | 117 | **LGRF-mode kernels** (256 regs/lane). |
| `esimd_kernel_moe.sycl` | 154 | MoE bookkeeping. |
| `esimd_kernel_topk_v2.sycl` | 46 | Vectorized argmax for routing. |
| `torch_extension*.cc` | — | PyTorch op registration. |

### Speculative decoding (`csrc/eagle/`)

| File | LoC | Notes |
|---|---|---|
| `eagle.sycl` | 433 | EAGLE/EAGLE-2 speculative decoding entry. |
| `eagle.kernels.fp16.h` | 469 | FP16 EAGLE inner kernels. |
| `eagle.kernels.bf16.h` | 469 | BF16 variant. |
| `page.attn.h` | 550 | **Paged attention** (decode-time FA over paged KV cache). Phase-1 / phase-2 decomposition with explicit SLM atomics. |

The paged attention kernel is the closest analog to your `fa-overhead-sycl` work but operates on a paged KV layout vLLM uses, not ggml's contiguous layout. Algorithmic ideas transfer; data-layout assumptions don't.

### Op surface (`include/kernel_ops.h`, 232 LoC, 27 ops)

Selected:

```cpp
at::Tensor esimd_gemv_fp8_pern(...);            // FP8 GEMV per-N-scale (decode)
at::Tensor esimd_gemv_fp8_pern_fused2(...);     // ... fused 2 weight matrices
at::Tensor esimd_gemv_fp8_pern_fused3(...);     // ... fused 3 (QKV)
at::Tensor esimd_gemv_int4(...);                // INT4 GEMV
at::Tensor esimd_gemv_int4_fused2(...);
at::Tensor esimd_norm_gemv_fp8_pert(...);       // RMSNorm + FP8 GEMV fused
at::Tensor esimd_norm_gemv_int4_pert(...);      // RMSNorm + INT4 GEMV fused
at::Tensor esimd_resadd_norm_gemv_fp8_pert(...);// res + RMSNorm + FP8 GEMV fused
at::Tensor esimd_resadd_norm_gemv_int4_pert(...);
at::Tensor esimd_qkv_split_norm_rope(...);      // QKV split + RMSNorm + RoPE fused
at::Tensor esimd_moe_gemm_fp8(...);             // FP8 grouped MoE GEMM
at::Tensor esimd_moe_gemm_fp8_pert(...);
at::Tensor esimd_moe_topk(...);
at::Tensor esimd_moe_scatter(...);
at::Tensor esimd_moe_scatter_fused(...);
at::Tensor esimd_moe_silu_mul(...);
at::Tensor esimd_moe_gather(...);
```

The **fusion patterns** (`resadd_norm_gemv_*`, `qkv_split_norm_rope`) are the most interesting ideas to port: these collapse 3-4 ggml ops into one kernel, removing the inter-kernel synchronization that limits SYCL queue concurrency. This is an alternative path to multi-streaming — instead of overlapping kernels, just stop having so many kernels.

### Build entry points

- `setup.py` — full build with all kernels.
- `setup_moe_int4_only.py` — **MoE INT4 only**, ~3700 LoC of kernels. Smallest viable extraction unit.
- `setup_sycl.py` — SYCL-only kernels (skip ESIMD).

The `setup_moe_int4_only.py` existence is significant: Intel themselves ship the MoE INT4 kernel as a standalone option. That's the natural starting point for a port.

## What's in `omni/omni_xpu_kernel/`

Originally for ComfyUI-GGUF (image-gen with GGUF-quantized weights). Crucially: **GGUF support is the bridge to llama.cpp.**

### `csrc/gguf_dequant.cpp` (528 LoC)

The single most directly portable file in either repo. Header comment:

```
// Supported formats (matching ComfyUI-GGUF layout):
//   Q4_0: Block=32, Size=18 bytes (2 scale + 16 data)
//   Q8_0: Block=32, Size=34 bytes (2 scale + 32 data)
//   Q4_K: Block=256, Size=144 bytes (2+2+12+128)
//   Q6_K: Block=256, Size=210 bytes (128+64+16+2)
```

**These are GGML's exact block layouts.** Not a re-implementation, not a re-quant — direct dequant from GGML byte format to fp16/bf16. The same blocks `ggml_quantize_q4_K_reference` produces in `ggml/src/ggml-quants.c`.

Exposed ops:

```cpp
gguf.dequantize_q4_0(input, dtype=fp16) -> Tensor
gguf.dequantize_q8_0(input, dtype=fp16) -> Tensor
gguf.dequantize_q4_k(input, dtype=fp16) -> Tensor   // ★
gguf.dequantize_q6_k(input, dtype=fp16) -> Tensor
gguf.dequantize_batch(...)  // batch multiple tensors, fewer kernel launches
```

The `dequantize_batch` op is interesting: groups inputs by format, concatenates, launches one kernel per format, then splits — explicit attempt to amortize launch overhead. Same problem ggml-sycl has with many small kernels per layer.

### `csrc/onednn_int4_gemm.cpp` (323 LoC)

**Fused INT4 dequant + GEMM via oneDNN's u4 matmul primitive.** From the file's own header:

> "Uses oneDNN's native u4 matmul primitive with per-group block quantization scales to fuse the dequantize_w4 + bf16 matmul into a single oneDNN call."
>
> "Performance: Primitive creation is expensive (~5ms). We cache primitives keyed by {ActDT, M, K, N, group_size} so repeated calls (1224 per image) only pay the creation cost once."

This is **exactly the path that closes the Q4_K-vs-F16 watt gap**: instead of `[dequant kernel] -> [fp16 matmul kernel]`, oneDNN dequants block-by-block in registers and feeds XMX directly. One primitive, one kernel, weights stay packed in DRAM.

Exposed ops:

```cpp
svdq.onednn_int4_gemm(...)                  // generic
svdq.onednn_int4_gemm_preconverted(...)     // pre-converted weights (skip per-call repack)
svdq.onednn_int4_gemm_add_to_output(...)    // + residual add via oneDNN append_sum post-op
```

The `_preconverted` and `_add_to_output` variants matter for ggml-sycl integration: weights can be repacked once at model load, and residual connections can be fused into the matmul instead of needing a separate add.

### Other `omni` kernels

| File | LoC | Notes |
|---|---|---|
| `csrc/onednn_fp8.cpp` | 533 | FP8 W8A16 fused dequant + GEMM (analog of the INT4 path for FP8). |
| `csrc/svdq_dequant.cpp` | 528 | SVDQuant-format dequant (W4A4 scheme used by Nunchaku diffusion models). |
| `csrc/svdq_fused_postproc.cpp` | 468 | SVDQuant post-processing fusion. |
| `csrc/norm.cpp` | 612 | RMSNorm, LayerNorm, **fused_add_rms_norm**, **fused_rms_norm_linear** (norm + linear in one kernel). |
| `csrc/rotary.cpp` | 277 | RoPE. |
| `csrc/sdp.cpp` + `sdp_kernels.h` | 380+38 | Scaled dot-product attention. |
| `lgrf_uni/sdp_kernels.cpp` | 202 | LGRF-mode SDP entry. |
| `lgrf_uni/single_kernels/flash.attn.*.h` | **5×~700** | **Five FA variants** in LGRF mode: `mha`/`mha128`, `fp16`/`bf16io`, `fp32accum`/`opt`. |

The five-way FA variant explosion is signal: Intel's engineers found that one parameterized FA kernel was leaving perf on the table, and ship handcrafted variants per (head_dim, dtype, accumulator) combination. Your `fa-overhead-sycl` work could borrow this taxonomy directly — pick the (FP16, head_dim=128, FP32 accum) variant and use it as a reference for what an actually-fast SYCL FA kernel looks like.

## Comparison to vllm-xpu-kernels

| Aspect | vllm-xpu-kernels | llm-scaler |
|---|---|---|
| Style | CUTLASS-via-sycl-tla + oneDNN | ESIMD + oneDNN |
| Hardware tuning | XE2 generic | Battlemage explicit, LGRF mode |
| GGUF awareness | None (vLLM uses its own quant layouts) | **Native Q4_0/Q8_0/Q4_K/Q6_K dequant** |
| Targeted MoE workload | Generic grouped GEMM | **Qwen3.5-35B-A3B specifically** |
| Fusion | Some (rms_norm + quant) | Aggressive (resadd+norm+gemv, qkv+norm+rope) |
| Build dep complexity | sycl-tla via FetchContent | None (oneDNN already ggml-sycl dep) |
| Code abstraction | High (CUTE templates) | Low (explicit SIMD lanes) |
| Easier to read | Yes if you know CUTE | No — but more direct mapping to hw |

**For ggml-sycl, llm-scaler is the closer fit.** ggml's quant philosophy and llm-scaler's GGUF orientation share a worldview; vllm's quant ecosystem is different.

## Highest-ROI ports for ggml-sycl

Re-ranked with `llm-scaler` material in mind:

### 1. `omni_xpu_kernel/csrc/onednn_int4_gemm.cpp` for Q4_K matmul

This is the single highest-leverage port. ~320 LoC, depends only on oneDNN (already a ggml-sycl dep), implements fused-dequant-plus-GEMM via oneDNN u4 matmul. **Closes the dequant-pass gap that explains the 25W difference between Q4_K and F16 in our benchmark.**

But: oneDNN expects a specific weight layout (`[N, K/2]` packed, per-group scales). Q4_K's super-block format with 6-bit nested scales doesn't map directly. Two options:

- **Repack Q4_K → flat W4 + group scales at model load** (one-shot cost). The arithmetic is well-defined; the bytes have to move.
- **Use the existing GGUF dequant kernel from `omni_xpu_kernel/csrc/gguf_dequant.cpp` as input to a separate matmul.** Worse than fused, but a stepping stone.

For Q4_0 (which the user's MoE model could be re-quantized to), the layout match is essentially direct.

### 2. `vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl` for MoE workloads

The Qwen3.5-35B-A3B-specific MoE INT4 kernel reads GGML Q4_0 layout directly. For users running this exact model on Battlemage — i.e., probably the user — this is a near-drop-in replacement for the entire MoE forward pass. The watt floor we saw at 98W is what this kernel was written to fix.

Effort: high. This is 3700 LoC of ESIMD with explicit DPAS, register tiling, SLM management. ESIMD is not for the faint of heart. But Intel did the work; we'd be transcribing, not designing.

### 3. `omni_xpu_kernel/csrc/gguf_dequant.cpp` standalone

Even without the fused-GEMM path, replacing ggml-sycl's current Q4_K dequant kernel with this ESIMD version probably wins watts. It's the closest thing to a free port: same input bytes, same output bytes, faster path. ~250 LoC for the Q4_K function alone (extractable from the 528-LoC file).

### 4. LGRF-mode flash attention

`lgrf_uni/single_kernels/flash.attn.b.mha.fp16.opt.h` (~740 LoC) is a reference for what a Battlemage-tuned FA kernel looks like. Your `fa-overhead-sycl` branch's 15% prefill regression vs non-FA is exactly the kind of thing this kernel was written to *not* have.

Effort: medium-high. Pure ESIMD, but isolated — no build deps beyond `<sycl/ext/intel/esimd.hpp>`.

### 5. Fusion patterns

The `esimd_resadd_norm_gemv_*` and `esimd_qkv_split_norm_rope` patterns are ideas to apply to ggml-sycl, not code to port. ggml's graph compute walks one op at a time; collapsing residual+norm+gemv into a single kernel removes the very kernel-launch overhead that the multi-stream approach is trying to hide. **Same problem, different axis.**

## What I'd skip

- `vllm/patches/` — vLLM-specific patches, irrelevant to llama.cpp.
- `vllm/tpp/` — only docs and licenses.
- `omni/omni_xpu_kernel/csrc/svdq_*` — SVDQuant is a diffusion-model quantization scheme; not present in GGUF/llama.cpp.
- `omni/omni_xpu_kernel/csrc/sdp.cpp` (the non-LGRF SDP) — likely superseded by the LGRF FA variants for our use case.

## Licensing

Both repos under the same Intel/vLLM-style permissive licenses (Apache 2.0 territory). Verify before merging code, but no surprises expected.

## Summary

If we're picking *one* thing to port to close the watt gap on the bench we ran:
**`omni_xpu_kernel/csrc/onednn_int4_gemm.cpp` + a Q4_K-aware adapter or one-time repack at model load.** ~500 LoC of new code in ggml-sycl, no new build deps, and it directly attacks the dequant-pass overhead that's costing 25W on Q4_K dense and likely more on Q4_K MoE.

If we're picking *one* thing to learn from: **read `moe_int4.sycl`** — it's a worked example of how to keep XMX fed with INT4 weights end-to-end, by people who measured it on this exact hardware.
