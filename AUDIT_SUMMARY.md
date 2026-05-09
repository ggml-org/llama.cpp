# Audit summary: vllm-xpu-kernels & llm-scaler

Quick-revisit summary of two audits. Full detail in:

- `vllm-xpu-kernels/AUDIT.md`
- `llm-scaler/AUDIT.md` (`LLM-SCALER-AUDIT.md` in cwd)

## Context

Investigation into why B60 (Battlemage, 200W rated) tops out around 95-145W
during llama.cpp inference. The watt gap traces to the dequant-then-FP16-matmul
split for quantized weights: dequant is memory-bound and runs as a separate
kernel, leaving XMX matrix engines partially idle. F16 weights (no dequant)
pull the chip closest to its sustained ceiling (~143W observed); Q4_K MoE
floors at 98W.

Two third-party SYCL kernel repos exist that have already solved pieces of this.

## The two repos at a glance

| | `vllm-xpu-kernels` | `llm-scaler` |
|---|---|---|
| Style | CUTLASS-via-sycl-tla + oneDNN | ESIMD + oneDNN |
| Hardware tuning | XE2 generic | **Battlemage explicit (LGRF mode)** |
| GGUF awareness | None | **Native Q4_0 / Q8_0 / Q4_K / Q6_K** |
| Targeted MoE workload | Generic grouped GEMM | **Qwen3.5-35B-A3B specifically** |
| Build dep complexity | sycl-tla via FetchContent | None new (oneDNN already there) |
| Closer fit for ggml-sycl | No | **Yes** |

`llm-scaler` is substantially more relevant. Read its audit first.

## Headline findings from llm-scaler

1. **`omni_xpu_kernel/csrc/gguf_dequant.cpp`** (528 LoC) — ESIMD dequant for the *exact* GGML byte layouts of Q4_0/Q8_0/**Q4_K**/Q6_K. Direct port candidate.

2. **`omni_xpu_kernel/csrc/onednn_int4_gemm.cpp`** (323 LoC) — fused INT4-dequant + GEMM via oneDNN's u4 matmul primitive. The exact path that closes the 25W Q4_K-vs-F16 gap. No new build deps.

3. **`vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl`** (3660 LoC) — ESIMD MoE INT4 kernel built specifically for Qwen3.5-35B-A3B reading GGML Q4_0 layout. Intel ships a `setup_moe_int4_only.py` for isolated build. Targets exactly the workload that floored at 98W on our bench.

4. **`lgrf_uni/single_kernels/flash.attn.*.h`** — five hand-tuned FA variants in Battlemage's Large-GRF mode (256 regs/lane). Reference for what a non-regressing SYCL FA kernel looks like; directly relevant to the `fa-overhead-sycl` branch (which currently sees -15% on F16 prefill with FA on).

## Highest-ROI port if picking one thing

**`omni_xpu_kernel/csrc/onednn_int4_gemm.cpp` + a Q4_K layout adapter (or one-time repack at model load).**

- ~500 LoC of new ggml-sycl code.
- No new build deps (oneDNN is already linked).
- Directly attacks the dequant-pass overhead.
- For Q4_0 the layout match is essentially direct; Q4_K's nested 6-bit super-block scales need a repack to flat W4 + group-scales format that oneDNN expects.

## Highest-ROI thing to read

**`moe_int4.sycl`** — worked example of keeping XMX fed with INT4 weights end-to-end on this exact hardware, by Intel engineers who measured it. Even if not ported wholesale, the patterns (DPAS dispatch, register tiling, SLM management) are the reference.

## Adjacent insight: fusion as alternative to multi-streaming

`llm-scaler` ships aggressively fused kernels (`esimd_resadd_norm_gemv_*`,
`esimd_qkv_split_norm_rope`, `fused_rms_norm_linear`). These collapse 3-4
ggml ops into one kernel.

This is the **inverse** of the multi-stream/concurrent-kernel approach:
instead of overlapping kernels to hide launch overhead, just stop having so
many kernels. For ggml-sycl this could mean defining fused ggml ops at the
backend level. Lower invasiveness than refactoring stream/queue plumbing,
arguably higher payoff per LoC.

## Effort ranking (rough)

| Port | LoC | Build deps | Effort | Impact |
|---|---|---|---|---|
| GGUF dequant kernels (standalone) | ~250 (Q4_K only) | none | low | watts on Q4_K |
| oneDNN INT4 GEMM + adapter | ~500 | none | medium | **closes Q4_K↔F16 gap** |
| MoE INT4 (Qwen3 35B-A3B) | ~3700 | none | high | **closes MoE 98W floor** |
| LGRF flash attention | ~700 | none | medium-high | unblocks `fa-overhead-sycl` |
| Fused ops (norm+gemv etc.) | varies | none | medium | reduces launch overhead globally |

## What to skip

- `vllm-xpu-kernels` LoRA / sampling / norm/activation kernels — orthogonal to base inference watts.
- `llm-scaler` SVDQuant kernels — diffusion-model scheme, not in GGUF.
- vLLM patches in `llm-scaler/vllm/patches/` — vLLM-specific, irrelevant to llama.cpp.

## Cross-references

- Multi-stream discussion that motivated this audit: see conversation context. Key code: `ggml/src/ggml-sycl/common.hpp:329-334` (the stub that aliases all 8 stream slots to one in-order queue).
- CUDA equivalent (real per-slot streams): `ggml/src/ggml-cuda/common.cuh:1436-1442`.
- Bench results that grounded the watt analysis:
  - 35B-A3B Q4_K: pp512=349 t/s, peak 98W
  - 4B dense Q4_K: pp512=1129 t/s, peak 123W
  - 7B dense F16: pp4096=632 t/s, sustained 138W, peak 143W
  - F16 with `-fa 1`: pp4096=539 t/s (-15%), tg128=26.3 t/s (+6%), same 130-143W band
