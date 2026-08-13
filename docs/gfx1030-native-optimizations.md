# Opt-in gfx1030 native optimizations

This branch contains experimental HIP paths tuned and validated on AMD RDNA2/gfx1030 (four Radeon Pro V620 GPUs). All behavior remains stock unless the master switch is enabled:

```bash
export GGML_HIP_GFX1030_NATIVE=1
```

Model-specific fusions and graph-scoped Q8_1 reuse require additional switches. Setting a secondary switch without `GGML_HIP_GFX1030_NATIVE=1` has no effect.

## Environment variables

| Variable | Default | Effect |
|---|---:|---|
| `GGML_HIP_GFX1030_NATIVE` | unset / `0` | Master opt-in for validated gfx1030 kernel specializations. Enables the Q4_0 DOT8 MMVQ path, bounded six-row Q4_K/Q6_K routed MMVQ dispatch, native tiled-FlashAttention arithmetic/reductions, and chunked GDN prefill loads. |
| `GGML_HIP_GFX1030_Q8_1_FUSION` | unset / `0` | In combination with the master switch, fuses routed SwiGLU evaluation into Q8_1 activation staging for eligible prompt-processing `MUL_MAT_ID` down projections. |
| `GGML_HIP_GFX1030_GDN_SIBLING_FUSION` | unset / `0` | In combination with the master switch, creates and uses fused Qwen3.5/Qwen3.6 DeltaNet sibling projection weights. |
| `GGML_HIP_GFX1030_Q8_CACHE` | unset / `0` | In combination with the master switch, enables graph-owned reuse of exact standard Q8_1 TG activations and the eligible dual RMSNorm F32/Q8_1 producer. |
| `GGML_HIP_GFX1030_Q8_CACHE_TELEMETRY` | unset / `0` | Reports eligible standard-MMVQ Q8_1 sources, safe reuses, cache hits, and storage. It does not enable reuse unless `GGML_HIP_GFX1030_Q8_CACHE=1` is also set. |

The selectors are read once during backend or model initialization. Set them before starting `llama-cli`, `llama-server`, `llama-bench`, or a test binary.

Example with every accepted path enabled:

```bash
GGML_HIP_GFX1030_NATIVE=1 \
GGML_HIP_GFX1030_Q8_1_FUSION=1 \
GGML_HIP_GFX1030_GDN_SIBLING_FUSION=1 \
GGML_HIP_GFX1030_Q8_CACHE=1 \
build/bin/llama-bench \
  -m model.gguf -ngl 999 -sm layer -ts 1/1/1/1 -fa on \
  -p 512 -n 128 -b 512 -ub 256 -r 5
```

To return to stock behavior, unset the optimization variables:

```bash
unset GGML_HIP_GFX1030_NATIVE
unset GGML_HIP_GFX1030_Q8_1_FUSION
unset GGML_HIP_GFX1030_GDN_SIBLING_FUSION
unset GGML_HIP_GFX1030_Q8_CACHE
unset GGML_HIP_GFX1030_Q8_CACHE_TELEMETRY
```

Multi-GPU ROCm state save/restore stability has a separate opt-in, `GGML_HIP_SAFE_STATE_IO=1`. It does not require the gfx1030 master switch and does not change inference kernels. See [the multi-GPU ROCm state-I/O workaround](rocm-multi-gpu-state-io.md).

## Master-switch paths

### Q4_0 DOT8 MMVQ

For Q4_0 decode with one destination column, the native MMVQ specialization preserves the ordinary Q4_0 weights and Q8_1 activation bytes. A two-byte `sum_hi` sidecar per Q8_1 block supplies the exact high-nibble correction needed by gfx1030 `UDOT8`/`SDOT8`; no model conversion or persistent layout change is required. Other quantization types remain on their normal vector-dot implementations.

The path is exact and opt-in, but the measured Qwen end-to-end result was neutral to approximately 0.7% slower. It is retained as a validated native arithmetic experiment rather than advertised as a default speedup. See [the Q4_0 DOT8 experiment](rdna2-v620-q4-0-dot8-experiment.md).

### Bounded six-row routed MMVQ

On RDNA2, stock Q4_K and Q6_K `MUL_MAT_ID` dispatch changes from MMVQ to MMQ above five routed token rows. Native mode extends MMVQ through six rows when all of the following hold:

- the expert weights are Q4_K or Q6_K;
- the destination has at most six token rows;
- the routed IDs select at most four experts per token.

The top-k bound is intentionally conservative. Across 48 Q4_K/Q6_K cases with top-k 2 or 4, K from 256 to 8192, N from 256 to 4096, 8 to 256 experts, and both uniform and concentrated routing, six-row MMVQ reduced operation latency by 15.9% to 61.2%. Top-k 6 regressed in one tested Q6_K case, while concentrated top-k 8 routing regressed in 12 of 16 grid cases by as much as 54.5%; those cases therefore retain stock dispatch.

The exact Qwen3.6 35B four-GPU layer-split configuration currently carries an advisory tensor flag that permits its validated top-k 8 MTP path to use six-row MMVQ. The flag is generic: another model loader may set it on validated Q4_K/Q6_K routed weights after equivalent testing. It is inert unless `GGML_HIP_GFX1030_NATIVE=1`, so native-off execution remains stock. Without the hint, every model automatically receives the validated top-k 1--4 path; higher top-k routing remains on stock dispatch until separately validated.

MMQ and MMVQ accumulate floating-point products in different orders and are not generally byte-identical. The validation sweep measured NMSE from `4.23e-10` to `9.15e-9`, compared with the backend `MUL_MAT_ID` allowance of `5e-4`; MMVQ graph and non-graph outputs were byte-identical. This path does not alter quantized weights or Q8_1 activation encoding.

### Tiled FlashAttention

The native tiled-F16 specialization uses gfx1030 `fdot2` accumulation and native wave-32 sum/max reductions. Host dispatch selects a separate compile-time kernel specialization, so the inner loops do not contain a runtime branch. Older ROCm compilers that do not expose the wave-reduction builtins compile the exact shuffle fallback instead. Vector and MMA FlashAttention variants are unchanged.

Both stock and native runs passed `2920/2920` `FLASH_ATTN_EXT` backend tests. Four-GPU PP4096 measurements remained within run-to-run variance, so no end-to-end FlashAttention gain is claimed. The guarded benchmark and verification workflow is documented in [the native FA harness](gfx1030-native-fa-harness.md).

### Chunked Gated DeltaNet prefill

For GDN calls with more than one token, the native specialization has lane 0 load scalar `beta` and per-column value inputs and broadcast them across the wave. In the non-KDA form it also loads and broadcasts the scalar gate; the KDA form retains its per-row gate loads. Decode (`n_tokens == 1`) keeps the stock specialization.

Direct GDN measurements improved by about 7.9% at 256 tokens and 17.7% at 512 tokens. The GDN backend suite passed all 36 cases across all five tested backends. Full-model PP4096 measurements were sensitive to process order and GPU temperature, so only the direct-kernel improvement is claimed.

## Secondary fusions

### Graph-scoped standard Q8_1 reuse

With both the master switch and `GGML_HIP_GFX1030_Q8_CACHE=1`, eligible TG matrix multiplications can share an exact standard Q8_1 activation instead of independently staging the same F32 source. The initial contract is deliberately narrow:

- a single-token, non-routed `MUL_MAT` using Q8_0 weights and MMVQ;
- standard `block_q8_1` activation layout only;
- the same source tensor/data, padded K, byte size, device graph, and stream;
- no packed 64/128/256 Q8_1 layouts, Q4_0 `sum_hi`, MMQ, or `MUL_MAT_ID`.

Storage belongs to the existing per-execution CUDA/HIP graph object and is freed when that graph is evicted. Readiness resets for every execution: the first consumer normally refreshes the entry, and later consumers reuse it only after an intervening-output overlap scan confirms that the F32 source remains live. This source-version and lifetime rule also applies when CUDA graph capture is disabled; it does not use an untyped tensor sidecar or a fixed node TTL.

After one execution has proved a reusable group, an eligible already-fused contiguous single-row RMSNorm+MUL can write both the normal F32 output and exact standard Q8_1 bytes into the planned entry. If no matching safe entry exists, the ordinary fused RMSNorm and MMVQ staging paths run unchanged. The dual producer preserves the materialized F32 arithmetic boundary before Q8_1 scale, sum, and rounding operations.

On the validated four-V620 Qwen3.6 35B graph, the ten full-attention `attn_norm` sources each feed three Q8_0 projections. The cache uses ten 2304-byte entries distributed across the four device graphs. Cache-only reuse removed 20 staging launches per token; the dual producer removed the remaining ten. In a matched rocprof run, standard `quantize_q8_1<false>` dispatches fell from 935 to 815.

Temporary verification compared every cached Q8_1 byte, scale, and sum with a fresh stock quantization and every dual-producer F32 byte with the stock fused RMSNorm+MUL result. A deterministic 64-token completion and every reported top-10 probability also matched. Four-process ABBA TG128 measurements improved from 86.2102 to 86.7479 tok/s (**+0.62%**). Packed layouts, prompt-processing MMQ, and routed operations retain stock behavior.

Set `GGML_HIP_GFX1030_Q8_CACHE_TELEMETRY=1` to print per-device source/hit summaries during graph warmup. Telemetry alone only reports opportunities; it does not allocate or reuse cache entries.

### Routed SwiGLU to Q8_1 staging

With both the master switch and `GGML_HIP_GFX1030_Q8_1_FUSION=1`, graph fusion can replace:

```text
F32 gate + F32 up -> SwiGLU F32 tensor -> Q8_1 staging -> routed down projection
```

with register-level SwiGLU evaluation inside the Q8_1 staging kernel. Eligibility is deliberately narrow:

- prompt processing only; batches eligible for MMVQ/decode are rejected immediately;
- routed `GGML_OP_MUL_MAT_ID` down projection using the MMQ path;
- F32 gate/up inputs with exact matching shapes and supported alignment;
- SwiGLU, quantized down weights, and the ordinary non-deduplicated routed layout.

TG retains normal MMVQ dispatch. Shared dense experts and broadcast/deduplicated MoE layouts retain the stock graph.

Unsafe-math can otherwise reassociate arithmetic after removing the materialized F32 tensor. The fused kernel therefore keeps an opaque register-level compiler boundary after SwiGLU. Verification compared about 330 MB across 280 Qwen dispatches with zero Q8_1 byte differences. The targeted GLU plus staging sequence fell from about 41.5 microseconds to 11.6 microseconds; alternating PP512 runs measured a smaller end-to-end improvement of roughly 0.3% (with earlier runs near 1%).

### DeltaNet sibling projections

With both the master switch and `GGML_HIP_GFX1030_GDN_SIBLING_FUSION=1`, model loading creates two persistent row-concatenated weights for recurrent Qwen35MoE 35B layers:

```text
Q8_0 [wqkv | z]       : [2048, 8192] + [2048, 4096] -> [2048, 12288]
F32  [beta | alpha]   : [2048,   32] + [2048,   32] -> [2048,    64]
```

Packed rows are copied byte-for-byte; no dequantization or requantization occurs. The graph performs two matrix multiplications instead of four and exposes the original logical tensors through correctly-strided views. Non-contiguous inputs to CUDA unary operations are materialized before use.

The loader enables this only for the Qwen35MoE 35B architecture in layer-split mode, matching ROCm buffer types, expected Q8_0/F32 types, contiguous row layouts, and models without per-weight or input scales. If an active LoRA adapter is present, graph construction conservatively falls back to all four original `build_lora_mm` paths.

The original weights remain resident to support fallback, so the fused weights add **780 MiB** total (about 195 MiB per GPU with an even four-way layer split). Observed model initialization increased by roughly 190 ms. An exact two-pointer TG prototype consumed the original Q8_0/F32 sibling weights directly but was 0.46% slower than the packed path in four-process ABBA testing; it would also forfeit the packed PP gain. The prototype was therefore rejected and the known-correct packed implementation remains.

Exact full-byte callback hashes matched for 181 canonical tensors in both PP and TG: 120 projection outputs, 30 convolution inputs, 30 recurrent final outputs, and final logits. A deterministic 32-token completion also matched byte-for-byte.

Four-V620 ABBA benchmarks with seven repetitions measured:

| Test | Sibling fusion off | Sibling fusion on | Change |
|---|---:|---:|---:|
| PP512 | 3052.22 tok/s | 3094.22 tok/s | **+1.38%** |
| TG128 | 82.57 tok/s | 86.19 tok/s | **+4.39%** |

Both arms used `GGML_HIP_GFX1030_NATIVE=1`, `GGML_HIP_GFX1030_Q8_1_FUSION=1`, `-sm layer -ts 1/1/1/1`, FlashAttention, and `-ub 256`; only the sibling-fusion switch differed.

## Validation commands

Build for gfx1030 using the normal HIP configuration, then run stock and native arms separately. Representative checks are:

```bash
# Gated DeltaNet
build/bin/test-backend-ops test -o GATED_DELTA_NET -b ROCm0
GGML_HIP_GFX1030_NATIVE=1 \
  build/bin/test-backend-ops test -o GATED_DELTA_NET -b ROCm0

# FlashAttention
build/bin/test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0
GGML_HIP_GFX1030_NATIVE=1 \
  build/bin/test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0

# Six-row selector contract
build/bin/test-mmvq-batch6-config

# Synthetic MMVQ and bounded generic routed MMID
GGML_HIP_GFX1030_NATIVE=1 build/bin/test-mmvq-rdna2
GGML_HIP_GFX1030_NATIVE=1 build/bin/test-mmid-rdna2 \
  --type q4_k --k 2048 --n 512 --batch 6 \
  --experts 64 --top-k 4 --routing hot

# Exercise the native-only validated high-top-k advisory override
GGML_HIP_GFX1030_NATIVE=1 build/bin/test-mmid-rdna2 \
  --type q4_k --k 2048 --n 512 --batch 6 \
  --experts 256 --top-k 8 --mmvq-batch6-hint

# Report and exercise exact graph-scoped standard-Q8_1 TG reuse
GGML_HIP_GFX1030_NATIVE=1 \
GGML_HIP_GFX1030_Q8_CACHE=1 \
GGML_HIP_GFX1030_Q8_CACHE_TELEMETRY=1 \
build/bin/llama-bench -m model.gguf -ngl 999 -fa on -p 0 -n 128 -r 5
```

Use the guarded scripts when collecting reproducible artifacts:

- `scripts/benchmark-gfx1030-mmvq.py`
- `scripts/verify-gfx1030-mmvq-run.py`
- `scripts/benchmark-gfx1030-native-fa.py`
- `scripts/verify-gfx1030-native-fa-run.py`

## Related model preparation

The native paths do not require a special GGUF. The separate Qwen Q4S8 quantization, calibration, quality, and benchmark report is maintained in [edwinbrowwn/gguf-q4s8](https://github.com/edwinbrowwn/gguf-q4s8). It is independent of the runtime environment variables above.