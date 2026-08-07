# V620 RDNA2 Q4_0 DOT8 experiment

Branch: `exp/rdna2-q4-0-dot8-v620`\\
Parent: `exp/rdna2-q4k-mmid-batch6-pr23685`\\
Implementation commits: `b4afc40ce`, `ed3565903`, `873653deb`

## Scope

This is an opt-in Q4_0 MMVQ/decode prototype for HIP RDNA2/gfx1030 only. GGUF files, Q4_0 weights, NVIDIA behavior, other quantizations, and the stock DP4A path are preserved. Enable it with:

```bash
GGML_HIP_Q4_0_DOT8=1
```

Unset or zero keeps the stock path. The selector is compiled as an A/B kernel specialization so stock execution does not enter the DOT8 arithmetic.

The first prototype used existing Q8_1 bytes and three DOT8 operations per eight Q4 values. The follow-up keeps the normal 36-byte `block_q8_1` unchanged and adds only a two-byte-per-block `block_q8_1_sum_hi` sidecar. The optional Q8 quantizer variant writes that sidecar while producing the ordinary Q8 bytes. The DOT8 vec-dot loads the existing Q8 bytes, constructs interleaved LO/HI words in registers, and uses two independent accumulators:

```text
acc_lo = UDOT8(Q4, q8_lo, acc_lo)
acc_hi = SDOT8(Q4_signed, q8_hi, acc_hi)
```

The two accumulators are combined after the loop with the sidecar correction. No activation repacking, W4A4, or Q4_K path is included.

## Reproducibility

```text
commit before experiment: 0711cdf685c4e8f1db6fe46102428d3f485aeaea
HIP/ROCm: 7.14.60850-0000000
AMD clang: 23.0.0git, patched LLVM 46fcb339fb61119b337f973c7ca9e710a319fdd0
GPU: gfx1030 / Radeon Pro V620
model SHA256: 52312daa5b2190c1f5723d33c3315c01c55af4206f6c6e6eb63f3d8dd52bb85e
```

Build:

```bash
make -C build llama-bench -j8
make -C build/tests test-mmid-rdna2 -j8
```

Qwen benchmark settings: four V620s, layer split `1/1/1/1`, `--flash-attn on`, `-b 2048`, `-ub 256`, Qwen3.6-35B-A3B-Q4_0.

## Correctness and ISA

- Exhaustive/random Q8 digit decomposition: **34,096 checks, zero mismatches**.
- Standalone gfx1030 probe: `sdot8=-4`, `udot8=60`.
- Q4_0 K=2048/N=512: **0/512 mismatches**, max abs `1.22e-4`.
- Q4_0 K=512/N=2048: **0/2048 mismatches**, max abs `1.53e-5`.
- Q4_0 K=8192/N=1024: **0/1024 mismatches**, max abs `4.88e-4`.
- Routed Q4_0 MMID, K=2048/N=512/batch=1/top-k=8/explicit-multi: **0/4096 mismatches**, max abs `1.22e-4`.
- Existing Q4_K routed MMID with DOT8 unset: **0/4096 mismatches**.
- Generated gfx1030 assembly contains native `v_dot8_u32_u4` and `v_dot8_i32_i4`; the hot loop has no SDOT8-against-ones correction.

## Synthetic Qwen-shaped kernels

Thirty timed graph executions on the first V620:

| Shape KxN | Stock | 36-byte + sidecar DOT8 | Time change |
|---:|---:|---:|---:|
| 2048x512 | 25.80 us | 30.97 us | +20.0% |
| 512x2048 | 29.37 us | 27.23 us | -7.3% |

The two shapes have opposite behavior, so neither is a standalone justification for enabling the path.

Routed Q4_0 MMID with 64 experts, batch=1, top-k=8, explicit multi-column activation, 30 timed executions:

| Path | Average | Logical GMAC/s |
|---|---:|---:|
| Stock DP4A | 31.43 us | 266.87 |
| 36-byte + sidecar DOT8 | 30.60 us | 274.14 |

Both routed runs were bit-equivalent under the A/B harness.

## Qwen3.6 end-to-end decode

Three repetitions per row, four V620s, matched settings:

| Generation | Stock | Sidecar DOT8 | Change |
|---:|---:|---:|---:|
| 128 | 91.129 | 90.429 | -0.77% |
| 256 | 92.023 | 91.364 | -0.72% |
| 512 | 92.180 | 91.452 | -0.79% |

A five-repetition TG512 rerun measured 92.268 tok/s stock versus 91.612 tok/s DOT8, or **-0.71%**. This is a small, repeatable-looking difference, not a decisive optimization result; it is close to the noise floor relative to the added implementation complexity.

## Kernel profiling

ROCm kernel traces confirmed that the actual Qwen Q4_0 MoE path executes the experimental kernels:

```text
mul_mat_vec_q<(ggml_type)2, 1, false, false, 32, true>(...)
quantize_q8_1<true>(...)
```

The same `...true` MMVQ kernel was observed for routed top-k=8 MMID. For Qwen non-fusion Q4_0 MMVQ, the trace reported:

| Metric | Stock | Sidecar DOT8 |
|---|---:|---:|
| VGPR count | 16 | 24 |
| SGPR count | 128 | 128 |
| Workgroup | 32x1 | 32x1 |
| Grid | 262144x1x1 | 262144x1x1 |

The routed trace used grid `16384x8x1` and the same 16-to-24 VGPR increase. The DOT8 assembly has the expected extra Q8 byte loads plus the sidecar load; the sidecar is generated in the existing quantization kernel, so there is no separate sidecar-launch penalty.

Aggregate Qwen trace statistics showed the non-fusion Q4_0 MMVQ kernel at 16.03 us stock versus 16.31 us DOT8, while the fusion kernel was 21.43 us versus 21.38 us. The experimental quantizer averaged 1.485 us per call versus 1.227 us for the stock quantizer in that trace. These measurements explain why the end-to-end result is effectively neutral despite the two-DOT8 arithmetic.

## Decision

The suggested narrow experiment is complete:

- Normal 36-byte Q8_1 storage is preserved.
- The correction is a minimal two-byte sidecar.
- LO/HI packing occurs in registers.
- Two independent DOT8 accumulators are emitted.
- Actual Qwen and routed MoE kernels execute the path.
- Output equivalence is established across ordinary and routed shapes.

The sidecar version is correct and can remain as a local opt-in experiment, but it does not provide a sufficiently strong W4A8 win on V620. Further lossless W4A8 DOT8 work should stop unless a larger gain appears with a lower-level scheduling/layout change. W4A4 is the next materially different direction because it can reduce the arithmetic and activation integer storage by another factor of two.
