# V620 RDNA2 Q4_0 DOT8 experiment

Branch: `exp/rdna2-q4-0-dot8-v620`\
Parent: `exp/rdna2-q4k-mmid-batch6-pr23685`\
Implementation commits: `b4afc40ce`, `ed3565903`

## Scope

This is an opt-in Q4_0 MMVQ/decode prototype for HIP RDNA2/gfx1030 only. It leaves the GGUF and Q4_0 weights unchanged, keeps the existing Q8_1 quantizer, and retains the stock DP4A path. Enable it with:

```bash
GGML_HIP_Q4_0_DOT8=1
```

Unset or zero keeps the stock path. The selector is compiled as an A/B kernel specialization so the stock path retains compile-time function selection and inlining.

The prototype reconstructs the existing integer accumulation using existing Q8 bytes and three DOT8 operations per eight Q4 values: UDOT8 for low digits, SDOT8 for high digits, and SDOT8 against ones for the high-digit correction. It intentionally does not yet add a split-Q8 activation buffer or correction sidecar.

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
```

Benchmark settings: four V620s, layer split `1/1/1/1`, `--flash-attn on`, `-b 2048`, `-ub 256`, `-r 3`, Qwen3.6-35B-A3B-Q4_0.

## Correctness and ISA

- Exhaustive/random decomposition checks: **34,096 checks, zero mismatches**.
- Standalone gfx1030 probe compiled and ran: `sdot8=-4`, `udot8=60`.
- Generated device assembly contains actual `v_dot8_u32_u4` and `v_dot8_i32_i4` instructions.
- Q4_0 MMVQ synthetic A/B, K=4096/N=512/batch=1: **0/512 mismatches**, max abs `0`.
- The stock path remains the default when `GGML_HIP_Q4_0_DOT8` is unset.

## Results

Synthetic Q4_0 MMVQ, five timed graph executions:

| Path | Average | Logical GMAC/s |
|---|---:|---:|
| Stock DP4A | 23.40 us | 89.62 |
| DOT8 prototype | 23.60 us | 88.86 |

Qwen3.6 raw TG, three repetitions per row:

| Generation | Stock | DOT8 | Change |
|---:|---:|---:|---:|
| 128 | 90.744 | 90.797 | +0.06% |
| 256 | 91.600 | 91.559 | -0.05% |
| 512 | 91.819 | 91.687 | -0.14% |

## Conclusion

The exact packed-Q4/lossless-split-Q8 arithmetic works and gfx1030 emits the intended DOT8 instructions. This first prototype is performance-neutral on the V620; it is not an end-to-end win. The likely next optimization, only if continued, is activation-side correction metadata to remove the third DOT8 from the hot loop. No automatic/default enablement is recommended yet.
