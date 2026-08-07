# V620 RDNA2 Q4_0 DOT8 experiment

Branch: `exp/rdna2-q4-0-dot8-v620`\
Parent: `exp/rdna2-q4k-mmid-batch6-pr23685`\
Implementation commits: `b4afc40ce`, `ed3565903`, `c760f124b`

## Scope

This is an opt-in Q4_0 MMVQ/decode prototype for HIP RDNA2/gfx1030 only. It leaves the GGUF and Q4_0 weights unchanged, keeps the existing Q8_1 quantizer, and retains the stock DP4A path. Enable it with:

```bash
GGML_HIP_Q4_0_DOT8=1
```

Unset or zero keeps the stock path. The selector is compiled as an A/B kernel specialization so the stock path retains compile-time function selection and inlining.

The first prototype reconstructed the existing integer accumulation from existing Q8 bytes using three DOT8 operations per eight Q4 values. The follow-up metadata variant quantizes directly into an internal 40-byte block containing `ds`, four packed LO words, four packed HI words, and two `int8_t` 16-value `sum_hi` corrections. It uses only UDOT8 plus SDOT8 in the weight loop; the stock Q8 scale and original floating-point sum remain unchanged.

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

## Activation-side metadata follow-up

The metadata variant was validated with the same A/B harness:

- Q4_0 K=4096/N=512/batch=1: **0/512 mismatches**.
- Q4_0 K=8192/N=1024/batch=1: **0/1024 mismatches**.
- Q4_K routed MMID with the selector unset: **0/4096 mismatches**, confirming the default non-Q4_0 path is unchanged.
- Final gfx1030 assembly contains `12` `v_dot8_u32_u4` and `12` `v_dot8_i32_i4` occurrences in the generated MMVQ device source, with no `SDOT8` ones operation in the hot path.

The final metadata microbenchmark measured 23.20 us stock versus 22.60 us DOT8 for K=4096/N=512/batch=1, but the end-to-end model result was negative:

| Generation | Stock | 2-DOT8 metadata | Change |
|---:|---:|---:|---:|
| 128 | 90.846 | 90.633 | -0.23% |
| 256 | 91.751 | 91.229 | -0.57% |
| 512 | 91.746 | 91.324 | -0.46% |

The activation-side metadata successfully removes the third DOT8 and preserves exact output, but the additional activation layout/quantization and 40-byte block cost outweigh the arithmetic savings on this V620/Qwen workload. The experimental path remains opt-in; no default enablement, repacking, W4A4, or Q4_K expansion is recommended.
