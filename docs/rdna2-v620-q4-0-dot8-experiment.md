# V620 RDNA2 Q4_0 DOT8 experiment

Branch: `exp/rdna2-q4-0-dot8-v620`\\
Parent: `exp/rdna2-q4k-mmid-batch6-pr23685`\\
Implementation commits: `b4afc40ce`, `ed3565903`, `873653deb`, `7dd6d1fd6`, `f8c7dce9b`

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

## Verified kernel-structure facts

- `block_q4_0` is exactly 18 bytes: a 2-byte FP16 scale followed by 16 bytes of packed quants. `qs` is therefore only guaranteed 2-byte aligned, and successive blocks advance by 18 bytes.
- `get_int_b2()` assumes 2-byte alignment and combines two 16-bit words. In generated gfx1030 ISA, LLVM coalesces the Q4 quant load into `global_load_dwordx2 ... offset:2`; it does not literally emit two separate 16-bit loads in the DOT8 hot kernel. The load remains based on the 18-byte AoS layout and can be unaligned relative to larger boundaries, so repacking remains plausible but is not yet proven necessary.
- No device calls were found around the vec-dot path; the `__forceinline__` functions are inlined.
- `MMVQ_PARAMETERS_RDNA2` exists, but `calc_nwarps()` and `calc_rows_per_block()` have no RDNA2-specific branches. RDNA2 Q4_0 MMVQ therefore defaults to one warp and one row per block.

## Activation-reuse / rows-per-block experiment

A temporary compile-time sweep tested rows-per-block values 1, 2, 4, and 8. On isolated synthetic Q4_0 kernels, rows-four sometimes reduced kernel time and the generated ISA showed one Q8 activation load set feeding multiple row accumulators. However, this did not translate to Qwen decode performance.

The rows-four Qwen trial used the temporary `GGML_HIP_Q4_0_DOT8_MMVQ_ROWS=4` build. Lower `avg_ts` means lower throughput, so the result is a regression:

| Generation | Stock rows=1 | Temporary DOT8 rows=4 | Time change |
|---:|---:|---:|---:|
| 128 | 91.002 tok/s | 88.431 tok/s | +2.91% |
| 256 | 91.884 tok/s | 89.227 tok/s | +2.98% |
| 512 | 92.125 tok/s | 89.454 tok/s | +2.98% |

The rows-four change was reverted. Stock and the final opt-in DOT8 path both remain rows-one.

Routed Q4_0 MMID with 64 experts, batch=1, top-k=8, explicit multi-column activation also regressed in the temporary rows-four trial: stock 27.60 us versus DOT8 rows-four 30.75 us. Both paths remained bit-equivalent.

## Qwen3.6 end-to-end decode with final rows-one path

Four V620s, layer split `1/1/1/1`, flash attention, `-b 2048`, `-ub 256`, three repetitions:

| Generation | Stock rows=1 | Sidecar DOT8 rows=1 | Throughput change |
|---:|---:|---:|---:|
| 128 | 91.129 | 90.429 | -0.77% |
| 256 | 92.023 | 91.364 | -0.72% |
| 512 | 92.180 | 91.452 | -0.79% |

A five-repetition TG512 rerun measured 92.268 tok/s stock versus 91.612 tok/s DOT8, or **-0.71%**. The rows-four reuse idea therefore did not rescue the lossless W4A8 path.

## Kernel profiling and compiler flag check

ROCm kernel traces confirmed that actual Qwen Q4_0 execution uses:

```text
mul_mat_vec_q<(ggml_type)2, 1, false, false, 32, true>(...)
quantize_q8_1<true>(...)
```

The temporary rows-four kernel increased VGPR usage from 24 to 40 for DOT8, while stock rows-one reported 16 and SGPR usage remained 128. The generated rows-four ISA showed activation loads feeding multiple row accumulators, so the proposed reuse exists; it is simply not beneficial end-to-end for this workload.

A clean HIP build with:

```text
-mllvm -amdgpu-unroll-threshold-local=600
```

showed no material improvement: TG512 measured 92.122 stock / 89.391 DOT8 with the flag versus approximately 92.125 / 89.454 without it.

## Decision

The new analysis correctly identified real structural issues: 18-byte Q4_0 AoS storage, conservative RDNA2 one-row geometry, and activation reuse opportunities. The fast rows-four isolated-kernel result was not representative of Qwen end-to-end behavior, so it was reverted.

The final sidecar DOT8 path remains correct and opt-in but is approximately neutral-to-0.7% slower on Qwen. Do not repack Q4_0 yet: the compiler already coalesces quant loads into 64-bit operations, and repacking would broaden model-load/storage changes without a demonstrated end-to-end bottleneck. W4A4 remains the higher-ceiling future direction.
