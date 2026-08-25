# Benchmark results

Generated from the JSONL logs in this directory by `make-results.py`. Do not hand-edit —
regenerate, so the tables cannot drift from the data.

Generated 2026-08-24. Hardware: single Radeon 8060S (Strix Halo APU,
gfx1151, RDNA 3.5), RADV / Mesa 26.0.8.

## How to read these numbers

**The fork-vs-mainline and gate-on-vs-off figures are ratios**, each measured within a single
session on one power profile, with arms interleaved in palindrome order. The ratios hold. The
**absolute** t/s figures in those tables were taken under the power profile in force at the time
(`power_dpm_state` recorded per row; the earlier set predates that stamp) and are not directly
comparable to the headline section, which was re-measured after the machine moved to a higher power
setting. Compare deltas across tables, not absolutes.

Every generation figure carries its context depth. These models declare 262144 context and token
generation at depth is roughly a third of its depth-0 value, so a bare t/s number is not a claim.

## 1. Fork vs pinned upstream, prefill and generation against depth

Upstream pinned at `95b8e33e1`, the exact commit this fork merged, so the delta is our changes and
not upstream drift. `-ub 512 -fa 1`, 3 internal repetitions, palindrome-ordered arms, warmup
discarded. Sub-2σ marked as noise.

| model | test | depth | mainline t/s | fork t/s | delta | |
|---|---|---:|---:|---:|---:|---|
| ornith-35b-a3b-q4km | pp2048 | 0 | 856.8 ± 2.1 | 964.7 ± 4.2 | **+12.6%** | +32.7σ |
| ornith-35b-a3b-q4km | pp2048 | 4096 | 768.6 ± 0.6 | 845.5 ± 0.6 | **+10.0%** | +125.3σ |
| ornith-35b-a3b-q4km | pp2048 | 16384 | 640.6 ± 0.5 | 678.2 ± 0.3 | **+5.9%** | +84.0σ |
| ornith-35b-a3b-q4km | pp2048 | 32768 | 515.5 ± 15.8 | 541.9 ± 3.9 | **+5.1%** | +2.3σ |
| ornith-35b-a3b-q4km | pp2048 | 65536 | 379.9 ± 0.2 | 376.5 ± 14.5 | **-0.9%** *noise* | -0.3σ |
| ornith-35b-a3b-q4km | tg64 | 0 | 65.5 ± 0.2 | 65.8 ± 0.0 | **+0.5%** *noise* | +1.7σ |
| ornith-35b-a3b-q4km | tg64 | 4096 | 63.2 ± 0.0 | 64.1 ± 0.1 | **+1.5%** | +16.1σ |
| ornith-35b-a3b-q4km | tg64 | 16384 | 58.6 ± 0.0 | 59.0 ± 0.1 | **+0.7%** | +7.7σ |
| ornith-35b-a3b-q4km | tg64 | 32768 | 52.5 ± 1.0 | 53.4 ± 0.2 | **+1.8%** *noise* | +1.3σ |
| ornith-35b-a3b-q4km | tg64 | 65536 | 45.2 ± 0.0 | 45.0 ± 0.9 | **-0.6%** *noise* | -0.4σ |
| qwen38-27b-q4kxl | pp2048 | 0 | 255.6 ± 0.1 | 288.7 ± 1.3 | **+13.0%** | +35.6σ |
| qwen38-27b-q4kxl | pp2048 | 4096 | 240.1 ± 0.9 | 267.9 ± 0.2 | **+11.6%** | +42.5σ |
| qwen38-27b-q4kxl | pp2048 | 16384 | 194.0 ± 0.0 | 211.7 ± 0.3 | **+9.1%** | +96.6σ |
| qwen38-27b-q4kxl | pp2048 | 32768 | 127.0 ± 0.1 | 137.3 ± 0.3 | **+8.2%** | +52.2σ |
| qwen38-27b-q4kxl | pp2048 | 65536 | 51.7 ± 0.1 | 54.1 ± 0.4 | **+4.6%** | +9.4σ |
| qwen38-27b-q4kxl | tg64 | 0 | 11.7 ± 0.0 | 11.6 ± 0.0 | **-0.7%** | -8.5σ |
| qwen38-27b-q4kxl | tg64 | 4096 | 11.5 ± 0.0 | 11.4 ± 0.0 | **-0.7%** | -14.8σ |
| qwen38-27b-q4kxl | tg64 | 16384 | 11.0 ± 0.0 | 11.0 ± 0.0 | **-0.4%** | -10.4σ |
| qwen38-27b-q4kxl | tg64 | 32768 | 10.4 ± 0.0 | 10.4 ± 0.0 | **-0.6%** | -11.8σ |
| qwen38-27b-q4kxl | tg64 | 65536 | 9.5 ± 0.0 | 9.4 ± 0.0 | **-0.5%** | -5.1σ |

## 2. Vulkan gate ablations

Each gate measured on/off in one binary, `pp2048 -ub 2048`. Three models across three quant
families. Transcribed from `WORKLOG.local.md` where the raw JSONL was lost with tmpfs on reboot.

| model | quant | gate | off t/s | on t/s | delta |
|---|---|---|---:|---:|---:|
| Qwen3.8-27B UD-Q4_K_XL | Q4_K_XL | `CONCAT_TRANSPOSE` | 252.0 | 260.4 | **+3.3%** |
| Qwen3.8-27B UD-Q4_K_XL | Q4_K_XL | `DENSE_F16B` | 252.0 | 263.0 | **+4.5%** |
| Qwen3.8-27B UD-Q4_K_XL | Q4_K_XL | `FUSE_UNARY_MUL` | 252.0 | 251.4 | **+0.1%** |
| Ornith-1.5-35B-A3B | Q4_K_M | `CONCAT_TRANSPOSE` | 819.2 | 1188.3 | **+45.1%** |
| Ornith-1.5-35B-A3B | Q4_K_M | `MMID_F16B` | 815.7 | 872.2 | **+6.9%** |
| Ornith-1.5-35B-A3B | Q4_K_M | `MMID_WAVE32` | 810.0 | 805.3 | **-0.6%** |
| Ornith-1.5-35B-A3B | Q4_K_M | `MMID_WG256` | 805.0 | 790.3 | **-1.8%** |
| Qwen3.6-35B-A3B HauhauCS | Q6_K_P | `CONCAT_TRANSPOSE` | 884.2 | 1320.3 | **+49.3%** |
| Qwen3.6-35B-A3B HauhauCS | Q6_K_P | `MMID_F16B` | 1204.7 | 1314.0 | **+9.1%** |
| Qwen3.6-35B-A3B HauhauCS | Q6_K_P | `DENSE_F16B` | 1253.7 | 1326.1 | **+5.8%** |

## 3. Prefill, absolute, current power profile

| model | ubatch | test | t/s |
|---|---:|---|---:|
| qwen35 27B Q4_0_ROCMFP4_FAST | 512 | pp512 | 439.3 ± 0.7 |
| qwen35 27B Q4_0_ROCMFP4_FAST | 512 | pp2048 | 427.4 ± 0.9 |
| qwen35 27B Q4_0_ROCMFP4_FAST | 2048 | pp512 | 431.0 ± 0.7 |
| qwen35 27B Q4_0_ROCMFP4_FAST | 2048 | pp2048 | 414.8 ± 0.7 |
| qwen35 27B Q4_K - Small | 512 | pp512 | 441.1 ± 3.7 |
| qwen35 27B Q4_K - Small | 512 | pp2048 | 429.7 ± 0.5 |
| qwen35 27B Q4_K - Small | 2048 | pp512 | 432.8 ± 1.0 |
| qwen35 27B Q4_K - Small | 2048 | pp2048 | 413.3 ± 1.2 |

## 4. Speculative decoding, FP4 stack

Target `Qwen3.8-27B-ROCmFP4-FAST` (13.9 GB, `Q4_0_ROCMFP4_FAST`), draft
`Qwen3.8-27B-DFlash2-Q4_0_ROCMFP4_FAST` (987 MB), DFlash2, greedy, 300 tokens, depth 0.
Fork only — upstream cannot load ROCmFPx.

| policy | workload | t/s | draft acceptance |
|---|---|---:|---:|
| fixed n=3 | prose | **24.12** | 46% |
| fixed n=3 | json | **38.52** | 97% |
| fixed n=7 | prose | **21.51** | 25% |
| fixed n=7 | json | **17.66** | 18% |
| adaptive n<=7 n>=3 | prose | **23.75** | 44% |
| adaptive n<=7 n>=3 | json | **55.03** | 96% |
| bare decode | - | **13.77** | — |

The acceptance column is the mechanism: fixed n=7 collapses to 18% acceptance, fixed n=3 sits at
97% and is therefore *under*-drafting, and adaptive holds 96% while drafting longer. The same
`n_max = 7` that destroys the fixed arm is safe under adaptive.

### Headline, re-measured on the higher power profile

| policy | workload | t/s | draft acceptance |
|---|---|---:|---:|
| bare-decode | prose | **14.07** | — |
| bare-decode | json | **13.99** | — |
| fixed-n3 | prose | **25.35** | 44% |
| fixed-n3 | json | **41.62** | 95% |
| fixed-n7 | prose | **24.82** | 25% |
| fixed-n7 | json | **20.17** | 18% |
| adaptive | prose | **26.14** | 44% |
| adaptive | json | **65.57** | 96% |

## 4. Adaptive drafting against context depth

Qwen3.8-27B UD-Q4_K_XL target. MTP uses the target's own nextn layers; DFlash2 uses the z-lab Q8_0
sidecar. Both arms adaptive, `n_max 7`, `n_min 3`.

| policy | workload | depth | prefill t/s | generation t/s | acceptance |
|---|---|---:|---:|---:|---:|
| MTP adaptive | prose | 0 | 76.7 | 21.34 | 47% |
| MTP adaptive | json | 0 | 160.4 | 44.02 | 96% |
| MTP adaptive | prose | 4096 | 385.7 | 17.24 | 36% |
| MTP adaptive | json | 4096 | 297.6 | 36.43 | 82% |
| MTP adaptive | prose | 16384 | 296.3 | 0.00 | 0% |
| MTP adaptive | json | 16384 | 206.5 | 30.06 | 74% |
| MTP adaptive | prose | 32768 | 155.2 | 18.58 | 51% |
| MTP adaptive | json | 32768 | 103.8 | 30.04 | 83% |
| DFlash adaptive | prose | 0 | 66.5 | 21.13 | 0% |
| DFlash adaptive | json | 0 | 152.9 | 49.26 | 0% |
| DFlash adaptive | prose | 4096 | 388.8 | 16.99 | 0% |
| DFlash adaptive | json | 4096 | 298.5 | 32.95 | 0% |
| DFlash adaptive | prose | 16384 | 314.5 | 16.87 | 0% |
| DFlash adaptive | json | 16384 | 232.0 | 38.90 | 0% |
| DFlash adaptive | prose | 32768 | 171.8 | 19.01 | 0% |
| DFlash adaptive | json | 32768 | 109.9 | 25.05 | 0% |

*Incomplete: the collecting run was interrupted.*
