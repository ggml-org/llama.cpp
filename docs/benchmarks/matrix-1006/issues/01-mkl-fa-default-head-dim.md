# sycl: oneMKL flash-attention route is on by default and regresses prefill at depth for head dim 128 on DG2

Draft, not filed. Evidence: /mnt/nvme1/oneapi-ab/matrix-1006/ (mkl/, mkl32k/, deep/, correct*/, oracle/).

## Summary
`GGML_SYCL_ENABLE_MKL_FA` defaults to 1. On the Arc A770 (acm-g10, xe driver, oneAPI 2026.1, fork b12327.c4b9d6430) it helps
head-dim-256 models and hurts head-dim-128 models once n_kv is above about 1k.

## Measurements (llama-bench, -fa 1, `-p 512 -n 0 -d <depth> -r 2`, mode 1 compute, quiet host unless noted)
pp512 t/s with the route off, relative to on:

| model (head dim, GQA) | KV | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|
| Llama-3.1-8B Q4_K_M (128, 4:1) | q8_0 | +32% | +20% | +57% | +58% | +64% (103.7 vs 63.4) |
| Llama-3.1-8B | f16 | +33% | +19% | +55% | +57% | |
| Qwen2.5-Coder-7B Q6_K (128, 7:1) | q8_0 | +4% | -7% | +30% | +38% | |
| Ornith-1.5-35B-A3B Q4_K_M (256) | q8_0 | -10% | -18% | -24% | -34% | -44% (116 vs 206); -53% at 64k |
| Qwen3.5-9B Q4_K_M (256) | q8_0 | -15% | -25% | -23% | -31% | |

Depth 0 is unchanged (n_kv < 1024, MKL does not engage). Decode is unchanged. The 32k rows are clean reruns (2 rounds each, foreign CPU <=12%): 8B 63.5 on / 103.7 off, Ornith 208 on / 118 off; Q8_0 Ornith at 8k: off -21.7%.
The 64k row is a single measurement.

## Numerics (llama-perplexity, ctx 8192, 3-4 chunks, deterministic on dense models)
MKL on minus off: Llama-8B -0.007 (off is higher), Qwen2.5-7B (GQA 7:1) +0.072 f16 / +0.069 q8_0, Qwen3.5-9B +0.008.
Ornith 6.807-6.822 either way, inside its known 0.01 run-to-run noise. So the MKL route is measurably less accurate than the
fallback on the 7:1 shape.

## Test coverage gap
tests/test-sycl-turbo-correctness.cpp section [4c] covers d=128, GQA 4:1, f16 and q8_0 only. With the route off, [4c] fails by design
("flash attention probe required MKL but selected another route", 6 GATE-FAIL), so the fallback at n_kv >= 1024 prefill has no oracle
check; [4]/[4b] stop at n_kv = 256. No d=256 and no 7:1 shape is covered at all.

## Suggested direction
Gate the default by head dim (on for 256, off for 128) or pick the route by a measured per-shape cost, and add a route-agnostic oracle
section for n_kv >= 1024 including d=256 and GQA 7:1.

## Not claimed
Measured on one A770 with one driver stack; other Xe parts untested. Two rounds per cell at most. The Qwen2.5 curve is non-monotonic
(4k favors MKL) and I do not know why. Perplexity has no CPU reference here, only route-to-route differences.
Head dim is the only variable that tracks the sign in this sample (two models per class); model family, expert count and layer
count change with it, so a head-dim cause is a hypothesis.
