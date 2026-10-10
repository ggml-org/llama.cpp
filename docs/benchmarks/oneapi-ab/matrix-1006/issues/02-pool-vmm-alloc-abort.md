# sycl: ggml_sycl_pool_vmm::alloc aborts with "Failed to allocate physical memory" when --fit-target leaves little headroom

Draft, not filed. Evidence: /mnt/nvme1/oneapi-ab/matrix-1006/ (spec3/, spec4/, correct2/, corefiles noted below).

## Symptom
```
Failed to allocate physical memory.
SYCL error: CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_size)): Exception caught in this line of code.
  ggml/src/ggml-sycl/common.hpp:174: SYCL error   (via ggml_sycl_pool_vmm::alloc <- ggml_sycl_op_mul_mat_sycl)
```
(and, with GGML_SYCL_ENABLE_VMM=0, `UR_RESULT_ERROR_OUT_OF_RESOURCES`). The process aborts (rc 134).

## Where it happened (A770 16 GB, xe, b12327.c4b9d6430, `--fit on --fit-target 1024`)
| workload | 1024 MiB | 2048 | 3072 |
|---|---|---|---|
| Qwen4Exp trunk, `--spec-type ngram-mod` / `ngram-simple` | abort at first request | runs | runs |
| Ornith Q4_K_M, `--prefetch-experts-slots 4 --spec-type ngram-mod` | abort (also with UR_L0_USE_COPY_ENGINE=1) | not run | runs |
| Ornith, llama-perplexity `-c 8192 -b 2048 -ub 512` (fitt 1024, q8_0 and f16 KV, MKL on and off) | abort in 8 of 8 (2 rounds) | not run | not run (retry used -b 512 -fitt 3072: ran) |

Runs that did not abort at 1024: the same Ornith server with ngram-mod alone, prefetch alone, MTP drafting on Qwen4Exp, and llama-bench
at depth up to 64k. One server log shows `SYCL0 has 84.3 MiB free, the device copy needs 62.8 MiB on top of the 1024.0 MiB margin`
mid-request, i.e. the free memory after load was far below the requested margin.

## Added evidence (later runs)
- With `--prefetch-experts-slots 4` the server reports 104.7 MiB free after load against 951 MiB without prefetch: the slots take about 850 MiB
  that the fit budget did not reserve. That config aborts at 1024 and runs at 3072.
- Qwen4Exp + n-gram abort with 953 MiB free after load, the same free amount as an Ornith run that does not abort, so there the trigger is the
  batch-time allocation, not low free memory at load.
- A leftover llama-server from a killed script (holding the card) made a plain Ornith `ngram-mod` launch abort at 1 request with 947 MiB reported free.
  A competing VRAM holder therefore reproduces the signature; I did not check for strays before every earlier abort.

## Hypothesis (not proven)
Per-op temporary buffers from the SYCL pool (mul_mat dequantization or MoE staging for large or batched inputs) are not part of the
budget `--fit` computes, so a batch larger than the planned one cannot get its reserve and the allocation aborts instead of
degrading.

## Asked
Either include the pool's worst-case reserve for the configured batch/verify sizes in the fit budget, or make the allocation failure
non-fatal (fall back to a smaller reserve or the host path).

## Not claimed
Single machine. The three aborts may have different triggers; I only know they share the allocation site and clear with more headroom.
The pool's actual peak was not measured. The earlier prefetch crash under `ngram-mod` was observed once at 1024 and not run at 2048.
