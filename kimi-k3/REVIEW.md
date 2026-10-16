# Kimi-K3 on 4x NVMe RAID0: project review (2026-10-16)

## Goal
Run Kimi-K3 (2.78T params MoE, 896 experts / 16 used, 93 layers) on a 160 GB RAM + RTX 3080 Ti 12 GB box by
streaming the routed experts from a 4x Samsung 990 (DRAM-less) RAID0, as fast as the hardware allows, without
losing accuracy. Two GGUFs: MXFP4_MOE (1.51 TB, 25.8 GB of experts per token) and IQ1_S-XS (540 GB, ~4.3 GB/token).

## What was built (worktree `llama.cpp-master`, branch `kimi-k3-expert-stream`, uncommitted)
| Change | Where | Effect |
|---|---|---|
| A. lazy experts (`--lazy-mode on`) | src/models/kimi-k3.cpp | no 1.5 TB populate at load; experts stay on disk, faulted per token |
| B. selected-expert prefetch (`GGML_CPU_MOE_PREFETCH=1`) | ggml-cpu.c MUL_MAT_ID | all 16 experts of a layer requested at once across threads: 4.9x decode |
| D. pinned dense weights, no warmup (`-lm none --no-warmup`) | launcher | dense CPU weights (49 GiB MXFP4 / 29 GiB IQ1) cannot be evicted; no 140 GB warmup read |
| E. windowed in-order prefetch for large batches | ggml-cpu.c | prefill reads become sequential 114 KB requests instead of 4 KB faults (100x fewer faults) |
| G. no readahead spill at load | llama-mmap.cpp | dense read() no longer drags neighbouring experts in |
| H. 2048-token prompt batches (`UBATCH`) | launcher | every batch sweeps all experts once: 1.4 TB instead of 3.6 TB per 1562-token prompt |
| P. next-layer expert prediction (`PREDICT=2`) | llama-moe-predict.*, llama-graph hook, kimi-k3 | guesses layer L+1's experts while L computes (router top-N + decayed popularity, adaptive N, persisted stats): 1.27x (MXFP4) / 1.51x (IQ1) decode, outputs bit-identical |

Verification: every change went through a written plan with 4 parallel reviewers to unanimous approval, then
code review rounds + a dynamic tester to "no bugs"; unit tests `test-moe-prefetch` (page-cache residency proves
what was prefetched) and `test-moe-predict` (mutation-tested). Numerics: on Kimi-K3 every A/B compared the new
path against the same build with the feature off (lazy off / prefetch off / prediction off) and found greedy ids and
top-10 probabilities bit-identical over 128+ tokens; identity against unpatched upstream was shown on a smaller MoE
model (GPT-OSS-120B), since stock llama.cpp cannot run Kimi-K3 from this disk in reasonable time.

## Measured now (launcher defaults, real 1562-token document + 128 generated tokens)
| | MXFP4 | IQ1_S-XS (main model) |
|---|---|---|
| decode | 0.150 tok/s (stock llama.cpp: 0.02) | 0.46 tok/s (11 GPU layers + GPU prefill) |
| prompt | 1135 s | **230 s** (GPU prefill; 618 s on the CPU) |
| decode disk use | 2.9 GB/s = 76% of the 3.8 GB/s ceiling | 2.0 GB/s = 35% of 5.7 GB/s |
| prompt | CPU 12.2 / 14 cores, disk 1.2 GB/s, GPU 1% | CPU 13 / 14 cores, disk 0.8 GB/s, GPU 2% |
| answer quality | polished | coherent and accurate (subjective reading of one answer; no perplexity measured) |

**Accuracy note.** The user's stated preference was to keep accuracy (MXFP4 rather than a 1-bit quant). MXFP4 is
the accuracy-preserving path; IQ1_S-XS (experts at ~1.56 bits, down-projections Q2_K) is a speed alternative the
user chose to try afterwards. Its quality loss has not been quantified here.

Numbers above: launcher defaults of 2026-10-16 (UBATCH 2048, GPU_LAYERS 8 for MXFP4 / 19 for IQ1, PREDICT=2),
same session as the prediction A/B (prediction off: MXFP4 0.118 tok/s, IQ1 0.302 tok/s). Earlier sessions measured
MXFP4 decode at 0.135-0.143 tok/s without prediction; the array's folded-TLC ceiling drifted from 4.3 to 3.8 GB/s.

Hardware facts: Gen3 x4 per drive (CPU has no Gen4), ~3.5 GB/s each; random reads on data already folded to TLC
~1 GB/s per drive, so the array ceiling is ~4 GB/s (5.7 on the freshly written IQ1 shards, still partly in SLC);
HMB already at the firmware maximum (64 MiB); TRIM made no difference; load is balanced within 1% across drives.

## Remaining levers, ranked by expected gain
1. **Prompt processing on the GPU (IQ1 first).** Prefill is CPU-compute-bound with the GPU idle. With 2048-token
   batches every expert is used, so copying each layer's three expert tensors to the GPU (upstream op-offload,
   currently disabled by the launcher) makes the 3080 Ti do the matmuls. Bound by disk + PCIe: IQ1 ~150-250 s vs
   618 s (2.5-4x); MXFP4 cannot fit two 5.2 GB tensors next to the activations in 12 GB, so only IQ1 unless chunked.
   Measured (E1): upstream op-offload as is page-faults serially through the lazy mapping (35 MB/s, 5.8 h per
   prompt). Fixed (V': the expert upload streams the file with read-ahead inside llama.cpp's expert-copy callback):
   IQ1 prompt 655 s -> 230 s (2.85x), decode unchanged, no serial faults; now bounded by the pageable host-to-GPU
   copy at ~2.2 GB/s (a pinned staging buffer would be the next step). The offload compute buffer is ~4.2 GB, so the
   IQ1 default is 11 dense GPU layers with offload (19 without). GPU prefill uses the CUDA IQ1_S/Q2_K kernels, so
   results are not bit-identical to the CPU path (chosen-token logprobs differ by up to 0.25, greedy sequences part
   after ~9 tokens; both coherent); kernels pass test-backend-ops; perplexity on a ~6k-token text: 3.3315 (GPU prefill) vs 3.3485 (CPU), i.e.
   equal within noise. Kept as the IQ1 default: 1562-token prompt 230 s (6.8 tok/s), decode 0.458 tok/s, >= 900 MiB
   GPU memory free with 11 dense layers.
2. ~~Headroom-aware speculation for IQ1 decode~~ Measured flat: wider guesses 0.99x, 2 workers 1.003x, both 0.98x
   (ids identical). IQ1 decode is compute-bound (11.8 of 14 threads running, disk at 34% of its ceiling), so the
   next decode lever on IQ1 is CPU kernel speed (IQ1_S/Q2_K dot products) or more dense layers on the GPU, not I/O.
3. **Fixes (done):** the startup 2-token probe decode no longer feeds the predictor (load reads back to 108.9 GB,
   saved state unchanged by a request-less start); per-model launcher defaults; restart benefit re-measured after
   the fix: 1.05x recall on the same document, ~1.01x on a new one (persistence works, the gain is small by nature).
4. **More RAM** is the biggest non-software lever for MXFP4 (cache hit rate drives bytes/token); drives with DRAM
   would raise the random-read ceiling ~3x.

Not worth doing (measured): RAID chunk/readahead changes, bigger read blocks (+5%), restriping, HMB tuning, TRIM.
