# gfx1030 next optimization plan

> Current implemented paths, supported environment variables, correctness evidence, and benchmark results are consolidated in [Opt-in gfx1030 native optimizations](gfx1030-native-optimizations.md).

## Audit result

The research is directionally correct, but parts are stale relative to this fork:

- `ggml/src/ggml-cuda/mmvq.cu` already has `MMVQ_PARAMETERS_RDNA2`, but `calc_nwarps()` has no RDNA2 branch. RDNA2 therefore falls through to `return 1` for standard MMVQ. This is the strongest confirmed next target.
- Q8_1 packed activation layouts are already present for Q4_K/Q5_K/Q6_K. The default is 128, and HIP can override it with `GGML_HIP_MMVQ_Q8_1_BLOCK_SIZE=64|128|256`; 32 is the standard layout. A combined layout/nwarps sweep is still missing.
- Routed-MoE MMQ and typical expert-width tile selection are already present in `mmq.cu/mmq.cuh`, including `GGML_HIP_RDNA2_MMQ_J` and `GGML_HIP_RDNA2_MMQ_J_Q4_K`. Re-test and tune these rather than reimplementing the upstream idea.
- `topk-moe.cu` still uses generic shuffle pair reductions for expert selection and has no RDNA2-specific dispatch.
- Native DPP-backed wave reductions now exist for tiled FA. The same wrappers can be applied selectively to MMVQ activation quantization and Top-K/softmax after profiling.

## DFlash2 sustained-2x frontier

The validated normal-width candidate combines automatic greedy target backend sampling with opt-in width-six Q4_0/Q4_K/Q6_K rows/block=2. It measures `77.783 tok/s` at `48.554 ms/speculative cycle` with a mean committed span of `3.724`; the 2x target is `89.784 tok/s` and about `41.477 ms/cycle`. A representative five-workload mean is `83.85 tok/s` (`1.87x` AR), so 2x has not been reached.

Only mechanisms with roughly a 7 ms/cycle exact end-to-end ceiling should be pursued next:

1. TP4 M6 skinny-GEMM or weight-reuse changes that preserve the existing reduction order and output bytes.
2. Exact collective scheduling/fusion or LM-head reduction that retains all mandatory TP shards.
3. GPU-side acceptance that preserves sampler, grammar, rollback, recurrent state, and fallback semantics.

Do not retry single-GPU target verification, naïve verification chunking, forced RCCL Ring/LL/channel/thread overrides, guessed width-two P2P schedules, width-six rows/block=4, Q4_0 eight-wave MMVQ, or `p_split` tuning. Those paths were measured as slower, neutral, or incorrect. Detailed traces remain outside Git in the sprint ledger.

## General priority order

1. Add an RDNA2-specific `calc_nwarps()` table and sweep Q4_0/Q8_0/Q4_K/Q5_K/Q6_K across nwarps 1/2/4/6/8. Preserve a stock fallback and keep Q4_0 DOT8 separate from generic DP4A assumptions.
2. Sweep the existing Q8_1 layouts together with nwarps. Do not assume the W7800 128-block choice is optimal for V620.
3. Profile Q4_K then Q6_K unpack/feed code (`mmq-load-tiles.cuh`, `vecdotq.cuh`) for VALU/VMEM/packing costs before adding `v_perm_b32` or other inline assembly.
4. Extend DPP reductions first to `quantize_q8_1()` and selected MMVQ/MMVF reductions, then evaluate Top-K pair reduction with deterministic tie-breaking.
5. Measure existing routed-MoE MMQ J/typical-width selection on V620 before changing it.
6. Defer broad LDS/VGPR/VMEM and RMSNorm/softmax changes until counters identify a bottleneck.
7. Keep INT16 sdot2 as a capability helper/microkernel only; it has no current numeric I16 inference consumer.

## Guardrails

Every native change must remain behind the native profile or an explicit fail-closed feature switch, retain stock behavior when unset, honor the global RDNA2 auto-off switch, pass backend correctness, and be benchmarked with RCCL tensor split. Do not claim a gain from a new intrinsic without a matching V620 profile and end-to-end TG/PP result.