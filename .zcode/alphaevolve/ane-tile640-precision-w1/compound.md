# compound.md - run: ane-tile640-precision-w1

## Final state
- No champions stacked. All 6 candidates evaluated; none improved over baseline.
- Baseline (g2 F16 pre-dequant, 4 x 1024 tile path): 128x4096x1 max_rel_err = 7.25e-03
- 1e-1 bar is met by the current g2 path. The matmul precision is not a bottleneck.

## Greedy restack
- No candidates to restack. The review branch is the baseline + documentation.

## Per-gene summary
- g1 (Kahan summation): scaffold, -9.02e-02 (worse than baseline). MIL-level approximation (split-K 2 halves) doesn't help. ANE opaque matmul blocks true in-loop Kahan.
- g2 (smaller tiles 8x512): scaffold, -7.25e-03 (same as baseline). ANE matmul precision similar at 512 and 1024 inner-dim.
- g3 (rescale before fp16): scaffold, -8.63e-02 (worse than baseline). Extra fp16 mul ops add rounding error.
- g4 (int8 matmul): scaffold, -3.21e-01 (much worse). Fixed scale 0.01 too coarse. ANE matmul doesn't accept int8.
- g5 (two-stage matmul): non-repro. Dispatch expects single output y; 2-output .mlmodelc fails to load. Requires dispatch change.
- g6 (fp32 accumulation): scaffold, -7.25e-03 (same as baseline). ANE matmul ignores compute_precision setting.

## Key finding
The spec's baseline of 1.66e-1 is from a pre-g2 state. The current g2 (F16 pre-dequant) path achieves 7.25e-3 on the dispatch path (C++ test, seed 0x8DDC), which is well under the 1e-1 bar. The 128x4096x1 dense case already passes the 1e-1 bar with the shipped g2 path. None of the 6 candidates (Kahan, smaller tiles, rescale, int8, two-stage, fp32 accumulation) improve over this baseline. The matmul precision is at the ANE's fp16 limit.

## Wall-clock
- Total: ~45 min (within budget)
- Per gene: ~5-10 min
- Build: ~3 min (dispatch rebuild for g2)
- Harness: ~30s per evaluation

## Cleanup
Safe to remove (after architect review):
- Worktrees: `.zcode/alphaevolve/ane-tile640-precision-w1/integration/worktrees/` (already purged)
- Branches: `evolve/ane-tile640-precision-w1/g1` through `g6` (already deleted)
- The review branch `evolve-review/ane-tile640-precision-w1` is the durable artifact.

## Cross-run findings
- ANE matmul is fp16 mac regardless of compute_precision setting
- ANE matmul doesn't accept int8 inputs directly
- ANE matmul precision is similar for 512 and 1024 inner-dim
- 2-output .mlmodelc requires dispatch changes
- The 1.66e-1 baseline in the spec is from a pre-g2 state
