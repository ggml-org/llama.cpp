# plan.md - run family: memopt-pipeline

The research surfaced 8 candidate approaches (S1-S8). They are run as three
waves, gated on the dependencies and the file-region analysis in research.md.
Each wave is its own run; waves 2 and 3 build on the finalized HEAD of the
prior wave so shipped work compounds.

## Wave 1: S1 (kernel path) - SINGLE GENE (proof of concept)

Scope: seed ONLY S1, single gene. This wave is the proof that the alphaevolve
  loop works end-to-end on tessera before committing compute to the bigger
  fan-out. One gene, focused, deeper budget.
Why single: S1 edits the paged-attn kernels on CPU AND Metal AND ANE at once
  (ggml-cpu/ops.cpp:9250+, ggml-metal.metal:11240+, ggml-ane.mm:1202+, plus
  src/llama-kv-cache.cpp:2776+). Multiple genes hitting those same files would
  collide on every commit. One gene is correct both for the file-conflict
  reason and for proof-of-concept discipline.
Approach within the gene: start with q8_0 (lower risk, ~50% KV win, near-
  lossless) before attempting q4_0. The gene's evolution can explore both, but
  the seed should bias toward q8_0-first so the early generations establish the
  dequant-on-read mechanism; q4_0 follows naturally as a mutation once the q8
  reader works.
Baseline: f02cda5d (the WIP stash snapshot from the dry Phase 0+1).
Run name: memopt-pipeline-w1

Commits: per the updated profile, commits ARE allowed on scratch refs
  (evolve/<run>/*, champions/<gene-id>, integration/main) and the final
  review branch (evolve-review/memopt-pipeline-w1). NEVER on master/main.
  After finalize, inspect the review branch:
    git log --oneline f02cda5d..evolve-review/memopt-pipeline-w1
    git diff f02cda5d...evolve-review/memopt-pipeline-w1
  Merge or cherry-pick to master when you choose.

After wave 1 finalize: record the new HEAD as wave1_head below.

  wave1_head: <to be filled after finalize>

## Wave 2: S2, S3, S7 (disjoint, parallel fan-out) - 3-4 ISLANDS

Scope: seed S2, S3, S7 (defer S4, S5, S6, S8).
Why parallel works here: these three are disjoint at the file level.
  - S2 (mmap KV residency)    -> src/llama-kv-cache.cpp, src/llama-memory*.cpp
  - S3 (KV eviction policy)   -> src/llama-kv-cache.{h,cpp}, src/llama-kv-cells.h
  - S7 (CacheGen compression) -> tools/server/server-context.cpp (radix store)
  S2 and S3 both touch kv-cache.cpp but typically different functions (buffer
  alloc vs eviction hooks); the merge step in Phase 4/5 will catch real
  conflicts. Each can be its own island with low collision risk.
Baseline: wave1_head (NOT the original f02cda5d). S1's quantized-KV reader is
  the foundation wave 2 builds on - running wave 2 against the pre-S1 baseline
  would waste the whole wave.
Run name: memopt-pipeline-w2
Invocation: init with the `reuse memopt-pipeline` clause so Phase 1 is skipped
  (research.md is already done) and `| S2, S3, S7` to scope the seed pool.

After wave 2 finalize: record the new HEAD as wave2_head below.

  wave2_head: <to be filled after finalize>

## Wave 3: S4, S5, S6, S8 (the rest, dependencies now satisfied) - up to 4 ISLANDS

Scope: seed S4, S5, S6, S8 (and revisit any wave-1/2 candidate that did NOT
  stack successfully - see "Re-promotion" below).
Dependency check at wave 3 start:
  - S4 (KIVI 2-bit KV) extends S1 -> S1 shipped in wave 1. OK.
  - S5 (InfiniGen prefetch) needs a host-tier KV cache -> S2 (mmap-backed KV)
    shipped in wave 2. OK if S2 stacked; if S2 was skipped/non-compounding,
    DEFER S5 again.
  - S6 (MoE disk offload) is independent of S1/S2/S3 - could have run in any
    wave. Held to here because it's the largest implementation surface and
    only pays off if the workload model is MoE. Confirm the workload model
    before starting S6.
  - S8 (speculative expert prefetch) extends S6 -> only start if S6 shipped.
Baseline: wave2_head.
Run name: memopt-pipeline-w3
Invocation: init with `reuse memopt-pipeline` and `| S4, S5, S6, S8`.

## Re-promotion across waves

A candidate that did NOT stack in its wave (logged non-compounding or conflict
in integration/compound.md) is NOT permanently lost. Its champion branch
survives in that wave's integration repo. At each later wave's finalize, the
candidate's exported patch can be re-tried against the richer baseline that
includes later waves' shipped work. Concretely: at wave 3 finalize, gather
patches from ALL three waves' integration/patches/ directories and re-stack
greedily - the compounding test runs against a baseline that has S1 + S2 + S3
+ S7 shipped, which may unlock a patch that under-gained against wave 1's
sparser main.

## Per-wave budgets

Wave 1 (kernel work, single island): allow deeper budget per candidate since
  there are few of them. Suggest 15 generations / 120 min / 100 candidates.
Wave 2 (3-4 parallel islands): standard budget. 12 gens / 90 min / 200
  candidates total across agents.
Wave 3 (up to 4 islands, larger surfaces): standard-to-generous. 12 gens /
  120 min / 250 candidates.

These are defaults for the joiners to inherit; tighten or relax in each wave's
spec.md at init time.

## What this plan does NOT do

- Does NOT auto-launch. Each wave's init/join/finalize is fired manually via
  the control script so you can review research.md and each wave's champions
  before committing to the next wave.
- Does NOT skip the Phase 2 checkpoint. Even with the plan fixed, each wave's
  init still pauses for your single "go" after seeding, so you can redirect
  (e.g. drop S6 if the workload model turns out to be dense).
- Does NOT assume success. If wave 1's S1 doesn't ship (every candidate
  regresses correctness or no q8_0 path beats f16), STOP and re-research
  before wave 2 - the dependency chain is broken and wave 2's baseline would
  be wrong.
