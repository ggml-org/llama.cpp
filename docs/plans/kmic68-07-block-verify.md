# P07 - Block verification

**Kind:** common-layer feature
**Depends on:** P06

## Context

P06 stops at the first rejected draft token. Block verification instead
computes the probability that each whole prefix survives, evaluates every
position, and selects the longest surviving prefix. Sun et al.,
arXiv:2403.10444, prove that the rule preserves the target distribution and
never accepts fewer tokens in expectation. P07 reuses P06's recorded target and
draft distributions and its exact per-token fallback.

The runtime contract is deliberately disabled by default:
`LLAMA_SPEC_BLOCK_VERIFY` unset or `0` selects P06; `1` selects block
verification; `2` selects the same block rule plus a structured expected-gain
line after every 256 eligible blocks. Any other or non-integer value is a startup
error. This plan does not change the default after measurement.

Source provenance: `Kmic-68/llama.cpp` branch `p100-optimizations`,
`common_spec_block_verify` and the block branch in
`common_sampler_sample_and_accept_n_dist`, based on Sun et al. Algorithm 2.

## EARS requirements

- **R07.1 (State-driven):** WHILE `LLAMA_SPEC_BLOCK_VERIFY` is `1` or `2` and the P06 block is eligible, `common_spec_block_verify` shall evaluate every drafted position without stopping at the first rejection.
- **R07.2 (Event-driven):** WHEN zero-based draft position `k` is evaluated, `common_spec_block_verify` shall compute `keep[k+1]=min(keep[k]*P[k](draft[k])/q[k](draft[k]),1)` from `keep[0]=1`.
- **R07.3 (Event-driven):** WHEN a position's survival draw fails, `common_spec_block_verify` shall continue evaluating the remaining positions.
- **R07.4 (Event-driven):** WHEN mode `2` completes 256 eligible blocks for one diagnostic identity, the block diagnostic shall report expected accepted draft tokens for P07 and P06, their draft-token difference, and the total drafted-token denominator while excluding every correction or bonus token.
- **R07.5 (Event-driven):** WHEN surviving prefix length `tau` is less than `G`, `common_spec_block_verify` shall sample its correction from normalized `max(keep[tau]*P[tau]-q[tau],0)`.
- **R07.6 (State-driven):** WHILE a sampler is stateful/constrained, emitted-token `q` mass is nonpositive, or any P06 fallback is active, speculative acceptance shall use P06's per-token rule.
- **R07.7 (State-driven):** WHILE `LLAMA_SPEC_BLOCK_VERIFY` is unset or `0`, speculative acceptance shall use P06's per-token rule.
- **R07.8 (Ubiquitous):** The block-verification acceptor shall preserve the target output distribution `p`.
- **R07.9 (Event-driven):** WHEN all block positions have been evaluated, `common_spec_block_verify` shall select the greatest prefix length whose survival draw passed.
- **R07.10 (Unwanted behaviour):** IF `LLAMA_SPEC_BLOCK_VERIFY` is not unset, `0`, `1`, or `2`, THEN the sampler initializer shall fail with the rejected value.
- **R07.11 (Event-driven):** WHEN block verification returns prefix length `tau` and token `y`, the distribution acceptor shall commit draft tokens `[0,tau)` followed by `y` to target sampler state in that order.
- **R07.12 (Ubiquitous):** The mode-2 diagnostic record shall identify its launch, sequence, acceptor instance, and proposal range.
- **R07.13 (Event-driven):** WHEN P06 atomically marks a single-use proposal consumed, the owning mode-2 diagnostic state shall add that proposal's expected-gain contribution exactly once.
- **R07.14 (Event-driven):** WHEN a server slot replaces its sampler, the slot's mode-2 diagnostic state shall start a fresh acceptor identity with zero counters.

## Approach

1. Parse `LLAMA_SPEC_BLOCK_VERIFY` once into mode 0, 1, or 2. Unset maps
   to zero. Mode and P06 RNG state clone/copy normally; diagnostic counters do
   not live in function-static or cloned sampler state.
2. Add pure `common_spec_block_verify` to `common/sampling.cpp`. Inputs are
   `P[0..G]`, `q[0..G-1]`, `draft[0..G-1]`, target draws, and P06 RNG.
   Before entry, P06 preflight must prove every emitted token has finite positive
   `q[k](draft[k])`; zero mass is whole-block P06 fallback, not a P07 branch.
3. Initialize `keep[0]=1`. For `k=0..G-1`, draw an independent uniform and
   compute `keep[k+1]` from `P[k]`, `q[k]`, and `draft[k]`. For a
   nonterminal position, compute
   `S=sum_y max(keep[k+1]*P[k+1](y)-q[k+1](y),0)` and
   `h[k+1]=S/(S+1-keep[k+1])`; for `k=G-1`, use `h[G]=keep[G]`. Set
   `tau=k+1` whenever the draw passes and continue through the block.
4. Initialize correction/bonus `y` from the target draw at `P[tau]`. If
   `tau<G`, form the residual as
   `max(keep[tau] * P[tau] - q[tau], 0)`, treating absent sparse IDs as zero.
   Sample it after normalization when mass is positive; otherwise retain the
   target draw. Commit only after `tau` and `y` are final.
5. Materialize all target distributions only after P06 whole-block preflight.
   Mode 1/2 call the helper; mode 0 and any fallback call P06 unchanged.
6. Add `common_spec_block_diag_state` owned by the long-lived acceptor caller:
   `server_slot` for server use and an explicit local object for examples/tests.
   Key it by launch, sequence, and acceptor-instance ID; retain first/last proposal
   IDs. Preserve it across `common_sampler_reset` only while the same acceptor
   and sequence remain active. Rotate its acceptor-instance ID and clear counters
   whenever `slot.smpl` is replaced, even with the same seed/configuration or a
   reused pointer address. Ordinary sampling requests create new samplers and
   cannot share a bucket. Discard any incomplete bucket on replacement; do not
   emit it as a 256-block record. Do not share state across slots, sequences, or
   cloned acceptors. Protect updates when one acceptor can be reached concurrently.
7. The pure helper computes provisional
   `E_block=sum_{r=1..G} (1-product_{i=r..G}(1-h[i]))` and
   `E_token=sum_{k=0..G-1} product_{j=0..k} min(1,P[j](draft[j])/q[j](draft[j]))`,
   then returns `(tau,y,keep,E_block,E_token)` without touching long-lived
   counters. Conditional on the recorded proposal, the final prefix reaches `r`
   if any independent draw at positions `r..G` passes; `keep[r]` is not that
   probability. Both expectations count accepted draft tokens only and exclude the
   always-emitted correction/bonus token `y`. The caller holds those
   contributions with the proposal result.
8. Only after P06 successfully copies committed sampler/RNG state and atomically
   marks that single-use proposal consumed may the owning acceptor add
   `E_block/E_token`. Fallback, failed preflight, clone discard, rollback,
   replay, or a repeated consumed proposal adds nothing. Mode 1 adds nothing.
   After exactly 256 committed eligible proposals, emit one identity-bearing
   record containing `expected_accepted_draft_tokens_block`,
   `expected_accepted_draft_tokens_tokenwise`,
   `expected_gain_draft_tokens`, and `drafted_tokens_total`; optionally also
   emit gain divided by `drafted_tokens_total`. Within one identity, reset only
   after successful emission. These fields exclude the always-emitted
   correction/bonus token `y`.
9. Inherit every P06 fallback for the whole block and keep unset/default mode 0.
   A 256-proposal record is descriptive and cannot authorize a default change.
   Any future default proposal needs P06 parity and repeated fixed-corpus runs
   with independent predeclared seeds and a confidence bound over run-level gain,
   as well as A770 throughput evidence. Conditional expectations remove the
   acceptance-draw noise for a given proposal, not variation in sampled proposals.

## Critical files & anchors

- `common/sampling.cpp:678-714` - current exact-match acceptor and P06 insertion point.
- `common/sampling.h:85-89` - acceptor declarations; add the pure block helper.
- `common/sampling.cpp:509-539` - mode and P06 RNG clone/copy; diagnostics stay caller-owned.
- `tools/server/server-context.cpp:386-400,4302-4308` - per-slot diagnostic ownership and acceptance call.
- `tools/server/server-context.cpp:1990-1992` - request-time sampler replacement must rotate diagnostic identity.
- `tests/test-sampling.cpp` - seeded helper and end-to-end distribution corpus.
- `scripts/perf/verify-spec-block.py` - planned mode comparison and diagnostic collector.
- `docs/research/kmic68-a770-block-verify.md` - planned default-eligibility evidence.
- Sun et al., arXiv:2403.10444, Algorithm 2 and Theorems 1-2 - algorithm contract.

## Verification

Extend the P06 corpus with block lengths 1, 2, 4, and 8; tiny positive emitted
mass; zero-mass whole-block fallback; near-zero residual; full acceptance; and
rejection at every position. Reuse P06's exact pooled Pearson procedure: reject
zero-probability outputs, pool bins until expected count is at least 5, run
per-case goodness-of-fit and enabled/mode homogeneity tests, and apply
Holm-Bonferroni at family-wise `alpha=0.01`. TV/max error remain diagnostics.

Add a diagnostic-lifetime fixture that explicitly retains the same acceptor and
sequence: accumulate 128 eligible blocks, call `common_sampler_reset`, then
accumulate 128 more and emit exactly one `blocks=256` record. Separately, two
ordinary server requests with 128 blocks each on the same slot must emit none:
sampler replacement rotates identity and discards the old partial bucket. Cover
both changed and identical seeds/configurations, distinct slots, and clones;
none may combine counters.

Add an exact diagnostic test enumerating all pass/fail combinations for short
blocks and weighting the greatest passing position by its independent-draw
probability. For `G=2`, `P[0]=P[1]=(0.25,0.75)`,
`q[0]=q[1]=(0.5,0.5)`, and `draft=(0,1)`, the helper gets
`keep=(1,0.5,0.75)`, `h=(0,0.75)`, and `E_block=1.5`, not
`sum(keep[1:])=1.25`. Include all-zero/all-one draws and `G=1`; the
diagnostic calculation must consume no RNG draws.

```bash
timeout 240 ctest --test-dir build-sycl -R '^test-sampling$' --output-on-failure
```

On A770, run an explicit MTP configuration with a sufficiently long request for
one acceptor to complete 256 eligible blocks. The collector must fail if only
separate request identities reach that total:

```bash
timeout 2400 python3 scripts/perf/verify-spec-block.py --server ./build-sycl/bin/llama-server --model target-qwen4exp.gguf --draft-model qwen4exp-mtp.gguf --prompts scripts/perf/prompts.jsonl --seed 123 --temperature 0.8 --modes 0,1,2 --min-blocks 256 --spec-args='--spec-type draft-mtp --no-spec-draft-backend-sampling' --report docs/research/kmic68-a770-block-verify.md
```

Expected evidence:

- modes 1 and 2 pass the same pooled Pearson/Holm distribution procedure as P06
  against mode 0 and target `p`;
- zero or invalid emitted-token `q` falls back before the helper, and every
  target/draft distribution index follows the declared zero-based mapping;
- mode 0 and unset are the P06 control path; invalid mode values fail startup;
- constrained, stateful, missing-`q`, backend-selected, replayed, and chained
  cases fall back for the whole block;
- one acceptor spanning a reset emits exactly one identity-bearing
  `blocks=256` record; sampler replacement starts a fresh bucket even on the
  same slot with the same seed, and separate acceptors never combine counters;
- mode 1 and mode 2 make identical decisions, but only mode 2 emits diagnostics;
- `expected_gain_draft_tokens` is reported with its sampled proposal identity
  regardless of sign; its P07/P06 operands exclude correction/bonus tokens and
  `drafted_tokens_total` is the matching denominator. Exact enumeration over a
  tiny proposal distribution checks the unconditional expectation separately;
  no single 256-proposal sign is a default-eligibility gate;
- actual accepted/drafted counts and the post-run two-driver fault gate are
  recorded.

Run the exact README regression floor before and after the feature.

## Assumptions & contingencies

- Unset means mode 0, intentionally differing from the source default-on
  behavior; P07 cannot change that default.
- Mode 2 changes diagnostics only and consumes no additional RNG draws.
- Diagnostic state survives reset of the same acceptor and sequence, not sampler
  replacement; it never mixes slots, sequences, or sampler generations.
- Expected gain is exact conditional on the recorded proposals, whose selection
  remains stochastic. These diagnostic records are neither an unconditional
  expectation nor a confidence bound; P07 makes no default or throughput claim.
- P07 never reconstructs missing P06 distributions or handles state dependent
  on a previously accepted token.
