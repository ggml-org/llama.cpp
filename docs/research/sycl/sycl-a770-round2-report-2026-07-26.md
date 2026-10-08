# SYCL A770 decode round 2: campaign report

Date: 2026-07-26. Detailed candidate list and evidence:
[`sycl-a770-round2-decode-candidates-2026-07-25.md`](sycl-a770-round2-decode-candidates-2026-07-25.md).

This report states what changed, what was learned, and what must be measured before any of it is
trusted. Nothing in this round executed on the A770.

## 1. What shipped

| Change | File | Risk |
|---|---|---|
| quants-first q8_0 KV row layout promoted from opt-in to default, with `GGML_SYCL_Q8_KV_QUANTS_FIRST=0` opt-out | `src/llama-kv-cache.cpp` | **Unverified at depths 0 and 2048** - see section 4 |
| Load alignment corrected on the two quants-first load sites, plus `static_assert(ne % 4 == 0)` | `ggml/src/ggml-sycl/fattn-common.hpp` | None. Hygiene only, no performance claim |
| `GGML_SYCL_Q8_KV_QUANTS_FIRST` documented (was undocumented) | `docs/backend/SYCL.md` | None |
| Compile-only Xe-HPG ISA probe | `scripts/perf/probe-q8-load-width.sh` | None. Does not execute on the GPU |

The default flip is a cutover of an already-measured mechanism, not a new kernel. P5.11 measured
it at +8.25/+8.01% at depth 4096, +13.81/+13.37% at 8192 and +20.62/+20.43% at 16384 (Mistral-7B
and Llama-3.1-8B, Xe-HPG A770, paired tg128 medians), with worst pp512 -0.39%, cosine at least
0.999957 and `0 GATE-FAIL`. The per-layer predicate is unchanged and still restricts the layout to
a SYCL device with q8_0 K and V and 128-element heads; every other cache keeps canonical blocks.

Session-file compatibility is already handled: state write repacks to canonical
(`src/llama-kv-cache.cpp`, `llama_kv_cache_q8_repack_groups(..., false)`) and state read repacks
back, so bytes on disk stay canonical `block_q8_0` in both directions.

## 2. The finding that reorders the programme

Held-binary paired benchmarks fit

```text
per-token time ~= 40.41 ms + 3.67 us * depth
```

so flash attention is **0.58% of a token at depth 0**, 27.4% at 4096 and 60.0% at 16384. A
conjunctive ">=5% at every depth" gate cannot be cleared by any attention change, because making
attention free gains 0.58% at depth 0. Every FA candidate in this project's history shows the same
deep-win/shallow-lose shape; that shape is Amdahl's law, not a tuning failure.

The gate adopted for this round is therefore disjunctive: **>=5% at the promotion depths 8192 and
16384, with the protected depths 0, 2048 and 4096 held to a -2% no-regression guard.** All five
depths are benched, since a guard cannot be evaluated on an unmeasured cell. The authoritative
definition lives in section 1 of the candidate artifact; this report does not restate it
differently.

The corollary is larger than the gate change. About 40.4 ms of every token is spent outside
attention, at 106 GB/s effective against a 560 GB/s part - **19% of the bandwidth roofline**. A
bandwidth decomposition accounts for only 9.64 ms of it. Of the remaining 30.77 ms, roughly
12.5 to 13.0 ms is submission overhead (measured-derived, about 14.9 us across an estimated 837
kernel enqueues per token) and **about 18.2 ms is unaccounted for**.

## 3. What was refuted, including two of our own claims

- **Vulkan is not faster than SYCL on the A770** (52.56 versus 55.53-59.03 t/s, llama-2-7b Q4_0).
  There is no backend tax to recover.
- **XMX is shelved for decode on two independent grounds.** DPAS is 4x the vector path on Xe-HPG
  while decode attention leaves that ALU about 15x idle, and the saturation crossover is 61-65
  query rows against the 4-8 that GQA supplies. Separately, sub-group-16 DPAS is a
  Xe-HPC-and-later hardware capability, so the reported IGC internal compiler error can never be
  fixed by a toolchain bump. No reason remains to move off the IGC 2.36.3 pin.
- **A shape-conditioned submission policy cannot be built on `UR_L0_BATCH_SIZE`** - it is read once
  at Level Zero adapter load. But submission mode is a per-queue property reachable from SYCL, so
  the objective survives by a different route.
- **The L2-capacity model proposed earlier in this round for the GQA depth crossover is refuted**
  (it was ours). The measured curve decelerates monotonically with no kink, the crossover is about
  2400 rather than 7710, and the kernel runs at 8% of peak bandwidth so a DRAM-capacity model
  answers the wrong question. Replaced by an occupancy hypothesis, which is decidable by a
  host-side probe before any kernel work.
- **A GRF figure used earlier in this round was wrong** (also ours): 128 x 64 B = 8 KB was Xe3
  data. Xe-HPG GRFs are 32 B, so the budget is 4 KB per thread. Any register-pressure estimate
  made against 8 KB is 2x optimistic.

## 4. What must be measured before this is trusted

**The default flip is not yet gated.** The protected depths are 0, 2048 and 4096. P5.11 measured
only 4096 (at +8.25/+8.01%, comfortably inside its guard) along with the two promotion depths, so
**depths 0 and 2048 are the missing cells**, and producing them is the merge gate for this branch.

```sh
scripts/bench-a770-fork-unique.py    # paired A/B, opt-out vs default
# depths 0, 2048, 4096, 8192, 16384; 6 launches per arm, repetition 0 discarded
# preconditions: ONEAPI_DEVICE_SELECTOR=level_zero:0, sole tenancy of /dev/dri/renderD128
```

Pre-registered kill criterion, stated before any run: rejected if any protected depth - 0, 2048 or
4096 - regresses more than -2% on any fleet model, or if the promotion depths 8192 and 16384 fail
to reproduce at >=5% on the newly fetched stock Q4_K_M files. Correctness gates:
`ctest -R test-sycl-turbo-correctness` at `0 GATE-FAIL`, the forced-TILE 4:1 and 8:1 quants-first
cells, and `scripts/perf/server_state_roundtrip.py` at all 8 assertions. With the opt-out set,
output must match the pre-change build.

The validation fleet was completed this round: stock `Meta-Llama-3.1-8B-Instruct` Q4_K_M and
`Qwen3-Coder-30B-A3B-Instruct` Q4_K_M were fetched and verified (`general.file_type = 15`; GQA
r=4 and r=8, head dim 128). **No P5 number measured on the previous Q3_K_XL Qwen or abliterated
Llama files is a valid baseline for these**, so baselines must be re-measured.

## 5. Next actions, in order

1. **Reproduce upstream's A770 number on this box.** Upstream SYCL reaches 226 GB/s effective on
   llama-2-7b Q4_0; normalised for size that predicts about 52.6 tok/s where this fork measures
   24.6 - roughly 2.1x. That is a cross-source comparison, not a paired run, and it is the single
   most decision-relevant number outstanding. If upstream also lands near 25 t/s on this driver
   stack, the regression belongs to compute-runtime 26.22 and the answer is a driver bisect.
2. **Run the depth 0 and 2048 cells** that gate this branch (section 4).
3. **Dual-queue shape-routed submission.** Largest measured mechanism, no determinism impact, no
   launch-path refactor.
4. **Un-gate SYCL graphs for MoE.** `MUL_MAT_ID` disables graphs model-wide on the strength of a
   host wait the fused token-generation path returns before reaching, so Qwen3-Coder can never use
   graphs despite having the most enqueues to save.
5. **The GQA occupancy probe**, before writing any depth gate - it decides whether a gate is the
   fix or a workaround.

Two correctness items were also raised and are written up with probes rather than acted on: the
`gqa_ratio >= 6` auto-asymmetric turbo-K predicate may be misspecified (published data has a GQA
2:1 model failing while a 4:1 model is fine, suggesting model family rather than ratio is the
discriminator - though the supporting evidence is 2-bit only and the observed failure here was at
turbo3), and the q8_0 nondeterminism under speculative verification is root-caused to the
VEC-to-TILE route flip at batch above 2, which f16 does not take.
