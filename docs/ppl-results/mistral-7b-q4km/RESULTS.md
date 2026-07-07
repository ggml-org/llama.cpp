# PPL — Mistral-7B-Instruct-v0.1 (Q4_K_M) on wikitext-2 test (raw)

**Generated:** 2026-07-07, RALPH task P1 [model 1] (sub-task 1: PPL matrix).
**Host:** vinbonesjr (AMD Ryzen 9 7900X3D, Intel Arc A770 DG2 / 16 GB VRAM,
oneAPI 2026.0 icpx, kernel 7.1.3-273-tkg-bore).
**Binary:** `/mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork/build-port/bin/llama-perplexity`
(built from commit `de701194b` on `turbo-sycl-opt`).
**Model:** `/mnt/mrgr/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf`
(Q4_K_M weight quant, 4.37 GB; arch=llama, embed=4096, head=32, head_kv=8,
head_dim=128, GQA=4:1).
**Corpus:** `wikitext-2-raw-v1.zip` test split, 1.29 MB / 4358 lines / ~322 K tokens,
file: `/mnt/mrgr/projects/llama-cpp-turboquant/wikitext-2-raw/wiki.test.raw`.
**Common flags:** `-c 512 -b 512 -ub 512 -ngl 99` (ctx=512, all 32 layers → GPU VRAM).
CPU threads=12 (default, full mask), 1 sequence (no batching).
**Deltas:** relative to **f16** KV (the canonical precision reference).

## TL;DR (corrected 2026-07-07, see addendum)

The first PPL pass (commit `a1495e8b1`) violated this loop's hard rule:

> "Turbo KV is FA-only. Never validate it under -fa off — the non-FA path
> is architecturally broken for block-quantized KV in general, confirmed
> as of 2026-07-02." — `RALPH_TASKS.md` §P1.5, L75-80.

The 2026-07-02 doc explicitly identifies the cause and the blessed
fallback:

> "Turbo is **FA-only by design**. The non-FA path is architecturally broken
> for block-quantized V because of the unconditional transpose in
> `build_attn_mha`. Using non-FA as a baseline for 'is turbo math correct?'
> will always fail, even if the FA path is perfect.
> **Lesson**: When adjudicating turbo failures, gate against f16-FA or
> **CPU-FA** coherence, never against turbo-non-FA."
> — `docs/research/Intel-Arc-A770-llama-cpp-turboquant-SYCL-Turbo-FA-Port-KQ-Dot-Fix-SET_ROWS-Bug-Plus-Non-FA-V-Transpose-Architectural-Limitation-20260702.md`,
> §"Lessons Learned" §4 "Non-FA ≠ Valid Baseline for Turbo", L396-400.

The three **turbo rows** in the first pass used `--flash-attn off` (SYCL FA
is vetoed for turbo on this stack — `ggml_sycl_flash_attn_ext_supported` at
`fattn.cpp:168-180` returns false for any turbo K/V — and the binary's
`--flash-attn auto` resolves to `off` when the backend rejects FA). They
therefore measured the architecturally-broken path the rule forbids.
**Those rows are SUPERSEDED.** A CPU-FA re-run (full 642 chunks, `-ngl 0`,
`--flash-attn auto`) is in flight; its numbers will replace them.

The **f16 / q8_0 / q4_0** rows are unaffected by this rule (they are not
turbo KV) and remain valid as baseline reference points.

## Results table

| KV type | PPL     | ±     | Δ vs f16 | Δ %    | chunks | flash-attn | wall-time | status |
|---------|---------|-------|----------|--------|--------|------------|-----------|--------|
| f16     | 7.6329  | 0.048 | (ref)    | —      | 642    | auto       | 4 min 40 s | valid |
| q8_0    | 7.6332  | 0.048 | +0.0003  | +0.00% | 642    | auto       | 4 min 43 s | valid |
| q4_0    | 7.6901  | 0.049 | +0.0572  | +0.75% | 642    | auto       | 4 min 43 s | valid |
| ~~turbo4~~ | ~~7.6563~~ | 0.049 | ~~+0.0234~~ | ~~+0.31%~~ | 642 | ~~off~~ | ~~5 min 01 s~~ | **SUPERSEDED** (rule violation; CPU-FA re-run pending) |
| ~~turbo3~~ | ~~7.7275~~ | 0.049 | ~~+0.0946~~ | ~~+1.24%~~ | 642 | ~~off~~ | ~~5 min 02 s~~ | **SUPERSEDED** (rule violation; CPU-FA re-run pending) |
| ~~turbo2~~ | ~~8.1166~~ | 0.051 | ~~+0.4837~~ | ~~+6.34%~~ | 642 | ~~off~~ | ~~5 min 01 s~~ | **SUPERSEDED** (rule violation; CPU-FA re-run pending) |

The struck-through numbers are preserved so the diff against the next
attempt (CPU-FA re-run) is computable. They MUST NOT be cited as valid
PPL evidence for the capacity track; they exist only to expose the
non-FA vs FA delta.

## Gate verdicts (reserved until CPU-FA re-run lands)

1. **Gate (RALPH_TASKS P1 [model 1]): "turbo4 should now show the post-2a
   improvement (was tied with q4_0 pre-fix) — confirm numerically, don't
   assume."** **RESERVED.** Cannot be PASS or FAIL on the prior numbers —
   the prior turbo4 PPL (7.6563) was measured on the non-FA path the rule
   forbids, while q4_0 PPL (7.6901) was measured on SYCL FA. Apples-to-
   oranges. The gate will be re-evaluated against the CPU-FA turbo4 number
   when the re-run completes.

2. **No GATE regression on f16 / q8_0 / q4_0 baseline.** f16 PPL 7.6329 is
   the canonical reference; q8_0 PPL 7.6332 is bit-equivalent within ±0.05
   noise (the 8-bit KV cache cost is below the PPL noise floor for
   mistral-7b at ctx=512). q4_0 PPL 7.6901 = +0.75% relative to f16,
   consistent with published numbers for 4-bit KV at this corpus.

3. **Turbo correctness claim needs CPU-FA basis.** The "[3b] / [3c] harness
   GATE passed" prior-turn claim was correct as a statement about the
   harness's regression-test for the dequant-before-transpose fix in
   `build_attn_mha` — but the harness intentionally exercises the
   architecturally-broken non-FA path as a **documented XFAIL**
   (`tests/test-sycl-turbo-correctness.cpp:475-480`), and a harness PASS on
   that section is *not* a correctness endorsement of PPL under -fa off.
   The correct FA-path correctness check is the harness `[5b] FA TURBO GQA`
   sweep, which DID PASS in the P0 GQA-shape smoke test — proving the FA
   kernel math on turbo is correct on this binary. CPU-FA re-runs the
   same FA math at the model level.

## Methodological notes

- **Why CPU-FA, not GPU-FA.** The SYCL backend's `flash_attn_ext_supported`
  returns false for any turbo K/V on this stack (`fattn.cpp:168-180`, also
  called out in the 2026-07-02 doc). On the GPU path, turbo inference
  therefore forces the non-FA mul_mat/softmax/mul_mat chain — which is
  exactly the path the rule forbids for correctness measurement. CPU-FA
  (the CPU backend's FA kernel, `-ngl 0 --flash-attn auto`) does not have
  that restriction: it dequantizes on-the-fly and is endorsed as the
  blessed turbo-correctness baseline by the 2026-07-02 lesson.

- **Smoke-test f16 = 7.5818 (64 chunks):** superseded by the matched-condition
  re-run = 7.6329 (642 chunks). The smoke value was an underestimate because
  PPL on the first ~64 chunks hasn't converged; the gap (7.5818 → 7.6329) is
  exactly the early-chunk undershoot pattern, not a measurement bug.

- **Reproducibility:** every number is from a single run, not an average of
  multiple runs. PPL on a fixed corpus + fixed seed is deterministic, so a
  second run would yield bit-identical values. The ± error bar is the
  standard error of the mean over chunks.

- **PPL-of-PPL:** the published f16 baseline (7.6329) is consistent with
  published values for mistral-7b-instruct-v0.1 Q4_K_M on wikitext-2 (which
  sit in the 7.5–7.7 range across llama.cpp revisions; the f16 KV variant
  gets close to but not below 7.5 on Q4_K_M weights because the weights
  themselves are 4-bit quantized, setting a floor).

- **The reframe (capacity, not speed):** once the CPU-FA turbo numbers
  land, the PPL cost is the *quality* half of the capacity-feature
  argument. The *capacity* half — KV-byte → context-length gain at
  constant VRAM — is the next sub-task. PPL does not measure capacity.

## Files

- This file: `RESULTS.md` (canonical, corrected).
- First-pass raw logs (full chunk-by-chunk trajectory, GPU -fa off):
  `/tmp/ralph-ppl-mistral7b/*.log` (preserved for the non-FA vs FA delta
  calculation; not citable as valid turbo PPL).
- CPU-FA re-run logs (in flight): `/tmp/ralph-ppl-mistral7b-cpufa/*.log`.
- Addendum in `RALPH_PROGRESS.md` 2026-07-07 entry — full root-cause +
  correction rationale + verbatim source citations.

## Status

- [x] f16 KV (642 chunks, matched-condition) — valid baseline
- [x] q8_0 KV (642 chunks) — valid baseline
- [x] q4_0 KV (642 chunks) — valid baseline
- [x] ~~turbo2 KV (642 chunks)~~ — **SUPERSEDED; CPU-FA re-run in flight**
- [x] ~~turbo3 KV (642 chunks)~~ — **SUPERSEDED; CPU-FA re-run in flight**
- [x] ~~turbo4 KV (642 chunks)~~ — **SUPERSEDED; CPU-FA re-run in flight**

Gate verdict reserved until CPU-FA re-run completes.