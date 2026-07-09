# Arc A770 benchmark results - incremental log

Date: 2026-07-09
Branch: `benchmark/fork-vs-upstream-a770`
Status: partial 16/60 at 2026-07-09T10:48:59Z
Matrix job: `bg_3` still running when this revision written
Matrix artifact: `docs/research/a770-fork-unique-2026-07-09/results.jsonl`
Important: current `--quick` matrix uses `p64/n16/r1`. Smoke only. Directional only. Not final throughput claim.

## Build parity used for comparison

Fork and upstream built locally with comparable oneAPI/SYCL settings:
- `Release`
- `icx` / `icpx`
- `GGML_SYCL=ON`
- `GGML_SYCL_F16=ON`
- `GGML_NATIVE=ON`
- fork tests enabled for harness work
- upstream tools built for control `llama-bench` runs

Build roots:
- fork: `build-port/`
- upstream: `compare/llama.cpp/build-sycl-a770/`

## Completed reachability benchmarks

### 1. Default SYCL turbo correctness harness

Command:
```bash
timeout 240 env ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build-port/bin/test-sycl-turbo-correctness
```

Result:
- standard SYCL FA gates pass
- turbo WHT pass
- turbo copy/dequant pass
- turbo quantize-store pass
- turbo `mul_mat` pass
- turbo non-FA attention pass for turbo3/turbo4; turbo2 warns, expected lossy behavior
- turbo FA section skipped by design because `LLAMA_TEST_TURBO_FA` absent
- d=256 generic FA section skipped by design because `LLAMA_TEST_FA256` absent
- InnerQ section skipped by design because `LLAMA_TEST_INNERQ` absent

Analysis:
- base fork kernel chain alive
- default run proves safe paths, not full turbo FA reachability

### 2. Turbo FA reachability harness

Command:
```bash
timeout 240 env ONEAPI_DEVICE_SELECTOR=level_zero:0 LLAMA_TEST_TURBO_FA=1 ./build-port/bin/test-sycl-turbo-correctness
```

Result:
- turbo FA harness probe labels `tile` / `vec` describe prefill-vs-decode shape, not selected kernel family
- turbo2 FA remains xfail across both probe shapes and GQA variants
- turbo3 FA passes d=128 prefill/decode-shape probes and GQA 4:1 / 8:1
- turbo4 FA passes d=128 prefill/decode-shape probes and GQA 4:1 / 8:1
- turbo3 FA passes d=256 prefill/decode-shape probes
- summary: `0 GATE-FAIL, 0 XPASS, 6 xfail`

Analysis:
- turbo FA work reachable on real A770
- default turbo FA router is VEC; combined XMX run covers turbo-XMX opt-in separately
- turbo3/turbo4 no longer theoretical path; harness reaches and validates them
- turbo2 still below quality floor, not production-clean

### 3. XMX FA reachability harness

Command:
```bash
timeout 240 env ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_FA_XMX=1 ./build-port/bin/test-sycl-turbo-correctness
```

Result:
- f16 XMX null-mask probes pass
- standard FA gates still pass under XMX env
- turbo FA still skipped because `LLAMA_TEST_TURBO_FA` absent

Analysis:
- XMX router live
- XMX implementation not dead code
- this run alone does not prove turbo-XMX path

### 4. Combined turbo + XMX reachability harness

Command:
```bash
timeout 240 env ONEAPI_DEVICE_SELECTOR=level_zero:0 LLAMA_TEST_TURBO_FA=1 GGML_SYCL_FA_XMX=1 ./build-port/bin/test-sycl-turbo-correctness
```

Result:
- same turbo FA pass/xfail shape as non-XMX turbo run
- same f16 XMX null-mask passes as XMX-only run
- no support-check skip on combined run
- summary: `0 GATE-FAIL, 0 XPASS, 6 xfail`

Analysis:
- combined env path reachable
- router does not wedge on turbo + XMX enablement
- covered turbo-XMX conditions stable enough for harness probes

## Matrix benchmark - partial results so far

Current file length: 16 rows out of planned 60.
Current completed rows all from model `llama31-8b-heretic`.

### Partial table

| model | case | prompt tok/s | gen tok/s | note |
|---|---|---:|---:|---|
| llama31-8b-heretic | upstream-f16-f16 | 327.67 | 24.83 | upstream control |
| llama31-8b-heretic | fork-f16-f16 | 326.65 | 24.85 | fork baseline |
| llama31-8b-heretic | upstream-q8_0-q8_0 | 318.51 | 23.45 | upstream control |
| llama31-8b-heretic | fork-q8_0-q8_0 | 324.31 | 25.29 | fork baseline |
| llama31-8b-heretic | fork-xmx-default-f16-f16 | 157.97 | 6.72 | XMX on |
| llama31-8b-heretic | fork-xmx-default-q8_0-q8_0 | 222.95 | 10.90 | XMX on |
| llama31-8b-heretic | fork-xmx-default-turbo3-turbo3 | 156.49 | 6.64 | XMX on, default policies |
| llama31-8b-heretic | fork-default-turbo2-turbo2 | 308.63 | 23.75 | default policies may mutate layout |
| llama31-8b-heretic | fork-pure-turbo2-turbo2 | 313.06 | 24.33 | auto policies forced off |
| llama31-8b-heretic | fork-xmx-pure-turbo2-turbo2 | 157.20 | 6.66 | XMX plus pure turbo |
| llama31-8b-heretic | fork-default-turbo3-turbo3 | 312.00 | 23.92 | default policies may mutate layout |
| llama31-8b-heretic | fork-pure-turbo3-turbo3 | 311.75 | 23.98 | pure turbo |
| llama31-8b-heretic | fork-xmx-pure-turbo3-turbo3 | 156.70 | 6.64 | turbo-XMX |
| llama31-8b-heretic | fork-default-turbo4-turbo4 | 315.55 | 24.35 | default policies may mutate layout |
| llama31-8b-heretic | fork-pure-turbo4-turbo4 | 315.10 | 24.25 | pure turbo |
| llama31-8b-heretic | fork-xmx-pure-turbo4-turbo4 | 151.68 | 6.66 | turbo-XMX |

## Partial analysis

### A. Fork baseline vs upstream baseline

f16/f16:
- prompt: fork 326.65 vs upstream 327.67, delta -0.31%
- gen: fork 24.85 vs upstream 24.83, delta +0.10%

q8_0/q8_0:
- prompt: fork 324.31 vs upstream 318.51, delta +1.82%
- gen: fork 25.29 vs upstream 23.45, delta +7.83%

Analysis:
- fork baseline not slower on this short smoke
- for f16, difference tiny
- for q8_0, fork ahead on this quick run
- because `r=1`, short prompt, short decode: treat as smoke-direction, not stable perf verdict

### B. XMX path

Compared to fork baseline:
- f16/f16 XMX prompt 157.97 vs 326.65, about 48.4% of baseline
- f16/f16 XMX gen 6.72 vs 24.85, about 27.0% of baseline
- q8_0/q8_0 XMX prompt 222.95 vs 324.31, about 68.7% of baseline
- q8_0/q8_0 XMX gen 10.90 vs 25.29, about 43.1% of baseline

Analysis:
- XMX path reachable
- XMX path currently big perf regression on this short workload
- matches fork source comments that XMX is bring-up path, not tuned fast path yet
- reachability good; speed bad so far

### C. Turbo non-XMX smoke

Pure turbo, same model, `p64/n16/r1`:
- turbo2/turbo2: 313.06 / 24.33
- turbo3/turbo3: 311.75 / 23.98
- turbo4/turbo4: 315.10 / 24.25

Relative to fork q8_0/q8_0 baseline 324.31 / 25.29:
- turbo2 prompt -3.47%, gen -3.80%
- turbo3 prompt -3.87%, gen -5.18%
- turbo4 prompt -2.84%, gen -4.11%

Analysis:
- on short smoke, pure turbo not collapsing
- turbo4 closest to q8_0 so far
- turbo3 slightly behind turbo2/turbo4 in this short bench
- these are synthetic bench numbers only; must pair with correctness and later coherence/perplexity

### D. Turbo XMX smoke

Pure turbo XMX rows:
- turbo2/turbo2: 157.20 / 6.66
- turbo3/turbo3: 156.70 / 6.64
- turbo4/turbo4: 151.68 / 6.66

Analysis:
- turbo-XMX path definitely dispatches
- all turbo-XMX rows much slower than non-XMX turbo smoke
- current XMX value on A770 looks diagnostic only, not performance feature

### E. Auto-policy impact, first partial sample

Default vs pure on llama31-8b:
- turbo2 default 308.63 / 23.75 vs pure 313.06 / 24.33
- turbo3 default 312.00 / 23.92 vs pure 311.75 / 23.98
- turbo4 default 315.55 / 24.35 vs pure 315.10 / 24.25

Analysis:
- policy effect small on this short smoke for llama31-8b
- still must keep cases separate because runtime may mutate K/V type selection on other models, especially high-GQA cases
- labeling default vs pure remains mandatory

## What still missing in this results log

Not done yet:
- remaining 44 matrix rows
- mistral-7b matrix rows
- qwen3-coder-30b-a3b matrix rows
- mixed `q8_0/turbo3` rows in partial analysis
- non-FA `llama-bench` row analysis
- coherence probes with `llama-cli` / `llama-completion`
- final repeated/full-context benchmark verdicts

## Interim verdict

Reachability verdict now strong:
- turbo kernels reachable
- turbo FA reachable
- XMX reachable
- turbo-XMX reachable

Performance verdict still partial:
- fork baseline roughly at parity or slightly ahead of upstream on early llama31 smoke
- XMX path currently much slower than baseline
- pure turbo non-XMX only few percent behind q8_0 on short synthetic smoke
- final conclusion waits on remaining matrix plus coherence checks

## Update 2026-07-09T10:53:28Z

Matrix status now partial 53/60. `bg_3` still running when this update appended.

Completed-row count by model:
- llama31-8b-heretic: 20/20
- mistral-7b: 20/20
- qwen3-coder-30b-a3b: 13/20

### New directional snapshot across completed models

Baseline upstream vs fork:
- llama31 f16: near parity, fork -0.31% prompt, +0.10% gen
- llama31 q8_0: fork +1.82% prompt, +7.83% gen
- mistral f16: near parity, fork -0.23% prompt, flat gen
- mistral q8_0: fork +1.79% prompt, +7.72% gen
- qwen3 f16: fork -5.50% prompt, flat gen
- qwen3 q8_0: fork -4.60% prompt, +6.69% gen

XMX default so far:
- llama31 f16: 157.97 / 6.72
- mistral f16: 158.48 / 6.79
- qwen3 f16: 43.92 / 4.32
- llama31 q8_0: 222.95 / 10.90
- mistral q8_0: 224.15 / 11.08
- qwen3 q8_0: 50.35 / 6.96

Read: XMX still reachable, still much slower than baseline on all completed controls.

Pure turbo non-XMX so far:
- llama31 turbo2: 313.06 / 24.33
- llama31 turbo3: 311.75 / 23.98
- llama31 turbo4: 315.10 / 24.25
- mistral turbo2: 315.28 / 25.18
- mistral turbo3: 314.45 / 24.83
- mistral turbo4: 317.01 / 25.14
- qwen3 turbo2: 55.94 / 14.35
- qwen3 turbo3: 55.97 / 14.11
- qwen3 turbo4: 54.91 / 14.34

Read:
- dense 7B/8B models: pure turbo still only few percent behind q8_0 on short smoke
- qwen3 smoke: fork turbo rows near fork f16/q8_0 prompt range; decode spread small
- still smoke only. No stable throughput claim yet. No correctness claim from `llama-bench` alone.

Non-FA smoke completed on dense models:
- llama31 `fork-nonfa-turbo3-turbo3`: 222.96 / 9.16
- mistral `fork-nonfa-turbo3-turbo3`: 224.60 / 9.27

Read:
- non-FA much slower than FA smoke on dense models
- matches expectation that graph-side non-FA path exists and is benchmarkable, but not preferred fast path

Still missing after this update:
- qwen3 remaining 7 rows
- mixed `q8_0/turbo3` row analysis
- coherence probes
- non-quick reruns

## Corroborating prior results (from RALPH/ASSUMPTIONS docs, not this session's runs)

Source docs: `RALPH_PROGRESS.md` and `docs/ppl-results/*` (in this repo); plus `RALPH_TASKS.md`, `ASSUMPTIONS.md`, and `docs/research/turbo-capacity-validation.md` (external, parent working tree, not in this repo).
These add the quality and capacity axes that this session's short `llama-bench` smoke does not measure. Attributed to prior work, re-cited here so all results live together.

### PPL quality (CPU-FA, wikitext-2, 564-chunk unless noted)

HARD RULE from prior work: turbo KV is FA-only. Prior PPL used CPU-FA (`-ngl 0 --flash-attn auto`) because non-FA `-fa off` transposes block-quant V and corrupts it. So prior PPL is CPU-FA, this session's `llama-bench` is GPU-FA throughput; different axes, do not merge cells.

| KV | mistral-7b | llama31-8b | qwen3-30b MoE |
|---|---|---|---|
| f16 | 7.6328 | 7.5433 | 9.7022 |
| q8_0 | 7.6332 (+0.00%) | 7.5456 (+0.03%) | 9.7030 (+0.01%) |
| q4_0 | 7.6913 (+0.77%) | 7.7722 (+3.03%) | 9.8740 (+1.77%) |
| turbo2 | 8.1216 (+6.40%) | 10.6345 (+41%) | KILLED chunk5 exp divergence |
| turbo3 | 7.7298 (+1.27%) | 8.0200 (+6.33%) | KILLED chunk8 NaN |
| turbo4 | 7.6534 (+0.27%) | 7.6625 (+1.58%) | 8.9105 (50ch, auto-asym K=q8_0) |

Gate `turbo4 < q4_0`: mistral PASS (-0.49%), llama31 PASS (-1.41%), qwen3 directional win.
Finding: turbo2/turbo3 not viable on MoE (per-expert MUL_MAT_ID roundoff past 2/3-bit V budget). Use turbo4 only on MoE.

### Capacity (single-stream max ctx, 16 GB A770)

- turbo4/f16 = 3.79x on both dense 7-8B models (model-invariant).
- q8_0/f16 = 1.89-1.90x, q4_0/f16 = 3.57-3.60x (bytes-per-token driven, model-invariant).
- turbo2 = 6.38x (max), turbo3 = 5.16x (llama31 numbers).
- qwen3 30B: model is 12.7 GB of 16 GB VRAM, so turbo capacity there is a host-RAM story (CPU-FA), not VRAM.

### Depth caveat on this session's short-smoke throughput parity

This session's `p64/n16` smoke shows pure turbo3/turbo4 within a few percent of q8_0. Prior perf-findings doc (`docs/research/2026-07-05-turbo3-sycl-perf-findings-and-roadmap.md`) measured turbo LOSING to f16/q8_0 at depth (turbo3 -6% at d=0, -17% at 16k) because the per-element centroid dequant tax dominates as context grows. So short-smoke parity is expected; it does NOT contradict "turbo is a capacity feature, not a speed feature." A long-context rerun would surface the crossover; not run this session.

### InnerQ runtime status (P3.2.2) - reachable in code, NOT proven live end-to-end

- Producer path fires: prior smoke logged `InnerQ publish_scale_inv issued from Vcur-0` (up to 17x/run).
- Consumer path never fired: `InnerQ scale_inv tensor updated ... finalized=1` = 0 occurrences.
- On Qwen3 GQA 8:1 InnerQ is inapplicable by design: auto-asymmetric K downgrade (`src/llama-kv-cache.cpp:152`) turns `type_k` turbo3 into q8_0, so the turbo-rotation-tensor alloc guard is false for every layer.
- Prior runtime proof was blocked by a SYCL AOT/IGC offload-link failure on `bin/libggml-sycl.so.0.15.1` in the separate AOT build tree `/home/svnbjrn/build-turbo-aot` (ocloc acm-g10).
- Contrast worth recording: this session's `build-port` is a generic JIT SYCL build, and it built and ran clean. The AOT blocker that stalled the RALPH InnerQ loop is AOT-specific, not reproduced on the JIT build used for these benchmarks.

### Speed-work closures (do not re-open without new evidence)

- Tier-1 SLM Q-centroid LUT: measured -8% at depth, reverted.
- Tier-3 XMX joint_matrix: SG=16 caused 3x IGC ICE; fixed at SG=8. P3.1 verified the existing SG=8 build runs on current toolchain. This session's XMX bench confirms XMX dispatches but is much slower than VEC on short smoke, matching "bring-up path, not tuned fast path."

### Correctness harness prior status

All 3 fleet configs (mistral/llama GQA 4:1, qwen3 GQA 8:1) previously ran `0 GATE-FAIL` in default and `LLAMA_TEST_TURBO_FA=1` env. This session reproduced that and additionally promoted turbo3/turbo4 FA probes from XFAIL to GATE (they now pass; turbo2 stays XFAIL at 2-bit).

## Final verdict (matrix complete + coherence complete + prior results folded)

Matrix: 60/60 rows ok. Coherence smoke: 15/15 coherent, all produce "Paris".

Reachability: every fork-only SYCL surface (turbo WHT, decode, quantize-store, mul_mat, non-FA turbo attention, turbo FA VEC, XMX f16, turbo-XMX) is reachable and benchmarkable on the A770 through `test-sycl-turbo-correctness` + `llama-bench` + fork env switches.

Performance (this session, short GPU-FA smoke):
- fork baseline == upstream on f16; fork slightly ahead on q8_0 (likely noise/short-run).
- pure turbo2/3/4 within ~2-5% of q8_0 on short smoke for dense models; near fork f16/q8_0 range on qwen3.
- XMX path much slower than VEC baseline everywhere; reachable, not yet a speed feature.
- non-FA turbo3 much slower than FA; benchmarkable but correctness-caveated (see ledger).

Quality + capacity (prior RALPH work): turbo4 is the production turbo pick - PPL within +1.58% of f16 on dense, beats q4_0, 3.79x capacity vs f16. turbo2/turbo3 unsafe on MoE.

Not closed this session: long-context throughput crossover, fresh full-corpus PPL, live InnerQ consumer proof, AOT build path.
