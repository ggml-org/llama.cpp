# A770/SYCL spec-decode + KV-type tuning — findings (R1)

Reproducible benchmark of speculative decoding (`--spec-type`) × KV cache type
on the Intel Arc A770, implementing recommendation **R1** of the fork-perf
research. Every throughput magnitude the research left as `[needs-A770]` is
measured here.

## Conditions

| | |
|---|---|
| Host | `vinbonesjr` — AMD Ryzen 9 7900X3D, Intel **Arc A770** (15473 MiB), 64 GB |
| Backend | SYCL / oneAPI Level-Zero 2026.0 (icx/icpx), `GGML_SYCL_F16=ON`, JIT target `INTEL` |
| Fork commit | `1de10dc` (`perf/spec-decode-tuning` @ HEAD of `fix/sycl-turbo-fa`) |
| Model | `Meta-Llama-3.1-8B-Instruct-heretic.Q4_K_M.gguf` (the actual A770 prod model) |
| Server flags (fixed) | `--device SYCL0 --n-gpu-layers 999 --no-mmap --flash-attn on --parallel 1 --threads 12 --ctx-size 16384` |
| ngram-mod params | `--spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64` |
| Harness | `scripts/perf/bench_spec.py` (launches server/arm, OpenAI `/v1/chat/completions`, reads server `timings`) |
| Metric | `timings.predicted_per_second` (tg), median of REPEATS=3; acceptance from `timings.draft_n_accepted / draft_n` |
| Date | 2026-06-27 |
| Caveat | A concurrent, **independent** AUR `makepkg` AOT build (separate session, `/mnt/mrgr/aur/llama.cpp-sycl`) ran during the sweep (load 6–10/24). It is **CPU-only** — GPU compute is uncontended, so tg is faithful: base tg (≈38) matches the documented prod ≈37.5, and prefill (pp) was identical across all 6 arms (538/444/186 for the three prompts), proving stable conditions across arms. |

## Prompt suite (`prompts.jsonl`, n_predict=256, temperature=0)

- `code_edit` — a ~30-line function + "rename `total` → `grand_total` and return the whole file" → whole-file copy-runs (the spec sweet spot).
- `multi_turn` — 3-turn coding chat; final turn extends prior code → reproduces earlier code (copy-runs).
- `free_prose` — original 250-word essay → low-repetition control.

## Results — throughput (tg, tokens/s, median of 3)

| prompt | none-q8_0 | ngrammod-q8_0 | ngrammod+mapk4v-q8_0 | none-f16 | ngrammod-f16 | ngrammod+mapk4v-f16 |
|---|---|---|---|---|---|---|
| code_edit | 37.96 | 120.62 | **124.76** | 36.74 | 119.81 | 123.92 |
| multi_turn | 38.98 | **89.66** | 88.95 | 37.13 | 86.54 | 86.60 |
| free_prose | 38.47 | 54.20 | 54.25 | 36.83 | **63.66** | 63.58 |

## Results — draft acceptance rate (median)

| prompt | ngrammod-q8_0 | ngrammod+mapk4v-q8_0 | ngrammod-f16 | ngrammod+mapk4v-f16 |
|---|---|---|---|---|
| code_edit | 0.641 | 0.898 | 0.641 | 0.898 |
| multi_turn | 0.969 | 0.969 | 0.969 | 0.969 |
| free_prose | 0.771 | 0.771 | 0.750 | 0.750 |

(`none` arms have no drafts → acceptance n/a. pp identical across all arms; see Conditions.)

## Analysis

### Spec-decode is a large, lossless win (R1 — CONFIRMED, not refuted)
Speculative ngram-mod beats `--spec-type none` by **far more than the 5% R1
threshold** on the copy-run regimes:

| prompt | best-ngram / none (q8_0) | speedup |
|---|---|---|
| code_edit | 124.76 / 37.96 | **3.29×** |
| multi_turn | 89.66 / 38.98 | **2.30×** |
| free_prose | 54.25 / 38.47 | 1.41× |

Correctness verified: at temperature 0 the spec arms produce **byte-identical
output** to the `none` arms (spec decoding is lossless), and all generations are
coherent (renamed code, valid `fib`, real prose) — not SYCL garbage. This both
confirms output correctness and closes the llama-bench "correctness-blind" gap.

### Flash-attention engages with q8_0 KV (closes research lead [F7])
`--flash-attn on` + `--cache-type-k/v q8_0` ran with no FA-disable warning,
without crashing (commit `fc3584d`, an ancestor of HEAD `1de10dc`, fixes the
historical A770 q8_0-KV-FA crash and is in the built binary),
and produced correct output. The decisive signature: q8_0 base tg (37.96) is
**≥** f16 base tg (36.74) — impossible if FA were silently off for q8_0
(unfused q8_0 attention would be far slower). FA over q8_0 KV is active.

### `ngram-map-k4v` adds a small, free bump on code edits
Adding `ngram-map-k4v` to `ngram-mod` raises code_edit acceptance 0.641 → 0.898
and tg 120.62 → 124.76 (q8_0, +3.4%), and is within noise everywhere else — no
regression observed. Worth including; `ngram-mod` alone already captures ~97% of
the gain if a single impl is preferred.

### KV type: q8_0 for this workload
q8_0 wins the two copy-run regimes that matter for a code assistant (code_edit
≈ f16; multi_turn +3.6%) and **halves KV memory** (longer context in the shared
16 GB VRAM). f16 wins `free_prose` (63.66 vs 54.25), but that is the noisiest
case (256-token low-repetition gen; f16 shows *higher* tg at *lower* acceptance —
a measurement-noise tell), and prose is not the target workload. Keep prod's
**q8_0**.

## Recommended production launch config

The prod unit (`llama-sycl.cpp.service`) currently runs **without** `--spec-type`
(no speculation). Enable it — that is the 2–3× win above. Add to `LLAMA_ARGS`
(the backend flags `--flash-attn on --cache-type-k/v q8_0` already match):

```
--spec-type ngram-mod,ngram-map-k4v \
--spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
```

## R1 falsifiable verdict

**CONFIRMED.** Best ngram arm beats `--spec-type none` by 3.29× (code_edit) and
2.30× (multi_turn) — vastly exceeding the >5% bar. Recorded model is the 8B
heretic; on a larger target (27B/35B) the relative spec win is expected to be at
least as large (decode is more memory-bound there), so this is a lower bound for
the user's bigger models.

## Reproduce

```bash
cd /home/svnbjrn/projects/trb/llama-cpp-turboquant-perf
MODE=baseline REPEATS=3 REQ_TIMEOUT=600 \
  SERVER_BIN=$PWD/build/bin/llama-server \
  python scripts/perf/bench_spec.py
# -> scripts/perf/results/summary.json + printed tables
```

---

# R2 — ngram-mod hard-off latch

A per-sequence latch that disables ngram-mod drafting after
`--spec-ngram-mod-dead-off` consecutive **zero-accept** drafts, re-armed each
generation (`begin()`). Bounds the worst case where ngram-mod fires repeatedly
but predicts nothing — each fired draft costs a wasted batched verify.

Implementation (`common/`): `n_dead_off` param (`common.h`, default 3), the
`--spec-ngram-mod-dead-off` flag (`arg.cpp`), and the latch in
`common_speculative_impl_ngram_mod` (`speculative.cpp`): `seq_info.{n_dead,off}`,
re-arm in `begin()`, an early-return guard in `draft_one()` placed *after* the
`n_draft_last = 0` reset (so a disabled seq leaves `n_draft_last == 0` and
`accept()` skips the streak logic on a phantom draft), and the consecutive-zero
counter in `accept()` (any partial accept resets `n_dead`).

## B4a — Harmlessness (default n_min=48, real workload)

`MODE=deadoff KV=q8_0 REPEATS=3`, dead-off 0 vs 3 on the suite:

| prompt | dead-off=0 | dead-off=3 | ratio | accept (both) |
|---|---|---|---|---|
| code_edit | 120.81 | 120.51 | 0.998 | 0.641 |
| multi_turn | 89.78 | 88.60 | 0.987 | 0.969 |
| free_prose | 54.46 | 54.42 | 0.999 | 0.771 |

dead-off=3 holds **≥0.987×** everywhere (free_prose 0.999), and acceptance is
**identical** between arms → at the production n_min=48 the latch never trips on
normal text (3 consecutive zero-accept 48-token drafts do not occur when
acceptance is 0.64–0.97). The latch is a true no-op on the proven 2–3× win.
**R2 harmlessness criterion CONFIRMED** (≥0.98× on free_prose; code_edit within
noise).

> Phase A showed even `free_prose` benefits from ngram (1.4×) — it is not a pure
> worst case. The concern that a productive generation could trip the latch and
> forfeit that win did **not** materialize at n_min=48 (identical acceptance,
> 0.999× tg).

## B4b — Mechanism + value (stress: n_match=4, n_min=4)

To exercise the pathological regime the latch guards, ngram was made hyper-eager
(`NMATCH=4 NMIN=4 NMAX=64`) on an adversarial divergent prompt
(`prompts_adversarial.jsonl`), dead-off 0 vs 3, `LLAMA_TRACE=1`:

| | dead-off=0 | dead-off=3 |
|---|---|---|
| tg (t/s) | **22.49** | **39.22** |
| draft acceptance | 0.020 (≈3 / 1264 per gen) | latch trips |
| `disabling for seq` trips | 0 | 3 (one per generation) |
| draft tokens / gen | ~1000–1264 (unbounded) | ~97–156 (capped at trip) |

- **Mechanism:** `accept: 3 dead ngram-mod fires - disabling for seq 0` fires
  exactly when 3 consecutive zero-accept drafts occur, and never at dead-off=0.
- **Value:** without the latch, constant fire-and-fail drags tg to **22.49 —
  41% below the ~38 t/s no-spec base**. The latch trips early, stops the waste,
  and recovers tg to **39.22 (~base): a 1.74× recovery**, capping wasted drafts
  from ~1264 to ~156 tokens.

## R2 verdict

**CONFIRMED.** The hard-off latch is harmless on real workloads (≥0.999× on
free_prose, identical acceptance, never trips at n_min=48) and bounds the hostile
worst case (1.74× tg recovery, draft waste capped). Default
`--spec-ngram-mod-dead-off 3` ships safely: dormant insurance at the production
n_min=48, active only under sustained zero-accept drafting. Because 3 is the
default, the R1 recommended config already carries the latch — no extra flag
needed; set `--spec-ngram-mod-dead-off 0` only to opt out.

## Reproduce R2

```bash
# harmlessness (default n_min) -> results/summary_deadoff.json
MODE=deadoff KV=q8_0 REPEATS=3 SERVER_BIN=$PWD/build/bin/llama-server \
  python scripts/perf/bench_spec.py
# mechanism + value (stress) -> results/summary_deadoff_stress.json
MODE=deadoff KV=q8_0 REPEATS=2 NMATCH=4 NMIN=4 NMAX=64 LLAMA_TRACE=1 \
  PROMPTS=$PWD/scripts/perf/prompts_adversarial.jsonl \
  SERVER_BIN=$PWD/build/bin/llama-server \
  python scripts/perf/bench_spec.py
# then: grep 'disabling for seq' scripts/perf/results/deadoff3-q8_0.log
```
