# Tessera polars integration — full scout

_Author: Mavis (mavis). Date: 2026-08-03. Scope: imatrix / calibration /
quantization / runtime-dequant / L3 / L4 / L5 / evaluation pipelines, end to
end. Companion to `docs/runtime-aware-pipeline.md` and
`docs/per-tensor-calibration.md`._

## 0. TL;DR (read this first)

Polars is **partially already adopted** in Tessera — it is not a new
dependency. It is the canonical bridge from the imatrix GGUF writer to
the rest of the evidence / observability surface (5 files, ~6,800 LoC of
polars-driven paths). What is missing is **the rest of the pipeline**:

- L1 / L1.5 sidecar readers (Python side) are pure numpy
- L3 outlier report aggregation is stdlib + numpy
- Per-layer error table rollups are stdlib + numpy
- L5 orchestrator / L4 report consumption is stdlib dicts
- Quantize consumer (C++) is fine; the Python analytical surface
  surrounding it is the missing piece
- llama-bench / perplexity CSVs land in JSON or CSV; no rollup tooling
  joins them with imatrix / sidecar evidence

The right framing is: **promote the polars evidence-store style to the
rest of the Python analytics layer.** Do not invent a second dataframe
library — polars already won, and the codebase never adopted pandas.

## 1. End-to-end data flow (the canonical pipeline)

```
                       PRODUCER (C++)                            CONSUMERS (C++ & Python)

imatrix  tools/imatrix/imatrix.cpp        ─┬─→  imatrix.gguf / imatrix.dat
                                           │         │
                                           │         ├─→  quantize.cpp  (load_imatrix, tools/quantize/quantize.cpp:194)
                                           │         │       └─→  tessera-imatrix.cpp (act_scales)
                                           │         │             └─→  tessera-mm-imatrix.cpp (MoE multi-matrix)
                                           │         │                   └─→  tessera-regime.cpp (Q2/K/T640 selector)
                                           │         └─→  evidence-store.py  (ingest_imatrix)  ★ polars
                                           │                └─→  observer/*.parquet  +  router/*.parquet

L4 probe  per_tensor_calibrate / awq-evolve ──→  policy JSON  (llama.speculative.calibration-policy.v1)
                                                                 │
l5 plan   l5_orchestrator.py (L4 + imatrix + L4 bwd-perturb)  ──→ requant plan JSON
                                                                 │
apply     per_tensor_calibrate / quantize                    ──→ policy applied → quantize.cpp

L1 dequant  tessera-debug (CPU/MM/Metal hooks)   ──→  <name>.dequant.f32    (v3 sidecar)
L1.5 ref    tessera-debug (W4A4 mode)            ──→  <name>.act.dequant.f16 (v3 sidecar)  +  provenance.json

L3 outlier  l3_outlier_report.py                  ←──  <name>.dequant.f32
L3.5 err    per_layer_error_table.py              ←──  pair (L1, L1.5)
Hessian     l3_hessian_trace.py                   ←──  weight + activation (npz)
Awq-evolve  awq-evolve.py (island GA)             ←──  .npz bundles → policy

llama-bench                                       ──→  .csv / .json / .jsonl / .sql
perplexity                                        ──→  .json
                                                                            │
                                                                            ▼
                                          (no polars join today; rollups done per-script)
```

**★ = polars already in use.** The only branch with a polars-driven path
is the evidence-store ingest. Everything below the sidecar line is
unframeworked Python.

## 2. Data-format inventory

| Format | Producer | Reader(s) | Where it lives today | Polars-ready? |
|---|---|---|---|---|
| `imatrix.dat` (legacy) | `imatrix.cpp:save_imatrix_legacy` | `common/imatrix-loader.cpp:common_imatrix_load_legacy` | disk | yes — convert via `evidence-store.py` |
| `imatrix.gguf` (default) | `imatrix.cpp:save_imatrix` (line 1206) | `common/imatrix-loader.cpp` + `gguf-py` | disk | **already wired** in `evidence-store.py:observer_frames` |
| `<name>.dequant.f32` (L1, v3) | `tessera-debug.cpp` CPU/CUDA/Metal hooks | `l3_sidecar_v3_reader.py` | per-tensor sidecar dir | numpy → flatten → polars |
| `<name>.act.dequant.f32|.f16` (L1.5, v3) | same, W4A4 mode | same reader | same dir | numpy → flatten → polars |
| `<name>.dequant.f32.provenance.json` | `tessera-debug.cpp` (close) | reader optional `provenance=True` | next to data | direct `pl.read_json` candidate |
| `<layer>.npz` (calibration bundle) | `make-awq-layer-bundles.py`, `make-moe-awq-bundles.py` | `per_tensor_calibrate.py`, `awq-evolve.py` | bundle dir | numpy → flatten → polars |
| `llama.tessera.awq-evolution.v1` (JSON) | `awq-evolve.py` | `evidence-store.py:ingest_evolution` | checkpoint dir | **already wired** |
| `llama.tessera.router` (parquet) | `evidence-store.py:router_frame` | self | `evidence/router/part-*.parquet` | **already polars** |
| `llama.tessera.observer` (parquet) | `evidence-store.py:observer_frames` | self | `evidence/observer/part-*.parquet` | **already polars** |
| `llama.speculative.calibration-policy.v1` (JSON) | `per_tensor_calibrate.py`, `l3_hessian_trace.py` | `tile640_quantize_v3.py`, `l5_orchestrator.py` | `policy.json` | direct polars `pl.read_json` |
| `llama.tessera.per-layer-error-table.v1` (JSON) | `per_layer_error_table.py` | none | per-run | direct polars `pl.read_json` |
| `tessera.l3-outlier-report.v1` (JSON) | `l3_outlier_report.py` | none | per-run | direct polars `pl.read_json` |
| `tessera.l4-e2e-probe.v1` (JSON) | `l5_demo.py:50` (synthetic) / L4 probe | `l5_orchestrator.py:_read_l4_report` | per-run | direct polars `pl.read_json` |
| L5 plan history | `l5_orchestrator.py:write_history` | none | plan sidecar | direct polars |
| llama-bench CSV/JSON | `tools/llama-bench/llama-bench.cpp` | `scripts/compare-llama-bench.py` (GitPython + tabulate) | per-run | direct polars |
| perplexity JSON | `tools/perplexity/perplexity.cpp` | none | per-run | direct polars |

## 3. Existing polars surface (the seed of the integration)

The 5 files where polars is in use today, in order of relevance to this
scout:

### 3.1 `tools/tessera/evidence-store.py` (482 lines) — **the canonical bridge**

- Already converts imatrix.gguf → `observer/part-*.parquet` with one
  row per (tensor, expert, channel): `rms, mean_abs, kurtosis,
  tail_ratio` plus the raw `sum2, sumabs, sum4, maxabs, count`.
- Also converts `*.ffn_moe_router` tensor groups → `router/part-*.parquet`
  with per-expert sufficient statistics (`selected, probability_sum,
  confidence_sum, margin_sum, output_error_sum,
  downstream_divergence_sum`).
- `ingest_evolution` and `ingest_shadow` and `ingest_acceptance_*` keep
  the same evidence schema across all paths.
- Schema: `llama.tessera.evidence.v1`. Writer is
  `write_part(...)` (line 39) with zstd compression, statistics on,
  row_group_size=65536. **This is the pattern to clone elsewhere.**

### 3.2 `tools/tessera/moe_calibration.py` (257 lines)

- Reads `router.parquet` produced by 3.1 with `pl.read_parquet(...).to_dicts()`
  and reconstructs a `RouterAccumulator` per layer.
- The numpy-first pattern is the only viable one (the accumulator class
  is mutated row-by-row); polars only feeds the loop. **This is correct
  — do not try to rewrite the accumulator in polars expressions.**

### 3.3 `scripts/alphaevolve-metrics.py` (218 lines) — **the precedent to copy**

- Reads `.zcode/alphaevolve/*/gene-ledger.json` + `findings.jsonl`,
  normalizes the heterogeneous `champion_scores` dicts into tidy long
  format via a regex-based family bucketing, and emits a unified table
  or CSV/JSON dumps.
- Already does what the calibration pipeline needs: schema-drift
  handling, family bucketing, long-format output.
- **Template to clone** for the imatrix × sidecar × L4 cross-rollup.

### 3.4 `tools/tessera/unsloth-policy.py`

- Lazy-imports polars; only the bridge / observer-evidence path needs
  it, so PE-QAT sub-mode still works on polars-free systems. **Good
  pattern** for any opt-in polars consumer that the calibration
  harness container (which ships without polars) might run.

### 3.5 `tools/tessera/hf-evidence.py`

- Polars imported at module top — different stance from
  `unsloth-policy.py`. Bridge to the Hugging Face evidence
  repository; the polars dependency is unconditional because this
  code only runs in the bridge context.

**Pattern synthesis across the 5 files:**

- `evidence-store.py` and `moe_calibration.py` show the **imatrix →
  polars bridge** (read gguf, flatten to long form, write parquet).
- `alphaevolve-metrics.py` shows the **cross-source tidy long-format
  join** (multiple JSON shapes, one tidy table).
- `unsloth-policy.py` shows the **opt-in polars** discipline for
  container-restricted environments.

The integration the rest of the calibration surface needs is the
**third leg of the stool**: cross-source tidy rollups of imatrix +
sidecar + L4 + policy. The 5 existing files give you the read and
write primitives; only the rollup layer is missing.

## 4. Non-polars workhorses (the natural candidates)

These three scripts own the cross-pipeline rollups that today are
re-implemented per-script with stdlib dicts. They are the highest-value
polars integration targets.

### 4.1 `tools/tessera/l3_outlier_report.py` (617 lines)

- Reads every `<name>.dequant.f32` in a sidecar dir (its own
  `read_sidecar` parser at line ~150 that duplicates v3 reader logic
  — known minor fork), aggregates outlier count + fraction per tensor,
  emits a console report, JSON, CSV, and a per-row PNG plot.
- Multiple `--sidecar-dir LABEL:PATH` arguments mean **the cross-dir
  rollup is already a first-class CLI feature** — the data shape is
  already wide (one column per dir). Polars would convert this to a
  single `pl.concat([...], how="diagonal_relaxed")` and a
  `df.pivot("tensor", "sidecar_label", "outlier_fraction")` join.
- **Polars win: collapse the SidecarGroup / SidecarGroup datatypes to
  one DataFrame; the L3 ceiling-breach warning becomes a single
  `df.filter(pl.col("outlier_fraction") > ceiling)` line.** The CSV
  writer goes away; `df.write_csv` is a one-liner.

### 4.2 `tools/tessera/per_layer_error_table.py` (434 lines)

- Pairs L1 and L1.5 sidecars by tensor name; computes per-tensor
  `epsilon = ||L15 - L1||^2_F / ||L15||^2_F`; aggregates per layer
  (sum of per-tensor epsilons, sorted by block index).
- Pure numpy per-tensor math (which is the right choice for
  Frobenius); the aggregation step is stdlib dicts and a
  `_BLK_RE.match` regex. **Polars should not touch the per-tensor
  math** — it should consume the (tensor, layer, epsilon) record list
  and turn it into a tidy per-layer table joined with
  imatrix/l3-outlier evidence.
- **Polars win: same join pattern as 4.1, plus a per-layer rollup
  (`group_by("layer").agg(pl.col("epsilon").sum())`) that today is
  re-implemented.**

### 4.3 `tools/tessera/l5_orchestrator.py` (972 lines)

- Reads L4 report (JSON, per-tensor: `current_qtype, mse,
  mse_minus_one, perplexity, top1_mismatch, n_weights`).
- Reads imatrix as `{tensor: float_magnitude}` JSON (pre-normalized
  somewhere upstream — there is a hand-rolled `_read_imatrix` helper
  at line 782).
- Reads gradient_proxy from two L4 samples; reads layer_position_prior
  from a block-index ramp.
- Combines into weighted ema, emits RequantPlan, applies via
  `per_tensor_calibrate` or directly to policy.
- **Polars win: turn the 4 component dicts into a single
  `tensors × components` DataFrame once, then run the EMA / plan /
  decay as vectorized expressions. The orchestrator code shrinks
  ~30%, and the data shape becomes inspectable (e.g. "which tensor
  has gradient=0 but imatrix=top decile?" becomes a 2-line polars
  query).**

The orchestrator's `requant_plan_history` (line 683) is also
serializeable to parquet alongside the L4 report; today it lives
only as a JSON sidecar.

## 5. New analyses polars would unlock

The 4.1 / 4.2 / 4.3 refactors each unlock a new class of analysis
that is impractical in stdlib + numpy. Concrete examples:

### 5.1 The "imatrix × sidecar × L4" 3-way join

For a given calibration run, the architect's question is usually
"which imatrix-flagged sensitive tensors also have the worst L1/L1.5
sidecar error?" Today you run three scripts in sequence and grep.
With polars it is one query:

```python
df = (
    imatrix_tidy  # (tensor, expert, channel, rms, kurtosis, tail_ratio, count)
    .join(l3_tidy, on="tensor", how="left")
    .join(layer_err_tidy, on="tensor", how="left")
    .with_columns(per_layer_zscore=pl.col("epsilon").rank() / pl.len())
    .filter(pl.col("per_layer_zscore") > 0.95)
    .select("tensor", "layer", "rms", "outlier_fraction", "epsilon",
            "per_layer_zscore")
    .sort("per_layer_zscore", descending=True)
)
```

This is the canonical "where should I requant" rollup. It is not
possible today without re-implementing the per-script loading and
joining in your head.

### 5.2 Cross-corpus calibration divergence

`imatrix.datasets` is an array of corpus paths in the GGUF header
(`LLM_KV_IMATRIX_DATASETS`, see `common/imatrix-loader.h:8`). With
multiple imatrix runs over different corpora, polars' long-format
join gives a per-tensor `(corpus, tensor, rms, kurtosis, …)` table
that today is impossible to compute without writing a custom merger.

### 5.3 Per-layer error heatmap across quant policies

After `per_tensor_calibrate.py` has produced a policy for awq/lrq/
flrq, the same join on `tensor → per-tensor mse` × `policy` × `layer`
gives the canonical "which policy wins at which layer" heatmap.
Today: run the pipeline once per policy, then eyeball JSONs.

### 5.4 L5 plan history × L4 outcome feedback loop

`l5_orchestrator.py:write_history` writes a plan history; the L4
re-probe writes a per-tensor mse delta. Joining them by tensor and
plan-index gives the "did this requant plan actually reduce error?"
trace. With parquet, this is a SQL-style join; with stdlib dicts
today it is impossible to even express the question cleanly.

### 5.5 L3 outlier × imatrix tail-ratio cross-check

The imatrix observer already computes `tail_ratio = max_abs / rms`
per channel (see `evidence-store.py:95`). The L3 sidecar gives a
per-row outlier count. Joining on (tensor, row) lets you ask "do
the imatrix tail-ratio outliers line up with the sidecar outlier
rows?" — the single most useful sanity check for both pipelines.
Polars turns this into a 5-line query; today it is a manual
eyeball-and-spreadsheet operation.

### 5.6 llama-bench × perplexity × quantize-policy correlation

The `scripts/compare-llama-bench.py` script already exists for
benchmark comparison (uses GitPython + tabulate, no polars).
Adding a third join against a `policy × perplexity × tok/s` tidy
table gives the canonical "did this calibration policy help
quality AND speed, or did we trade one for the other?" table.

## 6. Integration plan (phased, ordered by independence)

### Phase 0 — formalize the evidence schema (0.5 day, no behavior change)

- Codify the long-format `tensors × components` schema that
  `evidence-store.py` already produces. Add a one-page
  `docs/tessera-evidence-schema.md` (or extend
  `docs/runtime-aware-pipeline.md`) with the column list and the
  writer conventions (`write_part` zstd / row_group_size).
- Add `polars>=0.20` to a new `requirements/requirements-tessera.txt`
  (kept separate from the upstream llama.cpp requirements files so
  the upstream sync stays clean). The imatrix alphaevolve
  preflight already documents
  `pip install --user --break-system-packages polars`; capture
  that as the install recipe.
- **No code change to the 5 existing polars users.** This phase
  just unblocks the next two.

### Phase 1 — clone the evidence-store pattern in three places (2-3 days)

- **1a.** `l3_outlier_report.py`: keep the sidecar parser as-is (it
  is needed for the per-row PNG plots that need numpy arrays);
  replace the `SidecarGroup` / `TensorOutlierReport` dataclasses
  with a single polars DataFrame; replace `render_console` with a
  polars `select + sort + head(top_k) + str.format` chain; replace
  `render_csv` with `df.write_csv`. Keep JSON as
  `df.write_json(row_oriented=True)`. Net: ~150 lines deleted.
- **1b.** `per_layer_error_table.py`: keep the per-tensor epsilon
  math in numpy; replace `aggregate_per_layer` with a
  `group_by("layer").agg(pl.col("epsilon").sum())` and add an
  optional `--imatrix-tidy <parquet>` join column. Net: ~80 lines
  deleted; one new feature enabled.
- **1c.** `l5_orchestrator.py:TensorState` and the component-score
  dicts: convert the 4 component dicts to a single
  `pl.DataFrame({"tensor", "imatrix_mag", "gradient", "layer_prior",
  "w_imatrix", "w_gradient", "w_layer"})` once at `_load_tensors`
  time; the EMA / decay / plan-apply become vectorized polars
  expressions. The plan-history writer becomes
  `df.write_parquet(..., compression="zstd", row_group_size=4096)`
  matching the evidence-store convention.

After Phase 1, every analytical script in `tools/tessera/` reads
or writes evidence in the same tidy long format. This is the
"polars is the default" milestone.

### Phase 2 — cross-source rollups (1-2 days, no behavior change)

- Add `tools/tessera/calibration_rollup.py`: a new read-only script
  that takes `--evidence-store <dir>`, `--sidecar-dir ...`, `--l4
  <json>`, `--l5-plan <parquet>` and emits a single
  `rollup.parquet` joining everything on `tensor`. This is the
  artifact that the alphaevolve run script
  (`scripts/alphaevolve-metrics.py`) was already shaped to read
  — a long-format tidy table of one row per (run, tensor, metric).
- The `l3_hessian_trace.py` output (per-tile tr(H)) is a natural
  additional column for the same rollup; adding it as a join is
  one more read.

### Phase 3 — feedback loops that were not possible before (1-2 days, new features)

- `l5_orchestrator.py` adds a `--plan-history-in <parquet>` flag;
  reads the prior plan history, uses polars to compute the per-tensor
  `delta_mse = mse_after - mse_before` for each prior requant, and
  adjusts the EMA decay per-tensor based on whether prior requants
  helped or hurt. This is the "did this requant plan actually reduce
  error?" feedback loop from 5.4.
- `evidence-store.py` adds a `--corpus-summary` mode that joins
  the imatrix dataset paths (from `LLM_KV_IMATRIX_DATASETS`) across
  multiple imatrix runs and produces a per-corpus coverage table.
  This is 5.2.
- `l3_outlier_report.py` adds a `--imatrix-tidy <parquet>` join
  that produces the 5.5 sanity-check table.

### Phase 4 — cross-tool rollup (1 day, optional)

- Extend `scripts/compare-llama-bench.py` to also accept
  `--rollup <parquet>` and emit a combined `tok/s × perplexity ×
  epsilon × imatrix_rms` table for the compared commits. Drop
  GitPython + tabulate; replace with polars + the existing
  `compare-llama-bench.py` CSV output as a polars source.
- The same `rollup.parquet` becomes the input to any future
  alphaevolve reward signal that wants to optimize calibration
  quality *and* runtime speed jointly.

## 7. Risks and constraints

1. **C++ side is not the integration point.** The producer (imatrix,
   dequant sidecar) is C++; the C++ readers (`common/imatrix-loader`,
   `l3_sidecar_v3_reader` analogue in C++ if any) are fine and do
   not need to change. Polars is a Python-side framework; the
   integration lives in `tools/tessera/*.py`.

2. **Per-tensor numpy math stays.** Frobenius norms, KL, MSE,
   per-row outlier counts — these are dense linear algebra on
   per-tensor arrays and polars is the wrong tool. Do not be
   tempted to express them in polars expressions; keep them in
   numpy, feed the result into polars as a record-per-tensor
   frame.

3. **The MoE `router.parquet` is row-oriented and small.** The
   polars ingestion in `moe_calibration.py` is correct as-is
   (parquet → list of dicts → numpy accumulators). Do not try to
   express the per-expert accumulation in polars — the accumulator
   class is mutated and that is fine.

4. **Container environment may not have polars.** The calibration
   harness container is documented as "ships without polars" in
   `unsloth-policy.py:25`. Use the lazy-import pattern from
   `unsloth-policy.py` for any code path that runs inside the
   container; the rest of `tools/tessera/*.py` can take a
   top-level import.

5. **The `imatrix-impl-w1` alphaevolve wave is in flight.** Phase 1
   of that wave is "verify cheap wins" and does not touch the
   Python surface. Phase 2 (streaming reduction, Metal observer
   fusion, vDSP + sharded map) is producer-side and orthogonal
   to this scout. Coordinate the Phase 0 schema doc with the
   alphaevolve wave's findings; in particular, if Phase 2 ships
   `max_abs` accessor (PREFLIGHT item 2), the evidence-store
   `tail_ratio` column becomes 1:1 with the wire max_abs and
   should be re-verified in the new audit.

6. **No pandas. Do not introduce it.** The codebase has never
   imported pandas. Adding it would split the dataframe world
   into two libraries. If a future polars API gap appears, the
   right answer is to vendor the missing operation, not to
   introduce pandas.

## 8. Out of scope (deliberately not in this scout)

- The C++ `common_imatrix_load` and `tessera-imatrix.cpp` /
  `tessera-mm-imatrix.cpp` consumers are correct; the missing
  feature is `max_abs` exposure to the DartQuant/ChampQ path,
  which the imatrix-study identifies as Q3 (the cheapest quality
  win). That is the alphaevolve wave's job, not this scout's.
- DFlash / D-PACE / block-dataset pipelines. They are separate
  evidence ledgers and have their own data contracts.
- llama-bench / perplexity write paths (the C++ side is fine;
  the join is the missing piece and is in Phase 4 only).
- The DuckDB persistent pipeline store
  (`.zcode/alphaevolve/imatrix-study/study.md` §3.11,
  commit `d0ba47b49`). DuckDB and polars coexist; DuckDB is
  the upstream pipeline store, polars is the analysis layer.
  The integration question there is "should polars read
  DuckDB tables directly" — that is its own design question
  and out of scope here.

## 9. One-sentence summary

Promote the polars evidence-store pattern (already 482 lines of it,
5 files, ~6,800 LoC of polars-driven paths) to be the default
tidy-long-format backbone of every Python script in
`tools/tessera/` that does cross-pipeline rollups — start with
`l3_outlier_report.py`, `per_layer_error_table.py`, and
`l5_orchestrator.py`, then add a `calibration_rollup.py` joiner,
then close the L5 feedback loop on parquet — keep the per-tensor
numpy math and the C++ producer untouched, do not introduce
pandas, do not touch the in-flight imatrix alphaevolve wave.
