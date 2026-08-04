# Tessera unified store: `tessera.duckdb` + the write buffer pattern

_Author: Mavis (mavis). Date: 2026-08-04. Companion to
`docs/tessera-polars-integration-scout.md`. Supersedes the per-pipeline
storage sketch in scout §6._

## Why a single store

The scout identified the imatrix, calibration, quantization, and
analytics pipelines as four independent stores (imatrix parquet,
C++ DuckDB GA store, NDJSON analytical outputs, parquet
calibration rollup). The shared per-tensor statistics
(`kurtosis, eff_rank, rms, mean_abs, tail_ratio, family, layer, ...`)
were computed redundantly in three of the four pipelines, and
cross-pipeline joins required re-implementing the per-script
loader.

The unified `tessera.duckdb` consolidates the per-tensor summary
rows into one DuckDB file, owned by the C++ quantize binary's
schema setup and consumable by both sides. The bulky per-channel
data (imatrix observer, MoE router) stays in parquet / NDJSON;
only the per-tensor summary fields land in DuckDB. The cross-pipeline
join collapses to a `SELECT ... FROM tensor_stats JOIN l4_probe_summary
USING (model_hash, name)` instead of three hand-written parsers.

## The schema (additive on top of the existing C++ tables)

| Table | Rows | Writer(s) | Notes |
|---|---|---|---|
| `runs` | 1 per quantize run | C++ (`ts_quantize_db_begin_run`) | Per-run lifecycle; status `running` / `completed` / `failed`. |
| `tensors` | 1 per quantize 2D weight, per run | C++ (`ts_quantize_db_insert_tensor`) | Legacy per-run table; the GA-prep walk writes kurtosis / eff_rank here. Kept for backward compat; new code reads from `tensor_stats`. |
| `ga_evaluations` | 1.6M per run | C++ (`ts_db_buffer` for `ga_evaluations`) | Per-candidate GA telemetry. Hot path; goes through the buffer. |
| `ga_results` | 1 per converged tensor | C++ (`ts_quantize_db_insert_ga_result`) | Best (alpha, clip) per tensor. Warm-start seed source. |
| `acceptance` | 1 per tensor | C++ (`ts_quantize_db_insert_acceptance`) | Acceptance-gate verdict. |
| `l5_fixups` | 1 per requant fixup | C++ (`ts_quantize_db_insert_l5_fixup`) | L5 adaptive requantize fixup rows. |
| **`tensor_stats`** | 1 per tensor per model | C++ (GA-prep) + Python (cal) | **The cross-pipeline feature table.** PRIMARY KEY (model_hash, name) + ON CONFLICT DO UPDATE = upsert target. C++ writes kurtosis / eff_rank / dtype; Python writes rms / mean_abs / tail_ratio. `source` records which pipeline last wrote the row. |
| **`l3_outlier_summary`** | 1 per tensor per sidecar label | Python (analytics) | L3 outlier rate per tensor. model_hash + name joins to tensor_stats. |
| **`l4_probe_summary`** | 1 per tensor | Python (analytics) | L4 E2E probe (mse, perplexity, top1_mismatch). |
| **`l4_plan_outcome`** | 1 per (tensor, iteration, plan_id) | C++ (adaptive_requantize loop) | **The feedback-loop audit trail.** The per-iteration L4 measurement AFTER a requant plan was applied, with the before/after split. The C++ `ts_dispatch_run_l5_loop` writes one row per (tensor, gen) via `ts_quantize_db_append_l4_outcome`. |
| **`l5_plan_summary`** | 1 per (tensor, iteration, plan_id) | Python (l5_orchestrator) | L5 requant plan (sensitivity_score, recommended_alpha, recommended_clip). |
| **`l5_outcome`** | 1 per (tensor, iteration, plan_id) | Python (`tools/tessera/l5_outcome.py`) | **The feedback-loop verdict.** Computed by `l5_outcome.py` from a join of `l5_plan_summary` and `l4_plan_outcome`. `plan_accepted` is True if `delta_mse < accept_threshold`. `residual` is the per-(model, family) linear-fit residual of delta_mse on sensitivity_score (a running measure of how well the orchestrator's sensitivity scoring predicts the actual error delta). Phase 14: also carries the per-tensor components `imatrix_magnitude`, `gradient_proxy`, `layer_position_prior` (all nullable) so a future retune can fit a 3-coefficient model. |
| **`l5_weights`** | 1 per (model, family) | Python (`tools/tessera/l5_retune.py`) | **The feedback-loop consumer.** Per-(model, family) retuned `(w_imatrix, w_gradient, w_layer)` on the simplex. Computed by `l5_retune.py` from a per-(model, family) closed-form OLS of `delta_mse` on `sensitivity_score` and projected to the simplex. The orchestrator's next generation reads this table via `--retune-from-db`, closing the loop. Phase 14: also carries `requant_budget_bits` (nullable) — the dispatch-side budget the retune recommends for the next requant pass. |
| **`per_layer_error_summary`** | 1 per tensor | Python (analytics) | L1/L1.5 sidecar epsilon. |

The 7 bolded tables are the additions on top of the existing
6-table schema; see the scout §0 / §6 for the original inventory.

## The feedback loop: "did this requant plan actually reduce error?"

The scout §5.4 identified the cross-pipeline feedback loop as the
analytical force multiplier: a way to ask "for each L5 plan we
emitted, did the L4 measurement AFTER the plan was applied
actually show a reduction in error?" and use that signal to retune
the orchestrator's sensitivity scoring.

The loop lands here with three pieces:

1. **C++ producer (`l4_plan_outcome`).** The dispatch's
   `ts_dispatch_run_l5_loop` writes one row per
   `(tensor, gen)` with the before/after split (alpha / clip /
   outlier_thresh parameters + rel_frob) via
   `ts_quantize_db_append_l4_outcome`. plan_id is
   `cpp_quant_gen{N}_stage{S}` where S is the stage that won
   (A or B). The Python orchestrator's plan_id is
   `py_orch_iter{N}` and uses a different prefix; both coexist
   on the (model_hash, name, iteration, plan_id) primary key.
2. **Python producer (`l5_plan_summary`).** The existing
   `l5_orchestrator.py::write_history` writes one row per
   `(tensor, iteration, plan_id)` with `sensitivity_score` and
   the recommended `alpha / clip`. Already existed before this
   commit; the unified-schema migration just landed the table.
3. **Python consumer (`l5_outcome.py`).** Reads
   `l5_plan_summary` JOIN `l4_plan_outcome` on the primary key,
   computes `delta_mse = mse_after - mse_before` and
   `plan_accepted = (delta_mse < accept_threshold)`, fits a
   per-(model, family) linear model of `delta_mse` on
   `sensitivity_score` and records the residual, and writes
   the verdict to `l5_outcome`.

The hit-rate metric (per-run, per-family) and the sensitivity
calibration residual are the two new things the loop gives the
architect. The hit rate answers "of the plans we emitted this
run, how many actually helped?"; the calibration residual answers
"how well does our sensitivity scoring predict the actual
delta?" Both are first-class observables on the l5_outcome
table.

Production data flow:

```
                    C++ dispatch
                    ts_dispatch_run_l5_loop
                            │
                            ├── l4_plan_outcome (per-iter audit trail)
                            │
Python l5_orchestrator        │
    l5_plan_summary ──────────┴────────► l5_outcome.py ─────► l5_outcome
                                                                     │
                                                                     ▼
                                                              l5_retune.py
                                                              (per-family OLS)
                                                                     │
                                                                     ▼
                                                                l5_weights
                                                                     │
                                                                     ▼
                                              l5_orchestrator --retune-from-db
                                              (closes the loop)
```

The retune is the loop's second consumer (`l5_retune.py`). For
each (model, family) in `l5_outcome`, it fits a closed-form
OLS `delta_mse = a + b * sensitivity_score`, then projects the
result onto the (w_imatrix, w_gradient, w_layer) simplex:

* `b > 0` + low hit rate -> shift mass from `w_imatrix` to
  `w_gradient` (the imatrix signal is leading the orchestrator
  astray for this family).
* `b < 0` + low hit rate -> shift mass from `w_gradient` to
  `w_imatrix` (the imatrix is correctly identifying the
  sensitive tensors, and those tensors are being protected;
  amplify the signal that drove it).
* `hit_rate = 1.0` -> gate=0, no shift (the orchestrator is
  working for this family; don't fix it).
* `n < min_samples` (default 3) -> skip; the OLS is too noisy.

The shifted weights are projected to the simplex
(non-negative, sum=1) and written to `l5_weights` with
PRIMARY KEY `(model_hash, family)`. The orchestrator's next
generation reads the table back via `--retune-from-db` and
uses the n_samples-weighted average across families as the
starting `(w_imatrix, w_gradient, w_layer)` tuple. This is
the closed-loop optimization the feedback loop was designed
for.

## The write buffer pattern

The unified store has a heavy-parallel write problem: the GA
hot path is 16-64 candidate evaluators appending to
`ga_evaluations` simultaneously, the calibration pipeline is
multi-corpus (each corpus has its own worker), and the analytics
pipeline is polars-parallel. DuckDB is single-writer per file,
so all of this would serialize on the connection lock and become
the bottleneck.

The `ts_db_buffer` (C++) and `TesseraDBBuffer` (Python) abstractions
solve this with the standard multi-producer / single-consumer pattern:

```
N producer threads                1 flusher thread          DuckDB
  append(row)  ────────►  pending queue   ─────►  bulk INSERT
                              │
                              └─ count + time trigger
```

Properties:

- **`append` is hot-path cheap.** A vector copy under the buffer's
  mutex + maybe a `cv.notify`. No SQL, no network, no I/O.
- **One writer.** The flusher thread is the only code path that
  calls `db->conn->Query`. DuckDB's MVCC handles concurrent reads.
- **Batching.** 65,536-row batches (matches the evidence-store
  `row_group_size` convention) and a 1-second time flush.
- **Sync-on-exit.** The destructor forces a final drain. A process
  that exits cleanly never loses a buffered row.
- **Best-effort.** A failed flush logs to stderr, bumps
  `rows_dropped`, and the producer continues. The DB is a recording
  aid, never a correctness requirement.

`tensor_stats` is an exception: it has a primary key, so the
buffered append path would fail on a duplicate. The Python and
C++ sides both bypass the buffer for `tensor_stats` and use a
direct `INSERT ... ON CONFLICT (model_hash, name) DO UPDATE SET
...` with `COALESCE` on the update clause (the upsert preserves
the other side's columns when the new write's columns are NULL).

Production-tuned throughput (test 5, 8 threads x 50k rows = 400k):

```
append: ~500 ms (memory speed, no DuckDB)
flush:  ~3.4 sec total (~120k rows/sec sustained)
```

## CLI

The unified DB is opened by the dispatch with `--tessera-db PATH`.
The legacy `--quantize-db PATH` is a deprecated alias kept for one
release; it prints a warning to stderr when used. Both flags set
the same `tessera_params.tessera_db` field; if both are given, the
last one wins (CLI ordering convention).

## Files

| Layer | File | Role |
|---|---|---|
| C++ schema | `tools/quantize/tessera/tessera-quantize-db.{h,cpp}` | Owns the schema (CREATE TABLE IF NOT EXISTS) and the typed insert helpers. `--tessera-db PATH` opens or creates the file. |
| C++ buffer | `tools/quantize/tessera/tessera-db-buffer.{h,cpp}` | MPSC write buffer with count + time + explicit flush triggers and sync-on-exit. |
| C++ dispatch | `tools/quantize/tessera/tessera-dispatch.cpp` | The dispatch opens two buffers (`ga_evaluations` + `l4_plan_outcome`); the eval_recorder + L5 loop push rows into them. The GA-prep walk also upserts into `tensor_stats` via `ts_quantize_db_upsert_tensor_stat`. Phase 14: pre-loads `l5_weights` for the GA's `family_seed_lookup` warm-start; the L5 loop's early-exit consults `l5_outcome` for the converged-fast gate. |
| C++ tests | `tools/quantize/tessera/test_db_buffer.cpp`, `tools/quantize/tessera/test_quantize_db.cpp`, `tools/quantize/tessera/test_l5_dispatch.cpp` | 8 + 6 + 35 cases. |
| Python buffer | `tools/tessera/tessera_db_buffer.py` | Python mirror; same contract. |
| Python unified | `tools/tessera/tessera_db.py` | High-level API (`TesseraDB` class) that owns the DuckDB connection + per-table buffers. Typed helpers per summary table. |
| Python l5_outcome | `tools/tessera/l5_outcome.py` | The feedback-loop consumer. Joins `l5_plan_summary` and `l4_plan_outcome`, computes `delta_mse` + `plan_accepted` + per-(model, family) `residual`, writes `l5_outcome`. |
| Python l5_retune | `tools/tessera/l5_retune.py` | The retune (closed-loop) consumer. Per-(model, family) closed-form OLS of `delta_mse` on `sensitivity_score`, projects to the (w_imatrix, w_gradient, w_layer) simplex, writes `l5_weights`. The orchestrator's `--retune-from-db` reads this table. |
| Python cal | `tools/tessera/calibration_to_tensor_stats.py` | Reads the per-channel observer parquet, reduces to per-tensor summary, upserts into `tensor_stats` (source = `py_cal`). |
| Python l3_outlier | `tools/tessera/l3_outlier_report.py` | The L3 outlier report. `--tessera-db` is the fast path: reads per-tensor summary from `tensor_stats` and produces a conservative 0/1 outlier count tagged `source='tensor_stats_estimate'`. The dequant-sidecar path remains the default. |
| Python tests | `tools/tessera/test_tessera_db_buffer.py`, `tools/tessera/test_tessera_db.py`, `tools/tessera/test_l5_outcome.py`, `tools/tessera/test_calibration_to_tensor_stats.py`, `tools/tessera/test_l3_outlier_fast_path.py`, `tools/tessera/test_l5_retune.py` | 8 + 7 + 5 + 3 + 4 + 15 cases. |

## Usage

### C++ side (quantize pipeline)

```cpp
// Already wired in ts_dispatch_db_open / ts_dispatch_db_close.
ts_quantize_db * db = ts_quantize_db_open("tessera.duckdb", &err);
ts_db_buffer * eval_buf = ts_db_buffer_open(db, "ga_evaluations", cols,
                                            65536, 1s);
ts_db_buffer * l4_buf   = ts_db_buffer_open(db, "l4_plan_outcome", l4_cols,
                                            1024, 1s);
for (...) {
    ts_db_buffer_append(eval_buf, { run_id, tensor_name, ... });
}
// In the L5 adaptive loop:
ts_quantize_db_l4_outcome row;
row.model_hash = wrap->model_hash;
row.iteration  = gen;
row.plan_id    = "cpp_quant_gen" + std::to_string(gen) + "_stage" + strategy;
row.mse_before = before; row.mse_after = after;
ts_quantize_db_append_l4_outcome(l4_buf, row);
// At end:
ts_db_buffer_flush_now(eval_buf);
ts_db_buffer_flush_now(l4_buf);
ts_db_buffer_close(&eval_buf);
ts_db_buffer_close(&l4_buf);
ts_quantize_db_close(db, "completed");
```

### Python side (calibration + feedback loop)

```python
from tessera_db import TesseraDB

# Calibration side
with TesseraDB.open("tessera.duckdb") as db:
    db.insert_tensor_stats(model_hash=model_hash, rows=[
        {"name": "blk.0.attn_q.weight", "family": "attn_q",
         "kurtosis": 5.2, "eff_rank": 0.85, ...},
        ...
    ])
    db.insert_l5_plan(model_hash=model_hash, rows=[
        {"name": "blk.0.attn_q.weight", "iteration": 0, "plan_id": "p0",
         "sensitivity_score": 0.87, "recommended_alpha": 0.5, ...},
        ...
    ])
    db.insert_l4_plan_outcome(model_hash=model_hash, rows=[{
        "name": "blk.0.attn_q.weight", "iteration": 0, "plan_id": "p0",
        "mse_before": 0.012, "mse_after": 0.010, "family": "attn_q",
        ...
    }])
    db.insert_l4_probe(model_hash=model_hash, rows=[...])
```

```bash
# Run the feedback loop consumer (joins + verdict)
python3 tools/tessera/l5_outcome.py \
    --db tessera.duckdb \
    --model-hash <hash> \
    --accept-threshold 0.0 \
    --print-summary

# Pipeline run:
# ll  ok: plans: 24
#      accepted: 19
#      hit_rate: 0.792
#      mean_delta: -0.0008
#      mean_residual: 0.0012
#      per-family:
#             attn_q  0.875
#             ffn_gate  0.714
```

```bash
# Run the retune (per-(model, family) closed-form OLS on
# l5_outcome -> l5_weights). Defaults: alpha=0.5,
# min_samples=3.
python3 tools/tessera/l5_retune.py \
    --db tessera.duckdb \
    --model-hash <hash>

# Pipeline run:
# l5_weights: wrote 3 row(s), skipped 1 (insufficient
# samples), of 4 (model, family) group(s)

# Use the retuned weights in the next generation's
# orchestrator. --retune-from-db reads l5_weights and the
# n_samples-weighted across-family mean overrides the
# --w-imatrix / --w-gradient / --w-layer flags.
python3 tools/tessera/l5_orchestrator.py \
    --l4-report l4.json \
    --imatrix imatrix.json \
    --retune-from-db tessera.duckdb \
    --model-hash <hash> \
    --policy out/l5-policy.json
# [l5] retune-from-db: 3 family row(s) ->
#       w=(0.420, 0.380, 0.200)
```

## Migration status

- **Phase 0 (commits `e90adc4d8` + `b70ec075c`):** schema
  migration. 7 new tables (5 in the first commit, +2
  `l4_plan_outcome` + `l5_outcome` for the feedback loop).
  All `CREATE TABLE IF NOT EXISTS`. Existing 6 tables untouched.
- **Phase 1 (commit `e96c0c21c`):** C++ buffer. 8 cases green.
  Existing `tessera-quantize-db` and `tessera-quantize-db-e2e`
  tests pass.
- **Phase 2 (commit `db0b1b002`):** dispatch refactor.
  Per-tensor DuckDB Appender sharded map replaced with a
  single `ts_db_buffer` for `ga_evaluations`. E2E + L5 tests
  green.
- **Phase 3 (commit `54e047674`):** Python side. Buffer +
  unified `tessera_db.py`.
- **Phase 4 (commit `b8eac1a2a`):** Python tests stabilized.
- **Phase 5 (commits `17724cc3a` + `bd4caa2e1`):** feedback
  loop producer side. C++ helper for `l4_plan_outcome`,
  dispatched into the L5 adaptive loop.
- **Phase 6 (commits `54e047674` + later + test commits):**
  feedback loop consumer. `l5_outcome.py` joins plan + outcome,
  computes verdict, writes `l5_outcome`. 5-case test green.
- **Phase 7 (commits `516d91cc3` + later):** C++ GA-prep walks
  upserts `tensor_stats` (kurtosis / eff_rank / dtype, source =
  `cpp_quant`); Python calibration writes
  `rms / mean_abs / tail_ratio` (source = `py_cal`). The
  `COALESCE` upsert preserves the other side's columns on
  a subsequent write.
- **Phase 8 (commit `48f353088`):** CLI rename. `--tessera-db PATH`
  is the new canonical flag; `--quantize-db PATH` is a
  deprecated alias for one release. The internal field
  `tessera_params.tessera_db` and `ts_dispatch_params::
  tessera_db_path` are renamed in lockstep.
- **Phase 9 (commit `97293385b`):** C type rename. `ts_quantize_db`
  -> `ts_tessera_db` across 10 files for consistency with
  the user-facing flag. Mechanical, no behavior change.
- **Phase 10 (commit `91d904056`):** `--imatrix-tidy <parquet>`
  join in `calibration_rollup.py` — per-tensor reduction
  (mean rms / mean_abs / kurtosis, max tail_ratio), prefixed
  `imatrix.*` columns. The cross-pipeline query now includes
  the imatrix's per-tensor statistics without a separate
  tessera.duckdb table.
- **Phase 11 (commit `3723eee69`):** `l3_outlier_report
  --tessera-db` fast path. Reads per-tensor summary from
  `tensor_stats` and produces a conservative 0/1 outlier
  count tagged `source='tensor_stats_estimate'`. The
  dequant-sidecar path remains the default. 4 new tests.
- **Phase 12 (this commit):** feedback loop closes. The
  retune (`l5_retune.py`) writes per-(model, family)
  retuned `(w_imatrix, w_gradient, w_layer)` to the new
  `l5_weights` table; the orchestrator's
  `--retune-from-db` reads it back. C++ side: schema
  addition + `ts_tessera_db_upsert_l5_weight` helper
  mirrored on the Python `TesseraDB.insert_l5_weights`
  (direct `INSERT ... ON CONFLICT DO UPDATE`, like
  `tensor_stats`). 15 Python tests + 1 new C++ test
  case.
- **Phase 13:** (skipped — Phase 12 already includes the
  close-the-loop work; the candidate commit was merged
  into Phase 12.)
- **Phase 14 (this commit):** C++ side of the feedback
  loop. Schema additions: `l5_weights.requant_budget_bits
  BIGINT` (nullable) + `l5_outcome.{imatrix_magnitude,
  gradient_proxy, layer_position_prior}` (nullable).
  C++ API: `ts_tessera_db_list_l5_weights` typed reader,
  `ts_tessera_db_l5_outcome_stats_for` per-(model,
  family) hit_rate aggregate, `ts_tessera_db_test_insert_l5_outcome`
  test-only INSERT helper. Dispatch wiring: `ts_dispatch_db_open`
  pre-loads `l5_weights` into `l5_weight_map`;
  `ts_dispatch_family_seed_lookup` consumes it as the
  primary warm-start source ahead of the legacy
  `ga_results` lookup (hit_rate > 0.5 -> alpha/clip
  biased up). L5 loop: `ts_dispatch_run_l5_loop` gains a
  converged-fast early-exit at the top of each gen
  (gen >= 1) — when `l5_outcome` hit_rate > 0.95 the
  loop breaks before the L2 measurement; the report
  JSON carries a `converged_fast=true` marker. Tests:
  `test_quantize_db` round-trip on the new field + the
  two new readers; `test_l5_dispatch` end-to-end
  warm-start + converged-fast coverage (synthetic
  l5_outcome row -> loop exits one generation early).

## Open follow-ups (after this commit)

The unified-DB feedback loop is now feature-complete on both
the producer and consumer sides. Phase 14 closed the three
remaining follow-ups from this section (the C++ warm-start,
the C++ converged-fast early-exit, and the `l5_outcome`
per-tensor component columns). New items that have surfaced
during Phase 14 work are tracked below.

- Per-tensor component storage in `l5_outcome` (DONE in
  Phase 14). The schema now carries nullable
  `imatrix_magnitude`, `gradient_proxy`,
  `layer_position_prior` columns. The C++ side does not
  populate them (the orchestrator's `write_history` is the
  production writer); the C++ reader accepts them. A
  retune that fits a 3-coefficient model per (model,
  family) is now possible without a follow-up schema
  change.
- Cross-model retune. `l5_weights` is keyed by
  `(model_hash, family)`, so a new model starts from the
  base weights. A second pass could add a global
  `(family, w_im, w_grad, w_layer)` row that's the
  across-model n_samples-weighted mean; the orchestrator
  would warm-start new models from that.
- GA-prep walk: read `l5_weights` to bias the GA's
  initialization (DONE in Phase 14). The dispatch's
  `ts_dispatch_db_open` pre-loads `l5_weights` into a
  per-family registry; the GA's `family_seed_lookup`
  hook consumes it as the primary warm-start source
  (ahead of the legacy `ga_results` lookup). The bias
  rule: `hit_rate > 0.5` -> alpha 25% higher and clip
  20% higher than the 0.5/0.7 base; otherwise the base.
- Converged-fast early-exit in the L5 adaptive loop
  (DONE in Phase 14). When `l5_outcome` has rows for
  the model with `hit_rate > 0.95`, the l5 loop breaks
  before the L2 measurement on `gen >= 1`. The first
  generation still runs (to seed `l4_outcome` rows);
  the early-exit fires from the second onward. The
  report JSON carries a `converged_fast=true` marker
  with the `hit_rate` and `n_rows` the dispatch saw.

## Phase 14 (this commit)

Phase 14 closes the C++ side of the feedback loop. The
retune's per-(model, family) `(w_imatrix, w_gradient,
w_layer)` and `hit_rate` now feed the dispatch's GA warm-start,
and the orchestrator's `l5_outcome` verdict drives a
converged-fast early-exit in the L5 adaptive requantize loop.
All schema changes are additive; existing rows remain
readable.

### Schema additions (additive)

- `l5_weights.requant_budget_bits BIGINT` (nullable).
  The dispatch-side budget the orchestrator's
  `l5_retune` recommends for this family in the next
  requant pass; computed by `l5_retune.py` as
  `total_storage * (1 - hit_rate) *
  base_budget_fraction` and NULL when the family has
  too few samples to project a budget. The dispatch
  currently reads it for forward-compatibility but
  does not act on the value yet; the column is here
  so the producer/consumer agree on the schema. C++
  uses a `-1` sentinel for NULL; the upsert SQL
  emits the literal `NULL` token when the field is
  set to `-1`.
- `l5_outcome.imatrix_magnitude DOUBLE` (nullable).
- `l5_outcome.gradient_proxy DOUBLE` (nullable).
- `l5_outcome.layer_position_prior DOUBLE`
  (nullable). The orchestrator's `write_history`
  path populates these so the retune can fit a
  3-coefficient model and isolate which component
  drives the miscalibration per family. The C++
  side reads but does not write them; the columns
  are nullable so the pre-emptive schema change
  stays additive.

### C++ API additions

- `ts_tessera_db_l5_weight::requant_budget_bits`
  (default `-1` = NULL). Mirrors the new schema
  column.
- `ts_tessera_db_list_l5_weights(db, model_hash,
  family_or_empty, &out)`. Typed reader that
  returns one entry per `(model, family)` row,
  ordered by `hit_rate DESC`. Empty `family` means
  "all families for this model_hash". Mirrors
  `ts_tessera_db_list_converged_for_model`.
- `ts_tessera_db_l5_outcome_stats_for(db,
  model_hash, family_or_empty, &out)`. Per-(model,
  family) `hit_rate` / `n_rows` aggregate over
  `l5_outcome`. The C++ aggregation is built from
  two `COUNT(*) FILTER (WHERE ...)` calls because
  the DuckDB amalgamation in this build does not
  load the `core_functions` extension (so `SUM` /
  `AVG` are unavailable); `hit_rate` is the C++
  ratio of the two counts.
- `ts_tessera_db_test_insert_l5_outcome(db, row)`
  (test-only). The `l5_outcome` table is normally
  Python-written; this C++ helper lets the
  converged-fast test populate it without
  round-tripping through Python. The 3 per-tensor
  component columns are emitted as `NULL` when
  passed 0 (the C++ test path is round-trip only).

### Dispatch wiring

- `ts_dispatch_db` gains a `l5_weight_map`
  (`unordered_map<family, entry>`). Populated at
  `ts_dispatch_db_open` via
  `ts_tessera_db_list_l5_weights`; the dispatch
  logs the loaded entries when `verbose` is on.
- `ts_dispatch_family_seed_lookup` now consults
  `l5_weight_map` first: if the family has a row,
  the seed is biased by `hit_rate` (alpha 0.5 +
  0.25*hit_rate, clip 0.7 + 0.20*hit_rate when
  `hit_rate > 0.5`; base 0.5/0.7 otherwise). The
  legacy `ga_results` lookup is the fallback for
  families with no `l5_weight` row.
- `ts_dispatch_run_l5_loop` gains a converged-fast
  early-exit at the top of each generation
  (`gen >= 1`): if `l5_outcome` has rows for the
  model with `hit_rate > 0.95`, the loop logs +
  breaks before the L2 measurement. The first
  generation always runs (to seed `l4_outcome`
  rows); the early-exit fires from the second
  onward. The report JSON carries a
  `converged_fast=true` marker with the `hit_rate`
  and `n_rows` the dispatch saw.

### Tests

- `test_quantize_db`: round-trip for
  `requant_budget_bits` (NULL via `-1` sentinel,
  real value, zero edge case, untouched row stays
  NULL), `ts_tessera_db_list_l5_weights` (empty
  family = all, non-empty = single, unknown
  model/family = empty, ordering by `hit_rate
  DESC`), `ts_tessera_db_l5_outcome_stats_for`
  (empty case).
- `test_l5_dispatch`: end-to-end warm-start
  coverage (pre-populate `l5_weights` for both
  fixture families, re-run the dispatch with
  `--tessera-db`, verify the dispatch completed and
  the warm-start path didn't break the pipeline);
  end-to-end converged-fast coverage (pre-populate
  10 `l5_outcome` rows with `hit_rate=1.0`, re-run
  the dispatch with `l5_max_generations=4`, verify
  the loop produced exactly one
  `converged_fast=true` marker and broke early).
