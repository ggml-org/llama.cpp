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
| **`tensor_stats`** | 1 per (tensor, model_role) per model | C++ (GA-prep) + Python (cal) | **The cross-pipeline feature table.** PRIMARY KEY (model_hash, model_role, name) + ON CONFLICT DO UPDATE = upsert target. Phase 16: `model_role` is one of `trunk` / `dflash` / `dspark` / `mtp_nextn` / `shared_embd`; default `'trunk'` preserves the pre-Phase-16 contract. C++ writes kurtosis / eff_rank / dtype; Python writes rms / mean_abs / tail_ratio / `recommended_action`. `source` records which pipeline last wrote the row. `recommended_action` is the per-tensor verdict the calibration pipeline derives from `l5_weights` via the `l5_action` rules (one of `protect` / `requant_up` / `requant_down` / `monitor` / `noop`); the C++ side just carries the column through the upsert with no logic. |
| **`l3_outlier_summary`** | 1 per (tensor, model_role, sidecar_label) | Python (analytics) | L3 outlier rate per tensor. (model_hash, model_role, name) joins to tensor_stats. |
| **`l4_probe_summary`** | 1 per (tensor, model_role) | Python (analytics) | L4 E2E probe (mse, perplexity, top1_mismatch). |
| **`l4_plan_outcome`** | 1 per (tensor, model_role, iteration, plan_id) | C++ (adaptive_requantize loop) | **The feedback-loop audit trail.** The per-iteration L4 measurement AFTER a requant plan was applied, with the before/after split. The C++ `ts_dispatch_run_l5_loop` writes one row per (tensor, gen) via `ts_quantize_db_append_l4_outcome`. |
| **`l5_plan_summary`** | 1 per (tensor, model_role, iteration, plan_id) | Python (l5_orchestrator) | L5 requant plan (sensitivity_score, recommended_alpha, recommended_clip). |
| **`l5_outcome`** | 1 per (tensor, model_role, iteration, plan_id) | Python (`tools/tessera/l5_outcome.py`) | **The feedback-loop verdict.** Computed by `l5_outcome.py` from a join of `l5_plan_summary` and `l4_plan_outcome`. `plan_accepted` is True if `delta_mse < accept_threshold`. `residual` is the per-(model, family) linear-fit residual of delta_mse on sensitivity_score. Phase 14: also carries the per-tensor components `imatrix_magnitude`, `gradient_proxy`, `layer_position_prior` (all nullable) so a future retune can fit a 3-coefficient model. |
| **`l5_weights`** | 1 per (model, model_role, family) | Python (`tools/tessera/l5_retune.py`) | **The feedback-loop consumer.** Per-(model, model_role, family) retuned `(w_imatrix, w_gradient, w_layer)` on the simplex. Computed by `l5_retune.py` from a per-(model, role, family) closed-form OLS of `delta_mse` on `sensitivity_score` and projected to the simplex. The orchestrator's next generation reads this table via `--retune-from-db`, closing the loop. Phase 14: also carries `requant_budget_bits` (nullable) — the dispatch-side budget the retune recommends for the next requant pass. |
| **`per_layer_error_summary`** | 1 per tensor | Python (analytics) | L1/L1.5 sidecar epsilon. Not part of the Phase 16 migration. |
| **`tensor_stats`** | 1 per tensor per model | C++ (GA-prep) + Python (cal) | **The cross-pipeline feature table.** PRIMARY KEY (model_hash, name) + ON CONFLICT DO UPDATE = upsert target. C++ writes kurtosis / eff_rank / dtype; Python writes rms / mean_abs / tail_ratio / `recommended_action`. `source` records which pipeline last wrote the row. `recommended_action` is the per-tensor verdict the calibration pipeline derives from `l5_weights` via the `l5_action` rules (one of `protect` / `requant_up` / `requant_down` / `monitor` / `noop`); the C++ side just carries the column through the upsert with no logic. |
| **`l3_outlier_summary`** | 1 per tensor per sidecar label | Python (analytics) | L3 outlier rate per tensor. model_hash + name joins to tensor_stats. |
| **`l4_probe_summary`** | 1 per tensor | Python (analytics) | L4 E2E probe (mse, perplexity, top1_mismatch). |
| **`l4_plan_outcome`** | 1 per (tensor, iteration, plan_id) | C++ (adaptive_requantize loop) | **The feedback-loop audit trail.** The per-iteration L4 measurement AFTER a requant plan was applied, with the before/after split. The C++ `ts_dispatch_run_l5_loop` writes one row per (tensor, gen) via `ts_quantize_db_append_l4_outcome`. |
| **`l5_plan_summary`** | 1 per (tensor, iteration, plan_id) | Python (l5_orchestrator) | L5 requant plan (sensitivity_score, recommended_alpha, recommended_clip). |
| **`l5_outcome`** | 1 per (tensor, iteration, plan_id) | Python (`tools/tessera/l5_outcome.py`) | **The feedback-loop verdict.** Computed by `l5_outcome.py` from a join of `l5_plan_summary` and `l4_plan_outcome`. `plan_accepted` is True if `delta_mse < accept_threshold`. `residual` is the per-(model, family) linear-fit residual of delta_mse on sensitivity_score (a running measure of how well the orchestrator's sensitivity scoring predicts the actual error delta). Phase 14: also carries the per-tensor components `imatrix_magnitude`, `gradient_proxy`, `layer_position_prior` (all nullable) so a future retune can fit a 3-coefficient model. |
| **`l5_weights`** | 1 per (model, model_role, family) | Python (`tools/tessera/l5_retune.py`) | **The feedback-loop consumer.** Per-(model, model_role, family) retuned `(w_imatrix, w_gradient, w_layer)` on the simplex. Computed by `l5_retune.py` from a per-(model, model_role, family) closed-form OLS of `delta_mse` on `sensitivity_score` and projected to the simplex. The orchestrator's next generation reads this table via `--retune-from-db --model-role R`, closing the loop. Phase 16: the `model_role` dimension lets the same family in different architectural roles (trunk / dflash / dspark / mtp_nextn / shared_embd) get independent retune verdicts. Phase 15: also carries `top_fraction` (nullable) — the per-family requant aggressiveness recommendation. Phase 14: also carries `requant_budget_bits` (nullable) — the dispatch-side budget the retune recommends for the next requant pass. |
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
PRIMARY KEY `(model_hash, model_role, family)` (Phase 16;
the legacy 2-tuple `(model_hash, family)` PK is supported
via the `_l5_weights_pk_shape` fallback). The
orchestrator's next generation reads the table back via
`--retune-from-db --model-role R` and uses the
n_samples-weighted average across families as the starting
`(w_imatrix, w_gradient, w_layer)` tuple. This is the
closed-loop optimization the feedback loop was designed
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
| Python cal | `tools/tessera/calibration_to_tensor_stats.py` | Reads the per-channel observer parquet, reduces to per-tensor summary, upserts into `tensor_stats` (source = `py_cal`). Phase 13 also reads `l5_weights` + `l5_outcome` and writes `recommended_action` per tensor via the `l5_action` rules. |
| Python l5_action | `tools/tessera/l5_action.py` | The `recommended_action` derivation rules (Phase 13). Single source of truth: per-tensor `(slope, hit_rate, delta_mse, plan_accepted)` -> one of `protect` / `requant_up` / `requant_down` / `monitor` / `noop`. 8 unit tests. |
| Python l3_outlier | `tools/tessera/l3_outlier_report.py` | The L3 outlier report. `--tessera-db` is the fast path: reads per-tensor summary from `tensor_stats` and produces a conservative 0/1 outlier count tagged `source='tensor_stats_estimate'`. The dequant-sidecar path remains the default. |
| Python tests | `tools/tessera/test_tessera_db_buffer.py`, `tools/tessera/test_tessera_db.py`, `tools/tessera/test_l5_outcome.py`, `tools/tessera/test_calibration_to_tensor_stats.py`, `tools/tessera/test_l3_outlier_fast_path.py`, `tools/tessera/test_l5_retune.py`, `tools/tessera/test_l5_action.py` | 8 + 10 + 5 + 8 + 4 + 15 + 8 cases. |
| C++ tests | `tools/quantize/tessera/test_db_buffer.cpp`, `tools/quantize/tessera/test_quantize_db.cpp`, `tools/quantize/tessera/test_quantize_db_e2e.cpp` | 8 + 7 + n cases (Phase 13 adds a recommended_action round-trip). |

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
- **Phase 12 (commit `9e2778e01`):** feedback loop closes.
  The retune (`l5_retune.py`) writes per-(model, family)
  retuned `(w_imatrix, w_gradient, w_layer)` to the new
  `l5_weights` table; the orchestrator's
  `--retune-from-db` reads it back. C++ side: schema
  addition + `ts_tessera_db_upsert_l5_weight` helper
  mirrored on the Python `TesseraDB.insert_l5_weights`
  (direct `INSERT ... ON CONFLICT DO UPDATE`, like
  `tensor_stats`). 15 Python tests + 1 new C++ test
  case.
- **Phase 13 (this commit):** calibration side consumes
  the orchestrator's feedback. Three pieces:
  1. `tensor_stats` gains `recommended_action TEXT`
     (additive; C++ struct + CREATE TABLE + the
     `ts_tessera_db_upsert_tensor_stat` upsert all carry
     the column through). The C++ side does not derive
     the value; it just persists whatever the Python
     calibration side wrote.
  2. New `tools/tessera/l5_action.py` module is the
     single source of truth for the rules:
     `(miscalibration_score, hit_rate, delta_mse,
     plan_accepted)` -> one of `protect` / `requant_up` /
     `requant_down` / `monitor` / `noop`. The thresholds
     (0.5, -0.2, 0.001, 0.9) are KNOBs documented at the
     top of the module. 8 unit tests cover each branch
     + the priority order + the noop default.
  3. `calibration_to_tensor_stats.py` reads
     `l5_weights` + `l5_outcome` (the most recent per
     `(model_hash, name)`, picked by `ROW_NUMBER() OVER
     (PARTITION BY name ORDER BY iteration DESC,
     plan_id DESC)`) and upserts `recommended_action`
     into `tensor_stats` per tensor. 5 new test cases
     cover the protect / no-l5-weights / most-recent-wins
     / requant_up / re-upsert-stability contracts.
  4. `calibration_rollup.py` gains `--l5-outcome
     LABEL=PATH` (a `tessera.duckdb` path; the prefix is
     always `l5.*` regardless of the user-supplied label)
     + `--model-hash HASH` (required when `--l5-outcome`
     is set). The rollup joins `l5_outcome` (most-recent
     per name) + `l5_weights` (per family) and adds the
     `l5.*` columns + the derived `recommended_action` to
     the per-tensor rollup table. 4 new test cases
     cover the join, the most-recent-wins contract, the
     model-hash requirement, and the empty-DB fallback.
  Schema changes are additive: `CREATE TABLE IF NOT
  EXISTS` keeps existing DBs compatible; the new column
  is NULL for rows written before this commit. The
  C++ `ts_tessera_db_upsert_tensor_stat` upsert uses
  straight overwrite (not COALESCE) for the new column;
  the Python side is the authoritative writer.
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
- **Phase 15 (this commit):** feedback loop extends. The
  retune is now a **3-coefficient OLS** (per-tensor
  `imatrix_magnitude`, `gradient_proxy`,
  `layer_position_prior` components) instead of a
  2-coefficient OLS on the combined `sensitivity_score`.
  The orchestrator's `SensitivityScorer.score()` already
  emits the per-tensor components on the
  `l5_plan_summary` row; Phase 15 threads them through
  to `l5_outcome` (additive, nullable; the producer
  populates the columns from the scorer, the consumer
  reads them to fit the 3-coefficient OLS). The
  2-coefficient OLS on the combined `sensitivity_score`
  is the **fallback** when the components are NULL
  (pre-Phase-15 rows, the C++ side before the schema
  migration). Schema additions (Python-side, additive via
  `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` so the
  changes are forward-compatible with the C++ side's
  pre-Phase-15 schema):
  * `l5_plan_summary` gains `imatrix_magnitude`,
    `gradient_proxy`, `layer_position_prior`. The
    orchestrator's `write_history` (NDJSON) populates
    them from `RequantAction.imatrix_magnitude` etc.
  * `l5_outcome` gains the same three columns (populated
    by `l5_outcome.py` at read time via the
    `l5_plan_summary` join).
  * `l5_weights` gains `top_fraction DOUBLE` (nullable).
  New retune features:
  * **3-coefficient OLS** via numpy `lstsq` on
    `[1, im, grad, layer]` with sqrt-weighted rows. The
    closed-form `(X^T W X)^-1 X^T W y` would be brittle
    on near-singular inputs; lstsq is the standard tool
    for small weighted least squares.
  * **Sample weights** derived from
    `1 / (1 + in_sample_loss * 100) * sqrt(n_samples /
    max_n_samples)`. The in_sample_loss term damps rows
    whose post-fit loss is high; the n_samples term
    rewards rows with more data (sub-linear).
  * **Per-family `top_fraction` recommendation**:
    `top_fraction = base * (1 + tanh(2*slope) *
    (1 - hit_rate))`. The orchestrator's
    `RequantPlanner` consumes it via the
    `--per-family-top-fraction` flag and overrides the
    uniform `--top-fraction` for the families the retune
    has flagged.
  * **Cross-model retune** (`--retune-cross-model`):
    writes a per-family aggregate row with
    `model_hash = "*"` (n_samples-weighted mean across
    all models). The orchestrator's
    `--retune-from-db` falls back to the cross-model row
    for any family the per-model lookup missed
    (`--retune-cross-model-fallback`). The
    `read_l5_weights(cross_model_fallback=True)` path
    handles the union.
  * **EMA-aware retune**: optional join with
    `l5_plan_ema` (the per-iteration EMA of the
    sensitivity score) replaces the per-iteration
    `sensitivity_score` on the OLS. The EMA is stable
    across iterations; the per-iteration score is noisy.
    The retune fits on the EMA when the table is
    present, falls back to the per-iteration score
    otherwise. The join key is `(model_hash, name,
    iteration, plan_id)`.
  Algorithm tags written to `l5_weights.retune_source`:
  * `ols_3coef_v1`: the 3-coefficient OLS path (the
    production path when components are present).
  * `ols_slope_v1`: the 2-coefficient OLS fallback (the
    Phase 12 path; preserved for backward-compat).
  * `ols_3coef_crossmodel_v1`: the cross-model
    aggregate row.
  32 retune tests + 7 outcome tests + 11 db tests + 8
  buffer tests + 7 dataframe tests = 65 tests pass.
- **Phase 16 (this commit):** the unified Gemma4 12B +
  dspark + dflash + MTP arch disambiguates tensors with
  the same name in the trunk and the drafters. The
  shared `tessera.duckdb` gains a `model_role` column on
  7 of the cross-pipeline tables:
  * `tensor_stats`
  * `l3_outlier_summary`
  * `l4_probe_summary`
  * `l5_plan_summary`
  * `l4_plan_outcome`
  * `l5_outcome`
  * `l5_weights`
  The PKs are extended to include `model_role`. The
  enum:
  * `trunk` — the main model's transformer blocks
    (gemma4 `n_layer`). The default; the contract
    `model_role=""` -> `'trunk'` preserves
    pre-Phase-16 callers.
  * `dflash` — the dflash drafter's per-block layers
    (DFlash's `LLM_TENSOR_FC`, `D2T`, etc.).
  * `dspark` — the dspark markov/conf heads
    (`LLM_TENSOR_DSPARK_*`).
  * `mtp_nextn` — the MTP/nextn projections
    (`LLM_TENSOR_NEXTN_*`).
  * `shared_embd` — the shared `tok_embd` + `output`
    (`lm_head`); only one row per model. The
    `name` for `shared_embd` rows is `token_embd` /
    `output` (matching the trunk's existing names).
  Schema:
  ```
  tensor_stats:        PK (model_hash, model_role, name)
  l3_outlier_summary:  PK (model_hash, model_role, name, sidecar_label)
  l4_probe_summary:    PK (model_hash, model_role, name)
  l4_plan_outcome:     PK (model_hash, model_role, name, iteration, plan_id)
  l5_plan_summary:     PK (model_hash, model_role, name, iteration, plan_id)
  l5_outcome:          PK (model_hash, model_role, name, iteration, plan_id)
  l5_weights:          PK (model_hash, model_role, family)
  ```
  The `name` column for `dflash` / `dspark` /
  `mtp_nextn` rows is the **drafter-local** tensor
  name (NOT the global name with `dflash.` prefix);
  the consumer joins via `(model_hash, model_role,
  name)`.
  Implementation:
  * C++ side: 4 row structs gain a `model_role`
    `std::string` field (`ts_tessera_db_tensor_stat`,
    `ts_tessera_db_l4_outcome`,
    `ts_tessera_db_l5_outcome_row`,
    `ts_tessera_db_l5_weight` +
    `ts_tessera_db_l5_weight_list_entry`). The 4
    insert / upsert / append / test helpers carry
    the new column. `ts_tessera_db_list_l5_weights`
    reads it from the new column position.
  * Python side: the 6 `L*_COLS` tuples
    (`TENSOR_STATS_COLS`, `L3_OUTLIER_COLS`,
    `L4_PROBE_COLS`, `L4_PLAN_OUTCOME_COLS`,
    `L5_PLAN_COLS`, `L5_OUTCOME_COLS`,
    `L5_WEIGHTS_COLS`) include `model_role`; the 6
    `insert_*` helpers accept a `model_role` key in
    the row dict (default `'trunk'`).
  * `per_layer_error_summary` is unchanged (not in
    the Phase 16 list; it does not need
    disambiguation).
  Migration: two paths, both idempotent.
  * **C++ side**: `ts_tessera_db_migrate_model_role`
    runs the standard DuckDB PK-rebuild dance on each
    of the 7 affected tables
    (`CREATE TABLE <name>__p16_new (new schema)`
    -> `INSERT INTO <name>__p16_new SELECT *,
    'trunk' AS model_role FROM <name>`
    -> `DROP TABLE <name>`
    -> `ALTER TABLE <name>__p16_new RENAME TO <name>`).
    DuckDB does not support `ALTER TABLE ... DROP
    CONSTRAINT` in older versions, so the rebuild is
    the only way to change a PRIMARY KEY. The
    function is called from `ts_tessera_db_open` on
    every open. The per-table idempotency check
    (`information_schema.columns WHERE
    column_name='model_role'`) short-circuits on a
    fresh DB (the CREATE TABLE has the column) or a
    re-opened migrated DB.
  * **Python side**: `tools/tessera/
    migrate_model_role.py::migrate(db_path)`. The
    same PK-rebuild dance, run from Python
    (DuckDB's Python client). Use this when opening
    a pre-Phase-16 DB before the C++ side has
    touched it.
  Both migrations preserve the data: the INSERT
  ... SELECT explicitly lists every column (kurtosis,
  rms, sensitivity_score, plan_accepted, ...) so the
  per-tensor values are carried through. The new
  `model_role` column is backfilled with `'trunk'`
  (the single-model-run default; the disambiguation
  only matters for new rows written by the dflash /
  dspark / mtp_nextn / shared_embd producers).
  Tests:
  * C++: `test-tessera-quantize-db` round-trip on
    the 4 row structs (default-empty -> 'trunk',
    explicit 'dflash' / 'mtp_nextn' on the new PK,
    reader echoes model_role) + the migration
    round-trip on a raw-SQL-seeded pre-Phase-16 DB.
  * Python: 7 new `test_tessera_db.py` cases
    (round-trip on the 6 insert helpers) +
    `test_migrate_model_role.py` (4 cases: pre-16
    migration, idempotency, fresh-DB no-op, mixed
    model_role coexistence after migration).
  Schema + struct diff: `git diff --stat main..HEAD`
  on the branch. 21 Python tests + 3 C++ tests pass
  (test-tessera-quantize-db,
  test-tessera-db-buffer,
  test-tessera-quantize-db-e2e).

- **Phase 16 (this commit):** calibration memory-bound /
  spatial-temporal path.  Five categories of
  optimisations land here, each with its own commit:
  mmap streaming I/O, chunked processing, peak-RSS
  budget + graceful abort, spatial occupancy (per-
  layer round-robin across components), and temporal
  pipeline (double-buffered I/O + compute overlap).
  CLI flags: `--chunk-rows`, `--peak-rss-budget-gb`,
  `--spatial-occupancy`, `--temporal-pipeline-depth`.
  41 unit tests in `test_calibration_memory.py` +
  3 end-to-end tests in
  `test_per_tensor_calibrate_memory.py` pass.  The
  end-to-end test runs a synthetic 200-tensor
  calibration with `--peak-rss-budget-gb 1` and
  asserts the peak RSS stays under 1.5 GB.  See
  `docs/tessera-unified-db.md` Phase 16 section for
  the per-category docs.

## Open follow-ups (after this commit)

The unified-DB feedback loop is now feature-complete on
all three sides: the orchestrator side (3-coefficient OLS
retune + cross-model aggregate + per-family top_fraction +
EMA-aware + `--retune-from-db` closed-loop wiring) is in
production, the calibration side (`calibration_rollup
--l5-outcome` + `recommended_action` on `tensor_stats` +
the `l5_action` rules module) is in production, and the
C++ quantization side (`l5_weights`-driven GA warm-start
+ converged-fast early-exit) is in production. The
remaining items are about extending the loop further, not
about adding the next leg.

- **GA-prep walk**: read `l5_weights` to bias the GA's
  initialization for the families the retune flagged as
  miscalibrated. This is the C++ consumer side; the
  dispatch would call `ts_tessera_db_list_l5_weights` at
  `ts_dispatch_db_open` time and thread the per-family
  weights into the GA's seed-generation. Phase 15's
  per-family `top_fraction` is the natural input.
- **Producer-side EMA write**: the
  `l5_orchestrator.py:OrchestratorLoop` does not yet
  write to `l5_plan_ema`. Phase 15's retune consumes the
  EMA via a left-join; the producer side (the
  orchestrator's run loop) is the missing leg. A small
  follow-up commit adds the EMA write in
  `OrchestratorLoop.run` so the EMA path is the
  production default end-to-end.
- **Per-tensor residual decomposition**: the
  `l5_outcome.residual` column is the residual of the
  per-(model, family) 2-coefficient fit of `delta_mse`
  on `sensitivity_score`. Phase 15's 3-coefficient OLS
  can replace this with a per-(model, family, component)
  residual, surfaced as `residual_im`, `residual_grad`,
  `residual_layer` columns on `l5_outcome`. The
  diagnostics surface would change (one number per
  component per family instead of one number per
  family).

## Phase 16: retune role propagation (this branch)

The unified multi-component architecture (gemma4_12B +
dflash encoder + dspark drafter + MTP-NextN head +
shared embedding / output projection) puts the same
tensor family in different architectural roles: the
trunk's `attn_q` and the dflash encoder's `attn_q` are
the same family in the l5_outcome row's `family`
column, but they calibrate independently (the dflash
encoder consumes trunk hidden states; its residual
surface is very different from the trunk's). The Phase
15 retune's `(model_hash, family)` PK would conflate
them; the per-(model, family) OLS would see the union
of the trunk's and the dflash encoder's rows and
produce a single (w_imatrix, w_gradient, w_layer)
verdict that miscalibrates both.

Phase 16 introduces a `model_role` dimension on the
retune. The `l5_weights` PK is now `(model_hash,
model_role, family)`. The same family in different
roles gets independent (w_imatrix, w_gradient, w_layer)
recommendations. The orchestrator's
`--retune-from-db --model-role dflash` looks up the
dflash-specific row; the cross-model fallback is also
role-aware; the legacy `(model_hash, family)` lookup is
preserved as a fallback for new roles without their own
retune rows.

**Schema additions (additive via `ALTER TABLE ... ADD
COLUMN IF NOT EXISTS` so legacy DBs work; the
`evolve/unified-schema` worker owns the
`tessera-quantize-db.cpp` CREATE TABLE migration that
moves the l5_weights PK from 2-tuple to 3-tuple):

* `l5_plan_summary` gains `model_role TEXT DEFAULT
  'trunk'`. The orchestrator's `write_history`
  (NDJSON) populates the column from
  `RequantAction.model_role`.
* `l5_outcome` gains `model_role TEXT DEFAULT 'trunk'`.
  Populated by `l5_outcome.py` at read time via the
  l5_plan_summary join.
* `l5_weights` gains `model_role TEXT DEFAULT 'trunk'`.
  The `l5_weights` PK is now `(model_hash, model_role,
  family)` so the trunk's `attn_q` and the dflash
  encoder's `attn_q` get independent retune verdicts.

**New retune features:**

* **Per-(model, model_role, family) partition**. The
  retune's `partition_by` is now `["model_hash",
  "model_role", "family"]`. The 3-coefficient OLS is
  fit per (model, model_role, family); the verdict's
  FamilyWeights carries the role. The retune's
  write-back DELETE keys on `(model_hash, model_role)`
  so other roles' l5_weights rows are preserved.
* **Cross-model, per-role aggregate**. The
  `--retune-cross-model` write path is now
  per-(model_role, family), not per-family. The
  cross-model row's `model_role` is the same string as
  the per-model rows it aggregates.
* **Role-aware orchestrator lookup**. The orchestrator's
  `--retune-from-db --model-role R` looks up
  per-(model_hash, R, family) rows; the
  `--retune-cross-model-fallback` falls back to
  per-(R, family) cross-model rows; the legacy
  per-(model_hash, *, family) trunk row is the final
  fallback for new roles. The 3-tier lookup is
  implemented in `l5_orchestrator.py:main`.
* **Role-aware top_fraction consumer**. The
  `RequantPlanner`'s per-family top_fraction map is
  loaded via `read_per_family_top_fraction(model_role=R)`;
  the dflash encoder's attn_q top_fraction is
  independent of the trunk's.
* **Role plumbed through SensitivityScorer /
  RequantPlanner / RequantAction**. The
  `SensitivityScorer` and `RequantPlanner` each carry a
  `model_role`; every per-tensor `RequantAction` tags
  the row with the role so the l5_plan_summary writer
  can put the role on the l5_outcome side. The
  `l5_metrics.combine` and `l5_metrics.decompose`
  helpers accept the role as an optional pass-through
  parameter (the math is unchanged).

**Backward compatibility:**

* Pre-Phase-16 l5_outcome rows (no `model_role` column)
  are backfilled with a uniform `'trunk'` string in
  the verdict projection; the partition still produces
  a single group per (model, family) tagged with
  `model_role='trunk'`.
* Pre-Phase-16 l5_weights (2-tuple PK `(model_hash,
  family)`) is detected via the
  `_l5_weights_pk_shape` helper; the upsert falls back
  to a 2-tuple `ON CONFLICT` target. The role is still
  written to the row (the column was added by
  `_ensure_l5_weights_columns`); the legacy PK is
  upgraded to the 3-tuple by the schema worker's
  migration.
* The default `model_role` is `'trunk'`. The
  pre-Phase-16 callers that did not pass a role get
  the legacy `(model_hash, family)` retune verdict,
  tagged with `model_role='trunk'`.

**Algorithm tags written to `l5_weights.retune_source`**
(unchanged from Phase 15):
* `ols_3coef_v1`: the 3-coefficient OLS path (the
  production path when components are present).
* `ols_slope_v1`: the 2-coefficient OLS fallback (the
  Phase 12 path; preserved for backward-compat).
* `ols_3coef_crossmodel_v1`: the cross-model
  aggregate row (now per-(model_role, family)).

42 retune tests + 12 outcome tests + 19 db tests + 7
dataframe tests = 80 tests pass.
## Phase 16 (this commit): calibration memory-bound / spatial-temporal path

The unified gemma4_12B + dspark + dflash + MTP single-GGUF
calibration processes 4000+ tensors, with FFN gate/up
tensors as large as 16384x4096 = 256 MB F32. Loading
all of them at once (or even iterating them with full
retention) blows past 64 GB of RAM and the OS kills the
process. Phase 16 ships the memory-bound / spatial-
temporal pipeline that bounds the per-tensor RSS and
overlaps I/O with compute, so the 12B unified
calibration runs in bounded memory in hours rather than
days (or OOMing).

Five categories of optimisations land in this commit.
They are independent; each ships with its own
implementation, tests, and (where applicable) docs.

### 16.1: Streaming I/O (mmap instead of read)

The legacy `tools/tessera/per_tensor_calibrate.py`
`load_layer()` read the whole `.npz` file into RAM via
`np.load(path)`. The new path opens the bundle with
`mmap_mode="r"` and hands the consumer mmap-backed
views of the heavy keys (weight, activations,
observer moments). Peak RSS for a single tensor is
bounded to `max(weight, activations, observer)`
rather than `sum(all_tensors_in_turn)`.

The mmap utility is in `tools/tessera/calibration_memory.py`:
* `mmap_tensor(npz_path, key, dtype)` - single-tensor
  view; closes the `np.load` handle before returning
  so the OS keeps the zip mmap alive as long as the
  view is held.
* `mmap_layer(npz_path, keys=...)` - context manager;
  opens a single `np.load` handle for the legacy key
  set (weight, train_activations, heldout_activations,
  in_sum2, counts, name, family) and yields the dict.

### 16.2: Chunked processing (for tensors too big to fit)

The per-tensor training's working set is dominated by
`out_dim * in_dim * 4` F32 intermediates (the LRQ
forward produces 4 of them; the FLRQ sketch produces
1 + the SVD on `(K, n_projections * target_rank)`).
For a 12B FFN gate at 16384x4096 the working set is
~1 GB. The chunked path splits the out_dim axis into
row-chunks; each chunk processes a
`(chunk_rows, in_dim)` sub-matrix and the OS reclaims
the pages before the next chunk is read.

The chunked utility is also in
`calibration_memory.py`:
* `chunked_iter(n_rows, chunk_rows)` - yields
  `ChunkSpec` per row-chunk.
* `chunked_process(weight, activations, chunk_rows,
  compute)` - the row-chunked callback dispatcher;
  per-chunk peak RSS is `max(W_chunk, intermediates)`.

The per-tensor training is wired to the chunked path
in `per_tensor_calibrate.py`:
* `train_lrq_chunked` - chunked LRQ training. Uses
  `ChunkedAdam` (per-chunk U state, full V state) and
  accumulates `d_v` across chunks. The per-tensor
  result is bit-equivalent to `train_lrq` (modulo the
  float32 order-of-operations in the per-chunk matmul).
* `flrq_sketch_chunked` - row-chunked R1-Sketch. Each
  chunk produces a `(chunk_rows, total_width)` slice
  of `Y = W @ Omega`; the slices are concatenated
  before the SVD.

CLI flag: `--chunk-rows N` (default 4096; 0 or
negative disables chunking).

### 16.3: Peak-RSS budget with graceful abort

The legacy calibration had no peak-RSS cap; on a 12B
calibration it would OOM and the OS would kill the
process without a useful error message. Phase 16 ships
a `ResidencyTracker` that reads the current process RSS
(Linux: `/proc/self/status`; macOS/Windows: psutil) and
aborts with a clear error when the budget is exceeded.

The tracker is in
`tools/tessera/calibration_residency.py`:
* `read_rss_bytes()` - cross-platform RSS reader.
* `ResidencyTracker(budget_bytes, abort_on_exceed)` -
  the check method. Records `peak_bytes`, `n_checks`,
  `n_violations` for the final report.
* `residency_managed(budget_bytes)` - context manager
  for a block of per-tensor work.

The tracker is created at the top of `main()` in
`per_tensor_calibrate.py` and checked at the top of
each per-tensor iteration. The check is **advisory**:
it catches sustained over-budget states, not micro-
spikes from numpy's internal allocations. Operators
should choose the budget with a 1.5-2x safety margin
over the expected per-tensor working set so transient
over-runs don't false-positive.

CLI flag: `--peak-rss-budget-gb N` (default 32; 0
disables the check). The default 32 GB fits a 12B
unified run on a 64 GB host with the chunked path.

### 16.4: Spatial occupancy (interleave components for cache locality)

The per-tensor observer moments (per-input-channel
scales) are similar across components (they all share
`tok_embd` + `output`). Round-robining the per-
component tensors at the layer level keeps the cache
hot on the shared moments.

The interleave utility is in `calibration_memory.py`:
* `extract_layer_index(tensor_name)` - parses the
  layer index from a tensor name (e.g. `blk.0.attn_q`
  -> 0, `dflash.encoder.fc.0` -> 0, `token_embd.weight`
  -> -1 for "no layer").
* `interleave_components(components, roles)` - yields
  `(role, tensor_name)` pairs in spatial-interleaved
  order: round-robin per-component tensors at the
  layer level, with the per-role order preserved.

The per-component shell-out in `per_tensor_calibrate.py`
(the in-process multi-component driver; Worker 2's
`unified_calibrate.py` is the subprocess alternative)
uses the helper. The single-component path doesn't
need this optimisation.

CLI flag: `--spatial-occupancy {sequential,interleaved}`
(default `interleaved`). The per-tensor result is
identical (the spatial order is a pure refactor); the
wall-time and cache-hit rate differ.

### 16.5: Temporal occupancy (pipeline I/O with compute)

The legacy per-tensor loop is sequential:
1. mmap the weight (I/O, blocking)
2. mmap the activations (I/O, blocking)
3. compute the per-tensor qtype (CPU, blocking)
4. write the per-tensor policy entry (I/O, blocking)

The temporal pipeline overlaps steps 1+2 of the next
tensor with steps 3+4 of the current tensor, so the
next tensor's mmap is in flight while the current
tensor is computing. This is the standard double-
buffered producer/consumer pattern.

The pipeline utility is in `calibration_memory.py`:
* `CalibPipeline(layer_paths, depth, keys)` - the
  double-buffered pipeline. A daemon thread mmaps one
  tensor ahead of the consumer; the consumer reads the
  current tensor's mmap data, builds a `Layer` via
  `_layer_from_mmap_data`, and runs the per-tensor
  training. The depth is configurable; 1 is the legacy
  single-thread path; 2 (default) is double-buffered;
  3+ keeps more tensors in flight on slow I/O at the
  cost of more peak RSS.

CLI flag: `--temporal-pipeline-depth N` (default 2;
1 = legacy single-thread).

### 16.6: Performance and acceptance

The 12B unified calibration should:
* Run in bounded memory (default 32 GB; configurable
  down to 8 GB on a tight host)
* Finish in a few hours (the previous single-stream
  loop would take days or OOM)
* Use the spatial occupancy to keep the cache hot on
  the shared observer moments
* Use the temporal pipeline to overlap I/O with
  compute

The end-to-end test
(`tools/tessera/test_per_tensor_calibrate_memory.py`)
runs a synthetic 200-tensor calibration with
`--peak-rss-budget-gb 1` and asserts the peak RSS
stays under 1.5 GB.  Observed on a 2024-era CPU: 200
tensors at 1024x256 (the small variant for fast CI;
set `TESSERA_E2E_FULL=1` for the 12B variant)
finishes in ~2 s with peak RSS 0.60 GB.  The test
also pins the per-tensor result's independence of
the spatial-occupancy choice.

Tests:
* `test_calibration_memory.py` - 41 unit tests for
  the mmap / chunked / spatial / temporal /
  residency utilities.
* `test_per_tensor_calibrate_memory.py` - 3 end-to-end
  tests (200-tensor budget-bounded run, sequential
  vs interleaved equivalence, residency flag wiring).

The 12B-shape variant (`TESSERA_E2E_FULL=1`) needs
~30 GB free disk and ~2-3 min wall-time; the small
variant (default) needs ~0.5 GB and ~2-10 s.

### 16.7: Open follow-ups (after this commit)

* **AVX-512 / Metal dispatch for chunked processing**:
  the chunked LRQ / FLRQ paths are pure-numpy; an
  AVX-512 / Metal dispatch on the per-chunk matmul
  would give 2-4x wall-time. The 12B unified
  calibration's per-tensor training is matmul-heavy;
  the dispatch side is the natural extension.
* **Async file I/O on Windows**: the `CalibPipeline`
  producer thread uses `np.load(mmap_mode="r")` which
  is a synchronous read on Windows. An `aiofiles`-
  backed producer would let the Windows path overlap
  I/O with compute the same way the Linux path does.
* **BLC chunked (FLRQ)**: the FLRQ BLC step still
  needs the full weight (it iterates `W - U @ V`);
  the chunked sketch is the only FLRQ chunking Phase
  16 ships. A chunked BLC would cap the FLRQ peak
  to the chunked-sketch peak (~64 MB) instead of the
  full BLC peak (~200 MB).
* **12B-shape E2E test in CI**: the
  `TESSERA_E2E_FULL=1` variant is opt-in because it
  needs ~30 GB free disk. The default small variant
  is fast and validates the property; the 12B variant
  is the production validation.
||||||| 1a5d56ca2

## Phase 16: unified GGUF writer (C++ side)

The unified Gemma4 12B + dspark + dflash + MTP pipeline
(LLM_ARCH_GEMMA4_ASSISTANT) emits a single self-contained GGUF
in one quantization pass. The writer (`ts_unified_writer`,
`tools/quantize/tessera/tessera-unified-writer.{h,cpp}`) takes
4+ per-component source GGUFs (trunk / dflash / dspark /
mtp_nextn / shared_embd) and the per-tensor calibration policy
and produces a `gemma4-assistant` GGUF that the loader
(`src/models/gemma4-assistant.cpp`) can read end-to-end.

### Per-tensor qtype reader (Phase 16 schema)

The pre-Phase-16 `tensor_stats` table had `PRIMARY KEY
(model_hash, name)`. The unified arch's per-block tensors
collide on `name` (the trunk and the dflash drafter both
export `blk.0.attn_q.weight`); the new PK is
`(model_hash, model_role, name)` where `model_role` is one of
`trunk` / `dflash` / `dspark` / `mtp_nextn` / `shared_embd`.

The migration is the standard DuckDB PK-rebuild dance
(`CREATE new -> INSERT FROM old -> DROP old -> RENAME`)
on `tensor_stats`, idempotent via an
`information_schema.columns` check. The writer branch
(`evolve/unified-writer`) only needs the column on
`tensor_stats`; the schema branch (`evolve/unified-schema`)
extends the same dance to the other 6 tables that share the
collision pattern (`l3_outlier_summary`, `l4_probe_summary`,
`l5_plan_summary`, `l4_plan_outcome`, `l5_outcome`,
`l5_weights`). The writer's `ts_tessera_db_upsert_tensor_stat`
honors the new column; an empty `model_role` defaults to
`"trunk"` so the pre-Phase-16 contract (single-component
writes) is preserved.

The new reader is `ts_tessera_db_read_unified_policy(db,
model_hash, role, out, err)`:
* `role` is empty -> all roles for the model
* `role` is non-empty -> just that role
* Rows are returned in `(model_role, name)` order so the
  writer's per-component scan is cache-friendly.

The dispatch's GA-prep walk uses this reader (via the
`tessera_db` CLI flag) to feed the writer with the
calibration-verdict qtype per tensor. When the dispatch
opens a pre-Phase-16 DB (no `model_role` column yet), the
migration runs transparently on the first `ts_tessera_db_open`.

### The writer

`ts_unified_writer` is a C++ class that knows the
gemma4-assistant tensor layout. Per-component routing:

* `trunk`        -> copy as-is. The trunk's `blk.{i}.attn_q.weight`
                    IS the gemma4-assistant MTP-side per-block
                    attn_q (the MTP graph uses the same names).
* `dflash`       -> copy with a `dflash.` prefix to disambiguate
                    from the trunk's identically-named tensors.
* `dspark`       -> copy as-is. The dspark heads
                    (`markov_w1`, `markov_w2`, `conf_proj`) are
                    unique.
* `mtp_nextn`    -> copy as-is. The mtp_nextn tensors are
                    `blk.{i}.nextn.*` with unique suffixes.
* `shared_embd`  -> copy as-is. The shared embeddings
                    (`token_embd`, `output`) are unique.

Tile640 (`GGML_TYPE_TESSERA_T640`) cluster format is preserved
end-to-end: the writer detects the base tensor's tile640 type,
copies the 6+ sub-tensors by data pointer, and emits a
`TESSERA_T640` placeholder for the base (the loader reads the
sub-tensors via `get_tile640_tensor`).

Per-tensor qtype override: the calibration policy's
`(model_role, name) -> dtype` map is consulted for each tensor;
when the policy's dtype differs from the source's qtype, the
destination tensor's type is set to the policy value. A
no-op override (policy says F16, source is F16) is intentionally
not counted as an override (the per-tensor override counter
in the writer's stats reflects actual changes).

Duplicate detection: the writer tracks each destination name
emitted; on a collision (the same dst_name from a later
component), the tensor is skipped rather than calling
`gguf_add_tensor`'s `ABORT` path. The `shared_embd`-vs-`trunk`
case for `token_embd` is the common collision: trunk is
written first, `shared_embd`'s copy is silently skipped (they
should be the same tensor data anyway).

### CLI

```
llama-tessera unified-writer \
    --out <dest.gguf> \
    --arch gemma4-assistant \
    --hparams <hparams.json> \
    --policy <policy.json> \
    --trunk <trunk.gguf> \
    --dflash <dflash.gguf> \
    --dspark <dspark.gguf> \
    --mtp <mtp.gguf> \
    --shared-embd <embd.gguf>
```

At least one `--{component}` flag is required. `--hparams` is
a JSON file with the gemma4 arch's canonical field names
(`n_layer`, `n_embd`, `feed_forward_length`, `attention.head_count`,
`attention.head_count_kv`, `attention.key_length`, `attention.value_length`,
`attention.key_length_swa`, `attention.value_length_swa`,
`attention.sliding_window`, `attention.sliding_window_pattern` (uint8 array),
`attention.layer_norm_rms_epsilon`, `rope.freq_base_swa`,
`nextn_predict_layers`, `vocab_size`, `embedding_length_out`,
`n_kv_shared_layers`).

`--policy` is a sidecar JSON file with the same shape
`unified_calibrate.py` emits: a top-level `tensor_families` array
of `{model_role, name, dtype}` triples. When `--tessera-db` is
also set, the per-`(model_hash, model_role, name)` `tensor_stats`
rows override the sidecar on collision (the DB is the
production data source; the sidecar is a debugging affordance).

### End-to-end example

Synthetic 4-component input (1 trunk + 1 dflash + 1 dspark +
1 mtp_nextn + 1 shared_embd = 17 tensors, ~26 KB output):

```
$ llama-tessera unified-writer \
    --out unified.gguf \
    --arch gemma4-assistant \
    --hparams hparams.json \
    --policy policy.json \
    --trunk trunk.gguf \
    --dflash dflash.gguf \
    --dspark dspark.gguf \
    --mtp mtp.gguf \
    --shared-embd shared.gguf
unified-writer: unified.gguf -> ok
  tensors: trunk=6 dflash=3 dspark=3 mtp_nextn=4 shared_embd=1
  qtype overrides: 2 (per-tensor calibration policy)
  total bytes: 26122
```

The output GGUF's `general.architecture` is `gemma4-assistant`,
the gemma4-specific hparams land (block_count, embedding_length,
attention.key_length_swa, rope.freq_base_swa, ...), the per-tensor
qtype overrides from the policy take effect, and the byte-level
data round-trips through the writer (the writer copies tensors
by data pointer; the source GGUF's data is mmap'd and not
re-read or re-quantized). Tile640 cluster format is preserved
end-to-end (a tile640-encoded source tensor arrives at the
destination with all 6+ sub-tensors intact).

### Tests

`test-unified-writer` (`tools/quantize/tessera/test_unified_writer.cpp`,
95 cases):
1. qtype string round-trip (F16, Q4_K, TESSERA_T640, ...)
2. policy JSON round-trip (load + save + structural compare)
3. per-component qtype reader (`ts_tessera_db_read_unified_policy`)
   with 5 rows, all-roles + per-role + unknown-model +
   unknown-role coverage
4. synthetic 4-component GGUF build (trunk + dflash + dspark +
   mtp_nextn + shared_embd), write to a single gemma4-assistant
   GGUF, reopen, verify arch + hparams + tensor count + 7
   tensor name samples (including the dflash. prefix) +
   byte-identical data round-trip for a non-overridden tensor +
   type override verification for 2 tensors
5. invalid-hparams rejection
6. hparams JSON file round-trip (the CLI's --hparams path)

`test-tessera-quantize-db` extends the existing
`tensor_stats` round-trip with the new `model_role` column:
* Pre-Phase-16 DBs (no `model_role` column) open cleanly via
  the in-place migration
* Post-Phase-16 DBs (column already present) no-op
* `ts_tessera_db_read_unified_policy` returns the correct rows
  for `(model_hash, "")`, `(model_hash, "trunk")`, etc.

### Files

| File | Role |
|---|---|
| `tools/quantize/tessera/tessera-unified-writer.{h,cpp}` | The writer class (`ts_unified_writer`), the per-tensor policy struct, the qtype string<->enum helpers, the policy JSON load/save helpers. |
| `tools/quantize/tessera/tessera-quantize-db.{h,cpp}` | The minimal additive `model_role` migration on `tensor_stats` (the writer branch's scope); the new `ts_tessera_db_read_unified_policy` reader. |
| `tools/quantize/tessera/test_unified_writer.cpp` | Round-trip + CLI test. 95 cases. |
| `tools/quantize/tessera/test_quantize_db.cpp` | Extended with `ts_tessera_db_read_unified_policy` round-trip. |
| `common/common.h` | New `TESSERA_SC_UNIFIED_WRITER` enum value. |
| `common/tessera-args.h` | New `common_tessera_params` fields (`unified_out`, `unified_policy`, `unified_hparams`, `unified_{trunk,dflash,dspark,mtp,shared_embd}`, `unified_arch`). |
| `common/arg.cpp` | The `unified-writer` subcommand table entry + 9 new `--{out,arch,policy,hparams,trunk,dflash,dspark,mtp,shared-embd}` add_opt entries; the `--model-is-required` early-exit list extended to include the new subcommand. |
| `tools/quantize/quantize.cpp` | The `ts_cli_unified_writer` helper and the `TESSERA_SC_UNIFIED_WRITER` dispatch case. |
| `tools/quantize/CMakeLists.txt` | New `tessera-unified-writer.cpp` in the `llama-quantize-impl` source list; new `test-unified-writer` test target. |
