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
| **`tensor_stats`** | 1 per tensor per model | C++ (GA-prep) + Python (cal) | **The cross-pipeline feature table.** PRIMARY KEY (model_hash, name) makes this an upsert target. C++ writes kurtosis / eff_rank / dtype; Python writes rms / mean_abs / tail_ratio. `source` records which pipeline last wrote the row. |
| **`l3_outlier_summary`** | 1 per tensor per sidecar label | Python (analytics) | L3 outlier rate per tensor. model_hash + name joins to tensor_stats. |
| **`l4_probe_summary`** | 1 per tensor | Python (analytics) | L4 E2E probe (mse, perplexity, top1_mismatch). |
| **`l5_plan_summary`** | 1 per (tensor, iteration, plan_id) | Python (analytics) | L5 requant plan sensitivity. |
| **`per_layer_error_summary`** | 1 per tensor | Python (analytics) | L1/L1.5 sidecar epsilon. |

The 5 bolded tables are the additions on top of the existing
6-table schema; see the scout §0 / §6 for the original inventory.

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

Production-tuned throughput (test 5, 8 threads x 50k rows = 400k):

```
append: ~500 ms (memory speed, no DuckDB)
flush:  ~3.4 sec total (~120k rows/sec sustained)
```

## Files

| Layer | File | Role |
|---|---|---|
| C++ schema | `tools/quantize/tessera/tessera-quantize-db.{h,cpp}` | Owns the schema (CREATE TABLE IF NOT EXISTS) and the typed insert helpers. `--quantize-db PATH` (renamed `--tessera-db PATH` in a follow-up) opens or creates the file. |
| C++ buffer | `tools/quantize/tessera/tessera-db-buffer.{h,cpp}` | MPSC write buffer with count + time + explicit flush triggers and sync-on-exit. |
| C++ dispatch | `tools/quantize/tessera/tessera-dispatch.cpp` | The dispatch opens one buffer for `ga_evaluations`; the eval_recorder callback pushes rows into it. |
| C++ tests | `tools/quantize/tessera/test_db_buffer.cpp` | 8 cases covering the buffer in isolation. |
| Python buffer | `tools/tessera/tessera_db_buffer.py` | Python mirror; same contract. |
| Python unified | `tools/tessera/tessera_db.py` | High-level API (`TesseraDB` class) that owns the DuckDB connection + per-table buffers. Typed helpers per summary table. |
| Python tests | `tools/tessera/test_tessera_db_buffer.py`, `tools/tessera/test_tessera_db.py` | 8 + 7 cases. |

## Usage

### C++ side (quantize pipeline)

```cpp
// Already wired in ts_dispatch_db_open / ts_dispatch_db_close.
ts_quantize_db * db = ts_quantize_db_open("tessera.duckdb", &err);
ts_db_buffer * buf = ts_db_buffer_open(db, "ga_evaluations", cols,
                                       65536, 1s);
for (...) {
    ts_db_buffer_append(buf, { run_id, tensor_name, ... });
}
ts_db_buffer_flush_now(buf);
ts_db_buffer_close(&buf);   // sync-on-exit
ts_quantize_db_close(db, "completed");
```

### Python side (calibration / analytics)

```python
from tessera_db import TesseraDB

with TesseraDB.open("tessera.duckdb") as db:
    # Calibration writes per-tensor stats
    db.insert_tensor_stats(model_hash=model_hash, rows=[
        {"name": "blk.0.attn_q.weight", "family": "attn_q",
         "kurtosis": 5.2, "eff_rank": 0.85, "rms": 0.12, ...},
        ...
    ])

    # Analytics writes per-tensor summaries
    db.insert_l4_probe(model_hash=model_hash, rows=[...])
    db.insert_l5_plan(model_hash=model_hash, rows=[...])

    # Cross-pipeline query (the analytical force multiplier)
    df = db.query("""
        SELECT t.name, t.kurtosis, t.eff_rank,
               l4.mse, l5.sensitivity_score
        FROM tensor_stats t
        LEFT JOIN l4_probe_summary l4 USING (model_hash, name)
        LEFT JOIN l5_plan_summary  l5 USING (model_hash, name)
        WHERE t.model_hash = ? AND t.kurtosis > 5.0
        ORDER BY t.kurtosis DESC
    """)
```

## Schema migration status

- **Phase 0 (this doc + commit `e90adc4d8`):** schema migration
  landed. 5 new tables, all `CREATE TABLE IF NOT EXISTS`. Existing
  6 tables untouched. Existing tests pass.
- **Phase 1 (commit `e96c0c21c`):** C++ buffer landed. 8 cases
  green. Existing `tessera-quantize-db` and `tessera-quantize-db-e2e`
  tests pass.
- **Phase 2 (commit `db0b1b002`):** dispatch refactor landed.
  Per-tensor DuckDB Appender sharded map replaced with a single
  `ts_db_buffer` for `ga_evaluations`. E2E + L5 tests green.
- **Phase 3 (commit `54e047674`):** Python side landed. Buffer
  + unified `tessera_db.py` + 8 + 7 cases green.
- **Phase 4 (commit `b8eac1a2a`):** Python tests stabilized
  (race fix on the parallel-producers test).

## Open follow-ups (not in this commit)

- Rename `--quantize-db PATH` to `--tessera-db PATH` (with
  `--quantize-db` as a deprecated alias for one release).
- Wire the C++ GA-prep walk to upsert into `tensor_stats`
  (currently writes only `tensors`; the new `tensor_stats`
  table is waiting for the C++ side to start populating it).
- Wire the Python calibration pipeline to write
  `rms, mean_abs, tail_ratio` into `tensor_stats` (the imatrix
  observer already computes these; the buffer is the only
  missing piece).
- Wire the C++ L5 requant fixup to read `l4_probe_summary` and
  `l5_plan_summary` for cross-pipeline feedback (the "did this
  requant plan actually reduce error?" loop the scout §5.4
  describes).
