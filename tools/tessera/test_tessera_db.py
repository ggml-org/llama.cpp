"""Tests for tools/tessera/tessera_db.py (high-level unified DB API).

Exercises the typed insert helpers, multi-table writes, sync-on-exit
via the context manager, read-only mode, and the roundtrip with a
C++-created schema (the production path: C++ opens + creates the
schema, Python writes to it).

Run as a unittest module. Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import duckdb
import polars as pl

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from tessera_db import (
    L3_OUTLIER_COLS,
    L4_PROBE_COLS,
    L5_PLAN_COLS,
    PER_LAYER_ERROR_COLS,
    TENSOR_STATS_COLS,
    TesseraDB,
)


def _fresh_path(idx: int) -> str:
    return f"/tmp/tessera-db-py-test-{idx}.duckdb"


# Mirror of the C++ schema in tessera-quantize-db.cpp. Used when the
# test needs a fresh DB without depending on the C++ binary.
SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        model_path TEXT,
        model_hash TEXT,
        tessera_commit TEXT,
        config_json TEXT,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        status TEXT DEFAULT 'running'
    );
    CREATE TABLE IF NOT EXISTS tensor_stats (
        model_hash         TEXT NOT NULL,
        name               TEXT NOT NULL,
        family             TEXT,
        layer_depth        INTEGER,
        out_dim            BIGINT,
        in_dim             BIGINT,
        n_elements         BIGINT,
        dtype              TEXT,
        kurtosis           DOUBLE,
        eff_rank           DOUBLE,
        rms                DOUBLE,
        mean_abs           DOUBLE,
        tail_ratio         DOUBLE,
        source             TEXT,
        recommended_action TEXT,
        updated_at         TIMESTAMP,
        PRIMARY KEY (model_hash, name)
    );
    CREATE TABLE IF NOT EXISTS l3_outlier_summary (
        model_hash        TEXT NOT NULL,
        name              TEXT NOT NULL,
        layer             INTEGER,
        sidecar_label     TEXT,
        outlier_count     BIGINT,
        outlier_fraction  DOUBLE,
        max_abs           DOUBLE,
        rms               DOUBLE,
        updated_at        TIMESTAMP,
        PRIMARY KEY (model_hash, name, sidecar_label)
    );
    CREATE TABLE IF NOT EXISTS l4_probe_summary (
        model_hash        TEXT NOT NULL,
        name              TEXT NOT NULL,
        layer             INTEGER,
        current_qtype     TEXT,
        mse               DOUBLE,
        mse_minus_one     DOUBLE,
        perplexity        DOUBLE,
        top1_mismatch     DOUBLE,
        n_weights         BIGINT,
        updated_at        TIMESTAMP,
        PRIMARY KEY (model_hash, name)
    );
    CREATE TABLE IF NOT EXISTS l5_plan_summary (
        model_hash        TEXT NOT NULL,
        name              TEXT NOT NULL,
        layer             INTEGER,
        iteration         INTEGER,
        plan_id           TEXT,
        sensitivity_score DOUBLE,
        recommended_qtype TEXT,
        recommended_alpha DOUBLE,
        recommended_clip  DOUBLE,
        updated_at        TIMESTAMP,
        PRIMARY KEY (model_hash, name, iteration, plan_id)
    );
    CREATE TABLE IF NOT EXISTS per_layer_error_summary (
        model_hash        TEXT NOT NULL,
        name              TEXT NOT NULL,
        layer             INTEGER,
        epsilon           DOUBLE,
        reference_qtype   TEXT,
        updated_at        TIMESTAMP,
        PRIMARY KEY (model_hash, name)
    );
"""


def _create_fresh_db(path: str) -> None:
    con = duckdb.connect(path)
    try:
        for stmt in SCHEMA_SQL.strip().split(";"):
            s = stmt.strip()
            if s:
                con.execute(s)
    finally:
        con.close()


def _count(path: str, table: str) -> int:
    con = duckdb.connect(path, read_only=True)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


class TestTesseraDB(unittest.TestCase):
    def setUp(self) -> None:
        self.paths: list[str] = []

    def tearDown(self) -> None:
        for p in self.paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

    def _fresh(self, idx: int) -> str:
        p = _fresh_path(idx)
        self.paths.append(p)
        _create_fresh_db(p)
        return p

    # ---- 1. Open + tensor_stats insert + close ----------------------

    def test_insert_tensor_stats(self) -> None:
        path = self._fresh(1)
        with TesseraDB.open(path) as db:
            n = db.insert_tensor_stats(
                model_hash="hash_model_A",
                rows=[
                    {"name": "blk.0.attn_q.weight", "family": "attn_q",
                     "layer_depth": 0, "kurtosis": 3.2, "eff_rank": 0.85,
                     "rms": 0.12, "mean_abs": 0.10, "tail_ratio": 4.5,
                     "dtype": "f16", "out_dim": 4096, "in_dim": 4096,
                     "n_elements": 16_777_216},
                    {"name": "blk.0.attn_k.weight", "family": "attn_k",
                     "layer_depth": 0, "kurtosis": 3.1, "eff_rank": 0.86,
                     "rms": 0.11, "mean_abs": 0.09, "tail_ratio": 4.4,
                     "dtype": "f16", "out_dim": 4096, "in_dim": 4096,
                     "n_elements": 16_777_216},
                    {"name": "blk.0.ffn_gate.weight", "family": "ffn_gate",
                     "layer_depth": 0, "kurtosis": 7.5, "eff_rank": 0.42,
                     "rms": 0.05, "mean_abs": 0.04, "tail_ratio": 12.0,
                     "dtype": "f16", "out_dim": 4096, "in_dim": 11008,
                     "n_elements": 45_088_768},
                ],
            )
            self.assertEqual(n, 3)
        # After close, the sync-on-exit drain must have landed all rows.
        self.assertEqual(_count(path, "tensor_stats"), 3)

    # ---- 2. Multi-table writes (calibration + analytics) -------------

    def test_multi_table_writes(self) -> None:
        path = self._fresh(2)
        with TesseraDB.open(path) as db:
            db.insert_tensor_stats(model_hash="m", rows=[
                {"name": "blk.0.attn_q.weight", "family": "attn_q",
                 "layer_depth": 0, "kurtosis": 3.2, "eff_rank": 0.85},
            ])
            db.insert_l3_outlier(model_hash="m", rows=[
                {"name": "blk.0.attn_q.weight", "layer": 0,
                 "sidecar_label": "ckpt-v3", "outlier_count": 17,
                 "outlier_fraction": 0.001013, "max_abs": 4.5, "rms": 0.12},
            ])
            db.insert_l4_probe(model_hash="m", rows=[
                {"name": "blk.0.attn_q.weight", "layer": 0,
                 "current_qtype": "Q4_K", "mse": 0.012, "mse_minus_one": 0.0002,
                 "perplexity": 5.83, "top1_mismatch": 0.014, "n_weights": 16_777_216},
            ])
            db.insert_l5_plan(model_hash="m", rows=[
                {"name": "blk.0.attn_q.weight", "layer": 0,
                 "iteration": 0, "plan_id": "p0", "sensitivity_score": 0.87,
                 "recommended_qtype": "Q4_K", "recommended_alpha": 0.5,
                 "recommended_clip": 1.0},
            ])
            db.insert_per_layer_error(model_hash="m", rows=[
                {"name": "blk.0.attn_q.weight", "layer": 0,
                 "epsilon": 0.0012, "reference_qtype": "Q4_K"},
            ])
        self.assertEqual(_count(path, "tensor_stats"), 1)
        self.assertEqual(_count(path, "l3_outlier_summary"), 1)
        self.assertEqual(_count(path, "l4_probe_summary"), 1)
        self.assertEqual(_count(path, "l5_plan_summary"), 1)
        self.assertEqual(_count(path, "per_layer_error_summary"), 1)

    # ---- 3. Query roundtrip: insert then SELECT via polars -----------

    def test_query_returns_polars_dataframe(self) -> None:
        path = self._fresh(3)
        with TesseraDB.open(path) as db:
            # insert_tensor_stats uses direct INSERT ... ON CONFLICT
            # (bypasses the per-table buffer because the table has a
            # primary key), so the rows are visible immediately.
            db.insert_tensor_stats(model_hash="m", rows=[
                {"name": f"tensor_{i}", "family": "attn_q",
                 "layer_depth": i, "kurtosis": 3.0 + i * 0.1,
                 "eff_rank": 0.9 - i * 0.01}
                for i in range(10)
            ])
            df = db.query(
                "SELECT name, kurtosis FROM tensor_stats "
                "WHERE model_hash = 'm' ORDER BY kurtosis DESC LIMIT 3"
            )
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertEqual(df["name"].to_list()[0], "tensor_9")
        self.assertAlmostEqual(df["kurtosis"].to_list()[0], 3.9, places=2)

    # ---- 4. Roundtrip with the C++-created schema -------------------

    def test_roundtrip_with_cpp_schema(self) -> None:
        """Simulate the production path: C++ opens the DB and
        creates the schema, then Python opens the same file and
        writes via TesseraDB. The test uses a fresh DB created by
        the same SCHEMA_SQL the C++ binary uses (this is the
        authoritative schema; see tessera-quantize-db.cpp's
        TS_QDB_SCHEMA_SQL)."""
        path = self._fresh(4)
        with TesseraDB.open(path) as db:
            db.insert_tensor_stats(model_hash="cpp_then_py", rows=[
                {"name": "blk.0.attn_q.weight", "family": "attn_q",
                 "kurtosis": 5.0, "eff_rank": 0.8},
            ])
        # Verify via a read-only DuckDB connection.
        self.assertEqual(_count(path, "tensor_stats"), 1)
        con = duckdb.connect(path, read_only=True)
        try:
            row = con.execute(
                "SELECT model_hash, name, family, kurtosis, eff_rank "
                "FROM tensor_stats WHERE model_hash = 'cpp_then_py'"
            ).fetchone()
        finally:
            con.close()
        self.assertEqual(row[0], "cpp_then_py")
        self.assertEqual(row[1], "blk.0.attn_q.weight")
        self.assertEqual(row[2], "attn_q")
        self.assertAlmostEqual(row[3], 5.0, places=4)
        self.assertAlmostEqual(row[4], 0.8, places=4)

    # ---- 5. Read-only mode -------------------------------------------

    def test_read_only_rejects_writes(self) -> None:
        path = self._fresh(5)
        # Pre-populate via a writable TesseraDB.
        with TesseraDB.open(path) as db:
            db.insert_tensor_stats(model_hash="ro", rows=[
                {"name": "tensor_x", "family": "attn_q", "kurtosis": 1.0},
            ])
        # Reopen read-only.
        with TesseraDB.open(path, read_only=True) as db:
            n = db.insert_tensor_stats(model_hash="ro", rows=[
                {"name": "should_not_land", "family": "attn_q"},
            ])
            self.assertEqual(n, 0)  # write rejected
            df = db.query("SELECT COUNT(*) AS n FROM tensor_stats")
            self.assertEqual(df["n"].to_list()[0], 1)

    # ---- 6. Buffer stats: per-table counters reflect work -----------

    def test_buffer_stats_reflect_work(self) -> None:
        """Buffer stats reflect per-table work. l3_outlier_summary
        uses the buffered path (no primary key on the table), so the
        appended / flushed_rows counters track the writes."""
        path = self._fresh(6)
        with TesseraDB.open(path) as db:
            db.insert_l3_outlier(model_hash="s", rows=[
                {"name": f"t_{i}", "layer": 0, "sidecar_label": "ckpt",
                 "outlier_count": 1, "outlier_fraction": 0.001}
                for i in range(100)
            ])
            # Force a flush so the counter is updated.
            db._buffers["l3_outlier_summary"].flush_now()
            stats = db.buffer_stats()
        self.assertIn("l3_outlier_summary", stats)
        s = stats["l3_outlier_summary"]
        self.assertEqual(s.appended, 100)
        self.assertEqual(s.flushed_rows, 100)
        self.assertEqual(s.rows_dropped, 0)

    # ---- 7. table_names() returns the unified schema ----------------

    def test_table_names_lists_unified_schema(self) -> None:
        path = self._fresh(7)
        with TesseraDB.open(path) as db:
            names = db.table_names()
        expected_tables = {
            "tensor_stats", "l3_outlier_summary", "l4_probe_summary",
            "l5_plan_summary", "per_layer_error_summary",
        }
        self.assertTrue(expected_tables.issubset(set(names)),
                        f"missing tables: {expected_tables - set(names)}")

    # ---- 8. insert_tensor_stats with recommended_action ----------------

    def test_insert_tensor_stats_with_recommended_action(self) -> None:
        """The recommended_action column is upserted on the Python
        side. The COALESCE preservation rule means: if a row was
        written with a recommended_action and a subsequent write
        omits the column (None), the prior value is preserved."""
        path = self._fresh(8)
        with TesseraDB.open(path) as db:
            # First write: set recommended_action = "protect".
            n = db.insert_tensor_stats(
                model_hash="m",
                rows=[{
                    "name": "blk.0.attn_q.weight", "family": "attn_q",
                    "layer_depth": 0, "kurtosis": 3.2, "eff_rank": 0.85,
                    "rms": 0.12, "mean_abs": 0.10, "tail_ratio": 4.5,
                    "dtype": "f16", "out_dim": 4096, "in_dim": 4096,
                    "n_elements": 16_777_216,
                    "recommended_action": "protect",
                }],
            )
            self.assertEqual(n, 1)
            df = db.query(
                "SELECT recommended_action FROM tensor_stats "
                "WHERE model_hash = 'm'"
            )
            self.assertEqual(df.height, 1)
            self.assertEqual(df["recommended_action"].to_list()[0], "protect")

            # Second write on the same row: same key, no
            # recommended_action. COALESCE keeps the prior value.
            n = db.insert_tensor_stats(
                model_hash="m",
                rows=[{
                    "name": "blk.0.attn_q.weight", "family": "attn_q",
                    "layer_depth": 0, "kurtosis": 9.9,  # updated
                    "rms": 0.20,  # updated
                }],
            )
            self.assertEqual(n, 1)
            df = db.query(
                "SELECT kurtosis, rms, recommended_action "
                "FROM tensor_stats WHERE model_hash = 'm'"
            )
            row = df.row(0, named=True)
            # Updated columns took the new values.
            self.assertAlmostEqual(row["kurtosis"], 9.9, places=4)
            self.assertAlmostEqual(row["rms"], 0.20, places=4)
            # recommended_action preserved (Python side's prior
            # value survived the second write's NULL-pass).
            self.assertEqual(row["recommended_action"], "protect")

    def test_insert_tensor_stats_recommended_action_overwrite(self) -> None:
        """When the new write carries a recommended_action, it
        overwrites the prior value (the Python side is the
        authoritative writer for this column; same rule as rms /
        mean_abs / tail_ratio when both sides are Python)."""
        path = self._fresh(9)
        with TesseraDB.open(path) as db:
            db.insert_tensor_stats(model_hash="m", rows=[{
                "name": "blk.0.attn_q.weight", "family": "attn_q",
                "rms": 0.10,
                "recommended_action": "monitor",
            }])
            # Second write with a different recommended_action.
            db.insert_tensor_stats(model_hash="m", rows=[{
                "name": "blk.0.attn_q.weight", "family": "attn_q",
                "rms": 0.20,
                "recommended_action": "requant_up",
            }])
            df = db.query(
                "SELECT rms, recommended_action FROM tensor_stats "
                "WHERE model_hash = 'm'"
            )
            row = df.row(0, named=True)
            self.assertAlmostEqual(row["rms"], 0.20, places=4)
            self.assertEqual(row["recommended_action"], "requant_up")

    def test_insert_tensor_stats_recommended_action_none_on_fresh(self) -> None:
        """A fresh row written without recommended_action is NULL
        (the Python side has not produced a verdict for this tensor
        yet)."""
        path = self._fresh(10)
        with TesseraDB.open(path) as db:
            db.insert_tensor_stats(model_hash="m", rows=[{
                "name": "blk.0.attn_q.weight", "family": "attn_q",
                "rms": 0.10,
            }])
            df = db.query(
                "SELECT recommended_action FROM tensor_stats "
                "WHERE model_hash = 'm'"
            )
            self.assertEqual(df.height, 1)
            # recommended_action is NULL when not provided.
            self.assertIsNone(df["recommended_action"].to_list()[0])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTesseraDB)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
