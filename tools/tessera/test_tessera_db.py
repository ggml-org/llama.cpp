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
        model_role         TEXT NOT NULL DEFAULT 'trunk',
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
        PRIMARY KEY (model_hash, model_role, name)
    );
    CREATE TABLE IF NOT EXISTS l3_outlier_summary (
        model_hash        TEXT NOT NULL,
        model_role        TEXT NOT NULL DEFAULT 'trunk',
        name              TEXT NOT NULL,
        layer             INTEGER,
        sidecar_label     TEXT,
        outlier_count     BIGINT,
        outlier_fraction  DOUBLE,
        max_abs           DOUBLE,
        rms               DOUBLE,
        updated_at        TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, name, sidecar_label)
    );
    CREATE TABLE IF NOT EXISTS l4_probe_summary (
        model_hash        TEXT NOT NULL,
        model_role        TEXT NOT NULL DEFAULT 'trunk',
        name              TEXT NOT NULL,
        layer             INTEGER,
        current_qtype     TEXT,
        mse               DOUBLE,
        mse_minus_one     DOUBLE,
        perplexity        DOUBLE,
        top1_mismatch     DOUBLE,
        n_weights         BIGINT,
        updated_at        TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, name)
    );
    CREATE TABLE IF NOT EXISTS l5_plan_summary (
        model_hash            TEXT NOT NULL,
        model_role            TEXT NOT NULL DEFAULT 'trunk',
        model_role            TEXT DEFAULT 'trunk',
        name                  TEXT NOT NULL,
        layer                 INTEGER,
        iteration             INTEGER,
        plan_id               TEXT,
        sensitivity_score     DOUBLE,
        recommended_qtype     TEXT,
        recommended_alpha     DOUBLE,
        recommended_clip      DOUBLE,
        imatrix_magnitude     DOUBLE,
        gradient_proxy        DOUBLE,
        layer_position_prior  DOUBLE,
        updated_at            TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)
    );
    CREATE TABLE IF NOT EXISTS l5_outcome (
        model_hash            TEXT NOT NULL,
        model_role            TEXT NOT NULL DEFAULT 'trunk',
        model_role            TEXT DEFAULT 'trunk',
        name                  TEXT NOT NULL,
        layer                 INTEGER,
        iteration             INTEGER NOT NULL,
        plan_id               TEXT NOT NULL,
        family                TEXT,
        sensitivity_score     DOUBLE,
        imatrix_magnitude     DOUBLE,
        gradient_proxy        DOUBLE,
        layer_position_prior  DOUBLE,
        recommended_alpha     DOUBLE,
        recommended_clip      DOUBLE,
        mse_before            DOUBLE,
        mse_after             DOUBLE,
        delta_mse             DOUBLE,
        delta_frob            DOUBLE,
        plan_accepted         BOOLEAN,
        accept_threshold      DOUBLE,
        residual              DOUBLE,
        in_sample_loss        DOUBLE,
        updated_at            TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)
    );
    CREATE TABLE IF NOT EXISTS l5_weights (
        model_hash            TEXT NOT NULL,
        model_role            TEXT NOT NULL DEFAULT 'trunk',
        model_role            TEXT DEFAULT 'trunk',
        family                TEXT NOT NULL,
        w_imatrix             DOUBLE NOT NULL,
        w_gradient            DOUBLE NOT NULL,
        w_layer               DOUBLE NOT NULL,
        bias                  DOUBLE,
        n_samples             INTEGER,
        in_sample_loss        DOUBLE,
        hit_rate              DOUBLE,
        top_fraction          DOUBLE,
        retune_source         TEXT,
        updated_at            TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, family)
    );
    CREATE TABLE IF NOT EXISTS l4_plan_outcome (
        model_hash           TEXT NOT NULL,
        model_role           TEXT NOT NULL DEFAULT 'trunk',
        name                 TEXT NOT NULL,
        layer                INTEGER,
        iteration            INTEGER NOT NULL,
        plan_id              TEXT NOT NULL,
        strategy             TEXT,
        alpha_before         DOUBLE,
        alpha_after          DOUBLE,
        clip_before          DOUBLE,
        clip_after           DOUBLE,
        outlier_thresh_before DOUBLE,
        outlier_thresh_after  DOUBLE,
        mse_before           DOUBLE,
        mse_after            DOUBLE,
        frob_before          DOUBLE,
        frob_after           DOUBLE,
        family               TEXT,
        updated_at           TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)
        PRIMARY KEY (model_hash, model_role, family)
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


# ---- 8. Phase 15: l5_plan_summary with per-tensor component
    #         columns ----------------------------------------------

    def test_insert_l5_plan_with_component_columns(self) -> None:
        """Phase 15: insert_l5_plan writes the per-tensor
        component columns (imatrix_magnitude, gradient_proxy,
        layer_position_prior) and the row round-trips through
        a SELECT with the new columns populated.
        """
        path = self._fresh(8)
        with TesseraDB.open(path) as db:
            n = db.insert_l5_plan(
                model_hash="p15", rows=[
                    {
                        "name":                  "blk.0.attn_q.weight",
                        "layer":                 0,
                        "iteration":             0,
                        "plan_id":               "p0",
                        "sensitivity_score":     0.5,
                        "recommended_qtype":     "Q4_K",
                        "recommended_alpha":     0.5,
                        "recommended_clip":      1.0,
                        "imatrix_magnitude":     0.7,
                        "gradient_proxy":        0.4,
                        "layer_position_prior":  0.2,
                    },
                    {
                        "name":                  "blk.0.ffn_gate.weight",
                        "layer":                 0,
                        "iteration":             0,
                        "plan_id":               "p0",
                        "sensitivity_score":     0.3,
                        "recommended_qtype":     "Q4_K",
                        "recommended_alpha":     0.5,
                        "recommended_clip":      1.0,
                        # Components are None (the per-tensor
                        # components are optional; older
                        # producers / the C++ side before
                        # migration do not set them).
                        "imatrix_magnitude":     None,
                        "gradient_proxy":        None,
                        "layer_position_prior":  None,
                    },
                ],
            )
        # sync-on-exit drained the buffer.
        self.assertEqual(n, 2)
        # The rows are visible with the new columns populated
        # / NULL correctly.
        with TesseraDB.open(path, read_only=True) as db:
            df = db.query(
                "SELECT name, imatrix_magnitude, gradient_proxy, "
                "layer_position_prior FROM l5_plan_summary "
                "WHERE model_hash = 'p15' ORDER BY name"
            )
        self.assertEqual(df.height, 2)
        # attn_q: 0.7 / 0.4 / 0.2
        self.assertAlmostEqual(df["imatrix_magnitude"][0], 0.7, places=6)
        self.assertAlmostEqual(df["gradient_proxy"][0], 0.4, places=6)
        self.assertAlmostEqual(df["layer_position_prior"][0], 0.2, places=6)
        # ffn_gate: NULL / NULL / NULL
        self.assertIsNone(df["imatrix_magnitude"][1])
        self.assertIsNone(df["gradient_proxy"][1])
        self.assertIsNone(df["layer_position_prior"][1])

    def test_insert_l5_plan_adds_missing_columns_idempotently(self) -> None:
        """A pre-Phase-15 l5_plan_summary (without the
        per-tensor component columns) is upgraded in place
        on the first insert_l5_plan call. The ALTER TABLE
        ADD COLUMN IF NOT EXISTS is a no-op on subsequent
        calls. The end state has all three columns and the
        new insert round-trips correctly.
        """
        path = self._fresh(9)
        # Pre-Phase-15 schema: drop the per-tensor component
        # columns. The TesseraDB.open() in
        # _create_fresh_db() creates the full schema, so we
        # need to drop the Phase 15 columns to simulate a
        # pre-Phase-15 DB.
        import duckdb as _dd
        con = _dd.connect(path)
        try:
            for col in ("imatrix_magnitude", "gradient_proxy",
                        "layer_position_prior"):
                con.execute(f"ALTER TABLE l5_plan_summary DROP COLUMN {col}")
        finally:
            con.close()
        # Now insert_l5_plan() must add the columns back.
        with TesseraDB.open(path) as db:
            db.insert_l5_plan(
                model_hash="pre15", rows=[{
                    "name": "blk.0.attn_q.weight",
                    "iteration": 0, "plan_id": "p0",
                    "sensitivity_score": 0.5,
                    "recommended_qtype": "Q4_K",
                    "imatrix_magnitude": 0.6,
                    "gradient_proxy": 0.3,
                    "layer_position_prior": 0.1,
                }],
            )
        # The columns are present and the row is in the
        # table.
        with TesseraDB.open(path, read_only=True) as db:
            names = [c for c in db.query(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'l5_plan_summary'"
            )["column_name"].to_list()]
        for col in ("imatrix_magnitude", "gradient_proxy",
                    "layer_position_prior"):
            self.assertIn(col, names)
        # The row round-trips.
        with TesseraDB.open(path, read_only=True) as db:
            row = db.query(
                "SELECT imatrix_magnitude, gradient_proxy, "
                "layer_position_prior FROM l5_plan_summary "
                "WHERE model_hash = 'pre15'"
            ).row(0, named=True)
        self.assertAlmostEqual(row["imatrix_magnitude"], 0.6, places=6)
        self.assertAlmostEqual(row["gradient_proxy"], 0.3, places=6)
        self.assertAlmostEqual(row["layer_position_prior"], 0.1, places=6)

    def test_insert_l5_weights_with_top_fraction(self) -> None:
        """Phase 15: insert_l5_weights writes the per-family
        top_fraction column (nullable). The row round-trips
        through a SELECT.
        """
        path = self._fresh(10)
        with TesseraDB.open(path) as db:
            n = db.insert_l5_weights([
                {
                    "model_hash":     "tf",
                    "family":         "attn_q",
                    "w_imatrix":      0.6,
                    "w_gradient":     0.3,
                    "w_layer":        0.1,
                    "bias":           0.0,
                    "n_samples":      20,
                    "in_sample_loss": 0.001,
                    "hit_rate":       0.5,
                    "top_fraction":   0.18,
                },
                {
                    "model_hash":     "tf",
                    "family":         "ffn_gate",
                    "w_imatrix":      0.4,
                    "w_gradient":     0.5,
                    "w_layer":        0.1,
                    "bias":           0.0,
                    "n_samples":      20,
                    "in_sample_loss": 0.002,
                    "hit_rate":       0.4,
                    # top_fraction is None (older writers
                    # / the 2-coefficient fallback path).
                    "top_fraction":   None,
                },
            ])
        self.assertEqual(n, 2)
        # The rows round-trip.
        with TesseraDB.open(path, read_only=True) as db:
            df = db.query(
                "SELECT family, top_fraction FROM l5_weights "
                "WHERE model_hash = 'tf' ORDER BY family"
            )
        self.assertEqual(df.height, 2)
        # attn_q: top_fraction = 0.18
        self.assertAlmostEqual(df["top_fraction"][0], 0.18, places=6)
        # ffn_gate: top_fraction is NULL
        self.assertIsNone(df["top_fraction"][1])

    def test_insert_l5_weights_adds_top_fraction_idempotently(self) -> None:
        """A pre-Phase-15 l5_weights (without the top_fraction
        column) is upgraded in place on the first
        insert_l5_weights call. The column addition is
        idempotent across multiple calls.
        """
        path = self._fresh(11)
        # Drop the top_fraction column to simulate a
        # pre-Phase-15 DB.
        import duckdb as _dd
        con = _dd.connect(path)
        try:
            con.execute("ALTER TABLE l5_weights DROP COLUMN top_fraction")
        finally:
            con.close()
        with TesseraDB.open(path) as db:
            db.insert_l5_weights([{
                "model_hash":     "add",
                "family":         "attn_q",
                "w_imatrix":      0.5,
                "w_gradient":     0.3,
                "w_layer":        0.2,
                "n_samples":      10,
                "hit_rate":       0.5,
                "top_fraction":   0.12,
            }])
        # The column is back.
        with TesseraDB.open(path, read_only=True) as db:
            names = [c for c in db.query(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'l5_weights'"
            )["column_name"].to_list()]
        self.assertIn("top_fraction", names)
        # Idempotent: a second insert does not fail.
        with TesseraDB.open(path) as db:
            db.insert_l5_weights([{
                "model_hash":     "add",
                "family":         "ffn_gate",
                "w_imatrix":      0.4,
                "w_gradient":     0.4,
                "w_layer":        0.2,
                "n_samples":      10,
                "hit_rate":       0.5,
                "top_fraction":   0.08,
            }])
        # Both rows are present.
        with TesseraDB.open(path, read_only=True) as db:
            count = int(db.query(
                "SELECT COUNT(*) AS n FROM l5_weights "
                "WHERE model_hash = 'add'"
            )["n"][0])
        self.assertEqual(count, 2)

    # ---- Phase 16: model_role on l5_weights / l5_outcome / l5_plan ---

    def test_insert_l5_weights_with_model_role(self) -> None:
        """Phase 16: insert_l5_weights writes the
        model_role column; the (model_hash, model_role,
        family) PK is enforced (two rows with the same
        (model_hash, family) but different model_role
        coexist).
        """
        path = self._fresh(12)
        with TesseraDB.open(path) as db:
            n = db.insert_l5_weights([
                {
                    "model_hash":   "x", "model_role": "trunk",
                    "family":       "attn_q",
                    "w_imatrix":    0.6, "w_gradient": 0.3,
                    "w_layer":      0.1,
                    "n_samples":    10, "top_fraction": 0.18,
                },
                {
                    "model_hash":   "x", "model_role": "dflash",
                    "family":       "attn_q",
                    "w_imatrix":    0.4, "w_gradient": 0.5,
                    "w_layer":      0.1,
                    "n_samples":    10, "top_fraction": 0.30,
                },
            ])
        self.assertEqual(n, 2)
        with TesseraDB.open(path, read_only=True) as db:
            df = db.query(
                "SELECT model_role, family, top_fraction "
                "FROM l5_weights WHERE model_hash = 'x' "
                "ORDER BY model_role, family"
            )
        self.assertEqual(df.height, 2)
        # Same family, different roles, different
        # top_fraction. The 3-tuple PK is enforced.
        self.assertEqual(df["model_role"][0], "dflash")
        self.assertEqual(df["model_role"][1], "trunk")
        self.assertAlmostEqual(df["top_fraction"][0], 0.30, places=6)
        self.assertAlmostEqual(df["top_fraction"][1], 0.18, places=6)

    def test_insert_l5_weights_default_model_role_trunk(self) -> None:
        """When the row dict does not include model_role,
        the insert defaults to 'trunk' (the legacy
        pre-Phase-16 behaviour)."""
        path = self._fresh(13)
        with TesseraDB.open(path) as db:
            db.insert_l5_weights([{
                "model_hash":   "y", "family": "attn_q",
                "w_imatrix":    0.5, "w_gradient": 0.3,
                "w_layer":      0.2,
                "n_samples":    10,
            }])
        with TesseraDB.open(path, read_only=True) as db:
            role = db.query(
                "SELECT model_role FROM l5_weights "
                "WHERE model_hash = 'y'"
            )["model_role"][0]
        self.assertEqual(role, "trunk")

    def test_insert_l5_weights_adds_model_role_idempotently(self) -> None:
        """A pre-Phase-16 l5_weights (without the
        model_role column) is upgraded in place on the
        first insert_l5_weights call. The ALTER TABLE
        ADD COLUMN IF NOT EXISTS is a no-op on subsequent
        calls."""
        path = self._fresh(14)
        # Drop the model_role column to simulate a
        # pre-Phase-16 DB. The PRIMARY KEY in the test
        # schema is the post-Phase-16 3-tuple, but DuckDB
        # does not let us DROP a column that is part of
        # the PK, so we rebuild the table with the
        # pre-Phase-16 (model_hash, family) PK first.
        # The pre-Phase-16 table also lacks the
        # model_role column.
        import duckdb as _dd
        con = _dd.connect(path)
        try:
            con.execute("ALTER TABLE l5_weights RENAME TO l5_weights_old")
            con.execute(
                "CREATE TABLE l5_weights ("
                "  model_hash TEXT NOT NULL, family TEXT NOT NULL, "
                "  w_imatrix DOUBLE NOT NULL, w_gradient DOUBLE NOT NULL, "
                "  w_layer DOUBLE NOT NULL, bias DOUBLE, "
                "  n_samples INTEGER, in_sample_loss DOUBLE, "
                "  hit_rate DOUBLE, top_fraction DOUBLE, "
                "  retune_source TEXT, updated_at TIMESTAMP, "
                "  PRIMARY KEY (model_hash, family))"
            )
            # Copy only the pre-Phase-16 columns (the old
            # table had model_role; the new table does not).
            con.execute(
                "INSERT INTO l5_weights "
                "SELECT model_hash, family, w_imatrix, w_gradient, "
                "w_layer, bias, n_samples, in_sample_loss, hit_rate, "
                "top_fraction, retune_source, updated_at "
                "FROM l5_weights_old"
            )
            con.execute("DROP TABLE l5_weights_old")
        finally:
            con.close()
        with TesseraDB.open(path) as db:
            db.insert_l5_weights([{
                "model_hash":  "addrole", "model_role": "dflash",
                "family":      "attn_q",
                "w_imatrix":   0.5, "w_gradient": 0.3, "w_layer": 0.2,
                "n_samples":   10,
            }])
        # The model_role column is back.
        with TesseraDB.open(path, read_only=True) as db:
            names = [c for c in db.query(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'l5_weights'"
            )["column_name"].to_list()]
        self.assertIn("model_role", names)
        # The row was inserted.
        with TesseraDB.open(path, read_only=True) as db:
            df = db.query(
                "SELECT model_role, family FROM l5_weights "
                "WHERE model_hash = 'addrole' ORDER BY family"
            )
        self.assertEqual(df.height, 1)
        self.assertEqual(df["model_role"][0], "dflash")
        self.assertEqual(df["family"][0], "attn_q")
        # Idempotent: a second insert does not fail.
        with TesseraDB.open(path) as db:
            db.insert_l5_weights([{
                "model_hash":  "addrole", "model_role": "trunk",
                "family":      "ffn_gate",
                "w_imatrix":   0.4, "w_gradient": 0.4, "w_layer": 0.2,
                "n_samples":   10,
            }])

    def test_insert_l5_plan_with_model_role(self) -> None:
        """Phase 16: insert_l5_plan writes the model_role
        column; rows default to 'trunk' when not given."""
        path = self._fresh(15)
        with TesseraDB.open(path) as db:
            db.insert_l5_plan(
                model_hash="p1", model_role="dflash",
                rows=[{
                    "name": "blk.0.attn_q.weight", "layer": 0,
                    "iteration": 0, "plan_id": "p0",
                    "sensitivity_score": 0.5,
                    "recommended_qtype": "Q4_K",
                }],
            )
            db.insert_l5_plan(
                model_hash="p1",  # default role
                rows=[{
                    "name": "blk.0.ffn_gate.weight", "layer": 0,
                    "iteration": 0, "plan_id": "p0",
                    "sensitivity_score": 0.4,
                    "recommended_qtype": "Q4_K",
                }],
            )
        with TesseraDB.open(path, read_only=True) as db:
            df = db.query(
                "SELECT name, model_role FROM l5_plan_summary "
                "WHERE model_hash = 'p1' ORDER BY name"
            )
        self.assertEqual(df.height, 2)
        # The dflash row.
        attn = df.filter(df["name"] == "blk.0.attn_q.weight")
        self.assertEqual(attn["model_role"][0], "dflash")
        # The default trunk row.
        ffn = df.filter(df["name"] == "blk.0.ffn_gate.weight")
        self.assertEqual(ffn["model_role"][0], "trunk")

    def test_insert_l5_outcome_with_model_role(self) -> None:
        """Phase 16: insert_l5_outcome writes the
        model_role column; rows default to 'trunk' when
        not given."""
        path = self._fresh(16)
        with TesseraDB.open(path) as db:
            db.insert_l5_outcome(
                model_hash="o1", model_role="dflash",
                rows=[{
                    "name": "blk.0.attn_q.weight", "layer": 0,
                    "iteration": 0, "plan_id": "p0",
                    "family": "attn_q",
                    "sensitivity_score": 0.5,
                    "delta_mse": 0.001,
                }],
            )
            db.insert_l5_outcome(
                model_hash="o1",  # default role
                rows=[{
                    "name": "blk.0.ffn_gate.weight", "layer": 0,
                    "iteration": 0, "plan_id": "p0",
                    "family": "ffn_gate",
                    "sensitivity_score": 0.4,
                    "delta_mse": 0.001,
                }],
            )
        with TesseraDB.open(path, read_only=True) as db:
            df = db.query(
                "SELECT name, model_role FROM l5_outcome "
                "WHERE model_hash = 'o1' ORDER BY name"
            )
        self.assertEqual(df.height, 2)
        attn = df.filter(df["name"] == "blk.0.attn_q.weight")
        self.assertEqual(attn["model_role"][0], "dflash")
        ffn = df.filter(df["name"] == "blk.0.ffn_gate.weight")
        self.assertEqual(ffn["model_role"][0], "trunk")


# ---- 9. Phase 16: model_role round-trip on the 6 insert helpers -

    def test_insert_tensor_stats_model_role_round_trip(self) -> None:
        """Phase 16: insert_tensor_stats round-trips
        model_role. The default is 'trunk' when the row
        dict omits the key (preserves the pre-Phase-16
        contract); explicit values land in the new
        (model_hash, model_role, name) PK.
        """
        path = self._fresh(12)
        with TesseraDB.open(path) as db:
            # 3 rows: trunk (default), dflash, mtp_nextn.
            db.insert_tensor_stats(
                model_hash="p16", rows=[
                    {"name": "blk.0.attn_q.weight",
                     "family": "attn_q", "kurtosis": 3.0},
                    {"model_role": "dflash",
                     "name": "blk.0.attn_q.weight",
                     "family": "attn_q", "kurtosis": 4.0},
                    {"model_role": "mtp_nextn",
                     "name": "blk.0.nextn_proj.weight",
                     "family": "nextn", "kurtosis": 5.0},
                ],
            )
        with TesseraDB.open(path, read_only=True) as db:
            # Read back as a Python dict keyed by
            # (model_role, name) to make the assertions
            # order-independent.
            df = db.query(
                "SELECT model_role, name, kurtosis FROM tensor_stats "
                "WHERE model_hash = 'p16'"
            )
            n_blk0 = int(db.query(
                "SELECT COUNT(*) AS n FROM tensor_stats "
                "WHERE model_hash = 'p16' AND name = 'blk.0.attn_q.weight'"
            )["n"][0])
        self.assertEqual(df.height, 3)
        by_key = {
            (r[0], r[1]): r[2]
            for r in df.iter_rows(named=False)
        }
        self.assertEqual(by_key[("trunk", "blk.0.attn_q.weight")], 3.0)
        self.assertEqual(by_key[("dflash", "blk.0.attn_q.weight")], 4.0)
        self.assertEqual(
            by_key[("mtp_nextn", "blk.0.nextn_proj.weight")], 5.0
        )
        # Two rows for blk.0.attn_q.weight with different
        # model_role -> no PK collision.
        self.assertEqual(n_blk0, 2)

    def test_insert_l3_outlier_model_role_round_trip(self) -> None:
        """Phase 16: insert_l3_outlier round-trips
        model_role. A dflash-sidecar row coexists with the
        trunk's on the (model_hash, model_role, name,
        sidecar_label) PK.
        """
        path = self._fresh(13)
        with TesseraDB.open(path) as db:
            db.insert_l3_outlier(
                model_hash="p16", rows=[
                    {"name": "blk.0.attn_q.weight", "layer": 0,
                     "sidecar_label": "ckpt-v3",
                     "outlier_count": 17, "outlier_fraction": 0.001},
                    {"model_role": "dflash",
                     "name": "blk.0.attn_q.weight", "layer": 0,
                     "sidecar_label": "ckpt-v3-dflash",
                     "outlier_count": 9, "outlier_fraction": 0.0005},
                ],
            )
        with TesseraDB.open(path, read_only=True) as db:
            df = db.query(
                "SELECT model_role, name, sidecar_label, outlier_count "
                "FROM l3_outlier_summary WHERE model_hash = 'p16'"
            )
        self.assertEqual(df.height, 2)
        by_key = {
            (r[0], r[1], r[2]): r[3]
            for r in df.iter_rows(named=False)
        }
        self.assertEqual(
            by_key[("trunk", "blk.0.attn_q.weight", "ckpt-v3")], 17
        )
        self.assertEqual(
            by_key[("dflash", "blk.0.attn_q.weight", "ckpt-v3-dflash")], 9
        )

    def test_insert_l4_probe_model_role_round_trip(self) -> None:
        """Phase 16: insert_l4_probe round-trips model_role.
        """
        path = self._fresh(14)
        with TesseraDB.open(path) as db:
            db.insert_l4_probe(
                model_hash="p16", rows=[
                    {"name": "blk.0.attn_q.weight", "layer": 0,
                     "current_qtype": "Q4_K", "mse": 0.012,
                     "perplexity": 5.83, "top1_mismatch": 0.014},
                    {"model_role": "dflash",
                     "name": "blk.0.attn_q.weight", "layer": 0,
                     "current_qtype": "Q4_K", "mse": 0.020,
                     "perplexity": 6.10, "top1_mismatch": 0.020},
                ],
            )
        with TesseraDB.open(path, read_only=True) as db:
            df = db.query(
                "SELECT model_role, name, mse FROM l4_probe_summary "
                "WHERE model_hash = 'p16'"
            )
        self.assertEqual(df.height, 2)
        by_key = {
            (r[0], r[1]): r[2]
            for r in df.iter_rows(named=False)
        }
        self.assertAlmostEqual(
            by_key[("trunk", "blk.0.attn_q.weight")], 0.012, places=4
        )
        self.assertAlmostEqual(
            by_key[("dflash", "blk.0.attn_q.weight")], 0.020, places=4
        )

    def test_insert_l4_plan_outcome_model_role_round_trip(self) -> None:
        """Phase 16: insert_l4_plan_outcome round-trips
        model_role. The drafter-local tensor name (e.g.
        'blk.0.attn_q.weight' for the dflash encoder)
        coexists with the trunk's on the
        (model_hash, model_role, name, iteration, plan_id) PK.
        """
        path = self._fresh(15)
        with TesseraDB.open(path) as db:
            db.insert_l4_plan_outcome(
                model_hash="p16", rows=[
                    {"name": "blk.0.attn_q.weight", "layer": 0,
                     "iteration": 0, "plan_id": "p0",
                     "strategy": "A", "mse_before": 0.012,
                     "mse_after": 0.010, "family": "attn_q"},
                    {"model_role": "dflash",
                     "name": "blk.0.attn_q.weight", "layer": 0,
                     "iteration": 0, "plan_id": "p0",
                     "strategy": "A", "mse_before": 0.020,
                     "mse_after": 0.018, "family": "attn_q"},
                ],
            )
        with TesseraDB.open(path, read_only=True) as db:
            df = db.query(
                "SELECT model_role, name, mse_before, mse_after "
                "FROM l4_plan_outcome WHERE model_hash = 'p16'"
            )
        self.assertEqual(df.height, 2)
        by_key = {
            (r[0], r[1]): (r[2], r[3])
            for r in df.iter_rows(named=False)
        }
        self.assertAlmostEqual(
            by_key[("trunk", "blk.0.attn_q.weight")][1], 0.010, places=4
        )
        self.assertAlmostEqual(
            by_key[("dflash", "blk.0.attn_q.weight")][1], 0.018, places=4
        )

    def test_insert_l5_plan_model_role_round_trip(self) -> None:
        """Phase 16: insert_l5_plan round-trips model_role.
        """
        path = self._fresh(16)
        with TesseraDB.open(path) as db:
            db.insert_l5_plan(
                model_hash="p16", rows=[
                    {"name": "blk.0.attn_q.weight", "layer": 0,
                     "iteration": 0, "plan_id": "p0",
                     "sensitivity_score": 0.5,
                     "recommended_qtype": "Q4_K"},
                    {"model_role": "dflash",
                     "name": "blk.0.attn_q.weight", "layer": 0,
                     "iteration": 0, "plan_id": "p0",
                     "sensitivity_score": 0.6,
                     "recommended_qtype": "Q4_K"},
                ],
            )
        with TesseraDB.open(path, read_only=True) as db:
            df = db.query(
                "SELECT model_role, name, sensitivity_score "
                "FROM l5_plan_summary WHERE model_hash = 'p16'"
            )
        self.assertEqual(df.height, 2)
        by_key = {
            (r[0], r[1]): r[2]
            for r in df.iter_rows(named=False)
        }
        self.assertAlmostEqual(
            by_key[("trunk", "blk.0.attn_q.weight")], 0.5, places=4
        )
        self.assertAlmostEqual(
            by_key[("dflash", "blk.0.attn_q.weight")], 0.6, places=4
        )

    def test_insert_l5_outcome_model_role_round_trip(self) -> None:
        """Phase 16: insert_l5_outcome round-trips model_role.
        """
        path = self._fresh(17)
        with TesseraDB.open(path) as db:
            db.insert_l5_outcome(
                model_hash="p16", rows=[
                    {"name": "blk.0.attn_q.weight", "layer": 0,
                     "iteration": 0, "plan_id": "p0",
                     "family": "attn_q",
                     "sensitivity_score": 0.5,
                     "mse_before": 0.012, "mse_after": 0.010,
                     "plan_accepted": True},
                    {"model_role": "dflash",
                     "name": "blk.0.attn_q.weight", "layer": 0,
                     "iteration": 0, "plan_id": "p0",
                     "family": "attn_q",
                     "sensitivity_score": 0.6,
                     "mse_before": 0.020, "mse_after": 0.018,
                     "plan_accepted": True},
                ],
            )
        with TesseraDB.open(path, read_only=True) as db:
            df = db.query(
                "SELECT model_role, name, sensitivity_score, plan_accepted "
                "FROM l5_outcome WHERE model_hash = 'p16'"
            )
        self.assertEqual(df.height, 2)
        # Build a map: (model_role, name) -> (score, accepted)
        by_key = {
            (r[0], r[1]): (r[2], r[3])
            for r in df.iter_rows(named=False)
        }
        self.assertAlmostEqual(
            by_key[("trunk", "blk.0.attn_q.weight")][0], 0.5, places=4
        )
        self.assertTrue(by_key[("trunk", "blk.0.attn_q.weight")][1])
        self.assertAlmostEqual(
            by_key[("dflash", "blk.0.attn_q.weight")][0], 0.6, places=4
        )
        self.assertTrue(by_key[("dflash", "blk.0.attn_q.weight")][1])

    def test_insert_l5_weights_model_role_round_trip(self) -> None:
        """Phase 16: insert_l5_weights round-trips model_role.
        The (model_hash, model_role, family) PK lets the
        dflash family's retuned weights coexist with the
        trunk family's. Re-write on the same key overwrites
        (the upsert contract).
        """
        path = self._fresh(18)
        with TesseraDB.open(path) as db:
            db.insert_l5_weights([
                {"model_hash": "p16", "model_role": "trunk",
                 "family": "attn_q",
                 "w_imatrix": 0.4, "w_gradient": 0.3, "w_layer": 0.3,
                 "n_samples": 10, "hit_rate": 0.7},
                {"model_hash": "p16", "model_role": "dflash",
                 "family": "attn_q",
                 "w_imatrix": 0.2, "w_gradient": 0.5, "w_layer": 0.3,
                 "n_samples": 5, "hit_rate": 0.6},
            ])
        with TesseraDB.open(path, read_only=True) as db:
            df = db.query(
                "SELECT model_role, family, w_gradient FROM l5_weights "
                "WHERE model_hash = 'p16'"
            )
        self.assertEqual(df.height, 2)
        by_key = {
            (r[0], r[1]): r[2]
            for r in df.iter_rows(named=False)
        }
        self.assertAlmostEqual(
            by_key[("trunk", "attn_q")], 0.3, places=4
        )
        self.assertAlmostEqual(
            by_key[("dflash", "attn_q")], 0.5, places=4
        )

        # Re-write with the same model_role+family is an
        # upsert (overwrites via ON CONFLICT).
        with TesseraDB.open(path) as db:
            db.insert_l5_weights([
                {"model_hash": "p16", "model_role": "dflash",
                 "family": "attn_q",
                 "w_imatrix": 0.1, "w_gradient": 0.6, "w_layer": 0.3,
                 "n_samples": 8, "hit_rate": 0.65},
            ])
        with TesseraDB.open(path, read_only=True) as db:
            df = db.query(
                "SELECT w_gradient, n_samples FROM l5_weights "
                "WHERE model_hash = 'p16' AND model_role = 'dflash' "
                "AND family = 'attn_q'"
            )
        self.assertEqual(df.height, 1)
        self.assertAlmostEqual(df["w_gradient"][0], 0.6, places=4)
        self.assertEqual(df["n_samples"][0], 8)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTesseraDB)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
