"""Tests for tools/tessera/l5_outcome.py.

The feedback loop has three sides:

  1. The Python l5_orchestrator writes l5_plan_summary rows
     (one per (tensor, iteration, plan_id) with sensitivity_score,
     recommended_alpha, recommended_clip).
  2. The C++ dispatch (or any consumer of the L5 plan) writes
     l4_plan_outcome rows (one per (tensor, iteration, plan_id)
     with mse_before, mse_after).
  3. l5_outcome.py joins the two, computes delta_mse, plan_accepted,
     residual, and writes l5_outcome.

These tests simulate sides 1 and 2 with synthetic data and verify
side 3 produces the expected verdict, hit rate, and sensitivity
calibration residuals. The C++/Python cross is covered by the C++
test (test_quantize_db.cpp::test_l4_plan_outcome); this test focuses
on the Python consumer.

Run as a unittest module. Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import polars as pl

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from tessera_db import TesseraDB
import l5_outcome as l5o


def _fresh_path(idx: int) -> str:
    return f"/tmp/tessera-l5-outcome-test-{idx}.duckdb"


# Schema (only the tables the l5_outcome flow touches). Mirrored
# on the C++ side in tessera-quantize-db.cpp.
#
# Phase 15: the per-tensor sensitivity component columns
# (imatrix_magnitude, gradient_proxy, layer_position_prior) live
# on both l5_plan_summary and l5_outcome. The test schema
# includes them from the start so the test exercises the full
# column path; production code adds them via ALTER TABLE IF NOT
# EXISTS for DBs that pre-date the column addition.
SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS l5_plan_summary (
        model_hash            TEXT NOT NULL,
        model_role            TEXT NOT NULL DEFAULT 'trunk',
        name                  TEXT NOT NULL,
        layer                 INTEGER,
        iteration             INTEGER NOT NULL,
        plan_id               TEXT NOT NULL,
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
    );
    CREATE TABLE IF NOT EXISTS l5_outcome (
        model_hash            TEXT NOT NULL,
        model_role            TEXT NOT NULL DEFAULT 'trunk',
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
        updated_at            TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)
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


class TestL5Outcome(unittest.TestCase):
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

    # ---- 1. Happy path: a mix of accepted and rejected plans ----

    def test_basic_outcome_join(self) -> None:
        """10 plans across 3 iterations of 2 tensors. Mix of:
        - accepted: delta_mse < 0 (plan reduced error)
        - rejected: delta_mse > 0 (plan hurt)

        Verify the verdict classifies each correctly, the hit
        rate matches the manual count, and the per-family rollup
        is consistent.
        """
        path = self._fresh(1)
        # Synthetic l5_plan_summary: 5 plans for two tensors (A, B)
        # across iterations 0..2. The plan_id encodes the iteration
        # (matches the l5_orchestrator's write_history convention).
        plan_rows = []
        for it in range(3):
            for tname in ("blk.0.attn_q.weight", "blk.0.ffn_gate.weight"):
                plan_rows.append({
                    "name":              tname,
                    "layer":             0,
                    "iteration":         it,
                    "plan_id":           f"py_orch_iter{it}",
                    "sensitivity_score": 0.5 + 0.1 * it,
                    "recommended_qtype": "Q4_K",
                    "recommended_alpha": 0.5,
                    "recommended_clip":  1.0,
                })
        # Synthetic l4_plan_outcome: the post-apply measurement.
        # Build a deterministic table of (tensor, iter) -> mse_after.
        # Plan A (attn_q) at iter 0/1/2: mse 0.012 -> 0.011 -> 0.010
        #   (each iter reduces; deltas = -0.001, -0.001, -0.001)
        # Plan B (ffn_gate) at iter 0/1/2: mse 0.025 -> 0.024 -> 0.026
        #   (iter 0 reduces, iter 1 reduces, iter 2 increases)
        attn_mse = [(0.012, 0.011), (0.011, 0.010), (0.010, 0.009)]
        ffn_mse  = [(0.025, 0.024), (0.024, 0.023), (0.023, 0.026)]
        outcome_rows = []
        for it, (mb, ma) in enumerate(attn_mse):
            outcome_rows.append({
                "name":                 "blk.0.attn_q.weight",
                "layer":                0,
                "iteration":            it,
                "plan_id":              f"py_orch_iter{it}",
                "strategy":             "A",
                "mse_before":           mb,
                "mse_after":            ma,
                "frob_before":          mb,
                "frob_after":           ma,
                "family":               "attn_q",
            })
        for it, (mb, ma) in enumerate(ffn_mse):
            outcome_rows.append({
                "name":                 "blk.0.ffn_gate.weight",
                "layer":                0,
                "iteration":            it,
                "plan_id":              f"py_orch_iter{it}",
                "strategy":             "A",
                "mse_before":           mb,
                "mse_after":            ma,
                "frob_before":          mb,
                "frob_after":           ma,
                "family":               "ffn_gate",
            })

        with TesseraDB.open(path) as db:
            db.insert_l5_plan(model_hash="m", rows=plan_rows)
            db.insert_l4_plan_outcome(model_hash="m", rows=outcome_rows)
        # The context manager's __exit__ invokes close() on every
        # buffer, which drains the pending queue (sync-on-exit).
        # By the time the `with` block exits, both l5_plan_summary
        # and l4_plan_outcome are committed to disk.
        self.assertEqual(_count(path, "l5_plan_summary"), 6)
        self.assertEqual(_count(path, "l4_plan_outcome"), 6)

        # Run the verdict.
        verdict = l5o.compute_l5_outcome(path, model_hash="m",
                                          accept_threshold=0.0,
                                          write_back=True)
        self.assertEqual(verdict.height, 6, "6 joined rows")
        # Check per-row verdicts.
        for row in verdict.to_dicts():
            delta = row["delta_mse"]
            accepted = row["plan_accepted"]
            if delta < 0:
                self.assertTrue(accepted,
                    f"plan with delta_mse={delta} should be accepted")
            else:
                self.assertFalse(accepted,
                    f"plan with delta_mse={delta} should be rejected")

        # Hit rate: attn_q (3 iters) all reduced -> 3 accepted;
        # ffn_gate (3 iters) iters 0+1 reduced, iter 2 increased
        # -> 2 accepted. Total: 5/6 = 0.833.
        s = l5o.summarize(verdict)
        self.assertEqual(s.n_plans, 6)
        self.assertEqual(s.n_accepted, 5)
        self.assertAlmostEqual(s.hit_rate, 5 / 6, places=4)
        # Per-family: attn_q 3/3 = 1.0, ffn_gate 2/3 = 0.667.
        self.assertAlmostEqual(s.per_family_hit_rate["attn_q"], 1.0, places=4)
        self.assertAlmostEqual(s.per_family_hit_rate["ffn_gate"], 2 / 3, places=4)

        # l5_outcome landed.
        self.assertEqual(_count(path, "l5_outcome"), 6)

    # ---- 2. Sensitivity calibration: residual is small when the
    #         orchestrator's sensitivity score tracks the actual delta

    def test_sensitivity_calibration(self) -> None:
        """Build a calibration scenario where the orchestrator's
        sensitivity_score * 0.01 is a near-perfect predictor of
        delta_mse. The residual (delta_mse - (a + b*sensitivity))
        should be near 0.
        """
        path = self._fresh(2)
        # 4 plans where delta_mse = 0.005 * sensitivity_score (i.e.
        # the orchestrator's sensitivity is calibrated to predict
        # delta_mse at a 0.005 scale).
        plan_rows = []
        outcome_rows = []
        sensitivity_grid = [0.2, 0.4, 0.6, 0.8]
        for i, s in enumerate(sensitivity_grid):
            plan_rows.append({
                "name":              f"tensor_{i}",
                "layer":             0,
                "iteration":         0,
                "plan_id":           f"p_{i}",
                "sensitivity_score": s,
                "recommended_qtype": "Q4_K",
                "recommended_alpha": 0.5,
                "recommended_clip":  1.0,
            })
            # delta_mse = 0.005 * s, so mse_after - mse_before = 0.005*s.
            # Choose mse_before = 0.010, then mse_after = 0.010 + 0.005*s.
            mb = 0.010
            ma = mb + 0.005 * s
            outcome_rows.append({
                "name":                 f"tensor_{i}",
                "layer":                0,
                "iteration":            0,
                "plan_id":              f"p_{i}",
                "strategy":             "A",
                "mse_before":           mb,
                "mse_after":            ma,
                "frob_before":          mb,
                "frob_after":           ma,
                "family":               "attn_q",
            })
        with TesseraDB.open(path) as db:
            db.insert_l5_plan(model_hash="cal", rows=plan_rows)
            db.insert_l4_plan_outcome(model_hash="cal", rows=outcome_rows)
        # sync-on-exit drained both buffers.

        verdict = l5o.compute_l5_outcome(path, model_hash="cal",
                                          accept_threshold=0.0,
                                          write_back=False)
        # All 4 deltas are positive (plan hurt the error), so all
        # 4 are rejected. Mean delta is ~ 0.005*0.5 = 0.0025.
        s = l5o.summarize(verdict)
        self.assertEqual(s.n_plans, 4)
        self.assertEqual(s.n_accepted, 0)
        self.assertAlmostEqual(s.mean_delta_mse, 0.0025, places=4)
        # The residual of delta_mse on sensitivity_score should be
        # near 0 (the linear fit captures the relationship exactly).
        # Each row's residual is (delta_mse - (a + b*sensitivity));
        # the mean abs residual is the per-row calibration error.
        self.assertLess(s.mean_residual, 1e-9,
            f"residual too large for a calibrated scenario: {s.mean_residual}")

    # ---- 3. Empty DB / missing schema -> clear error

    def test_missing_tables_raises(self) -> None:
        path = self._fresh(3)
        # Remove the l5_plan_summary table to simulate a fresh
        # C++ DB that hasn't been touched by the orchestrator.
        con = duckdb.connect(path)
        con.execute("DROP TABLE l5_plan_summary")
        con.close()
        with self.assertRaises(RuntimeError) as ctx:
            l5o.compute_l5_outcome(path, write_back=False)
        self.assertIn("l5_plan_summary", str(ctx.exception))

    # ---- 4. Dry-run does not write --------------------

    def test_dry_run_does_not_write(self) -> None:
        path = self._fresh(4)
        with TesseraDB.open(path) as db:
            db.insert_l5_plan(model_hash="d", rows=[{
                "name": "tensor_d", "iteration": 0, "plan_id": "p0",
                "sensitivity_score": 0.5, "recommended_qtype": "Q4_K",
            }])
            db.insert_l4_plan_outcome(model_hash="d", rows=[{
                "name": "tensor_d", "iteration": 0, "plan_id": "p0",
                "mse_before": 0.01, "mse_after": 0.005,
                "frob_before": 0.01, "frob_after": 0.005,
                "family": "attn_q",
            }])
        # sync-on-exit drained both buffers.

        verdict = l5o.compute_l5_outcome(path, model_hash="d",
                                          write_back=False)
        self.assertEqual(verdict.height, 1)
        self.assertEqual(_count(path, "l5_outcome"), 0,
            "dry-run must not write to l5_outcome")

    # ---- 5. Idempotent re-run: the second run replaces the first --

    def test_idempotent_rerun(self) -> None:
        """Running compute_l5_outcome twice for the same model
        replaces the prior verdict rows (transactional
        DELETE+INSERT). The l5_outcome table ends up with one
        row per (model, name, iteration, plan_id), not duplicates
        from the first run.
        """
        path = self._fresh(5)
        plan_rows = [{
            "name": "t_x", "iteration": 0, "plan_id": "p0",
            "sensitivity_score": 0.3, "recommended_qtype": "Q4_K",
        }]
        outcome_rows = [{
            "name": "t_x", "iteration": 0, "plan_id": "p0",
            "mse_before": 0.01, "mse_after": 0.008,
            "frob_before": 0.01, "frob_after": 0.008,
            "family": "attn_q",
        }]
        with TesseraDB.open(path) as db:
            db.insert_l5_plan(model_hash="idem", rows=plan_rows)
            db.insert_l4_plan_outcome(model_hash="idem", rows=outcome_rows)
        # sync-on-exit drained both buffers.

        # First run.
        l5o.compute_l5_outcome(path, model_hash="idem", write_back=True)
        self.assertEqual(_count(path, "l5_outcome"), 1)
        # Second run.
        l5o.compute_l5_outcome(path, model_hash="idem", write_back=True)
        self.assertEqual(_count(path, "l5_outcome"), 1,
            "second run must replace, not append")

    # ---- 6. Phase 15: per-tensor sensitivity component columns ----

    def test_per_tensor_components_flow_through(self) -> None:
        """The per-tensor sensitivity components
        (imatrix_magnitude, gradient_proxy, layer_position_prior)
        live on l5_plan_summary and must flow through the join
        to l5_outcome. The 3-coefficient retune (Phase 15) reads
        them; the verdict writeback preserves them so the retune
        does not need a second join.
        """
        path = self._fresh(6)
        # Two plans, with distinct per-tensor component values.
        # The component sum equals the (combined) sensitivity_score
        # under the default weights 0.5/0.3/0.2, so a downstream
        # consumer can cross-check.
        plan_rows = [
            {
                "name":              "blk.0.attn_q.weight",
                "layer":             0,
                "iteration":         0,
                "plan_id":           "p0",
                "sensitivity_score": 0.5 * 0.6 + 0.3 * 0.4 + 0.2 * 0.2,
                "recommended_qtype": "Q4_K",
                "imatrix_magnitude":  0.6,
                "gradient_proxy":     0.4,
                "layer_position_prior": 0.2,
            },
            {
                "name":              "blk.0.ffn_gate.weight",
                "layer":             0,
                "iteration":         0,
                "plan_id":           "p0",
                "sensitivity_score": 0.5 * 0.3 + 0.3 * 0.7 + 0.2 * 0.5,
                "recommended_qtype": "Q4_K",
                "imatrix_magnitude":  0.3,
                "gradient_proxy":     0.7,
                "layer_position_prior": 0.5,
            },
        ]
        outcome_rows = [
            {
                "name":        "blk.0.attn_q.weight",
                "layer":       0,
                "iteration":   0,
                "plan_id":     "p0",
                "strategy":    "A",
                "mse_before":  0.010,
                "mse_after":   0.009,
                "frob_before": 0.010,
                "frob_after":  0.009,
                "family":      "attn_q",
            },
            {
                "name":        "blk.0.ffn_gate.weight",
                "layer":       0,
                "iteration":   0,
                "plan_id":     "p0",
                "strategy":    "A",
                "mse_before":  0.020,
                "mse_after":   0.022,
                "frob_before": 0.020,
                "frob_after":  0.022,
                "family":      "ffn_gate",
            },
        ]
        with TesseraDB.open(path) as db:
            db.insert_l5_plan(model_hash="comp", rows=plan_rows)
            db.insert_l4_plan_outcome(model_hash="comp", rows=outcome_rows)
        # sync-on-exit drained both buffers.

        verdict = l5o.compute_l5_outcome(
            path, model_hash="comp", write_back=True,
        )
        self.assertEqual(verdict.height, 2)
        # The component columns are surfaced on the returned
        # DataFrame. Sort by name for a deterministic order.
        rows = sorted(verdict.to_dicts(), key=lambda r: r["name"])
        # attn_q: im=0.6, grad=0.4, layer=0.2.
        self.assertAlmostEqual(rows[0]["imatrix_magnitude"], 0.6, places=6)
        self.assertAlmostEqual(rows[0]["gradient_proxy"], 0.4, places=6)
        self.assertAlmostEqual(rows[0]["layer_position_prior"], 0.2, places=6)
        # ffn_gate: im=0.3, grad=0.7, layer=0.5.
        self.assertAlmostEqual(rows[1]["imatrix_magnitude"], 0.3, places=6)
        self.assertAlmostEqual(rows[1]["gradient_proxy"], 0.7, places=6)
        self.assertAlmostEqual(rows[1]["layer_position_prior"], 0.5, places=6)
        # The components also land on the l5_outcome table.
        self.assertEqual(_count(path, "l5_outcome"), 2)
        import duckdb as _dd
        con = _dd.connect(path, read_only=True)
        try:
            dbr = con.execute(
                "SELECT name, imatrix_magnitude, gradient_proxy, "
                "layer_position_prior FROM l5_outcome "
                "WHERE model_hash = 'comp' ORDER BY name"
            ).fetchall()
        finally:
            con.close()
        self.assertEqual(dbr[0][0], "blk.0.attn_q.weight")
        self.assertAlmostEqual(dbr[0][1], 0.6, places=6)
        self.assertAlmostEqual(dbr[0][2], 0.4, places=6)
        self.assertAlmostEqual(dbr[0][3], 0.2, places=6)
        self.assertEqual(dbr[1][0], "blk.0.ffn_gate.weight")
        self.assertAlmostEqual(dbr[1][1], 0.3, places=6)
        self.assertAlmostEqual(dbr[1][2], 0.7, places=6)
        self.assertAlmostEqual(dbr[1][3], 0.5, places=6)

    def test_pre_phase15_schema_falls_back_gracefully(self) -> None:
        """A DB whose l5_plan_summary lacks the per-tensor
        component columns (pre-Phase-15 schema, C++ side that has
        not yet migrated) is handled by the SELECT fallback. The
        verdict surfaces the components as NULL so the retune can
        fall back to the 2-coefficient OLS on the combined
        sensitivity_score.
        """
        path = self._fresh(7)
        # Build a pre-Phase-15 schema: l5_plan_summary without
        # the per-tensor component columns. The TesseraDB's
        # _ensure_l5_plan_columns() is what would normally add
        # them on a writable connection; we DROP and recreate
        # l5_plan_summary so it lacks the columns entirely.
        import duckdb as _dd
        con = _dd.connect(path)
        try:
            con.execute("DROP TABLE l5_plan_summary")
            con.execute("""
                CREATE TABLE l5_plan_summary (
                    model_hash            TEXT NOT NULL,
                    name                  TEXT NOT NULL,
                    layer                 INTEGER,
                    iteration             INTEGER NOT NULL,
                    plan_id               TEXT NOT NULL,
                    sensitivity_score     DOUBLE,
                    recommended_qtype     TEXT,
                    recommended_alpha     DOUBLE,
                    recommended_clip      DOUBLE,
                    updated_at            TIMESTAMP,
                    PRIMARY KEY (model_hash, name, iteration, plan_id)
                )
            """)
        finally:
            con.close()
        # Write a plan row + matching l4 outcome via raw duckdb
        # (the TesseraDB.insert_l5_plan helper would re-add the
        # columns via ALTER; we want to keep the pre-Phase-15
        # shape for this test). Note: updated_at IS still on the
        # pre-Phase-15 l5_plan_summary (it's not one of the
        # Phase 15 additions); only the per-tensor component
        # columns are missing. Phase 16 added model_role to
        # l4_plan_outcome (not l5_plan_summary in this test's
        # pre-Phase-15 schema), so the l4 insert includes
        # model_role='trunk'.
        con = _dd.connect(path)
        try:
            con.execute(
                "INSERT INTO l5_plan_summary VALUES "
                "('old', 'blk.0.attn_q.weight', 0, 0, 'p0', 0.5, "
                " 'Q4_K', 0.5, 1.0, NULL)"
            )
            con.execute(
                "INSERT INTO l4_plan_outcome VALUES "
                "('old', 'trunk', 'blk.0.attn_q.weight', 0, 0, 'p0', 'A', "
                " 0.5, 0.5, 1.0, 1.0, 4.0, 4.0, 0.01, 0.009, 0.01, 0.009, "
                " 'attn_q', NULL)"
            )
        finally:
            con.close()

        # Run the verdict. The SELECT fallback in
        # _read_l5_plan_safe should kick in; the per-tensor
        # component columns surface as NULL on the verdict.
        verdict = l5o.compute_l5_outcome(
            path, model_hash="old", write_back=False,
        )
        self.assertEqual(verdict.height, 1)
        row = verdict.to_dicts()[0]
        for col in ("imatrix_magnitude", "gradient_proxy",
                    "layer_position_prior"):
            self.assertIsNone(
                row[col],
                f"pre-Phase-15 row should have NULL {col}, "
                f"got {row[col]!r}",
            )
        # The combined sensitivity_score is still present.
        self.assertAlmostEqual(row["sensitivity_score"], 0.5, places=6)

    # ---- Phase 16: model_role propagation ----

    def _seed_plan_outcome_pair(
        self,
        path: str,
        *,
        model_hash: str,
        name: str,
        plan_id: str,
        family: str,
        model_role: str = "trunk",
        sens: float = 0.5,
        im: float | None = 0.5,
        grad: float | None = 0.5,
        layer: float | None = 0.5,
        mse_before: float = 0.01,
        mse_after: float = 0.011,
    ) -> None:
        """Insert one (plan, outcome) pair across the
        l5_plan_summary / l4_plan_outcome tables. Used by
        the model_role tests below."""
        with TesseraDB.open(path) as db:
            db.insert_l5_plan(
                model_hash=model_hash,
                rows=[{
                    "name": name, "layer": 0, "iteration": 0,
                    "plan_id": plan_id, "model_role": model_role,
                    "sensitivity_score": sens,
                    "recommended_qtype": "Q4_K",
                    "recommended_alpha": 0.5,
                    "recommended_clip": 1.0,
                    "imatrix_magnitude": im,
                    "gradient_proxy": grad,
                    "layer_position_prior": layer,
                }],
                model_role=model_role,
            )
            db.insert_l4_plan_outcome(
                model_hash=model_hash,
                rows=[{
                    "name": name, "layer": 0, "iteration": 0,
                    "plan_id": plan_id, "strategy": "noop",
                    "mse_before": mse_before, "mse_after": mse_after,
                    "frob_before": 0.0, "frob_after": 0.0,
                    "family": family,
                }],
            )

    def test_outcome_surfaces_model_role(self) -> None:
        """The l5_outcome projection surfaces model_role."""
        idx = len(self.paths) + 1
        path = self._fresh(idx)
        self.paths.append(path)
        _create_fresh_db(path)
        self._seed_plan_outcome_pair(
            path, model_hash="m1", name="blk.0.attn_q.weight",
            plan_id="p0", family="attn_q", model_role="trunk",
        )
        verdict = l5o.compute_l5_outcome(
            path, model_hash="m1", write_back=False,
        )
        self.assertEqual(verdict.height, 1)
        self.assertIn("model_role", verdict.columns)
        self.assertEqual(verdict["model_role"][0], "trunk")

    def test_outcome_role_filter(self) -> None:
        """``compute_l5_outcome(model_role=...)`` filters the
        SELECT to the given role; the l5_outcome rows
        carry the role."""
        idx = len(self.paths) + 1
        path = self._fresh(idx)
        self.paths.append(path)
        _create_fresh_db(path)
        # Two roles, two plans.
        for role in ("trunk", "dflash"):
            self._seed_plan_outcome_pair(
                path, model_hash="m1",
                name=f"blk.0.attn_q.{role}.weight",
                plan_id=f"p{role}", family="attn_q",
                model_role=role,
            )
        # Filter on trunk only.
        verdict = l5o.compute_l5_outcome(
            path, model_hash="m1", model_role="trunk",
            write_back=False,
        )
        self.assertEqual(verdict.height, 1)
        self.assertEqual(verdict["model_role"][0], "trunk")
        # Filter on dflash only.
        verdict = l5o.compute_l5_outcome(
            path, model_hash="m1", model_role="dflash",
            write_back=False,
        )
        self.assertEqual(verdict.height, 1)
        self.assertEqual(verdict["model_role"][0], "dflash")
        # No role filter: both rows.
        verdict_all = l5o.compute_l5_outcome(
            path, model_hash="m1", write_back=False,
        )
        self.assertEqual(verdict_all.height, 2)

    def test_outcome_role_filter_requires_model_hash(self) -> None:
        """A bare model_role filter (no model_hash) is
        rejected: it would silently mix roles across models.
        """
        idx = len(self.paths) + 1
        path = self._fresh(idx)
        self.paths.append(path)
        _create_fresh_db(path)
        with self.assertRaises(ValueError):
            l5o.compute_l5_outcome(
                path, model_role="dflash", write_back=False,
            )

    def test_outcome_write_back_role_aware_delete(self) -> None:
        """``replace_l5_outcome``'s DELETE key is
        ``(model_hash, model_role)``: writing one role's
        outcome rows does not clobber other roles' rows
        for the same model.
        """
        idx = len(self.paths) + 1
        path = self._fresh(idx)
        self.paths.append(path)
        _create_fresh_db(path)
        for role in ("trunk", "dflash"):
            self._seed_plan_outcome_pair(
                path, model_hash="m1",
                name=f"blk.0.attn_q.{role}.weight",
                plan_id=f"p{role}", family="attn_q",
                model_role=role,
            )
        # First, write all roles for the model.
        l5o.compute_l5_outcome(
            path, model_hash="m1", write_back=True,
        )
        # Now the trunk's row is in l5_outcome with
        # model_role='trunk'; the dflash's row is in
        # l5_outcome with model_role='dflash'.
        import duckdb as _dd
        con = _dd.connect(path, read_only=True)
        try:
            n = con.execute(
                "SELECT COUNT(*) FROM l5_outcome WHERE model_hash = 'm1'"
            ).fetchone()[0]
        finally:
            con.close()
        self.assertEqual(n, 2)
        # Re-run the trunk-only retune. The trunk's row is
        # replaced (1 row), the dflash's row is preserved
        # (1 row). Total stays at 2.
        l5o.compute_l5_outcome(
            path, model_hash="m1", model_role="trunk",
            write_back=True,
        )
        con = _dd.connect(path, read_only=True)
        try:
            rows = con.execute(
                "SELECT model_role, COUNT(*) FROM l5_outcome "
                "WHERE model_hash = 'm1' GROUP BY model_role "
                "ORDER BY model_role"
            ).fetchall()
        finally:
            con.close()
        self.assertEqual(
            rows, [("dflash", 1), ("trunk", 1)],
        )

    def test_outcome_backfills_role_for_pre_phase16_db(self) -> None:
        """A pre-Phase-16 DB (no model_role column on
        l5_plan_summary) still produces l5_outcome rows
        with model_role='trunk' (the backfill)."""
        # Use a unique path that is NOT created via
        # _create_fresh_db (which builds the post-Phase-16
        # schema). The DB file is built from a literal
        # pre-Phase-16 SCHEMA_SQL.
        idx = len(self.paths) + 100
        path = _fresh_path(idx)
        self.paths.append(path)
        PRE_PHASE16_SCHEMA = """
            CREATE TABLE IF NOT EXISTS l5_plan_summary (
                model_hash            TEXT NOT NULL,
                name                  TEXT NOT NULL,
                layer                 INTEGER,
                iteration             INTEGER NOT NULL,
                plan_id               TEXT NOT NULL,
                sensitivity_score     DOUBLE,
                recommended_qtype     TEXT,
                recommended_alpha     DOUBLE,
                recommended_clip      DOUBLE,
                imatrix_magnitude     DOUBLE,
                gradient_proxy        DOUBLE,
                layer_position_prior  DOUBLE,
                updated_at            TIMESTAMP,
                PRIMARY KEY (model_hash, name, iteration, plan_id)
            );
            CREATE TABLE IF NOT EXISTS l4_plan_outcome (
                model_hash           TEXT NOT NULL,
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
                PRIMARY KEY (model_hash, name, iteration, plan_id)
            );
            CREATE TABLE IF NOT EXISTS l5_outcome (
                model_hash            TEXT NOT NULL,
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
                updated_at            TIMESTAMP,
                PRIMARY KEY (model_hash, name, iteration, plan_id)
            );
        """
        import duckdb as _dd
        con = _dd.connect(path)
        try:
            for stmt in PRE_PHASE16_SCHEMA.strip().split(";"):
                s = stmt.strip()
                if s:
                    con.execute(s)
            con.execute(
                "INSERT INTO l5_plan_summary VALUES ("
                "  'm1', 'blk.0.attn_q.weight', 0, 0, 'p0', 0.5, "
                "  'Q4_K', 0.5, 1.0, 0.5, 0.5, 0.5, NULL)"
            )
            con.execute(
                "INSERT INTO l4_plan_outcome VALUES ("
                "  'm1', 'blk.0.attn_q.weight', 0, 0, 'p0', "
                "  'noop', 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, "
                "  0.01, 0.011, 0.0, 0.0, 'attn_q', NULL)"
            )
        finally:
            con.close()
        verdict = l5o.compute_l5_outcome(
            path, model_hash="m1", write_back=False,
        )
        self.assertEqual(verdict.height, 1)
        # The backfill substitutes 'trunk' for the role.
        self.assertIn("model_role", verdict.columns)
        self.assertEqual(verdict["model_role"][0], "trunk")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestL5Outcome)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
