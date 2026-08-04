"""Tests for tools/tessera/l5_retune.py.

The l5_retune module is the consumer-side of the "did this
requant plan reduce error?" feedback loop. It reads
``l5_outcome``, runs a per-(model, family) closed-form OLS on
``delta_mse`` against ``sensitivity_score``, and projects the
result onto the (w_imatrix, w_gradient, w_layer) simplex. The
projection lands in ``l5_weights`` with PRIMARY KEY
(model_hash, family); the orchestrator's next generation reads
the table back via ``--retune-from-db``.

These tests simulate the join side with synthetic l5_outcome
rows and verify:
  1. A well-calibrated family (b near 0): weights stay close
     to the base.
  2. A miscalibrated family (b > 0, low hit rate): weights
     shift from im to gradient.
  3. A family with high hit rate (gate=0): no shift.
  4. A family with n < min_samples: skipped (no row written).
  5. Idempotent re-run: re-running the retune on the same
     l5_outcome yields the same l5_weights (the upsert
     overwrites in place).
  6. ``aggregate_weights`` n_samples-weighted average: when
     the orchestrator reads l5_weights back, the per-family
     rows combine into one tuple with the correct simplex
     projection.
  7. The l5_orchestrator --retune-from-db path: a synthetic
     l5_weights row in a DB is read by the orchestrator's
     main() and overrides the --w-* flag values.

Run as a unittest module. Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from l5_retune import (
    DEFAULT_ALPHA,
    DEFAULT_BASE_WEIGHTS,
    DEFAULT_MIN_SAMPLES,
    RETUNE_SOURCE_TAG,
    aggregate_weights,
    compute_l5_weights,
    read_l5_weights,
    _ols_slope_intercept,
    _project_simplex,
    _retune_family,
)
from tessera_db import TesseraDB


# Minimal schema for the retune flow. Mirrors the C++ side's
# CREATE TABLE statements in tessera-quantize-db.cpp.
SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS l5_outcome (
        model_hash            TEXT NOT NULL,
        name                  TEXT NOT NULL,
        layer                 INTEGER,
        iteration             INTEGER NOT NULL,
        plan_id               TEXT NOT NULL,
        family                TEXT,
        sensitivity_score     DOUBLE,
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
    CREATE TABLE IF NOT EXISTS l5_weights (
        model_hash            TEXT NOT NULL,
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
        PRIMARY KEY (model_hash, family)
    );
"""


def _create_fresh_db(path: str) -> None:
    import duckdb
    con = duckdb.connect(path)
    try:
        for stmt in SCHEMA_SQL.strip().split(";"):
            s = stmt.strip()
            if s:
                con.execute(s)
    finally:
        con.close()


def _seed_l5_outcome(path: str, model_hash: str, rows: list[dict]) -> None:
    with TesseraDB.open(path) as db:
        db.insert_l5_outcome(model_hash=model_hash, rows=rows)


def _count(path: str, table: str) -> int:
    import duckdb
    con = duckdb.connect(path, read_only=True)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def _extract_top_level_json(text: str) -> dict:
    """Find the first parseable top-level JSON object in ``text``.

    The orchestrator's stdout may have a retune-from-db warning
    or other noise before the summary JSON. Brace-count from
    each '{' candidate and return the first one that parses
    cleanly. Raises json.JSONDecodeError when no candidate
    works.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
            # Sanity: the parse should consume the rest of the
            # text (modulo whitespace), not just a fragment.
            if end >= len(text) - 2:
                return obj
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError(
        "no top-level JSON object found in text", text, 0,
    )


class TestL5Retune(unittest.TestCase):
    def setUp(self) -> None:
        self._td = Path(tempfile.mkdtemp(prefix="l5_retune_test_"))
        self.db_path = str(self._td / "tessera.duckdb")
        _create_fresh_db(self.db_path)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._td, ignore_errors=True)

    # ---- 1. Unit: OLS slope/intercept -----------------------

    def test_ols_slope_exact_for_linear_signal(self) -> None:
        """On y = 0.005 * x, OLS recovers a=0, b=0.005 with
        near-zero in-sample loss."""
        a, b, loss = _ols_slope_intercept(
            [0.2, 0.4, 0.6, 0.8],
            [0.001, 0.002, 0.003, 0.004],
        )
        self.assertAlmostEqual(a, 0.0, places=12)
        self.assertAlmostEqual(b, 0.005, places=12)
        self.assertLess(loss, 1e-9)

    def test_ols_slope_constant_x_returns_intercept_only(self) -> None:
        """When all x values are equal, the slope is 0 and the
        intercept is the mean of y."""
        a, b, loss = _ols_slope_intercept(
            [0.5, 0.5, 0.5], [0.1, 0.2, 0.3],
        )
        self.assertAlmostEqual(b, 0.0)
        self.assertAlmostEqual(a, 0.2)
        # residual = (y - mean(y)), mean abs = 0.0666..
        self.assertAlmostEqual(loss, (0.1 + 0.0 + 0.1) / 3, places=6)

    # ---- 2. Unit: simplex projection -----------------------

    def test_simplex_projection_basic(self) -> None:
        """Sum-to-1 is preserved; negatives are clipped to 0."""
        self.assertEqual(
            _project_simplex((0.5, 0.3, 0.2)),
            (0.5, 0.3, 0.2),
        )
        # (-0.5, 1.0, 0.5) -> (0.0, 0.667, 0.333)
        w = _project_simplex((-0.5, 1.0, 0.5))
        self.assertAlmostEqual(w[0], 0.0, places=9)
        self.assertAlmostEqual(w[1], 2.0 / 3.0, places=9)
        self.assertAlmostEqual(w[2], 1.0 / 3.0, places=9)
        self.assertAlmostEqual(sum(w), 1.0, places=9)

    def test_simplex_all_negative_returns_uniform(self) -> None:
        """All-negative input -> uniform (1/3, 1/3, 1/3)."""
        w = _project_simplex((-1.0, -1.0, -1.0))
        self.assertAlmostEqual(w[0], 1.0 / 3.0, places=9)
        self.assertAlmostEqual(w[1], 1.0 / 3.0, places=9)
        self.assertAlmostEqual(w[2], 1.0 / 3.0, places=9)

    # ---- 3. Unit: per-family retune verdict --------------------

    def test_retune_well_calibrated_stays_near_base(self) -> None:
        """A family where sensitivity_score is uninformative
        (slope ~ 0) keeps the base weights. This is the
        'don't fix what isn't broken' guard."""
        v = _retune_family(
            model_hash="m", family="attn_q",
            # random-ish deltas; slope is small relative to noise
            sensitivity=[0.2, 0.4, 0.6, 0.8],
            delta_mse=[0.001, 0.0009, 0.0011, 0.001],
            plan_accepted=[True, True, True, True],
            base_weights=DEFAULT_BASE_WEIGHTS,
            alpha=DEFAULT_ALPHA, min_samples=3,
        )
        self.assertTrue(v.was_acted_on)
        # The slope is small, so the shift is small; the
        # weights stay within ~5% of the base.
        for w, base in zip(v.weights, DEFAULT_BASE_WEIGHTS):
            self.assertLess(abs(w - base), 0.05,
                f"weight {w} too far from base {base}")

    def test_retune_b_positive_shifts_im_to_gradient(self) -> None:
        """b > 0 + low hit rate -> shift mass from
        w_imatrix to w_gradient."""
        v = _retune_family(
            model_hash="m", family="ffn_gate",
            sensitivity=[0.2, 0.4, 0.6, 0.8],
            delta_mse=[0.001, 0.002, 0.003, 0.004],  # b=+0.005
            plan_accepted=[True, True, False, False],  # hit_rate=0.5
            base_weights=DEFAULT_BASE_WEIGHTS,
            alpha=DEFAULT_ALPHA, min_samples=3,
        )
        self.assertTrue(v.was_acted_on)
        # w_im should DROP, w_grad should RISE.
        self.assertLess(v.weights[0], DEFAULT_BASE_WEIGHTS[0])
        self.assertGreater(v.weights[1], DEFAULT_BASE_WEIGHTS[1])
        # w_layer unchanged (the retune doesn't touch the prior).
        self.assertAlmostEqual(v.weights[2], DEFAULT_BASE_WEIGHTS[2], places=9)
        # Sum-to-1 preserved.
        self.assertAlmostEqual(sum(v.weights), 1.0, places=9)
        # Hit rate recorded.
        self.assertAlmostEqual(v.hit_rate, 0.5, places=6)
        # Slope is +0.005.
        self.assertAlmostEqual(v.slope, 0.005, places=6)

    def test_retune_b_negative_shifts_gradient_to_im(self) -> None:
        """b < 0 + low hit rate -> shift mass from
        w_gradient to w_imatrix."""
        v = _retune_family(
            model_hash="m", family="attn_k",
            sensitivity=[0.2, 0.4, 0.6, 0.8],
            delta_mse=[0.004, 0.003, 0.002, 0.001],  # b=-0.005
            plan_accepted=[True, False, False, True],  # hit_rate=0.5
            base_weights=DEFAULT_BASE_WEIGHTS,
            alpha=DEFAULT_ALPHA, min_samples=3,
        )
        self.assertTrue(v.was_acted_on)
        # w_im should RISE, w_grad should DROP.
        self.assertGreater(v.weights[0], DEFAULT_BASE_WEIGHTS[0])
        self.assertLess(v.weights[1], DEFAULT_BASE_WEIGHTS[1])

    def test_retune_high_hit_rate_no_shift(self) -> None:
        """When hit_rate = 1.0, the gate (1 - hit_rate) is 0
        and the weights stay at the base regardless of slope."""
        v = _retune_family(
            model_hash="m", family="token_embd",
            sensitivity=[0.2, 0.4, 0.6, 0.8],
            delta_mse=[0.001, 0.002, 0.003, 0.004],  # b=+0.005
            plan_accepted=[True, True, True, True],   # hit_rate=1.0
            base_weights=DEFAULT_BASE_WEIGHTS,
            alpha=DEFAULT_ALPHA, min_samples=3,
        )
        self.assertTrue(v.was_acted_on)
        for w, base in zip(v.weights, DEFAULT_BASE_WEIGHTS):
            self.assertAlmostEqual(w, base, places=9)

    def test_retune_min_samples_skips(self) -> None:
        """A (model, family) group with n < min_samples is
        skipped (was_acted_on=False) and the base weights
        are returned."""
        v = _retune_family(
            model_hash="m", family="rare_family",
            sensitivity=[0.5, 0.7],
            delta_mse=[0.001, 0.002],
            plan_accepted=[True, False],
            base_weights=DEFAULT_BASE_WEIGHTS,
            alpha=DEFAULT_ALPHA, min_samples=3,
        )
        self.assertFalse(v.was_acted_on)
        # Base weights returned.
        for w, base in zip(v.weights, DEFAULT_BASE_WEIGHTS):
            self.assertAlmostEqual(w, base, places=9)

    # ---- 4. End-to-end: compute_l5_weights on a synthetic DB ----

    def test_compute_l5_weights_writes_per_family(self) -> None:
        """Two families with different calibration quality.
        The miscalibrated family shifts; the well-calibrated
        one stays near the base. The skipped family (n<3)
        doesn't write a row."""
        rows: list[dict] = []

        # Family A: 4 rows, slope=+0.005, hit_rate=0.5 -> shift.
        sens_a = [0.2, 0.4, 0.6, 0.8]
        deltas_a = [0.001, 0.002, 0.003, 0.004]
        acc_a = [True, True, False, False]
        for i, (s, d, a) in enumerate(zip(sens_a, deltas_a, acc_a)):
            rows.append({
                "name": f"blk.0.attn_q.{i}",
                "layer": 0,
                "iteration": 0,
                "plan_id": f"p{i}",
                "family": "attn_q",
                "sensitivity_score": s,
                "mse_before": 0.01,
                "mse_after":  0.01 + d,
                "delta_mse":  d,
                "plan_accepted": a,
                "accept_threshold": 0.0,
            })

        # Family B: 4 rows, slope=+0.005, hit_rate=1.0 -> no shift.
        for i, (s, d) in enumerate(zip(sens_a, deltas_a)):
            rows.append({
                "name": f"blk.0.ffn_gate.{i}",
                "layer": 0,
                "iteration": 0,
                "plan_id": f"q{i}",
                "family": "ffn_gate",
                "sensitivity_score": s,
                "mse_before": 0.01,
                "mse_after":  0.01 + d,
                "delta_mse":  d,
                "plan_accepted": True,
                "accept_threshold": 0.0,
            })

        # Family C: 2 rows (n<min_samples=3) -> skipped, no write.
        for i, s in enumerate([0.3, 0.7]):
            rows.append({
                "name": f"blk.0.token_embd.{i}",
                "layer": 0,
                "iteration": 0,
                "plan_id": f"r{i}",
                "family": "token_embd",
                "sensitivity_score": s,
                "mse_before": 0.01,
                "mse_after":  0.01,
                "delta_mse":  0.0,
                "plan_accepted": True,
                "accept_threshold": 0.0,
            })

        _seed_l5_outcome(self.db_path, "m", rows)

        verdicts = compute_l5_weights(
            self.db_path, model_hash="m",
            write_back=True,
        )
        # 3 groups, 2 acted on.
        self.assertEqual(len(verdicts), 3)
        acted = [v for v in verdicts if v.was_acted_on]
        skipped = [v for v in verdicts if not v.was_acted_on]
        self.assertEqual(len(acted), 2)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].family, "token_embd")

        # attn_q: im dropped, grad rose.
        attn_q = next(v for v in verdicts if v.family == "attn_q")
        self.assertLess(attn_q.weights[0], DEFAULT_BASE_WEIGHTS[0])
        self.assertGreater(attn_q.weights[1], DEFAULT_BASE_WEIGHTS[1])
        # ffn_gate: no shift (hit_rate=1.0 -> gate=0).
        ffn = next(v for v in verdicts if v.family == "ffn_gate")
        for w, base in zip(ffn.weights, DEFAULT_BASE_WEIGHTS):
            self.assertAlmostEqual(w, base, places=9)

        # l5_weights table has 2 rows (one per acted family).
        self.assertEqual(_count(self.db_path, "l5_weights"), 2)

        # Read back, verify the schema.
        df = read_l5_weights(self.db_path, model_hash="m")
        self.assertEqual(df.height, 2)
        self.assertIn("w_imatrix", df.columns)
        self.assertIn("w_gradient", df.columns)
        self.assertIn("w_layer", df.columns)
        # retune_source is the algorithm tag.
        sources = sorted(df["retune_source"].to_list())
        self.assertEqual(sources, [RETUNE_SOURCE_TAG, RETUNE_SOURCE_TAG])

    def test_compute_l5_weights_idempotent(self) -> None:
        """Re-running on the same l5_outcome yields the same
        l5_weights (the upsert overwrites in place)."""
        rows = []
        for i, (s, d) in enumerate(zip(
            [0.2, 0.4, 0.6, 0.8], [0.001, 0.002, 0.003, 0.004],
        )):
            rows.append({
                "name": f"blk.0.attn_q.{i}",
                "layer": 0, "iteration": 0,
                "plan_id": f"p{i}", "family": "attn_q",
                "sensitivity_score": s,
                "mse_before": 0.01, "mse_after": 0.01 + d,
                "delta_mse": d, "plan_accepted": True,
                "accept_threshold": 0.0,
            })
        _seed_l5_outcome(self.db_path, "idem", rows)

        compute_l5_weights(self.db_path, model_hash="idem", write_back=True)
        first = read_l5_weights(self.db_path, model_hash="idem")
        self.assertEqual(first.height, 1)

        compute_l5_weights(self.db_path, model_hash="idem", write_back=True)
        second = read_l5_weights(self.db_path, model_hash="idem")
        self.assertEqual(second.height, 1,
            "re-run replaces, doesn't duplicate")

        # The two runs should produce the same weights.
        self.assertAlmostEqual(
            first["w_imatrix"][0], second["w_imatrix"][0], places=9,
        )
        self.assertAlmostEqual(
            first["w_gradient"][0], second["w_gradient"][0], places=9,
        )

    # ---- 5. aggregate_weights: n_samples-weighted average -------

    def test_aggregate_weights_n_samples_weighted(self) -> None:
        """Two families with different n_samples: the aggregate
        weight is the n_samples-weighted average of the per-family
        rows, projected to the simplex."""
        df = pl.DataFrame({
            "model_hash":  ["m", "m"],
            "family":      ["attn_q", "ffn_gate"],
            "w_imatrix":   [0.6, 0.4],   # 30 samples vs 10 samples
            "w_gradient":  [0.3, 0.5],
            "w_layer":     [0.1, 0.1],
            "n_samples":   [30, 10],
        })
        w_im, w_grad, w_layer = aggregate_weights(
            df, base_weights=DEFAULT_BASE_WEIGHTS,
        )
        # Expected (un-normalized): im = (0.6*30 + 0.4*10)/40 = 0.55
        #                            grad = (0.3*30 + 0.5*10)/40 = 0.35
        #                            layer = (0.1*30 + 0.1*10)/40 = 0.10
        # Sum = 1.0 -> no projection shift.
        self.assertAlmostEqual(w_im, 0.55, places=6)
        self.assertAlmostEqual(w_grad, 0.35, places=6)
        self.assertAlmostEqual(w_layer, 0.10, places=6)
        self.assertAlmostEqual(w_im + w_grad + w_layer, 1.0, places=9)

    def test_aggregate_weights_empty_returns_base(self) -> None:
        """Empty DataFrame -> base weights."""
        df = pl.DataFrame(schema={
            "model_hash":  pl.Utf8,
            "family":      pl.Utf8,
            "w_imatrix":   pl.Float64,
            "w_gradient":  pl.Float64,
            "w_layer":     pl.Float64,
            "n_samples":   pl.Int64,
        })
        w = aggregate_weights(df, base_weights=DEFAULT_BASE_WEIGHTS)
        self.assertEqual(w, DEFAULT_BASE_WEIGHTS)

    # ---- 6. l5_orchestrator --retune-from-db end-to-end ----

    def test_orchestrator_retune_from_db_overrides_flag_weights(self) -> None:
        """A synthetic l5_weights row in a DB is read by the
        orchestrator's main() and overrides the --w-* flag
        values. We test via a subprocess so we exercise the
        actual CLI wiring (not the import-side path)."""
        # Seed l5_weights with a deliberate override: 0.7/0.2/0.1.
        with TesseraDB.open(self.db_path) as db:
            db.insert_l5_weights([{
                "model_hash":    "abc",
                "family":        "attn_q",
                "w_imatrix":     0.7,
                "w_gradient":    0.2,
                "w_layer":       0.1,
                "bias":          0.0,
                "n_samples":     100,
                "in_sample_loss": 0.001,
                "hit_rate":      0.6,
                "retune_source": RETUNE_SOURCE_TAG,
            }])
            # The context manager's __exit__ drains the upsert.

        # Build a minimal L4 report that won't trigger the
        # orchestrator's loop body (we only care about the
        # weights used). 4 tensors, all converged, so the
        # loop runs one iteration and stops.
        l4_report = {
            "tensors": {
                f"blk.{i}.attn_q.weight": {
                    "current_qtype": "Q4_K",
                    "mse": 0.001,
                    "mse_minus_one": 0.002,
                    "n_weights": 4096,
                } for i in range(4)
            }
        }
        l4_path = self._td / "l4.json"
        l4_path.write_text(json.dumps(l4_report))

        # Run the orchestrator with --retune-from-db and the
        # --w-* flags set to the default (0.5, 0.3, 0.2). The
        # DB row should override the flag values.
        result = subprocess.run(
            [sys.executable, str(THIS_DIR / "l5_orchestrator.py"),
             "--l4-report", str(l4_path),
             "--retune-from-db", self.db_path,
             "--model-hash", "abc",
             "--max-iterations", "1",
             "--top-fraction", "0.5"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
            f"orchestrator failed: {result.stderr}")
        # The summary is the last JSON document on stdout.
        out = result.stdout.strip()
        # Find the top-level JSON object (the summary). The
        # orchestrator may print earlier noise; we walk
        # brace-counting from each '{' candidate until we find
        # one that parses cleanly.
        summary = _extract_top_level_json(out)
        weights = summary["weights"]
        # The DB's w_imatrix=0.7 is the dominant signal
        # (n_samples=100), so the aggregate should be close.
        try:
            self.assertAlmostEqual(weights[0], 0.7, places=6)
            self.assertAlmostEqual(weights[1], 0.2, places=6)
            self.assertAlmostEqual(weights[2], 0.1, places=6)
        except AssertionError:
            self.fail(f"weights not overridden by DB: {weights}")
        # The retune source is recorded in the summary.
        self.assertEqual(summary["retune_source"], self.db_path)

    def test_orchestrator_retune_from_db_missing_model_warns(self) -> None:
        """When --retune-from-db is given but the model_hash
        has no l5_weights row, the orchestrator prints a
        warning and falls back to the --w-* flag values."""
        # DB has the table but no rows for "missing".
        l4_report = {
            "tensors": {
                f"blk.{i}.attn_q.weight": {
                    "current_qtype": "Q4_K", "mse": 0.001,
                    "mse_minus_one": 0.002, "n_weights": 4096,
                } for i in range(4)
            }
        }
        l4_path = self._td / "l4.json"
        l4_path.write_text(json.dumps(l4_report))

        result = subprocess.run(
            [sys.executable, str(THIS_DIR / "l5_orchestrator.py"),
             "--l4-report", str(l4_path),
             "--retune-from-db", self.db_path,
             "--model-hash", "missing",
             "--max-iterations", "1",
             "--top-fraction", "0.5"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
            f"orchestrator failed: {result.stderr}")
        self.assertIn("WARN: --retune-from-db", result.stderr)
        # The summary should still print the --w-* flag values.
        out = result.stdout.strip()
        summary = _extract_top_level_json(out)
        weights = summary["weights"]
        self.assertAlmostEqual(weights[0], DEFAULT_BASE_WEIGHTS[0], places=6)
        self.assertAlmostEqual(weights[1], DEFAULT_BASE_WEIGHTS[1], places=6)
        self.assertAlmostEqual(weights[2], DEFAULT_BASE_WEIGHTS[2], places=6)
        # retune_source is None (no DB contribution).
        self.assertIsNone(summary["retune_source"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestL5Retune)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
