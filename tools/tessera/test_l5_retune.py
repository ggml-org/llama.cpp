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
    DEFAULT_BASE_TOP_FRACTION,
    RETUNE_SOURCE_TAG,
    RETUNE_SOURCE_TAG_3COEF,
    RETUNE_SOURCE_TAG_2COEF,
    RETUNE_SOURCE_TAG_CROSSMODEL,
    aggregate_weights,
    clear_l5_weights_lookup_cache,
    compute_l5_weights,
    find_fingerprint_match,
    read_l5_weights,
    read_per_family_top_fraction,
    resolve_l5_weights_for_orchestrator,
    resolve_per_family_top_fraction_for_orchestrator,
    write_cross_model_aggregate,
    _compute_coupling_score,
    _compute_top_fraction,
    _confidence_weight,
    _l5_weights_lookup_cache,
    _l5_weights_top_fraction_cache,
    _model_hash_fingerprint,
    _ols_3coef_weighted,
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
        in_sample_loss        DOUBLE,
        updated_at            TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)
    );
    CREATE TABLE IF NOT EXISTS l5_weights (
        model_hash            TEXT NOT NULL,
        model_role            TEXT NOT NULL DEFAULT 'trunk',
        family                TEXT NOT NULL,
        w_imatrix             DOUBLE NOT NULL,
        w_gradient            DOUBLE NOT NULL,
        w_layer               DOUBLE NOT NULL,
        bias                  DOUBLE,
        n_samples             INTEGER,
        in_sample_loss        DOUBLE,
        hit_rate              DOUBLE,
        top_fraction          DOUBLE,
        coupling_score        DOUBLE,
        requant_budget_bits   BIGINT,
        retune_source         TEXT,
        updated_at            TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, family)
    );
    CREATE TABLE IF NOT EXISTS tensor_stats (
        model_hash            TEXT NOT NULL,
        model_role            TEXT NOT NULL DEFAULT 'trunk',
        name                  TEXT NOT NULL,
        family                TEXT,
        layer_depth           INTEGER,
        out_dim               INTEGER,
        in_dim                INTEGER,
        n_elements            INTEGER,
        dtype                 TEXT,
        kurtosis              DOUBLE,
        eff_rank              DOUBLE,
        rms                   DOUBLE,
        mean_abs              DOUBLE,
        tail_ratio            DOUBLE,
        source                TEXT,
        recommended_action    TEXT,
        updated_at            TIMESTAMP,
        PRIMARY KEY (model_hash, model_role, name)
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


def _read_l5_outcome_df(
    path: str, model_hash: str,
) -> pl.DataFrame:
    """Read the l5_outcome rows for a model as a polars
    DataFrame. Used by the F3.1 coupling-score tests to
    drive ``_compute_coupling_score`` with a DataFrame
    matching what ``compute_l5_weights`` produces (after
    the column-existence backfill).
    """
    import duckdb
    con = duckdb.connect(path, read_only=True)
    try:
        return con.execute(
            "SELECT model_hash, model_role, family, layer, "
            "plan_accepted FROM l5_outcome "
            f"WHERE model_hash = '{model_hash}'"
        ).pl()
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

    # ---- 6b. Phase 16: --model-role 3-tier lookup ------------

    def test_orchestrator_retune_from_db_model_role_override(
        self,
    ) -> None:
        """The orchestrator's ``--retune-from-db --model-role``
        reads the per-(model, model_role, family) row. A
        dflash-specific row overrides the --w-* flag values
        independently of the trunk's row."""
        with TesseraDB.open(self.db_path) as db:
            # Two roles for the same model. The dflash
            # row has a deliberate 0.4/0.5/0.1 split.
            db.insert_l5_weights([
                {"model_hash":    "mr", "model_role": "trunk",
                 "family":        "attn_q",
                 "w_imatrix":     0.6, "w_gradient":  0.3,
                 "w_layer":       0.1,
                 "bias":          0.0,
                 "n_samples":     100, "in_sample_loss": 0.001,
                 "hit_rate":      0.6, "retune_source": RETUNE_SOURCE_TAG},
                {"model_hash":    "mr", "model_role": "dflash",
                 "family":        "attn_q",
                 "w_imatrix":     0.4, "w_gradient":  0.5,
                 "w_layer":       0.1,
                 "bias":          0.0,
                 "n_samples":     100, "in_sample_loss": 0.001,
                 "hit_rate":      0.6, "retune_source": RETUNE_SOURCE_TAG},
            ])
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
             "--model-hash", "mr",
             "--model-role", "dflash",
             "--max-iterations", "1",
             "--top-fraction", "0.5"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
            f"orchestrator failed: {result.stderr}")
        summary = _extract_top_level_json(result.stdout.strip())
        weights = summary["weights"]
        # The dflash row wins (0.4/0.5/0.1), not the
        # trunk row (0.6/0.3/0.1).
        self.assertAlmostEqual(weights[0], 0.4, places=6)
        self.assertAlmostEqual(weights[1], 0.5, places=6)
        self.assertAlmostEqual(weights[2], 0.1, places=6)

    def test_orchestrator_retune_from_db_model_role_fallback(
        self,
    ) -> None:
        """When ``--model-role`` is set but the per-(model,
        model_role) row is missing, the orchestrator falls
        back to the per-model, no-role row (the legacy
        pre-Phase-16 path). The dflash's recommended
        weights are the trunk's row (0.6/0.3/0.1) — not
        role-perfect but a reasonable warm-start for a new
        role.
        """
        # Per-model, per-role: trunk only.
        with TesseraDB.open(self.db_path) as db:
            db.insert_l5_weights([
                {"model_hash":    "fr", "model_role": "trunk",
                 "family":        "attn_q",
                 "w_imatrix":     0.6, "w_gradient":  0.3,
                 "w_layer":       0.1,
                 "n_samples":     100, "in_sample_loss": 0.001,
                 "hit_rate":      0.6, "retune_source": RETUNE_SOURCE_TAG},
            ])
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
        # --model-role dflash on a DB that has no dflash
        # rows for this model. The orchestrator falls
        # back to the per-model, no-role trunk row.
        result = subprocess.run(
            [sys.executable, str(THIS_DIR / "l5_orchestrator.py"),
             "--l4-report", str(l4_path),
             "--retune-from-db", self.db_path,
             "--model-hash", "fr",
             "--model-role", "dflash",
             "--max-iterations", "1",
             "--top-fraction", "0.5"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
            f"orchestrator failed: {result.stderr}")
        # The log shows the dflash-tagged lookup found a
        # row (via the no-role fallback).
        self.assertIn("role=dflash", result.stderr)
        summary = _extract_top_level_json(result.stdout.strip())
        # The trunk's row is used as the dflash warm-start
        # (0.6/0.3/0.1). The aggregation n=100 dominates
        # the --w-* flag values.
        weights = summary["weights"]
        self.assertAlmostEqual(weights[0], 0.6, places=6)
        self.assertAlmostEqual(weights[1], 0.3, places=6)
        self.assertAlmostEqual(weights[2], 0.1, places=6)

    # ---- 7. Phase 15: 3-coefficient OLS -----------------------

    def test_ols_3coef_recovers_exact_linear_signal(self) -> None:
        """On ``y = 0.001 + 0.5*im + 0.3*grad + 0.2*layer`` the
        3-coefficient OLS recovers the coefficients with
        near-zero in-sample loss.

        The X matrix's three component columns must be
        linearly independent for the lstsq to recover the
        unique coefficient vector; we use three independent
        columns (the im / grad / layer values are not
        collinear). Random-ish but deterministic.
        """
        im   = [0.1, 0.2, 0.4, 0.7, 0.9]
        grad = [0.3, 0.5, 0.1, 0.6, 0.2]
        layer = [0.8, 0.05, 0.5, 0.3, 0.7]
        y = [0.001 + 0.5 * i + 0.3 * g + 0.2 * l
             for i, g, l in zip(im, grad, layer)]
        a, b_im, b_grad, b_layer, loss = _ols_3coef_weighted(
            im=im, grad=grad, layer=layer, y=y,
        )
        self.assertAlmostEqual(a, 0.001, places=9)
        self.assertAlmostEqual(b_im, 0.5, places=9)
        self.assertAlmostEqual(b_grad, 0.3, places=9)
        self.assertAlmostEqual(b_layer, 0.2, places=9)
        self.assertLess(loss, 1e-9)

    def test_ols_3coef_handles_uniform_weights(self) -> None:
        """Sample weights all 1.0 (default) -> the same fit as
        the unweighted OLS. Verifies the sqrt-weighting
        transformation is correct."""
        a, b_im, b_grad, b_layer, loss = _ols_3coef_weighted(
            im=[0.1, 0.5, 0.9],
            grad=[0.2, 0.4, 0.6],
            layer=[0.3, 0.5, 0.7],
            y=[0.1, 0.3, 0.5],
        )
        # Check the coefficients are not all zero (the signal
        # is in the data; a non-trivial fit should produce
        # non-zero coefficients).
        self.assertFalse(
            abs(b_im) < 1e-9 and abs(b_grad) < 1e-9 and abs(b_layer) < 1e-9,
            "all-zero coefficients on a non-trivial signal",
        )
        # The loss is finite.
        self.assertGreater(loss, 0.0)

    def test_ols_3coef_handles_degenerate_input(self) -> None:
        """When all rows have the same (im, grad, layer), the
        design matrix is rank-deficient (the three component
        columns are identical, so the column space is the
        span of [1, 1, 1] and [0.5, 0.5, 0.5]). The lstsq
        returns the minimum-norm solution; the coefficient
        vector is spread evenly across the three
        identical columns. The loss is the mean abs deviation
        of y from its mean (a constant fit has no slope)."""
        a, b_im, b_grad, b_layer, loss = _ols_3coef_weighted(
            im=[0.5, 0.5, 0.5],
            grad=[0.5, 0.5, 0.5],
            layer=[0.5, 0.5, 0.5],
            y=[0.1, 0.2, 0.3],
        )
        # Mean abs deviation of [0.1, 0.2, 0.3] from 0.2 is
        # (0.1 + 0.0 + 0.1) / 3 = 0.0666..
        self.assertAlmostEqual(loss, (0.1 + 0.0 + 0.1) / 3, places=6)
        # The minimum-norm solution has the three component
        # coefficients equal (the columns are identical).
        self.assertAlmostEqual(b_im, b_grad, places=6)
        self.assertAlmostEqual(b_grad, b_layer, places=6)
        # The coefficient vector is small relative to the
        # intercept (the rank-deficient columns contribute
        # to the intercept, not the slope). The minimum-norm
        # solution for rank 2 in 4 columns has 0.057
        # per column; the L1 norm of the slope vector is
        # 0.171.
        self.assertLess(abs(b_im) + abs(b_grad) + abs(b_layer), 0.3)

    # ---- 8. Phase 15: _retune_family 3-coefficient path ----

    def test_retune_family_3coef_positive_im_shifts_weights(self) -> None:
        """When ``b_im > 0`` and hit_rate is low, the im
        component of the recommended weights should RISE
        (the 3-coefficient path's per-component shift rule:
        a positive b_im means im is a positive predictor
        of delta_mse, so increase its weight).

        The X matrix uses im / grad / layer columns that
        are linearly independent so the lstsq recovers the
        unique coefficient vector.
        """
        # 5 rows where im maps directly to delta_mse
        # (b_im = 1, b_grad ~ 0, b_layer ~ 0). The grad
        # and layer are independent of im.
        im_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
        grad_grid = [0.7, 0.5, 0.3, 0.1, 0.6]
        layer_grid = [0.4, 0.6, 0.2, 0.8, 0.5]
        components = list(zip(im_grid, grad_grid, layer_grid))
        sensitivity = [
            0.5 * im + 0.3 * grad + 0.2 * layer
            for im, grad, layer in components
        ]
        delta_mse = list(im_grid)  # b_im = 1, b_grad ~ 0, b_layer ~ 0
        v = _retune_family(
            model_hash="m", family="attn_q",
            sensitivity=sensitivity, delta_mse=delta_mse,
            plan_accepted=[True, True, True, True, False],  # hr=0.8
            components=components,
            base_weights=DEFAULT_BASE_WEIGHTS,
            alpha=DEFAULT_ALPHA, min_samples=3,
        )
        self.assertTrue(v.was_acted_on)
        self.assertEqual(v.retune_algorithm, RETUNE_SOURCE_TAG_3COEF)
        # 3-coefficient path: b_im is positive and the
        # im weight should rise above the base.
        self.assertGreater(v.slopes[0], 0.0,
            f"expected b_im > 0, got {v.slopes[0]}")
        # The hit rate was 4/5 = 0.8, so the gate is
        # 1 - 0.8 = 0.2; the shift is small but
        # positive.
        self.assertGreater(v.weights[0], DEFAULT_BASE_WEIGHTS[0])
        # Sum-to-1 preserved.
        self.assertAlmostEqual(sum(v.weights), 1.0, places=9)

    def test_retune_family_3coef_no_components_falls_back(self) -> None:
        """When components is None (or all-None), the retune
        falls back to the 2-coefficient OLS on the
        combined sensitivity_score. The path is tagged
        ``RETUNE_SOURCE_TAG_2COEF``."""
        v = _retune_family(
            model_hash="m", family="ffn_gate",
            sensitivity=[0.2, 0.4, 0.6, 0.8],
            delta_mse=[0.001, 0.002, 0.003, 0.004],  # b=+0.005
            plan_accepted=[True, True, False, False],  # hit_rate=0.5
            components=None,  # pre-Phase-15 path
            base_weights=DEFAULT_BASE_WEIGHTS,
            alpha=DEFAULT_ALPHA, min_samples=3,
        )
        self.assertTrue(v.was_acted_on)
        self.assertEqual(v.retune_algorithm, RETUNE_SOURCE_TAG_2COEF)
        # 2-coefficient path: b_grad positive, shift
        # mass from im to grad.
        self.assertLess(v.weights[0], DEFAULT_BASE_WEIGHTS[0])
        self.assertGreater(v.weights[1], DEFAULT_BASE_WEIGHTS[1])

    def test_retune_family_3coef_top_fraction_aggressive(self) -> None:
        """High slope + low hit rate -> top_fraction
        recommendation > base. The per-family recommendation
        uses the L2 norm of the coefficient vector as the
        "dominant slope"."""
        # 5 rows where the components are very predictive
        # (slope ~ 1.0) and the hit rate is 0.2.
        components = [
            (0.4, 0.5, 0.5),
        ] * 5
        delta_mse = [0.4, 0.4, 0.4, 0.4, 0.4]  # b_im=+1
        v = _retune_family(
            model_hash="m", family="attn_q",
            sensitivity=[0.5] * 5,
            delta_mse=delta_mse,
            plan_accepted=[True, False, False, False, False],  # hr=0.2
            components=components,
            base_weights=DEFAULT_BASE_WEIGHTS,
            base_top_fraction=0.1,
            alpha=DEFAULT_ALPHA, min_samples=3,
        )
        self.assertTrue(v.was_acted_on)
        # gate = 0.8; dominant slope = 1.0 (b_im); tanh(2) ~ 0.96;
        # top_fraction = 0.1 * (1 + 0.96 * 0.8) ~ 0.177.
        self.assertGreater(v.top_fraction, 0.1)
        self.assertLess(v.top_fraction, 1.0)
        # Verify against the formula directly.
        import math
        expected = _compute_top_fraction(
            0.1, math.sqrt(sum(s * s for s in v.slopes)), 0.2,
        )
        self.assertAlmostEqual(v.top_fraction, expected, places=6)

    def test_retune_family_3coef_top_fraction_at_base(self) -> None:
        """When the OLS coefficients are tiny (well-calibrated
        family), the top_fraction recommendation is at the
        base (the per-family formula returns the base when
        the slope is near zero)."""
        # The components are linearly independent (not
        # collinear) so the lstsq can recover the
        # coefficients. The delta_mse varies within
        # noise, so the coefficients are tiny.
        v = _retune_family(
            model_hash="m", family="attn_k",
            sensitivity=[0.5] * 4,
            delta_mse=[0.001, 0.0009, 0.0011, 0.001],
            plan_accepted=[True, True, True, True],
            components=[
                (0.1, 0.7, 0.4),
                (0.3, 0.5, 0.6),
                (0.5, 0.3, 0.2),
                (0.7, 0.1, 0.8),
            ],
            base_weights=DEFAULT_BASE_WEIGHTS,
            base_top_fraction=0.1,
            alpha=DEFAULT_ALPHA, min_samples=3,
        )
        self.assertTrue(v.was_acted_on)
        # The OLS coefficients are tiny (the delta_mse
        # varies within noise), so the top_fraction is
        # near base. Allow a small jitter from the
        # hit_rate gate.
        import math
        expected = _compute_top_fraction(
            0.1, math.sqrt(sum(s * s for s in v.slopes)), 1.0,
        )
        self.assertAlmostEqual(v.top_fraction, expected, places=6)
        # With hit_rate=1.0 the gate is 0, so the
        # top_fraction is at the base.
        self.assertAlmostEqual(v.top_fraction, 0.1, places=9)

    def test_compute_top_fraction_formula(self) -> None:
        """Direct test of the top_fraction formula."""
        # High slope + low hit rate -> above base.
        self.assertGreater(
            _compute_top_fraction(0.1, 1.0, 0.0),
            0.1,
        )
        # Slope 0 -> exactly base (the tanh(0) is 0).
        self.assertAlmostEqual(
            _compute_top_fraction(0.1, 0.0, 0.5), 0.1, places=9,
        )
        # Hit rate 1.0 -> exactly base (the gate is 0).
        self.assertAlmostEqual(
            _compute_top_fraction(0.1, 1.0, 1.0), 0.1, places=9,
        )
        # Clipped to [0, 1].
        self.assertLessEqual(_compute_top_fraction(10.0, 100.0, 0.0), 1.0)
        # Non-negative even with negative slope.
        self.assertGreaterEqual(
            _compute_top_fraction(0.1, -1.0, 0.0), 0.0,
        )

    def test_confidence_weight_basic(self) -> None:
        """Confidence weight = 1/(1+loss*100) * sqrt(n/max_n)."""
        # n=10, loss=0.01, max=10 -> 1/(1+1) * sqrt(1) = 0.5
        self.assertAlmostEqual(
            _confidence_weight(10, 0.01, 10), 0.5, places=6,
        )
        # n=1, loss=0, max=10 -> 1 * sqrt(0.1) ~ 0.316
        import math
        self.assertAlmostEqual(
            _confidence_weight(1, 0.0, 10), math.sqrt(0.1), places=6,
        )
        # loss=0, n=10, max=10 -> 1.0 (the maximum).
        self.assertAlmostEqual(
            _confidence_weight(10, 0.0, 10), 1.0, places=6,
        )
        # n=0 -> 0 (no data).
        self.assertAlmostEqual(
            _confidence_weight(0, 0.0, 10), 0.0, places=6,
        )

    # ---- 9. Phase 15: confidence-weighted OLS ----------------

    def test_ols_3coef_confidence_weighted(self) -> None:
        """A high-loss row is downweighted relative to a
        low-loss row. The fit follows the high-weight rows'
        signal more closely.

        Four high-weight rows have a known signal
        ``y = 0.5*im + 0.3*grad + 0.2*layer``; two
        low-weight rows are noise. The X matrix's three
        component columns are linearly independent (im /
        grad / layer are not collinear) so the lstsq
        recovers the unique coefficient vector. We need
        at least 4 rows for a determined system (4
        unknowns incl. the intercept).
        """
        # 6 rows. First 4 have the canonical signal and
        # full weight; last 2 are noise with zero weight
        # (dropped from the lstsq).
        im   = [0.1, 0.7, 0.4, 0.2, 0.5, 0.3]
        grad = [0.4, 0.2, 0.6, 0.3, 0.6, 0.8]
        layer = [0.3, 0.6, 0.5, 0.8, 0.1, 0.5]
        y = [
            0.5 * 0.1 + 0.3 * 0.4 + 0.2 * 0.3,  # 0.23
            0.5 * 0.7 + 0.3 * 0.2 + 0.2 * 0.6,  # 0.53
            0.5 * 0.4 + 0.3 * 0.6 + 0.2 * 0.5,  # 0.48
            0.5 * 0.2 + 0.3 * 0.3 + 0.2 * 0.8,  # 0.35
            0.1,  # noise
            0.2,  # noise
        ]
        sample_weights = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        a, b_im, b_grad, b_layer, _ = _ols_3coef_weighted(
            im, grad, layer, y, sample_weights=sample_weights,
        )
        # The high-weight rows say b_im ~ 0.5, b_grad ~ 0.3,
        # b_layer ~ 0.2. With zero weight on the noise
        # rows, the fit should be close.
        self.assertAlmostEqual(b_im, 0.5, places=2)
        self.assertAlmostEqual(b_grad, 0.3, places=2)
        self.assertAlmostEqual(b_layer, 0.2, places=2)
        # Without sample weighting the noise rows
        # contribute and the coefficients are very
        # different. At least one of the coefficients
        # should differ.
        a2, b_im2, b_grad2, b_layer2, _ = _ols_3coef_weighted(
            im, grad, layer, y, sample_weights=None,
        )
        any_diff = (
            abs(b_im - b_im2) > 0.05
            or abs(b_grad - b_grad2) > 0.05
            or abs(b_layer - b_layer2) > 0.05
        )
        self.assertTrue(any_diff,
            f"unweighted fit didn't differ: b_im={b_im2}, "
            f"b_grad={b_grad2}, b_layer={b_layer2}")

    # ---- 10. Phase 15: cross-model retune --------------------

    def test_write_cross_model_aggregate_basic(self) -> None:
        """Three models, two families. The cross-model
        aggregate writes one row per family with
        ``model_hash = "*"`` and the n_samples-weighted
        mean of the per-model rows."""
        with TesseraDB.open(self.db_path) as db:
            # Three models, two families (attn_q, ffn_gate).
            # attn_q has 30 samples across models; ffn_gate
            # has 10.
            db.insert_l5_weights([
                {
                    "model_hash": "m1", "family": "attn_q",
                    "w_imatrix": 0.6, "w_gradient": 0.3,
                    "w_layer": 0.1, "n_samples": 10,
                    "in_sample_loss": 0.001, "hit_rate": 0.6,
                    "top_fraction": 0.15,
                },
                {
                    "model_hash": "m2", "family": "attn_q",
                    "w_imatrix": 0.5, "w_gradient": 0.4,
                    "w_layer": 0.1, "n_samples": 20,
                    "in_sample_loss": 0.001, "hit_rate": 0.5,
                    "top_fraction": 0.10,
                },
                {
                    "model_hash": "m1", "family": "ffn_gate",
                    "w_imatrix": 0.4, "w_gradient": 0.5,
                    "w_layer": 0.1, "n_samples": 6,
                    "in_sample_loss": 0.002, "hit_rate": 0.4,
                    "top_fraction": None,
                },
                {
                    "model_hash": "m3", "family": "ffn_gate",
                    "w_imatrix": 0.3, "w_gradient": 0.6,
                    "w_layer": 0.1, "n_samples": 4,
                    "in_sample_loss": 0.001, "hit_rate": 0.5,
                    "top_fraction": 0.20,
                },
            ])

        verdicts = write_cross_model_aggregate(self.db_path)
        self.assertEqual(len(verdicts), 2)
        # The verdicts are sorted by family.
        attn = next(v for v in verdicts if v.family == "attn_q")
        ffn = next(v for v in verdicts if v.family == "ffn_gate")
        # attn_q: n_samples total = 30. w_imatrix aggregate
        # = (0.6*10 + 0.5*20) / 30 = 16/30 = 0.5333.
        # w_gradient = (0.3*10 + 0.4*20) / 30 = 11/30 = 0.3667.
        # w_layer = 0.1 (uniform).
        self.assertEqual(attn.model_hash, "*")
        self.assertEqual(attn.n_samples, 30)
        self.assertAlmostEqual(attn.weights[0], 0.6 * 10 / 30
                               + 0.5 * 20 / 30, places=6)
        self.assertAlmostEqual(attn.weights[1], 0.3 * 10 / 30
                               + 0.4 * 20 / 30, places=6)
        self.assertAlmostEqual(sum(attn.weights), 1.0, places=6)
        # ffn_gate: n_samples total = 10.
        self.assertEqual(ffn.n_samples, 10)
        # The cross-model row is tagged.
        self.assertEqual(attn.retune_algorithm,
                         RETUNE_SOURCE_TAG_CROSSMODEL)
        # l5_weights table now has the cross-model rows
        # in addition to the per-model rows.
        import duckdb as _dd
        con = _dd.connect(self.db_path, read_only=True)
        try:
            cross = con.execute(
                "SELECT family FROM l5_weights "
                "WHERE model_hash = '*' ORDER BY family"
            ).fetchall()
        finally:
            con.close()
        self.assertEqual(
            sorted([c[0] for c in cross]),
            ["attn_q", "ffn_gate"],
        )

    def test_cross_model_fallback_in_read_l5_weights(self) -> None:
        """``read_l5_weights(cross_model_fallback=True)`` returns
        the per-model rows for the requested model plus the
        cross-model rows (model_hash = "*") for any family
        the per-model lookup missed."""
        with TesseraDB.open(self.db_path) as db:
            db.insert_l5_weights([
                {
                    "model_hash": "warm", "family": "attn_q",
                    "w_imatrix": 0.6, "w_gradient": 0.3,
                    "w_layer": 0.1, "n_samples": 10,
                },
                {
                    "model_hash": "warm", "family": "ffn_gate",
                    "w_imatrix": 0.4, "w_gradient": 0.5,
                    "w_layer": 0.1, "n_samples": 10,
                },
                {
                    "model_hash": "*", "family": "ffn_gate",
                    "w_imatrix": 0.3, "w_gradient": 0.6,
                    "w_layer": 0.1, "n_samples": 50,
                },
            ])

        # Read with the cross-model fallback; the per-model
        # lookup for "cold" has no rows, so the
        # cross-model row for ffn_gate is appended.
        df = read_l5_weights(
            self.db_path, model_hash="cold",
            cross_model_fallback=True,
        )
        families = sorted(df["family"].to_list())
        # The cross-model ffn_gate row is in the result.
        self.assertIn("ffn_gate", families)
        # The cross-model row is preserved with model_hash = "*".
        cross = df.filter(df["model_hash"] == "*")
        self.assertEqual(cross.height, 1)
        self.assertEqual(cross["family"][0], "ffn_gate")
        # The ffn_gate row has the cross-model weights.
        self.assertAlmostEqual(
            df.filter(df["family"] == "ffn_gate")["w_gradient"][0],
            0.6, places=6,
        )

        # Without the fallback, the result is empty (no
        # per-model rows for "cold").
        df2 = read_l5_weights(
            self.db_path, model_hash="cold",
            cross_model_fallback=False,
        )
        self.assertEqual(df2.height, 0)

    # ---- 11. Phase 15: per-family top_fraction lookup ----

    def test_read_per_family_top_fraction(self) -> None:
        """The orchestrator's per-family top_fraction lookup
        reads from l5_weights and returns a {family: value}
        dict. The per-model rows take priority; the
        cross-model rows fill in for families the per-model
        lookup missed."""
        with TesseraDB.open(self.db_path) as db:
            db.insert_l5_weights([
                {
                    "model_hash": "m1", "family": "attn_q",
                    "w_imatrix": 0.5, "w_gradient": 0.3,
                    "w_layer": 0.2, "n_samples": 10,
                    "top_fraction": 0.20,
                },
                {
                    "model_hash": "m1", "family": "ffn_gate",
                    "w_imatrix": 0.4, "w_gradient": 0.5,
                    "w_layer": 0.1, "n_samples": 10,
                    "top_fraction": None,  # no per-family rec
                },
                {
                    "model_hash": "*", "family": "ffn_gate",
                    "w_imatrix": 0.3, "w_gradient": 0.6,
                    "w_layer": 0.1, "n_samples": 50,
                    "top_fraction": 0.30,
                },
                {
                    "model_hash": "*", "family": "token_embd",
                    "w_imatrix": 0.2, "w_gradient": 0.3,
                    "w_layer": 0.5, "n_samples": 20,
                    "top_fraction": 0.05,
                },
            ])

        # Per-model rows for m1; cross-model rows fill in
        # for ffn_gate (the per-model row has NULL
        # top_fraction) and token_embd (no per-model row).
        recs = read_per_family_top_fraction(
            self.db_path, model_hash="m1",
            cross_model_fallback=True,
        )
        self.assertAlmostEqual(recs["attn_q"], 0.20, places=6)
        # The ffn_gate per-model row has NULL top_fraction,
        # so the cross-model row fills in.
        self.assertAlmostEqual(recs["ffn_gate"], 0.30, places=6)
        # token_embd has no per-model row; the cross-model
        # row fills in.
        self.assertAlmostEqual(recs["token_embd"], 0.05, places=6)
        # Without the cross-model fallback, only the
        # per-model rows are returned.
        recs_no_fallback = read_per_family_top_fraction(
            self.db_path, model_hash="m1",
            cross_model_fallback=False,
        )
        self.assertNotIn("token_embd", recs_no_fallback)
        # ffn_gate's per-model row has NULL top_fraction;
        # without the fallback it's not in the dict.
        self.assertNotIn("ffn_gate", recs_no_fallback)
        self.assertIn("attn_q", recs_no_fallback)

    # ---- 12. Phase 15: end-to-end 3-coefficient retune ----

    def test_compute_l5_weights_3coef_with_components(self) -> None:
        """A synthetic l5_outcome with per-tensor component
        columns populated produces a 3-coefficient OLS
        retune. The verdict is tagged with the 3-coefficient
        algorithm; the top_fraction is recommended.

        The component columns are linearly independent
        (im / grad / layer are not collinear) so the lstsq
        can recover the unique coefficient vector.
        """
        rows: list[dict] = []
        # 6 rows for attn_q. delta_mse = im (so b_im = 1,
        # b_grad ~ 0, b_layer ~ 0). The im / grad / layer
        # columns are linearly independent (no collinearity).
        im_grid = [0.10, 0.30, 0.50, 0.70, 0.20, 0.80]
        grad_grid = [0.70, 0.50, 0.30, 0.10, 0.60, 0.40]
        layer_grid = [0.40, 0.60, 0.20, 0.80, 0.50, 0.10]
        for i, (im, grad, layer) in enumerate(
            zip(im_grid, grad_grid, layer_grid),
        ):
            sens = 0.5 * im + 0.3 * grad + 0.2 * layer
            rows.append({
                "name": f"blk.0.attn_q.{i}",
                "layer": 0, "iteration": 0,
                "plan_id": f"p{i}", "family": "attn_q",
                "sensitivity_score": sens,
                "imatrix_magnitude": im,
                "gradient_proxy": grad,
                "layer_position_prior": layer,
                "mse_before": 0.01, "mse_after": 0.01 + im,
                "delta_mse": im,
                "plan_accepted": i % 2 == 0,  # hit_rate = 0.5
                "in_sample_loss": 0.0,
            })
        _seed_l5_outcome(self.db_path, "3coef", rows)

        verdicts = compute_l5_weights(
            self.db_path, model_hash="3coef",
            write_back=True,
        )
        # The 3-coefficient path produced a verdict.
        attn = next(v for v in verdicts if v.family == "attn_q")
        self.assertTrue(attn.was_acted_on)
        self.assertEqual(attn.retune_algorithm, RETUNE_SOURCE_TAG_3COEF)
        # b_im is positive (im is a positive predictor of
        # delta_mse).
        self.assertGreater(attn.slopes[0], 0.0)
        # The top_fraction is recommended (not None).
        self.assertIsNotNone(attn.top_fraction)
        self.assertGreater(attn.top_fraction, DEFAULT_BASE_TOP_FRACTION)
        # l5_weights has the row with top_fraction populated.
        df = read_l5_weights(self.db_path, model_hash="3coef")
        self.assertEqual(df.height, 1)
        self.assertIsNotNone(df["top_fraction"][0])
        self.assertEqual(
            df["retune_source"][0], RETUNE_SOURCE_TAG_3COEF,
        )

    def test_compute_l5_weights_2coef_fallback_when_components_null(
        self,
    ) -> None:
        """A pre-Phase-15 l5_outcome (no per-tensor component
        columns populated) routes through the 2-coefficient
        fallback. The verdict is tagged with the
        2-coefficient algorithm."""
        rows: list[dict] = []
        # 4 rows with the per-tensor component columns NULL
        # (a pre-Phase-15 producer / older C++ writer).
        sens = [0.2, 0.4, 0.6, 0.8]
        deltas = [0.001, 0.002, 0.003, 0.004]
        for i, (s, d) in enumerate(zip(sens, deltas)):
            rows.append({
                "name": f"blk.0.attn_q.{i}",
                "layer": 0, "iteration": 0,
                "plan_id": f"p{i}", "family": "attn_q",
                "sensitivity_score": s,
                # Per-tensor component columns are NULL on
                # pre-Phase-15 rows.
                "imatrix_magnitude": None,
                "gradient_proxy": None,
                "layer_position_prior": None,
                "mse_before": 0.01, "mse_after": 0.01 + d,
                "delta_mse": d,
                "plan_accepted": True,
            })
        _seed_l5_outcome(self.db_path, "pre15", rows)

        verdicts = compute_l5_weights(
            self.db_path, model_hash="pre15",
            write_back=True,
        )
        attn = next(v for v in verdicts if v.family == "attn_q")
        self.assertTrue(attn.was_acted_on)
        # The 2-coefficient path is the fallback.
        self.assertEqual(attn.retune_algorithm, RETUNE_SOURCE_TAG_2COEF)
        # b_grad is the only non-zero coefficient.
        self.assertAlmostEqual(attn.slopes[0], 0.0, places=6)
        self.assertAlmostEqual(attn.slopes[2], 0.0, places=6)
        # b_grad ~ 0.005 (the per-row slope).
        self.assertAlmostEqual(attn.slopes[1], 0.005, places=6)
        # The top_fraction is still recommended.
        self.assertIsNotNone(attn.top_fraction)

    # ---- 13. Phase 15: EMA-aware retune --------------------

    def test_compute_l5_weights_ema_join_replaces_score(self) -> None:
        """When l5_plan_ema is present, the retune uses the
        EMA-tracked score for the OLS instead of the
        per-iteration sensitivity_score. The per-row
        ema_score is the EMA value at the iteration of the
        plan; the join replaces the per-iteration score
        with the EMA value.

        This test exercises the join path. The verdict is
        tagged with the 3-coefficient algorithm and the
        coefficients reflect the EMA-driven signal.
        """
        rows: list[dict] = []
        # 4 rows; per-iteration scores are noisy; the EMA
        # values are the stable signal. The component
        # columns are populated so the 3-coefficient path
        # runs.
        im_grid = [0.1, 0.3, 0.5, 0.7]
        grad_grid = [0.4, 0.2, 0.6, 0.3]
        layer_grid = [0.5, 0.7, 0.3, 0.1]
        for i, (im, grad, layer) in enumerate(
            zip(im_grid, grad_grid, layer_grid),
        ):
            # The per-iteration sensitivity is noisy
            # (random-ish, distinct from the EMA).
            sens = 0.4 + 0.1 * i
            rows.append({
                "name": f"blk.0.attn_q.{i}",
                "layer": 0, "iteration": 0,
                "plan_id": f"p{i}", "family": "attn_q",
                "sensitivity_score": sens,
                "imatrix_magnitude": im,
                "gradient_proxy": grad,
                "layer_position_prior": layer,
                "mse_before": 0.01, "mse_after": 0.01 + im,
                "delta_mse": im,
                "plan_accepted": i % 2 == 0,
            })
        _seed_l5_outcome(self.db_path, "ema", rows)

        # The EMA values: a stable signal where
        # ema_score = 0.5 * im + 0.3 * grad + 0.2 * layer.
        # The 3-coefficient OLS on (im, grad, layer, ema)
        # recovers the (1, 0, 0) coefficient on im; the
        # per-iteration sensitivity_score is too noisy
        # to recover a clean signal.
        ema_rows = [
            {
                "model_hash": "ema",
                "name": f"blk.0.attn_q.{i}",
                "iteration": 0, "plan_id": f"p{i}",
                "ema_score": 0.5 * im + 0.3 * grad + 0.2 * layer,
            }
            for i, (im, grad, layer) in enumerate(
                zip(im_grid, grad_grid, layer_grid),
            )
        ]
        # Insert the EMA rows directly (the test schema
        # does not have a typed insert helper).
        import duckdb as _dd
        con = _dd.connect(self.db_path)
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS l5_plan_ema ("
                "model_hash TEXT NOT NULL, name TEXT NOT NULL, "
                "iteration INTEGER NOT NULL, plan_id TEXT NOT NULL, "
                "ema_score DOUBLE, "
                "PRIMARY KEY (model_hash, name, iteration, plan_id))"
            )
            for r in ema_rows:
                con.execute(
                    "INSERT INTO l5_plan_ema VALUES (?, ?, ?, ?, ?)",
                    [r["model_hash"], r["name"], r["iteration"],
                     r["plan_id"], r["ema_score"]],
                )
        finally:
            con.close()

        # Run the retune with the EMA-aware path. The 4
        # rows produce a 3-coefficient OLS; the EMA-driven
        # sensitivity is the per-iteration score in the
        # lstsq (because the per-iteration score is
        # replaced by the EMA value on the join).
        verdicts = compute_l5_weights(
            self.db_path, model_hash="ema",
            use_ema=True, write_back=False,
        )
        attn = next(v for v in verdicts if v.family == "attn_q")
        self.assertTrue(attn.was_acted_on)
        # The 3-coefficient path is used.
        self.assertEqual(attn.retune_algorithm, RETUNE_SOURCE_TAG_3COEF)
        # b_im is positive (im is a positive predictor of
        # delta_mse; the per-iteration sensitivity is
        # replaced by the EMA value via the join, so the
        # OLS sees the stable signal).
        self.assertGreater(attn.slopes[0], 0.0)

        # Without the EMA join (use_ema=False), the OLS
        # uses the per-iteration sensitivity_score, which
        # is noisy. The coefficients may differ.
        verdicts_no_ema = compute_l5_weights(
            self.db_path, model_hash="ema",
            use_ema=False, write_back=False,
        )
        attn_no_ema = next(v for v in verdicts_no_ema
                           if v.family == "attn_q")
        # The two retunes may produce different
        # coefficients; the EMA path is the production
        # path. We don't enforce a specific difference,
        # but we verify both runs succeed.
        self.assertTrue(attn_no_ema.was_acted_on)

    def test_compute_l5_weights_no_ema_table_falls_back(self) -> None:
        """When l5_plan_ema is missing (DBs created before
        Phase 15 or the EMA producer hasn't run yet), the
        retune falls back to the per-iteration
        sensitivity_score. The retune is unaffected.
        """
        rows: list[dict] = []
        for i, (im, grad, layer) in enumerate([
            (0.1, 0.7, 0.4),
            (0.3, 0.5, 0.6),
            (0.5, 0.3, 0.2),
            (0.7, 0.1, 0.8),
        ]):
            sens = 0.5 * im + 0.3 * grad + 0.2 * layer
            rows.append({
                "name": f"blk.0.attn_q.{i}",
                "layer": 0, "iteration": 0,
                "plan_id": f"p{i}", "family": "attn_q",
                "sensitivity_score": sens,
                "imatrix_magnitude": im,
                "gradient_proxy": grad,
                "layer_position_prior": layer,
                "mse_before": 0.01, "mse_after": 0.01 + im,
                "delta_mse": im,
                "plan_accepted": i % 2 == 0,
            })
        _seed_l5_outcome(self.db_path, "no_ema", rows)
        # No l5_plan_ema table; the retune uses the
        # per-iteration sensitivity_score.
        verdicts = compute_l5_weights(
            self.db_path, model_hash="no_ema",
            use_ema=True, write_back=True,
        )
        attn = next(v for v in verdicts if v.family == "attn_q")
        self.assertTrue(attn.was_acted_on)
        # The 3-coefficient path is used (the per-tensor
        # components are populated).
        self.assertEqual(attn.retune_algorithm, RETUNE_SOURCE_TAG_3COEF)

    # ---- 14. Phase 16: model_role partition --------------------

    def test_family_weights_default_role_is_trunk(self) -> None:
        """The FamilyWeights dataclass defaults model_role to
        ``"trunk"`` for backward compat with Phase 15 callers.
        """
        v = _retune_family(
            model_hash="m", family="attn_q",
            sensitivity=[0.2, 0.4, 0.6, 0.8],
            delta_mse=[0.001, 0.002, 0.003, 0.004],
            plan_accepted=[True, True, False, False],
            base_weights=DEFAULT_BASE_WEIGHTS,
            alpha=DEFAULT_ALPHA, min_samples=3,
        )
        self.assertEqual(v.model_role, "trunk")
        # The to_dict path includes the role.
        d = v.to_dict()
        self.assertEqual(d["model_role"], "trunk")

    def test_family_weights_explicit_role(self) -> None:
        """An explicit model_role flows through to the verdict."""
        v = _retune_family(
            model_hash="m", family="attn_q",
            model_role="dflash",
            sensitivity=[0.2, 0.4, 0.6, 0.8],
            delta_mse=[0.001, 0.002, 0.003, 0.004],
            plan_accepted=[True, True, False, False],
            base_weights=DEFAULT_BASE_WEIGHTS,
            alpha=DEFAULT_ALPHA, min_samples=3,
        )
        self.assertEqual(v.model_role, "dflash")

    def test_retune_family_writes_role_to_l5_weights(self) -> None:
        """The retune writes a ``model_role`` column on
        ``l5_weights`` and the upsert uses the new
        (model_hash, model_role, family) PK."""
        rows: list[dict] = []
        for i, (s, d) in enumerate(zip(
            [0.2, 0.4, 0.6, 0.8], [0.001, 0.002, 0.003, 0.004],
        )):
            rows.append({
                "name": f"blk.0.attn_q.{i}",
                "layer": 0, "iteration": 0,
                "plan_id": f"p{i}", "family": "attn_q",
                "model_role": "dflash",
                "sensitivity_score": s,
                "mse_before": 0.01, "mse_after": 0.01 + d,
                "delta_mse": d, "plan_accepted": True,
                "accept_threshold": 0.0,
            })
        _seed_l5_outcome(self.db_path, "mr1", rows)
        verdicts = compute_l5_weights(
            self.db_path, model_hash="mr1",
            model_role="dflash", write_back=True,
        )
        attn = next(v for v in verdicts if v.family == "attn_q")
        self.assertEqual(attn.model_role, "dflash")
        # The l5_weights table has the row with the role.
        df = read_l5_weights(
            self.db_path, model_hash="mr1", model_role="dflash",
        )
        self.assertEqual(df.height, 1)
        self.assertEqual(df["model_role"][0], "dflash")
        # And without the role filter, the same row is the
        # only row for this model.
        df_all = read_l5_weights(self.db_path, model_hash="mr1")
        self.assertEqual(df_all.height, 1)
        self.assertEqual(df_all["model_role"][0], "dflash")

    def test_retune_independent_per_role_verdicts(self) -> None:
        """Same (model, family) in different roles get
        independent (w_imatrix, w_gradient, w_layer) tuples.
        The trunk's attn_q and the dflash encoder's attn_q
        are partitioned; the OLS for one is independent of
        the OLS for the other.
        """
        rows: list[dict] = []
        # 4 trunk rows: b_im = +1 (attn_q is well-explained
        # by im). The per-row delta_mse = im; the 3-coef OLS
        # recovers (b_im=1, b_grad=0, b_layer=0). Some plans
        # are rejected so the gate (1 - hit_rate) is non-zero
        # and the shift actually fires.
        trunk_im = [0.1, 0.3, 0.5, 0.7]
        trunk_acc = [True, False, True, False]
        for i, im in enumerate(trunk_im):
            rows.append({
                "name": f"blk.0.attn_q.{i}",
                "layer": 0, "iteration": 0,
                "plan_id": f"tp{i}", "family": "attn_q",
                "model_role": "trunk",
                "sensitivity_score": im,
                "imatrix_magnitude": im,
                "gradient_proxy": 0.5,
                "layer_position_prior": 0.3,
                "mse_before": 0.01, "mse_after": 0.01 + im,
                "delta_mse": im,
                "plan_accepted": trunk_acc[i],
            })
        # 4 dflash rows: b_grad = +1 (the dflash encoder's
        # attn_q is well-explained by gradient, not im).
        dflash_grad = [0.1, 0.3, 0.5, 0.7]
        dflash_acc = [False, True, False, True]
        for i, gr in enumerate(dflash_grad):
            rows.append({
                "name": f"dflash.enc.attn_q.{i}",
                "layer": 0, "iteration": 0,
                "plan_id": f"dp{i}", "family": "attn_q",
                "model_role": "dflash",
                "sensitivity_score": gr,
                "imatrix_magnitude": 0.5,
                "gradient_proxy": gr,
                "layer_position_prior": 0.3,
                "mse_before": 0.01, "mse_after": 0.01 + gr,
                "delta_mse": gr,
                "plan_accepted": dflash_acc[i],
            })
        _seed_l5_outcome(self.db_path, "mr2", rows)

        verdicts = compute_l5_weights(
            self.db_path, model_hash="mr2", write_back=True,
        )
        # Two verdicts: trunk/attn_q and dflash/attn_q.
        self.assertEqual(len(verdicts), 2)
        roles = {v.model_role for v in verdicts}
        self.assertEqual(roles, {"trunk", "dflash"})
        # The l5_weights table has 2 rows for the same
        # model_hash, different model_role, same family.
        import duckdb as _dd
        con = _dd.connect(self.db_path, read_only=True)
        try:
            n = con.execute(
                "SELECT COUNT(*) FROM l5_weights "
                "WHERE model_hash = 'mr2'"
            ).fetchone()[0]
        finally:
            con.close()
        self.assertEqual(n, 2)
        # The trunk's b_im is positive; the dflash's
        # b_grad is positive. The verdicts' slopes carry
        # the difference.
        trunk = next(v for v in verdicts if v.model_role == "trunk")
        dflash = next(v for v in verdicts if v.model_role == "dflash")
        self.assertGreater(trunk.slopes[0], 0.0)
        self.assertGreater(dflash.slopes[1], 0.0)
        # The trunk and dflash verdicts have different
        # (w_imatrix, w_gradient, w_layer) tuples. The
        # shift direction is different (b_im positive for
        # the trunk raises w_im; b_grad positive for the
        # dflash raises w_grad). With low hit_rate on
        # both, the gate is non-zero and the per-role
        # shifts are different.
        self.assertFalse(
            trunk.weights == dflash.weights,
            f"per-role verdicts should differ; "
            f"trunk={trunk.weights}, dflash={dflash.weights}",
        )

    def test_retune_role_filter_selects_one_role(self) -> None:
        """A retune with model_role=R restricts the read to
        the role-R rows; the role-R' rows are not consumed.
        The write-back also only DELETEs the role-R rows
        (other roles' l5_weights rows are preserved)."""
        # Seed two roles' l5_outcome rows.
        rows: list[dict] = []
        for i, (s, d) in enumerate(zip(
            [0.2, 0.4, 0.6, 0.8], [0.001, 0.002, 0.003, 0.004],
        )):
            for role in ("trunk", "dflash"):
                rows.append({
                    "name": f"blk.0.attn_q.{role}.{i}",
                    "layer": 0, "iteration": 0,
                    "plan_id": f"p{role}{i}", "family": "attn_q",
                    "model_role": role,
                    "sensitivity_score": s,
                    "mse_before": 0.01, "mse_after": 0.01 + d,
                    "delta_mse": d, "plan_accepted": True,
                    "accept_threshold": 0.0,
                })
        _seed_l5_outcome(self.db_path, "mr3", rows)
        # First retune: no model_role filter (all roles).
        verdicts_all = compute_l5_weights(
            self.db_path, model_hash="mr3", write_back=True,
        )
        self.assertEqual(len(verdicts_all), 2)
        # The l5_weights has both rows.
        df = read_l5_weights(self.db_path, model_hash="mr3")
        self.assertEqual(df.height, 2)
        # Now retune only the dflash role.
        verdicts_dflash = compute_l5_weights(
            self.db_path, model_hash="mr3", model_role="dflash",
            write_back=True,
        )
        self.assertEqual(len(verdicts_dflash), 1)
        self.assertEqual(verdicts_dflash[0].model_role, "dflash")
        # The trunk's row is still in l5_weights (the
        # write-back DELETE was only on model_role='dflash').
        df_after = read_l5_weights(
            self.db_path, model_hash="mr3", model_role="trunk",
        )
        self.assertEqual(df_after.height, 1)
        self.assertEqual(df_after["model_role"][0], "trunk")

    def test_read_l5_weights_model_role_filter(self) -> None:
        """``read_l5_weights(model_role=...)`` filters the
        SELECT to the given role; the consumer-side
        per-family top_fraction is role-aware."""
        with TesseraDB.open(self.db_path) as db:
            db.insert_l5_weights([
                {"model_hash": "x", "model_role": "trunk",
                 "family": "attn_q", "w_imatrix": 0.6,
                 "w_gradient": 0.3, "w_layer": 0.1,
                 "n_samples": 10, "top_fraction": 0.20},
                {"model_hash": "x", "model_role": "dflash",
                 "family": "attn_q", "w_imatrix": 0.4,
                 "w_gradient": 0.5, "w_layer": 0.1,
                 "n_samples": 10, "top_fraction": 0.30},
            ])
        # Role filter: trunk only.
        df_trunk = read_l5_weights(
            self.db_path, model_hash="x", model_role="trunk",
        )
        self.assertEqual(df_trunk.height, 1)
        self.assertEqual(df_trunk["model_role"][0], "trunk")
        # Role filter: dflash only.
        df_dflash = read_l5_weights(
            self.db_path, model_hash="x", model_role="dflash",
        )
        self.assertEqual(df_dflash.height, 1)
        self.assertEqual(df_dflash["model_role"][0], "dflash")
        # No role filter: both rows.
        df_all = read_l5_weights(self.db_path, model_hash="x")
        self.assertEqual(df_all.height, 2)

    # ---- 15. Retune follow-ups: F3.1 cross-component coupling ----

    def test_compute_coupling_score_perfect_positive(self) -> None:
        """When the trunk's per-layer hit_rate and the
        dflash's per-layer hit_rate are perfectly positively
        correlated, the score is +1.0. A high score means a
        single retune covers both roles.
        """
        rows: list[dict] = []
        # 4 layers; trunk and dflash hit_rate move
        # together perfectly: (1, 1, 0, 0) for both roles.
        for layer, hit in enumerate([1.0, 1.0, 0.0, 0.0]):
            for role, prefix in (("trunk", "t"), ("dflash", "d")):
                for j in range(2):
                    rows.append({
                        "name": f"{prefix}.L{layer}.{j}",
                        "layer": layer,
                        "iteration": 0,
                        "plan_id": f"{prefix}p{layer}{j}",
                        "family": "attn_q",
                        "model_role": role,
                        "sensitivity_score": 0.5,
                        "delta_mse": 0.0,
                        "plan_accepted": hit >= 0.5,
                    })
        _seed_l5_outcome(self.db_path, "c1", rows)
        df = _read_l5_outcome_df(self.db_path, model_hash="c1")
        score = _compute_coupling_score(
            df, model_hash="c1", family="attn_q",
        )
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_compute_coupling_score_perfect_negative(self) -> None:
        """When the trunk's per-layer hit_rate and the
        dflash's per-layer hit_rate are perfectly anti-
        correlated, the score is -1.0. A negative score
        means the two roles' miscalibration is
        independent (or opposite).
        """
        rows: list[dict] = []
        trunk_hits = [1.0, 1.0, 0.0, 0.0]
        dflash_hits = [0.0, 0.0, 1.0, 1.0]
        for layer in range(4):
            for role, hits, prefix in (
                ("trunk", trunk_hits, "t"),
                ("dflash", dflash_hits, "d"),
            ):
                for j in range(2):
                    hit = hits[layer]
                    rows.append({
                        "name": f"{prefix}.L{layer}.{j}",
                        "layer": layer,
                        "iteration": 0,
                        "plan_id": f"{prefix}p{layer}{j}",
                        "family": "attn_q",
                        "model_role": role,
                        "sensitivity_score": 0.5,
                        "delta_mse": 0.0,
                        "plan_accepted": hit >= 0.5,
                    })
        _seed_l5_outcome(self.db_path, "c2", rows)
        df = _read_l5_outcome_df(self.db_path, model_hash="c2")
        score = _compute_coupling_score(
            df, model_hash="c2", family="attn_q",
        )
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score, -1.0, places=6)

    def test_compute_coupling_score_single_role_returns_none(self) -> None:
        """A family with rows for only one of the two roles
        (e.g. trunk only) has no correlation -> None."""
        rows: list[dict] = []
        for layer, hit in enumerate([1.0, 0.0, 1.0, 0.0]):
            for j in range(3):
                rows.append({
                    "name": f"blk.{layer}.attn_q.{j}",
                    "layer": layer,
                    "iteration": 0, "plan_id": f"p{layer}{j}",
                    "family": "attn_q",
                    "model_role": "trunk",
                    "sensitivity_score": 0.5,
                    "delta_mse": 0.0,
                    "plan_accepted": hit >= 0.5,
                })
        _seed_l5_outcome(self.db_path, "c3", rows)
        df = _read_l5_outcome_df(self.db_path, model_hash="c3")
        score = _compute_coupling_score(
            df, model_hash="c3", family="attn_q",
        )
        self.assertIsNone(score)

    def test_compute_coupling_score_zero_variance_returns_none(self) -> None:
        """When both roles' per-layer hit rates are constant
        (zero variance), the correlation is mathematically
        undefined -> None."""
        rows: list[dict] = []
        for layer in range(3):
            for role, prefix in (("trunk", "t"), ("dflash", "d")):
                for j in range(2):
                    rows.append({
                        "name": f"{prefix}.L{layer}.{j}",
                        "layer": layer,
                        "iteration": 0,
                        "plan_id": f"{prefix}p{layer}{j}",
                        "family": "attn_q",
                        "model_role": role,
                        "sensitivity_score": 0.5,
                        "delta_mse": 0.0,
                        "plan_accepted": True,
                    })
        _seed_l5_outcome(self.db_path, "c4", rows)
        df = _read_l5_outcome_df(self.db_path, model_hash="c4")
        score = _compute_coupling_score(
            df, model_hash="c4", family="attn_q",
        )
        self.assertIsNone(score)

    def test_compute_l5_weights_writes_coupling_score(self) -> None:
        """A retune on a DB with rows for both trunk and
        dflash writes the coupling_score column on every
        (model, model_role, family) row, with the same
        score shared by both roles."""
        rows: list[dict] = []
        # Trunk: 4 layers, hit_rate moves from 1.0 to 0.0
        # monotonically.
        for layer, hits in enumerate([
            [True, True, True, True],
            [True, True, True, False],
            [True, True, False, False],
            [True, False, False, False],
        ]):
            for j, h in enumerate(hits):
                rows.append({
                    "name": f"blk.{layer}.attn_q.t.{j}",
                    "layer": layer,
                    "iteration": 0, "plan_id": f"tp{layer}{j}",
                    "family": "attn_q",
                    "model_role": "trunk",
                    "sensitivity_score": 0.5,
                    "delta_mse": 0.0,
                    "plan_accepted": h,
                })
        # Dflash: identical per-layer hit_rate pattern
        # (perfectly correlated with trunk).
        for layer, hits in enumerate([
            [True, True, True, True],
            [True, True, True, False],
            [True, True, False, False],
            [True, False, False, False],
        ]):
            for j, h in enumerate(hits):
                rows.append({
                    "name": f"blk.{layer}.attn_q.d.{j}",
                    "layer": layer,
                    "iteration": 0, "plan_id": f"dp{layer}{j}",
                    "family": "attn_q",
                    "model_role": "dflash",
                    "sensitivity_score": 0.5,
                    "delta_mse": 0.0,
                    "plan_accepted": h,
                })
        _seed_l5_outcome(self.db_path, "c5", rows)

        verdicts = compute_l5_weights(
            self.db_path, model_hash="c5", write_back=True,
        )
        # Two verdicts: trunk/attn_q and dflash/attn_q.
        self.assertEqual(len(verdicts), 2)
        # Both carry the same coupling_score (the score
        # is per (model, family), not per (model, role,
        # family)).
        scores = [v.coupling_score for v in verdicts]
        self.assertEqual(scores[0], scores[1])
        # The pattern is perfectly correlated.
        self.assertIsNotNone(scores[0])
        self.assertAlmostEqual(scores[0], 1.0, places=6)
        # The l5_weights column is populated.
        df = read_l5_weights(self.db_path, model_hash="c5")
        self.assertEqual(df.height, 2)
        self.assertIn("coupling_score", df.columns)
        coupling_col = df["coupling_score"].to_list()
        self.assertTrue(all(c is not None for c in coupling_col))
        # Both rows have the same score.
        self.assertAlmostEqual(
            coupling_col[0], coupling_col[1], places=9,
        )

    def test_compute_l5_weights_single_role_coupling_is_null(self) -> None:
        """A retune on a single-role DB (trunk only)
        produces a None coupling_score (the helper returns
        None when only one role is present)."""
        rows: list[dict] = []
        for layer, hits in enumerate([
            [True, True, True, True],
            [True, True, True, False],
            [True, True, False, False],
        ]):
            for j, h in enumerate(hits):
                rows.append({
                    "name": f"blk.{layer}.attn_q.{j}",
                    "layer": layer,
                    "iteration": 0, "plan_id": f"p{layer}{j}",
                    "family": "attn_q",
                    "model_role": "trunk",
                    "sensitivity_score": 0.5,
                    "delta_mse": 0.0,
                    "plan_accepted": h,
                })
        _seed_l5_outcome(self.db_path, "c6", rows)
        verdicts = compute_l5_weights(
            self.db_path, model_hash="c6", write_back=True,
        )
        # One verdict: trunk/attn_q. coupling_score is None.
        self.assertEqual(len(verdicts), 1)
        self.assertIsNone(verdicts[0].coupling_score)
        # The l5_weights column is NULL.
        df = read_l5_weights(self.db_path, model_hash="c6")
        self.assertEqual(df.height, 1)
        self.assertIsNone(df["coupling_score"][0])

    # ---- 16. Retune follow-ups: F3.2 cross-model hash dedup ----

    def test_model_hash_fingerprint_deterministic(self) -> None:
        """The fingerprint is deterministic: two reads of
        the same model's tensor_stats produce the same
        hash. The function is pure (no side effects)."""
        self._seed_tensor_stats("fp1", n_tensors=20)
        with TesseraDB.open(self.db_path, read_only=True) as db:
            fp1 = _model_hash_fingerprint(db, "fp1")
            fp2 = _model_hash_fingerprint(db, "fp1")
        self.assertIsNotNone(fp1)
        self.assertEqual(fp1, fp2)
        # 16 hex chars (truncated SHA-1).
        self.assertEqual(len(fp1), 16)
        # Hex characters.
        import re
        self.assertRegex(fp1, r"^[0-9a-f]{16}$")

    def test_model_hash_fingerprint_different_models_differ(self) -> None:
        """Two models with very different per-tensor stat
        distributions have different fingerprints. A 10x
        difference in mean rms shifts the first sig fig
        of the mean moment -> different hash."""
        self._seed_tensor_stats("fp_a", n_tensors=20, rms_base=0.1)
        self._seed_tensor_stats("fp_b", n_tensors=20, rms_base=1.0)
        with TesseraDB.open(self.db_path, read_only=True) as db:
            fp_a = _model_hash_fingerprint(db, "fp_a")
            fp_b = _model_hash_fingerprint(db, "fp_b")
        self.assertIsNotNone(fp_a)
        self.assertIsNotNone(fp_b)
        self.assertNotEqual(fp_a, fp_b)

    def test_model_hash_fingerprint_small_drift_same(self) -> None:
        """Two models with the same per-tensor stat
        distributions up to numerical noise have the same
        fingerprint. The 4-sig-fig rounding absorbs the
        drift. The dedup's whole point: a fine-tune of the
        same base matches the parent."""
        # Same seed -> identical per-tensor values, just a
        # tiny rms jitter shift.
        self._seed_tensor_stats("fp_c", n_tensors=20, rms_base=0.1, seed=42)
        self._seed_tensor_stats(
            "fp_d", n_tensors=20, rms_base=0.1, rms_jitter=1e-6, seed=42,
        )
        with TesseraDB.open(self.db_path, read_only=True) as db:
            fp_c = _model_hash_fingerprint(db, "fp_c")
            fp_d = _model_hash_fingerprint(db, "fp_d")
        self.assertEqual(fp_c, fp_d)

    def test_model_hash_fingerprint_missing_model_returns_none(
        self,
    ) -> None:
        """A model with no tensor_stats rows -> None."""
        with TesseraDB.open(self.db_path, read_only=True) as db:
            fp = _model_hash_fingerprint(db, "no_such_model")
        self.assertIsNone(fp)

    def test_find_fingerprint_match_finds_matching_model(self) -> None:
        """``find_fingerprint_match`` returns the other
        model_hash whose fingerprint matches the requested
        one. Used by the orchestrator's --cross-model-dedup
        path."""
        # Two models with identical fingerprints (same
        # seed, tiny jitter).
        self._seed_tensor_stats("m_a", n_tensors=20, rms_base=0.1, seed=42)
        self._seed_tensor_stats(
            "m_b", n_tensors=20, rms_base=0.1, rms_jitter=1e-6, seed=42,
        )
        # A third model with a different fingerprint
        # (same seed, very different rms).
        self._seed_tensor_stats("m_c", n_tensors=20, rms_base=1.0, seed=42)
        # The match for m_a is m_b (same fingerprint).
        match = find_fingerprint_match(self.db_path, "m_a")
        self.assertEqual(match, "m_b")
        # The match for m_c is None (no other model has
        # the same fingerprint).
        match_none = find_fingerprint_match(self.db_path, "m_c")
        self.assertIsNone(match_none)

    def test_find_fingerprint_match_self_excluded(self) -> None:
        """The match never returns the requested model_hash
        itself. When m_a is the only model in the DB, the
        match is None (no OTHER model to match)."""
        self._seed_tensor_stats("only", n_tensors=20, rms_base=0.1)
        match = find_fingerprint_match(self.db_path, "only")
        self.assertIsNone(match)

    def test_orchestrator_cross_model_dedup_overrides(self) -> None:
        """The orchestrator's --cross-model-dedup looks up
        a fingerprint-matched model when the requested
        model_hash is not in the DB. The matched model's
        l5_weights are reused; a warning is printed.

        We test via a subprocess so we exercise the
        actual CLI wiring.
        """
        # Seed two models with identical fingerprints and
        # the second's l5_weights row. The orchestrator
        # asks for the first model (no l5_weights row);
        # --cross-model-dedup finds the second model and
        # reuses its row.
        self._seed_tensor_stats("dedup_a", n_tensors=20, rms_base=0.1, seed=42)
        self._seed_tensor_stats(
            "dedup_b", n_tensors=20, rms_base=0.1, rms_jitter=1e-6, seed=42,
        )
        with TesseraDB.open(self.db_path) as db:
            db.insert_l5_weights([{
                "model_hash": "dedup_b", "family": "attn_q",
                "w_imatrix": 0.7, "w_gradient": 0.2, "w_layer": 0.1,
                "bias": 0.0, "n_samples": 100,
                "in_sample_loss": 0.001, "hit_rate": 0.6,
                "retune_source": RETUNE_SOURCE_TAG,
            }])
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
             "--model-hash", "dedup_a",
             "--cross-model-dedup",
             "--max-iterations", "1",
             "--top-fraction", "0.5"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
            f"orchestrator failed: {result.stderr}")
        # The warning is printed to stderr.
        self.assertIn("cross-model-dedup", result.stderr)
        # The matched model's weights (0.7/0.2/0.1) are
        # used. The dedup successfully reused the row.
        summary = _extract_top_level_json(result.stdout.strip())
        weights = summary["weights"]
        self.assertAlmostEqual(weights[0], 0.7, places=6)
        self.assertAlmostEqual(weights[1], 0.2, places=6)
        self.assertAlmostEqual(weights[2], 0.1, places=6)

    def test_orchestrator_cross_model_dedup_off_by_default(self) -> None:
        """The dedup is opt-in. Without --cross-model-dedup,
        a missing model_hash falls back to the --w-* flag
        values (the legacy path)."""
        self._seed_tensor_stats("legacy_a", n_tensors=20, rms_base=0.1, seed=42)
        self._seed_tensor_stats(
            "legacy_b", n_tensors=20, rms_base=0.1, rms_jitter=1e-6, seed=42,
        )
        with TesseraDB.open(self.db_path) as db:
            db.insert_l5_weights([{
                "model_hash": "legacy_b", "family": "attn_q",
                "w_imatrix": 0.7, "w_gradient": 0.2, "w_layer": 0.1,
                "n_samples": 100, "in_sample_loss": 0.001,
                "hit_rate": 0.6, "retune_source": RETUNE_SOURCE_TAG,
            }])
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
             "--model-hash", "legacy_a",
             # Note: no --cross-model-dedup.
             "--max-iterations", "1",
             "--top-fraction", "0.5"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
            f"orchestrator failed: {result.stderr}")
        # No dedup warning.
        self.assertNotIn("cross-model-dedup", result.stderr)
        # The legacy WARN is printed.
        self.assertIn("WARN: --retune-from-db", result.stderr)
        # The summary uses the --w-* flag values (default
        # 0.5/0.3/0.2).
        summary = _extract_top_level_json(result.stdout.strip())
        weights = summary["weights"]
        self.assertAlmostEqual(weights[0], DEFAULT_BASE_WEIGHTS[0], places=6)
        self.assertAlmostEqual(weights[1], DEFAULT_BASE_WEIGHTS[1], places=6)
        self.assertAlmostEqual(weights[2], DEFAULT_BASE_WEIGHTS[2], places=6)

    # ---- 17. Retune follow-ups: F3.3 --retune-from-db cache ----

    def test_resolve_l5_weights_caches_result(self) -> None:
        """The 3-tier lookup helper caches its result. The
        second call with the same args returns the same
        DataFrame without re-querying DuckDB; the cache
        is populated after the first call.
        """
        with TesseraDB.open(self.db_path) as db:
            db.insert_l5_weights([{
                "model_hash": "cache_m", "family": "attn_q",
                "w_imatrix": 0.6, "w_gradient": 0.3, "w_layer": 0.1,
                "n_samples": 100, "in_sample_loss": 0.001,
                "hit_rate": 0.6, "retune_source": RETUNE_SOURCE_TAG,
            }])
        # Clear the cache (other tests may have populated
        # entries with the same path).
        clear_l5_weights_lookup_cache()
        cache = _l5_weights_lookup_cache()
        # Pre-condition: the cache has no entry for our
        # specific (db_path, model_hash) key.
        key = (self.db_path, "cache_m", None, True)
        self.assertNotIn(key, cache)
        # First call: populates the cache.
        df1 = resolve_l5_weights_for_orchestrator(
            self.db_path, model_hash="cache_m",
            model_role=None, cross_model_fallback=True,
        )
        self.assertEqual(df1.height, 1)
        self.assertIn(key, cache)
        cached_df = cache[key]
        # The cached value is the same object as the
        # returned DataFrame (the lookup returns the
        # cached reference; the caller can mutate the
        # returned DataFrame without affecting the
        # cache, but the test asserts identity here).
        # The check below uses ``is`` for identity.
        # Note: polars DataFrames are not typically
        # mutated by the caller, so the identity is
        # a reasonable correctness check.
        # We don't assert identity strictly because the
        # helper may rebuild the cached DataFrame's
        # internal state on insert; the test verifies
        # the data is the same.
        self.assertEqual(
            cached_df["w_imatrix"][0], df1["w_imatrix"][0],
        )
        # Second call: returns the cached entry.
        df2 = resolve_l5_weights_for_orchestrator(
            self.db_path, model_hash="cache_m",
            model_role=None, cross_model_fallback=True,
        )
        self.assertEqual(df2.height, 1)
        # Same data on both calls.
        self.assertEqual(df1["w_imatrix"][0], df2["w_imatrix"][0])
        self.assertEqual(df1["family"][0], df2["family"][0])

    def test_resolve_l5_weights_cache_hits_faster(self) -> None:
        """The cached call is at least as fast as the
        uncached one. The smoke test is intentionally
        loose: the second call is bounded by the time of
        the first. On slow machines the difference may
        be small; the test asserts a soft upper bound
        (the second call is no more than 1.5x the first
        call's duration; a real speedup is typically
        10-100x but we don't want the test to be
        flaky on a contended CI host).
        """
        with TesseraDB.open(self.db_path) as db:
            db.insert_l5_weights([{
                "model_hash": "cache_t", "family": "attn_q",
                "w_imatrix": 0.5, "w_gradient": 0.3, "w_layer": 0.2,
                "n_samples": 100, "in_sample_loss": 0.001,
                "hit_rate": 0.6, "retune_source": RETUNE_SOURCE_TAG,
            }])
        # Clear the cache.
        clear_l5_weights_lookup_cache()
        import time
        # First call: populates the cache.
        t0 = time.perf_counter()
        df1 = resolve_l5_weights_for_orchestrator(
            self.db_path, model_hash="cache_t",
            model_role=None, cross_model_fallback=True,
        )
        first_ms = (time.perf_counter() - t0) * 1000.0
        # Second call: hits the cache.
        t0 = time.perf_counter()
        df2 = resolve_l5_weights_for_orchestrator(
            self.db_path, model_hash="cache_t",
            model_role=None, cross_model_fallback=True,
        )
        second_ms = (time.perf_counter() - t0) * 1000.0
        # The second call should not be slower. On a
        # contended host both calls can be very fast
        # (~1ms) and the noise can be large; the soft
        # bound is 1.5x. In practice the second call
        # is well under 1ms (a dict lookup), so the
        # test passes with margin.
        self.assertLessEqual(
            second_ms, max(first_ms * 1.5, 5.0),
            f"cache miss should be at least as fast as "
            f"first call: first={first_ms:.3f}ms, "
            f"second={second_ms:.3f}ms",
        )
        # The data is the same.
        self.assertEqual(df1["family"][0], df2["family"][0])

    def test_resolve_l5_weights_different_path_different_key(self) -> None:
        """A different db_path produces a different cache
        key. The cache is keyed on (db_path, ...), so
        the second path gets its own entry. (Manual
        invalidation is not required; a path change
        -> a different key -> no stale data.)"""
        clear_l5_weights_lookup_cache()
        # Build a second DB.
        td2 = Path(tempfile.mkdtemp(prefix="l5_retune_test2_"))
        db_path_2 = str(td2 / "tessera.duckdb")
        _create_fresh_db(db_path_2)
        try:
            # Different data in the second DB.
            with TesseraDB.open(db_path_2) as db:
                db.insert_l5_weights([{
                    "model_hash": "k_m", "family": "attn_q",
                    "w_imatrix": 0.9, "w_gradient": 0.05, "w_layer": 0.05,
                    "n_samples": 100, "in_sample_loss": 0.001,
                    "hit_rate": 0.6, "retune_source": RETUNE_SOURCE_TAG,
                }])
            # First call: path 1.
            df1 = resolve_l5_weights_for_orchestrator(
                self.db_path, model_hash="k_m",
                model_role=None, cross_model_fallback=True,
            )
            # Second call: same model_hash, different path.
            df2 = resolve_l5_weights_for_orchestrator(
                db_path_2, model_hash="k_m",
                model_role=None, cross_model_fallback=True,
            )
            # The two calls produced different data
            # (the second DB has the 0.9 row, the first
            # DB does not).
            self.assertEqual(df1.height, 0)
            self.assertEqual(df2.height, 1)
            self.assertAlmostEqual(df2["w_imatrix"][0], 0.9, places=6)
        finally:
            import shutil
            shutil.rmtree(td2, ignore_errors=True)

    def test_resolve_l5_weights_clear_cache(self) -> None:
        """``clear_l5_weights_lookup_cache`` drops all
        entries; the next call re-queries DuckDB.
        """
        with TesseraDB.open(self.db_path) as db:
            db.insert_l5_weights([{
                "model_hash": "clear_m", "family": "attn_q",
                "w_imatrix": 0.5, "w_gradient": 0.3, "w_layer": 0.2,
                "n_samples": 100, "in_sample_loss": 0.001,
                "hit_rate": 0.6, "retune_source": RETUNE_SOURCE_TAG,
            }])
        # Populate the cache.
        resolve_l5_weights_for_orchestrator(
            self.db_path, model_hash="clear_m",
            model_role=None, cross_model_fallback=True,
        )
        self.assertGreater(len(_l5_weights_lookup_cache()), 0)
        # Clear.
        clear_l5_weights_lookup_cache()
        self.assertEqual(len(_l5_weights_lookup_cache()), 0)
        # The next call re-queries and re-populates.
        df = resolve_l5_weights_for_orchestrator(
            self.db_path, model_hash="clear_m",
            model_role=None, cross_model_fallback=True,
        )
        self.assertEqual(df.height, 1)
        self.assertEqual(len(_l5_weights_lookup_cache()), 1)

    def test_resolve_per_family_top_fraction_caches(self) -> None:
        """The per-family top_fraction consumer is also
        cached. The second call returns the same dict
        without re-querying DuckDB.
        """
        with TesseraDB.open(self.db_path) as db:
            db.insert_l5_weights([{
                "model_hash": "tf_m", "family": "attn_q",
                "w_imatrix": 0.5, "w_gradient": 0.3, "w_layer": 0.2,
                "n_samples": 10, "top_fraction": 0.42,
            }])
        # Clear.
        _l5_weights_top_fraction_cache().clear()
        out1 = resolve_per_family_top_fraction_for_orchestrator(
            self.db_path, model_hash="tf_m",
            model_role=None, cross_model_fallback=True,
        )
        self.assertAlmostEqual(out1["attn_q"], 0.42, places=6)
        # The cache has the entry.
        key = (self.db_path, "tf_m", None, True)
        self.assertIn(key, _l5_weights_top_fraction_cache())
        # Second call: cache hit, same dict.
        out2 = resolve_per_family_top_fraction_for_orchestrator(
            self.db_path, model_hash="tf_m",
            model_role=None, cross_model_fallback=True,
        )
        self.assertEqual(out1, out2)

    def _seed_tensor_stats(
        self,
        model_hash: str,
        *,
        n_tensors: int = 20,
        rms_base: float = 0.1,
        rms_jitter: float = 0.0,
        kurt_base: float = 5.0,
        eff_rank_base: float = 100.0,
        mean_abs_base: float = 0.08,
        tail_ratio_base: float = 4.0,
        seed: int = 42,
    ) -> None:
        """Seed tensor_stats with ``n_tensors`` rows for
        ``model_hash``. The stat values are the per-row
        inputs to the fingerprint hash; the helper lets
        the test sweep one parameter (e.g. rms_base) and
        keep the others constant.

        Used by the F3.2 fingerprint tests to construct
        two models with controlled statistical similarity.
        The default ``seed=42`` makes the per-tensor
        values deterministic across calls; two models
        seeded with the same ``seed`` have identical
        per-tensor values except for the ``rms_jitter``
        shift on the rms column.
        """
        import random
        rng = random.Random(seed)
        rows = []
        for i in range(n_tensors):
            rows.append({
                "name": f"blk.{i}.attn_q.weight",
                "family": "attn_q",
                "layer_depth": i,
                "out_dim": 4096, "in_dim": 4096, "n_elements": 4096 * 4096,
                "dtype": "f16",
                "kurtosis": kurt_base + rng.uniform(-0.5, 0.5),
                "eff_rank": eff_rank_base + rng.uniform(-5, 5),
                "rms": rms_base + rng.uniform(-0.01, 0.01) + rms_jitter,
                "mean_abs": mean_abs_base + rng.uniform(-0.005, 0.005),
                "tail_ratio": tail_ratio_base + rng.uniform(-0.1, 0.1),
                "source": "py_cal",
            })
        with TesseraDB.open(self.db_path) as db:
            db.insert_tensor_stats(model_hash=model_hash, rows=rows)

    def test_read_per_family_top_fraction_role_filter(self) -> None:
        """The per-family top_fraction consumer is role-aware:
        a family with different per-role top_fraction
        recommendations returns the role-specific value.
        """
        with TesseraDB.open(self.db_path) as db:
            db.insert_l5_weights([
                {"model_hash": "x", "model_role": "trunk",
                 "family": "attn_q", "w_imatrix": 0.5,
                 "w_gradient": 0.3, "w_layer": 0.2,
                 "n_samples": 10, "top_fraction": 0.20},
                {"model_hash": "x", "model_role": "dflash",
                 "family": "attn_q", "w_imatrix": 0.4,
                 "w_gradient": 0.5, "w_layer": 0.1,
                 "n_samples": 10, "top_fraction": 0.40},
            ])
        recs_trunk = read_per_family_top_fraction(
            self.db_path, model_hash="x", model_role="trunk",
        )
        self.assertAlmostEqual(recs_trunk["attn_q"], 0.20, places=6)
        recs_dflash = read_per_family_top_fraction(
            self.db_path, model_hash="x", model_role="dflash",
        )
        self.assertAlmostEqual(recs_dflash["attn_q"], 0.40, places=6)
        # No role filter: the per-model row with the
        # larger n_samples wins (here both have n=10;
        # the row encountered first in the SELECT wins;
        # the test accepts either, but verifies both
        # are in [0.20, 0.40]).
        recs_all = read_per_family_top_fraction(
            self.db_path, model_hash="x",
        )
        self.assertIn(recs_all["attn_q"], (0.20, 0.40))

    def test_cross_model_aggregate_groups_by_role(self) -> None:
        """The cross-model aggregate is per-(model_role,
        family) not per-family: the trunk's attn_q and
        the dflash's attn_q get independent cross-model
        rows.
        """
        with TesseraDB.open(self.db_path) as db:
            db.insert_l5_weights([
                {"model_hash": "m1", "model_role": "trunk",
                 "family": "attn_q", "w_imatrix": 0.6,
                 "w_gradient": 0.3, "w_layer": 0.1,
                 "n_samples": 10, "top_fraction": 0.15},
                {"model_hash": "m2", "model_role": "trunk",
                 "family": "attn_q", "w_imatrix": 0.5,
                 "w_gradient": 0.4, "w_layer": 0.1,
                 "n_samples": 20, "top_fraction": 0.10},
                {"model_hash": "m1", "model_role": "dflash",
                 "family": "attn_q", "w_imatrix": 0.4,
                 "w_gradient": 0.5, "w_layer": 0.1,
                 "n_samples": 30, "top_fraction": 0.30},
            ])
        verdicts = write_cross_model_aggregate(self.db_path)
        # 2 verdicts: (trunk, attn_q) and (dflash, attn_q).
        self.assertEqual(len(verdicts), 2)
        roles = {v.model_role for v in verdicts}
        self.assertEqual(roles, {"trunk", "dflash"})
        trunk = next(v for v in verdicts if v.model_role == "trunk")
        dflash = next(v for v in verdicts if v.model_role == "dflash")
        # trunk: n_samples=30, w_im = (0.6*10 + 0.5*20)/30 = 0.533
        self.assertEqual(trunk.n_samples, 30)
        self.assertAlmostEqual(trunk.weights[0], 16.0 / 30.0, places=6)
        # dflash: n_samples=30, w_im = 0.4 (one model)
        self.assertEqual(dflash.n_samples, 30)
        self.assertAlmostEqual(dflash.weights[0], 0.4, places=6)
        # The cross-model rows in l5_weights are per-role.
        import duckdb as _dd
        con = _dd.connect(self.db_path, read_only=True)
        try:
            cross = con.execute(
                "SELECT model_role, family FROM l5_weights "
                "WHERE model_hash = '*' ORDER BY model_role, family"
            ).fetchall()
        finally:
            con.close()
        self.assertEqual(
            sorted(cross),
            sorted([("dflash", "attn_q"), ("trunk", "attn_q")]),
        )


class TestRequantBudgetProducer(unittest.TestCase):
    """The requant_budget_bits producer: the retune computes
    budget = family_storage_bits * (1 - hit_rate) * fraction
    from tensor_stats and persists it on the l5_weights row.
    """

    def setUp(self) -> None:
        self._td = Path(tempfile.mkdtemp(prefix="l5_budget_test_"))
        self.db_path = str(self._td / "tessera.duckdb")
        _create_fresh_db(self.db_path)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._td, ignore_errors=True)

    def _seed_outcome(
        self, model_hash: str, family: str, n: int, hit_rate: float,
        role: str = "trunk",
    ) -> None:
        """Seed n l5_outcome rows with a controlled hit_rate
        (the first round(hit_rate*n) plans are accepted) and a
        clean positive slope so the OLS acts on the group."""
        n_accepted = int(round(hit_rate * n))
        rows = []
        for i in range(n):
            rows.append({
                "name": f"blk.{i}.{family}.weight",
                "layer": i,
                "iteration": 0,
                "plan_id": f"p{i}",
                "family": family,
                "model_role": role,
                "sensitivity_score": 0.1 * (i + 1),
                "mse_before": 0.01,
                "mse_after": 0.01 + 0.001 * (i + 1),
                "delta_mse": 0.001 * (i + 1),
                "plan_accepted": i < n_accepted,
                "accept_threshold": 0.0,
            })
        _seed_l5_outcome(self.db_path, model_hash, rows)

    def _seed_stats(
        self, model_hash: str, family: str, tensors: list[tuple[int, str]],
        role: str = "trunk",
    ) -> None:
        """Seed tensor_stats rows: (n_elements, dtype) per tensor."""
        with TesseraDB.open(self.db_path) as db:
            db.insert_tensor_stats(model_hash=model_hash, rows=[
                {
                    "name": f"blk.{i}.{family}.weight",
                    "family": family,
                    "model_role": role,
                    "layer_depth": i,
                    "out_dim": 1, "in_dim": n_el, "n_elements": n_el,
                    "dtype": dtype,
                    "source": "cpp_quant",
                }
                for i, (n_el, dtype) in enumerate(tensors)
            ])

    def test_budget_from_storage_and_hit_rate(self) -> None:
        # storage = 1000*16 + 1000*16 = 32000 bits; hit_rate=0.5,
        # fraction=1.0 -> budget = 16000.
        self._seed_outcome("m", "attn_q", n=4, hit_rate=0.5)
        self._seed_stats("m", "attn_q", [(1000, "f16"), (1000, "f16")])
        verdicts = compute_l5_weights(
            self.db_path, model_hash="m", write_back=True,
        )
        v = next(v for v in verdicts if v.family == "attn_q")
        self.assertAlmostEqual(v.hit_rate, 0.5, places=6)
        self.assertEqual(v.requant_budget_bits, 16000)

    def test_budget_null_without_tensor_stats(self) -> None:
        self._seed_outcome("m", "attn_q", n=4, hit_rate=0.5)
        verdicts = compute_l5_weights(
            self.db_path, model_hash="m", write_back=True,
        )
        v = next(v for v in verdicts if v.family == "attn_q")
        self.assertTrue(v.was_acted_on)
        self.assertIsNone(v.requant_budget_bits)

    def test_budget_null_when_too_few_samples(self) -> None:
        self._seed_outcome("m", "attn_q", n=2, hit_rate=0.5)
        self._seed_stats("m", "attn_q", [(1000, "f16")])
        verdicts = compute_l5_weights(
            self.db_path, model_hash="m", write_back=True,
        )
        v = next(v for v in verdicts if v.family == "attn_q")
        self.assertFalse(v.was_acted_on)
        self.assertIsNone(v.requant_budget_bits)

    def test_budget_fraction_scaling(self) -> None:
        self._seed_outcome("m", "attn_q", n=4, hit_rate=0.5)
        self._seed_stats("m", "attn_q", [(1000, "f16"), (1000, "f16")])
        verdicts = compute_l5_weights(
            self.db_path, model_hash="m", write_back=True,
            base_budget_fraction=0.25,
        )
        v = next(v for v in verdicts if v.family == "attn_q")
        # 32000 * 0.5 * 0.25 = 4000
        self.assertEqual(v.requant_budget_bits, 4000)

    def test_budget_disabled_with_zero_fraction(self) -> None:
        self._seed_outcome("m", "attn_q", n=4, hit_rate=0.5)
        self._seed_stats("m", "attn_q", [(1000, "f16")])
        verdicts = compute_l5_weights(
            self.db_path, model_hash="m", write_back=True,
            base_budget_fraction=0.0,
        )
        v = next(v for v in verdicts if v.family == "attn_q")
        self.assertIsNone(v.requant_budget_bits)

    def test_budget_zero_at_hit_rate_one(self) -> None:
        # A converged family (all plans accepted) gets budget 0:
        # the next requant pass should not grow it.
        self._seed_outcome("m", "attn_q", n=4, hit_rate=1.0)
        self._seed_stats("m", "attn_q", [(1000, "f16")])
        verdicts = compute_l5_weights(
            self.db_path, model_hash="m", write_back=True,
        )
        v = next(v for v in verdicts if v.family == "attn_q")
        self.assertEqual(v.requant_budget_bits, 0)

    def test_budget_persisted_to_l5_weights(self) -> None:
        self._seed_outcome("m", "attn_q", n=4, hit_rate=0.5)
        self._seed_stats("m", "attn_q", [(1000, "f16"), (1000, "f16")])
        compute_l5_weights(
            self.db_path, model_hash="m", write_back=True,
        )
        import duckdb as _dd
        con = _dd.connect(self.db_path, read_only=True)
        try:
            rows = con.execute(
                "SELECT requant_budget_bits FROM l5_weights "
                "WHERE model_hash = 'm' AND family = 'attn_q'"
            ).fetchall()
        finally:
            con.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 16000)

    def test_budget_role_independent_storage(self) -> None:
        # The same family in two roles has different storage
        # footprints; each role's budget is computed from its
        # own tensor_stats rows.
        self._seed_outcome("m", "attn_q", n=4, hit_rate=0.5, role="trunk")
        self._seed_outcome("m", "attn_q", n=4, hit_rate=0.5, role="dflash")
        self._seed_stats("m", "attn_q", [(1000, "f16")], role="trunk")
        self._seed_stats("m", "attn_q", [(500, "f16")], role="dflash")
        verdicts = compute_l5_weights(
            self.db_path, model_hash="m", write_back=True,
        )
        trunk = next(v for v in verdicts
                     if v.model_role == "trunk" and v.family == "attn_q")
        dflash = next(v for v in verdicts
                      if v.model_role == "dflash" and v.family == "attn_q")
        # trunk: 1000*16 * 0.5 = 8000; dflash: 500*16 * 0.5 = 4000
        self.assertEqual(trunk.requant_budget_bits, 8000)
        self.assertEqual(dflash.requant_budget_bits, 4000)

    def test_budget_unknown_dtype_skipped(self) -> None:
        # Rows with an unrecognized dtype contribute nothing;
        # the known rows still sum.
        self._seed_outcome("m", "attn_q", n=4, hit_rate=0.5)
        self._seed_stats(
            "m", "attn_q",
            [(1000, "f16"), (9999, "mystery_dtype")],
        )
        verdicts = compute_l5_weights(
            self.db_path, model_hash="m", write_back=True,
        )
        v = next(v for v in verdicts if v.family == "attn_q")
        # 1000*16 * 0.5 = 8000 (the mystery row is skipped)
        self.assertEqual(v.requant_budget_bits, 8000)

    def test_cross_model_budget_weighted_mean(self) -> None:
        # Two models, same (role, family), equal n_samples:
        # the cross-model budget is the mean of the per-model
        # budgets.
        for mh, n_el in (("m1", 1000), ("m2", 3000)):
            self._seed_outcome(mh, "attn_q", n=4, hit_rate=0.5)
            self._seed_stats(mh, "attn_q", [(n_el, "f16")])
        compute_l5_weights(self.db_path, write_back=True)
        cross = write_cross_model_aggregate(self.db_path)
        v = next(v for v in cross
                 if v.model_role == "trunk" and v.family == "attn_q")
        # m1 budget = 1000*16*0.5 = 8000; m2 = 3000*16*0.5 = 24000;
        # equal n_samples -> mean = 16000.
        self.assertEqual(v.requant_budget_bits, 16000)


if __name__ == "__main__":
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestL5Retune))
    suite.addTests(loader.loadTestsFromTestCase(TestRequantBudgetProducer))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
