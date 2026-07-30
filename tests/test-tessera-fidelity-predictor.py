#!/usr/bin/env python3
"""Smoke test for tools/tessera/fidelity_predictor.py.

Covers the four public contracts:

* ``predict_error`` returns a ``Prediction`` with the right shape and a
  scalar ``total`` in the expected range on normalised inputs.
* ``train`` returns a ``Predictor`` whose ``alpha`` is a 6-vector and
  whose ``beta`` is an ``(n_layers, n_layers)`` matrix.
* ``inspect`` returns plain Python types (no numpy leakage).
* The fitted model achieves ``R^2 > 0.5`` on the held-out 20% of the
  synthetic 10-tensor bundle, and the fit is deterministic across two
  training runs with the same seed.

The tests deliberately avoid mocking the lstsq call; the synthetic data
is a faithful stand-in for the real per-tensor error target (the
``per_layer_error_table`` output) so the regression exercise exercises
the same code path the integrator will hit.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "tessera" / "fidelity_predictor.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fidelity_predictor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["fidelity_predictor"] = module
    spec.loader.exec_module(module)
    return module


class FidelityPredictorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fp = load_module()
        cls.bundle = cls.fp.generate_synthetic_bundle(seed=42)
        cls.train_idx = list(cls.fp.DEFAULT_TRAIN_INDICES)
        cls.val_idx = list(cls.fp.DEFAULT_VAL_INDICES)
        cls.predictor = cls.fp.train(
            cls.bundle["scores_arr"][cls.train_idx],
            cls.bundle["errors_arr"][cls.train_idx],
            cls.bundle["layer_indices"][cls.train_idx],
        )

    # -- shape contracts -------------------------------------------------

    def test_train_returns_predictor_with_six_vector_alpha(self):
        alpha = self.predictor.alpha
        self.assertEqual(alpha.shape, (6,))
        self.assertTrue(np.issubdtype(alpha.dtype, np.floating))

    def test_train_returns_symmetric_band_beta(self):
        beta = self.predictor.beta
        n_layers = self.predictor.n_layers
        self.assertEqual(beta.shape, (n_layers, n_layers))
        # Diagonal must be zero (no self-interaction).
        for l in range(n_layers):
            self.assertEqual(beta[l, l], 0.0)
        # Symmetric.
        for l in range(n_layers):
            for m in range(n_layers):
                self.assertAlmostEqual(beta[l, m], beta[m, l], places=12)
        # Sparse: |l - m| > 1 must be zero.
        for l in range(n_layers):
            for m in range(n_layers):
                if abs(l - m) > 1:
                    self.assertEqual(beta[l, m], 0.0)

    # -- predict_error contract ----------------------------------------

    def test_predict_error_returns_prediction_with_finite_total(self):
        s = self.bundle["scores_arr"][0]
        # Layer 0 has one neighbour (layer 1) and one distant (layer 2,
        # which is out of band and contributes zero via beta=0).
        neighbours = [
            (1, self.bundle["scores_arr"][4]),
            (2, self.bundle["scores_arr"][7]),
        ]
        pred = self.fp.predict_error(
            self.predictor, s, neighbours, current_layer=0
        )
        self.assertTrue(math.isfinite(pred.intercept))
        self.assertTrue(math.isfinite(pred.linear))
        self.assertTrue(math.isfinite(pred.interaction))
        self.assertTrue(math.isfinite(pred.total))
        self.assertAlmostEqual(
            pred.total,
            pred.intercept + pred.linear + pred.interaction,
            places=12,
        )

    def test_predict_error_in_unit_interval_on_normalised_inputs(self):
        # The synthetic scores are uniform in [0, 1] and the fitted
        # model produces small-magnitude predictions in the same range.
        for tensor_idx in range(self.bundle["scores_arr"].shape[0]):
            layer = int(self.bundle["layer_indices"][tensor_idx])
            neighbours = []
            for other_layer in range(self.predictor.n_layers):
                if other_layer == layer:
                    continue
                if abs(other_layer - layer) > 1:
                    continue
                mask = self.bundle["layer_indices"] == other_layer
                # Use the first tensor in that layer as the representative
                # neighbour.  This is deterministic; the test asserts a
                # property of the fitted model, not of the synthetic data.
                first = int(np.argmax(mask))
                neighbours.append(
                    (other_layer, self.bundle["scores_arr"][first])
                )
            pred = self.fp.predict_error(
                self.predictor,
                self.bundle["scores_arr"][tensor_idx],
                neighbours,
                current_layer=layer,
            )
            self.assertGreaterEqual(pred.total, 0.0)
            self.assertLessEqual(pred.total, 1.0)

    def test_predict_error_rejects_wrong_signal_length(self):
        with self.assertRaises(ValueError):
            self.fp.predict_error(
                self.predictor,
                [0.0, 0.0, 0.0, 0.0, 0.0],  # 5, not 6
                [],
                current_layer=0,
            )

    def test_predict_error_rejects_non_finite_scores(self):
        bad = [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            self.fp.predict_error(self.predictor, bad, [], current_layer=0)

    # -- inspect() contract ---------------------------------------------

    def test_inspect_returns_python_types_only(self):
        payload = self.fp.inspect(self.predictor)
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["schema"], self.fp.PREDICTOR_SCHEMA)
        # No numpy types in the JSON payload.
        for entry in payload["alpha"]:
            self.assertIsInstance(entry, float)
        for row in payload["beta"]:
            for entry in row:
                self.assertIsInstance(entry, float)
        self.assertIsInstance(payload["intercept"], float)
        self.assertEqual(payload["signal_order"], list(self.fp.SIGNAL_NAMES))

    def test_inspect_is_json_serialisable(self):
        payload = self.fp.inspect(self.predictor)
        encoded = json.dumps(payload)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["alpha"], payload["alpha"])
        self.assertEqual(decoded["beta"], payload["beta"])

    # -- predictive quality on the synthetic bundle --------------------

    def test_val_r2_above_threshold(self):
        val_r2 = self.fp.coefficient_r2(
            self.predictor,
            self.bundle["scores_arr"][self.val_idx],
            self.bundle["errors_arr"][self.val_idx],
            self.bundle["layer_indices"][self.val_idx],
        )
        # The synthetic bundle is essentially linear with 0.005 noise; the
        # recovered alpha should match the true alpha to within numerical
        # precision, so the val R^2 is well above the 0.5 spec threshold.
        self.assertTrue(
            math.isfinite(val_r2),
            f"val R^2 is not finite: {val_r2}",
        )
        self.assertGreater(
            val_r2,
            0.5,
            f"val R^2 = {val_r2:.4f} below the 0.5 spec threshold",
        )

    def test_top_alpha_coefficients_returns_three_pairs(self):
        top = self.fp.top_alpha_coefficients(self.predictor, k=3)
        self.assertEqual(len(top), 3)
        for name, value in top:
            self.assertIn(name, self.fp.SIGNAL_NAMES)
            self.assertIsInstance(value, float)

    # -- determinism ----------------------------------------------------

    def test_train_is_deterministic_under_fixed_seed(self):
        bundle_a = self.fp.generate_synthetic_bundle(seed=1234)
        bundle_b = self.fp.generate_synthetic_bundle(seed=1234)
        train_a = self.fp.train(
            bundle_a["scores_arr"][:8],
            bundle_a["errors_arr"][:8],
            bundle_a["layer_indices"][:8],
        )
        train_b = self.fp.train(
            bundle_b["scores_arr"][:8],
            bundle_b["errors_arr"][:8],
            bundle_b["layer_indices"][:8],
        )
        self.assertEqual(train_a.intercept, train_b.intercept)
        np.testing.assert_array_equal(train_a.alpha, train_b.alpha)
        np.testing.assert_array_equal(train_a.beta, train_b.beta)

    # -- CLI smoke ------------------------------------------------------

    def test_cli_train_demo_runs_and_emits_predictor(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--train-demo",
                "--seed",
                "7",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"CLI failed:\nstdout={completed.stdout}\nstderr={completed.stderr}",
        )
        # The --quiet mode prints the JSON sidecar to stdout after a
        # header.  Parse the JSON block by stripping the leading
        # "# Predictor sidecar (JSON):" comment.
        out = completed.stdout
        marker = "# Predictor sidecar (JSON):"
        self.assertIn(marker, out)
        json_block = out.split(marker, 1)[1].strip()
        payload = json.loads(json_block)
        self.assertEqual(payload["schema"], self.fp.PREDICTOR_SCHEMA)
        self.assertEqual(len(payload["alpha"]), 6)
        self.assertEqual(len(payload["beta"]), 3)
        self.assertEqual(len(payload["beta"][0]), 3)


if __name__ == "__main__":
    unittest.main()
