#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import polars as pl


SCRIPT = Path(__file__).parents[1] / "tools" / "tessera" / "hf-evidence.py"
SPEC = importlib.util.spec_from_file_location("tessera_hf_evidence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HFEvidenceTests(unittest.TestCase):
    def test_model_fingerprint_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            config = {"model_type": "gemma4", "hidden_size": 8, "architectures": ["Test"]}
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            first, _ = MODULE.model_identity(model)
            (model / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            second, _ = MODULE.model_identity(model)
            self.assertEqual(first, second)

    def test_contribution_license_names_grantee_and_is_hashed(self):
        record = MODULE.contribution_license_record(
            MODULE.CONTRIBUTION_LICENSE_ID, "contributor"
        )
        self.assertEqual(
            record["license_grantee"], "Julian Alejandro Torres Nieto"
        )
        self.assertEqual(len(record["license_sha256"]), 64)
        self.assertTrue(record["commercial_relicensing"])
        self.assertEqual(record["royalty_obligation"], "none")

    def test_contribution_license_requires_exact_explicit_assent(self):
        with self.assertRaisesRegex(ValueError, "explicit acceptance"):
            MODULE.contribution_license_record("", "contributor")

    def test_public_license_is_noncommercial_and_attributed(self):
        record = MODULE.public_license_record()
        self.assertEqual(record["license_id"], "CC-BY-NC-SA-4.0")
        self.assertEqual(
            record["attribution"],
            "Julian Alejandro Torres Nieto, Tribunus.dev",
        )
        self.assertFalse(record["commercial_use"])
        self.assertTrue(record["share_alike"])
        self.assertTrue(record["legacy_grants_preserved"])
        self.assertEqual(len(record["notice_sha256"]), 64)

    def test_observer_aggregation_preserves_sufficient_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            target = store / "observer"
            target.mkdir()
            pl.DataFrame([
                {"run_id": "r", "tensor": "x", "expert": 0, "channel": 0,
                 "count": 2.0, "sum2": 8.0, "sumabs": 4.0, "sum4": 32.0,
                 "maxabs": 3.0, "rms": 2.0, "mean_abs": 2.0,
                 "kurtosis": 2.0, "tail_ratio": 1.5},
                {"run_id": "r", "tensor": "x", "expert": 0, "channel": 0,
                 "count": 3.0, "sum2": 12.0, "sumabs": 6.0, "sum4": 48.0,
                 "maxabs": 4.0, "rms": 2.0, "mean_abs": 2.0,
                 "kurtosis": 2.0, "tail_ratio": 2.0},
            ]).write_parquet(target / "part.parquet")
            frame = MODULE.aggregate_observer(store, "r", min_tokens=1)
            self.assertEqual(frame.height, 1)
            row = frame.row(0, named=True)
            self.assertEqual(row["count"], 5.0)
            self.assertEqual(row["sum2"], 20.0)
            self.assertEqual(row["maxabs"], 4.0)

    def test_privacy_schema_rejects_identifier_columns(self):
        frame = pl.DataFrame({"tensor": ["x"], "request_id": ["secret"]})
        with self.assertRaises(ValueError):
            MODULE.audit_columns(frame, {"tensor", "request_id"}, "test")

    def test_small_observer_population_is_not_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            target = store / "observer"
            target.mkdir()
            pl.DataFrame([{
                "run_id": "r", "tensor": "x", "expert": 0, "channel": 0,
                "count": 2.0, "sum2": 8.0, "sumabs": 4.0, "sum4": 32.0,
                "maxabs": 3.0, "rms": 2.0, "mean_abs": 2.0,
                "kurtosis": 2.0, "tail_ratio": 1.5,
            }]).write_parquet(target / "part.parquet")
            frame = MODULE.aggregate_observer(store, "r", min_tokens=128)
            self.assertTrue(frame.is_empty())

    def test_router_aggregation_enforces_per_expert_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            target = store / "router"
            target.mkdir()
            pl.DataFrame([
                {
                    "run_id": "r", "layer": 2, "expert": 0,
                    "observations": 100, "selected": 20,
                    "probability_sum": 25.0, "confidence_sum": 12.0,
                    "margin_sum": 4.0, "output_error_sum": 2.0,
                    "downstream_divergence_sum": 1.0,
                },
                {
                    "run_id": "r", "layer": 2, "expert": 0,
                    "observations": 100, "selected": 30,
                    "probability_sum": 35.0, "confidence_sum": 18.0,
                    "margin_sum": 6.0, "output_error_sum": 3.0,
                    "downstream_divergence_sum": 1.5,
                },
                {
                    "run_id": "r", "layer": 2, "expert": 1,
                    "observations": 200, "selected": 2,
                    "probability_sum": 5.0, "confidence_sum": 1.0,
                    "margin_sum": 0.1, "output_error_sum": 1.0,
                    "downstream_divergence_sum": 0.5,
                },
            ]).write_parquet(target / "part.parquet")
            frame = MODULE.aggregate_router(
                store, "r", min_observations=128, min_expert_selections=16
            )
            self.assertEqual(frame.height, 1)
            row = frame.row(0, named=True)
            self.assertEqual(row["expert"], 0)
            self.assertEqual(row["selected"], 50)
            self.assertAlmostEqual(row["frequency"], 0.25)

    def test_epoch_advances_and_requests_requantization(self):
        manifests = [
            {
                "aggregate_id": "a",
                "model_fingerprint": "model",
                "observer_calibration_tokens": 600,
                "acceptance_observations": 20,
            },
            {
                "aggregate_id": "b",
                "model_fingerprint": "model",
                "observer_calibration_tokens": 500,
                "acceptance_observations": 30,
            },
        ]
        state = MODULE.epoch_state(
            "model", manifests, model_epoch=0, observer_tokens_per_epoch=1000
        )
        self.assertEqual(state["epoch"], 1)
        self.assertTrue(state["requantization_due"])
        self.assertEqual(state["observer_calibration_tokens"], 1100)

    def test_epoch_deduplicates_aggregate_ids(self):
        manifest = {
            "aggregate_id": "same",
            "model_fingerprint": "model",
            "observer_calibration_tokens": 800,
        }
        state = MODULE.epoch_state(
            "model", [manifest, manifest], model_epoch=1, observer_tokens_per_epoch=500
        )
        self.assertEqual(state["epoch"], 1)
        self.assertEqual(state["aggregate_count"], 1)
        self.assertFalse(state["requantization_due"])

    def test_epoch_deduplicates_republished_observer_component(self):
        manifests = [
            {
                "aggregate_id": "first",
                "model_fingerprint": "model",
                "observer_digest": "observer",
                "observer_calibration_tokens": 800,
            },
            {
                "aggregate_id": "second",
                "model_fingerprint": "model",
                "observer_digest": "observer",
                "observer_calibration_tokens": 800,
            },
        ]
        state = MODULE.epoch_state(
            "model", manifests, model_epoch=0, observer_tokens_per_epoch=1000
        )
        self.assertEqual(state["observer_calibration_tokens"], 800)
        self.assertEqual(state["epoch"], 0)


if __name__ == "__main__":
    unittest.main()
