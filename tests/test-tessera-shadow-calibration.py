#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AWQ = load("shadow_awq", "tools/tessera/awq-evolve.py")
SHADOW = load("shadow_calibrate", "tools/tessera/shadow-calibrate.py")


class ShadowCalibrationTest(unittest.TestCase):
    def layer(self, index, scale=1.0):
        rng = np.random.default_rng(index)
        weight = (rng.normal(size=(8, 12)) * scale).astype(np.float32)
        activations = rng.normal(size=(24, 12)).astype(np.float32)
        return AWQ.Layer(
            f"blk.{index}.ffn_down.weight", "ffn", weight,
            activations[:16], activations[16:],
            np.mean(activations ** 2, axis=0),
            np.mean(activations ** 4, axis=0),
            np.max(np.abs(activations), axis=0),
        )

    def test_shadow_refinement_keeps_policy_and_allocates_bounded_residuals(self):
        layers = [self.layer(index, 1.0 + index * 0.3) for index in range(5)]
        policy = {
            "schema": "llama.speculative.calibration-policy.v1",
            "tensor_families": {
                "ffn": {
                    "match": ["ffn_down"], "exact": False,
                    "awq_alpha": 0.4, "awq_clip": 0.9,
                    "outlier_fraction": 0.005, "moment_mix": 0.2,
                    "tail_guard": 0.3,
                },
            },
        }
        records = SHADOW.score_layers(layers, policy, AWQ)
        selected = SHADOW.select_stratified(records, 0.4, 3)
        refined = SHADOW.refine_policy(policy, records, selected, 2.0)
        overrides = [value for key, value in refined["tensor_families"].items()
                     if key.startswith("shadow_")]
        self.assertTrue(overrides)
        self.assertLessEqual(len(overrides), 3)
        self.assertTrue(all(0.0001 <= entry["outlier_fraction"] <= 0.05 for entry in overrides))
        self.assertEqual(refined["tessera_shadow_calibration"]["layers_scored"], 5)

    def test_exact_shadow_override_wins_over_family_match(self):
        layer = self.layer(3)
        policy = {
            "tensor_families": {
                "ffn": {"match": ["ffn_down"], "awq_alpha": 0.2, "outlier_fraction": 0.002},
                "exact": {"match": [layer.name], "exact": True, "awq_alpha": 0.8, "outlier_fraction": 0.01},
            },
        }
        candidate = SHADOW.candidate_for(layer, policy, AWQ)
        self.assertAlmostEqual(candidate.alpha, 0.8)
        self.assertAlmostEqual(candidate.outlier_fraction, 0.01)

    def test_shadow_expert_bundle_emits_per_expert_policy(self):
        layer = self.layer(4)
        layer.name = "blk.4.ffn_down_exps.weight.expert-7"
        layer.family = "routed_expert"
        layer.sample_count = 4
        policy = {
            "schema": "llama.speculative.calibration-policy.v1",
            "tensor_families": {
                "expert": {"match": ["ffn_down_exps"], "outlier_fraction": 0.005},
            },
        }
        record = SHADOW.score_layers([layer], policy, AWQ)[0]
        refined = SHADOW.refine_policy(policy, [record], [record], 1.0)
        entry = refined["moe_residual_allocation"]["layers"]["4"]["tensors"][
            "ffn_down_exps.weight"
        ]["experts"]["7"]
        self.assertIn("outlier_fraction", entry)
        self.assertIn("coverage_uncertainty", entry)

    def test_shadow_preserves_multimodal_family_stratum(self):
        vision = self.layer(1)
        vision.name = "v.embed_vision_proj.weight"
        vision.family = "fusion"
        text = self.layer(2)
        policy = {
            "schema": "llama.speculative.calibration-policy.v1",
            "tensor_families": {
                "fusion": {"match": ["embed_vision_proj"], "outlier_fraction": 0.004},
                "ffn": {"match": ["ffn_down"], "outlier_fraction": 0.004},
            },
        }
        records = SHADOW.score_layers([vision, text], policy, AWQ)
        selected = SHADOW.select_stratified(records, 0.5, 4)
        self.assertEqual({item["family"] for item in selected}, {"fusion", "ffn"})


if __name__ == "__main__":
    unittest.main()
