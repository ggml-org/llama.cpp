#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tools" / "tessera" / "observer-delta-policy.py"
SPEC = importlib.util.spec_from_file_location("tessera_observer_delta", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TesseraObserverDeltaTest(unittest.TestCase):
    def test_delta_ranks_divergent_tensor(self):
        reference = {
            "blk.0.attn_q.weight": {
                "in_sum2": np.ones(4),
                "in_maxabs": np.ones(4),
                "counts": np.ones(1),
            },
            "blk.1.attn_q.weight": {
                "in_sum2": np.ones(4),
                "in_maxabs": np.ones(4),
                "counts": np.ones(1),
            },
        }
        candidate = {
            **reference,
            "blk.1.attn_q.weight": {
                "in_sum2": np.full(4, 4.0),
                "in_maxabs": np.full(4, 2.0),
                "counts": np.ones(1),
            },
        }
        records = MODULE.observer_delta(reference, candidate)
        self.assertEqual(records[0]["tensor"], "blk.1.attn_q.weight")
        self.assertGreater(records[0]["score"], records[1]["score"])

    def test_repair_overrides_precede_base_policy(self):
        base = {
            "schema": MODULE.SCHEMA,
            "tensor_families": {
                "attention": {
                    "match": ["attn_q"],
                    "outlier_fraction": 0.005,
                }
            },
        }
        policy = MODULE.build_repair_policy(
            base,
            [{
                "tensor": "blk.7.attn_q.weight",
                "score": 1.0,
                "moment_delta": 0.8,
                "tail_delta": 0.8,
                "channels": 16,
            }],
            1.0,
            4,
            2.0,
            {},
        )
        entries = list(policy["tensor_families"].values())
        self.assertEqual(entries[0]["match"], ["blk.7.attn_q.weight"])
        self.assertEqual(entries[0]["outlier_fraction"], 0.01)


if __name__ == "__main__":
    unittest.main()
