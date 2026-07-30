#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE = Path(__file__).parents[1] / "tools/tessera/policy-prior.py"
SPEC = importlib.util.spec_from_file_location("tessera_prior", MODULE)
PRIOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PRIOR
SPEC.loader.exec_module(PRIOR)


class PolicyPriorTest(unittest.TestCase):
    def test_only_portable_family_entries_are_imported(self):
        base = {"schema": PRIOR.SCHEMA, "tensor_families": {"local": {"match": ["ffn"], "outlier_fraction": 0.01}}}
        prior = {"schema": PRIOR.SCHEMA, "tensor_families": {
            "attention": {"match": ["attn_"], "outlier_fraction": 0.004},
            "shadow_tensor": {"match": ["blk.1.attn_q.weight"], "outlier_fraction": 0.02},
            "exact": {"match": ["norm"], "exact": True, "outlier_fraction": 1.0},
        }}
        merged = PRIOR.merge(base, prior, "gemma4")
        families = merged["tensor_families"]
        self.assertIn("local", families)
        self.assertIn("prior:gemma4:attention", families)
        self.assertNotIn("prior:gemma4:shadow_tensor", families)
        self.assertNotIn("prior:gemma4:exact", families)
        self.assertEqual(merged["tessera_calibration_prior"]["rejected_nonportable_entries"], ["exact", "shadow_tensor"])


if __name__ == "__main__":
    unittest.main()
