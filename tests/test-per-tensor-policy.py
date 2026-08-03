#!/usr/bin/env python3
"""test-per-tensor-policy.py

Verifies the per-tensor calibration policy API in tools/tile640/quantize_v3.py
(the `load_calibration_policy` and `tensor_policy` functions) plus the
shared schema that quantize_v3.py and per_tensor_calibrate.py are
expected to agree on. The audit (docs/audit-2026-07-29.md, section 12)
flags the lack of a shared schema definition between the AWQ evolution
candidate dataclass and the quantizer's `tensor_policy()` function as a
real risk. This test pins down the current contract so that future
schema changes are visible.

Why this matters
----------------
The quantizer's per-tensor `ternary_threshold` (added in 190e9a72 to fix
the 0.86% accept rate on tessera Q4_K_M) is loaded from a JSON file
produced by `tools/tessera/per_tensor_calibrate.py`. The contract is:
1. The JSON file has a `schema` field naming the schema version.
2. `load_calibration_policy` only accepts two schemas today:
       llama.dflash.calibration-policy.v1
       llama.speculative.calibration-policy.v1
   (quantize_v3.py:1967-1971).
3. `tensor_policy` looks up the tensor name in `tensor_families`,
   preferring the most specific (highest-ranked) match.

What this test verifies
-----------------------
1. Schema validation: well-formed JSON files with the right schema are
   accepted; bad schemas (missing, unknown) are rejected.
2. Family matching: a tensor name that matches multiple families is
   resolved to the most specific one (longest fragment, with `exact`
   winning over substring).
3. Field propagation: each family field (outlier_fraction, awq_alpha,
   awq_clip, ternary_threshold, exact) flows through to the policy.
4. Default fallback: tensors not covered by any family get the defaults
   passed in.
5. The two accepted schemas resolve identically under `tensor_policy`
   (the quantizer treats them as the same wire format).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools" / "tile640" / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


QUANTIZE_V3 = load_script("tessera_quantize_v3", "quantize_v3.py")


# These are the two schemas the quantizer accepts today. Both resolve to
# the same `tensor_policy` behaviour; the test treats them as aliases of
# the same wire format (the audit's section 12 future work is to unify
# them under a single schema — the test pins the current behaviour so
# the unification is a conscious change).
SCHEMA_DFLASH  = "llama.dflash.calibration-policy.v1"
SCHEMA_SPEC    = "llama.speculative.calibration-policy.v1"


def write_policy(tmp, schema, families):
    """Serialize a policy dict to a temp file in the schema's expected shape."""
    doc = {
        "schema": schema,
        "tensor_families": families,
    }
    fd = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=tmp
    )
    json.dump(doc, fd)
    fd.close()
    return Path(fd.name)


class LoadCalibrationPolicyTests(unittest.TestCase):
    def test_accepts_both_supported_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            for schema in (SCHEMA_DFLASH, SCHEMA_SPEC):
                path = write_policy(tmp, schema, {})
                policy = QUANTIZE_V3.load_calibration_policy(str(path))
                self.assertEqual(policy["schema"], schema)

    def test_rejects_missing_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            fd = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, dir=tmp
            )
            json.dump({"tensor_families": {}}, fd)
            fd.close()
            with self.assertRaises(ValueError):
                QUANTIZE_V3.load_calibration_policy(fd.name)

    def test_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_policy(tmp, "unknown-schema", {})
            with self.assertRaises(ValueError):
                QUANTIZE_V3.load_calibration_policy(str(path))

    def test_empty_path_returns_none(self):
        # load_calibration_policy treats empty/None as "no policy".
        self.assertIsNone(QUANTIZE_V3.load_calibration_policy(None))
        self.assertIsNone(QUANTIZE_V3.load_calibration_policy(""))


class TensorPolicyTests(unittest.TestCase):
    """End-to-end checks on tensor_policy() with various family configs."""

    # Default values are arbitrary; the test only checks that they're
    # surfaced for tensors not covered by any family.
    DEFAULT_FRACTION = 0.012
    DEFAULT_ALPHA = 0.5

    def test_uncovered_tensor_returns_defaults(self):
        policy = {
            "schema": SCHEMA_DFLASH,
            "tensor_families": {
                "attention": {"match": ["attn_"], "outlier_fraction": 0.01},
            },
        }
        fraction, alpha, clip, exact, threshold = QUANTIZE_V3.tensor_policy(
            policy, "blk.0.ffn_gate.weight", self.DEFAULT_FRACTION, self.DEFAULT_ALPHA
        )
        self.assertAlmostEqual(fraction, self.DEFAULT_FRACTION)
        self.assertEqual(alpha, self.DEFAULT_ALPHA)
        self.assertAlmostEqual(clip, 1.0)
        self.assertFalse(exact)
        self.assertAlmostEqual(threshold, 1.0)

    def test_no_policy_returns_defaults(self):
        fraction, alpha, clip, exact, threshold = QUANTIZE_V3.tensor_policy(
            None, "blk.0.attn_q.weight", self.DEFAULT_FRACTION, self.DEFAULT_ALPHA
        )
        self.assertAlmostEqual(fraction, self.DEFAULT_FRACTION)
        self.assertEqual(alpha, self.DEFAULT_ALPHA)
        self.assertAlmostEqual(clip, 1.0)
        self.assertFalse(exact)
        self.assertAlmostEqual(threshold, 1.0)

    def test_substring_match_propagates_fields(self):
        # "ffn" matches "blk.0.ffn_gate.weight" but not "attn_q".
        policy = {
            "schema": SCHEMA_DFLASH,
            "tensor_families": {
                "ffn": {
                    "match": ["ffn"],
                    "outlier_fraction": 0.005,
                    "awq_alpha": 0.7,
                    "awq_clip": 1.2,
                    "ternary_threshold": 1.5,
                },
            },
        }
        fraction, alpha, clip, exact, threshold = QUANTIZE_V3.tensor_policy(
            policy, "blk.0.ffn_gate.weight", self.DEFAULT_FRACTION, self.DEFAULT_ALPHA
        )
        self.assertAlmostEqual(fraction, 0.005)
        self.assertAlmostEqual(alpha, 0.7)
        self.assertAlmostEqual(clip, 1.2)
        self.assertFalse(exact)
        self.assertAlmostEqual(threshold, 1.5)

    def test_exact_match_wins_over_substring(self):
        # "ffn" matches "blk.0.ffn_gate.weight" as a substring, but
        # `exact: True` with the full name should win on rank.
        policy = {
            "schema": SCHEMA_DFLASH,
            "tensor_families": {
                "ffn_substring": {
                    "match": ["ffn"],
                    "outlier_fraction": 0.005,
                },
                "ffn_exact": {
                    "match": ["blk.0.ffn_gate.weight"],
                    "exact": True,
                    "outlier_fraction": 1.0,  # bypass: keep all weights as outliers
                },
            },
        }
        fraction, alpha, clip, exact, threshold = QUANTIZE_V3.tensor_policy(
            policy, "blk.0.ffn_gate.weight", self.DEFAULT_FRACTION, self.DEFAULT_ALPHA
        )
        self.assertAlmostEqual(fraction, 1.0)
        self.assertTrue(exact)

    def test_awq_alpha_auto_resolves_to_none(self):
        # The "auto" sentinel for awq_alpha means "let the quantizer
        # pick"; tensor_policy surfaces this as None.
        policy = {
            "schema": SCHEMA_DFLASH,
            "tensor_families": {
                "attn": {
                    "match": ["attn_"],
                    "outlier_fraction": 0.01,
                    "awq_alpha": "auto",
                },
            },
        }
        _, alpha, _, _, _ = QUANTIZE_V3.tensor_policy(
            policy, "blk.0.attn_q.weight", self.DEFAULT_FRACTION, self.DEFAULT_ALPHA
        )
        self.assertIsNone(alpha)

    def test_ternary_threshold_field_round_trip(self):
        # The ternary_threshold is the post-audit addition that fixes
        # the 0.86% accept rate. Pin down its propagation: every value
        # we set in the policy must come out of tensor_policy unchanged.
        for value in (0.5, 1.0, 1.5, 2.0):
            policy = {
                "schema": SCHEMA_DFLASH,
                "tensor_families": {
                    "ffn": {
                        "match": ["ffn"],
                        "ternary_threshold": value,
                    },
                },
            }
            *_, threshold = QUANTIZE_V3.tensor_policy(
                policy, "blk.0.ffn_up.weight", self.DEFAULT_FRACTION, self.DEFAULT_ALPHA
            )
            self.assertAlmostEqual(threshold, value, places=6)

    def test_schemas_are_interchangeable(self):
        # The quantizer accepts both dflash and speculative schemas.
        # tensor_policy does not look at `schema`; the two wire
        # formats are equivalent for the policy lookup logic. Pin this
        # down so the audit's "unify the schemas" work is a conscious
        # change.
        families = {
            "ffn": {
                "match": ["ffn"],
                "outlier_fraction": 0.005,
                "ternary_threshold": 1.3,
            },
        }
        results = []
        for schema in (SCHEMA_DFLASH, SCHEMA_SPEC):
            policy = {"schema": schema, "tensor_families": families}
            results.append(QUANTIZE_V3.tensor_policy(
                policy, "blk.3.ffn_down.weight",
                self.DEFAULT_FRACTION, self.DEFAULT_ALPHA
            ))
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
