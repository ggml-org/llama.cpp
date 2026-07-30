#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "tools" / "tessera" / "engram_hash.py"
SPEC = importlib.util.spec_from_file_location("tessera_engram_hash", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EngramHashTests(unittest.TestCase):
    def test_hash_manifest_round_trip(self):
        spec = MODULE.make_hash_spec(
            compressed_vocab_size=256,
            vocab_size_per_ngram=[1000, 2000],
            max_ngram_size=3,
            heads_per_ngram=2,
            layer_ids=[1, 15],
            pad_id=2,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hash.json"
            path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
            loaded = MODULE.load_spec(path)
        self.assertEqual(spec, loaded)

    def test_hash_is_deterministic_and_bounded(self):
        spec = MODULE.make_hash_spec(
            compressed_vocab_size=128,
            vocab_size_per_ngram=[997, 1997],
            max_ngram_size=3,
            heads_per_ngram=3,
            layer_ids=[1],
            pad_id=2,
            seed=0,
        )
        tokens = np.array([[11, 29, 7, 101]], dtype=np.int64)
        first = MODULE.hash_layer(tokens, spec, 1)
        second = MODULE.hash_layer(tokens, spec, 1)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (1, 4, 6))
        for index, modulus in enumerate(
            spec.layer_moduli[1][0] + spec.layer_moduli[1][1]
        ):
            self.assertTrue(np.all(first[:, :, index] >= 0))
            self.assertTrue(np.all(first[:, :, index] < modulus))

    def test_normalizer_collapses_case_accents_and_space(self):
        self.assertEqual(MODULE.normalize_piece("  CAFÉ\t"), "cafe")

    def test_digest_rejects_mutation(self):
        spec = MODULE.make_hash_spec(128, [997], 2, 1, [1], 2, 0)
        payload = spec.to_dict()
        payload["pad_id"] = 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hash.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                MODULE.load_spec(path)


if __name__ == "__main__":
    unittest.main()
