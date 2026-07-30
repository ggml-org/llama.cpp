#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "tessera" / "engram_contract.py"
SPEC = importlib.util.spec_from_file_location("tessera_engram_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EngramContractTests(unittest.TestCase):
    def manifest(self):
        return {
            "schema": "llama.tessera.engram-hash.v1",
            "digest": "hash",
            "max_ngram_size": 3,
            "heads_per_ngram": 2,
            "layer_moduli": {
                "1": [[101, 103], [107, 109]],
                "15": [[113, 127], [131, 137]],
            },
        }

    def test_contract_has_bounded_rowwise_memory(self):
        contract = MODULE.make_contract(self.manifest(), 256, 16, [1, 15], "abc")
        self.assertEqual(contract["memory_encoding"], "rowwise-q8")
        self.assertEqual(contract["estimated_memory_bytes"], 14848)
        metadata = MODULE.gguf_metadata(contract)
        self.assertEqual(metadata["tessera.engram.layers"], [1, 15])

    def test_layer_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "layers"):
            MODULE.make_contract(self.manifest(), 256, 16, [1], "abc")


if __name__ == "__main__":
    unittest.main()
