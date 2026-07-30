#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "tessera" / "make-awq-layer-bundles.py"
SPEC = importlib.util.spec_from_file_location("tessera_awq_bundles", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TesseraBundleSelectionTest(unittest.TestCase):
    def test_selection_covers_model_depth(self):
        sources = [
            MODULE.TensorSource(
                Path("model.safetensors"),
                f"model.layers.{layer}.self_attn.q_proj.weight",
                f"blk.{layer}.attn_q.weight",
                "attention",
                (4096, 4096),
            )
            for layer in range(48)
        ]
        selected = MODULE.stratified_sources(sources, 8)
        indices = [MODULE.layer_index(source.observer_name) for source in selected]
        self.assertEqual(len(selected), 8)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 47)
        self.assertGreater(len(set(indices)), 6)

    def test_selection_preserves_small_families(self):
        sources = [
            MODULE.TensorSource(
                Path("model.safetensors"),
                "model.embed_tokens.weight",
                "token_embd.weight",
                "output_embedding",
                (256000, 4096),
            )
        ]
        self.assertEqual(MODULE.stratified_sources(sources, 24), sources)


if __name__ == "__main__":
    unittest.main()
