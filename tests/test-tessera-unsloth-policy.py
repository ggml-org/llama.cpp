#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "tessera" / "unsloth-policy.py"
SPEC = importlib.util.spec_from_file_location("tessera_unsloth_policy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class UnslothPolicyTests(unittest.TestCase):
    def test_reads_skip_list_without_importing_unsloth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "unsloth_zoo"
            package.mkdir()
            source = package / "peft_utils.py"
            source.write_text(
                'raise RuntimeError("must not import")\n'
                'SKIP_QUANTIZATION_MODULES = ["lm_head", "router"]\n',
                encoding="utf-8",
            )
            modules, resolved = MODULE.read_unsloth_skip_modules(root)
            self.assertEqual(modules, ["lm_head", "router"])
            self.assertEqual(resolved, source)

    def test_reads_nested_hugging_face_skip_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({
                "text_config": {
                    "quantization_config": {
                        "llm_int8_skip_modules": ["lm_head", "model.layers.3"]
                    }
                }
            }), encoding="utf-8")
            self.assertEqual(
                MODULE.config_skip_modules(config),
                ["lm_head", "model.layers.3"],
            )

    def test_gguf_aliases_cover_output_router_and_vision(self):
        self.assertIn("output.weight", MODULE.unique_fragments("lm_head"))
        self.assertIn("ffn_gate_inp", MODULE.unique_fragments("router"))
        self.assertIn("v.", MODULE.unique_fragments("vision_tower"))


if __name__ == "__main__":
    unittest.main()
