#!/usr/bin/env python3

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestConvertHfToGguf(unittest.TestCase):
    def test_phi3_medium_arch_is_registered(self):
        source = (ROOT / "convert_hf_to_gguf.py").read_text(encoding="utf-8")
        module = ast.parse(source)

        for node in module.body:
            if not isinstance(node, ast.ClassDef) or node.name != "Phi3MiniModel":
                continue

            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                if not isinstance(decorator.func, ast.Attribute):
                    continue
                if decorator.func.attr != "register":
                    continue

                args = [
                    arg.value for arg in decorator.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ]

                self.assertIn("Phi3ForCausalLM", args)
                self.assertIn("Phi3MediumForCausalLM", args)
                return

        self.fail("Phi3MiniModel registration was not found in convert_hf_to_gguf.py")


if __name__ == "__main__":
    unittest.main()
