import re
import unittest
from pathlib import Path


class TestTurboTables(unittest.TestCase):
    def test_cpu_sycl_vulkan_centroids_agree(self):
        root = Path(__file__).resolve().parents[2] / "ggml" / "src"
        paths = [
            "ggml-turbo-quant.c",
            "ggml-sycl/turbo-quants.hpp",
            "ggml-vulkan/vulkan-shaders/copy_to_quant.comp",
            "ggml-vulkan/vulkan-shaders/dequant_turbo3_0.comp",
            "ggml-vulkan/vulkan-shaders/dequant_funcs.glsl",
            "ggml-vulkan/vulkan-shaders/dequant_funcs_cm2.glsl",
            "ggml-vulkan/vulkan-shaders/flash_attn_dequant.glsl",
        ]
        pattern = r"float\s+\w+\[8\]\s*=\s*(?:float\[8\])?[({]([^})]+)[})]"
        expected = None
        for path in paths:
            with self.subTest(path=path):
                tables = re.findall(pattern, (root / path).read_text())
                self.assertGreaterEqual(len(tables), 1)
                # Later eight-entry tables belong to the TQ3 weight format.
                values = [float(x) for x in re.findall(r"-?\d+\.\d+", tables[0])]
                self.assertEqual(len(values), 8)
                if expected is None:
                    expected = values
                self.assertEqual(values, expected)


if __name__ == "__main__":
    unittest.main()
