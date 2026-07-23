#!/usr/bin/env python3

import unittest
from pathlib import Path
import os
import sys

import numpy as np

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf


class TestGGUFWriter(unittest.TestCase):

    def test_add_key_value_rejects_string_declared_type_with_list_value(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        with self.assertRaisesRegex(ValueError, "declared STRING"):
            writer.add_key_value("general.license", ["apache-2.0"], gguf.GGUFValueType.STRING)

    def test_add_key_value_rejects_array_with_invalid_declared_sub_type_item(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        with self.assertRaisesRegex(ValueError, "declared ARRAY"):
            writer.add_key_value("general.tags", [1], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)

    def test_add_key_value_rejects_mixed_type_array_value(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        with self.assertRaisesRegex(ValueError, "declared ARRAY"):
            writer.add_key_value("general.tags", [1, "apache-2.0"], gguf.GGUFValueType.ARRAY)

    def test_add_key_value_rejects_empty_array_value(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        with self.assertRaisesRegex(ValueError, "declared ARRAY"):
            writer.add_key_value("general.tags", [], gguf.GGUFValueType.ARRAY)

    def test_add_key_value_accepts_string_array_with_declared_string_sub_type(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        writer.add_key_value("general.tags", ["apache-2.0"], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)

    def test_add_key_value_accepts_bytes_array_value(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        writer.add_key_value("tokenizer.ggml.precompiled_charsmap", b"\x01\x02", gguf.GGUFValueType.ARRAY)

    def test_add_key_value_accepts_homogeneous_integer_array_value(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        writer.add_key_value("test.array", [1, 2], gguf.GGUFValueType.ARRAY)

    def test_add_uint32_still_accepts_python_int(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        writer.add_uint32("answer", 42)

    def test_add_key_value_accepts_numpy_uint32_for_declared_uint32(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        writer.add_key_value("answer", np.uint32(42), gguf.GGUFValueType.UINT32)

    def test_add_key_value_rejects_bool_for_declared_uint32(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        with self.assertRaisesRegex(ValueError, "declared UINT32"):
            writer.add_key_value("answer", True, gguf.GGUFValueType.UINT32)

    def test_add_key_value_rejects_numpy_bool_for_declared_uint32(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        with self.assertRaisesRegex(ValueError, "declared UINT32"):
            writer.add_key_value("answer", np.bool_(True), gguf.GGUFValueType.UINT32)

    def test_add_key_value_accepts_numpy_float32_for_declared_float32(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        writer.add_key_value("temperature", np.float32(1.5), gguf.GGUFValueType.FLOAT32)

    def test_add_key_value_rejects_python_int_for_declared_float32(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        with self.assertRaisesRegex(ValueError, "declared FLOAT32"):
            writer.add_key_value("temperature", 1, gguf.GGUFValueType.FLOAT32)

    def test_add_key_value_rejects_python_int_for_declared_float64(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        with self.assertRaisesRegex(ValueError, "declared FLOAT64"):
            writer.add_key_value("temperature", 1, gguf.GGUFValueType.FLOAT64)

    def test_add_key_value_accepts_numpy_bool_for_declared_bool(self):
        writer = gguf.GGUFWriter("/tmp/test.gguf", "llama")

        writer.add_key_value("flag", np.bool_(True), gguf.GGUFValueType.BOOL)


if __name__ == "__main__":
    unittest.main()
