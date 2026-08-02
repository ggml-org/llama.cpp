#!/usr/bin/env python3

import unittest
from pathlib import Path
import os
import sys
import tempfile

import numpy as np

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf


class TestGGUFWriterTypeValidation(unittest.TestCase):

    def _make_writer(self):
        fd, path = tempfile.mkstemp(suffix=".gguf")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        return gguf.GGUFWriter(path, "llama")

    # --- STRING validation ---

    def test_rejects_list_for_string(self):
        """The original bug: model card license as a list instead of string."""
        w = self._make_writer()
        with self.assertRaises(TypeError) as ctx:
            w.add_key_value("general.license", ["Apache-2.0"], gguf.GGUFValueType.STRING)
        self.assertIn("general.license", str(ctx.exception))
        self.assertIn("STRING", str(ctx.exception))

    def test_rejects_int_for_string(self):
        w = self._make_writer()
        with self.assertRaises(TypeError) as ctx:
            w.add_key_value("general.name", 42, gguf.GGUFValueType.STRING)
        self.assertIn("STRING", str(ctx.exception))

    def test_accepts_str_for_string(self):
        w = self._make_writer()
        w.add_key_value("general.name", "test", gguf.GGUFValueType.STRING)

    def test_accepts_bytes_for_string(self):
        w = self._make_writer()
        w.add_key_value("tokenizer.ggml.precompiled_charsmap", b"\x01\x02", gguf.GGUFValueType.STRING)

    # --- BOOL vs INT trap ---

    def test_rejects_bool_for_uint32(self):
        """Python bool is subclass of int -- must be explicitly rejected."""
        w = self._make_writer()
        with self.assertRaises(TypeError) as ctx:
            w.add_key_value("general.file_type", True, gguf.GGUFValueType.UINT32)
        self.assertIn("UINT32", str(ctx.exception))

    def test_rejects_bool_for_int32(self):
        w = self._make_writer()
        with self.assertRaises(TypeError) as ctx:
            w.add_key_value("answer", False, gguf.GGUFValueType.INT32)
        self.assertIn("INT32", str(ctx.exception))

    def test_rejects_numpy_bool_for_uint32(self):
        """np.bool_ must also be rejected for integer types."""
        w = self._make_writer()
        with self.assertRaises(TypeError):
            w.add_key_value("general.file_type", np.bool_(True), gguf.GGUFValueType.UINT32)

    def test_accepts_python_bool_for_bool(self):
        w = self._make_writer()
        w.add_key_value("general.use_parallel_residual", True, gguf.GGUFValueType.BOOL)

    def test_accepts_numpy_bool_for_bool(self):
        w = self._make_writer()
        w.add_key_value("general.use_parallel_residual", np.bool_(True), gguf.GGUFValueType.BOOL)

    # --- FLOAT accepts int ---

    def test_accepts_int_for_float32(self):
        """int literals should be accepted for float types (struct.pack handles this)."""
        w = self._make_writer()
        w.add_key_value("attention.layer_norm_rms_eps", 0, gguf.GGUFValueType.FLOAT32)

    def test_accepts_int_for_float64(self):
        w = self._make_writer()
        w.add_key_value("test", 1, gguf.GGUFValueType.FLOAT64)

    def test_rejects_bool_for_float32(self):
        """bool should NOT be accepted for float types even though bool is int subclass."""
        w = self._make_writer()
        with self.assertRaises(TypeError):
            w.add_key_value("temperature", True, gguf.GGUFValueType.FLOAT32)

    def test_accepts_float_for_float32(self):
        w = self._make_writer()
        w.add_key_value("temperature", 0.7, gguf.GGUFValueType.FLOAT32)

    # --- NumPy scalars ---

    def test_accepts_numpy_int32_for_uint32(self):
        w = self._make_writer()
        w.add_key_value("general.file_type", np.int32(7), gguf.GGUFValueType.UINT32)

    def test_accepts_numpy_uint64_for_uint64(self):
        w = self._make_writer()
        w.add_key_value("test", np.uint64(42), gguf.GGUFValueType.UINT64)

    def test_accepts_numpy_float32_for_float32(self):
        w = self._make_writer()
        w.add_key_value("temperature", np.float32(0.5), gguf.GGUFValueType.FLOAT32)

    def test_accepts_numpy_float64_for_float64(self):
        w = self._make_writer()
        w.add_key_value("test", np.float64(1.5), gguf.GGUFValueType.FLOAT64)

    # --- ARRAY validation ---

    def test_rejects_non_sequence_for_array(self):
        w = self._make_writer()
        with self.assertRaises(TypeError):
            w.add_key_value("general.tags", 42, gguf.GGUFValueType.ARRAY)

    def test_rejects_mixed_type_array_with_string_sub_type(self):
        w = self._make_writer()
        with self.assertRaises(TypeError):
            w.add_key_value("general.tags", ["ok", 1], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)

    def test_accepts_string_array_with_string_sub_type(self):
        w = self._make_writer()
        w.add_key_value("general.tags", ["conversational", "code"], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)

    def test_accepts_int_array_with_int32_sub_type(self):
        w = self._make_writer()
        w.add_key_value("test", [1, 2, 3], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT32)

    def test_rejects_bool_in_int_array(self):
        """bool must be rejected even inside integer arrays."""
        w = self._make_writer()
        with self.assertRaises(TypeError):
            w.add_key_value("test", [1, True], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT32)

    # --- convenience methods still work ---

    def test_add_uint32_still_works(self):
        w = self._make_writer()
        w.add_uint32("general.file_type", 7)

    def test_add_float32_still_works(self):
        w = self._make_writer()
        w.add_float32("attention.layer_norm_rms_eps", 1e-5)

    def test_add_bool_still_works(self):
        w = self._make_writer()
        w.add_bool("general.use_parallel_residual", False)

    def test_add_string_still_works(self):
        w = self._make_writer()
        w.add_string("general.name", "TestModel")

    def test_add_array_still_works(self):
        w = self._make_writer()
        w.add_array("general.tags", ["conversational", "code"])

    # --- additional edge cases from code review ---

    def test_rejects_str_for_array(self):
        """str is a Sequence subclass but never a valid GGUF array value."""
        w = self._make_writer()
        with self.assertRaises(TypeError) as ctx:
            w.add_key_value("general.tags", "not-a-list", gguf.GGUFValueType.ARRAY)
        self.assertIn("ARRAY", str(ctx.exception))
        self.assertIn("str", str(ctx.exception))

    def test_accepts_bytes_for_array(self):
        """bytes is a legitimate UINT8 array representation (e.g. precompiled_charsmap)."""
        w = self._make_writer()
        w.add_key_value("test", b"\x01\x02", gguf.GGUFValueType.ARRAY)

    def test_rejects_none_for_string(self):
        w = self._make_writer()
        with self.assertRaises(TypeError):
            w.add_key_value("general.name", None, gguf.GGUFValueType.STRING)

    def test_rejects_none_for_uint32(self):
        w = self._make_writer()
        with self.assertRaises(TypeError):
            w.add_key_value("general.file_type", None, gguf.GGUFValueType.UINT32)

    def test_rejects_float_for_uint32(self):
        """float must be rejected for integer types."""
        w = self._make_writer()
        with self.assertRaises(TypeError) as ctx:
            w.add_key_value("general.file_type", 7.5, gguf.GGUFValueType.UINT32)
        self.assertIn("UINT32", str(ctx.exception))

    def test_rejects_float_for_int32(self):
        w = self._make_writer()
        with self.assertRaises(TypeError):
            w.add_key_value("test", 1.0, gguf.GGUFValueType.INT32)

    # --- error message quality ---

    def test_error_message_includes_key_name_and_value(self):
        """Error message must show the key and actual value for easy debugging."""
        w = self._make_writer()
        with self.assertRaises(TypeError) as ctx:
            w.add_key_value("general.license", ["Apache-2.0"], gguf.GGUFValueType.STRING)
        msg = str(ctx.exception)
        self.assertIn("general.license", msg)
        self.assertIn("list", msg)
        self.assertIn("['Apache-2.0']", msg)


if __name__ == "__main__":
    unittest.main()
