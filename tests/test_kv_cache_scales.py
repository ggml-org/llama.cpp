"""Test KV cache scales with synthetic tensors; no GPU or model downloads required.

Install dependencies and run from the repository root:
    python -m pip install -r requirements/requirements-convert_hf_to_gguf.txt
    python -m pip install -e ./gguf-py
    python -m unittest discover -s tests -p 'test_kv_cache_scales.py' -v

Use unittest discovery so the repository root is available for imports.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from conversion.base import ModelBase
import gguf


class TestKVCacheScales(unittest.TestCase):
    def test_projection_names(self):
        cases = (
            (gguf.MODEL_ARCH.PHI3, "model.layers.2.self_attn.qkv_proj", "model.layers.2.self_attn.qkv_proj"),
            (gguf.MODEL_ARCH.GPT2, "transformer.h.2.attn.c_attn", "transformer.h.2.attn.c_attn"),
            (gguf.MODEL_ARCH.LLAMA, "model.layers.2.self_attn.k_proj", "model.layers.2.self_attn.v_proj"),
        )
        for arch, k_name, v_name in cases:
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "scales.gguf"
                model = object.__new__(ModelBase)
                model.tensor_map = gguf.get_tensor_name_map(arch, 3)
                model.gguf_writer = gguf.GGUFWriter(path, gguf.MODEL_ARCH_NAMES[arch])
                model.model_tensors = {
                    k_name + ".k_scale": lambda: torch.tensor(0.5, dtype=torch.float32),
                    v_name + ".v_scale": lambda: torch.tensor(1.0, dtype=torch.float32),
                }
                model._prepare_kv_cache_scales()
                self.assertFalse(model.model_tensors)
                model.gguf_writer.write_header_to_file()
                model.gguf_writer.write_kv_data_to_file()
                model.gguf_writer.write_tensors_to_file()
                model.gguf_writer.close()

                reader = gguf.GGUFReader(path)
                self.assertEqual([t.name for t in reader.tensors], ["blk.2.attn_k.k_scale", "blk.2.attn_v.v_scale"])
                self.assertEqual([float(t.data[0]) for t in reader.tensors], [0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
