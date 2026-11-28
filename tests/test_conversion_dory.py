"""Dory metadata validation and scale-vector storage regression tests."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from conversion.base import gguf
from conversion.dory import DoryModel


def config():
    return dict(architectures=["DoryForCausalLM"], model_type="dory", vocab_size=32,
                hidden_size=256, num_hidden_layers=6, intermediate_size=256,
                num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=64,
                n_input_layers=2, n_recurrent_layers=2, n_output_layers=2, n_recurrent_loops=2,
                head_dim=64, hidden_act="silu", initializer_range=.0177, ngpt_alpha_init=.05,
                attention_output_gate_logit_scale_init=None, sliding_window=4,
                rope_parameters={
                    "sliding_attention": dict(rope_type="default", rope_theta=10000),
                    "full_attention": dict(rope_type="default", rope_theta=10000000)},
                layer_types=["full_attention", "mlp", "sliding_attention", "mlp", "full_attention", "mlp"])


class TestDoryConversion(unittest.TestCase):
    def make_model(self, path, hparams, ftype=gguf.LlamaFileType.MOSTLY_BF16, tensors=None):
        (path / "config.json").write_text(json.dumps(hparams))
        save_file(tensors or {"lm_head.weight": torch.ones(32, 256)}, str(path / "model.safetensors"))
        return DoryModel(path, ftype, path / "out.gguf", hparams=hparams, eager=True)

    def test_metadata(self):
        for loops in (1, 2, 4):
            for mode in ("per_loop", "last_loop"):
                with self.subTest(loops=loops, mode=mode), tempfile.TemporaryDirectory() as directory:
                    h = dict(config(), n_recurrent_loops=loops, recurrent_kv_cache_mode=mode)
                    model = self.make_model(Path(directory), h)
                    model.set_gguf_parameters()
                    kv = model.gguf_writer.kv_data[0]
                    self.assertEqual(kv["dory.num_loops"].value, loops)
                    self.assertEqual(kv["dory.recurrent_kv_cache_mode"].value, mode)
                    self.assertEqual(kv["dory.gate_init"].value, 16)
                    self.assertEqual(kv["dory.rope.profile"].value, [1, 1, 2, 1, 1, 1])

    def test_required_counts(self):
        for key in ("n_input_layers", "n_recurrent_layers", "n_output_layers", "n_recurrent_loops"):
            for value in (None, 0, -1, 1.5, True):
                with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                    h = config()
                    if value is None:
                        del h[key]
                    else:
                        h[key] = value
                    model = self.make_model(Path(directory), h)
                    with self.assertRaisesRegex(ValueError, key):
                        model.set_gguf_parameters()

    def test_invalid_gate_scale(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                h = dict(config(), attention_output_gate_logit_scale_init=value)
                model = self.make_model(Path(directory), h)
                with self.assertRaisesRegex(ValueError, "attention_output_gate_logit_scale_init"):
                    model.set_gguf_parameters()

    def test_explicit_gate_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.make_model(Path(directory), dict(config(), attention_output_gate_logit_scale_init=2.5))
            model.set_gguf_parameters()
            self.assertEqual(model.gguf_writer.kv_data[0]["dory.gate_init"].value, 2.5)

    def test_scale_storage(self):
        tensors = {
            "backbone.layers.0.attn_alpha": torch.ones(256),
            "backbone.layers.0.mixer.sqk": torch.ones(512),
            "backbone.layers.0.mixer.s_gate": torch.ones(512),
            "backbone.layers.1.mixer.suv_gate": torch.ones(256),
            "backbone.layers.1.mixer.suv_up": torch.ones(256),
            "lm_head.logit_scale": torch.ones(32),
        }
        for ftype, dtype in ((gguf.LlamaFileType.MOSTLY_BF16, gguf.GGMLQuantizationType.BF16),
                             (gguf.LlamaFileType.MOSTLY_F16, gguf.GGMLQuantizationType.F16),
                             (gguf.LlamaFileType.ALL_F32, gguf.GGMLQuantizationType.F32),
                             (gguf.LlamaFileType.MOSTLY_Q8_0, gguf.GGMLQuantizationType.F32)):
            with self.subTest(ftype=ftype), tempfile.TemporaryDirectory() as directory:
                model = self.make_model(Path(directory), config(), ftype, tensors)
                model.prepare_tensors()
                stored = model.gguf_writer.tensors[0]
                self.assertEqual(set(stored), {"blk.0.residual_alpha.weight", "blk.0.attn_qk_scale.weight",
                                               "blk.0.attn_gate_scale.weight", "blk.1.ffn_gate_scale.weight",
                                               "blk.1.ffn_up_scale.weight", "output_scale.weight"})
                self.assertTrue(all(info.dtype == dtype for info in stored.values()))


if __name__ == "__main__":
    unittest.main()
