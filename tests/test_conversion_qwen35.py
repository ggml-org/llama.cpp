"""Run with python -m unittest discover -s tests -p test_conversion_qwen35.py -v."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

from conversion.base import gguf
from conversion.qwen import Qwen3_5TextModel, Qwen3_5MoeTextModel
from conversion.qwen4exp import Qwen4ExpTextModel


class SeparateArchitectureModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.CLEF


class TestQwen35Conversion(unittest.TestCase):
    def projections(self):
        return {
            f"model.layers.0.linear_attn.{name}.weight": torch.arange(rows * 16, dtype=torch.float32).reshape(rows, 16) + offset
            for name, rows, offset in (("in_proj_qkv", 20, 0), ("in_proj_z", 12, 1000), ("in_proj_b", 6, 2000), ("in_proj_a", 6, 3000))
        }

    def convert(self, tensors, eager=False, model_class=Qwen3_5TextModel, quantized=False, hadamard=False, ftype=gguf.LlamaFileType.ALL_F32, infer_layer_types=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            hparams = {
                "architectures": ["Qwen3_5ForCausalLM"],
                "num_hidden_layers": 2,
                "mtp_num_hidden_layers": 1,
                "hidden_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 6,
                "linear_key_head_dim": 2,
                "linear_value_head_dim": 2,
                "layer_types": ["linear_attention", "full_attention"],
            }
            if infer_layer_types:
                del hparams["layer_types"]
                hparams["full_attention_interval"] = 2
            if quantized:
                hparams["quantization_config"] = {"quant_method": "modelopt"}
            if hadamard:
                (path / "hadamard_packing.json").write_text('{"tensors": [{"name": "model.layers.0.linear_attn.in_proj_qkv.weight"}]}')
            save_file(tensors, str(path / "model.safetensors"))
            model = model_class(path, ftype, path / "out.gguf", hparams=hparams, eager=eager)
            model.prepare_tensors()
            return {
                name: gguf.LazyNumpyTensor.to_eager(info.tensor).copy()
                for name, info in model.gguf_writer.tensors[0].items()
            }

    def test_merges_after_v_head_reorder(self):
        tensors = self.projections()
        v_rows = [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11]
        head_rows = [0, 3, 1, 4, 2, 5]
        qkv = tensors["model.layers.0.linear_attn.in_proj_qkv.weight"]
        expected_qkvz = torch.cat((qkv[:8], qkv[8:][v_rows], tensors["model.layers.0.linear_attn.in_proj_z.weight"][v_rows]))
        expected_ba = torch.cat((tensors["model.layers.0.linear_attn.in_proj_b.weight"][head_rows], tensors["model.layers.0.linear_attn.in_proj_a.weight"][head_rows]))
        for eager in (False, True):
            with self.subTest(eager=eager):
                converted = self.convert(tensors, eager=eager)
                self.assertEqual(set(converted), {"blk.0.attn_qkv_gate.weight", "blk.0.ssm_beta_alpha.weight"})
                np.testing.assert_array_equal(converted["blk.0.attn_qkv_gate.weight"], expected_qkvz.numpy())
                np.testing.assert_array_equal(converted["blk.0.ssm_beta_alpha.weight"], expected_ba.numpy())

    def test_missing_half(self):
        tensors = self.projections()
        del tensors["model.layers.0.linear_attn.in_proj_a.weight"]
        with self.assertRaisesRegex(ValueError, "merged weights missing a half"):
            self.convert(tensors)

    def test_inferred_layer_types(self):
        converted = self.convert(self.projections(), infer_layer_types=True)
        self.assertEqual(set(converted), {"blk.0.attn_qkv_gate.weight", "blk.0.ssm_beta_alpha.weight"})

    def test_prequantized_and_hadamard_stay_separate(self):
        for options in ({"quantized": True}, {"hadamard": True}):
            with self.subTest(**options):
                converted = self.convert(self.projections(), **options)
                self.assertEqual(set(converted), {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight"})

    def test_full_attention_stays_separate(self):
        tensors = {f"model.layers.1.self_attn.{name}_proj.weight": torch.ones(rows, 16) for name, rows in (("q", 32), ("k", 8), ("v", 8))}
        converted = self.convert(tensors)
        self.assertEqual(set(converted), {"blk.1.attn_q.weight", "blk.1.attn_k.weight", "blk.1.attn_v.weight"})

    def test_preserved_fp8_without_quantization_config(self):
        tensors = self.projections()
        name = "model.layers.0.linear_attn.in_proj_qkv.weight"
        tensors[name] = tensors[name].to(torch.float8_e4m3fn)
        tensors[name + "_scale"] = torch.tensor(0.5)
        converted = self.convert(tensors, ftype=gguf.LlamaFileType.GUESSED)
        self.assertIn("blk.0.attn_qkv.weight", converted)
        self.assertIn("blk.0.attn_qkv.scale", converted)
        self.assertIn("blk.0.attn_gate.weight", converted)
        self.assertNotIn("blk.0.attn_qkv_gate.weight", converted)

    def test_moe_stays_separate(self):
        converted = self.convert(self.projections(), model_class=Qwen3_5MoeTextModel)
        self.assertEqual(set(converted), {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight"})

    def test_other_architecture_stays_separate(self):
        converted = self.convert(self.projections(), model_class=SeparateArchitectureModel)
        self.assertEqual(set(converted), {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight"})

    def test_flash_next_merges(self):
        tensors = self.projections()
        tensors.update({
            f"model.layers.0.{module}.{name}.weight": torch.ones(rows, 16)
            for module in ("attn_hyper_connection", "mlp_hyper_connection")
            for name, rows in (("input_mix_weight_down", 4), ("block_inject_weight", 2))
        })
        converted = self.convert(tensors, model_class=Qwen4ExpTextModel)
        self.assertEqual(set(converted), {"blk.0.attn_qkv_gate.weight", "blk.0.ssm_beta_alpha.weight", "blk.0.hc_attn_down_inject.weight", "blk.0.hc_ffn_down_inject.weight"})


if __name__ == "__main__":
    unittest.main()
