"""Run with python -m unittest discover -s tests -p 'test_conversion_mtp.py'."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from conversion.base import gguf
from conversion.deepseek import DeepseekV2Model, DeepseekV32Model
from conversion.gigachat import Gigachat35Model
from conversion.qwen import Qwen3NextModel


class TestMTPConversion(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.hparams: dict[str, Any] = {
            "model_type": "gigachat3_5",
            "architectures": ["GigaChat35ForCausalLM"],
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 2,
            "hidden_size": 8,
            "vocab_size": 16,
            "max_position_embeddings": 128,
            "intermediate_size": 16,
            "moe_intermediate_size": 8,
            "num_attention_heads": 2,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "first_k_dense_replace": 1,
            "n_group": 1,
            "topk_group": 1,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "rms_norm_eps": 1e-6,
            "rope_scaling": {"type": "yarn", "factor": 2, "original_max_position_embeddings": 64, "mscale": 1, "mscale_all_dim": 1},
            "use_mla_scaling_factor": True,
            "rope_interleave": True,
            "hidden_act": "silu",
            "norm_type": "ZeroCenteredGatedNorm",
            "layernorm_type": "pre_post",
            "layernorm_gating_weight": 2.0,
            "gated_attention": True,
            "linear_attention_type": "GigaChat35GatedDeltaNet",
            "linear_gating_type": "gated_rmsnorm_sigmoid_zero_centered",
            "linear_sigmoid_gate_scale": 2.0,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "full_attention_layers": [1],
            "swiglu_limit": 10,
            "index_n_heads": 2,
            "index_head_dim": 4,
            "index_topk": 8,
        }
        tensors = {
            "model.embed_tokens.weight": torch.ones(16, 8),
            "model.norm.weight": torch.ones(8),
            "lm_head.weight": torch.ones(16, 8),
            "model.norm.gate_up_projection.weight": torch.ones(3, 8),
            "model.norm.gate_down_projection.weight": torch.ones(8, 3),
        }
        tensors.update({f"model.layers.{il}.input_layernorm.weight": torch.ones(8) for il in range(4)})
        save_file(tensors, str(self.path / "model.safetensors"))

    def model(self, cls, no_mtp=False, mtp_only=False):
        # Export flags live on the converter class; keep test cases independent.
        cls = type("TestModel", (cls,), {"model_arch": cls.model_arch, "no_mtp": no_mtp, "mtp_only": mtp_only})
        hparams = copy.deepcopy(self.hparams)
        (self.path / "config.json").write_text(json.dumps(hparams))
        return cls(self.path, gguf.LlamaFileType.ALL_F32, self.path / "out.gguf", hparams=hparams, eager=True)

    def test_mtp_filtering(self):
        shared = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        gates = {"model.norm.gate_up_projection.weight", "model.norm.gate_down_projection.weight"}
        trunk = {f"model.layers.{il}.input_layernorm.weight" for il in range(2)}
        mtp = {f"model.layers.{il}.input_layernorm.weight" for il in range(2, 4)}
        for cls in (Gigachat35Model, DeepseekV32Model):
            for no_mtp, mtp_only in ((False, False), (True, False), (False, True)):
                with self.subTest(model=cls.__name__, no_mtp=no_mtp, mtp_only=mtp_only):
                    model = self.model(cls, no_mtp, mtp_only)
                    expected = shared | (set() if no_mtp else mtp) | (set() if mtp_only else trunk)
                    if cls is Gigachat35Model or not mtp_only:
                        expected |= gates
                    self.assertEqual(set(model.model_tensors), expected)
                    self.assertEqual(model.block_count, 2 if no_mtp else 4)
                    model.set_gguf_parameters()
                    key = gguf.Keys.LLM.NEXTN_PREDICT_LAYERS.format(arch=gguf.MODEL_ARCH_NAMES[model.model_arch])
                    if no_mtp:
                        self.assertNotIn(key, model.gguf_writer.kv_data[0])
                    else:
                        self.assertEqual(model.gguf_writer.kv_data[0][key].value, 2)
                        name = "model.layers.2.input_layernorm.weight"
                        self.assertTrue(list(model.modify_tensors(torch.ones(8), name, 2)))

    def test_deepseek2_still_skips_mtp(self):
        model = self.model(DeepseekV2Model)
        self.assertEqual(model.block_count, 2)
        self.assertEqual(list(model.modify_tensors(torch.ones(8), "model.layers.2.input_layernorm.weight", 2)), [])

    def test_norm_gate_metadata(self):
        for no_mtp, mtp_only in ((False, False), (True, False), (False, True)):
            with self.subTest(no_mtp=no_mtp, mtp_only=mtp_only):
                model = self.model(Gigachat35Model, no_mtp, mtp_only)
                with patch.object(model, "set_vocab"):
                    model.prepare_metadata(vocab_only=False)
                key = gguf.Keys.LLM.NORM_GATE_RANK.format(arch="gigachat35")
                self.assertEqual(model.gguf_writer.kv_data[0][key].value, 3)

    def test_qkvz_order(self):
        self.hparams.update(linear_num_key_heads=2, linear_num_value_heads=4,
                            linear_key_head_dim=2, linear_value_head_dim=3)
        data = torch.arange(256, dtype=torch.float32).reshape(32, 8)
        qkv_rows = [0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25]
        z_rows = [10, 11, 12, 13, 14, 15, 26, 27, 28, 29, 30, 31]
        for cls in (Gigachat35Model, Qwen3NextModel):
            with self.subTest(model=cls.__name__):
                model = self.model(cls, no_mtp=True)
                tensors = dict(model.modify_tensors(data, "model.layers.0.self_attn.in_proj_qkvz.weight", 0))
                torch.testing.assert_close(tensors["blk.0.attn_qkv.weight"], data[qkv_rows], rtol=0, atol=0)
                torch.testing.assert_close(tensors["blk.0.attn_gate.weight"], data[z_rows], rtol=0, atol=0)

    def test_beta_alpha_order(self):
        self.hparams.update(linear_num_key_heads=2, linear_num_value_heads=4)
        model = self.model(Gigachat35Model)
        data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        tensors = dict(model.modify_tensors(data, "model.layers.0.self_attn.in_proj_ba.weight", 0))
        torch.testing.assert_close(tensors["blk.0.ssm_beta.weight"], data[[0, 1, 4, 5]], rtol=0, atol=0)
        torch.testing.assert_close(tensors["blk.0.ssm_alpha.weight"], data[[2, 3, 6, 7]], rtol=0, atol=0)

    def test_zero_centered_norms(self):
        model = self.model(Gigachat35Model)
        for name, output, size in (
            ("model.norm.weight", "output_norm.weight", 8),
            ("model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight", 8),
            ("model.layers.0.self_attn.norm.weight", "blk.0.ssm_norm.weight", 4),
        ):
            with self.subTest(tensor=name):
                data = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6][:size], dtype=torch.float16)
                tensors = dict(model.modify_tensors(data, name, None if name == "model.norm.weight" else 0))
                expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7][:size], dtype=torch.float32)
                torch.testing.assert_close(tensors[output], expected, rtol=0, atol=0)

    def test_a_log(self):
        model = self.model(Gigachat35Model)
        data = torch.tensor([0, math.log(2), math.log(4), math.log(8)], dtype=torch.float32)
        tensors = dict(model.modify_tensors(data, "model.layers.0.self_attn.A_log", 0))
        torch.testing.assert_close(tensors["blk.0.ssm_a"], torch.tensor([-1, -2, -4, -8], dtype=torch.float32))

    def test_optional_output_norm_eps(self):
        for eps in ("absent", None, 1e-6, 2e-6):
            with self.subTest(eps=eps):
                if eps != "absent":
                    self.hparams["linear_attn_o_norm_eps"] = eps
                model = self.model(Gigachat35Model)
                if eps == 2e-6:
                    with self.assertRaisesRegex(ValueError, "linear_attn_o_norm_eps"):
                        model.set_gguf_parameters()
                else:
                    model.set_gguf_parameters()

    def test_nondefault_output_norm_eps(self):
        self.hparams["rms_norm_eps"] = 1e-5
        with self.assertRaisesRegex(ValueError, "explicit linear_attn_o_norm_eps"):
            self.model(Gigachat35Model).set_gguf_parameters()
        for eps in (None, 1e-5):
            self.hparams["linear_attn_o_norm_eps"] = eps
            self.model(Gigachat35Model).set_gguf_parameters()


if __name__ == "__main__":
    unittest.main()
