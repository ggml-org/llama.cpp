"""Test MiniCPM5 MoE conversion with small local checkpoints.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_conversion_minicpm.py' -v
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

from conversion.base import gguf
from conversion.minicpm import MiniCPM5MoEModel


class TestMiniCPM5MoEConversion(unittest.TestCase):
    def convert(self, tensors, eager=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        hparams = {
            "model_type": "minicpm5_moe",
            "architectures": ["MiniCPM5MoEForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 8,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "n_shared_experts": 1,
            "moe_intermediate_size": 3,
            "first_k_dense_replace": 1,
            "routed_scaling_factor": 3.66,
            "use_gated_attention": True,
        }
        (path / "config.json").write_text(json.dumps(hparams))
        save_file(tensors, str(path / "model.safetensors"))
        model = MiniCPM5MoEModel(path, gguf.LlamaFileType.ALL_F32, path / "out.gguf", hparams=hparams, eager=eager)
        model.prepare_tensors()
        result = {
            name: gguf.LazyNumpyTensor.to_eager(info.tensor).copy()
            for name, info in model.gguf_writer.tensors[0].items()
        }
        return model, result

    def test_query_gate_and_gqa_key_layout(self):
        query_gate = torch.arange(64, dtype=torch.float32).reshape(16, 4)
        key = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        for eager in (False, True):
            with self.subTest(eager=eager):
                _, result = self.convert({
                    "model.layers.0.self_attn.q_proj.weight": query_gate,
                    "model.layers.0.self_attn.k_proj.weight": key,
                }, eager=eager)
                np.testing.assert_array_equal(
                    result["blk.0.attn_q.weight"],
                    query_gate[[0, 1, 2, 3, 8, 9, 10, 11]].numpy(),
                )
                np.testing.assert_array_equal(
                    result["blk.0.attn_gate.weight"],
                    query_gate[[4, 5, 6, 7, 12, 13, 14, 15]].numpy(),
                )
                np.testing.assert_array_equal(result["blk.0.attn_k.weight"], key.numpy())

    def test_expert_axis_and_router_bias(self):
        tensors = {}
        for expert in (1, 0):
            for projection in ("down_proj", "up_proj", "gate_proj"):
                shape = (4, 3) if projection == "down_proj" else (3, 4)
                values = torch.arange(12, dtype=torch.float32).reshape(shape) + 100 * expert
                tensors[f"model.layers.1.mlp.experts.{expert}.{projection}.weight"] = values
        tensors["model.layers.1.mlp.gate.e_score_correction_bias"] = torch.tensor([0.25, -0.5])
        _, result = self.convert(tensors)
        for projection, tensor_name in (("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"), ("down_proj", "ffn_down_exps")):
            for expert in range(2):
                np.testing.assert_array_equal(
                    result[f"blk.1.{tensor_name}.weight"][expert],
                    tensors[f"model.layers.1.mlp.experts.{expert}.{projection}.weight"].numpy(),
                )
        np.testing.assert_array_equal(result["blk.1.exp_probs_b.bias"], [0.25, -0.5])

    def test_metadata_preserves_expanded_attention_and_moe(self):
        model, _ = self.convert({"model.layers.0.self_attn.q_proj.weight": torch.zeros(16, 4)})
        model.set_gguf_parameters()
        metadata = model.gguf_writer.kv_data[0]
        expected = {
            "embedding_length": 4,
            "attention.head_count": 2,
            "attention.head_count_kv": 1,
            "attention.key_length": 4,
            "attention.value_length": 4,
            "rope.dimension_count": 4,
            "feed_forward_length": 8,
            "expert_count": 2,
            "expert_used_count": 1,
            "expert_shared_count": 1,
            "expert_feed_forward_length": 3,
            "leading_dense_block_count": 1,
            "expert_weights_norm": True,
            "expert_weights_scale": 3.66,
        }
        for name, value in expected.items():
            self.assertEqual(metadata[f"minicpm5_moe.{name}"].value, value)


if __name__ == "__main__":
    unittest.main()
