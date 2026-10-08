#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))

import numpy as np
import torch
from safetensors.torch import save_file

import gguf
from conversion import get_model_class
from conversion.base import LazyTorchTensor
from conversion.decision2 import _Decision2Mixin, _decision2_base, _load_decision2_hparams


class TestDecision2Conversion(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "backbone").mkdir()
        self.config = {
            "model_type": "decision2", "architectures": ["Decision2Model"],
            "backbone": {"config": "backbone/config.json"},
            "decision_weights": {"decision_head": "decision_head.safetensors"},
            "model_config": "decision_config.json", "max_input_tokens": 256,
            "manifest": "MODEL_MANIFEST.json", "calibration": None,
        }
        self.hparams = {
            "model_type": "qwen3", "architectures": ["Qwen3Model"],
            "hidden_size": 4, "head_dim": 2, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "intermediate_size": 8, "max_position_embeddings": 1024,
            "vocab_size": 4, "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 2,
            "linear_value_head_dim": 2, "linear_num_key_heads": 1,
            "linear_num_value_heads": 1, "full_attention_interval": 4,
        }
        self.write_json("config.json", self.config)
        self.write_json("backbone/config.json", self.hparams)
        self.write_json("decision_config.json", {
            "prompt_version": "decision2-segmented-options-global-query-v1",
            "head_dim": 2, "max_options": 255, "head_variant": "shared",
        })
        self.head = {}
        for name in ("candidate_norm", "query_norm"):
            self.head[name + ".weight"] = torch.full((4,), 1.25)
            self.head[name + ".bias"] = torch.full((4,), -0.125)
        for name in ("key", "query", "candidate_mlp", "query_mlp"):
            self.head[name + ".weight"] = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 7
        self.head["candidate_mlp.bias"] = torch.tensor([0.125, -0.25])
        self.head["scalar.weight"] = torch.tensor([[0.125, -0.25]])
        save_file(self.head, self.root / "decision_head.safetensors")
        save_file({"embed_tokens.weight": torch.ones(4, 4), "norm.weight": torch.ones(4)}, self.root / "backbone/model.safetensors")

    def write_json(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def model(self, model_type="qwen3"):
        self.hparams["model_type"] = model_type
        self.write_json("backbone/config.json", self.hparams)
        hparams = _load_decision2_hparams(self.root)
        cls = get_model_class(hparams["architectures"][0])
        return cls(self.root, gguf.LlamaFileType.MOSTLY_BF16, self.root / "model.gguf", hparams=hparams)

    def test_dispatch_and_package_context_limit(self):
        for model_type, expected in (("qwen3", gguf.MODEL_ARCH.QWEN3), ("qwen3_5_text", gguf.MODEL_ARCH.QWEN35)):
            model = self.model(model_type)
            self.assertEqual(model.model_arch, expected)
            self.assertEqual(model.hparams["max_position_embeddings"], 256)
            self.assertEqual(model.dir_model, self.root)
            self.assertEqual(set(model.model_tensors), {"embed_tokens.weight", "norm.weight"})

    def test_head_is_not_offset_by_qwen35_norm_conversion(self):
        model = self.model("qwen3_5_text")
        name = "decision2.candidate_norm.weight"
        converted = dict(model.modify_tensors(self.head["candidate_norm.weight"], name, None))
        torch.testing.assert_close(converted[name], self.head["candidate_norm.weight"], rtol=0, atol=0)
        backbone = dict(model.modify_tensors(torch.ones(4), "norm.weight", None))
        torch.testing.assert_close(backbone["output_norm.weight"], torch.full((4,), 2.0))

    def test_all_head_tensors_remain_f32_in_bf16_file(self):
        model = self.model()
        model.prepare_tensors()
        model.set_gguf_parameters()
        model.gguf_writer.write_header_to_file(path=model.fname_out)
        model.gguf_writer.write_kv_data_to_file()
        model.gguf_writer.write_tensors_to_file()
        model.gguf_writer.close()
        reader = gguf.GGUFReader(model.fname_out)
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        for name, expected in self.head.items():
            tensor = tensors["decision2." + name]
            self.assertEqual(tensor.tensor_type, gguf.GGMLQuantizationType.F32)
            np.testing.assert_array_equal(tensor.data.reshape(expected.shape), expected.numpy())
        self.assertEqual(tensors["token_embd.weight"].tensor_type, gguf.GGMLQuantizationType.BF16)
        decision_type = reader.get_field("qwen3.decision.type")
        head_dim = reader.get_field("qwen3.decision.head_dim")
        assert decision_type is not None and head_dim is not None
        self.assertEqual(decision_type.contents(), "decision2")
        self.assertEqual(head_dim.contents(), 2)

    def test_malformed_head_is_rejected(self):
        self.head["key.weight"] = torch.ones(3, 4)
        save_file(self.head, self.root / "decision_head.safetensors")
        with self.assertRaisesRegex(ValueError, "head tensor"):
            self.model()

    def test_score_bias_preserved_and_validated(self):
        offsets = [0.039188, 0.203049, 0.079362, -0.15162, -0.169979]
        self.write_json("score_bias.json", {"format": "dev2-score-bias-v1", "offsets": {"5": offsets}})
        model = self.model()
        self.assertEqual(model.score_bias, {5: offsets})
        model.set_gguf_parameters()
        for index, value in enumerate(offsets):
            self.assertEqual(model.gguf_writer.kv_data[0][f"qwen3.decision.score_bias.5.{index}"].value, value)
        self.write_json("score_bias.json", {"format": "dev2-score-bias-v1", "offsets": {"5": [float("nan")] * 5}})
        with self.assertRaisesRegex(ValueError, "score offsets"):
            _Decision2Mixin._load_score_bias(self.root)

    def test_adapter_merge_is_fp32_and_checks_unmerged_weights(self):
        model = self.model()
        (self.root / "adapter").mkdir()
        self.write_json("adapter/adapter_config.json", {"peft_type": "LORA", "r": 2, "lora_alpha": 4, "bias": "none"})
        a = torch.tensor([[0.01, 0.03, 0.07, 0.11], [-0.02, 0.05, -0.13, 0.17]], dtype=torch.bfloat16)
        b = torch.arange(16, dtype=torch.bfloat16).reshape(8, 2) / 9
        save_file({"base_model.model.layers.0.mlp.gate_proj.lora_A.weight": a,
                   "base_model.model.layers.0.mlp.gate_proj.lora_B.weight": b}, self.root / "adapter/adapter_model.safetensors")
        model._load_adapter(self.root, {"config": "adapter/adapter_config.json", "weights": ["adapter/adapter_model.safetensors"]})
        with self.assertRaisesRegex(ValueError, "Unmerged"):
            model.prepare_tensors()
        weight = torch.ones(8, 4, dtype=torch.bfloat16)
        name, converted = next(iter(model.modify_tensors(weight, "model.layers.0.mlp.gate_proj.weight", 0)))
        self.assertEqual(name, "blk.0.ffn_gate.weight")
        converted = LazyTorchTensor.to_eager(converted)
        self.assertEqual(converted.dtype, torch.float32)
        torch.testing.assert_close(converted, weight.float() + 2 * (b.float() @ a.float()), rtol=0, atol=0)
        self.assertEqual(model.lora_merged, {"layers.0.mlp.gate_proj.weight"})

    def test_pinned_base_is_required(self):
        self.config["backbone"] = {"repository": "example/base", "revision": "a" * 40}
        self.write_json("MODEL_MANIFEST.json", {"base": {"repo_id": "example/base", "revision": "b" * 40}})
        with self.assertRaisesRegex(ValueError, "pinned manifest"):
            _decision2_base(self.root, self.config)
        self.write_json("MODEL_MANIFEST.json", {"base": {"repo_id": "example/base", "revision": "a" * 40}})
        self.assertEqual(_decision2_base(self.root, self.config), ("example/base", "a" * 40))

    def test_pinned_base_config_download(self):
        self.config["backbone"] = {"repository": "example/base", "revision": "a" * 40}
        self.write_json("config.json", self.config)
        path = self.root / "backbone/config.json"
        self.write_json("MODEL_MANIFEST.json", {"base": {"repo_id": "example/base", "revision": "a" * 40,
                         "files_sha256": {"config.json": hashlib.sha256(path.read_bytes()).hexdigest()}}})
        with patch("huggingface_hub.hf_hub_download", return_value=str(path)) as download:
            hparams = _load_decision2_hparams(self.root)
        download.assert_called_once_with("example/base", "config.json", revision="a" * 40)
        self.assertEqual(hparams["architectures"], ["Decision2Qwen3Model"])


if __name__ == "__main__":
    unittest.main()
