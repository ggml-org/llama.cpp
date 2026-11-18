"""Test Loop conversion using a synthetic checkpoint without model downloads."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from conversion import get_model_class
from conversion.base import gguf
from conversion.sensenova_loop import SenseNovaLoopModel, SenseNovaLoopVisionModel


class TestSenseNovaLoopConversion(unittest.TestCase):
    def test_training_architecture(self):
        arch = "Qwen3_5LoopForConditionalGeneration"
        self.assertIs(get_model_class(arch), SenseNovaLoopModel)
        self.assertIs(get_model_class(arch, mmproj=True), SenseNovaLoopVisionModel)

    @staticmethod
    def hparams() -> dict[str, Any]:
        return {
            "architectures": ["SenseNovaLoopForConditionalGeneration"],
            "model_type": "sensenova_loop",
            "num_hidden_layers": 4,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "layer_types": ["linear_attention", "full_attention", "linear_attention", "linear_attention"],
            "loop_settings": {"start": 1, "end": 3, "eval_count": 2, "counts": [2], "use_gate": False},
            "mhc": {"start": 1, "end": 3, "streams": 2, "iterations": 20, "fixed_identity": False},
        }

    def test_shared_layers_and_precision(self):
        hp = self.hparams()
        tensors = {
            "model.language_model.loop_gate.logits": torch.ones(8),
            "model.language_model.layers.1.mhc_attn.weight": torch.full((8, 16), 0.1234567),
            "model.language_model.layers.1.mhc_attn.bias": torch.arange(8, dtype=torch.float32),
            "model.language_model.layers.2.input_layernorm.weight": torch.zeros(8),
            "model.language_model.layers.3.linear_attn.in_proj_qkv.weight": torch.arange(128, dtype=torch.float32).reshape(16, 8),
            "model.language_model.layers.3.linear_attn.in_proj_z.weight": torch.ones(8, 8),
            "model.language_model.layers.3.linear_attn.conv1d.weight": torch.ones(16, 1, 4),
            "mtp.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8),
        }
        original = copy.deepcopy(hp)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(hp))
            save_file(tensors, str(path / "model.safetensors"))
            model = SenseNovaLoopModel(path, gguf.LlamaFileType.MOSTLY_BF16, path / "out.gguf", hparams=hp)
            model.set_gguf_parameters()
            model.prepare_tensors()
            self.assertEqual(hp, original)
            self.assertEqual(model.plan, [0, 1, 2, 1, 2, 3])
            kv = model.gguf_writer.kv_data[0]
            self.assertEqual(kv["sensenova_loop.block_count"].value, 6)
            self.assertEqual(kv["sensenova_loop.attention.recurrent_layers"].value, [True, False, True, False, True, True])
            result = model.gguf_writer.tensors[0]
            self.assertEqual(set(result), {"blk.1.hc_attn_fn.weight", "blk.1.hc_attn_fn.bias", "blk.2.attn_norm.weight", "blk.5.attn_qkv_gate.weight", "blk.5.ssm_conv1d.weight"})
            router = result["blk.1.hc_attn_fn.weight"]
            self.assertEqual(router.dtype, gguf.GGMLQuantizationType.F32)
            np.testing.assert_array_equal(gguf.LazyNumpyTensor.to_eager(router.tensor), tensors["model.language_model.layers.1.mhc_attn.weight"].numpy())
            self.assertEqual(result["blk.5.ssm_conv1d.weight"].dtype, gguf.GGMLQuantizationType.F32)
            np.testing.assert_array_equal(gguf.LazyNumpyTensor.to_eager(result["blk.2.attn_norm.weight"].tensor), np.ones(8))
            self.assertEqual(result["blk.5.attn_qkv_gate.weight"].shape, (24, 8))

            for settings in ({"eval_count": 256, "counts": [256]}, {"start": 3, "end": 2}, {"counts": [1, 2]}, {"use_gate": True}):
                invalid = copy.deepcopy(original)
                invalid["loop_settings"].update(settings)
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    SenseNovaLoopModel(path, gguf.LlamaFileType.MOSTLY_BF16, path / "invalid.gguf", hparams=invalid)

    def test_mtp_export_modes(self):
        hp = self.hparams()
        hp["mtp_num_hidden_layers"] = 1
        tensors = {
            "model.language_model.embed_tokens.weight": torch.ones(32, 8),
            "model.language_model.norm.weight": torch.zeros(8),
            "lm_head.weight": torch.ones(32, 8),
            "model.language_model.layers.3.input_layernorm.weight": torch.zeros(8),
            "mtp.fc.weight": torch.ones(8, 16),
            "mtp.pre_fc_norm_embedding.weight": torch.zeros(8),
            "mtp.pre_fc_norm_hidden.weight": torch.zeros(8),
            "mtp.norm.weight": torch.zeros(8),
            "mtp.layers.0.self_attn.q_proj.weight": torch.ones(16, 8),
        }
        common = {"token_embd.weight", "output_norm.weight", "output.weight"}
        draft = {"blk.6.nextn.eh_proj.weight", "blk.6.nextn.enorm.weight", "blk.6.nextn.hnorm.weight", "blk.6.nextn.shared_head_norm.weight", "blk.6.attn_q.weight"}
        for layout in ("embedded", "indexed", "standalone"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as directory:
                self.check_mtp_export_modes(Path(directory), hp, tensors, common, draft, layout)

    def check_mtp_export_modes(self, path, hp, tensors, common, draft, layout):
        (path / "config.json").write_text(json.dumps(hp))
        main = {name: tensor for name, tensor in tensors.items() if not name.startswith("mtp.")}
        mtp = {name: tensor for name, tensor in tensors.items() if name.startswith("mtp.")}
        if layout == "embedded":
            save_file(tensors, str(path / "model.safetensors"))
            # An indexed head takes precedence over a stale sidecar.
            save_file({name: tensor + 3 for name, tensor in mtp.items()}, str(path / "mtp.safetensors"))
            weight_map = {name: "model.safetensors" for name in tensors}
        else:
            save_file(main, str(path / "model.safetensors"))
            save_file(mtp, str(path / "mtp.safetensors"))
            weight_map = {name: "model.safetensors" for name in main}
            if layout == "indexed":
                weight_map.update({name: "mtp.safetensors" for name in mtp})
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        for no_mtp, mtp_only in ((False, False), (True, False), (False, True)):
            with self.subTest(no_mtp=no_mtp, mtp_only=mtp_only), patch.object(SenseNovaLoopModel, "no_mtp", no_mtp), patch.object(SenseNovaLoopModel, "mtp_only", mtp_only):
                model = SenseNovaLoopModel(path, gguf.LlamaFileType.MOSTLY_BF16, path / "out.gguf", hparams=copy.deepcopy(hp))
                model.set_gguf_parameters()
                model.prepare_tensors()
                kv = model.gguf_writer.kv_data[0]
                self.assertEqual(kv["sensenova_loop.block_count"].value, 6 if no_mtp else 7)
                self.assertEqual(kv["sensenova_loop.attention.recurrent_layers"].value, [True, False, True, False, True, True] + ([] if no_mtp else [False]))
                result = model.gguf_writer.tensors[0]
                expected = common | (set() if mtp_only else {"blk.5.attn_norm.weight"}) | (set() if no_mtp else draft)
                self.assertEqual(set(result), expected)
                if not no_mtp:
                    self.assertEqual(kv["sensenova_loop.nextn_predict_layers"].value, 1)
                    np.testing.assert_array_equal(gguf.LazyNumpyTensor.to_eager(result["blk.6.nextn.hnorm.weight"].tensor), np.ones(8))
                else:
                    self.assertNotIn("sensenova_loop.nextn_predict_layers", kv)
        for count in (0, 2, -1, True):
            invalid = {**hp, "mtp_num_hidden_layers": count}
            with self.subTest(mtp_layers=count), patch.object(SenseNovaLoopModel, "mtp_only", True), self.assertRaises(ValueError):
                SenseNovaLoopModel(path, gguf.LlamaFileType.MOSTLY_BF16, path / "invalid.gguf", hparams=invalid)

    def test_vision_pixel_limits(self):
        hp = {
            "architectures": ["SenseNovaLoopForConditionalGeneration"],
            "text_config": {"hidden_size": 8},
            "vision_config": {
                "depth": 1, "hidden_size": 8, "intermediate_size": 16, "num_heads": 2,
                "num_position_embeddings": 16, "patch_size": 2, "spatial_merge_size": 2,
            },
        }
        processor = {
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            "size": {"shortest_edge": 65536, "longest_edge": 16777216},
        }
        for nested in (False, True):
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "config.json").write_text(json.dumps(hp))
                name = "processor_config.json" if nested else "preprocessor_config.json"
                config = {"image_processor": processor} if nested else processor
                (path / name).write_text(json.dumps(config))
                save_file({"model.visual.pos_embed.weight": torch.zeros(16, 8)}, str(path / "model.safetensors"))
                model = SenseNovaLoopVisionModel(path, gguf.LlamaFileType.MOSTLY_F16, path / "out.gguf", hparams=copy.deepcopy(hp))
                model.set_gguf_parameters()
                kv = model.gguf_writer.kv_data[0]
                self.assertEqual(kv["clip.vision.image_min_pixels"].value, 65536)
                self.assertEqual(kv["clip.vision.image_max_pixels"].value, 16777216)


if __name__ == "__main__":
    unittest.main()
