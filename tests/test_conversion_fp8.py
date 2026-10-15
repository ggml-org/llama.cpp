from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversion.base import LazyTorchTensor, gguf
from conversion.deepseek import DeepseekV2Model
from conversion.qwen import Qwen2MoeModel


class ForcedF16Model(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN2MOE

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if name.endswith("q_proj.weight"):
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)


class TestFP8Conversion(unittest.TestCase):
    def convert(self, model_class, tensors, eager=False, check_lazy=False, scan_nvfp4=False, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            hparams = {
                "model_type": "qwen2_moe",
                "architectures": ["Qwen2MoeForCausalLM"],
                "num_hidden_layers": 1,
                "hidden_size": 4,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_experts": 2,
                "qk_nope_head_dim": 2,
                "v_head_dim": 2,
                "quantization_config": {"quant_method": "modelopt"},
            }
            (path / "config.json").write_text(json.dumps(hparams))
            save_file(tensors, str(path / "model.safetensors"))
            model = model_class(path, gguf.LlamaFileType.MOSTLY_F16, path / "out.gguf", hparams=hparams, eager=eager, **kwargs)
            reads: list[str] = []
            if check_lazy:
                self.assertFalse(eager)

                def track(name, gen):
                    tensor = gen()

                    def load(source):
                        reads.append(name)
                        return LazyTorchTensor.to_eager(source())

                    return LazyTorchTensor(meta=torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="meta"), args=(gen,), func=load)

                model.model_tensors = {
                    name: lambda name=name, gen=gen: track(name, gen)
                    for name, gen in model.model_tensors.items()
                }
            if scan_nvfp4:
                model._generate_nvfp4_tensors()
                self.assertEqual(reads, [])
            model.prepare_tensors()
            if check_lazy:
                deferred = {name for name, tensor in tensors.items() if tensor.numel() != 1 or name.endswith((".weight", ".bias"))}
                self.assertFalse(deferred.intersection(reads), f"Materialized during preparation: {reads}")
                for name, info in model.gguf_writer.tensors[0].items():
                    if name.endswith((".weight", ".bias")):
                        self.assertIsInstance(info.tensor, gguf.LazyNumpyTensor)
            result = {
                name: (info.dtype, gguf.LazyNumpyTensor.to_eager(info.tensor))
                for name, info in model.gguf_writer.tensors[0].items()
            }
            if check_lazy:
                self.assertTrue(deferred.intersection(reads))
            return result

    def test_lazy_fp8_export(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        weight = torch.tensor([[1., -1.], [2., -2.]]).to(torch.float8_e4m3fn)
        for channel_scale in (False, True):
            for scan_nvfp4 in (False, True):
                with self.subTest(channel_scale=channel_scale, scan_nvfp4=scan_nvfp4):
                    tensors = {
                        name: weight,
                        name + "_scale": torch.full((2, 1) if channel_scale else (1,), 0.25),
                    }
                    result = self.convert(Qwen2MoeModel, tensors, check_lazy=True, scan_nvfp4=scan_nvfp4)
                    dtype, data = result["blk.0.attn_q.weight"]
                    if channel_scale:
                        self.assertEqual(dtype, gguf.GGMLQuantizationType.F16)
                        np.testing.assert_array_equal(data, weight.float().numpy() * 0.25)
                    else:
                        self.assertEqual(dtype, gguf.GGMLQuantizationType.F8_E4M3)
                        np.testing.assert_array_equal(gguf.dequantize(data, dtype), weight.float().numpy())
                        np.testing.assert_array_equal(result["blk.0.attn_q.scale"][1], [0.25])

    def test_mla_scale_fallback(self):
        weight = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.float8_e4m3fn)
        prefix = "model.layers.0.self_attn."
        tensors = {}
        for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"):
            tensors[prefix + proj + ".weight"] = weight.clone()
            tensors[prefix + proj + ".weight_scale"] = torch.tensor([0.25])
        for eager in (False, True):
            with self.subTest(eager=eager):
                result = self.convert(DeepseekV2Model, tensors, eager=eager)
                self.assertFalse(any(name.endswith(".scale") for name in result))
                np.testing.assert_array_equal(result["blk.0.attn_k_b.weight"][1], (weight.float()[:2].T * 0.25).numpy()[None])
                np.testing.assert_array_equal(result["blk.0.attn_v_b.weight"][1], (weight.float()[2:] * 0.25).numpy()[None])

    def test_output_type_override(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        weight = torch.tensor([[1., -1.], [2., -2.]]).to(torch.float8_e4m3fn)
        for eager in (False, True):
            for scaled in (False, True):
                with self.subTest(eager=eager, scaled=scaled):
                    tensors = {name: weight}
                    if scaled:
                        tensors[name + "_scale"] = torch.tensor([0.25])
                    result = self.convert(ForcedF16Model, tensors, eager=eager, check_lazy=not eager)
                    dtype, data = result["blk.0.attn_q.weight"]
                    self.assertEqual(dtype, gguf.GGMLQuantizationType.F16)
                    self.assertNotIn("blk.0.attn_q.scale", result)
                    np.testing.assert_array_equal(data, weight.float().numpy() * (0.25 if scaled else 1.))

    def test_fp8_bias_is_numeric_f32(self):
        bias = torch.tensor([1., -1.]).to(torch.float8_e4m3fn)
        result = self.convert(Qwen2MoeModel, {"model.layers.0.self_attn.q_proj.bias": bias}, check_lazy=True)
        dtype, data = result["blk.0.attn_q.bias"]
        self.assertEqual(dtype, gguf.GGMLQuantizationType.F32)
        np.testing.assert_array_equal(data, [1., -1.])

    def test_mixed_experts(self):
        for eager in (False, True):
            for other in ("f16", "channel", "unscaled"):
                with self.subTest(eager=eager, other=other):
                    tensors = {}
                    for expert in range(2):
                        for proj in ("gate", "up", "down"):
                            name = f"model.layers.0.mlp.experts.{expert}.{proj}_proj.weight"
                            dtype = torch.float16 if expert == 1 and other == "f16" else torch.float8_e4m3fn
                            tensors[name] = torch.ones((4, 4), dtype=torch.float32).to(dtype)
                            if expert == 0:
                                tensors[name + "_scale"] = torch.tensor([0.25])
                            elif other == "channel":
                                tensors[name + "_scale"] = torch.full((4, 1), 0.5)
                    result = self.convert(Qwen2MoeModel, tensors, eager=eager, check_lazy=not eager)
                    self.assertEqual(len(result), 3)
                    for proj in ("gate", "up", "down"):
                        dtype, data = result[f"blk.0.ffn_{proj}_exps.weight"]
                        self.assertEqual(dtype, gguf.GGMLQuantizationType.F16)
                        np.testing.assert_array_equal(data[0], np.full((4, 4), 0.25))
                        np.testing.assert_array_equal(data[1], np.full((4, 4), 0.5 if other == "channel" else 1.))

    def test_uniform_experts_stay_fp8(self):
        tensors = {}
        for expert in range(2):
            for proj in ("gate", "up", "down"):
                name = f"model.layers.0.mlp.experts.{expert}.{proj}_proj.weight"
                tensors[name] = torch.ones((4, 4), dtype=torch.float32).to(torch.float8_e4m3fn)
                tensors[name + "_scale"] = torch.tensor([0.25 * (expert + 1)])
        result = self.convert(Qwen2MoeModel, tensors, fuse_gate_up_exps=True, check_lazy=True)
        for proj in ("gate", "up", "down"):
            dtype, data = result[f"blk.0.ffn_{proj}_exps.weight"]
            self.assertEqual(dtype, gguf.GGMLQuantizationType.F8_E4M3)
            np.testing.assert_array_equal(gguf.dequantize(data, dtype), np.ones((2, 4, 4)))
            np.testing.assert_array_equal(result[f"blk.0.ffn_{proj}_exps.scale"][1], [0.25, 0.5])

        unscaled = {name: tensor for name, tensor in tensors.items() if name.endswith(".weight")}
        result = self.convert(Qwen2MoeModel, unscaled, check_lazy=True)
        self.assertEqual(len(result), 3)
        for dtype, data in result.values():
            self.assertEqual(dtype, gguf.GGMLQuantizationType.F8_E4M3)
            np.testing.assert_array_equal(gguf.dequantize(data, dtype), np.ones((2, 4, 4)))

    def test_fp8_as_q8(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        tensors = {
            name: torch.ones((4, 32), dtype=torch.float32).to(torch.float8_e4m3fn),
            name + "_scale": torch.tensor([0.25]),
        }
        result = self.convert(Qwen2MoeModel, tensors, fp8_as_q8=True)
        self.assertNotIn("blk.0.attn_q.scale", result)
        dtype, data = result["blk.0.attn_q.weight"]
        self.assertEqual(dtype, gguf.GGMLQuantizationType.Q8_0)
        np.testing.assert_allclose(gguf.dequantize(data, dtype), np.full((4, 32), 0.25), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
