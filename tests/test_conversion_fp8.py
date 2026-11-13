"""Test FP8 conversion with small synthetic checkpoints; no GPU or model downloads required.

Install dependencies and run from the repository root:
    python -m pip install -r requirements/requirements-convert_hf_to_gguf.txt
    python -m unittest discover -s tests -p 'test_conversion_fp8.py' -v

Use unittest discovery so the repository root is available for imports.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

from conversion.base import LazyTorchTensor, gguf
from conversion.deepseek import DeepseekV2Model
from conversion.inkling2 import Inkling2Model, _E4M3, _E2M1
from conversion.llama import LlamaModel
from conversion.nemotron import NemotronHModel
from conversion.phi import PhiMoeModel
from conversion.qwen import Qwen2MoeModel, Qwen3NextModel


class ForcedF16Model(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN2MOE

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if name.endswith("q_proj.weight"):
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)


class OffsetNormModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN2MOE

    def modify_tensors(self, data_torch, name, bid):
        if name.endswith("input_layernorm.weight"):
            data_torch = data_torch + 1.0
        yield from super().modify_tensors(data_torch, name, bid)


class TestFP8Conversion(unittest.TestCase):
    def convert(self, model_class, tensors, eager=False, check_lazy=False, scan_nvfp4=False, ftype=gguf.LlamaFileType.GUESSED, hparams_override=None, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            return self._convert(Path(directory), model_class, tensors, eager, check_lazy, scan_nvfp4, ftype, hparams_override, **kwargs)

    def _convert(self, path, model_class, tensors, eager, check_lazy, scan_nvfp4, ftype, hparams_override, **kwargs):
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
        hparams.update(hparams_override or {})
        (path / "config.json").write_text(json.dumps(hparams))
        save_file(tensors, str(path / "model.safetensors"))
        model = model_class(path, ftype, path / "out.gguf", hparams=hparams, eager=eager, **kwargs)
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
            name: (info.dtype, gguf.LazyNumpyTensor.to_eager(info.tensor).copy())
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

    def test_mixed_compressed_fp8_block_scale(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        scale = torch.tensor([[0.25, 0.5], [1.0, 2.0]])
        tensors = {
            name: torch.ones((3, 32)).to(torch.float8_e4m3fn),
            name + "_scale": scale,
        }
        config = {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "config_groups": {
                "nvfp4": {"format": "nvfp4-pack-quantized"},
                "fp8": {"format": "float-quantized", "weights": {
                    "type": "float", "num_bits": 8, "strategy": "block", "block_structure": [2, 16],
                }},
            },
        }
        result = self.convert(Qwen2MoeModel, tensors, check_lazy=True, hparams_override={"quantization_config": config})
        self.assertEqual(set(result), {"blk.0.attn_q.weight"})
        dtype, data = result["blk.0.attn_q.weight"]
        self.assertEqual(dtype, gguf.GGMLQuantizationType.F16)
        expected = scale.repeat_interleave(2, 0).repeat_interleave(16, 1)[:3, :32]
        np.testing.assert_array_equal(data, expected.numpy())

    def test_mixed_compressed_fp8_block_layout_errors(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        tensors = {
            name: torch.ones((3, 32)).to(torch.float8_e4m3fn),
            name + "_scale": torch.ones((2, 2)),
        }
        for configs, message in (
            ([], "requires a block layout"),
            ([{"strategy": "block"}], "requires a block layout"),
            ([{"strategy": "block", "block_structure": [2, 16], "group_size": 16}], "requires a block layout"),
            ([{"strategy": "block", "block_structure": [2, 16]}, {"strategy": "block", "block_structure": [2, 32]}], "ambiguous block layouts"),
        ):
            with self.subTest(configs=configs):
                groups: dict[str, dict[str, object]] = {"nvfp4": {"format": "nvfp4-pack-quantized"}}
                for i, config in enumerate(configs):
                    groups[str(i)] = {"format": "float-quantized", "weights": {"type": "float", "num_bits": 8, **config}}
                config = {"quant_method": "compressed-tensors", "format": "mixed-precision", "config_groups": groups}
                with self.assertRaisesRegex(ValueError, message):
                    self.convert(Qwen2MoeModel, tensors, hparams_override={"quantization_config": config})

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

    def test_fp8_mandatory_f32(self):
        for name, target, shape in (
            ("model.layers.0.linear_attn.conv1d.weight", "blk.0.ssm_conv1d.weight", (4, 1, 4)),
            ("model.layers.0.mlp.gate.weight", "blk.0.ffn_gate_inp.weight", (2, 4)),
        ):
            for eager in (False, True):
                for scaled in (False, True):
                    with self.subTest(name=name, eager=eager, scaled=scaled):
                        weight = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape).to(torch.float8_e4m3fn)
                        tensors = {name: weight}
                        if scaled:
                            tensors[name + "_scale"] = torch.tensor([0.25])
                        result = self.convert(Qwen3NextModel, tensors, eager=eager, check_lazy=not eager, hparams_override={"mtp_num_hidden_layers": 1})
                        dtype, data = result[target]
                        self.assertEqual(dtype, gguf.GGMLQuantizationType.F32)
                        np.testing.assert_array_equal(data, weight.float().squeeze().numpy() * (0.25 if scaled else 1.))
                        self.assertNotIn(target.removesuffix("weight") + "scale", result)

    def test_fp8_norm_transform(self):
        for shape in ((4,), (2, 2)):
            with self.subTest(shape=shape):
                weight = torch.ones(shape, dtype=torch.float32).to(torch.float8_e4m3fn)
                result = self.convert(OffsetNormModel, {"model.layers.0.input_layernorm.weight": weight}, check_lazy=True)
                dtype, data = result["blk.0.attn_norm.weight"]
                self.assertEqual(dtype, gguf.GGMLQuantizationType.F32)
                np.testing.assert_array_equal(data, np.full(shape, 2.))

    def test_explicit_outtypes(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        weight = torch.ones((4, 256), dtype=torch.float32).to(torch.float8_e4m3fn)
        for ftype, qtype in (
            (gguf.LlamaFileType.ALL_F32, gguf.GGMLQuantizationType.F32),
            (gguf.LlamaFileType.MOSTLY_F16, gguf.GGMLQuantizationType.F16),
            (gguf.LlamaFileType.MOSTLY_BF16, gguf.GGMLQuantizationType.BF16),
            (gguf.LlamaFileType.MOSTLY_Q8_0, gguf.GGMLQuantizationType.Q8_0),
            (gguf.LlamaFileType.MOSTLY_TQ1_0, gguf.GGMLQuantizationType.TQ1_0),
            (gguf.LlamaFileType.MOSTLY_TQ2_0, gguf.GGMLQuantizationType.TQ2_0),
        ):
            for eager in (False, True):
                for scaled in (False, True):
                    with self.subTest(ftype=ftype, eager=eager, scaled=scaled):
                        tensors = {name: weight}
                        if scaled:
                            tensors[name + "_scale"] = torch.tensor([0.25])
                            tensors[name.removesuffix("weight") + "input_scale"] = torch.tensor([0.5])
                        result = self.convert(Qwen2MoeModel, tensors, ftype=ftype, eager=eager, check_lazy=not eager)
                        self.assertEqual(set(result), {"blk.0.attn_q.weight"})
                        dtype, data = result["blk.0.attn_q.weight"]
                        self.assertEqual(dtype, qtype)
                        np.testing.assert_allclose(gguf.dequantize(data, dtype), np.full((4, 256), 0.25 if scaled else 1.), atol=1e-4)

    def test_embedding_scale_fallback(self):
        name = "model.embed_tokens.weight"
        weight = torch.ones((4, 4), dtype=torch.float32).to(torch.float8_e4m3fn)
        for eager in (False, True):
            with self.subTest(eager=eager):
                result = self.convert(Qwen2MoeModel, {name: weight, name + "_scale": torch.tensor([0.25])}, eager=eager, check_lazy=not eager)
                self.assertEqual(set(result), {"token_embd.weight"})
                dtype, data = result["token_embd.weight"]
                self.assertEqual(dtype, gguf.GGMLQuantizationType.F16)
                np.testing.assert_array_equal(data, np.full((4, 4), 0.25))

    def test_fp8_scale_formats(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        weight = torch.ones((4, 4), dtype=torch.float32).to(torch.float8_e4m3fn)
        for suffix, config in (
            ("_scale_inv", {"quant_method": "fp8"}),
            ("_scale", {"quant_method": "compressed-tensors", "format": "float-quantized", "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 8, "strategy": "tensor"}}}}),
        ):
            for ftype in (gguf.LlamaFileType.GUESSED, gguf.LlamaFileType.MOSTLY_F16):
                with self.subTest(suffix=suffix, ftype=ftype):
                    result = self.convert(Qwen2MoeModel, {name: weight, name + suffix: torch.tensor([0.25])}, ftype=ftype, hparams_override={"quantization_config": config}, check_lazy=True)
                    dtype, data = result["blk.0.attn_q.weight"]
                    if ftype == gguf.LlamaFileType.GUESSED:
                        self.assertEqual(dtype, gguf.GGMLQuantizationType.F8_E4M3)
                        np.testing.assert_array_equal(result["blk.0.attn_q.scale"][1], [0.25])
                    else:
                        self.assertEqual(dtype, gguf.GGMLQuantizationType.F16)
                        self.assertEqual(set(result), {"blk.0.attn_q.weight"})
                        np.testing.assert_array_equal(data, np.full((4, 4), 0.25))

    def test_expert_naming(self):
        for model_class, prefix, projections, hparams in (
            (LlamaModel, "model.layers.0.block_sparse_moe", (("w1", "gate"), ("w3", "up"), ("w2", "down")), {"num_local_experts": 2}),
            (PhiMoeModel, "model.layers.0.block_sparse_moe", (("w1", "gate"), ("w3", "up"), ("w2", "down")), {"max_position_embeddings": 16, "rope_scaling": {"original_max_position_embeddings": 16}}),
            (NemotronHModel, "backbone.layers.0.mixer", (("up_proj", "up"), ("down_proj", "down")), {"num_experts_per_tok": 1, "n_routed_experts": 2, "head_dim": 4, "num_heads": 1, "n_groups": 1, "hybrid_override_pattern": "E"}),
        ):
            for eager in (False, True):
                for mixed in (False, True):
                    with self.subTest(model=model_class.__name__, eager=eager, mixed=mixed):
                        tensors = {}
                        for expert in (1, 0):
                            for proj, _ in projections:
                                name = f"{prefix}.experts.{expert}.{proj}.weight"
                                dtype = torch.float16 if mixed and expert == 1 else torch.float8_e4m3fn
                                tensors[name] = torch.ones((4, 4), dtype=torch.float32).to(dtype)
                                tensors[name + "_scale"] = torch.tensor([0.25 * (expert + 1)])
                        result = self.convert(model_class, tensors, hparams_override=hparams, eager=eager, check_lazy=not eager)
                        for _, proj in projections:
                            dtype, data = result[f"blk.0.ffn_{proj}_exps.weight"]
                            if mixed:
                                self.assertEqual(dtype, gguf.GGMLQuantizationType.F16)
                                np.testing.assert_array_equal(data, np.array([np.full((4, 4), 0.25), np.full((4, 4), 0.5)]))
                                self.assertNotIn(f"blk.0.ffn_{proj}_exps.scale", result)
                            else:
                                self.assertEqual(dtype, gguf.GGMLQuantizationType.F8_E4M3)
                                np.testing.assert_array_equal(gguf.dequantize(data, dtype), np.ones((2, 4, 4)))
                                np.testing.assert_array_equal(result[f"blk.0.ffn_{proj}_exps.scale"][1], [0.25, 0.5])

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


class TestInkling2PackedConversion(unittest.TestCase):
    def fixture(self):
        tensors, groups, expected = {}, [], {}
        for layer, nvfp4 in enumerate((True, False)):
            names = []
            for projection, shape in (("w13", (2, 320, 64)), ("w2", (2, 64, 160))):
                name = f"model.llm.layers.{layer}.mlp.experts.{projection}_weight"
                names.append(name)
                e, r, c = shape
                grid = torch.arange(e * r * c).reshape(shape)
                if nvfp4:
                    codes = ((grid * 7 + grid // c) % 16).to(torch.uint8)
                    packed = codes[:, :, ::2] | (codes[:, :, 1::2] << 4)
                    scales = (torch.arange(e * r * (c // 16)).reshape(e, r, c // 16) % 126 + 1).to(torch.uint8).view(torch.float8_e4m3fn)
                    amax = torch.tensor([0.7, 1.9])
                    den = torch.tensor([3.0 if projection == "w13" else 7.0])
                    tensors[name + "_amax"] = amax
                    tensors[name + "_outer_scale_denominator"] = den
                    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
                    decoded = lut[codes.long()] * scales.float().repeat_interleave(16, -1)
                    decoded *= (amax / den)[:, None, None]
                else:
                    packed = ((grid * 13 + grid // c) % 256).to(torch.uint8)
                    packed[(packed & 127) == 127] = 126
                    packed = packed.view(torch.float8_e4m3fn)
                    scales = (torch.arange(e * r * (c // 32)).reshape(e, r, c // 32) % 7 + 123).to(torch.uint8)
                    scales[0, 0, 0] = 0
                    decoded = packed.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
                tensors[name] = packed
                tensors[name + "_scale"] = scales
                if projection == "w13":
                    decoded = decoded.reshape(e, r // 2, 2, c).transpose(1, 2).reshape(shape)
                key = f"blk.{layer}.ffn_{'gate_up' if projection == 'w13' else 'down'}_exps.weight"
                expected[key] = decoded.numpy()
            groups.append({"tensors": names, "weight_transform": "nvfp4_nearest" if nvfp4 else "mxfp8_nearest", "weight_scale_format": "amax_denominator", "weight_block_shape": [1, 16] if nvfp4 else [32, 32]})
        return tensors, groups, expected

    def model(self, path, tensors, groups, ftype=gguf.LlamaFileType.ALL_F32, eager=False):
        hp = {"model_type": "inkling2_text", "architectures": ["Inkling2ForCausalLM"], "num_hidden_layers": 2, "hidden_size": 64, "moe_intermediate_size": 160, "n_routed_experts": 2}
        (path / "config.json").write_text(json.dumps(hp))
        (path / "hf_quant_config.json").write_text(json.dumps({"model": {"parameter_groups": groups}}))
        save_file(tensors, path / "model.safetensors")
        return Inkling2Model(path, ftype, path / "unused.gguf", hparams=hp, eager=eager)

    def test_lookup_semantics(self):
        reference = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
        np.testing.assert_array_equal(_E4M3.view(np.uint32)[np.isfinite(reference)], reference.view(np.uint32)[np.isfinite(reference)])
        np.testing.assert_array_equal(np.flatnonzero(np.isnan(_E4M3)), [127, 255])
        self.assertTrue(np.signbit(_E2M1[8]))

    def test_mixed_schemes_and_streaming_outtypes(self):
        tensors, groups, expected = self.fixture()
        for eager in (False, True):
            for ftype, qtype in ((gguf.LlamaFileType.GUESSED, gguf.GGMLQuantizationType.F16), (gguf.LlamaFileType.ALL_F32, gguf.GGMLQuantizationType.F32), (gguf.LlamaFileType.MOSTLY_F16, gguf.GGMLQuantizationType.F16), (gguf.LlamaFileType.MOSTLY_BF16, gguf.GGMLQuantizationType.BF16)):
                with self.subTest(eager=eager, outtype=qtype), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    model = self.model(path, tensors, groups, ftype, eager)
                    with self.assertLogs("hf-to-gguf", level="WARNING") as logs:
                        model.prepare_tensors()
                    self.assertTrue(any("[32, 32]" in m and "[1, 32]" in m for m in logs.output))
                    infos = model.gguf_writer.tensors[0]
                    self.assertEqual(infos.keys(), expected.keys())
                    for name, info in infos.items():
                        self.assertEqual(info.dtype, qtype)
                        self.assertEqual(tuple(info.shape), expected[name].shape)
                        self.assertIsInstance(info.tensor, gguf.LazyChunkedTensor)
                        with self.assertRaises(TypeError):
                            np.asarray(info.tensor)
                        for load in info.tensor._chunks:
                            self.assertLessEqual(load().shape[0], 128)
                        output = path / "tensor.bin"
                        with output.open("wb") as f:
                            info.tensor.tofile(f)
                        reference = gguf.quantize(expected[name], qtype)
                        self.assertEqual(output.read_bytes(), reference.tobytes(), name)

    def test_projection_and_companions_in_separate_shards(self):
        tensors, groups, expected = self.fixture()
        name = groups[0]["tensors"][0]
        tensors = {k: v for k, v in tensors.items() if k.startswith(name)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.model(path, tensors, groups)
            (path / "model.safetensors").rename(path / "model-00001-of-00002.safetensors")
            save_file({name: tensors.pop(name)}, path / "model-00001-of-00002.safetensors")
            save_file(tensors, path / "model-00002-of-00002.safetensors")
            model = Inkling2Model(path, gguf.LlamaFileType.ALL_F32, path / "unused.gguf")
            model.prepare_tensors()
            key = "blk.0.ffn_gate_up_exps.weight"
            infos = model.gguf_writer.tensors[0]
            self.assertEqual(list(infos), [key])
            tensor = infos[key].tensor
            assert tensor is not None
            with (path / "tensor.bin").open("wb") as f:
                tensor.tofile(f)
            self.assertEqual((path / "tensor.bin").read_bytes(), expected[key].tobytes())

    def test_float_experts_with_stale_quant_config(self):
        _, groups, expected = self.fixture()
        tensors = {}
        for layer in range(2):
            for projection, target in (("w13", "gate_up"), ("w2", "down")):
                values = torch.from_numpy(expected[f"blk.{layer}.ffn_{target}_exps.weight"]).double()
                e, r, c = values.shape
                if projection == "w13":
                    values = values.reshape(e, 2, r // 2, c).transpose(1, 2).reshape(e, r, c)
                tensors[f"model.llm.layers.{layer}.mlp.experts.{projection}_weight"] = values
        with tempfile.TemporaryDirectory() as directory:
            model = self.model(Path(directory), tensors, groups)
            model.prepare_tensors()
            for name, info in model.gguf_writer.tensors[0].items():
                actual = gguf.LazyNumpyTensor.to_eager(info.tensor)
                np.testing.assert_array_equal(actual.view(np.uint32), expected[name].view(np.uint32))

    def test_invalid_metadata_and_companions(self):
        cases = ("missing", "dtype", "amax_shape", "den_shape", "grid", "block", "scheme", "orphan", "mxfp8_amax")
        for case in cases:
            tensors, groups, _ = self.fixture()
            name = groups[0]["tensors"][0]
            if case == "missing":
                del tensors[name + "_amax"]
            elif case == "dtype":
                tensors[name + "_scale"] = tensors[name + "_scale"].view(torch.uint8)
            elif case == "amax_shape":
                tensors[name + "_amax"] = torch.ones(1)
            elif case == "den_shape":
                tensors[name + "_outer_scale_denominator"] = torch.ones(2)
            elif case == "grid":
                tensors[name + "_scale"] = tensors[name + "_scale"][:, :, :3].contiguous()
            elif case == "block":
                tensors[name + "_scale"] = tensors[name + "_scale"][:, ::2].contiguous()
            elif case == "scheme":
                groups[0]["weight_transform"] = "unknown"
            elif case == "orphan":
                del tensors[name]
            else:
                tensors[groups[1]["tensors"][0] + "_amax"] = torch.ones(2)
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                model = self.model(Path(directory), tensors, groups)
                with self.assertRaisesRegex(ValueError, "Inkling2:"):
                    model.prepare_tensors()

    def test_invalid_values_name_expert(self):
        for layer, suffix, value in ((0, "_scale", 127), (0, "_scale", 255), (1, "", 127), (1, "", 255), (1, "_scale", 255), (0, "_amax", float("nan")), (0, "_outer_scale_denominator", 0.0)):
            tensors, groups, _ = self.fixture()
            name = groups[layer]["tensors"][0]
            target = tensors[name + suffix]
            if suffix in ("_amax", "_outer_scale_denominator"):
                target[-1] = value
            else:
                target.view(torch.uint8)[1, 0, 0] = value
            with self.subTest(layer=layer, suffix=suffix, value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                model = self.model(path, tensors, groups)
                model.prepare_tensors()
                tensor = model.gguf_writer.tensors[0][f"blk.{layer}.ffn_gate_up_exps.weight"].tensor
                with (path / "tensor.bin").open("wb") as f, self.assertRaisesRegex(ValueError, r"w13_weight.*expert [01]"):
                    tensor.tofile(f)


if __name__ == "__main__":
    unittest.main()
