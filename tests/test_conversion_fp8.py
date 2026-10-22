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


class ReorderedRowsModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN2MOE

    def modify_tensors(self, data_torch, name, bid):
        if name.endswith("q_proj.weight"):
            data_torch = data_torch.flip(0)
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
            deferred = {name for name, tensor in tensors.items() if name in model.model_tensors and (tensor.numel() != 1 or name.endswith((".weight", ".bias")))}
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
                    self.assertEqual(dtype, gguf.GGMLQuantizationType.F8_E4M3)
                    np.testing.assert_array_equal(gguf.dequantize(data, dtype), weight.float().numpy())
                    np.testing.assert_array_equal(result["blk.0.attn_q.scale"][1], [0.25, 0.25] if channel_scale else [0.25])

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

    def test_channel_scales_follow_rope_permutation(self):
        for n_head, n_kv_head in ((1, 1), (2, 1)):
            for proj in ("q", "k"):
                n_rows = 8 if proj == "q" else 8 * n_kv_head // n_head
                rows = [0, 4, 1, 5, 2, 6, 3, 7] if n_head == 1 else ([0, 2, 1, 3, 4, 6, 5, 7] if proj == "q" else [0, 2, 1, 3])
                weight = torch.arange(n_rows * 8, dtype=torch.float32).reshape(n_rows, 8).to(torch.float8_e4m3fn)
                for channel_scale in (False, True):
                    scales = (torch.arange(1, n_rows + 1, dtype=torch.float32).reshape(n_rows, 1) / 8) if channel_scale else torch.tensor([0.25])
                    for eager in (False, True):
                        with self.subTest(proj=proj, n_head=n_head, channel_scale=channel_scale, eager=eager):
                            name = f"model.layers.0.self_attn.{proj}_proj.weight"
                            hparams = {"hidden_size": 8, "num_attention_heads": n_head, "num_key_value_heads": n_kv_head}
                            result = self.convert(LlamaModel, {name: weight, name + "_scale": scales}, eager=eager,
                                                  check_lazy=not eager, hparams_override=hparams)
                            target = f"blk.0.attn_{proj}"
                            dtype, data = result[target + ".weight"]
                            self.assertEqual(dtype, gguf.GGMLQuantizationType.F8_E4M3)
                            actual_scale = result[target + ".scale"][1]
                            np.testing.assert_array_equal(actual_scale, scales.numpy().flatten()[rows] if channel_scale else [0.25])
                            np.testing.assert_array_equal(gguf.dequantize(data, dtype) * actual_scale.reshape(-1, 1), (weight.float() * scales).numpy()[rows])

    def test_channel_scales_follow_projection_split(self):
        name = "model.layers.0.linear_attn.in_proj_qkvz.weight"
        for n_v_heads, qkv_rows, z_rows in (
            (2, [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13], [6, 7, 14, 15]),
            (4, [0, 1, 12, 13, 2, 3, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19], [8, 9, 10, 11, 20, 21, 22, 23]),
        ):
            n_rows = len(qkv_rows) + len(z_rows)
            weight = torch.arange(n_rows * 4, dtype=torch.float32).reshape(n_rows, 4).to(torch.float8_e4m3fn)
            hparams = {"mtp_num_hidden_layers": 1, "linear_key_head_dim": 2, "linear_value_head_dim": 2,
                       "linear_num_key_heads": 2, "linear_num_value_heads": n_v_heads}
            for channel_scale in (False, True):
                scales = torch.arange(1, n_rows + 1, dtype=torch.float32).reshape(n_rows, 1) / 32 if channel_scale else torch.tensor([0.25])
                expected = (weight.float() * scales).numpy()
                for eager in (False, True):
                    with self.subTest(n_v_heads=n_v_heads, channel_scale=channel_scale, eager=eager):
                        result = self.convert(Qwen3NextModel, {name: weight, name + "_scale": scales}, eager=eager,
                                              check_lazy=not eager, hparams_override=hparams)
                        for target, rows in (("blk.0.attn_qkv", qkv_rows), ("blk.0.attn_gate", z_rows)):
                            dtype, data = result[target + ".weight"]
                            self.assertEqual(dtype, gguf.GGMLQuantizationType.F8_E4M3)
                            actual_scale = result[target + ".scale"][1]
                            np.testing.assert_array_equal(actual_scale, scales.numpy().flatten()[rows] if channel_scale else [0.25])
                            np.testing.assert_array_equal(gguf.dequantize(data, dtype) * actual_scale.reshape(-1, 1), expected[rows])

    def test_channel_scales_unknown_transform_fallback(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        weight = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.float8_e4m3fn)
        scales = torch.tensor([0.125, 0.25, 0.5, 1.]).reshape(4, 1)
        for eager in (False, True):
            with self.subTest(eager=eager):
                result = self.convert(ReorderedRowsModel, {name: weight, name + "_scale": scales}, eager=eager, check_lazy=not eager)
                dtype, data = result["blk.0.attn_q.weight"]
                self.assertEqual(dtype, gguf.GGMLQuantizationType.F16)
                np.testing.assert_array_equal(data, (weight.float() * scales).flip(0).numpy())
                self.assertNotIn("blk.0.attn_q.scale", result)

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

    def test_missing_weight_scale_rejected(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        for nvfp4 in (False, True):
            for eager in (False, True):
                for scale in (None, 1.0, 0.25):
                    with self.subTest(nvfp4=nvfp4, eager=eager, scale=scale):
                        if nvfp4:
                            tensors = {
                                name: torch.zeros((4, 32), dtype=torch.uint8),
                                name + "_scale": torch.ones((4, 4)).to(torch.float8_e4m3fn),
                            }
                            scale_name = name + "_scale_2"
                            config = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
                        else:
                            tensors = {name: torch.ones((4, 4)).to(torch.float8_e4m3fn)}
                            scale_name = name + "_scale"
                            config = {"quant_method": "modelopt"}
                        if scale is not None:
                            tensors[scale_name] = torch.tensor([scale])
                        kwargs = {"eager": eager, "hparams_override": {"quantization_config": config}}
                        if scale == 0.25:
                            result = self.convert(Qwen2MoeModel, tensors, **kwargs)
                            np.testing.assert_array_equal(result["blk.0.attn_q.scale"][1], [scale])
                        else:
                            error = "source weight scale is missing" if scale is None else "required scale tensor .* is missing"
                            with self.assertRaisesRegex(ValueError, error):
                                self.convert(Qwen2MoeModel, tensors, **kwargs)

    def test_output_type_override(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        weight = torch.tensor([[1., -1.], [2., -2.]]).to(torch.float8_e4m3fn)
        for eager in (False, True):
            for scaled in (False, True):
                with self.subTest(eager=eager, scaled=scaled):
                    tensors = {name: weight}
                    if scaled:
                        tensors[name + "_scale"] = torch.tensor([0.25])
                    else:
                        with self.assertRaisesRegex(ValueError, "source weight scale is missing"):
                            self.convert(ForcedF16Model, tensors, eager=eager)
                        continue
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
                        else:
                            with self.assertRaisesRegex(ValueError, "source weight scale is missing"):
                                self.convert(Qwen3NextModel, tensors, eager=eager, hparams_override={"mtp_num_hidden_layers": 1})
                            continue
                        result = self.convert(Qwen3NextModel, tensors, eager=eager, check_lazy=not eager, hparams_override={"mtp_num_hidden_layers": 1})
                        dtype, data = result[target]
                        self.assertEqual(dtype, gguf.GGMLQuantizationType.F32)
                        np.testing.assert_array_equal(data, weight.float().squeeze().numpy() * (0.25 if scaled else 1.))
                        self.assertNotIn(target.removesuffix("weight") + "scale", result)

    def test_fp8_norm_transform(self):
        for shape in ((4,), (2, 2)):
            with self.subTest(shape=shape):
                weight = torch.ones(shape, dtype=torch.float32).to(torch.float8_e4m3fn)
                result = self.convert(OffsetNormModel, {"model.layers.0.input_layernorm.weight": weight, "model.layers.0.input_layernorm.weight_scale": torch.tensor([1.0])}, check_lazy=True)
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
                        else:
                            with self.assertRaisesRegex(ValueError, "source weight scale is missing"):
                                self.convert(Qwen2MoeModel, tensors, ftype=ftype, eager=eager)
                            continue
                        if qtype in (gguf.GGMLQuantizationType.TQ1_0, gguf.GGMLQuantizationType.TQ2_0):
                            with self.assertRaisesRegex(ValueError, "required scale tensor 'blk.0.attn_q.scale' is missing"):
                                self.convert(Qwen2MoeModel, tensors, ftype=ftype, eager=eager)
                            continue
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
                    if other == "unscaled":
                        with self.assertRaisesRegex(ValueError, "source weight scale is missing"):
                            self.convert(Qwen2MoeModel, tensors, eager=eager)
                        continue
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
        with self.assertRaisesRegex(ValueError, "source weight scale is missing"):
            self.convert(Qwen2MoeModel, unscaled, check_lazy=True)

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

    def test_fp8_as_q8_requires_source_scale(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for eager in (False, True):
                with self.subTest(dtype=dtype, eager=eager):
                    tensors = {name: torch.ones((4, 32)).to(dtype)}
                    with self.assertRaisesRegex(ValueError, "source weight scale is missing"):
                        self.convert(Qwen2MoeModel, tensors, fp8_as_q8=True, eager=eager)

    def test_nvfp4_rejects_omitted_unit_scales(self):
        for experts in (False, True):
            with self.subTest(experts=experts):
                names = ["model.layers.0.self_attn.q_proj.weight"]
                if experts:
                    names = [f"model.layers.0.mlp.experts.{expert}.{proj}_proj.weight"
                             for expert in range(2) for proj in ("gate", "up", "down")]
                tensors = {}
                for name in names:
                    tensors[name] = torch.zeros((2, 32), dtype=torch.uint8)
                    tensors[name + "_scale"] = torch.ones((2, 4)).to(torch.float8_e4m3fn)
                with self.assertRaisesRegex(ValueError, "source weight scale is missing"):
                    self.convert(Qwen2MoeModel, tensors, eager=True, scan_nvfp4=True)
                for name in names:
                    tensors[name + "_scale_2"] = torch.tensor([1.0])
                with self.assertRaisesRegex(ValueError, "required scale tensor .* is missing"):
                    self.convert(Qwen2MoeModel, tensors, eager=True, scan_nvfp4=True)

    def test_nvfp4_requires_block_scale(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        tensors = {name: torch.zeros((2, 32), dtype=torch.uint8), name + "_scale_2": torch.tensor([1.0])}
        with self.assertRaisesRegex(ValueError, "source block scale .* is missing"):
            self.convert(Qwen2MoeModel, tensors, hparams_override={"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}})

    def test_fp8_rejects_expert_unit_scales(self):
        tensors = {}
        for expert in range(2):
            for proj in ("gate", "up", "down"):
                name = f"model.layers.0.mlp.experts.{expert}.{proj}_proj.weight"
                tensors[name] = torch.ones((4, 4)).to(torch.float8_e4m3fn)
                tensors[name + "_scale"] = torch.tensor([1.0])
        for eager in (False, True):
            with self.subTest(eager=eager):
                with self.assertRaisesRegex(ValueError, "required scale tensor .* is missing"):
                    self.convert(Qwen2MoeModel, tensors, eager=eager, check_lazy=not eager)


if __name__ == "__main__":
    unittest.main()
