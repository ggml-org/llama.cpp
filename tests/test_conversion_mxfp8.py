"""Test ModelOpt MXFP8 group scales without GPU or model downloads."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import cast
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

from conversion.base import ModelBase, gguf
from conversion.qwen import Qwen2MoeModel
from conversion.qwen4exp import Qwen4ExpTextModel
import test_conversion_fp8 as fp8_tests
import test_conversion_nvfp4_ple as ple_tests


class TestModelOptMXFP8Conversion(unittest.TestCase):
    name = "model.layers.0.self_attn.q_proj.weight"
    source = "model.language_model.layers.0.self_attn.q_proj.weight"

    def fixture(self, rows=3, cols=96):
        weight = ((torch.arange(rows * cols).reshape(rows, cols) % 9 - 4).float() / 2).to(torch.float8_e4m3fn)
        scale = (torch.arange(rows * ((cols + 31) // 32)).reshape(rows, -1) % 5 + 125).to(torch.uint8)
        exponents = np.repeat(scale.numpy().astype(np.int32) - 127, 32, axis=-1)[:, :cols]
        expected = np.ldexp(weight.float().numpy(), exponents)
        return weight, scale, expected

    def config(self, method="modelopt"):
        return {"quant_method": method, "quant_algo": "MIXED_PRECISION", "quantized_layers": {
            "model.language_model.layers.0.self_attn": {"quant_algo": "MXFP8", "group_size": 32},
        }}

    def dequant(self, weight, scale, config=None, name=None):
        name = name or self.name
        state = SimpleNamespace(_is_nvfp4=True, _fp8_as_q8=False, _fp8_dequantized=set(),
                                _source_tensor_names={name.replace("model.", "model.language_model.", 1): name},
                                hparams={"quantization_config": config or self.config()},
                                model_tensors={name: lambda: weight, name + "_scale": lambda: scale})
        ModelBase.dequant_model(cast(ModelBase, state))
        self.assertEqual(set(state.model_tensors), {name})
        return state.model_tensors[name]()

    def test_checkpoint_shapes_and_shared_expert_metadata(self):
        weight, scale, expected = self.fixture(640, 2560)
        name = "model.layers.0.mlp.shared_expert.gate_proj.weight"
        config = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": {
            name.replace("model.", "model.language_model.", 1).removesuffix(".weight"): {"quant_algo": "MXFP8", "group_size": 32},
        }}
        np.testing.assert_array_equal(self.dequant(weight, scale, config, name).numpy(), expected)

    def test_lazy_eager_conversion_and_single_group(self):
        for rows, cols in ((1, 32), (3, 96)):
            weight, scale, expected = self.fixture(rows, cols)
            for eager in (False, True):
                for method in ("modelopt", "modelopt_mixed"):
                    for fp8_as_q8 in (False, True):
                        for source in (self.name, self.source):
                            with self.subTest(shape=(rows, cols), eager=eager, method=method, fp8_as_q8=fp8_as_q8, source=source):
                                config = self.config(method)
                                if source == self.name:
                                    config["quantized_layers"] = {key.replace("language_model.", ""): value for key, value in config["quantized_layers"].items()}
                                result = fp8_tests.TestFP8Conversion().convert(
                                    Qwen2MoeModel, {source: weight, source + "_scale": scale},
                                    eager=eager, check_lazy=not eager and source == self.name, fp8_as_q8=fp8_as_q8,
                                    hparams_override={"quantization_config": config})
                                self.assertEqual(set(result), {"blk.0.attn_q.weight"})
                                dtype, actual = result["blk.0.attn_q.weight"]
                                qtype = gguf.GGMLQuantizationType.Q8_0 if fp8_as_q8 else gguf.GGMLQuantizationType.F16
                                self.assertEqual(dtype, qtype)
                                np.testing.assert_array_equal(actual, gguf.quantize(expected, qtype))

    def test_global_metadata_and_partial_final_group(self):
        weight, scale, expected = self.fixture(2, 65)
        config = {"quant_method": "modelopt", "quant_algo": "MXFP8", "group_size": 32}
        np.testing.assert_array_equal(self.dequant(weight, scale, config).numpy(), expected)
        # MXFP8's standard group size applies when it is omitted from metadata.
        del config["group_size"]
        np.testing.assert_array_equal(self.dequant(weight, scale, config).numpy(), expected)

    def test_all_e8m0_bytes_and_typed_scales(self):
        weight = torch.ones((1, 256 * 32), dtype=torch.float8_e4m3fn)
        scale = torch.arange(256).to(torch.uint8).reshape(1, 256)
        config = {"quant_method": "modelopt", "quant_algo": "MXFP8"}
        expected_scales = np.empty(256, dtype=np.float32)
        expected_scales[:255] = np.ldexp(np.ones(255, dtype=np.float32), np.arange(255) - 127)
        expected_scales[255] = np.nan
        expected = np.repeat(expected_scales, 32).reshape(1, -1)
        np.testing.assert_array_equal(self.dequant(weight, scale, config).numpy(), expected)
        if hasattr(torch, "float8_e8m0fnu"):
            np.testing.assert_array_equal(self.dequant(weight, scale.view(torch.float8_e8m0fnu), config).numpy(), expected)

    def test_invalid_groups_and_shapes(self):
        weight, scale, _ = self.fixture()
        for group in (0, -1, None, True, 32.0, 16, 64, 1 << 40):
            with self.subTest(group=group):
                with self.assertRaisesRegex(ValueError, "requires group_size=32"):
                    self.dequant(weight, scale, {"quant_method": "modelopt", "quant_algo": "MXFP8", "group_size": group})
        for bad_weight, bad_scale, message in (
            (weight, scale[:, :-1], "expected"),
            (weight, scale[:1], "expected"),
            (weight, scale.float(), "requires FP8 weights and E8M0"),
            (weight.float(), scale, "requires FP8 weights and E8M0"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.dequant(bad_weight, bad_scale)

    def test_sidecar_quantization_metadata(self):
        weight, scale, expected = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            sidecar = {"producer": {"name": "modelopt"}, "quantization": self.config()}
            (path / "hf_quant_config.json").write_text(json.dumps(sidecar))
            result = fp8_tests.TestFP8Conversion()._convert(
                path, Qwen2MoeModel, {self.source: weight, self.source + "_scale": scale},
                eager=False, check_lazy=False, scan_nvfp4=False, ftype=gguf.LlamaFileType.GUESSED,
                hparams_override={"quantization_config": {"quant_method": "modelopt"}})
            dtype, actual = result["blk.0.attn_q.weight"]
            self.assertEqual(dtype, gguf.GGMLQuantizationType.F16)
            np.testing.assert_array_equal(actual, expected.astype(np.float16))

    def test_mxfp8_csf_and_ple_convert_together(self):
        tensors, ple_expected = ple_tests.TestNVFP4PLEConversion().fixture(rows=(2,))
        fixed, exceptions, scale_bytes = fp8_tests.TestNVFP4CSFConversion().fixture()
        expert = "model.language_model.layers.0.mlp.experts.0.down_proj.weight"
        packed = (torch.arange(32 * 64).reshape(32, 64) % 256).to(torch.uint8)
        tensors.update({expert: packed, expert + "_scale.nvfp4_csf_fixed": fixed,
                        expert + "_scale.nvfp4_csf_exceptions": exceptions, expert + "_scale_2": torch.tensor([0.25])})
        dense = "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"
        weight, scale, dense_expected = self.fixture()
        tensors.update({dense: weight, dense + "_scale": scale})
        hp = {"model_type": "qwen4exp", "architectures": ["Qwen4ExpForCausalLM"], "num_hidden_layers": 2,
              "mtp_num_hidden_layers": 1, "split_ngram_parts": 1, "num_experts": 1,
              "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": {
                  "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf"},
                  dense.removesuffix(".weight"): {"quant_algo": "MXFP8", "group_size": 32},
                  "model.language_model.layers.1.ple.ple_embedding.ngram_embedding": {"quant_algo": "NVFP4", "group_size": 16},
              }}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_file(tensors, str(path / "model.safetensors"))
            model = Qwen4ExpTextModel(path, gguf.LlamaFileType.MOSTLY_F16, path / "out.gguf", hparams=hp)
            model.prepare_tensors()
            writer = model.gguf_writer
            writer.write_header_to_file(path / "out.gguf")
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            reader = gguf.GGUFReader(path / "out.gguf")
            outputs = {tensor.name: tensor for tensor in reader.tensors}
            ple_name = ple_tests.TestNVFP4PLEConversion.output_name
            self.assertEqual(outputs[ple_name].tensor_type, gguf.GGMLQuantizationType.F16)
            np.testing.assert_array_equal(outputs[ple_name].data, ple_expected.astype(np.float16))
            np.testing.assert_array_equal(outputs["blk.0.ffn_gate_shexp.weight"].data, dense_expected.astype(np.float16))
            expert_output = outputs["blk.0.ffn_down_exps.weight"]
            self.assertEqual(expert_output.tensor_type, gguf.GGMLQuantizationType.NVFP4)
            expected_raw, _ = ModelBase._nvfp4_pack(packed, torch.from_numpy(scale_bytes).view(torch.float8_e4m3fn))
            np.testing.assert_array_equal(expert_output.data.reshape(expected_raw.shape), expected_raw)
            np.testing.assert_array_equal(outputs["blk.0.ffn_down_exps.scale"].data, [0.25])
            del expert_output, outputs, reader, writer, model


if __name__ == "__main__":
    unittest.main()
