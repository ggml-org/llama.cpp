"""Test NVFP4 PLE fallback with small checkpoints; no GPU or model downloads required."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from conversion.base import LazyTorchTensor, gguf
from conversion.qwen4exp import Qwen4ExpTextModel


class TestNVFP4PLEConversion(unittest.TestCase):
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    output_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD] + ".weight"

    def fixture(self, rows=(130, 3), scale2=3.324236240587197e-05):
        tensors = {self.prefix + "weight_scale_2": torch.tensor([scale2])}
        expected = []
        levels = [0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.]
        for index, count in enumerate(rows):
            codes = (np.arange(count * 160).reshape(count, 160) + index * 3) % 16
            scales = np.array([[0.25, 0.5, 1., 2.][(r + c + index) % 4] for r in range(count) for c in range(10)], dtype=np.float32).reshape(count, 10)
            packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
            name = self.prefix + f"shard_{index}.weight"
            tensors[name] = torch.from_numpy(packed)
            tensors[name + "_scale"] = torch.from_numpy(scales).to(torch.float8_e4m3fn)
            reference = np.array([[levels[int(codes[r, c])] * float(scales[r, c // 16]) for c in range(160)] for r in range(count)], dtype=np.float32)
            with np.errstate(invalid="ignore"):
                reference *= np.float32(scale2)
            expected.append(reference)
        # Registration order must not determine the final table's row order.
        return dict(reversed(list(tensors.items()))), np.concatenate(expected)

    def prepare(self, path, tensors, *, eager=False, ftype=gguf.LlamaFileType.GUESSED, parts=2):
        hp = {"model_type": "qwen4exp", "architectures": ["Qwen4ExpForCausalLM"],
              "num_hidden_layers": 2, "mtp_num_hidden_layers": 1, "split_ngram_parts": parts,
              "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}
        (path / "config.json").write_text(json.dumps(hp))
        save_file(tensors, str(path / "model.safetensors"))
        model = Qwen4ExpTextModel(path, ftype, path / "out.gguf", hparams=hp, eager=eager)
        with patch("conversion.qwen4exp._NVFP4_PLE_CHUNK_ROWS", 128):
            model.prepare_tensors()
        return model

    def test_f16_table_round_trip_and_both_scale_levels(self):
        for eager in (False, True):
            for ftype in (gguf.LlamaFileType.GUESSED, gguf.LlamaFileType.MOSTLY_Q8_0, gguf.LlamaFileType.MOSTLY_BF16):
                with self.subTest(eager=eager, ftype=ftype), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    tensors, expected = self.fixture()
                    model = self.prepare(path, tensors, eager=eager, ftype=ftype)
                    self.assertEqual(model._ple_row_dim, 160)
                    self.assertEqual(set(model.gguf_writer.tensors[0]), {self.output_name})
                    info = model.gguf_writer.tensors[0][self.output_name]
                    self.assertEqual(info.dtype, gguf.GGMLQuantizationType.F16)
                    self.assertIsInstance(info.tensor, gguf.LazyChunkedTensor)
                    self.assertEqual(info.tensor.shape, (133, 160))
                    self.assertEqual(info.tensor.dtype, np.float16)
                    self.assertEqual(len(info.tensor._chunks), 3)
                    self.assertFalse(any(name.endswith(("_scale", "_scale_2")) for name in model.model_tensors))
                    model.gguf_writer.add_embedding_length_per_layer_input(model._ple_row_dim)
                    model.gguf_writer.write_header_to_file(path / "out.gguf")
                    model.gguf_writer.write_kv_data_to_file()
                    model.gguf_writer.write_tensors_to_file()
                    model.gguf_writer.close()
                    reader = gguf.GGUFReader(path / "out.gguf")
                    tensor = reader.tensors[0]
                    self.assertEqual(tensor.tensor_type, gguf.GGMLQuantizationType.F16)
                    np.testing.assert_array_equal(tensor.shape, [160, 133])
                    np.testing.assert_array_equal(tensor.data.view(np.uint16), expected.astype(np.float16).view(np.uint16))
                    del tensor, reader, info, model

    def test_preparation_does_not_materialize_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            tensors, expected = self.fixture()
            with patch.object(LazyTorchTensor, "to_eager", wraps=LazyTorchTensor.to_eager) as materialize:
                model = self.prepare(Path(directory), tensors, ftype=gguf.LlamaFileType.MOSTLY_F16)
                materialize.assert_not_called()
            table = model.gguf_writer.tensors[0][self.output_name].tensor
            for name in ("first.bin", "second.bin"):
                with (Path(directory) / name).open("wb") as output:
                    table.tofile(output)
                actual = np.fromfile(Path(directory) / name, dtype=np.float16).reshape(expected.shape)
                np.testing.assert_array_equal(actual.view(np.uint16), expected.astype(np.float16).view(np.uint16))
            self.assertTrue(all(name.endswith(".weight") for name in model.model_tensors))
            del table, model, materialize

    def test_missing_scales_and_invalid_layouts(self):
        name = self.prefix + "shard_0.weight"
        scale2 = self.prefix + "weight_scale_2"
        tensors, _ = self.fixture(rows=(1,))
        for remove, extra, message in (
            (name + "_scale", {}, "requires block scales"),
            (scale2, {}, "requires block scales"),
            (None, {name + "_scale": torch.ones((1, 9), dtype=torch.float8_e4m3fn)}, "layout"),
            (None, {name + "_scale": torch.ones((1, 10))}, "layout"),
            (None, {scale2: torch.ones(2)}, "scalar"),
            (None, {scale2: torch.ones(1, dtype=torch.int32)}, "scalar"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                case = tensors | extra
                if remove:
                    del case[remove]
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare(Path(directory), case, parts=1)

    def test_invalid_scale_values_fail_when_streamed(self):
        for value in (float("nan"), float("inf"), -1.):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                tensors, _ = self.fixture(rows=(1,), scale2=value)
                model = self.prepare(Path(directory), tensors, parts=1)
                table = model.gguf_writer.tensors[0][self.output_name].tensor
                with self.assertRaisesRegex(ValueError, "invalid scales"):
                    table._chunks[0]()
                del table, model

    def test_missing_and_out_of_range_shards(self):
        tensors, _ = self.fixture(rows=(1,))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "expected 2"):
                self.prepare(Path(directory), tensors)
        tensors = {name.replace("shard_0", "shard_2"): value for name, value in tensors.items()}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "outside the expected range"):
                self.prepare(Path(directory), tensors, parts=1)

    def test_plain_ple_and_native_nvfp4_weights_still_convert(self):
        with tempfile.TemporaryDirectory() as directory:
            tensors, expected = self.fixture(rows=(2,), scale2=0.125)
            dense = "model.language_model.layers.0.self_attn.q_proj.weight"
            tensors[dense] = torch.zeros((2, 32), dtype=torch.uint8)
            tensors[dense + "_scale"] = torch.ones((2, 4), dtype=torch.float8_e4m3fn)
            tensors[dense + "_scale_2"] = torch.tensor([0.5])
            model = self.prepare(Path(directory), tensors, parts=1)
            self.assertEqual(model.gguf_writer.tensors[0]["blk.0.attn_q.weight"].dtype, gguf.GGMLQuantizationType.NVFP4)
            self.assertEqual(model.gguf_writer.tensors[0][self.output_name].dtype, gguf.GGMLQuantizationType.F16)
            del model
        with tempfile.TemporaryDirectory() as directory:
            name = self.prefix + "shard_0.weight"
            plain = torch.arange(320, dtype=torch.float32).reshape(2, 160) / 16
            model = self.prepare(Path(directory), {name: plain}, ftype=gguf.LlamaFileType.ALL_F32, parts=1)
            self.assertEqual(model.gguf_writer.tensors[0][self.output_name].dtype, gguf.GGMLQuantizationType.F32)
            self.assertEqual(model._nvfp4_ple_shards, {})
            del model


class TestMTPQuantizationMetadata(unittest.TestCase):
    canonical = "mtp.layers.0.mlp.experts"
    alias = "mtp.layers.48.mlp.experts"
    settings = {"quant_algo": "W4A16_NVFP4", "group_size": 16}

    def convert(self, layers, *, eager=False, sidecar=False, no_mtp=False, mtp_only=False, mtp_indices=(0,)):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            tensors = {}
            for index in mtp_indices:
                name = f"mtp.layers.{index}.mlp.experts.0.down_proj.weight"
                tensors[name] = (torch.arange(32 * 32).reshape(32, 32) % 256).to(torch.uint8)
                tensors[name + "_scale"] = torch.ones((32, 4), dtype=torch.float8_e4m3fn)
                tensors[name + "_scale_2"] = torch.tensor([0.25])
            save_file(tensors, str(path / "model.safetensors"))
            quantization = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": layers}
            hp = {"model_type": "qwen4exp", "architectures": ["Qwen4ExpForCausalLM"], "num_hidden_layers": 48,
                  "mtp_num_hidden_layers": len(mtp_indices), "num_experts": 1, "quantization_config": quantization}
            if sidecar:
                hp["quantization_config"] = {"quant_method": "modelopt"}
                (path / "hf_quant_config.json").write_text(json.dumps({"producer": {"name": "modelopt"}, "quantization": quantization}))
            config = json.dumps(hp)
            (path / "config.json").write_text(config)
            with patch.object(Qwen4ExpTextModel, "no_mtp", no_mtp), patch.object(Qwen4ExpTextModel, "mtp_only", mtp_only):
                model = Qwen4ExpTextModel(path, gguf.LlamaFileType.MOSTLY_F16, path / "out.gguf", hparams=hp, eager=eager)
                model.prepare_tensors()
            self.assertEqual((path / "config.json").read_text(), config)
            self.assertEqual(model.hparams["quantization_config"]["quantized_layers"], layers)
            result = {}
            for name, info in model.gguf_writer.tensors[0].items():
                assert info.tensor is not None
                result[name] = (info.dtype, info.tensor.copy())
            return result

    def test_redundant_mtp_alias_preserves_native_bytes(self):
        canonical = {self.canonical: self.settings}
        expected = self.convert(canonical)
        self.assertEqual(set(expected), {"blk.48.ffn_down_exps.weight", "blk.48.ffn_down_exps.scale"})
        self.assertEqual(expected["blk.48.ffn_down_exps.weight"][0], gguf.GGMLQuantizationType.NVFP4)
        for eager in (False, True):
            for sidecar in (False, True):
                for no_mtp, mtp_only in ((False, False), (True, False), (False, True)):
                    with self.subTest(eager=eager, sidecar=sidecar, no_mtp=no_mtp, mtp_only=mtp_only):
                        with self.assertLogs("hf-to-gguf", level="WARNING") as logs:
                            actual = self.convert(canonical | {self.alias: self.settings}, eager=eager, sidecar=sidecar, no_mtp=no_mtp, mtp_only=mtp_only)
                        self.assertTrue(any(self.alias in message and self.canonical in message for message in logs.output))
                        reference = {} if no_mtp else expected
                        self.assertEqual(actual.keys(), reference.keys())
                        for name, (dtype, data) in reference.items():
                            self.assertEqual(actual[name][0], dtype)
                            np.testing.assert_array_equal(actual[name][1], data)

    def test_conflicting_alias_metadata_is_rejected(self):
        for settings in (self.settings | {"group_size": 32}, self.settings | {"quant_algo": "NVFP4"}, self.settings | {"weight_scale_encoding": "csf"}):
            with self.subTest(settings=settings), self.assertRaisesRegex(AssertionError, "quantized_layers entries not in the model tensors"):
                self.convert({self.canonical: self.settings, self.alias: settings})

    def test_alias_requires_matching_canonical_metadata_and_tensors(self):
        for layers, indices in (({self.alias: self.settings}, (0,)),
                                ({self.canonical: self.settings, self.alias: self.settings}, (1,))):
            with self.subTest(indices=indices, layers=layers), self.assertRaisesRegex(AssertionError, "quantized_layers entries not in the model tensors"):
                self.convert(layers, mtp_indices=indices)

    def test_unrelated_and_out_of_range_metadata_is_rejected(self):
        for name in ("mtp.layers.49.mlp.experts", "mtp.layers.47.mlp.experts", "mtp.layers.bad.mlp.experts",
                     "model.layers.48.mlp.experts", "mtp.layers.48.mlp.typo"):
            with self.subTest(name=name), self.assertRaisesRegex(AssertionError, "quantized_layers entries not in the model tensors"):
                self.convert({self.canonical: self.settings, name: self.settings})

    def test_multiple_mtp_layers_use_their_own_metadata(self):
        layers = {f"mtp.layers.{index}.mlp.experts": self.settings for index in (0, 1)}
        expected = self.convert(layers, mtp_indices=(0, 1))
        aliases = {f"mtp.layers.{index}.mlp.experts": self.settings for index in (48, 49)}
        with self.assertLogs("hf-to-gguf", level="WARNING"):
            actual = self.convert(layers | aliases, mtp_indices=(0, 1))
        self.assertEqual(actual.keys(), expected.keys())
        for name in expected:
            np.testing.assert_array_equal(actual[name][1], expected[name][1])

    def test_offset_module_with_source_tensors_is_not_treated_as_redundant(self):
        layers = {self.canonical: self.settings, self.alias: self.settings,
                  self.alias + ".0.down_proj": self.settings}
        with self.assertRaisesRegex(AssertionError, "quantized_layers entries not in the model tensors"):
            self.convert(layers, mtp_indices=(0, 48))


if __name__ == "__main__":
    unittest.main()
