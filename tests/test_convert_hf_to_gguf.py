from __future__ import annotations

import unittest
from typing import Any, cast

import gguf
import numpy as np
import torch

from conversion.base import ModelBase


def mixed_config_groups() -> dict[str, dict[str, Any]]:
    return {
        "group_0": {
            "format": "float-quantized",
            "weights": {
                "type": "float",
                "num_bits": 8,
                "strategy": "channel",
                "group_size": None,
                "block_structure": None,
            },
        },
        "group_1": {
            "format": "nvfp4-pack-quantized",
            "weights": {
                "type": "float",
                "num_bits": 4,
                "strategy": "tensor_group",
                "group_size": 16,
                "block_structure": None,
            },
        },
    }


def make_model(tensors: dict[str, torch.Tensor], *, fp8_as_q8: bool = False) -> ModelBase:
    model = ModelBase.__new__(ModelBase)
    model.hparams = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "config_groups": mixed_config_groups(),
        },
    }
    model.model_tensors = {name: (lambda tensor=tensor: tensor) for name, tensor in tensors.items()}
    model._is_nvfp4 = True
    model._fp8_as_q8 = fp8_as_q8
    model._fp8_dequantized = set()
    return model


class TensorWriter:
    def __init__(self) -> None:
        self.tensors: dict[str, tuple[np.ndarray, gguf.GGMLQuantizationType | None]] = {}

    def add_tensor(self, name: str, data: np.ndarray, raw_dtype: gguf.GGMLQuantizationType | None = None) -> None:
        self.tensors[name] = data, raw_dtype


class TestMixedCompressedTensors(unittest.TestCase):
    def test_dequantizes_fp8(self) -> None:
        weight_name = "model.layers.32.mlp.experts.0.gate_proj.weight"
        scale_name = weight_name + "_scale"
        weight = torch.tensor([[1.0, -2.0, 3.0, -4.0], [2.0, 4.0, -1.0, -3.0]], dtype=torch.float8_e4m3fn)
        scale = torch.tensor([[0.5], [0.25]], dtype=torch.bfloat16)
        model = make_model({weight_name: weight, scale_name: scale}, fp8_as_q8=True)

        model.dequant_model()

        self.assertNotIn(scale_name, model.model_tensors)
        self.assertIn(weight_name, model._fp8_dequantized)
        self.assertEqual(model.tensor_force_quant(weight_name, weight_name, None, 2), gguf.GGMLQuantizationType.Q8_0)
        torch.testing.assert_close(model.model_tensors[weight_name](), weight.float() * scale.float())

    def test_nvfp4_repacking_uses_packed_weight_provenance(self) -> None:
        nvfp4_name = "model.layers.31.mlp.shared_expert.gate_proj.weight"
        fp8_name = "model.layers.32.mlp.shared_expert.gate_proj.weight"
        model = make_model({
            nvfp4_name: torch.arange(64, dtype=torch.uint8).reshape(2, 32),
            nvfp4_name + "_scale": torch.ones((2, 4), dtype=torch.float8_e4m3fn),
            fp8_name: torch.ones((2, 32), dtype=torch.float8_e4m3fn),
            fp8_name + "_scale": torch.ones((2, 1), dtype=torch.bfloat16),
        })
        writer = TensorWriter()
        cast(Any, model).gguf_writer = writer
        model.hparams["num_local_experts"] = 0
        cast(Any, model).map_tensor_name = lambda name: name

        model._generate_nvfp4_tensors({nvfp4_name})

        self.assertEqual(writer.tensors[nvfp4_name][1], gguf.GGMLQuantizationType.NVFP4)
        self.assertNotIn(nvfp4_name, model.model_tensors)
        self.assertIn(fp8_name, model.model_tensors)
        self.assertIn(fp8_name + "_scale", model.model_tensors)

    def test_nvfp4_repacking_rejects_malformed_packed_weight(self) -> None:
        weight_name = "model.layers.31.mlp.shared_expert.gate_proj.weight"
        model = make_model({
            weight_name: torch.ones((2, 32), dtype=torch.float8_e4m3fn),
            weight_name + "_scale": torch.ones((2, 4), dtype=torch.float8_e4m3fn),
        })

        with self.assertRaisesRegex(ValueError, "Invalid packed NVFP4 tensors"):
            model._generate_nvfp4_tensors({weight_name})

    def test_rejects_unknown_group(self) -> None:
        model = make_model({})
        model._is_nvfp4 = False
        groups = model.hparams["quantization_config"]["config_groups"]
        groups["group_2"] = {"format": "int-quantized", "weights": {"num_bits": 4}}

        with self.assertRaisesRegex(NotImplementedError, "multiple config groups"):
            model.dequant_model()


if __name__ == "__main__":
    unittest.main()
