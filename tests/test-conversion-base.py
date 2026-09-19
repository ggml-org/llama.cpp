from typing import Any

import torch

from conversion.base import ModelBase, _is_compressed_tensors_nvfp4


MIXED_CONFIG: dict[str, Any] = {
    "quant_method": "compressed-tensors",
    "format": "mixed-precision",
    "config_groups": {
        "fp8": {
            "format": "float-quantized",
            "weights": {
                "type": "float",
                "num_bits": 8,
                "strategy": "channel",
                "group_size": None,
            },
        },
        "nvfp4": {
            "format": "nvfp4-pack-quantized",
            "weights": {
                "type": "float",
                "num_bits": 4,
                "strategy": "tensor_group",
                "group_size": 16,
            },
        },
    },
}


def test_detects_nvfp4_group_in_mixed_precision_config():
    assert _is_compressed_tensors_nvfp4(
        MIXED_CONFIG["quant_method"],
        MIXED_CONFIG["format"],
        MIXED_CONFIG["config_groups"],
    )


def test_dequantizes_fp8_tensors_remaining_after_nvfp4_repack():
    model = object.__new__(ModelBase)
    model.hparams = {"quantization_config": MIXED_CONFIG}
    model._is_nvfp4 = True
    model._fp8_as_q8 = True
    model._fp8_dequantized = set()
    model.model_tensors = {
        "model.layers.0.self_attn.q_proj.weight": lambda: torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn
        ),
        "model.layers.0.self_attn.q_proj.weight_scale": lambda: torch.tensor([[0.5], [0.25]]),
    }

    model.dequant_model()

    weight_name = "model.layers.0.self_attn.q_proj.weight"
    assert set(model.model_tensors) == {weight_name}
    weight = model.model_tensors[weight_name]()
    assert weight.dtype == torch.float32
    torch.testing.assert_close(weight, torch.tensor([[0.5, 1.0], [0.75, 1.0]]))
    assert model._fp8_dequantized == {weight_name}


def test_nvfp4_repacking_skips_fp8_with_2d_channel_scale():
    model = object.__new__(ModelBase)
    model.hparams = {}
    model.model_tensors = {
        "lm_head.weight": lambda: torch.ones((2, 2), dtype=torch.float8_e4m3fn),
        "lm_head.weight_scale": lambda: torch.ones((2, 1)),
    }

    model._generate_nvfp4_tensors()

    assert set(model.model_tensors) == {"lm_head.weight", "lm_head.weight_scale"}
