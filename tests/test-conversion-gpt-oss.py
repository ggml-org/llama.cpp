from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from conversion.base import LazyTorchTensor, gguf
from conversion.gpt_oss import GptOssModel


@dataclass
class TensorRecord:
    name: str
    tensor: object
    raw_dtype: gguf.GGMLQuantizationType


class TensorWriter:
    def __init__(self) -> None:
        self.records: list[TensorRecord] = []

    def add_tensor(self, name, tensor, raw_dtype) -> None:
        self.records.append(TensorRecord(name, tensor, raw_dtype))


def make_lazy_tensor(data: torch.Tensor, loads: list[str], name: str):
    return LazyTorchTensor(
        meta=LazyTorchTensor.meta_with_dtype_and_shape(data.dtype, tuple(data.shape)),
        func=lambda: loads.append(name) or data,
    )


def test_repack_mxfp4_parts_stays_lazy_and_keeps_gate_up_order() -> None:
    model = object.__new__(GptOssModel)
    writer = TensorWriter()
    model.gguf_writer = writer

    gate_data = torch.arange(2 * 3 * 4 * 16).to(torch.uint8).reshape(2, 3, 4, 16)
    up_data = torch.flip(gate_data, dims=(1,))
    gate_scales_data = torch.arange(2 * 3 * 4, dtype=torch.uint8).reshape(2, 3, 4)
    up_scales_data = gate_scales_data + 32
    loads: list[str] = []

    gate = make_lazy_tensor(gate_data, loads, "gate")
    up = make_lazy_tensor(up_data, loads, "up")
    gate_scales = make_lazy_tensor(gate_scales_data, loads, "gate_scales")
    up_scales = make_lazy_tensor(up_scales_data, loads, "up_scales")

    model.repack_mxfp4_parts("blk.0.ffn_gate_up_exps.weight", ((gate, gate_scales), (up, up_scales)))

    assert loads == []
    assert len(writer.records) == 1
    record = writer.records[0]
    assert record.name == "blk.0.ffn_gate_up_exps.weight"
    assert record.raw_dtype == gguf.GGMLQuantizationType.MXFP4
    assert isinstance(record.tensor, gguf.LazyNumpyTensor)
    assert record.tensor.shape == (2, 6, 68)

    actual = gguf.LazyNumpyTensor.to_eager(record.tensor)
    expected_gate = torch.concat(
        (gate_scales_data.unsqueeze(-1), model.transform_nibble_layout(gate_data)), dim=-1)
    expected_up = torch.concat(
        (up_scales_data.unsqueeze(-1), model.transform_nibble_layout(up_data)), dim=-1)
    expected = torch.concat((expected_gate, expected_up), dim=1).reshape(2, 6, 68).numpy()

    assert sorted(loads) == ["gate", "gate_scales", "up", "up_scales"]
    np.testing.assert_array_equal(actual, expected)


def test_generate_extra_tensors_pairs_mxfp4_tensors_by_name() -> None:
    model = object.__new__(GptOssModel)
    writer = TensorWriter()
    model.gguf_writer = writer
    model.fuse_gate_up_exps = True
    model.map_tensor_name = lambda name: name

    down_blocks = torch.zeros((2, 3, 4, 16), dtype=torch.uint8)
    down_scales = torch.zeros((2, 3, 4), dtype=torch.uint8)
    gate_up_blocks = torch.zeros((2, 6, 4, 16), dtype=torch.uint8)
    gate_up_scales = torch.zeros((2, 6, 4), dtype=torch.uint8)
    tensors = (
        ("model.layers.0.mlp.experts.down_proj_blocks", down_blocks),
        ("model.layers.0.mlp.experts.gate_up_proj_blocks", gate_up_blocks),
        ("model.layers.0.mlp.experts.down_proj_scales", down_scales),
        ("model.layers.0.mlp.experts.gate_up_proj_scales", gate_up_scales),
    )
    model.get_tensors = lambda: iter(tensors)

    assert model.generate_extra_tensors() == []
    assert [record.name for record in writer.records] == [
        "model.layers.0.mlp.experts.down_proj.weight",
        "model.layers.0.mlp.experts.gate_up_proj.weight",
    ]


def test_modify_tensors_fuses_bias_gate_first() -> None:
    model = object.__new__(GptOssModel)
    model.fuse_gate_up_exps = True
    model._gate_exp_buffer = {}
    model._up_exp_buffer = {}
    model.map_tensor_name = lambda name: "blk.0.ffn_gate_up_exps.bias"
    bias = torch.tensor(((1.0, 11.0, 2.0, 12.0), (3.0, 13.0, 4.0, 14.0)))

    tensors = list(model.modify_tensors(bias, "model.layers.0.mlp.experts.gate_up_proj_bias", 0))

    assert len(tensors) == 1
    assert tensors[0][0] == "blk.0.ffn_gate_up_exps.bias"
    torch.testing.assert_close(tensors[0][1], torch.tensor(((1.0, 2.0, 11.0, 12.0), (3.0, 4.0, 13.0, 14.0))))
