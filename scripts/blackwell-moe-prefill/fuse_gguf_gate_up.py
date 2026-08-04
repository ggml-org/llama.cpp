#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import gguf
import numpy as np


@dataclass(frozen=True)
class TensorCopy:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    tensor_type: gguf.GGMLQuantizationType
    segments: tuple[tuple[int, int], ...]

    @property
    def nbytes(self) -> int:
        return sum(length for _, length in self.segments)

    def tofile(self, output: BinaryIO) -> None:
        source_fd = self.source_fd
        output_fd = output.fileno()
        for offset, length in self.segments:
            copied = 0
            while copied < length:
                count = os.copy_file_range(
                    source_fd,
                    output_fd,
                    min(length - copied, 1 << 30),
                    offset_src=offset + copied,
                )
                if count == 0:
                    raise OSError(f"Short copy at source offset {offset + copied}")
                copied += count

    source_fd: int = -1


def field_contents(reader: gguf.GGUFReader, name: str):
    field = reader.get_field(name)
    return field.contents() if field else None


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)


def tensor_copy(tensor: gguf.ReaderTensor, source_fd: int) -> TensorCopy:
    return TensorCopy(
        name=tensor.name,
        shape=tuple(tensor.data.shape),
        dtype=tensor.data.dtype,
        tensor_type=tensor.tensor_type,
        segments=((tensor.data_offset, tensor.n_bytes),),
        source_fd=source_fd,
    )


def fused_tensor_copy(
    gate: gguf.ReaderTensor,
    up: gguf.ReaderTensor,
    source_fd: int,
) -> TensorCopy:
    if gate.tensor_type != up.tensor_type or gate.data.dtype != up.data.dtype:
        raise ValueError(f"Mismatched tensor types for {gate.name} and {up.name}")
    if gate.data.ndim not in (2, 3) or gate.data.ndim != up.data.ndim:
        raise ValueError(f"Unexpected tensor ranks for {gate.name} and {up.name}")
    if gate.data.shape[0] != up.data.shape[0] or gate.data.shape[2:] != up.data.shape[2:]:
        raise ValueError(f"Mismatched tensor shapes for {gate.name} and {up.name}")

    gate_stride = gate.n_bytes // gate.data.shape[0]
    up_stride = up.n_bytes // up.data.shape[0]
    if gate_stride * gate.data.shape[0] != gate.n_bytes or up_stride * up.data.shape[0] != up.n_bytes:
        raise ValueError(f"Non-integral expert stride for {gate.name} and {up.name}")

    segments = []
    for expert in range(gate.data.shape[0]):
        segments.append((gate.data_offset + expert * gate_stride, gate_stride))
        segments.append((up.data_offset + expert * up_stride, up_stride))

    shape = list(gate.data.shape)
    shape[1] += up.data.shape[1]
    return TensorCopy(
        name=gate.name.replace(".ffn_gate_exps.", ".ffn_gate_up_exps."),
        shape=tuple(shape),
        dtype=gate.data.dtype,
        tensor_type=gate.tensor_type,
        segments=tuple(segments),
        source_fd=source_fd,
    )


def build_tensor_copies(reader: gguf.GGUFReader, source_fd: int) -> list[TensorCopy]:
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    copies = []
    fused_weights = 0
    fused_biases = 0

    for tensor in reader.tensors:
        if ".ffn_up_exps." in tensor.name:
            continue
        if ".ffn_gate_exps." not in tensor.name:
            copies.append(tensor_copy(tensor, source_fd))
            continue

        up_name = tensor.name.replace(".ffn_gate_exps.", ".ffn_up_exps.")
        up = tensors.get(up_name)
        if up is None:
            raise ValueError(f"Missing paired tensor {up_name}")
        copies.append(fused_tensor_copy(tensor, up, source_fd))
        if tensor.name.endswith(".weight"):
            fused_weights += 1
        elif tensor.name.endswith(".bias"):
            fused_biases += 1
        else:
            raise ValueError(f"Unexpected expert tensor name {tensor.name}")

    if fused_weights != 36 or fused_biases != 36:
        raise ValueError(f"Expected 36 fused weights and biases, got {fused_weights} and {fused_biases}")
    if len(copies) != 615:
        raise ValueError(f"Expected 615 output tensors, got {len(copies)}")
    return copies


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse GPT-OSS gate/up expert tensors in an existing GGUF")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)

    reader = gguf.GGUFReader(args.input, "r")
    architecture = field_contents(reader, gguf.Keys.General.ARCHITECTURE)
    if architecture != "gpt-oss":
        raise ValueError(f"Expected gpt-oss architecture, got {architecture!r}")

    with args.input.open("rb") as source:
        copies = build_tensor_copies(reader, source.fileno())
        writer = gguf.GGUFWriter(args.output, arch=architecture, endianess=reader.endianess)
        alignment = field_contents(reader, gguf.Keys.General.ALIGNMENT)
        if alignment is not None:
            writer.data_alignment = alignment
        copy_metadata(reader, writer)

        for tensor in copies:
            writer.add_tensor_info(
                tensor.name,
                tensor.shape,
                tensor.dtype,
                tensor.nbytes,
                raw_dtype=tensor.tensor_type,
            )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        total = sum(tensor.nbytes for tensor in copies)
        written = 0
        for tensor in copies:
            writer.write_tensor_data(tensor, tensor_endianess=reader.endianess)
            written += tensor.nbytes
            print(f"{written}/{total} {tensor.name}", flush=True)
        writer.close()


if __name__ == "__main__":
    main()
