#!/usr/bin/env python3
"""Create an experimental dynamic INT8 variant of M5-TTS depth_frame.onnx."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize(source, output):
    model = onnx.load(source)
    aliases = {tensor.name: tensor.name for tensor in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Identity" and node.input[0] in aliases:
            aliases[node.output[0]] = aliases[node.input[0]]
        elif node.op_type == "MatMul" and node.input[1] in aliases:
            # The unrolled decoder gives the same weight different Identity aliases.
            node.input[1] = aliases[node.input[1]]
    with TemporaryDirectory() as temporary:
        canonical = Path(temporary) / "depth.onnx"
        onnx.save(model, canonical)
        quantize_dynamic(
            str(canonical), str(output),
            op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    quantize(args.source, args.output)
