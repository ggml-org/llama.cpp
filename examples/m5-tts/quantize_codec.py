#!/usr/bin/env python3
"""Create experimental dynamic INT8 variants of the M5-TTS codec decoder.

Full-graph quantization makes ORT abort while creating the session on the
device, so "conv" and "top" modes quantize fewer nodes.
"""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

# Highest-cost Conv nodes from the ORT codec profile (cumulative kernel time).
TOP_CONV = (
    "/conv_14/Conv", "/conv_23/Conv", "/conv_18/Conv", "/conv_21/Conv",
    "/conv_16/Conv", "/conv_25/Conv", "/conv_7/Conv", "/conv_9/Conv",
    "/conv_11/Conv", "/conv_28/Conv", "/conv_30/Conv", "/conv_32/Conv",
)


def quantize(source: Path, output: Path, mode: str) -> None:
    model = onnx.load(source)
    # The exporter aliases the same weight through Identity nodes.
    aliases = {t.name: t.name for t in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Identity" and node.input[0] in aliases:
            aliases[node.output[0]] = aliases[node.input[0]]
        elif (node.op_type in ("Conv", "ConvTranspose", "MatMul", "Gemm")
              and len(node.input) > 1 and node.input[1] in aliases):
            node.input[1] = aliases[node.input[1]]

    convs = [n.name for n in model.graph.node if n.op_type == "Conv"]
    kwargs = {"weight_type": QuantType.QInt8}
    if mode == "full":
        kwargs["op_types_to_quantize"] = ["Conv", "MatMul"]
    elif mode == "conv":
        kwargs["op_types_to_quantize"] = ["Conv"]
        kwargs["nodes_to_quantize"] = convs
    else:
        kwargs["op_types_to_quantize"] = ["Conv"]
        kwargs["nodes_to_quantize"] = [n for n in TOP_CONV if n in convs]

    with TemporaryDirectory() as tmp:
        canonical = Path(tmp) / "codec.onnx"
        onnx.save(model, canonical)
        quantize_dynamic(canonical, output, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("full", "conv", "top"), default="full")
    args = parser.parse_args()
    quantize(args.source, args.output, args.mode)
