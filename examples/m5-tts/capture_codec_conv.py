#!/usr/bin/env python3
"""Capture real /conv_14/Conv inputs and outputs over three streaming chunks."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort


def capture(model_path, metadata, voices, destination):
    model = onnx.load(model_path)
    node = next(node for node in model.graph.node if node.name == "/conv_14/Conv")
    weights = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer
               if tensor.name in node.input[1:]}
    weight = weights[node.input[1]]
    bias = weights[node.input[2]]
    if weight.shape != (384, 384, 7) or bias.shape != (384,):
        raise ValueError("unexpected convolution weights")
    for name in (node.input[0], node.output[0]):
        model.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 384, None]))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    meta = json.loads(metadata.read_text())
    codes = np.asarray(json.loads(voices.read_text())["voices"][0]["codes"], dtype=np.int64)
    states = [np.zeros((8, 2, 1, 16, 0, 64), np.float32),
              *[np.zeros(item["shape"], np.float32) for item in meta["states"]]]
    destination.mkdir(parents=True, exist_ok=True)
    weight.astype("<f4").tofile(destination / "weight.f32")
    bias.astype("<f4").tofile(destination / "bias.f32")
    offset = 0
    for count in (4, 8, 8):
        chunk = codes[np.arange(offset, offset + count) % len(codes)]
        feed = dict(zip(meta["inputs"], [chunk.T[None], np.array([offset], np.int64), *states]))
        _, *states, activation, expected = session.run(None, feed)
        rows = expected.shape[-1]
        if activation.shape != (1, 384, rows + 6):
            raise ValueError("unexpected convolution activation shape")
        activation.astype("<f4").tofile(destination / f"input-{offset}.f32")
        expected[0].T.astype("<f4").tofile(destination / f"output-{offset}.f32")
        print(f"offset={offset} frames={count} positions={rows}")
        offset += count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "metadata", "voices", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    capture(args.model, args.metadata, args.voices, args.destination)
