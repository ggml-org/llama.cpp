#!/usr/bin/env python3
"""Replay official sampling after substituting each depth core hidden state."""

import argparse

import numpy as np
import onnx
from onnx import numpy_helper


def check(model, reference, hidden_path):
    graph = onnx.load(model).graph
    weights = {tensor.name: tensor for tensor in graph.initializer}
    nodes = {node.name: node for node in graph.node}
    hidden = np.fromfile(hidden_path, dtype="<f4").reshape(15, 512)
    with np.load(reference) as data:
        noise = data["noise"][1:, 0]
        expected = data["codes"][0, 1:]
    actual = []
    for index, state in enumerate(hidden):
        suffix = f"_{index}" if index else ""
        head = numpy_helper.to_array(weights[nodes[f"/MatMul{suffix}"].input[1]])
        logits = (state @ head) / np.float32(0.9)
        cutoff = np.partition(logits, -10)[-10]
        logits = np.where(logits >= cutoff, logits, -np.inf)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        actual.append(np.argmax(probabilities / noise[index]))
    actual = np.asarray(actual)
    mismatches = np.flatnonzero(actual != expected) + 1
    print(f"matched={np.count_nonzero(actual == expected)}/15 differing_codebooks={mismatches.tolist()}")
    return not len(mismatches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("model", "reference", "hidden"):
        parser.add_argument(argument)
    args = parser.parse_args()
    raise SystemExit(not check(args.model, args.reference, args.hidden))
