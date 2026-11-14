#!/usr/bin/env python3
"""Represent codec Conv1D as equivalent Conv2D with a singleton height."""

import argparse

import onnx
from onnx import helper, numpy_helper


def lift(source, destination):
    model = onnx.load(source)
    graph = model.graph
    weights = {tensor.name: tensor for tensor in graph.initializer}
    opset = next(item.version for item in model.opset_import if not item.domain)
    if opset < 13:
        raise ValueError("expected ONNX opset 13 or newer")
    axes = "__m5_conv_height_axis"
    graph.initializer.append(helper.make_tensor(axes, onnx.TensorProto.INT64, [1], [2]))
    lifted = {}
    nodes = []
    for index, node in enumerate(graph.node):
        weight = weights.get(node.input[1]) if len(node.input) > 1 else None
        if node.op_type not in ("Conv", "ConvTranspose") or weight is None or len(weight.dims) != 3:
            nodes.append(node)
            continue
        prefix = f"__m5_conv_{index}"
        if weight.name not in lifted:
            name = prefix + "_weight"
            graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(weight)[:, :, None, :], name))
            lifted[weight.name] = name
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        for attr in ("dilations", "kernel_shape", "strides"):
            if attr in attrs:
                attrs[attr] = [1, *attrs[attr]]
        if "pads" in attrs:
            before, after = attrs["pads"]
            attrs["pads"] = [0, before, 0, after]
        if "output_padding" in attrs:
            attrs["output_padding"] = [0, *attrs["output_padding"]]
        if "output_shape" in attrs:
            attrs["output_shape"] = [1, *attrs["output_shape"]]
        nodes.extend([
            helper.make_node("Unsqueeze", [node.input[0], axes], [prefix + "_input"]),
            helper.make_node(node.op_type, [prefix + "_input", lifted[weight.name], *node.input[2:]],
                             [prefix + "_output"], name=node.name, **attrs),
            helper.make_node("Squeeze", [prefix + "_output", axes], list(node.output)),
        ])
    del graph.node[:]
    graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input} | {value.name for value in graph.output}
    initializers = [tensor for tensor in graph.initializer if tensor.name not in lifted or tensor.name in used]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    onnx.checker.check_model(model)
    onnx.save(model, destination)
    print(f"Lifted {len(lifted)} distinct convolution weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    lift(args.source, args.destination)
