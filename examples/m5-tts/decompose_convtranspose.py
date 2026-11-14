#!/usr/bin/env python3
"""Rewrite codec ConvTranspose nodes as a single MatMul + interleave/pad.

ORT CPU has no integer ConvTranspose kernel, but it does have MatMulInteger.
For pad=0, dilation=1, groups=1 a ConvTranspose(x, W, stride=S, kernel=K) is

    out[b, c, i*S + r] += x[b, :, i] @ W[:, c, r]      r in 0..K-1

which folds into one GEMM of shape [B, L, Cin] x [Cin, K*Cout] followed by a
reshape/interleave and, when K = 2S, an overlap-add of the two S-sized groups.
The resulting MatMul carries a constant weight and can be QDQ-quantized onto
the MatMulIntegerToFloat path.

Also replaces Pow(x, 2) with Mul(x, x): exact and much cheaper than std::pow.
"""
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def rename(prefix, name):
    return f"{prefix}/{name}"


def build_ct_replacement(prefix, node, graph_initializers):
    """Returns (nodes, initializers, output_name) replacing one ConvTranspose."""
    x_name, w_name = node.input[0], node.input[1]
    bias_name = node.input[2] if len(node.input) > 2 else None
    out_name = node.output[0]
    attrs = {a.name: (list(a.ints) if a.ints else a.i) for a in node.attribute}
    K = attrs.get("kernel_shape", [1])[0]
    S = attrs.get("strides", [1])[0]
    pads = attrs.get("pads", [0, 0])
    dil = attrs.get("dilations", [1])
    group = attrs.get("group", 1)
    if pads[:2] != [0, 0] or dil != [1] or group != 1:
        raise ValueError(f"{node.name}: unsupported ConvTranspose config {attrs}")
    if K % S != 0:
        raise ValueError(f"{node.name}: kernel {K} not a multiple of stride {S}")
    G = K // S                       # groups of S taps; G=1 or 2 expected
    if G not in (1, 2):
        raise ValueError(f"{node.name}: kernel/stride ratio {G} not handled")

    W = numpy_helper.to_array(graph_initializers[w_name])   # [Cin, Cout, K]
    Cin, Cout, _ = W.shape
    # W_flat[cin, c*K + r] = W[cin, c, r] - W is already (Cin, Cout, K), so a
    # plain reshape flattens (c, r) in exactly that order.
    W_flat = W.reshape(Cin, K * Cout).astype(np.float32)

    n = helper.make_node
    inits = [numpy_helper.from_array(W_flat, rename(prefix, "w"))]
    nodes = [
        n("Transpose", [x_name], [rename(prefix, "xt")], perm=[0, 2, 1]),
        n("MatMul", [rename(prefix, "xt"), rename(prefix, "w")], [rename(prefix, "y")]),
        n("Reshape", [rename(prefix, "y"), rename(prefix, "shape_kc")], [rename(prefix, "yr")]),
        n("Transpose", [rename(prefix, "yr")], [rename(prefix, "yt")], perm=[0, 2, 1, 3]),
    ]
    inits.append(numpy_helper.from_array(
        np.asarray([1, -1, Cout, K], np.int64), rename(prefix, "shape_kc")))

    if G == 1:
        # [B, Cout, L, S] -> [B, Cout, L*S]
        nodes.append(n("Reshape", [rename(prefix, "yt"), rename(prefix, "shape_out")],
                       [rename(prefix, "acc")]))
        inits.append(numpy_helper.from_array(
            np.asarray([0, 0, -1], np.int64), rename(prefix, "shape_out")))
        acc = rename(prefix, "acc")
    else:
        nodes.append(n("Reshape", [rename(prefix, "yt"), rename(prefix, "shape_g")],
                       [rename(prefix, "yg")]))
        inits.append(numpy_helper.from_array(
            np.asarray([0, 0, -1, G, S], np.int64), rename(prefix, "shape_g")))
        inits.append(numpy_helper.from_array(np.asarray([3], np.int64), rename(prefix, "axg")))
        inits.append(numpy_helper.from_array(np.asarray([1], np.int64), rename(prefix, "one")))
        # k = g*S + s, so [B,Cout,L,K] -> [B,Cout,L,G,S]; group g lands at
        # output offset g*S and contributes length L*S each.
        branches = []
        for g in range(2):
            # Slice opset 10+ input order: (data, starts, ends, axes, steps)
            nodes.append(n("Slice", [rename(prefix, "yg"), rename(prefix, f"st{g}"),
                                     rename(prefix, f"en{g}"), rename(prefix, "axg"),
                                     rename(prefix, "one")],
                           [rename(prefix, f"bg{g}r")]))
            nodes.append(n("Reshape", [rename(prefix, f"bg{g}r"), rename(prefix, "shape_out")],
                           [rename(prefix, f"bg{g}")]))
            nodes.append(n("Pad", [rename(prefix, f"bg{g}"), rename(prefix, f"pads{g}")],
                           [rename(prefix, f"bp{g}")]))
            inits += [
                numpy_helper.from_array(np.asarray([g], np.int64), rename(prefix, f"st{g}")),
                numpy_helper.from_array(np.asarray([g + 1], np.int64), rename(prefix, f"en{g}")),
                numpy_helper.from_array(np.asarray([0, 0, g * S, 0, 0, (1 - g) * S], np.int64),
                                        rename(prefix, f"pads{g}")),
            ]
            branches.append(rename(prefix, f"bp{g}"))
        inits.append(numpy_helper.from_array(
            np.asarray([0, 0, -1], np.int64), rename(prefix, "shape_out")))
        nodes.append(n("Add", branches, [rename(prefix, "acc")]))
        acc = rename(prefix, "acc")

    if bias_name:
        # [Cout] right-aligns against the time axis of [B, Cout, T]; make it
        # [1, Cout, 1] so it broadcasts along the channel axis instead.
        nodes.append(n("Reshape", [bias_name, rename(prefix, "shape_bias")],
                       [rename(prefix, "bias3")]))
        inits.append(numpy_helper.from_array(
            np.asarray([1, -1, 1], np.int64), rename(prefix, "shape_bias")))
        nodes.append(n("Add", [acc, rename(prefix, "bias3")], [rename(prefix, "biased")]))
        acc = rename(prefix, "biased")
    # Reshape/identity to the original output name keeps downstream untouched.
    if acc != out_name:
        nodes.append(n("Identity", [acc], [out_name]))
    # ORT's graph optimizer derives fusion names from node names; unnamed nodes
    # collide (e.g. two "/GatherSliceToSplitFusion/"). Name everything.
    for i, node in enumerate(nodes):
        if not node.name:
            node.name = f"{prefix}/n{i}"
    return nodes, inits


def rewrite(source, dest):
    m = onnx.load(source)
    graph = m.graph
    inits = {t.name: t for t in graph.initializer}
    out_nodes, added = [], []
    replaced_weights = set()
    replaced = 0
    for node in graph.node:
        if node.op_type == "ConvTranspose":
            prefix = f"__ct{replaced}"
            nodes, new_inits = build_ct_replacement(prefix, node, inits)
            out_nodes.extend(nodes)
            added.extend(new_inits)
            replaced_weights.add(node.input[1])
            replaced += 1
            continue
        if node.op_type == "Pow":
            exp = inits.get(node.input[1]) if len(node.input) > 1 else None
            if exp is not None and exp.data_type == TensorProto.FLOAT:
                vals = numpy_helper.to_array(exp)
                if vals.size == 1 and float(vals.reshape(-1)[0]) == 2.0:
                    out_nodes.append(helper.make_node(
                        "Mul", [node.input[0], node.input[0]], list(node.output),
                        name=node.name + "_sq"))
                    continue
        out_nodes.append(node)

    del graph.node[:]
    graph.node.extend(out_nodes)
    live_names = {name for node in out_nodes for name in node.input}
    live_names.update(value.name for value in graph.input)
    live_names.update(value.name for value in graph.output)
    kept = [tensor for tensor in graph.initializer
            if tensor.name not in replaced_weights or tensor.name in live_names]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    graph.initializer.extend(added)
    onnx.save(m, dest)
    print(f"rewrote {replaced} ConvTranspose nodes -> {dest} "
          f"({len(graph.node)} nodes, {dest.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("dest", type=Path)
    args = ap.parse_args()
    rewrite(args.source, args.dest)
