#!/usr/bin/env python3
"""Fuse the Snake activation chain into one m5::Snake custom op.

The codec applies snake(x) = x + c * sin(alpha*x)^2 at ~29 sites as

    alpha = Exp(const)                     (constant subgraph)
    c     = Mul(Reciprocal(Add(Exp(const), const)), const)
    Mul(x, alpha) -> Sin -> Pow(., 2) -> Mul(., c) -> Add(x, .)

Five full passes over big tensors plus the constant bookkeeping. The m5::Snake
op (m5_snake_op.h) computes the whole activation in one pass with a
range-reduced sine polynomial; this script pattern-matches the chain,
evaluates the alpha/c constant subgraphs once via a tiny ORT session, and
replaces each chain with

    Snake(x, alpha_const, c_const)

Shared intermediates are handled: a chain node is dropped only when every one
of its consumers was replaced.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def static_value(name, by_out, inits):
    """Resolve a tensor that is an initializer or the output of a Constant node."""
    if name in inits:
        t = inits[name]
        return numpy_helper.to_array(t) if (t.raw_data or t.dims) else None
    node = by_out.get(name)
    if node is not None and node.op_type == "Constant":
        for a in node.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
    return None


def chain_of(add, by_out):
    """Match the Add(x, Mul(c, Pow(Sin(Mul(x, a)), 2))) pattern."""
    if add.op_type != "Add":
        return None
    for a in (0, 1):
        x_name = add.input[a]
        mul2 = by_out.get(add.input[1 - a])
        if mul2 is None or mul2.op_type != "Mul":
            continue
        pow_in = c_name = None
        for i in (0, 1):
            src = by_out.get(mul2.input[i])
            if src is not None and src.op_type == "Pow":
                pow_in = src
                c_name = mul2.input[1 - i]
        if pow_in is None or len(pow_in.input) < 2:
            continue
        exp_val = static_value(pow_in.input[1], by_out, INIT)
        if exp_val is None or float(np.asarray(exp_val).reshape(-1)[0]) != 2.0:
            continue
        sin_n = by_out.get(pow_in.input[0])
        if sin_n is None or sin_n.op_type != "Sin":
            continue
        mul1 = by_out.get(sin_n.input[0])
        if mul1 is None or mul1.op_type != "Mul":
            continue
        for i in (0, 1):
            if mul1.input[1 - i] == x_name:
                return {
                    "x": x_name,
                    "alpha_src": mul1.input[i],
                    "c_src": c_name,
                    "nodes": [mul1, sin_n, pow_in, mul2, add],
                }
    return None


def const_closure(roots, by_out, inits):
    """All nodes/initializers needed to compute `roots` from initializers.
    Returns (nodes, init_names, ok). ok is False when a root is not constant."""
    nodes = {}
    init_used = set()
    stack = list(roots)
    seen = set()
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in inits:
            init_used.add(name)
            continue
        node = by_out.get(name)
        if node is None:
            return None, None, False           # graph input - not constant
        nodes[node.name] = node
        stack.extend(node.input)
    return list(nodes.values()), init_used, True


def eval_const(nodes, init_names, inits, outputs):
    g_inits = [inits[n] for n in init_names]
    out_vi = [helper.make_tensor_value_info(o, TensorProto.FLOAT, None) for o in outputs]
    g = helper.make_graph(nodes, "const", [], out_vi, g_inits)
    m = helper.make_model(g, opset_imports=[
        helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx", 17)])
    m.ir_version = 8
    sess = ort.InferenceSession(m.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    return sess.run(outputs, {})


def fuse(source: Path, dest: Path, dry_run: bool = False) -> None:
    global INIT, BY_INIT
    m = onnx.load(source)
    g = m.graph
    INIT = {t.name: t for t in g.initializer}
    BY_INIT = g.initializer
    by_out = {}
    for n in g.node:
        for o in n.output:
            by_out[o] = n

    matches = []
    for n in g.node:
        hit = chain_of(n, by_out)
        if hit:
            matches.append(hit)

    # evaluate alpha/c constants
    plans = []
    for hit in matches:
        ok = True
        const_nodes, const_inits = set(), set()
        vals = {}
        for key in ("alpha_src", "c_src"):
            ns, ins, is_const = const_closure([hit[key]], by_out, INIT)
            if not is_const:
                ok = False
                break
            const_nodes.update(id(x) for x in ns)
            const_inits.update(ins)
        if not ok:
            continue
        ns, ins, _ = const_closure([hit["alpha_src"], hit["c_src"]], by_out, INIT)
        try:
            a_val, c_val = eval_const(ns, ins, INIT, [hit["alpha_src"], hit["c_src"]])
        except Exception as e:
            print(f"skip {hit['nodes'][-1].name}: const eval failed: {e}")
            continue
        hit["alpha_val"] = np.asarray(a_val, np.float32)
        hit["c_val"] = np.asarray(c_val, np.float32)
        hit["const_nodes"] = {x.name for x in ns}
        hit["const_inits"] = ins
        plans.append(hit)

    if dry_run:
        print(f"matched {len(plans)} Snake chains")
        for hit in plans:
            add = hit["nodes"][-1]
            print(f"  {add.name:34s} x=...{hit['x'][-26:]}  "
                  f"alpha{hit['alpha_val'].shape} c{hit['c_val'].shape}")
        return

    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    by_name = {n.name: n for n in g.node}
    snake_nodes = {id(h["nodes"][-1]) for h in plans}
    chain_ids = set()
    for h in plans:
        for n in h["nodes"][:-1]:
            chain_ids.add(id(n))
        for name in h["const_nodes"]:
            if name in by_name:
                chain_ids.add(id(by_name[name]))
    all_match_outputs = {o for h in plans for n in [h["nodes"][-1]] for o in n.output}
    consumed_ok_outputs = all_match_outputs | {
        o for h in plans for n in h["nodes"] for o in n.output}

    def drop_ok(node):
        for o in node.output:
            for cons in consumers.get(o, []):
                if id(cons) in snake_nodes or id(cons) in chain_ids:
                    continue
                return False
        return True

    out_nodes = []
    added_inits = []
    replaced = dropped = 0
    for n in g.node:
        hit = next((h for h in plans if h["nodes"][-1] is n), None)
        if hit:
            tag = n.name.replace("/", "_")
            a_name = f"__snake_{tag}_alpha"
            c_name = f"__snake_{tag}_c"
            added_inits += [
                numpy_helper.from_array(hit["alpha_val"], a_name),
                numpy_helper.from_array(hit["c_val"], c_name),
            ]
            out_nodes.append(helper.make_node(
                "Snake", [hit["x"], a_name, c_name], list(n.output),
                name=n.name + "_snake", domain="m5"))
            replaced += 1
            continue
        if (id(n) in chain_ids) and drop_ok(n):
            dropped += 1
            continue
        out_nodes.append(n)

    del g.node[:]
    g.node.extend(out_nodes)
    g.initializer.extend(added_inits)
    if not any(o.domain == "m5" for o in m.opset_import):
        m.opset_import.extend([helper.make_opsetid("m5", 1)])
    onnx.save(m, dest)
    print(f"fused {replaced} Snake chains (dropped {dropped} intermediates) -> "
          f"{dest} ({len(g.node)} nodes, {dest.stat().st_size / 1e6:.0f} MB)")


INIT = {}
BY_INIT = []

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    fuse(args.source, args.dest, args.dry_run)
