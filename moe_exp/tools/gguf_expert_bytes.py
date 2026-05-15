#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict
from gguf import GGUFReader

LAYER_RE = re.compile(r"^blk\.(\d+)\.")
EXPERT_TENSOR_RE = re.compile(r"^blk\.(\d+)\..*exps.*")


def tensor_nbytes(tensor):
    if hasattr(tensor.data, "nbytes"):
        return int(tensor.data.nbytes)

    raise TypeError(f"Cannot determine nbytes for tensor {tensor.name}")


def gib(x):
    return x / (1024**3)


def mib(x):
    return x / (1024**2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--n-expert", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=48)
    ap.add_argument("--show-tensors", action="store_true")
    args = ap.parse_args()

    reader = GGUFReader(args.gguf)

    total_bytes = 0
    expert_bytes = 0
    non_expert_bytes = 0

    expert_by_layer = defaultdict(int)
    all_by_layer = defaultdict(int)

    expert_tensors = []
    non_expert_tensors = []

    for t in reader.tensors:
        name = t.name
        nbytes = tensor_nbytes(t)
        total_bytes += nbytes

        m_layer = LAYER_RE.match(name)
        if m_layer:
            layer = int(m_layer.group(1))
            all_by_layer[layer] += nbytes

        m_expert = EXPERT_TENSOR_RE.match(name)
        if m_expert:
            layer = int(m_expert.group(1))
            expert_bytes += nbytes
            expert_by_layer[layer] += nbytes
            expert_tensors.append((name, nbytes, list(t.shape), str(t.tensor_type)))
        else:
            non_expert_bytes += nbytes
            non_expert_tensors.append((name, nbytes, list(t.shape), str(t.tensor_type)))

    print(f"file: {args.gguf}")
    print()
    print("== total ==")
    print(f"total tensor bytes:      {total_bytes:15d}  {gib(total_bytes):8.3f} GiB")
    print(f"expert tensor bytes:     {expert_bytes:15d}  {gib(expert_bytes):8.3f} GiB")
    print(
        f"non-expert tensor bytes: {non_expert_bytes:15d}  {gib(non_expert_bytes):8.3f} GiB"
    )

    if total_bytes:
        print(f"expert fraction:         {expert_bytes / total_bytes:8.3%}")

    print()
    print("== per-layer expert bytes ==")
    print(
        "layer expert_bytes_MiB avg_expert_MiB all_layer_MiB expert_fraction_in_layer"
    )

    for layer in range(args.n_layer):
        eb = expert_by_layer[layer]
        ab = all_by_layer[layer]
        avg = eb / args.n_expert if args.n_expert else 0
        frac = eb / ab if ab else 0.0

        print(
            f"{layer:02d} "
            f"{mib(eb):16.3f} "
            f"{mib(avg):14.3f} "
            f"{mib(ab):13.3f} "
            f"{frac:24.3%}"
        )

    print()
    print("== resident expert budget estimate ==")
    print("budget_per_layer resident_expert_GiB fraction_of_all_experts")

    for budget in [8, 16, 24, 32, 48, 64, 96, 128]:
        # This assumes expert bytes are roughly evenly split across experts within each layer.
        # For Qwen-style *_exps tensors this is usually true enough for first-order estimate.
        resident = 0.0
        for layer in range(args.n_layer):
            resident += (
                expert_by_layer[layer] * min(budget, args.n_expert) / args.n_expert
            )

        print(
            f"{budget:16d} "
            f"{gib(int(resident)):19.3f} "
            f"{budget / args.n_expert:22.3%}"
        )

    if args.show_tensors:
        print()
        print("== expert tensors ==")
        for name, nbytes, shape, typ in expert_tensors:
            print(
                f"{name:80s} shape={shape} type={typ} bytes={nbytes} MiB={mib(nbytes):.3f}"
            )


if __name__ == "__main__":
    main()
