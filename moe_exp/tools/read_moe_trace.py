#!/usr/bin/env python3

import argparse
import os
import struct
from collections import Counter, defaultdict

REC_STRUCT = struct.Struct("<I H B B 8h")
REC_SIZE = REC_STRUCT.size

PHASE_NAME = {
    0: "prefill",
    1: "decode",
}


def read_records(path):
    size = os.path.getsize(path)

    if size % REC_SIZE != 0:
        raise ValueError(
            f"Bad file size: {size} bytes is not divisible by record size {REC_SIZE}"
        )

    with open(path, "rb") as f:
        idx = 0
        while True:
            data = f.read(REC_SIZE)
            if not data:
                break

            token_idx, layer_idx, phase, n_expert_used, *expert_ids = REC_STRUCT.unpack(
                data
            )

            yield {
                "record_idx": idx,
                "token_idx": token_idx,
                "layer_idx": layer_idx,
                "phase": phase,
                "n_expert_used": n_expert_used,
                "expert_ids": expert_ids[:n_expert_used],
                "raw_expert_ids": expert_ids,
            }

            idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--n-expert", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=48)
    parser.add_argument("--expected-k", type=int, default=8)
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    records = list(read_records(args.path))

    print(f"file: {args.path}")
    print(f"record_size: {REC_SIZE}")
    print(f"records: {len(records)}")

    if not records:
        return

    phase_counts = Counter(r["phase"] for r in records)
    layer_counts = Counter(r["layer_idx"] for r in records)
    token_counts = Counter(r["token_idx"] for r in records)
    k_counts = Counter(r["n_expert_used"] for r in records)

    bad = []

    for r in records:
        if r["phase"] not in (0, 1):
            bad.append((r, "bad phase"))

        if r["layer_idx"] < 0 or r["layer_idx"] >= args.n_layer:
            bad.append((r, "bad layer"))

        if r["n_expert_used"] != args.expected_k:
            bad.append((r, "bad n_expert_used"))

        if len(r["expert_ids"]) != r["n_expert_used"]:
            bad.append((r, "bad expert id count"))

        for expert_id in r["expert_ids"]:
            if expert_id < 0 or expert_id >= args.n_expert:
                bad.append((r, f"bad expert id {expert_id}"))
                break

    print()
    print("phase_counts:")
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase} ({PHASE_NAME.get(phase, 'unknown')}): {count}")

    print()
    print("n_expert_used counts:")
    for k, count in sorted(k_counts.items()):
        print(f"  {k}: {count}")

    print()
    print(f"token_idx range: {min(token_counts)}..{max(token_counts)}")
    print(f"unique token_idx: {len(token_counts)}")

    print()
    print(f"layer_idx range: {min(layer_counts)}..{max(layer_counts)}")
    print(f"unique layers: {len(layer_counts)}")
    print("layer counts sample:")
    for layer in range(args.n_layer):
        if layer in layer_counts:
            print(f"  layer {layer:02d}: {layer_counts[layer]}")

    expert_global = Counter()
    expert_by_layer = defaultdict(Counter)

    for r in records:
        layer = r["layer_idx"]
        for expert_id in r["expert_ids"]:
            expert_global[expert_id] += 1
            expert_by_layer[layer][expert_id] += 1

    print()
    print("top global experts:")
    for expert_id, count in expert_global.most_common(20):
        print(f"  expert {expert_id:03d}: {count}")

    print()
    print(f"validation_errors: {len(bad)}")
    for r, reason in bad[:20]:
        print(
            f"  record={r['record_idx']} reason={reason} "
            f"token={r['token_idx']} layer={r['layer_idx']} "
            f"phase={r['phase']} k={r['n_expert_used']} experts={r['expert_ids']}"
        )

    print()
    print("first records:")
    for r in records[: args.show]:
        phase_name = PHASE_NAME.get(r["phase"], "unknown")
        print(
            f"  rec={r['record_idx']} token={r['token_idx']} "
            f"layer={r['layer_idx']} phase={phase_name} "
            f"k={r['n_expert_used']} experts={r['expert_ids']}"
        )


if __name__ == "__main__":
    main()
