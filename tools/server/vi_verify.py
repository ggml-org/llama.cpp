#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
import sys


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _hex(b: bytes) -> str:
    return b.hex()


def _leaf_hash(tensor: dict, opened_bytes: bytes) -> bytes:
    # Must match common/vi-proof.cpp hash_trace_leaf()
    ne = tensor["ne"]
    hdr = (
        "llama.cpp/vi-leaf/v1\0"
        + tensor["name"] + "\0"
        + tensor["op"] + "\0"
        + tensor["type"] + "\0"
        + f"{ne[0]},{ne[1]},{ne[2]},{ne[3]}\0"
        + str(tensor["nbytes"]) + "\0"
    ).encode("utf-8")
    return hashlib.sha256(hdr + opened_bytes).digest()


def _merkle_recompute_root(leaf_index: int, leaf: bytes, siblings_hex: list[str]) -> bytes:
    cur = leaf
    idx = leaf_index
    for sib_hex in siblings_hex:
        sib = bytes.fromhex(sib_hex)
        if idx % 2 == 0:
            cur = _sha256(cur + sib)
        else:
            cur = _sha256(sib + cur)
        idx //= 2
    return cur


def verify_response(obj: dict) -> None:
    if "proof" not in obj:
        raise SystemExit("missing proof in response")
    vi = obj["proof"]
    root = bytes.fromhex(vi["merkle_root_sha256"])

    openings = vi.get("openings", [])
    if not isinstance(openings, list) or len(openings) == 0:
        raise SystemExit("proof.openings missing/empty")

    for i, o in enumerate(openings):
        tensor = o["tensor"]
        b64 = o["opened_bytes_b64"]
        opened = base64.b64decode(b64)

        # bytes hash
        bytes_hash = _sha256(opened)
        if _hex(bytes_hash) != o["opened_bytes_sha256"]:
            raise SystemExit(f"opening[{i}]: opened_bytes_sha256 mismatch")

        # leaf hash
        leaf = _leaf_hash(tensor, opened)
        if _hex(leaf) != o["leaf_hash_sha256"]:
            raise SystemExit(f"opening[{i}]: leaf_hash_sha256 mismatch")

        # merkle root
        root2 = _merkle_recompute_root(o["leaf_index"], leaf, o["merkle_path_siblings_sha256"])
        if root2 != root:
            raise SystemExit(f"opening[{i}]: merkle root mismatch")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="response JSON file (defaults to stdin)")
    args = ap.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        obj = json.load(sys.stdin)

    verify_response(obj)
    print("verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

