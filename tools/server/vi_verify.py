#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
import sys
import urllib.request
import urllib.parse
from typing import Optional


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
def _fetch_bytes(url: str, server_base: Optional[str], api_key: Optional[str]) -> bytes:
    if server_base and url.startswith("/"):
        url = urllib.parse.urljoin(server_base.rstrip("/") + "/", url.lstrip("/"))

    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    with urllib.request.urlopen(req) as resp:
        return resp.read()


def verify_response(obj: dict, *, server_base: Optional[str] = None, api_key: Optional[str] = None) -> None:
    if "verifiable_inference" not in obj:
        raise SystemExit("missing verifiable_inference in response")
    vi = obj["verifiable_inference"]
    root = bytes.fromhex(vi["merkle_root_sha256"])

    openings = vi.get("openings", [])
    if not isinstance(openings, list) or len(openings) == 0:
        raise SystemExit("verifiable_inference.openings missing/empty")

    for i, o in enumerate(openings):
        tensor = o["tensor"]
        if "opened_bytes_b64" in o:
            opened = base64.b64decode(o["opened_bytes_b64"])
        else:
            url = o.get("opened_bytes_url")
            if not url:
                raise SystemExit(f"opening[{i}]: missing opened_bytes_url (and no opened_bytes_b64)")
            if url.startswith("/") and not server_base:
                raise SystemExit(f"opening[{i}]: opened_bytes_url is relative; provide --server-base")
            opened = _fetch_bytes(url, server_base, api_key)

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
    ap.add_argument("--server-base", help="Base URL for relative opened_bytes_url, e.g. http://127.0.0.1:8080")
    ap.add_argument("--api-key", help="API key (sent as Authorization: Bearer ...)")
    args = ap.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        obj = json.load(sys.stdin)

    verify_response(obj, server_base=args.server_base, api_key=args.api_key)
    print("verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

