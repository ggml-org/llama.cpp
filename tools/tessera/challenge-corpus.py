#!/usr/bin/env python3
"""Build a deterministic high-information Tessera calibration challenge set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


SCHEMA = "llama.tessera.calibration-challenge.v1"


def rank(seed: int, record: dict) -> tuple[int, int]:
    text = record["text"]
    # Prefer structured, long, and non-ASCII samples while retaining a stable
    # hash tie-breaker. These are proxy stressors, never user-content scores.
    complexity = (
        4 * sum(character in "{}[]<>`|" for character in text)
        + 2 * sum(character == "\n" for character in text)
        + sum(ord(character) > 127 for character in text)
        + min(len(text) // 80, 32)
    )
    tie = int.from_bytes(hashlib.sha256(
        f"{seed}\0{record['id']}".encode()).digest()[:8], "big")
    return complexity, tie


def select(records: list[dict], per_category: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[record["category"]].append(record)
    selected = []
    for category in sorted(groups):
        ordered = sorted(groups[category], key=lambda record: rank(seed, record), reverse=True)
        selected.extend(ordered[:per_category])
    return sorted(selected, key=lambda record: record["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic Tessera calibration challenge corpus")
    parser.add_argument("--index", required=True, help="clean-room calibration samples.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--seed", type=int, default=640)
    args = parser.parse_args()
    if args.per_category < 1:
        raise ValueError("per-category must be positive")
    index = Path(args.index)
    records = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
    if not records or any(record.get("schema") != "llama.tessera.calibration-corpus.v1" for record in records):
        raise ValueError("index is not a Tessera clean-room calibration corpus")
    chosen = select(records, args.per_category, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(record["text"] for record in chosen) + "\n", encoding="utf-8")
    receipt = {
        "schema": SCHEMA,
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "seed": args.seed,
        "per_category": args.per_category,
        "samples": len(chosen),
        "categories": sorted({record["category"] for record in chosen}),
        "holdout": True,
        "statistical_role": "first-pass-challenge",
    }
    Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}: samples={len(chosen)} categories={len(receipt['categories'])}")


if __name__ == "__main__":
    main()
