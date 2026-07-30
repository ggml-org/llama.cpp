#!/usr/bin/env python3
"""Prepare a reproducible, distribution-cleared Tessera Engram pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


MODULE_PATH = Path(__file__).with_name("engram_hash.py")
SPEC = importlib.util.spec_from_file_location("tessera_engram_hash", MODULE_PATH)
HASH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = HASH
SPEC.loader.exec_module(HASH)

CORPUS_SCHEMA = "llama.tessera.training-corpus.v1"
INDEX_SCHEMA = "llama.tessera.engram-index.v1"
EMAIL = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SECRET = re.compile(r"(?i)\b(?:api[_-]?key|secret|password|token)\s*[:=]\s*\S+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require_cleared_receipt(path: Path, corpus: Path) -> dict:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != CORPUS_SCHEMA:
        raise ValueError("unsupported training corpus receipt")
    if receipt.get("distribution_cleared") is not True:
        raise ValueError("training corpus is not distribution-cleared")
    if receipt.get("contains_user_inference") is not False:
        raise ValueError("training corpus may contain user inference data")
    if not receipt.get("license"):
        raise ValueError("training corpus license is unresolved")
    if receipt.get("sha256") != sha256_file(corpus):
        raise ValueError("training corpus digest does not match receipt")
    return receipt


def sensitive(text: str) -> bool:
    return bool(EMAIL.search(text) or IPV4.search(text) or SECRET.search(text))


def count_ngrams(
    corpus: Path,
    tokenizer,
    orders: range,
    minimum_count: int,
    maximum_entries: int,
) -> tuple[list[dict], dict]:
    counts: Counter[tuple[int, ...]] = Counter()
    lines = 0
    rejected_lines = 0
    tokens = 0
    with corpus.open("r", encoding="utf-8", errors="replace") as source:
        for raw in source:
            text = raw.strip()
            if not text:
                continue
            lines += 1
            if sensitive(text):
                rejected_lines += 1
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            tokens += len(ids)
            for order in orders:
                counts.update(
                    tuple(ids[index:index + order])
                    for index in range(len(ids) - order + 1)
                )
    selected = []
    for token_ids, count in counts.most_common():
        if count < minimum_count or len(selected) >= maximum_entries:
            break
        text = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if sensitive(text):
            continue
        selected.append({"token_ids": list(token_ids), "count": count})
    return selected, {
        "lines": lines,
        "rejected_lines": rejected_lines,
        "tokens": tokens,
        "distinct_ngrams": len(counts),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def command_spec(args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
    )
    lookup = HASH.build_compressed_lookup(tokenizer)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has neither a pad nor EOS token")
    spec = HASH.make_hash_spec(
        compressed_vocab_size=int(lookup.max()) + 1,
        vocab_size_per_ngram=[args.vocab_size] * (args.max_ngram - 1),
        max_ngram_size=args.max_ngram,
        heads_per_ngram=args.heads,
        layer_ids=args.layers,
        pad_id=int(lookup[pad_token_id]),
        seed=args.seed,
    )
    output = Path(args.output)
    write_json(output, spec.to_dict())
    np.save(output.with_suffix(".token-map.npy"), lookup, allow_pickle=False)


def command_index(args) -> None:
    corpus = Path(args.corpus)
    receipt = require_cleared_receipt(Path(args.receipt), corpus)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
    )
    entries, stats = count_ngrams(
        corpus,
        tokenizer,
        range(args.min_ngram, args.max_ngram + 1),
        args.minimum_count,
        args.maximum_entries,
    )
    payload = {
        "schema": INDEX_SCHEMA,
        "corpus_epoch": receipt["epoch"],
        "corpus_sha256": receipt["sha256"],
        "tokenizer_sha256": sha256_file(Path(args.tokenizer) / "tokenizer.json"),
        "orders": list(range(args.min_ngram, args.max_ngram + 1)),
        "minimum_count": args.minimum_count,
        "maximum_entries": args.maximum_entries,
        "stats": stats,
        "entries": entries,
    }
    payload["digest"] = hashlib.sha256(HASH.canonical_json(payload)).hexdigest()
    write_json(Path(args.output), payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    spec = subparsers.add_parser("spec")
    spec.add_argument("--tokenizer", required=True)
    spec.add_argument("--output", required=True)
    spec.add_argument("--layers", type=int, nargs="+", default=[1, 15])
    spec.add_argument("--max-ngram", type=int, default=3)
    spec.add_argument("--heads", type=int, default=8)
    spec.add_argument("--vocab-size", type=int, default=65_521)
    spec.add_argument("--seed", type=int, default=0)
    spec.set_defaults(function=command_spec)
    index = subparsers.add_parser("index")
    index.add_argument("--tokenizer", required=True)
    index.add_argument("--corpus", required=True)
    index.add_argument("--receipt", required=True)
    index.add_argument("--output", required=True)
    index.add_argument("--min-ngram", type=int, default=2)
    index.add_argument("--max-ngram", type=int, default=4)
    index.add_argument("--minimum-count", type=int, default=8)
    index.add_argument("--maximum-entries", type=int, default=1_000_000)
    index.set_defaults(function=command_index)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
