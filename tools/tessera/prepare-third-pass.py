#!/usr/bin/env python3
"""Seal the non-training inputs for a Tessera compression-aware Engram pass."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


CONTRACT_PATH = Path(__file__).with_name("engram_contract.py")
SPEC = importlib.util.spec_from_file_location("tessera_engram_contract", CONTRACT_PATH)
CONTRACT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CONTRACT
SPEC.loader.exec_module(CONTRACT)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_corpus_receipt(path: Path, corpus: Path) -> dict:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "llama.tessera.training-corpus.v1":
        raise ValueError("unsupported training corpus receipt")
    if receipt.get("sha256") != sha256_file(corpus):
        raise ValueError("training corpus receipt digest mismatch")
    if receipt.get("distribution_cleared") is not True:
        raise ValueError("training corpus is not distribution-cleared")
    if receipt.get("contains_user_inference") is not False:
        raise ValueError("training corpus may contain user inference data")
    if not receipt.get("license"):
        raise ValueError("training corpus license is unresolved")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--corpus")
    parser.add_argument("--corpus-receipt")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 15])
    parser.add_argument("--embedding-dimension", type=int, default=16)
    parser.add_argument("--value-dimension", type=int, default=256)
    args = parser.parse_args()
    hash_manifest = json.loads(Path(args.hash_manifest).read_text(encoding="utf-8"))
    contract = CONTRACT.make_contract(
        hash_manifest,
        value_dimension=args.value_dimension,
        embedding_dimension=args.embedding_dimension,
        insertion_layers=args.layers,
        source_revision=args.source_revision,
    )
    if bool(args.corpus) != bool(args.corpus_receipt):
        raise ValueError("--corpus and --corpus-receipt must be provided together")
    corpus_receipt = None
    if args.corpus:
        corpus_receipt = load_corpus_receipt(
            Path(args.corpus_receipt),
            Path(args.corpus),
        )
    payload = {
        "contract": contract,
        "gguf_metadata": CONTRACT.gguf_metadata(contract),
        "training": {
            "backbone": "gemma-4-12b-unified",
            "teacher_cache": {"k": "f16", "v": "f16"},
            "student_cache_curriculum": [
                {"k": "q8_0", "v": "q8_0"},
                {"k": "q5_1", "v": "q5_1"},
                {"k": "q4_1", "v": "q4_1"},
                {"k": "iq4_nl", "v": "iq4_nl"},
            ],
            "flash_attention": True,
            "trainable": [
                "engram-memory",
                "engram-gates",
                "engram-projections",
                "attention-lora",
                "attention-norms",
                "mtp-alignment",
                "dflash-alignment",
            ],
            "losses": [
                "next-token",
                "teacher-logit-kl",
                "hidden-state",
                "attention-output",
                "draft-acceptance",
            ],
        },
        "corpus": (
            {
                "status": "licensed",
                "schema": corpus_receipt["schema"],
                "epoch": corpus_receipt["epoch"],
                "sha256": corpus_receipt["sha256"],
                "license": corpus_receipt["license"],
                "license_uri": corpus_receipt.get("license_uri"),
                "attribution": corpus_receipt.get("attribution"),
                "distribution_cleared": True,
                "contains_user_inference": False,
                "commercial_use": bool(corpus_receipt.get("commercial_use", False)),
                "share_alike": bool(corpus_receipt.get("share_alike", False)),
            }
            if corpus_receipt is not None
            else {
                "required_schema": "llama.tessera.training-corpus.v1",
                "distribution_cleared": True,
                "contains_user_inference": False,
                "status": "awaiting-approved-receipt",
            }
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
