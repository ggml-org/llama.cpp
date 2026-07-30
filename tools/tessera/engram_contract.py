#!/usr/bin/env python3
"""GGUF and training-manifest contract for Tessera Engram adapters."""

from __future__ import annotations

import hashlib
import json
from typing import Any


SCHEMA = "llama.tessera.engram-contract.v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def make_contract(
    hash_manifest: dict,
    value_dimension: int,
    embedding_dimension: int,
    insertion_layers: list[int],
    source_revision: str,
) -> dict:
    if hash_manifest.get("schema") != "llama.tessera.engram-hash.v1":
        raise ValueError("unsupported Engram hash manifest")
    if value_dimension <= 0 or embedding_dimension <= 0:
        raise ValueError("Engram dimensions must be positive")
    layers = sorted(set(int(layer) for layer in insertion_layers))
    hash_layers = sorted(int(layer) for layer in hash_manifest["layer_moduli"])
    if layers != hash_layers:
        raise ValueError("contract layers do not match hash manifest")
    rows_by_layer = {
        str(layer): sum(
            modulus
            for order in hash_manifest["layer_moduli"][str(layer)]
            for modulus in order
        )
        for layer in layers
    }
    contract = {
        "schema": SCHEMA,
        "version": 1,
        "hash_digest": hash_manifest["digest"],
        "source": {
            "repo": "deepseek-ai/Engram",
            "revision": source_revision,
        },
        "insertion_layers": layers,
        "ngram_orders": list(range(2, int(hash_manifest["max_ngram_size"]) + 1)),
        "heads_per_ngram": int(hash_manifest["heads_per_ngram"]),
        "embedding_dimension": embedding_dimension,
        "value_dimension": value_dimension,
        "memory_encoding": "rowwise-q8",
        "rows_by_layer": rows_by_layer,
        "tensor_names": {
            "token_map": "engram.token_map",
            "memory": "blk.{layer}.engram.memory.weight",
            "value_projection": "blk.{layer}.engram.value_proj.weight",
            "key_projection": "blk.{layer}.engram.key_proj.weight",
            "query_norm": "blk.{layer}.engram.query_norm.weight",
            "key_norm": "blk.{layer}.engram.key_norm.weight",
            "short_conv": "blk.{layer}.engram.short_conv.weight",
        },
    }
    contract["estimated_memory_bytes"] = sum(rows_by_layer.values()) * embedding_dimension
    contract["digest"] = hashlib.sha256(canonical_json(contract)).hexdigest()
    return contract


def gguf_metadata(contract: dict) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported Engram contract")
    return {
        "tessera.engram.version": int(contract["version"]),
        "tessera.engram.contract_digest": contract["digest"],
        "tessera.engram.hash_digest": contract["hash_digest"],
        "tessera.engram.layers": contract["insertion_layers"],
        "tessera.engram.ngram_orders": contract["ngram_orders"],
        "tessera.engram.heads_per_ngram": contract["heads_per_ngram"],
        "tessera.engram.embedding_dimension": contract["embedding_dimension"],
        "tessera.engram.value_dimension": contract["value_dimension"],
        "tessera.engram.memory_encoding": contract["memory_encoding"],
        "tessera.engram.source_revision": contract["source"]["revision"],
    }
