#!/usr/bin/env python3
"""Deterministic Engram hashing shared by corpus preparation and parity tests."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "llama.tessera.engram-hash.v1"
PRIME_1 = 10007


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def normalize_piece(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.lower()
    return " ".join(text.split())


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    seen.add(candidate)
    return candidate


def build_compressed_lookup(tokenizer: Any) -> np.ndarray:
    keys: dict[str, int] = {}
    lookup = np.empty(len(tokenizer), dtype=np.int64)
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if "\ufffd" in text:
            key = tokenizer.convert_ids_to_tokens(token_id)
        else:
            key = normalize_piece(text) or text
        lookup[token_id] = keys.setdefault(key, len(keys))
    return lookup


@dataclass(frozen=True)
class HashSpec:
    max_ngram_size: int
    heads_per_ngram: int
    pad_id: int
    layer_multipliers: dict[int, list[int]]
    layer_moduli: dict[int, list[list[int]]]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": SCHEMA,
            "max_ngram_size": self.max_ngram_size,
            "heads_per_ngram": self.heads_per_ngram,
            "pad_id": self.pad_id,
            "layer_multipliers": {
                str(layer): values
                for layer, values in sorted(self.layer_multipliers.items())
            },
            "layer_moduli": {
                str(layer): values
                for layer, values in sorted(self.layer_moduli.items())
            },
        }
        payload["digest"] = hashlib.sha256(canonical_json(payload)).hexdigest()
        return payload


def make_hash_spec(
    compressed_vocab_size: int,
    vocab_size_per_ngram: list[int],
    max_ngram_size: int,
    heads_per_ngram: int,
    layer_ids: list[int],
    pad_id: int,
    seed: int,
) -> HashSpec:
    if compressed_vocab_size <= 0:
        raise ValueError("compressed vocabulary must be positive")
    if max_ngram_size < 2:
        raise ValueError("Engram requires at least bigrams")
    if len(vocab_size_per_ngram) != max_ngram_size - 1:
        raise ValueError("one vocabulary size is required for every n-gram order")
    max_long = np.iinfo(np.int64).max
    half_bound = max(1, int(max_long // compressed_vocab_size) // 2)
    multipliers: dict[int, list[int]] = {}
    moduli: dict[int, list[list[int]]] = {}
    seen: set[int] = set()
    for layer_id in layer_ids:
        rng = np.random.default_rng(seed + PRIME_1 * layer_id)
        values = rng.integers(
            low=0,
            high=half_bound,
            size=max_ngram_size,
            dtype=np.int64,
        )
        multipliers[layer_id] = [int(value * 2 + 1) for value in values]
        per_order: list[list[int]] = []
        for vocabulary_size in vocab_size_per_ngram:
            search = vocabulary_size - 1
            per_head = []
            for _ in range(heads_per_ngram):
                prime = next_prime(search, seen)
                per_head.append(prime)
                search = prime
            per_order.append(per_head)
        moduli[layer_id] = per_order
    return HashSpec(
        max_ngram_size=max_ngram_size,
        heads_per_ngram=heads_per_ngram,
        pad_id=pad_id,
        layer_multipliers=multipliers,
        layer_moduli=moduli,
    )


def hash_layer(
    input_ids: np.ndarray,
    spec: HashSpec,
    layer_id: int,
) -> np.ndarray:
    tokens = np.asarray(input_ids, dtype=np.int64)
    if tokens.ndim != 2:
        raise ValueError("input token IDs must have shape [batch, tokens]")
    batch, length = tokens.shape
    shifted = [tokens]
    for offset in range(1, spec.max_ngram_size):
        prefix = np.full((batch, offset), spec.pad_id, dtype=np.int64)
        shifted.append(np.concatenate((prefix, tokens), axis=1)[:, :length])
    multipliers = spec.layer_multipliers[layer_id]
    outputs = []
    for ngram_size in range(2, spec.max_ngram_size + 1):
        mixed = np.multiply(
            shifted[0].view(np.uint64),
            np.uint64(multipliers[0]),
        )
        for offset in range(1, ngram_size):
            term = np.multiply(
                shifted[offset].view(np.uint64),
                np.uint64(multipliers[offset]),
            )
            mixed = np.bitwise_xor(mixed, term)
        signed = mixed.view(np.int64)
        for modulus in spec.layer_moduli[layer_id][ngram_size - 2]:
            outputs.append(np.mod(signed, modulus))
    return np.stack(outputs, axis=2)


def load_spec(path: Path) -> HashSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("digest")
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported Engram hash schema")
    if hashlib.sha256(canonical_json(payload)).hexdigest() != digest:
        raise ValueError("Engram hash manifest digest mismatch")
    return HashSpec(
        max_ngram_size=int(payload["max_ngram_size"]),
        heads_per_ngram=int(payload["heads_per_ngram"]),
        pad_id=int(payload["pad_id"]),
        layer_multipliers={
            int(layer): [int(value) for value in values]
            for layer, values in payload["layer_multipliers"].items()
        },
        layer_moduli={
            int(layer): [[int(value) for value in order] for order in values]
            for layer, values in payload["layer_moduli"].items()
        },
    )
