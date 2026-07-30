#!/usr/bin/env python3
"""Create a deterministic approved-corpus replay subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path


SCHEMA = "llama.tessera.replay-corpus.v1"

FAMILY_TERMS = {
    "code": {"code", "function", "class", "python", "javascript", "rust", "c++", "sql", "debug", "algorithm"},
    "math": {"calculate", "equation", "proof", "theorem", "probability", "matrix", "integral", "derivative"},
    "reasoning": {"explain", "compare", "analyze", "reason", "because", "therefore", "tradeoff", "hypothesis"},
    "factual": {"what", "when", "where", "who", "history", "science", "definition", "describe"},
    "creative": {"story", "poem", "creative", "imagine", "character", "scene", "dialogue", "rewrite"},
    "tool": {"json", "api", "tool", "schema", "command", "terminal", "search", "database"},
    "safety": {"safe", "risk", "harm", "medical", "legal", "privacy", "security", "policy"},
    "multimodal": {"image", "video", "audio", "vision", "picture", "diagram", "chart"},
}


def stable_hash(salt: str, text: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{salt}\0{text}".encode("utf-8")).digest()[:8],
        "big",
    )


def semantic_family(paragraph: str) -> str:
    words = set(re.findall(r"[a-z0-9+#]+", paragraph.lower()))
    scores = {
        family: len(words.intersection(terms))
        for family, terms in FAMILY_TERMS.items()
    }
    family, score = max(scores.items(), key=lambda item: (item[1], item[0]))
    if score == 0:
        family = "multilingual" if any(ord(c) > 127 for c in paragraph) else "general"
    word_count = max(1, len(re.findall(r"\S+", paragraph)))
    length_bucket = "short" if word_count < 80 else "medium" if word_count < 240 else "long"
    return f"{family}:{length_bucket}"


def select_semantic_replay(
    paragraphs: list[str],
    fraction: float,
    salt: str,
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    families: dict[str, list[str]] = defaultdict(list)
    for paragraph in paragraphs:
        families[semantic_family(paragraph)].append(paragraph)
    for values in families.values():
        values.sort(key=lambda value: stable_hash(salt, value))

    target = min(len(paragraphs), max(1, math.ceil(len(paragraphs) * fraction)))
    selected: list[str] = []
    selected_counts: dict[str, int] = defaultdict(int)
    active = sorted(
        families,
        key=lambda family: (-len(families[family]), stable_hash(salt, family)),
    )
    round_index = 0
    while active and len(selected) < target:
        next_active = []
        for family in active:
            values = families[family]
            if round_index < len(values) and len(selected) < target:
                selected.append(values[round_index])
                selected_counts[family] += 1
            if round_index + 1 < len(values):
                next_active.append(family)
        active = next_active
        round_index += 1
    return selected, dict(sorted(
        (family, len(values)) for family, values in families.items()
    )), dict(sorted(selected_counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic calibration replay subset")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--salt", default="tessera-repair-v1")
    parser.add_argument(
        "--strategy",
        choices=("hash", "semantic-family"),
        default="semantic-family",
    )
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    source = Path(args.input)
    paragraphs = [
        paragraph.strip()
        for paragraph in source.read_text(encoding="utf-8").split("\n\n")
        if paragraph.strip()
    ]
    source_families: dict[str, int] = {}
    selected_families: dict[str, int] = {}
    if args.strategy == "semantic-family":
        selected, source_families, selected_families = select_semantic_replay(
            paragraphs, args.fraction, args.salt
        )
    else:
        selected = []
        threshold = int(args.fraction * (1 << 64))
        for paragraph in paragraphs:
            if stable_hash(args.salt, paragraph) < threshold:
                selected.append(paragraph)
    if not selected and paragraphs:
        selected.append(paragraphs[0])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(selected) + "\n", encoding="utf-8")
    receipt = {
        "schema": SCHEMA,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "fraction": args.fraction,
        "salt": args.salt,
        "strategy": args.strategy,
        "source_paragraphs": len(paragraphs),
        "selected_paragraphs": len(selected),
        "source_semantic_families": source_families,
        "selected_semantic_families": selected_families,
        "covered_semantic_families": len(selected_families),
        "semantic_family_count": len(source_families),
        "recommended_convergence_min_chunks": max(
            64, 4 * len(selected_families)
        ),
        "statistical_role": "replay",
        "holdout": False,
    }
    Path(args.receipt).write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
