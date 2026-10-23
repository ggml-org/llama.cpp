#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent.parent / 'gguf-py'))
import gguf  # noqa: E402

DESCRIPTION = """
Build the token id list for --spec-draft-vocab: the control and user-defined ids, the ids below --id-floor, then the
most frequent ids of the corpus (ideally text the target model generated), then the lowest unused ids.
An evenly spread --holdout of each file is left out of the counts to report the coverage of unseen text.

Example:
    python scripts/build-draft-vocab.py -m model.gguf --tokenize build/bin/llama-tokenize \\
        --text-file outputs.jsonl --num-tokens 32768 -o draft-vocab.json

    llama-server -m model.gguf --spec-type draft-mtp --spec-draft-vocab draft-vocab.json
"""

KEEP_TYPES = {gguf.TokenType.CONTROL, gguf.TokenType.USER_DEFINED}


def read_docs(path: str) -> list[str]:
    # .jsonl: one document per line, a string or an object ("text", chat "messages"/"conversations", else its strings)
    if not path.endswith(".jsonl"):
        return [Path(path).read_text(encoding="utf-8")]
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                turns = row.get("messages") or row.get("conversations")
                if isinstance(row.get("text"), str):
                    row = row["text"]
                elif isinstance(turns, list):
                    row = "\n".join(str(m.get("content") or m.get("value") or "") for m in turns)
                else:
                    row = "\n".join(v for v in row.values() if isinstance(v, str))
            docs.append(row)
    return docs


def tokenize(exe: str, model: str, docs: list[str]) -> list[int]:
    if not docs:
        return []
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", suffix=".txt", delete=False) as f:
        f.write("\n".join(docs))
    try:
        res = subprocess.run([exe, "-m", model, "-f", f.name, "--ids", "--no-bos", "--no-escape", "--log-disable"],
                             capture_output=True, text=True, check=True)
    finally:
        os.unlink(f.name)
    m = re.search(r"\[([0-9,\s]*)\]", res.stdout)
    if m is None:
        raise RuntimeError(f"no token ids in the output of {exe}")
    return [int(x) for x in m.group(1).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True, help="a GGUF of the target model (only the vocabulary is read)")
    parser.add_argument("--tokenize", required=True, help="path to llama-tokenize")
    parser.add_argument("--text-file", action="append", required=True,
                        help="plain text (one document) or .jsonl (one document per line), may be repeated")
    parser.add_argument("--num-tokens", type=int, default=32768, help="number of ids in the list (default: 32768)")
    parser.add_argument("--id-floor", type=int, default=0, help="always keep the ids below this value (default: 0)")
    parser.add_argument("--holdout", type=float, default=0.05, help="fraction of each file left out of the counts, every 1/holdout-th document (default: 0.05)")
    parser.add_argument("-o", "--output", required=True, help="output JSON file, ids sorted")
    args = parser.parse_args()

    reader = gguf.GGUFReader(args.model, "r")
    types = [int(t) for t in reader.fields["tokenizer.ggml.token_type"].contents()]
    n_vocab = len(types)
    if not 0 < args.num_tokens < n_vocab:
        parser.error(f"--num-tokens must be in 1..{n_vocab - 1}")

    counts: Counter[int] = Counter()
    held: dict[str, list[int]] = {}
    n_docs = 0
    # every k-th document is held out, so a file sorted by language or source is sampled evenly
    k = round(1 / args.holdout) if args.holdout > 0 else 0
    for path in args.text_file:
        docs = read_docs(path)
        is_held = [k > 0 and i % k == k - 1 for i in range(len(docs))]
        train = tokenize(args.tokenize, args.model, [d for d, h in zip(docs, is_held) if not h])
        held[path] = tokenize(args.tokenize, args.model, [d for d, h in zip(docs, is_held) if h])
        counts.update(train)
        n_docs += len(docs) - sum(is_held)
        print(f"{path}: {len(docs) - sum(is_held)} documents, {len(train)} tokens, {sum(is_held)} documents held out")

    keep = set(range(min(args.id_floor, n_vocab)))
    keep.update(i for i, t in enumerate(types) if t in KEEP_TYPES)
    n_fixed = len(keep)
    if n_fixed > args.num_tokens:
        parser.error(f"{n_fixed} ids are always kept, more than --num-tokens {args.num_tokens}")
    for i in sorted(counts, key=lambda i: (-counts[i], i)):
        if len(keep) >= args.num_tokens:
            break
        keep.add(i)
    n_counted = len(keep) - n_fixed
    for i in range(n_vocab):
        if len(keep) >= args.num_tokens:
            break
        keep.add(i)
    ids = sorted(keep)

    with open(args.output, "w") as f:
        json.dump(ids, f)

    print(f"wrote {len(ids)} of {n_vocab} ids to {args.output}: {n_fixed} control/user-defined/floor, {n_counted} by frequency "
          f"({len(counts)} distinct ids in {sum(counts.values())} tokens of {n_docs} documents), {len(ids) - n_fixed - n_counted} lowest unused")
    for path, toks in held.items():
        if toks:
            print(f"held-out coverage of {path}: {100 * sum(t in keep for t in toks) / len(toks):.2f}% of {len(toks)} tokens")


if __name__ == "__main__":
    main()
