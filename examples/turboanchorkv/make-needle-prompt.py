#!/usr/bin/env python3

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--depth", type=float, default=0.5)
    parser.add_argument("--needle", default="TURBO-ANCHOR-7391")
    args = parser.parse_args()

    if not 0.0 <= args.depth <= 1.0:
        raise ValueError("depth must be between 0 and 1")

    haystack = args.source.read_text(encoding="utf-8")
    offset = int(len(haystack) * args.depth)
    needle = f"\n\nImportant memory: the secret code is {args.needle}.\n\n"
    body = haystack[:offset] + needle + haystack[offset:]
    prompt = (
        "<start_of_turn>user\n"
        "Read the following text and remember the secret code.\n\n"
        f"{body}\n\n"
        "What is the secret code? Answer with only the code.\n"
        "<end_of_turn>\n<start_of_turn>model\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prompt, encoding="utf-8")


if __name__ == "__main__":
    main()
