#!/usr/bin/env python3
"""Turn plain source text into a stream.jsonl for llama-self-spec-bias.

Segmentation is data, not a decoder flag. Add a new policy here; the decoder
never has to know about it.

  python3 segment.py --input flores200.devtest.en --policy interval --n 3 \
      --output stream.jsonl
"""

import argparse
import json
from pathlib import Path


def seg_interval(words, n):
    """Emit a prefix every n words, always ending with the full sentence."""
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        if (i + 1) % n == 0:
            out.append(" ".join(cur))
    full = " ".join(cur)
    if not out or out[-1] != full:
        out.append(full)
    return out


def seg_whole(words, _n):
    return [" ".join(words)]


POLICIES = {"interval": seg_interval, "whole": seg_whole}

# policies that do not use --n, so it is left out of the recorded segmentation
POLICIES_WITHOUT_N = {"whole"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--policy", default="interval", choices=sorted(POLICIES))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--id-prefix", default=None,
                    help="defaults to the input file name")
    args = ap.parse_args()

    stem = args.id_prefix or Path(args.input).name
    fn = POLICIES[args.policy]

    n_rec = n_req = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for i, line in enumerate(Path(args.input).read_text(encoding="utf-8").splitlines()):
            words = line.split()
            if not words:
                continue
            ins = fn(words, args.n)
            seg = {"policy": args.policy}
            if args.policy not in POLICIES_WITHOUT_N:
                seg["n"] = args.n

            rec = {
                "id": f"{stem}:{i}",
                "source": line,
                "segmentation": seg,
                "stream_ins": ins,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_rec += 1
            n_req += len(ins)

    print(f"wrote {n_rec} records, {n_req} requests -> {args.output}")


if __name__ == "__main__":
    main()
