#!/usr/bin/env python3
"""Score llama-self-spec-bias jsonl output.

Quality is measured on the last answer of each record. Stability is measured
across the answers within a record. References live in their own file and are
joined by id, never by line order.

  python3 score.py --hyp draft_02.jsonl --refs refs.jsonl
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from sacrebleu.tokenizers.tokenizer_spm import TokenizerSPM
except ImportError:
    TokenizerSPM = None

try:
    import sacrebleu
except ImportError:
    sacrebleu = None


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_refs(path):
    """refs.jsonl of {"id","ref"}, or a plain text file joined by line order."""
    p = Path(path)
    if p.suffix == ".jsonl":
        return {r["id"]: r["ref"] for r in load_jsonl(p)}
    return {i: l for i, l in enumerate(p.read_text(encoding="utf-8").splitlines())}


def validate(recs):
    """Fail loudly. A schema only helps if violations stop the run."""
    errs = []
    seen = set()
    for i, r in enumerate(recs):
        for k in ("id", "source", "stream_ins", "stream_outs"):
            if k not in r:
                errs.append(f"record {i}: missing '{k}'")
        if errs:
            break
        if r["id"] in seen:
            errs.append(f"duplicate id {r['id']!r}")
        seen.add(r["id"])
        if len(r["stream_ins"]) != len(r["stream_outs"]):
            errs.append(f"{r['id']}: {len(r['stream_ins'])} inputs vs {len(r['stream_outs'])} outputs")
        if not r["stream_outs"]:
            errs.append(f"{r['id']}: empty stream_outs")
    if errs:
        sys.exit("invalid hypothesis file:\n  " + "\n  ".join(errs[:20]))


def one_line(text):
    """Metric files are line oriented, so a newline inside a hypothesis would
    shift every later sentence against its source and reference."""
    return " ".join(text.split())


def normalized_erasure(outs, tok, mask_k=0):
    """Tokens retracted between consecutive answers, over the final length.

    mask_k models a reader that is shown all but the last k tokens of each
    partial answer. Rewriting a token nobody saw is not erasure. The final
    answer is compared whole, because that one is shown in full.
    """
    if len(outs) < 2:
        return 0.0
    n = 0.0
    seq_new = tok(outs[0])
    for i in range(len(outs) - 1):
        seq_old, seq_new = tok(outs[i]), tok(outs[i + 1])
        if mask_k > 0:
            seq_old = seq_old[:-mask_k]
            if i != len(outs) - 2:
                seq_new = seq_new[:-mask_k]
        lcp = 0
        while lcp < min(len(seq_old), len(seq_new)) and seq_old[lcp] == seq_new[lcp]:
            lcp += 1
        n += len(seq_old) - lcp
    return n / len(seq_new) if seq_new else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--refs")
    ap.add_argument("--out", help="write a results json here")
    ap.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--display-mask-k", default="0",
                    help="erasure ignores the last k tokens of each partial answer, "
                         "modelling a reader shown all but the last k. Takes a comma "
                         "separated list, since this costs nothing to vary")
    ap.add_argument("--bleu-tokenize", default=None,
                    help="sacrebleu tokenizer; zh and ja need this set explicitly")
    args = ap.parse_args()

    recs = load_jsonl(args.hyp)
    if not recs:
        sys.exit("no records")
    validate(recs)

    res = {
        "hyp": args.hyp,
        "records": len(recs),
        "requests": sum(len(r["stream_outs"]) for r in recs),
        "segmentation": recs[0].get("segmentation"),
    }
    print(f"records: {res['records']}  requests: {res['requests']}  segmentation: {res['segmentation']}")

    # stability, needs the whole history
    if TokenizerSPM is None:
        print("sacrebleu missing, skipping erasure")
    else:
        # One multilingual spm model for every language, on purpose. Erasure
        # compares a hypothesis against its own earlier versions, so a single
        # tokenizer keeps the numbers comparable across language pairs. Only
        # BLEU needs a per language tokenizer.
        #
        # Strip the line, then drop the leading spm word marker, as mt
        # retrieve_streams.py does, or the numbers will not line up with it.
        spm = TokenizerSPM()
        tok = lambda s: spm(s.strip()).lstrip("\u2581").split()
        # varying the display mask needs no re-decoding, so report the curve
        ks = [int(x) for x in str(args.display_mask_k).split(",") if x.strip() != ""]
        by_k = {}
        for k in ks:
            es = [normalized_erasure(r["stream_outs"], tok, k) for r in recs]
            by_k[k] = sum(es) / len(es)

        res["normalized_erasure"] = by_k[ks[0]]
        res["display_mask_k"] = ks[0]
        res["erasure_by_display_mask_k"] = by_k

        print(f"System Normalized Erasure: {by_k[ks[0]]:.2f} (display mask k={ks[0]})")
        if len(ks) > 1:
            print("  erasure by display mask k: " + "  ".join(f"{k}={v:.2f}" for k, v in by_k.items()))

    if not args.refs:
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        return

    refs = load_refs(args.refs)
    by_line = not str(args.refs).endswith(".jsonl")

    srcs, hyps, rfs, missing = [], [], [], []
    for i, r in enumerate(recs):
        key = i if by_line else r["id"]
        if key not in refs:
            missing.append(r["id"])
            continue
        srcs.append(one_line(r["source"]))
        hyps.append(one_line(r["stream_outs"][-1]))
        rfs.append(one_line(refs[key]))

    if missing:
        sys.exit(f"{len(missing)} ids have no reference, first few: {missing[:5]}")

    if sacrebleu is not None:
        # BLEU is not comparable across tokenizers, so record the one actually
        # used. ja-mecab and ko-mecab need extras that may not be installed.
        want = args.bleu_tokenize
        try:
            bleu = sacrebleu.corpus_bleu(hyps, [rfs], **({"tokenize": want} if want else {}))
            used = want or "default"
        except Exception as e:
            print(f"BLEU: {want} unavailable ({str(e).splitlines()[0][:60]}), falling back to char")
            bleu = sacrebleu.corpus_bleu(hyps, [rfs], tokenize="char")
            used = "char"

        res["bleu"] = bleu.score
        res["bleu_tokenize"] = used
        res["chrf"] = sacrebleu.corpus_chrf(hyps, [rfs]).score
        print(f"BLEU: {res['bleu']:.2f} (tok={used})  chrF: {res['chrf']:.2f}")

    if not args.no_comet:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, rows in (("src", srcs), ("hyp", hyps), ("ref", rfs)):
                (d / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
                n_lines = len((d / name).read_text(encoding="utf-8").splitlines())
                if n_lines != len(rows):
                    sys.exit("%s: wrote %d lines for %d rows" % (name, n_lines, len(rows)))
            try:
                out = subprocess.run(
                    ["comet-score", "-s", str(d / "src"), "-t", str(d / "hyp"),
                     "-r", str(d / "ref"), "--model", args.comet_model, "--only_system"],
                    capture_output=True, text=True,
                )
            except FileNotFoundError:
                print("COMET: comet-score not on PATH, skipping. "
                      "pip install -r requirements.txt to get it")
                out = None
            sys.stderr.write(out.stderr if out else "")
            for line in (out.stdout.splitlines() if out else []):
                if "score:" in line:
                    print("COMET:", line.split("score:")[-1].strip())
                    res["comet"] = float(line.split("score:")[-1])

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
