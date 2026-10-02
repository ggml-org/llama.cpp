"""Compare llama-laya-cli output against the golden fixtures.

Runs the self-contained inference CLI on every golden case and reports the
per-case deviation of the raw scorer logits at the marker positions.

Usage:
    PYTHONPATH=gguf-py python3 tests/laya/verify_inference.py <build-bin> <laya.gguf>
"""

import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    cli = sys.argv[1]
    model = sys.argv[2]

    manifest = json.load(open(os.path.join(GOLDEN, "manifest.json")))
    print("%-18s %-8s %-8s %-16s %-16s %s" % (
        "case", "ids", "markers", "golden", "mine", "max_atol"))
    ok = True
    for name, fn in manifest.items():
        g = json.load(open(os.path.join(GOLDEN, fn)))
        inp = {"state": g["state"], "questions": g["questions"]}
        inp_path = os.path.join(ROOT, "build", "laya_input.json")
        with open(inp_path, "w", encoding="utf-8") as f:
            json.dump(inp, f, ensure_ascii=False)
        r = subprocess.run(
            [cli, "-m", model, "-f", inp_path, "-t", "8"],
            capture_output=True, text=True)
        if r.returncode != 0:
            print("%-18s FAILED: %s" % (name, r.stderr[-200:]))
            ok = False
            continue
        o = json.loads(r.stdout)
        gq = g["per_question"]["q1"]
        oq = o["per_question"]["q1"]
        ids_ok = gq["input_ids"] == oq["input_ids"]
        mk_ok = gq["marker_pos"] == oq["marker_pos"]
        atol = max(abs(a - b) for a, b in zip(gq["raw_logits"], oq["raw_logits"]))
        ok = ok and ids_ok and mk_ok
        print("%-18s %-8s %-8s %-16s %-16s %.4f" % (
            name, ids_ok, mk_ok,
            str([round(x, 3) for x in gq["raw_logits"]]),
            str([round(x, 3) for x in oq["raw_logits"]]),
            atol))
    print("\nsequence construction (ids/markers) all match:", ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
