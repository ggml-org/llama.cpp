"""Performance and stability benchmarks for the self-contained laya runtime.

Measures, with ``llama-laya-cli``'s in-process ``--bench`` mode:

  * single-question forward latency (mean / median / min / max / std)
  * multi-question forward latency (one batch, K questions)
  * the batching speed-up: K questions in one forward vs K separate forwards
  * thread scaling (``-t 1`` vs ``-t 8``)
  * determinism: in-process repeated runs and cross-process repeated runs

Usage:
    python3 tests/laya/bench.py <build-bin> [--models m1,m2] [--runs N]

Defaults to the four standard artifacts in the repo root.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden")
TMP = os.path.join(ROOT, "build")

DEFAULT_MODELS = ["laya-f16.gguf", "laya-q4_k_m.gguf", "laya-q5_k_m.gguf", "laya-q8_0.gguf"]


def load_case(name):
    return json.load(open(os.path.join(GOLDEN, name + ".json"), encoding="utf-8"))


def write_input(cases, path, n_questions):
    """Build an input from `cases`; repeat questions until n_questions or exhausted."""
    pool = []
    state = None
    for name in cases:
        c = load_case(name)
        state = state or c["state"]
        pool.append(c["questions"]["q1"])
    while len(pool) < n_questions:
        pool.append(dict(pool[len(pool) % len(cases)]))
    questions = {"q%d" % (i + 1): q for i, q in enumerate(pool[:n_questions])}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"state": state, "questions": questions}, f, ensure_ascii=False)
    return path


def run(cli, model, inp, threads=8, bench=0):
    cmd = [cli, "-m", model, "-f", inp, "-t", str(threads)]
    if bench:
        cmd += ["-b", str(bench)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("cli failed: %s" % r.stderr[-300:])
    return json.loads(r.stdout), r.stderr


def fmt(x):
    return "%.2f" % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cli")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()

    models = [m if os.path.isabs(m) else os.path.join(ROOT, m) for m in args.models.split(",")]
    single_in = write_input(["choice_single_zh"], os.path.join(TMP, "laya_bench_single.json"), 1)
    batch_in = write_input(["choice_single_zh", "choice_multi_zh", "score_zh", "noul_zh"],
                           os.path.join(TMP, "laya_bench_batch.json"), 4)

    n_single = json.load(open(single_in))["questions"]
    n_batch = json.load(open(batch_in))["questions"]
    print("inputs: single=%d question, batch=%d questions" % (len(n_single), len(n_batch)))

    print("\n== forward latency (threads=8, in-process) ==")
    print("%-14s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %s" % (
        "model", "input", "tokens", "mean", "median", "min", "max", "std", "determ."))

    results = {}
    for m in models:
        name = os.path.splitext(os.path.basename(m))[0]
        results[name] = {}
        for label, inp in (("single", single_in), ("batch4", batch_in)):
            t0 = time.time()
            out, stderr = run(args.cli, m, inp, threads=8, bench=args.runs)
            load_ms = (time.time() - t0) * 1000.0 - out["bench"]["mean_ms"] * args.runs
            b = out["bench"]
            results[name][label] = b
            results[name][label + "_load_ms"] = load_ms
            print("%-14s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %s" % (
                name, label, b["tokens"], fmt(b["mean_ms"]), fmt(b["median_ms"]),
                fmt(b["min_ms"]), fmt(b["max_ms"]), fmt(b["std_ms"]),
                "yes" if b["deterministic"] else "NO"))

    print("\n== batching: 4 questions (batch4 vs 4x single) ==")
    print("%-14s %-12s %-12s %-10s %-10s" % ("model", "batch4(ms)", "4x single(ms)", "speedup", "load(ms)"))
    for name in results:
        s = results[name]["single"]["mean_ms"]
        b4 = results[name]["batch4"]["mean_ms"]
        print("%-14s %-12s %-12s %-10s %-10s" % (
            name, fmt(b4), fmt(4 * s), "%.2fx" % ((4 * s) / b4 if b4 else 0),
            fmt(results[name]["single_load_ms"])))

    print("\n== thread scaling (single question) ==")
    print("%-14s %-10s %-10s %s" % ("model", "t=1(ms)", "t=8(ms)", "speedup"))
    for name, m in zip(results, models):
        o1, _ = run(args.cli, m, single_in, threads=1, bench=args.runs)
        o8, _ = run(args.cli, m, single_in, threads=8, bench=args.runs)
        s1 = o1["bench"]["mean_ms"]
        s8 = o8["bench"]["mean_ms"]
        print("%-14s %-10s %-10s %s" % (name, fmt(s1), fmt(s8), "%.2fx" % (s1 / s8 if s8 else 0)))

    print("\n== stability: %d serial in-process runs + cross-process ==" % args.runs)
    model = models[0]
    name = os.path.splitext(os.path.basename(model))[0]
    o1, _ = run(args.cli, model, batch_in, threads=8)
    o2, _ = run(args.cli, model, batch_in, threads=8)
    same = json.dumps(o1["answers"], sort_keys=True) == json.dumps(o2["answers"], sort_keys=True)
    inproc = results[name]["batch4"]["deterministic"]
    print("in-process deterministic: %s; cross-process answers identical: %s" % (
        "yes" if inproc else "NO", "yes" if same else "NO"))
    if not same:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
