"""End-to-end precision regression for the self-contained laya runtime.

Runs ``llama-laya-cli`` on every golden case for each model variant
(F16 + the standard quantization tiers) and compares the outputs against:

  * the PyTorch reference (``golden/<case>.json``)
  * the F16 GGUF, as an implementation-consistent baseline

Reported per model variant: raw scorer logits (max / mean absolute
deviation), answer probabilities, the decision fields (choice / score /
noul) and the action-head probability. The F16 model is also reported
against the PyTorch reference so the pure quantization error can be read
off by subtracting the F16 row from the quant rows.

Usage:
    python3 tests/laya/verify_precision.py <build-bin> [model.gguf ...]

With no model arguments it uses the four standard artifacts in the repo
root: laya-f16.gguf, laya-q4_k_m.gguf, laya-q5_k_m.gguf, laya-q8_0.gguf.
"""

import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden")

DEFAULT_MODELS = ["laya-f16.gguf", "laya-q4_k_m.gguf", "laya-q5_k_m.gguf", "laya-q8_0.gguf"]


def run_cli(cli, model, case):
    """Run the CLI on one golden case; return the parsed output."""
    g = json.load(open(os.path.join(GOLDEN, case + ".json"), encoding="utf-8"))
    inp = {"state": g["state"], "questions": g["questions"]}
    inp_path = os.path.join(ROOT, "build", "laya_prec_input.json")
    with open(inp_path, "w", encoding="utf-8") as f:
        json.dump(inp, f, ensure_ascii=False)
    r = subprocess.run([cli, "-m", model, "-f", inp_path, "-t", "8"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("cli failed on %s/%s: %s" % (model, case, r.stderr[-300:]))
    return g, json.loads(r.stdout)


def decisions(out, qid):
    """Decision fields + probabilities for one question, normalized."""
    ans = out["answers"][qid]
    pq = out["per_question"][qid]
    d = {
        "probs": None,
        "choice": None,
        "score": None,
        "noul": None,
        "act": pq.get("act_probability"),
        "logits": pq["raw_logits"],
    }
    if pq["qtype"] == "choice":
        d["probs"] = list(ans["probabilities"].values())
        d["choice"] = ans["choice"]
    elif pq["qtype"] == "score":
        d["probs"] = list(ans["probabilities"].values())
        d["score"] = ans["score"]
    else:
        d["noul"] = ans["noul"]
    return d


def golden_decisions(g):
    """Same normalization for a golden fixture."""
    ans = g["answers"]["q1"]
    pq = g["per_question"]["q1"]
    d = {
        "probs": None,
        "choice": None,
        "score": None,
        "noul": None,
        "act": pq["act_probability"],
        "logits": pq["raw_logits"],
    }
    if pq["qtype"] == "choice":
        d["probs"] = list(ans["probabilities"].values())
        d["choice"] = ans["choice"]
    elif pq["qtype"] == "score":
        d["probs"] = list(ans["probabilities"].values())
        d["score"] = ans["score"]
    else:
        d["noul"] = ans["noul"]
    return d


def compare(a, b):
    """Deviations of reference `a` from tested `b`."""
    logits = max(abs(x - y) for x, y in zip(a["logits"], b["logits"])) if a["logits"] else 0.0
    mean_logits = (sum(abs(x - y) for x, y in zip(a["logits"], b["logits"])) / len(a["logits"])
                   if a["logits"] else 0.0)
    probs = max(abs(x - y) for x, y in zip(a["probs"], b["probs"])) if a["probs"] else 0.0
    choice_ok = (a["choice"] == b["choice"]) if a["choice"] is not None else None
    score_d = abs(a["score"] - b["score"]) if a["score"] is not None else None
    noul_d = abs(a["noul"] - b["noul"]) if a["noul"] is not None else None
    act_d = abs((a["act"] or 0.0) - (b["act"] or 0.0))
    return {
        "logits_max": logits,
        "logits_mean": mean_logits,
        "probs_max": probs,
        "choice_ok": choice_ok,
        "score_delta": score_d,
        "noul_delta": noul_d,
        "act_delta": act_d,
    }


def fmt(x, nd=4):
    return "-" if x is None else ("%." + str(nd) + "f") % x


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cli = sys.argv[1]
    models = sys.argv[2:] or DEFAULT_MODELS
    models = [os.path.join(ROOT, m) if not os.path.isabs(m) else m for m in models]

    manifest = json.load(open(os.path.join(GOLDEN, "manifest.json")))
    cases = list(manifest.keys())

    # outputs[model][case] -> normalized decision dict
    outputs = {}
    for m in models:
        name = os.path.splitext(os.path.basename(m))[0]
        outputs[name] = {}
        for case in cases:
            _g, out = run_cli(cli, m, case)
            outputs[name][case] = decisions(out, "q1")

    golden = {case: golden_decisions(json.load(open(os.path.join(GOLDEN, manifest[case]), encoding="utf-8")))
              for case in cases}

    # ---- per model vs PyTorch golden -----------------------------------
    print("== vs PyTorch golden ==")
    header = "%-14s %-10s %-10s %-10s %-8s %-10s %-10s %s" % (
        "model", "logit_max", "logit_mean", "prob_max", "choice", "score_d", "noul_d", "act_d")
    print(header)
    choice_ok_by_model = {}
    for name in outputs:
        agg = {"logits_max": 0.0, "logits_mean": 0.0, "probs_max": 0.0,
               "score_delta": 0.0, "noul_delta": 0.0, "act_delta": 0.0, "choice_ok": True}
        for case in cases:
            c = compare(golden[case], outputs[name][case])
            agg["logits_max"] = max(agg["logits_max"], c["logits_max"])
            agg["logits_mean"] = max(agg["logits_mean"], c["logits_mean"])
            agg["probs_max"] = max(agg["probs_max"], c["probs_max"])
            agg["act_delta"] = max(agg["act_delta"], c["act_delta"])
            if c["choice_ok"] is False:
                agg["choice_ok"] = False
            if c["score_delta"] is not None:
                agg["score_delta"] = max(agg["score_delta"], c["score_delta"])
            if c["noul_delta"] is not None:
                agg["noul_delta"] = max(agg["noul_delta"], c["noul_delta"])
        choice_ok_by_model[name] = agg["choice_ok"]
        print("%-14s %-10s %-10s %-10s %-8s %-10s %-10s %s" % (
            name, fmt(agg["logits_max"], 3), fmt(agg["logits_mean"], 3), fmt(agg["probs_max"]),
            "OK" if agg["choice_ok"] else "MISMATCH",
            fmt(agg["score_delta"]), fmt(agg["noul_delta"]), fmt(agg["act_delta"])))

    # ---- per case detail for the F16 baseline --------------------------
    print("\n== per-case (F16) ==")
    print("%-20s %-11s %-11s %-8s %s" % ("case", "golden", "f16", "atol", "choice"))
    f16 = "laya-f16" if "laya-f16" in outputs else list(outputs)[0]
    for case in cases:
        a, b = golden[case], outputs[f16][case]
        c = compare(a, b)
        print("%-20s %-11s %-11s %-8s %s" % (
            case,
            "[" + ",".join("%.2f" % v for v in a["logits"]) + "]",
            "[" + ",".join("%.2f" % v for v in b["logits"]) + "]",
            fmt(c["logits_max"], 3),
            "OK" if c["choice_ok"] is not False else "MISMATCH"))

    # ---- quant vs F16 (implementation-consistent baseline) -------------
    if f16 in outputs:
        print("\n== quantization error vs F16 (same runtime) ==")
        print("%-14s %-10s %-10s %-8s %s" % ("model", "logit_max", "prob_max", "choice", "score_d/noul_d"))
        for name in outputs:
            if name == f16:
                continue
            agg = {"logits_max": 0.0, "probs_max": 0.0, "choice_ok": True,
                   "score_delta": 0.0, "noul_delta": 0.0}
            for case in cases:
                c = compare(outputs[f16][case], outputs[name][case])
                agg["logits_max"] = max(agg["logits_max"], c["logits_max"])
                agg["probs_max"] = max(agg["probs_max"], c["probs_max"])
                if c["choice_ok"] is False:
                    agg["choice_ok"] = False
                if c["score_delta"] is not None:
                    agg["score_delta"] = max(agg["score_delta"], c["score_delta"])
                if c["noul_delta"] is not None:
                    agg["noul_delta"] = max(agg["noul_delta"], c["noul_delta"])
            print("%-14s %-10s %-10s %-8s %s / %s" % (
                name, fmt(agg["logits_max"], 4), fmt(agg["probs_max"]),
                "OK" if agg["choice_ok"] else "MISMATCH",
                fmt(agg["score_delta"]), fmt(agg["noul_delta"])))

    f16_ok = choice_ok_by_model.get(f16, False)
    all_ok = all(choice_ok_by_model.values())
    print("\nchoice/argmax agreement with PyTorch golden: F16=%s, all tiers=%s" % (
        "PASS" if f16_ok else "FAIL", "PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
