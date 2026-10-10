#!/usr/bin/env python3
import json, itertools
from pathlib import Path
D = Path("/mnt/nvme1/oneapi-ab/matrix-1006/wgcheck/raw")
def load(tag, ctx):
    p = D / f"{tag}.{ctx}.json"
    return json.loads(p.read_text()) if p.exists() else None
def cmp(a, b):
    ta, tb = [x["t"] for x in a["toks"]], [x["t"] for x in b["toks"]]
    n = min(len(ta), len(tb)); same = 0
    for i in range(n):
        if ta[i] != tb[i]: break
        same += 1
    dlp = [abs(a["toks"][i]["lp"] - b["toks"][i]["lp"]) for i in range(same)]
    div = None
    if same < n:
        x = a["toks"][same]; margin = x["top"][0][1] - x["top"][1][1] if len(x["top"]) > 1 else None
        div = {"pos": same, "a": ta[same], "b": tb[same], "margin_a": margin}
    return {"len": (len(ta), len(tb)), "identical_prefix": same, "max_dlogprob": max(dlp) if dlp else 0.0, "mean_dlogprob": sum(dlp)/len(dlp) if dlp else 0.0, "first_divergence": div, "text_equal": a["text"] == b["text"]}
pairs = [("wg16.l0", "wg16.l3", "wg16 vs wg16 (launch-to-launch)"), ("wg32.l1", "wg32.l2", "wg32 vs wg32"), ("wg16.l0", "wg32.l1", "wg16 vs wg32"), ("wg16.l3", "wg32.l2", "wg16 vs wg32 (other pair)")]
for ctx in ("d28k", "d60k"):
    for ta, tb, label in pairs:
        a, b = load(ta, ctx), load(tb, ctx)
        if a and b: print(ctx, label, json.dumps(cmp(a, b)))
        else: print(ctx, label, "missing")
for ctx in ("d28k", "d60k"):
    for t in ("wg16.l0", "wg32.l1", "wg32.l2", "wg16.l3"):
        a = load(t, ctx)
        if a: print(ctx, t, "prompt_n", (a.get("usage") or {}).get("prompt_tokens"), "tg", round(a["timings"].get("predicted_per_second", 0), 2), "|", a["text"].replace("\n", " / ")[:110])
