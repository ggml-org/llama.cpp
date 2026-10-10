#!/usr/bin/env python3
"""Per-run and per-arm complete matrix with effective env, sysfs state and bench args -> COMPLETE-MATRIX.{md,csv}."""
import csv, json, re, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path("/mnt/nvme1/oneapi-ab/matrix-1006")
BASE_ENV = {"ONEAPI_DEVICE_SELECTOR": "level_zero:0", "GGML_SYCL_ENABLE_GRAPH": "1", "GGML_SYCL_GRAPH_EVICTION_TIMEOUT": "300",
            "GGML_SYCL_XE_COPY_ENGINE_DEFAULT": "0", "GGML_SYCL_FA_LARGE_GRF": "1"}
B8 = "-m Llama-3.1-8B-Q4_K_M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 64 -d 0,8192 -r 3 -t 12"
BORN = "-m Ornith-1.5-35B-Q4_K_M -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 -p 512 -n 64 -d 0,8192 -r 3 -t 12"
ORN_ARGS = {"prod": "--moe-cache off --load-mode none", "mcauto_none": "--moe-cache auto --load-mode none",
            "mcauto_auto": "--moe-cache auto --load-mode auto", "mcoff_auto": "--moe-cache off --load-mode auto",
            "mcsoft_none": "--moe-cache soft --load-mode none", "mcon_none": "--moe-cache on --load-mode none",
            "mc4096_none": "--moe-cache 4096 --load-mode none"}
FREQ = {"def": (600, 2400, "base"), "pin2400": (2400, 2400, "base"), "pin2000": (2000, 2000, "base"), "pin1500": (1500, 1500, "base"),
        "powersave": (600, 2400, "power_saving"), "floor1500": (1500, 2400, "base")}
TESTS = ["pp512", "tg64", "pp512 @ d8192", "tg64 @ d8192"]

# arm env deltas, from index.tsv column 3 (what the harness passed on top of BASE_ENV)
idx = {}
for f in ("index-part1.tsv", "index.tsv"):
    for line in (ROOT / f).read_text().splitlines():
        c = line.split("\t")
        if len(c) >= 8 and c[0] in ("ccs", "envs", "il", "freq", "ornith"):
            idx[(c[0], c[1])] = c

def metrics(p: Path):
    out = {}
    for line in p.read_text().splitlines():
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) > 2 and c[-2] in TESTS:
            m = re.match(r"([\d.]+)\s*±\s*([\d.]+)", c[-1])
            if m: out[c[-2]] = (float(m[1]), float(m[2]))
    return out

def freqstat(p: Path):
    v = [int(m[1]) for l in p.read_text().splitlines() if (m := re.search(r"act=(\d+)", l))] if p.exists() else []
    return f"{st.mean(v):.0f}" if v else ""

rows = []
for sec in ("ccs", "envs", "il", "freq", "ornith"):
    for p in sorted((ROOT / sec).glob("*.md")):
        tag = p.stem; arm, rnd = tag.rsplit(".r", 1)
        env = dict(BASE_ENV); ccs, fmin, fmax, prof, args, model = 1, 600, 2400, "base", "", "8B"
        extra = idx.get((sec, tag), [None] * 8)[2] or ""
        for kv in extra.split():
            k, v = kv.split("=", 1); env[k] = v
        if sec == "ccs": ccs = int(arm[3:])
        if sec == "il": ccs = int(re.match(r"ccs(\d)", arm)[1])
        if sec == "freq": fmin, fmax, prof = FREQ[arm]
        if sec == "ornith": model = "Ornith-Q4_K_M"; args = ORN_ARGS.get(arm, ORN_ARGS["prod"])
        m = metrics(p); c = idx.get((sec, tag), [""] * 8)
        rows.append({"section": sec, "arm": arm, "round": rnd, "model": model, "ccs_mode": ccs, "freq_min": fmin, "freq_max": fmax,
                     "power_profile": prof, "timeslice_us": 1000, "env_extra": extra or "(none)",
                     "env_full": " ".join(f"{k}={v}" for k, v in env.items()),
                     "bench_args": (B8 if model == "8B" else BORN) + (" " + args if args else ""),
                     **{t: f"{m[t][0]:.2f}±{m[t][1]:.2f}" if t in m else "" for t in TESTS},
                     "cpu_idle": c[3].replace("idle=", ""), "load": c[4].replace("load=", ""), "rc": c[6].replace("rc=", ""),
                     "act_freq_mean": freqstat(p.with_suffix(".freq"))})
# two concurrent processes
for p in sorted((ROOT / "two").glob("*.a.md")):
    tag = p.name[:-5]; m = re.match(r"ccs(\d)\.ts(\d+)\.r(\d)", tag)
    a = metrics(p); b = metrics(p.with_name(tag + ".b.md"))
    rows.append({"section": "two", "arm": f"ccs{m[1]}.ts{m[2]}", "round": m[3], "model": "8B x2 concurrent", "ccs_mode": int(m[1]),
                 "freq_min": 600, "freq_max": 2400, "power_profile": "base", "timeslice_us": int(m[2]), "env_extra": "(none)",
                 "env_full": " ".join(f"{k}={v}" for k, v in BASE_ENV.items()),
                 "bench_args": B8.replace("-d 0,8192", "-d 0").replace("-t 12", "-t 6") + " (two processes)",
                 "pp512": f"A {a['pp512'][0]:.1f} + B {b['pp512'][0]:.1f} = {a['pp512'][0]+b['pp512'][0]:.1f}",
                 "tg64": f"A {a['tg64'][0]:.2f} + B {b['tg64'][0]:.2f} = {a['tg64'][0]+b['tg64'][0]:.2f}"})
# n-gram server
srv = {"none": "--spec-type none --moe-cache off --load-mode none", "ngmod": "--spec-type ngram-mod --moe-cache off --load-mode none",
       "ngmod_small": "--spec-type ngram-mod --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32 --moe-cache off --load-mode none",
       "ngsimple": "--spec-type ngram-simple --moe-cache off --load-mode none", "ngmapk": "--spec-type ngram-map-k --moe-cache off --load-mode none",
       "ngmapk4v": "--spec-type ngram-map-k4v --moe-cache off --load-mode none", "ngcache": "--spec-type ngram-cache --moe-cache off --load-mode none",
       "none_mcauto": "--spec-type none --moe-cache auto --load-mode none", "ngmod_mcauto": "--spec-type ngram-mod --moe-cache auto --load-mode none",
       "ngmod_lmauto": "--spec-type ngram-mod --moe-cache off --load-mode auto", "ngmod_prefetch4": "--spec-type ngram-mod --moe-cache off --load-mode none --prefetch-experts-slots 4"}
ng = defaultdict(list)
for l in (ROOT / "ngram/results.jsonl").read_text().splitlines():
    j = json.loads(l); arm, r = j["cfg"].rsplit(".r", 1); ng[(arm, r, j["prompt"])].append(j)
for (arm, r, pr), v in ng.items():
    rows.append({"section": "ngram", "arm": arm, "round": r, "model": "Ornith-Q4_K_M server", "ccs_mode": 1, "freq_min": 600, "freq_max": 2400,
                 "power_profile": "base", "timeslice_us": 1000, "env_extra": "(none)", "env_full": " ".join(f"{k}={v}" for k, v in BASE_ENV.items()),
                 "bench_args": f"llama-server --ctx-size 131072 --fit on --fit-target 1024 --flash-attn on -ctk q8_0 -ctv q8_0 {srv.get(arm,'')} | prompt={pr}",
                 "tg64": f"{st.mean(x['tg'] for x in v):.2f} tg t/s (n=3)", "pp512": f"{st.mean(x['pp'] for x in v):.0f} pp t/s",
                 "rc": f"draft {sum(x['draft_acc'] or 0 for x in v)}/{sum(x['draft_n'] or 0 for x in v)}"})

cols = ["section", "arm", "round", "model", "ccs_mode", "freq_min", "freq_max", "power_profile", "timeslice_us", "env_extra", "env_full",
        "bench_args", *TESTS, "cpu_idle", "load", "rc", "act_freq_mean"]
with open(ROOT / "COMPLETE-MATRIX.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

# per-arm markdown: mean over rounds, env delta shown
def num(s): return float(re.match(r"[\d.]+", s)[0]) if s else None
arms = defaultdict(list)
for r in rows:
    if r["section"] in ("two", "ngram"): continue
    arms[(r["section"], r["arm"])].append(r)
md = ["# Complete matrix, b12327, Arc A770 (xe)", "",
      "Baseline env (every run): `" + " ".join(f"{k}={v}" for k, v in BASE_ENV.items()) + "`; ambient GGML_/SYCL_/UR_/ZE_/LLAMA_ARG vars were unset first. 'env delta' = what the arm changed on top. Per-run rows with full env, bench args, idle, load and mean act_freq: COMPLETE-MATRIX.csv.", ""]
for sec in ("ccs", "envs", "il", "freq", "ornith"):
    md += [f"## {sec}", "", "| arm | ccs | freq min/max/profile | env delta | extra args | n | pp512 | tg64 | pp512@8k | tg64@8k |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (s, arm), rs in arms.items():
        if s != sec: continue
        r0 = rs[0]; vals = [f"{st.mean(num(r[t]) for r in rs if r[t]):.1f}" for t in TESTS]
        extra = r0["bench_args"].split("-t 12")[1].strip()
        md.append(f"| {arm} | {r0['ccs_mode']} | {r0['freq_min']}/{r0['freq_max']}/{r0['power_profile']} | {r0['env_extra']} | {extra or '-'} | {len(rs)} | " + " | ".join(vals) + " |")
    md.append("")
md += ["## two concurrent 8B processes (aggregate of A+B, mean of 3 rounds)", "", "| arm | ccs | timeslice us | env delta | agg pp512 | agg tg64 |", "|---|---|---|---|---|---|"]
tw = defaultdict(list)
for r in rows:
    if r["section"] == "two": tw[r["arm"]].append(r)
for arm, rs in tw.items():
    agg = lambda t: st.mean(float(r[t].split("=")[-1]) for r in rs)
    md.append(f"| {arm} | {rs[0]['ccs_mode']} | {rs[0]['timeslice_us']} | (none) | {agg('pp512'):.1f} | {agg('tg64'):.2f} |")
md += ["", "## spec-type server (Ornith production flags; mean of 2 launches x 3 reps)", "", "| arm | server args | prompt | tg t/s | draft acc/n |", "|---|---|---|---|---|"]
sv = defaultdict(list)
for r in rows:
    if r["section"] == "ngram": sv[(r["arm"], r["bench_args"].split("|")[1].strip())].append(r)
for (arm, pr), rs in sv.items():
    acc = [list(map(int, r["rc"].split()[1].split("/"))) for r in rs]
    md.append(f"| {arm} | {srv[arm]} | {pr.replace('prompt=','')} | {st.mean(float(r['tg64'].split()[0]) for r in rs):.1f} | {sum(a[0] for a in acc)}/{sum(a[1] for a in acc)} |")
(ROOT / "COMPLETE-MATRIX.md").write_text("\n".join(md) + "\n")
print(len(rows), "rows")
