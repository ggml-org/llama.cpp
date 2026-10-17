#!/usr/bin/env python3
"""Summarize a decode trace (LLAMA_TRACE / run-server.sh TRACE=): where the time of each token goes.

  python3 -I trace_report.py /tmp/k3.json [--layers] [--json out.json]

Per token: wall time (TOKEN span + output SYNC), worker 0 CPU op time, GPU device op time, expert compute time,
expert residency at compute start (mean / min / slices not fully resident), residency sample delay (median),
I/O wait (sum of the per-thread stall over all experts), host waits (SYNC, incl. the output SYNC), COPY / XFER
bytes, and the part of the wall time no op, sync or copy accounts for.
I/O: issue -> resident latency (median / p95) per source, cached and unresolved counts, and the predictor hit rate
per source and guess rank (a guess is a hit when its expert is computed in the layer it was made for, in the step
that follows the guess). --layers adds the per-layer averages over the traced tokens.
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trace_check import TID_WORKER0, Trace, kind_of, t0, t1, union_len  # noqa: E402


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def med(v):
    return statistics.median(v) if v else float("nan")


def rank_bucket(rank):
    if rank is None:
        return "selected"
    kind, n = rank[0], int(rank[1:])
    lo = n // 4 * 4
    return f"{'router' if kind == 'r' else 'popular'} {lo}-{lo + 3}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--layers", action="store_true", help="per-layer table")
    ap.add_argument("--json", help="also write the numbers to this file")
    args = ap.parse_args()

    tr = Trace(args.trace)
    meta = tr.meta
    print(f"trace {args.trace}: {meta.get('model', '?')}, {meta.get('tokens_traced')} tokens, detail {tr.detail}, "
          f"{meta.get('events_written')} events, {meta.get('dropped')} dropped records, "
          f"poll {meta.get('config', {}).get('poll_us')} us ({meta.get('poll_adaptations')} adaptations), "
          f"residency valid: {meta.get('residency_valid')}")

    exp_by_step = defaultdict(list)
    for x in tr.experts:
        exp_by_step[x["step"]].append(x)

    rows = []
    hdr = (f"{'step':>4} {'token':>7} {'pos':>6} {'wall ms':>8} {'cpu op':>7} {'gpu op':>7} {'expert':>7} {'res mean':>8} "
           f"{'res min':>7} {'not res':>7} {'smp us':>6} {'io wait':>8} {'sync ms':>7} {'copy KiB':>8} {'xfer KiB':>8} {'unacc ms':>8}")
    print("\nper token (ms unless noted; io wait = sum over threads and experts)")
    print(hdr)
    for tok in tr.tokens:
        s = tok["args"]["step"]
        lo, hi = t0(tok), t1(tok)
        out_sync = sum(e.get("dur", 0.0) for e in tr.output_sync(s))
        wall = (hi - lo) + out_sync
        cpu_ops = [e for e in tr.ops_cpu if e["args"]["step"] == s and e["tid"] == TID_WORKER0]
        gpu_ops = [e for e in tr.ops_gpu if e["args"]["step"] == s]
        ex = exp_by_step.get(s, [])
        rf = [x["resident_frac"] for x in ex if isinstance(x.get("resident_frac"), (int, float))]
        syncs = [e for e in tr.syncs if e["args"]["step"] == s]
        copies = [e for e in tr.copies if e["args"]["step"] == s]
        xfers = [e for e in tr.xfers if e["args"]["step"] == s]
        iv = [(t0(e), t1(e)) for e in cpu_ops]
        iv += [(t0(e), t1(e)) for e in syncs if not e["args"].get("output")]
        iv += [(t0(e), t1(e)) for e in copies if e["ph"] == "X"]
        iv += [(t0(e), t1(e)) for e in tr.splits if e["args"]["step"] == s and e["args"]["backend"] != "CPU"]
        iv += [(t0(e), t1(e)) for e in xfers]
        covered = union_len(iv, lo, hi)
        row = {
            "step": s,
            "token": tok["args"]["token"],
            "pos": tok["args"]["pos"],
            "token_out": tok["args"]["token_out"],
            "wall_ms": wall / 1000,
            "cpu_op_ms": sum(e["dur"] for e in cpu_ops) / 1000,
            "gpu_op_ms": sum(e["dur"] for e in gpu_ops) / 1000,
            "expert_ms": sum((x["t1"] or x["t0"]) - x["t0"] for x in ex) / 1000,
            "res_mean": statistics.mean(rf) if rf else float("nan"),
            "res_min": min(rf) if rf else float("nan"),
            "not_resident": sum(1 for v in rf if v < 1.0),
            "sample_delay_us": med([x["sample_delay_ns"] / 1000 for x in ex if isinstance(x.get("sample_delay_ns"), (int, float))]),
            "io_wait_ms": sum(x["stall_ns"] for x in ex) / 1e6,
            "sync_ms": sum(e["dur"] for e in syncs) / 1000,
            "output_sync_ms": out_sync / 1000,
            "copy_kib": sum(e["args"]["bytes"] for e in copies) / 1024,
            "xfer_kib": sum(e["args"]["bytes"] for e in xfers) / 1024,
            "unaccounted_ms": (hi - lo - covered) / 1000,
        }
        rows.append(row)
        print(f"{s:>4} {row['token']:>7} {row['pos']:>6} {row['wall_ms']:>8.1f} {row['cpu_op_ms']:>7.1f} {row['gpu_op_ms']:>7.2f} "
              f"{row['expert_ms']:>7.1f} {row['res_mean']:>8.3f} {row['res_min']:>7.3f} {row['not_resident']:>7} "
              f"{row['sample_delay_us']:>6.0f} {row['io_wait_ms']:>8.1f} {row['sync_ms']:>7.2f} {row['copy_kib']:>8.1f} "
              f"{row['xfer_kib']:>8.1f} {row['unaccounted_ms']:>8.2f}")

    # I/O latency per source
    print("\nI/O issue -> resident (" + ("per slice" if tr.detail == 1 else "per piece") + ")")
    by_src = defaultdict(lambda: {"lat": [], "cached": 0, "unresolved": 0, "n": 0, "dup": 0, "issue": []})
    for a in tr.io_spans:
        d = by_src[a["source"]]
        d["n"] += 1
        d["issue"].append(a.get("issue_dur_ns", 0) / 1000)
        if a.get("cached"):
            d["cached"] += 1
            continue
        end = a["end"] or {}
        if end.get("unresolved") or end.get("resolution") == "unresolved":
            d["unresolved"] += 1
        elif a["t1"] is not None:
            d["lat"].append((a["t1"] - a["t0"]) / 1000)
    for i in tr.issues:
        if i["args"]["duplicate"]:
            by_src[i["args"]["source"]]["dup"] += 1
    print(f"{'source':>14} {'spans':>7} {'cached':>7} {'unres':>6} {'dup pcs':>7} {'issue us med':>12} {'lat ms med':>10} {'lat ms p95':>10}")
    io_rows = {}
    for src, d in sorted(by_src.items()):
        io_rows[src] = {k: d[k] for k in ("n", "cached", "unresolved", "dup")}
        io_rows[src].update({"issue_us_median": med(d["issue"]), "lat_ms_median": med(d["lat"]), "lat_ms_p95": pct(d["lat"], 0.95)})
        print(f"{src:>14} {d['n']:>7} {d['cached']:>7} {d['unresolved']:>6} {d['dup']:>7} {med(d['issue']):>12.1f} "
              f"{med(d['lat']):>10.2f} {pct(d['lat'], 0.95):>10.2f}")

    # predictor hit rate: the guess for (layer, expert) is a hit if that expert is computed in that layer in the step
    # whose TOKEN span contains the issue, or in the next traced step for a guess issued in IDLE
    computed = defaultdict(set)
    for x in tr.experts:
        computed[x["step"]].add((x["layer"], x["expert"]))
    starts = sorted((t0(t), t["args"]["step"]) for t in tr.tokens)
    hits = defaultdict(lambda: [0, 0])
    seen = set()
    for i in tr.issues:
        a = i["args"]
        if a["source"] == "prefetch" or a["piece"] != 0 or kind_of(a["name"]) != "gate":
            continue  # one count per guessed expert (the first piece of its first tensor)
        s = a["step"]
        if s < 0:
            s = next((st for ts0, st in starts if ts0 >= t0(i)), None)
        if s is None:
            continue
        key = (s, a["source"], a["layer"], a["expert"])
        if key in seen:
            continue
        seen.add(key)
        h = hits[(a["source"], rank_bucket(a.get("rank")))]
        h[1] += 1
        h[0] += (a["layer"], a["expert"]) in computed[s]
    print("\npredictor hits (guessed experts that the layer then computed)")
    print(f"{'source':>14} {'rank':>14} {'guesses':>8} {'hit rate':>8}")
    pred_rows = {}
    for (src, rb), (h, n) in sorted(hits.items()):
        pred_rows[f"{src} {rb}"] = {"guesses": n, "hits": h}
        print(f"{src:>14} {rb:>14} {n:>8} {h / n if n else 0:>8.3f}")

    layer_rows = {}
    if args.layers:
        print("\nper layer (mean over traced tokens; ms)")
        n_tok = max(1, len(tr.tokens))
        agg = defaultdict(lambda: {"cpu": 0.0, "gpu": 0.0, "exp": 0.0, "stall": 0.0, "rf": [], "notres": 0})
        for e in tr.ops_cpu:
            if e["tid"] == TID_WORKER0 and e["args"]["step"] >= 0:
                agg[e["args"]["layer"]]["cpu"] += e["dur"] / 1000
        for e in tr.ops_gpu:
            if e["args"]["step"] >= 0:
                agg[e["args"]["layer"]]["gpu"] += e["dur"] / 1000
        for x in tr.experts:
            d = agg[x["layer"]]
            d["exp"] += ((x["t1"] or x["t0"]) - x["t0"]) / 1000
            d["stall"] += x["stall_ns"] / 1e6
            if isinstance(x.get("resident_frac"), (int, float)):
                d["rf"].append(x["resident_frac"])
                d["notres"] += x["resident_frac"] < 1.0
        print(f"{'layer':>5} {'cpu op':>8} {'gpu op':>8} {'expert':>8} {'io wait':>8} {'res mean':>8} {'not res':>7}")
        for l in sorted(agg):
            d = agg[l]
            layer_rows[l] = {"cpu_op_ms": d["cpu"] / n_tok, "gpu_op_ms": d["gpu"] / n_tok, "expert_ms": d["exp"] / n_tok,
                             "io_wait_ms": d["stall"] / n_tok, "res_mean": statistics.mean(d["rf"]) if d["rf"] else float("nan"),
                             "not_resident": d["notres"] / n_tok}
            r = layer_rows[l]
            print(f"{l:>5} {r['cpu_op_ms']:>8.2f} {r['gpu_op_ms']:>8.3f} {r['expert_ms']:>8.2f} {r['io_wait_ms']:>8.2f} "
                  f"{r['res_mean']:>8.3f} {r['not_resident']:>7.1f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "tokens": rows, "io": io_rows, "predictor": pred_rows, "layers": layer_rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
