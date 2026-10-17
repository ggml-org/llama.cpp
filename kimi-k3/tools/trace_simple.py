#!/usr/bin/env python3
"""Convert a DETAILED decode trace (LLAMA_TRACE_DETAIL=2, or a trace written before the simple layout existed) to
the simple view that the tracer writes by default (LLAMA_TRACE_DETAIL=1): which layer and expert each CPU worker,
the GPU, the expert loads and the host <-> device copies are busy with. Same layer + expert = same span name = same
color in ui.perfetto.dev. New traces need no conversion.

  python3 -I trace_simple.py /tmp/k3.json [-o /tmp/k3.simple.json]
  python3 -I trace_simple.py --selftest

Output: a Chrome-trace JSON of "X" events only, process "simple", rows (threads):
  token      one span per TOKEN: "token <step> -> <token_out>"
  GPU        "L<layer>": the device time of the GPU ops, merged over consecutive ops of the same layer less than
             100 us apart (concurrent streams are clipped so that the row does not overlap itself)
  worker N   "L<layer> e<expert>" while CPU thread N computes that expert (one span per tensor, consecutive
             spans of the same name merged); "L<layer>" for the layer's other CPU ops, merged over consecutive ops
             of the same layer (worker 0 only: the trace records the non-expert ops on worker 0)
  copy N     "L<layer>" (the layer of the copied tensor; "input" / "output" without one) for every host <-> device
             transfer of a step: scheduler copies of split inputs and the application's tensor set / get (logits)
  load N     "L<layer> e<expert>" from the first read issue to the last piece resident, merged over the expert's
             pieces, its gate/up/down tensors and the sources (prefetch, predictor) into one span per
             (layer, expert, step); greedy lanes so that no row overlaps itself. Loads that were cached at issue
             (no read at all) are SKIPPED; a load still unresolved at the end of the trace ends with its step.
Span names are exactly "L<layer> e<expert>" / "L<layer>" in every row; tensor, source, step, etc. go to args.

Detail 1 traces have no per-thread expert records: the merged EXPERT span (first thread start .. last thread end)
is put on every worker that took part, clipped so that the experts of one node do not overlap on a row; the
output's metadata says so. Detail 2 traces use the per-thread records as they are.
Reads both the current writer's form (complete events on lanes) and the older one (async b/e pairs).
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

PID_HOST, PID_DEV0, PID_EXPERT, PID_IO = 1, 2, 100, 101
TID_WORKER0 = 100
OUT_PID = 1
TID_TOKEN, TID_GPU, TID_WORKER, TID_LOAD, TID_COPY = 1, 2, 100, 1000, 2000
SOURCES = ("prefetch", "predict_high", "predict_low")  # in the tracer's order

# the args the tracer writes per row kind (optional ones in brackets), mirrored here; --selftest compares
TRACER_ARGS = {
    "token": ({"step", "token", "pos", "token_out", "n_out_sync", "n_sample", "sample_token", "n_sync", "sync_ms", "n_split",
               "n_predict", "n_experts", "n_expert_threads", "n_residency", "n_pieces_prefetch", "n_pieces_predict_high",
               "n_pieces_predict_low", "n_pieces_cached", "n_pieces_dup", "n_loads", "n_loads_cached", "n_copies", "copy_bytes",
               "n_gpu_ops"}, {"out_sync_ts", "out_sync_dur", "sample_ts", "sample_dur"}),
    "worker_expert": ({"layer", "expert", "step", "tensor", "bytes", "stall_ns", "n_threads"}, {"resident_frac", "sample_delay_ns"}),
    "worker_dense": ({"layer", "step", "n_ops"}, set()),
    "GPU": ({"layer", "step", "n_ops", "n_weight_ops"}, set()),
    "copy": ({"layer", "step", "direction", "bytes", "kind", "async", "weight", "tensor", "backend"}, set()),
    "load": ({"layer", "expert", "step", "pieces", "cached_pieces", "duplicate_pieces", "bytes", "tensors", "sources",
              "unresolved", "computed"}, {"t_resident_ns"}),
}
TRACER_TYPES = {"sources": str, "tensors": int, "unresolved": bool, "computed": bool, "weight": bool, "async": bool,
                "direction": str, "tensor": str, "backend": str, "kind": str}


def row_kind(row, args):
    if row == "token" or row == "GPU":
        return row
    if row.startswith("worker "):
        return "worker_expert" if "expert" in args else "worker_dense"
    return row.split(" ")[0]


def arg_problems(out):
    """Differences between the args of the output's spans and the tracer's (keys, types)"""
    names, rows = rows_of(out)
    bad = set()
    for r, spans in rows.items():
        for e in spans:
            k = row_kind(r, e["args"])
            req, opt = TRACER_ARGS[k]
            keys = set(e["args"])
            if not req <= keys or keys - req - opt:
                bad.add(f"{k}: missing {sorted(req - keys)}, extra {sorted(keys - req - opt)}")
            for key, ty in TRACER_TYPES.items():
                if key in e["args"] and not isinstance(e["args"][key], ty):
                    bad.add(f"{k}: {key} is {type(e['args'][key]).__name__}, not {ty.__name__}")
    return sorted(bad)


def load_events(path):
    with open(path) as f:
        s = f.read()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(s.rstrip().rstrip(",") + "]")


def spans_of(events, cat, pid):
    """Complete spans of a category on a process: "X" events, or "b"/"e" pairs by id; (t0, t1, b args, e args)."""
    out, opened = [], {}
    for e in events:
        if e.get("cat") != cat or e.get("pid") != pid:
            continue
        ph = e.get("ph")
        if ph == "X":
            out.append((e["ts"], e["ts"] + e.get("dur", 0.0), e.get("args", {}), e.get("args", {})))
        elif ph == "b":
            opened[e["id"]] = e
        elif ph == "e" and e["id"] in opened:
            b = opened.pop(e["id"])
            out.append((b["ts"], e["ts"], b.get("args", {}), e.get("args", {})))
    return out


def parse_layer(name):
    """The layer tag of a tensor name: "-<il>" suffix, "blk.<il>.", "_l<il>"; None if none."""
    m = re.search(r"blk\.(\d+)\.", name) or re.search(r"-(\d+)$", name) or re.search(r"_l(\d+)", name)
    return int(m.group(1)) if m else None


def ename(layer, expert):
    return f"L{layer} e{expert}"


def lname(layer):
    return f"L{layer}"


GPU_GAP_US = 100.0  # the tracer's GGML_TRACE_GPU_GAP_NS


def merge_runs(items, max_gap=None):
    """items: (t0, t1, name, args) sorted by t0; consecutive spans of the same name that are not expert spans become
    one span (n_ops counts them); the result does not overlap itself (a start before the previous end is moved to
    that end, empty spans dropped)."""
    out = []
    for t0, t1, name, args in items:
        if out and out[-1][2] == name and "expert" not in args and (max_gap is None or t0 - out[-1][1] < max_gap):
            out[-1][1] = max(out[-1][1], t1)
            out[-1][3]["n_ops"] = out[-1][3].get("n_ops", 1) + args.get("n_ops", 1)
            out[-1][3]["n_weight_ops"] = out[-1][3].get("n_weight_ops", 0) + args.get("n_weight_ops", 0)
            if "n_weight_ops" not in args:
                out[-1][3].pop("n_weight_ops")
            continue
        if out and t0 < out[-1][1]:
            t0 = out[-1][1]
        if t1 <= t0:
            continue
        out.append([t0, t1, name, dict(args)])
    return out


def place_lanes(items):
    """Greedy lanes: each span (sorted by t0) on the first lane whose last span has ended."""
    ends, lanes = [], []
    for it in sorted(items, key=lambda x: x[0]):
        for i, end in enumerate(ends):
            if end <= it[0]:
                ends[i] = it[1]
                lanes[i].append(it)
                break
        else:
            ends.append(it[1])
            lanes.append([it])
    return lanes


def simplify(events):
    meta_in = next((e.get("args", {}) for e in events if e.get("name") == "meta" and e.get("cat") == "meta"), {})
    detail = meta_in.get("config", {}).get("detail", 1)
    tokens = sorted((e for e in events if e.get("cat") == "token" and e.get("ph") == "X"), key=lambda e: e["ts"])
    rows = defaultdict(list)  # (tid, name) -> [t0, t1, name, args]
    notes = []

    # token row, with the per-step counts the tracer puts there (for trace_check.py)
    cnt = defaultdict(lambda: defaultdict(float))
    experts_seen = defaultdict(set)
    for e in events:
        a = e.get("args", {})
        st = a.get("step")
        cat = e.get("cat")
        if not isinstance(st, int):
            continue
        c = cnt[st]
        if cat == "sync":
            if a.get("output"):
                c["n_out_sync"] += 1
                c["out_sync_ts"], c["out_sync_dur"] = e["ts"], e["dur"]
            else:
                c["n_sync"] += 1
                c["sync_ms"] += e["dur"] / 1000
        elif cat == "sample":
            c["n_sample"] += 1
            c["sample_ts"], c["sample_dur"], c["sample_token"] = e["ts"], e["dur"], a["token_out"]
        elif cat == "split":
            c["n_split"] += 1
        elif cat == "predict":
            c["n_predict"] += 1
        elif cat == "residency":
            c["n_residency"] += 1
        elif cat in ("copy", "xfer"):
            c["n_copies"] += 1
            c["copy_bytes"] += a.get("bytes", 0)
        elif cat == "op" and e.get("pid", 0) >= PID_DEV0 and e.get("pid") not in (PID_EXPERT, PID_IO) and e.get("tid") == 1:
            c["n_gpu_ops"] += 1
        elif cat == "expert_thread":
            c["n_expert_threads"] += 1
            experts_seen[st].add((a["name"], a["expert"]))
        elif cat == "expert" and e.get("ph") in ("X", "b"):
            experts_seen[st].add((a["name"], a["expert"]))
        elif cat in ("io", "io_issue") and e.get("pid") == PID_HOST and e.get("ph") == "X":
            src = a.get("source", "prefetch")
            c["n_pieces_" + src] += 1
            c["n_pieces_cached"] += bool(a.get("cached"))
            c["n_pieces_dup"] += bool(a.get("duplicate"))
    for t in tokens:
        a = t["args"]
        c = cnt[a["step"]]
        args = {"step": a["step"], "token": a["token"], "pos": a["pos"], "token_out": a["token_out"],
                "n_experts": len(experts_seen[a["step"]]), "sample_token": int(c.get("sample_token", -1))}
        for k in ("n_out_sync", "n_sample", "n_sync", "n_split", "n_predict", "n_residency", "n_copies", "copy_bytes", "n_gpu_ops",
                  "n_expert_threads", "n_pieces_prefetch", "n_pieces_predict_high", "n_pieces_predict_low", "n_pieces_cached", "n_pieces_dup"):
            args[k] = int(c.get(k, 0))
        args["sync_ms"] = round(c.get("sync_ms", 0.0), 3)
        for k in ("out_sync_ts", "out_sync_dur", "sample_ts", "sample_dur"):
            if k in c:
                args[k] = c[k]
        rows[(TID_TOKEN, "token")].append([t["ts"], t["ts"] + t["dur"], f"token {a['step']} -> {a['token_out']}", args])

    # GPU row: device times, per layer run
    def gpu_name(e):
        return "output" if e["args"].get("name", e["name"]).startswith("result_") else lname(e["args"]["layer"])
    gpu = sorted((e for e in events if e.get("cat") == "op" and e.get("ph") == "X" and e.get("pid", 0) >= PID_DEV0
                  and e.get("pid") not in (PID_EXPERT, PID_IO) and e.get("tid") == 1
                  and (e["args"].get("layer", -1) >= 0 or e["args"].get("name", "").startswith("result_"))),
                 key=lambda e: e["ts"])
    rows[(TID_GPU, "GPU")] = merge_runs([(e["ts"], e["ts"] + e["dur"], gpu_name(e),
                                          {"layer": -2 if gpu_name(e) == "output" else e["args"]["layer"], "step": e["args"]["step"],
                                           "n_ops": 1, "n_weight_ops": int(bool(e["args"].get("weight")))}) for e in gpu], GPU_GAP_US)

    # copy rows: scheduler copies and the application's transfers of traced steps
    copy_items = []
    for e in events:
        cat = e.get("cat")
        if cat not in ("copy", "xfer") or e["args"].get("step", -1) < 0:
            continue
        a = e["args"]
        if cat == "xfer":
            h2d = a.get("direction") == "input"
            layer = parse_layer(a.get("name", ""))
        else:
            h2d = a.get("backend") != "CPU"
            layer = a.get("layer", -1)
        name = lname(layer) if layer is not None and layer >= 0 else ("input" if h2d else "output")
        t1 = e["ts"] + e.get("dur", 0.0)
        copy_items.append((e["ts"], max(t1, e["ts"] + 0.001), name,
                           {"layer": layer if layer is not None else -1, "step": a["step"], "direction": "h2d" if h2d else "d2h",
                            "bytes": a.get("bytes", 0), "kind": cat, "async": bool(a.get("async")),
                            "weight": bool(a.get("weight")) or ".weight" in a.get("name", ""), "tensor": a.get("name", ""),
                            "backend": a.get("backend") or "CPU"}))
    for i, lane in enumerate(place_lanes(copy_items)):
        rows[(TID_COPY + i, f"copy {i}")] = [list(x) for x in lane]

    # worker rows: expert compute; the merged EXPERT span has the bytes, threads and residency sample
    exp_args = {}
    for _, _, a, _ in spans_of(events, "expert", PID_EXPERT):
        ea = {"bytes": a.get("bytes", 0), "n_threads": a.get("n_threads", 1)}
        if isinstance(a.get("resident_frac"), (int, float)):
            ea["resident_frac"], ea["sample_delay_ns"] = a["resident_frac"], a.get("sample_delay_ns", 0)
        exp_args[(a["step"], a["name"], a["expert"])] = ea
    worker_items = defaultdict(list)
    threads = [e for e in events if e.get("cat") == "expert_thread" and e.get("ph") == "X"]
    if threads:
        n_thr = defaultdict(int)
        for e in threads:
            n_thr[(e["args"]["step"], e["args"]["name"], e["args"]["expert"])] += 1
        for e in threads:
            a = e["args"]
            k = (a["step"], a["name"], a["expert"])
            ex = dict(exp_args.get(k, {"bytes": 0}))
            ex["n_threads"] = n_thr[k]
            worker_items[e["tid"] - TID_WORKER0].append((e["ts"], e["ts"] + e["dur"], ename(a["layer"], a["expert"]),
                                                       dict(ex, layer=a["layer"], expert=a["expert"], step=a["step"], tensor=a["name"],
                                                            stall_ns=a.get("stall_ns", 0))))
    else:
        notes.append("detail 1: no per-thread expert records; each expert's merged span (first thread start .. last "
                     "thread end) is shown on every worker that took part, clipped so that the experts of one node do "
                     "not overlap")
        by_node = defaultdict(list)
        for t0, t1, a, _ in spans_of(events, "expert", PID_EXPERT):
            by_node[(a["step"], a["name"])].append((t0, t1, a))
        for node in by_node.values():
            node.sort(key=lambda x: x[0])
            for i, (t0, t1, a) in enumerate(node):
                if i + 1 < len(node):
                    t1 = min(t1, node[i + 1][0])
                for w in range(a.get("n_threads", 1)):
                    worker_items[w].append((t0, t1, ename(a["layer"], a["expert"]),
                                            dict(exp_args.get((a["step"], a["name"], a["expert"]), {}), layer=a["layer"], expert=a["expert"],
                                                 step=a["step"], tensor=a["name"], stall_ns=a.get("stall_ns", 0) // max(1, a.get("n_threads", 1)))))
    # the layer's other CPU ops (worker 0 records them; MUL_MAT_ID is shown per expert above)
    for e in events:
        if e.get("cat") == "op" and e.get("ph") == "X" and e.get("pid") == PID_HOST and e.get("tid") == TID_WORKER0:
            a = e["args"]
            if a.get("op") != "MUL_MAT_ID" and a.get("layer", -1) >= 0:
                worker_items[0].append((e["ts"], e["ts"] + e["dur"], lname(a["layer"]), {"layer": a["layer"], "step": a["step"], "n_ops": 1}))
    for w, items in worker_items.items():
        rows[(TID_WORKER + w, f"worker {w}")] = merge_runs(sorted(items, key=lambda x: x[0]))
    if not threads:
        per_step = defaultdict(int)
        for items in worker_items.values():
            for it in items:
                per_step[it[3]["step"]] += "expert" in it[3]
        for span in rows[(TID_TOKEN, "token")]:
            span[3]["n_expert_threads"] = per_step[span[3]["step"]]

    # load rows: one span per (layer, expert, step), from the first issue to the last resident
    tok_ends = [(t["ts"], t["ts"] + t["dur"], t["args"]["step"]) for t in tokens]

    def step_for(ts, step):
        if step >= 0:
            return step
        # an IDLE hint is for the step that is running (its span may start late) or the next traced one
        nxt = [s for _, t1, s in tok_ends if t1 >= ts]
        return nxt[0] if nxt else -1

    # loads, as the tracer builds them: the counts from the issue records, the ends from the I/O spans
    def new_load():
        return {"t0": float("inf"), "t_issue_end": 0.0, "t_res": None, "pieces": 0, "cached": 0, "dup": 0, "bytes": 0,
                "unresolved": False, "sources": set(), "tensors": set()}
    loads = defaultdict(new_load)
    # older traces also recorded the hints of prompt batches and skipped steps (before the first TOKEN); the tracer
    # no longer does, and they are left out here too
    first_tok = tok_ends[0][0] if tok_ends else float("-inf")
    n_early = 0
    for e in events:
        if e.get("cat") in ("io", "io_issue") and e.get("pid") == PID_HOST and e.get("ph") == "X":
            if e["ts"] < first_tok:
                n_early += 1
                continue
            a = e["args"]
            d = loads[(a["layer"], a["expert"], step_for(e["ts"], a["step"]))]
            d["t0"] = min(d["t0"], e["ts"])
            d["t_issue_end"] = max(d["t_issue_end"], e["ts"] + e["dur"])
            d["pieces"] += 1
            d["cached"] += bool(a.get("cached"))
            d["dup"] += bool(a.get("duplicate"))
            d["bytes"] += a.get("bytes", 0)
            d["sources"].add(a.get("source", "prefetch"))
            d["tensors"].add(a.get("name", "?"))
    for t0, t1, b, e in spans_of(events, "io", PID_IO):
        cached = bool(b.get("cached")) or e.get("resolution") == "cached"
        if cached or t0 < first_tok:
            continue
        d = loads[(b["layer"], b["expert"], step_for(t0, b["step"]))]
        d["t_res"] = max(d["t_res"] or 0.0, t1)
        d["unresolved"] = d["unresolved"] or bool(e.get("unresolved")) or e.get("resolution") == "unresolved" or bool(e.get("partial"))
    computed = defaultdict(set)
    for items in worker_items.values():
        for it in items:
            if "expert" in it[3]:
                computed[it[3]["step"]].add(it[2])
    step_end = {s: t1 for _, t1, s in tok_ends}
    load_items = []
    n_cached = 0
    for (layer, expert, step), d in loads.items():
        if d["cached"] == d["pieces"]:
            n_cached += 1  # no read at all
            continue
        t1 = max(d["t_res"] or 0.0, d["t_issue_end"])
        if d["unresolved"]:
            t1 = min(t1, max(step_end[step], d["t_issue_end"])) if step in step_end else d["t_issue_end"]
        if t1 <= d["t0"]:
            t1 = d["t0"] + 0.001
        args = {"layer": layer, "expert": expert, "step": step, "pieces": d["pieces"], "cached_pieces": d["cached"],
                "duplicate_pieces": d["dup"], "bytes": d["bytes"], "tensors": len(d["tensors"]),
                "sources": ",".join(x for x in SOURCES if x in d["sources"]), "unresolved": d["unresolved"],
                "computed": ename(layer, expert) in computed[step]}
        if d["t_res"] is not None:
            args["t_resident_ns"] = int(round(d["t_res"] * 1000))
        load_items.append((d["t0"], t1, ename(layer, expert), args))
    for i, lane in enumerate(place_lanes(load_items)):
        rows[(TID_LOAD + i, f"load {i}")] = [list(x) for x in lane]
    notes.append(f"{n_cached} loads cached at issue (no read) are not shown")
    if n_early:
        notes.append(f"{n_early} read hints before the first TOKEN (prompt batches, skipped steps) are left out")
    n_loads, n_loads_cached = defaultdict(int), defaultdict(int)
    for (layer, expert, step), d in loads.items():
        n_loads[step] += 1
        n_loads_cached[step] += d["cached"] == d["pieces"]
    for span in rows[(TID_TOKEN, "token")]:
        span[3]["n_loads"] = n_loads[span[3]["step"]]
        span[3]["n_loads_cached"] = n_loads_cached[span[3]["step"]]

    out = [{"name": "meta", "cat": "meta", "ph": "i", "s": "g", "ts": 0, "pid": OUT_PID, "tid": TID_TOKEN,
            "args": dict(meta_in, layout="simple", converted_by="trace_simple.py", source_detail=detail, notes=notes)},
           {"name": "process_name", "ph": "M", "pid": OUT_PID, "args": {"name": "simple"}}]
    for (tid, tname), spans in sorted(rows.items()):
        out.append({"name": "thread_name", "ph": "M", "pid": OUT_PID, "tid": tid, "args": {"name": tname}})
        out.append({"name": "thread_sort_index", "ph": "M", "pid": OUT_PID, "tid": tid, "args": {"sort_index": tid}})
        for t0, t1, name, args in spans:
            out.append({"name": name, "cat": "simple", "ph": "X", "ts": round(t0, 3), "dur": round(t1 - t0, 3), "pid": OUT_PID, "tid": tid, "args": args})
    return out, notes


def rows_of(out):
    names = {e["tid"]: e["args"]["name"] for e in out if e.get("name") == "thread_name"}
    rows = defaultdict(list)
    for e in out:
        if e["ph"] == "X":
            rows[names[e["tid"]]].append(e)
    return names, rows


def check(out, n_workers=None):
    """Invariants of the output; returns a list of problems."""
    bad = []
    names, rows = rows_of(out)
    for e in out:
        if e["ph"] not in ("X", "M") and e.get("name") != "meta":
            bad.append(f"non-X event {e['ph']}")
    for rname, spans in rows.items():
        spans = sorted(spans, key=lambda e: e["ts"])
        for a, b in zip(spans, spans[1:]):
            if b["ts"] < a["ts"] + a["dur"] - 1e-3:
                bad.append(f"overlap in {rname}: {a['name']} / {b['name']} at {b['ts']}")
                break
    for rname in names.values():
        ok = rname in ("token", "GPU") or rname.split(" ")[0] in ("worker", "load", "copy")
        if not ok:
            bad.append(f"unexpected row {rname}")
    return bad


def expert_names(rows, prefix):
    return {e["name"] for r, spans in rows.items() if r.startswith(prefix) for e in spans if " e" in e["name"]}


def selftest():
    """Synthetic trace: 2 workers at detail 2, one GPU layer, two experts loaded (one cached)."""
    def X(cat, label, ts, dur, pid, tid, **args):
        return {"name": label, "cat": cat, "ph": "X", "ts": ts, "dur": dur, "pid": pid, "tid": tid, "args": args}
    ev = [
        {"name": "meta", "cat": "meta", "ph": "i", "ts": 0, "pid": 1, "tid": 1, "args": {"config": {"detail": 2}}},
        X("token", "TOKEN", 0, 1000, 1, 1, step=0, token=5, pos=9, token_out=7),
        X("op", "attn_norm-0", 10, 20, 2, 1, step=0, layer=0, op="RMS_NORM"),
        X("op", "attn_out-0", 30, 20, 2, 1, step=0, layer=0, op="MUL_MAT"),
        X("op", "result_output", 900, 20, 2, 1, step=0, layer=1, op="MUL_MAT", layer_inferred=True),
        X("op", "ffn_norm-1", 100, 10, 1, 100, step=0, layer=1, op="RMS_NORM"),
        X("op", "ffn_moe_gate-1", 110, 200, 1, 100, step=0, layer=1, op="MUL_MAT_ID"),
        X("op", "ffn_out-1", 310, 10, 1, 100, step=0, layer=1, op="ADD"),
    ]
    for w in (0, 1):
        ev.append(X("expert_thread", "e3", 115 + w, 40, 1, 100 + w, step=0, layer=1, expert=3, name="ffn_moe_gate-1"))
        ev.append(X("expert_thread", "e8", 160 + w, 40, 1, 100 + w, step=0, layer=1, expert=8, name="ffn_moe_gate-1"))
    # loads: expert 3 read by the prefetch (2 pieces) and hinted by the predictor; expert 8 read; expert 9 cached
    for (ts, dur, st, ex, nm, src, cached) in [(50, 2, 0, 3, "blk.1.ffn_gate_exps.weight", "prefetch", False),
                                               (40, 3, -1, 3, "blk.1.ffn_up_exps.weight", "predict_low", False),
                                               (60, 2, 0, 8, "blk.1.ffn_gate_exps.weight", "prefetch", False),
                                               (60, 1, 0, 9, "blk.1.ffn_gate_exps.weight", "prefetch", True)]:
        ev.append(X("io_issue", f"ISSUE L1 e{ex} p0", ts, dur, 1, 100, step=st, layer=1, expert=ex, name=nm, source=src,
                    piece=0, n_pieces=1, bytes=4096, cached=cached, duplicate=False))
    ev.append(X("io", "L1 e3", 50, 30, 101, 800, step=0, layer=1, expert=3, name="blk.1.ffn_gate_exps.weight", source="prefetch", cached=False))
    ev.append(X("io", "L1 e3", 40, 70, 101, 801, step=-1, layer=1, expert=3, name="blk.1.ffn_up_exps.weight", source="predict_low", cached=False))
    ev.append(X("io", "L1 e8", 60, 50, 101, 802, step=0, layer=1, expert=8, name="blk.1.ffn_gate_exps.weight", source="prefetch", cached=False))
    ev.append(X("io", "L1 e9", 60, 0, 101, 803, step=0, layer=1, expert=9, name="blk.1.ffn_gate_exps.weight", source="prefetch", cached=True))
    ev.append(X("xfer", "inp_tokens", 5, 2, 1, 1, step=0, direction="input", bytes=4, name="inp_tokens"))
    ev.append(X("copy", "COPY attn_out-0", 60, 5, 1, 1, step=0, backend="CPU", bytes=1024, name="attn_out-0", layer=0, **{"async": False}))
    ev.append(X("xfer", "result_output", 950, 3, 1, 1, step=0, direction="output", bytes=640, name="result_output"))
    out, _ = simplify(ev)
    bad = check(out) + arg_problems(out)
    names, rows = rows_of(out)
    assert not bad, bad
    assert set(names.values()) == {"token", "GPU", "worker 0", "worker 1", "load 0", "load 1", "copy 0"}, names
    assert [(e["name"], e["args"]["direction"]) for e in sorted(rows["copy 0"], key=lambda e: e["ts"])] == \
        [("input", "h2d"), ("L0", "d2h"), ("output", "d2h")], rows["copy 0"]
    w = expert_names(rows, "worker ")
    ld = expert_names(rows, "load ")
    assert w == ld == {"L1 e3", "L1 e8"}, (w, ld)  # byte-identical names on compute and load rows
    assert [e["name"] for e in rows["GPU"]] == ["L0", "output"], rows["GPU"]
    assert [e["name"] for e in sorted(rows["worker 0"], key=lambda e: e["ts"])] == ["L1", "L1 e3", "L1 e8", "L1"]
    assert [e["name"] for e in rows["worker 1"]] == ["L1 e3", "L1 e8"]
    l3 = [e for r, s in rows.items() if r.startswith("load") for e in s if e["name"] == "L1 e3"]
    assert len(l3) == 1 and l3[0]["ts"] == 40 and l3[0]["dur"] == 70, l3  # pieces, tensors, sources merged
    assert rows["token"][0]["name"] == "token 0 -> 7"
    loads = {e["name"]: e["args"] for r, s in rows.items() if r.startswith("load") for e in s}
    assert loads["L1 e3"]["sources"] == "prefetch,predict_low" and loads["L1 e3"]["pieces"] == 2 and loads["L1 e3"]["computed"] is True
    assert rows["token"][0]["args"]["n_loads"] == 3 and rows["token"][0]["args"]["n_loads_cached"] == 1
    # one expert per node and thread: consecutive spans of the same expert (gate, then up) stay apart
    ev2 = [dict(e) for e in ev if e.get("cat") != "expert_thread"]
    for tensor, t in (("ffn_moe_gate-1", 115), ("ffn_moe_up-1", 160)):
        ev2.append(X("expert_thread", "e3", t, 40, 1, 100, step=0, layer=1, expert=3, name=tensor))
    _, rows2 = rows_of(simplify(ev2)[0])
    assert [e["name"] for e in rows2["worker 0"]].count("L1 e3") == 2, rows2["worker 0"]
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?")
    ap.add_argument("-o", "--out", help="output file (default: <trace>.simple.json next to the input)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.trace:
        ap.error("a trace file is required")
    out, notes = simplify(load_events(args.trace))
    bad = check(out)
    if bad:
        print("problems:", bad[:10], file=sys.stderr)
        return 1
    path = args.out or (args.trace[:-5] if args.trace.endswith(".json") else args.trace) + ".simple.json"
    with open(path, "w") as f:
        f.write("[\n")
        f.write(",\n".join(json.dumps(e, separators=(",", ":")) for e in out))
        f.write("\n]\n")
    names, rows = rows_of(out)
    w, ld = expert_names(rows, "worker "), expert_names(rows, "load ")
    print(f"{path}: {os.path.getsize(path)} bytes, {sum(len(s) for s in rows.values())} spans in {len(rows)} rows "
          f"({', '.join(f'{r}: {len(s)}' for r, s in sorted(rows.items(), key=lambda kv: min(t for t, n in names.items() if n == kv[0])))})")
    print(f"expert names: {len(w)} computed, {len(ld)} loaded, {len(w & ld)} on both (the rest: cached loads, or "
          f"hinted experts that were not computed)")
    for n in notes:
        print("note:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
