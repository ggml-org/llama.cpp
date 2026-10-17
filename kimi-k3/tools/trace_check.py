#!/usr/bin/env python3
"""Check a decode trace (LLAMA_TRACE / run-server.sh TRACE=) of Kimi-K3 against what a decode step must contain.

The trace is the Chrome Trace Event JSON written by ggml-trace.cpp (one event per line, loads in ui.perfetto.dev).
Each check prints PASS / FAIL with the numbers behind it; the exit status is 1 if any check failed.

  python3 -I trace_check.py /tmp/k3.json [--tokens 8] [--gpu-layers 11] [--moe-layers 92] [--n-used 16]

Both layouts are read: the default simple one (LLAMA_TRACE_DETAIL=1: process "simple", rows token / GPU / worker N
/ load N / copy N, the exact per-step counts in the token spans' args) and the detailed one (LLAMA_TRACE_DETAIL=2).

Checks (per traced token unless noted; per piece / per sync / per IO_ISSUE ones need the detailed layout):
  * TOKEN spans: the expected count, consecutive steps and positions
  * exactly one output SYNC after the TOKEN span, exactly one SAMPLE after it whose token is the next TOKEN's input
  * no event of a traced step outside its TOKEN span, no IDLE (step -1) event inside one, no compute or prefetch
    event outside a traced step (prompt batches are not traced)
  * EXPERT: moe-layers x 3 tensors x n-used spans, n-used distinct experts per layer, the same set on gate/up/down;
    one residency sample each (not sampled < 1%), median sample delay < 200 us; an earlier IO_ISSUE for each
  * PREDICT: moe-layers per token (when the predictor is on)
  * I/O: one prefetch slice span per EXPERT, every non-duplicate issue resolved exactly once
  * OP_GPU for every layer < gpu-layers and the output; for the CPU layers no op with a weight (the KV / recurrent
    caches of all layers are on CUDA0, so their weight-free cache ops run there; listed as INFO), no expert matmul
    on the GPU, no weight matrix copied to a device (small norm vectors are, listed as INFO); SPLIT / SYNC present
  * nothing (I/O, residency, compute, PREDICT) before the first TOKEN: prompt batches and skipped steps not traced
  * prefetch: exactly ceil(slice bytes / 2 MiB) pieces per used expert slice
  * XFER input before the first op, output after the last split
  * coverage: worker 0 ops + syncs + copies + GPU split launches >= 90% of TOKEN span + output SYNC
  * overhead: counted trace syscalls x measured per-call cost < 5% of the token time
  * with the python 'perfetto' module: no expert / io slice on an async (global) track, all of them on the lanes of
    the "expert compute" / "expert I/O" processes, lane counts bounded, slices per category == JSON events
"""

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

PID_HOST, PID_EXPERT, PID_IO = 1, 100, 101
PIECE = 2 * 1024 * 1024  # GGML_CPU_MOE_PREFETCH_PIECE
TID_SCHED, TID_PREDICTOR, TID_POLLER, TID_SAMPLER, TID_WORKER0 = 1, 2, 3, 4, 100

# measured on this box (PLAN-decode-trace.md section 2), per call
COST_READ_BYTES_US = 3.5
COST_CLOCK_US = 0.03
COST_CUDA_EVENT_US = 2.0


thread_names = {}  # (pid, tid) -> name, of the last file loaded


def load_trace(path):
    """All events of the file; a file truncated after any line (crashed process) loads too."""
    with open(path, "r") as f:
        s = f.read()
    try:
        events = json.loads(s)
    except json.JSONDecodeError:
        s = s.rstrip().rstrip(",")
        events = json.loads(s + "]")
    meta = {}
    out = []
    thread_names.clear()
    for e in events:
        if e.get("ph") == "M":
            if e.get("name") == "thread_name":
                thread_names[(e["pid"], e["tid"])] = e["args"]["name"]
            continue
        if e.get("name") == "meta" and e.get("cat") == "meta":
            meta = e.get("args", {})
            continue
        out.append(e)
    return meta, out


def t0(e):
    return e["ts"]


def t1(e):
    return e["ts"] + e.get("dur", 0.0)


class Trace:
    """Events of a trace grouped by kind, with the async b/e pairs joined."""

    def __init__(self, path):
        self.meta, self.events = load_trace(path)
        self.thread_names = dict(thread_names)
        self.layout = self.meta.get("layout") or ("simple" if any(e.get("cat") == "simple" for e in self.events) else "detailed")
        self.tokens = sorted((e for e in self.events if e.get("cat") == "token"), key=lambda e: e["args"]["step"])
        self.by_step = {e["args"]["step"]: e for e in self.tokens}
        self.syncs = [e for e in self.events if e.get("cat") == "sync"]
        self.samples = [e for e in self.events if e.get("cat") == "sample"]
        self.xfers = [e for e in self.events if e.get("cat") == "xfer"]
        self.splits = [e for e in self.events if e.get("cat") == "split"]
        self.copies = [e for e in self.events if e.get("cat") == "copy"]
        self.ops_cpu = [e for e in self.events if e.get("cat") == "op" and e["pid"] == PID_HOST]
        self.ops_gpu = [e for e in self.events if e.get("cat") == "op" and e["pid"] not in (PID_HOST, PID_EXPERT, PID_IO)]
        self.launches = [e for e in self.events if e.get("cat") == "launch"]
        self.predicts = [e for e in self.events if e.get("cat") == "predict"]
        self.issues = [e for e in self.events if e.get("cat") == "io_issue"]
        self.residency = [e for e in self.events if e.get("cat") == "residency"]
        self.expert_threads = [e for e in self.events if e.get("cat") == "expert_thread"]
        self.detail = self.meta.get("config", {}).get("detail", 1)

        # EXPERT spans: complete events on the lanes of the expert compute process
        self.experts = []
        for e in self.events:
            if e.get("cat") == "expert" and e["ph"] == "X":
                a = dict(e["args"])
                a["t0"] = e["ts"]
                a["t1"] = e["ts"] + e["dur"]
                self.experts.append(a)

        # issue -> resident spans on the lanes of the expert I/O process (per slice at detail 1, per piece at
        # detail 2); the resolution (unresolved, resolution, t_prev_poll_ns, partial) is in the args
        self.io_spans = []
        for e in self.events:
            if e.get("cat") == "io" and e["pid"] == PID_IO and e["ph"] == "X":
                a = dict(e["args"])
                a["t0"] = e["ts"]
                a["t1"] = None if a.get("partial") else e["ts"] + e["dur"]
                a["end"] = e["args"]
                self.io_spans.append(a)
        if self.layout == "simple":
            self._simple()

    def _simple(self):
        """The rows of the simple layout (process "simple"): token, GPU, worker N, load N, copy N."""
        self.rows = defaultdict(list)
        for e in self.events:
            if e.get("cat") == "simple":
                self.rows[self.thread_names.get((e["pid"], e["tid"]), "?")].append(e)
        self.tokens = sorted(self.rows.get("token", []), key=lambda e: e["args"]["step"])
        self.by_step = {e["args"]["step"]: e for e in self.tokens}
        self.workers = {int(r.split()[1]): v for r, v in self.rows.items() if r.startswith("worker ")}
        self.worker_experts = [e for v in self.workers.values() for e in v if "expert" in e["args"]]
        self.worker_dense = [e for v in self.workers.values() for e in v if "expert" not in e["args"]]
        self.gpu = self.rows.get("GPU", [])
        self.loads = [e for r, v in self.rows.items() if r.startswith("load ") for e in v]
        self.copies_s = [e for r, v in self.rows.items() if r.startswith("copy ") for e in v]

    def step_of_time(self, ts):
        """The traced step whose TOKEN span contains ts, else None."""
        for tok in self.tokens:
            if t0(tok) <= ts <= t1(tok):
                return tok["args"]["step"]
        return None

    def output_sync(self, step):
        return [e for e in self.syncs if e["args"].get("output") and e["args"]["step"] == step]


def is_weight(a):
    """A copied tensor is a weight: the tracer's flag, else (older traces) the name"""
    return bool(a.get("weight")) or ".weight" in a.get("tensor", a.get("name", ""))


def kind_of(name):
    for k in ("gate", "up", "down"):
        if k in name:
            return k
    return "other"


def union_len(intervals, lo, hi):
    """Length of the union of the intervals, clipped to [lo, hi]."""
    iv = sorted((max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi)
    total, cur_a, cur_b = 0.0, None, None
    for a, b in iv:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


class Checker:
    def __init__(self):
        self.n_fail = 0

    def check(self, ok, what, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {what}" + (f": {detail}" if detail else ""))
        if not ok:
            self.n_fail += 1
        return ok

    def info(self, what, detail=""):
        print(f"INFO  {what}" + (f": {detail}" if detail else ""))



def check_simple(tr, args, c):
    """The checks on the default (simple) layout: the exact per-step counts are in the token spans' args, the spans
    of the worker / GPU / load / copy rows carry layer, expert, tensor, bytes and the residency sample."""
    meta = tr.meta
    c.info("layout", "simple (LLAMA_TRACE_DETAIL=1); per-piece and per-sync checks need LLAMA_TRACE_DETAIL=2")
    c.check(bool(meta), "metadata record present")
    c.check(meta.get("dropped", 0) == 0, "no dropped records", f"dropped {meta.get('dropped')}, emitted {meta.get('records_emitted')}")
    c.check(meta.get("ring_drops", 0) == 0, "no dropped residency requests", f"{meta.get('ring_drops')}")
    rows = set(tr.rows)
    bad_rows = [r for r in rows if not (r in ("token", "GPU") or r.split(" ")[0] in ("worker", "load", "copy") and r.split(" ")[-1].isdigit())]
    c.check(not bad_rows and "token" in rows and any(r.startswith("worker ") for r in rows) and any(r.startswith("copy ") for r in rows),
            "rows token / GPU / worker N / load N / copy N", f"{sorted(rows)[:8]}..., unexpected {bad_rows}")
    overlap = []
    for r, v in tr.rows.items():
        v = sorted(v, key=t0)
        overlap += [r for a, b in zip(v, v[1:]) if t0(b) < t1(a) - 1e-3][:1]
    c.check(not overlap, "no overlap within a row", f"{overlap}")
    def code(v, n):
        out = ""
        for _ in range(n):
            out = chr(ord("a") + v % 26) + out
            v //= 26
        return out

    def name_ok(name, layer_only, args=None):
        # "L<layer> e<expert> <matrix> <4 letters>" / "L<layer> <2 letters>": the letters encode the numbers (Perfetto
        # colors by the name without digits), so they must match them, and the numbers and matrix match the args
        m = re.fullmatch(r"L(\d+) ([a-z]{2})", name) if layer_only else re.fullmatch(r"L(\d+) e(\d+) (gate|up|down|other) ([a-z]{4})", name)
        if not m:
            return False
        if layer_only:
            return m.group(2) == code(int(m.group(1)), 2)
        if args is not None:
            matrix = args.get("matrix") or kind_of(args.get("tensor", ""))
            if (int(m.group(1)), int(m.group(2)), m.group(3)) != (args["layer"], args["expert"], matrix):
                return False
            if "tensor" in args and kind_of(args["tensor"]) != matrix:
                return False
        return m.group(4) == code(int(m.group(1)) * 1024 + int(m.group(2)), 4)

    names_ok = all(name_ok(e["name"], False, e["args"]) for e in tr.worker_experts + tr.loads) and \
        all(name_ok(e["name"], True) for e in tr.worker_dense) and \
        all(e["name"] == "output" or name_ok(e["name"], True) for e in tr.gpu) and \
        all(e["name"] in ("input", "output") or name_ok(e["name"], True) for e in tr.copies_s)
    c.check(names_ok, "span names exactly L<layer> e<expert> <matrix> <code> / L<layer> <code> (input / output for copies, "
            "output on the GPU), matching their args")

    n_tok = args.tokens
    if n_tok is None:
        n_tok = meta.get("config", {}).get("tokens") or meta.get("tokens_traced")
    steps = [t["args"]["step"] for t in tr.tokens]
    c.check(len(tr.tokens) == n_tok, "TOKEN count", f"{len(tr.tokens)}, expected {n_tok}")
    c.check(steps == list(range(len(steps))), "consecutive steps", f"{steps[:4]}...")
    pos = [t["args"]["pos"] for t in tr.tokens]
    c.check(all(b == a + 1 for a, b in zip(pos, pos[1:])), "consecutive positions", f"{pos[:4]}...")
    want_exp = args.moe_layers * 3 * args.n_used
    first = t0(tr.tokens[0]) if tr.tokens else 0.0
    early = [e for e in tr.events if e.get("cat") == "simple" and t0(e) < first - 1e-3]
    c.check(not early, "nothing before the first TOKEN (prompt and skipped steps not traced)", f"{len(early)}")

    ex_by = defaultdict(list)
    for e in tr.worker_experts:
        ex_by[e["args"]["step"]].append(e)
    load_names_all = {e["name"] for e in tr.loads}
    comp_names_all = {e["name"] for e in tr.worker_experts}
    for i, tok in enumerate(tr.tokens):
        a = tok["args"]
        s = a["step"]
        nxt = tr.tokens[i + 1]["args"]["token"] if i + 1 < len(tr.tokens) else None
        out_ok = a.get("n_out_sync") == 1 and a.get("out_sync_ts", -1) >= t1(tok) - 1e-3
        smp_ok = a.get("n_sample") == 1 and a.get("sample_ts", -1) >= a.get("out_sync_ts", 0) + a.get("out_sync_dur", 0) - 1e-3
        c.check(out_ok, f"step {s}: one output SYNC after the TOKEN", f"{a.get('n_out_sync')}")
        c.check(smp_ok, f"step {s}: one SAMPLE after the output SYNC", f"{a.get('n_sample')}")
        c.check(a["token_out"] == a.get("sample_token") and (nxt is None or a["token_out"] == nxt),
                f"step {s}: sampled token stamped and is the next TOKEN's input", f"{a['token_out']} -> {nxt}")

        ex = ex_by.get(s, [])
        keys = Counter((e["args"]["layer"], e["args"]["expert"], kind_of(e["args"]["tensor"])) for e in ex)
        n_threads = ex[0]["args"]["n_threads"] if ex else 0
        sets = defaultdict(set)
        for (l, x, k) in keys:
            sets[(l, k)].add(x)
        layers = sorted({l for l, _ in sets})
        bad_l = [l for l in layers if any(len(sets[(l, k)]) != args.n_used for k in ("gate", "up", "down"))]
        bad_same = [l for l in layers if not (sets[(l, "gate")] == sets[(l, "up")] == sets[(l, "down")])]
        c.check(len(keys) == want_exp and a["n_experts"] == want_exp, f"step {s}: EXPERT count", f"{len(keys)} in the rows, {a['n_experts']} counted, expected {want_exp}")
        c.check(all(n == n_threads for n in keys.values()) and a["n_expert_threads"] == want_exp * n_threads,
                f"step {s}: one span per thread and expert matrix", f"{a['n_expert_threads']}, {n_threads} threads")
        per_thread = Counter()
        for w, v in tr.workers.items():
            per_thread[w] = len({e["name"] for e in v if e["args"]["step"] == s and "expert" in e["args"]})
        c.check(len(per_thread) >= n_threads and all(n == want_exp for w, n in per_thread.items() if n),
                f"step {s}: {want_exp} distinct expert matrix names per thread", f"{sorted(set(per_thread.values()))}")
        c.check(len(layers) == args.moe_layers and not bad_l, f"step {s}: {args.n_used} distinct experts per MoE layer and tensor", f"bad {bad_l[:5]}")
        c.check(not bad_same, f"step {s}: same expert set on gate/up/down", f"bad layers {bad_same[:5]}")
        res = {}
        for e in ex:
            if isinstance(e["args"].get("resident_frac"), (int, float)):
                res[(e["args"]["layer"], e["args"]["expert"], kind_of(e["args"]["tensor"]))] = e["args"]["sample_delay_ns"]
        c.check(a["n_residency"] == want_exp, f"step {s}: EXPERT_RESIDENCY count", f"{a['n_residency']}")
        c.check(len(keys) - len(res) <= 0.01 * max(1, len(keys)), f"step {s}: EXPERT not sampled < 1%", f"{len(keys) - len(res)}")
        if res:
            med = statistics.median(res.values()) / 1000.0
            c.check(med < 200.0, f"step {s}: median residency sample delay < 200 us", f"{med:.1f} us")
        if meta.get("moe_predictor", 1) != 0:
            c.check(a["n_predict"] == args.moe_layers, f"step {s}: PREDICT count", f"{a['n_predict']}")
        bytes_of = {}
        for e in ex:
            bytes_of[(e["args"]["layer"], e["args"]["expert"], kind_of(e["args"]["tensor"]))] = e["args"]["bytes"]
        want_pieces = sum(-(-b // PIECE) for b in bytes_of.values())
        c.check(a["n_pieces_prefetch"] == want_pieces, f"step {s}: prefetch pieces", f"{a['n_pieces_prefetch']}, expected {want_pieces}")

        g = [e for e in tr.gpu if e["args"]["step"] == s]
        gl = {e["args"]["layer"] for e in g}
        missing = [l for l in range(args.gpu_layers) if l not in gl]
        c.check(not missing, f"step {s}: GPU spans for every layer 0..{args.gpu_layers - 1}", f"missing {missing}")
        c.check(any(e["name"] == "output" for e in g), f"step {s}: GPU span for the output")
        wrong = [e["name"] for e in g if e["args"]["layer"] >= args.gpu_layers and e["args"].get("n_weight_ops", 0) > 0]
        c.check(not wrong, f"step {s}: no op with a weight on the GPU for CPU layers", f"{wrong[:3]}")
        if s == tr.tokens[0]["args"]["step"]:
            cpu_l = sorted(l for l in gl if l >= args.gpu_layers)
            c.info(f"step {s}: GPU spans of CPU layers", f"{sum(e['args'].get('n_ops', 1) for e in g if e['args']['layer'] >= args.gpu_layers)} ops in {len(cpu_l)} layers")

        cp = [e for e in tr.copies_s if e["args"]["step"] == s]
        cbytes = sum(e["args"]["bytes"] for e in cp)
        c.check(len(cp) == a["n_copies"] and cbytes == a["copy_bytes"], f"step {s}: copy row == COPY + XFER of the step",
                f"{len(cp)} spans / {cbytes} bytes, counted {a['n_copies']} / {a['copy_bytes']}")
        big = [e["args"]["tensor"] for e in cp if is_weight(e["args"]) and e["args"]["bytes"] > (1 << 20)]
        c.check(not big, f"step {s}: no weight matrix (> 1 MiB) copied to a device", f"{big[:3]}")
        if s == tr.tokens[0]["args"]["step"]:
            small = [e for e in cp if is_weight(e["args"])]
            c.info(f"step {s}: copies", f"{len(cp)} ({cbytes} bytes), of which small weights {len(small)} "
                   f"({sum(e['args']['bytes'] for e in small)} bytes, {sum(e['dur'] for e in small) / 1000:.2f} ms)")
        host = [e for e in ex + [e for e in tr.worker_dense if e["args"]["step"] == s] + cp]
        outside = sum(1 for e in host if t0(e) < t0(tok) - 1e-3 or t1(e) > t1(tok) + 1e-3)
        c.check(outside == 0, f"step {s}: worker and copy spans inside the TOKEN span", f"{outside} outside")
        # any worker: worker 0 waits at the barrier of a MUL_MAT_ID while other threads finish their experts
        wk = [(t0(e), t1(e)) for v in tr.workers.values() for e in v if e["args"]["step"] == s]
        iv = wk + [(t0(e), t1(e)) for e in cp] + [(t0(e), t1(e)) for e in g]
        covered = union_len(iv, t0(tok), t1(tok))
        total = tok["dur"] + a.get("out_sync_dur", 0.0)
        # the share of the token in which some CPU worker computes (worker 0's merged runs of a layer's other ops
        # include the barrier waits between them), a copy runs or the GPU runs (device time)
        cov = (covered + a.get("out_sync_dur", 0.0)) / total if total > 0 else 0.0
        c.check(cov >= 0.90, f"step {s}: coverage (workers + copies + GPU device time) of TOKEN + output SYNC",
                f"{100 * cov:.1f}%, unaccounted {(total - covered - a.get('out_sync_dur', 0.0)) / 1000:.2f} ms; host sync waits {a['sync_ms']:.2f} ms")
        c.check(a.get("n_split", 0) > 0 and a.get("n_sync", 0) > 0, f"step {s}: SPLIT / SYNC counted", f"{a.get('n_split')} splits, {a.get('n_sync')} syncs")
        n_threads_m = meta.get("n_threads", 14) or 14
        n_pieces = a["n_pieces_prefetch"] + a["n_pieces_predict_high"] + a["n_pieces_predict_low"]
        cost_us = (2 * n_pieces * COST_READ_BYTES_US + 4 * a["n_expert_threads"] * COST_CLOCK_US) / n_threads_m + 2 * a["n_gpu_ops"] * COST_CUDA_EVENT_US
        c.check(cost_us < 0.05 * total, f"step {s}: estimated trace overhead < 5%", f"{cost_us / 1000:.2f} ms of {total / 1000:.2f} ms")

    # a load is a hint or the prefetch of a node: a prefetched expert is computed in its step by definition, a
    # predictor guess may be wrong (then nothing computes it: a wasted read)
    comp = defaultdict(set)
    for e in tr.worker_experts:
        comp[e["args"]["step"]].add(e["name"])
    not_comp = [e for e in tr.loads if e["name"] not in comp.get(e["args"]["step"], set())]
    bad_pf = [e["name"] for e in not_comp if "prefetch" in e["args"].get("sources", "")]
    flag_bad = [e["name"] for e in tr.loads if "computed" in e["args"] and e["args"]["computed"] != (e["name"] in comp.get(e["args"]["step"], set()))]
    c.check(not bad_pf, "every load with source prefetch has a worker span of the same (layer, expert, matrix) name in its step",
            f"{len(bad_pf)}, e.g. {bad_pf[:3]}")
    c.check(not flag_bad, "load 'computed' flags match the worker rows", f"{len(flag_bad)}, e.g. {flag_bad[:3]}")
    by_src = Counter(e["args"].get("sources", "?") for e in not_comp)
    c.info("loads not computed in their step (wrong guesses)", f"{len(not_comp)} of {len(tr.loads)}: {dict(by_src)}")
    unres = sum(1 for e in tr.loads if e["args"].get("unresolved"))
    c.info("loads", f"{len(tr.loads)} spans in {sum(1 for r in tr.rows if r.startswith('load '))} rows, {unres} unresolved; "
           f"{len(load_names_all & comp_names_all)} names also on worker rows; late records {meta.get('late_records', 0)}")


def perfetto_simple(tr, args, c):
    try:
        from perfetto.trace_processor import TraceProcessor  # type: ignore
    except ImportError:
        c.info("perfetto python module not installed: trace_processor checks skipped (pip install perfetto)")
        return
    tp = TraceProcessor(trace=args.trace)
    q = lambda sql: list(tp.query(sql))  # noqa: E731
    n_async = q("select count(*) as n from track where type like '%async%'")[0].n
    c.check(n_async == 0, "trace_processor: no async tracks", f"{n_async}")
    rows = q("select t.name as tname, p.name as pname, count(*) as n from slice s join thread_track tt on s.track_id = tt.id "
             "join thread t using (utid) left join process p using (upid) group by 1, 2")
    got = {r.tname: r.n for r in rows if r.pname == "simple"}
    other = [(r.tname, r.pname) for r in rows if r.pname != "simple"]
    want = {r: len(v) for r, v in tr.rows.items()}
    c.check(got == want and not other, "trace_processor: every span on its row of process 'simple'",
            f"{len(got)} rows ({sum(got.values())} slices), json {len(want)} rows ({sum(want.values())}); other {other[:3]}")
    tp.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--tokens", type=int, default=None,
                    help="expected traced tokens (default: LLAMA_TRACE_TOKENS from the metadata config; when that is 0 = unlimited, tokens_traced)")
    ap.add_argument("--gpu-layers", type=int, default=11, help="dense layers on the GPU (run-server.sh GPU_LAYERS)")
    ap.add_argument("--moe-layers", type=int, default=92)
    ap.add_argument("--first-moe", type=int, default=1)
    ap.add_argument("--n-used", type=int, default=16)
    args = ap.parse_args()

    tr = Trace(args.trace)
    c = Checker()
    meta = tr.meta
    if tr.layout == "simple":
        check_simple(tr, args, c)
        perfetto_simple(tr, args, c)
        print(f"\n{'FAILED' if c.n_fail else 'OK'}: {c.n_fail} check(s) failed")
        return 1 if c.n_fail else 0

    c.check(bool(meta), "metadata record present")
    c.check(meta.get("dropped", 0) == 0, "no dropped records", f"dropped {meta.get('dropped')}, emitted {meta.get('records_emitted')}")
    c.check(meta.get("ring_drops", 0) == 0, "no dropped residency requests", f"{meta.get('ring_drops')}")
    if not meta.get("residency_valid", True):
        c.info("residency_valid is false: the weight files are neither owned nor writable, mincore sees only mapped pages")

    # TOKEN spans
    n_tok = args.tokens
    if n_tok is None:
        n_tok = meta.get("config", {}).get("tokens") or meta.get("tokens_traced")
    steps = [t["args"]["step"] for t in tr.tokens]
    c.check(len(tr.tokens) == n_tok, "TOKEN count", f"{len(tr.tokens)}, expected {n_tok}")
    c.check(steps == list(range(len(steps))), "consecutive steps", f"{steps[:4]}...")
    pos = [t["args"]["pos"] for t in tr.tokens]
    c.check(all(b == a + 1 for a, b in zip(pos, pos[1:])), "consecutive positions", f"{pos[:4]}...")

    # output SYNC and SAMPLE
    bad_sync, bad_sample, bad_tok = [], [], []
    for i, tok in enumerate(tr.tokens):
        s = tok["args"]["step"]
        so = tr.output_sync(s)
        if len(so) != 1 or t0(so[0]) < t1(tok) - 1e-3:
            bad_sync.append((s, len(so)))
        sa = [e for e in tr.samples if e["args"]["step"] == s]
        if len(sa) != 1 or (so and t0(sa[0]) < t1(so[0]) - 1e-3):
            bad_sample.append((s, len(sa)))
        elif i + 1 < len(tr.tokens) and sa[0]["args"]["token_out"] != tr.tokens[i + 1]["args"]["token"]:
            bad_tok.append((s, sa[0]["args"]["token_out"], tr.tokens[i + 1]["args"]["token"]))
        if tok["args"]["token_out"] != (sa[0]["args"]["token_out"] if len(sa) == 1 else None):
            bad_tok.append((s, "TOKEN.token_out", tok["args"]["token_out"]))
    c.check(not bad_sync, "one output SYNC per TOKEN, after it", f"bad (step, count): {bad_sync[:5]}")
    c.check(not bad_sample, "one SAMPLE per TOKEN, after its output SYNC", f"bad (step, count): {bad_sample[:5]}")
    c.check(not bad_tok, "sampled token is the next TOKEN's input and is stamped", f"{bad_tok[:5]}")

    # step consistency
    stepped = [e for e in tr.events if isinstance(e.get("args", {}).get("step"), int)
               and e.get("cat") not in ("token", "sync", "sample", "meta")]
    outside, idle_inside, untraced_compute = 0, 0, 0
    for e in stepped:
        s = e["args"]["step"]
        if s >= 0:
            tok = tr.by_step.get(s)
            # residency samples and I/O resolution happen on other threads, possibly a little later
            if e.get("cat") in ("op", "expert_thread", "xfer", "split", "copy") and e["pid"] == PID_HOST:
                if tok is None or t0(e) < t0(tok) - 1e-3 or t1(e) > t1(tok) + 1e-3:
                    outside += 1
        else:
            if e["ph"] == "X" and e["pid"] == PID_HOST and tr.step_of_time(t0(e)) is not None and tr.step_of_time(t1(e)) == tr.step_of_time(t0(e)):
                idle_inside += 1
            if e.get("cat") in ("op", "expert", "expert_thread") or e["args"].get("source") == "prefetch":
                untraced_compute += 1
    c.check(outside == 0, "host events of a traced step inside its TOKEN span", f"{outside} outside")
    c.check(idle_inside == 0, "no IDLE (step -1) event inside a TOKEN span", f"{idle_inside}")
    c.check(untraced_compute == 0, "no compute / prefetch event outside a traced step (prompt not traced)", f"{untraced_compute}")
    if tr.tokens:
        first = t0(tr.tokens[0])
        early = [e for e in tr.events if e["ts"] < first - 1e-3 and (e.get("cat") in ("io", "residency", "op", "expert", "expert_thread", "predict"))]
        c.check(not early, "no I/O, residency, compute or predict event before the first TOKEN (prompt and skipped steps)",
                f"{len(early)}, e.g. {[(e.get('cat'), e.get('name')) for e in early[:3]]}")

    # EXPERT
    want_exp = args.moe_layers * 3 * args.n_used
    exp_by_step = defaultdict(list)
    for x in tr.experts:
        exp_by_step[x["step"]].append(x)
    for tok in tr.tokens:
        s = tok["args"]["step"]
        ex = exp_by_step.get(s, [])
        sets = defaultdict(set)
        for x in ex:
            sets[(x["layer"], kind_of(x["name"]))].add(x["expert"])
        layers = sorted({l for l, _ in sets})
        n_distinct_bad = [l for l in layers if any(len(sets[(l, k)]) != args.n_used for k in ("gate", "up", "down"))]
        n_same_bad = [l for l in layers if not (sets[(l, "gate")] == sets[(l, "up")] == sets[(l, "down")])]
        c.check(len(ex) == want_exp, f"step {s}: EXPERT count", f"{len(ex)}, expected {want_exp}")
        c.check(len(layers) == args.moe_layers and not n_distinct_bad, f"step {s}: {args.n_used} distinct experts per MoE layer and tensor",
                f"{len(layers)} layers, bad {n_distinct_bad[:5]}")
        c.check(not n_same_bad, f"step {s}: same expert set on gate/up/down", f"bad layers {n_same_bad[:5]}")
        partial = sum(1 for x in ex if x.get("partial"))
        c.check(partial == 0, f"step {s}: no partial EXPERT spans", f"{partial}")

        res = [r for r in tr.residency if r["args"]["step"] == s]
        not_sampled = sum(1 for x in ex if not isinstance(x.get("resident_frac"), (int, float)))
        c.check(len(res) == want_exp, f"step {s}: EXPERT_RESIDENCY count", f"{len(res)}, expected {want_exp}")
        c.check(not_sampled <= 0.01 * max(1, len(ex)), f"step {s}: EXPERT not sampled < 1%", f"{not_sampled}")
        delays = [x["sample_delay_ns"] for x in ex if isinstance(x.get("sample_delay_ns"), (int, float))]
        if delays:
            med = statistics.median(delays) / 1000.0
            c.check(med < 200.0, f"step {s}: median residency sample delay < 200 us", f"{med:.1f} us")

        # an earlier I/O issue for each expert slice in the same token (the prefetch of its own node, or a hint)
        issued = defaultdict(lambda: float("inf"))
        for i in tr.issues:
            a = i["args"]
            if a["step"] == s or (a["step"] == -1 and t0(i) <= t1(tok)):
                key = (a["layer"], a["expert"], kind_of(a["name"]))
                issued[key] = min(issued[key], t0(i))
        late = [(x["layer"], x["expert"]) for x in ex if issued[(x["layer"], x["expert"], kind_of(x["name"]))] > x["t0"] + 1e-3]
        c.check(not late, f"step {s}: every EXPERT has an earlier IO_ISSUE", f"{len(late)} without, e.g. {late[:3]}")

        if meta.get("moe_predictor", 1) != 0:
            npred = sum(1 for p in tr.predicts if p["args"]["step"] == s)
            c.check(npred == args.moe_layers, f"step {s}: PREDICT count", f"{npred}, expected {args.moe_layers}")

        # prefetch issues: per piece X events, per slice spans at detail 1
        # prefetch pieces: exactly ceil(slice bytes / 2 MiB) per used expert slice, the slice size from its EXPERT
        pf = [i for i in tr.issues if i["args"]["step"] == s and i["args"]["source"] == "prefetch"]
        got_pieces = Counter((i["args"]["layer"], i["args"]["expert"], kind_of(i["args"]["name"])) for i in pf)
        want = {(x["layer"], x["expert"], kind_of(x["name"])): -(-x["bytes"] // PIECE) for x in ex}
        bad = [k for k in set(want) | set(got_pieces) if got_pieces.get(k, 0) != want.get(k, 0)]
        c.check(not bad and len(pf) == sum(want.values()), f"step {s}: prefetch pieces",
                f"{len(pf)} pieces, expected {sum(want.values())} for {len(want)} slices; mismatched slices {sorted(bad)[:3]}")
        if tr.detail == 1:  # legacy detailed trace at detail 1 (before the simple layout): one span per slice
            spans = [a for a in tr.io_spans if a["step"] == s and a["source"] == "prefetch"]
            c.check(len(spans) == want_exp, f"step {s}: prefetch slice spans on the I/O track", f"{len(spans)}")

        # GPU: the dense layers < gpu-layers run there; for the CPU layers only the weight-free ops whose inputs are
        # the KV / recurrent caches (on CUDA0 for every layer) may, never a matmul of a weight and never the experts
        g_ops = [e for e in tr.ops_gpu if e["args"]["step"] == s]
        gl = Counter(e["args"]["layer"] for e in g_ops if not e["args"].get("layer_inferred"))
        names = {e["args"]["name"] for e in g_ops}
        missing = [l for l in range(args.gpu_layers) if gl.get(l, 0) == 0]
        c.check(not missing, f"step {s}: OP_GPU for every layer 0..{args.gpu_layers - 1}", f"missing {missing}")
        c.check("result_output" in names, f"step {s}: OP_GPU for the output", "")
        wrong = [(e["args"]["name"], e["args"]["op"]) for e in g_ops
                 if e["args"]["layer"] >= args.gpu_layers and e["args"].get("weight")
                 and not e["args"]["name"].startswith("result_")]
        c.check(not wrong, f"step {s}: no op with a weight on the GPU for CPU layers", f"{len(wrong)}, e.g. {wrong[:3]}")
        mmid = [e["args"]["name"] for e in g_ops if e["args"]["op"] == "MUL_MAT_ID"]
        c.check(not mmid, f"step {s}: no expert matmul on the GPU", f"{mmid[:3]}")
        # small CPU-layer weights (norm vectors of the cache ops above) are copied per token; a matrix must not be
        wcopy = [e for e in tr.copies if e["args"]["step"] == s and is_weight(e["args"])]
        big = [e["args"]["name"] for e in wcopy if e["args"]["bytes"] > (1 << 20)]
        c.check(not big, f"step {s}: no weight matrix (> 1 MiB) copied to a device", f"{len(big)}, e.g. {big[:3]}")
        if s == tr.tokens[0]["args"]["step"] and wcopy:
            kinds = Counter(e["args"]["name"].split(".")[2] if e["args"]["name"].count(".") >= 2 else e["args"]["name"] for e in wcopy)
            c.info(f"step {s}: small weights copied to a device per token",
                   f"{len(wcopy)} copies, {sum(e['args']['bytes'] for e in wcopy)} bytes, "
                   f"{sum(e.get('dur', 0.0) for e in wcopy) / 1000:.2f} ms: {dict(kinds)}")
        if s == tr.tokens[0]["args"]["step"]:
            cpu_l = sorted(l for l in gl if l >= args.gpu_layers)
            mix = Counter(e["args"]["op"] for e in g_ops if e["args"]["layer"] >= args.gpu_layers)
            c.info(f"step {s}: GPU ops of CPU layers", f"{sum(mix.values())} launches in {len(cpu_l)} layers "
                   f"({cpu_l[:1]}..{cpu_l[-1:]}), ops {dict(mix.most_common(8))}")
        nsplit = sum(1 for e in tr.splits if e["args"]["step"] == s)
        nsync = sum(1 for e in tr.syncs if e["args"]["step"] == s and not e["args"].get("output"))
        ncopy = sum(1 for e in tr.copies if e["args"]["step"] == s)
        c.check(nsplit > 0 and nsync > 0, f"step {s}: SPLIT / SYNC events", f"{nsplit} splits, {nsync} syncs, {ncopy} copies")

        # XFER order
        ops = [e for e in tr.ops_cpu if e["args"]["step"] == s] + [e for e in tr.launches if e["args"]["step"] == s]
        xin = [e for e in tr.xfers if e["args"]["step"] == s and e["args"]["direction"] == "input"]
        xout = [e for e in tr.xfers if e["args"]["step"] == s and e["args"]["direction"] == "output"]
        sp = [e for e in tr.splits if e["args"]["step"] == s]
        if ops:
            first_op = min(t0(e) for e in ops)
            c.check(bool(xin) and min(t0(e) for e in xin) <= first_op, f"step {s}: XFER input before the first op")
        if sp:
            last_split = max(t1(e) for e in sp)
            c.check(bool(xout) and max(t0(e) for e in xout) >= last_split - 1e-3, f"step {s}: XFER output after the last split")

        # coverage of the host time of the token
        so = tr.output_sync(s)
        lo, hi = t0(tok), t1(tok)
        iv = [(t0(e), t1(e)) for e in tr.ops_cpu if e["args"]["step"] == s and e["tid"] == TID_WORKER0]
        iv += [(t0(e), t1(e)) for e in tr.syncs if e["args"]["step"] == s and not e["args"].get("output")]
        iv += [(t0(e), t1(e)) for e in tr.copies if e["args"]["step"] == s and e["ph"] == "X"]
        iv += [(t0(e), t1(e)) for e in tr.splits if e["args"]["step"] == s and e["args"]["backend"] != "CPU"]
        iv += [(t0(e), t1(e)) for e in tr.xfers if e["args"]["step"] == s]
        covered = union_len(iv, lo, hi)
        out_sync = sum(e.get("dur", 0.0) for e in so)
        total = (hi - lo) + out_sync
        cov = (covered + out_sync) / total if total > 0 else 0.0
        c.check(cov >= 0.90, f"step {s}: coverage of TOKEN + output SYNC", f"{100 * cov:.1f}%, unaccounted {(total - covered - out_sync) / 1000:.2f} ms of {total / 1000:.2f} ms")

        # overhead estimate from counted calls
        n_pieces = sum(1 for i in tr.issues if i["args"]["step"] == s)
        n_clock = sum(1 for e in tr.events if e.get("args", {}).get("step") == s) * 2 + 4 * len(ex) * max(1, ex[0]["n_threads"] if ex else 1)
        n_cuda = 2 * sum(1 for e in tr.ops_gpu if e["args"]["step"] == s)
        n_threads = meta.get("n_threads", 14) or 14
        cost_us = (2 * n_pieces * COST_READ_BYTES_US) / n_threads + (n_clock * COST_CLOCK_US) / n_threads + n_cuda * COST_CUDA_EVENT_US
        c.check(cost_us < 0.05 * total, f"step {s}: estimated trace overhead < 5%", f"{cost_us / 1000:.2f} ms of {total / 1000:.2f} ms")

    # I/O resolution; tr.detail == 1 with this (detailed) layout: a legacy trace written before the simple layout
    # became the default (one span per slice instead of per piece)
    if tr.detail == 1:
        no_end = sum(1 for a in tr.io_spans if a["t1"] is None)
        dup_id = sum(n - 1 for n in Counter(a["id"] for a in tr.io_spans).values())
        c.check(no_end == 0 and dup_id == 0, "every I/O slice span ends exactly once", f"{no_end} open, {dup_id} repeated ids")
    else:
        nondup = [i for i in tr.issues if not i["args"]["duplicate"]]
        cookies = Counter(a["id"] for a in tr.io_spans if a["t1"] is not None)
        bad = [i["args"]["cookie"] for i in nondup if cookies.get(i["args"]["cookie"], 0) != 1]
        c.check(not bad, "every non-duplicate issue resolved exactly once", f"{len(bad)} not, of {len(nondup)}")
    unresolved = sum(1 for a in tr.io_spans if a["end"] and a["end"].get("unresolved") or (a["end"] or {}).get("resolution") == "unresolved")
    c.info("unresolved I/O spans", f"{unresolved} of {len(tr.io_spans)}")

    # Perfetto: the spans that may overlap must land on lanes (thread tracks) of their process, not on the global
    # async tracks that "b"/"e" events become, and every category must import completely
    try:
        from perfetto.trace_processor import TraceProcessor  # type: ignore
    except ImportError:
        c.info("perfetto python module not installed: trace_processor checks skipped (pip install perfetto)")
    else:
        tp = TraceProcessor(trace=args.trace)

        def q(sql):
            return list(tp.query(sql))

        n_async = q("select count(*) as n from slice s join track t on s.track_id = t.id "
                    "where s.category in ('expert', 'io') and t.type like '%async%'")[0].n
        c.check(n_async == 0, "trace_processor: no expert / io slice on an async track", f"{n_async}")
        rows = q("select s.category as cat, p.name as pname, count(*) as n, count(distinct tt.utid) as lanes "
                 "from slice s join thread_track tt on s.track_id = tt.id join thread using (utid) "
                 "left join process p using (upid) where s.category in ('expert', 'io') group by 1, 2")
        by = {(r.cat, r.pname): (r.n, r.lanes) for r in rows}
        want_proc = {"expert": "expert compute", "io": "expert I/O"}
        for cat, pname in want_proc.items():
            n_json = sum(1 for e in tr.events if e.get("cat") == cat)
            n_on, lanes = by.get((cat, pname), (0, 0))
            elsewhere = sum(n for (k, p), (n, _) in by.items() if k == cat and p != pname)
            c.check(n_on == n_json and elsewhere == 0, f"trace_processor: all {cat} slices on the '{pname}' process",
                    f"{n_on} there, {elsewhere} elsewhere, {n_json} in the JSON; {lanes} lanes")
        exp_lanes = by.get(("expert", "expert compute"), (0, 0))[1]
        io_lanes = by.get(("io", "expert I/O"), (0, 0))[1]
        c.check(exp_lanes <= 3 * args.n_used, "trace_processor: expert compute lanes <= 3 x n-used", f"{exp_lanes}")
        c.check(io_lanes <= 512, "trace_processor: expert I/O lanes bounded (<= 512)", f"{io_lanes}")
        per_track = q("select t.name as tname from slice s join thread_track tt on s.track_id = tt.id join thread t using (utid) "
                      "where s.category = 'io' group by t.name")
        kinds = Counter(r.tname.rsplit(" ", 1)[0] for r in per_track)
        c.info("expert I/O lanes per track", f"{dict(kinds)}")

        # every category imports completely: complete and instant events become slices (global instants included)
        json_cnt = Counter(e.get("cat") for e in tr.events if e["ph"] in ("X", "i"))
        json_cnt["meta"] += 1 if meta else 0
        tp_cnt = {r.cat: r.n for r in q("select category as cat, count(*) as n from slice group by 1")}
        bad = {k: (v, tp_cnt.get(k, 0)) for k, v in json_cnt.items() if tp_cnt.get(k, 0) != v}
        bad.update({k: (0, v) for k, v in tp_cnt.items() if k not in json_cnt})
        c.check(not bad, "trace_processor: slices per category == JSON events", f"mismatch (json, tp): {bad}" if bad else f"{dict(json_cnt)}")
        tp.close()

    print(f"\n{'FAILED' if c.n_fail else 'OK'}: {c.n_fail} check(s) failed")
    return 1 if c.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
