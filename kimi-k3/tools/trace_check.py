#!/usr/bin/env python3
"""Check a decode trace (LLAMA_TRACE / run-server.sh TRACE=) of Kimi-K3 against what a decode step must contain.

The trace is the Chrome Trace Event JSON written by ggml-trace.cpp (one event per line, loads in ui.perfetto.dev).
Each check prints PASS / FAIL with the numbers behind it; the exit status is 1 if any check failed.

  python3 -I trace_check.py /tmp/k3.json [--tokens 8] [--gpu-layers 11] [--moe-layers 92] [--n-used 16]

Checks (per traced token unless noted):
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
  * with the python 'perfetto' module: trace_processor sees as many slices as there are X + b (+ instant) events
"""

import argparse
import json
import os
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
    for e in events:
        if e.get("ph") == "M":
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
        self.issues = [e for e in self.events if e.get("cat") == "io" and e["ph"] == "X"]
        self.residency = [e for e in self.events if e.get("cat") == "residency"]
        self.expert_threads = [e for e in self.events if e.get("cat") == "expert_thread"]
        self.detail = self.meta.get("config", {}).get("detail", 1)

        # EXPERT spans: b args + e time
        e_ts = {e["id"]: e["ts"] for e in self.events if e.get("cat") == "expert" and e["ph"] == "e"}
        self.experts = []
        for e in self.events:
            if e.get("cat") == "expert" and e["ph"] == "b":
                a = dict(e["args"])
                a["t0"] = e["ts"]
                a["t1"] = e_ts.get(e["id"])
                self.experts.append(a)

        # I/O spans on the expert I/O tracks: b args + e args/time (per slice at detail 1, per piece at detail 2)
        io_e = {}
        for e in self.events:
            if e.get("cat") == "io" and e["pid"] == PID_IO and e["ph"] == "e":
                io_e[e["id"]] = e
        self.io_spans = []
        for e in self.events:
            if e.get("cat") == "io" and e["pid"] == PID_IO and e["ph"] == "b":
                a = dict(e["args"])
                a["t0"] = e["ts"]
                end = io_e.get(e["id"])
                a["t1"] = end["ts"] if end else None
                a["end"] = end["args"] if end else None
                a["id"] = e["id"]
                self.io_spans.append(a)
        self.io_e_unmatched = len(io_e) - len({a["id"] for a in self.io_spans} & set(io_e))

    def step_of_time(self, ts):
        """The traced step whose TOKEN span contains ts, else None."""
        for tok in self.tokens:
            if t0(tok) <= ts <= t1(tok):
                return tok["args"]["step"]
        return None

    def output_sync(self, step):
        return [e for e in self.syncs if e["args"].get("output") and e["args"]["step"] == step]


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
            if e["ph"] == "X" and tr.step_of_time(t0(e)) is not None and tr.step_of_time(t1(e)) == tr.step_of_time(t0(e)):
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
        if tr.detail == 1:
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
        wcopy = [e for e in tr.copies if e["args"]["step"] == s and ".weight" in e["args"]["name"]]
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

    # I/O resolution
    if tr.detail == 1:
        no_end = sum(1 for a in tr.io_spans if a["t1"] is None)
        c.check(no_end == 0 and tr.io_e_unmatched == 0, "every I/O slice span ends exactly once", f"{no_end} open, {tr.io_e_unmatched} unmatched ends")
    else:
        nondup = [i for i in tr.issues if not i["args"]["duplicate"]]
        cookies = Counter(a["id"] for a in tr.io_spans if a["t1"] is not None)
        bad = [i["args"]["cookie"] for i in nondup if cookies.get(i["args"]["cookie"], 0) != 1]
        c.check(not bad, "every non-duplicate issue resolved exactly once", f"{len(bad)} not, of {len(nondup)}")
    unresolved = sum(1 for a in tr.io_spans if a["end"] and a["end"].get("unresolved") or (a["end"] or {}).get("resolution") == "unresolved")
    c.info("unresolved I/O spans", f"{unresolved} of {len(tr.io_spans)}")

    # Perfetto
    try:
        from perfetto.trace_processor import TraceProcessor  # type: ignore
    except ImportError:
        c.info("perfetto python module not installed: trace_processor check skipped (pip install perfetto)")
    else:
        # instants are imported as zero-length slices too (the metadata record included)
        n_x_b = sum(1 for e in tr.events if e["ph"] in ("X", "b"))
        n_i = sum(1 for e in tr.events if e["ph"] == "i") + (1 if meta else 0)
        tp = TraceProcessor(trace=args.trace)
        n_slice = next(iter(tp.query("select count(*) as n from slice"))).n
        tp.close()
        c.check(n_slice == n_x_b + n_i, "trace_processor slices == X + b + i events", f"{n_slice} vs {n_x_b} X/b + {n_i} i")

    print(f"\n{'FAILED' if c.n_fail else 'OK'}: {c.n_fail} check(s) failed")
    return 1 if c.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
