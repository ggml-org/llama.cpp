#!/usr/bin/env python3
"""Decode matmul times of Kimi-K3 from a trace (LLAMA_TRACE): which weight takes how long on the CPU and on the GPU.
Part A of PLAN-cpu-gpu-matmul.md. Stdlib only.

  python3 -I trace_matmul.py TRACE.json [--model /mnt/k3/kimi-k3-iq1] [-o out/partA.md]

Weight bytes and types come from the GGUF headers (read only, O_RDONLY, no tensor data).
A detailed trace (LLAMA_TRACE_DETAIL=2) gives all tables. A simple trace (detail 1) has no per-op records: it gives
A3 and a reduced A4 only.

Pairing (detailed trace): per step, all MUL_MAT ops of worker 0 (CPU) and of the device (GPU) in time order are cut
into layers at the named ops ffn_out-0 / ffn_shexp-L. Each layer must match the graph of src/models/kimi-k3.cpp
op by op, with the named ops at fixed ranks (kda_out / mla_out / prefix_sum_attn, ffn_moe_logits, fattn_mla,
ffn_shexp), else the script stops. "pred" is the next layer's router applied by the MoE predictor hint.
"""

import argparse
import glob
import json
import os
import statistics
import struct
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trace_check import TID_WORKER0, Trace, union_len  # noqa: E402

PID_GPU = 2
MATMUL_REF_S = 1.67  # plan: CPU dense MUL_MAT + MUL_MAT_ID compute of step 0 of the d2 trace
PRED = "pred"

# op order of one layer in the graph (src/models/kimi-k3.cpp); None in a slot = unnamed op
KDA_ATTN = ["attn_q", "attn_k", "attn_v", "ssm_f_a", "ssm_f_b", "ssm_beta", "ssm_g", "attn_output"]
MLA_ATTN = ["attn_q_a", "attn_q_b", "attn_kv_a_mqa", "attn_k_b", "attn_v_b", "attn_gate", "attn_output"]
MOE_FFN = ["ffn_gate_inp", PRED, "ffn_routed_down", "ffn_routed_up", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"]
DENSE_FFN = ["ffn_gate", "ffn_up", "ffn_down"]

# ggml type id -> (name, block size, type size)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
}


def fail(msg):
    sys.exit(f"trace_matmul: FAIL: {msg}")


# ---------------------------------------------------------------- GGUF header (stdlib)

def _rd(f, fmt):
    fmt = "<" + fmt
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def _rstr(f):
    (n,) = _rd(f, "Q")
    return f.read(n).decode("utf-8", "replace")


_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _rval(f, t):
    if t in _SCALAR:
        return _rd(f, _SCALAR[t])[0]
    if t == 8:
        return _rstr(f)
    if t == 9:
        at, n = _rd(f, "IQ")
        if at in _SCALAR:
            return list(_rd(f, "%d%s" % (n, _SCALAR[at])))
        return [_rval(f, at) for _ in range(n)]
    raise ValueError(f"gguf value type {t}")


def read_gguf(path):
    with os.fdopen(os.open(path, os.O_RDONLY), "rb") as f:
        if f.read(4) != b"GGUF":
            fail(f"{path}: not a GGUF file")
        _ver, n_t, n_kv = _rd(f, "IQQ")
        kv = {}
        for _ in range(n_kv):
            k = _rstr(f)
            (t,) = _rd(f, "I")
            kv[k] = _rval(f, t)
        tensors = {}
        for _ in range(n_t):
            name = _rstr(f)
            (nd,) = _rd(f, "I")
            ne = list(_rd(f, "%dQ" % nd))
            (ty,) = _rd(f, "I")
            _rd(f, "Q")
            tensors[name] = (ne, ty)
    return kv, tensors


class Model:
    def __init__(self, model_dir):
        files = sorted(glob.glob(os.path.join(model_dir, "*.gguf")))
        if not files:
            fail(f"no *.gguf in {model_dir}")
        self.kv, self.t = {}, {}
        for p in files:
            kv, t = read_gguf(p)
            self.kv.update(kv)
            self.t.update(t)
        arch = self.kv.get("general.architecture")
        self.n_layer = self.kv[f"{arch}.block_count"]
        hkv = self.kv[f"{arch}.attention.head_count_kv"]
        self.is_mla = [h != 0 for h in hkv] if isinstance(hkv, list) else [hkv != 0] * self.n_layer

    def has(self, name):
        return name in self.t

    def bytes(self, name):
        ne, ty = self.t[name]
        if ty not in GGML_TYPES:
            fail(f"{name}: unknown ggml type {ty}")
        _, bs, ts = GGML_TYPES[ty]
        n = 1
        for x in ne[1:]:
            n *= x
        return ne[0] // bs * ts * n

    def tname(self, name):
        return GGML_TYPES[self.t[name][1]][0]

    def dense_bytes(self, il):
        """All matmul weights of a layer: 2D+ tensors except the experts and the conv kernels."""
        pre = f"blk.{il}."
        tot = 0
        for n, (ne, _ty) in self.t.items():
            if n.startswith(pre) and len(ne) >= 2 and "_exps" not in n and "conv1d" not in n:
                tot += self.bytes(n)
        return tot


# ---------------------------------------------------------------- helpers

def med(v):
    return statistics.median(v) if v else float("nan")


def p90(v):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(0.9 * (len(v) - 1) + 0.5))]


def gbs(nbytes, us):
    return nbytes / (us * 1e3) if us and us > 0 else float("nan")


def f(x, nd=1):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{x:.{nd}f}"


class Out:
    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        print(s)
        self.lines.append(s)

    def table(self, hdr, rows):
        self("| " + " | ".join(hdr) + " |")
        self("|" + "|".join("---" for _ in hdr) + "|")
        for r in rows:
            self("| " + " | ".join(str(x) for x in r) + " |")
        self()


# ---------------------------------------------------------------- pairing (detailed trace)

def template(model, il, has_pred):
    if il < model.kv.get("kimi-k3.leading_dense_block_count", 1):
        ffn = DENSE_FFN
    else:
        ffn = [w for w in MOE_FFN if has_pred or w != PRED]
    return (MLA_ATTN if model.is_mla[il] else KDA_ATTN) + ffn


def check_anchors(seg, names, il, mla):
    """Named ops must sit at their slot, else the mapping is wrong."""
    want = {}
    out_slot = names.index("attn_output")
    want[out_slot] = {f"kda_out-{il}", f"mla_out-{il}", f"prefix_sum_attn-{il}"}
    if mla:
        want[names.index("attn_v_b")] = {f"fattn_mla-{il}"}
    if "ffn_gate_inp" in names:
        want[names.index("ffn_gate_inp")] = {f"ffn_moe_logits-{il}"}
        want[len(names) - 1] = {f"ffn_shexp-{il}"}
    else:
        want[len(names) - 1] = {f"ffn_out-{il}"}
    for slot, ok in want.items():
        got = seg[slot]["name"]
        if got not in ok:
            fail(f"layer {il} step {seg[0]['args']['step']}: rank {slot} ({names[slot]}) is '{got}', expected one of {sorted(ok)}")
    # any other named op of this layer must not sit elsewhere
    for slot, e in enumerate(seg):
        n = e["name"]
        if n.endswith(f"-{il}") and slot not in want and not n.startswith("node_"):
            fail(f"layer {il}: named op '{n}' at rank {slot} ({names[slot]}) is not an anchor slot")


def pair_layers(tr, model):
    """-> per step: list of (layer, [(weight, event, device)]), output op"""
    predictor = bool(tr.meta.get("moe_predictor"))
    out = {}
    for tok in tr.tokens:
        s = tok["args"]["step"]
        mm = [e for e in tr.events if e.get("cat") == "op" and e["args"].get("step") == s and e["args"].get("op") == "MUL_MAT"
              and ((e["pid"] == 1 and e["tid"] == TID_WORKER0) or (e["pid"] == PID_GPU and "stream" in e["args"]))]
        mm.sort(key=lambda e: e["ts"])
        idx = {e["name"]: i for i, e in enumerate(mm)}
        layers = []
        lo = 0
        for il in range(model.n_layer):
            end_name = f"ffn_out-{il}" if il < model.kv.get("kimi-k3.leading_dense_block_count", 1) else f"ffn_shexp-{il}"
            if end_name not in idx:
                fail(f"step {s}: anchor {end_name} not found")
            hi = idx[end_name] + 1
            seg = mm[lo:hi]
            names = None
            for has_pred in (True, False):
                t = template(model, il, has_pred)
                if len(t) == len(seg):
                    names = t
                    break
            if names is None:
                fail(f"step {s} layer {il}: {len(seg)} MUL_MAT ops, template has {len(template(model, il, True))} (or one less without pred): "
                     + ", ".join(e["name"] for e in seg))
            check_anchors(seg, names, il, model.is_mla[il])
            if (PRED in names) != (predictor and 1 <= il < model.n_layer - 1):
                fail(f"step {s} layer {il}: pred op {'present' if PRED in names else 'missing'}")
            layers.append((il, [(w, e, "gpu" if e["pid"] == PID_GPU else "cpu") for w, e in zip(names, seg)]))
            lo = hi
        rest = mm[lo:]
        if len(rest) != 1 or rest[0]["name"] != "result_output":
            fail(f"step {s}: after the last layer expected only result_output, got {[e['name'] for e in rest]}")
        # pred runs where its weight (the next layer's router) is; routed_down where this layer's router is
        dev_logits = {il: next(d for w, _e, d in ops if w == "ffn_gate_inp") for il, ops in layers if any(w == "ffn_gate_inp" for w, _e, _d in ops)}
        for il, ops in layers:
            for w, e, d in ops:
                if w == PRED and il + 1 in dev_logits and d != dev_logits[il + 1]:
                    fail(f"step {s} layer {il}: pred op {e['name']} on {d}, next router on {dev_logits[il + 1]}")
                if w == "ffn_routed_down" and d != dev_logits[il]:
                    fail(f"step {s} layer {il}: routed_down op {e['name']} on {d}, router on {dev_logits[il]}")
        out[s] = (layers, rest[0])
    return out


def wbytes(model, il, w):
    if w == PRED:
        return model.bytes(f"blk.{il + 1}.ffn_gate_inp.weight")
    return model.bytes(f"blk.{il}.{w}.weight")


def wtype(model, il, w):
    if w == PRED:
        return model.tname(f"blk.{il + 1}.ffn_gate_inp.weight")
    return model.tname(f"blk.{il}.{w}.weight")


# ---------------------------------------------------------------- A1 / A2

def tables_a1_a2(o, tr, model, paired, res):
    n_layer = model.n_layer
    first, last = 0, n_layer - 1
    # bytes check per layer: mapped weights (without pred) == all dense weights of the layer
    for il, ops in paired[min(paired)][0]:
        mapped = sum(wbytes(model, il, w) for w, _e, _d in ops if w != PRED)
        dense = model.dense_bytes(il)
        if mapped != dense:
            fail(f"layer {il}: mapped weight bytes {mapped} != dense bytes {dense}")
    o(f"Pairing: {len(paired)} steps x {n_layer} layers matched the graph op by op; mapped weight bytes == dense bytes for every layer.")
    o()

    cpu_layers = sorted({il for il, ops in paired[min(paired)][0] if all(d == "cpu" for w, _e, d in ops if w != PRED)})
    gpu_layers = sorted({il for il, ops in paired[min(paired)][0] if all(d == "gpu" for w, _e, d in ops if w != PRED)})
    n_kda_cpu = sum(1 for il in cpu_layers if not model.is_mla[il])
    n_mla_cpu = sum(1 for il in cpu_layers if model.is_mla[il])
    o(f"CPU layers {cpu_layers[0]}-{cpu_layers[-1]}: {n_kda_cpu} KDA + {n_mla_cpu} MLA (from the trace; MLA layers: "
      f"{','.join(str(i) for i in range(n_layer) if model.is_mla[i])}). GPU layers: {gpu_layers[0]}-{gpu_layers[-1]}.")
    if (n_kda_cpu, n_mla_cpu) != (61, 21):
        o(f"NOTE: the plan assumed 61 KDA + 21 MLA CPU layers; the trace and the GGUF give {n_kda_cpu} + {n_mla_cpu} (layer {last} is MLA).")
    o()

    # per (type, rank): durations by device
    stats = {}
    for kind in ("KDA", "MLA"):
        is_mla = kind == "MLA"
        names = (MLA_ATTN if is_mla else KDA_ATTN) + MOE_FFN
        rows = {w: {"cpu": [], "gpu": [], "cpu_gbs": [], "gpu_gbs": [], "bytes": set(), "type": set()} for w in names}
        for s, (layers, _out) in paired.items():
            for il, ops in layers:
                if il in (first, last) or model.is_mla[il] != is_mla:
                    continue
                for w, e, d in ops:
                    b = wbytes(model, il, w)
                    r = rows[w]
                    r[d].append(e["dur"])
                    r[d + "_gbs"].append(gbs(b, e["dur"]))
                    r["bytes"].add(b)
                    r["type"].add(wtype(model, il, w))
        stats[kind] = rows

    o("### A1 per layer type and rank (median over steps x layers; p90 in brackets; layers 0 and last excluded)")
    o()
    for kind, rows in stats.items():
        o(f"**{kind}**")
        o()
        trows = []
        for rank, (w, r) in enumerate(rows.items()):
            b = max(r["bytes"]) if r["bytes"] else 0
            cm, gm = med(r["cpu"]), med(r["gpu"])
            name = "pred (next layer ffn_gate_inp)" if w == PRED else w
            trows.append([rank, name, "/".join(sorted(r["type"])), f(b / 1e6, 2), f"{f(cm)} ({f(p90(r['cpu']))})",
                          f"{f(gm)} ({f(p90(r['gpu']))})", f(cm / gm if gm else float('nan')), f(gbs(b, cm)), f(gbs(b, gm)),
                          f"{len(r['cpu'])}/{len(r['gpu'])}"])
            if gbs(b, cm) > 80:
                o(f"WARN: {kind} rank {rank} {w}: CPU {gbs(b, cm):.0f} GB/s is above DDR4 speed; check the rank mapping")
        o.table(["rank", "weight", "type", "MB", "CPU us", "GPU dev us", "CPU/GPU", "CPU GB/s", "GPU GB/s", "n cpu/gpu"], trows)
        res["A1"][kind] = {w: {"bytes": max(r["bytes"]) if r["bytes"] else 0, "type": "/".join(sorted(r["type"])),
                               "cpu_us_med": med(r["cpu"]), "cpu_us_p90": p90(r["cpu"]), "gpu_us_med": med(r["gpu"]),
                               "gpu_us_p90": p90(r["gpu"]), "n_cpu": len(r["cpu"]), "n_gpu": len(r["gpu"])}
                           for w, r in rows.items()}

    # layers 0 and last, alone
    o(f"**layer {first} (dense FFN) and layer {last} (no pred), alone (median over steps)**")
    o()
    trows = []
    for il in (first, last):
        per = defaultdict(list)
        for s, (layers, _out) in paired.items():
            for w, e, d in layers[il][1]:
                per[(w, d)].append(e["dur"])
        for (w, d), v in per.items():
            b = wbytes(model, il, w)
            trows.append([il, w, wtype(model, il, w), f(b / 1e6, 2), d, f(med(v)), f(gbs(b, med(v)))])
    o.table(["layer", "weight", "type", "MB", "device", "us", "GB/s"], trows)

    outs = [out for _l, out in paired.values()]
    ob = model.bytes("output.weight")
    dev = "gpu" if outs[0]["pid"] == PID_GPU else "cpu"
    om = med([e["dur"] for e in outs])
    o(f"output.weight {model.tname('output.weight')} {ob / 1e6:.1f} MB on {dev}: {om:.1f} us median ({gbs(ob, om):.1f} GB/s). "
      f"token_embd {model.tname('token_embd.weight')} {model.bytes('token_embd.weight') / 1e6:.1f} MB: GET_ROWS on the CPU, 1 row per token, not timed.")
    o()
    res["output"] = {"bytes": ob, "device": dev, "us_med": om}

    # A2
    o("### A2 per layer type: sum of the MUL_MAT medians per layer")
    o()
    trows = []
    per_tok = {"cpu": 0.0, "gpu": 0.0}
    has_pred = {il: any(w == PRED for w, _e, _d in ops) for il, ops in paired[min(paired)][0]}
    for kind, rows in stats.items():
        is_mla = kind == "MLA"
        layers_k = [il for il in cpu_layers if model.is_mla[il] == is_mla]
        n = len(layers_k)
        cs = sum(med(r["cpu"]) for r in rows.values() if r["cpu"]) / 1000
        gsum = sum(med(r["gpu"]) for r in rows.values() if r["gpu"]) / 1000
        mb = sum(max(r["bytes"]) for r in rows.values() if r["bytes"]) / 1e6
        # a layer without pred (the last one) does not pay for it
        n_nopred = sum(1 for il in layers_k if not has_pred[il])
        cpu_tok = n * cs - n_nopred * med(rows[PRED]["cpu"]) / 1000
        gpu_tok = n * gsum - n_nopred * med(rows[PRED]["gpu"]) / 1000
        trows.append([kind, f(mb), f(cs, 2), f(gsum, 3), f(cs / gsum if gsum else float('nan')), f"{n} ({n_nopred} w/o pred)", f(cpu_tok, 1), f(gpu_tok, 2)])
        per_tok["cpu"] += cpu_tok
        per_tok["gpu"] += gpu_tok
        res["A2"][kind] = {"mb": mb, "cpu_ms": cs, "gpu_ms": gsum, "n_cpu_layers": n, "n_without_pred": n_nopred,
                           "cpu_ms_per_token": cpu_tok, "gpu_ms_per_token": gpu_tok}
    o.table(["type", "MB/layer (incl. pred)", "CPU ms/layer", "GPU ms/layer", "CPU/GPU", "CPU layers", "CPU ms/token", "same on GPU ms/token"], trows)
    measured = []
    for s, (layers, _out) in paired.items():
        measured.append(sum(e["dur"] for il, ops in layers for w, e, d in ops if d == "cpu") / 1000)
    o(f"Per token, projected from the medians ({n_kda_cpu} KDA + {n_mla_cpu} MLA): CPU {per_tok['cpu']:.1f} ms, the same on the GPU "
      f"{per_tok['gpu']:.2f} ms. Measured sum of all CPU MUL_MAT per token: {', '.join(f(x) for x in measured)} ms.")
    o()
    res["A2"]["per_token_cpu_ms"] = per_tok["cpu"]
    res["A2"]["per_token_gpu_ms"] = per_tok["gpu"]
    res["A2"]["measured_cpu_mm_ms"] = measured
    res["layers"] = {"cpu": cpu_layers, "gpu": gpu_layers, "n_kda_cpu": n_kda_cpu, "n_mla_cpu": n_mla_cpu}


# ---------------------------------------------------------------- A3

def slice_class(model, tensor_name):
    """'ffn_moe_gate-12' -> ('gate/up IQ1_S', slice bytes)"""
    kind, il = tensor_name.rsplit("-", 1)
    il = int(il)
    kind = kind.replace("ffn_moe_", "")
    gname = f"blk.{il}.ffn_{kind}_exps.weight"
    ne, _ty = model.t[gname]
    b = model.bytes(gname) // ne[2]
    ty = model.tname(gname)
    lo_hi = "L1-10" if il <= 10 else f"L11-{model.n_layer - 1}"
    if kind in ("gate", "up"):
        return f"gate/up {ty} {lo_hi}", b
    return f"down {ty} {lo_hi}", b


def table_a3(o, tr, model, res):
    """per-thread records: detail 2 expert_thread events, detail 1 worker expert spans"""
    if tr.layout == "simple":
        recs = [(e["args"]["step"], e["args"]["tensor"], e["args"]["expert"], e["tid"], e["ts"], e["dur"], e["args"]["stall_ns"] / 1000,
                 e["args"].get("n_threads", 14)) for e in tr.worker_experts]
        ops = {}
    else:
        recs = [(e["args"]["step"], e["args"]["name"], e["args"]["expert"], e["tid"], e["ts"], e["dur"], e["args"]["stall_ns"] / 1000, None)
                for e in tr.expert_threads]
        ops = {(e["args"]["step"], e["name"]): e for e in tr.ops_cpu if e["tid"] == TID_WORKER0 and e["args"]["op"] == "MUL_MAT_ID"}

    by_op = defaultdict(list)
    for r in recs:
        by_op[(r[0], r[1])].append(r)
    cls = {}
    per = defaultdict(lambda: {"thr_us": [], "thr_gbs": [], "bytes": 0, "t_mean": 0.0, "t_crit": 0.0, "n_ops": 0, "agg_op": [], "slice": 0})
    stall_tok = defaultdict(float)
    comp_tok = defaultdict(float)
    for (s, name), rr in by_op.items():
        if name not in cls:
            cls[name] = slice_class(model, name)
        c, sb = cls[name]
        threads = defaultdict(lambda: [0.0, 0.0])
        experts = set()
        for _s, _n, ex, tid, ts, dur, st, nth in rr:
            experts.add(ex)
            threads[tid][0] += dur
            threads[tid][1] += st
        nth = len(threads)
        for _s, _n, ex, tid, ts, dur, st, _nth in rr:
            comp = dur - st
            if comp > 0:
                per[c]["thr_us"].append(comp)
                per[c]["thr_gbs"].append(gbs(sb / nth, comp))
        if (s, name) in ops:
            wall = ops[(s, name)]["dur"]
        else:
            wall = max(r[4] + r[5] for r in rr) - min(r[4] for r in rr)
        mean_stall = sum(v[1] for v in threads.values()) / nth
        crit = max(v[0] - v[1] for v in threads.values())
        nbytes = len(experts) * sb
        d = per[c]
        d["bytes"] += nbytes
        d["t_mean"] += wall - mean_stall
        d["t_crit"] += crit
        d["n_ops"] += 1
        d["slice"] = sb
        d["agg_op"].append(gbs(nbytes, wall - mean_stall))
        stall_tok[s] += mean_stall
        comp_tok[s] += wall - mean_stall

    n_steps = len({k[0] for k in by_op})
    o("### A3 routed experts per slice type (CPU, 14 threads)")
    o()
    o("per-thread compute = record dur - stall_ns (one thread's share of one expert); aggregate = expert bytes / (MUL_MAT_ID dur - mean "
      "per-thread stall of the op), as a ratio of sums over all ops; 'crit' uses the slowest thread's compute instead.")
    o()
    trows = []
    tot_b, tot_t = 0, 0.0
    for c in sorted(per):
        d = per[c]
        tot_b += d["bytes"]
        tot_t += d["t_mean"]
        agg = gbs(d["bytes"], d["t_mean"])
        trows.append([c, f(d["slice"] / 1e6, 3), d["n_ops"] // n_steps, f(med(d["thr_us"])), f(p90(d["thr_us"])), f(med(d["thr_gbs"]), 2),
                      f(14 * med(d["thr_gbs"])), f(agg), f(med(d["agg_op"])), f(gbs(d["bytes"], d["t_crit"])),
                      f(d["bytes"] / n_steps / 1e9, 3), f(d["t_mean"] / n_steps / 1000)])
        res["A3"][c] = {"slice_bytes": d["slice"], "ops_per_token": d["n_ops"] // n_steps, "thr_us_med": med(d["thr_us"]),
                        "thr_gbs_med": med(d["thr_gbs"]), "agg_gbs": agg, "agg_gbs_crit": gbs(d["bytes"], d["t_crit"]),
                        "gb_per_token": d["bytes"] / n_steps / 1e9, "compute_ms_per_token": d["t_mean"] / n_steps / 1000}
    o.table(["slice type", "MB/slice", "ops/token", "thread us med", "p90", "thread GB/s", "x14", "AGGREGATE GB/s", "agg med/op",
             "agg crit GB/s", "GB/token", "compute ms/token"], trows)
    agg_all = gbs(tot_b, tot_t)
    o("Layers 1-10 (dense part on the GPU) and 11-92 are split: in layers 1-10 the same IQ1_S slices run slower per thread, so the Q2_K "
      "down slices (only in layers 1-10) are not comparable with a standalone benchmark of Q2_K.")
    o(f"All experts: {tot_b / n_steps / 1e9:.2f} GB/token, compute {tot_t / n_steps / 1000:.1f} ms/token -> AGGREGATE {agg_all:.1f} GB/s "
      f"(the number the go rule uses).")
    o("Stall per token (sum over ops of the mean per-thread stall): " + ", ".join(f"step {s}: {v / 1000:.1f} ms" for s, v in sorted(stall_tok.items())))
    o()
    res["A3"]["all"] = {"gb_per_token": tot_b / n_steps / 1e9, "compute_ms_per_token": tot_t / n_steps / 1000, "agg_gbs": agg_all}
    res["stall_ms"] = {s: v / 1000 for s, v in stall_tok.items()}
    res["expert_compute_ms"] = {s: v / 1000 for s, v in comp_tok.items()}
    return stall_tok, comp_tok


# ---------------------------------------------------------------- A4

def overlap(iv_sorted, starts, a, b):
    """length of [a, b] covered by the merged sorted intervals"""
    import bisect
    i = max(0, bisect.bisect_right(starts, a) - 1)
    tot = 0.0
    while i < len(iv_sorted) and iv_sorted[i][0] < b:
        x, y = iv_sorted[i]
        tot += max(0.0, min(y, b) - max(x, a))
        i += 1
    return tot


def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def table_a4(o, tr, model, stall_tok, comp_tok, res, cpu_first):
    o("### A4 token budget (ms per step)")
    o()
    rows = []
    for tok in tr.tokens:
        s = tok["args"]["step"]
        if tr.layout == "simple":
            out_sync = tok["args"].get("out_sync_dur", 0.0)
        else:
            out_sync = sum(e.get("dur", 0.0) for e in tr.output_sync(s))
        wall = (tok["dur"] + out_sync) / 1000
        r = {"step": s, "wall": wall, "stall": stall_tok.get(s, 0.0) / 1000, "mmid": comp_tok.get(s, 0.0) / 1000}
        if tr.layout == "simple":
            dense = [e for e in tr.worker_dense if e["args"]["step"] == s]
            r["cpu_dense_mm"] = None
            r["cpu_other"] = sum(e["dur"] for e in dense) / 1000  # all non-expert CPU ops
            gpu = [e for e in tr.gpu if e["args"]["step"] == s]
            r["gpu_w"] = sum(e["dur"] for e in gpu if e["args"]["layer"] < cpu_first or e["args"]["layer"] < 0) / 1000
            r["gpu_c"] = sum(e["dur"] for e in gpu if e["args"]["layer"] >= cpu_first) / 1000
            r["n_gpu_c"] = sum(e["args"]["n_ops"] for e in gpu if e["args"]["layer"] >= cpu_first)
            r["n_sync"] = tok["args"].get("n_sync")
            r["sync"] = tok["args"].get("sync_ms")
            r["sync_net"] = None
            cp = [e for e in tr.copies_s if e["args"]["step"] == s]
            r["n_copy"] = len(cp)
            r["copy"] = sum(e["dur"] for e in cp) / 1000
            r["launch"] = None
        else:
            cpu = [e for e in tr.ops_cpu if e["tid"] == TID_WORKER0 and e["args"]["step"] == s]
            r["cpu_dense_mm"] = sum(e["dur"] for e in cpu if e["args"]["op"] == "MUL_MAT") / 1000
            r["cpu_other"] = sum(e["dur"] for e in cpu if e["args"]["op"] not in ("MUL_MAT", "MUL_MAT_ID")) / 1000
            gpu = [e for e in tr.ops_gpu if e["args"]["step"] == s]
            gw = [e for e in gpu if e["args"]["layer"] < cpu_first or e["name"] == "result_output"]
            gc = [e for e in gpu if not (e["args"]["layer"] < cpu_first or e["name"] == "result_output")]
            r["gpu_w"] = sum(e["dur"] for e in gw) / 1000
            r["gpu_c"] = sum(e["dur"] for e in gc) / 1000
            r["n_gpu_c"] = len(gc)
            busy = merge([(e["ts"], e["ts"] + e["dur"]) for e in gpu])
            starts = [a for a, _b in busy]
            syncs = [e for e in tr.syncs if e["args"]["step"] == s and not e["args"].get("output")]
            r["n_sync"] = len(syncs)
            r["sync"] = sum(e["dur"] for e in syncs) / 1000
            r["sync_net"] = sum(e["dur"] - overlap(busy, starts, e["ts"], e["ts"] + e["dur"]) for e in syncs) / 1000
            cp = [e for e in tr.copies if e["args"]["step"] == s and e["ph"] == "X"]
            r["n_copy"] = len(cp)
            r["copy"] = sum(e["dur"] for e in cp) / 1000
            r["launch"] = sum(e["dur"] for e in tr.launches if e["args"].get("step") == s) / 1000
        acc = r["stall"] + r["mmid"] + r["cpu_other"] + (r["cpu_dense_mm"] or 0) + r["gpu_w"] + r["gpu_c"] + r["copy"] + (r["sync_net"] or 0)
        r["unacc"] = r["wall"] - acc
        r["today_compute"] = r["wall"] - r["stall"]
        rows.append(r)

    lab = [("wall (TOKEN + output SYNC)", "wall", 1), ("CPU dense MUL_MAT", "cpu_dense_mm", 1), ("MUL_MAT_ID compute (dur - stall)", "mmid", 1),
           ("stall (mean per-thread, sum over expert ops)", "stall", 1),
           ("CPU non-matmul ops" if tr.layout != "simple" else "CPU non-expert ops (incl. dense matmuls)", "cpu_other", 1),
           (f"GPU ops of layers 0-{cpu_first - 1} + output", "gpu_w", 1), ("GPU ops of CPU layers (KV / recurrent cache)", "gpu_c", 1),
           ("  count", "n_gpu_c", 0), ("SYNC (host, excl. output)", "sync", 1), ("  count", "n_sync", 0),
           ("  SYNC not overlapped by GPU ops", "sync_net", 1), ("COPY", "copy", 1), ("  count", "n_copy", 0),
           ("host launch (info, not added)", "launch", 1), ("unaccounted", "unacc", 1), ("today_compute = wall - stall", "today_compute", 1)]
    o.table(["item"] + [f"step {r['step']}" for r in rows] + ["median"],
            [[name] + [f(r[k], nd) if r[k] is not None else "-" for r in rows]
             + [f(med([r[k] for r in rows if r[k] is not None]), nd) if any(r[k] is not None for r in rows) else "-"]
             for name, k, nd in lab])
    tc0 = rows[0]["today_compute"] / 1000
    if rows[0]["cpu_dense_mm"] is not None:
        mm0 = (rows[0]["cpu_dense_mm"] + rows[0]["mmid"]) / 1000
        dev = mm0 / MATMUL_REF_S - 1
        o(f"Self-consistency: step 0 CPU dense MUL_MAT + MUL_MAT_ID compute = {mm0:.3f} s vs the plan's {MATMUL_REF_S} s (matmuls only): {dev * 100:+.1f}% -> "
          f"{'PASS' if abs(dev) <= 0.15 else 'FAIL'} (15% rule; the reference is the d2 trace, for another trace this line is information only).")
        res["matmul_step0_s"] = mm0
        res["matmul_ref_dev"] = dev
    wall_med = med([r["wall"] for r in rows]) / 1000
    tc_med = med([r["today_compute"] for r in rows]) / 1000
    o(f"B4 baselines: today = wall {rows[0]['wall'] / 1000:.3f} s step 0, {wall_med:.3f} s median ({1 / wall_med:.3f} tok/s); "
      f"today, zero stall = wall - stall {tc0:.3f} s step 0, {tc_med:.3f} s median ({1 / tc_med:.3f} tok/s).")
    o("The 1.37 s all-cached figure is in no trace: unverified.")
    if tr.layout == "simple":
        o("Simple layout: the worker 0 spans merge consecutive ops with the gaps between them and the expert op wall is first thread start .. "
          "last thread end, so 'unaccounted' can be negative.")
    o()
    res["A4"] = rows
    res["today_compute_step0_s"] = tc0
    res["b4_baseline"] = {"wall_s_step0": rows[0]["wall"] / 1000, "wall_s_median": wall_med, "today_compute_s_step0": tc0,
                          "today_compute_s_median": tc_med}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--model", default="/mnt/k3/kimi-k3-iq1", help="directory of the GGUF shards (headers only are read)")
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "out", "partA.md"), help="markdown output; a .json with the numbers is written next to it")
    args = ap.parse_args()

    model = Model(args.model)
    tr = Trace(args.trace)
    o = Out()
    res = {"trace": args.trace, "layout": tr.layout, "A1": {}, "A2": {}, "A3": {}}
    o(f"## Part A: {os.path.basename(args.trace)} ({tr.layout} layout, {len(tr.tokens)} tokens, n_threads {tr.meta.get('n_threads')})")
    o()
    cpu_first = 11
    if tr.layout != "simple":
        paired = pair_layers(tr, model)
        tables_a1_a2(o, tr, model, paired, res)
        cpu_first = res["layers"]["cpu"][0]
    else:
        o("A1 / A2 need per-op records (LLAMA_TRACE_DETAIL=2); this trace gives A3 and a reduced A4 only.")
        o()
    stall_tok, comp_tok = table_a3(o, tr, model, res)
    table_a4(o, tr, model, stall_tok, comp_tok, res, cpu_first)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fp:
        fp.write("\n".join(o.lines) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fp:
        json.dump(res, fp, indent=1, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
