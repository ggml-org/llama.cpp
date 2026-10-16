#!/usr/bin/env python3
"""Profile repeated patterns in the Kimi-K3 IQ1_S-XS routed-expert weights, to see whether they compress further.

What is actually stored (not literally 1 bit per weight):
  IQ1_S (ffn_gate_exps, ffn_up_exps, ffn_down_exps of layers 11+): 50 bytes per 256 weights = 1.5625 bpw
      d (fp16), qs[32] (low 8 bits), qh[8] (uint16 per 32 weights: 3x3 high index bits, 3-bit scale, delta sign).
      Every 8 consecutive weights of a row are ONE 11-bit index into a fixed 2048-entry codebook of {-1,0,+1} patterns.
  Q2_K (ffn_down_exps of layers 1-10): 84 bytes per 256 weights = 2.625 bpw: scales[16], qs[64] (2-bit codes), d, dmin (fp16).

So the "1/0 pin patterns" are codebook indices. The script measures, on a sample of layers and experts:
  * order-0 entropy of the indices / scales / fp16 d   -> what a plain entropy coder would save
  * conditional entropy given the left neighbour, the row above, the same position in another expert of the
    same layer, and the same expert id in another sampled layer -> whether a context model would save more
  * exact repeats of 32-weight sub-blocks and whole 256-weight blocks, and how much of the matrix the
    top-K most repeated patterns cover -> the "dictionary of most repeated patterns" idea
  * real zlib / lzma ratios on raw expert slices (raw bytes and with the fields split into separate streams)

Reads only the sampled expert slices (~2 MB IQ1_S / ~3.6 MB Q2_K each), never the whole 540 GB.

  venv/bin/python -I profile_weights.py [--dir /mnt/k3/kimi-k3-iq1] [--layers 1,10,31,61,92] [--experts 8] [--out r.json]
"""

import argparse
import glob
import json
import lzma
import math
import os
import random
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402
from gguf.quants import IQ1_S  # noqa: E402

QK_K = 256
IQ1S_BYTES = 50
Q2K_BYTES = 84
NIDX = 2048
_j = np.arange(16)
Q2K_GROUP_BYTES = (32 * (_j // 8) + 16 * (_j % 2))[:, None] + _j[None, :]      # (16 groups, 16 weights) qs byte
Q2K_GROUP_SHIFT = (2 * ((_j % 8) // 2)).astype(np.uint8)[None, :, None]
Q2K_CODE_SHIFTS = (2 * _j).astype(np.uint64)[None, None, :]
KINDS = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}


def entropy(counts):
    c = np.asarray(counts, dtype=np.float64).ravel()
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def cond_entropy(joint, nx):
    """H(Y|X) from a flat joint histogram with key x*ny + y."""
    j = np.asarray(joint, dtype=np.float64).reshape(nx, -1)
    return entropy(j) - entropy(j.sum(axis=1))


def xent(hist2, alpha=0.5):
    """Held-out code length in bits/symbol: 2-fold cross-entropy of fold k under the other fold's counts with
    additive smoothing. Unlike the plug-in entropy it is not biased low when the table has more cells than samples,
    so it is what an adaptive coder trained on other experts could really achieve."""
    h = np.asarray(hist2, dtype=np.float64)
    bits = n = 0.0
    for k in (0, 1):
        tr, te = h[1 - k], h[k]
        p = (tr + alpha) / (tr.sum() + alpha * tr.size)
        bits += -(te * np.log2(p)).sum()
        n += te.sum()
    return float(bits / n) if n else 0.0


def cond_xent(joint2, nx, alpha=0.5):
    """Held-out H(Y|X): 2-fold, P(y|x) from the other fold with additive smoothing."""
    j = np.asarray(joint2, dtype=np.float64).reshape(2, nx, -1)
    bits = n = 0.0
    for k in (0, 1):
        tr, te = j[1 - k], j[k]
        p = (tr + alpha) / (tr.sum(axis=1, keepdims=True) + alpha * tr.shape[1])
        bits += -(te * np.log2(p)).sum()
        n += te.sum()
    return float(bits / n) if n else 0.0


def mix64(words):
    """Hash rows of a uint64 matrix to one uint64 per row (for duplicate counting; collisions are negligible)."""
    h = np.full(words.shape[0], 0xCBF29CE484222325, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(words.shape[1]):
            h ^= words[:, k]
            h *= np.uint64(0x9E3779B97F4A7C15)
            h ^= h >> np.uint64(29)
    return h


def coverage(keys, tops=(1, 16, 256, 4096, 65536)):
    """Count repeats in a key array: unique fraction, share covered by the top-K most frequent keys."""
    _, counts = np.unique(keys, return_counts=True)
    counts = np.sort(counts)[::-1]
    n = int(keys.size)
    cum = np.cumsum(counts)
    return {
        "n": n,
        "unique": int(counts.size),
        "unique_frac": counts.size / n if n else 0.0,
        "seen_more_than_once_frac": float(counts[counts > 1].sum()) / n if n else 0.0,
        "top_cover": {str(k): float(cum[min(k, counts.size) - 1]) / n for k in tops if counts.size},
    }


def top_keys(keys, k):
    vals, counts = np.unique(keys, return_counts=True)
    order = np.argsort(counts)[::-1][:k]
    return [(int(vals[i]), int(counts[i])) for i in order]


def ternary_str(row):
    return "".join({-1: "-", 0: "0", 1: "+"}[int(v)] for v in row)


def compress_ratios(streams, do_lzma):
    raw = sum(len(s) for s in streams)
    out = {"zlib9": raw / sum(len(zlib.compress(s, 9)) for s in streams)}
    if do_lzma:
        out["lzma6"] = raw / sum(len(lzma.compress(s, preset=6)) for s in streams)
    return out


class Stats:
    """Accumulators for one tensor kind."""

    def __init__(self, qtype):
        self.qtype = qtype
        self.slices = 0
        self.weights = 0
        self.d_hist = np.zeros((2, 65536), np.int64)
        self.comp = []  # per compressed slice: dict of ratios
        self.cross_expert_pairs = 0
        self.cross_layer_pairs = 0
        if qtype == GGMLQuantizationType.IQ1_S:
            n = NIDX
            self.ss_hist = np.zeros((2, 16), np.int64)  # scale | sign << 3
        else:
            n = 256  # Q2_K: one qs byte = 4 consecutive 2-bit codes
            self.dmin_hist = np.zeros((2, 65536), np.int64)
            self.sc_hist = np.zeros((2, 256), np.int64)
        self.n = n
        self.sym_hist = np.zeros((2, n), np.int64)
        self.left = np.zeros((2, n * n), np.int64)
        self.up = np.zeros((2, n * n), np.int64)
        self.xexp = np.zeros((2, n * n), np.int64)
        self.xlay = np.zeros((2, n * n), np.int64)
        self.sub_keys = []    # small exact patterns (IQ1_S: 32-weight sub-block; Q2_K: 16-weight group + scale)
        self.blk_pat = []     # whole block without fp16 scales (pattern only)
        self.blk_full = []    # whole block, bit-exact

    def add_joint(self, acc, a, b, fold):
        acc[fold] += np.bincount((a.astype(np.int64) * self.n + b).ravel(), minlength=self.n * self.n)


def decode_iq1s(raw, ne0):
    """raw: (rows, row_bytes) uint8 -> dict of fields; idx is (rows, ne0/8) of 11-bit codebook indices."""
    rows = raw.shape[0]
    b = raw.reshape(-1, IQ1S_BYTES)
    nb = b.shape[0]
    d = b[:, 0:2].copy().view("<u2").ravel()
    qs = b[:, 2:34].astype(np.uint16).reshape(nb, 8, 4)
    qh = b[:, 34:50].copy().view("<u2").reshape(nb, 8)
    hi = (qh[:, :, None] >> np.array([0, 3, 6, 9], np.uint16)) & 7
    idx = (qs | (hi << 8)).reshape(rows, ne0 // 8)
    scale = (qh >> 12) & 7
    sign = qh >> 15
    return {"b": b, "d": d, "qh": qh, "idx": idx, "ss": (scale | (sign << 3)).reshape(rows, -1)}


def decode_q2k(raw, ne0):
    rows = raw.shape[0]
    b = raw.reshape(-1, Q2K_BYTES)
    nb = b.shape[0]
    return {
        "b": b,
        "sc": b[:, 0:16].reshape(rows, -1),
        "qs": b[:, 16:80].reshape(rows, -1),  # one byte = 4 codes; layout as stored, rows kept intact
        "d": b[:, 80:82].copy().view("<u2").ravel(),
        "dmin": b[:, 82:84].copy().view("<u2").ravel(),
        "nb": nb,
    }


def profile_slice(st, raw, ne0, fo, compress, do_lzma):
    """fo: held-out fold of this slice; it follows the expert, so no expert is ever in both folds."""
    st.slices += 1
    st.weights += raw.shape[0] * ne0
    if st.qtype == GGMLQuantizationType.IQ1_S:
        f = decode_iq1s(raw, ne0)
        sym = f["idx"]
        st.ss_hist[fo] += np.bincount(f["ss"].ravel(), minlength=16)
        st.d_hist[fo] += np.bincount(f["d"], minlength=65536)
        # 32-weight sub-block: 4 indices + scale + sign = 48 bits, exact key
        i4 = sym.reshape(-1, 4).astype(np.uint64)
        key = i4[:, 0] | (i4[:, 1] << 11) | (i4[:, 2] << 22) | (i4[:, 3] << 33) | \
            (f["ss"].ravel().astype(np.uint64) << 44)
        st.sub_keys.append(key)
        b = f["b"]
        pat = np.ascontiguousarray(b[:, 2:50]).view("<u8")           # 48 bytes = 6 words
        st.blk_pat.append(mix64(pat))
        full = np.concatenate([pat, b[:, 0:2].copy().view("<u2").astype(np.uint64)], axis=1)
        st.blk_full.append(mix64(full))
        if compress:
            st.comp.append({
                "raw": compress_ratios([b.tobytes()], do_lzma),
                "split": compress_ratios([b[:, 0:2].tobytes(), b[:, 2:34].tobytes(), b[:, 34:50].tobytes()], do_lzma),
            })
    else:
        f = decode_q2k(raw, ne0)
        sym = f["qs"]
        st.d_hist[fo] += np.bincount(f["d"], minlength=65536)
        st.dmin_hist[fo] += np.bincount(f["dmin"], minlength=65536)
        st.sc_hist[fo] += np.bincount(f["sc"].ravel(), minlength=256)
        # 16-weight group sharing scales[j] (dequantize_row_q2_K): codes (qs[32*(j//8) + 16*(j%2) + l] >> 2*((j%8)//2)) & 3,
        # l = 0..15; key = 16 codes x 2 bits | scales[j] << 32
        qsb = f["b"][:, 16:80]
        codes = (qsb[:, Q2K_GROUP_BYTES] >> Q2K_GROUP_SHIFT) & 3                  # (nb, 16 groups, 16 weights)
        key = (codes.astype(np.uint64) << Q2K_CODE_SHIFTS).sum(axis=2, dtype=np.uint64)
        st.sub_keys.append((key | (f["b"][:, 0:16].astype(np.uint64) << np.uint64(32))).ravel())
        b = f["b"]
        words = np.ascontiguousarray(b[:, 0:80]).view("<u8")         # scales + qs = 80 bytes = 10 words
        st.blk_pat.append(mix64(words))
        full = np.concatenate([words, b[:, 80:84].copy().view("<u4").astype(np.uint64)], axis=1)
        st.blk_full.append(mix64(full))
        if compress:
            st.comp.append({
                "raw": compress_ratios([b.tobytes()], do_lzma),
                "split": compress_ratios([b[:, 0:16].tobytes(), b[:, 16:80].tobytes(), b[:, 80:84].tobytes()],
                                         do_lzma),
            })
    st.sym_hist[fo] += np.bincount(sym.ravel(), minlength=st.n)
    st.add_joint(st.left, sym[:, :-1], sym[:, 1:], fo)
    st.add_joint(st.up, sym[:-1, :], sym[1:, :], fo)
    return sym


def summarize(st, top):
    n = st.n
    hist = st.sym_hist.sum(axis=0)
    both = lambda a: bool((a.sum(axis=1) > 0).all())  # noqa: E731  held-out needs data in both folds
    hx = lambda a: xent(a) if both(a) else None  # noqa: E731
    hc = lambda a: cond_xent(a, n) if both(a) else None  # noqa: E731
    r = {
        "qtype": st.qtype.name,
        "slices_sampled": st.slices,
        "weights_sampled": st.weights,
        "symbol": "11-bit codebook index (8 weights)" if st.qtype == GGMLQuantizationType.IQ1_S
        else "qs byte (4 x 2-bit codes)",
        "symbols_used": int((hist > 0).sum()),
        "symbols_possible": n,
        # *_bits: held-out code lengths (honest); *_plugin: plug-in entropy of all samples (biased low)
        "H0_bits": hx(st.sym_hist),
        "H0_plugin": entropy(hist),
        "H_given_left_bits": hc(st.left),
        "H_given_up_bits": hc(st.up),
        "H_given_same_pos_other_expert_bits": hc(st.xexp),
        "H_given_same_pos_other_layer_bits": hc(st.xlay),
        "H_given_left_plugin": cond_entropy(st.left.sum(axis=0), n),
        "match_rate_other_expert": float(np.trace(st.xexp.sum(axis=0).reshape(n, n))) / st.xexp.sum()
        if st.cross_expert_pairs else None,
        "match_rate_by_chance": float(((hist / hist.sum()) ** 2).sum()),
        "H_d_fp16_bits": hx(st.d_hist),
        "sub_patterns": coverage(np.concatenate(st.sub_keys)),
        "block_patterns_no_scale": coverage(np.concatenate(st.blk_pat)),
        "blocks_bit_exact": coverage(np.concatenate(st.blk_full)),
    }
    conds = [r[k] for k in ("H0_bits", "H_given_left_bits", "H_given_up_bits",
                            "H_given_same_pos_other_expert_bits", "H_given_same_pos_other_layer_bits")
             if r[k] is not None]
    best_cond = min(conds) if conds else None
    if st.qtype == GGMLQuantizationType.IQ1_S:
        ssh = st.ss_hist.sum(axis=0)
        hss = hx(st.ss_hist)
        r["H_scale_sign_bits"] = hss
        r["scale_hist"] = [int(x) for x in ssh.reshape(2, 8).sum(axis=0)]
        r["delta_negative_frac"] = float(ssh[8:].sum()) / ssh.sum()
        grid = IQ1_S.grid.reshape(NIDX, 8)
        tern = np.array([(grid == v).sum(axis=1) for v in (-1, 0, 1)])  # (3, 2048)
        tc = tern @ hist
        r["ternary_frac_minus_zero_plus"] = (tc / tc.sum()).tolist()
        r["top_indices"] = [{"index": int(i), "pattern": ternary_str(grid[i]), "share": float(hist[i]) /
                             hist.sum()} for i in np.argsort(hist)[::-1][:top]]
        subs = top_keys(np.concatenate(st.sub_keys), top)
        r["top_sub_blocks"] = [{
            "indices": [(k >> (11 * j)) & 0x7FF for j in range(4)],
            "scale": (k >> 44) & 7, "delta_neg": (k >> 47) & 1,
            "pattern": " ".join(ternary_str(grid[(k >> (11 * j)) & 0x7FF]) for j in range(4)),
            "count": c} for k, c in subs]
        raw_bits = 16 + 8 * 4 + 32 * 11
        parts = (r["H_d_fp16_bits"], hss)
        est = lambda hidx: (parts[0] + 8 * parts[1] + 32 * hidx) / QK_K  # noqa: E731
    else:
        r["H_scale_byte_bits"] = hx(st.sc_hist)
        r["H_dmin_fp16_bits"] = hx(st.dmin_hist)
        codes = np.zeros(4, np.int64)
        for s in range(0, 8, 2):
            codes += np.bincount((np.arange(256) >> s) & 3, weights=hist, minlength=4).astype(np.int64)
        r["code_frac_0_1_2_3"] = (codes / codes.sum()).tolist()
        r["H_code_bits"] = entropy(codes)
        raw_bits = 16 * 8 + 64 * 8 + 32
        parts = (r["H_scale_byte_bits"], r["H_d_fp16_bits"], r["H_dmin_fp16_bits"])
        est = lambda hq: (16 * parts[0] + 64 * hq + parts[1] + parts[2]) / QK_K  # noqa: E731
    r["bpw_stored"] = raw_bits / QK_K
    ok = r["H0_bits"] is not None and all(v is not None for v in parts)
    r["bpw_order0_estimate"] = est(r["H0_bits"]) if ok else None
    r["bpw_best_context_estimate"] = est(best_cond) if ok else None
    if st.comp:
        r["compress_ratio_mean"] = {
            view: {alg: float(np.mean([c[view][alg] for c in st.comp])) for alg in st.comp[0][view]}
            for view in st.comp[0]}
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="/mnt/k3/kimi-k3-iq1", help="directory with the GGUF shards")
    ap.add_argument("--layers", default="1,10,31,61,92", help="comma list of MoE layers to sample (down is Q2_K in 1-10, IQ1_S above)")
    ap.add_argument("--experts", type=int, default=8, help="random experts per layer (same ids in every layer)")
    ap.add_argument("--expert-ids", default="", help="explicit comma list of expert ids (overrides --experts)")
    ap.add_argument("--kinds", default="gate,up,down", help="subset of gate,up,down")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compress-slices", type=int, default=4, help="slices per kind run through zlib/lzma")
    ap.add_argument("--no-lzma", action="store_true", help="zlib only (lzma is slow)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="", help="write the full JSON report here")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.dir, "*.gguf")))
    if not shards:
        sys.exit(f"no .gguf files in {args.dir}")
    tensors = {}
    for p in shards:
        for t in GGUFReader(p, "r").tensors:
            tensors[t.name] = t
    IQ1_S.init_grid()

    layers = [int(x) for x in args.layers.split(",") if x]
    kinds = [k for k in args.kinds.split(",") if k]
    for k in kinds:
        if k not in KINDS:
            sys.exit(f"unknown kind {k}")

    probe = tensors.get(f"blk.{layers[0]}.{KINDS[kinds[0]]}.weight")
    if probe is None:
        sys.exit(f"tensor blk.{layers[0]}.{KINDS[kinds[0]]}.weight not found (dense layer?)")
    n_expert = int(probe.shape[2])
    if args.expert_ids:
        experts = sorted({int(x) for x in args.expert_ids.split(",") if x})
    else:
        experts = sorted(random.Random(args.seed).sample(range(n_expert), min(args.experts, n_expert)))
    if any(e < 0 or e >= n_expert for e in experts):
        sys.exit(f"expert ids must be in 0..{n_expert - 1}")

    report = {"dir": args.dir, "layers": layers, "experts": experts, "kinds": {}}
    t0 = time.time()
    read_bytes = 0
    for kind in kinds:
        # a kind can mix quant types across layers (this model: down is Q2_K in layers 1-10, IQ1_S above),
        # so statistics, cross-layer pairs and model byte totals are kept per (kind, type)
        per_type = {}  # qtype -> [Stats, prev_layer]
        for li in layers:
            name = f"blk.{li}.{KINDS[kind]}.weight"
            t = tensors.get(name)
            if t is None:
                sys.exit(f"tensor {name} not found")
            qt = GGMLQuantizationType(int(t.tensor_type))
            if qt not in (GGMLQuantizationType.IQ1_S, GGMLQuantizationType.Q2_K):
                sys.exit(f"{name}: type {qt.name} not handled (IQ1_S / Q2_K only)")
            st, prev_layer = per_type.setdefault(qt, [Stats(qt), {}])
            ne0, ne1 = int(t.shape[0]), int(t.shape[1])
            if ne0 % QK_K or t.data.shape[0] != int(t.shape[2]):
                sys.exit(f"{name}: unexpected shape {list(t.shape)} / data {t.data.shape}")
            cur = {}
            for pos, e in enumerate(experts):
                fo = pos % 2
                raw = np.array(t.data[e], dtype=np.uint8).reshape(ne1, -1)  # reads this expert's slice only
                read_bytes += raw.nbytes
                compress = len(st.comp) < args.compress_slices
                sym = profile_slice(st, raw, ne0, fo, compress, not args.no_lzma)
                for ofo, other in cur.values():
                    if ofo == fo:  # pair only within a fold, so the held-out fold never shares an expert
                        st.add_joint(st.xexp, other, sym, fo)
                        st.cross_expert_pairs += 1
                if e in prev_layer:
                    st.add_joint(st.xlay, prev_layer[e][1], sym, fo)
                    st.cross_layer_pairs += 1
                cur[e] = (fo, sym)
            per_type[qt][1] = cur
            print(f"  {kind} layer {li} ({qt.name}): {len(experts)} experts, {read_bytes / 1e9:.2f} GB read, "
                  f"{time.time() - t0:.0f} s", file=sys.stderr)
        for qt, (st, _) in per_type.items():
            r = summarize(st, args.top)
            same = [n for n, t in tensors.items() if n.endswith(f".{KINDS[kind]}.weight") and
                    int(t.tensor_type) == int(qt)]
            r["model_layers_of_this_type"] = len(same)
            r["model_bytes_all_layers"] = sum(int(tensors[n].n_bytes) for n in same)
            r["model_bytes_at_best_context_estimate"] = None if r["bpw_best_context_estimate"] is None else \
                r["model_bytes_all_layers"] * r["bpw_best_context_estimate"] / r["bpw_stored"]
            report["kinds"][kind if len(per_type) == 1 else f"{kind}_{qt.name}"] = r

    # expert bytes of (kind, type) combinations that were not sampled: not covered by the estimate
    sampled = {(k.split("_")[0], r["qtype"]) for k, r in report["kinds"].items()}
    report["unsampled_expert_bytes"] = sum(
        int(t.n_bytes) for n, t in tensors.items()
        for k in kinds if n.endswith(f".{KINDS[k]}.weight") and
        (k, GGMLQuantizationType(int(t.tensor_type)).name) not in sampled)

    report["seconds"] = time.time() - t0
    report["bytes_read"] = read_bytes
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)

    # short text summary
    print(f"\nsampled layers {layers}, experts {experts}  ({read_bytes / 1e9:.2f} GB read, {report['seconds']:.0f} s)")
    f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
    tot_now = tot_best = 0.0
    complete = True
    for kind, r in report["kinds"].items():
        tot_now += r["model_bytes_all_layers"]
        if r["model_bytes_at_best_context_estimate"] is None:
            complete = False
        else:
            tot_best += r["model_bytes_at_best_context_estimate"]
        print(f"\n== {kind} ({r['qtype']}): symbol = {r['symbol']}, {r['symbols_used']}/{r['symbols_possible']} used")
        print(f"  entropy bits/symbol: order0 {f(r['H0_bits'])} (plug-in {r['H0_plugin']:.3f}) | given left {f(r['H_given_left_bits'])} | "
              f"given row above {f(r['H_given_up_bits'])} | given other expert same pos "
              f"{f(r['H_given_same_pos_other_expert_bits'])} | other layer same expert id (null control: ids are "
              f"not aligned across layers) "
              f"{f(r['H_given_same_pos_other_layer_bits'])}")
        if r["match_rate_other_expert"] is not None:
            print(f"  same symbol at same position in another expert: {r['match_rate_other_expert']:.4f} "
                  f"(chance {r['match_rate_by_chance']:.4f})")
        for label, key in (("small patterns", "sub_patterns"), ("blocks (no scale)", "block_patterns_no_scale"),
                           ("blocks (bit-exact)", "blocks_bit_exact")):
            c = r[key]
            cov = ", ".join(f"top{k} {v * 100:.2f}%" for k, v in c["top_cover"].items())
            print(f"  {label}: {c['unique']}/{c['n']} unique ({c['unique_frac'] * 100:.1f}%), "
                  f"repeated {c['seen_more_than_once_frac'] * 100:.2f}%; cover {cov}")
        if "top_indices" in r:
            print("  most common 8-weight patterns: " + ", ".join(
                f"{x['pattern']} {x['share'] * 100:.2f}%" for x in r["top_indices"][:5]))
            print(f"  ternary -/0/+: " + "/".join(f"{v * 100:.1f}%" for v in r["ternary_frac_minus_zero_plus"]) +
                  f"; scale+sign H {f(r['H_scale_sign_bits'], 2)}/4 bits; d H {f(r['H_d_fp16_bits'], 2)}/16 bits")
        else:
            print(f"  2-bit codes 0/1/2/3: " + "/".join(f"{v * 100:.1f}%" for v in r["code_frac_0_1_2_3"]) +
                  f" (H {r['H_code_bits']:.3f}/2 bits); scale byte H {f(r['H_scale_byte_bits'], 2)}/8 bits")
        if "compress_ratio_mean" in r:
            print("  real compressors (ratio): " + "; ".join(
                f"{view} " + ", ".join(f"{a} {v:.3f}x" for a, v in algs.items())
                for view, algs in r["compress_ratio_mean"].items()))
        print(f"  bpw: stored {r['bpw_stored']:.4f} | order-0 entropy coder {f(r['bpw_order0_estimate'], 4)} | "
              f"best context model {f(r['bpw_best_context_estimate'], 4)}")
        if r["model_bytes_at_best_context_estimate"] is not None:
            print(f"  all {r['model_layers_of_this_type']} layers of this type: {r['model_bytes_all_layers'] / 1e9:.1f} GB -> ~"
                  f"{r['model_bytes_at_best_context_estimate'] / 1e9:.1f} GB at the best estimate")
    if report["unsampled_expert_bytes"]:
        print(f"\nnote: {report['unsampled_expert_bytes'] / 1e9:.1f} GB of experts are of a type no sampled layer had; "
              f"not in the total below (sample layers of every type to cover them)")
    if not complete:
        print("\n(held-out estimates need >= 2 experts per sampled type; the cross-expert context needs >= 4)")
    elif tot_now:
        print(f"\nexperts total {tot_now / 1e9:.1f} GB -> ~{tot_best / 1e9:.1f} GB "
              f"({(1 - tot_best / tot_now) * 100:.1f}% smaller) at the best estimate "
              f"(held-out code lengths; a real coder also pays for its tables and decode time)")


if __name__ == "__main__":
    main()
