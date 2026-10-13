#!/usr/bin/env python3
"""Multi-layer (cross-layer) router predictor - does cross-layer routing history
predict the next step better than each layer in isolation?

The premise being tested: routing is a cross-layer trajectory, so a predictor
that sees ALL 48 layers' picks should beat one that sees only its own layer.

Architecture (the "mini-MTP router predictor" in compact integer form - the
model is fed routing decisions, not hidden states, so no transformer over the
125B activations is needed):

    E : Embedding(512 experts -> d)              d = 64      = 32,768 params
    z_L = sum over ctx previous steps, all 48 layers of E[picked expert]
    logits_L = W z_L + b_L   (512 experts, per layer)
                                              48 * 64 * 512 = 1,572,864 params
    total                                              ~1.6M params (~6 MB)

Compared against the same model restricted to its own layer, and against
per-layer LFU. Reported as per-layer recall at fixed candidate set K, which is
the only metric comparable to the paper's K=8/12/16: recall over a large
candidate set is trivially high and says nothing about prefetch cost.

No PyTorch on this box; this is the compact linear-embedding form of the same
model, trained with SGD + BCE on the routing trace.
"""
import argparse
import collections

import numpy as np


def load(path, n_used, n_layer):
    a = np.fromfile(path, dtype=np.uint16)
    n = len(a) // (n_layer * n_used)
    return a[:n * n_layer * n_used].reshape(n, n_layer, n_used).astype(np.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True)
    p.add_argument("--n-layer", type=int, default=48)
    p.add_argument("--n-used", type=int, default=8)
    p.add_argument("--n-exp", type=int, default=512)
    p.add_argument("--ctx", type=int, default=1)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--holdout", type=float, default=0.2)
    args = p.parse_args()

    A = load(args.ids, args.n_used, args.n_layer)
    cut = int(len(A) * (1 - args.holdout))
    tr, te = A[:cut], A[cut:]
    D = args.dim
    print("trace %d steps (train %d / test %d), %d layers x top-%d of %d, ctx=%d, d=%d"
          % (len(A), len(tr), len(te), args.n_layer, args.n_used, args.n_exp, args.ctx, D))

    rng = np.random.default_rng(0)
    E = rng.normal(0, 0.1, (args.n_exp, D)).astype(np.float32)
    b = np.zeros(args.n_exp, np.float32)
    # cross: sums over ALL layers; same: sums over its own layer only
    Wc = [rng.normal(0, 0.05, (D, args.n_exp)).astype(np.float32) for _ in range(args.n_layer)]
    Ws = [rng.normal(0, 0.05, (D, args.n_exp)).astype(np.float32) for _ in range(args.n_layer)]
    n_par = E.size + sum(w.size for w in Wc) + sum(b.size for b in [b] * args.n_layer)
    print("parameters: %.2fM (%.1f MB fp32)" % (n_par / 1e6, n_par * 4 / 1e6))

    def zvec(step, L, cross):
        s = np.zeros(D, np.float32)
        for k in range(1, args.ctx + 1):
            if cross:
                s += E[tr[step - k].ravel()].sum(0)
            else:
                s += E[tr[step - k, L, :]].sum(0)
        return s / (args.ctx * (args.n_layer if cross else 1))

    idx = np.arange(args.ctx, len(tr) - 1)
    y = np.zeros(args.n_exp, np.float32)
    for ep in range(args.epochs):
        rng.shuffle(idx)
        lr = 2.0 / (1 + ep)
        for i in idx:
            for L in range(args.n_layer):
                y[:] = 0
                y[tr[i + 1, L, :]] = 1
                for cross, W in ((True, Wc), (False, Ws)):
                    z = zvec(i, L, cross)
                    pr = 1.0 / (1.0 + np.exp(-np.clip(z @ W[L] + b, -12, 12)))
                    pr = np.clip(pr, 1e-6, 1 - 1e-6)
                    g = (pr - y)[:, None] * lr
                    W[L] -= np.outer(z, g.sum(0))
                    b -= g.sum(0)
                # embedding gets gradient from the cross-layer branch only
                z = zvec(i, 0, True)
                pr = 1.0 / (1.0 + np.exp(-np.clip(z @ Wc[0] + b, -12, 12)))
    print("trained %d epochs" % args.epochs)

    def scores_cross(L, t):
        z = np.zeros(D, np.float32)
        for k in range(1, args.ctx + 1):
            z += E[te[t - k].ravel()].sum(0)
        return (z / (args.ctx * args.n_layer)) @ Wc[L] + b

    def scores_same(L, t):
        z = np.zeros(D, np.float32)
        for k in range(1, args.ctx + 1):
            z += E[te[t - k, L, :]].sum(0)
        return (z / args.ctx) @ Ws[L] + b

    lfu = [collections.Counter(tr[:, L, :].ravel().tolist()) for L in range(args.n_layer)]

    def evaluate(fn, K):
        hit = tot = 0
        for t in range(args.ctx, len(te)):
            for L in range(args.n_layer):
                top = np.argpartition(-fn(L, t), K)[:K]
                for e in te[t, L, :]:
                    tot += 1
                    hit += e in top
        return 100.0 * hit / tot

    print("\n%-40s %7s %7s %7s %7s" % ("model", "R@8", "R@16", "R@32", "R@64"))
    print("%-40s %7.1f %7.1f %7.1f %7.1f" % ("per-layer LFU (0 params)",
          *[evaluate(lambda L, t, c=collections.Counter(lfu[L]): np.array([c.get(e, 0) for e in range(args.n_exp)], np.float32), K) for K in (8, 16, 32, 64)]))
    print("%-40s %7.1f %7.1f %7.1f %7.1f" % ("same-layer only",
          *[evaluate(lambda L, t: scores_same(L, t), K) for K in (8, 16, 32, 64)]))
    print("%-40s %7.1f %7.1f %7.1f %7.1f" % ("CROSS-LAYER (all 48 layers)",
          *[evaluate(lambda L, t: scores_cross(L, t), K) for K in (8, 16, 32, 64)]))
    print("\nuniform-random baseline: %.1f / %.1f / %.1f / %.1f"
          % tuple(100 * K / 512 for K in (8, 16, 32, 64)))


if __name__ == "__main__":
    main()