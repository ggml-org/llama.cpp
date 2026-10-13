#!/usr/bin/env python3
"""Benchmark oracle architectures for MoE prefetch: which gets the most right?

Split-free decode needs token t+1's experts resident before t+1's graph runs.
The router runs inside that graph, so something must predict them. This trains
and scores every predictor that is buildable here, against the capacity
ceiling, so the choice is made on numbers rather than intuition.

Reported per architecture (held-out top-K recall of the ACTUAL router picks,
i.e. the fraction of experts a prefetcher would have to fetch):

    lfu            static per-layer top-K (what the pool does today)
    decayed        per-layer frequency with an exponential half-life
    recency        the previous step's picks dominate
    transition     P(e | x was picked last step at this layer)
    mlp            a trained per-layer MLP over the last K steps' routing
    oracle         hindsight top-K over the whole trace: the ceiling

Usage:
    moe_oracle_bench.py --ids trace.ids.u16 [--n-layer 48] [--n-used 8]
"""
import argparse
import collections

import numpy as np


def load(path, n_used, n_layer):
    a = np.fromfile(path, dtype=np.uint16)
    n = len(a) // (n_layer * n_used)
    return a[:n * n_layer * n_used].reshape(n, n_layer, n_used).astype(np.int32)


def recall_topk(scores, truth, k):
    """scores: (S, n_exp) higher is better. truth: (S, n_used) expert ids."""
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    hit = 0
    for i in range(len(truth)):
        hit += len(set(top[i].tolist()) & set(truth[i].tolist()))
    return 100.0 * hit / truth.size


def train_mlp(tr, L, K, n_exp, n_used, rng, hidden=64, epochs=15, lr=0.05):
    """Per-layer MLP: last K steps' routing at this layer -> expert logits."""
    def feat(i):
        v = np.zeros(K * n_used, np.float32)
        for k in range(1, K + 1):
            for j in range(n_used):
                v[(k - 1) * n_used + j] = tr[i - k, L, j] / float(n_exp)
        return v
    ntr = len(tr) - K
    X = np.stack([feat(i) for i in range(K, len(tr))])
    Y = np.zeros((ntr, n_exp), np.float32)
    for j, i in enumerate(range(K, len(tr))):
        Y[j, tr[i, L, :]] = 1.0
    D = K * n_used
    W1 = (rng.normal(0, 0.05, (D, hidden)) / np.sqrt(D)).astype(np.float32)
    b1 = np.zeros(hidden, np.float32)
    W2 = (rng.normal(0, 0.05, (hidden, n_exp)) / np.sqrt(hidden)).astype(np.float32)
    b2 = np.zeros(n_exp, np.float32)
    for _ in range(epochs):
        idx = rng.permutation(ntr)
        for s in range(0, ntr, 64):
            b = idx[s:s + 64]
            x, y = X[b], Y[b]
            z1 = np.maximum(0, x @ W1 + b1)
            z2 = z1 @ W2 + b2
            z2 -= z2.max(1, keepdims=True)
            p = np.exp(z2)
            p /= p.sum(1, keepdims=True)
            p = np.clip(p, 1e-9, 1.0)
            p /= p.sum(1, keepdims=True)
            g2 = (p - y) * (lr / len(b))
            W2 -= z1.T @ g2
            b2 -= g2.sum(0)
            g1 = (g2 @ W2.T) * (z1 > 0)
            W1 -= x.T @ g1
            b1 -= g1.sum(0)
            np.clip(W1, -1, 1, out=W1)
            np.clip(W2, -1, 1, out=W2)
    return feat, (W1, b1, W2, b2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True)
    p.add_argument("--n-layer", type=int, default=48)
    p.add_argument("--n-used", type=int, default=8)
    p.add_argument("--n-exp", type=int, default=512)
    p.add_argument("--holdout", type=float, default=0.3)
    p.add_argument("--ctx", type=int, default=4, help="MLP context steps")
    args = p.parse_args()

    A = load(args.ids, args.n_used, args.n_layer)
    cut = int(len(A) * (1 - args.holdout))
    tr, te = A[:cut], A[cut:]
    rng = np.random.default_rng(0)
    K = args.n_used
    print("trace %d steps: train %d, held-out %d; top-%d of %d per layer per step\n"
          % (len(A), len(tr), len(te), args.n_used, args.n_exp))

    tally = collections.defaultdict(float)
    for L in range(args.n_layer):
        truth = te[:, L, :]
        cnt = collections.Counter(tr[:, L, :].ravel().tolist())
        base = np.array([cnt.get(e, 0) for e in range(args.n_exp)], np.float32)

        tally["lfu"] += recall_topk(np.tile(base, (len(te), 1)), truth, K)

        dec = collections.Counter()
        for s in range(len(tr)):
            for e, v in list(dec.items()):
                dec[e] = v * 0.98
            for e in tr[s, L, :]:
                dec[e] += 1
        dsc = np.array([dec.get(e, 0) for e in range(args.n_exp)], np.float32)
        tally["decayed"] += recall_topk(np.tile(dsc, (len(te), 1)), truth, K)

        rsc = np.tile(base, (len(te), 1))
        for i in range(1, len(te)):
            for e in te[i - 1, L, :]:
                rsc[i, e] += 1e6
        tally["recency"] += recall_topk(rsc, truth, K)

        pair = collections.defaultdict(collections.Counter)
        for s in range(1, len(tr)):
            for x in tr[s - 1, L, :].tolist():
                for y in tr[s, L, :].tolist():
                    pair[x][y] += 1
        tsc = np.zeros((len(te), args.n_exp), np.float32)
        for i in range(len(te)):
            prev = te[i - 1, L, :].tolist() if i else te[0, L, :].tolist()
            sc = collections.Counter()
            for x in prev:
                sc.update(pair[x])
            for e in range(args.n_exp):
                tsc[i, e] = sc.get(e, 0) + base[e] * 0.1
        tally["transition"] += recall_topk(tsc, truth, K)

        feat, params = train_mlp(tr, L, args.ctx, args.n_exp, args.n_used, rng)
        W1, b1, W2, b2 = params
        X = np.stack([feat(i) for i in range(args.ctx, len(te))])
        msc = np.maximum(0, X @ W1 + b1) @ W2 + b2
        ttruth = te[args.ctx:, L, :]
        tally["mlp"] += recall_topk(msc, ttruth, K)

        allc = collections.Counter(A[:, L, :].ravel().tolist())
        asc = np.array([allc.get(e, 0) for e in range(args.n_exp)], np.float32)
        tally["oracle"] += recall_topk(np.tile(asc, (len(te), 1)), truth, K)

    nL = args.n_layer
    print("%-12s %8s" % ("arch", "top-%d hit" % K))
    for k in ("recency", "mlp", "lfu", "transition", "decayed", "oracle"):
        print("%-12s %7.1f%%" % (k, tally[k] / nL))

    # coverage ceiling vs pool size, and the VRAM that would need
    print("\npool size vs coverage ceiling (this router is near-uniform,")
    print("entropy 8.76 of 9.00 bits):")
    print("%-10s %10s %12s" % ("slots/layer", "coverage", "VRAM GiB"))
    for s in (8, 32, 104, 250, 512):
        print("%-10d %9.1f%% %11.1f" % (s, 100.0 * s / args.n_exp,
                                         s * nL * 1.5104 * 1.51 / 1024 / 1024))


if __name__ == "__main__":
    main()