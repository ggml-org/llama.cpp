#!/usr/bin/env python3
"""Train a next-token MoE routing predictor and measure what it can reach.

Split-free decode needs token t+1's experts resident before token t+1's graph
runs, so the router (which runs inside that graph) has to be predicted. This
trains the predictors that are actually buildable here and reports their
held-out recall, against the capacity ceiling the pool imposes.

  static      per-layer top-K over the training trace (token-free)
  transition  per-layer P(e | picked last step at this layer)
  oracle      hindsight top-K over the WHOLE trace (the capacity ceiling)

Usage:
    moe_predict_eval.py --ids trace.ids.u16 [--n-layer 48] [--n-used 8]
"""
import argparse
import collections

import numpy as np


def load(path, n_used, n_layer):
    a = np.fromfile(path, dtype=np.uint16)
    n = len(a) // (n_layer * n_used)
    return a[:n * n_layer * n_used].reshape(n, n_layer, n_used).astype(np.int32)


def static_topk(train, test, K, n_layer):
    hits = tot = 0
    for L in range(n_layer):
        keep = set(e for e, _ in collections.Counter(train[:, L, :].ravel().tolist()).most_common(K))
        for s in range(len(test)):
            for e in test[s, L, :]:
                tot += 1
                hits += e in keep
    return 100.0 * hits / max(tot, 1)


def transition_topk(train, test, K, n_layer):
    """Rank candidates by P(e | some x was picked at this layer last step)."""
    hits = tot = 0
    for L in range(n_layer):
        pair = collections.Counter()
        for s in range(1, len(train)):
            for x in train[s - 1, L, :].tolist():
                for y in train[s, L, :].tolist():
                    pair[(x, y)] += 1
        by_prev = collections.defaultdict(collections.Counter)
        for (x, y), c in pair.items():
            by_prev[x][y] += c
        for s in range(len(test)):
            prev = test[s - 1, L, :].tolist() if s > 0 else test[0, L, :].tolist()
            sc = collections.Counter()
            for x in prev:
                sc.update(by_prev[x])
            top = set(e for e, _ in sc.most_common(K))
            for e in test[s, L, :]:
                tot += 1
                hits += e in top
    return 100.0 * hits / max(tot, 1)


def oracle_topk(all_steps, K, n_layer):
    hits = tot = 0
    for L in range(n_layer):
        keep = set(e for e, _ in collections.Counter(all_steps[:, L, :].ravel().tolist()).most_common(K))
        for s in range(len(all_steps)):
            for e in all_steps[s, L, :]:
                tot += 1
                hits += e in keep
    return 100.0 * hits / max(tot, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True)
    p.add_argument("--n-layer", type=int, default=48)
    p.add_argument("--n-used", type=int, default=8)
    p.add_argument("--holdout", type=float, default=0.3)
    args = p.parse_args()

    A = load(args.ids, args.n_used, args.n_layer)
    cut = int(len(A) * (1 - args.holdout))
    tr, te = A[:cut], A[cut:]
    print("trace %d steps, train %d, held-out %d" % (len(A), len(tr), len(te)))

    print("\n%-6s %14s %18s %16s" % ("K/layer", "static top-K", "transition top-K", "oracle (ceiling)"))
    for K in (32, 64, 104, 250, 400):
        print("%-6d %13.1f%% %17.1f%% %15.1f%%"
              % (K, static_topk(tr, te, K, args.n_layer),
                 transition_topk(tr, te, min(K, 64), args.n_layer),
                 oracle_topk(A, K, args.n_layer)))
    print("\noracle is hindsight over the whole trace: no predictor can pass it.")


if __name__ == "__main__":
    main()