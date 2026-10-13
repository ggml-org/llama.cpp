#!/usr/bin/env python3
"""How predictable is this model's MoE routing from the input token?

The split-free decode path needs the experts for token t+1 resident before
token t+1's graph runs. The router runs inside that graph, so the only way
to know them in advance is to predict them. This measures whether a
token-conditioned map can do that.

Inputs (produced with MOE_TRACE_DIR=<prefix> plus the server's own text,
retokenized so step t's routing pairs with the token fed into step t):
    <prefix>.ids.u16   u16[n_steps, n_layer, n_used]
    <prefix>.tokens.json   the generated token ids, in step order
"""
import argparse
import collections
import json

import numpy as np


def load(prefix, n_used, n_layer):
    a = np.fromfile(prefix + ".ids.u16", dtype=np.uint16)
    toks = json.load(open(prefix + ".tokens.json"))
    n = min(len(a) // (n_layer * n_used), len(toks))
    return (a[:n * n_layer * n_used].reshape(n, n_layer, n_used), toks[:n])


def loo_token_recall(a, toks, n_layer):
    """Leave-one-out recall of the best fixed top-n_used set per (token, layer)."""
    by = collections.defaultdict(list)
    for i, w in enumerate(toks):
        by[w].append(i)
    hits = tot = 0
    for steps in by.values():
        if len(steps) < 3:
            continue
        for L in range(n_layer):
            occ = [set(a[s, L, :].tolist()) for s in steps]
            for i in range(len(steps)):
                cnt = collections.Counter()
                for j in range(len(steps)):
                    if j != i:
                        for e in occ[j]:
                            cnt[e] += 1
                top = set(e for e, _ in cnt.most_common(a.shape[2]))
                for e in occ[i]:
                    tot += 1
                    hits += e in top
    return 100.0 * hits / max(tot, 1), tot


def global_lfu_recall(a, n_layer):
    hits = tot = 0
    for L in range(n_layer):
        cnt = collections.Counter(a[:, L, :].ravel().tolist())
        top = set(e for e, _ in cnt.most_common(a.shape[2]))
        for s in range(len(a)):
            for e in a[s, L, :]:
                tot += 1
                hits += e in top
    return 100.0 * hits / max(tot, 1)


def oracle_capacity(a, n_layer, k):
    hits = tot = 0
    for L in range(n_layer):
        cnt = collections.Counter(a[:, L, :].ravel().tolist())
        keep = set(e for e, _ in cnt.most_common(k))
        for s in range(len(a)):
            for e in a[s, L, :]:
                tot += 1
                hits += e in keep
    return 100.0 * hits / max(tot, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True, help="trace prefix (MOE_TRACE_DIR)")
    p.add_argument("--n-layer", type=int, default=48)
    p.add_argument("--n-used", type=int, default=8)
    p.add_argument("--slots-per-layer", type=int, default=104)
    args = p.parse_args()
    a, toks = load(args.prefix, args.n_used, args.n_layer)

    by = collections.defaultdict(list)
    for i, w in enumerate(toks):
        by[w].append(i)
    repeated = sum(len(v) for v in by.values() if len(v) >= 3)

    print("steps=%d  tokens=%d  distinct=%d  steps on a token seen >=3x: %d"
          % (len(a), len(toks), len(by), repeated))

    ident = tot_i = 0
    for steps in by.values():
        if len(steps) < 2:
            continue
        for L in range(args.n_layer):
            sets = [frozenset(a[s, L, :].tolist()) for s in steps]
            for i in range(1, len(sets)):
                tot_i += 1
                ident += sets[i] == sets[i - 1]
    print("repeated-token routing set identical to the previous occurrence: %.1f%%"
          % (100.0 * ident / max(tot_i, 1)))

    tok_recall, n = loo_token_recall(a, toks, args.n_layer)
    print("leave-one-out recall, best token-conditioned top-%d : %.1f%% (n=%d)"
          % (args.n_used, tok_recall, n))
    print("global per-layer LFU top-%d recall (no tokens)     : %.1f%%"
          % (args.n_used, global_lfu_recall(a, args.n_layer)))
    print("oracle top-%d/layer coverage (pool capacity)        : %.1f%%"
          % (args.slots_per_layer,
             oracle_capacity(a, args.n_layer, args.slots_per_layer)))


if __name__ == "__main__":
    main()