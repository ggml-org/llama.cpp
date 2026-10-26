#!/usr/bin/env python3
"""Offline simulator for the llama.cpp MoE expert cache.

Replays an expert-selection trace (LLAMA_MOE_CACHE_TRACE=path) against several
replacement policies and reports hit rate and host->device upload volume.

The SLRU implementation mirrors src/llama-moe-cache.cpp exactly so the numbers
can be validated against the hit rates that llama.cpp itself reports.

Usage:
  python moe_sim.py trace.csv --layout layout.json [--sizes 256,512,...] [--policies ...]
"""

import argparse
import collections
import csv
import json
import math
import random
import sys

# ---------------------------------------------------------------------------
# trace loading
# ---------------------------------------------------------------------------

def load_trace(path):
    """Returns a list of (layer, [expert ids]) in call order."""
    rows = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                il = int(r["layer"])
                ids = [int(x) for x in r["ids"].split()] if r["ids"] else []
            except (TypeError, ValueError):
                continue  # truncated final line
            if ids:
                rows.append((il, ids, int(r["n_ids"])))
    return rows


# ---------------------------------------------------------------------------
# policies.  each policy owns n_slots slots shared by the layers of one group
# and must implement plan(il, ids) -> (n_hit, n_fill)
# ---------------------------------------------------------------------------

class SlruCache:
    """Faithful port of moe_cache_slru from src/llama-moe-cache.cpp.

    max_protected == 0 degenerates to plain LRU, which is what the upstream
    ramp does for caches at or below one token's worth of experts.
    """

    def __init__(self, n_layer, n_expert, n_slots, max_protected):
        self.n_expert = n_expert
        self.n_slots = n_slots
        self.max_protected = max_protected
        self.slot_map = {il: [-1] * n_expert for il in range(n_layer)}
        self.key_of = [-1] * n_slots
        self.prev = [s - 1 for s in range(n_slots)]
        self.next = [s + 1 if s + 1 < n_slots else -1 for s in range(n_slots)]
        self.head = [0 if n_slots else -1, -1]
        self.tail = [n_slots - 1 if n_slots else -1, -1]
        self.is_protected = [False] * n_slots
        self.n_protected = 0
        self.seen = [0] * n_expert
        self.seen_gen = 0

    def move(self, s, protect):
        frm = 1 if self.is_protected[s] else 0
        to = 1 if protect else 0
        if frm == to and s == self.tail[frm]:
            return
        if self.prev[s] >= 0:
            self.next[self.prev[s]] = self.next[s]
        else:
            self.head[frm] = self.next[s]
        if self.next[s] >= 0:
            self.prev[self.next[s]] = self.prev[s]
        else:
            self.tail[frm] = self.prev[s]

        self.prev[s] = self.tail[to]
        self.next[s] = -1
        if self.tail[to] >= 0:
            self.next[self.tail[to]] = s
        else:
            self.head[to] = s
        self.tail[to] = s
        self.is_protected[s] = protect
        self.n_protected += to - frm

    def touch(self, s):
        self.move(s, self.max_protected > 0)
        if self.n_protected > self.max_protected:
            self.move(self.head[1], False)

    def victim(self, il):
        for seg in (0, 1):
            s = self.head[seg]
            while s >= 0:
                key = self.key_of[s]
                if key < 0 or key // self.n_expert != il or self.seen[key % self.n_expert] != self.seen_gen:
                    return s
                s = self.next[s]
        raise RuntimeError("no SLRU victim")

    def plan(self, il, ids):
        self.seen_gen += 1
        if self.seen_gen > 0xFFFFFFFF:
            self.seen = [0] * self.n_expert
            self.seen_gen = 1
        uniq = []
        for e in ids:
            if self.seen[e] != self.seen_gen:
                self.seen[e] = self.seen_gen
                uniq.append(e)
        if len(uniq) > self.n_slots:
            return None  # llama.cpp aborts; caller treats the config as invalid

        slots = self.slot_map[il]
        n_hit = 0
        for e in uniq:
            if slots[e] >= 0:
                self.touch(slots[e])
                n_hit += 1
        uniq.sort()
        n_fill = 0
        for e in uniq:
            if slots[e] >= 0:
                continue
            s = self.victim(il)
            if self.key_of[s] >= 0:
                self.slot_map[self.key_of[s] // self.n_expert][self.key_of[s] % self.n_expert] = -1
            self.key_of[s] = il * self.n_expert + e
            slots[e] = s
            self.move(s, False)
            n_fill += 1
        return n_hit, n_fill


class LfuCache:
    """Frequency-first eviction with an LRU tie-break, for comparison."""

    def __init__(self, n_layer, n_expert, n_slots, **_):
        self.n_expert = n_expert
        self.n_slots = n_slots
        self.cached = {}            # key -> freq
        self.recency = {}           # key -> last use tick
        self.freq = collections.Counter()  # key -> lifetime use count
        self.tick = 0

    def plan(self, il, ids):
        uniq = sorted(set(ids))
        if len(uniq) > self.n_slots:
            return None
        self.tick += 1
        n_hit = n_fill = 0
        keys = [il * self.n_expert + e for e in uniq]
        for k in keys:
            self.freq[k] += 1
        for k in keys:
            if k in self.cached:
                self.recency[k] = self.tick
                n_hit += 1
        protected = set(keys)
        for k in keys:
            if k in self.cached:
                continue
            if len(self.cached) >= self.n_slots:
                cand = [c for c in self.cached if c not in protected]
                if not cand:
                    cand = list(self.cached)
                ev = min(cand, key=lambda c: (self.freq[c], self.recency[c]))
                del self.cached[ev]
                self.recency.pop(ev, None)
            self.cached[k] = 1
            self.recency[k] = self.tick
            n_fill += 1
        return n_hit, n_fill


class S3FifoCache:
    """S3-FIFO: small probationary FIFO (10%), main FIFO, ghost queue."""

    def __init__(self, n_layer, n_expert, n_slots, **_):
        self.n_expert = n_expert
        self.n_slots = n_slots
        self.small_cap = max(1, n_slots // 10)
        self.main_cap = n_slots - self.small_cap
        self.small = collections.OrderedDict()  # key -> freq
        self.main = collections.OrderedDict()
        self.ghost = collections.OrderedDict()
        self.ghost_cap = self.main_cap

    def _evict(self, protected):
        while len(self.small) + len(self.main) >= self.n_slots:
            if len(self.small) >= self.small_cap and self.small:
                k, fr = self.small.popitem(last=False)
                if fr > 1:
                    self.main[k] = 0
                    self.main.move_to_end(k)
                else:
                    self.ghost[k] = 1
                    if len(self.ghost) > self.ghost_cap:
                        self.ghost.popitem(last=False)
            elif self.main:
                k = next(iter(self.main))
                fr = self.main.pop(k)
                if fr > 0 and k not in protected:
                    self.main[k] = fr - 1
                    self.main.move_to_end(k)
                elif k in protected:
                    self.main[k] = fr
                    self.main.move_to_end(k)
                    # everything in main is protected: fall back to small
                    if all(c in protected for c in self.main):
                        if self.small:
                            k2, _ = self.small.popitem(last=False)
                            self.ghost[k2] = 1
                        else:
                            return
            elif self.small:
                self.small.popitem(last=False)
            else:
                return

    def plan(self, il, ids):
        uniq = sorted(set(ids))
        if len(uniq) > self.n_slots:
            return None
        keys = [il * self.n_expert + e for e in uniq]
        protected = set(keys)
        n_hit = n_fill = 0
        for k in keys:
            if k in self.small:
                self.small[k] = min(3, self.small[k] + 1)
                n_hit += 1
            elif k in self.main:
                self.main[k] = min(3, self.main[k] + 1)
                n_hit += 1
        for k in keys:
            if k in self.small or k in self.main:
                continue
            self._evict(protected)
            if k in self.ghost:
                del self.ghost[k]
                self.main[k] = 0
            else:
                self.small[k] = 1
            n_fill += 1
        return n_hit, n_fill


class SlruFreqCache(SlruCache):
    """SLRU whose victim is the least used of `n_sample` candidates.

    The candidates are taken from the head (least recently used end) of the
    probation list, then the protected list, in the same order the upstream
    victim() walks them, so the change is confined to which of the evictable
    slots is picked.  Counters are halved every `aging` fills so that an expert
    that was hot long ago cannot hold a slot forever.
    """

    def __init__(self, n_layer, n_expert, n_slots, max_protected, n_sample=8, aging=0):
        super().__init__(n_layer, n_expert, n_slots, max_protected)
        self.use = collections.Counter()  # key -> use count
        self.n_sample = n_sample
        self.aging = aging if aging else 16 * max(1, n_slots)
        self.n_fills = 0

    def plan(self, il, ids):
        for e in set(ids):
            self.use[il * self.n_expert + e] += 1
        return super().plan(il, ids)

    def victim(self, il):
        best = -1
        best_use = None
        n = 0
        for seg in (0, 1):
            s = self.head[seg]
            while s >= 0 and n < self.n_sample:
                key = self.key_of[s]
                if key < 0 or key // self.n_expert != il or self.seen[key % self.n_expert] != self.seen_gen:
                    if key < 0:
                        return s  # empty slot, nothing better exists
                    u = self.use[key]
                    if best_use is None or u < best_use:
                        best, best_use = s, u
                    n += 1
                s = self.next[s]
        if best >= 0:
            self.n_fills += 1
            if self.n_fills % self.aging == 0:
                for k in list(self.use):
                    self.use[k] >>= 1
                    if not self.use[k]:
                        del self.use[k]
            return best
        return super().victim(il)


class SlruRandCache(SlruCache):
    """SLRU with a uniformly random victim: the cheapest cure for LRU thrash.

    A cyclic sweep over more keys than there are slots makes LRU evict exactly
    the entries it is about to need, while random replacement keeps a random
    (and therefore partly reused) subset.
    """

    def __init__(self, n_layer, n_expert, n_slots, max_protected, seed=1234):
        super().__init__(n_layer, n_expert, n_slots, max_protected)
        self.rng = random.Random(seed)

    def victim(self, il):
        for _ in range(16):
            s = self.rng.randrange(self.n_slots)
            key = self.key_of[s]
            if key < 0 or key // self.n_expert != il or self.seen[key % self.n_expert] != self.seen_gen:
                return s
        return super().victim(il)


class SlruSampleFreqCache(SlruCache):
    """SLRU whose victim is the least used of k uniformly sampled slots.

    Sampling the whole cache rather than the LRU end approximates LFU (the same
    trick Redis uses for its LFU eviction) and needs no extra data structure
    beyond one counter per cached expert.
    """

    def __init__(self, n_layer, n_expert, n_slots, max_protected, n_sample=8, seed=1234, aging=0):
        super().__init__(n_layer, n_expert, n_slots, max_protected)
        self.use = collections.Counter()
        self.n_sample = n_sample
        self.rng = random.Random(seed)
        self.aging = aging
        self.n_fills = 0

    def plan(self, il, ids):
        for e in set(ids):
            self.use[il * self.n_expert + e] += 1
        return super().plan(il, ids)

    def victim(self, il):
        best = -1
        best_use = None
        n = 0
        tries = 0
        while n < self.n_sample and tries < 4 * self.n_sample:
            tries += 1
            s = self.rng.randrange(self.n_slots)
            key = self.key_of[s]
            if key >= 0 and key // self.n_expert == il and self.seen[key % self.n_expert] == self.seen_gen:
                continue
            if key < 0:
                return s
            u = self.use[key]
            if best_use is None or u < best_use:
                best, best_use = s, u
            n += 1
        if self.aging:
            self.n_fills += 1
            if self.n_fills % self.aging == 0:
                for k in list(self.use):
                    self.use[k] >>= 1
                    if not self.use[k]:
                        del self.use[k]
        return best if best >= 0 else super().victim(il)


class BeladyCache:
    """Offline optimum: evict the entry whose next use is furthest away."""

    def __init__(self, n_layer, n_expert, n_slots, future=None, **_):
        self.n_expert = n_expert
        self.n_slots = n_slots
        self.cached = set()
        self.future = future  # key -> deque of positions in this group's call sequence
        self.call = -1

    def plan(self, il, ids):
        self.call += 1
        uniq = sorted(set(ids))
        if len(uniq) > self.n_slots:
            return None
        keys = [il * self.n_expert + e for e in uniq]
        for k in keys:
            q = self.future.get(k)
            while q and q[0] <= self.call:
                q.popleft()
        n_hit = sum(1 for k in keys if k in self.cached)
        protected = set(keys)

        def next_use(k):
            q = self.future.get(k)
            return q[0] if q else math.inf

        n_fill = 0
        for k in keys:
            if k in self.cached:
                continue
            if len(self.cached) >= self.n_slots:
                cand = [c for c in self.cached if c not in protected]
                if not cand:
                    cand = list(self.cached)
                ev = max(cand, key=next_use)
                self.cached.discard(ev)
            self.cached.add(k)
            n_fill += 1
        return n_hit, n_fill


# ---------------------------------------------------------------------------
# group model: mirrors the per-group budget split in llama-moe-cache.cpp
# ---------------------------------------------------------------------------

GGML_ALIGN = 256


def ggml_pad(x, n):
    return ((x + n - 1) // n) * n


class Group:
    def __init__(self, layers, per_tensor_bytes, n_expert):
        self.layers = list(layers)
        self.per_tensor = list(per_tensor_bytes)  # bytes of one expert per tensor
        self.n_expert = n_expert

    @property
    def host_bytes(self):
        return sum(self.per_tensor) * self.n_expert * len(self.layers)

    def alloc_size(self, n_slots):
        return sum(ggml_pad(nb * (n_slots + 1), GGML_ALIGN) for nb in self.per_tensor)

    def n_slots_for(self, budget):
        max_slots = len(self.layers) * self.n_expert
        n = 0
        while n < max_slots and self.alloc_size(n + 1) <= budget:
            n += 1
        return n

    def expert_bytes(self):
        return sum(self.per_tensor)


def build_groups(layout):
    groups = []
    for g in layout["groups"]:
        groups.append(Group(g["layers"], g["per_tensor_bytes"], layout["n_expert"]))
    return groups


def max_protected_ramp(n_slots, n_expert_token):
    """PR 30848: 0% protected at 1x a token's experts, 50% at 1.8x."""
    if n_expert_token == 0:
        return 0
    ratio = n_slots / n_expert_token
    progress = min(max((ratio - 1.0) / 0.8, 0.0), 1.0)
    return int(n_slots * progress * 0.5)


POLICIES = {
    "lru":        lambda n_layer, ne, ns, net, fut: SlruCache(n_layer, ne, ns, 0),
    "slru50":     lambda n_layer, ne, ns, net, fut: SlruCache(n_layer, ne, ns, ns // 2),
    "slru-ramp":  lambda n_layer, ne, ns, net, fut: SlruCache(n_layer, ne, ns, max_protected_ramp(ns, net)),
    "lfu":        lambda n_layer, ne, ns, net, fut: LfuCache(n_layer, ne, ns),
    "s3fifo":     lambda n_layer, ne, ns, net, fut: S3FifoCache(n_layer, ne, ns),
    "belady":     lambda n_layer, ne, ns, net, fut: BeladyCache(n_layer, ne, ns, future=fut),
    # sampled least-frequently-used victim on top of the upstream ramp
    "ramp-freq4":  lambda n_layer, ne, ns, net, fut: SlruFreqCache(n_layer, ne, ns, max_protected_ramp(ns, net), 4),
    "ramp-freq8":  lambda n_layer, ne, ns, net, fut: SlruFreqCache(n_layer, ne, ns, max_protected_ramp(ns, net), 8),
    "ramp-freq16": lambda n_layer, ne, ns, net, fut: SlruFreqCache(n_layer, ne, ns, max_protected_ramp(ns, net), 16),
    "lru-freq8":   lambda n_layer, ne, ns, net, fut: SlruFreqCache(n_layer, ne, ns, 0, 8),
    # uniformly random victim
    "ramp-rand":   lambda n_layer, ne, ns, net, fut: SlruRandCache(n_layer, ne, ns, max_protected_ramp(ns, net)),
    "lru-rand":    lambda n_layer, ne, ns, net, fut: SlruRandCache(n_layer, ne, ns, 0),
    # least used of k uniformly sampled slots
    "ramp-s8":     lambda n_layer, ne, ns, net, fut: SlruSampleFreqCache(n_layer, ne, ns, max_protected_ramp(ns, net), 8),
    "ramp-s16":    lambda n_layer, ne, ns, net, fut: SlruSampleFreqCache(n_layer, ne, ns, max_protected_ramp(ns, net), 16),
    "ramp-s32":    lambda n_layer, ne, ns, net, fut: SlruSampleFreqCache(n_layer, ne, ns, max_protected_ramp(ns, net), 32),
    "lru-s8":      lambda n_layer, ne, ns, net, fut: SlruSampleFreqCache(n_layer, ne, ns, 0, 8),
    "lru-s16":     lambda n_layer, ne, ns, net, fut: SlruSampleFreqCache(n_layer, ne, ns, 0, 16),
    # same, with the counters halved every N fills to forget stale popularity
    "ramp-s16-age1":  lambda n_layer, ne, ns, net, fut: SlruSampleFreqCache(n_layer, ne, ns, max_protected_ramp(ns, net), 16, aging=1*max(1, ns)),
    "ramp-s16-age4":  lambda n_layer, ne, ns, net, fut: SlruSampleFreqCache(n_layer, ne, ns, max_protected_ramp(ns, net), 16, aging=4*max(1, ns)),
    "ramp-s16-age2":  lambda n_layer, ne, ns, net, fut: SlruSampleFreqCache(n_layer, ne, ns, max_protected_ramp(ns, net), 16, aging=2*max(1, ns)),
    "ramp-s16-age8":  lambda n_layer, ne, ns, net, fut: SlruSampleFreqCache(n_layer, ne, ns, max_protected_ramp(ns, net), 16, aging=8*max(1, ns)),
    "ramp-s16-age16": lambda n_layer, ne, ns, net, fut: SlruSampleFreqCache(n_layer, ne, ns, max_protected_ramp(ns, net), 16, aging=16*max(1, ns)),
}


def simulate(trace, layout, cache_mib, policy, n_expert_used, small_only=True):
    groups = build_groups(layout)
    total_host = sum(g.host_bytes for g in groups)
    n_layer = layout["n_layer"]
    size = cache_mib * 1024 * 1024

    layer_group = {}
    caches = {}
    stats = {}
    for gi, g in enumerate(groups):
        budget = int(size * g.host_bytes / total_host)
        n_slots = g.n_slots_for(budget)
        n_expert_token = len(g.layers) * n_expert_used
        for il in g.layers:
            layer_group[il] = gi
        if n_slots < n_expert_used:
            caches[gi] = None
            stats[gi] = dict(n_slots=0, hits=0, fills=0, bytes=0, skipped=0,
                             expert_bytes=g.expert_bytes(), n_slots_ratio=0.0)
            continue
        fut = None
        if policy == "belady":
            # index the future by position in THIS group's call sequence, which is
            # the order plan() sees, so BeladyCache.call lines up with it
            fut = collections.defaultdict(collections.deque)
            pos = 0
            layers_of_g = set(g.layers)
            for il, ids, n_ids in trace:
                if il not in layers_of_g:
                    continue
                if small_only and n_ids > 8 * n_expert_used:
                    continue
                for e in set(ids):
                    fut[il * layout["n_expert"] + e].append(pos)
                pos += 1
        caches[gi] = POLICIES[policy](n_layer, layout["n_expert"], n_slots, n_expert_token, fut)
        stats[gi] = dict(n_slots=n_slots, hits=0, fills=0, bytes=0, skipped=0,
                         expert_bytes=g.expert_bytes(),
                         n_slots_ratio=n_slots / n_expert_token if n_expert_token else 0.0)

    for il, ids, n_ids in trace:
        gi = layer_group.get(il)
        if gi is None or caches[gi] is None:
            continue
        if small_only and n_ids > 8 * n_expert_used:
            continue
        res = caches[gi].plan(il, ids)
        if res is None:
            stats[gi]["skipped"] += 1
            continue
        n_hit, n_fill = res
        stats[gi]["hits"] += n_hit
        stats[gi]["fills"] += n_fill
        stats[gi]["bytes"] += n_fill * stats[gi]["expert_bytes"]

    hits = sum(s["hits"] for s in stats.values())
    fills = sum(s["fills"] for s in stats.values())
    byts = sum(s["bytes"] for s in stats.values())
    vram = 0
    for gi, g in enumerate(groups):
        if stats[gi]["n_slots"]:
            vram += g.alloc_size(stats[gi]["n_slots"])
    return dict(
        cache_mib=cache_mib, policy=policy, hits=hits, fills=fills,
        hit_rate=100.0 * hits / (hits + fills) if hits + fills else 0.0,
        uploaded_mib=byts / 2 ** 20, vram_mib=vram / 2 ** 20,
        per_group={gi: dict(s) for gi, s in stats.items()},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--layout", required=True)
    ap.add_argument("--sizes", default="128,256,384,512,768,1024,1536,2048,3072,4096")
    ap.add_argument("--policies", default="lru,slru50,slru-ramp,lfu,s3fifo,belady")
    ap.add_argument("--n-expert-used", type=int, default=8)
    ap.add_argument("--all-batches", action="store_true",
                    help="include large-ubatch calls instead of generation only")
    ap.add_argument("--json", help="write raw results here")
    args = ap.parse_args()

    layout = json.load(open(args.layout))
    trace = load_trace(args.trace)
    n_small = sum(1 for _, _, n in trace if n <= 8 * args.n_expert_used)
    print(f"trace: {len(trace)} calls ({n_small} generation-sized), "
          f"{len({il for il, _, _ in trace})} layers", file=sys.stderr)

    sizes = [int(x) for x in args.sizes.split(",")]
    policies = args.policies.split(",")
    out = []
    print(f"{'MiB':>6} {'policy':>10} {'slots/tok':>10} {'hit%':>7} {'uploaded MiB':>13} {'VRAM MiB':>9}")
    for mib in sizes:
        for pol in policies:
            r = simulate(trace, layout, mib, pol, args.n_expert_used,
                         small_only=not args.all_batches)
            ratios = [f"{s['n_slots_ratio']:.2f}" for s in r["per_group"].values()]
            print(f"{mib:>6} {pol:>10} {'/'.join(ratios):>10} {r['hit_rate']:>7.2f} "
                  f"{r['uploaded_mib']:>13.1f} {r['vram_mib']:>9.1f}")
            out.append(r)
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
