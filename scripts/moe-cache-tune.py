#!/usr/bin/env python3
# Picks --moe-cache-mib and --moe-cache-cpu-share for a llama-server command (see docs/moe-cache.md):
#   python3 scripts/moe-cache-tune.py [--max-mib N] -- llama-server -m model.gguf -cmoe [args]
# Runs the server at 2 cache sizes x 3 CPU shares, fits the time per token, and replays the recorded
# routing through a copy of the cache logic (src/llama-moe-cache.cpp) to predict other sizes and shares.

import argparse
import json
import logging
import math
import os
import re
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.request

logger = logging.getLogger("moe-cache-tune")

MASK64 = (1 << 64) - 1
MAX_BATCH = 32  # llama_moe_cache::impl::max_batch
SHARES = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98]
CALIB_SHARES = [0.0, 0.5, 0.95]
N_FIRST = 2048  # tokens per request, first run
N_CALIB = 512   # tokens per request, other runs
PROMPT = ("The lighthouse keeper had kept a logbook for thirty-one years, and in all that time he had never once written down a lie. "
          "So when the ship appeared on the horizon on the first calm morning of October, with no flag, no lights and no one on deck, "
          "he sat at his small wooden table for a long while before he picked up the pen. Below him the sea was flat and grey, and the "
          "gulls had gone quiet. He wrote the date, the hour and the direction of the wind, and then he stopped, because the next line "
          "would have to say what he had seen, and he was no longer sure that he had seen it at all. This is what he finally wrote:")


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def pad(x, n):
    return (x + n - 1) // n * n


# routing trace (LLAMA_MOE_CACHE_TRACE)

class Group:
    n_expert: int
    layers: list
    split: float
    split_sum: float
    host_bytes: int
    dev_host_bytes: int
    alignment: int
    nb2: list
    expert_bytes: int


def read_trace(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"MCT1":
        sys.exit(f"{path}: not a MoE cache trace")
    pos = 4

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return v

    n_layer, n_expert_used, n_group = take("3I")
    groups = []
    for _ in range(n_group):
        g = Group()
        _, g.n_expert, n_layers = take("3I")
        g.layers = list(take(f"{n_layers}I"))
        g.split, g.split_sum = take("2d")
        g.host_bytes, g.dev_host_bytes, g.alignment = take("3Q")
        (n_tensors,) = take("I")
        g.nb2 = list(take(f"{n_tensors}Q"))
        g.expert_bytes = sum(g.nb2)
        groups.append(g)

    records = []
    while pos < len(data):
        il, n_used, n_ids = struct.unpack_from("<3i", data, pos)
        pos += 12
        records.append((il, n_used, struct.unpack_from(f"<{n_ids}i", data, pos)))
        pos += 4 * n_ids
    return n_layer, n_expert_used, groups, records


def concat_traces(paths):
    traces = [read_trace(p) for p in paths]
    n_layer, n_expert_used, groups, _ = traces[0]
    return n_layer, n_expert_used, groups, [rec for t in traces for rec in t[3]]


def trace_tokens(trace):
    first = min(il for il, _, _ in trace[3])
    return sum(len(ids) // n_used for il, n_used, ids in trace[3] if il == first)


# cache simulation, mirrors src/llama-moe-cache.cpp

def n_slots_of(g, size_mib, n_zero, n_expert_used):
    budget = int(float(size_mib * 1024 * 1024) * g.split / g.split_sum * float(g.host_bytes) / float(g.dev_host_bytes))

    def alloc(n):
        return sum(pad(nb2 * (n + n_zero + 1), g.alignment) for nb2 in g.nb2)

    if alloc(0) > budget:
        return 0
    lo, hi = 0, len(g.layers) * g.n_expert
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if alloc(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo if lo >= n_expert_used else 0


def full_mib(trace, share):
    """smallest cache that holds all experts"""
    _, n_expert_used, groups, _ = trace
    n_zero = min(n_expert_used, 64) if f32(share) > 0.0 else 0
    res = 0.0
    for g in groups:
        alloc = sum(pad(nb2 * (len(g.layers) * g.n_expert + n_zero + 1), g.alignment) for nb2 in g.nb2)
        res = max(res, alloc / (g.split / g.split_sum * g.host_bytes / g.dev_host_bytes))
    return math.ceil(res / 2**20) + 1


class SLRU:
    """moe_cache_slru"""

    def __init__(self, layers, n_expert, n_slots):
        self.n_expert = n_expert
        self.n_slots = n_slots
        self.slot_map = {il: [n_slots] * n_expert for il in layers}
        self.key_of = [-1] * n_slots
        self.prev = [s - 1 for s in range(n_slots)]
        self.next = [s + 1 if s + 1 < n_slots else -1 for s in range(n_slots)]
        self.head = [0, -1]
        self.tail = [n_slots - 1, -1]
        self.is_protected = [False] * n_slots
        self.n_protected = 0
        self.n_filled = 0
        self.seen = [0] * n_expert
        self.seen_gen = 0

    def move(self, s, protect):
        frm = self.is_protected[s]
        if frm == protect and s == self.tail[frm]:
            return
        prev, nxt, head, tail = self.prev, self.next, self.head, self.tail
        if prev[s] >= 0:
            nxt[prev[s]] = nxt[s]
        else:
            head[frm] = nxt[s]
        if nxt[s] >= 0:
            prev[nxt[s]] = prev[s]
        else:
            tail[frm] = prev[s]
        prev[s] = tail[protect]
        nxt[s] = -1
        if tail[protect] >= 0:
            nxt[tail[protect]] = s
        else:
            head[protect] = s
        tail[protect] = s
        self.is_protected[s] = protect
        self.n_protected += int(protect) - int(frm)

    def touch(self, s):
        self.move(s, True)
        if self.n_protected > self.n_slots // 2:
            self.move(self.head[1], False)

    def victim(self, il):
        n_expert, key_of, seen, gen = self.n_expert, self.key_of, self.seen, self.seen_gen
        for segment in (0, 1):
            s = self.head[segment]
            while s >= 0:
                key = key_of[s]
                if key < 0 or key // n_expert != il or seen[key % n_expert] != gen:
                    return s
                s = self.next[s]
        raise RuntimeError("no SLRU victim")

    def plan(self, il, ids, n_used, n_zero, n_cpu, rng):
        """returns (uploaded, computed on the CPU, rng)"""
        self.seen_gen += 1
        seen, gen = self.seen, self.seen_gen
        uniq = []
        for e in ids:
            if seen[e] != gen:
                seen[e] = gen
                uniq.append(e)
        if len(uniq) > self.n_slots:
            raise RuntimeError(f"layer {il}: {len(uniq)} experts for {self.n_slots} slots")

        slots = self.slot_map[il]
        n_slots = self.n_slots
        misses = []
        for e in uniq:
            if slots[e] < n_slots:
                self.touch(slots[e])
            else:
                misses.append(e)

        n_c = min(n_cpu(len(misses)), len(misses))
        for i in range(n_c):
            rng = (rng * 6364136223846793005 + 1442695040888963407) & MASK64
            j = i + (rng >> 33) % (len(misses) - i)
            misses[i], misses[j] = misses[j], misses[i]
        n_cpu_done = 0
        if n_c > 0:
            # distinct zero slots per token, a miss without one is uploaded
            n_tokens = len(ids) // n_used
            token_zero = [0] * n_tokens
            n_keep = n_c
            i = 0
            while i < n_keep:
                e = misses[i]
                toks = [it for it in range(n_tokens) if e in ids[it * n_used:(it + 1) * n_used]]
                used = 0
                for it in toks:
                    used |= token_zero[it]
                z = ((~used) & (used + 1)).bit_length() - 1
                if z >= n_zero:
                    n_keep -= 1
                    misses[i], misses[n_keep] = misses[n_keep], misses[i]
                    continue
                for it in toks:
                    token_zero[it] |= 1 << z
                slots[e] = n_slots + z
                n_cpu_done += 1
                i += 1
            del misses[:n_keep]

        misses.sort()
        for e in misses:
            s = self.victim(il)
            key = self.key_of[s]
            if key >= 0:
                self.slot_map[key // self.n_expert][key % self.n_expert] = n_slots
            else:
                self.n_filled += 1
            self.key_of[s] = il * self.n_expert + e
            slots[e] = s
            self.move(s, False)
        return len(misses), n_cpu_done, rng


def replay(trace, size_mib, share, skip_tokens):
    """GiB of experts uploaded and computed on the CPU per token, after skip_tokens"""
    n_layer, n_expert_used, groups, records = trace
    share = f32(share)
    share_q = int(math.floor(f32(share * 65536.0) + 0.5))  # std::lround(cpu_share*65536.0f)
    n_zero = min(n_expert_used, 64) if share > 0.0 else 0
    caches, group_of = [], {}
    for ig, g in enumerate(groups):
        n = n_slots_of(g, size_mib, n_zero, n_expert_used)
        caches.append(SLRU(g.layers, g.n_expert, n) if n > 0 else None)
        group_of.update((il, ig) for il in g.layers)

    first = min(il for il, _, _ in records)
    acc = [0] * n_layer
    rng = 0x853c49e6748fea9b
    tokens = seen = up = cpu = 0
    for il, n_used, ids in records:
        n_tokens = len(ids) // n_used
        if il == first:
            seen += n_tokens
            if seen > skip_tokens:
                tokens += n_tokens
        count = seen > skip_tokens
        g, c = groups[group_of[il]], caches[group_of[il]]
        if c is None or n_tokens > MAX_BATCH or min(n_tokens * n_used, g.n_expert) > c.n_slots:
            if count:
                cpu += len(set(ids)) * g.expert_bytes  # not cached
            continue

        def n_cpu(n_miss):
            v = n_miss * (share_q if c.n_filled == c.n_slots else 0) + acc[il]  # the share applies once the cache is full
            acc[il] = v & 0xFFFF
            return v >> 16

        n_fill, n_c, rng = c.plan(il, ids, n_used, n_zero, n_cpu, rng)
        if count:
            up += n_fill * g.expert_bytes
            cpu += n_c * g.expert_bytes
    tokens = max(tokens, 1)
    return up / 2**30 / tokens, cpu / 2**30 / tokens


# time model: ms/token = base + up*up_ms + cpu*cpu_ms + split*(share > 0) + overlap*min(up, cpu), up and cpu in GiB/token

def model_ms(m, up, cpu, share):
    return m["base"] + m["up"] * up + m["cpu"] * cpu + (m["split"] if share > 0 else 0.0) + m["overlap"] * min(up, cpu)


def solve(A, b):
    """least squares by the normal equations"""
    n = len(A[0])
    M: list = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)] + [sum(A[k][i] * b[k] for k in range(len(A)))] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        if abs(M[i][i]) < 1e-12:
            sys.exit("the runs do not determine the time model")
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]
                M[r] = [x - f * y for x, y in zip(M[r], M[i])]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit(runs):
    """returns the model, the relative error per run and whether the model is physical"""
    y = [r["ms"] / r["tokens"] for r in runs]
    rows = [[1.0, r["up_gib"], r["cpu_gib"], float(r["share"] > 0), min(r["up_gib"], r["cpu_gib"])] for r in runs]

    def solve_model(cols):
        coef = dict(zip(cols, solve([[row[i] for i in cols.values()] for row in rows], y)))
        if coef.get("split", 0) < 0:  # noise
            del cols["split"]
            return solve_model(cols)
        return {k: coef.get(k, 0.0) for k in ("base", "up", "cpu", "split", "overlap")}

    def physical(m):
        # overlap hides at most the cheaper path, the rest of the token is a good part of the fastest run
        return m["up"] > 0 and m["cpu"] > 0 and m["base"] >= 0.2 * min(y) and -m["overlap"] <= min(m["up"], m["cpu"])

    m = solve_model({"base": 0, "up": 1, "cpu": 2, "split": 3, "overlap": 4})
    if not physical(m):
        m = solve_model({"base": 0, "up": 1, "cpu": 2, "split": 3})
    return m, [(model_ms(m, r["up_gib"], r["cpu_gib"], r["share"]) - yy) / yy for r, yy in zip(runs, y)], physical(m)


# output

FMT = {"markdown": False, "tty": False}


def say(text=""):
    print(text)  # noqa: NP100


def best(text):
    if FMT["markdown"]:
        return f"**{text}**"
    return f"\033[1m{text}\033[0m" if FMT["tty"] else f"{text}*"


def code(text):
    return f"`{text}`" if FMT["markdown"] else text


def table(header, rows):
    rows = [[str(c) for c in row] for row in rows]
    if FMT["markdown"]:
        return "\n".join(["| " + " | ".join(header) + " |", "|" + "---:|" * len(header)] + ["| " + " | ".join(row) + " |" for row in rows])

    def width(text):
        return len(re.sub(r"\033\[[0-9;]*m", "", text))

    w = [max(width(c) for c in col) for col in zip(header, *rows)]
    return "\n".join("  " + "  ".join(" " * (wi - width(c)) + c for c, wi in zip(row, w)) for row in [header] + rows)


# model file

GGUF_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
EXPS_RE = re.compile(r"\.ffn_(up|down|gate|gate_up)_(ch|)exps")  # LLM_FFN_EXPS_REGEX


def model_files(path):
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
    return [f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf" for i in range(1, int(m.group(3)) + 1)] if m else [path]


def gguf_tensor_sizes(path):
    """tensor sizes from the data offsets"""
    with open(path, "rb") as f:
        def take(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

        def string():
            (n,) = take("Q")
            return f.read(n).decode("utf-8", "replace")

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                et, n = take("IQ")
                if et in (8, 9):
                    for _ in range(n):
                        value(et)
                else:
                    f.seek(n * struct.calcsize("<" + GGUF_SCALAR[et]), 1)
                return None
            return take(GGUF_SCALAR[t])[0]

        if f.read(4) != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        take("I")
        n_tensors, n_kv = take("QQ")
        kv = {}
        for _ in range(n_kv):
            key = string()
            kv[key] = value(take("I")[0])
        infos = []
        for _ in range(n_tensors):
            name = string()
            take(f"{take('I')[0]}Q")
            take("I")
            infos.append((take("Q")[0], name))
        data = pad(f.tell(), kv.get("general.alignment", 32))
    end = os.path.getsize(path) - data
    infos.sort()
    return {name: (infos[i + 1][0] if i + 1 < len(infos) else end) - off for i, (off, name) in enumerate(infos)}


def expert_mib(path):
    sizes = {}
    for p in model_files(path):
        sizes.update(gguf_tensor_sizes(p))
    return sum(v for k, v in sizes.items() if EXPS_RE.search(k)) / 2**20


# server runs

def http(port, endpoint, body=None, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{endpoint}", data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def cache_stats(log):
    """the statistics the cache logs at exit (-lv 4)"""
    r = {"hits": 0, "misses": 0, "cpu": 0, "up_mib": 0.0}
    for m in re.finditer(r"llama_moe_cache: ubatch .. 8: hits = (\d+), misses = (\d+)(?: \((\d+) on the CPU\))?, hit rate = [0-9.]+%, uploaded = ([0-9.]+) MiB", log):
        r["hits"] += int(m.group(1))
        r["misses"] += int(m.group(2))
        r["cpu"] += int(m.group(3) or 0)
        r["up_mib"] += float(m.group(4))
    return r


def device_room(log):
    """host experts, cache device, its free memory at load and the memory the model uses on it"""
    host = sum(float(x) for x in re.findall(r"MoE cache size =\s+[0-9.]+ MiB for ([0-9.]+) MiB of host experts", log))
    m = re.search(r"(\S+) MoE cache size =", log)
    if host == 0 or not m:
        sys.exit("no MoE cache: the experts must be in host memory (-cmoe, -ncmoe or -ot)")
    free = {d: int(x) for d, x in re.findall(r"using device (\S+) \(.*- (\d+) MiB free", log)}
    used = sum(float(x) for d, x in re.findall(r"(\S+) +(?:model|KV|RS|compute|output) +buffer size =\s+([0-9.]+) MiB", log) if d == m.group(1))
    if m.group(1) not in free:
        sys.exit(f"the free memory of {m.group(1)} is not in the log, pass --max-mib")
    return host, m.group(1), free[m.group(1)], used


def run_server(cmd, out, mib, share, n_predict, seed, label, warm):
    """two generations with the cache, reused if already in out"""
    base = os.path.join(out, f"run-{mib}-{share:g}")
    if os.path.isfile(base + ".json") and os.path.isfile(base + ".trace"):
        r = json.load(open(base + ".json"))
    else:
        warm()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        args = cmd + ["--host", "127.0.0.1", "--port", str(port), "-lv", "4", "--moe-cache-mib", str(mib), "--moe-cache-cpu-share", str(share)]
        timings: list = []
        for attempt in range(2):
            with open(base + ".log", "w") as log:
                proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, LLAMA_MOE_CACHE_TRACE=base + ".trace"))
            try:
                while True:
                    if proc.poll() is not None:
                        sys.exit(f"llama-server exited ({proc.returncode}), see {base}.log")
                    try:
                        if http(port, "/health", timeout=5).get("status") == "ok":
                            break
                    except OSError:
                        pass
                    time.sleep(1)
                n = min(n_predict, http(port, "/props")["default_generation_settings"]["n_ctx"] - 256)
                timings = [http(port, "/completion", {"prompt": PROMPT, "n_predict": n, "ignore_eos": True, "cache_prompt": False,
                                                      "temperature": 0.8, "top_p": 0.95, "seed": seed + k})["timings"] for k in (0, 1)]
            finally:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
            # the second generation has a warmer cache, it is much slower only if the machine is busy
            if timings[1]["predicted_per_second"] > timings[0]["predicted_per_second"] / 1.5 or attempt == 1:
                break
            logger.info("  slower second generation, machine busy? running again")
        r = cache_stats(open(base + ".log", errors="replace").read())
        r.update(mib=mib, share=share, ms=sum(t["predicted_ms"] for t in timings), tokens=sum(t["predicted_n"] for t in timings))
        json.dump(r, open(base + ".json", "w"))
        logger.info(f"{label} {mib} MiB, share {share:g}: {1000 * r['tokens'] / r['ms']:.1f} t/s")
    r["hit"] = 100 * r["hits"] / max(r["hits"] + r["misses"], 1)
    return r


def tune(cmd, out, max_mib):
    model = next((cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a in ("-m", "--model")), None)
    if model is None:
        sys.exit("the server command needs -m")
    os.makedirs(out, exist_ok=True)
    command = os.path.join(out, "command.txt")
    if os.path.isfile(command) and open(command).read() != shlex.join(cmd):
        for f in os.listdir(out):
            os.remove(os.path.join(out, f))
    open(command, "w").write(shlex.join(cmd))

    warmed = []

    def warm():
        # read an mmap'd model once, so that no run reads it from disk
        if not warmed and not any(a in cmd for a in ("--no-mmap", "-lm", "--load-mode")):
            for p in model_files(model):
                with open(p, "rb") as f:
                    while f.read(64 << 20):
                        pass
        warmed.append(True)

    def rnd(x):
        return max(64, int(x) // 64 * 64)

    n = 2 * len(CALIB_SHARES)
    small = rnd(0.10 * expert_mib(model))
    runs = [run_server(cmd, out, small, 0.0, N_FIRST, 1, f"1/{n}", warm)]
    host, dev, free, used = device_room(open(os.path.join(out, f"run-{small}-0.log"), errors="replace").read())
    room = int(free - used - 512)
    cap = min(max_mib, room) if max_mib else room
    large = rnd(min(0.30 * host, room))
    if large <= small:
        sys.exit(f"no room on {dev} for a cache larger than {small} MiB")
    for i, (mib, share) in enumerate([(small, s) for s in CALIB_SHARES[1:]] + [(large, s) for s in CALIB_SHARES]):
        runs.append(run_server(cmd, out, mib, share, N_CALIB, 2 * i + 3, f"{i + 2}/{n}", warm))

    trace = concat_traces([os.path.join(out, f"run-{r['mib']}-{r['share']:g}.trace") for r in runs])
    groups = trace[2]
    gib = sum(g.expert_bytes * len(g.layers) for g in groups) / sum(len(g.layers) for g in groups) / 2**30
    for r in runs:
        r["up_gib"], r["cpu_gib"] = r["up_mib"] / 1024 / r["tokens"], r["cpu"] * gib / r["tokens"]
    m, errs, ok = fit(runs)

    full = full_mib(trace, max(SHARES))
    top = min(cap, full)
    sizes = sorted({rnd(min(f * host, top)) for f in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)} | {small, large, full if cap >= full else rnd(top)})
    skip = trace_tokens(trace) // 2  # warm-up
    rows, pick = [], {}
    for mib in [s for s in sizes if s <= top]:
        ms = [model_ms(m, *replay(trace, mib, share, skip), share) for share in SHARES]
        k = min(i for i in range(len(ms)) if ms[i] <= 1.05 * min(ms))  # lowest share within 5% of the fastest
        rows.append([mib] + [best(f"{1000 / x:.1f}") if i == k else f"{1000 / x:.1f}" for i, x in enumerate(ms)])
        pick[mib] = (SHARES[k], 1000 / ms[k], 1000 / ms[0])

    say(f"{code(os.path.basename(model))}: {host:.0f} MiB of experts, {dev} {free} MiB free, cache up to {cap} MiB")
    say()
    say(table(["cache MiB", "share", "t/s", "hits"], [[r["mib"], f"{r['share']:g}", f"{1000 * r['tokens'] / r['ms']:.1f}", f"{r['hit']:.0f}%"] for r in runs]))
    say()
    say(f"machine: uploads {1000 / m['up']:.1f} GiB/s, CPU {1000 / m['cpu']:.1f} GiB/s, CPU part {m['split']:.2f} ms, overlap {m['overlap']:.1f} ms/GiB,"
        f" rest {m['base']:.2f} ms per token, fit within {100 * max(abs(e) for e in errs):.1f}%")
    say()
    say("predicted t/s, warm cache:")
    say(table(["cache MiB"] + [f"{s:g}" for s in SHARES], rows))
    say()
    if not ok or max(abs(e) for e in errs) > 0.10:
        sys.exit(f"the runs do not fit the time model, was the machine busy? delete {out} and run again")
    if cap >= full:
        say(f"all experts fit on {dev} ({full} MiB): drop {code('-cmoe')}, or tune a smaller cache with {code('--max-mib')}")
        return
    share, tps, tps0 = pick[max(pick)]
    say(f"recommended: {code(f'--moe-cache-mib {max(pick)} --moe-cache-cpu-share {share:g}')} ({tps:.1f} t/s" + (f", {tps0:.1f} with share 0)" if share > 0 else ")"))


def main():
    ap = argparse.ArgumentParser(description="picks --moe-cache-mib and --moe-cache-cpu-share for a llama-server command",
                                 usage="%(prog)s [--max-mib N] [--out DIR] [--markdown] -- llama-server -m model.gguf -cmoe [args]")
    ap.add_argument("--max-mib", type=int, help="largest cache in MiB (default: what fits)")
    ap.add_argument("--out", default="moe-cache-tune", help="run directory, reused for the same command (default: moe-cache-tune)")
    ap.add_argument("--markdown", action="store_true", help="Markdown tables")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="llama-server command")
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("pass the llama-server command after --")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FMT["markdown"] = args.markdown
    FMT["tty"] = sys.stdout.isatty() and not args.markdown
    tune(cmd, args.out, args.max_mib)


if __name__ == "__main__":
    main()
