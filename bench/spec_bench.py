#!/usr/bin/env python3
"""Speculative-decoding benchmark: server lifecycle, probing and recording in one process.

This replaces a bash script that embedded Python heredocs. The bash version failed repeatedly, and
always at the seam rather than in the measurement: `set -euo pipefail` turned a single unparseable
output line into a dead matrix, a stale server kept holding the port, a heredoc silently failed to
write its own script, and manual nohup/setsid detachment did not survive the caller. Process
lifecycle, HTTP and JSON all live here now, where they are ordinary library calls.

Usage:
    python3 bench/spec_bench.py --preset fp4
    python3 bench/spec_bench.py --preset kquant --depths 0,4096,16384
"""
from __future__ import annotations

import argparse, contextlib, datetime, json, pathlib, socket, subprocess, sys, time, urllib.error, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
HF = pathlib.Path.home() / ".cache/huggingface/hub"

SERVER = REPO / "build/bin/llama-server"   # overridable with --server, to compare against mainline

PRESETS = {
    "fp4": {
        "target": HF / "models--julianmb--Qwen-3.8-27B-ROCmFP4-FAST-GGUF/snapshots/1e67b67e0e23324a29a7a4449279685fac364b37/Qwen3.8-27B-ROCmFP4-FAST.gguf",
        "draft":  pathlib.Path("/models/dflash2-fpx/Qwen3.8-27B-DFlash2-Q4_0_ROCMFP4_FAST.gguf"),
        "target_quant": "Q4_0_ROCMFP4_FAST", "draft_quant": "Q4_0_ROCMFP4_FAST",
        "method": "draft-dflash",
    },
    "ornith-mtp": {
        "target": HF / "models--ornith-ai--Ornith-1.5-35B-A3B-GGUF/snapshots/12393612fd4f730ff5aadc23e9b8f9648aa49ceb/Ornith-1.5-35B-Q4_K_M.gguf",
        "draft":  None,                      # MTP drafts from the target's own nextn layers
        "target_quant": "Q4_K_M", "draft_quant": None,
        "method": "draft-mtp",
    },
    "kquant": {
        "target": HF / "models--unsloth--Qwen3.8-27B-GGUF/snapshots/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/Qwen3.8-27B-UD-Q4_K_XL.gguf",
        "draft":  HF / "models--z-lab--Qwen3.8-27B-DFlash2-GGUF/snapshots/57ab3265056d4024870b0621cfc2c127537020ed/Qwen3.8-27B-DFlash2-Q8_0.gguf",
        "target_quant": "Q4_K_XL", "draft_quant": "Q8_0",
        "method": "draft-dflash",
    },
}

ARMS = [
    ("bare-decode", False, []),
    ("fixed-n3",    True,  ["--spec-draft-n-max", "3"]),
    ("fixed-n7",    True,  ["--spec-draft-n-max", "7"]),
    ("adaptive",    True,  ["--spec-draft-n-max", "7", "--spec-draft-adaptive", "--spec-draft-n-min", "3"]),
    # MTP wants a tight cap: the EMA maximises accepted tokens, not throughput, and for MTP those
    # diverge because later nextn layers are less accurate while cost stays linear in n. Capping at
    # 4 is what the measured data points at, and is what this fork recommends.
    ("fixed-n4",    True,  ["--spec-draft-n-max", "4"]),
    ("adaptive-2-4", True, ["--spec-draft-n-max", "4", "--spec-draft-adaptive", "--spec-draft-n-min", "2"]),
]

TASKS = {
    "prose": ("Write a continuous, flowing essay of several paragraphs about how tidal patterns "
              "shaped the settlement of coastal towns. Do not use lists or headings."),
    "json":  ('Return ONLY a JSON array of 12 objects, no prose. Each object must have exactly the '
              'keys "id" (integer), "name" (string), "category" (string), "score" (number) and '
              '"active" (boolean). Begin your reply with [ and end it with ].'),
}


def sysfs(path, default="unknown"):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return default


def vram_mb() -> int:
    try:
        return int(sysfs("/sys/class/drm/card1/device/mem_info_vram_used", "0")) // 1048576
    except ValueError:
        return 0


def port_free(port: int) -> bool:
    with contextlib.closing(socket.socket()) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def gpu_containers(stop: bool):
    """Stop or restart containers holding /dev/dri or /dev/kfd. Returns the names it stopped."""
    try:
        names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                               capture_output=True, text=True, timeout=30).stdout.split()
    except Exception:
        return []
    hit = []
    for name in names:
        try:
            devs = subprocess.run(
                ["docker", "inspect", name, "--format",
                 "{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}"],
                capture_output=True, text=True, timeout=30).stdout
        except Exception:
            continue
        if "/dev/dri" in devs or "/dev/kfd" in devs:
            hit.append(name)
    if stop:
        for n in hit:
            subprocess.run(["docker", "stop", n], capture_output=True, timeout=120)
    return hit


@contextlib.contextmanager
def server(target, draft, method, port, extra, ctx, log_path, spec=True):
    """Start llama-server, wait for health, and guarantee it is gone afterwards."""
    for _ in range(60):
        if port_free(port):
            break
        time.sleep(1)
    if not port_free(port):
        raise RuntimeError(f"port {port} still busy")
    for _ in range(90):
        if vram_mb() <= 4096:
            break
        time.sleep(2)

    cmd = [str(SERVER), "-m", str(target)]
    if spec:
        cmd += ["--spec-type", method]
        if draft:                      # MTP has no sidecar; it drafts from the target's nextn layers
            cmd += ["-md", str(draft), "--spec-draft-ngl", "99"]
    cmd += ["--host", "127.0.0.1", "--port", str(port), "-fa", "on", "-ngl", "999",
            "-c", str(ctx)] + extra

    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    try:
        for _ in range(300):
            if proc.poll() is not None:
                raise RuntimeError(f"server exited early (rc={proc.returncode}), see {log_path}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).read()
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("server never became healthy")
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


def build_filler(depth: int) -> str:
    """Coherent in-distribution context. Word soup made the model stop after one token at 16k."""
    if not depth:
        return ""
    corpus = ""
    for rel in ("README.md", "CONTRIBUTING.md", "docs/build.md"):
        f = REPO / rel
        if f.is_file():
            corpus += f.read_text(errors="ignore") + "\n\n"
    if not corpus:
        corpus = "Long context behaviour depends on how much of the sequence must be attended. " * 200
    need = depth * 4
    while len(corpus) < need:
        corpus += corpus
    return corpus[:need] + "\n\n"


def probe(port, prompt, n_predict, timeout=1800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0,
                    "cache_prompt": True}).encode(),
        {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout)).get("timings", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="fp4")
    ap.add_argument("--depths", default="0")
    ap.add_argument("--n-predict", type=int, default=300)
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of: " + ",".join(a[0] for a in ARMS))
    ap.add_argument("--label", default=None, help="override the policy label written to rows")
    ap.add_argument("--out", default=None)
    ap.add_argument("--server", default=None,
                    help="llama-server to drive. Use a mainline build to compare against upstream.")
    ap.add_argument("--build", default="fork",
                    help="label recorded on every row, so fork and mainline rows stay separable")
    args = ap.parse_args()

    global SERVER
    if args.server:
        SERVER = pathlib.Path(args.server).resolve()

    cfg = PRESETS[args.preset]
    depths = [int(d) for d in args.depths.split(",")]
    ctx = args.ctx or max(8192, max(depths) + args.n_predict + 4096)
    out = pathlib.Path(args.out) if args.out else HERE / f"results-spec-{args.preset}.jsonl"

    for path in [SERVER, cfg["target"]] + ([cfg["draft"]] if cfg["draft"] else []):
        if not pathlib.Path(path).exists():
            sys.exit(f"missing: {path}")

    power = f'{sysfs("/sys/class/drm/card1/device/power_dpm_force_performance_level")}/' \
            f'{sysfs("/sys/class/drm/card1/device/power_dpm_state")}'
    # hwmon index is not stable across boots; glob for it rather than hardcoding hwmon0
    temp = "0"
    for t in sorted(pathlib.Path("/sys/class/drm/card1/device/hwmon").glob("hwmon*/temp1_input")):
        temp = sysfs(t, "0")
        break

    build_commit = subprocess.run(
        ["git", "-C", str(SERVER.parent.parent.parent), "rev-parse", "--short=9", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"

    stopped = gpu_containers(stop=True)
    if stopped:
        print(f"stopped GPU containers: {' '.join(stopped)}", flush=True)
    print(f"preset={args.preset} build={args.build} server={SERVER}", flush=True)
    print(f"power={power} ctx={ctx} depths={depths} n_predict={args.n_predict}", flush=True)

    try:
        with open(out, "a") as fh:
            wanted = set(args.arms.split(",")) if args.arms else None
            for label, use_draft, extra in ARMS:
                if wanted and label not in wanted:
                    continue
                if args.label:
                    label = args.label
                try:
                    with server(cfg["target"], cfg["draft"] if use_draft else None, cfg["method"],
                                args.port, extra, ctx, f"/tmp/spec-{args.preset}-{label}.log",
                                spec=use_draft):
                        for depth in depths:
                            filler = build_filler(depth)
                            line = []
                            for workload, task in TASKS.items():
                                try:
                                    t = probe(args.port, filler + task, args.n_predict)
                                except Exception as exc:
                                    line.append(f"{workload} FAILED {type(exc).__name__}")
                                    continue
                                drafted = t.get("draft_n")
                                acc = (100.0 * (t.get("draft_n_accepted") or 0) / drafted
                                       if drafted else None)
                                n = t.get("predicted_n") or 0
                                fh.write(json.dumps({
                                    "suite": "spec", "preset": args.preset, "policy": label,
                                    "build": args.build, "build_commit": build_commit,
                                    "workload": workload, "depth": depth, "power": power,
                                    "gpu_temp_c": int(temp) // 1000 if temp.isdigit() else None,
                                    "target": cfg["target"].name, "target_quant": cfg["target_quant"],
                                    "draft": (cfg["draft"].name if cfg["draft"] else "(target nextn)") if use_draft else None,
                                    "draft_quant": cfg["draft_quant"] if use_draft else None,
                                    "spec_method": cfg["method"] if use_draft else "none",
                                    "prompt_n": t.get("prompt_n"),
                                    "prompt_per_second": t.get("prompt_per_second"),
                                    "predicted_n": n,
                                    "predicted_per_second": t.get("predicted_per_second"),
                                    "accept_pct": acc, "degenerate": n < args.n_predict * 0.9,
                                    "recorded_at": datetime.datetime.now(
                                        datetime.timezone.utc).isoformat(),
                                }) + "\n")
                                fh.flush()
                                flag = "  DEGENERATE" if n < args.n_predict * 0.9 else ""
                                line.append(f"{workload} {t.get('predicted_per_second') or 0:6.2f} t/s"
                                            + (f" acc {acc:.0f}%" if acc is not None else "") + flag)
                            print(f"  {args.build:8} {label:12} d={depth:<6} " + "   ".join(line), flush=True)
                except Exception as exc:
                    print(f"  {label:12} ARM FAILED: {exc}", flush=True)
    finally:
        for n in stopped:
            subprocess.run(["docker", "start", n], capture_output=True, timeout=120)
        if stopped:
            print(f"restarted: {' '.join(stopped)}", flush=True)
    print(f"SPEC_BENCH_DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
