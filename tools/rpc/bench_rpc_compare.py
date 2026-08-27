#!/usr/bin/env python3
import argparse
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path


def run(cmd, *, env=None, timeout=None, stdout=None):
    return subprocess.run(cmd, env=env, timeout=timeout, check=True, text=True, stdout=stdout)


def wait_for_port(host, port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for {host}:{port}")


def load_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize(rows):
    result = {}
    for row in rows:
        kind = "pp" if row["n_prompt"] else "tg"
        samples = row["samples_ts"]
        result[kind] = {
            "avg_ts": row["avg_ts"],
            "median_ts": statistics.median(samples),
            "min_ts": min(samples),
            "max_ts": max(samples),
            "samples_ts": samples,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare two llama.cpp RPC builds with repeatable llama-bench runs.")
    parser.add_argument("--base-bin", required=True, type=Path, help="baseline build/bin directory")
    parser.add_argument("--patch-bin", required=True, type=Path, help="patched build/bin directory")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", default=50061, type=int)
    parser.add_argument("--patch-port", default=50062, type=int)
    parser.add_argument("--threads", default=8, type=int)
    parser.add_argument("--prompt", default=32, type=int)
    parser.add_argument("--gen", default=32, type=int)
    parser.add_argument("--repetitions", default=7, type=int)
    parser.add_argument("--ngl", default=99, type=int)
    parser.add_argument("--device", default=None, help="llama-bench device selector, for example RPC0")
    parser.add_argument("--server-device", default=None, help="rpc-server device selector, for example CUDA0")
    parser.add_argument("--out-dir", default=Path("rpc-bench-results"), type=Path)
    parser.add_argument("--keep-servers", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = [("base", args.base_bin, args.base_port), ("patch", args.patch_bin, args.patch_port)]
    summary = {
        "model": str(args.model),
        "host": args.host,
        "prompt": args.prompt,
        "gen": args.gen,
        "repetitions": args.repetitions,
        "ngl": args.ngl,
        "cases": {},
    }

    for name, bin_dir, port in cases:
        server_log = args.out_dir / f"{name}-rpc-server.log"
        bench_jsonl = args.out_dir / f"{name}-llama-bench.jsonl"
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        server = subprocess.Popen(
            [
                str(bin_dir / "rpc-server"),
                "-H", args.host,
                "-p", str(port),
                "-t", str(args.threads),
                *([] if args.server_device is None else ["--device", args.server_device]),
            ],
            stdout=server_log.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        try:
            wait_for_port(args.host, port, 15)
            with bench_jsonl.open("w", encoding="utf-8") as fh:
                bench_cmd = [
                    str(bin_dir / "llama-bench"),
                    "-m", str(args.model),
                    "--rpc", f"{args.host}:{port}",
                    "-ngl", str(args.ngl),
                    "-p", str(args.prompt),
                    "-n", str(args.gen),
                    "-r", str(args.repetitions),
                    "-o", "jsonl",
                ]
                if args.device is not None:
                    bench_cmd.extend(["--device", args.device])
                run(
                    bench_cmd,
                    env=env,
                    timeout=1800,
                    stdout=fh,
                )
            summary["cases"][name] = summarize(load_jsonl(bench_jsonl))
        finally:
            if not args.keep_servers:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()

    base_tg = summary["cases"].get("base", {}).get("tg", {}).get("avg_ts")
    patch_tg = summary["cases"].get("patch", {}).get("tg", {}).get("avg_ts")
    if base_tg and patch_tg:
        summary["decode_avg_ts_delta_pct"] = (patch_tg / base_tg - 1.0) * 100.0
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
