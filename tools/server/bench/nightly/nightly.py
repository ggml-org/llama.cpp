#!/usr/bin/env python3
"""Nightly llama.cpp perf benchmark for DGX Spark.

  nightly.py prefetch --home DIR [--tag bNNNN]   pick + download a pre-release build, models and data
  nightly.py run      --home DIR [--suites ...]  run the suites on the prefetched build, write night.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
SPEED_BENCH = REPO_ROOT / "tools" / "server" / "bench" / "speed-bench" / "speed_bench.py"
WIKITEXT_SCRIPT = REPO_ROOT / "scripts" / "get-wikitext-2.sh"
SUITES = ["engine", "speedbench", "serving", "agentic", "quality", "decision"]
BINARIES = ["llama-server", "llama-bench", "llama-perplexity"]


class BenchError(RuntimeError):
    pass


class DryRun(Exception):
    pass


def log(msg: str) -> None:
    print(f"[nightly {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for name, model in cfg["models"].items():
        for item in [model, model.get("tokenizer") or {}, *(model.get("extras") or {}).values()]:
            if item and not re.fullmatch(r"[0-9a-f]{40}", str(item.get("revision", ""))):
                raise ValueError(f"model {name}: every repo needs a 40-hex 'revision'")
        unknown = set(model["suites"]) - set(SUITES)
        if unknown:
            raise ValueError(f"model {name}: unknown suites {sorted(unknown)}")
    return cfg


def repo_dir(home: Path, kind: str, repo: str, revision: str) -> Path:
    return home / kind / f"{repo.replace('/', '--')}@{revision[:12]}"


def model_path(home: Path, item: dict[str, Any]) -> Path:
    return repo_dir(home, "models", item["repo"], item["revision"]) / item["file"]


def wikitext_path(home: Path) -> Path:
    return home / "data" / "wikitext-2-raw" / "wiki.test.raw"


# --- prefetch ---

def github_get(url: str) -> Any:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def pick_release(releases: list[dict[str, Any]], assets: list[str]) -> dict[str, Any]:
    # newest first; the release job uploads assets over time, so skip tags that are still incomplete
    for rel in releases:
        if rel.get("draft"):
            continue
        names = {a["name"] for a in rel.get("assets", [])}
        if all(a.format(tag=rel["tag_name"]) in names for a in assets):
            return rel
    raise BenchError(f"no release has all of {assets}")


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def fetch_build(cfg: dict[str, Any], home: Path, tag: str | None) -> dict[str, Any]:
    bcfg = cfg["build"]
    api = f"https://api.github.com/repos/{bcfg['repo']}/releases"
    if tag:
        rel = pick_release([github_get(f"{api}/tags/{tag}")], bcfg["assets"])
    else:
        rel = pick_release(github_get(f"{api}?per_page=20"), bcfg["assets"])
    tag = str(rel["tag_name"])
    bin_dir = home / "builds" / tag
    info = {
        "tag": tag,
        "commit": rel.get("target_commitish", ""),
        "published_at": rel.get("published_at", ""),
        "url": rel.get("html_url", ""),
        "bin_dir": str(bin_dir),
    }
    if not all((bin_dir / b).exists() for b in BINARIES):
        log(f"downloading build {tag}")
        shutil.rmtree(bin_dir, ignore_errors=True)
        bin_dir.mkdir(parents=True)
        urls = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
        for pattern in bcfg["assets"]:
            name = pattern.format(tag=tag)
            archive = home / "builds" / name
            download(urls[name], archive)
            # both tarballs have one top-level dir; flatten them together so $ORIGIN rpath finds cudart
            subprocess.run(["tar", "-xzf", str(archive), "--strip-components=1", "-C", str(bin_dir)], check=True)
            archive.unlink()
        missing = [b for b in BINARIES if not (bin_dir / b).exists()]
        if missing:
            raise BenchError(f"build {tag} has no {missing}")
    write_json(bin_dir / "build.json", info)

    builds = sorted((p for p in (home / "builds").iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in builds[bcfg.get("keep", 3):]:
        if old != bin_dir:
            shutil.rmtree(old, ignore_errors=True)
    return info


def prefetch(cfg: dict[str, Any], home: Path, tag: str | None, models: list[str]) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    info = fetch_build(cfg, home, tag)
    write_json(home / "selected_build.json", info)
    log(f"build {info['tag']} ({info['commit']})")

    for name in models:
        model = cfg["models"][name]
        for item in [model, *(model.get("extras") or {}).values()]:
            dest = model_path(home, item)
            if not dest.exists():
                log(f"downloading {item['repo']}/{item['file']}")
                hf_hub_download(repo_id=item["repo"], filename=item["file"], revision=item["revision"], local_dir=dest.parent)
        if model.get("tokenizer"):
            tok = model["tokenizer"]
            # tokenizer files only, one by one: snapshot_download caches the repo file list, and then AIPerf's
            # offline decode workers reject the snapshot because the weights are missing
            siblings = HfApi().model_info(tok["repo"], revision=tok["revision"]).siblings or []
            for f in (s.rfilename for s in siblings):
                if "/" not in f and f.endswith((".json", ".jinja", ".txt", ".model")):
                    hf_hub_download(repo_id=tok["repo"], filename=f, revision=tok["revision"])

    suites = {s for name in models for s in cfg["models"][name]["suites"]}
    if "speedbench" in suites:
        from datasets import load_dataset
        # same call as speed_bench.py, so the offline run hits the cache
        load_dataset("nvidia/SPEED-Bench", name=cfg["suites"]["speedbench"]["bench"], split="test")
    if "quality" in suites and not wikitext_path(home).exists():
        (home / "data").mkdir(parents=True, exist_ok=True)
        subprocess.run(["sh", str(WIKITEXT_SCRIPT)], cwd=home / "data", check=True)


# --- process + server helpers ---

class Context:
    def __init__(self, cfg: dict[str, Any], home: Path, build: dict[str, Any], out_dir: Path, dry_run: bool):
        self.cfg = cfg
        self.home = home
        self.build = build
        self.bin_dir = Path(build["bin_dir"])
        self.out_dir = out_dir
        self.dry_run = dry_run
        self.port = int(cfg["server"]["port"])
        self.url = f"http://127.0.0.1:{self.port}"
        self.commands: list[list[str]] = []
        self.first_server = True

    def run(self, cmd: list[str], log_path: Path, timeout_s: float, stdout_path: Path | None = None) -> None:
        self.commands.append(cmd)
        if self.dry_run:
            raise DryRun()
        log(" ".join(cmd))
        with open(log_path, "ab") as logf:
            out = open(stdout_path, "wb") if stdout_path else logf
            try:
                proc = subprocess.Popen(cmd, stdout=out, stderr=logf, start_new_session=True)
                try:
                    rc = proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    kill_group(proc)
                    raise BenchError(f"timeout after {timeout_s:.0f} s")
                except BaseException:
                    kill_group(proc)
                    raise
            finally:
                if stdout_path:
                    out.close()
        if rc != 0:
            raise BenchError(f"exit code {rc}: {tail(log_path)}")


def kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def tail(path: Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


class Server:
    def __init__(self, ctx: Context, args: list[str], log_path: Path):
        self.ctx = ctx
        self.cmd = [str(ctx.bin_dir / "llama-server"), "--host", "127.0.0.1", "--port", str(ctx.port), *args]
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> Server:
        self.ctx.commands.append(self.cmd)
        if self.ctx.dry_run:
            return self
        if not self.ctx.first_server:
            time.sleep(float(self.ctx.cfg["server"]["cooldown_s"]))
        self.ctx.first_server = False
        deadline = time.monotonic() + 60
        while not port_free(self.ctx.port):
            if time.monotonic() > deadline:
                raise BenchError(f"port {self.ctx.port} still in use")
            time.sleep(1)
        log(" ".join(self.cmd))
        self.logf = open(self.log_path, "ab")
        try:
            self.proc = subprocess.Popen(self.cmd, stdout=self.logf, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            self.logf.close()
            raise BenchError(f"cannot start llama-server: {exc}") from exc
        deadline = time.monotonic() + float(self.ctx.cfg["server"]["health_timeout_s"])
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.__exit__()
                raise BenchError(f"llama-server exited with {self.proc.returncode}: {tail(self.log_path)}")
            try:
                if requests.get(f"{self.ctx.url}/health", timeout=5).status_code == 200:
                    return self
            except requests.RequestException:
                pass
            time.sleep(2)
        self.__exit__()
        raise BenchError(f"llama-server not healthy in time: {tail(self.log_path)}")

    def __exit__(self, *exc: Any) -> None:
        if self.proc is not None:
            kill_group(self.proc)
            self.proc = None
            self.logf.close()

    def check_alive(self) -> None:
        if self.proc is not None and self.proc.poll() is not None:
            raise BenchError(f"llama-server died: {tail(self.log_path)}")

    def post(self, path: str, body: dict[str, Any], timeout: float = 120) -> dict[str, Any]:
        if self.ctx.dry_run:
            raise DryRun()
        resp = requests.post(f"{self.ctx.url}{path}", json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def metrics(self) -> dict[str, float]:
        if self.ctx.dry_run:
            return {}
        out: dict[str, float] = {}
        for line in requests.get(f"{self.ctx.url}/metrics", timeout=10).text.splitlines():
            parts = line.split()
            if len(parts) == 2 and not line.startswith("#"):
                out[parts[0]] = float(parts[1])
        return out


class SysMonitor:
    # GB10 has unified memory, so nvidia-smi memory.used is N/A; use /proc/meminfo instead
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> SysMonitor:
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop_event.set()
        self.thread.join()

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            self.samples.append(self._sample())

    @staticmethod
    def _sample() -> dict[str, float]:
        s: dict[str, float] = {}
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw,clocks.sm,temperature.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.splitlines()
            for key, raw in zip(("power_w", "sm_clock_mhz", "gpu_temp_c"), (out[0] if out else "").split(",")):
                try:
                    s[key] = float(raw)
                except ValueError:
                    pass
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            s["mem_used_gib"] = (int(mem["MemTotal"].split()[0]) - int(mem["MemAvailable"].split()[0])) / 2**20
        except (OSError, KeyError, ValueError):
            pass
        return s

    def summary(self) -> dict[str, float | None]:
        def agg(key: str, fn: Callable[[list[float]], float]) -> float | None:
            vals = [s[key] for s in self.samples if key in s]
            return round(fn(vals), 2) if vals else None
        return {
            "power_w_avg": agg("power_w", lambda v: sum(v) / len(v)),
            "power_w_max": agg("power_w", max),
            "sm_clock_mhz_avg": agg("sm_clock_mhz", lambda v: sum(v) / len(v)),
            "gpu_temp_c_max": agg("gpu_temp_c", max),
            "mem_used_gib_peak": agg("mem_used_gib", max),
        }


def run_point(ctx: Context, records: list[dict[str, Any]], suite: str, model: str, config: str,
              fn: Callable[[Path], dict[str, float]]) -> None:
    run_dir = ctx.out_dir / suite / model / config
    run_dir.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"suite": suite, "model": model, "config": config, "status": "ok", "error": "", "metrics": {}}
    ctx.commands = []
    start = time.monotonic()
    with SysMonitor() as mon:
        try:
            rec["metrics"] = fn(run_dir)
        except DryRun:
            rec["status"] = "skipped"
            rec["error"] = "dry-run"
        except Exception as exc:
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["duration_s"] = round(time.monotonic() - start, 1)
    rec["telemetry"] = mon.summary()
    rec["commands"] = [" ".join(c) for c in ctx.commands]
    rec["artifact_dir"] = str(run_dir.relative_to(ctx.out_dir))
    log(f"{suite}/{model}/{config}: {rec['status']} {rec['error'][:300]}")
    records.append(rec)


def run_group(ctx: Context, records: list[dict[str, Any]], suite: str, model: str, server_args: list[str],
              log_path: Path, points: list[tuple[str, Callable[[Server, Path], dict[str, float]]]]) -> None:
    # one server for several points; if it fails to start or dies, the remaining points fail
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Server(ctx, server_args, log_path) as server:
            for config, fn in points:
                def point(run_dir: Path, fn=fn) -> dict[str, float]:
                    ctx.commands.append(server.cmd)
                    server.check_alive()
                    return fn(server, run_dir)
                run_point(ctx, records, suite, model, config, point)
    except BenchError as exc:
        def fail(_run_dir: Path, exc=exc) -> dict[str, float]:
            raise exc

        done = {(r["suite"], r["model"], r["config"]) for r in records}
        for config, _ in points:
            if (suite, model, config) not in done:
                run_point(ctx, records, suite, model, config, fail)


# --- suites ---

def resolve_spec(ctx: Context, model: dict[str, Any], spec: str) -> list[str]:
    if spec == "none":
        return []
    extras = {k: str(model_path(ctx.home, v)) for k, v in (model.get("extras") or {}).items()}
    return [a.format(**extras) for a in model["spec"][spec]]


def suite_engine(ctx: Context, records: list[dict[str, Any]], name: str, model: dict[str, Any]) -> None:
    scfg = ctx.cfg["suites"]["engine"]

    def fn(run_dir: Path) -> dict[str, float]:
        out = run_dir / "llama-bench.json"
        cmd = [str(ctx.bin_dir / "llama-bench"), "-m", str(model_path(ctx.home, model)), "-o", "json", *scfg["args"]]
        ctx.run(cmd, run_dir / "llama-bench.log", scfg["timeout_s"], stdout_path=out)
        metrics: dict[str, float] = {}
        for row in json.loads(out.read_text()):
            kind = f"tg{row['n_gen']}" if row["n_gen"] else f"pp{row['n_prompt']}"
            metrics[f"{kind}_d{row['n_depth']}_tps"] = row["avg_ts"]
            metrics[f"{kind}_d{row['n_depth']}_tps_stddev"] = row["stddev_ts"]
        return metrics

    run_point(ctx, records, "engine", name, "llama-bench", fn)


def suite_speedbench(ctx: Context, records: list[dict[str, Any]], name: str, model: dict[str, Any]) -> None:
    scfg = ctx.cfg["suites"]["speedbench"]
    specs = [s for s in scfg["spec"] if s == "none" or s in (model.get("spec") or {})]
    for spec in specs:
        for c in scfg["concurrency"]:
            def fn(server: Server, run_dir: Path, spec=spec, c=c) -> dict[str, float]:
                out = run_dir / "speedbench.json"
                cmd = [sys.executable, str(SPEED_BENCH), "--url", ctx.url, "--bench", scfg["bench"], "--limit", str(scfg["limit"]),
                       "--osl", str(scfg["osl"]), "--concurrency", str(c), "--extra-inputs", json.dumps(scfg["extra_inputs"]),
                       "--output", str(out)]
                ctx.run(cmd, run_dir / "speedbench.log", scfg["timeout_s"])
                data = json.loads(out.read_text())
                metrics: dict[str, float] = {"failed_samples": data["failed_samples"]}
                for row in data["summary"]:
                    for key, metric in (("avg_pred_t_s", "pred_tps"), ("avg_prompt_t_s", "prompt_tps"),
                                        ("avg_latency", "latency_s"), ("accept_rate", "accept_rate")):
                        if row.get(key) is not None:
                            metrics[f"{row['category']}.{metric}"] = row[key]
                overall = next(r for r in data["summary"] if r["category"] == "overall")
                if spec != "none" and not overall["draft_n"]:
                    raise BenchError("spec_inactive: speculative decoding on, but no tokens were drafted")
                return metrics

            server_args = [*ctx.cfg["server"]["args"], "-m", str(model_path(ctx.home, model)),
                           "-np", str(c), "-c", str(scfg["ctx_per_slot"] * c), *resolve_spec(ctx, model, spec)]
            config = f"spec-{spec}_c{c}"
            run_group(ctx, records, "speedbench", name, server_args, ctx.out_dir / "speedbench" / name / f"server_{config}.log", [(config, fn)])


AIPERF_METRICS = [
    ("ttft_p50_ms", "time_to_first_token", "p50"),
    ("ttft_p99_ms", "time_to_first_token", "p99"),
    ("itl_p50_ms", "inter_token_latency", "p50"),
    ("itl_p99_ms", "inter_token_latency", "p99"),
    ("out_tps_per_user", "output_token_throughput_per_user", "avg"),
    ("out_tps_total", "output_token_throughput", "avg"),
    ("req_per_s", "request_throughput", "avg"),
]


def aiperf_bin() -> str:
    local = Path(sys.executable).parent / "aiperf"
    return str(local) if local.exists() else "aiperf"


def aiperf_base(ctx: Context, model: dict[str, Any], run_dir: Path, timeout_s: int) -> list[str]:
    return [aiperf_bin(), "profile", "--model", "bench", "--url", ctx.url, "--endpoint-type", "chat", "--streaming",
            "--tokenizer", model["tokenizer"]["repo"], "--tokenizer-revision", model["tokenizer"]["revision"],
            "--use-server-token-count", "--random-seed", "42", "--request-timeout-seconds", str(timeout_s),
            "--artifact-dir", str(run_dir / "aiperf"), "--ui-type", "none"]


def aiperf_metrics(run_dir: Path) -> dict[str, float]:
    files = sorted(run_dir.rglob("profile_export_aiperf.json"))
    if not files:
        raise BenchError("aiperf wrote no profile_export_aiperf.json")
    profile = json.loads(files[0].read_text())
    metrics: dict[str, float] = {}
    for name, tag, stat in AIPERF_METRICS:
        value = (profile.get(tag) or {}).get(stat)
        if value is None:
            raise BenchError(f"parse_error: aiperf output has no {tag}.{stat}")
        metrics[name] = float(value)
    for name, tag in (("requests", "request_count"), ("errors", "error_request_count")):
        value = (profile.get(tag) or {}).get("avg")
        if value is not None:
            metrics[name] = float(value)
    return metrics


def slot_ctx(tokens: int) -> int:
    return math.ceil((tokens + 1024) / 256) * 256


def suite_serving(ctx: Context, records: list[dict[str, Any]], name: str, model: dict[str, Any]) -> None:
    scfg = ctx.cfg["suites"]["serving"]
    osl = scfg["osl"]
    for c in scfg["concurrency"]:
        points = []
        for shape in scfg["shapes"]:
            isl, depth = shape["isl"], shape["depth"]

            def fn(server: Server, run_dir: Path, isl=isl, depth=depth, c=c) -> dict[str, float]:
                cmd = aiperf_base(ctx, model, run_dir, scfg["timeout_s"])
                cmd += ["--isl", str(isl), "--isl-stddev", "0", "--osl", str(osl), "--osl-stddev", "0",
                        "--extra-inputs", json.dumps({"ignore_eos": True, "min_tokens": osl}),
                        "--concurrency", str(c), "--request-count", str(scfg["requests"][c]),
                        "--warmup-request-count", str(scfg["warmup"])]
                if depth:
                    cmd += ["--shared-system-prompt-length", str(depth)]
                ctx.run(cmd, run_dir / "aiperf.log", scfg["timeout_s"])
                return aiperf_metrics(run_dir)

            points.append((f"depth{depth}_isl{isl}_osl{osl}_c{c}", fn))
        n_ctx = max(slot_ctx(s["isl"] + s["depth"] + osl) for s in scfg["shapes"]) * c
        server_args = [*ctx.cfg["server"]["args"], "-m", str(model_path(ctx.home, model)), "-np", str(c), "-c", str(n_ctx)]
        run_group(ctx, records, "serving", name, server_args, ctx.out_dir / "serving" / name / f"server_c{c}.log", points)


def suite_agentic(ctx: Context, records: list[dict[str, Any]], name: str, model: dict[str, Any]) -> None:
    scfg = ctx.cfg["suites"]["agentic"]
    cap = scfg["max_output_tokens"]
    for trace in scfg["traces"]:
        rows = [json.loads(line) for line in (HERE / trace).read_text().splitlines() if line.strip()]
        for row in rows:
            row["output_length"] = min(row["output_length"], cap)

        def fn(server: Server, run_dir: Path, rows=rows) -> dict[str, float]:
            capped = run_dir / "trace.jsonl"
            capped.write_text("".join(json.dumps(r) + "\n" for r in rows))
            cmd = aiperf_base(ctx, model, run_dir, scfg["timeout_s"])
            cmd += ["--input-file", str(capped), "--custom-dataset-type", "mooncake_trace", "--fixed-schedule",
                    "--fixed-schedule-auto-offset", "--isl-block-size", str(scfg["block_size"]),
                    "--extra-inputs", json.dumps({"ignore_eos": True})]
            before = server.metrics()
            ctx.run(cmd, run_dir / "aiperf.log", scfg["timeout_s"])
            after = server.metrics()
            metrics = aiperf_metrics(run_dir)
            processed = after.get("llamacpp:prompt_tokens_total", 0) - before.get("llamacpp:prompt_tokens_total", 0)
            cached = after.get("llamacpp:prompt_tokens_cached_total", 0) - before.get("llamacpp:prompt_tokens_cached_total", 0)
            if processed + cached > 0:
                metrics["prefix_reuse"] = cached / (processed + cached)
            return metrics

        n_ctx = slot_ctx(max(r["input_length"] + r["output_length"] for r in rows)) * scfg["slots"]
        server_args = [*ctx.cfg["server"]["args"], "-m", str(model_path(ctx.home, model)), "-np", str(scfg["slots"]), "-c", str(n_ctx)]
        config = f"mooncake_{Path(trace).stem.split('_')[0]}_osl{cap}"
        run_group(ctx, records, "agentic", name, server_args, ctx.out_dir / "agentic" / name / f"server_{config}.log", [(config, fn)])


def suite_quality(ctx: Context, records: list[dict[str, Any]], name: str, model: dict[str, Any]) -> None:
    scfg = ctx.cfg["suites"]["quality"]

    def fn(run_dir: Path) -> dict[str, float]:
        log_path = run_dir / "perplexity.log"
        cmd = [str(ctx.bin_dir / "llama-perplexity"), "-m", str(model_path(ctx.home, model)), "-f", str(wikitext_path(ctx.home)),
               "-c", str(scfg["ctx"]), "--chunks", str(scfg["chunks"]), "-ngl", "999", "-fa", "on"]
        ctx.run(cmd, log_path, scfg["timeout_s"])
        m = re.search(r"Final estimate: PPL = ([0-9.]+)", log_path.read_text(errors="replace"))
        if not m:
            raise BenchError("parse_error: no 'Final estimate' line")
        return {"ppl": float(m.group(1))}

    run_point(ctx, records, "quality", name, f"ppl_c{scfg['ctx']}_n{scfg['chunks']}", fn)


DECISION_QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": None, "shipping": None, "technical": None, "account": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["can wait", "this week", "today", "right now", "already late"],
    },
    "needs_human": {"type": "noul", "instructions": "Does this need a human agent?"},
}

DECISION_FILLER = (
    "Customer message: I ordered a replacement part two weeks ago and the tracking page has not changed since. "
    "I was also charged twice on my card and the support chat closed before anyone answered. "
)


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def make_state(server: Server, n_tokens: int) -> str:
    text = DECISION_FILLER * (n_tokens // 8 + 1)
    tokens = server.post("/tokenize", {"content": text})["tokens"][:n_tokens]
    return server.post("/detokenize", {"tokens": tokens})["content"]


def systemone(url: str, state: str, timeout: float) -> tuple[float, dict[str, Any]]:
    start = time.perf_counter()
    resp = requests.post(f"{url}/v1/systemone", json={"state": state, "questions": DECISION_QUESTIONS}, timeout=timeout)
    latency = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()
    answers = data.get("answers") or {}
    if set(answers) != set(DECISION_QUESTIONS):
        raise BenchError(f"parse_error: answers {sorted(answers)}")
    if abs(sum(answers["route"]["probabilities"].values()) - 1) > 1e-3:
        raise BenchError("parse_error: choice probabilities do not sum to 1")
    return latency, data


def suite_decision(ctx: Context, records: list[dict[str, Any]], name: str, model: dict[str, Any]) -> None:
    scfg = ctx.cfg["suites"]["decision"]
    points = []
    for n_tokens in scfg["state_tokens"]:
        for c in scfg["concurrency"]:
            def fn(server: Server, run_dir: Path, n_tokens=n_tokens, c=c) -> dict[str, float]:
                state = make_state(server, n_tokens)
                for _ in range(scfg["warmup"]):
                    systemone(ctx.url, state, scfg["timeout_s"])
                start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                    results = list(pool.map(lambda _: systemone(ctx.url, state, scfg["timeout_s"]), range(scfg["requests"])))
                wall = time.perf_counter() - start
                latencies = [r[0] * 1000 for r in results]
                (run_dir / "last_response.json").write_text(json.dumps(results[-1][1], indent=2))
                return {
                    "latency_p50_ms": percentile(latencies, 50),
                    "latency_p99_ms": percentile(latencies, 99),
                    "decisions_per_s": len(results) * len(DECISION_QUESTIONS) / wall,
                    "input_tokens": float(results[-1][1].get("usage", {}).get("input_tokens", 0)),
                }

            points.append((f"sys_s{n_tokens}_c{c}", fn))
    server_args = ["-m", str(model_path(ctx.home, model)), *scfg["args"]]
    run_group(ctx, records, "decision", name, server_args, ctx.out_dir / "decision" / name / "server.log", points)


SUITE_FNS = {
    "engine": suite_engine,
    "speedbench": suite_speedbench,
    "serving": suite_serving,
    "agentic": suite_agentic,
    "quality": suite_quality,
    "decision": suite_decision,
}

HEADLINE = {
    "engine": ["pp2048_d0_tps", "tg128_d0_tps", "pp2048_d32768_tps", "tg128_d32768_tps"],
    "speedbench": ["overall.pred_tps", "overall.latency_s", "overall.accept_rate"],
    "serving": ["ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms", "out_tps_total"],
    "agentic": ["ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms", "out_tps_total", "prefix_reuse"],
    "quality": ["ppl"],
    "decision": ["latency_p50_ms", "latency_p99_ms", "decisions_per_s"],
}


# --- report ---

def previous_night(home: Path, night_id: str) -> dict[str, Any] | None:
    nights = sorted(p for p in (home / "results").glob("*/night.json")
                    if p.parent.name < night_id and not p.parent.name.endswith("_dry-run"))
    for path in reversed(nights):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return None


def render_summary(manifest: dict[str, Any], prev: dict[str, Any] | None) -> str:
    repo = "https://github.com/ggml-org/llama.cpp"
    b = manifest["build"]
    lines = [f"## llama.cpp nightly bench on {manifest['runner']}: {b['tag']}", ""]
    lines.append(f"- build: [{b['tag']}]({b.get('url', '')}) at `{b['commit'][:10]}`")
    if prev and prev["build"]["commit"] and prev["build"]["commit"] != b["commit"]:
        lines.append(f"- changes since last night: [{prev['build']['tag']}...{b['tag']}]({repo}/compare/{prev['build']['commit']}...{b['commit']})")
    elif prev:
        lines.append("- same build as last night")
    statuses = [r["status"] for r in manifest["records"]]
    lines.append(f"- runs: {statuses.count('ok')} ok, {statuses.count('failed')} failed, {statuses.count('skipped')} skipped")
    lines.append("")
    for suite in SUITES:
        recs = [r for r in manifest["records"] if r["suite"] == suite]
        if not recs:
            continue
        cols = HEADLINE[suite]
        lines += [f"### {suite}", "", "| model | config | status | " + " | ".join(cols) + " | time |",
                  "|---|---|---|" + "---|" * len(cols) + "---|"]
        for r in recs:
            vals = [f"{r['metrics'][c]:.4g}" if c in r["metrics"] else "-" for c in cols]
            status = r["status"] if r["status"] != "failed" else f"failed: {r['error'][:80]}".replace("|", "/").replace("\n", " ")
            lines.append(f"| {r['model']} | {r['config']} | {status} | " + " | ".join(vals) + f" | {r['duration_s']:.0f} s |")
        lines.append("")
    return "\n".join(lines)


def run(cfg: dict[str, Any], home: Path, suites: list[str], models: list[str], dry_run: bool, summary_path: Path | None) -> int:
    selected = home / "selected_build.json"
    if selected.exists():
        build = json.loads(selected.read_text())
    elif dry_run:
        build = {"tag": "dry-run", "commit": "", "bin_dir": str(home / "builds" / "dry-run")}
    else:
        raise BenchError(f"{selected} not found, run prefetch first")

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    night_id = f"{date}_{build['tag']}" + ("_dry-run" if dry_run else "")
    out_dir = home / "results" / night_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"night_dir={out_dir}\nnight_id={night_id}\n")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "night_id": night_id,
        "date": date,
        "build": build,
        "runner": os.environ.get("RUNNER_NAME", socket.gethostname()),
        "workflow_url": os.environ.get("NIGHTLY_WORKFLOW_URL", ""),
        "models": {n: {k: cfg["models"][n][k] for k in ("repo", "revision", "file")} for n in models},
        "records": [],
    }
    # a re-run on the same night keeps the records of the suites and models it does not run
    if (out_dir / "night.json").exists():
        old = json.loads((out_dir / "night.json").read_text())
        manifest["records"] = [r for r in old.get("records", []) if r["suite"] not in suites or r["model"] not in models]
        manifest["models"] = {**old.get("models", {}), **manifest["models"]}
    ctx = Context(cfg, home, build, out_dir, dry_run)
    started = time.monotonic()
    for suite in suites:
        for name in models:
            model = cfg["models"][name]
            if suite in model["suites"]:
                log(f"suite {suite} on {name}")
                SUITE_FNS[suite](ctx, manifest["records"], name, model)
                write_json(out_dir / "night.json", manifest)
    manifest["duration_s"] = round(time.monotonic() - started, 1)
    write_json(out_dir / "night.json", manifest)

    summary = render_summary(manifest, None if dry_run else previous_night(home, night_id))
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(summary + "\n")
    print(summary)
    return 1 if any(r["status"] == "failed" for r in manifest["records"]) else 0


def split_list(value: str | None, allowed: list[str]) -> list[str]:
    if not value:
        return list(allowed)
    items = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [v for v in items if v not in allowed]
    if unknown:
        raise SystemExit(f"unknown {unknown}, choose from {allowed}")
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nightly llama.cpp perf benchmark for DGX Spark")
    parser.add_argument("command", choices=["prefetch", "run"])
    parser.add_argument("--config", type=Path, default=HERE / "suite.yaml")
    parser.add_argument("--home", type=Path, default=Path(os.environ.get("NIGHTLY_HOME", "/opt/llama-nightly")))
    parser.add_argument("--tag", default=None, help="release tag to test (default: newest with all assets)")
    parser.add_argument("--suites", default=None, help=f"comma-separated subset of {SUITES}")
    parser.add_argument("--models", default=None, help="comma-separated subset of the models in the config")
    parser.add_argument("--dry-run", action="store_true", help="print the commands, start nothing")
    parser.add_argument("--summary", type=Path, default=None, help="append the Markdown summary to this file")
    args = parser.parse_args(argv)

    # turn SIGTERM (job cancel) into SystemExit so the servers get stopped
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    cfg = load_config(args.config)
    models = split_list(args.models, list(cfg["models"]))
    suites = split_list(args.suites, SUITES)
    args.home.mkdir(parents=True, exist_ok=True)
    if args.command == "prefetch":
        prefetch(cfg, args.home, args.tag, models)
        return 0
    return run(cfg, args.home, suites, models, args.dry_run, args.summary)


if __name__ == "__main__":
    sys.exit(main())
