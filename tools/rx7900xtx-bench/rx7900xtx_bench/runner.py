from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

from .server_client import ServerClient
from .metrics import counter_deltas, summarize_samples
from .process import ServerProcess
from .workloads import FIXTURE_TEXT_SHA256, build_prompt_tokens


def build_server_command(config: dict[str, Any], host: str, port: int) -> list[str]:
    command = [
        str(config["server"]),
        "--model", str(config["model"]),
        "--host", host,
        "--port", str(port),
        "--metrics",
        "--no-warmup",
        "--no-cache-prompt",
    ]
    mmproj = config.get("mmproj")
    if mmproj:
        command.extend(["--mmproj", str(mmproj)])
    command.extend(str(value) for value in config["server_args"])
    return command


def run_benchmark(config: dict[str, Any], options: dict[str, Any]) -> Path:
    depth = options["depth"]
    if depth not in config["depths"]:
        raise ValueError(f"depth {depth} is not in the preset workload matrix")
    results_dir = make_results_dir(Path(options["results_dir"]), options["label"])
    logs_dir = results_dir / "logs"
    responses_dir = results_dir / "responses"
    responses_dir.mkdir(parents=True)

    command = build_server_command(config, options["host"], options["port"])
    require_port_available(options["host"], options["port"])
    manifest = make_manifest(config, options, command)
    manifest["device_inventory"] = verify_requested_device(config)
    write_json(results_dir / "manifest.json", manifest)

    client = ServerClient(options["host"], options["port"], options["timeout_seconds"])
    server = ServerProcess(command, logs_dir / "server.log")
    samples: list[dict[str, Any]] = []
    try:
        server.start()
        server.wait_ready(client, options["startup_timeout_seconds"])
        server.wait_mtp(options["startup_timeout_seconds"])
        context_effective = verify_server_configuration(client, config)
        manifest["runtime"]["context_requested"] = config["context_size"]
        manifest["runtime"]["context_effective"] = context_effective
        prompt_tokens, prompt_hash = build_prompt_tokens(client, depth)
        manifest["prompt"] = {
            "mode": "fresh",
            "fixture_id": "project-technical-discussion-v1",
            "fixture_source": "project-authored",
            "fixture_text_sha256": FIXTURE_TEXT_SHA256,
            "requested_depth": depth,
            "observed_depth": len(prompt_tokens),
            "token_sha256": prompt_hash,
        }
        write_json(results_dir / "manifest.json", manifest)

        warmup = measure_request(client, prompt_tokens, options["output_tokens"], "warmup-0")
        write_json(responses_dir / "warmup-0.json", warmup.pop("raw_response"))
        warmup["warmup"] = True
        write_json(results_dir / "warmups.json", [warmup])

        for index in range(options["runs"]):
            sample_id = f"sample-{index:03d}"
            sample = measure_request(client, prompt_tokens, options["output_tokens"], sample_id)
            write_json(responses_dir / f"{sample_id}.json", sample.pop("raw_response"))
            samples.append(sample)
            append_jsonl(results_dir / "samples.jsonl", sample)
    finally:
        server.stop()

    summary = summarize_samples(samples)
    summary.update({
        "preset": options["preset"],
        "depth": depth,
        "runs": options["runs"],
        "output_tokens_requested": options["output_tokens"],
    })
    write_json(results_dir / "summary.json", summary)
    write_csv(results_dir / "summary.csv", samples)
    write_report(results_dir / "report.md", summary, manifest)
    return results_dir


def measure_request(client: ServerClient, prompt_tokens: list[int], output_tokens: int, sample_id: str) -> dict[str, Any]:
    metrics_before = client.metrics()
    request_started = time.monotonic()
    first_content_time: float | None = None
    final_response: dict[str, Any] | None = None
    for event_time, event in client.completion_stream({
        "prompt": prompt_tokens,
        "n_predict": output_tokens,
        "stream": True,
        "timings_per_token": True,
    }):
        if first_content_time is None and (event.get("content") or event.get("tokens")):
            first_content_time = event_time
        if event.get("stop") is True:
            final_response = event
    request_finished = time.monotonic()
    metrics_after = client.metrics()
    if final_response is None:
        raise RuntimeError("stream completed without a final completion response")

    final_output_tokens = final_response.get("tokens_predicted")
    if not isinstance(final_output_tokens, int):
        raise RuntimeError("completion response has no authoritative final token count")
    request_seconds = request_finished - request_started
    delivery_seconds = request_finished - first_content_time if first_content_time is not None else None
    timings = final_response.get("timings") if isinstance(final_response.get("timings"), dict) else {}
    deltas = counter_deltas(metrics_before, metrics_after)
    draft_proposed = deltas["llamacpp:spec_decode_num_draft_tokens_total"]
    draft_accepted = deltas["llamacpp:spec_decode_num_accepted_tokens_total"]
    return {
        "sample_id": sample_id,
        "warmup": False,
        "final_output_tokens": final_output_tokens,
        "request_ms": request_seconds * 1000.0,
        "ttft_ms": None if first_content_time is None else (first_content_time - request_started) * 1000.0,
        "end_to_end_output_tps": final_output_tokens / request_seconds if request_seconds > 0 else None,
        "client_delivery_tps": final_output_tokens / delivery_seconds if delivery_seconds and delivery_seconds > 0 else None,
        "server_predicted_tps_unvalidated": timings.get("predicted_per_second"),
        "prompt_eval_tokens": timings.get("prompt_n"),
        "prompt_tps": timings.get("prompt_per_second"),
        "draft_proposed": draft_proposed,
        "draft_accepted": draft_accepted,
        "draft_acceptance": draft_accepted / draft_proposed if draft_proposed and draft_accepted is not None else None,
        "draft_steps": deltas["llamacpp:spec_decode_num_drafts_total"],
        "raw_response": final_response,
    }


def context_size_effective(context_size: int) -> int:
    return ((context_size + 255) // 256) * 256


def verify_requested_device(config: dict[str, Any]) -> str:
    inventory = command_output([str(config["server"]), "--list-devices"])
    if inventory is None:
        raise RuntimeError("llama-server --list-devices failed")
    if "Vulkan0" not in inventory or "7900 XTX" not in inventory:
        raise RuntimeError("llama-server --list-devices did not confirm RX 7900 XTX as Vulkan0")
    return inventory


def verify_server_configuration(client: ServerClient, config: dict[str, Any]) -> int:
    props = client.props()
    settings = props.get("default_generation_settings")
    if not isinstance(settings, dict):
        raise RuntimeError("llama-server /props returned no default generation settings")
    expected_context = context_size_effective(config["context_size"])
    if settings.get("n_ctx") != expected_context:
        raise RuntimeError(
            f"llama-server context is {settings.get('n_ctx')}, expected {expected_context} "
            f"after 256-token alignment of requested {config['context_size']}"
        )
    if props.get("total_slots") != 1:
        raise RuntimeError(f"llama-server has {props.get('total_slots')} slots, expected 1")
    model_path = props.get("model_path")
    if not isinstance(model_path, str) or Path(model_path).resolve() != Path(config["model"]).resolve():
        raise RuntimeError("llama-server reported a different model path")
    return expected_context


def make_manifest(config: dict[str, Any], options: dict[str, Any], command: list[str]) -> dict[str, Any]:
    model_path = Path(config["model"])
    return {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "preset": options["preset"],
        "model": {
            "path": str(model_path),
            "filename": model_path.name,
            "size_bytes": model_path.stat().st_size,
            "sha256": file_sha256(model_path),
        },
        "server_command": command,
        "resolved_config": config,
        "runtime": {
            "host": options["host"],
            "port": options["port"],
            "context_requested": config["context_size"],
            "context_effective_expected": context_size_effective(config["context_size"]),
            "runs": options["runs"],
            "output_tokens": options["output_tokens"],
        },
        "server_version": command_output([str(config["server"]), "--version"]),
        "source_revision": command_output(["git", "rev-parse", "HEAD"]),
    }


def require_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_client:
        if socket_client.connect_ex((host, port)) == 0:
            raise RuntimeError(f"refusing to use occupied port {host}:{port}")


def make_results_dir(root: Path, label: str) -> Path:
    safe_label = "".join(char if char.isalnum() or char in "-_" else "-" for char in label)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{stamp}-{safe_label}"
    path.mkdir(parents=True)
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, sort_keys=True))
        file.write("\n")


def write_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    import csv

    if not samples:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)


def write_report(path: Path, summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    end_to_end_tps = summary["end_to_end_output_tps"]
    delivery_tps = summary["client_delivery_tps"]
    with path.open("w", encoding="utf-8") as file:
        file.write("# RX 7900 XTX MTP benchmark\n\n")
        file.write(f"- Preset: `{summary['preset']}`\n")
        file.write(f"- Prompt depth: {summary['depth']}\n")
        file.write(f"- Measured runs: {summary['runs']}\n")
        file.write(f"- Model SHA-256: `{manifest['model']['sha256']}`\n\n")
        runtime = manifest["runtime"]
        file.write(f"- Context requested: {runtime['context_requested']} tokens\n")
        file.write(f"- Context effective: {runtime.get('context_effective', runtime['context_effective_expected'])} tokens\n\n")
        file.write("## End-to-end final output throughput\n\n")
        file.write(f"- Mean: {end_to_end_tps['mean']} tokens/s\n")
        file.write(f"- Median: {end_to_end_tps['median']} tokens/s\n\n")
        file.write("## Client delivery throughput\n\n")
        file.write(f"- Mean: {delivery_tps['mean']} tokens/s\n")
        file.write(f"- Median: {delivery_tps['median']} tokens/s\n")
        file.write("\nServer predicted throughput is retained as unvalidated raw data.\n")
