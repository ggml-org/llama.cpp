#!/usr/bin/env python3
#
# Run generated HRX model-scoped Loom benchmarks.

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any, NamedTuple


SCRIPT_DIR = Path(__file__).resolve().parent
HRX_DIR = SCRIPT_DIR.parents[1]
REPO_DIR = HRX_DIR.parents[2]
BENCHMARK_DIR = HRX_DIR / "benchmarks"


def fail(message: str) -> None:
    raise SystemExit(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_tool(env_name: str, binary_name: str, build_dir: Path | None, build_suffix: str) -> Path:
    override = os.environ.get(env_name)
    if override:
        path = Path(override)
        if path.is_file() and os.access(path, os.X_OK):
            return path
        fail(f"{env_name} is not executable: {path}")
    if build_dir is not None:
        matches = sorted(build_dir.glob(build_suffix))
        for match in matches:
            if match.is_file() and os.access(match, os.X_OK):
                return match
    found = shutil.which(binary_name)
    if found:
        return Path(found)
    fail(f"could not find {binary_name}; set {env_name} or pass --build-dir")


def source_path(entry: dict[str, Any], source: str) -> Path:
    corpus_dir = entry.get("corpus_dir")
    if not corpus_dir:
        fail(f"{entry['kernel']} has no corpus_dir in manifest")
    return HRX_DIR / corpus_dir / source


def loom_link_source_args(entry: dict[str, Any]) -> list[str]:
    sources = entry.get("sources")
    if sources is None:
        sources = [*entry.get("primary_sources", []), *entry.get("library_sources", [])]
    return [str(source_path(entry, source)) for source in sources]


def loom_link_provider_source_args(entry: dict[str, Any]) -> list[str]:
    primary_sources = entry.get("primary_sources")
    library_sources = entry.get("library_sources")
    closure = [str(source_path(entry, source)) for source in library_sources or []]
    providers = [f"--library={source_path(entry, source)}" for source in primary_sources or []]
    return [*closure, *providers]


def missed_exact_source_selection(result: CommandResult) -> bool:
    if result.state == "ok" or not result.stderr.is_file():
        return False
    return "exact link selection missed reachable source symbol ref" in result.stderr.read_text(encoding="utf-8")


def manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if "dispatches" in manifest:
        return manifest.get("dispatches", [])
    return manifest.get("benchmarks", [])


def entry_status(entry: dict[str, Any]) -> str:
    status = entry.get("status")
    if status is not None:
        return str(status)
    if entry.get("benchmark"):
        return "generated"
    return "unsupported"


def entry_count(entry: dict[str, Any]) -> int:
    return int(entry.get("count", entry.get("shape_multiplicity", 1)))


def parse_bool(value: str) -> bool:
    value = value.lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


class CommandResult(NamedTuple):
    state: str
    error: str | None
    returncode: int | None
    signal_name: str | None
    stdout: Path
    stderr: Path
    command: Path


def format_signal(returncode: int | None) -> str | None:
    if returncode is None or returncode >= 0:
        return None
    signum = -returncode
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"SIG{signum}"


def process_output_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def run_command(command: list[str], cwd: Path, timeout_sec: float | None, bench_dir: Path, stage: str) -> CommandResult:
    stdout_path = bench_dir / f"{stage}.stdout.txt"
    stderr_path = bench_dir / f"{stage}.stderr.txt"
    command_path = bench_dir / f"{stage}.command.json"
    command_record: dict[str, Any] = {
        "argv": command,
        "cwd": str(cwd),
        "timeout_sec": timeout_sec,
    }
    try:
        result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(process_output_text(exc.stdout), encoding="utf-8")
        stderr_path.write_text(process_output_text(exc.stderr), encoding="utf-8")
        command_record.update(
            {
                "state": "timeout",
                "returncode": None,
                "signal": None,
                "timed_out": True,
            }
        )
        command_path.write_text(json.dumps(command_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return CommandResult(
            "timeout",
            f"{stage} timed out after {exc.timeout} seconds",
            None,
            None,
            stdout_path,
            stderr_path,
            command_path,
        )

    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    sig_name = format_signal(result.returncode)
    command_record.update(
        {
            "state": "ok" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "signal": sig_name,
            "timed_out": False,
        }
    )
    command_path.write_text(json.dumps(command_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result.returncode == 0:
        return CommandResult("ok", None, result.returncode, sig_name, stdout_path, stderr_path, command_path)
    if sig_name:
        return CommandResult(
            "failed",
            f"{stage} exited with status {result.returncode} ({sig_name})",
            result.returncode,
            sig_name,
            stdout_path,
            stderr_path,
            command_path,
        )
    return CommandResult(
        "failed",
        f"{stage} exited with status {result.returncode}",
        result.returncode,
        sig_name,
        stdout_path,
        stderr_path,
        command_path,
    )


def command_result_json(result: CommandResult) -> dict[str, Any]:
    return {
        "state": result.state,
        "error": result.error,
        "returncode": result.returncode,
        "signal": result.signal_name,
        "stdout": str(result.stdout),
        "stderr": str(result.stderr),
        "command": str(result.command),
    }


def find_matching_char(text: str, open_index: int, open_char: str, close_char: str) -> int:
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unmatched {open_char!r}")


def parse_root_index_parameters(source_text: str, symbol: str) -> list[str]:
    marker = f"@{symbol}("
    start = source_text.find(marker)
    if start < 0:
        raise ValueError(f"could not find root @{symbol}")
    params_start = start + len(marker) - 1
    params_end = find_matching_char(source_text, params_start, "(", ")")
    params_text = source_text[params_start + 1 : params_end]
    names = []
    for raw_param in params_text.split(","):
        parts = raw_param.strip().split()
        if len(parts) < 2 or parts[-1] != "index":
            continue
        name = parts[-2].rstrip(":")
        if not name.startswith("%"):
            continue
        names.append(name[1:])
    return names


def replace_index_uses(region: str, name: str, constant_name: str) -> str:
    token = "%" + name
    output = []
    index = 0
    while True:
        match = region.find(token, index)
        if match < 0:
            output.append(region[index:])
            return "".join(output)
        before_ok = match == 0 or not (region[match - 1].isalnum() or region[match - 1] == "_")
        end = match + len(token)
        after_ok = end == len(region) or not (region[end].isalnum() or region[end] == "_")
        output.append(region[index:match])
        if before_ok and after_ok:
            output.append("%" + constant_name)
        else:
            output.append(token)
        index = end


def specialize_region(region: str, workload_values: list[tuple[str, int]]) -> str:
    body = region
    constants = []
    for name, value in workload_values:
        constant_name = f"ggml_hrx_specialized_{name}"
        constants.append(f"  %{constant_name} = index.constant {value} : index\n")
        body = replace_index_uses(body, name, constant_name)
    return "\n" + "".join(constants) + body


def specialize_workload_text(source_text: str, symbol: str, workload_values: list[tuple[str, int]]) -> str:
    if not workload_values:
        return source_text
    marker = f"@{symbol}("
    root_start = source_text.find(marker)
    if root_start < 0:
        raise ValueError(f"could not find root @{symbol}")
    params_start = root_start + len(marker) - 1
    params_end = find_matching_char(source_text, params_start, "(", ")")

    config_open = source_text.find("{", params_end)
    if config_open < 0:
        raise ValueError("could not find kernel config block")
    config_close = find_matching_char(source_text, config_open, "{", "}")

    launch_marker = "launch("
    launch_start = source_text.find(launch_marker, config_close)
    if launch_start < 0:
        raise ValueError("could not find launch region")
    launch_body_open = source_text.find("{", launch_start)
    if launch_body_open < 0:
        raise ValueError("could not find launch body")
    launch_body_close = find_matching_char(source_text, launch_body_open, "{", "}")

    config_body = source_text[config_open + 1 : config_close]
    launch_body = source_text[launch_body_open + 1 : launch_body_close]
    specialized_config = specialize_region(config_body, workload_values)
    specialized_launch = specialize_region(launch_body, workload_values)

    return (
        source_text[: config_open + 1]
        + specialized_config
        + source_text[config_close: launch_body_open + 1]
        + specialized_launch
        + source_text[launch_body_close:]
    )


def specialize_linked_source(entry: dict[str, Any], linked_source: Path, output_source: Path) -> bool:
    symbol = entry.get("symbol") or entry["kernel"].split(":")[-1]
    integer_parameters = entry.get("integer_parameters", {})
    workload_parameters = entry.get("workload_parameters") or []
    workload_names = [param["name"] for param in workload_parameters if param.get("type") == "index"]
    source_text = linked_source.read_text(encoding="utf-8")
    if not workload_names:
        workload_names = parse_root_index_parameters(source_text, symbol)
    workload_values = []
    for name in workload_names:
        if name not in integer_parameters:
            continue
        workload_values.append((name, int(integer_parameters[name])))
    if not workload_values:
        return False
    output_source.write_text(specialize_workload_text(source_text, symbol, workload_values), encoding="utf-8")
    return True


def select_entries(manifest: dict[str, Any], benchmark: str | None, include_unsupported: bool) -> list[dict[str, Any]]:
    entries = manifest_entries(manifest)
    if benchmark:
        requested = benchmark if benchmark.startswith("@") else "@" + benchmark
        entries = [entry for entry in entries if entry.get("benchmark") == requested]
        if not entries:
            fail(f"benchmark not found in manifest: {requested}")
    if include_unsupported:
        return entries
    return [entry for entry in entries if entry_status(entry) == "generated"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model slug, for example llama32_3b_f16.")
    parser.add_argument("--scenario", required=True, help="Scenario slug, for example pp512.")
    parser.add_argument("--build-dir", type=Path, help="llama.cpp build directory.")
    parser.add_argument("--output-dir", type=Path, help="Directory for linked sources, plans, and results.")
    parser.add_argument("--manifest", type=Path, help="Generated model scenario manifest.")
    parser.add_argument("--benchmark", help="Single benchmark symbol to run.")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "amdgpu"), help="HAL device. Defaults to amdgpu.")
    parser.add_argument("--dry-run-only", action="store_true", help="Stop after benchmark planning.")
    parser.add_argument("--list", action="store_true", help="List selected generated benchmarks and exit.")
    parser.add_argument("--include-unsupported", action="store_true", help="Include unsupported manifest entries when listing.")
    parser.add_argument("--batch-size", default="64")
    parser.add_argument("--iterations", default="10")
    parser.add_argument("--warmup-iterations", default="1")
    parser.add_argument("--continue-on-failure", action="store_true", help="Record failed benchmarks and continue running the rest.")
    parser.add_argument("--benchmark-timeout-sec", type=float, default=300.0, help="Timeout per benchmark tool invocation. Defaults to 300.")
    parser.add_argument("--profile-final-batch", type=parse_bool, default=True, help="Pass --profile-final-batch to iree-benchmark-loom. Defaults to true.")
    parser.add_argument("--input-ring-count", help="Optional --input-ring-count value for iree-benchmark-loom.")
    parser.add_argument("--benchmark-extra-arg", action="append", default=[], help="Additional argument to pass to benchmark invocations. May be repeated.")
    parser.add_argument("--runtime-specialization", action=argparse.BooleanOptionalAction, default=True, help="Specialize workload index parameters after loom-link. Defaults to enabled.")
    args = parser.parse_args()

    build_dir = args.build_dir
    if build_dir is not None and not build_dir.is_absolute():
        build_dir = REPO_DIR / build_dir
    manifest_path = args.manifest or (BENCHMARK_DIR / "loom" / f"{args.model}.{args.scenario}.json")
    manifest = load_json(manifest_path)
    loom_source = HRX_DIR / manifest["loom_source"]
    entries = select_entries(manifest, args.benchmark, args.include_unsupported)

    if args.list:
        for entry in entries:
            count = entry_count(entry)
            print(f"{entry.get('benchmark') or '-'} {entry_status(entry)} {entry['kernel']} x{count}")
        return

    entries = [entry for entry in entries if entry_status(entry) == "generated"]
    if not entries:
        fail("no generated benchmarks selected")
    if args.output_dir is None:
        fail("--output-dir is required unless --list is used")

    loom_link = find_tool("LOOM_LINK", "loom-link", build_dir, "**/loom-link")
    iree_benchmark_loom = find_tool("IREE_BENCHMARK_LOOM", "iree-benchmark-loom", build_dir, "**/iree-benchmark-loom")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "results.jsonl"
    with summary_path.open("w", encoding="utf-8") as summary:
        for entry in entries:
            benchmark = entry["benchmark"]
            safe_name = benchmark[1:]
            bench_dir = args.output_dir / safe_name
            artifact_dir = bench_dir / "artifacts"
            bench_dir.mkdir(parents=True, exist_ok=True)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            linked_source = bench_dir / "linked.loom"
            specialized_source = bench_dir / "linked.runtime-specialized.loom"
            plan_output = bench_dir / "plan.json"
            result_output = bench_dir / "results.json"
            stage_results: dict[str, Any] = {}
            config_args = [f"--config={name}={value}" for name, value in sorted(entry.get("compile_parameters", {}).items())]

            link_inputs = loom_link_source_args(entry)
            state = "ok"
            error = None
            link_result = run_command(
                [
                    str(loom_link),
                    str(loom_source),
                    *link_inputs,
                    "--mode=link",
                    "--to=text",
                    f"--root={benchmark}",
                    f"--output={linked_source}",
                    *config_args,
                ],
                REPO_DIR,
                args.benchmark_timeout_sec,
                bench_dir,
                "loom-link",
            )
            stage_results["loom-link"] = command_result_json(link_result)
            if missed_exact_source_selection(link_result) and entry.get("primary_sources"):
                provider_link_inputs = loom_link_provider_source_args(entry)
                link_result = run_command(
                    [
                        str(loom_link),
                        str(loom_source),
                        *provider_link_inputs,
                        "--mode=link",
                        "--to=text",
                        f"--root={benchmark}",
                        f"--output={linked_source}",
                        *config_args,
                    ],
                    REPO_DIR,
                    args.benchmark_timeout_sec,
                    bench_dir,
                    "loom-link-provider-retry",
                )
                stage_results["loom-link-provider-retry"] = command_result_json(link_result)
            state = link_result.state
            error = link_result.error
            if state == "ok":
                benchmark_source = linked_source
                runtime_specialized = False
                if args.runtime_specialization:
                    try:
                        runtime_specialized = specialize_linked_source(entry, linked_source, specialized_source)
                    except ValueError as exc:
                        state = "failed"
                        error = f"runtime specialization failed: {exc}"
                    if runtime_specialized:
                        benchmark_source = specialized_source
                else:
                    runtime_specialized = False
            else:
                benchmark_source = linked_source
                runtime_specialized = False
            if state == "ok":
                dry_run = [
                    str(iree_benchmark_loom),
                    str(benchmark_source),
                    f"--benchmark={benchmark}",
                    f"--device={args.device}",
                    "--dry-run",
                    "--measure=dispatch_complete",
                    "--output-format=jsonl",
                    f"--output={plan_output}",
                    *config_args,
                ]
                dry_run_result = run_command(dry_run, REPO_DIR, args.benchmark_timeout_sec, bench_dir, "dry-run")
                stage_results["dry-run"] = command_result_json(dry_run_result)
                state = dry_run_result.state
                error = dry_run_result.error
            if state == "ok" and not args.dry_run_only:
                benchmark_command = [
                    str(iree_benchmark_loom),
                    str(benchmark_source),
                    f"--benchmark={benchmark}",
                    f"--device={args.device}",
                    "--measure=dispatch_complete",
                    f"--batch-size={args.batch_size}",
                    f"--iterations={args.iterations}",
                    f"--warmup-iterations={args.warmup_iterations}",
                    f"--profile-final-batch={str(args.profile_final_batch).lower()}",
                    f"--artifact-bundle-dir={artifact_dir}",
                    "--output-format=jsonl",
                    f"--output={result_output}",
                    *config_args,
                    *args.benchmark_extra_arg,
                ]
                if args.input_ring_count is not None:
                    benchmark_command.append(f"--input-ring-count={args.input_ring_count}")
                benchmark_result = run_command(benchmark_command, REPO_DIR, args.benchmark_timeout_sec, bench_dir, "benchmark")
                stage_results["benchmark"] = command_result_json(benchmark_result)
                state = benchmark_result.state
                error = benchmark_result.error
            summary.write(
                json.dumps(
                    {
                        "benchmark": benchmark,
                        "kernel": entry["kernel"],
                        "count": entry_count(entry),
                        "linked_source": str(linked_source),
                        "benchmark_source": str(benchmark_source),
                        "runtime_specialization": runtime_specialized,
                        "plan": str(plan_output),
                        "results": str(result_output) if not args.dry_run_only else None,
                        "state": state,
                        "error": error,
                        "stages": stage_results,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            summary.flush()
            print(f"{state} {benchmark}")
            if state != "ok" and not args.continue_on_failure:
                fail(f"{benchmark}: {error}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
