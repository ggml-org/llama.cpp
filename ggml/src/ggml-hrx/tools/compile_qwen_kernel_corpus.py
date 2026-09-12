#!/usr/bin/env python3
"""Compiles the pinned Qwen corpus using only BUILD.bazel-authored recipes."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys


BENCHMARK_RE = re.compile(
    r"check\.benchmark<@(?P<case>[A-Za-z0-9_]+)>\s+@(?P<name>[A-Za-z0-9_]+)"
    r"(?:\s*\{(?P<attrs>[^}]*)\})?"
)
ATTR_RE = re.compile(r"(?P<name>[A-Za-z0-9_.]+)\s*=\s*(?P<value>-?[0-9]+)")
CASE_RE = re.compile(r"check\.case(?:\s+public)?\s+@(?P<name>[A-Za-z0-9_]+)\s*\{")
LITERAL_RE = re.compile(r"%(?P<name>[A-Za-z0-9_]+)\s*=\s*check\.literal\s+value\((?P<value>-?[0-9]+)\)\s*:\s*index")
FUNC_CALL_RE = re.compile(r"func\.call\s+@(?P<name>[A-Za-z0-9_]+)\s*\(")


def run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False)


def require_run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = run(command, env=env)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--corpus-dir", type=pathlib.Path, required=True)
    parser.add_argument("--hrx-source", type=pathlib.Path, required=True)
    parser.add_argument("--loom-link", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark-tool", type=pathlib.Path, required=True)
    parser.add_argument("--compiler", type=pathlib.Path, required=True)
    parser.add_argument("--target", choices=("gfx1100", "gfx1151"), required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--schedule", type=pathlib.Path, action="append", default=[],
                        help="materialized program.json whose exact specializations must compile")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    revision = require_run(["git", "-C", str(args.hrx_source), "rev-parse", "HEAD"]).stdout.strip()
    if revision != manifest["upstream_revision"]:
        raise RuntimeError(f"HRX checkout {revision} does not match corpus {manifest['upstream_revision']}")
    if require_run(["git", "-C", str(args.hrx_source), "status", "--porcelain"]).stdout.strip():
        raise RuntimeError("refusing to compile against a dirty pinned HRX checkout")
    override = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if args.target == "gfx1100" and override != "11.0.0":
        raise RuntimeError("gfx1100 validation on this gfx1151 host requires HSA_OVERRIDE_GFX_VERSION=11.0.0")
    if args.target == "gfx1151" and override:
        raise RuntimeError("native gfx1151 validation must run without HSA_OVERRIDE_GFX_VERSION")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modules = {item["name"]: item for item in manifest["link_modules"]}
    linked_dir = args.output_dir / "linked"
    linked_dir.mkdir(exist_ok=True)
    linked_paths: dict[str, pathlib.Path] = {}

    def link_module(name: str) -> pathlib.Path:
        if name in linked_paths:
            return linked_paths[name]
        recipe = modules[name]
        output = linked_dir / f"{name}.loom"
        command = [str(args.loom_link), "--mode=archive", f"--output={output}"]
        command.extend(str(args.corpus_dir / source) for source in recipe["srcs"])
        for library in recipe["libraries"]:
            path = link_module(library[1:]) if library.startswith(":") else args.corpus_dir / library
            command.append(f"--library={path}")
        require_run(command)
        linked_paths[name] = output
        return output

    exports = {item["symbol"]: item for item in manifest["exports"]}

    def source_for_root(root: str) -> pathlib.Path:
        source_name = exports[root]["source"]
        direct_modules = [name for name, recipe in modules.items() if source_name in recipe["srcs"]]
        if direct_modules:
            return link_module(direct_modules[0])
        library_modules = [name for name, recipe in modules.items() if source_name in recipe["libraries"]]
        if library_modules:
            return link_module(library_modules[0])
        return args.corpus_dir / source_name

    results: list[dict[str, object]] = []
    compiled_keys: set[tuple[str, tuple[int, ...], tuple[str, ...]]] = set()
    root_sources: dict[str, pathlib.Path] = {}
    planned_invocation_count = 0
    resolved_invocation_count = 0
    for case in manifest["plan_cases"]:
        source = link_module(case["link_module"]) if "link_module" in case else args.corpus_dir / case["source"]
        plan_command = [str(args.benchmark_tool)]
        for item in case["args"]:
            if item.startswith("$(location"):
                plan_command.append(str(source))
            else:
                plan_command.append(item)
        plan = require_run(plan_command)
        case_dir = args.output_dir / "recipes" / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "plan.jsonl").write_text(plan.stdout, encoding="utf-8")
        (case_dir / "plan.stderr.txt").write_text(plan.stderr, encoding="utf-8")
        plan_rows = [json.loads(line) for line in plan.stdout.splitlines() if line.strip()]
        plan_rows = [row for row in plan_rows if row.get("row") == "plan"]
        if not plan_rows:
            raise RuntimeError(f"BUILD recipe {case['name']} produced no planner rows")
        source_text = source.read_text(encoding="utf-8")
        benchmark_defs = {
            match.group("name"): (match.group("case"),
                {item.group("name"): int(item.group("value")) for item in ATTR_RE.finditer(match.group("attrs") or "")})
            for match in BENCHMARK_RE.finditer(source.read_text(encoding="utf-8"))
        }
        case_literals: dict[str, dict[str, int]] = {}
        case_calls: dict[str, list[str]] = {}
        for match in CASE_RE.finditer(source_text):
            depth = 1
            cursor = match.end()
            while cursor < len(source_text) and depth:
                depth += source_text[cursor] == "{"
                depth -= source_text[cursor] == "}"
                cursor += 1
            body = source_text[match.end():cursor - 1]
            case_literals[match.group("name")] = {
                item.group("name"): int(item.group("value")) for item in LITERAL_RE.finditer(body)
            }
            case_calls[match.group("name")] = [item.group("name") for item in FUNC_CALL_RE.finditer(body)]
        configs = [item.removeprefix("--config=") for item in case["args"] if item.startswith("--config=")]
        selects_benchmark = any(item.startswith("--benchmark=") for item in case["args"])
        owned_sources = set(modules[case["link_module"]]["srcs"]) if "link_module" in case else {case["source"]}
        for row in plan_rows:
            benchmark_case, attrs = benchmark_defs.get(row["benchmark"], (row["case"], {}))
            concrete_values = dict(case_literals.get(benchmark_case, {}))
            concrete_values.update(attrs)
            expected_count = int(row.get("actual_invocation_count", 1))
            planned_invocation_count += expected_count
            if row.get("actual_entry"):
                roots = [row["actual_entry"]]
            else:
                roots = [root for root in case_calls.get(benchmark_case, []) if root in exports]
            if len(roots) != expected_count:
                raise RuntimeError(
                    f"BUILD recipe {case['name']} planner reports {expected_count} invocations for "
                    f"{benchmark_case}, but source resolves {len(roots)} exported calls: {roots}"
                )
            resolved_invocation_count += len(roots)
            for root in roots:
                if root not in exports:
                    raise RuntimeError(f"BUILD recipe {case['name']} selected unmanifested kernel {root}")
                if not selects_benchmark and exports[root]["source"] not in owned_sources:
                    continue
                root_sources.setdefault(root, source)
                parameter_names = [item["name"] for item in exports[root]["workload_parameters"]]
                missing = [name for name in parameter_names if name not in concrete_values]
                if missing:
                    raise RuntimeError(f"{case['name']} recipe omits concrete {missing} for {root}")
                workload = tuple(concrete_values[name] for name in parameter_names)
                key = (root, workload, tuple(configs))
                if key in compiled_keys:
                    continue
                compiled_keys.add(key)
                compile_dir = case_dir / f"{root}-{len(compiled_keys):03d}"
                command = [str(args.compiler), "--target", args.target, "--source", str(source),
                           "--root", root, "--output", str(compile_dir)]
                for config in configs:
                    command.extend(("--config", config))
                for value in workload:
                    command.extend(("--workload", str(value)))
                compile_result = run(command, env=os.environ.copy())
                (case_dir / f"{root}-{len(compiled_keys):03d}.stdout.txt").write_text(compile_result.stdout, encoding="utf-8")
                (case_dir / f"{root}-{len(compiled_keys):03d}.stderr.txt").write_text(compile_result.stderr, encoding="utf-8")
                result = {"case": case["name"], "root": root, "workload": workload,
                          "configs": configs, "artifact_dir": str(compile_dir),
                          "status": "ok" if compile_result.returncode == 0 else "failed"}
                results.append(result)

    schedule_requirement_count = 0
    schedule_unique_requirement_count = 0
    for schedule_path in args.schedule:
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        schedule_dir = args.output_dir / "schedules" / schedule_path.parent.name
        schedule_dir.mkdir(parents=True, exist_ok=True)
        for invocation in schedule.get("invocations", []):
            for dispatch in invocation.get("dispatches", []):
                specialization = dispatch["kernel"]
                if specialization.get("execution") != "native":
                    raise RuntimeError(f"schedule {schedule_path} contains non-native dispatch {specialization['variant']}")
                root = specialization["variant"]
                if root not in exports:
                    raise RuntimeError(f"schedule {schedule_path} selects unmanifested kernel {root}")
                schedule_requirement_count += 1
                parameters = specialization.get("parameters", {})
                parameter_names = [item["name"] for item in exports[root]["workload_parameters"]]
                missing = [name for name in parameter_names if name not in parameters]
                if missing:
                    raise RuntimeError(f"schedule {schedule_path} omits concrete {missing} for {root}")
                workload = tuple(int(parameters[name]) for name in parameter_names)
                configs = tuple(f"{name}={value}" for name, value in
                                sorted(specialization.get("compile_parameters", {}).items()))
                key = (root, workload, configs)
                if key in compiled_keys:
                    continue
                schedule_unique_requirement_count += 1
                compiled_keys.add(key)
                source = root_sources[root] if root in root_sources else source_for_root(root)
                compile_dir = schedule_dir / f"{root}-{schedule_unique_requirement_count:03d}"
                command = [str(args.compiler), "--target", args.target, "--source", str(source),
                           "--root", root, "--output", str(compile_dir)]
                for config in configs:
                    command.extend(("--config", config))
                for value in workload:
                    command.extend(("--workload", str(value)))
                compile_result = run(command, env=os.environ.copy())
                (schedule_dir / f"{root}-{schedule_unique_requirement_count:03d}.stdout.txt").write_text(
                    compile_result.stdout, encoding="utf-8")
                (schedule_dir / f"{root}-{schedule_unique_requirement_count:03d}.stderr.txt").write_text(
                    compile_result.stderr, encoding="utf-8")
                results.append({"case": f"schedule:{schedule_path}", "root": root, "workload": workload,
                                "configs": configs, "artifact_dir": str(compile_dir),
                                "status": "ok" if compile_result.returncode == 0 else "failed"})

    summary = {
        "schema": "ggml-hrx-qwen-compile-report-v1",
        "target": args.target,
        "hsa_override_gfx_version": override,
        "hrx_revision": revision,
        "corpus_digest": manifest["corpus_sha256"],
        "recipe_digest": manifest["build_bazel_sha256"],
        "plan_case_count": len(manifest["plan_cases"]),
        "planned_invocation_count": planned_invocation_count,
        "resolved_invocation_count": resolved_invocation_count,
        "schedule_requirement_count": schedule_requirement_count,
        "schedule_unique_requirement_count": schedule_unique_requirement_count,
        "compile_count": len(results),
        "failed_count": sum(item["status"] != "ok" for item in results),
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("target", "plan_case_count", "compile_count", "failed_count")}))
    return 0 if summary["failed_count"] == 0 and planned_invocation_count == resolved_invocation_count else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"compile corpus: {error}", file=sys.stderr)
        raise SystemExit(2)
