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
KERNEL_LAUNCH_RE = re.compile(
    r"kernel\.launch\s+@(?P<name>[A-Za-z0-9_]+)"
    r"(?:\[(?P<workload>[^]]*)\])?\s*\("
)
SSA_VALUE_RE = re.compile(r"%(?P<name>[A-Za-z0-9_]+)")
TARGET_DEF_RE = re.compile(r"amdgpu\.target<(?P<selector>[^>]+)>\s+@(?P<name>[A-Za-z0-9_]+)")
KERNEL_TARGET_RE = re.compile(
    r"kernel\.def[^\n{]*target\(@(?P<target>[A-Za-z0-9_]+)\)[^\n{]*@(?P<name>[A-Za-z0-9_]+)\s*\("
)
KERNEL_DEF_RE = re.compile(r"kernel\.def[^\n{]*@(?P<name>[A-Za-z0-9_]+)\s*\(")


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
    upstream_revision = manifest.get("upstream_revision", "local")
    if upstream_revision != "local" and revision != upstream_revision:
        raise RuntimeError(f"HRX checkout {revision} does not match corpus {upstream_revision}")
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

    def module_inputs(name: str) -> list[pathlib.Path]:
        paths: list[pathlib.Path] = []
        seen: set[pathlib.Path] = set()

        def add(path: pathlib.Path) -> None:
            if path not in seen:
                seen.add(path)
                paths.append(path)

        def visit(module_name: str) -> None:
            recipe = modules[module_name]
            for source in recipe["srcs"]:
                add(args.corpus_dir / source)
            for library in recipe["libraries"]:
                if library.startswith(":"):
                    visit(library[1:])
                else:
                    add(args.corpus_dir / library)

        visit(name)
        return paths

    def link_module(name: str) -> pathlib.Path:
        if name in linked_paths:
            return linked_paths[name]
        recipe = modules[name]
        output = linked_dir / f"{name}.loom"
        command = [str(args.loom_link), "--mode=merge", f"--output={output}"]
        command.extend(str(args.corpus_dir / source) for source in recipe["srcs"])
        for library in recipe["libraries"]:
            path = link_module(library[1:]) if library.startswith(":") else args.corpus_dir / library
            command.append(f"--library={path}")
        require_run(command)
        linked_paths[name] = output
        return output

    exports = {item["symbol"]: item for item in manifest["exports"]}
    kernel_defs_by_source: dict[pathlib.Path, set[str]] = {}

    def kernel_defs(path: pathlib.Path) -> set[str]:
        if path not in kernel_defs_by_source:
            text = path.read_text(encoding="utf-8")
            kernel_defs_by_source[path] = {match.group("name") for match in KERNEL_DEF_RE.finditer(text)}
        return kernel_defs_by_source[path]

    def internal_inputs(root: str, inputs: list[pathlib.Path]) -> list[pathlib.Path]:
        owners = {path for path in inputs if root in kernel_defs(path)}
        if not owners:
            return inputs
        return [path for path in inputs if path in owners or not kernel_defs(path)]

    def export_inputs(root: str) -> list[pathlib.Path]:
        recipe = exports[root]["compile_recipe"]
        return [
            args.corpus_dir / source
            for source in recipe["primary_sources"] + recipe["library_sources"]
        ]

    def link_root(root: str, configs: tuple[str, ...] | list[str], inputs: list[pathlib.Path], output: pathlib.Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        target = "amdgpu:gfx11-generic" if args.target == "gfx1100" else "amdgpu:gfx1151"
        command = [
            str(args.loom_link),
            "--mode=link",
            f"--root=@{root}",
            f"--target={target}",
            "--require-resolved-config",
            "--to=text",
            f"--output={output}",
        ]
        command.extend(f"--config={config}" for config in configs)
        command.extend(str(path) for path in inputs)
        require_run(command)

    results: list[dict[str, object]] = []
    compiled_keys: set[tuple[str, tuple[int, ...], tuple[str, ...]]] = set()
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
        target_selectors = {
            match.group("name"): match.group("selector") for match in TARGET_DEF_RE.finditer(source_text)
        }
        kernel_selectors = {
            match.group("name"): target_selectors.get(match.group("target"), "")
            for match in KERNEL_TARGET_RE.finditer(source_text)
        }
        benchmark_defs = {
            match.group("name"): (match.group("case"),
                {item.group("name"): int(item.group("value")) for item in ATTR_RE.finditer(match.group("attrs") or "")})
            for match in BENCHMARK_RE.finditer(source.read_text(encoding="utf-8"))
        }
        case_literals: dict[str, dict[str, int]] = {}
        case_invocations: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
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
            case_invocations[match.group("name")] = [
                (
                    item.group("name"),
                    tuple(value.group("name") for value in SSA_VALUE_RE.finditer(item.group("workload") or "")),
                )
                for item in KERNEL_LAUNCH_RE.finditer(body)
            ]
        configs = [item.removeprefix("--config=") for item in case["args"] if item.startswith("--config=")]
        selects_benchmark = any(item.startswith("--benchmark=") for item in case["args"])
        owned_sources = set(modules[case["link_module"]]["srcs"]) if "link_module" in case else {case["source"]}
        case_inputs = module_inputs(case["link_module"]) if "link_module" in case else [args.corpus_dir / case["source"]]
        for row in plan_rows:
            benchmark_case, attrs = benchmark_defs.get(row["benchmark"], (row["case"], {}))
            concrete_values = dict(case_literals.get(benchmark_case, {}))
            concrete_values.update(attrs)
            expected_count = int(row.get("kernel_launch_count", row.get("actual_invocation_count", 1)))
            planned_invocation_count += expected_count
            invocations = case_invocations.get(benchmark_case, [])
            actual_entry = row.get("kernel_entry", row.get("actual_entry"))
            if actual_entry:
                invocations = [item for item in invocations if item[0] == actual_entry]
                if not invocations and actual_entry in exports:
                    invocations = [
                        (
                            actual_entry,
                            tuple(item["name"] for item in exports[actual_entry]["workload_parameters"]),
                        )
                    ]
            if len(invocations) != expected_count:
                raise RuntimeError(
                    f"BUILD recipe {case['name']} planner reports {expected_count} invocations for "
                    f"{benchmark_case}, but source resolves {len(invocations)} kernel launches: "
                    f"{[item[0] for item in invocations]}"
                )
            resolved_invocation_count += len(invocations)
            for root, parameter_names in invocations:
                if not selects_benchmark and root in exports and exports[root]["source"] not in owned_sources:
                    continue
                missing = [name for name in parameter_names if name not in concrete_values]
                if missing:
                    raise RuntimeError(f"{case['name']} recipe omits concrete {missing} for {root}")
                workload = tuple(concrete_values[name] for name in parameter_names)
                key = (root, workload, tuple(sorted(configs)))
                if key in compiled_keys:
                    continue
                compiled_keys.add(key)
                compile_dir = case_dir / f"{root}-{len(compiled_keys):03d}"
                selector = kernel_selectors.get(root, "")
                if selector and selector not in ("gfx11-generic", args.target):
                    results.append({"case": case["name"], "root": root, "workload": workload,
                                    "configs": configs, "artifact_dir": str(compile_dir),
                                    "status": "skipped-incompatible-target", "target_selector": selector})
                    continue
                linked_root = compile_dir / "linked.loom"
                inputs = export_inputs(root) if root in exports else internal_inputs(root, case_inputs)
                link_root(root, configs, inputs, linked_root)
                command = [str(args.compiler), "--target", args.target, "--source", str(linked_root),
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
        specializations: list[dict[str, object]] = []
        if schedule.get("schema") == "ggml-hrx-command-program-v1":
            for command in schedule.get("commands", []):
                if command.get("kind") != "Kernel":
                    continue
                qualified_root = str(command.get("kernel", ""))
                if not qualified_root.startswith("qwen3_moe:"):
                    continue
                specializations.append({
                    "variant": qualified_root.removeprefix("qwen3_moe:"),
                    "parameters": command.get("integer_parameters", {}),
                    "compile_parameters": command.get("compile_parameters", {}),
                })
        else:
            for invocation in schedule.get("invocations", []):
                for dispatch in invocation.get("dispatches", []):
                    specialization = dispatch["kernel"]
                    if specialization.get("execution") != "native":
                        raise RuntimeError(
                            f"schedule {schedule_path} contains non-native dispatch {specialization['variant']}"
                        )
                    specializations.append(specialization)
        if not specializations:
            raise RuntimeError(f"schedule {schedule_path} contains no Qwen MoE kernel requirements")
        for specialization in specializations:
            root = str(specialization["variant"])
            if root not in exports:
                raise RuntimeError(f"schedule {schedule_path} selects unmanifested Qwen MoE kernel {root}")
            schedule_requirement_count += 1
            parameters = specialization.get("parameters", {})
            parameter_names = [item["name"] for item in exports[root]["workload_parameters"]]
            missing = [name for name in parameter_names if name not in parameters]
            if missing:
                raise RuntimeError(f"schedule {schedule_path} omits concrete {missing} for {root}")
            workload = tuple(int(parameters[name]) for name in parameter_names)
            compile_parameters = specialization.get("compile_parameters", {})
            configs = tuple(f"{name}={value}" for name, value in sorted(compile_parameters.items()))
            key = (root, workload, configs)
            if key in compiled_keys:
                continue
            schedule_unique_requirement_count += 1
            compiled_keys.add(key)
            compile_dir = schedule_dir / f"{root}-{schedule_unique_requirement_count:03d}"
            linked_root = compile_dir / "linked.loom"
            link_root(root, configs, export_inputs(root), linked_root)
            command = [str(args.compiler), "--target", args.target, "--source", str(linked_root),
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
        "corpus_digest": manifest.get("corpus_sha256", ""),
        "recipe_digest": manifest.get("build_bazel_sha256", ""),
        "plan_case_count": len(manifest["plan_cases"]),
        "planned_invocation_count": planned_invocation_count,
        "resolved_invocation_count": resolved_invocation_count,
        "schedule_requirement_count": schedule_requirement_count,
        "schedule_unique_requirement_count": schedule_unique_requirement_count,
        "compile_count": len(results),
        "skipped_incompatible_target_count": sum(
            item["status"] == "skipped-incompatible-target" for item in results
        ),
        "failed_count": sum(item["status"] == "failed" for item in results),
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
