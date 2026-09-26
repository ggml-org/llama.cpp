#!/usr/bin/env python3
"""Mirrors the bounded Qwen MoE Loom corpus with reproducible provenance."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile


SOURCE_SUBDIR = pathlib.Path("experimental/qwen_moe/kernels")
QWEN_ENDPOINT_SOURCE_SUBDIR = pathlib.Path("experimental/qwen/kernels")
CORPUS_FILES = (
    "ggml/linear_q6k_f32.loom",
    "ggml/linear_q6k_q8_1_x4.loom",
    "ggml/quantize_q8_1_x4.loom",
    "qwen3_moe/attention_postprocess_f32_f16.loom",
    "qwen3_moe/attention_prepare_quantized.loom",
    "qwen3_moe/attention_qkv_postprocess_fused.loom",
    "qwen3_moe/attention_qkv_quantized.loom",
    "qwen3_moe/attention_qkv_same_format_prefill.loom",
    "qwen3_moe/batched_decode_expert_dispatch.loom",
    "qwen3_moe/batched_decode_gate_up_q4k.loom",
    "qwen3_moe/dense_linear_quantized_f16_wmma.loom",
    "qwen3_moe/expert_table_partition_fused.loom",
    "qwen3_moe/flash_attention_decode_f32_f16_wmma.loom",
    "qwen3_moe/flash_attention_decode_q128_f32_f16_wmma.loom",
    "qwen3_moe/flash_attention_decode_split_f32_f16_wmma.loom",
    "qwen3_moe/flash_attention_decode_split_next_q8_test.loom",
    "qwen3_moe/flash_attention_f32_f16_wmma.loom",
    "qwen3_moe/model_config.loom",
    "qwen3_moe/routed_down_q4k.loom",
    "qwen3_moe/routed_down_q6k.loom",
    "qwen3_moe/routed_down_next_q8.loom",
    "qwen3_moe/routed_down_quantized_f16_wmma.loom",
    "qwen3_moe/routed_down_weighted_reduce_next_rmsnorm_f32.loom",
    "qwen3_moe/routed_down_weighted_reduce_next_rmsnorm_q8_1_x4.loom",
    "qwen3_moe/routed_gate_up_swiglu_q4k.loom",
    "qwen3_moe/routed_linear_q4k_f16_wmma.loom",
    "qwen3_moe/router_projection_f32.loom",
    "qwen3_moe/router_projection_top8_fused_f32.loom",
    "qwen3_moe/router_top8_f32.loom",
)
QWEN_ENDPOINT_FILES = (
    ("token_embedding_q4k.loom", "qwen_owned/token_embedding_q4k.loom"),
    ("attention_metadata.loom", "qwen_owned/attention_metadata.loom"),
)

# These integration kernels are deliberately owned by the llama.cpp HRX
# backend. They are not attributed to the pinned qwen_moe corpus or its BUILD
# recipes.
OWNED_KERNEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "kernel-corpus" / "kernels"
OWNED_FILES = (
    "qwen_owned/token_embedding_bringup_workaround.loom",
    "qwen_owned/attention_state_initialize.loom",
    "qwen_owned/attention_metadata_bringup_workaround.loom",
    "hrx_owned/gather_add_f32.loom",
    "hrx_owned/add_f32.loom",
)

KERNEL_RE = re.compile(
    r"kernel\.def(?P<modifiers>(?:\s+(?:target\([^)]*\)|export\(\"[^\"]+\"\)))*)"
    r"\s+@(?P<symbol>[A-Za-z0-9_]+)"
    r"\((?P<workload>.*?)\)\s*\{.*?\}\s*launch\((?P<launch>.*?)\)"
    r"(?:\s+where\s+\[[^\]]*\])?\s*\{",
    re.DOTALL,
)
ARG_RE = re.compile(r"%(?P<name>[A-Za-z0-9_]+)\s*:\s*(?P<type>[A-Za-z0-9<>?]+)")
TARGET_MODIFIER_RE = re.compile(r"target\(@(?P<symbol>[A-Za-z0-9_]+)\)")
EXPORT_MODIFIER_RE = re.compile(r"export\(\"(?P<name>[^\"]+)\"\)")
AMDGPU_TARGET_RE = re.compile(
    r"amdgpu\.target<(?P<selector>[A-Za-z0-9_.-]+)>\s+@(?P<symbol>[A-Za-z0-9_]+)"
)


def binding_access(symbol: str, name: str) -> str:
    """Authoritative launch ABI access contract; no name inference at runtime."""
    if symbol == "qwen_attention_context_base_capture":
        return "read" if name == "positions" else "write"
    if symbol == "qwen_attention_decode_state_initialize":
        return "read" if name == "positions" else "write"
    if symbol == "qwen_attention_metadata_bringup_workaround" and name != "control":
        return "read_write"
    if symbol == "qwen_attention_metadata" and name != "control":
        return "read_write"
    if symbol == "qwen_decode_attention_metadata" and name != "control":
        return "read_write"
    if symbol == "qwen3_moe_router_top8_f32" and name in ("route_ids", "route_weights"):
        return "write"
    if symbol == "qwen3_moe_router_projection_top8_fused_decode_f32" and name in (
            "logits", "completion_counter", "route_ids", "route_weights"):
        return "read_write"
    if symbol == "qwen3_moe_attention_qkv_postprocess_fused_decode" and name in (
            "query_output_raw", "key_output_raw", "value_output_raw",
            "query_output", "key_cache", "value_cache", "completion_counters"):
        return "read_write"
    if symbol == "qwen3_moe_build_expert_table" and name == "expert_table":
        return "write"
    if symbol == "qwen3_moe_build_expert_partition_table" and name == "partition_table":
        return "write"
    if symbol == "qwen3_moe_build_expert_table_partition_prefill_512" and name in ("expert_table", "partition_table"):
        return "write"
    if symbol == "qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32":
        if name == "hidden_state":
            return "read_write"
        if name == "next_projection_input":
            return "write"
    if symbol == "ggml_q8_1_x4_inspect_one_group" and name != "packed":
        return "write"
    if name in ("output", "query_output", "key_output", "value_output", "normalized_output", "q8_output",
                "next_q8_output", "key_cache", "value_cache", "partial_max", "partial_sum", "partial_output",
                "completion_counter", "completion_counters"):
        return "read_write"
    return "read"


def starlark_calls(text: str, function: str) -> list[str]:
    """Extracts the literal-only calls used by the pinned kernel BUILD file."""
    result: list[str] = []
    marker = function + "("
    cursor = 0
    while (start := text.find(marker, cursor)) != -1:
        index = start + len(marker)
        depth = 1
        quote: str | None = None
        escaped = False
        while index < len(text) and depth:
            character = text[index]
            if quote:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
            elif character in "\"'":
                quote = character
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            index += 1
        if depth:
            raise RuntimeError(f"unterminated {function} call in BUILD.bazel")
        result.append(text[start + len(marker) : index - 1])
        cursor = index
    return result


def literal_assignment(call: str, name: str, default: object = None) -> object:
    match = re.search(rf"(?:^|\n)\s*{re.escape(name)}\s*=\s*(\[[\s\S]*?\]|\"[^\"]*\")\s*,", call)
    return default if match is None else ast.literal_eval(match.group(1))


def parse_build_recipes(text: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    modules: list[dict[str, object]] = []
    for call in starlark_calls(text, "loom_link_module"):
        modules.append({
            "name": literal_assignment(call, "name"),
            "srcs": literal_assignment(call, "srcs", []),
            "libraries": literal_assignment(call, "libraries", []),
        })
    module_names = {str(module["name"]) for module in modules}
    cases: list[dict[str, object]] = []
    for call in starlark_calls(text, "iree_executable_test"):
        name = literal_assignment(call, "name")
        if not isinstance(name, str) or not name.endswith("_plan_test"):
            continue
        args = literal_assignment(call, "args", [])
        data = literal_assignment(call, "data", [])
        linked = [item[1:] for item in data if isinstance(item, str) and item.startswith(":") and item[1:] in module_names]
        direct_sources = [item for item in data if isinstance(item, str) and item.endswith(".loom")]
        if len(linked) + len(direct_sources) != 1:
            raise RuntimeError(f"plan test {name} does not name exactly one linked module or direct Loom source")
        case = {
            "name": name,
            "args": args,
        }
        if linked:
            case["link_module"] = linked[0]
        else:
            case["source"] = direct_sources[0]
        cases.append(case)
    if not modules or not cases:
        raise RuntimeError("BUILD.bazel contains no pinned Loom link/plan recipes")
    return modules, cases


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def optional_git(repo: pathlib.Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def normalize_repository_url(url: str) -> str:
    """Normalizes GitHub transport spelling without changing repository identity."""
    match = re.fullmatch(r"(?:ssh://)?git@github\.com[:/](?P<path>.+)", url)
    if match:
        return f"https://github.com/{match.group('path')}"
    return url


def upstream_repository(repo: pathlib.Path) -> str:
    """Returns provenance without imposing a local Git remote name."""
    remotes: list[str] = []
    branch = optional_git(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
    if branch:
        branch_remote = optional_git(repo, "config", "--get", f"branch.{branch}.remote")
        if branch_remote and branch_remote != ".":
            remotes.append(branch_remote)
    remotes.append("origin")
    remotes.extend(git(repo, "remote").splitlines())
    for remote in dict.fromkeys(remotes):
        url = optional_git(repo, "config", "--get", f"remote.{remote}.url")
        if url:
            return normalize_repository_url(url)
    raise RuntimeError("HRX source tree has no repository remote for provenance")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_exports(text: str, source: str) -> list[dict[str, object]]:
    exports: list[dict[str, object]] = []
    for match in KERNEL_RE.finditer(text):
        modifiers = match.group("modifiers")
        target_match = TARGET_MODIFIER_RE.search(modifiers)
        export_match = EXPORT_MODIFIER_RE.search(modifiers)
        workload = [item.groupdict() for item in ARG_RE.finditer(match.group("workload"))]
        launch = [item.groupdict() for item in ARG_RE.finditer(match.group("launch"))]
        bindings = [item["name"] for item in launch if item["type"] == "buffer"]
        exports.append(
            {
                "name": export_match.group("name") if export_match else match.group("symbol"),
                "symbol": match.group("symbol"),
                "target_symbol": target_match.group("symbol") if target_match else "",
                "source": source,
                "workload_parameters": workload,
                "launch_parameters": [item for item in launch if item["type"] != "buffer"],
                "bindings": bindings,
                "binding_access": [binding_access(match.group("symbol"), name) for name in bindings],
            }
        )
    return exports


def parse_amdgpu_targets(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in AMDGPU_TARGET_RE.finditer(text):
        symbol = match.group("symbol")
        selector = match.group("selector")
        if symbol in result and result[symbol] != selector:
            raise RuntimeError(
                f"AMDGPU target @{symbol} is declared as both {result[symbol]} and {selector}")
        result[symbol] = selector
    return result


def merge_amdgpu_targets(target_selectors: dict[str, str], additions: dict[str, str]) -> None:
    for symbol, selector in additions.items():
        if symbol in target_selectors and target_selectors[symbol] != selector:
            raise RuntimeError(
                f"AMDGPU target @{symbol} is declared as both {target_selectors[symbol]} and {selector}")
        target_selectors[symbol] = selector


def resolve_export_variants(exports: list[dict[str, object]], target_selectors: dict[str, str]) -> None:
    groups: dict[str, list[dict[str, object]]] = {}
    for item in exports:
        groups.setdefault(str(item["name"]), []).append(item)

    for name, variants in groups.items():
        selectors: set[str] = set()
        for item in variants:
            target_symbol = str(item.pop("target_symbol"))
            selector = ""
            if target_symbol:
                if target_symbol not in target_selectors:
                    raise RuntimeError(f"kernel export {name} references unknown target @{target_symbol}")
                selector = target_selectors[target_symbol]
                if selector.endswith("-generic"):
                    selector = ""
            if selector in selectors:
                label = selector or "default"
                raise RuntimeError(f"kernel export {name} repeats target variant {label}")
            selectors.add(selector)
            item["target_selector"] = selector


def construct(source_root: pathlib.Path, destination: pathlib.Path, expected_revision: str | None) -> None:
    revision = git(source_root, "rev-parse", "HEAD")
    if expected_revision and revision != expected_revision:
        raise RuntimeError(f"HRX revision {revision} does not match expected {expected_revision}")
    if git(source_root, "status", "--porcelain"):
        raise RuntimeError("refusing to mirror a dirty HRX source tree")

    source_directory = source_root / SOURCE_SUBDIR
    build_data = (source_directory / "BUILD.bazel").read_bytes()
    all_link_modules, all_plan_cases = parse_build_recipes(build_data.decode("utf-8"))

    modules_by_name = {str(item["name"]): item for item in all_link_modules}
    direct_plan_sources = {
        str(item["source"])
        for item in all_plan_cases
        if "source" in item
    }

    def module_files(name: str) -> list[str]:
        module = modules_by_name[name]
        result = list(module["srcs"])
        for library in module["libraries"]:
            if str(library).startswith(":"):
                result.extend(module_files(str(library)[1:]))
            else:
                result.append(str(library))
        return list(dict.fromkeys(result))

    def compile_recipe(source: str) -> dict[str, object]:
        direct = [str(item["name"]) for item in all_link_modules if source in item["srcs"]]
        indirect = [str(item["name"]) for item in all_link_modules if source in item["libraries"]]
        if not direct and (source in direct_plan_sources or not indirect):
            return {"mode": "direct", "primary_sources": [source], "library_sources": []}
        module_name = (direct or indirect)[0]
        module = modules_by_name[module_name]
        files = module_files(module_name)
        return {
            "mode": "archive",
            "link_module": module_name,
            "primary_sources": list(module["srcs"]),
            "library_sources": [item for item in files if item not in module["srcs"]],
        }
    file_rows: list[dict[str, object]] = []
    exports: list[dict[str, object]] = []
    target_selectors: dict[str, str] = {}
    upstream_aggregate = hashlib.sha256()
    for relative_text in CORPUS_FILES:
        relative = pathlib.Path(relative_text)
        source = source_directory / relative
        if not source.is_file():
            raise RuntimeError(f"missing required corpus source: {source}")
        data = source.read_bytes()
        digest = sha256(data)
        upstream_aggregate.update(relative_text.encode())
        upstream_aggregate.update(b"\0")
        upstream_aggregate.update(bytes.fromhex(digest))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        file_rows.append({"path": relative_text, "sha256": digest, "size": len(data)})
        source_text = data.decode("utf-8")
        exports.extend(parse_exports(source_text, relative_text))
        merge_amdgpu_targets(target_selectors, parse_amdgpu_targets(source_text))

    endpoint_source_directory = source_root / QWEN_ENDPOINT_SOURCE_SUBDIR
    for source_text_name, local_text_name in QWEN_ENDPOINT_FILES:
        source = endpoint_source_directory / source_text_name
        if not source.is_file():
            raise RuntimeError(f"missing required Qwen endpoint source: {source}")
        data = source.read_bytes()
        digest = sha256(data)
        provenance_path = f"{QWEN_ENDPOINT_SOURCE_SUBDIR.as_posix()}/{source_text_name}"
        upstream_aggregate.update(provenance_path.encode())
        upstream_aggregate.update(b"\0")
        upstream_aggregate.update(bytes.fromhex(digest))
        relative_text = f"../{local_text_name}"
        target = destination.parent / local_text_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        file_rows.append({
            "path": relative_text,
            "sha256": digest,
            "size": len(data),
            "upstream_path": provenance_path,
        })
        source_text = data.decode("utf-8")
        exports.extend(parse_exports(source_text, relative_text))
        merge_amdgpu_targets(target_selectors, parse_amdgpu_targets(source_text))

    owned_aggregate = hashlib.sha256()
    for filename in OWNED_FILES:
        source = OWNED_KERNEL_DIR / filename
        if not source.is_file():
            raise RuntimeError(f"missing required backend-owned kernel source: {source}")
        data = source.read_bytes()
        digest = sha256(data)
        relative_text = f"../{filename}"
        owned_aggregate.update(filename.encode())
        owned_aggregate.update(b"\0")
        owned_aggregate.update(bytes.fromhex(digest))
        file_rows.append({"path": relative_text, "sha256": digest, "size": len(data), "owner": "ggml-hrx"})
        source_text = data.decode("utf-8")
        exports.extend(parse_exports(source_text, relative_text))
        merge_amdgpu_targets(target_selectors, parse_amdgpu_targets(source_text))

    resolve_export_variants(exports, target_selectors)

    for item in exports:
        recipe = compile_recipe(str(item["source"]))
        item["compile_recipe"] = recipe
        item["compile_dependencies"] = list(recipe["library_sources"])

    required_modules: set[str] = set()

    def require_module(name: str) -> None:
        if name in required_modules:
            return
        if name not in modules_by_name:
            raise RuntimeError(f"selected kernel recipe references unknown link module {name}")
        required_modules.add(name)
        for library in modules_by_name[name]["libraries"]:
            if str(library).startswith(":"):
                require_module(str(library)[1:])

    required_files = set(CORPUS_FILES)
    for item in exports:
        if str(item["source"]).startswith("../"):
            continue
        recipe = item["compile_recipe"]
        link_module = str(recipe.get("link_module", ""))
        if link_module:
            require_module(link_module)
        required_files.update(str(path) for path in recipe["primary_sources"])
        required_files.update(str(path) for path in recipe["library_sources"])

    mirrored_files = set(CORPUS_FILES)
    for relative_text in sorted(required_files - mirrored_files):
        relative = pathlib.Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"kernel recipe escapes the source corpus: {relative_text}")
        source = source_directory / relative
        if not source.is_file():
            raise RuntimeError(f"missing required kernel dependency: {source}")
        data = source.read_bytes()
        digest = sha256(data)
        upstream_aggregate.update(relative_text.encode())
        upstream_aggregate.update(b"\0")
        upstream_aggregate.update(bytes.fromhex(digest))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        file_rows.append({"path": relative_text, "sha256": digest, "size": len(data)})

    link_modules = [
        module for module in all_link_modules
        if str(module["name"]) in required_modules
    ]
    plan_cases = [
        case for case in all_plan_cases
        if (str(case.get("link_module", "")) in required_modules or
            str(case.get("source", "")) in required_files)
    ]
    plan_cases.extend([
        {
            "name": "owned_token_embedding_decode_plan_test",
            "args": ["$(location ../qwen_owned/token_embedding_q4k.loom)",
                     "--benchmark=@qwen_token_embedding_q4k_decode", "--dry-run",
                     "--output-format=jsonl", "--sample-compilation=per_sample"],
            "source": "../qwen_owned/token_embedding_q4k.loom",
        },
        {
            "name": "owned_token_embedding_prefill_plan_test",
            "args": ["$(location ../qwen_owned/token_embedding_q4k.loom)",
                     "--benchmark=@qwen_token_embedding_q4k_prefill_512", "--dry-run",
                     "--output-format=jsonl", "--sample-compilation=per_sample"],
            "source": "../qwen_owned/token_embedding_q4k.loom",
        },
        {
            "name": "owned_attention_context_base_capture_plan_test",
            "args": ["$(location ../qwen_owned/attention_state_initialize.loom)",
                     "--benchmark=@qwen_attention_context_base_capture_benchmark", "--dry-run",
                     "--output-format=jsonl", "--sample-compilation=per_sample"],
            "source": "../qwen_owned/attention_state_initialize.loom",
            "owner": "ggml-hrx",
        },
        {
            "name": "owned_attention_decode_state_initialize_plan_test",
            "args": ["$(location ../qwen_owned/attention_state_initialize.loom)",
                     "--benchmark=@qwen_attention_decode_state_initialize_benchmark", "--dry-run",
                     "--output-format=jsonl", "--sample-compilation=per_sample"],
            "source": "../qwen_owned/attention_state_initialize.loom",
            "owner": "ggml-hrx",
        },
        {
            "name": "owned_attention_metadata_prefill_plan_test",
            "args": ["$(location ../qwen_owned/attention_metadata.loom)",
                     "--benchmark=@qwen_attention_metadata_prefill_512", "--dry-run",
                     "--output-format=jsonl", "--sample-compilation=per_sample"],
            "source": "../qwen_owned/attention_metadata.loom",
        },
        {
            "name": "owned_gather_add_plan_test",
            "args": ["$(location ../hrx_owned/gather_add_f32.loom)",
                     "--benchmark=@ggml_gather_add_noncontiguous", "--dry-run",
                     "--output-format=jsonl", "--sample-compilation=per_sample"],
            "source": "../hrx_owned/gather_add_f32.loom",
            "owner": "ggml-hrx",
        },
    ])

    upstream_digest = upstream_aggregate.hexdigest()
    owned_digest = owned_aggregate.hexdigest()
    combined_aggregate = hashlib.sha256()
    combined_aggregate.update(bytes.fromhex(upstream_digest))
    combined_aggregate.update(bytes.fromhex(owned_digest))

    manifest = {
        "schema": "ggml-hrx-qwen-kernel-corpus-v2",
        "upstream_repository": upstream_repository(source_root),
        "upstream_revision": revision,
        "source_subdirectory": SOURCE_SUBDIR.as_posix(),
        "qwen_endpoint_source_subdirectory": QWEN_ENDPOINT_SOURCE_SUBDIR.as_posix(),
        "corpus_sha256": combined_aggregate.hexdigest(),
        "upstream_corpus_sha256": upstream_digest,
        "owned_corpus_sha256": owned_digest,
        "build_bazel_sha256": sha256(build_data),
        "files": file_rows,
        "exports": sorted(exports, key=lambda item: (str(item["symbol"]), str(item["source"]))),
        "link_modules": link_modules,
        "plan_cases": plan_cases,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def trees_equal(lhs: pathlib.Path, rhs: pathlib.Path) -> bool:
    lhs_files = sorted(path.relative_to(lhs) for path in lhs.rglob("*") if path.is_file())
    rhs_files = sorted(path.relative_to(rhs) for path in rhs.rglob("*") if path.is_file())
    return lhs_files == rhs_files and all((lhs / path).read_bytes() == (rhs / path).read_bytes() for path in lhs_files)


def endpoint_files_equal(generated_qwen_moe: pathlib.Path, destination_qwen_moe: pathlib.Path) -> bool:
    generated_kernel_root = generated_qwen_moe.parent
    destination_kernel_root = destination_qwen_moe.parent
    for _, local_text_name in QWEN_ENDPOINT_FILES:
        relative = pathlib.Path(local_text_name)
        generated = generated_kernel_root / relative
        destination = destination_kernel_root / relative
        if not destination.is_file() or generated.read_bytes() != destination.read_bytes():
            return False
    return True


def copy_endpoint_files(generated_qwen_moe: pathlib.Path, destination_qwen_moe: pathlib.Path) -> None:
    generated_kernel_root = generated_qwen_moe.parent
    destination_kernel_root = destination_qwen_moe.parent
    for _, local_text_name in QWEN_ENDPOINT_FILES:
        relative = pathlib.Path(local_text_name)
        source = generated_kernel_root / relative
        target = destination_kernel_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrx-source", type=pathlib.Path, required=True)
    parser.add_argument("--destination", type=pathlib.Path, required=True)
    parser.add_argument("--expect-revision")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="hrx-qwen-corpus-") as temporary:
        generated = pathlib.Path(temporary) / args.destination.name
        construct(args.hrx_source.resolve(), generated, args.expect_revision)
        if args.check:
            if (not args.destination.is_dir() or not trees_equal(generated, args.destination) or
                    not endpoint_files_equal(generated, args.destination)):
                print("mirrored Qwen kernel corpus is stale", file=sys.stderr)
                return 1
            return 0
        args.destination.mkdir(parents=True, exist_ok=True)
        for child in args.destination.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        shutil.copytree(generated, args.destination, dirs_exist_ok=True)
        copy_endpoint_files(generated, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
