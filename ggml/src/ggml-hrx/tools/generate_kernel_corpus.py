#!/usr/bin/env python3

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List, Tuple


DEFAULT_KERNEL_FAMILY = "qwen3_moe"

SOURCE_ARRAY_TEMPLATE = """static const unsigned char {symbol}[] = {{
{bytes}
}};
static constexpr size_t {symbol}Size = {size};
"""

DEPENDENCY_TABLE_TEMPLATE = """static const KernelSourceSpan {dependency_table}[] = {{
{dependencies}
}};
"""

SOURCE_RECORD_TEMPLATE = """static const KernelSource {record} = {{
    {{ reinterpret_cast<const char *>({source_symbol}), {source_symbol}Size, {source_format} }},
    {dependency_table},
    {dependency_count},
}};
"""

CORPUS_ARRAY_TEMPLATE = """static {type} {symbol}[] = {{
{values}
}};
"""

KERNEL_RECORD_TEMPLATE = """    {{
        {family},
        {name},
        kernel_catalog_id({family}, {name}),
        {source},
        {dependencies},
        {symbol},
        "amdgpu",
        {target_selector},
        {{ nullptr, 0 }},
        {scalar_parameters},
        {bindings},
        {source_digest},
        {workload_parameters},
        {launch_parameters},
        {{
            {compile_mode},
            {link_module},
            {primary_sources},
            {library_sources},
        }},
    }},"""

SOURCE_DATA_TEMPLATE = """{source_arrays}
{dependency_tables}
{source_records}
static const KernelSourceRecordEntry kKernelSourceRecords[] = {{
{lookup_entries}
}};
"""

CORPUS_DATA_TEMPLATE = """{kernel_arrays}
static const KernelDefinition kQwenKernelDefinitions[] = {{
{kernel_records}
}};

static const KernelCorpus kQwenKernelCorpus = {{
    "ggml-hrx-kernel-corpus-v2",
    {upstream_revision},
    {corpus_digest},
    {recipe_digest},
    {plan_case_count},
    {{ kQwenKernelDefinitions, {kernel_count} }},
}};
"""

CATALOG_DATA_TEMPLATE = """struct KernelCatalogEntry {{
    const char * family;
    const char * name;
}};

static constexpr KernelCatalogEntry kKernelCatalogEntries[] = {{
{kernel_entries}
}};

constexpr bool kernel_catalog_entry_exists(const char * family, const char * name) {{
    for (const KernelCatalogEntry & known : kKernelCatalogEntries) {{
        if (kernel_catalog_name_equal(family, known.family) && kernel_catalog_name_equal(name, known.name)) {{
            return true;
        }}
    }}
    return false;
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embedded kernel corpus data includes.")
    parser.add_argument("--source-output", type=pathlib.Path, required=True)
    parser.add_argument("--corpus-output", type=pathlib.Path, required=True)
    parser.add_argument("--catalog-output", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--corpus-dir", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--source-format", choices=("text", "binary"), default="text")
    parser.add_argument("--loom-link", type=pathlib.Path)
    parser.add_argument("--loom-format", type=pathlib.Path)
    parser.add_argument("--depfile", type=pathlib.Path)
    return parser.parse_args()


def read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text()
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc


def read_bytes(path: pathlib.Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc


def run_tool(command: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def require_tool(command: List[str]) -> None:
    result = run_tool(command)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}")


def convert_source_to_bytecode(source_path: pathlib.Path, loom_link: pathlib.Path, loom_format: pathlib.Path) -> bytes:
    with tempfile.TemporaryDirectory(prefix="ggml-hrx-loom-") as temp_dir_name:
        temp_dir = pathlib.Path(temp_dir_name)
        stripped = temp_dir / "stripped.loom"
        bytecode = temp_dir / "stripped.loombc"
        require_tool([
            str(loom_link),
            "--verify=false",
            "--mode=merge",
            "--strip-check",
            "--to=text",
            f"--output={stripped}",
            str(source_path),
        ])
        format_result = run_tool([
            str(loom_format),
            "--from=text",
            "--to=bc",
            f"--output={bytecode}",
            str(stripped),
        ])
        if format_result.returncode:
            require_tool([
                str(loom_link),
                "--verify=false",
                "--mode=merge",
                "--strip-check",
                "--to=bc",
                f"--output={bytecode}",
                str(source_path),
            ])
        return read_bytes(bytecode)


def sanitize_symbol(path: str, index: int) -> str:
    stem = re.sub(r"[^0-9A-Za-z_]", "_", path)
    if not stem or stem[0].isdigit():
        stem = f"_{stem}"
    return f"kernel_source_{index}_{stem}"


def format_byte_array(data: bytes) -> str:
    if not data:
        return ""
    lines = []
    for offset in range(0, len(data), 16):
        chunk = data[offset : offset + 16]
        lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")
    return "\n".join(lines) + "\n"


def escape_cpp_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def cpp_string(text: str) -> str:
    return f"\"{escape_cpp_string(text)}\""


def resource_access_value(access: str) -> str:
    if access == "read":
        return "ResourceAccess::Read"
    if access == "write":
        return "ResourceAccess::Write"
    if access == "read_write":
        return "ResourceAccess::ReadWrite"
    raise RuntimeError(f"invalid kernel binding access metadata: {access}")


def span_initializer(symbol: str, count: int) -> str:
    if count == 0:
        return "{ nullptr, 0 }"
    return "{ " + symbol + ", " + str(count) + " }"


def typed_array(symbol: str, value_type: str, values: List[str]) -> Tuple[str, str]:
    if not values:
        return "", "{ nullptr, 0 }"
    array = CORPUS_ARRAY_TEMPLATE.format(
        type=value_type,
        symbol=symbol,
        values="\n".join("    " + value + "," for value in values),
    )
    return array, span_initializer(symbol, len(values))


def string_array(symbol: str, items: Iterable[str]) -> Tuple[str, str]:
    return typed_array(symbol, "const char * const", [cpp_string(item) for item in items])


def source_ref_array(symbol: str, items: Iterable[str], source_records: Dict[str, str]) -> Tuple[str, str]:
    values = [
        "{ " + cpp_string(item) + ", &" + source_records[item] + " }"
        for item in items
    ]
    return typed_array(symbol, "const KernelSourceRef", values)


def scalar_array(symbol: str, items: Iterable[dict]) -> Tuple[str, str]:
    values = [
        "{ " + cpp_string(item["name"]) + ", " + cpp_string(item["type"]) + " }"
        for item in items
    ]
    return typed_array(symbol, "const KernelScalarDefinition", values)


def binding_array(symbol: str, names: List[str], access: List[str]) -> Tuple[str, str]:
    if len(names) != len(access):
        raise RuntimeError("kernel binding access metadata has the wrong arity")
    values = [
        "{ " + cpp_string(name) + ", " + resource_access_value(access_value) + " }"
        for name, access_value in zip(names, access)
    ]
    return typed_array(symbol, "const KernelBindingDefinition", values)


def scalar_parameter_names(workload_parameters: List[dict], launch_parameters: List[dict]) -> List[str]:
    names: List[str] = []
    for parameter in [*workload_parameters, *launch_parameters]:
        name = parameter["name"]
        if name not in names:
            names.append(name)
    return names


def depfile_escape(path: pathlib.Path) -> str:
    return str(path).replace("\\", "\\\\").replace(" ", "\\ ")


def collect_sources(manifest: dict) -> Tuple[List[str], Dict[str, List[str]]]:
    source_dependencies: Dict[str, List[str]] = {}
    all_sources = set()

    for export in manifest.get("exports", []):
        recipe = export.get("compile_recipe", {})
        primary_sources = recipe.get("primary_sources", [])
        library_sources = recipe.get("library_sources", [])
        if len(primary_sources) != 1:
            raise RuntimeError("expected each export compile_recipe to have exactly one primary source")

        primary = primary_sources[0]
        dependencies = list(library_sources)
        previous = source_dependencies.get(primary)
        if previous is not None and previous != dependencies:
            raise RuntimeError(f"conflicting dependency list for {primary}")
        source_dependencies[primary] = dependencies
        all_sources.add(primary)
        all_sources.update(dependencies)

    return sorted(all_sources), source_dependencies


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def complete_manifest_source_metadata(manifest: dict, source_paths: Dict[str, pathlib.Path]) -> dict:
    completed = dict(manifest)
    files = []
    for file in manifest.get("files", []):
        path = str(file["path"])
        file_record = {key: value for key, value in file.items() if key not in ("owner", "sha256", "size")}
        data = read_bytes(source_paths[path])
        file_record["sha256"] = sha256(data)
        files.append(file_record)
    completed["files"] = files
    corpus_digest_input = {
        "files": [{"path": file["path"], "sha256": file["sha256"]} for file in files],
        "exports": [
            {
                "family": export.get("family", DEFAULT_KERNEL_FAMILY),
                "name": export["name"],
                "source": export["source"],
                "symbol": export["symbol"],
                "target_selector": export.get("target_selector", ""),
                "compile_recipe": export["compile_recipe"],
            }
            for export in completed.get("exports", [])
        ],
    }
    recipe_digest_input = {
        "link_modules": completed.get("link_modules", []),
        "plan_cases": completed.get("plan_cases", []),
    }
    completed["corpus_sha256"] = json_digest(corpus_digest_input)
    completed["build_bazel_sha256"] = json_digest(recipe_digest_input)
    return completed


def merge_unique_text(values: Iterable[str]) -> str:
    unique = list(dict.fromkeys(value for value in values if value))
    return ";".join(unique)


def require_unique_export_variants(exports: List[dict]) -> None:
    variants: Dict[Tuple[str, str, str], dict] = {}
    for export in exports:
        key = (
            str(export.get("family", DEFAULT_KERNEL_FAMILY)),
            str(export["name"]),
            str(export.get("target_selector", "")),
        )
        previous = variants.get(key)
        if previous is not None:
            family, name, target = key
            label = target or "default"
            raise RuntimeError(f"kernel corpus repeats target variant {family}:{name}@{label}")
        variants[key] = export


def merge_manifests(manifest_paths: List[pathlib.Path], corpus_dirs: List[pathlib.Path]) -> Tuple[dict, Dict[str, pathlib.Path]]:
    if len(manifest_paths) != len(corpus_dirs):
        raise RuntimeError("--manifest and --corpus-dir must be provided the same number of times")
    if not manifest_paths:
        raise RuntimeError("at least one --manifest is required")

    loaded = [(manifest_path, corpus_dir, json.loads(read_text(manifest_path)))
              for manifest_path, corpus_dir in zip(manifest_paths, corpus_dirs)]
    if len(loaded) == 1:
        manifest_path, corpus_dir, manifest = loaded[0]
        source_paths = {file["path"]: corpus_dir / file["path"] for file in manifest.get("files", [])}
        return complete_manifest_source_metadata(manifest, source_paths), source_paths

    files = []
    file_digests: Dict[str, str] = {}
    source_paths: Dict[str, pathlib.Path] = {}
    exports = []
    link_modules = []
    plan_cases = []
    metadata = []

    for manifest_path, corpus_dir, manifest in loaded:
        manifest_source_paths = {file["path"]: corpus_dir / file["path"] for file in manifest.get("files", [])}
        manifest = complete_manifest_source_metadata(manifest, manifest_source_paths)
        metadata.append({
            "manifest": str(manifest_path),
            "upstream_revision": manifest.get("upstream_revision", ""),
            "corpus_sha256": manifest.get("corpus_sha256", ""),
            "build_bazel_sha256": manifest.get("build_bazel_sha256", ""),
        })
        for file in manifest.get("files", []):
            path = str(file["path"])
            digest = str(file["sha256"])
            previous = file_digests.get(path)
            if previous is not None:
                if previous != digest:
                    raise RuntimeError(f"manifest file digest conflict for {path}: {previous} vs {digest}")
                continue
            file_digests[path] = digest
            source_paths[path] = corpus_dir / path
            files.append(dict(file))
        exports.extend(dict(export) for export in manifest.get("exports", []))
        link_modules.extend(dict(module) for module in manifest.get("link_modules", []))
        plan_cases.extend(dict(case) for case in manifest.get("plan_cases", []))

    require_unique_export_variants(exports)

    corpus_digest_input = {
        "files": [{"path": file["path"], "sha256": file["sha256"]} for file in files],
        "exports": [
            {
                "family": export.get("family", DEFAULT_KERNEL_FAMILY),
                "name": export["name"],
                "source": export["source"],
                "symbol": export["symbol"],
                "target_selector": export.get("target_selector", ""),
                "compile_recipe": export["compile_recipe"],
            }
            for export in exports
        ],
    }
    recipe_digest_input = {
        "manifests": metadata,
        "link_modules": link_modules,
        "plan_cases": plan_cases,
    }

    return {
        "schema": "ggml-hrx-kernel-corpus-v2",
        "upstream_revision": merge_unique_text(str(manifest.get("upstream_revision", "")) for _, _, manifest in loaded),
        "corpus_sha256": json_digest(corpus_digest_input),
        "build_bazel_sha256": json_digest(recipe_digest_input),
        "files": files,
        "exports": exports,
        "link_modules": link_modules,
        "plan_cases": plan_cases,
    }, source_paths


def kernel_catalog_id(family: str, name: str) -> int:
    hash_value = 1469598103934665603
    for byte in family.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    hash_value ^= 0
    hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    for byte in name.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return hash_value


def generate_corpus_records(manifest: dict, source_records: Dict[str, str], source_digests: Dict[str, str]) -> Tuple[str, str, int]:
    arrays = []
    records = []
    exports = manifest.get("exports", [])
    for index, export in enumerate(exports):
        recipe = export["compile_recipe"]
        dependencies = list(export["compile_dependencies"])
        library_sources = list(recipe["library_sources"])
        if dependencies != library_sources:
            raise RuntimeError("legacy dependency closure disagrees with compile recipe")
        workload_parameters = list(export["workload_parameters"])
        launch_parameters = list(export["launch_parameters"])

        dependencies_array, dependencies_span = string_array(f"kKernelDependencies{index}", dependencies)
        scalar_array_text, scalar_span = string_array(
            f"kKernelScalarParameters{index}",
            scalar_parameter_names(workload_parameters, launch_parameters),
        )
        bindings_array, bindings_span = binding_array(
            f"kKernelBindings{index}",
            list(export["bindings"]),
            list(export.get("binding_access", [])),
        )
        workload_array, workload_span = scalar_array(f"kKernelWorkloadParameters{index}", workload_parameters)
        launch_array, launch_span = scalar_array(f"kKernelLaunchParameters{index}", launch_parameters)
        primary_array, primary_span = source_ref_array(f"kKernelPrimarySources{index}", recipe["primary_sources"], source_records)
        library_array, library_span = source_ref_array(f"kKernelLibrarySources{index}", library_sources, source_records)
        arrays.extend(
            item for item in [
                dependencies_array,
                scalar_array_text,
                bindings_array,
                workload_array,
                launch_array,
                primary_array,
                library_array,
            ] if item
        )
        records.append(
            KERNEL_RECORD_TEMPLATE.format(
                family=cpp_string(export.get("family", DEFAULT_KERNEL_FAMILY)),
                name=cpp_string(export["name"]),
                symbol=cpp_string(export["symbol"]),
                target_selector=cpp_string(export.get("target_selector", "")),
                source=cpp_string(export["source"]),
                dependencies=dependencies_span,
                source_digest=cpp_string(source_digests[export["source"]]),
                scalar_parameters=scalar_span,
                bindings=bindings_span,
                workload_parameters=workload_span,
                launch_parameters=launch_span,
                compile_mode=cpp_string(recipe["mode"]),
                link_module=cpp_string(recipe.get("link_module", "")),
                primary_sources=primary_span,
                library_sources=library_span,
            )
        )
    return "\n".join(arrays), "\n".join(records), len(exports)


def generate_catalog_verifier(manifest: dict) -> str:
    kernel_entries = sorted(set(
        (export.get("family", DEFAULT_KERNEL_FAMILY), export["name"])
        for export in manifest.get("exports", [])
    ))
    ids: Dict[int, Tuple[str, str]] = {}
    for family, kernel_name in kernel_entries:
        catalog_id = kernel_catalog_id(family, kernel_name)
        if catalog_id in ids:
            previous_family, previous_name = ids[catalog_id]
            raise RuntimeError(
                "kernel catalog id collision: "
                f"{previous_family}/{previous_name} and {family}/{kernel_name}")
        ids[catalog_id] = (family, kernel_name)
    return CATALOG_DATA_TEMPLATE.format(
        kernel_entries="\n".join(
            "    { " + cpp_string(family) + ", " + cpp_string(kernel_name) + " },"
            for family, kernel_name in kernel_entries
        ),
    )


def generate_includes(args: argparse.Namespace, manifest: dict, source_paths: Dict[str, pathlib.Path]) -> Tuple[str, str, str, List[pathlib.Path], int]:
    if args.source_format == "binary" and (args.loom_link is None or args.loom_format is None):
        raise RuntimeError("binary source format requires --loom-link and --loom-format")

    sources, source_dependencies = collect_sources(manifest)
    source_digests = {str(file["path"]): str(file["sha256"]) for file in manifest.get("files", [])}
    source_bytes: Dict[str, bytes] = {}
    source_format = "KERNEL_SOURCE_FORMAT_BINARY" if args.source_format == "binary" else "KERNEL_SOURCE_FORMAT_TEXT"
    input_files: List[pathlib.Path] = []

    for export in manifest.get("exports", []):
        recipe = export.get("compile_recipe", {})
        for primary in recipe.get("primary_sources", []):
            if primary not in sources:
                raise RuntimeError(f"export primary source is not embedded: {primary}")

    for source in sources:
        if source not in source_digests:
            raise RuntimeError(f"embedded source is missing from manifest file table: {source}")
        path = source_paths[source]
        input_files.append(path)
        data = read_bytes(path)
        source_digests[source] = sha256(data)
        if args.source_format == "binary":
            data = convert_source_to_bytecode(path, args.loom_link, args.loom_format)
        source_bytes[source] = data

    source_symbols: Dict[str, str] = {}
    source_arrays = []
    for index, source in enumerate(sources):
        symbol = sanitize_symbol(source, index)
        source_symbols[source] = symbol
        data = source_bytes[source]
        source_arrays.append(
            SOURCE_ARRAY_TEMPLATE.format(
                symbol=symbol,
                bytes=format_byte_array(data),
                size=len(data),
            )
        )

    dependency_tables = []
    source_record_definitions = []
    source_record_symbols = {}
    lookup_entries = []
    for index, source in enumerate(sources):
        dependencies = source_dependencies.get(source, [])
        record = f"kernel_source_record_{index}"
        source_record_symbols[source] = record
        if dependencies:
            dependency_table = f"kernel_source_dependencies_{index}"
            entries = []
            for dependency in dependencies:
                symbol = source_symbols[dependency]
                entries.append(
                    f"    {{ reinterpret_cast<const char *>({symbol}), {symbol}Size, {source_format} }},"
                )
            dependency_tables.append(
                DEPENDENCY_TABLE_TEMPLATE.format(
                    dependency_table=dependency_table,
                    dependencies="\n".join(entries),
                )
            )
        else:
            dependency_table = "nullptr"

        source_record_definitions.append(
            SOURCE_RECORD_TEMPLATE.format(
                record=record,
                source_symbol=source_symbols[source],
                source_format=source_format,
                dependency_table=dependency_table,
                dependency_count=len(dependencies),
            )
        )
        lookup_entries.append(f"    {{ {cpp_string(source)}, &{record} }},")

    kernel_arrays, kernel_records, kernel_count = generate_corpus_records(manifest, source_record_symbols, source_digests)
    return (
        SOURCE_DATA_TEMPLATE.format(
            source_arrays="\n".join(source_arrays),
            dependency_tables="\n".join(dependency_tables),
            source_records="\n".join(source_record_definitions),
            lookup_entries="\n".join(lookup_entries),
        ),
        CORPUS_DATA_TEMPLATE.format(
            kernel_arrays=kernel_arrays,
            kernel_records=kernel_records,
            upstream_revision=cpp_string(manifest["upstream_revision"]),
            corpus_digest=cpp_string(manifest["corpus_sha256"]),
            recipe_digest=cpp_string(manifest["build_bazel_sha256"]),
            plan_case_count=len(manifest["plan_cases"]),
            kernel_count=kernel_count,
        ),
        generate_catalog_verifier(manifest),
        input_files,
        sum(len(data) for data in source_bytes.values()),
    )


def write_depfile(path: pathlib.Path, outputs: List[pathlib.Path], inputs: List[pathlib.Path]) -> None:
    targets = " ".join(depfile_escape(output_path) for output_path in outputs)
    entries = [depfile_escape(input_path) for input_path in inputs]
    path.write_text(f"{targets}: {' '.join(entries)}\n")


def main() -> int:
    args = parse_args()
    try:
        manifest, source_paths = merge_manifests(args.manifest, args.corpus_dir)
        source_include, corpus_include, catalog_include, input_files, byte_count = generate_includes(
            args, manifest, source_paths)
        args.source_output.parent.mkdir(parents=True, exist_ok=True)
        args.corpus_output.parent.mkdir(parents=True, exist_ok=True)
        args.catalog_output.parent.mkdir(parents=True, exist_ok=True)
        args.source_output.write_text(source_include)
        args.corpus_output.write_text(corpus_include)
        args.catalog_output.write_text(catalog_include)
        if args.depfile is not None:
            args.depfile.parent.mkdir(parents=True, exist_ok=True)
            write_depfile(args.depfile, [args.source_output, args.corpus_output, args.catalog_output],
                          [*args.manifest, *input_files])
    except Exception as exc:
        print(f"generate_kernel_corpus.py: {exc}", file=sys.stderr)
        return 1

    print(
        f"embedded kernel corpus: source_files={len(input_files)} source_bytes={byte_count} "
        f"source_format={args.source_format}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
