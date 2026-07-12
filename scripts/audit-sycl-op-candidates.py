#!/usr/bin/env python3

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


CANDIDATES: list[tuple[str, tuple[str, ...], str]] = [
    ("COL2IM_1D", ("ggml/src/ggml-sycl/col2im-1d.cpp", "ggml/src/ggml-sycl/col2im-1d.hpp"), "COL2IM_1D"),
    ("CONV_2D", ("ggml/src/ggml-sycl/conv2d.cpp", "ggml/src/ggml-sycl/conv2d.hpp"), "CONV_2D_DIRECT_IMPL"),
    ("CONV_2D_DW", ("ggml/src/ggml-sycl/conv2d-dw.cpp", "ggml/src/ggml-sycl/conv2d-dw.hpp"), "CONV_2D_DW"),
    ("CONV_TRANSPOSE_2D", ("ggml/src/ggml-sycl/conv2d-transpose.cpp", "ggml/src/ggml-sycl/conv2d-transpose.hpp"), "CONV_TRANSPOSE_2D"),
    ("CONV_3D", ("ggml/src/ggml-sycl/conv3d.cpp", "ggml/src/ggml-sycl/conv3d.hpp"), "CONV_3D"),
    ("POOL_1D", ("ggml/src/ggml-sycl/pool.cpp", "ggml/src/ggml-sycl/pool.hpp"), "POOL_1D"),
    ("CROSS_ENTROPY_LOSS", ("ggml/src/ggml-sycl/cross_entropy_loss.cpp", "ggml/src/ggml-sycl/cross_entropy_loss.hpp"), "CROSS_ENTROPY_LOSS"),
    ("CROSS_ENTROPY_LOSS_BACK", ("ggml/src/ggml-sycl/cross_entropy_loss.cpp", "ggml/src/ggml-sycl/cross_entropy_loss.hpp"), "CROSS_ENTROPY_LOSS_BACK"),
]


class AuditError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit graph ops against candidate SYCL op ports."
    )
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_ggml_ops(repo_root: Path) -> dict[int, str]:
    ggml_h = repo_root / "ggml" / "include" / "ggml.h"
    if not ggml_h.is_file():
        raise AuditError(f"ggml.h not found: {ggml_h}")

    text = ggml_h.read_text(encoding="utf-8")
    match = re.search(r"enum\s+ggml_op\s*\{(?P<body>.*?)\n\s*\};", text, re.S)
    if not match:
        raise AuditError(f"could not parse enum ggml_op in {ggml_h}")

    ops: dict[int, str] = {}
    value = 0
    for raw_line in match.group("body").splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        item = line.rstrip(",").strip()
        if not item.startswith("GGML_OP_"):
            continue
        if "=" in item:
            name, raw_value = (part.strip() for part in item.split("=", 1))
            try:
                value = int(raw_value, 0)
            except ValueError as exc:
                raise AuditError(f"unsupported enum value for {name}: {raw_value}") from exc
        else:
            name = item
        ops[value] = name
        value += 1

    if "GGML_OP_COUNT" not in ops.values():
        raise AuditError(f"enum ggml_op parse did not find GGML_OP_COUNT in {ggml_h}")
    return ops


def parse_graph_ops(graph: Path, enum_ops: dict[int, str]) -> Counter[str]:
    if not graph.is_file():
        raise AuditError(f"graph file not found: {graph}")

    counts: Counter[str] = Counter()
    valid_rows = 0
    for line_number, raw_line in enumerate(graph.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        first = line.split(maxsplit=1)[0]
        try:
            op_id = int(first, 0)
        except ValueError as exc:
            raise AuditError(f"invalid graph op id at {graph}:{line_number}: {first}") from exc
        if op_id not in enum_ops:
            raise AuditError(f"graph op id {op_id} at {graph}:{line_number} is not in enum ggml_op")
        op_name = enum_ops[op_id]
        if op_name == "GGML_OP_COUNT":
            raise AuditError(f"graph op id {op_id} at {graph}:{line_number} maps to GGML_OP_COUNT")
        counts[op_name.removeprefix("GGML_OP_")] += 1
        valid_rows += 1

    if valid_rows == 0:
        raise AuditError(f"no valid graph rows read from {graph}")
    return counts


def case_count(repo_root: Path, op: str) -> int:
    sycl_cpp = repo_root / "ggml" / "src" / "ggml-sycl" / "ggml-sycl.cpp"
    if not sycl_cpp.is_file():
        raise AuditError(f"ggml-sycl.cpp not found: {sycl_cpp}")
    text = sycl_cpp.read_text(encoding="utf-8")
    return text.count(f"case GGML_OP_{op}:")


def target_status(repo_root: Path, op: str, files: tuple[str, ...]) -> tuple[str, bool]:
    missing = [path for path in files if not (repo_root / path).is_file()]
    cases = case_count(repo_root, op)
    file_status = "files=present" if not missing else "missing=" + ", ".join(missing)
    complete = not missing and cases >= 2
    return f"{file_status}; case_count={cases}", complete


def markdown_table(repo_root: Path, graph: Path, counts: Counter[str]) -> str:
    rows = [
        "# SYCL op candidate audit",
        "",
        f"- Graph: `{graph}`",
        f"- Repo root: `{repo_root}`",
        "",
        "| Op | Present in graph | Node count | Action | Target status | Source files to port if present | Verification filter |",
        "|---|---:|---:|---|---|---|---|",
    ]

    for op, files, verify_filter in CANDIDATES:
        node_count = counts[op]
        status, complete = target_status(repo_root, op, files)
        if node_count == 0:
            action = "SKIP"
        elif complete:
            action = "VERIFY_EXISTING"
        else:
            action = "PORT"
        rows.append(
            f"| {op} | {'YES' if node_count else 'NO'} | {node_count} | {action} | {status} | {', '.join(files)} | {verify_filter} |"
        )

    rows.append("")
    return "\n".join(rows)


def main() -> int:
    args = parse_args()
    try:
        repo_root = args.repo_root.resolve()
        graph = args.graph.resolve()
        enum_ops = parse_ggml_ops(repo_root)
        counts = parse_graph_ops(graph, enum_ops)
        output = markdown_table(repo_root, graph, counts)
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
