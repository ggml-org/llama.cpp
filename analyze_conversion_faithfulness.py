#!/usr/bin/env python3
"""
Analyze faithfulness of the split from monolithic convert_hf_to_gguf.py
to the modular conversion/ package.

Usage:
    # Save a copy of original convert_hf_to_gguf.py in reference/ before running.
    python3 analyze_conversion_faithfulness.py

Compares each class in reference/convert_hf_to_gguf.py with its counterpart
in conversion/*.py, reporting missing classes, differing registrations,
and code differences (ignoring whitespace).
"""

import ast
import sys
from pathlib import Path
import difflib


class ClassInfo:
    def __init__(self, name: str, decorators: list[str], source: str, file_path: str):
        self.name = name
        self.decorators = decorators
        self.source = source
        self.file_path = file_path
        self.registrations = self._extract_registrations(decorators)

    @staticmethod
    def _extract_registrations(decorators: list[str]) -> set[str]:
        regs: set[str] = set()
        for dec in decorators:
            if "ModelBase.register" in dec:
                regs.add(dec.strip())
        return regs


def extract_class_info(file_path: Path) -> dict[str, ClassInfo]:
    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {file_path}: {e}") # noqa: NP100
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}") # noqa: NP100
        return {}

    classes: dict[str, ClassInfo] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            decorators = [ast.unparse(d) for d in node.decorator_list]
            class_source = ast.get_source_segment(source, node)
            if class_source is None:
                lines = source.split("\n")
                class_source = "\n".join(lines[node.lineno - 1:node.end_lineno])
            classes[node.name] = ClassInfo(
                name=node.name,
                decorators=decorators,
                source=class_source,
                file_path=str(file_path),
            )
    return classes


def normalize_source(source: str, strip_blank: bool = True) -> str:
    lines = [line.rstrip() for line in source.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    if strip_blank:
        lines = [x for x in lines if x.strip()]
    return "\n".join(lines)


def run() -> int:
    script_dir = Path(__file__).parent
    reference_file = script_dir / "reference" / "convert_hf_to_gguf.py"
    conversion_dir = script_dir / "conversion"

    if not reference_file.exists():
        print(f"Error: reference file not found: {reference_file}") # noqa: NP100
        print("Place the original monolithic convert_hf_to_gguf.py at that path to compare.") # noqa: NP100
        return 1
    if not conversion_dir.exists():
        print(f"Error: conversion directory not found: {conversion_dir}")# noqa: NP100
        return 1

    reference = extract_class_info(reference_file)

    converted: dict[str, ClassInfo] = {}
    for py in conversion_dir.glob("*.py"):
        if py.name.startswith("_"):
            continue
        for name, info in extract_class_info(py).items():
            if name in converted:
                print(f"Warning: duplicate class {name} in {py} and {converted[name].file_path}") # noqa: NP100
            converted[name] = info

    base_classes = {"ModelBase", "TextModel", "MmprojModel", "LazyTorchTensor",
                    "SentencePieceTokenTypes", "ModelType"}

    missing = []
    code_mismatch = []
    reg_mismatch = []
    perfect = []

    for name, ref in reference.items():
        if name in base_classes:
            continue
        if name not in converted:
            missing.append(name)
            continue
        conv = converted[name]
        ref_norm = normalize_source(ref.source)
        conv_norm = normalize_source(conv.source)
        if ref.registrations != conv.registrations:
            reg_mismatch.append((name, ref.registrations, conv.registrations))
        if ref_norm != conv_norm:
            code_mismatch.append((name, ref_norm, conv_norm, ref.file_path, conv.file_path))
        elif ref.registrations == conv.registrations:
            perfect.append(name)

    extra = [n for n in converted if n not in reference and n not in base_classes]

    print("=" * 80) # noqa: NP100
    print("CONVERSION FAITHFULNESS REPORT") # noqa: NP100
    print("=" * 80) # noqa: NP100
    print(f"Reference classes: {len(reference)}") # noqa: NP100
    print(f"Converted classes: {len(converted)}") # noqa: NP100
    print(f"Perfect matches:   {len(perfect)}") # noqa: NP100
    print(f"Missing:           {len(missing)}") # noqa: NP100
    print(f"Extra:             {len(extra)}") # noqa: NP100
    print(f"Reg mismatches:    {len(reg_mismatch)}") # noqa: NP100
    print(f"Code mismatches:   {len(code_mismatch)}") # noqa: NP100

    if missing:
        print("\nMISSING:") # noqa: NP100
        for n in sorted(missing):
            print(f"  - {n}") # noqa: NP100
    if extra:
        print("\nEXTRA:") # noqa: NP100
        for n in sorted(extra):
            print(f"  - {n}") # noqa: NP100
    if reg_mismatch:
        print("\nREGISTRATION MISMATCHES:") # noqa: NP100
        for name, ref_r, conv_r in reg_mismatch:
            print(f"  {name}:") # noqa: NP100
            print(f"    ref:  {ref_r}") # noqa: NP100
            print(f"    conv: {conv_r}") # noqa: NP100
    if code_mismatch:
        print("\nCODE MISMATCHES:") # noqa: NP100
        for name, ref_n, conv_n, rp, cp in code_mismatch:
            print(f"\n  {name}") # noqa: NP100
            print(f"    {rp} -> {cp}") # noqa: NP100
            diff = list(difflib.unified_diff(
                ref_n.split("\n"), conv_n.split("\n"),
                fromfile=f"reference/{name}", tofile=f"converted/{name}",
                lineterm=""))
            for line in diff[:40]:
                print(f"      {line}") # noqa: NP100
            if len(diff) > 40:
                print(f"      ... ({len(diff) - 40} more lines)") # noqa: NP100

    print()
    if not missing and not reg_mismatch and not code_mismatch:
        print("CONVERSION IS FAITHFUL") # noqa: NP100
        return 0
    print("CONVERSION HAS DIFFERENCES") # noqa: NP100
    return 1


if __name__ == "__main__":
    sys.exit(run())
