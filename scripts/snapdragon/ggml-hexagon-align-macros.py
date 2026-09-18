#!/usr/bin/env python3
"""
align-macros.py - Inspect and align trailing backslashes in multiline C/C++ macros.

Usage:
    align-macros.py [paths...]                 # Check and report misaligned macros
    align-macros.py --diff [paths...]          # Show unified diff of fixes
    align-macros.py --fix [paths...]           # Fix misaligned macros in-place
    align-macros.py --fix --mode majority ...  # Align to the dominant column
    align-macros.py --fix --pad 2 ...          # Align to (max_content_len + pad)

Safety rules:
    - Macros that are ALREADY aligned are NEVER touched (unless --all is given).
    - Whitespace after trailing backslashes is flagged and cleaned.
"""

import argparse
import difflib
import logging
import os
import re
import sys
from collections import Counter
from typing import List, Optional, Tuple, NamedTuple

logger = logging.getLogger("ggml-hexagon-align-macros")


class MacroLine(NamedTuple):
    line_num: int       # 1-indexed
    raw: str            # Original line including newline
    content: str        # Line content before trailing backslash (stripped of trailing whitespace)
    bs_col: Optional[int]  # 1-indexed column of backslash, or None if last line has no backslash
    trailing_ws: bool   # True if whitespace existed after the backslash


class MacroDef(NamedTuple):
    name: str
    filepath: str
    start_line: int
    end_line: int
    lines: List[MacroLine]


def parse_macros(filepath: str) -> List[MacroDef]:
    """Extract all multiline macros from a C/C++ source file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return []

    macros: List[MacroDef] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        m = re.match(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if m:
            macro_name = m.group(1)
            macro_start = i + 1
            macro_lines: List[MacroLine] = []
            cur = i

            while cur < n:
                l_raw = lines[cur]
                l_rstrip = l_raw.rstrip("\r\n")

                # Check if line has a trailing backslash
                # Note: handle possible accidental spaces after backslash
                match_bs = re.search(r"\\([ \t]*)$", l_rstrip)
                if match_bs:
                    has_trailing_ws = len(match_bs.group(1)) > 0
                    bs_index = match_bs.start()
                    content = l_rstrip[:bs_index].rstrip()
                    # 1-indexed column of the backslash
                    bs_col = bs_index + 1
                    macro_lines.append(MacroLine(
                        line_num=cur + 1,
                        raw=l_raw,
                        content=content,
                        bs_col=bs_col,
                        trailing_ws=has_trailing_ws
                    ))
                    cur += 1
                else:
                    # Line does not end with backslash
                    if cur == i:
                        # Single-line macro, not multiline
                        break
                    else:
                        # Final line of a multiline macro
                        macro_lines.append(MacroLine(
                            line_num=cur + 1,
                            raw=l_raw,
                            content=l_rstrip.rstrip(),
                            bs_col=None,
                            trailing_ws=False
                        ))
                        break

            # Only record if it is a multiline macro (has at least one continuation line)
            continuation_lines = [ml for ml in macro_lines if ml.bs_col is not None]
            if continuation_lines:
                macro_end = macro_lines[-1].line_num
                macros.append(MacroDef(
                    name=macro_name,
                    filepath=filepath,
                    start_line=macro_start,
                    end_line=macro_end,
                    lines=macro_lines
                ))
            i = cur
        i += 1

    return macros


def is_macro_aligned(macro: MacroDef) -> bool:
    """A macro is aligned if all continuation lines have backslashes at the same column."""
    bs_cols = [ml.bs_col for ml in macro.lines if ml.bs_col is not None]
    if not bs_cols:
        return True
    has_trailing_ws = any(ml.trailing_ws for ml in macro.lines)
    return len(set(bs_cols)) == 1 and not has_trailing_ws


def compute_target_column(macro: MacroDef, mode: str, pad: int, target_col: Optional[int]) -> int:
    """Determine the column where backslashes should be aligned."""
    max_content_len = max(len(ml.content) for ml in macro.lines)
    min_needed = max_content_len + pad

    if target_col is not None:
        return max(target_col, min_needed)

    bs_cols = [ml.bs_col for ml in macro.lines if ml.bs_col is not None]
    if not bs_cols:
        return min_needed

    if mode == "min":
        return min_needed
    elif mode == "max":
        return max(max(bs_cols), min_needed)
    elif mode == "majority":
        counts = Counter(bs_cols)
        # Sort by frequency descending, then by column descending
        majority_col = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))[0][0]
        return max(majority_col, min_needed)
    else:
        return min_needed


def realign_macro_lines(macro: MacroDef, target_col: int) -> List[str]:
    """Format macro lines with backslashes aligned at target_col."""
    new_lines: List[str] = []
    for ml in macro.lines:
        nl = "\r\n" if ml.raw.endswith("\r\n") else "\n"
        if ml.bs_col is None:
            # Last line without backslash
            new_lines.append(ml.raw)
        else:
            if not ml.content:
                spaces = " " * (target_col - 1)
                new_lines.append(f"{spaces}\\{nl}")
            else:
                spaces_needed = max(1, target_col - len(ml.content) - 1)
                new_lines.append(f"{ml.content}{' ' * spaces_needed}\\{nl}")
    return new_lines


def process_file(filepath: str, args: argparse.Namespace) -> Tuple[int, int, Optional[str]]:
    macros = parse_macros(filepath)
    if not macros:
        return 0, 0, None

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        file_lines = f.readlines()

    misaligned_count = 0
    modified = False
    new_file_lines = list(file_lines)

    for macro in macros:
        aligned = is_macro_aligned(macro)
        if not aligned or args.all:
            if not aligned:
                misaligned_count += 1

            bs_cols = [ml.bs_col for ml in macro.lines if ml.bs_col is not None]
            max_content = max(len(ml.content) for ml in macro.lines)
            col_counts = Counter(bs_cols)

            if not args.quiet:
                logger.info(f"{filepath}:{macro.start_line}-{macro.end_line} [{macro.name}]")
                logger.info(f"  Max content width: {max_content}, Min needed column (+{args.pad}): {max_content + args.pad}")
                logger.info(f"  Current backslash columns: {dict(sorted(col_counts.items()))}")
                trailing_ws_lines = [ml.line_num for ml in macro.lines if ml.trailing_ws]
                if trailing_ws_lines:
                    logger.warning(f"  Warning: Trailing whitespace after backslash on line(s): {trailing_ws_lines}")

            target_col = compute_target_column(macro, args.mode, args.pad, args.target_col)
            if not args.quiet:
                logger.info(f"  -> Target alignment column: {target_col}")

            realigned = realign_macro_lines(macro, target_col)

            start_idx = macro.start_line - 1
            end_idx = start_idx + len(macro.lines)
            if new_file_lines[start_idx:end_idx] != realigned:
                new_file_lines[start_idx:end_idx] = realigned
                modified = True

    diff_text = None
    if modified:
        diff = difflib.unified_diff(
            file_lines,
            new_file_lines,
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
            lineterm=""
        )
        diff_text = "\n".join(diff)

        if args.fix:
            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(new_file_lines)
            if not args.quiet:
                logger.info(f"  [FIXED] Updated {filepath}")

    return len(macros), misaligned_count, diff_text


def find_source_files(paths: List[str]) -> List[str]:
    extensions = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".inl"}
    result: List[str] = []
    for p in paths:
        if os.path.isfile(p):
            result.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for file in sorted(files):
                    _, ext = os.path.splitext(file)
                    if ext.lower() in extensions:
                        result.append(os.path.join(root, file))
    return sorted(result)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Inspect and align backslashes in multiline C/C++ macros."
    )
    parser.add_argument("paths", nargs="*", default=["."], help="Files or directories to scan (default: current dir)")
    parser.add_argument("--fix", action="store_true", help="Fix misaligned macros in-place")
    parser.add_argument("--diff", action="store_true", help="Display unified diff of suggested fixes")
    parser.add_argument("--check", action="store_true", help="Exit with code 1 if misaligned macros exist")
    parser.add_argument("--mode", choices=["min", "max", "majority"], default="min",
                        help="Alignment mode: 'min' (max_len + pad), 'max' (max existing col), 'majority' (dominant col)")
    parser.add_argument("--pad", type=int, default=2, help="Spaces between longest line and backslash (default: 2)")
    parser.add_argument("--target-col", type=int, default=None, help="Force alignment to an exact column")
    parser.add_argument("--all", action="store_true", help="Realign all macros even if already aligned (default: only misaligned)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only output errors and diffs/summary")

    args = parser.parse_args()

    files = find_source_files(args.paths)
    if not files:
        logger.error("No C/C++ source files found.")
        sys.exit(0)

    total_macros = 0
    total_misaligned = 0
    diffs: List[str] = []

    for filepath in files:
        num_macros, num_misaligned, diff_text = process_file(filepath, args)
        total_macros += num_macros
        total_misaligned += num_misaligned
        if diff_text:
            diffs.append(diff_text)

    if args.diff and diffs:
        logger.info("\n--- Proposed Changes ---\n")
        for d in diffs:
            logger.info(d)

    logger.info(f"\nSummary: scanned {len(files)} files, {total_macros} multiline macros, {total_misaligned} misaligned.")

    if args.check and total_misaligned > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
