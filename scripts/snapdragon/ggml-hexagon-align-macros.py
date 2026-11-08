#!/usr/bin/env python3
"""
align-macros.py - Inspect and align trailing backslashes in multiline C/C++ macros.

Usage:
    align-macros.py [paths...]                 # Check and report misaligned macros
    align-macros.py --diff [paths...]          # Show unified diff of fixes
    align-macros.py --fix [paths...]           # Fix misaligned macros in-place
    align-macros.py --fix --reflow false ...   # Disable line reflowing in mostly-aligned macros
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


def str_to_bool(val: str) -> bool:
    if isinstance(val, bool):
        return val
    v = val.strip().lower()
    if v in ("true", "1", "yes", "y", "on"):
        return True
    if v in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{val}'")


def find_paren_comma_groups(s: str) -> List[Tuple[int, int, List[int]]]:
    """Find parenthesized groups (open_idx, close_idx, comma_indices) ordered from outermost to innermost."""
    stack: List[Tuple[int, List[int]]] = []
    groups: List[Tuple[int, int, List[int]]] = []
    in_str: Optional[str] = None
    escape = False

    for idx, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_str is not None:
            if ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            continue
        if ch == "(":
            stack.append((idx, []))
        elif ch == ")" and stack:
            open_idx, commas = stack.pop()
            if commas:
                groups.append((open_idx, idx, commas))
        elif ch == "," and stack:
            stack[-1][1].append(idx)

    groups.sort(key=lambda g: g[0])
    return groups


def find_top_level_op_splits(s: str) -> Tuple[Optional[int], List[int]]:
    """Find split positions after binary operators (+, -, &&, ||) at paren depth 0."""
    depth = 0
    in_str: Optional[str] = None
    escape = False
    assign_idx: Optional[int] = None
    splits: List[int] = []
    n = len(s)
    i = 0

    while i < n:
        ch = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if ch == "\\":
            escape = True
            i += 1
            continue
        if in_str is not None:
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = ch
            i += 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0:
            if s.startswith("&&", i) or s.startswith("||", i):
                splits.append(i + 2)
                i += 2
                continue
            if ch == "=" and (i == 0 or s[i - 1] not in "=!<>+-*/%&|^") and (i + 1 >= n or s[i + 1] != "="):
                if assign_idx is None:
                    assign_idx = i
            elif ch in "+-" and i > 0 and s[i - 1] == " " and i + 1 < n and s[i + 1] == " ":
                splits.append(i + 1)
        i += 1

    return assign_idx, splits


def reflow_c_line(content: str, max_len: int) -> Optional[List[str]]:
    """Try to cleanly reflow a C statement/signature line so each piece has len <= max_len."""
    if len(content) <= max_len:
        return [content]

    stripped = content.lstrip()
    if not stripped or stripped.startswith("#") or "//" in content or "/*" in content:
        return None

    base_indent = content[:len(content) - len(stripped)]

    # 1. Try splitting comma-separated arguments/parameters inside (...)
    for open_idx, _, commas in find_paren_comma_groups(content):
        pts = [0] + [c + 1 for c in commas] + [len(content)]
        segs: List[str] = []
        for idx in range(len(pts) - 1):
            seg = content[pts[idx]:pts[idx + 1]].rstrip()
            if idx > 0:
                seg = seg.lstrip()
            segs.append(seg)

        for cont_indent in [" " * (open_idx + 1), base_indent + "        ", base_indent + "    "]:
            lines = [segs[0]]
            for seg in segs[1:]:
                cand = f"{lines[-1]} {seg}"
                if len(cand) <= max_len:
                    lines[-1] = cand
                else:
                    lines.append(f"{cont_indent}{seg}")
            if len(lines) > 1 and all(len(line) <= max_len for line in lines):
                return lines

    # 2. Try splitting after top-level binary operators (+, -, &&, ||) or assignment (=)
    assign_idx, op_splits = find_top_level_op_splits(content)
    if not op_splits and assign_idx is not None:
        op_splits = [assign_idx + 1]
    if op_splits:
        pts = [0] + op_splits + [len(content)]
        segs = []
        for idx in range(len(pts) - 1):
            seg = content[pts[idx]:pts[idx + 1]].rstrip()
            if idx > 0:
                seg = seg.lstrip()
            segs.append(seg)

        indents = []
        if assign_idx is not None:
            rhs_start = assign_idx + 1
            while rhs_start < len(content) and content[rhs_start] == " ":
                rhs_start += 1
            indents.append(" " * rhs_start)
        indents.extend([base_indent + "        ", base_indent + "    "])

        for cont_indent in indents:
            lines = [segs[0]]
            for seg in segs[1:]:
                cand = f"{lines[-1]} {seg}"
                if len(cand) <= max_len:
                    lines[-1] = cand
                else:
                    lines.append(f"{cont_indent}{seg}")
            if len(lines) > 1 and all(len(line) <= max_len for line in lines):
                return lines

    return None


def reflow_macro_if_needed(
    macro: MacroDef,
    pad: int,
    dom_ratio_threshold: float = 0.80,
    fit_ratio_threshold: float = 0.90,
) -> Tuple[MacroDef, Optional[int]]:
    """
    If a macro is already mostly aligned at a dominant column (>= dom_ratio_threshold)
    and >= fit_ratio_threshold of continuation lines fit within that column, reflow any
    exceeding lines so the macro can keep its dominant column instead of shifting all lines.
    """
    cont_lines = [ml for ml in macro.lines if ml.bs_col is not None]
    if not cont_lines:
        return macro, None

    counts = Counter(ml.bs_col for ml in cont_lines if ml.bs_col is not None)
    dom_col, dom_count = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))[0]
    max_allowed_len = dom_col - pad
    if max_allowed_len <= 0:
        return macro, None

    dom_ratio = dom_count / len(cont_lines)
    fit_count = sum(1 for ml in cont_lines if len(ml.content) <= max_allowed_len)
    fit_ratio = fit_count / len(cont_lines)

    effective_dom_threshold = 0.70 if fit_count == len(cont_lines) else dom_ratio_threshold
    if dom_ratio < effective_dom_threshold or fit_ratio < fit_ratio_threshold:
        return macro, None

    new_macro_lines: List[MacroLine] = []
    for ml in macro.lines:
        if ml.bs_col is None or len(ml.content) <= max_allowed_len:
            new_macro_lines.append(ml)
        else:
            reflowed = reflow_c_line(ml.content, max_allowed_len)
            if not reflowed:
                return macro, None
            nl = "\r\n" if ml.raw.endswith("\r\n") else "\n"
            for piece in reflowed:
                new_macro_lines.append(MacroLine(
                    line_num=ml.line_num,
                    raw=f"{piece} \\{nl}",
                    content=piece,
                    bs_col=dom_col,
                    trailing_ws=False,
                ))

    return MacroDef(
        name=macro.name,
        filepath=macro.filepath,
        start_line=macro.start_line,
        end_line=macro.end_line,
        lines=new_macro_lines,
    ), dom_col


def compute_target_column(macro: MacroDef, mode: str, pad: int, target_col: Optional[int]) -> int:
    """Determine the column where backslashes should be aligned."""
    cont_lines = [ml for ml in macro.lines if ml.bs_col is not None]
    lines_for_len = cont_lines if cont_lines else macro.lines
    max_content_len = max(len(ml.content) for ml in lines_for_len)
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
    line_offset = 0

    for macro in macros:
        aligned = is_macro_aligned(macro)
        if not aligned or args.all:
            if not aligned:
                misaligned_count += 1

            cont_lines = [ml for ml in macro.lines if ml.bs_col is not None]
            bs_cols = [ml.bs_col for ml in cont_lines if ml.bs_col is not None]
            max_content = max(len(ml.content) for ml in (cont_lines or macro.lines))
            col_counts = Counter(bs_cols)

            if not args.quiet:
                logger.info(f"{filepath}:{macro.start_line}-{macro.end_line} [{macro.name}]")
                logger.info(f"  Max content width: {max_content}, Min needed column (+{args.pad}): {max_content + args.pad}")
                logger.info(f"  Current backslash columns: {dict(sorted(col_counts.items()))}")
                trailing_ws_lines = [ml.line_num for ml in macro.lines if ml.trailing_ws]
                if trailing_ws_lines:
                    logger.warning(f"  Warning: Trailing whitespace after backslash on line(s): {trailing_ws_lines}")

            reflow_col: Optional[int] = None
            target_macro = macro
            if args.reflow and args.target_col is None:
                target_macro, reflow_col = reflow_macro_if_needed(macro, args.pad)

            if reflow_col is not None:
                target_col = reflow_col
                if not args.quiet and len(target_macro.lines) != len(macro.lines):
                    logger.info(f"  -> Reflowed {len(target_macro.lines) - len(macro.lines)} line(s) to preserve column {target_col}")
            else:
                target_col = compute_target_column(target_macro, args.mode, args.pad, args.target_col)

            if not args.quiet:
                logger.info(f"  -> Target alignment column: {target_col}")

            realigned = realign_macro_lines(target_macro, target_col)

            start_idx = macro.start_line - 1 + line_offset
            end_idx = start_idx + len(macro.lines)
            if new_file_lines[start_idx:end_idx] != realigned:
                new_file_lines[start_idx:end_idx] = realigned
                line_offset += len(realigned) - len(macro.lines)
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
    parser.add_argument("--reflow", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Reflow long lines in mostly-aligned macros instead of shifting the whole macro (default: true)")
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
