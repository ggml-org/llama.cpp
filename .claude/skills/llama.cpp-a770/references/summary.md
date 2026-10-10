This file is a merged representation of a subset of the codebase, containing files not matching ignore patterns, combined into a single document by Repomix.
The content has been processed where content has been compressed (code blocks are separated by ⋮---- delimiter).

# Summary

## Purpose

This is a reference codebase organized into multiple files for AI consumption.
It is designed to be easily searchable using grep and other text-based tools.

## File Structure

This skill contains the following reference files:

| File | Contents |
|------|----------|
| `project-structure.md` | Directory tree with line counts per file |
| `files.md` | All file contents (search with `## File: <path>`) |
| `tech-stacks.md` | Languages, frameworks, and dependencies per package (search with `## Tech Stack: <path>`) |
| `summary.md` | This file - purpose and format explanation |

## Usage Guidelines

- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes

- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching these patterns are excluded: docs/benchmarks/**/product.json
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Content has been compressed - code blocks are separated by ⋮---- delimiter
- Long base64 data strings (e.g., data:image/png;base64,...) have been truncated to reduce token count
- Files are sorted by Git change count (files with more changes are at the bottom)

## Statistics

11940 files | 1,828,048 lines

| Language | Files | Lines |
|----------|------:|------:|
| JSON | 3145 | 138,803 |
| ERR | 1492 | 216,270 |
| Markdown | 1478 | 46,119 |
| FREQ | 982 | 64,702 |
| FDINFO | 956 | 5,100 |
| C++ | 710 | 77,328 |
| Text | 574 | 411,751 |
| C/C++ Header | 511 | 64,242 |
| TypeScript | 416 | 20,313 |
| Svelte | 324 | 28,736 |
| Other | 1352 | 754,684 |

**Largest files:**
- `docs/benchmarks/N-xe-x4-pinned-ceon/bench.stdout` (75,024 lines)
- `docs/benchmarks/oneapi-ab/N-xe-x4-pinned-ceon/bench.stdout` (75,024 lines)
- `docs/benchmarks/N-xe-x6-noscratch/bench.stdout` (43,556 lines)
- `docs/benchmarks/oneapi-ab/N-xe-x6-noscratch/bench.stdout` (43,556 lines)
- `docs/benchmarks/N-xe-x1-trace/bench.stdout` (43,426 lines)
- `docs/benchmarks/oneapi-ab/N-xe-x1-trace/bench.stdout` (43,426 lines)
- `docs/benchmarks/N-xe-realtext-moecache/err-soft.txt` (40,175 lines)
- `docs/benchmarks/oneapi-ab/N-xe-realtext-moecache/err-soft.txt` (40,175 lines)
- `benches/dgx-spark/aime25_openai__gpt-oss-120b-high_temp1.0_20251109_094547.html` (35,144 lines)
- `docs/ops/BLAS.csv` (30,737 lines)