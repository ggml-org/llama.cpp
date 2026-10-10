# Repository Guidelines

## Project Structure & Module Organization

This directory is the Arc A770 benchmark archive for the TurboQuant/SYCL
llama.cpp fork. Follow the repository-root `AGENTS.md` alongside this guide.
Paths below are repository-relative unless stated otherwise.

- `docs/benchmarks/README.md`, `MATRIX-GUIDE.md`, and `CAMPAIGNS.md` explain the archive; `FILES.tsv` records artifact identities.
- `matrix-1006/`, `decode/`, and `N-xe-*` hold campaign reports and raw evidence. `oneapi-ab/` contains overlapping artifacts with some unique evidence; preserve both copies.
- `scripts/` contains maintained harnesses and analysis tools; `tests/` contains correctness tests.
- `src/`, `common/`, and `ggml/src/ggml-sycl/` implement inference and the SYCL backend. `docs/research/` records dated findings.

## Build, Test, and Development Commands

Run these from the repository root:

```bash
# Check archive inventory behavior without GPU access.
python3 -m unittest discover -s scripts -p 'test_index_benchmarks.py'
# Emit an inventory for inspection without overwriting archived data.
python3 scripts/index_benchmarks.py docs/benchmarks > /tmp/benchmark-inventory.tsv
# Rebuild an already configured SYCL development tree.
ninja -C build-sycl
# Run registered SYCL tests with per-test timeouts.
ctest --test-dir build-sycl -L sycl --timeout 180 -V
```

For initial oneAPI/CMake configuration, follow `docs/backend/SYCL.md`.
Archived runners may contain obsolete absolute paths; inspect them before execution.

## Coding Style & Naming Conventions

Use ASCII text, LF endings, final newlines, and four-space indentation, following
`.editorconfig`. Match nearby naming: `matrix-*.sh`, descriptive campaign directories,
and `test_*.py` harness tests. Reuse maintained scripts before adding runners.
The root pre-commit configuration checks whitespace, YAML, large files, and Python
style with Flake8; C/C++ formatting follows `.clang-format`.

## Testing & Benchmark Evidence

Use Python `unittest` for inventory changes and the synthetic
`test-sycl-turbo-correctness` oracle for affected SYCL behavior. Documentation changes
need link and command review. Preserve raw logs and distinguish correctness from throughput.

Before timing, stop `llama-sycl.cpp.service`, verify exclusive GPU use, capture driver
and stack details, and wrap GPU commands in `timeout`. Restore the service afterward.
Compare builds on the same boot and driver; historical i915 results are not xe baselines.

## Commit & Pull Request Guidelines

Use one logical change per commit and descriptive prefixes, such as `docs:`,
`fix(perf):`, or `test(spec):`, matching recent history. Preserve unrelated work.
Create PRs only from the current branch to `Raudbjorn/ggml-llama.cpp:master`.
Explain the change and verification, link relevant issues or evidence, and end every
PR body with a `Not claimed` section covering untested behavior, evidence limits,
residual risks, and new state or dependencies.
