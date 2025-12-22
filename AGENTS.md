# Repository Guidelines (llama.cpp / ggml)

This repo is `llama.cpp` + `ggml`. Prefer minimal diffs and match existing patterns.

Note: `AGENTS.md` is hierarchical. Prefer putting highly-specific rules under the closest directory (e.g. `src/`, `ggml/src/ggml-cpu/`) instead of bloating this file.

## 0) Tooling / Search Priority (must follow)

When locating code, avoid broad “whole-repo” searches until needed:

1) File-level narrowing: `fd`
2) Text search: `rg` (ripgrep)
3) Optional structural search: `sg` (ast-grep)

Exclude bulky dirs in searches: `.git`, `build*`, `models`, `tmp`, `node_modules` (if any).

Optional (recommended for C/C++ hot-path edits): generate `compile_commands.json` and run `clang-tidy` on the files you touched (avoid full-repo tidy runs).

Required after code changes (fast, targeted only):
- `clang-format` check: use `git clang-format` against the merge-base of your target branch, scoped to C/C++ paths.
  - Example (check only): `BASE=$(git merge-base HEAD origin/master 2>/dev/null || git merge-base HEAD origin/main); git clang-format --style=file --diff "$BASE" -- '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hh' '*.hpp'`
  - To apply formatting: drop `--diff`.
- `clang-tidy` check: run on the C/C++ source files you touched using `build-rel/compile_commands.json` (avoid headers and full-repo tidy runs).
  - Example (macOS): `BASE=$(git merge-base HEAD origin/master 2>/dev/null || git merge-base HEAD origin/main); FILES=$(git diff --name-only "$BASE" -- '*.c' '*.cc' '*.cpp' '*.cxx'); [ -n "$FILES" ] && clang-tidy -p build-rel --extra-arg="-isysroot$(xcrun --show-sdk-path)" --checks="-misc-include-cleaner" $FILES`

## 1) Project Structure & Ownership Map

- `src/`: model loading, GGUF KV parsing, tensor mapping, graph building, decode/prefill logic.
- `include/llama.h`: public API (keep backward-compatible unless explicitly changing API).
- `ggml/` + `ggml/src/`: tensor runtime + CPU/GPU kernels (treat as upstream-like: smallest necessary changes).
- `gguf-py/` + `scripts/`: GGUF tooling / conversion helpers.
- `tests/` + `ci/`: tests and CI harness.

When adding a **new model architecture**, most changes should stay in `src/` (see `src/AGENTS.md` for entry points).

## 2) Build, Test, and Development Commands (authoritative)

### Configure + build (CPU Release)
- `cmake -B build-rel -DCMAKE_BUILD_TYPE=Release`
- `cmake --build build-rel -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)` (multi-config: add `--config Release`)

### Run tests
- `ctest --test-dir build-rel --output-on-failure -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)` (multi-config: add `-C Release`)

### Full CI parity (optional)
- `bash ci/run.sh ./tmp/results ./tmp/mnt` (add `GG_BUILD_CUDA=1`, `GG_BUILD_SYCL=1`, etc. as needed)

### Quick functional run (example)
- `./build-rel/bin/llama-cli -m <path/to/model.gguf> --gpu-layers 0 -t 4 -p "I believe life is" -n 128 -no-cnv`

### Targeted rebuild (examples)
- `cmake --build build-rel --target ggml-base test-ifairy -j $(nproc 2>/dev/null || sysctl -n hw.ncpu)`

## 3) Stage-Gated Workflow (for non-trivial work)

For complex changes (e.g. adding a new architecture, changing hot-path kernels), do not jump straight to edits:

1) **Spec**: restate requirements + invariants + acceptance criteria
2) **Plan**: short checklist tied to acceptance criteria
3) **Do**: implement minimal diffs, run validation commands, summarize evidence

## 4) Adding a New Model Architecture: Checklist (Definition of Done)

### Scope definition (must be explicit)
- Architecture string/id (GGUF `general.architecture`) and how it maps to `llm_arch`
- Required GGUF keys / hyperparams and their defaults
- Tensor naming scheme and required shapes
- Which inference shapes must work (prefill vs decode)

### Implementation steps (typical)
1) Extend `llm_arch` + name mapping (`src/llama-arch.*`)
2) Parse GGUF metadata into a dedicated hyperparams struct (`src/llama-model-loader.*`)
3) Map GGUF tensors → internal tensors; validate shapes early with actionable errors (`src/llama-model*.cpp`)
4) Build graph paths for prefill + decode (`llama_model::build_graph`, `src/llama-graph.*`, `src/llama-context.cpp`)
5) Add/extend minimal tests (load succeeds, one forward path runs)
6) Update docs with a minimal repro command and any env flags

### Acceptance criteria
- Release build succeeds (`build-rel`)
- Targeted tests pass (`ctest` and any architecture-specific tests)
- `llama-cli` loads the GGUF and runs deterministically for a fixed seed
- If performance-related: provide reproducible `eval tok/s` logs (raw output) + summary

## 5) iFairy / LUT Special Rules (global summary only)

This repo includes an iFairy 2-bit complex LUT path. When touching it:

- Semantic invariant: must match baseline exactly (`w * conj(x)`)
- Correctness gate: `./build-rel/bin/test-ifairy` + strict mode must pass:
  - `GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_VALIDATE_STRICT=1 ./build-rel/bin/test-ifairy`
- Performance claims must include reproducible commands + raw `eval tok/s` logs
- Edge-case coverage lives in `tests/test-ifairy.cpp` (alignment, small/large dims, env semantics); keep docs in sync.

Detailed iFairy rules live in:
- `ggml/src/ggml-cpu/AGENTS.md`
- `IFAIRY_ARM_3W_LUT_DESIGN.md`, `IFAIRY_ARM_3W_LUT_API_PLAN.md`, `IFAIRY_ARM_3W_LUT_STATUS.md`

## 6) Coding Style & Naming Conventions

- C/C++: 4-space indentation, middle-aligned pointers (`void * ptr`), `snake_case` identifiers; match local alignment.
- Avoid formatting-only diffs; prefer `git clang-format` when needed.
- Do not introduce new third-party dependencies without prior discussion.
- Python: PEP 8 + repo typing configs (`pyproject.toml`, `pyrightconfig.json`).

## 7) Testing / Benchmarking Guidelines

- Use `LLAMACPP_TEST_MODELFILE=/path/model.gguf` when tests need weights.
- For perf/quality spot checks: `./build-rel/bin/llama-bench ...`, `./build-rel/bin/llama-perplexity ...`.
- For targeted suites: `ctest -R tokenizer` or `scripts/debug-test.sh`; server tests: `tools/server/tests/tests.sh`.

## 8) Commit & Pull Request Guidelines

- Commits: `<module>: concise message (#issue)`; keep scope focused.
- PRs should include: what changed + why, repro commands, tests run, and perf logs (if relevant).

## 9) Security & Configuration Tips

- Consult `SECURITY.md` before disclosing vulnerabilities.
- Never commit model weights, API keys, or proprietary datasets.
- Store environment-specific overrides outside the repo (e.g. `~/.config/llama.cpp/`).
