# Merge upstream into the Raudbjorn llama.cpp fork

This runbook updates `/mnt/mrgr/llama-resync` from `ggml-org/llama.cpp` while preserving the fork's deliberately smaller backend surface and its TurboQuant/SYCL implementation. It is written for an agent operating the repository, but every command can also be run by a human.

The default procedure is a sanitized snapshot merge, not a direct merge or rebase. Upstream is first copied into a scratch tree, backends excluded by fork policy are removed, and that tree is committed with the real merge base as its parent. Merging the synthetic commit gives Git a correct three-way base without importing thousands of incompatible backend files.

This procedure preserves the fork's first-parent history. It imports upstream content as one synthetic delta; it does **not** make every upstream commit an ancestor of the result. Do not describe it as a commit-by-commit history sync.

## Outcome and non-goals

A completed merge has all of the following properties:

- The merge commit's first parent is the fork tip that was selected before the merge.
- The second parent is the sanitized snapshot commit.
- Current upstream code is present except for explicitly excluded backends and their build/runtime registrations.
- CPU, BLAS, SYCL, Vulkan, and OpenVINO remain available.
- TurboQuant types, CPU reference code, graph wiring, KV policy, SYCL kernels, tests, and runtime state survive.
- The CPU build, fallback-disabled UI source build, and safe SYCL correctness gate pass.
- The working tree is clean after the merge commit.
- Nothing is pushed unless the user explicitly requests it.

This runbook does not restore removed backends, run destructive GPU probes, establish model-level quality, or prove production AOT behavior. Those require separate scope and evidence.

## Source of truth

Read these before editing:

- [`AGENTS.md`](../../AGENTS.md): current repository policy, supported backend set, TurboQuant architecture, build commands, and verification rules.
- [`docs/build.md`](../build.md): current build options.
- [`docs/backend/SYCL.md`](../backend/SYCL.md): SYCL toolchain and runtime details.
- [`TURBOQUANT_UPSTREAM_MERGE.md`](../../TURBOQUANT_UPSTREAM_MERGE.md): historical merge notes only. It describes an older fork shape and is not current backend policy.

When these disagree, current code and `AGENTS.md` win. Never copy stale line numbers or old conflict choices without re-reading the current files.

## Fixed fork invariants

### Supported backends

The fork keeps these backend directories and registrations:

| Backend | Directory or integration |
| --- | --- |
| CPU | `ggml/src/ggml-cpu/` |
| BLAS | `ggml/src/ggml-blas/` |
| SYCL | `ggml/src/ggml-sycl/` |
| Vulkan | `ggml/src/ggml-vulkan/` |
| OpenVINO | `ggml/src/ggml-openvino/` |

The sanitized upstream tree excludes these backend directories:

```text
ggml-cann
ggml-cuda
ggml-et
ggml-hexagon
ggml-hip
ggml-metal
ggml-musa
ggml-opencl
ggml-rpc
ggml-virtgpu
ggml-webgpu
ggml-zdnn
ggml-zendnn
```

Documentation may name an excluded backend for comparison or migration guidance. Build files, registries, compiled sources, and runtime dispatch must not register or depend on one.

### Stable TurboQuant ABI

GGUF serializes `ggml_type` numerically. Preserve these values unless a separately designed migration explicitly changes the format:

| Type | Numeric value |
| --- | ---: |
| `GGML_TYPE_TURBO2_0` | 43 |
| `GGML_TYPE_TURBO3_0` | 44 |
| `GGML_TYPE_TURBO4_0` | 45 |
| `GGML_TYPE_TQ3_1S` | 46 |
| `GGML_TYPE_TQ4_1S` | 47 |
| `GGML_TYPE_Q8_CR` | 48 |
| `GGML_TYPE_Q5_CR` | 49 |
| `GGML_TYPE_Q6_CR` | 50 |
| `GGML_TYPE_COUNT` | 51 |

Never resolve an enum conflict by accepting whichever side compiles. First compare numeric assignments, serialized consumers, block definitions, static assertions, quantizers, and backend switches. A renumbered type can compile and still silently misread existing GGUF files.

The separate `llama_ftype` / Python `LlamaFileType` namespace is serialized as
`general.file_type`. PR 51 assigns its new Q8_CR/Q5_CR/Q6_CR values 512/513/514,
below the `GUESSED=1024` flag and away from the next upstream slots. This is a
fork-selected range, not an upstream reservation. Existing TQ3_1S/TQ4_1S values
43/44 remain unchanged. Audit both enum namespaces on every future sync; never
confuse these file-type IDs with tensor-type IDs 48/49/50.

`GGML_OP_TURBO_WHT` must also remain present in the op enum, name tables, graph builder, CPU implementation, and SYCL dispatch.

### Fork-owned surfaces

At minimum, preserve these files or their current replacements:

```text
ggml/include/ggml-innerq.h
ggml/src/ggml-innerq.c
ggml/src/ggml-turbo-quant.c
ggml/src/ggml-turbo-wht-signs.h
ggml/src/ggml-sycl/fattn-xmx.cpp
ggml/src/ggml-sycl/fattn-xmx.hpp
ggml/src/ggml-sycl/innerq.cpp
ggml/src/ggml-sycl/turbo-quants.hpp
ggml/src/ggml-sycl/turbo-wht.cpp
ggml/src/ggml-sycl/turbo-wht.hpp
ggml/src/ggml-sycl/sycl-mutable-command-list-probe.cpp
ggml/src/ggml-sycl/sycl-mutable-command-list-probe.cl
src/llama-turbo-innerq-runtime.cpp
```

Also preserve TurboQuant template instances under `ggml/src/ggml-sycl/template-instances/`, graph-side WHT calls in `src/llama-graph.cpp`, and auto-asymmetric/layer-adaptive policy in `src/llama-kv-cache.cpp`.

Treat `CONTRIBUTING.md`, `.github/`, `ci/run.sh`, `scripts/`, `vendor/CMakeLists.txt`, and the server proxy policy as fork-owned review surfaces. In particular, preserve `tools/server/server-cors-proxy.h`, its routing in `tools/server/server.cpp`, and `tools/server/tests/unit/test_proxy.py`. Upstream versions may be useful inputs, but they must not restore removed backend automation, weaken private reporting, or remove proxy destination policy.

The list is a floor, not a substitute for comparing the current fork against the merge base. New fork-owned files added after this guide must be inventoried before each merge.

## Phase 1: establish immutable inputs

Do not start from a dirty checkout. Do not stash unrelated user work to make the merge possible. Use a clean branch or a dedicated worktree.

```bash
set -euo pipefail

REPO=/mnt/mrgr/llama-resync
BASE_BRANCH=resync
UPSTREAM_URL=https://github.com/ggml-org/llama.cpp
EXPECTED_UPSTREAM_TIP=3466812d1
MERGE_BRANCH="merge/upstream-$(date +%F)"

cd "$REPO"
test -z "$(git status --porcelain)"

git remote get-url upstream >/dev/null 2>&1 || git remote add upstream "$UPSTREAM_URL"
git fetch upstream master

FORK_TIP=$(git rev-parse "$BASE_BRANCH")
UPSTREAM_TIP=$(git rev-parse FETCH_HEAD)
EXPECTED_UPSTREAM_TIP=$(git rev-parse "${EXPECTED_UPSTREAM_TIP}^{commit}")
test "$UPSTREAM_TIP" = "$EXPECTED_UPSTREAM_TIP"
MERGE_BASE=$(git merge-base "$FORK_TIP" "$UPSTREAM_TIP")

printf 'REPO=%s\nBASE_BRANCH=%s\nMERGE_BRANCH=%s\nFORK_TIP=%s\nEXPECTED_UPSTREAM_TIP=%s\nUPSTREAM_TIP=%s\nMERGE_BASE=%s\n' \
  "$REPO" "$BASE_BRANCH" "$MERGE_BRANCH" "$FORK_TIP" "$EXPECTED_UPSTREAM_TIP" "$UPSTREAM_TIP" "$MERGE_BASE" \
  > .git/upstream-merge.env

git switch -c "$MERGE_BRANCH" "$FORK_TIP"
```

Completion criteria:

- `git status --porcelain` was empty before branch creation.
- `.git/upstream-merge.env` records the repository, branches, and three full object IDs.
- `git branch --show-current` prints the new merge branch.
- No push occurred.

### Inventory the fork delta before choosing resolutions

The merge base, not memory, defines fork-owned work:

```bash
source .git/upstream-merge.env

git diff --name-status "$MERGE_BASE..$FORK_TIP" -- \
  .github ci scripts vendor CONTRIBUTING.md docs/development/upstream-merge.md \
  ggml/include ggml/src src common tests tools CMakeLists.txt cmake \
  > /tmp/llama-fork-delta.txt

git diff --stat "$MERGE_BASE..$FORK_TIP"
```

Classify changed paths into four groups:

1. Fork-only files that upstream does not have.
2. Shared files with fork-only behavior that must be replayed.
3. Backend removals and build-surface pruning.
4. Historical edits already superseded by upstream.

Record the classification before merging. Completion means every fork-changed path has an explicit disposition; "probably upstream" is not a disposition.

## Phase 2: create the sanitized upstream tree

Use `git archive` so the scratch tree contains tracked upstream content only. Put the scratch tree outside the repository.

```bash
source .git/upstream-merge.env

SCRATCH=$(mktemp -d -p /mnt/mrgr upstream-scratch-XXXXXX)
git archive "$UPSTREAM_TIP" | tar -x -C "$SCRATCH"

REMOVED_BACKENDS=(
  ggml-cann ggml-cuda ggml-et ggml-hexagon ggml-hip
  ggml-metal ggml-musa ggml-opencl ggml-rpc ggml-virtgpu
  ggml-webgpu ggml-zdnn ggml-zendnn
)

for backend in "${REMOVED_BACKENDS[@]}"; do
  rm -rf "$SCRATCH/ggml/src/$backend"
done
```

### Strip build and runtime references surgically

Directory deletion is insufficient. Inspect at least:

```text
CMakeLists.txt
cmake/
ggml/CMakeLists.txt
ggml/include/
ggml/src/CMakeLists.txt
ggml/src/ggml-backend-reg.cpp
ggml/src/ggml-sycl/
ggml/src/ggml-vulkan/
common/
src/
tests/
tools/
.github/
ci/
scripts/
vendor/
```

Remove only blocks that register, compile, include, dispatch, or require excluded backends. Keep shared code merely because an excluded backend also uses it. Keep explanatory documentation unless the documentation falsely claims that the fork ships the backend.

The exact reference sites change upstream-to-upstream. Do not preserve a static patch script that assumes old line numbers. Search the fresh scratch tree, read each containing construct, and edit the complete construct.

Useful search families are:

```bash
REMOVED_DIR_RE='ggml-(cann|cuda|et|hexagon|hip|metal|musa|opencl|rpc|virtgpu|webgpu|zdnn|zendnn)'
REMOVED_MACRO_RE='GGML_(USE_)?(CANN|CUDA|ET|HEXAGON|HIP|METAL|MUSA|OPENCL|RPC|VIRTGPU|WEBGPU|ZDNN|ZENDNN)'

grep -RInE "$REMOVED_DIR_RE|$REMOVED_MACRO_RE" \
  "$SCRATCH/.github" "$SCRATCH/CMakeLists.txt" "$SCRATCH/ci" \
  "$SCRATCH/cmake" "$SCRATCH/ggml" "$SCRATCH/common" "$SCRATCH/src" \
  "$SCRATCH/scripts" "$SCRATCH/tests" "$SCRATCH/tools" "$SCRATCH/vendor"

grep -RInE \
  'ui_mcp_proxy_allow|proxy_handler_(get|post)|proxy_request|metadata\.(google\.internal|goog)' \
  "$SCRATCH/common" "$SCRATCH/tools/server"
```

Treat every hit as a review item, not an automatic deletion instruction. A comment describing unsupported CUDA behavior is different from `add_subdirectory(src/ggml-cuda)`.

### Validate the sanitized tree before creating an object

Check both absence and presence:

```bash
for backend in "${REMOVED_BACKENDS[@]}"; do
  test ! -e "$SCRATCH/ggml/src/$backend"
done

for backend in ggml-cpu ggml-blas ggml-sycl ggml-vulkan ggml-openvino; do
  test -d "$SCRATCH/ggml/src/$backend"
done
```

Configure a CPU-only build from the scratch tree. This catches dangling CMake registrations before the real merge becomes involved:

```bash
SANITIZED_BUILD=$(mktemp -d -p /tmp llama-sanitized-build-XXXXXX)
cmake -S "$SCRATCH" -B "$SANITIZED_BUILD" -G Ninja \
  -DGGML_SYCL=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_OPENVINO=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF
```

Completion criteria:

- Every excluded backend directory is absent.
- Every retained backend directory is present.
- No build/runtime reference to an excluded backend remains without a documented reason.
- CPU-only CMake configuration succeeds.

## Phase 3: create an ancestry-correct snapshot commit

The synthetic commit must have the real merge base as its parent. A root commit made by `git init` does not provide a useful three-way base and turns the merge into a near-total add/add conflict.

Use a private temporary index to write the sanitized tree into the main repository's object database without touching the working index:

```bash
cd "$REPO"
source .git/upstream-merge.env

TMP_INDEX=$(mktemp)
rm -f "$TMP_INDEX"

GIT_INDEX_FILE="$TMP_INDEX" \
  git --git-dir="$REPO/.git" --work-tree="$SCRATCH" add -A

SANITIZED_TREE=$(GIT_INDEX_FILE="$TMP_INDEX" \
  git --git-dir="$REPO/.git" write-tree)

SNAPTIP=$(printf '%s\n\nupstream: %s\n' \
  'Snapshot sanitized upstream tree' "$UPSTREAM_TIP" |
  git commit-tree "$SANITIZED_TREE" -p "$MERGE_BASE")

rm -f "$TMP_INDEX"
printf 'SNAPTIP=%s\n' "$SNAPTIP" >> .git/upstream-merge.env
```

Verify the object before merging:

```bash
source .git/upstream-merge.env

test "$(git show -s --format=%P "$SNAPTIP")" = "$MERGE_BASE"
test "$(git rev-parse "$SNAPTIP^{tree}")" = "$SANITIZED_TREE"
test "$(git merge-base "$FORK_TIP" "$SNAPTIP")" = "$MERGE_BASE"
git merge-base --is-ancestor "$MERGE_BASE" "$FORK_TIP"
git merge-base --is-ancestor "$MERGE_BASE" "$UPSTREAM_TIP"
git diff --name-status "$UPSTREAM_TIP" "$SNAPTIP" > /tmp/llama-sanitized-delta.txt
git diff --stat "$UPSTREAM_TIP" "$SNAPTIP"
```

The second command must show only intentional sanitization differences. A broad unrelated diff means the scratch tree is incomplete or contaminated; discard it and regenerate it.

## Phase 4: merge without committing

```bash
cd "$REPO"
source .git/upstream-merge.env

git merge --no-ff --no-commit "$SNAPTIP"
```

Conflicts are expected. The merge remains uncommitted until all structural, build, and runtime verification passes.

### Conflict-resolution discipline

For every conflict:

1. Read the base, ours, and theirs versions.
2. Identify the behavior owned by each side.
3. Resolve the smallest complete construct: enum block, class, function, CMake target, or test case.
4. Search all declarations, definitions, and callers before changing an exported symbol.
5. Stage the path only after all conflict markers in that path are gone.
6. Compile the smallest affected target before moving to broad verification.

Do not use "ours means fork" and "theirs means upstream" as a universal policy. A shared file usually needs upstream structure plus a small replay of fork behavior.

If the harness exposes conflicts as `conflict://N`, resolve one ID per write. Never use a wildcard or bulk conflict replacement. If a conflict operation corrupts a file, run `git merge --abort`, return to the clean branch tip, and restart rather than repairing nested or synthetic markers.

### Resolution matrix

| Area | Default resolution |
| --- | --- |
| Backend registry | Keep the curated CPU/BLAS/SYCL/Vulkan/OpenVINO registrations. Add new upstream registry mechanics only when they serve retained backends. |
| Root and ggml CMake | Start from upstream structure, remove excluded backend options/subdirectories, then restore fork-owned source files and targets. |
| `ggml/include/ggml.h` | Union upstream API changes with stable TurboQuant type numbers and `GGML_OP_TURBO_WHT`. Verify numeric ABI before staging. |
| `ggml/src/ggml-common.h` | Preserve block layouts and static assertions; integrate upstream shared definitions around them. |
| `src/llama-graph.cpp` | Keep upstream graph classes and new model paths, then replay WHT wrapping at every current attention construction path. |
| `src/llama-kv-cache.cpp` | Keep upstream cache architecture, then replay auto-asymmetric, layer-adaptive, padding, and rotation policy. |
| Shared SYCL files | Prefer upstream organization and fixes; replay TurboQuant dispatch, dequantization, WHT, InnerQ, template instances, and supported-op policy. |
| Fork-only SYCL files | Keep unless the behavior was deliberately retired and the retirement is documented. |
| `common/arg.cpp` and `common/common.h` | Union declarations and options. Reject duplicate flags, orphaned structs, and flags whose implementation was removed. |
| Tests | Preserve upstream coverage and fork gates. Remove duplicate cases and ensure every loop/block has one owner. |
| UI source | Prefer upstream barrel imports and component APIs. Remove duplicated props/imports introduced by unioning old and new forms. |

### ABI and API checks during resolution

Before staging `ggml.h`, assert the serialized type values from the working tree:

```bash
python3 - <<'PY'
from pathlib import Path
import re

text = Path('ggml/include/ggml.h').read_text()
expected = {
    'GGML_TYPE_TURBO2_0': 43,
    'GGML_TYPE_TURBO3_0': 44,
    'GGML_TYPE_TURBO4_0': 45,
    'GGML_TYPE_TQ3_1S': 46,
    'GGML_TYPE_TQ4_1S': 47,
    'GGML_TYPE_Q8_CR': 48,
    'GGML_TYPE_Q5_CR': 49,
    'GGML_TYPE_Q6_CR': 50,
    'GGML_TYPE_COUNT': 51,
}
for name, value in expected.items():
    match = re.search(rf'\b{name}\s*=\s*(\d+)', text)
    assert match, f'missing {name}'
    assert int(match.group(1)) == value, (name, match.group(1), value)
print('TurboQuant ggml_type ABI: OK')
PY
```

Public and internal parameter structs must move together. In particular, after this synchronization the public context parameters include `n_outputs_max_per_seq`, `samplers`, and `n_samplers`, while `src/llama-cparams.h` carries the corresponding internal output limit. Treat missing or duplicated fields as a merge defect, not a compiler quirk.

### Prefer targeted replay over wholesale replacement

Replacing a `.cpp` file wholesale is safe only when its headers, sibling implementations, source lists, and callers are advanced to the same interface. Otherwise it produces a cascade of missing declarations, undefined vtables, or duplicate definitions.

Before replacing a shared implementation file, check:

- Its public and private headers.
- Every declared virtual method and constructor.
- CMake source registration.
- Callers and derived classes.
- Whether upstream moved an implementation from `.cpp` to `.hpp`.
- Whether the fork added behavior inside the replaced functions.

If several files in one subsystem are structurally coupled, compare and integrate the subsystem as a unit. Do not repeatedly patch one linker error at a time while leaving the interface split across revisions.

## Phase 5: run the re-strip and preservation audits

After all paths are staged as resolved, search the working tree again. Upstream code can reintroduce excluded registrations outside the originally expected files.

```bash
source .git/upstream-merge.env

REMOVED_BACKENDS=(
  ggml-cann ggml-cuda ggml-et ggml-hexagon ggml-hip
  ggml-metal ggml-musa ggml-opencl ggml-rpc ggml-virtgpu
  ggml-webgpu ggml-zdnn ggml-zendnn
)

for backend in "${REMOVED_BACKENDS[@]}"; do
  test ! -e "ggml/src/$backend"
done

git grep -nE \
  'ggml-(cann|cuda|et|hexagon|hip|metal|musa|opencl|rpc|virtgpu|webgpu|zdnn|zendnn)|GGML_(USE_)?(CANN|CUDA|ET|HEXAGON|HIP|METAL|MUSA|OPENCL|RPC|VIRTGPU|WEBGPU|ZDNN|ZENDNN)' \
  -- .github CMakeLists.txt ci cmake ggml/include ggml/src common scripts src tests tools vendor
```

Review any output. The completion condition is not mechanically zero text matches; it is zero compiled or runtime dependency on excluded backends.

Verify fork-owned files:

```bash
FORK_FILES=(
  ggml/include/ggml-innerq.h
  ggml/src/ggml-innerq.c
  ggml/src/ggml-turbo-quant.c
  ggml/src/ggml-turbo-wht-signs.h
  ggml/src/ggml-sycl/fattn-xmx.cpp
  ggml/src/ggml-sycl/fattn-xmx.hpp
  ggml/src/ggml-sycl/innerq.cpp
  ggml/src/ggml-sycl/turbo-quants.hpp
  ggml/src/ggml-sycl/turbo-wht.cpp
  ggml/src/ggml-sycl/turbo-wht.hpp
  ggml/src/ggml-sycl/sycl-mutable-command-list-probe.cpp
  ggml/src/ggml-sycl/sycl-mutable-command-list-probe.cl
  src/llama-turbo-innerq-runtime.cpp
  .github/labeler.yml
  .github/ISSUE_TEMPLATE/010-bug-compilation.yml
  .github/ISSUE_TEMPLATE/011-bug-results.yml
  ci/run.sh
  CONTRIBUTING.md
  vendor/CMakeLists.txt
  scripts/check-required-targets.sh
  scripts/check-upstream-sync-invariants.sh
  docs/development/upstream-merge.md
  tools/server/server-cors-proxy.h
  tools/server/server.cpp
  tools/server/tests/unit/test_proxy.py
)

for path in "${FORK_FILES[@]}"; do
  test -f "$path" || { echo "missing: $path"; exit 1; }
done

test ! -e .github/workflows/build-wasm.yml
test ! -d scripts/snapdragon
```

Then verify symbols and wiring, not only file presence:

```bash
git grep -n 'GGML_OP_TURBO_WHT' -- ggml/include ggml/src src tests
git grep -n 'ggml_turbo_wht' -- ggml/include ggml/src src tests
git grep -nE 'TURBO_AUTO_ASYMMETRIC|TURBO_LAYER_ADAPTIVE' -- src
git grep -n 'llama_memory_clear_data_only' -- include src tests
git grep -nE 'ui_mcp_proxy_allow|proxy_handler_(get|post)|proxy_request' -- common tools/server
git grep -nE 'loopback|link-local|metadata|allowlist|private' -- tools/server/tests/unit/test_proxy.py
```

Run the checked-in invariant gate after the manual review:

```bash
scripts/check-upstream-sync-invariants.sh
```

The gate requires a production consumer for `ui_mcp_proxy_allow`, the proxy policy and global-address classifiers, the compiled proxy-policy target, and direct private/metadata rejection cases. Its removed-backend scan covers build and CI registrations. It excludes prose, dynamic plugin calls, OpenVINO's OpenCL dependency, and SYCL OpenCL target strings.

Completion means every required symbol has a declaration, implementation, dispatch or call site as appropriate, and a source file registered in CMake.

## Phase 6: verification ladder

Use the cheapest discriminating check first. Fix the first failure at its source, then resume the ladder. Do not run a production AOT link to diagnose a syntax error.

Use build directories outside `/mnt/mrgr`; mergerfs space accounting can fail long SYCL links. A ZFS-backed directory under the user's home is preferred.

```bash
BUILD_ROOT=${BUILD_ROOT:-/home/svnbjrn/.cache/llama-resync-merge}
CPU_BUILD="$BUILD_ROOT/cpu"
SYCL_BUILD="$BUILD_ROOT/sycl-jit"
UI_BUILD="$BUILD_ROOT/ui-source"
mkdir -p "$BUILD_ROOT"
```

### 1. Static merge integrity

```bash
test -z "$(git diff --name-only --diff-filter=U)"
git diff --check
git diff --cached --check

if git grep -nE '^(<<<<<<<|=======|>>>>>>>)' -- \
  .github CMakeLists.txt ci cmake common docs ggml include scripts src tests tools vendor; then
  echo "conflict markers remain" >&2
  exit 1
fi
```

Expected: no unmerged paths, no whitespace errors, and no conflict markers.

### 2. CPU configure and build

```bash
cmake -S . -B "$CPU_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DGGML_SYCL=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_OPENVINO=OFF

scripts/check-required-targets.sh "$CPU_BUILD" cpu

ninja -C "$CPU_BUILD"
```

A successful link is required. It catches interface mismatches that syntax-only checks miss, including absent graph-input definitions, stale source lists, and library version metadata drift.

### 3. Force a UI source build without fallback

The server build may download a prebuilt UI after the local Svelte build fails. That makes the overall build green while the checked-in UI source is broken. Verify the source separately with Hugging Face fallback disabled.

```bash
UI_SOURCE=$(mktemp -d -p /tmp llama-ui-source-XXXXXX)
mkdir -p "$UI_SOURCE/tools"
cp -a tools/ui "$UI_SOURCE/tools/"
rm -rf "$UI_SOURCE/tools/ui/node_modules" "$UI_SOURCE/tools/ui/dist"

test ! -e "$UI_SOURCE/tools/ui/dist/index.html"

cmake \
  -DUI_SOURCE_DIR="$UI_SOURCE/tools/ui" \
  -DUI_BINARY_DIR="$UI_BUILD" \
  -DLLAMA_SOURCE_DIR="$PWD" \
  -DHF_BUCKET=ggml-org/llama-ui \
  -DHF_ENABLED=OFF \
  -DBUILD_UI=ON \
  -DLLAMA_UI_EMBED="$CPU_BUILD/tools/ui/llama-ui-embed" \
  -DLLAMA_UI_GZIP=OFF \
  -P scripts/ui-assets.cmake
```

Required evidence includes `UI: npm build succeeded` and successful generation of `ui.cpp`/`ui.h`. A downloaded archive is not acceptable evidence.

### 4. CLI parser and metadata smoke

```bash
"$CPU_BUILD/bin/llama-cli" --version
"$CPU_BUILD/bin/llama-cli" --help >/tmp/llama-cli-help.txt
```

Check the expected version/commit, Turbo KV types, token and endpoint precedence text, speculative options, and unknown-environment diagnostics. Duplicate argument registrations commonly fail during `--help` construction before inference starts.

### 5. CPU TurboQuant gates

```bash
"$CPU_BUILD/bin/test-turbo-innerq-runtime"
"$CPU_BUILD/bin/test-kv-cache-adaptive-mode"

"$CPU_BUILD/bin/test-backend-ops" test -b CPU -o TURBO_WHT
"$CPU_BUILD/bin/test-backend-ops" test -b CPU -o SET_ROWS \
  -p '.*(turbo3|tq4_1s).*'
"$CPU_BUILD/bin/test-backend-ops" test -b CPU -o MUL_MAT \
  -p '.*tq4_1s.*m=16,n=1,k=256.*'
```

Each command must exit zero. Preserve unsupported-case semantics: a path documented as unsupported should remain an explicit unsupported result, not be converted into a fake pass.

### 6. SYCL JIT build

Use oneAPI `icx`/`icpx`, disable compiler launchers, and avoid AOT during iteration:

```bash
source /opt/intel/oneapi/setvars.sh

cmake -S . -B "$SYCL_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_COMPILER_LAUNCHER= \
  -DCMAKE_CXX_COMPILER_LAUNCHER= \
  -DLLAMA_BUILD_TESTS=ON

scripts/check-required-targets.sh "$SYCL_BUILD" sycl

ninja -C "$SYCL_BUILD" test-sycl-turbo-correctness
```

If compilation reports duplicate device helpers, check whether upstream moved their definitions into a header before deleting anything. During the 2026-09-01 merge, `op_tanh`, `op_gelu`, `op_exp`, and `op_silu` had moved to `element_wise.hpp`; only their stale duplicate `.cpp` definitions were removed. Related helpers still owned by the `.cpp` file were retained.

### 7. Safe Arc A770 correctness oracle

Inspect device users first. This is a correctness run, not a timing run; do not stop user services without authorization.

```bash
fuser -v /dev/dri/renderD128 || true

ONEAPI_DEVICE_SELECTOR=level_zero:0 \
SYCL_CACHE_PERSISTENT=1 \
  "$SYCL_BUILD/bin/test-sycl-turbo-correctness"
```

The binding summary is:

```text
0 GATE-FAIL
0 XPASS
```

Do not set `LLAMA_TEST_TURBO_FA`, `LLAMA_TEST_FA256`, or `LLAMA_TEST_INNERQ` as part of the default merge gate. Turbo FA and d=256 have explicit A770 hang risk; opt-in runs need their own authorization, timeout, and recovery plan.

### 8. Optional production AOT build

Run this only when the merge is intended for immediate A770 production deployment:

```bash
AOT_BUILD="$BUILD_ROOT/sycl-aot"

cmake -S . -B "$AOT_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_DEVICE_ARCH=acm-g10 \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_COMPILER_LAUNCHER= \
  -DCMAKE_CXX_COMPILER_LAUNCHER=

ninja -C "$AOT_BUILD"
```

A JIT build does not prove AOT linkability. If AOT is skipped, say so in the final report.

## Failure signatures and root causes

These failures occurred during the 2026-09-01 synchronization and are likely to recur when upstream restructures the same areas.

| Failure | Likely cause | Correct response |
| --- | --- | --- |
| Linker reports missing graph-input vtables or constructors | Upstream declarations or model call sites landed without all definitions, or `src/CMakeLists.txt` omitted new source files | Compare the full declaration/definition family and upstream source manifest; restore the complete family, not one symbol. |
| Duplicate CLI option during `llama-cli --help` | Both old fork and new upstream registration blocks survived | Keep one canonical registration and its documented precedence; verify help construction. |
| CMake build succeeds after a Svelte error | UI provisioning downloaded a prebuilt fallback | Run `scripts/ui-assets.cmake` with `HF_ENABLED=OFF` against a clean archived source tree. |
| Svelte reports duplicate identifier or prop | Merge kept both direct and barrel imports, or both old and new component properties | Keep the current upstream component API/barrel import and remove only the obsolete duplicate. Rebuild source with fallback disabled. |
| Duplicate SYCL helper definition | Upstream moved implementation into a header while fork retained the `.cpp` body | Confirm each helper's current owner, then remove only exact duplicates. |
| `tests/test-backend-ops.cpp` compiles with brace/case errors | Large test unions duplicated a loop, case block, or closing brace | Read the entire enclosing test block and compare upstream/fork cases; do not patch isolated braces blindly. |
| New model `.cpp` files compile but fail at link | CMake source list did not advance with upstream | Diff `src/CMakeLists.txt` against the upstream snapshot and retain fork-only sources in addition. |
| Public context parameter errors | `include/llama.h`, `src/llama-cparams.h`, and context implementation came from different revisions | Advance the interface family together and verify all fields once. |
| Unexpected removed backend returns | Final re-strip was limited to the old known files | Search the entire compiled/build/runtime surface after conflict resolution. |
| A merge appears to contain upstream but has unrelated-root conflicts | Synthetic snapshot was committed as a root rather than as a child of the real merge base | Abort and recreate the snapshot with `git commit-tree ... -p "$MERGE_BASE"`. |

## Phase 7: stage and commit only after verification

Run final checks after the last source edit:

```bash
test -z "$(git diff --name-only --diff-filter=U)"
git diff --check

git add -A
git diff --cached --check
```

Inspect the staged change at subsystem level. Confirm the merge is still active and the expected snapshot is `MERGE_HEAD`:

```bash
source .git/upstream-merge.env

test "$(git rev-parse MERGE_HEAD)" = "$SNAPTIP"
test "$(git show -s --format=%P "$SNAPTIP")" = "$MERGE_BASE"
test "$(git merge-base "$FORK_TIP" "$SNAPTIP")" = "$MERGE_BASE"
test "$(git branch --show-current)" = "$MERGE_BRANCH"
```

Create one merge commit:

```bash
git commit -m 'Merge sanitized upstream snapshot'
```

Verify parent order and cleanliness:

```bash
source .git/upstream-merge.env

set -- $(git show -s --format='%P' HEAD)
test "$#" -eq 2
test "$1" = "$FORK_TIP"
test "$2" = "$SNAPTIP"
test "$(git merge-base "$1" "$2")" = "$MERGE_BASE"
test -z "$(git status --porcelain)"

git show -s --format='%H%n%P%n%s' HEAD
```

Do not push. Repository policy permits PRs only from the current branch to `master` on `Raudbjorn/ggml-llama.cpp`, and creating or pushing one still requires explicit user authorization.

## Abort and recovery

Before the merge commit exists, the clean recovery path is:

```bash
git merge --abort
```

Then verify that the merge branch is back at `FORK_TIP` and regenerate any suspect scratch/snapshot objects. Do not retain a file containing nested conflict markers or a conflict resolution whose provenance is unclear.

If verification reveals a design-level incompatibility rather than a local merge defect:

1. Leave the merge uncommitted.
2. Record the exact failing command and first causal error.
3. Identify whether the incompatibility is in upstream structure, fork behavior, backend policy, or ABI.
4. Finish all independent verification that remains safe.
5. Ask for a scope decision only when the available choices materially change fork behavior.

Never "fix" a merge by suppressing a test, deleting a supported fork feature, enabling an excluded backend, or accepting a downloaded UI artifact in place of broken source.

## Merge report template

Record this at completion:

```text
Upstream URL:
Upstream tip:
Fork tip:
Merge base:
Sanitized snapshot:
Merge commit:
Excluded backends:
Preserved fork surfaces:

Verified:
- Static merge integrity:
- CPU configure/build:
- Fallback-disabled UI source build:
- CLI help/version:
- CPU TurboQuant tests:
- SYCL JIT build:
- Safe Arc A770 correctness gate:

Not claimed:
- AOT build:
- Model-level inference/perplexity:
- Turbo FA opt-in:
- d=256 FA opt-in:
- InnerQ FA opt-in:
- Performance:

Push/PR status:
```

A verification statement must name the exact command or observable result. "Build looks good" and "upstream merged" are not evidence.

## Completion checklist

- [ ] Upstream tip is exactly `3466812d1` and resolves to the recorded full object ID.
- [ ] Clean fork tip, upstream tip, and merge base recorded before editing.
- [ ] Current fork delta classified from the merge base.
- [ ] Sanitized upstream tree excludes all policy-removed backends.
- [ ] Sanitized tree configures independently.
- [ ] Snapshot commit has the real merge base as its sole parent.
- [ ] Snapshot differs from upstream only by intentional sanitization.
- [ ] Snapshot tree matches the sanitized tree object and its sole parent is the real merge base.
- [ ] All conflicts resolved by behavior, not blanket side selection.
- [ ] TurboQuant numeric type ABI remains 43-47 with count 48.
- [ ] Fork-owned sources, graph wiring, KV policy, and SYCL dispatch remain present.
- [ ] Fork governance, `.github` automation, `ci/run.sh`, `scripts`, vendor policy, and server proxy policy survived with explicit dispositions.
- [ ] No excluded backend is compiled, registered, or required at runtime.
- [ ] Removed backend workflows, labels, issue choices, CI branches, and Snapdragon scripts are absent.
- [ ] Proxy allowlist consumption and private/metadata rejection tests remain present.
- [ ] No unresolved paths, conflict markers, or whitespace errors remain.
- [ ] CPU build and TurboQuant CPU gates pass.
- [ ] UI source builds with fallback disabled.
- [ ] SYCL JIT target builds and safe correctness oracle reports zero gate failures and XPASS.
- [ ] Optional AOT/model/performance gaps are stated explicitly.
- [ ] Merge commit parent order is correct and the worktree is clean.
- [ ] Nothing was pushed without explicit authorization.

## API shims and kept divergences (2026-09-05)

Upstream call sites must compile unchanged after a sync. Two upstream signatures
are preserved as thin shims; keep them when resolving conflicts:

- `ggml_gated_delta_net(ctx, q, k, v, g, beta, state, K)` is the upstream form and
  wraps `ggml_gated_delta_net_ext(..., K, emit_mode)` with `emit_mode = 0`.
  `op_params[0] = K` as upstream, `op_params[1] = emit_mode` (0 for upstream callers).
  Fork ingredient-replay callers (`src/models/delta-net-base.cpp`,
  `tests/test-gdn-ingredient-replay.cpp`) use the `_ext` form.
- `llama_memory_recurrent(model, type_r, type_s, offload, mem_size, n_seq_max, n_rs_seq, filter)`
  is the upstream constructor and delegates to the fork constructor with
  `gdn_replay_req = false`. Fork callers pass the flag explicitly.
- `tests/test-upstream-api-shims.cpp` fails to compile if either shim is removed.

Kept product divergences (owner decision, expect conflicts in these hunks):

- `--cache-ram -1` means half of the free host memory at startup
  (`tools/server/server-context.cpp`, `common/common.h`, `common/arg.cpp` help text,
  `tools/cli/README.md`, `tools/server/README.md`); upstream means unlimited.
- `common/arg.cpp` tensor-split and fit-target parsing accept exactly
  `llama_max_devices()` entries (`>` bound); upstream rejects that count (`>=`).
