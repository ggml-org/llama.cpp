# Canonical SYCL build and runtime pins

Last verified: 2026-07-11 on `vinbonesjr`.

This is the source of truth for identifying the Arc-focused fork source,
locally built binaries, and the dormant systemd service. Do not infer
provenance from mtimes or build-directory names.

## Current identities

| Role | Package label | Source identity | Artifact/runtime path | Status |
|---|---|---|---|---|
| Loop source | Git checkout (not a package) | branch `turbo-sycl-opt`; P4.2 started from parent `415e4111aebe7dbf7a992c37a46ad73da451d3c7`; use `git rev-parse HEAD` for the moving tip | `/mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork` | active source checkout; one unrelated untracked audit script |
| Configured service | `unpackaged-local:build-turbofix` | executable reports build 9926, commit `904738505` | `/home/svnbjrn/build-turbofix/bin/llama-server` | unit is disabled and inactive; older than loop HEAD |
| P4.1 baseline | `unpackaged-local:p4.1-pinned-aot` | clean detached commit `10a70cde8161b5f2d3a50ed6e95ea74ec862ce25` | `/home/svnbjrn/build-turbo-aot-decode-10a70cde8/bin/llama-bench` and `/home/svnbjrn/build-turbo-aot-discriminator/bin/llama-perplexity` | validation-only; not deployed |

The configured service executable is not owned by a pacman package. Its
content identifiers are:

```text
llama-server --version: version 9926 (904738505)
llama-server sha256:    724d5aa5d9992d8fa3fe4314893e5223dace1fc251297c7bf32e8763b5b99c3b
libggml-sycl sha256:    7999fd6b3fb2fefa159ce98cab51812caba60c412dd75666c3c6397a0b10f0da
```

The service unit is `/etc/systemd/system/llama-sycl.cpp.service`. Its
`ExecStart` sources `/opt/intel/oneapi/setvars.sh`, then executes the
`build-turbofix` server against Mistral-7B with `SYCL0`, q8_0/q8_0 KV,
flash attention enabled, context 8192, and port 8088. The unit was
`inactive` and `disabled` when verified. This note does not authorize starting,
stopping, enabling, or rewriting it.

The P4.1 baseline hashes and results are in
`docs/research/standard-sycl-baseline-2026-07-11.md`. Those binaries are
pinned evidence, not the service artifact and not a substitute for a fresh
current-HEAD build.

## JIT-first build recipe

JIT is the default development posture on this host. Use a commit-scoped build
directory on ZFS and omit `GGML_SYCL_DEVICE_ARCH`; setting it to `acm-g10`
selects the slow offline AOT device-link path.

```bash
set +u
source /opt/intel/oneapi/setvars.sh
set -u
unset CFLAGS CXXFLAGS

SRC=/mnt/mrgr/llama-cpp-sycl-turbo/Raudbjorn-fork
# Replace SHORT_SHA with: git -C "$SRC" rev-parse --short=12 HEAD
BUILD=/home/svnbjrn/build-turbo-jit-SHORT_SHA

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_COMPILER_LAUNCHER= \
  -DCMAKE_CXX_COMPILER_LAUNCHER= \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_TARGET=INTEL \
  -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_SUPPORT_LEVEL_ZERO=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON
cmake --build "$BUILD" -j12 --target llama-bench llama-perplexity llama-server
```

Run JIT artifacts with the same oneAPI environment and an explicit device pin:

```bash
set +u
source /opt/intel/oneapi/setvars.sh
set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export LD_LIBRARY_PATH="$BUILD/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$BUILD/bin/llama-bench" --list-devices
```

Before a timing run, `fuser -v /dev/dri/renderD128` must show no other holder.
Correctness-only runs may proceed under documented contention; timing numbers
may not.

The configure half of this recipe was smoke-tested at `415e4111a` in
`/home/svnbjrn/build-turbo-jit-p42-verify`: CMake selected icx/icpx 2026.0,
`GGML_SYCL_DEVICE_ARCH` remained empty, and generated the `llama-bench`,
`llama-perplexity`, and `llama-server` targets.

## Content-based provenance checklist

Record all of the following with every result:

```bash
git -C "$SRC" rev-parse HEAD
git -C "$SRC" status --short --branch
"$BUILD/bin/llama-bench" --version
sha256sum "$BUILD/bin/llama-bench" \
  "$BUILD/bin/llama-perplexity" \
  "$BUILD/bin/libggml-sycl.so.0.15.1"
```

Also record branch, full commit, binary path, build directory, host, oneAPI
version, JIT/AOT mode, exact command, model path, and render-node occupancy.
For benchmark JSONL, retain the `build_commit` field. A binary version plus a
matching SHA-256 is the minimum identity proof; an mtime is never proof.

## AOT exception

AOT is opt-in proof work only, currently reserved for the canonical 564-chunk
acceptance probe or an explicitly AOT-scoped task. Configure a separate clean
ZFS build directory with `-DGGML_SYCL_DEVICE_ARCH=acm-g10`, record the source
commit before launch, and run detached with a durable log, PID, exact command,
and exit marker in `RALPH_PROGRESS.md`. The observed clean-build ceiling is
approximately 45 minutes. Do not reuse a JIT directory for AOT or present an
AOT artifact as the default product build.
