# llamalib SDK

The `lib` branch of this repository is a generated, library-only tree ("llamalib"):
libllama + libggml + backends (CPU, BLAS, SYCL, Vulkan, OpenVINO) + a trimmed
`llama-common` helper library, plus the fork's correctness tests and the minimal
tool set (`llama-bench`, `llama-perplexity`, `llama-quantize`) needed by the
quality gate.

The branch is produced by `scripts/prune-to-lib.sh` from `master`. Never commit
to it by hand - changes belong on `master`, then regenerate.

## What LLAMA_DOWNLOAD=OFF means

SDK builds default to providing models as local GGUF files. With
`-DLLAMA_DOWNLOAD=OFF` the model downloader is replaced by a stub
(`common/download-stub.cpp`): no cpp-httplib, no OpenSSL, and any `-hf`,
`--model-url`, or `docker://` request fails with a clear runtime error.
Everything else in `llama-common` (sampling, chat templates, arg parsing)
works unchanged.

## Environment surface

`LLAMA_ARG_*` option parsing remains part of `llama-common`. Diagnostics for
unknown names are opt-in (`--warn-unknown-env`) and log names only, never
values. With `LLAMA_DOWNLOAD=OFF`, downloader and token environment variables
have no network effect. Server and UI executables and their runtime surfaces
are absent from the generated `lib` branch, while server option definitions
remain available through `llama-common` parser examples.

After an explicit parameter, token precedence is `HF_TOKEN` then
`HUGGING_FACE_HUB_TOKEN`; endpoint precedence is `MODEL_ENDPOINT` then
`HF_ENDPOINT`. The `lib` branch is generated from `master` and must never be
edited directly.

## Consuming the SDK

### Option 1: subproject (add_subdirectory / FetchContent)

```cmake
include(FetchContent)
FetchContent_Declare(llamalib
    GIT_REPOSITORY <fork-url>
    GIT_TAG        lib)   # or pin a specific lib-branch commit
set(GGML_SYCL ON)
set(GGML_SYCL_TARGET INTEL)
set(GGML_SYCL_F16 ON)
set(LLAMA_DOWNLOAD OFF)
set(LLAMA_BUILD_COMMON ON)  # optional: sampler/chat helpers
FetchContent_MakeAvailable(llamalib)

target_link_libraries(myapp PRIVATE llama)         # core
target_link_libraries(myapp PRIVATE llama-common)  # optional helpers
```

In subproject mode `LLAMA_STANDALONE` is OFF, so tools, tests, and examples are
skipped automatically.

### Option 2: installed package

Build and install the SDK once:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF \
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DLLAMA_DOWNLOAD=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DCMAKE_INSTALL_PREFIX=/opt/llamalib
cmake --build build && cmake --install build
```

Consume with CMake:

```cmake
find_package(llama CONFIG REQUIRED HINTS /opt/llamalib/lib/cmake/llama)
target_link_libraries(myapp PRIVATE llama::llama)
target_link_libraries(myapp PRIVATE llama::common)  # if installed with LLAMA_BUILD_COMMON
```

or pkg-config: `pkg-config --cflags --libs llama` (no `ggml.pc` is installed;
`llama.pc` covers the core). Note the helper library is shared-build only:
`llama-common-base` is folded into `libllama-common.so` and not installed as a
separate archive, so static consumption of the helpers is unsupported.

## SYCL runtime requirements (Intel Arc)

- The consumer process needs the oneAPI runtime on `LD_LIBRARY_PATH` (compiler
  runtime, MKL, TBB) - same env block as the build.
- First GPU launch pays a one-time cold-JIT (~37 s on Arc A770), cached under
  `~/.cache`. AOT (`-DGGML_SYCL_DEVICE_ARCH=acm-g10`) avoids it at the cost of
  a ~45 min build.
- Head dims must be multiples of 128 for turbo KV types; see the repository
  CLAUDE.md / AGENTS.md architecture notes.

## Runtime env knobs

All fork knobs are read inside the library and keep working in SDK builds:
`GGML_SYCL_Q8_KV_QUANTS_FIRST`, `TURBO_LAYER_ADAPTIVE`, `GGML_SYCL_FA_XMX`,
`GGML_SYCL_FA_ONEDNN`, `GGML_SYCL_FA_PROFILE`, and the upstream
`GGML_SYCL_*` set.

## Known issues

- **Teardown segfault after GPU inference (pre-existing).** A minimal C-API
  consumer (llama_backend_init, load, decode, llama_free) that fully offloads
  to SYCL0 can segfault inside `llama_free` -> `~llama_kv_cache` -> libc free
  after inference completes successfully. Reproduces against pre-SDK builds of
  the fork (so it is not introduced by the SDK restructure) and does not
  reproduce with the fork's own binaries (llama-completion, llama-bench) at
  identical parameters, nor with CPU-only inference (`n_gpu_layers = 0`).
  Impact: exit-time crash only; generated output is correct. Workaround until
  root-caused: skip `llama_free`/`llama_backend_free` on process exit (the OS
  reclaims), or run teardown before backend-heavy destruction ordering matters.

## Regenerating the lib branch

```bash
scripts/prune-to-lib.sh              # prunes master -> branch "lib"
scripts/prune-to-lib.sh --from HEAD  # prune a different source ref
```

The script refuses a dirty tree, prunes in a throwaway worktree, runs a
CPU-only CMake configure as a sanity gate, and only then force-updates the
branch.
