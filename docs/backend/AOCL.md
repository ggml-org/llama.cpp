# AMD AOCL-BLAS

> [!NOTE]
> The [ZenDNN backend](ZenDNN.md) is the recommended path for inference on AMD CPUs. Refer to its documentation for the currently supported operations and data types. This page covers AOCL-BLAS as a vendor option for the generic `GGML_BLAS` backend.

AOCL-BLAS is AMD's BLAS library, optimized for AMD EPYC and Ryzen CPUs.
llama.cpp can link against it through the existing BLAS backend (`GGML_BLAS`).

The BLAS backend can use AOCL-BLAS for eligible large prompt GEMMs and generally does not participate in token generation.
F32 weights are passed to `cblas_sgemm` directly. Other types are converted to F32 first, so a quantized model can be slower than the native CPU kernels.
See [BLAS Build](../build.md#blas-build).

AOCL download / install: https://www.amd.com/en/developer/aocl.html

Use the Quick start for a short command list. The later sections cover install layout, the single-threaded tree, threading, and checks.

### Quick start (MT, 64 threads)

If AOCL is not installed yet, follow [Prepare](#prepare) first. Adjust the `amd-libs.cfg` path to your install. CMake flags are set once. Source `amd-libs.cfg` again in every new shell before you launch, or the loader will not find AOCL-BLAS.

```bash
source /opt/aocl/<version>/aocc/MT/amd-libs.cfg

cmake -B build \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=AOCL_mt \
  -DBLAS_INCLUDE_DIRS="${AOCL_ROOT}/include" \
  -DGGML_NATIVE=ON
cmake --build build --config Release

source /opt/aocl/<version>/aocc/MT/amd-libs.cfg
./build/bin/llama-cli -m model.gguf -t 64
```

`-t 64` is the thread count to pass on each launch (`--threads 64` is the same flag). Details, the single-threaded tree, and NUMA binding are below.

CMake vendor names `AOCL` and `AOCL_mt` are recognized by [FindBLAS](https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors) (CMake 3.27+).
When either vendor is selected, llama.cpp enables the BLIS code path (`GGML_BLAS_USE_BLIS`): it includes `blis.h` and calls `bli_thread_set_num_threads()` before GEMM. The same vendors also set `GGML_BLAS_USE_AOCL`, which only changes the device description to `AOCL-BLAS`. Upstream BLIS (`FLAME`) sets `GGML_BLAS_USE_BLIS` alone, so its description stays `BLIS`.

### Prepare

1. Install AOCL from AMD (package or tarball). Current releases ship **ST** (single-threaded) and **MT** (multi-threaded) libraries in separate folders. The default `AOCL_ROOT` is the **MT** tree. A typical layout (replace `<version>` and `aocc` with your install):

```
<aocl-prefix>/<version>/aocc/MT/
<aocl-prefix>/<version>/aocc/ST/
```

2. Source `amd-libs.cfg` from the tree you want. Current AOCL versions use this file (not a separate `aocl-env.sh`). Adjust the prefix, version, and compiler (`aocc` vs `gcc`) to match your install:

```bash
# Multi-threaded (default AOCL_ROOT):
source /opt/aocl/<version>/aocc/MT/amd-libs.cfg

# Single-threaded:
# source /opt/aocl/<version>/aocc/ST/amd-libs.cfg
```

This sets library and include paths so the linker can find AOCL-BLAS. Skipping it is a common cause of BLAS not found / unresolved symbol errors.

Optional, if your install provides an environment module:

```bash
cd /opt/aocl/<version>/aocc/MT
module load ./aocl-linux-aocc-<version>_module
# module unload ./aocl-linux-aocc-<version>_module
```

3. Prefer the **MT** libraries for llama.cpp. Use `-DGGML_BLAS_VENDOR=AOCL_mt` after sourcing the MT `amd-libs.cfg`. Use `-DGGML_BLAS_VENDOR=AOCL` if you sourced the ST tree instead.

### llama.cpp compilation

Requires **CMake 3.27 or newer** for `-DGGML_BLAS_VENDOR=AOCL` / `AOCL_mt`.

FindBLAS does not detect AOCL headers via pkg-config. After sourcing `amd-libs.cfg`, `AOCL_ROOT` is set and `$AOCL_ROOT/include` is a symlink to the active integer ABI (`include_LP64` by default). Pass that path to CMake:

```bash
source /opt/aocl/<version>/aocc/MT/amd-libs.cfg   # adjust path

cmake -B build \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=AOCL_mt \
  -DBLAS_INCLUDE_DIRS="${AOCL_ROOT}/include" \
  -DGGML_NATIVE=ON

cmake --build build --config Release
```

#### CMake older than 3.27

`AOCL` / `AOCL_mt` may be unknown to FindBLAS. After sourcing the AOCL env, you can try:

```bash
cmake -B build \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=Generic \
  -DBLAS_LIBRARIES="-lblis -lm" \
  -DBLAS_INCLUDE_DIRS="${AOCL_ROOT}/include" \
  -DGGML_NATIVE=ON
```

Library names differ between AOCL packages (`blis`, `blis-mt`, etc.). Pass whatever your install provides.
`GGML_BLAS_VENDOR=Generic` does not enable the BLIS header and thread path (`GGML_BLAS_USE_BLIS`). Upgrade CMake so `AOCL` or `AOCL_mt` is recognized.

### llama.cpp execution

`--threads` / `--threads-batch` are the thread budget for **every** backend that implements `set_n_threads` (CPU and BLAS).

On each large BLAS `MUL_MAT`, the backend calls `bli_thread_set_num_threads()` with that **same** value, so BLIS GEMM may use up to `--threads-batch` threads during prompt processing. Other ops stay on the CPU backend with the same limit. Token generation usually does not use BLAS.

`BLIS_NUM_THREADS` is **not** a reliable way to cap BLIS here: the per-GEMM `bli_thread_set_num_threads()` call overrides it. To use fewer cores, lower `--threads` and/or `--threads-batch`.

Thread scaling depends on the CPU, NUMA layout, model, and batch size. Benchmark the thread counts used for deployment. Nested OpenMP (ggml type conversion, then BLIS GEMM, both using OpenMP) can still oversubscribe even though those two steps are sequential.

On a multi-socket machine, bind the process to one NUMA node. To skip SMT, bind to that node's physical cores only (check `lscpu -e`; on many AMD layouts the first range is the physical cores and a higher range is the sibling threads):

```bash
numactl --physcpubind=0-127 --membind=0 ./build/bin/llama-cli -m model.gguf
```

Keep the sourced AOCL env (or `LD_LIBRARY_PATH`) set when running binaries, or dynamic linking to AOCL libs will fail. Source the same `amd-libs.cfg` you used at build time.

### Verify

- Configure output should show BLAS found, with libraries under the AOCL MT tree and includes at `$AOCL_ROOT/include`.
- `ldd` on `llama-bench` should list that same AOCL-BLAS library; `blis-mt` or `blis`.
- `--list-devices` prints the description `AOCL-BLAS` (`BLAS: AOCL-BLAS`). Upstream BLIS (`FLAME`) still prints `BLIS`.
- `llama-bench` reports the backend as `BLAS` for this build and `CPU` for a build with `-DGGML_BLAS=OFF`.
- `test-backend-ops -b BLAS` checks that BLAS GEMMs match the CPU reference.

### Notes

- Optional `-march=znver3` / `znver4` / `znver5` (or similar) can be passed via `CMAKE_C_FLAGS` / `CMAKE_CXX_FLAGS` for a known CPU, but `-DGGML_NATIVE=ON` is usually enough and is safer across Ryzen / EPYC generations.
- For building AOCL-BLAS (AMD's BLIS fork) from source instead of AOCL packages, see https://github.com/amd/blis

### Reference

1. https://www.amd.com/en/developer/aocl.html
2. https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors
3. https://github.com/amd/blis
