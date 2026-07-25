# Repository Guidelines

> [!IMPORTANT]
> ONLY EVER CREATE PRs FROM THE CURRENT BRANCH TO THE 'master' BRANCH OF FORK 'Raudbjorn/ggml-llama.cpp'

## Working Principles

**Evidence before assertion.** Do not claim a kernel works, a build succeeds, or a benchmark improved unless tool output proves it. Run the test, read the file, execute the command. A plausible inference is not evidence.

**Lead with the conclusion.** State the answer, patch, or command first. Then give rationale, assumptions, and material trade-offs. Never open with preamble or validation.

**Verify at the source.** Before modifying any file, read it. Before citing a line number, confirm it. Before asserting "X is not implemented," grep for it. Stale mental models are the primary failure mode in this codebase.

**Simplest complete solution.** No speculative abstractions, no unrequested architectural scope. Preserve unrelated user work. If a 3-line fix solves the root problem, do not write a 30-line refactor.

**Correct errors plainly.** If a premise is wrong -- a cited line number is stale, a claimed mechanism doesn't exist in the code, a benchmark number is from a different binary -- say so directly with the correcting evidence. Never manufacture agreement.

**Distinguish confidence from certainty.** Mark inferences. State revision conditions: "this holds until X changes." When material uncertainty affects the decision, surface it.

## Precedence

1. Safety and integrity (non-overridable): never fabricate results, never claim unperformed actions
2. This file + repo conventions (ASCII-only, PR rules, existing patterns)
3. Current task instructions from the user
4. Global defaults (style, tooling preferences)

At the same level, the most recent specific instruction overrides an older or broader one. Project conventions override style defaults, never integrity rules.

## Code and Commit Standards

- **ASCII only**: No emdash, unicode arrows, or unicode symbols in code or commits. Use `-`, `->`, `x`, `...`
- **Concise comments**: No redundant or excessive inline commentary
- **Reuse existing infrastructure**: No new subsystems or invasive changes that risk breaking existing behavior
- **Read before write**: Understand existing patterns; your changes must blend in with the surrounding codebase
- **One commit per logical change**: Atomic, revertible. No "wip" commits left on branches intended for PR

## Project Overview

Fork of ggml-org/llama.cpp adding **TurboQuant** KV-cache quantization (WHT rotation + PolarQuant centroid quantization) ported to the **Intel SYCL backend** for **Arc A770** (acm-g10, Xe-HPG/DG2).

TurboQuant compresses KV caches to 2/3/4-bit (`GGML_TYPE_TURBO2_0`/`TURBO3_0`/`TURBO4_0`, enum 43/44/45) using 128-element blocks. The graph applies forward-WHT to Q before attention and inverse-WHT to output after; FA kernels receive Q already rotated and only centroid-dequant K/V.

**Lineage**: upstream -> TheTom-llama-cpp-turboquant (CPU+CUDA+Metal+Vulkan oracle) -> this fork (SYCL port). CUDA has tensor-core turbo FA + SLM LUT + InnerQ; SYCL has VEC kernel (default) + experimental XMX/DPAS path (`GGML_SYCL_FA_XMX=1`, same-type K=V, D=128/256) + InnerQ hooks (dormant -- no calibration computes scale_inv).

**Performance reality**: The SYCL turbo path is dequant-compute-bound. turbo3 is slower than f16/q8_0 at every depth despite 5x smaller KV. Root cause: per-element centroid gather + scalar multiply vs q8_0's single dp4a per 4 elements. This is the central unsolved problem.

## Architecture & Data Flow

### TurboQuant Attention Pipeline

```
Q (f32/f16)
  -> ggml_turbo_wht(direction=0, group=128)     [forward WHT + InnerQ scale_inv]
  -> GGML_OP_FLASH_ATTN_EXT                     [VEC kernel; centroid-dequant K/V in-kernel]
  -> ggml_turbo_wht(direction=1, group=128)     [inverse WHT, self-inverse butterfly]
  -> [optional] ggml_view_3d strips V zero-padding
```

### Op Dispatch

`ggml-sycl.cpp` switch(op->op) at ~:4970-5280 calls `ggml_sycl_op_*` handlers:

- `GGML_OP_SET_ROWS` -> `ggml_sycl_op_set_rows()` -- KV cache write with turbo quantize
- `GGML_OP_FLASH_ATTN_EXT` -> `ggml_sycl_flash_attn_ext()` -- FA routing
- `GGML_OP_TURBO_WHT` -> `ggml_sycl_op_turbo_wht()` -- WHT butterfly

`MUL_MAT` routing: batch 1 -> DMMV; batch 2-8 -> MMVQ; batch >= 9 -> oneMKL GEMM. MMQ disabled.

### FA Kernel Routing

`fattn.cpp:ggml_sycl_get_best_fattn_kernel()` returns `{NONE=0, VEC=100, ONEDNN=150, TILE=200, XMX=300}`:

- **Turbo K or V**: VEC by default (D % 128 == 0). XMX opt-in for same-type K=V, D in {128, 256} (`GGML_SYCL_FA_XMX=1`). TILE does not support turbo.
- Non-turbo: VEC if D <= 512 && D % 64 == 0; TILE otherwise or GQA opt. XMX opt-in for f16/q8_0 same-type, D in {128, 256}.
- XMX kernel (`fattn-xmx.hpp`): off by default. Ignores ALiBi, softcap, sinks, multi-seq. SG=16 hits IGC ICE; SG=8 functional but 4-7x slower than VEC -- not production-viable yet.
- Master enable: `SYCL_FLASH_ATTN` macro (common.hpp:46).

### KV Cache Policy (src/llama-kv-cache.cpp)

- **Auto-asymmetric K downgrade** (:213-252): turbo K + GQA >= 6 + symmetric -> K downgraded to Q8_0. Override: `TURBO_AUTO_ASYMMETRIC=0`.
- **Layer-adaptive** (`TURBO_LAYER_ADAPTIVE` env): 0=uniform, 1-2=q8_0 boundary layers, 5-7=boundary-V variants.
- **Zero-padding**: non-128-aligned head_dim padded; WHT preserves inner products.
- **q8_0 quants-first**: `GGML_SYCL_Q8_KV_QUANTS_FIRST` env (SYCL, head_dim==128).

### Type System (ggml/include/ggml.h:435-440)

| Type | Enum | Block | Layout |
|------|------|-------|--------|
| TURBO2_0 | 43 | 34B | norm(f16) + qs[32] (2-bit) |
| TURBO3_0 | 44 | 50B | norm(f16) + qs[32] (low 2-bit) + signs[16] (high 1-bit) |
| TURBO4_0 | 45 | 68B | norm(f16) + rnorm(f16) + qs[64] (4-bit nibble) |
| TQ3_1S | 46 | 16B | weight quant, block=32 |
| TQ4_1S | 47 | 20B | weight quant, block=32 |
| COUNT | 48 | | |

All turbo KV: `QK_TURBO* = 128`. Block layouts in `ggml-common.h:260-343` with `static_assert` guards.

**ABI hazard**: Enum 42 is upstream's next free slot. GGUF serializes enums numerically -- collision risk on rebase.

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `ggml/src/ggml-sycl/` | SYCL backend (~110 files): FA kernels, turbo quant, op handlers, MMVQ/DMMV |
| `ggml/src/ggml-sycl/template-instances/` | 50 pre-compiled FA instantiation units (39 VEC + 11 TILE) |
| `ggml/src/` | ggml core: `ggml-turbo-quant.c` (CPU reference), `ggml-common.h`, `ggml-innerq.c` |
| `ggml/include/` | Public headers: `ggml.h`, `ggml-backend.h`, `ggml-innerq.h` |
| `src/` | llama core: `llama-graph.cpp` (WHT wiring), `llama-kv-cache.cpp` (turbo policy) |
| `common/` | Arg parsing, sampling, chat templates (Jinja), speculative decode |
| `tools/` | CLI: `llama-server`, `llama-bench`, `llama-perplexity`, `llama-completion` |
| `tests/` | Oracle gate + backend ops + turbo unit tests |
| `scripts/` | Bench harnesses, quality gates, sweep orchestrators |
| `docs/research/` | Per-campaign benchmark reports, build pins, fork-vs-upstream diffs |
| `docs/backend/SYCL.md` | Upstream SYCL docs port (39.7KB): build recipes, env vars, known-good stack |

## Development Commands

### Build (JIT -- development)

```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build-sycl -GNinja \
  -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER= \
  -DGGML_SYCL_F16=ON
ninja -C build-sycl
```

JIT: ~200s cold start on first run. `SYCL_CACHE_PERSISTENT=1` caches kernels.

### Build (AOT -- production)

```bash
cmake -B build-aot -GNinja \
  -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL_DEVICE_ARCH=acm-g10 -DGGML_SYCL_F16=ON
ninja -C build-aot   # ~14 min
```

### Key CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| `GGML_SYCL` | OFF | Enable SYCL backend |
| `GGML_SYCL_F16` | OFF | 16-bit float SYCL calculations |
| `GGML_SYCL_DEVICE_ARCH` | "" (JIT) | AOT target (`acm-g10`) |
| `GGML_SYCL_GRAPH` | ON | SYCL graph capture |
| `GGML_SYCL_DEVICE_CODE_SPLIT` | ON | Per-kernel device code split |
| `GGML_SYCL_SUPPORT_LEVEL_ZERO_API` | ON | Level Zero direct allocation |

`GGML_SYCL_WARP_SIZE=16` hardcoded for INTEL (`ggml-sycl/CMakeLists.txt:209`). Beware: some headers define `QK_WARP_SIZE`/`WARP_32_SIZE` as 32.

### Runtime Environment

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export SYCL_CACHE_PERSISTENT=1
export GGML_SYCL_DISABLE_GRAPHS=1          # default
# GGML_SYCL_FA_XMX=1                       # experimental XMX FA
# TURBO_AUTO_ASYMMETRIC=0                  # disable K downgrade
# TURBO_LAYER_ADAPTIVE=7                   # layer-adaptive KV
# GGML_SYCL_Q8_KV_QUANTS_FIRST=1           # q8_0 quants-first layout
```

### GPU Discipline (mandatory before timing runs)

```bash
sudo systemctl stop llama-sycl.cpp.service
fuser -v /dev/dri/renderD128               # verify sole tenancy
dmesg | grep -iE 'xe.*(reset|hang|timeout|GuC)'
# ... run benchmark wrapped in timeout ...
sudo systemctl start llama-sycl.cpp.service
```

### Benchmark Commands

```bash
# Product bench
./build-sycl/bin/llama-bench -m model.gguf -ngl 99 -fa 1 \
  -ctk turbo3 -ctv turbo3 -p 512 -n 128 -r 3

# Depth sweep
./build-sycl/bin/llama-bench -m model.gguf -ngl 99 -fa 1 \
  -ctk turbo3 -ctv turbo3 -p 0 -n 128 -d 0,4096,16384

# Perplexity
./build-sycl/bin/llama-perplexity -m model.gguf -ngl 99 -fa 1 \
  -ctk turbo3 -ctv turbo3 -f wikitext-2-raw/wiki.test.raw -c 4096

# Paired-CI A/B (sole tenancy required)
python3 scripts/bench-a770-fork-unique.py --campaign product \
  --candidate build-sycl --baseline build-baseline

# Full quality gate
bash scripts/turbo-quality-gate.sh
```

## Code Conventions & Patterns

### Naming

| Prefix | Scope | Example |
|--------|-------|---------|
| `ggml_sycl_op_*` | Backend op handler | `ggml_sycl_op_turbo_wht` |
| `ggml_sycl_flash_attn_ext*` | FA entry/variants | `ggml_sycl_flash_attn_ext_vec_case` |
| `flash_attn_ext_*` | Internal FA helpers | `flash_attn_ext_vec<D, ncols, type_K, type_V>` |
| `k_*` | Device kernel lambdas | `k_turbo_wht_f32_sycl` |
| `turbo_*` | TurboQuant helpers | `turbo_nearest_centroid_3bit` |
| `quantize_row_turbo*_ref` | CPU reference quantizers | `quantize_row_turbo3_0_ref` |
| `dequantize_turbo*` | Device inline dequant | `dequantize_turbo3_0` |

### File Naming

- `ggml-sycl/<op>.{cpp,hpp}` -- op pair
- `fattn-{vec,tile,xmx,onednn}.{hpp,cpp}` -- FA kernel families
- `template-instances/fattn-{vec,tile}-instance-<types>.cpp` -- explicit instantiation
- `turbo-{wht,quants}.{cpp,hpp}` -- TurboQuant

### Error Handling

Two-level model, no silent exception propagation:

1. `GGML_ASSERT(cond)` -- precondition bugs (type/shape/alignment)
2. `GGML_ABORT(fmt, ...)` -- unrecoverable runtime mismatch
3. `SYCL_CHECK(expr)` -- wraps every SYCL API call -> `GGML_ABORT` on failure
4. `CHECK_TRY_ERROR(expr)` -- catches `std::exception`, returns `dpct::err0`

### SYCL Kernel Conventions

- `[[sycl::reqd_sub_group_size(WARP_SIZE)]]` on kernel lambdas (WARP_SIZE=16)
- Reductions: `dpct::permute_sub_group_by_xor` + `sycl::group_barrier`
- SLM: `syclex::work_group_static<char[N]>` (TILE) or `local_accessor` (WHT)
- VEC template: `flash_attn_ext_vec<D, ncols, type_K, type_V, q8_quants_first, warp_size>`
- `nthreads_KQ = min(D/4, warp_size)` for quantized K
- `static_assert(warp_size % nthreads_KQ == 0)` guards

### Graph Construction (src/llama-graph.cpp)

```cpp
// Forward WHT on Q (direction=0)
q = ggml_turbo_wht(ctx0, q, 0, 0, innerq_scale);
// Inverse WHT on output (direction=1)
cur = ggml_turbo_wht(ctx0, cur, 1, turbo_group, innerq_scale);
```

Gate: `v->type == GGML_TYPE_TURBO{2,3,4}_0`. Three call sites: standard KV, K-only, ISWA.

## Important Files

### SYCL Backend

| File | Role |
|------|------|
| `ggml-sycl/ggml-sycl.cpp` | Entry: device init, op dispatch, mul_mat routing, env flags (273KB) |
| `ggml-sycl/common.hpp` | Macros: `WARP_SIZE`, `SYCL_FLASH_ATTN`, `SYCL_CHECK` |
| `ggml-sycl/fattn.cpp` | FA routing: `ggml_sycl_get_best_fattn_kernel()` |
| `ggml-sycl/fattn-common.hpp` | Shared FA: `vec_dot_KQ`, `dequantize_V` dispatch (53KB) |
| `ggml-sycl/fattn-vec.hpp` | VEC kernel: register-tile, nthreads, reductions (29KB) |
| `ggml-sycl/fattn-tile.hpp` | TILE kernel: SLM, barrier per KV iter (57KB) |
| `ggml-sycl/turbo-quants.hpp` | Device centroid tables, dequant/quantize helpers |
| `ggml-sycl/turbo-wht.cpp` | WHT kernel: forward/inverse, group 32/64/128 |
| `ggml-sycl/set_rows.cpp` | SET_ROWS turbo quantize dispatch (:428-454) |
| `ggml-sycl/innerq.cpp` | K-squared profile kernel + C fallback |
| `ggml-sycl/convert.cpp` | to_fp16/to_fp32 with turbo branches (:766, :850) |

### Llama Core

| File | Role |
|------|------|
| `src/llama-graph.cpp` | WHT wiring around `build_attn_mha()` |
| `src/llama-kv-cache.cpp` | Auto-asymmetric, layer-adaptive, zero-padding, rotation init |
| `ggml/src/ggml-turbo-quant.c` | CPU reference: centroids, WHT, quantize/dequantize |
| `ggml/src/ggml-innerq.c` | InnerQ host policy: decide, k_squared_scale, recovery |
| `ggml/src/ggml-common.h` | Block layouts + static_asserts (:260-343) |

## Runtime/Tooling

### Required Stack (known-good on A770)

- **Compiler**: oneAPI `icpx`/`icx` 2026.0 (NOT open-source clang++ -- produces unrunnable binaries)
- **IGC**: 2.36.3+
- **compute-runtime**: 26.22.x
- **level-zero-loader**: 1.28.6+
- **Kernel driver**: i915 (xe blacklisted on this host)
- **Build**: CMake + Ninja
- **GPU**: Arc A770 16GB (acm-g10, DG2, Xe-HPG)

### Build Gotchas

1. **sccache substitution**: Pass empty `-DCMAKE_C_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER=` to prevent sccache replacing icpx with c++
2. **JIT cold start**: ~200s first run. `SYCL_CACHE_PERSISTENT=1` for warm, `=0` for cold benchmarks
3. **AOT time**: ~14 min spir64_gen device link. Use JIT for iteration
4. **mergerfs ENOSPC**: Builds on `/mnt/mrgr` can fail. Use ZFS-backed dirs
5. **oneDNN**: Installed DNNL is CPU-only. All TUs compile `-DGGML_SYCL_DNNL=0`. Prefill GEMM = oneMKL, not oneDNN

## Testing & QA

### Primary Gate: test-sycl-turbo-correctness

`tests/test-sycl-turbo-correctness.cpp` (1565 lines). CPU-vs-SYCL oracle. **No external model files** -- all synthetic data with fixed seeds.

| Section | Tests |
|---------|-------|
| [1/1b] | WHT isolation (group 64/128, with/without scale_inv) |
| [2/2b/2c] | CPY turbo->F32, SET_ROWS quantize, Q8_0 layout |
| [3] | MUL_MAT turbo (MMVQ single column) |
| [4/4b] | FA turbo (gated), non-FA path |
| [5/5b] | FA f16 baseline, TILE sweep |
| [6/6b] | VEC FA sweep, GQA 4:1/8:1 |
| [7] | FA d=256 (opt-in, known-hang) |
| [8] | InnerQ state machine + K-squared profile |

**Oracle metrics**: nmse, cosine, norm_ratio, max_abs. Tiers: `Tol::STD` (nmse < 1e-3, cosine > 0.999) for exact paths; `Tol::LOSSY` (cosine > 0.95, norm_ratio in [0.85, 1.15]) for turbo vs f16.

Exit: `(g_failures > 0 || g_xpass > 0) ? 1 : 0`. XPASS also fails.

### Running Tests

```bash
# Default (safe)
./build-sycl/tests/test-sycl-turbo-correctness

# Turbo FA (HANG RISK if kernel broken)
LLAMA_TEST_TURBO_FA=1 ./build-sycl/tests/test-sycl-turbo-correctness

# InnerQ (host-only, safe)
LLAMA_TEST_INNERQ=1 ./build-sycl/tests/test-sycl-turbo-correctness

# d=256 (KNOWN HANG on A770)
LLAMA_TEST_FA256=1 ./build-sycl/tests/test-sycl-turbo-correctness

# Backend ops filtered
./build-sycl/tests/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT

# ctest
ctest --test-dir build-sycl -L sycl --timeout 180 -V
```

`LLAMA_TEST_TURBO_FA` uses exact `strcmp(v, "1")` -- `"true"`/`"on"` do NOT enable.

### Other SYCL Tests

| Test | Purpose |
|------|---------|
| `test-sycl-turbo.cpp` | Smoke: SET_ROWS + MUL_MAT for turbo/TQ |
| `test-sycl-fuzz.cpp` | Random-index SET_ROWS fuzzer (10k iter) |
| `test-sycl-stress-deep.cpp` | Memory pressure (100k iter, 32k ctx, 512MB) |
| `test-turbo-innerq-runtime.cpp` | State machine: publish/consume/abort/freeze |
| `test-kv-cache-adaptive-mode.cpp` | Adaptive policy + Q8_0 repack round-trip |
| `test-turbo-quant.c` | C round-trip: quant->dequant->inverse-WHT |
| `test-backend-ops.cpp` | Exhaustive per-op cross-backend (432KB) |

### CI

`.github/workflows/build-sycl.yml`: FP32/FP16 matrix, oneAPI 2025.3.3, `continue-on-error: true`. Triggers on `ggml/src/ggml-sycl/**`.

## Performance Context

### Current State (measured on A770, Llama-3.1-8B Q4_K_M)

- turbo3 pp512: 854 t/s vs q8_0 1178 (-28%)
- turbo3 tg128 @ depth: -6% (d=0), -11% (d=4096), -17% (d=16384) vs q8_0
- f16 KV fastest at depth; q8_0 loses 32% to f16 @ d=16384
- Root cause: per-element centroid gather + scalar multiply vs dp4a

### Killed (do not re-propose without new external evidence)

| Avenue | Kill evidence |
|--------|--------------|
| SLM centroid LUT in FA VEC | Measured -8% at depth; Intel 16x4B bank conflicts + GRF spill |
| nthreads_KQ=1 for turbo | Bundled with reverted LUT; SG=16 occupancy didn't transfer |
| joint_matrix XMX @ SG=16 | Hard IGC ICE, reproduced 3x |
| SG=8 XMX FA | Functional but 4-7x slower than VEC |
| InnerQ trivial scales | Calibration yields [0.997, 1.000] -> no-op |
| turbo2/3 on MoE | PPL diverges/NaN |
| Cold-JIT/AOT/cache tuning | 0% steady-state |
| SYCL graph replay (current driver) | DG2 lacks `ext_oneapi_graph` aspect; per-token re-record is added work |
| dp4a intrinsic swap | Intrinsic doesn't exist in oneAPI 2026.0 (ESIMD-only) |

### Open Avenues (ranked by evidence strength)

1. **q8_0 depth regression** (-32% vs f16 @ 16k): f16 routes to TILE with GQA batching; q8_0 stuck on per-head VEC. Attribution experiment: force f16 to VEC, compare.
2. **RMS_NORM+MUL fusion**: SYCL is lone no-fusion backend. ~64-96 fewer launches/token. +2-5% tg expected.
3. **ngram-mod speculative decoding**: Measured 2.3-3.3x on code/multi-turn. Pure config, zero GPU code.
4. **Upstream cherry-picks**: fused top-k MoE (#25217), UAF fix (#24676), softmax clamp (#24941).
5. **q8_0 VEC -> TILE routing**: If attribution confirms routing gap, instantiate TILE for type_K=q8_0.

### Reference Repos (read-only, local)

| Path | Role |
|------|------|
| `/mnt/mrgr/llama-cpp-sycl-turbo/compare/llama.cpp` | Upstream baseline (no turbo) |
| `/mnt/mrgr/llama-cpp-sycl-turbo/TheTom-llama-cpp-turboquant` | CUDA/Vulkan/Metal turbo oracle |

Key CUDA files: `fattn-mma-turbo.cuh` (tensor-core FA), `fattn-vec.cuh` (SLM LUT + nthreads_KQ=1), `turbo-quant.cuh`, `turbo-innerq.cu`, `set-rows.cu`.
