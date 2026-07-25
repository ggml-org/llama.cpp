# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## What this repo is

Single-maintainer fork of `ggml-org/llama.cpp` carrying the **TurboQuant+** codec stack
(Walsh-Hadamard rotation + polar-codebook KV/weight quantization) with **Intel Arc A770
(Alchemist / acm-g10, Xe-HPG)** as the canonical target. See `README.md` for the codec/policy
rationale and the paper corpus.

- Backends shipped: **CPU, BLAS, SYCL, Vulkan, OpenVINO**. CUDA, HIP/ROCm, Metal, OpenCL, CANN,
  MUSA, WebGPU, RPC and Hexagon are **deleted from this tree** - do not add code paths for them,
  and do not be surprised when upstream merges delete large amounts of those directories.
- Never push or open PRs to upstream `ggml-org`. Per AGENTS.md, PRs go from the current branch to
  `master` of `Raudbjorn/ggml-llama.cpp` only.
- Style (AGENTS.md): ASCII only in code and comments (no em dash, `->` arrows, `x`, `...`), concise
  comments, reuse existing infrastructure over new subsystems, read surrounding code first.
  Commits carry an `Assisted-by:` trailer.
- This checkout usually sits under the multi-checkout workspace `/mnt/mrgr/llama-cpp-sycl-turbo/`,
  whose own `CLAUDE.md` describes the sibling reference repos and the autonomous-loop state files.
  In-repo `TOPOLOGY.md` is historical - trust live `git` output over it.

## Operating contract

**Precedence when instructions conflict:** current task intent > this file, `AGENTS.md`, and the
pinned toolchain versions in `docs/research/sycl-build-runtime-pins.md` > scoped file/platform
rules > global defaults. At the same level the more recent and more specific instruction wins.
Project conventions override style and tool defaults; they never override safety or integrity
rules. When a material conflict cannot be resolved from context or tools, state it plainly and ask
only for the missing decision.

**Evidence rules.** This repo has repeatedly lost sessions to stale premises - the perf ledger's own
conclusion is that avenues die at the boundary between narrative and hardware:

- Never claim a build, test, benchmark, deploy, or fix succeeded without tool output showing it.
  No fabricated verification, no "should work", no invented file:line citations.
- Evidence precedence when sources disagree: (1) live command output and current source,
  (2) dated `docs/research/` artifacts, (3) commit messages and prior-session narrative. Source and
  live output beat prose - **including the prose in this file**, which drifts across upstream merges.
- Prefer a five-minute probe (`grep`, `ocloc`, a compiled `aspect` query, `ldd`, `dmesg`,
  `fuser /dev/dri/renderD128`) over an argument. Type enum numbers, block sizes, env-var names,
  driver aspects, and which oneDNN runtime is installed have all changed under this fork; re-verify
  before relying on any of them.
- Report failures with the shortest decisive line of output, not a log dump. If a step was skipped,
  a test failed, or a number is contended, say so explicitly.

**Engineering defaults.** Simplest change that fixes the root cause; no speculative abstraction, no
unrequested scope, no rewriting unrelated files or reformatting untouched code. Lead with the
conclusion, patch, or command, then evidence, assumptions, material trade-offs, and what would
change the answer. Correct verifiable errors even when agreement is expected - and do not
manufacture disagreement to seem rigorous. Distinguish measured from analytic: label estimates as
estimates, since most expected-gain figures in the research corpus were arithmetic guesses that
measurement later refuted.

## Build

`setvars.sh` prints success but exports nothing on some hosts; use the explicit oneAPI env block:

```bash
OA=/opt/intel/oneapi
export CMPLR_ROOT="$OA/compiler/latest" MKLROOT="$OA/mkl/latest" TBBROOT="$OA/tbb/latest"
export PATH="$OA/compiler/latest/bin:$PATH"
export LD_LIBRARY_PATH="$OA/compiler/latest/lib:$OA/compiler/latest/opt/compiler/lib:$OA/mkl/latest/lib:$OA/tbb/latest/lib/intel64/gcc4.8:${LD_LIBRARY_PATH:-}"

cmake -S . -B ~/build-<name> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF \
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DMKL_DIR="$MKLROOT/lib/cmake/mkl" -DTBB_DIR="$TBBROOT/lib/cmake/tbb" \
  -DIntelSYCL_DIR="$CMPLR_ROOT/lib/cmake/IntelSYCL" \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF
cmake --build ~/build-<name> --target llama-server llama-completion llama-bench \
  test-sycl-turbo-correctness -j 20
```

Hard rules:

- **JIT by default (~200 s build, ~37 s one-time cold-JIT on first GPU launch, cached in
  `~/.cache`). AOT (`-DGGML_SYCL_DEVICE_ARCH=acm-g10`) is opt-in proof work** and takes ~45 min
  wall clean. Do not kill an AOT build under 45 min - the icpx/ocloc subtree restarts from zero.
- Build dirs belong on local/ZFS storage (`/home/...`), not on the mergerfs mount.
- Never build via `makepkg` with default flags: injected CFLAGS corrupt the SYCL device pipeline.
- **`GGML_SYCL_DNN=ON` is only a request.** The effective `GGML_SYCL_DNNL` compile definition and
  the runtime `GGML_SYCL_DNNL:` line are authoritative; a CPU-only oneDNN package yields
  `GGML_SYCL_DNNL=0` and disables the oneDNN FA/GEMM paths entirely.
- Fork-added CMake knobs: `GGML_SYCL_DEVICE_CODE_SPLIT` (default ON, `-fsycl-device-code-split=per_kernel`),
  build provenance JSON written when `GGML_SYCL_DEVICE_ARCH` is set. `GGML_SYCL_FA_ALL_QUANTS` is a
  compile define (not a CMake option) that widens FA type coverage.
- Vulkan/CPU-only builds are the quick way to smoke-test non-SYCL changes (`cmake -B build && cmake --build build -j`).

## Tests

Primary gate is the CPU-vs-SYCL differential harness `tests/test-sycl-turbo-correctness.cpp`
(target `test-sycl-turbo-correctness`): runs identical graphs on CPU (reference) and SYCL and
diffs with NMSE/cosine. Non-zero exit on any FAIL. Sections:

`[1]` WHT - `[2]` centroid decode (cpy turbo->f32) - `[3]` turbo mat-vec - `[4]` FA TILE prefill
(standard KV) - `[5]` turbo-KV FA (opt-in) - `[6]` FA VEC decode (standard KV) - `[7]` XMX FA /
d=256 (opt-in) - `[8]` InnerQ (opt-in).

```bash
timeout 600 ~/build-<name>/bin/test-sycl-turbo-correctness       # default sweep
LLAMA_TEST_TURBO_FA=1 timeout 600 ...                            # section [5] turbo FA
LLAMA_TEST_FA256=1    timeout 600 ...                            # d=256 (historically hangs)
LLAMA_TEST_INNERQ=1   timeout 600 ...                            # section [8]
```

Always wrap GPU runs in `timeout` - a bad kernel can hang the IGC JIT indefinitely.

Other fork-added targets: `test-sycl-turbo`, `test-sycl-fuzz`, `test-sycl-stress-deep`,
`test-stress-context` (SYCL-gated), `test-kv-cache-adaptive-mode`, `test-turbo-innerq-runtime`,
`test-turbo-quant.c`, `test-validate-dense-turbo4-capacity.sh`. Upstream `test-backend-ops` and
`test-quantize-fns` carry turbo cases.

```bash
ctest --test-dir ~/build-<name> -R test-kv-cache-adaptive-mode -V   # single test
scripts/turbo-quality-gate.sh                                       # pre-push correctness+PPL+ctx gate
pre-commit run --all-files                                          # whitespace/yaml/flake8
```

## Benchmarks and GPU discipline

Never hand-roll paired timing; use the harnesses catalogued in `scripts/README.md`:

- `scripts/bench-a770-fork-unique.py --campaign` - product mode: sole-tenancy gate (exit 70),
  alternating arms, 6 launches/arm with sample-zero discard, paired 95% CIs, dmesg fault gate.
- `scripts/bench-sycl-cold-jit.py` - cold-JIT campaigns (forces `SYCL_CACHE_PERSISTENT=0`).
- `scripts/sweep-a770-mmvq-geometry.py` - MMV_Y x MMVQ_NUM_SUBGROUPS geometry sweeps.
- `scripts/perf/bench_spec.py` - speculative-decoding acceptance/throughput.

Before any timing run: stop competing llama services, check `fuser /dev/dri/renderD128` (a foreign
holder such as a browser or compositor makes numbers noise, not data), and check
`sudo dmesg | grep -iE 'xe .*(reset|hang|timeout|GuC)'` before and after. Re-bench the baseline
binary alongside any candidate - stale baselines mislead, and two internal baselines have disagreed
by 1.75x on pp512 in the past.

## Architecture

### TurboQuant KV pipeline

Types `GGML_TYPE_TURBO2_0/3_0/4_0` = **43/44/45**, weight types `TQ3_1S`/`TQ4_1S` = 46/47
(`ggml/include/ggml.h`). Enum numbers are serialized into GGUF/session files - treat renumbering as
an ABI break, and re-check the slots after every upstream merge.

Block layouts live in `ggml/src/ggml-common.h`: turbo2 = 34 B, turbo3 = 50 B (2-bit `qs` + 1-bit
`signs` forming a split 3-bit index), turbo4 = 68 B with a compile switch `TURBO4_USE_4BIT`
(default 1 = 16-centroid nibble-packed; `rnorm` is a reserved field in that mode). All are
128-element blocks (`QK_TURBO*`), so **turbo FA requires head dims that are multiples of 128**;
the graph-level WHT additionally supports a 64-element rotation group for non-FA paths.

Data flow:

1. **Quantize (K/V store)** - `SET_ROWS`, SYCL kernel in `ggml/src/ggml-sycl/set_rows.cpp`, CPU
   reference in `ggml/src/ggml-turbo-quant.c`. L2-normalize each group, apply `TURBO_WHT_SIGNS1`,
   WHT butterfly, `(1/sqrt(128)) * TURBO_WHT_SIGNS2`, then nearest-centroid pack. The block's f16
   `norm` stores the *correction factor* `grp_norm / recon_norm`, not the raw norm.
2. **Graph-level rotation** - `GGML_OP_TURBO_WHT` (`ggml/src/ggml.c`, `ggml_turbo_wht`), wired in
   `src/llama-graph.cpp`: forward-WHT on Q before attention, inverse-WHT on the attention output.
   FA kernels therefore receive Q **already rotated** and must not rotate again.
3. **Dequantize** - `CENTROIDS[idx] * norm`, output stays in the **rotated domain**
   (`ggml/src/ggml-sycl/turbo-quants.hpp`, `ggml/src/ggml-sycl/dequantize.hpp`).
4. **InnerQ** per-channel equalization - CPU `ggml/src/ggml-innerq.c`, SYCL
   `ggml/src/ggml-sycl/innerq.cpp`, host state machine `src/llama-turbo-innerq-runtime.{h,cpp}`.
   The scale is threaded into the WHT op as `innerq_scale_inv` from the KV-cache context; a null
   scale makes the hook inert.
5. **Copy/convert** - `ggml/src/ggml-sycl/cpy.cpp` handles turbo<->turbo raw copies and
   turbo->f32 dequant-copies (used by defrag, state save/load, and harness section [2]).

### KV-cache policy layer (`src/llama-kv-cache.cpp`)

This file, not the kernels, decides which types each layer actually gets:

- **Auto-asymmetric K downgrade**: symmetric turbo K+V requests get K rewritten (typically to
  `q8_0`) on high-GQA models, because turbo K blows up PPL there (Qwen2.5 7:1 GQA measured 2887 vs
  7.4 baseline; Mistral 4:1 is fine). Downstream code must tolerate `K=q8_0, V=turbo*`.
- **Layer-adaptive modes** via `TURBO_LAYER_ADAPTIVE` (modes 1/2/5/6/7; "Boundary V" mode 7
  auto-enables for turbo2-V, opt out with `=0`). Modes are inert for non-turbo types and log a
  warning when requested inertly.
- **q8_0 "quants-first" KV layout**: `GGML_SYCL_Q8_KV_QUANTS_FIRST=1` repacks groups of 4 q8_0
  blocks so all quants precede all scales; the flag rides on the tensor and is queried by
  `ggml_tensor_is_kv_q8_quants_first()` (`ggml/include/ggml.h`). Kernels have distinct
  `*_quants_first` variants - any new q8_0 KV consumer must handle both layouts or reject one.

### SYCL flash-attention routing (`ggml/src/ggml-sycl/fattn.cpp`)

`ggml_sycl_get_best_fattn_kernel()` is the single decision point; kernels are VEC
(`fattn-vec.hpp`), TILE (`fattn-tile.hpp`), XMX/DPAS (`fattn-xmx.cpp`) and oneDNN Graph SDPA
(`fattn-onednn.cpp`), with shared dequant/combine helpers in `fattn-common.hpp` and scratch
management in `fattn-buffers.cpp`. Decision order, roughly:

1. Head-dim and type gates; without `GGML_SYCL_FA_ALL_QUANTS`, mixed K/V types are rejected except
   the `K=q8_0, V=turbo*` pair produced by the auto-asymmetric downgrade.
2. **Turbo K or V routes to VEC exclusively** (TILE turbo is unsupported: only VEC has complete
   K *and* V turbo dequant with `need_f16 = false`), gated to `K->ne[0] % 128 == 0`. Opt-in XMX
   turbo requires same turbo type on both sides and D in {128, 256} (D=512 exceeds the 64 KB SLM
   budget).
3. Forced overrides: `GGML_SYCL_FA_Q8_GQA_TILE`, `GGML_SYCL_FA_FORCE_VEC_STANDARD`.
4. Opt-in XMX for f16/q8_0 KV, then oneDNN SDPA if statically supported, else VEC/TILE by
   `Q->ne[1]` and `gqa_opt_applies`.

XMX and oneDNN paths are **off by default and feature-gated**: XMX ignores ALiBi, logit softcap,
attention sinks and multi-sequence batches, so those must fall through to VEC/TILE rather than
silently changing results.

### VEC kernel data contract (the historical footgun)

The kernel loads Q into **per-thread register slices** `Q_reg[ncols][(D/2)/nthreads_KQ]`. Every
`vec_dot_fattn_vec_KQ_*` receives that slice, *not* a full Q row, and must index it as
`Q_v[k_KQ_0/nthreads + k_KQ_1]` with K elements at `k_KQ_0 + (lane % nthreads) * cpy_ne`; partial
sums are combined by the caller's `warp_reduce_sum<nthreads_KQ>`. Reading `Q_v[i]` for `i in 0..D`
was the root cause of the historical "turbo FA garbage output + IGC JIT hang" bug.

SYCL `WARP_SIZE` is **16** on Intel (`GGML_SYCL_WARP_SIZE=16` for `GGML_SYCL_TARGET=INTEL`); the
unrelated `QK_WARP_SIZE` / `WARP_32_SIZE` macros are 32. ~17 files pin
`[[sycl::reqd_sub_group_size(WARP_SIZE)]]`, so SIMD-width env/compiler overrides are no-ops or
hazards.

### Runtime env knobs (fork-specific)

`GGML_SYCL_FA_XMX`, `GGML_SYCL_FA_XMX_DEBUG`, `GGML_SYCL_FA_ONEDNN`, `GGML_SYCL_FA_Q8_GQA_TILE`,
`GGML_SYCL_FA_FORCE_VEC_STANDARD`, `GGML_SYCL_FA_PROFILE` (per-route launch/us buckets),
`GGML_SYCL_GRAPH_PROFILE`, `GGML_SYCL_ROPE_FUSION_PROFILE`, `GGML_SYCL_Q8_KV_QUANTS_FIRST`,
`TURBO_LAYER_ADAPTIVE`. Read them through `ggml_sycl_get_env` (upstream helper) rather than bare
`getenv` in new code. Upstream knobs (`GGML_SYCL_ENABLE_GRAPH`, `GGML_SYCL_ENABLE_DNN`,
`GGML_SYCL_USE_LEVEL_ZERO_API`, `GGML_SYCL_DEBUG`, ...) keep their upstream meaning.

## Standing decisions - do not re-litigate without new evidence

Full evidence lives in `docs/research/` (dated artifacts, notably
`sycl-a770-p5-performance-campaign-2026-07-19.md`, `standard-sycl-baseline-2026-07-11.md`,
`sycl-build-runtime-pins.md`) and `turbo-fa-research-artifact.md`.

- **Turbo is a CAPACITY feature**, not a speed feature: more context or a bigger model in the same
  VRAM. Parity with f16/q8_0 decode t/s is not the bar, and the turbo FA-speed chase is closed.
- Measured dead ends, do not re-run without a driver/compiler change: SLM centroid-LUT dequant in
  the FA VEC path (-8% at depth), global large-GRF mode, non-PVC direct upload, exact SYCL graph
  replay, GPU-oneDNN prefill, alternate MMVQ geometry, DMMV/reorder rerouting, MoE reorder,
  radix-4 / tensor-core WHT (already at parity, and WHT is a graph op outside the FA hot loop).
- `joint_matrix` XMX at sub-group 16 hits an IGC internal compiler error on A770; SG=8 is verified
  viable but 4-7x slower than VEC, hence the XMX kernel ships off by default.
- SYCL-Graph replay cannot amortize on this driver: DG2 lacks `aspect::ext_oneapi_graph`, so each
  pass re-records and re-finalizes. Re-probe after any compute-runtime upgrade before reopening.
- Speculative decoding changing temperature-0 output is **expected upstream behavior** (kernels are
  not batch-invariant), not a fork bug - gate acceptance on logit tolerance, not exact hashes.
- Promoted and retained: per-kernel device-code split (default ON), opt-in
  `GGML_SYCL_Q8_KV_QUANTS_FIRST`, FA KV scratch buffers that pre-grow in 16 MiB chunks before
  graph capture (`fattn-buffers.hpp`), fused MoE `mul_mat_id` MMVQ, and the graph-fusion entry
  point `ggml_sycl_fuse` (`topk-moe.cpp`, called from `ggml-sycl.cpp`, gated by
  `GGML_SYCL_ENABLE_FUSION`) - which currently fuses top-k MoE only, so it is the hook to extend
  for any new fusion.
- Open and unresolved: q8_0 KV decode degrades vs f16 as context grows (~-32% at 16k). Attribution
  points at VEC-vs-TILE routing and per-element dequant cost, not missing dp4a. Any fix must keep
  the CPU oracle green.
