# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

Project roles in `.claude/agents/` and their Codex counterparts in `.codex/agents/` share the
[subagent roster and contract](docs/development/agents.md). Their Markdown bodies are shared
domain runbooks; keep the corresponding Codex description in sync when changing routing.

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
- Session record for the xe KMD blitter root cause, PR #88, the NEO patch, the large-GRF knob
  and PR #89 (2026-09-30/10-01):
  `~/.docs/2026-09-30-to-10-01-arc-a770-xe-kmd-blitter-root-cause-pr88-pr89-session/`
  (README index, timeline, every run with numbers, all review findings and dispositions,
  open items). The committed evidence is `docs/research/xe-kmd-bcs-copy-engine-2026-09-30.md`,
  `docs/research/sycl/sycl-fa-large-grf-2026-09-30.md` and
  `docs/research/patches/0001-neo-retry-userptr-bind-readonly-on-eperm.patch`. Read the
  session README before touching `xe-kmd.cpp`, the FA GRF code, or the production unit.

## Operating contract

**Precedence when instructions conflict:** current task intent > this file, `AGENTS.md`, and the
pinned toolchain versions in `docs/research/software-stack/sycl-build-runtime-pins.md` > scoped file/platform
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
- Fork-added CMake knobs: `GGML_SYCL_DEVICE_CODE_SPLIT` (default ON),
  `GGML_SYCL_XMX_GATHER` (AUTO/ON/OFF), and build provenance are documented in
  `docs/backend/SYCL.md`. `GGML_SYCL_FA_ALL_QUANTS` is a compile define, not a CMake option.
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
`test-quantize-fns` carry turbo cases. CPU-only (no GPU) tests for the two newest knobs:
`test-sycl-xe-defaults` (fake sysfs trees and the copy-engine env policy, Linux only) and
`test-sycl-fa-large-grf` (whole-value parse and launch decision). In builds with
`GGML_SYCL_FA_LARGE_GRF=ON`, `ctest -R large-grf` also runs the oracle through
`tests/check-sycl-fa-large-grf.cmake`, which fails unless the FA geometry profile shows a
`route=TILE phase=prefill ... grf=256` launch (a green sweep alone proves nothing there).

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

Before any timing run: stop competing llama services (the production unit is
`llama-gpu@Ornith-1.5-35B-Q4_K_M.service`; stop it first and restart it in a trap), check
`fuser /dev/dri/renderD128` (a foreign holder such as a browser or compositor makes numbers noise,
not data), and check `sudo dmesg | grep -iE 'xe .*(reset|Timedout|CAT)'` before and after.
Re-bench the baseline binary alongside any candidate - stale baselines mislead, and two internal
baselines have disagreed by 1.75x on pp512 in the past. Two more gates learned the hard way:
the product harness compares the binary's `build_commit` with the HEAD of the checkout it runs
from, so run it from the checkout (or worktree) that built the binary or the product is marked
invalid; and note the host load in every run header, because with host-resident MoE experts the
scheduler syncs with the GPU once per MoE layer per token, so a compile or a busy database on
the host halves decode (15.8 vs 30 t/s observed at load 26 on 24 threads) without any GPU-side
cause. Pass/fail and bind counts from a loaded run stand, its throughput does not.

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
- **q8_0 "quants-first" KV layout**: **default-on** since #33 (`754dc99e2`) for SYCL q8_0 K *and* V
  with 128-wide heads; `GGML_SYCL_Q8_KV_QUANTS_FIRST=0` opts out. It repacks groups of 4 q8_0
  blocks so all quants precede all scales; the flag rides on the tensor and is queried by
  `ggml_tensor_is_kv_q8_quants_first()` (`ggml/include/ggml.h`). Kernels have distinct
  `*_quants_first` variants - any new q8_0 KV consumer must handle both layouts or reject one,
  and any consumer that converts K/V through `ggml_get_to_fp16_sycl` /
  `ggml_get_to_fp16_nc_sycl` must pass the K/V tensor itself (the converters read the flag from
  it). The MKL FA route passed the dst tensor instead and decoded quants-first rows as canonical
  q8_0, which produced NaN in the validated Arc A770 q8_0/q8_0 workloads routed to MKL
  at n_kv >= 1024 (fixed 2026-09-05; oracle section [4c] now covers the route).

### SYCL flash-attention routing (`ggml/src/ggml-sycl/fattn.cpp`)

`ggml_sycl_get_best_fattn_kernel()` is the single decision point; kernels are VEC
(`fattn-vec.hpp`), TILE (`fattn-tile.hpp`), MKL GEMM prefill (`fattn-mkl.cpp`), XMX/DPAS
(`fattn-xmx.cpp`) and oneDNN Graph SDPA (`fattn-onednn.cpp`), with shared dequant/combine
helpers in `fattn-common.hpp` and scratch management in `fattn-buffers.cpp` (TILE/VEC only; MKL
allocates from the context pool). Decision order, roughly:

1. Head-dim and type gates; without `GGML_SYCL_FA_ALL_QUANTS`, mixed K/V types are rejected except
   the `K=q8_0, V=turbo*` pair produced by the auto-asymmetric downgrade.
2. **Turbo K or V routes to VEC exclusively** (TILE turbo is unsupported: only VEC has complete
   K *and* V turbo dequant with `need_f16 = false`), gated to `K->ne[0] % 128 == 0`. Opt-in XMX
   turbo requires same turbo type on both sides and D in {128, 256} (D=512 exceeds the 64 KB SLM
   budget).
3. **MKL prefill (default on, `GGML_SYCL_ENABLE_MKL_FA=0` disables)**: non-turbo K/V, mask
   present, no sinks/ALiBi/softcap, `gqa_ratio >= 2`, D a multiple of 64 in [64, 512],
   `Q->ne[1] >= 32` **and `K->ne[1] >= 1024`**, compatible batch dimensions, and valid
   type/stride conditions. Stages K/V to dense f16 per chunk through the tensor-aware
   converters, then oneMKL GEMM plus an f32 online softmax. In the validated Arc A770
   configuration, ctx-512 stays below the n_kv gate; larger contexts reach MKL only when
   all selector requirements hold. Test eligible shapes at n_kv >= 1024.
4. Forced overrides (decode only, `Q->ne[1] == 1`): `GGML_SYCL_FA_Q8_GQA_TILE`,
   `GGML_SYCL_FA_FORCE_VEC_STANDARD`. They do not affect prefill.
5. Opt-in XMX for f16/q8_0 KV (canonical rows only), then oneDNN SDPA if statically supported,
   else VEC/TILE by `Q->ne[1]` and `gqa_opt_applies`.

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

### xe KMD copy-engine default (`ggml/src/ggml-sycl/xe-kmd.{hpp,cpp}`)

On the xe kernel driver, every userptr `VM_BIND` of the read-only mmap'd GGUF fails with
`EPERM`; intel-compute-runtime (NEO) answers each failure with its eviction sweep, which unbinds
the blitter's KMD-submitted command buffer under a pending job; the blitter halts and the next
LR-mode suspend becomes `Engine reset: engine_class=bcs` after the 640 ms preempt timeout. The
kernel is not at fault. Three fixes exist, keep them straight:

- The hook: a load-time constructor in `xe-kmd.cpp` probes `/sys/class/drm` and, only for a
  DG2 device (`0x5690..0x56ff`) bound to `xe`, sets `UR_L0_USE_COPY_ENGINE=0` and
  `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` unless already set or unless any copy-engine
  variable asks for copy engines (non-zero on the v1 family or its `SYCL_PI_` aliases, 0 on
  the v2 variable). It also removes empty copy-engine variables there (the v1 adapter aborts
  at `dlopen` on an empty `UR_L0_USE_COPY_ENGINE_FOR_*`). `GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0`
  disables it. Log lines are queued and printed at backend init. Off DG2/xe the environment is
  untouched. Consequence: `--prefetch-experts-slots` is unavailable under the default (needs
  `UR_L0_USE_COPY_ENGINE=1`, refused on the v2 adapter regardless).
- `--load-mode none`: host-resident experts go to pinned `SYCL_Host` memory, so no userptr
  bind happens at all (prefill +12 %, decode flat, 6.65 GB owned RAM for Ornith). Production
  runs it.
- The NEO patch (`docs/research/patches/0001-*`): retry the userptr bind read-only on `EPERM`
  before the sweep. Verified A/B on one library (X7G clean, X7H control stalls). The host's
  `intel-compute-runtime-git` package carries it as patch 050 since 2026-09-30; the fork
  default stays until a compute-runtime release carries it, and the production unit opts out
  with the kill switch. If a runtime update drops the patch, put `UR_L0_USE_COPY_ENGINE=0`
  back in the unit's `xe-copy-engine.conf`.

### FA large-GRF knob (`ggml/src/ggml-sycl/fattn-grf.hpp`, `fattn.cpp`, `fattn-common.hpp`)

The FA tile kernels spill 8-15 KB per thread at 128 GRF on DG2 (IGC shader dumps).
`GGML_SYCL_FA_LARGE_GRF=1` requests `grf_size<256>` for tile launches with more than one query
row (prefill); single-row decode that routes to TILE and every vec launch stay at 128 GRF.
Needs the CMake option, a supported architecture (`acm_*`, `pvc*`, `bmg_*`, `lnl_m`; one
warning per device elsewhere), and the value must be exactly `0` or `1` (anything else warns
and is off). A 256-GRF launch plans with half the work-groups per Xe-core unless
`GGML_SYCL_MAX_WG_PER_CU` was set and accepted. Measured: pp512 +2.4 % (d=256) / +10 % (d=128)
at short context, decode flat; the gain sits below the MKL prefill gate (n_kv < 1024). Default
off. A tile-and-vec mode existed and was dropped: with the occupancy halving it lost 4-8 %
decode at 8k.

### Scheduler input-copy policy

`GGML_SCHED_COPY_SYNC` defaults to synchronous copies. Only the exact value `0`
opts into experimental stream-ordered copies; unset, `1`, empty, or other strings
keep synchronization. It is cached process-wide on first scheduler use, so set
it before launch. The opt-in requires single-device SYCL, a single-copy scheduler,
compatible non-mapped host input, and an upload-completion event for mutable
sources. Disabled mode allocates no extra upload event. CPU lifetime tests and the
A770 gate pass, but model-output equivalence remains unproven; do not enable this
by default or claim a speedup from the current evidence. For A/B runs use explicit
`0` versus `1` and require a nonzero `stream-ordered input copies` counter in the
opt-in arm (`-lv 5` for completion, `-v` for bench).
Full contract, examples, exclusions, and evidence:
[SYCL scheduler input-copy synchronization](docs/backend/SYCL.md#scheduler-input-copy-synchronization).

### Runtime env knobs (fork-specific)

`GGML_SYCL_FA_XMX`, `GGML_SYCL_FA_XMX_DEBUG`, `GGML_SYCL_FA_ONEDNN`, `GGML_SYCL_FA_Q8_GQA_TILE`,
`GGML_SYCL_FA_FORCE_VEC_STANDARD`, `GGML_SYCL_ENABLE_MKL_FA` (default 1), `GGML_SYCL_MKL_FA_DEBUG`,
`GGML_SYCL_MKL_FA_Q_TILE`, `GGML_SYCL_FA_PROFILE` (per-route launch/us buckets, now including the
MKL and ONEDNN routes), `GGML_SYCL_GRAPH_PROFILE`, `GGML_SYCL_ROPE_FUSION_PROFILE`,
`GGML_SYCL_Q8_KV_QUANTS_FIRST` (default on, `=0` opts out), `TURBO_LAYER_ADAPTIVE`,
`GGML_SYCL_XE_COPY_ENGINE_DEFAULT` (`=0` disables the xe hook), `GGML_SYCL_FA_LARGE_GRF`
(`0`/`1`, whole-value parse), `GGML_SYCL_MAX_WG_PER_CU` (occupancy target, strict parse).
Read them through `ggml_sycl_get_env` (upstream helper) rather than bare `getenv` in new
code; a knob that must reject `1junk`-style values parses the raw text itself, as
`fattn-grf.hpp` does. Upstream knobs (`GGML_SYCL_ENABLE_GRAPH`, `GGML_SYCL_ENABLE_DNN`,
`GGML_SYCL_USE_LEVEL_ZERO_API`, `GGML_SYCL_DEBUG`, ...) keep their upstream meaning.

## Standing decisions - do not re-litigate without new evidence

Full evidence lives in `docs/research/` (dated artifacts sorted by topic, index in
`docs/research/README.md`; notably `sycl/sycl-a770-p5-performance-campaign-2026-07-19.md`,
`sycl/standard-sycl-baseline-2026-07-11.md`, `software-stack/sycl-build-runtime-pins.md`
and `turbo/turbo-fa-research-artifact.md`).

- **Turbo is a CAPACITY feature**, not a speed feature: more context or a bigger model in the same
  VRAM. Parity with f16/q8_0 decode t/s is not the bar, and the turbo FA-speed chase is closed.
- Measured dead ends, do not re-run without a driver/compiler change: SLM centroid-LUT dequant in
  the FA VEC path (-8% at depth), global large-GRF mode, non-PVC direct upload, GPU-oneDNN prefill,
  alternate MMVQ geometry, DMMV/reorder rerouting, MoE reorder, radix-4 / tensor-core WHT (already
  at parity, and WHT is a graph op outside the FA hot loop).
- `joint_matrix` XMX at sub-group 16 hits an IGC internal compiler error on A770; SG=8 is verified
  viable but 4-7x slower than VEC, hence the XMX kernel ships off by default.
- **Superseded 2026-09-27 (PR #62):** the prior "SYCL-Graph replay cannot amortize" note described
  the old per-call re-record-and-`update()` design. `GGML_SYCL_ENABLE_GRAPH=1` now records once and
  replays for stable-shape decode; measured `graph_calls=1 replay_calls=46` on this driver
  (real decode, `GGML_SYCL_GRAPH_PROFILE=1`), byte-identical vs `GGML_SYCL_ENABLE_GRAPH=0`. DG2
  still lacks `aspect::ext_oneapi_graph` (in-place update support: `update_calls=0` always), so a
  property change forces a full re-finalize rather than an update - that remains the real limit,
  not an inability to amortize. `--n-cpu-moe`'s multi-graph-per-context map exists and falls back
  safely, but has not been observed actually holding more than zero live graphs (the one model
  tested had graph-incompatible dense layers for an unrelated reason - a library GEMM route).
- Speculative decoding changing temperature-0 output is **expected upstream behavior** (kernels are
  not batch-invariant), not a fork bug - gate acceptance on logit tolerance, not exact hashes.
- Promoted and retained: per-kernel device-code split (default ON), default-on
  `GGML_SYCL_Q8_KV_QUANTS_FIRST` (#33; measured +8 to +21% decode at depth <= 512, and since
  2026-09-05 also correct on the MKL prefill route), FA KV scratch buffers that pre-grow in 16 MiB chunks before
  graph capture (`fattn-buffers.hpp`), fused MoE `mul_mat_id` MMVQ, and the graph-fusion entry
  point `ggml_sycl_fuse` (`topk-moe.cpp`, called from `ggml-sycl.cpp`, gated by
  `GGML_SYCL_ENABLE_FUSION`) - which currently fuses top-k MoE only, so it is the hook to extend
  for any new fusion.
- **xe KMD blitter failure is a NEO residency defect, not a kernel bug (2026-09-30, X1 trace
  plus coredump, confirmed by the patch A/B X7G/X7H).** Do not re-open "is the kernel at
  fault" without a new coredump signature. `DirectSubmissionOverrideBlitterSupport=1` survives
  but decodes at 8.7 t/s: not a production option.
- **Per-kernel 256 GRF for FA tile prefill launches is a measured, retained opt-in**; the
  standing "global large-GRF is a dead end" decision is untouched, and the tile-and-vec mode
  was measured and dropped. Do not re-run mode 2 without a kernel-level change to the vec
  decode kernels.
- Open and unresolved: q8_0 KV decode degrades vs f16 as context grows (~-32% at 16k). Attribution
  points at VEC-vs-TILE routing and per-element dequant cost, not missing dp4a. Any fix must keep
  the CPU oracle green.
- Open: PR #89 left two post-merge bot threads (`strtol` accepts `" 1"`, `"+1"`, `"01"`);
  the upstream report to intel/compute-runtime is drafted, not filed. The partial
  `docs/xe-fix-docs/` archive and standalone diagnostic sources are committed; consult the
  archive manifest for omitted artifacts and label owned-memory probes as controls.
