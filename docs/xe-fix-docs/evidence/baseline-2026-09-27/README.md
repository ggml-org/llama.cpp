# SYCL on Arc A770: oneAPI 2026.0 vs 2026.1, oneDNN, and fork b12275 -> b12305

Benchmarks run 2026-09-27 and 2026-09-28 while rebuilding the local
`llama.cpp-sycl-f16-git` package (`/mnt/mrgr/llama.cpp-sycl-f16-git`, branch
`feature/raudbjorn-fork-arc-tuning`) from the Raudbjorn/ggml-llama.cpp fork.

**Bottom line**

- **oneAPI 2026.1 over 2026.0:** pp512 at depth 0 went from about 54 to about
  124 t/s (2.3x) in both rounds. tg64 improved about 7% in round 2; round 1 was
  noisier. Prefill at 8k depth tied in the cleaner round.
- **oneDNN (`GGML_SYCL_DNN=ON`, oneAPI 2026.1):** pp512 at depth 0 was 45-65%
  slower, with +/-30-40 t/s spread in both rounds. It gave no convincing gain
  elsewhere. Kept **off**.
- **Fork `4e7400c3a` (b12305) over `bb6908513` (b12275), both on 2026.1 without
  oneDNN:** tg64 +2-4% (18.1 vs 17.4-17.8), pp512@8k +1-8% depending on
  which rounds are paired (107-109 vs 101-106). tg64@8k and clean-run pp512 at
  depth 0 are ties.
- **Installed now:** `b12305.4e7400c3a-1` on intel-deep-learning-essentials
  2026.1.4 (icpx 2026.1.1), oneDNN off, SYCL graphs on.

## Method (identical for every run below)

`bench.sh` (in `raw/`) runs offline `llama-bench` with the systemd Ornith
service stopped, then restarts it.

```
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 \
llama-bench -m /mnt/ssd2/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf \
  -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 \
  -p 512 -n 64 -d 0,8192 -r 5 -t 12 -o md
```

- **Model:** Ornith-1.5-35B-A3B Q4_K_M (`qwen35moe`, 35.51 B params,
  20.21 GiB). This is the production Ornith config: q8_0 KV, FA on, `--fit`
  expert placement.
- **Host:**
  - Ryzen 9 7900X3D, 64 GB RAM.
  - Arc A770 16 GB (i915), headless compute device.
  - intel-compute-runtime 26.35.39758.10.
  - intel-graphics-compiler 2.41.5.
- **Value format:** mean +/- standard deviation over 5 repetitions, in t/s.
  pp512 is prompt processing of 512 tokens; tg64 is generation of 64 tokens.
  `@8k` means at a KV depth of 8192.
- **Build flags:**
  - Common to all builds: Release, icx/icpx, `GGML_NATIVE=ON`,
    `GGML_SYCL_F16=ON`, AOT `GGML_SYCL_DEVICE_ARCH=acm-g10`,
    `GGML_SYCL_GRAPH=ON`, `DEVICE_CODE_SPLIT=ON`, Level Zero API on,
    `LLAMA_SUBPROCESS=ON`, makepkg `!buildflags !lto`.
  - Only the oneAPI version and `GGML_SYCL_DNN` differ, as listed per variant.

## Variants

| Variant | Fork commit | oneAPI toolchain | oneDNN | How it was run |
|---|---|---|---|---|
| A | `bb6908513` (b12275) | 2026.0 (intel-dpcpp-cpp-compiler 2026.0.0) | off | Package binary extracted to `A-pkg/`, run with 2026.0 runtime libs from the rollback kit via `LD_LIBRARY_PATH` (`raw/A.ldpath`). The system already had 2026.1. |
| C | `bb6908513` (b12275) | 2026.1 (DLE 2026.1.4, icpx 2026.1.1) | off | From its build dir |
| D | `bb6908513` (b12275) | 2026.1 | **on**: Intel oneDNN 2026.0.2, SYCL GPU runtime, `GGML_SYCL_DNNL: yes` | From its build dir |
| N | `4e7400c3a` (b12305) | 2026.1 | off | Installed package (`/usr/bin`) |

A 2026.0 + oneDNN variant was not possible on this host. Arch's `onednn`
3.11.3 is CPU-only (`DNNL_GPU_RUNTIME NONE`). That rules out this setup, not a
GPU-enabled 2026.0 oneDNN in general.

C and D passed `test-sycl-turbo-correctness` with `GGML_SYCL_ENABLE_GRAPH=1`
(0 GATE-FAIL, 0 SKIP; `raw/gate-C.log`, `raw/gate-D.log`). So did N
(`raw/gate-6.log`).

## Valid results

### Round 1: 2026-09-27, order A -> C -> D

Load before each run: A 2.26, C 11.07, D 8.60. The C and D figures are
mostly the decaying load of the preceding 12-thread `llama-bench`.

| Variant | pp512 | tg64 | pp512 @8k | tg64 @8k |
|---|---|---|---|---|
| A (2026.0) | 53.65 +/- 20.67 | 13.58 +/- 3.34 | 81.40 +/- 11.49 | 12.57 +/- 5.36 |
| C (2026.1) | 124.90 +/- 7.37 | 17.41 +/- 0.47 | 100.57 +/- 1.36 | 16.91 +/- 0.33 |
| D (2026.1 + oneDNN) | 68.61 +/- 40.92 | 14.93 +/- 4.70 | 105.11 +/- 4.54 | 12.97 +/- 6.65 |

### Round 2: 2026-09-27, reversed order D -> C -> A

Load before each run: D 2.43, C 6.30, A 7.00.

| Variant | pp512 | tg64 | pp512 @8k | tg64 @8k |
|---|---|---|---|---|
| D (2026.1 + oneDNN) | 43.64 +/- 30.06 | 17.56 +/- 0.17 | 104.48 +/- 3.75 | 16.48 +/- 0.58 |
| C (2026.1) | 123.46 +/- 5.86 | 17.83 +/- 0.11 | 105.90 +/- 3.29 | 16.36 +/- 0.52 |
| A (2026.0) | 54.37 +/- 22.59 | 16.57 +/- 1.08 | 106.69 +/- 1.45 | 15.87 +/- 0.52 |

### Fork update: 2026-09-28, N `4e7400c3a`, two rounds

Load before each round: 1.05 and 0.71. Nothing else was on the GPU.

| Variant | pp512 | tg64 | pp512 @8k | tg64 @8k |
|---|---|---|---|---|
| N round 1 | 62.38 +/- 23.82 | 18.12 +/- 0.05 | 108.74 +/- 1.23 | 17.19 +/- 0.32 |
| N round 2 | 125.30 +/- 5.67 | 18.13 +/- 0.15 | 107.13 +/- 1.93 | 16.81 +/- 0.70 |

### Summary against C

| Comparison | pp512 | tg64 | pp512 @8k | tg64 @8k |
|---|---|---|---|---|
| A -> C (oneAPI 2026.0 -> 2026.1) | **2.27-2.33x** | +7.6% (round 2); round 1 too noisy to state | tie in round 2 (105.9 vs 106.7) | +3-35%, round 1 noisy |
| C -> D (oneDNN on) | **-45% to -65%**, spread +/-30-40 | tie in round 2 | tie | tie in round 2 |
| C -> N (fork b12275 -> b12305) | tie on the clean round (125.3 vs 123.5-124.9) | **+2-4%** | **+1-8%** | tie |

**Caveats**

- **Load was not equal across runs.** Round 2 load ranged from 2.4 to 7.0, and
  A used an alternate runtime setup. The A -> C differences are gains observed
  on this host. They are not proof that 2026.1 code generation alone causes
  them.
- **oneDNN's slowdown repeated in both rounds**, but its size was not stable.
- **pp512 at depth 0 collapses intermittently.**
  - It happened to A (both rounds), D (both rounds) and N round 1, but never
    to C in these runs.
  - It is the first test of each `llama-bench` invocation.
  - The fork's MoE-cache fit logs `MoE cache fit selected main-device dense
    placement with ~9.5 GiB projected cache capacity for 18600 MiB of routed
    expert weights (up to ~52% coverage)`.
  - A cold expert cache during the first repetitions is a plausible but
    **unverified** cause. A longer warm-up (`-w`) or ignoring the first test
    would separate a warm-up effect from a real stall.
- **Swap was 15/15 GiB full** during all 2026-09-27 runs.

## Invalid or contaminated runs (kept for the record, not for comparison)

| Run | When | Result (pp512 / tg64 / pp512@8k / tg64@8k) | Why it's invalid |
|---|---|---|---|
| A-2026.0-nodnn.contaminated | 09-27, ~17:48 | 37.06+/-18.92 / 5.20+/-3.17 / 62.88+/-28.43 / 13.91+/-2.42 | A slow `pacman -Qi` scan of every package ran concurrently |
| A-2026.0-nodnn | 09-27, 18:16 start | 39.77+/-22.38 / 3.98+/-1.55 / 62.37+/-22.56 / 14.99+/-0.63 | Load average 18.9 at start from another session's cargo/rustc builds; swap full |
| C-installed-quiet | 09-28, ~15:40 | 97.54+/-16.85 / 12.02+/-2.82 / 105.67+/-2.11 / 17.42+/-0.11 | A second, manually started Ornith `llama-server` held VRAM, RAM and port 8089 (see below). Raw files deleted; values are from the session transcript. |
| N-4e7400c3a (first) | 09-28, ~16:15 | 57.09+/-22.95 / 18.32+/-0.12 / 109.68+/-1.97 / 17.48+/-0.03 | Same second server, plus load 20.5. Raw files deleted; values are from the session transcript. |

The 2026-09-26 figures in
`~/projects/local-models/bench-2026-09-26-subproc.md` (pp512 105, tg64 17.3)
were taken through the running server with `llama-benchy`, not offline
`llama-bench`. They are only a rough reference for these numbers.

## What changed between runs

### Toolchain: oneAPI 2026.0 -> 2026.1 (09-27, between A and C/D)

- **Removed** (`pacman -Rdd`):
  - intel-dpcpp-cpp-compiler 2026.0.0
  - intel-oneapi-mkl 2026.0.0.909
  - intel-oneapi-common
  - intel-oneapi-compiler-shared-runtime 2026.0.0
  - onednn 3.11.3 (CPU-only)
  - oneccl-arc
  - intel-pti
- **Installed:** intel-deep-learning-essentials 2026.1.4-2. The local recipe
  is `~/projects/intel-pytorch/intel-deep-learning-essentials/PKGBUILD`. It
  includes compiler 2026.1.1, MKL 2026.1, and Intel oneDNN 2026.0.2 with a SYCL
  GPU runtime.
- **intel-pti 0.17 reinstalled alongside DLE.** python-pytorch-opt-xpu, kivi
  and gear link `libpti_view.so.0`, while DLE ships PTI 1.1 (`.so.1`).
  - The first DLE build declared an intel-pti conflict and broke
    `import torch`.
  - pkgrel 2 dropped that conflict.
- `libsycl` stayed at `.so.9`.
- **Post-swap checks passed:**
  - torch 2.14 XPU on the A770: 1024x1024 matmul max error 2e-4 against CPU.
  - kivi, gear and triton import.
  - `sycl-ls` shows the A770 over Level Zero.
- **Rollback kit:** `/mnt/nvme1/oneapi-2026.0-rollback`. The compiler package
  there was rebuilt from installed files; the original artifact and the Intel
  installer were gone.

### oneDNN: C vs D (09-27)

`GGML_SYCL_DNN` is OFF in C and ON in D; everything else is identical. The
PKGBUILD now has an `LLAMA_SYCL_DNN` toggle, default OFF (commit `d6dcf14`).

### Fork: bb6908513 -> 4e7400c3a (09-28, C vs N)

26 commits: https://github.com/Raudbjorn/ggml-llama.cpp/compare/589bf18cb...4e7400c3a
`589bf18cb` is `bb6908513` plus 4 packaging-only commits. Performance-relevant
content:

- SYCL Q5_K reorder MMVQ and fused GLU (`f045c2e50`), plus A770 row pairing
  from 6 columns (`0dca7bcae`).
- SYCL Q8_0 DMMV ESIMD and MMVQ wide load (`69db254ec`).
- Grouped MoE XMX GEMM for IQ weights (PR #67, `e985ccb9e`). Compiled out in
  this package; see below.
- PR #63: ngxson MUL_MAT_ID -1 skip and common_batch.
- PR #70: `GGML_SYCL_SEPARATE_BUILD`, default OFF, unused here.
- `--prefetch-experts-slots`, SYCL private copy queue. This is opt-in and was
  not enabled in these runs.
- Vulkan and doc changes, which don't affect this SYCL build.

Ornith is Q4_K_M, so the Q5_K and Q8_0 kernel changes may only touch the few
tensors in those types. The measured gain is small (+2-4% tg64), and I haven't
attributed it to a specific commit.

**Local patch required for N**
(`raw/0001-sycl-no-xmx-gather-for-dg2-aot.patch`, commit `28e9839`):

- **The failure:** PR #67's `fused_dequant_gemm_launch` and
  `grouped_dequant_gemm_launch` use a sub-group-16 8x16x16 fp16
  `joint_matrix`. AOT for acm-g10 crashed IGC 2.41.5 with
  `Internal Compiler Error: Floating point exception` on all 18 IQ-type
  instantiations (9 IQ types x 2 kernels).
  - It reproduced with both the ALHP and the generic Arch IGC.
  - The failing kernels were identified through an `ocloc` wrapper that saved
    the failing SPIR-V, followed by `spirv-dis`.
- **Why the patch is safe:** the runtime gate (`matrix_combinations`) already
  rejects DG2.
- **What it does:** under `-DGGML_SYCL_NO_XMX_GATHER`, the patch stubs the two
  entry points to return `false`.
- **Behaviour on the A770 is unchanged.**
- **Proper fix (fork):** skip these kernels for `acm-*` AOT targets, or add a
  sub-group-8, N=8 variant.

### Other state differences between runs

- **Serving instance:**
  - The systemd Ornith service (58 MCP tools plus server tools) was stopped by
    `bench.sh` during each run.
  - From 09-28 02:14 until 17:21, a manually started `llama-server` (old build
    C, no tools, `--moe-cache off`, from Claude Code session `e88a6564` in
    `/mnt/mrgr/upstream-prs`) held port 8089 and a full model copy in
    VRAM/RAM. This invalidated the two 09-28 afternoon runs listed above.
  - It also made the systemd service crash-loop (`couldn't bind HTTP server
    socket`, 561+ restarts).
- **Identity-question side note:** the same prompt ("who are you...") gets
  "I'm Claude, ... Anthropic" from the systemd instance with tools in context,
  and "Qwen3.5-30B-A3B" from the manual tool-less instance.
  - This comes from context and sampling (temp 0.6), not from the build.
  - Ornith is a Qwen3.5-MoE fine-tune.
  - It reproduced on the new build through the systemd instance.

## Files

- `raw/<run>/bench.md`, `raw/<run>/bench.err`: llama-bench output and
  backend/fit log for every valid run and the first two invalid A runs.
- `raw/bench.sh`, `raw/run-*.sh`, `raw/run-*.log`: runners with their
  quiet-host wait (load under 4 for 3 minutes) and the load recorded at each
  start.
- `raw/gate-*.log`: correctness gate outputs. `gate-5.log` belongs to the
  failed first build of N; its test binary did not exist, exit 127.
- Package history: `/mnt/mrgr/llama.cpp-sycl-f16-git` commits `d6dcf14`
  (oneAPI 2026.1 dependency names, oneDNN toggle) and `28e9839` (DG2 XMX
  patch, b12305).

## 2026-09-29: xe KMD follow-up moved

The A770 was switched from i915 to the xe kernel driver on 2026-09-29 and variant N
was re-run there (dense 8B A/B, five Ornith rounds, hangs, two blitter resets with
device coredumps, a production-server crash, encode and multi-CCS findings). That set
lives in `../sycl-oneapi-benchmarks-2026-09-29/` (README + `raw/N-xe-*`). Short
version: prefill +55-62 %, decode -20 to -29 %, 4 failures in 7 long-context
exercises; production Ornith stays on i915. The N r1/r2 rounds above are the i915
reference it compares against.
