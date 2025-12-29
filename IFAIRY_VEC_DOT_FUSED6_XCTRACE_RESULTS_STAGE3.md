# Stage 3 Results: VecDot Fused-6 (K=1536) Fastpath + NEON asm tuning (macOS M4)

This stage targets `ggml_vec_dot_ifairy_q16_K()` on Apple Silicon (AArch64 + NEON + DOTPROD), focusing on the common `K=1536` case (`nb=6` blocks).

Baseline for comparison: commit `bc90479d` (Stage 2).

---

## 1) What changed

### 1.1 Fused-6 accumulation (when x scales are uniform)

File: `ggml/src/ggml-cpu/arch/arm/quants.c`

When `nb == 6` and `x[0].d_real/d_imag` matches `x[1..5].d_real/d_imag`, the kernel:

- Accumulates integer dot sums across all 6 blocks:
  - `sum_ac_total`, `sum_ad_total`, `sum_bc_total`, `sum_bd_total`
- Applies activation scales `x_real/x_imag` **once** (instead of per-block)
- Keeps the semantic invariant `w * conj(x)` unchanged:
  - `Re = w_r * (x_r * Σac) + w_i * (x_i * Σbd)`
  - `Im = w_i * (x_r * Σbc) - w_r * (x_i * Σad)`

This is intended to pair with Stage 2 activation tensor-scale quantization (`GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1`), which makes `x[i].d_*` uniform for `K=1536`.

### 1.2 NEON asm load tuning

File: `ggml/src/ggml-cpu/arch/arm/quants.c`

In the DOTPROD asm hot loop, replaced 8x `ldr q*` loads of `xr/xi` with 4x `ldp q*,q*` pairs for each half-block, reducing load instruction count.

---

## 2) Validation

### 2.1 Build (Release)

```bash
cmake --build build-rel --target ggml-cpu test-ifairy ifairy-vecdot-microbench llama-bench -j $(sysctl -n hw.ncpu)
```

### 2.2 Unit tests

```bash
./build-rel/bin/test-ifairy
GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_VALIDATE_STRICT=1 ./build-rel/bin/test-ifairy
```

Raw logs:

- `tmp/ifairy-stage3/test-ifairy_stage3.txt`
- `tmp/ifairy-stage3/test-ifairy_stage3_strict.txt`

---

## 3) Benchmark evidence

### 3.1 Microbench (vec_dot only, K=1536, x-scale=tensor)

Command:

```bash
./build-rel/bin/ifairy-vecdot-microbench --iters 50000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify
```

Run-to-run noise exists (DVFS/thermal). For a more stable A/B, Stage 2 was built from a separate worktree at `bc90479d` and compared against current.

Observed results (3 runs each, ns/iter):

- Stage 2 (`bc90479d`): `56.85`, `59.67`, `60.93`
- Stage 3 (current): `55.16`, `55.74`, `55.93`

Artifacts:

- `tmp/ifairy-stage3/vecdot_microbench_baseline.txt` (short-run baseline)
- `tmp/ifairy-stage3/vecdot_microbench_opt.txt` (short-run after changes)

### 3.2 CPU Counters (cycles + bottleneck ratios)

This uses the standard `CPU Counters` template (not a custom template):

```bash
xcrun xctrace record --template 'CPU Counters' --time-limit 20s --output tmp/xctrace/vecdot_cpu_counters_current.trace \
  --launch -- ./build-rel/bin/ifairy-vecdot-microbench --iters 50000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify

xcrun xctrace export --input tmp/xctrace/vecdot_cpu_counters_current.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="MetricAggregationForThread"]' \
  | python3 scripts/ifairy_xctrace_cpu_counters_summary.py
```

Summaries (captured):

- Stage 2: `tmp/ifairy-stage3/vecdot_cpu_counters_stage2.summary.txt`
- Stage 3: `tmp/ifairy-stage3/vecdot_cpu_counters_current.summary.txt`

Notes:

- `cycles` is the hardware cycle counter aggregated over the run.
- `useful/processing/delivery/discarded` are the CPU Counters template’s derived metrics (they are **not** raw PMU event names).

---

## 4) xctrace CPU Profiler leaf summaries (cycles-weighted samples)

Stage 3 adds two helper scripts:

- `scripts/ifairy_xctrace_leaf_cpu_profile.py` (summarize `cpu-profile` leaf cycles)
- `scripts/ifairy_xctrace_cpu_counters_summary.py` (summarize `MetricAggregationForThread`)

Baseline (Stage 2) and Stage 3 cpu-profile exports live in:

- `tmp/ifairy-stage3/vecdot_cpu_profile_base.xml`
- `tmp/ifairy-stage3/vecdot_cpu_profile_opt.xml`
- `tmp/ifairy-stage3/llama_bench_cpu_profile_base.xml`
- `tmp/ifairy-stage3/llama_bench_cpu_profile_opt.xml`

Example usage:

```bash
xcrun xctrace export --input <trace>.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="cpu-profile"]' \
  | python3 scripts/ifairy_xctrace_leaf_cpu_profile.py --top 20
```

