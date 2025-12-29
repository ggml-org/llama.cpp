# Stage 3B Results: VecDot nibble-LUT decode (reduce decode ops) on macOS M4

This stage iterates on `ggml_vec_dot_ifairy_q16_K()` (AArch64 + NEON + DOTPROD) and focuses on the common `K=1536` (`nb=6`) case used by iFairy.

Baseline for comparison: Stage 3 (commit `d597471f`) before this patch.

---

## 1) What changed

File: `ggml/src/ggml-cpu/arch/arm/quants.c`

### 1.1 Nibble-based weight decode (keep LUT, reduce shifts/masks)

The DOTPROD asm path decodes 2-bit weight indices using:

- `and v1, v0, 0x0F` (low nibble)
- `ushr v2, v0, #4` (high nibble)

Then uses 4 LUT vectors:

- low nibble:
  - `lut_real` / `lut_imag`
- high nibble:
  - `lut_wr_idx1` / `lut_wi_idx1`

This removes the previous pattern of `mask 0x3` + shifts (`#2/#4/#6`) for `idx1/idx2/idx3`, reducing decode ALU ops inside the hot loop while keeping the semantic invariant unchanged: `w * conj(x)`.

---

## 2) Validation

Build:

```bash
cmake --build build-rel --target ggml-cpu test-ifairy ifairy-vecdot-microbench llama-bench -j $(sysctl -n hw.ncpu)
```

Tests:

```bash
./build-rel/bin/test-ifairy
GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_VALIDATE_STRICT=1 ./build-rel/bin/test-ifairy
```

Microbench correctness check (diff vs generic kernel):

```bash
./build-rel/bin/ifairy-vecdot-microbench --iters 1000 --x-scale tensor
```

Raw logs:

- `tmp/ifairy-stage3b/test-ifairy_nibble.txt`
- `tmp/ifairy-stage3b/test-ifairy_nibble_strict.txt`
- `tmp/ifairy-stage3b/vecdot_verify_nibble.txt`

---

## 3) Benchmark evidence

### 3.1 Microbench (vec_dot only, K=1536, x-scale=tensor)

Command:

```bash
./build-rel/bin/ifairy-vecdot-microbench --iters 50000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify
```

Results:

- Baseline: `54.82 ns/iter` (`tmp/ifairy-stage3b/vecdot_microbench_base.txt`)
- Stage 3B: `47.47 ns/iter` (`tmp/ifairy-stage3b/vecdot_microbench_opt.txt`)

### 3.2 llama-bench (user methodology: warmup enabled, repetitions=3)

Command:

```bash
GGML_IFAIRY_LUT=0 GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1 \
  ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf \
  --threads 4 --n-prompt 64 --n-gen 128 -ngl 0 --device none --repetitions 3
```

Results:

- Baseline: pp64 `127.15 ± 12.02`, tg128 `83.89 ± 0.73` (`tmp/ifairy-stage3b/llama-bench_base.txt`)
- Stage 3B: pp64 `155.88 ± 0.90`, tg128 `90.58 ± 0.37` (`tmp/ifairy-stage3b/llama-bench_opt.txt`)

Note: `llama-bench` variance on macOS can be affected by DVFS/thermal; use microbench + CPU Counters for tighter A/B.

---

## 4) xctrace sampling evidence (ifairy-vecdot-microbench)

### 4.1 CPU Counters (cycles + derived bottleneck ratios)

Command:

```bash
xcrun xctrace record --template 'CPU Counters' --time-limit 20s \
  --output tmp/xctrace/vecdot_cpu_counters_opt.trace \
  --launch -- ./build-rel/bin/ifairy-vecdot-microbench --iters 50000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify

xcrun xctrace export --input tmp/xctrace/vecdot_cpu_counters_opt.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="MetricAggregationForThread"]' \
  | python3 scripts/ifairy_xctrace_cpu_counters_summary.py
```

Summaries:

- Baseline: `tmp/ifairy-stage3b/vecdot_cpu_counters_base.metric.summary.txt`
  - `duration_ns: 2764000000`, `cycles: 11038939699`
- Stage 3B: `tmp/ifairy-stage3b/vecdot_cpu_counters_opt.metric.summary.txt`
  - `duration_ns: 2409000000`, `cycles: 9635707075`

### 4.2 CPU Profiler (cycles-weighted leaf samples)

Command:

```bash
xcrun xctrace record --template 'CPU Profiler' --time-limit 20s \
  --output tmp/xctrace/vecdot_cpu_profiler_opt.trace \
  --launch -- ./build-rel/bin/ifairy-vecdot-microbench --iters 50000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify

xcrun xctrace export --input tmp/xctrace/vecdot_cpu_profiler_opt.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="cpu-profile"]' \
  | python3 scripts/ifairy_xctrace_leaf_cpu_profile.py --top 20
```

Leaf summaries:

- Baseline: `tmp/ifairy-stage3b/vecdot_cpu_profiler_leaf_base.txt`
- Stage 3B: `tmp/ifairy-stage3b/vecdot_cpu_profiler_leaf_opt.txt`

