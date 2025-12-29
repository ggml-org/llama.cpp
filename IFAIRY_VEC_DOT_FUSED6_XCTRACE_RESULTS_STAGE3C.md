# Stage 3C Results: VecDot register-pressure + prologue tuning (macOS M4)

This stage continues optimizing `ggml_vec_dot_ifairy_q16_K()` on Apple Silicon (AArch64 + NEON + DOTPROD), targeting the common `K=1536` (`nb=6`) vec_dot path (non-LUT).

Baseline for comparison: Stage 3B commit `e8319fa0`.

---

## 1) What changed

File: `ggml/src/ggml-cpu/arch/arm/quants.c`

### 1.1 Reduce asm register pressure (keep nibble-LUT decode)

- Keep the Stage 3B nibble-based decode (`mask 0x0F` + `ushr #4` + 4 LUT vectors).
- Restructure the DOTPROD asm to decode and consume:
  - low nibble → 4 decoded vectors (`wr0/wi0/wr1/wi1`)
  - then high nibble → 4 decoded vectors
  This reduces peak live temporaries compared to “decode all 8 vectors then sdot”.

### 1.2 Bind int32 accumulators to caller-saved NEON registers

To reduce prologue save/restore and stabilize register allocation, bind the 8 int32 accumulators to caller-saved regs:

- `v20..v26` + `v7`

### 1.3 Load scalar scales late

Move `w[0].d_real/d_imag` and (when `x_scales_uniform`) `x0_d_real/x0_d_imag` fp16→fp32 conversions to the end of the function to reduce hot-loop register pressure.

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

- `tmp/ifairy-stage3c/test-ifairy_opt3_fixedregs.txt`
- `tmp/ifairy-stage3c/test-ifairy_opt3_fixedregs_strict.txt`
- `tmp/ifairy-stage3c/vecdot_verify_opt3_fixedregs.txt`

---

## 3) Benchmark evidence

### 3.1 Microbench (vec_dot only, K=1536, x-scale=tensor)

Command:

```bash
./build-rel/bin/ifairy-vecdot-microbench --iters 50000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify
```

Notes:

- `ns/iter` is DVFS/thermal sensitive on macOS; Stage 3C uses CPU Counters cycles for tighter A/B.

Artifacts:

- Stage 3C runs: `tmp/ifairy-stage3c/vecdot_microbench_opt3_fixedregs.txt`, `tmp/ifairy-stage3c/vecdot_microbench_opt3_runs3.txt`
- Stage 3B reference: `tmp/ifairy-stage3b/vecdot_microbench_opt.txt`

### 3.2 CPU Counters (cycle-based A/B)

Command:

```bash
xcrun xctrace record --template 'CPU Counters' --time-limit 20s \
  --output tmp/xctrace/vecdot_cpu_counters_opt3.trace \
  --launch -- ./build-rel/bin/ifairy-vecdot-microbench --iters 50000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify

xcrun xctrace export --input tmp/xctrace/vecdot_cpu_counters_opt3.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="MetricAggregationForThread"]' \
  | python3 scripts/ifairy_xctrace_cpu_counters_summary.py
```

Summaries:

- Stage 3B: `tmp/ifairy-stage3b/vecdot_cpu_counters_opt.metric.summary.txt`
  - `cycles: 9635707075`
- Stage 3C: `tmp/ifairy-stage3c/vecdot_cpu_counters_opt3.metric.summary.txt`
  - `cycles: 9507182542`

### 3.3 CPU Profiler leaf summary (cycles-weighted samples)

Artifacts:

- Stage 3C: `tmp/ifairy-stage3c/vecdot_cpu_profiler_leaf_opt3.txt`
- Stage 3B reference: `tmp/ifairy-stage3b/vecdot_cpu_profiler_leaf_opt.txt`

### 3.4 llama-bench (user methodology: warmup enabled, repetitions=3)

Command:

```bash
GGML_IFAIRY_LUT=0 GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1 \
  ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf \
  --threads 4 --n-prompt 64 --n-gen 128 -ngl 0 --device none --repetitions 3
```

Artifacts:

- Stage 3B: `tmp/ifairy-stage3b/llama-bench_opt.txt`
- Stage 3C: `tmp/ifairy-stage3c/llama-bench_opt3_fixedregs.txt`

