# Stage 3D Results: Tensor-scale activation fusion for any `nb` (macOS M4)

This stage continues optimizing `ggml_vec_dot_ifairy_q16_K()` on Apple Silicon (AArch64 + NEON + DOTPROD), focusing on the **vec_dot (non-LUT)** path.

Key idea: when activations are quantized with **tensor-scale** (`GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1`), all `x[i].d_real/d_imag` are uniform across blocks. In that case we can **accumulate integer dot sums across all blocks** and apply the activation scale **once**, not per-block. Stage 3C only did this for `nb==6` (`K=1536`); Stage 3D generalizes it to **any `nb`** while preserving the `nb==6` fast-path.

Baseline for comparison: Stage 3C (commit `a6e511cc`).

---

## 1) What changed

File: `ggml/src/ggml-cpu/arch/arm/quants.c`

### 1.1 Generalize `x_scales_uniform` detection beyond `nb==6`

- Keep the existing fused integer accumulation path (`sum_*_total`) and final scaling.
- Extend the uniform-scale detection:
  - `nb==6`: keep the previous straight-line compares (fast path).
  - `nb!=6`: scan blocks until a mismatch is found.

This enables the fused path for any `K` (any multiple of 256) when the activation quantizer produces uniform scales across blocks.

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

Microbench correctness (diff vs generic kernel):

```bash
./build-rel/bin/ifairy-vecdot-microbench --k 1536 --iters 1000 --x-scale tensor
./build-rel/bin/ifairy-vecdot-microbench --k 4096 --iters 1000 --x-scale tensor
```

Raw logs:

- `tmp/ifairy-stage3d-final/test-ifairy.txt`
- `tmp/ifairy-stage3d-final/test-ifairy_strict.txt`
- `tmp/ifairy-stage3d-final/vecdot_verify_k1536.txt`
- `tmp/ifairy-stage3d-final/vecdot_verify_k4096.txt`

---

## 3) Benchmark evidence

### 3.1 Microbench (ns/iter; DVFS-sensitive)

Commands:

```bash
./build-rel/bin/ifairy-vecdot-microbench --k 1536 --iters 50000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify
./build-rel/bin/ifairy-vecdot-microbench --k 4096 --iters 20000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify
```

Artifacts:

- `tmp/ifairy-stage3d-final/vecdot_microbench_k1536.txt`
- `tmp/ifairy-stage3d-final/vecdot_microbench_k4096.txt`

### 3.2 CPU Counters (cycle-based A/B; focus on `K=4096`)

Stage 3C recheck (baseline):

- `tmp/ifairy-stage3c-recheck/vecdot_cpu_counters_k4096.metric.summary.txt`
- `cycles: 25587086872`

Stage 3D (this change):

- `tmp/ifairy-stage3d-final/vecdot_cpu_counters_k4096.metric.summary.txt`
- `cycles: 24766531526`

Δ cycles (lower is better): **-3.21%**

Record command:

```bash
xcrun xctrace record --template 'CPU Counters' --time-limit 20s \
  --output tmp/xctrace/vecdot_cpu_counters_stage3d_final_k4096.trace \
  --launch -- ./build-rel/bin/ifairy-vecdot-microbench --k 4096 --iters 50000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify

xcrun xctrace export --input tmp/xctrace/vecdot_cpu_counters_stage3d_final_k4096.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="MetricAggregationForThread"]' \
  > tmp/ifairy-stage3d-final/vecdot_cpu_counters_k4096.metric.xml

python3 scripts/ifairy_xctrace_cpu_counters_summary.py \
  < tmp/ifairy-stage3d-final/vecdot_cpu_counters_k4096.metric.xml \
  > tmp/ifairy-stage3d-final/vecdot_cpu_counters_k4096.metric.summary.txt
```

### 3.3 CPU Profiler (cycles-weighted leaf samples)

Artifacts:

- `tmp/ifairy-stage3d-final/vecdot_cpu_profile_k4096.xml`
- `tmp/ifairy-stage3d-final/vecdot_cpu_profiler_leaf_k4096.txt`

Command:

```bash
xcrun xctrace record --template 'CPU Profiler' --time-limit 10s \
  --output tmp/xctrace/vecdot_cpu_profiler_stage3d_final_k4096.trace \
  --launch -- ./build-rel/bin/ifairy-vecdot-microbench --k 4096 --iters 50000000 --warmup 2000 --seed 1 --x-scale tensor --no-verify

xcrun xctrace export --input tmp/xctrace/vecdot_cpu_profiler_stage3d_final_k4096.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="cpu-profile"]' \
  > tmp/ifairy-stage3d-final/vecdot_cpu_profile_k4096.xml

python3 scripts/ifairy_xctrace_leaf_cpu_profile.py \
  < tmp/ifairy-stage3d-final/vecdot_cpu_profile_k4096.xml \
  > tmp/ifairy-stage3d-final/vecdot_cpu_profiler_leaf_k4096.txt
```

### 3.4 llama-bench (warmup enabled, repetitions=3)

Command:

```bash
GGML_IFAIRY_LUT=0 GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1 \
  ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf \
  --threads 4 --n-prompt 64 --n-gen 128 -ngl 0 --device none --repetitions 3
```

Artifact:

- `tmp/ifairy-stage3d-final/llama-bench.txt`

---

## 4) Notes / next

- Next bottleneck iteration should stay focused on the asm kernel (`tbl` latency/throughput, `sdot` dependency chains, and register pressure) and should be validated via:
  - CPU Counters (cycles + bottleneck ratios)
  - CPU Profiler (where cycles concentrate)
  - `llama-bench` (pp/tg throughput, repetitions=3)

