# Stage 2 Results: VecDot Activation Tensor-Scale Quantization (macOS M4 / AArch64+NEON)

This file records the **Stage 2** implementation + validation evidence for:

- Goal: when using the **vec_dot** (non-LUT) path, quantize activations (`src1`) with a **tensor-scale** (full-K) scale, instead of per-256-block scales.
- Scope: Apple Silicon (AArch64 + NEON) fast path; reference implementation added for correctness parity.
- Runtime gate: `GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1` (default is off).

Commit base for this stage: `8fcf871e` (`tools: add ifairy vecdot microbench`).

---

## 1) What changed (Stage 2)

### New quantizer (tensor-scale over full K)

- Added ref function: `quantize_row_ifairy_q16_tensor_ref()` in:
  - `ggml/src/ggml-quants.h`
  - `ggml/src/ggml-quants.c`
- Added AArch64+NEON implementation: `quantize_row_ifairy_q16_tensor()` in:
  - `ggml/src/ggml-cpu/arch/arm/quants.c`
  - Declared in `ggml/src/ggml-cpu/quants.h`

Implementation model (2-pass):

1) Pass 1 scans the whole row (all `nb = K / QK_K` blocks) and finds a **global max** for real/imag.
2) Pass 2 quantizes each block using the shared scale and writes identical `d_real/d_imag` into every block.

### Integrate into vec_dot (non-LUT) conversion path

- `ggml/src/ggml-cpu/ggml-cpu.c`:
  - Added env knob `GGML_IFAIRY_VEC_DOT_ACT_TENSOR`.
  - When enabled and `src0->type == GGML_TYPE_IFAIRY` and `vec_dot_type == GGML_TYPE_IFAIRY_Q16`, activation conversion uses `quantize_row_ifairy_q16_tensor()`.

### Tests & microbench

- `tests/test-ifairy.cpp`:
  - Added **Test 1.1**: checks tensor-scale quantization produces uniform `d_real/d_imag` across all 6 blocks when `K=1536`.
- Added tool `ifairy-actq-microbench`:
  - `tools/ifairy-actq-microbench/ifairy-actq-microbench.cpp`
  - `tools/ifairy-actq-microbench/CMakeLists.txt`
  - Wired into `tools/CMakeLists.txt`

---

## 2) Build & unit tests

### Build (Release, targeted)

```bash
cmake --build build-rel --target ifairy-actq-microbench ifairy-vecdot-microbench test-ifairy llama-bench -j $(sysctl -n hw.ncpu)
```

### Unit test

```bash
./build-rel/bin/test-ifairy
```

### Strict LUT validation (should remain unaffected by this stage; still run as gate)

```bash
GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_VALIDATE_STRICT=1 ./build-rel/bin/test-ifairy
```

Result: both `test-ifairy` and strict mode passed.

---

## 3) Functional sanity (llama-bench)

Model used:

- `models/Fairy-plus-minus-i-700M/ifairy.gguf`

Commands:

```bash
GGML_IFAIRY_LUT=0 GGML_IFAIRY_VEC_DOT_ACT_TENSOR=0 ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 64 --n-gen 128 -ngl 0 --device none --repetitions 1 --no-warmup
GGML_IFAIRY_LUT=0 GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1 ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 64 --n-gen 128 -ngl 0 --device none --repetitions 1 --no-warmup
```

Raw outputs (captured):

- Baseline: `tmp/ifairy-stage2/llama-bench_vecdot_base.txt`
- Tensor-scale: `tmp/ifairy-stage2/llama-bench_vecdot_tensor.txt`

Summary table snippet:

```
baseline (GGML_IFAIRY_VEC_DOT_ACT_TENSOR=0)
  pp64   139.34 t/s
  tg128   83.48 t/s

tensor   (GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1)
  pp64   135.96 t/s
  tg128   82.69 t/s
```

Note: single-run `llama-bench` numbers can be noisy on Apple Silicon due to DVFS/thermal; this stage is correctness-first and uses xctrace to confirm routing.

---

## 4) xctrace sampling evidence (Time Profiler)

### 4.1 Key CLI rule: do NOT wrap the target in `/usr/bin/env`

If you do:

```bash
xcrun xctrace record --launch -- /usr/bin/env VAR=1 ./build-rel/bin/llama-bench ...
```

you mostly profile `/usr/bin/env` (startup), not `llama-bench`.

Correct usage is `--env` flags:

```bash
xcrun xctrace record --env VAR=1 --launch -- ./build-rel/bin/llama-bench ...
```

### 4.2 actq microbench (block vs tensor)

Commands:

```bash
xcrun xctrace record --template 'Time Profiler' --time-limit 30s --output tmp/xctrace/ifairy-actq_block.trace \
  --launch -- ./build-rel/bin/ifairy-actq-microbench --impl block --k 1536 --iters 6000000 --warmup 2000 --seed 1 --no-verify

xcrun xctrace export --input tmp/xctrace/ifairy-actq_block.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  | python3 scripts/ifairy_xctrace_leaf.py > tmp/xctrace/ifairy-actq_block.leaf.txt

xcrun xctrace record --template 'Time Profiler' --time-limit 30s --output tmp/xctrace/ifairy-actq_tensor.trace \
  --launch -- ./build-rel/bin/ifairy-actq-microbench --impl tensor --k 1536 --iters 6000000 --warmup 2000 --seed 1 --no-verify

xcrun xctrace export --input tmp/xctrace/ifairy-actq_tensor.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  | python3 scripts/ifairy_xctrace_leaf.py > tmp/xctrace/ifairy-actq_tensor.leaf.txt
```

Leaf highlights:

```
block  total_ms: 3815
  99.76%  quantize_row_ifairy_q16

tensor total_ms: 3760
  99.76%  quantize_row_ifairy_q16_tensor
```

### 4.3 llama-bench (vec_dot baseline vs tensor-scale activation quant)

Commands:

```bash
xcrun xctrace record --template 'Time Profiler' --time-limit 120s --output tmp/xctrace/llama-bench_vecdot_base_stage2.trace \
  --env GGML_IFAIRY_LUT=0 --env GGML_IFAIRY_VEC_DOT_ACT_TENSOR=0 \
  --launch -- ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 64 --n-gen 128 -ngl 0 --device none --repetitions 1 --no-warmup

xcrun xctrace export --input tmp/xctrace/llama-bench_vecdot_base_stage2.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  | python3 scripts/ifairy_xctrace_leaf.py > tmp/xctrace/llama-bench_vecdot_base_stage2.leaf.txt

xcrun xctrace record --template 'Time Profiler' --time-limit 120s --output tmp/xctrace/llama-bench_vecdot_tensor_stage2.trace \
  --env GGML_IFAIRY_LUT=0 --env GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1 \
  --launch -- ./build-rel/bin/llama-bench -m models/Fairy-plus-minus-i-700M/ifairy.gguf --threads 4 --n-prompt 64 --n-gen 128 -ngl 0 --device none --repetitions 1 --no-warmup

xcrun xctrace export --input tmp/xctrace/llama-bench_vecdot_tensor_stage2.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  | python3 scripts/ifairy_xctrace_leaf.py > tmp/xctrace/llama-bench_vecdot_tensor_stage2.leaf.txt
```

Leaf highlights (routing proof):

```
baseline (GGML_IFAIRY_VEC_DOT_ACT_TENSOR=0)
  65.06%  ggml_vec_dot_ifairy_q16_K
   0.54%  quantize_row_ifairy_q16

tensor   (GGML_IFAIRY_VEC_DOT_ACT_TENSOR=1)
  64.36%  ggml_vec_dot_ifairy_q16_K
   0.39%  quantize_row_ifairy_q16_tensor
```

Artifacts:

- `tmp/xctrace/ifairy-actq_block.trace`
- `tmp/xctrace/ifairy-actq_tensor.trace`
- `tmp/xctrace/llama-bench_vecdot_base_stage2.trace`
- `tmp/xctrace/llama-bench_vecdot_tensor_stage2.trace`
- `tmp/xctrace/*.leaf.txt` (leaf summaries)

---

## 5) Notes / known issues

- `ctest --test-dir build-rel ...` has pre-existing unrelated failures on this machine (e.g. `test-backend-ops` segfault, `test-thread-safety` model mismatch, and a curl-related failure). Stage 2 validation relies on `test-ifairy` + strict mode and the A/B functional `llama-bench` runs above.

