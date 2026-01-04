# MoE Prompt Processing Optimization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 4x MoE prompt processing regression while maintaining decode speedup, achieving performance parity with master on prompt AND better-than-master on decode.

**Architecture:** The regression is caused by our custom fused ESIMD MoE kernel being slower than master's oneDNN-batching approach for prompt processing. Master groups tokens by expert and calls `ggml_sycl_mul_mat` which uses oneDNN for batch >32 (282 t/s). Our fused kernel achieves only 87 t/s. Solution: fall back to oneDNN batching for large prompts while keeping MMVQ for decode.

**Tech Stack:** Intel SYCL, oneDNN GEMM, VTune profiler, llama-bench

---

## Benchmark Targets

| Model | Test | Master | Current | Target |
|-------|------|--------|---------|--------|
| GPT-OSS 20B Q8_0 | pp512 | 282 t/s | 87 t/s | ≥280 t/s |
| GPT-OSS 20B Q8_0 | tg128 | 14 t/s | 31 t/s | ≥31 t/s (maintain) |
| Mistral 7B Q4_0 | tg128 | 31 t/s | 42 t/s | ≥42 t/s (canary) |

---

### Task 1: Establish Baseline with VTune Profiling

**Files:**
- Read: `ggml/src/ggml-sycl/ggml-sycl.cpp:11192-11320` (fused MoE kernel)
- Read: `/Apps/llama.cpp/.worktrees/master-benchmark/ggml/src/ggml-sycl/ggml-sycl.cpp:3498-3600` (master batching)

**Step 1: Profile master branch prompt processing**

```bash
source /opt/intel/oneapi/setvars.sh --force
cd /Apps/llama.cpp/.worktrees/master-benchmark
ONEAPI_DEVICE_SELECTOR=level_zero:1 vtune -collect gpu-hotspots \
  -result-dir /tmp/vtune_moe_master -- \
  ./build/bin/llama-bench -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 0 -ngl 99 -fa 1
```

Expected: See oneDNN GEMM kernels dominating GPU time.

**Step 2: Profile our branch prompt processing**

```bash
cd /Apps/llama.cpp/.worktrees/sycl-coalescing
ONEAPI_DEVICE_SELECTOR=level_zero:1 vtune -collect gpu-hotspots \
  -result-dir /tmp/vtune_moe_coalescing -- \
  ./build/bin/llama-bench -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 0 -ngl 99 -fa 1
```

Expected: See fused_moe ESIMD kernels dominating GPU time.

**Step 3: Compare profile reports**

```bash
vtune -report summary -result-dir /tmp/vtune_moe_master > /tmp/moe_master_summary.txt
vtune -report summary -result-dir /tmp/vtune_moe_coalescing > /tmp/moe_coalescing_summary.txt
diff /tmp/moe_master_summary.txt /tmp/moe_coalescing_summary.txt
```

Expected: Identify which kernels are slowest in our branch.

---

### Task 2: Implement Batch Size Threshold for Fused Kernel

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:11192-11220`

**Step 1: Write the failing test**

Create a shell test that validates prompt processing speed:

```bash
# tests/test-moe-prompt-perf.sh
#!/bin/bash
# Test that MoE prompt processing is not slower than baseline

source /opt/intel/oneapi/setvars.sh --force
BASELINE=250  # t/s - conservative threshold

RESULT=$(ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 0 -ngl 99 -fa 1 2>&1 | grep "pp512" | awk '{print $NF}' | cut -d'±' -f1)

echo "MoE pp512: ${RESULT} t/s (target >= ${BASELINE})"

# Compare as integers (multiply by 100 to handle decimals)
RESULT_INT=$(echo "$RESULT * 100" | bc | cut -d'.' -f1)
BASELINE_INT=$((BASELINE * 100))

if [ "$RESULT_INT" -lt "$BASELINE_INT" ]; then
  echo "FAIL: Performance below threshold"
  exit 1
fi
echo "PASS"
```

**Step 2: Run test to verify it fails**

```bash
chmod +x tests/test-moe-prompt-perf.sh
./tests/test-moe-prompt-perf.sh
```

Expected: FAIL with ~87 t/s (below 250 threshold).

**Step 3: Add batch size check to bypass fused kernel**

Modify `ggml_sycl_mul_mat_id_fused` to return false for large batches:

```cpp
// In ggml-sycl.cpp around line 11205
// After: if (ne12 <= 1) { return false; }

// For large batch sizes, master's oneDNN batching approach is faster
// Fused ESIMD kernel: ~87 t/s for pp512, oneDNN batching: ~282 t/s
// Threshold determined empirically - fused kernel only wins for small batches
constexpr int64_t FUSED_MOE_MAX_BATCH = 32;
if (ne12 > FUSED_MOE_MAX_BATCH) {
    GGML_SYCL_DEBUG("[MoE FUSED] Batch %ld > %d, using oneDNN batching\n",
                    (long)ne12, FUSED_MOE_MAX_BATCH);
    return false;  // Fall back to master's batching approach
}
```

**Step 4: Run test to verify it passes**

```bash
./scripts/quick-rebuild.sh ggml-sycl.cpp
./tests/test-moe-prompt-perf.sh
```

Expected: PASS with ~280 t/s.

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp tests/test-moe-prompt-perf.sh
git commit -m "perf(sycl): use oneDNN batching for large MoE prefill

Fused ESIMD kernel is optimized for decode (single-token) performance.
For large batch sizes (>32 tokens), master's approach of grouping tokens
by expert and calling oneDNN GEMM is ~3x faster.

Benchmark: GPT-OSS 20B Q8_0
- pp512: 87 t/s → 280 t/s (+221%)
- tg128: 31 t/s → 31 t/s (maintained)"
```

---

### Task 3: Verify Decode Performance Not Regressed

**Files:**
- Test: existing benchmarks

**Step 1: Run decode benchmark**

```bash
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 0 -n 128 -ngl 99 -fa 1
```

Expected: tg128 ≥ 31 t/s (same as before change).

**Step 2: Run dense model canary test**

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

Expected: No regression from previous 42 t/s decode.

**Step 3: Commit verification results**

Document results in commit message or update docs.

---

### Task 4: Tune Batch Threshold (Optional Optimization)

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:11205`

**Step 1: Test different thresholds**

Run benchmarks with different FUSED_MOE_MAX_BATCH values:

```bash
for threshold in 8 16 32 64 128; do
  echo "=== Testing threshold=$threshold ==="
  # Modify code, rebuild, benchmark
done
```

**Step 2: Find optimal threshold**

The threshold should maximize both prompt AND decode performance.
Document findings.

**Step 3: Commit optimal threshold**

If threshold differs from 32, update and commit.

---

### Task 5: Final Validation

**Files:**
- All modified files

**Step 1: Run full benchmark matrix**

```bash
echo "=== GPT-OSS 20B Q8_0 ==="
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1

echo "=== Mistral 7B Q4_0 ==="
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

**Step 2: Verify correctness**

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10..."

**Step 3: Document results**

Update CLAUDE.md with new benchmark numbers if significant.

---

## Summary of Changes

1. **ggml-sycl.cpp:11205**: Add batch size check to skip fused ESIMD kernel for batch >32
2. **tests/test-moe-prompt-perf.sh**: Performance regression test

## Risk Assessment

- **Low risk**: Change only affects MoE models with large batch sizes
- **Fallback path**: Uses well-tested master code path (oneDNN batching)
- **Canary tests**: Mistral benchmarks ensure dense models unaffected
