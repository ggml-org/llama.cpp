# Lab 3: Performance Optimization Challenge

**Module 3** | **Estimated Time**: 3-4 hours | **Difficulty**: Advanced

## Learning Objectives

By completing this lab, you will:
- Profile and identify performance bottlenecks
- Apply optimization techniques (threading, SIMD, memory)
- Measure and validate performance improvements
- Understand hardware-specific optimizations
- Build a performance optimization methodology

## Prerequisites

- Completed Labs 1 and 2
- Understanding of performance optimization concepts
- Profiling tools installed (perf, valgrind, or equivalent)
- Python with psutil, pandas, matplotlib

## Lab Overview

This is a hands-on optimization challenge where you'll profile a baseline system, identify bottlenecks, apply optimizations, and achieve measurable speedups. You'll work with real llama.cpp inference and optimize for your specific hardware.

## Part 1: Baseline Profiling (45 minutes)

### Step 1: Establish Baseline Performance

Choose a model for this lab (7B recommended):

```bash
export MODEL="llama-2-7b-q4_k_m.gguf"
export TEST_PROMPT="The quick brown fox jumps over the lazy dog. This is a test prompt for performance profiling and optimization analysis."
```

### Run Baseline Benchmark

```bash
#!/bin/bash
# baseline_benchmark.sh

MODEL="llama-2-7b-q4_k_m.gguf"
PROMPT="The quick brown fox jumps over the lazy dog."
N_PREDICT=256

echo "=== Baseline Performance Benchmark ==="
echo "Model: $MODEL"
echo "Tokens to generate: $N_PREDICT"
echo ""

# Run 5 trials
for i in {1..5}; do
    echo "Trial $i:"
    ./llama-cli \
        -m $MODEL \
        -p "$PROMPT" \
        -n $N_PREDICT \
        --log-disable \
        2>&1 | grep -E "eval time|tokens"
    echo ""
done
```

**Question 1**: Record your baseline metrics:

| Trial | Prompt Eval Time (ms) | Token Gen Time (ms) | Tokens/sec |
|-------|-----------------------|---------------------|------------|
| 1     | _____                 | _____               | _____      |
| 2     | _____                 | _____               | _____      |
| 3     | _____                 | _____               | _____      |
| 4     | _____                 | _____               | _____      |
| 5     | _____                 | _____               | _____      |
| **Avg** | _____               | _____               | _____      |

### Step 2: System Information

Collect your system specs:

```bash
# CPU Info
lscpu | grep -E "Model name|Thread|Core|Socket|MHz"

# Memory
free -h

# Operating System
uname -a

# CPU Governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

**Question 2**: Fill in your system information:

- **CPU**: _____________________________________
- **Cores**: ____ physical, ____ logical
- **RAM**: ____ GB
- **OS**: _____________________________________
- **CPU Governor**: _________________________

### Step 3: Initial Profiling

Run with profiling enabled:

```bash
# Linux perf profiling
perf stat -d ./llama-cli \
    -m $MODEL \
    -p "$TEST_PROMPT" \
    -n 100 \
    2>&1 | tee baseline_perf.txt

# Look for:
# - Instructions per cycle (IPC)
# - Cache miss rate
# - Branch mispredictions
```

**Question 3**: Record profiling metrics:

- **Instructions per cycle (IPC)**: _____
- **L1 cache miss rate**: _____
- **L3 cache miss rate**: _____
- **Branch mispredictions**: _____

## Part 2: Thread Optimization (45 minutes)

### Step 1: Thread Scaling Analysis

Test different thread counts:

```bash
#!/bin/bash
# thread_scaling.sh

MODEL="llama-2-7b-q4_k_m.gguf"
PROMPT="Test prompt for thread scaling analysis"
N_PREDICT=128

echo "Threads,PromptTime_ms,TokenGenTime_ms,TokensPerSec" > thread_results.csv

for threads in 1 2 4 6 8 12 16; do
    echo "Testing $threads threads..."

    output=$(./llama-cli \
        -m $MODEL \
        -p "$PROMPT" \
        -n $N_PREDICT \
        -t $threads \
        --log-disable 2>&1)

    # Parse output and save to CSV
    # (You'll need to extract timing information)

    echo "$threads,$output" >> thread_results.csv
done
```

**Question 4**: Fill in thread scaling results:

| Threads | Tokens/sec | Speedup vs 1 thread | Efficiency (%) |
|---------|------------|---------------------|----------------|
| 1       | _____      | 1.00x               | 100%           |
| 2       | _____      | _____               | _____          |
| 4       | _____      | _____               | _____          |
| 6       | _____      | _____               | _____          |
| 8       | _____      | _____               | _____          |
| 12      | _____      | _____               | _____          |
| 16      | _____      | _____               | _____          |

**Efficiency** = (Speedup / Thread_count) × 100%

### Step 2: Analyze Thread Scaling

Plot the results:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('thread_results.csv')

# Calculate speedup
df['speedup'] = df['TokensPerSec'] / df.iloc[0]['TokensPerSec']
df['efficiency'] = (df['speedup'] / df['Threads']) * 100

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Speedup plot
ax1.plot(df['Threads'], df['speedup'], marker='o', label='Actual')
ax1.plot(df['Threads'], df['Threads'], '--', alpha=0.5, label='Ideal')
ax1.set_xlabel('Thread Count')
ax1.set_ylabel('Speedup')
ax1.set_title('Thread Scaling')
ax1.legend()
ax1.grid(alpha=0.3)

# Efficiency plot
ax2.plot(df['Threads'], df['efficiency'], marker='o')
ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
ax2.set_xlabel('Thread Count')
ax2.set_ylabel('Efficiency (%)')
ax2.set_title('Threading Efficiency')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('thread_scaling_analysis.png', dpi=150)
```

**Question 5**: What is your optimal thread count based on the analysis?

**Answer**: _____

**Question 6**: At what thread count does efficiency drop below 50%?

**Answer**: _____

**Question 7**: Why does adding more threads eventually hurt performance?

**Answer**: _________________________________________________________________

### Step 3: Thread Affinity Optimization

Test with CPU pinning:

```bash
# Without pinning
./llama-cli -m $MODEL -p "$TEST_PROMPT" -n 100 -t 8

# With pinning to first 8 cores
taskset -c 0-7 ./llama-cli -m $MODEL -p "$TEST_PROMPT" -n 100 -t 8

# With pinning to cores on same socket (NUMA)
numactl --cpunodebind=0 --membind=0 \
    ./llama-cli -m $MODEL -p "$TEST_PROMPT" -n 100 -t 8
```

**Question 8**: Does CPU pinning improve performance? By how much?

**Answer**: _________________________________________________________________

## Part 3: Memory and Cache Optimization (45 minutes)

### Step 1: Memory Bandwidth Analysis

Test memory impact:

```bash
# Monitor memory bandwidth
sudo perf stat -e \
    cycles,instructions,\
    cache-references,cache-misses,\
    mem_load_retired.l1_miss,\
    mem_load_retired.l3_miss \
    ./llama-cli -m $MODEL -p "$TEST_PROMPT" -n 100
```

**Question 9**: What is your cache miss rate?

- L1 cache miss rate: _____ %
- L3 cache miss rate: _____ %

**Question 10**: Is your inference memory-bound or compute-bound?

**Answer**: _________________________________________________________________

### Step 2: Context Size Impact

Test different context sizes:

```bash
for ctx_size in 512 1024 2048 4096; do
    echo "Context size: $ctx_size"
    ./llama-cli \
        -m $MODEL \
        -p "$TEST_PROMPT" \
        -n 100 \
        -c $ctx_size \
        --log-disable
done
```

**Question 11**: How does context size affect performance?

| Context Size | Tokens/sec | Memory Usage (GB) |
|--------------|------------|-------------------|
| 512          | _____      | _____             |
| 1024         | _____      | _____             |
| 2048         | _____      | _____             |
| 4096         | _____      | _____             |

### Step 3: Memory Prefetching

Examine memory access patterns:

```bash
# Cache profiling
perf record -e cache-misses ./llama-cli -m $MODEL -p "$TEST_PROMPT" -n 50
perf report

# Look for hot functions with high cache miss rates
```

**Question 12**: Which functions have the highest cache miss rates?

**Answer**: _________________________________________________________________

## Part 4: Compiler and SIMD Optimizations (45 minutes)

### Step 1: Verify SIMD Support

Check what SIMD instructions are being used:

```bash
# Check build flags
./llama-cli --version

# Check CPU SIMD support
lscpu | grep -i flags

# Verify binary uses SIMD
objdump -d ./build/bin/llama-cli | grep -i "avx\|sse\|neon"
```

**Question 13**: What SIMD instructions does your binary use?

**Answer**: _________________________________________________________________

### Step 2: Rebuild with Optimizations

If not already optimal, rebuild:

```bash
cd llama.cpp

# Clean build
rm -rf build
mkdir build && cd build

# Configure with optimizations
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_NATIVE=ON \
    -DLLAMA_AVX2=ON \
    -DLLAMA_FMA=ON \
    -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native"

# Build
make -j$(nproc)
```

**Question 14**: Does rebuilding with native optimizations improve performance?

- Before: _____ tokens/sec
- After: _____ tokens/sec
- Improvement: _____ %

### Step 3: Quantization Optimization

Compare quantization performance:

```bash
# Test different quantizations
for format in q4_0 q4_k_m q5_k_m q8_0; do
    echo "Testing $format..."
    ./llama-cli \
        -m llama-2-7b-${format}.gguf \
        -p "$TEST_PROMPT" \
        -n 100 \
        -t 8 \
        --log-disable
done
```

**Question 15**: Which quantization gives the best tokens/sec on your hardware?

**Answer**: _________________________________________________________________

## Part 5: Optimization Challenge (60 minutes)

### Goal: Achieve 30% Speedup

Starting from your baseline, achieve at least 30% speedup by applying the techniques learned.

### Optimization Checklist

Create a systematic optimization plan:

```
[ ] 1. Set CPU governor to performance
[ ] 2. Find optimal thread count
[ ] 3. Pin threads to physical cores
[ ] 4. Use NUMA-aware binding (if applicable)
[ ] 5. Rebuild with native optimizations
[ ] 6. Choose fastest quantization format
[ ] 7. Tune context size
[ ] 8. Disable unnecessary features
[ ] 9. Run isolated from other processes
[ ] 10. (Your custom optimization)
```

### Step 1: Apply Optimizations

Document each optimization and its impact:

```bash
#!/bin/bash
# optimized_benchmark.sh

# Apply all optimizations
sudo cpupower frequency-set -g performance

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true

numactl --cpunodebind=0 --membind=0 \
    taskset -c 0-7 \
    nice -n -20 \
    ./llama-cli \
        -m llama-2-7b-q4_k_m.gguf \
        -p "$TEST_PROMPT" \
        -n 256 \
        -t 8 \
        -c 2048 \
        --log-disable
```

### Step 2: Measure Final Performance

Run the optimized version 5 times:

**Question 16**: Final optimized results:

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Tokens/sec | _____ | _____ | _____ % |
| Prompt Time (ms) | _____ | _____ | _____ % |
| Memory (GB) | _____ | _____ | _____ % |

**Question 17**: Did you achieve 30% speedup?

**Answer**: _________________________________________________________________

### Step 3: Optimization Log

Document each optimization:

| Optimization | Impact (%) | Cumulative (%) | Notes |
|--------------|------------|----------------|-------|
| Baseline     | -          | 0%             | -     |
| ____________ | _____      | _____          | _____ |
| ____________ | _____      | _____          | _____ |
| ____________ | _____      | _____          | _____ |
| ____________ | _____      | _____          | _____ |
| **Total**    | -          | _____          | -     |

## Part 6: Advanced Challenges (Optional)

### Challenge 1: Batch Processing

Implement batch processing for higher throughput:

```python
# Process multiple prompts simultaneously
prompts = [
    "Prompt 1...",
    "Prompt 2...",
    "Prompt 3...",
    "Prompt 4..."
]

# Compare:
# - Sequential processing
# - Batched processing
```

**Question 18**: What throughput gain does batching provide?

**Answer**: _________________________________________________________________

### Challenge 2: GPU Offloading

If you have a GPU:

```bash
# Test GPU layer offloading
for layers in 0 16 32 40 48; do
    echo "GPU layers: $layers"
    ./llama-cli \
        -m $MODEL \
        -p "$TEST_PROMPT" \
        -n 100 \
        -ngl $layers
done
```

**Question 19**: What is the optimal number of GPU layers?

**Answer**: _________________________________________________________________

### Challenge 3: Custom Optimization

Implement your own optimization:

Ideas:
- Optimize prompt caching
- Implement speculative decoding
- Custom memory allocator
- Profile-guided optimization (PGO)

**Question 20**: Describe your custom optimization and its impact:

**Answer**: _________________________________________________________________

## Lab Deliverables

Submit:
1. Complete baseline and optimized benchmark results
2. Thread scaling analysis with plots
3. Optimization log documenting each change
4. Final speedup report
5. System-specific optimization recommendations

## Key Insights

**Question 21**: What was the single most effective optimization?

**Answer**: _________________________________________________________________

**Question 22**: Were there any surprising findings?

**Answer**: _________________________________________________________________

**Question 23**: What would you optimize next if you had more time?

**Answer**: _________________________________________________________________

## Production Recommendations

Based on your findings, write recommendations for production deployment:

**Recommendations**:
1. **Thread Configuration**: _________________________________________________
2. **Memory Configuration**: _________________________________________________
3. **Quantization Format**: _________________________________________________
4. **Hardware Requirements**: _________________________________________________
5. **Monitoring Metrics**: _________________________________________________

## Further Exploration

- Profile with Intel VTune or AMD μProf
- Implement custom GGML operations
- Compare ARM vs x86 performance
- Build automated performance regression tests
- Test with different model sizes

## Resources

- [Performance Optimization Guide](../docs/03-performance-optimization.md)
- [GGML Tensor Operations](../docs/04-ggml-tensor-operations.md)
- [Performance Profiler Tool](../code/performance_profiler.py)
- [Linux perf tutorial](https://perf.wiki.kernel.org/index.php/Tutorial)

---

**Completion Criteria**:
- [ ] Baseline benchmarks completed
- [ ] Thread scaling analyzed
- [ ] Memory profiling done
- [ ] Compiler optimizations applied
- [ ] 30% speedup achieved (or documented attempt)
- [ ] Optimization log complete
- [ ] Production recommendations written

**Estimated Time**: 3-4 hours
**Difficulty**: Advanced
**Author**: Agent 4 (Lab Designer)
**Module**: 3 - Quantization & Optimization
