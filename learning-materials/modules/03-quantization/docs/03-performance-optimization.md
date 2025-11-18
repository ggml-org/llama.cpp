# Performance Optimization Guide

**Module 3, Lesson 3** | **Estimated Time**: 4 hours | **Difficulty**: Advanced

## Table of Contents
1. [Introduction](#introduction)
2. [Performance Fundamentals](#performance-fundamentals)
3. [Profiling Tools and Techniques](#profiling-tools-and-techniques)
4. [CPU Optimization](#cpu-optimization)
5. [Memory Optimization](#memory-optimization)
6. [SIMD Vectorization](#simd-vectorization)
7. [Threading and Parallelism](#threading-and-parallelism)
8. [Batch Processing](#batch-processing)
9. [Hardware-Specific Optimizations](#hardware-specific-optimizations)
10. [Interview Questions](#interview-questions)

---

## Introduction

Performance optimization is critical for production LLM deployment. Even with quantization, a 7B model requires billions of operations per token. Understanding and applying optimization techniques can yield 2-10x speedups.

**Learning Objectives:**
- Profile and identify performance bottlenecks
- Apply SIMD optimizations for matrix operations
- Optimize memory access patterns
- Implement effective parallelization
- Tune for specific hardware architectures

**Prerequisites:**
- Quantization fundamentals
- Basic understanding of computer architecture
- Familiarity with C/C++ performance concepts

---

## Performance Fundamentals

### LLM Inference Bottlenecks

#### 1. Memory Bandwidth (Primary Bottleneck)

Most modern systems are **memory-bound**, not compute-bound:

```
Typical 7B Model Inference (Q4_K_M):
- Model size: ~4 GB
- Per-token: ~4 GB read (full model pass)
- Modern DDR4: ~50 GB/s
- Time: ~80ms per token (memory-limited)

vs.

Compute requirement:
- ~7B ops per token (FP32 equivalent)
- Modern CPU: ~100 GFLOPS
- Time: ~70ms (compute)
```

**Key Insight**: Reducing memory footprint (quantization) helps more than compute optimization.

#### 2. Cache Utilization

Memory hierarchy matters:

```
L1 Cache:  ~32 KB,   ~1 cycle,    ~200 GB/s
L2 Cache:  ~256 KB,  ~4 cycles,   ~100 GB/s
L3 Cache:  ~8 MB,    ~20 cycles,  ~50 GB/s
RAM:       ~16 GB,   ~100 cycles, ~40 GB/s
```

**Goal**: Maximize data reuse in cache before eviction.

#### 3. Instruction-Level Parallelism

Modern CPUs can execute multiple operations per cycle:
- **SIMD**: Process 8-16 values per instruction (AVX2/AVX-512)
- **Pipelining**: Execute different stages of multiple instructions simultaneously
- **Out-of-order execution**: CPU reorders instructions for efficiency

---

## Profiling Tools and Techniques

### Essential Profiling Tools

#### 1. llama.cpp Built-in Profiling

```bash
# Enable timing information
./llama-cli -m model.gguf -p "Test prompt" --verbose

# Output includes:
# - Time per layer
# - Total inference time
# - Tokens per second
```

#### 2. perf (Linux)

```bash
# Profile CPU cycles and instructions
perf stat ./llama-cli -m model.gguf -p "Test prompt" -n 100

# Sample output:
# Performance counter stats:
#   100,234,567,890 cycles
#    85,123,456,789 instructions    # 0.85 insn per cycle
#     5,234,567,890 cache-misses    # 5.2% miss rate
```

```bash
# Record and analyze hotspots
perf record -g ./llama-cli -m model.gguf -p "Test"
perf report

# Shows:
# - Which functions consume most time
# - Call graphs
# - Assembly-level analysis
```

#### 3. Intel VTune / AMD μProf

For detailed microarchitecture analysis:
```bash
# VTune
vtune -collect hotspots -r result ./llama-cli -m model.gguf

# Shows:
# - Pipeline stalls
# - Branch mispredictions
# - Memory access patterns
# - SIMD utilization
```

#### 4. Valgrind (Callgrind)

```bash
# Profile function calls and cache behavior
valgrind --tool=callgrind ./llama-cli -m model.gguf -p "Test"
kcachegrind callgrind.out.*

# Visualize:
# - Function call tree
# - Time per function
# - Cache hit/miss rates
```

### Custom Profiling Macros

```cpp
// timing.h
#include <chrono>

class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}

    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

// Usage in code
#define TIME_BLOCK(name) \
    Timer timer_##name; \
    defer { printf("%s: %.2f ms\n", #name, timer_##name.elapsed_ms()); }

// In inference code:
void forward_pass() {
    TIME_BLOCK(embedding);
    compute_embedding(...);

    TIME_BLOCK(attention);
    compute_attention(...);

    TIME_BLOCK(ffn);
    compute_ffn(...);
}
```

### Profiling Best Practices

1. **Warm up before measurement**
   ```cpp
   // Run a few iterations to warm cache
   for (int i = 0; i < 5; i++) {
       inference(model, dummy_input);
   }

   // Now measure
   Timer timer;
   for (int i = 0; i < 100; i++) {
       inference(model, input);
   }
   double avg_time = timer.elapsed_ms() / 100;
   ```

2. **Measure in production-like conditions**
   - Use realistic prompts
   - Test with typical batch sizes
   - Include full pipeline (tokenization, sampling, etc.)

3. **Profile iteratively**
   - Optimize biggest bottleneck first (Amdahl's law)
   - Re-profile after each optimization
   - Avoid premature optimization

---

## CPU Optimization

### 1. Compiler Optimization Flags

```cmake
# CMakeLists.txt optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")

# Breakdown:
# -O3: Maximum optimization
# -march=native: Use CPU-specific instructions (AVX2, AVX-512, etc.)
# -mtune=native: Optimize for specific CPU microarchitecture
```

**Impact**: 20-40% speedup over default flags

```bash
# Check what -march=native enables
gcc -march=native -Q --help=target | grep enabled
# Look for:
# -mavx2
# -mfma
# -mavx512f (if available)
```

### 2. Function Inlining

```cpp
// Force inline hot functions
inline __attribute__((always_inline))
float dot_product(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
```

**Impact**: Eliminates function call overhead (5-10% for hot paths)

### 3. Loop Unrolling

```cpp
// Manual loop unrolling
void vec_add(float* c, const float* a, const float* b, int n) {
    int i;
    // Process 4 elements per iteration
    for (i = 0; i < n - 3; i += 4) {
        c[i+0] = a[i+0] + b[i+0];
        c[i+1] = a[i+1] + b[i+1];
        c[i+2] = a[i+2] + b[i+2];
        c[i+3] = a[i+3] + b[i+3];
    }
    // Handle remainder
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

**Impact**: 10-20% speedup (reduces loop overhead, better pipelining)

### 4. Branch Prediction

```cpp
// Bad: Unpredictable branch in hot loop
for (int i = 0; i < n; i++) {
    if (values[i] > threshold) {  // 50/50 branch
        process_high(values[i]);
    } else {
        process_low(values[i]);
    }
}

// Better: Branchless version
for (int i = 0; i < n; i++) {
    int is_high = values[i] > threshold;
    result[i] = is_high * high_value + (!is_high) * low_value;
}

// Best: Separate loops (predictable branches)
for (int i = 0; i < n; i++) {
    if (values[i] > threshold) {
        high_indices[high_count++] = i;
    }
}
for (int i = 0; i < high_count; i++) {
    process_high(values[high_indices[i]]);
}
```

**Impact**: Up to 2x speedup for branch-heavy code

---

## Memory Optimization

### 1. Data Layout and Alignment

```cpp
// Bad: Structure of arrays (poor cache locality)
struct Tensors {
    std::vector<float> layer0_weights;
    std::vector<float> layer1_weights;
    std::vector<float> layer2_weights;
};

// Good: Contiguous allocation
struct Tensors {
    float* all_weights;  // Single contiguous allocation
    // Offsets into all_weights for each layer
    size_t layer_offsets[NUM_LAYERS];
};
```

**Alignment for SIMD:**
```cpp
// Ensure 32-byte alignment for AVX2
float* weights = (float*)aligned_alloc(32, size * sizeof(float));

// Or use compiler attributes
alignas(32) float weights[SIZE];
```

### 2. Memory Access Patterns

```cpp
// Bad: Non-sequential access (cache thrashing)
for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
        result[i] += matrix[j][i] * vector[j];  // Column-major in row-major array
    }
}

// Good: Sequential access (cache-friendly)
for (int j = 0; j < m; j++) {
    float v = vector[j];
    for (int i = 0; i < n; i++) {
        result[i] += matrix[j * n + i] * v;  // Sequential in memory
    }
}
```

### 3. Prefetching

```cpp
// Explicit prefetching for upcoming data
for (int i = 0; i < n; i++) {
    __builtin_prefetch(&data[i + 8], 0, 3);  // Prefetch 8 ahead
    process(data[i]);
}
```

**Impact**: 10-30% speedup for memory-bound operations

### 4. Memory Mapping

```cpp
// Use mmap for large model files
int fd = open("model.gguf", O_RDONLY);
void* model_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

// Benefits:
// - Lazy loading (only used pages loaded)
// - OS manages memory
// - Shared across processes
```

---

## SIMD Vectorization

### AVX2 Basics

AVX2 processes 8 floats (256 bits) per instruction:

```cpp
#include <immintrin.h>

// Scalar version
void add_scalar(float* c, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// AVX2 version (8x parallelism)
void add_avx2(float* c, const float* a, const float* b, int n) {
    int i;
    for (i = 0; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&c[i], vc);
    }
    // Handle remainder
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

**Speedup**: ~6-7x (ideal 8x, some overhead)

### Dot Product with AVX2

```cpp
float dot_product_avx2(const float* a, const float* b, int n) {
    __m256 sum_vec = _mm256_setzero_ps();

    int i;
    for (i = 0; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);  // Fused multiply-add
    }

    // Horizontal sum
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    sum_low = _mm_add_ps(sum_low, sum_high);

    sum_low = _mm_hadd_ps(sum_low, sum_low);
    sum_low = _mm_hadd_ps(sum_low, sum_low);

    float result = _mm_cvtss_f32(sum_low);

    // Add remainder
    for (; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}
```

### Quantization with SIMD

Example: Dequantize Q4_0 with AVX2

```cpp
void dequantize_q4_0_avx2(const block_q4_0* x, float* y, int k) {
    const int qk = QK4_0;  // 32

    for (int i = 0; i < k / qk; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t* qs = x[i].qs;

        __m256 d_vec = _mm256_set1_ps(d);

        for (int j = 0; j < qk/2; j += 8) {
            // Load 8 bytes (16 4-bit values)
            __m128i q4bits = _mm_loadl_epi64((__m128i*)&qs[j]);

            // Unpack to 8-bit
            __m128i q8_0 = _mm_and_si128(q4bits, _mm_set1_epi8(0xF));
            __m128i q8_1 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), _mm_set1_epi8(0xF));

            // Convert to float
            __m256i q32_0 = _mm256_cvtepi8_epi32(q8_0);
            __m256i q32_1 = _mm256_cvtepi8_epi32(q8_1);

            __m256 f_0 = _mm256_cvtepi32_ps(q32_0);
            __m256 f_1 = _mm256_cvtepi32_ps(q32_1);

            // Subtract 8 and multiply by scale
            f_0 = _mm256_mul_ps(_mm256_sub_ps(f_0, _mm256_set1_ps(8.0f)), d_vec);
            f_1 = _mm256_mul_ps(_mm256_sub_ps(f_1, _mm256_set1_ps(8.0f)), d_vec);

            // Store
            _mm256_storeu_ps(&y[i * qk + j * 2], f_0);
            _mm256_storeu_ps(&y[i * qk + j * 2 + 8], f_1);
        }
    }
}
```

**Speedup**: 4-8x over scalar dequantization

### ARM NEON

For ARM processors (Apple Silicon, mobile):

```cpp
#include <arm_neon.h>

void add_neon(float* c, const float* a, const float* b, int n) {
    int i;
    for (i = 0; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(&c[i], vc);
    }
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

---

## Threading and Parallelism

### Thread Pool Design

```cpp
class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

public:
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }
};
```

### Parallel Matrix Operations

```cpp
// Parallelize matrix-vector multiplication
void matmul_parallel(float* y, const float* A, const float* x,
                     int rows, int cols, int num_threads) {
    ThreadPool pool(num_threads);
    int rows_per_thread = (rows + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; t++) {
        int start_row = t * rows_per_thread;
        int end_row = std::min(start_row + rows_per_thread, rows);

        pool.enqueue([=] {
            for (int i = start_row; i < end_row; i++) {
                float sum = 0.0f;
                for (int j = 0; j < cols; j++) {
                    sum += A[i * cols + j] * x[j];
                }
                y[i] = sum;
            }
        });
    }
}
```

### Thread Scaling Guidelines

```cpp
// Determine optimal thread count
int get_optimal_threads() {
    int hw_threads = std::thread::hardware_concurrency();

    // For CPU inference:
    // - Use physical cores, not logical (hyperthreading)
    // - Leave 1-2 cores for OS/other tasks
    int physical_cores = hw_threads / 2;  // Rough estimate
    int optimal = std::max(1, physical_cores - 1);

    return optimal;
}
```

**Scaling Results** (typical):
- 1 thread: 1.0x (baseline)
- 2 threads: 1.8x
- 4 threads: 3.2x
- 8 threads: 5.5x
- 16 threads: 7.0x (diminishing returns)

### False Sharing Prevention

```cpp
// Bad: False sharing (threads writing to adjacent memory)
struct PerThreadData {
    int count;  // Only 4 bytes, but cache line is 64 bytes
};
PerThreadData thread_data[NUM_THREADS];  // Adjacent in memory!

// Good: Pad to cache line size
struct alignas(64) PerThreadData {
    int count;
    char padding[60];  // Pad to 64 bytes (cache line size)
};
```

---

## Batch Processing

### Batching Benefits

Process multiple sequences simultaneously:

```
Single sequence: [seq1] → 30 tokens/sec
Batch of 4:      [seq1, seq2, seq3, seq4] → 80 tokens/sec total (20 each)
Batch of 8:      [seq1, ..., seq8] → 120 tokens/sec total (15 each)
```

**Throughput increases** but **per-sequence latency increases**.

### Batch Implementation

```cpp
struct Batch {
    std::vector<int> tokens;          // All tokens
    std::vector<int> seq_lengths;     // Length of each sequence
    std::vector<int> seq_ids;         // Sequence ID for each token
    int num_sequences;
};

void process_batch(const Batch& batch, Model& model) {
    // Single forward pass for all sequences
    Tensor output = model.forward(batch.tokens);

    // Distribute outputs to sequences
    int offset = 0;
    for (int i = 0; i < batch.num_sequences; i++) {
        int len = batch.seq_lengths[i];
        process_sequence_output(&output[offset], len, i);
        offset += len;
    }
}
```

### Dynamic Batching

```cpp
class BatchScheduler {
    std::queue<Request> pending;
    const int max_batch_size = 8;
    const int max_wait_ms = 10;

public:
    Batch get_next_batch() {
        Batch batch;
        auto start = std::chrono::steady_clock::now();

        while (batch.num_sequences < max_batch_size) {
            if (pending.empty()) {
                auto elapsed = std::chrono::steady_clock::now() - start;
                if (batch.num_sequences > 0 &&
                    elapsed > std::chrono::milliseconds(max_wait_ms)) {
                    break;  // Don't wait too long
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            Request req = pending.front();
            pending.pop();
            batch.add_sequence(req);
        }

        return batch;
    }
};
```

---

## Hardware-Specific Optimizations

### Apple Silicon (M1/M2/M3)

```cpp
#if defined(__ARM_NEON)
    // Use NEON intrinsics
    #define USE_NEON 1
#endif

#if defined(__APPLE__)
    // Use Metal for GPU acceleration
    #define USE_METAL 1
#endif

// Optimize for unified memory
void* allocate_metal_buffer(size_t size) {
    // Use shared memory between CPU and GPU
    return malloc(size);  // Automatically shared on Apple Silicon
}
```

### Intel CPUs

```bash
# Build with AVX-512 if available
cmake -DLLAMA_AVX512=ON ..

# Or detect at runtime
if (__builtin_cpu_supports("avx512f")) {
    use_avx512_kernels();
} else if (__builtin_cpu_supports("avx2")) {
    use_avx2_kernels();
} else {
    use_scalar_kernels();
}
```

### AMD CPUs

```cmake
# AMD-specific optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma -mavx2")

# Use AMD AOCC compiler for best performance
set(CMAKE_CXX_COMPILER "clang++")
```

---

## Interview Questions

### Profiling

1. **Q: How would you identify the bottleneck in slow LLM inference?**

   A: Systematic profiling approach:
   ```bash
   # 1. Overall timing
   time ./llama-cli -m model.gguf -p "test" -n 100

   # 2. CPU profiling
   perf stat ./llama-cli ...
   # Check: IPC (instructions per cycle), cache miss rate

   # 3. Hotspot analysis
   perf record -g ./llama-cli ...
   perf report
   # Identify hot functions

   # 4. Memory bandwidth test
   # Compare actual throughput vs. memory bandwidth
   # If close to bandwidth limit → memory-bound

   # 5. Test with different threads
   # If doesn't scale → synchronization or memory-bound
   ```

   Typical finding: Memory bandwidth limited (70%+ of cases)

### Optimization Techniques

2. **Q: Explain why quantization improves inference speed even on compute-bound systems.**

   A: Multiple reasons:
   - **Memory bandwidth**: Less data to transfer (4GB vs 14GB for Q4 vs FP16)
   - **Cache efficiency**: More of model fits in cache
   - **SIMD**: Can process more values (16 INT8 vs 8 FP32 per AVX2 instruction)
   - **Memory latency**: Fewer cache misses
   - **Dequantization cost**: Usually cheaper than memory transfer

   Even if compute-bound, reduced data movement helps overall pipeline.

3. **Q: How does SIMD vectorization improve performance and what are its limitations?**

   A: SIMD processes multiple values per instruction:
   - AVX2: 8 floats/instruction (8x theoretical speedup)
   - AVX-512: 16 floats/instruction (16x theoretical)

   Limitations:
   - **Data alignment**: May need padding/alignment overhead
   - **Horizontal operations**: Reductions (sum, max) harder to vectorize
   - **Branching**: Can't vectorize unpredictable branches
   - **Memory bandwidth**: Still limited by memory even with SIMD
   - **Real speedup**: 4-6x typical (not full 8x) due to overhead

4. **Q: Design a threading strategy for inference. How many threads should you use?**

   A: Strategy:
   ```cpp
   int optimal_threads() {
       int hw_cores = std::thread::hardware_concurrency();
       int physical_cores = hw_cores / 2;  // Account for HT

       // For inference:
       // - Matrix multiply: Scale well to physical cores
       // - Quantization: Limited by memory bandwidth

       // Heuristic:
       if (model_size_gb < ram_bandwidth_gb_per_sec * 0.1) {
           // Compute-bound: use more threads
           return physical_cores - 1;
       } else {
           // Memory-bound: fewer threads often better
           return std::min(4, physical_cores - 1);
       }
   }
   ```

   Always benchmark: Start with 4, test 1/2/4/8, pick best.

### Advanced Topics

5. **Q: Explain the trade-off between batch size and latency. How would you optimize for a production API?**

   A: Trade-offs:
   ```
   Batch Size 1:
   - Latency: Low (50ms)
   - Throughput: Low (20 req/sec)
   - GPU utilization: Poor (30%)

   Batch Size 8:
   - Latency: Higher (150ms)
   - Throughput: High (50 req/sec)
   - GPU utilization: Good (80%)

   Batch Size 32:
   - Latency: Very high (500ms)
   - Throughput: Maximum (100 req/sec)
   - GPU utilization: Excellent (95%)
   ```

   Production strategy:
   ```python
   # Dynamic batching with timeout
   - Max batch size: 8-16 (balance)
   - Max wait time: 10-20ms (acceptable latency hit)
   - Priority queue: User-facing requests first
   - Separate pools: Low-latency vs high-throughput
   ```

6. **Q: Why might adding more CPU threads actually slow down inference?**

   A: Several reasons:
   - **Memory bandwidth saturation**: More threads competing for same bandwidth
   - **Cache thrashing**: Working sets don't fit in cache
   - **False sharing**: Adjacent data structures on different threads
   - **Synchronization overhead**: Lock contention, barriers
   - **NUMA effects**: Cross-socket memory access on multi-CPU systems

   Example:
   ```
   4 threads: 40 tokens/sec (each thread uses L2 cache)
   8 threads: 35 tokens/sec (cache thrashing, bandwidth saturated)
   ```

   Always profile thread scaling on target hardware.

7. **Q: How would you optimize for a specific CPU architecture (e.g., Intel Sapphire Rapids with AVX-512)?**

   A: Multi-level optimization:
   ```cmake
   # 1. Compiler flags
   -march=sapphirerapids  # CPU-specific optimizations
   -mtune=sapphirerapids

   # 2. Use AVX-512 kernels
   #ifdef __AVX512F__
   void matmul_avx512(...) {
       // Process 16 floats per instruction
       __m512 va = _mm512_loadu_ps(&a[i]);
       __m512 vb = _mm512_loadu_ps(&b[i]);
       __m512 vc = _mm512_fmadd_ps(va, vb, sum);
   }
   #endif

   # 3. Optimize for cache hierarchy
   # Sapphire Rapids: 2MB L2 per core, shared L3
   # Tile matrices to fit in L2

   # 4. Use AMX (Advanced Matrix Extensions) for INT8
   # New instruction set for matrix multiplication
   # Can replace manual SIMD for some operations

   # 5. NUMA awareness
   # Bind threads to cores, memory to local NUMA node
   numactl --cpunodebind=0 --membind=0 ./llama-cli ...
   ```

---

## Summary

**Key Takeaways:**

1. **Memory bandwidth is the primary bottleneck** - optimize data movement first
2. **Profile before optimizing** - measure, don't guess
3. **SIMD is essential** - use AVX2/AVX-512/NEON for hot paths
4. **Threading scales differently per operation** - matrix ops scale well, memory-bound ops don't
5. **Batch processing increases throughput** at the cost of latency

**Optimization Priority:**
1. Quantization (biggest impact)
2. Memory access patterns
3. SIMD vectorization
4. Threading
5. Compiler optimizations

**Next Steps:**
- Lesson 4: GGML tensor operations deep dive
- Lab 3: Hands-on performance optimization
- Tutorial: Building a performance dashboard

---

**Further Reading:**

- [Intel Optimization Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [Agner Fog's Optimization Guide](https://www.agner.org/optimize/)
- [SIMD for C++ Developers](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)

**Author**: Agent 5 (Documentation Specialist)
**Module**: 3 - Quantization & Optimization
**Last Updated**: 2025-11-18
