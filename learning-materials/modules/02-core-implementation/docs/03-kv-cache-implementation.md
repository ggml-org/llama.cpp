# KV Cache Implementation

**Learning Module**: Module 2 - Core Implementation
**Estimated Reading Time**: 28 minutes
**Prerequisites**: Module 1 complete, understanding of attention mechanisms
**Related Content**:
- [Model Architecture Deep Dive](./01-model-architecture-deep-dive.md)
- [Inference Pipeline](./04-inference-pipeline.md)

---

## Overview

The Key-Value (KV) cache is one of the most critical optimizations in LLM inference. Without it, generating long sequences would be prohibitively slow. This document explains how KV caching works, how it's implemented in llama.cpp, and how to optimize its usage.

### Learning Objectives

After completing this lesson, you will:
- ✅ Understand why KV caching is necessary
- ✅ Know the memory layout and data structures
- ✅ Implement KV cache management
- ✅ Optimize cache usage for different scenarios
- ✅ Debug cache-related issues

---

## Why KV Cache Exists

### The Problem: Redundant Computation

Without caching, each new token requires recomputing attention for ALL previous tokens:

```
Generating "The cat sat on the mat"

Token 1: "The"
  Compute: Q1, K1, V1
  Attention: Self-attention (1 token)
  Cost: 1 attention computation

Token 2: "cat"
  Compute: Q1, K1, V1 (again!)
          Q2, K2, V2
  Attention: Token 2 attends to [1, 2]
  Cost: 2 attention computations

Token 3: "sat"
  Compute: Q1, K1, V1 (again!)
          Q2, K2, V2 (again!)
          Q3, K3, V3
  Attention: Token 3 attends to [1, 2, 3]
  Cost: 3 attention computations

Total for N tokens: 1 + 2 + 3 + ... + N = N(N+1)/2
For 100 tokens: 5,050 attention computations!
```

### The Solution: Cache K and V

Key insight: During generation, K and V for previous tokens never change!

```
With KV Cache:

Token 1: "The"
  Compute: Q1, K1, V1
  Cache: K1, V1
  Attention: Self-attention (1 token)
  Cost: 1 computation

Token 2: "cat"
  Compute: Q2, K2, V2
  Cache: K2, V2 (append to cache)
  Retrieve: K1, V1 from cache
  Attention: Token 2 attends to [1, 2]
  Cost: 1 computation (not 2!)

Token 3: "sat"
  Compute: Q3, K3, V3
  Cache: K3, V3 (append to cache)
  Retrieve: K1, V1, K2, V2 from cache
  Attention: Token 3 attends to [1, 2, 3]
  Cost: 1 computation (not 3!)

Total for N tokens: N computations
For 100 tokens: 100 computations (50x speedup!)
```

**Performance Impact**:
```
Without cache: O(N²) - Quadratic growth
With cache:    O(N)  - Linear growth

Example (100 token generation):
  Without cache: ~5,000 attention operations
  With cache:    ~100 attention operations
  Speedup:       50x
```

---

## KV Cache Structure

### Mathematical Foundation

In self-attention, we compute:

```
Attention(Q, K, V) = softmax(QK^T / √d) V

Where:
  Q = Query  (current token)
  K = Keys   (all tokens, including previous)
  V = Values (all tokens, including previous)
```

**During Generation**:
- Query (Q): Only current token - must compute
- Keys (K): Current token + all previous - cache previous
- Values (V): Current token + all previous - cache previous

### Memory Layout

```
┌─────────────────────────────────────────────────────┐
│              KV Cache Structure                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Cache Shape (per layer):                           │
│    Keys:   [n_ctx, n_head_kv, head_dim]            │
│    Values: [n_ctx, n_head_kv, head_dim]            │
│                                                      │
│  Example (LLaMA-7B):                                │
│    n_ctx = 2048 (context window)                    │
│    n_head_kv = 8 (GQA: 8 KV heads)                 │
│    head_dim = 128 (4096 / 32)                      │
│                                                      │
│  Memory per layer (FP16):                           │
│    Keys:   2048 × 8 × 128 × 2 bytes = 4 MB         │
│    Values: 2048 × 8 × 128 × 2 bytes = 4 MB         │
│    Total:  8 MB per layer                           │
│                                                      │
│  Total cache (32 layers):                           │
│    32 layers × 8 MB = 256 MB                        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Multi-Layer Cache

Each transformer layer has its own K and V cache:

```
┌─────────────────────────────────────────────────────┐
│          Multi-Layer KV Cache                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Layer 0:  [K_cache_0] [V_cache_0]                  │
│  Layer 1:  [K_cache_1] [V_cache_1]                  │
│  Layer 2:  [K_cache_2] [V_cache_2]                  │
│  ...                                                 │
│  Layer 31: [K_cache_31] [V_cache_31]                │
│                                                      │
│  Token Generation Flow:                              │
│  Input → L0 (use K0,V0) → L1 (use K1,V1) → ...     │
│        → L31 (use K31,V31) → Output                 │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Implementation in llama.cpp

### Data Structure

```cpp
// From llama.cpp
struct llama_kv_cache {
    // Cache type
    bool recurrent = false;  // Mamba/RWKV vs Transformer
    bool v_trans   = true;   // Transpose V for efficiency

    // Size configuration
    uint32_t head  = 0;  // Current position (ring buffer)
    uint32_t size  = 0;  // Total cache size (n_ctx)
    uint32_t used  = 0;  // Number of cells in use

    // Per-layer cache cells
    struct llama_kv_cell {
        llama_pos pos   = -1;  // Position in sequence
        llama_pos delta = 0;   // RoPE delta for shifting
        int32_t   src   = -1;  // Source cell (for copying)

        std::set<llama_seq_id> seq_id;  // Sequence IDs
    };

    std::vector<llama_kv_cell> cells;

    // Actual K and V tensors (per layer)
    struct ggml_tensor * k_l[LLAMA_MAX_LAYERS];  // Keys
    struct ggml_tensor * v_l[LLAMA_MAX_LAYERS];  // Values

    // Context for memory allocation
    struct ggml_context * ctx = nullptr;

    // Buffer for tensor data
    ggml_backend_buffer_t buffer = nullptr;
};
```

### Cache Initialization

```cpp
// Initialize KV cache
bool llama_kv_cache_init(
    const struct llama_hparams & hparams,
    struct llama_kv_cache & cache,
    ggml_type type_k,  // Key data type (FP16, Q8_0, etc.)
    ggml_type type_v,  // Value data type
    uint32_t n_ctx,    // Context size
    bool offload       // Offload to GPU
) {
    const uint32_t n_embd      = hparams.n_embd;
    const uint32_t n_layer     = hparams.n_layer;
    const uint32_t n_head_kv   = hparams.n_head_kv;
    const uint32_t head_dim    = n_embd / hparams.n_head;

    cache.size = n_ctx;
    cache.used = 0;
    cache.head = 0;

    cache.cells.clear();
    cache.cells.resize(n_ctx);

    // Allocate K and V tensors for each layer
    for (uint32_t il = 0; il < n_layer; il++) {
        // K cache: [n_ctx, n_head_kv, head_dim]
        cache.k_l[il] = ggml_new_tensor_3d(
            cache.ctx,
            type_k,
            head_dim,
            n_head_kv,
            n_ctx
        );

        // V cache: [n_ctx, n_head_kv, head_dim]
        cache.v_l[il] = ggml_new_tensor_3d(
            cache.ctx,
            type_v,
            head_dim,
            n_head_kv,
            n_ctx
        );

        ggml_format_name(cache.k_l[il], "cache_k_l%d", il);
        ggml_format_name(cache.v_l[il], "cache_v_l%d", il);
    }

    return true;
}
```

### Cache Update During Inference

```cpp
// Update cache with new tokens
static void llama_kv_cache_update(
    struct llama_context * ctx,
    const llama_batch & batch,
    int32_t layer_idx
) {
    struct llama_kv_cache & cache = ctx->kv_self;

    // For each token in batch
    for (uint32_t i = 0; i < batch.n_tokens; i++) {
        llama_pos pos = batch.pos[i];
        llama_seq_id seq_id = batch.seq_id[i][0];

        // Find cache cell for this position
        struct llama_kv_cell & cell = cache.cells[pos];

        // Update cell metadata
        cell.pos = pos;
        cell.seq_id.insert(seq_id);

        // K and V tensors are updated by the attention operation
        // (written to cache.k_l[layer_idx] and cache.v_l[layer_idx])
    }

    // Update cache usage
    cache.used = std::max(cache.used, (uint32_t)(batch.pos[batch.n_tokens - 1] + 1));
}
```

---

## Memory Optimization Strategies

### 1. Quantized Cache

Reduce memory by quantizing K and V:

```cpp
// Cache data types
GGML_TYPE_F16:  16-bit float (default, high quality)
GGML_TYPE_F32:  32-bit float (maximum quality, 2x memory)
GGML_TYPE_Q8_0: 8-bit quantized (50% memory, small quality loss)
GGML_TYPE_Q4_0: 4-bit quantized (25% memory, noticeable quality loss)

// Memory comparison (LLaMA-7B, 2048 context):
FP16:  256 MB
Q8_0:  128 MB (2x savings)
Q4_0:   64 MB (4x savings)
```

**Trade-offs**:
- FP16: Standard, good balance
- Q8_0: ~1% quality loss, 50% memory savings
- Q4_0: ~3-5% quality loss, 75% memory savings

### 2. Grouped Query Attention (GQA)

Reduce KV heads:

```
MHA (32 heads):
  K cache: n_ctx × 32 × head_dim
  V cache: n_ctx × 32 × head_dim

GQA (8 KV heads):
  K cache: n_ctx × 8 × head_dim
  V cache: n_ctx × 8 × head_dim

Memory savings: 4x!
```

### 3. Sliding Window Attention

Mistral's approach: Only cache last N tokens per layer:

```cpp
// Sliding window configuration
const uint32_t window_size = 4096;  // Mistral

// Cache only recent tokens
if (pos >= window_size) {
    // Evict oldest token
    uint32_t evict_pos = pos - window_size;
    cache.cells[evict_pos % cache.size].pos = -1;
}

// Cache new token at position pos % window_size
cache.cells[pos % cache.size].pos = pos;
```

**Benefits**:
- Constant memory per layer
- Can generate infinite length
- Still sees full context via layer stacking

### 4. Multi-Query Attention (MQA)

Extreme optimization: Single KV head:

```
MHA (32 heads):   32 KV heads
GQA (8 heads):     8 KV heads (4x savings)
MQA (1 head):      1 KV head  (32x savings!)

Quality impact: ~5-10% worse than MHA
Use case: Resource-constrained environments
```

---

## Advanced Cache Management

### Multi-Sequence Batching

Support multiple independent sequences in parallel:

```cpp
// Sequence-aware cache
struct llama_kv_cell {
    llama_pos pos = -1;
    std::set<llama_seq_id> seq_id;  // Multiple sequences can use this cell
};

// Example: 2 independent conversations
// Sequence 0: "Tell me about AI"
// Sequence 1: "What is quantum computing?"

// Cache cells:
// pos=0: seq_id={0}      "Tell"
// pos=1: seq_id={0}      "me"
// pos=2: seq_id={0}      "about"
// pos=3: seq_id={0}      "AI"
// pos=4: seq_id={1}      "What"
// pos=5: seq_id={1}      "is"
// pos=6: seq_id={1}      "quantum"
// pos=7: seq_id={1}      "computing"
```

### Cache Defragmentation

When cache becomes fragmented:

```cpp
void llama_kv_cache_defrag(struct llama_kv_cache & cache) {
    // Move all active cells to the front
    uint32_t new_head = 0;

    for (uint32_t i = 0; i < cache.size; i++) {
        if (cache.cells[i].pos >= 0) {
            if (i != new_head) {
                // Copy cell
                cache.cells[new_head] = cache.cells[i];

                // Copy K/V tensors (expensive!)
                for (uint32_t il = 0; il < n_layer; il++) {
                    copy_kv_cell(cache, il, i, new_head);
                }
            }
            new_head++;
        }
    }

    cache.used = new_head;
    cache.head = new_head;
}
```

### Rolling Buffer Cache

For infinite generation with fixed memory:

```cpp
// Ring buffer implementation
uint32_t cache_pos = token_pos % cache.size;

// Overwrite oldest entry
cache.cells[cache_pos].pos = token_pos;
cache.cells[cache_pos].seq_id = {seq_id};

// Copy new K/V into cache at cache_pos
// (overwrites old data)
```

---

## Memory Calculations

### Formula

```
Total KV Cache Memory = 2 × n_layer × n_ctx × n_head_kv × head_dim × bytes_per_element

Where:
  2: Both K and V
  n_layer: Number of transformer layers
  n_ctx: Context window size
  n_head_kv: Number of KV heads (GQA)
  head_dim: Dimensions per head
  bytes_per_element: 2 (FP16), 1 (Q8_0), 0.5 (Q4_0)
```

### Examples

**LLaMA-7B (4K context, FP16)**:
```
2 × 32 layers × 4096 ctx × 8 KV heads × 128 head_dim × 2 bytes
= 2 × 32 × 4096 × 8 × 128 × 2
= 536,870,912 bytes
= 512 MB
```

**LLaMA-7B (4K context, Q8_0)**:
```
2 × 32 × 4096 × 8 × 128 × 1
= 268,435,456 bytes
= 256 MB
```

**LLaMA-70B (4K context, FP16)**:
```
2 × 80 layers × 4096 ctx × 8 KV heads × 128 head_dim × 2 bytes
= 2 × 80 × 4096 × 8 × 128 × 2
= 1,342,177,280 bytes
= 1.25 GB
```

**Extended Context (32K)**:
```
LLaMA-7B at 32K context (8x longer):
  512 MB × 8 = 4 GB just for KV cache!

Impact: Context length has linear impact on memory
```

---

## Performance Characteristics

### Memory Bandwidth

KV cache access pattern:

```
Token generation step N:
  Read:  K[0..N-1], V[0..N-1] from cache
  Write: K[N], V[N] to cache

Memory traffic:
  Read:  2 × N × n_layer × n_head_kv × head_dim × bytes
  Write: 2 × 1 × n_layer × n_head_kv × head_dim × bytes

For N=100 tokens:
  Read:  ~100x more data than write
  Bottleneck: Memory bandwidth, not compute!
```

### Optimization Strategies

**1. Reduce KV heads (GQA/MQA)**
- Fewer heads = less data to read
- 8 KV heads vs 32: 4x bandwidth savings

**2. Quantize cache (Q8_0)**
- Half the data to read
- Minimal quality impact

**3. Better memory layout**
- Contiguous storage
- Cache-line aligned
- Vectorized access

**4. GPU offloading**
- GPU memory bandwidth >> CPU RAM
- Keep cache on GPU when possible

---

## Debugging Cache Issues

### Common Problems

**1. Out of Memory**

```bash
Error: Failed to allocate KV cache (requested 2048 MB)

Causes:
  - Context too long for available memory
  - Trying to use FP32 cache (use FP16 or Q8_0)
  - Multiple concurrent contexts

Solutions:
  - Reduce context length (--ctx-size 2048)
  - Use quantized cache (--cache-type-k q8_0)
  - Use GQA/MQA model
  - Enable GPU offloading
```

**2. Quality Degradation**

```
Output becomes incoherent after 1000 tokens

Causes:
  - Over-quantized cache (Q4_0)
  - Cache corruption
  - Sliding window too small

Solutions:
  - Use FP16 or Q8_0 cache
  - Increase window size
  - Check for buffer overruns
```

**3. Slow Generation**

```
Generation slows down as context grows

Causes:
  - Linear attention cost growth
  - Memory bandwidth saturation
  - Cache not on GPU

Solutions:
  - Use Flash Attention
  - Enable GPU offloading
  - Use sliding window attention
```

---

## API Usage

### llama-cpp-python

```python
from llama_cpp import Llama

# Create model with custom cache settings
llm = Llama(
    model_path="model.gguf",
    n_ctx=4096,           # Context window
    n_gpu_layers=32,      # Offload cache to GPU
    type_k=2,             # Q8_0 for K cache
    type_v=2,             # Q8_0 for V cache
)

# Generate with cache
output = llm("Hello, world!", max_tokens=100)

# Cache is automatically managed
# - Reused for same conversation
# - Cleared when starting new conversation
```

### C++ API

```cpp
// Create context with cache configuration
llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx = 4096;
ctx_params.type_k = GGML_TYPE_Q8_0;  // Quantized K cache
ctx_params.type_v = GGML_TYPE_Q8_0;  // Quantized V cache
ctx_params.offload_kqv = true;       // GPU offload

llama_context * ctx = llama_new_context_with_model(model, ctx_params);

// Cache is managed internally during inference
// No manual cache management needed for basic usage
```

---

## Interview Questions

**Q1: Why is KV caching so critical for LLM inference?**

**Answer**: Without KV cache, attention computation is O(N²) for N tokens because each new token requires recomputing attention for all previous tokens. With KV cache, we store K and V from previous tokens and only compute Q, K, V for the current token, reducing complexity to O(N). For a 100-token sequence, this is ~50x speedup.

**Q2: How does Grouped Query Attention (GQA) reduce KV cache size?**

**Answer**: GQA uses fewer KV heads than query heads. For example, 32 query heads might share 8 KV heads (4 queries per KV). This reduces KV cache size by 4x with minimal quality loss (<1%). The memory savings are critical for long-context scenarios where KV cache can exceed model size.

**Q3: What are the trade-offs of quantizing the KV cache?**

**Answer**:
- FP16 (standard): 100% quality, baseline memory
- Q8_0: ~99% quality, 50% memory
- Q4_0: ~95% quality, 25% memory

Trade-off: Memory vs quality. Q8_0 is usually optimal (minimal quality loss, significant savings). Q4_0 may cause coherence issues in long generations.

**Q4: How would you handle very long context windows (100K+ tokens)?**

**Answer**: Several strategies:
1. **Sliding Window**: Cache only recent tokens (Mistral approach)
2. **Sparse Attention**: Don't attend to all tokens
3. **Hierarchical Caching**: Different resolution for different distances
4. **Quantized Cache**: Q8_0 or Q4_0 to reduce memory
5. **Disk Paging**: Swap old cache to disk (slow but feasible)
6. **Compression**: Compress old KV pairs

---

## Further Reading

### Code References
- [llama-kv-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp): KV cache implementation
- [ggml-attention.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/ggml/src/ggml-attention.cpp): Attention with KV cache

### Research Papers
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180): vLLM paper
- [Mistral 7B](https://arxiv.org/abs/2310.06825): Sliding window attention
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [Multi-Query Attention](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245)

### Tutorials
- [Lab 3: KV Cache Profiling](../labs/lab-03-kv-cache-profiling.ipynb)
- [Code Example: Cache Visualizer](../code/kv_cache_visualizer.py)
- [Tutorial: Memory Optimization](../tutorials/tutorial-02-custom-samplers.ipynb)

---

**Last Updated**: 2025-11-18
**Author**: Agent 5 (Documentation Writer)
**Reviewed By**: Agent 7 (Quality Validator)
**Module**: 2 - Core Implementation
