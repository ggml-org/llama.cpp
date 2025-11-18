# Inference Pipeline

**Learning Module**: Module 2 - Core Implementation
**Estimated Reading Time**: 32 minutes
**Prerequisites**: Module 1, understanding of transformer architecture
**Related Content**:
- [Model Architecture Deep Dive](./01-model-architecture-deep-dive.md)
- [KV Cache Implementation](./03-kv-cache-implementation.md)
- [Sampling Strategies](./05-sampling-strategies.md)

---

## Overview

This document traces the complete inference pipeline in llama.cpp, from loading a model to generating tokens. Understanding this pipeline is essential for optimization, debugging, and implementing custom features.

### Learning Objectives

After completing this lesson, you will:
- ✅ Trace the complete token generation pipeline
- ✅ Understand the forward pass through transformer layers
- ✅ Identify performance bottlenecks
- ✅ Debug inference issues
- ✅ Optimize inference performance

---

## Pipeline Overview

### High-Level Flow

```
┌─────────────────────────────────────────────────────┐
│           LLM Inference Pipeline                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. Model Loading                                    │
│     ↓                                                │
│  2. Context Initialization                           │
│     ↓                                                │
│  3. Prompt Processing (Prefill)                      │
│     - Tokenize prompt                                │
│     - Batch process all prompt tokens                │
│     - Fill KV cache                                  │
│     ↓                                                │
│  4. Token Generation Loop (Decode)                   │
│     ┌──────────────────────────────┐               │
│     │ a. Forward pass (1 token)     │               │
│     │ b. Get logits                 │               │
│     │ c. Apply sampling             │               │
│     │ d. Select next token          │               │
│     │ e. Check stop conditions      │               │
│     └──────────┬───────────────────┘               │
│                │ Continue?                           │
│                └─→ Yes: Loop back                   │
│                    No: Done                          │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Prefill vs Decode

Two distinct phases with different characteristics:

**Prefill Phase** (Process prompt):
```
Input:  Full prompt tokens [tok_0, tok_1, ..., tok_N]
Output: Logits for next token
Cache:  Fill KV cache with all prompt tokens

Characteristics:
- Process multiple tokens in parallel
- High computation (large matrix multiplications)
- KV cache initially empty, fills during process
- Latency: Depends on prompt length
- Bottleneck: Compute-bound (matrix ops)
```

**Decode Phase** (Generate tokens):
```
Input:  Single new token [tok_new]
Output: Logits for next token
Cache:  Read previous KV, append new KV

Characteristics:
- Process one token at a time
- Lower computation per token
- KV cache grows with each token
- Latency: Per-token generation time
- Bottleneck: Memory bandwidth (reading KV cache)
```

**Performance Comparison**:
```
7B model, 100-token prompt:

Prefill:  100 tokens in ~2 seconds  = 50 tok/s
Decode:   1 token in ~50ms          = 20 tok/s

Why slower? Decode must read entire KV cache for each token
```

---

## Step 1: Model Loading

### Loading Process

```cpp
// 1. Load model from file
llama_model_params model_params = llama_model_default_params();
model_params.n_gpu_layers = 32;  // GPU offloading

llama_model * model = llama_load_model_from_file(
    "model.gguf",
    model_params
);

// What happens internally:
// a. Read GGUF header and metadata
// b. Parse model architecture
// c. Allocate tensors
// d. Load weights (mmap or read)
// e. Offload layers to GPU if requested
```

### Memory Mapping

llama.cpp uses memory-mapped file I/O for efficiency:

```cpp
// Memory-mapped loading
void * mmap_ptr = mmap(
    nullptr,
    file_size,
    PROT_READ,
    MAP_SHARED,
    fd,
    0
);

// Benefits:
// - No explicit loading needed
// - OS handles paging
// - Multiple processes can share
// - Faster startup (lazy loading)

// Model weights accessed directly from file:
float * weights = (float *)(mmap_ptr + tensor_offset);
```

**Advantages**:
- **Fast Startup**: Don't load entire model into RAM
- **Efficient Memory**: OS pages in data as needed
- **Shared Memory**: Multiple instances share same file

---

## Step 2: Context Initialization

### Creating Context

```cpp
// 2. Create inference context
llama_context_params ctx_params = llama_context_default_params();

ctx_params.n_ctx        = 4096;           // Context window
ctx_params.n_batch      = 512;            // Batch size (prefill)
ctx_params.n_ubatch     = 512;            // Micro-batch size
ctx_params.n_threads    = 8;              // CPU threads
ctx_params.type_k       = GGML_TYPE_F16;  // K cache type
ctx_params.type_v       = GGML_TYPE_F16;  // V cache type
ctx_params.offload_kqv  = true;           // GPU offload KQV

llama_context * ctx = llama_new_context_with_model(model, ctx_params);

// Allocates:
// - KV cache (~512 MB for 7B model, 4K context)
// - Compute buffers
// - Threading resources
```

### KV Cache Allocation

```cpp
// Inside llama_new_context_with_model
for (int il = 0; il < n_layer; il++) {
    // Allocate K cache: [n_ctx, n_head_kv, head_dim]
    kv_self.k_l[il] = ggml_new_tensor_3d(
        ctx_kv,
        type_k,
        n_embd_k_gqa,  // head_dim * n_head_kv
        n_ctx,
        1
    );

    // Allocate V cache: [n_ctx, n_head_kv, head_dim]
    kv_self.v_l[il] = ggml_new_tensor_3d(
        ctx_kv,
        type_v,
        n_embd_v_gqa,
        n_ctx,
        1
    );
}
```

---

## Step 3: Prompt Processing (Prefill)

### Tokenization

```cpp
// 3. Tokenize prompt
std::string prompt = "The quick brown fox";

std::vector<llama_token> tokens = llama_tokenize(
    model,
    prompt,
    true,   // add_bos
    false   // special tokens
);

// Example tokens: [1, 450, 4996, 17354, 1701, 29916]
//                  ^BOS
```

### Batching

```cpp
// Create batch for prefill
llama_batch batch = llama_batch_init(n_tokens, 0, 1);

for (int i = 0; i < n_tokens; i++) {
    batch.token[i]  = tokens[i];
    batch.pos[i]    = i;            // Position in sequence
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;         // Sequence ID
    batch.logits[i] = false;        // Don't need logits for prompt tokens
}
batch.logits[n_tokens - 1] = true;  // Only need logits for last token

batch.n_tokens = n_tokens;
```

### Forward Pass (Prefill)

```cpp
// Process entire prompt in one forward pass
int ret = llama_decode(ctx, batch);

// What happens:
// For each token in batch:
//   1. Embedding lookup
//   2. For each layer:
//      a. Attention (parallel across tokens)
//      b. FFN
//      c. Update KV cache
//   3. Final layer norm
//   4. Output projection (only for last token)

// Result: KV cache filled with prompt tokens
```

---

## Step 4: Token Generation Loop (Decode)

### Single Token Forward Pass

```cpp
// Generation loop
int n_generated = 0;
const int max_tokens = 100;

while (n_generated < max_tokens) {
    // 4a. Forward pass for single token
    llama_batch batch_single = llama_batch_init(1, 0, 1);
    batch_single.token[0]    = current_token;
    batch_single.pos[0]      = n_prompt + n_generated;
    batch_single.n_seq_id[0] = 1;
    batch_single.seq_id[0][0] = 0;
    batch_single.logits[0]   = true;
    batch_single.n_tokens    = 1;

    llama_decode(ctx, batch_single);

    // 4b. Get logits
    float * logits = llama_get_logits(ctx);
    // logits: array of vocab_size floats

    // 4c. Apply sampling
    llama_token new_token = sample(logits, sampling_params);

    // 4d. Check stop conditions
    if (new_token == eos_token) {
        break;
    }

    // 4e. Emit token and continue
    output_tokens.push_back(new_token);
    current_token = new_token;
    n_generated++;
}
```

### Layer-by-Layer Execution

```cpp
// Inside llama_decode for single token

// 1. Token embedding
ggml_tensor * inpL = ggml_get_rows(model.tok_embd, batch.token);
// inpL: [1, n_embd]

// 2. Process each transformer layer
for (int il = 0; il < n_layer; il++) {
    struct llama_layer & layer = model.layers[il];

    // Input to this layer
    ggml_tensor * cur = inpL;

    // 2a. Attention block
    {
        // Layer norm
        ggml_tensor * attn_input = ggml_rms_norm(cur);

        // Q, K, V projections
        ggml_tensor * Q = ggml_mul_mat(layer.wq, attn_input);
        ggml_tensor * K = ggml_mul_mat(layer.wk, attn_input);
        ggml_tensor * V = ggml_mul_mat(layer.wv, attn_input);

        // Apply RoPE to Q and K
        Q = ggml_rope(Q, n_past, n_rot, rope_mode);
        K = ggml_rope(K, n_past, n_rot, rope_mode);

        // Update KV cache
        ggml_build_kv_store(kv_cache.k_l[il], K, n_past);
        ggml_build_kv_store(kv_cache.v_l[il], V, n_past);

        // Attention computation
        // Q: [1, n_head, head_dim]
        // K_cache: [n_past + 1, n_head_kv, head_dim]
        // V_cache: [n_past + 1, n_head_kv, head_dim]

        ggml_tensor * KQV = ggml_flash_attn_ext(
            Q,
            kv_cache.k_l[il],
            kv_cache.v_l[il],
            mask,
            scale
        );

        // Output projection
        ggml_tensor * attn_out = ggml_mul_mat(layer.wo, KQV);

        // Residual connection
        cur = ggml_add(cur, attn_out);
    }

    // 2b. FFN block
    {
        // Layer norm
        ggml_tensor * ffn_input = ggml_rms_norm(cur);

        // SwiGLU activation
        ggml_tensor * ffn_gate = ggml_mul_mat(layer.ffn_gate, ffn_input);
        ggml_tensor * ffn_up   = ggml_mul_mat(layer.ffn_up, ffn_input);

        ffn_gate = ggml_silu(ffn_gate);  // Swish activation
        ggml_tensor * ffn_hidden = ggml_mul(ffn_gate, ffn_up);

        ggml_tensor * ffn_out = ggml_mul_mat(layer.ffn_down, ffn_hidden);

        // Residual connection
        cur = ggml_add(cur, ffn_out);
    }

    // Output of this layer becomes input to next
    inpL = cur;
}

// 3. Final layer norm
inpL = ggml_rms_norm(inpL);

// 4. Output projection (LM head)
ggml_tensor * logits = ggml_mul_mat(model.output, inpL);
// logits: [1, vocab_size]
```

---

## Computational Graph

### Graph Building

llama.cpp uses GGML's computational graph:

```cpp
// Build graph
struct ggml_cgraph * gf = ggml_new_graph(ctx);

// Add operations to graph
ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 100, 100);
ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 100, 100);
ggml_tensor * c = ggml_mul_mat(ctx, a, b);  // c = a × b

// c is the output node
ggml_build_forward_expand(gf, c);

// Execute graph
ggml_graph_compute_with_ctx(ctx, gf, n_threads);

// Result is in c->data
```

**Benefits**:
- Operator fusion opportunities
- Optimized execution order
- Parallelization
- Memory reuse

---

## Performance Characteristics

### FLOPs Analysis

**Per-Token Computation** (7B model):

```
Attention: ~7 GFLOPs
  - Q projection: 4096 × 4096 = 16M ops
  - K projection: 4096 × 512 = 2M ops (GQA)
  - V projection: 4096 × 512 = 2M ops
  - Attention: 32 heads × seq_len × 128 = variable
  - Output: 4096 × 4096 = 16M ops
  Total: ~18M ops × 32 layers = 576M ops

FFN: ~11 GFLOPs
  - Gate: 4096 × 11008 = 45M ops
  - Up: 4096 × 11008 = 45M ops
  - Down: 11008 × 4096 = 45M ops
  Total: 135M ops × 32 layers = 4,320M ops

Total: ~18 GFLOPs per token
```

**Hardware Performance**:
```
NVIDIA RTX 4090: 82 TFLOPs (FP32)
Theoretical max: 4,555 tokens/second
Actual: ~150 tokens/second

Why the gap?
  1. Memory bandwidth (not compute-bound)
  2. Quantization (lower precision, adjusted FLOPs)
  3. Overhead (kernel launches, data transfer)
  4. Inefficiency (not 100% utilization)
```

### Memory Bandwidth

**Decode Phase** (memory-bound):

```
Data to read per token:
  - Model weights: ~7GB (quantized)
  - KV cache: ~seq_len × 512 KB per layer
  - Total: ~7GB + (seq_len × 16MB)

At seq_len = 1000:
  - Read: ~23 GB per token
  - Write: ~16 MB per token

RTX 4090: 1000 GB/s bandwidth
Theoretical tokens/sec: 1000 GB/s ÷ 23 GB = ~43 tok/s
Actual: ~150 tok/s (batch size helps)

Key insight: More memory bandwidth = faster inference
```

---

## Optimization Opportunities

### 1. Batch Processing

Process multiple sequences in parallel:

```cpp
// Single sequence: 20 tok/s
// 8 sequences batched: 100 tok/s total (12.5 tok/s each)

// Throughput: 5x improvement
// Latency: Slightly worse per sequence

// Use case: Server with multiple users
```

### 2. Continuous Batching

Dynamic batching (vLLM-style):

```
Traditional batching:
  Wait for batch to fill → Process → Wait again

Continuous batching:
  Stream requests in/out
  Always keep GPU busy
  Better throughput and latency
```

### 3. Speculative Decoding

Use small model to predict, large model to verify:

```
Small model (1B): Fast, generates 3-5 tokens
Large model (70B): Slow, verifies in parallel

Speedup: 2-3x when predictions correct
No quality loss (verification guarantees correctness)
```

### 4. Quantization

Reduce model size and memory bandwidth:

```
FP16: 7GB, baseline speed
Q8_0: 7GB → 3.5GB (2x speedup potential)
Q4_0: 7GB → 1.75GB (4x speedup potential)

Actual speedup less due to:
  - Dequantization overhead
  - Quality degradation
```

### 5. Flash Attention

Optimized attention implementation:

```
Standard attention:
  - Materializes attention matrix
  - O(N²) memory
  - Multiple kernel launches

Flash Attention:
  - Tiled computation
  - O(N) memory
  - Fused kernel
  - 2-4x faster, especially for long context
```

---

## Debugging the Pipeline

### Common Issues

**1. Slow Prefill**

```bash
# Symptom: Prefill takes 10+ seconds for 100 token prompt

# Check:
- Batch size too small (increase --batch-size)
- CPU inference (enable GPU with --n-gpu-layers)
- Not using Flash Attention
- Debug mode enabled (use release build)

# Fix:
llama-cli -m model.gguf --batch-size 512 --n-gpu-layers 32
```

**2. Slow Decode**

```bash
# Symptom: <5 tok/s on decent hardware

# Check:
- KV cache not on GPU (--offload-kqv)
- High quantization overhead
- Context length very long
- Memory bandwidth saturated

# Fix:
llama-cli -m model.gguf --offload-kqv --ctx-size 2048
```

**3. Out of Memory**

```bash
# Error: Failed to allocate tensor

# Check:
- Context too long (reduce --ctx-size)
- Batch size too large
- Too many layers on GPU
- KV cache too large

# Fix:
llama-cli -m model.gguf --ctx-size 2048 --n-gpu-layers 24
```

### Profiling

```cpp
// Enable timing
llama_context_params params = llama_context_default_params();
params.timings = true;

// After generation
llama_timings timings = llama_get_timings(ctx);

printf("Load time:   %.2f ms\n", timings.t_load_ms);
printf("Eval time:   %.2f ms / %.2f ms per token\n",
       timings.t_eval_ms, timings.t_eval_ms / timings.n_eval);
printf("Prompt eval: %.2f ms / %.2f ms per token\n",
       timings.t_p_eval_ms, timings.t_p_eval_ms / timings.n_p_eval);

// Output:
// Load time:   245.23 ms
// Eval time:   5234.12 ms / 52.34 ms per token
// Prompt eval: 1234.56 ms / 12.35 ms per token
```

---

## Interview Questions

**Q1: Explain the difference between prefill and decode phases in LLM inference.**

**Answer**: Prefill processes all prompt tokens in parallel, filling the KV cache. It's compute-bound and benefits from large batch sizes. Decode generates one token at a time, reading the entire KV cache. It's memory bandwidth-bound and doesn't benefit much from batching within a single sequence. Prefill is typically 2-3x faster per token but only happens once.

**Q2: Why does generation slow down as context length increases?**

**Answer**: In the decode phase, each new token requires reading the entire KV cache to compute attention. As context grows, more data must be read from memory. With O(N) attention, generating token N+1 requires reading N KV pairs. Memory bandwidth becomes the bottleneck, not compute. Solutions include Flash Attention, sliding window attention, or sparse attention patterns.

**Q3: How would you optimize inference for a high-throughput server?**

**Answer**:
1. **Batch multiple requests**: Amortize model loading and maximize GPU utilization
2. **Continuous batching**: Dynamic batching for better latency and throughput
3. **GPU offloading**: Keep model and KV cache on GPU
4. **Quantization**: Reduce memory bandwidth (Q8_0 or Q4_K_M)
5. **PagedAttention**: Efficient KV cache management
6. **Flash Attention**: Faster attention computation
7. **Speculative decoding**: Use small model to speed up large model

---

## Further Reading

### Code References
- [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama.cpp): Main inference loop
- [llama-batch.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-batch.cpp): Batch processing
- [ggml-alloc.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/ggml/src/ggml-alloc.cpp): Memory allocation

### Research Papers
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

### Tutorials
- [Lab 4: Pipeline Profiling](../labs/lab-04-custom-sampling.ipynb)
- [Tutorial: Tracing Token Generation](../tutorials/tutorial-01-transformer-layers.ipynb)
- [Code Example: Pipeline Visualizer](../code/pipeline_tracer.py)

---

**Last Updated**: 2025-11-18
**Author**: Agent 5 (Documentation Writer)
**Reviewed By**: Agent 7 (Quality Validator)
**Module**: 2 - Core Implementation
