# Inference Fundamentals

**Learning Module**: Module 1 - Foundations
**Estimated Reading Time**: 16 minutes
**Prerequisites**: Basic understanding of neural networks, transformers, and attention mechanisms
**Related Content**:
- [What is llama.cpp?](./01-what-is-llama-cpp.md)
- [GGUF Format Deep Dive](./02-gguf-format-deep-dive.md)
- [Lab 3: Understanding KV Cache](../../labs/lab-03/)

---

## What is LLM Inference?

Inference is the process of generating output from a trained neural network model given input data. For Large Language Models (LLMs), inference means generating text tokens one at a time based on an input prompt.

### Key Difference: Training vs. Inference

```
Training:
├─ Learns patterns from data
├─ Updates model weights
├─ Requires large datasets
├─ Computationally expensive
└─ Happens once (or infrequently)

Inference:
├─ Uses learned patterns
├─ Weights are frozen (no updates)
├─ Works with single examples
├─ Must be fast and efficient
└─ Happens millions of times
```

---

## The Token Generation Loop

### Overview

LLM text generation works in an iterative loop where each token is predicted based on all previous tokens:

```
User Input: "The capital of France is"

Iteration 1: → predict → " Paris"
Iteration 2: → predict → "."
Iteration 3: → predict → " It"
Iteration 4: → predict → " is"
...

Final Output: "The capital of France is Paris. It is..."
```

### Detailed Generation Process

```
┌───────────────────────────────────────────────────────┐
│         Token Generation Loop (Autoregressive)        │
├───────────────────────────────────────────────────────┤
│                                                        │
│  1. INPUT PHASE (Prompt Processing)                   │
│     ┌────────────────────────────────────┐           │
│     │ Tokenize: "Hello" → [156, 1245]   │           │
│     └──────────────┬─────────────────────┘           │
│                    ↓                                   │
│     ┌────────────────────────────────────┐           │
│     │ Embed: tokens → vectors            │           │
│     └──────────────┬─────────────────────┘           │
│                    ↓                                   │
│     ┌────────────────────────────────────┐           │
│     │ Process: Full forward pass         │           │
│     │ - All tokens processed in parallel │           │
│     │ - Build KV cache                   │           │
│     └──────────────┬─────────────────────┘           │
│                    ↓                                   │
│     ┌────────────────────────────────────┐           │
│     │ Output: Logits for last token      │           │
│     └──────────────┬─────────────────────┘           │
│                    │                                   │
│  ──────────────────┼─────────────────────────────────│
│                    │                                   │
│  2. GENERATION PHASE (Token-by-Token)                 │
│     │                                                  │
│     └──→ Loop until EOS or max length:                │
│          ┌────────────────────────────────────┐      │
│          │ Sample: Pick next token from logits│      │
│          │ - Apply temperature                 │      │
│          │ - Apply top-k/top-p                │      │
│          │ - Select token                     │      │
│          └──────────────┬─────────────────────┘      │
│                         ↓                              │
│          ┌────────────────────────────────────┐      │
│          │ Update: Add token to context       │      │
│          └──────────────┬─────────────────────┘      │
│                         ↓                              │
│          ┌────────────────────────────────────┐      │
│          │ Process: Forward pass (one token)  │      │
│          │ - Use cached KV for previous tokens│      │
│          │ - Only compute new token           │      │
│          │ - Update KV cache                  │      │
│          └──────────────┬─────────────────────┘      │
│                         ↓                              │
│          ┌────────────────────────────────────┐      │
│          │ Output: Logits for next token      │      │
│          └──────────────┬─────────────────────┘      │
│                         │                              │
│                         └────→ Repeat                  │
│                                                        │
└───────────────────────────────────────────────────────┘
```

### Phases in Detail

#### Phase 1: Prompt Processing (Prefill)

```c
// Example pseudo-code for prompt processing
void process_prompt(context* ctx, token* tokens, int n_tokens) {
    // 1. Tokenization (already done)

    // 2. Create batch with all prompt tokens
    llama_batch batch = llama_batch_init(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;              // Position in sequence
        batch.seq_id[i] = 0;           // Sequence ID (for multi-sequence)
        batch.logits[i] = (i == n_tokens - 1); // Only need logits for last token
    }

    // 3. Process entire prompt in one forward pass
    llama_decode(ctx, batch);

    // KV cache is now populated with K/V vectors for all prompt tokens
}
```

**Key Points**:
- All prompt tokens processed in parallel (efficient on GPU)
- KV cache populated for all positions
- Only the last token's logits are needed for generation
- This is the "slow" part for long prompts

#### Phase 2: Token Generation (Decode)

```c
// Example pseudo-code for token generation
token generate_next_token(context* ctx, int current_pos) {
    // 1. Sample token from previous logits
    token next_token = sample_from_logits(ctx->logits);

    // 2. Create batch with single token
    llama_batch batch = llama_batch_init(1);
    batch.token[0] = next_token;
    batch.pos[0] = current_pos;
    batch.seq_id[0] = 0;
    batch.logits[0] = true; // Need logits for next prediction

    // 3. Process single token
    // - Reuses cached K/V for all previous tokens
    // - Only computes K/V for this new token
    llama_decode(ctx, batch);

    // 4. Return token and continue
    return next_token;
}
```

**Key Points**:
- Only one token processed at a time
- KV cache reused (fast!)
- This is the "fast" part (memory-bound, not compute-bound)
- Repeated until EOS token or max length

---

## Context and Context Window

### What is Context?

The **context** is the sequence of tokens the model can "see" when generating the next token. It includes:
1. The original prompt
2. All previously generated tokens

### Context Window (Context Length)

The **context window** is the maximum number of tokens the model can handle at once.

```
Model: LLaMA-7B (context window: 2048)

Tokens:  [prompt tokens] + [generated tokens] = total tokens
Limit:   Must not exceed 2048 tokens

If total = 2048:
  ├─ No more tokens can be generated
  └─ Must either:
      ├─ Stop generation
      ├─ Truncate old tokens
      └─ Use context extension techniques
```

### Context Management Strategies

#### 1. Simple Truncation

```python
# When context is full, drop oldest tokens
if current_pos >= max_context:
    # Remove first N tokens
    tokens = tokens[N:]
    current_pos -= N
```

**Pros**: Simple
**Cons**: Loses information, model "forgets" beginning

#### 2. Sliding Window

```python
# Keep a fixed-size window of recent tokens
window_size = 2048
if len(tokens) > window_size:
    tokens = tokens[-window_size:]
```

**Pros**: Maintains recent context
**Cons**: Still loses old information

#### 3. Summary + Recent Tokens

```python
# Keep summary of old context + recent tokens
if len(tokens) > max_context - summary_length:
    # Summarize old tokens
    summary = generate_summary(tokens[:-recent_length])
    # Keep summary + recent tokens
    tokens = summary + tokens[-recent_length:]
```

**Pros**: Preserves key information
**Cons**: Summary might lose details

---

## KV Cache: The Performance Secret

### What is the KV Cache?

The **KV cache** (Key-Value cache) stores intermediate attention computations to avoid redundant calculations.

### Why KV Cache is Essential

Without KV cache, generating N tokens requires:

```
Token 1: Process 1 token    →  1 operation
Token 2: Process 2 tokens   →  2 operations
Token 3: Process 3 tokens   →  3 operations
...
Token N: Process N tokens   →  N operations

Total: 1 + 2 + 3 + ... + N = N(N+1)/2 ≈ O(N²)
```

With KV cache:

```
Token 1: Process 1 token    →  1 operation (cache K/V)
Token 2: Process 1 token    →  1 operation (reuse cached K/V)
Token 3: Process 1 token    →  1 operation (reuse cached K/V)
...
Token N: Process 1 token    →  1 operation (reuse cached K/V)

Total: N operations → O(N)
```

**Speedup**: N/2 times faster! (For N=100, that's 50x faster!)

### How KV Cache Works

#### Attention Mechanism (Simplified)

```python
# Standard attention (no cache)
def attention(Q, K, V):
    scores = Q @ K.T / sqrt(d_k)      # Attention scores
    weights = softmax(scores)          # Attention weights
    output = weights @ V               # Weighted values
    return output

# For each new token, we need K and V for ALL previous tokens
```

#### With KV Cache

```python
# Attention with cache
def attention_with_cache(Q_new, kv_cache, layer, pos):
    # Get cached K/V for all previous tokens
    K_prev = kv_cache[layer]["K"][:pos]  # Shape: [pos, d_k]
    V_prev = kv_cache[layer]["V"][:pos]  # Shape: [pos, d_v]

    # Compute K/V for new token only
    K_new = compute_K(Q_new)
    V_new = compute_V(Q_new)

    # Update cache
    kv_cache[layer]["K"][pos] = K_new
    kv_cache[layer]["V"][pos] = V_new

    # Concatenate for attention
    K_all = concat([K_prev, K_new])  # All keys
    V_all = concat([V_prev, V_new])  # All values

    # Compute attention (Q_new only attends to all K/V)
    scores = Q_new @ K_all.T / sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V_all

    return output
```

### KV Cache Memory Requirements

For a transformer model, the KV cache size is:

```
KV_cache_size = 2 × n_layers × seq_len × n_embd × bytes_per_element

Example (LLaMA-7B):
- n_layers = 32
- seq_len = 2048 (context window)
- n_embd = 4096
- bytes_per_element = 2 (FP16) or 4 (FP32)

FP16: 2 × 32 × 2048 × 4096 × 2 = 1,073,741,824 bytes = 1 GB
FP32: 2 × 32 × 2048 × 4096 × 4 = 2,147,483,648 bytes = 2 GB
```

**Memory Impact**:
```
7B model (Q4_K_M):
├─ Model weights: ~3.5 GB
├─ KV cache (FP16): ~1 GB (per sequence!)
└─ Runtime overhead: ~0.5 GB
Total: ~5 GB minimum
```

### KV Cache Optimization Techniques

#### 1. Quantized KV Cache

```c
// Quantize KV cache to save memory
// Q8 KV cache: 2x memory reduction
// Q4 KV cache: 4x memory reduction (with quality loss)

kv_cache_type = Q8_0;  // 8-bit quantization
// Memory: 1 GB → 500 MB for 7B model
```

#### 2. Multi-Query Attention (MQA)

```python
# Standard Multi-Head Attention
n_heads = 32
n_kv_heads = 32  # Same as n_heads

# Multi-Query Attention (MQA)
n_heads = 32
n_kv_heads = 1   # Share K/V across all heads
# Memory reduction: 32x for KV cache!

# Grouped-Query Attention (GQA) - compromise
n_heads = 32
n_kv_heads = 8   # Share K/V across groups
# Memory reduction: 4x for KV cache
```

#### 3. PagedAttention (vLLM-style)

```
Traditional KV cache:
┌────────────────────────────────┐
│ Contiguous memory allocation   │
│ [seq1] [seq2] [seq3] ...       │
└────────────────────────────────┘
Problem: Fragmentation, cannot share

PagedAttention:
┌──────┬──────┬──────┬──────┐
│ Block│ Block│ Block│ Block│
│  1   │  2   │  3   │  4   │
└──────┴──────┴──────┴──────┘
     ↑       ↑       ↑
     Shared across sequences if identical
```

---

## Memory Management

### Memory Layout

```
┌─────────────────────────────────────────────┐
│          llama.cpp Memory Layout             │
├─────────────────────────────────────────────┤
│                                              │
│  ┌─────────────────────────────────┐       │
│  │ Model Weights (mmap'd or loaded)│ ~3.5GB│
│  │ - Quantized tensors             │       │
│  │ - Loaded from GGUF file         │       │
│  └─────────────────────────────────┘       │
│                                              │
│  ┌─────────────────────────────────┐       │
│  │ KV Cache                         │ ~1GB  │
│  │ - Per layer, per position        │       │
│  │ - Size: n_layers × seq_len × d   │       │
│  └─────────────────────────────────┘       │
│                                              │
│  ┌─────────────────────────────────┐       │
│  │ Computation Buffers              │ ~0.5GB│
│  │ - Activation tensors             │       │
│  │ - Temporary buffers              │       │
│  └─────────────────────────────────┘       │
│                                              │
│  ┌─────────────────────────────────┐       │
│  │ Input/Output Buffers             │ ~0.1GB│
│  │ - Token buffers                  │       │
│  │ - Logits                         │       │
│  └─────────────────────────────────┘       │
│                                              │
└─────────────────────────────────────────────┘
Total: ~5 GB for 7B Q4_K_M model
```

### Memory Allocation Strategies

#### 1. Memory Mapping (mmap)

```c
// Load model with mmap (default)
struct llama_model_params params = llama_model_default_params();
params.use_mmap = true;  // Enable mmap

// Benefits:
// - Fast loading (instant)
// - OS manages memory
// - Shared across processes
// - Lazy loading (only load what's used)
```

#### 2. Memory Locking (mlock)

```c
// Keep model in RAM (prevent swapping)
params.use_mlock = true;

// Benefits:
// - Prevents disk swapping
// - More predictable performance
// - No page faults during inference

// Drawbacks:
// - Uses system resources
// - May require increased ulimit
```

#### 3. Explicit Loading

```c
// Load entire model into RAM
params.use_mmap = false;

// Benefits:
// - Full control over memory
// - May be faster on some systems

// Drawbacks:
// - Slower loading
// - More memory used
// - No sharing across processes
```

### Memory Budgeting

```python
def estimate_memory_usage(model_size, context_length, n_layers, d_model):
    """Estimate memory usage for inference"""

    # Model weights (quantized)
    model_memory = model_size * (4 / 32)  # Assuming Q4 quantization

    # KV cache
    bytes_per_element = 2  # FP16
    kv_cache = 2 * n_layers * context_length * d_model * bytes_per_element

    # Computation buffers (rough estimate)
    compute_buffer = 0.5 * 1e9  # 500 MB

    # Total
    total = model_memory + kv_cache + compute_buffer

    return {
        "model": model_memory / 1e9,
        "kv_cache": kv_cache / 1e9,
        "compute": compute_buffer / 1e9,
        "total": total / 1e9
    }

# Example: LLaMA-7B
memory = estimate_memory_usage(
    model_size=7e9,      # 7B parameters
    context_length=2048,
    n_layers=32,
    d_model=4096
)
# Output: ~5 GB total
```

---

## Batching and Throughput

### Single vs. Batched Inference

#### Single Request (Latency-Optimized)

```python
# Process one request at a time
for request in requests:
    output = model.generate(request.prompt)
    respond(output)

# Throughput: ~10-20 tokens/sec per request
# Latency: Optimal (no waiting)
# GPU Utilization: Low (25-40%)
```

#### Batched Inference (Throughput-Optimized)

```python
# Process multiple requests together
batch = collect_requests(max_size=8)
outputs = model.generate_batch(batch)
for output in outputs:
    respond(output)

# Throughput: ~100-150 tokens/sec total (5-10 per request)
# Latency: Higher (wait for batch)
# GPU Utilization: High (80-95%)
```

### Continuous Batching

```
Traditional Batching:
Request 1: ████████████████████ (wait for all to finish)
Request 2: ████████████████████
Request 3: ████████████████████

Continuous Batching (vLLM-style):
Request 1: ████████████████████
Request 2:     ████████████████████
Request 3:         ████████████████████
           └─ Add new requests dynamically
```

---

## Sampling and Decoding Strategies

### Sampling Methods

#### 1. Greedy Decoding

```python
# Always pick the most likely token
next_token = argmax(logits)

# Deterministic, but may be repetitive
# Example: "The cat sat on the mat. The cat sat on the mat."
```

#### 2. Temperature Sampling

```python
# Adjust probability distribution
temperature = 0.7  # Lower = more focused, Higher = more random

# Apply temperature
logits = logits / temperature

# Sample from distribution
probs = softmax(logits)
next_token = sample(probs)
```

**Temperature Effect**:
```
Temperature 0.0: Greedy (most likely token)
Temperature 0.5: Focused (slightly random)
Temperature 1.0: Standard (original probabilities)
Temperature 1.5: Creative (more random)
Temperature 2.0: Very random (chaotic)
```

#### 3. Top-K Sampling

```python
# Only consider top K most likely tokens
K = 40

# Sort and keep top K
top_k_logits = topk(logits, K)

# Sample from top K only
probs = softmax(top_k_logits)
next_token = sample(probs)
```

#### 4. Top-P (Nucleus) Sampling

```python
# Keep tokens until cumulative probability exceeds P
P = 0.9  # Keep 90% probability mass

# Sort by probability
sorted_probs = sort(softmax(logits))

# Find cutoff where cumulative sum exceeds P
cumsum = cumulative_sum(sorted_probs)
cutoff = find_first(cumsum > P)

# Sample from nucleus
probs = sorted_probs[:cutoff]
next_token = sample(probs)
```

#### 5. Repetition Penalty

```python
# Penalize tokens that were recently generated
penalty = 1.2  # > 1.0 = discourage repetition

for token in recent_tokens:
    if logits[token] > 0:
        logits[token] /= penalty
    else:
        logits[token] *= penalty
```

### Complete Sampling Pipeline

```python
def sample_next_token(logits, config):
    """Complete sampling with all techniques"""

    # 1. Apply repetition penalty
    logits = apply_repetition_penalty(logits, config.recent_tokens, config.penalty)

    # 2. Apply temperature
    logits = logits / config.temperature

    # 3. Apply top-k
    if config.top_k > 0:
        logits = apply_top_k(logits, config.top_k)

    # 4. Apply top-p
    if config.top_p < 1.0:
        logits = apply_top_p(logits, config.top_p)

    # 5. Sample from final distribution
    probs = softmax(logits)
    token = sample(probs)

    return token
```

---

## Performance Characteristics

### Typical Inference Speed (7B Model Q4_K_M)

| Hardware | Prefill (tok/s) | Generation (tok/s) | Latency (ms/tok) |
|----------|----------------|-------------------|------------------|
| M1 CPU | 100-200 | 15-20 | 50-67 |
| M1 GPU (Metal) | 500-800 | 40-60 | 17-25 |
| i7 CPU | 80-150 | 10-15 | 67-100 |
| RTX 3090 | 2000-3000 | 80-120 | 8-12 |
| RTX 4090 | 3000-5000 | 120-180 | 6-8 |

### Bottlenecks

#### Prefill Phase (Prompt Processing)
- **Bottleneck**: Compute (matrix multiplications)
- **Optimized by**: GPU acceleration, larger batch sizes
- **Scaling**: Linear with prompt length

#### Generation Phase (Token Decoding)
- **Bottleneck**: Memory bandwidth (loading KV cache)
- **Optimized by**: Quantization, KV cache quantization, faster memory
- **Scaling**: Constant per token (with cache)

---

## Interview Questions

**Q: "Why is inference with KV cache O(N) instead of O(N²)?"**

**A**: Discuss:
- Without cache: Each token re-computes attention over all previous tokens
- With cache: Store K/V vectors, only compute for new token
- Attention still looks at all previous tokens, but K/V are cached
- Computation: O(1) per token × N tokens = O(N)

**Q: "What's the trade-off between batch size and latency?"**

**A**: Cover:
- Larger batch → higher throughput (GPU utilization)
- Larger batch → higher latency (waiting for batch to fill)
- Smaller batch → lower latency (immediate processing)
- Smaller batch → lower throughput (underutilized GPU)
- Continuous batching provides best of both worlds

**Q: "Why is the KV cache so memory-intensive?"**

**A**: Explain:
- Size: 2 (K+V) × n_layers × seq_len × d_model × bytes
- Example: 2 × 32 × 2048 × 4096 × 2 = 1 GB
- Grows with sequence length (not batch size)
- FP16 precision (2 bytes per element)
- Solutions: Quantization, MQA/GQA, paged attention

---

## Further Reading

### Official Documentation
- [llama.cpp API Reference](../../../reference/llama-h-api.md)
- [Performance Tips](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/token_generation_performance_tips.md)

### Related Content
- [What is llama.cpp?](./01-what-is-llama-cpp.md)
- [Lab 3: Understanding KV Cache](../../labs/lab-03/)
- [Advanced Inference Optimization](../../../modules/05-performance/docs/01-optimization-techniques.md)

### Research Papers
- [Attention is All You Need](../../../papers/summaries/attention-is-all-you-need.md) (Agent 1)
- [FlashAttention Paper](../../../papers/summaries/flash-attention.md) (Agent 1)
- [PagedAttention (vLLM)](../../../papers/summaries/paged-attention.md) (Agent 1)

---

**Last Updated**: 2025-11-18
**Author**: Agent 5 (Documentation Writer)
**Reviewed By**: Agent 7 (Quality Validator)
**Feedback**: [Submit feedback](../../../feedback/)
