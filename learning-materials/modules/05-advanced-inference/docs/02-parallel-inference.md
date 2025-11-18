# Parallel Inference: Batch Processing and Request Handling

**Module 5, Lesson 2**
**Estimated Time**: 3-4 hours
**Difficulty**: Advanced

## Overview

Parallel inference enables serving multiple requests simultaneously, dramatically improving throughput and resource utilization. This lesson covers batch processing strategies, memory management, and optimization techniques for production LLM serving.

## Learning Objectives

By the end of this lesson, you will:
- ✅ Understand static vs dynamic batching
- ✅ Implement efficient batch processing
- ✅ Optimize KV cache memory across batches
- ✅ Handle variable-length sequences
- ✅ Maximize GPU utilization through parallelism

## Why Batch Processing Matters

### Sequential vs Parallel Serving

**Sequential (Naive) Approach**:
```
Request 1: [Generate 100 tokens] → 2.0s
Request 2: [Generate 100 tokens] → 2.0s
Request 3: [Generate 100 tokens] → 2.0s
Total: 6.0s, 150 tokens/sec
```

**Batched Approach**:
```
Batch [Req1, Req2, Req3]: [Generate 100 tokens each] → 2.5s
Total: 2.5s, 400 tokens/sec (2.7x faster!)
```

### Key Benefits

| Metric | Sequential | Batched (B=8) | Improvement |
|--------|-----------|---------------|-------------|
| Throughput | 50 tok/s | 280 tok/s | 5.6x |
| GPU Utilization | 15% | 75% | 5x |
| Latency (avg) | 2.0s | 2.8s | 1.4x slower |
| Cost/Token | $0.001 | $0.0002 | 5x cheaper |

**Trade-off**: Higher throughput, slightly higher latency per request.

## Static Batching

### Concept

Collect B requests, process them together, return all results when the **slowest request** completes.

```python
def static_batch_inference(requests, model, batch_size=8):
    """
    Simple static batching
    """
    results = []

    # Process in batches of batch_size
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i+batch_size]

        # Pad to same length
        max_len = max(len(r.prompt) for r in batch)
        padded_batch = [pad_to_length(r, max_len) for r in batch]

        # Generate for all requests in parallel
        batch_results = model.generate_batch(
            prompts=[r.prompt for r in padded_batch],
            max_tokens=max(r.max_tokens for r in batch)
        )

        results.extend(batch_results)

    return results
```

### Implementation in llama.cpp

```cpp
#include "llama.h"

// Initialize model with batch support
llama_model_params model_params = llama_model_default_params();
model_params.n_batch = 512;  // Maximum batch size

llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx = 4096;      // Context size
ctx_params.n_batch = 512;     // Batch size for processing
ctx_params.n_ubatch = 512;    // Physical batch size

llama_context* ctx = llama_new_context_with_model(model, ctx_params);

// Create batch
llama_batch batch = llama_batch_init(512, 0, 1);

// Add multiple sequences
for (int seq_id = 0; seq_id < num_requests; seq_id++) {
    auto& request = requests[seq_id];

    for (size_t i = 0; i < request.tokens.size(); i++) {
        llama_batch_add(
            batch,
            request.tokens[i],  // token
            i,                   // position
            {seq_id},           // sequence IDs
            false               // logits (false except last)
        );
    }

    // Request logits for last token
    batch.logits[batch.n_tokens - 1] = true;
}

// Process entire batch
llama_decode(ctx, batch);

// Sample from each sequence
for (int seq_id = 0; seq_id < num_requests; seq_id++) {
    llama_token token = llama_sample_token(ctx, seq_id);
    results[seq_id].push_back(token);
}
```

### Memory Layout

Static batching requires careful memory management:

```
KV Cache Layout (B=4, seq_len=100):
┌─────────────────────────────────────┐
│ Sequence 0: [K₀] [V₀] (100 tokens) │
│ Sequence 1: [K₁] [V₁] (100 tokens) │
│ Sequence 2: [K₂] [V₂] (100 tokens) │
│ Sequence 3: [K₃] [V₃] (100 tokens) │
└─────────────────────────────────────┘
Memory = B × seq_len × layers × dim × 2 (K+V)
```

### Limitations

1. **Wasted compute**: All sequences process until longest one finishes
2. **Memory inefficiency**: Pad short sequences to longest
3. **Head-of-line blocking**: One slow request delays entire batch
4. **Fixed batch size**: Can't add new requests mid-generation

## Advanced: Batch Decoding with llama.cpp

### Using llama_batch

```cpp
// Modern llama.cpp batching API
struct BatchInferenceEngine {
    llama_model* model;
    llama_context* ctx;
    llama_batch batch;

    std::unordered_map<int, RequestState> active_requests;

    void add_request(int req_id, const std::vector<llama_token>& prompt) {
        RequestState state;
        state.tokens = prompt;
        state.n_past = 0;
        state.completed = false;

        active_requests[req_id] = state;

        // Add prompt tokens to batch
        for (size_t i = 0; i < prompt.size(); i++) {
            llama_batch_add(
                batch,
                prompt[i],
                state.n_past + i,
                {req_id},
                i == prompt.size() - 1  // logits only for last token
            );
        }
    }

    void generate_step() {
        // Clear batch
        llama_batch_clear(batch);

        // Add next token from each active sequence
        for (auto& [req_id, state] : active_requests) {
            if (state.completed) continue;

            // Add last generated token
            llama_batch_add(
                batch,
                state.tokens.back(),
                state.n_past,
                {req_id},
                true  // need logits
            );

            state.n_past++;
        }

        // Decode batch
        if (batch.n_tokens > 0) {
            llama_decode(ctx, batch);
        }

        // Sample from each sequence
        for (auto& [req_id, state] : active_requests) {
            if (state.completed) continue;

            llama_token token = llama_sample_token(ctx, req_id);

            if (token == EOS_TOKEN || state.tokens.size() >= state.max_len) {
                state.completed = true;
            } else {
                state.tokens.push_back(token);
            }
        }

        // Remove completed requests
        std::erase_if(active_requests, [](const auto& item) {
            return item.second.completed;
        });
    }
};
```

### Optimizing Batch Size

```python
def find_optimal_batch_size(model, hardware):
    """
    Binary search for maximum batch size that fits in memory
    """
    low, high = 1, 128
    optimal = 1

    while low <= high:
        mid = (low + high) // 2

        try:
            # Test batch size
            success = test_batch_size(model, mid, hardware)

            if success:
                optimal = mid
                low = mid + 1
            else:
                high = mid - 1
        except OutOfMemoryError:
            high = mid - 1

    return optimal

def test_batch_size(model, batch_size, hardware):
    """
    Test if batch_size fits in available memory
    """
    # Calculate memory requirements
    kv_cache_memory = (
        batch_size *
        model.n_ctx *
        model.n_layers *
        model.n_embd *
        2 *  # K + V
        2    # fp16
    )

    activation_memory = (
        batch_size *
        model.n_ctx *
        model.n_embd *
        4    # fp32 activations
    )

    total_memory = kv_cache_memory + activation_memory

    return total_memory < hardware.available_memory * 0.9  # 90% safety margin
```

## Handling Variable-Length Sequences

### The Padding Problem

```
Batch of 4 requests:
Req 1: 100 tokens ████████████████████
Req 2:  50 tokens ██████████
Req 3:  75 tokens ███████████████
Req 4: 120 tokens ████████████████████████

With padding to max_len=120:
Req 1: ████████████████████____
Req 2: ██████████______________  (70 tokens wasted!)
Req 3: ███████████████_________  (45 tokens wasted!)
Req 4: ████████████████████████

Total waste: 115 / 345 = 33% compute wasted
```

### Solution 1: Bucketing

Group requests by similar lengths:

```python
def bucket_requests(requests, num_buckets=4):
    """
    Cluster requests into length buckets to minimize padding
    """
    # Sort by length
    sorted_requests = sorted(requests, key=lambda r: len(r.prompt))

    # Create buckets
    buckets = [[] for _ in range(num_buckets)]
    bucket_size = len(requests) // num_buckets

    for i, req in enumerate(sorted_requests):
        bucket_idx = min(i // bucket_size, num_buckets - 1)
        buckets[bucket_idx].append(req)

    # Process each bucket separately
    results = []
    for bucket in buckets:
        if bucket:
            batch_results = static_batch_inference(bucket, model)
            results.extend(batch_results)

    return results
```

### Solution 2: Attention Masking

Use attention masks to avoid computing over padding:

```python
def create_attention_mask(batch_lengths, max_len):
    """
    Create mask: 1 for valid tokens, 0 for padding
    """
    batch_size = len(batch_lengths)
    mask = torch.zeros(batch_size, max_len)

    for i, length in enumerate(batch_lengths):
        mask[i, :length] = 1

    return mask

# In forward pass:
attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
```

### Solution 3: FlashAttention with Variable Lengths

FlashAttention supports ragged sequences natively:

```cpp
// FlashAttention with sequence lengths
flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q,  // Cumulative sequence lengths
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal=true
);
```

## Performance Optimization

### GPU Utilization Analysis

```python
import torch
from torch.profiler import profile, ProfilerActivity

def profile_batch_inference(model, batch_sizes):
    """
    Profile GPU utilization for different batch sizes
    """
    results = {}

    for batch_size in batch_sizes:
        batch = create_dummy_batch(batch_size)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True
        ) as prof:
            model.generate_batch(batch)

        # Analyze results
        gpu_time = sum(
            evt.cuda_time_total
            for evt in prof.key_averages()
            if evt.device_type == torch.device('cuda')
        )

        total_time = prof.profiler.total_average().self_cuda_time_total

        results[batch_size] = {
            'gpu_time_ms': gpu_time / 1000,
            'utilization': gpu_time / total_time,
            'throughput': batch_size / (gpu_time / 1e6)
        }

    return results
```

### Benchmark Results

Typical GPU utilization vs batch size (LLaMA-13B on A100):

| Batch Size | GPU Util | Throughput | Latency | Efficiency |
|-----------|----------|------------|---------|------------|
| 1 | 12% | 28 tok/s | 35ms | 1.0x |
| 4 | 38% | 95 tok/s | 42ms | 3.4x |
| 8 | 62% | 168 tok/s | 48ms | 6.0x |
| 16 | 81% | 285 tok/s | 56ms | 10.2x |
| 32 | 92% | 410 tok/s | 78ms | 14.6x |
| 64 | 95% | 485 tok/s | 132ms | 17.3x |

**Sweet spot**: Batch size 16-32 for most workloads.

### Memory vs Compute Trade-offs

```python
def analyze_tradeoffs(model_size, batch_size, seq_len):
    """
    Calculate memory and compute requirements
    """
    # Memory (GB)
    model_memory = model_size * 2  # fp16
    kv_cache = batch_size * seq_len * model.n_layers * model.n_embd * 2 * 2 / 1e9
    activations = batch_size * seq_len * model.n_embd * 4 / 1e9

    total_memory = model_memory + kv_cache + activations

    # Compute (TFLOPs)
    attention_flops = 4 * batch_size * seq_len**2 * model.n_embd * model.n_layers
    ffn_flops = 8 * batch_size * seq_len * model.n_embd**2 * model.n_layers

    total_flops = (attention_flops + ffn_flops) / 1e12

    return {
        'memory_gb': total_memory,
        'compute_tflops': total_flops,
        'memory_bound': total_memory > hardware.memory_bandwidth,
        'compute_bound': total_flops > hardware.compute_capacity
    }
```

## Production Patterns

### Request Queue Management

```python
class BatchedInferenceQueue:
    def __init__(self, model, batch_size=8, max_wait_ms=50):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []
        self.lock = threading.Lock()

    async def add_request(self, request):
        """
        Add request to queue and wait for result
        """
        future = asyncio.Future()

        with self.lock:
            self.queue.append((request, future))

            # Trigger batch if full
            if len(self.queue) >= self.batch_size:
                self._process_batch()

        # Also trigger batch after timeout
        await asyncio.wait_for(future, timeout=self.max_wait_ms / 1000)
        return future.result()

    def _process_batch(self):
        """
        Process accumulated requests as a batch
        """
        with self.lock:
            if not self.queue:
                return

            # Extract requests
            batch_requests = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]

        # Process batch
        requests, futures = zip(*batch_requests)
        results = self.model.generate_batch(requests)

        # Return results
        for future, result in zip(futures, results):
            future.set_result(result)

    async def _timeout_processor(self):
        """
        Background task to process partial batches
        """
        while True:
            await asyncio.sleep(self.max_wait_ms / 1000)

            with self.lock:
                if self.queue:
                    self._process_batch()
```

### Load Balancing

```python
class MultiGPUBatchScheduler:
    def __init__(self, gpus):
        self.gpu_queues = {
            gpu_id: BatchedInferenceQueue(model, gpu_id)
            for gpu_id, model in enumerate(gpus)
        }

    async def route_request(self, request):
        """
        Route request to least loaded GPU
        """
        # Get queue lengths
        queue_lengths = {
            gpu_id: len(queue.queue)
            for gpu_id, queue in self.gpu_queues.items()
        }

        # Choose GPU with shortest queue
        best_gpu = min(queue_lengths, key=queue_lengths.get)

        # Submit to that GPU
        return await self.gpu_queues[best_gpu].add_request(request)
```

### Monitoring

```python
class BatchMetrics:
    def __init__(self):
        self.batch_sizes = []
        self.wait_times = []
        self.processing_times = []
        self.utilizations = []

    def log_batch(self, size, wait_time, proc_time, utilization):
        self.batch_sizes.append(size)
        self.wait_times.append(wait_time)
        self.processing_times.append(proc_time)
        self.utilizations.append(utilization)

    def summary(self):
        return {
            'avg_batch_size': np.mean(self.batch_sizes),
            'p50_wait_ms': np.percentile(self.wait_times, 50),
            'p95_wait_ms': np.percentile(self.wait_times, 95),
            'avg_processing_ms': np.mean(self.processing_times),
            'avg_gpu_util': np.mean(self.utilizations),
            'throughput_tok_s': sum(self.batch_sizes) / sum(self.processing_times)
        }
```

## Hands-On Exercises

### Exercise 1: Implement Static Batching

```bash
# Run single request
time ./llama-cli -m model.gguf -p "Tell me a story" -n 100

# Run with batching (parallel)
time ./llama-parallel -m model.gguf -np 4 \
    -p "Tell me a story" \
    -p "Explain quantum physics" \
    -p "Write a poem" \
    -p "Debug this code"
```

### Exercise 2: Optimize Batch Size

```python
# See code/batch_optimizer.py
python code/batch_optimizer.py \
    --model models/llama-13b.gguf \
    --batch-sizes 1,2,4,8,16,32 \
    --requests 100 \
    --output batch_analysis.json
```

### Exercise 3: Measure Memory Usage

```python
# Track KV cache memory across batch sizes
python code/memory_profiler.py \
    --model models/llama-13b.gguf \
    --batch-size 16 \
    --seq-len 2048 \
    --plot memory_usage.png
```

## Interview Questions

1. **Explain the difference between batch size and sequence length.**
   - Batch size: Number of independent requests processed in parallel
   - Sequence length: Number of tokens in each request
   - Memory = O(batch_size × seq_len), Compute = O(batch_size × seq_len²)

2. **Why does batching improve throughput but increase latency?**
   - Throughput: More requests processed per unit time (GPU utilization)
   - Latency: Each request waits for others in batch to complete
   - Trade-off is worth it when throughput > latency in importance

3. **How would you handle a batch where one sequence is much longer?**
   - Option 1: Dynamic batching (next lesson) - remove completed sequences
   - Option 2: Bucketing - group similar-length sequences
   - Option 3: Time limit - abort long sequences
   - Option 4: Attention masking - compute only on valid tokens

4. **Calculate memory for batch=16, seq=2048, model=13B LLaMA.**
   - KV cache: 16 × 2048 × 40 layers × 5120 dim × 2 (K+V) × 2 bytes
   - KV cache ≈ 13.4 GB
   - Add model weights (13B × 2 bytes = 26 GB)
   - Total ≈ 40 GB (need A100-40GB or better)

5. **Design a batching strategy for a production chatbot.**
   - Use dynamic batching with continuous batching (next lesson)
   - Set max_batch_size based on GPU memory
   - Set max_wait_time based on latency SLA (e.g., 100ms)
   - Implement priority queues for premium users
   - Monitor and auto-scale based on queue depth

## Summary

Parallel inference through batching is essential for production LLM serving:

✅ **Static batching**: Simple, processes B requests together
✅ **Batch size optimization**: Balance memory, compute, and latency
✅ **Variable-length handling**: Use bucketing or attention masking
✅ **Production patterns**: Queue management, load balancing, monitoring

**Key insight**: Batching trades off slight latency increase for massive throughput gains (5-10x typical).

In the next lesson, we'll explore **continuous batching** - a more sophisticated approach that removes completed sequences dynamically.

---

**Next**: [03-continuous-batching.md](./03-continuous-batching.md)
