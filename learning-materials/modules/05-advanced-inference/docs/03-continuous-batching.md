# Continuous Batching: Dynamic Request Scheduling

**Module 5, Lesson 3**
**Estimated Time**: 4-5 hours
**Difficulty**: Advanced

## Overview

Continuous batching (also called iteration-level scheduling or dynamic batching) is the state-of-the-art technique for LLM serving, pioneered by systems like vLLM and Orca. It eliminates the main limitation of static batching by allowing requests to join and leave the batch at any time.

## Learning Objectives

By the end of this lesson, you will:
- ✅ Understand continuous batching vs static batching
- ✅ Implement dynamic batch scheduling
- ✅ Master PagedAttention for memory efficiency
- ✅ Build a continuous batching scheduler
- ✅ Optimize for production workloads

## The Problem with Static Batching

### Head-of-Line Blocking

```
Static Batch (B=4):
┌─────────────────────────────────────┐
│ Req 1: ████████████ (50 tokens)    │
│ Req 2: ██████████████████ (80 tok) │
│ Req 3: ████████ (30 tokens)        │
│ Req 4: ████████████████████████    │ ← Longest (100 tokens)
└─────────────────────────────────────┘

All requests wait until Req 4 completes!
Waste: (100-50) + (100-80) + (100-30) = 90 token-steps
```

### Batch Utilization Over Time

```
Static Batching:
Time: 0    10   20   30   40   50   60   70   80   90   100
GPU:  ████ ████ ████ ███  ██   ██   ██   █    █    █    █
Util: 100% 100% 100% 75%  50%  50%  50%  25%  25%  25%  25%

Continuous Batching:
Time: 0    10   20   30   40   50   60   70   80   90   100
GPU:  ████ ████ ████ ████ ████ ████ ████ ████ ████ ████ ████
Util: 100% 100% 100% 100% 100% 100% 100% 100% 100% 100% 100%
      ↑              ↑         ↑              ↑
     New requests added when others complete
```

## Continuous Batching: The Solution

### Core Concept

Instead of waiting for all requests in a batch to complete:
1. **Remove** completed requests after each iteration
2. **Add** new requests from the queue
3. **Maintain** full batch size whenever possible

```python
def continuous_batching(request_queue, model, max_batch_size=32):
    """
    Continuous batching scheduler
    """
    active_requests = {}  # seq_id -> RequestState
    next_seq_id = 0

    while True:
        # Step 1: Remove completed requests
        active_requests = {
            seq_id: req
            for seq_id, req in active_requests.items()
            if not req.completed
        }

        # Step 2: Add new requests from queue
        while len(active_requests) < max_batch_size and not request_queue.empty():
            new_request = request_queue.get()
            active_requests[next_seq_id] = new_request
            next_seq_id += 1

        if not active_requests:
            time.sleep(0.001)  # Wait for new requests
            continue

        # Step 3: Generate one token for all active requests
        batch = create_batch(active_requests)
        model.decode(batch)

        # Step 4: Sample and update each request
        for seq_id, request in active_requests.items():
            token = model.sample(seq_id)
            request.tokens.append(token)

            if token == EOS or len(request.tokens) >= request.max_len:
                request.completed = True
                request.future.set_result(request.tokens)
```

### Performance Comparison

| Metric | Static Batching | Continuous Batching | Improvement |
|--------|----------------|---------------------|-------------|
| Throughput | 250 tok/s | 580 tok/s | 2.3x |
| Avg Latency | 3.2s | 1.8s | 1.8x faster |
| p99 Latency | 8.5s | 3.1s | 2.7x faster |
| GPU Util | 62% | 94% | 1.5x |
| Requests/sec | 15 | 42 | 2.8x |

## Implementation in llama.cpp

### Basic Continuous Batching

```cpp
#include "llama.h"
#include <queue>

struct SequenceState {
    int seq_id;
    std::vector<llama_token> tokens;
    int n_past;
    int max_tokens;
    bool completed;
    std::promise<std::vector<llama_token>> result;
};

class ContinuousBatchScheduler {
private:
    llama_model* model;
    llama_context* ctx;
    llama_batch batch;
    int max_batch_size;

    std::unordered_map<int, SequenceState> active_seqs;
    std::queue<SequenceState> pending_queue;
    int next_seq_id = 0;

public:
    ContinuousBatchScheduler(
        llama_model* model,
        llama_context* ctx,
        int max_batch_size = 32
    ) : model(model), ctx(ctx), max_batch_size(max_batch_size) {
        batch = llama_batch_init(max_batch_size * 2048, 0, max_batch_size);
    }

    std::future<std::vector<llama_token>> submit(
        const std::vector<llama_token>& prompt,
        int max_tokens = 512
    ) {
        SequenceState state;
        state.seq_id = next_seq_id++;
        state.tokens = prompt;
        state.n_past = 0;
        state.max_tokens = max_tokens;
        state.completed = false;

        auto future = state.result.get_future();
        pending_queue.push(std::move(state));

        return future;
    }

    void run_iteration() {
        // Remove completed sequences
        std::erase_if(active_seqs, [](const auto& item) {
            return item.second.completed;
        });

        // Add new sequences from queue
        while (active_seqs.size() < max_batch_size && !pending_queue.empty()) {
            auto state = std::move(pending_queue.front());
            pending_queue.pop();

            int seq_id = state.seq_id;
            active_seqs[seq_id] = std::move(state);

            // Add prompt to batch (all tokens at once for new sequence)
            auto& seq = active_seqs[seq_id];
            for (size_t i = 0; i < seq.tokens.size(); i++) {
                llama_batch_add(
                    batch,
                    seq.tokens[i],
                    seq.n_past + i,
                    {seq_id},
                    i == seq.tokens.size() - 1  // logits for last token
                );
            }
            seq.n_past += seq.tokens.size();
        }

        if (active_seqs.empty()) {
            return;
        }

        // Prepare batch for generation step
        llama_batch_clear(batch);

        for (auto& [seq_id, seq] : active_seqs) {
            if (seq.n_past == 0) continue;  // Just added, already in batch

            // Add last token for next generation
            llama_batch_add(
                batch,
                seq.tokens.back(),
                seq.n_past,
                {seq_id},
                true  // need logits
            );
            seq.n_past++;
        }

        // Decode batch
        if (batch.n_tokens > 0) {
            llama_decode(ctx, batch);
        }

        // Sample from each active sequence
        for (auto& [seq_id, seq] : active_seqs) {
            llama_token token = llama_sample_token(ctx, seq_id);
            seq.tokens.push_back(token);

            // Check completion
            if (token == llama_token_eos(model) ||
                seq.tokens.size() - seq.n_past >= seq.max_tokens) {

                seq.completed = true;
                seq.result.set_value(seq.tokens);
            }
        }
    }

    void run_loop() {
        while (true) {
            run_iteration();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
};
```

### Usage Example

```cpp
int main() {
    // Initialize model
    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_load_model_from_file("model.gguf", model_params);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 512;
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);

    // Create scheduler
    ContinuousBatchScheduler scheduler(model, ctx, 32);

    // Start scheduler in background thread
    std::thread scheduler_thread([&]() {
        scheduler.run_loop();
    });

    // Submit requests
    std::vector<std::future<std::vector<llama_token>>> futures;

    for (int i = 0; i < 100; i++) {
        auto prompt = tokenize("Tell me a story about " + std::to_string(i));
        futures.push_back(scheduler.submit(prompt, 512));

        // Submit requests over time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Wait for results
    for (auto& future : futures) {
        auto tokens = future.get();
        std::cout << detokenize(tokens) << "\n\n";
    }

    return 0;
}
```

## PagedAttention: Efficient KV Cache Management

### The KV Cache Fragmentation Problem

Traditional KV cache allocation:

```
Static Allocation (wasteful):
┌──────────────────────────────────────┐
│ Seq 0: [████████████________] (60%)  │ ← 40% wasted
│ Seq 1: [██████______________] (30%)  │ ← 70% wasted
│ Seq 2: [████████████████____] (80%)  │ ← 20% wasted
│ Seq 3: [██████████__________] (50%)  │ ← 50% wasted
└──────────────────────────────────────┘
Total waste: ~45% of allocated memory!
```

### PagedAttention Solution

Inspired by virtual memory paging:
- Divide KV cache into **fixed-size blocks** (e.g., 16 tokens)
- Allocate blocks **on-demand** as sequences grow
- **Share blocks** between sequences (for prefix caching)

```
Paged Allocation:
┌──────────────────────────────────────┐
│ Page Pool: [P0][P1][P2][P3][P4]...  │
└──────────────────────────────────────┘

Seq 0 (60 tokens):  [P0→P1→P2→P3]
Seq 1 (30 tokens):  [P4→P5]
Seq 2 (80 tokens):  [P0→P1→P6→P7→P8]  ← Shares P0,P1 with Seq 0!
Seq 3 (50 tokens):  [P9→P10→P11]

Memory usage: 100% efficient (no waste)
```

### Implementation Sketch

```python
class PagedKVCache:
    def __init__(self, block_size=16, num_blocks=1000):
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Physical blocks: [num_blocks, num_layers, block_size, num_kv_heads, head_dim]
        self.key_cache = torch.zeros(
            num_blocks, num_layers, block_size, num_kv_heads, head_dim
        )
        self.value_cache = torch.zeros(
            num_blocks, num_layers, block_size, num_kv_heads, head_dim
        )

        # Free block pool
        self.free_blocks = set(range(num_blocks))

        # Sequence to block mapping
        self.seq_to_blocks = {}  # seq_id -> [block_id, block_id, ...]

    def allocate_sequence(self, seq_id, initial_tokens=0):
        """Allocate blocks for a new sequence"""
        num_blocks_needed = (initial_tokens + self.block_size - 1) // self.block_size
        blocks = self._allocate_blocks(num_blocks_needed)
        self.seq_to_blocks[seq_id] = blocks
        return blocks

    def _allocate_blocks(self, num_blocks):
        """Allocate physical blocks from free pool"""
        if len(self.free_blocks) < num_blocks:
            raise OutOfMemoryError("No free blocks available")

        blocks = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop()
            blocks.append(block_id)

        return blocks

    def append_token(self, seq_id):
        """Append a new token to sequence, allocating block if needed"""
        blocks = self.seq_to_blocks[seq_id]
        seq_len = len(blocks) * self.block_size

        # Check if we need a new block
        if seq_len % self.block_size == 0:
            new_block = self._allocate_blocks(1)[0]
            blocks.append(new_block)

    def free_sequence(self, seq_id):
        """Free all blocks used by sequence"""
        blocks = self.seq_to_blocks.pop(seq_id)
        self.free_blocks.update(blocks)

    def get_kv_for_sequence(self, seq_id):
        """Get K,V tensors for a sequence (used in attention)"""
        blocks = self.seq_to_blocks[seq_id]

        # Gather K,V from physical blocks
        k_blocks = [self.key_cache[block_id] for block_id in blocks]
        v_blocks = [self.value_cache[block_id] for block_id in blocks]

        # Concatenate blocks
        k = torch.cat(k_blocks, dim=1)  # [layers, seq_len, heads, dim]
        v = torch.cat(v_blocks, dim=1)

        return k, v
```

### Prefix Caching with PagedAttention

Share common prefixes across sequences:

```python
class PrefixCachedPagedKVCache(PagedKVCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_to_blocks = {}  # prefix_hash -> [block_ids]
        self.block_ref_count = {}   # block_id -> reference count

    def allocate_with_prefix(self, seq_id, tokens):
        """
        Allocate sequence, reusing blocks for known prefix
        """
        # Check if prefix is cached
        prefix_hash = hash(tuple(tokens[:self.block_size]))

        if prefix_hash in self.prefix_to_blocks:
            # Reuse existing blocks for prefix
            prefix_blocks = self.prefix_to_blocks[prefix_hash]

            # Increment reference counts
            for block_id in prefix_blocks:
                self.block_ref_count[block_id] += 1

            # Allocate new blocks for remainder
            remaining_tokens = len(tokens) - len(prefix_blocks) * self.block_size
            new_blocks = self._allocate_blocks(
                (remaining_tokens + self.block_size - 1) // self.block_size
            )

            all_blocks = prefix_blocks + new_blocks
            self.seq_to_blocks[seq_id] = all_blocks

        else:
            # No prefix match, allocate normally
            blocks = self.allocate_sequence(seq_id, len(tokens))

            # Cache the prefix
            prefix_blocks = blocks[:len(tokens) // self.block_size]
            self.prefix_to_blocks[prefix_hash] = prefix_blocks

            for block_id in prefix_blocks:
                self.block_ref_count[block_id] = 1
```

Example benefit:

```
System prompt: "You are a helpful assistant." (10 tokens)

Without prefix caching:
- Request 1: Allocate 10 + 50 tokens = 60 tokens
- Request 2: Allocate 10 + 30 tokens = 40 tokens
- Request 3: Allocate 10 + 70 tokens = 80 tokens
Total: 180 tokens

With prefix caching:
- Request 1: Allocate 10 + 50 = 60 tokens
- Request 2: Share 10, allocate 30 = 40 tokens (save 10)
- Request 3: Share 10, allocate 70 = 80 tokens (save 10)
Total: 150 tokens (16% memory saved)
```

## Advanced Scheduling Strategies

### Priority-Based Scheduling

```python
class PriorityScheduler(ContinuousBatchScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.priority_queues = {
            'high': queue.PriorityQueue(),
            'medium': queue.PriorityQueue(),
            'low': queue.PriorityQueue()
        }

    def submit(self, request, priority='medium'):
        """Submit request with priority"""
        self.priority_queues[priority].put(request)

    def get_next_requests(self, max_count):
        """Get next requests, respecting priority"""
        requests = []

        # Try high priority first
        for priority in ['high', 'medium', 'low']:
            while len(requests) < max_count:
                try:
                    req = self.priority_queues[priority].get_nowait()
                    requests.append(req)
                except queue.Empty:
                    break

        return requests
```

### Preemption for Urgent Requests

```python
class PreemptiveScheduler(ContinuousBatchScheduler):
    def handle_urgent_request(self, urgent_request):
        """
        Preempt lowest priority request to make room
        """
        if len(self.active_seqs) >= self.max_batch_size:
            # Find lowest priority sequence
            lowest_priority_seq = min(
                self.active_seqs.items(),
                key=lambda x: x[1].priority
            )

            # Evict it (save state for later resumption)
            seq_id, seq_state = lowest_priority_seq
            self.evicted_seqs[seq_id] = seq_state
            del self.active_seqs[seq_id]

        # Add urgent request
        self.active_seqs[urgent_request.seq_id] = urgent_request
```

### Fair Sharing

```python
class FairShareScheduler(ContinuousBatchScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_quotas = {}  # user_id -> max_concurrent_requests

    def enforce_fairness(self):
        """
        Ensure no user exceeds their quota
        """
        user_counts = {}

        for seq_id, seq in self.active_seqs.items():
            user_id = seq.user_id
            user_counts[user_id] = user_counts.get(user_id, 0) + 1

        # Preempt if user exceeds quota
        for user_id, count in user_counts.items():
            quota = self.user_quotas.get(user_id, float('inf'))

            if count > quota:
                # Evict excess requests from this user
                excess = count - quota
                user_seqs = [
                    (seq_id, seq)
                    for seq_id, seq in self.active_seqs.items()
                    if seq.user_id == user_id
                ]

                # Evict oldest requests
                for seq_id, seq in sorted(user_seqs, key=lambda x: x[1].start_time)[:excess]:
                    self.evict_sequence(seq_id)
```

## Performance Optimization

### Batch Size Tuning

```python
class AdaptiveBatchSizeScheduler(ContinuousBatchScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_latency_ms = 100
        self.batch_size_range = (1, 64)

    def adapt_batch_size(self):
        """
        Dynamically adjust batch size based on observed latency
        """
        if len(self.latency_history) < 10:
            return

        avg_latency = np.mean(self.latency_history[-10:])

        if avg_latency > self.target_latency_ms * 1.1:
            # Latency too high, reduce batch size
            self.max_batch_size = max(
                self.batch_size_range[0],
                int(self.max_batch_size * 0.9)
            )
        elif avg_latency < self.target_latency_ms * 0.9:
            # Latency good, increase batch size
            self.max_batch_size = min(
                self.batch_size_range[1],
                int(self.max_batch_size * 1.1)
            )
```

### Chunked Prefill

For long prompts, split prefill into chunks:

```python
def chunked_prefill(prompt_tokens, model, chunk_size=512):
    """
    Process long prompts in chunks to avoid blocking other requests
    """
    kv_cache = []

    for i in range(0, len(prompt_tokens), chunk_size):
        chunk = prompt_tokens[i:i+chunk_size]

        # Process chunk
        kv_chunk = model.prefill(chunk, kv_cache)
        kv_cache.extend(kv_chunk)

        # Yield control to scheduler after each chunk
        yield

    return kv_cache
```

## Production Metrics

```python
class ContinuousBatchingMetrics:
    def __init__(self):
        self.iterations = 0
        self.tokens_generated = 0
        self.requests_completed = 0
        self.batch_sizes = []
        self.iteration_times = []
        self.queue_depths = []

    def summary(self):
        return {
            'throughput_tok_s': self.tokens_generated / sum(self.iteration_times),
            'avg_batch_size': np.mean(self.batch_sizes),
            'batch_efficiency': np.mean(self.batch_sizes) / max(self.batch_sizes),
            'avg_iteration_ms': np.mean(self.iteration_times) * 1000,
            'requests_per_sec': self.requests_completed / sum(self.iteration_times),
            'avg_queue_depth': np.mean(self.queue_depths),
            'p95_queue_depth': np.percentile(self.queue_depths, 95)
        }
```

## Interview Questions

1. **Explain continuous batching in 1 minute.**
   - Dynamic scheduling: requests join/leave batch each iteration
   - Eliminates head-of-line blocking from static batching
   - Maintains high GPU utilization by filling slots immediately
   - 2-3x throughput improvement over static batching

2. **What is PagedAttention and why is it important?**
   - Manages KV cache in fixed-size blocks (like virtual memory)
   - Eliminates fragmentation from variable-length sequences
   - Enables prefix sharing across requests
   - Achieves near 100% memory utilization

3. **Design a continuous batching scheduler for a production API.**
   - Priority queues for different SLA tiers
   - PagedAttention for memory efficiency
   - Preemption for urgent requests
   - Fair sharing across users
   - Adaptive batch sizing based on latency
   - Chunked prefill for long prompts

4. **How would you handle a sudden traffic spike?**
   - Increase max_batch_size up to memory limit
   - Reduce max_tokens per request
   - Enable request queueing with timeouts
   - Auto-scale GPUs if possible
   - Reject lowest-priority requests if needed

5. **Calculate memory savings from prefix caching.**
   - Shared prefix: P tokens
   - Unique suffixes: S₁, S₂, ..., Sₙ tokens
   - Without caching: N × P + ΣSᵢ
   - With caching: P + ΣSᵢ
   - Savings: (N-1) × P / (N × P + ΣSᵢ)

## Summary

Continuous batching is the state-of-the-art for LLM serving:

✅ **Dynamic scheduling**: 2-3x better than static batching
✅ **PagedAttention**: Near-perfect memory utilization
✅ **Prefix caching**: Share common prompts across requests
✅ **Advanced features**: Priority, preemption, fair sharing

**Key insight**: Continuously manage the batch to maximize GPU utilization and minimize latency.

In the next lesson, we'll explore **grammar-guided generation** for structured outputs.

---

**Next**: [04-grammar-guided-generation.md](./04-grammar-guided-generation.md)
