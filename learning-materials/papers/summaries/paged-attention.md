# PagedAttention: Detailed Technical Analysis

**Paper**: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
**Module**: 5 - Advanced Inference Optimization
**Impact**: ⭐⭐⭐⭐⭐

---

## Summary

PagedAttention is the core memory management innovation in vLLM. This document provides deeper technical details beyond the vLLM architecture overview.

**Note**: See continuous-batching-vllm.md for full context. This document focuses specifically on PagedAttention implementation details.

---

## 1. Block Table Management

```python
class BlockAllocator:
    """
    Manages physical KV-cache blocks (similar to OS page allocator)
    """
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = set(range(num_blocks))
        self.reference_counts = {}

    def allocate(self):
        """Allocate a free block"""
        if not self.free_blocks:
            raise OutOfMemoryError("No free KV-cache blocks")
        block_id = self.free_blocks.pop()
        self.reference_counts[block_id] = 1
        return block_id

    def free(self, block_id):
        """Free a block (dec ref count)"""
        self.reference_counts[block_id] -= 1
        if self.reference_counts[block_id] == 0:
            self.free_blocks.add(block_id)
            del self.reference_counts[block_id]

    def share(self, block_id):
        """Share block (inc ref count)"""
        self.reference_counts[block_id] += 1
```

---

## 2. Copy-on-Write for Beam Search

```python
# Beam search often requires copying KV-cache
# PagedAttention uses COW (copy-on-write) to avoid duplication

class COWBlockTable:
    def fork_sequence(self, parent_seq_id, child_seq_id):
        """
        Fork sequence for beam search candidate

        Initially shares blocks with parent (zero-copy)
        Copy block only when child modifies it
        """
        parent_blocks = self.block_tables[parent_seq_id]
        child_blocks = parent_blocks.copy()  # Copy table, not blocks!

        # Increment reference counts
        for block_id in child_blocks:
            self.allocator.share(block_id)

        self.block_tables[child_seq_id] = child_blocks

    def append_token_cow(self, seq_id, k, v):
        """Append token with copy-on-write"""
        blocks = self.block_tables[seq_id]
        last_block = blocks[-1]

        # Check if shared
        if self.allocator.reference_counts[last_block] > 1:
            # COW: Allocate new block, copy data
            new_block = self.allocator.allocate()
            self.copy_block(last_block, new_block)
            self.allocator.free(last_block)  # Dec old block ref
            blocks[-1] = new_block
            last_block = new_block

        # Now safe to modify
        self.write_to_block(last_block, k, v)

# Memory savings for beam search:
# Without COW: beam_width × KV-cache (huge!)
# With COW: Only modified blocks duplicated (~10-20% typically)
```

---

## 3. Kernel Implementation

### 3.1 Gather-Attention CUDA Kernel

```cuda
// Optimized PagedAttention kernel
__global__ void paged_attention_v2_kernel(
    const scalar_t* __restrict__ q,             // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,       // [num_blocks, num_heads, block_size, head_size]
    const scalar_t* __restrict__ v_cache,
    scalar_t* __restrict__ out,                 // [num_seqs, num_heads, head_size]
    const int* __restrict__ block_tables,       // [num_seqs, max_num_blocks]
    const int* __restrict__ context_lens,
    const int max_num_blocks,
    const float scale
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int num_heads = gridDim.y;
    const int head_size = blockDim.x;
    const int tid = threadIdx.x;

    // Load query into shared memory
    __shared__ scalar_t q_smem[HEAD_SIZE];
    if (tid < head_size) {
        q_smem[tid] = q[seq_idx * num_heads * head_size + head_idx * head_size + tid];
    }
    __syncthreads();

    // Iterate over blocks for this sequence
    const int* block_table = block_tables + seq_idx * max_num_blocks;
    const int context_len = context_lens[seq_idx];
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float max_logit = -FLT_MAX;
    float sum_exp = 0.0f;

    // First pass: Compute max and sum for softmax
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block = block_table[block_idx];
        const scalar_t* k_block = k_cache + physical_block * num_heads * BLOCK_SIZE * head_size;

        // Compute attention scores for this block
        for (int i = 0; i < BLOCK_SIZE; i++) {
            int token_idx = block_idx * BLOCK_SIZE + i;
            if (token_idx >= context_len) break;

            float score = 0.0f;
            for (int j = tid; j < head_size; j += blockDim.x) {
                score += q_smem[j] * k_block[head_idx * BLOCK_SIZE * head_size + i * head_size + j];
            }
            score = blockReduceSum(score);  // Warp reduction
            score *= scale;

            max_logit = max(max_logit, score);
        }
    }
    max_logit = blockReduceMax(max_logit);

    // Second pass: Compute exp and sum
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // ... (similar to first pass, compute exp(score - max_logit))
        sum_exp += block_sum;
    }

    // Third pass: Compute weighted sum of values
    __shared__ float output_smem[HEAD_SIZE];
    if (tid < head_size) {
        output_smem[tid] = 0.0f;
    }
    __syncthreads();

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block = block_table[block_idx];
        const scalar_t* v_block = v_cache + physical_block * num_heads * BLOCK_SIZE * head_size;

        // Accumulate attention-weighted values
        // ... (omitted for brevity)
    }

    // Write output
    if (tid < head_size) {
        out[seq_idx * num_heads * head_size + head_idx * head_size + tid] =
            output_smem[tid] / sum_exp;
    }
}
```

### Performance
- Comparable to FlashAttention for contiguous KV-cache
- ~10-15% overhead for block gathering
- Vastly outweighed by memory savings (4× batch size)

---

## 4. Prefix Sharing

```python
# Multiple requests with same prefix (e.g., system prompt)
# PagedAttention can share prefix blocks!

class PrefixCache:
    def __init__(self):
        self.prefix_to_blocks = {}  # prefix_hash → block_list

    def get_or_compute_prefix(self, prefix_tokens, model):
        """
        Compute KV-cache for prefix, or reuse if cached
        """
        prefix_hash = hash(tuple(prefix_tokens))

        if prefix_hash in self.prefix_to_blocks:
            # Reuse cached prefix blocks
            blocks = self.prefix_to_blocks[prefix_hash]
            for block in blocks:
                self.allocator.share(block)  # Increment ref count
            return blocks
        else:
            # Compute prefix KV-cache
            blocks = model.compute_prefix_kv(prefix_tokens)
            self.prefix_to_blocks[prefix_hash] = blocks
            return blocks

# Example: ChatGPT system prompt
# - Shared across ALL conversations
# - Computed once, reused thousands of times
# - Massive memory savings
```

---

## 5. Key Takeaways

**Memory Management**:
- Block-based allocation (16 tokens/block typical)
- Copy-on-write for beam search
- Reference counting for sharing
- Prefix caching for common prompts

**Performance**:
- ~10% overhead vs contiguous attention
- 4× memory efficiency → 4× batch size → 3-4× throughput
- Kernel fusion with attention computation

---

**Status**: Complete | Module 5 Complete (3/3) papers
