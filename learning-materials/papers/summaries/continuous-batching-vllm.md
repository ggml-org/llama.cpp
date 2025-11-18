# Continuous Batching and vLLM Architecture

**Paper**: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
**Authors**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. (UC Berkeley)
**Published**: September 2023
**Link**: https://arxiv.org/abs/2309.06180
**Module**: 5 - Advanced Inference Optimization
**Impact**: ⭐⭐⭐⭐⭐

---

## Executive Summary

vLLM introduces PagedAttention and continuous batching to dramatically improve LLM serving throughput. By treating KV-cache like OS virtual memory (paging) and dynamically adding/removing requests from batches, vLLM achieves 2-24× higher throughput than traditional serving systems.

**Key Innovations**: PagedAttention (memory management) + Continuous batching (scheduling)

---

## 1. Problem: Memory Fragmentation and Static Batching

### 1.1 KV-Cache Memory Waste

```python
# Traditional KV-cache allocation
def allocate_kv_cache_traditional(batch_size, max_seq_len, num_layers):
    """
    Allocate contiguous memory for each sequence

    Problem: Must allocate for max_seq_len upfront,
             even though actual length is unknown
    """
    kv_cache_size = batch_size * max_seq_len * num_layers * hidden_dim * 2
    kv_cache = torch.empty(kv_cache_size)  # Pre-allocate maximum

    # Actual usage: Often only 20-40% of allocated memory!
    # Waste: 60-80% of KV-cache memory

# Example: 13B model, batch=32, max_seq=2048
# Allocated: 32 × 2048 × 40 layers × 5120 × 2 bytes = 26 GB
# Actually used (avg seq=512): 6.5 GB
# Waste: 19.5 GB (75%!)
```

### 1.2 Static Batching Inefficiency

```python
# Traditional serving: Static batches
def static_batch_serve(requests, model, batch_size):
    """
    Wait for batch_size requests, process together, repeat

    Problems:
    1. Latency: Early requests wait for batch to fill
    2. Idle time: GPU idles when batch not full
    3. Different lengths: Padding to longest sequence
    """
    batches = []
    current_batch = []

    for request in requests:
        current_batch.append(request)

        if len(current_batch) == batch_size:
            # Process batch
            outputs = model.generate(current_batch)
            batches.append(outputs)
            current_batch = []

    # Last partial batch: either wait or process with padding
    # Both options are inefficient!

# Throughput limited by synchronization barriers
```

---

## 2. PagedAttention

### 2.1 Concept: Virtual Memory for KV-Cache

```
OS Virtual Memory:
- Large virtual address space
- Physical memory in pages (4KB)
- On-demand allocation
- Non-contiguous physical memory

PagedAttention:
- Large logical KV-cache
- Physical KV-cache in blocks (e.g., 16 tokens)
- On-demand allocation as tokens generated
- Non-contiguous physical blocks

Benefits:
- No waste: Allocate only what's needed
- Flexible: Grow dynamically
- Shareable: Multiple sequences can share blocks
```

### 2.2 Block-based KV-Cache

```python
BLOCK_SIZE = 16  # tokens per block

class PagedKVCache:
    def __init__(self, num_blocks, num_layers, num_heads, head_dim):
        # Physical blocks: Pre-allocated pool
        self.kv_blocks = torch.empty(
            num_blocks, 2,  # K and V
            num_layers, num_heads, BLOCK_SIZE, head_dim
        )
        self.block_table = {}  # seq_id → list of block indices
        self.free_blocks = list(range(num_blocks))

    def allocate_sequence(self, seq_id):
        """Allocate first block for new sequence"""
        block_idx = self.free_blocks.pop(0)
        self.block_table[seq_id] = [block_idx]
        return block_idx

    def append_token(self, seq_id, k, v):
        """Append K, V for new token"""
        blocks = self.block_table[seq_id]
        last_block = blocks[-1]

        # Check if current block has space
        block_offset = len(self.get_sequence_length(seq_id)) % BLOCK_SIZE

        if block_offset == 0:
            # Need new block
            new_block = self.free_blocks.pop(0)
            blocks.append(new_block)
            last_block = new_block
            block_offset = 0

        # Write K, V to block
        self.kv_blocks[last_block, 0, :, :, block_offset, :] = k  # K
        self.kv_blocks[last_block, 1, :, :, block_offset, :] = v  # V

    def get_kv(self, seq_id):
        """Retrieve all K, V for sequence"""
        blocks = self.block_table[seq_id]
        seq_len = self.get_sequence_length(seq_id)

        # Gather from non-contiguous blocks
        k_list = []
        v_list = []
        for block_idx in blocks:
            k_block = self.kv_blocks[block_idx, 0]
            v_block = self.kv_blocks[block_idx, 1]
            k_list.append(k_block)
            v_list.append(v_block)

        k = torch.cat(k_list, dim=-2)[:, :, :seq_len, :]  # Trim to actual length
        v = torch.cat(v_list, dim=-2)[:, :, :seq_len, :]
        return k, v
```

### 2.3 Memory Savings

```python
# Traditional: Allocate max_seq_len upfront
traditional_memory = batch_size * max_seq_len * model_params

# PagedAttention: Allocate only used tokens
paged_memory = batch_size * actual_avg_len * model_params

# Savings
memory_saved = traditional_memory - paged_memory

# Example: max_seq=2048, avg_actual=512
# Savings: 75% of KV-cache memory!
# Enables 4× larger batch size → 4× higher throughput
```

---

## 3. Continuous Batching

### 3.1 Dynamic Request Scheduling

```python
class ContinuousBatchScheduler:
    def __init__(self, model, max_batch_size):
        self.model = model
        self.max_batch_size = max_batch_size
        self.active_sequences = []  # Currently generating
        self.waiting_queue = deque()

    def add_request(self, request):
        """Add new request (can happen anytime)"""
        self.waiting_queue.append(request)

    def schedule_iteration(self):
        """Schedule next iteration (dynamic batching)"""

        # Remove completed sequences
        self.active_sequences = [
            seq for seq in self.active_sequences
            if not seq.is_finished()
        ]

        # Add new sequences from queue if space available
        while (len(self.active_sequences) < self.max_batch_size and
               self.waiting_queue):
            new_seq = self.waiting_queue.popleft()
            self.active_sequences.append(new_seq)

        # Generate next token for ALL active sequences
        if self.active_sequences:
            self.model.generate_next_token(self.active_sequences)

        return len(self.active_sequences)

# Key: Batch composition changes every iteration!
# Finished sequences leave, new sequences join
# GPU always at max utilization
```

### 3.2 Benefits

```
Traditional Static Batching:
┌────────┬────────┬────────┬────────┐
│ Wait   │Process │ Wait   │Process │
│Batch 1 │Batch 1 │Batch 2 │Batch 2 │
└────────┴────────┴────────┴────────┘
   idle      GPU      idle     GPU

Continuous Batching:
┌────────────────────────────────────┐
│ Process Request 1, 2, 3, ...       │
│ (requests join/leave dynamically)  │
└────────────────────────────────────┘
        GPU always busy

Throughput improvement: 2-5× typical
```

---

## 4. Attention Kernel with PagedAttention

```python
def paged_attention_kernel(q, block_table, kv_cache, block_size=16):
    """
    Attention with non-contiguous KV-cache

    Args:
        q: Query [batch, num_heads, 1, head_dim] (current token)
        block_table: [batch, max_num_blocks] (logical → physical mapping)
        kv_cache: [num_blocks, 2, num_layers, num_heads, block_size, head_dim]
    """
    batch_size, num_heads, _, head_dim = q.shape

    outputs = []
    for i in range(batch_size):
        # Get blocks for this sequence
        blocks_i = block_table[i]  # List of physical block indices

        # Gather K, V from blocks
        k_blocks = []
        v_blocks = []
        for block_idx in blocks_i:
            if block_idx >= 0:  # -1 = padding
                k_blocks.append(kv_cache[block_idx, 0])  # K
                v_blocks.append(kv_cache[block_idx, 1])  # V

        k = torch.cat(k_blocks, dim=-2)  # [num_heads, seq_len, head_dim]
        v = torch.cat(v_blocks, dim=-2)

        # Standard attention
        scores = (q[i] @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v

        outputs.append(out)

    return torch.stack(outputs, dim=0)

# Optimized CUDA kernel fuses block gathering + attention
# Performance: Comparable to contiguous KV-cache
```

---

## 5. vLLM System Architecture

```
┌─────────────────────────────────────────────────────┐
│                 vLLM Server                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐      ┌──────────────┐            │
│  │  API Server  │      │  Scheduler   │            │
│  │ (FastAPI)    │─────▶│ (Continuous  │            │
│  │              │      │  Batching)   │            │
│  └──────────────┘      └──────┬───────┘            │
│                               │                     │
│                               ▼                     │
│                    ┌──────────────────┐             │
│                    │  Block Manager   │             │
│                    │ (PagedAttention) │             │
│                    └─────────┬────────┘             │
│                              │                      │
│                              ▼                      │
│                    ┌──────────────────┐             │
│                    │  Model Executor  │             │
│                    │  (GPU Workers)   │             │
│                    └──────────────────┘             │
│                                                     │
└─────────────────────────────────────────────────────┘

Workflow:
1. Requests arrive at API server
2. Scheduler adds to continuous batch
3. Block Manager allocates KV-cache blocks
4. Model Executor generates tokens
5. Scheduler removes finished requests, adds new ones
```

---

## 6. Performance Results

### 6.1 Throughput Comparison

**LLaMA 13B on A100 (input=512, output=128 tokens)**

| System | Throughput (req/s) | Speedup vs HF |
|--------|-------------------|---------------|
| HuggingFace (static) | 0.8 | 1.0× |
| TGI (basic batching) | 2.1 | 2.6× |
| vLLM (paged + continuous) | **9.7** | **12.1×** |

**LLaMA 70B on 4× A100**

| System | Throughput (req/s) | Speedup |
|--------|-------------------|---------|
| FasterTransformer | 1.2 | 1.0× |
| TGI | 2.4 | 2.0× |
| vLLM | **8.5** | **7.1×** |

---

### 6.2 Memory Efficiency

```
Without PagedAttention:
- KV-cache waste: 60-80%
- Max batch size: 16

With PagedAttention:
- KV-cache waste: <5%
- Max batch size: 64 (4× increase)
- Throughput: 4-8× higher (larger batch + better utilization)
```

---

## 7. Integration and Usage

### 7.1 vLLM Server

```python
from vllm import LLM, SamplingParams

# Initialize vLLM (automatically uses PagedAttention + continuous batching)
llm = LLM(
    model="meta-llama/Llama-2-13b-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
    block_size=16,               # PagedAttention block size
    max_num_batched_tokens=2048,
    max_num_seqs=64              # Continuous batching: up to 64 concurrent requests
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=128
)

# Generate (automatically uses continuous batching)
prompts = [
    "Tell me about quantum computing",
    "Explain transformers",
    # ... can have many concurrent requests
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### 7.2 OpenAI-Compatible API

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-13b-hf \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 4096 \
  --block-size 16

# Use with OpenAI client
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

response = client.completions.create(
    model="meta-llama/Llama-2-13b-hf",
    prompt="Once upon a time",
    max_tokens=100
)
```

---

## 8. llama.cpp Comparison

### Similarities
- Both focus on inference optimization
- Both support dynamic batching (llama.cpp recent versions)
- Both minimize memory waste

### Differences

| Feature | vLLM | llama.cpp |
|---------|------|-----------|
| Target | GPU serving | CPU/GPU local inference |
| Batching | Continuous (advanced) | Basic dynamic batching |
| KV-Cache | PagedAttention (blocks) | Contiguous allocation |
| Multi-GPU | Yes (tensor parallelism) | Limited (layer split) |
| Use case | Production API serving | Local/embedded inference |

---

## 9. Key Takeaways

### PagedAttention
✅ Virtual memory-style KV-cache management
✅ 3-4× memory savings vs traditional allocation
✅ Enables larger batch sizes

### Continuous Batching
✅ Dynamic request scheduling (join/leave anytime)
✅ GPU always at max utilization
✅ 2-5× throughput improvement

### For llama.cpp Users
- vLLM is for high-throughput GPU serving
- llama.cpp is for local/CPU inference
- Both valuable in different contexts
- Consider vLLM for production API workloads

---

## Further Reading

- **Paper**: https://arxiv.org/abs/2309.06180
- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **Blog**: https://blog.vllm.ai/

---

**Status**: Complete | Module 5 (2/3) papers
