# Multi-GPU Inference Strategies

**Module 4, Lesson 4** | **Duration: 4 hours** | **Level: Advanced**

## Table of Contents
1. [Multi-GPU Fundamentals](#multi-gpu-fundamentals)
2. [Tensor Parallelism](#tensor-parallelism)
3. [Pipeline Parallelism](#pipeline-parallelism)
4. [NVLink and GPU Interconnects](#nvlink-and-gpu-interconnects)
5. [llama.cpp Multi-GPU Implementation](#llamacpp-multi-gpu-implementation)
6. [Scaling Performance Analysis](#scaling-performance-analysis)

---

## Learning Objectives

By the end of this lesson, you will:
- ✅ Understand multi-GPU parallelism strategies
- ✅ Implement tensor parallelism for LLMs
- ✅ Design pipeline parallel inference systems
- ✅ Optimize inter-GPU communication
- ✅ Configure llama.cpp for multi-GPU
- ✅ Analyze scaling efficiency

---

## Multi-GPU Fundamentals

### Why Multi-GPU?

**Problem Sizes:**
```
Model         FP16 Size   Single GPU (A100 80GB)
────────────────────────────────────────────────
LLaMA-7B      14 GB       ✓ Fits easily
LLaMA-13B     26 GB       ✓ Fits
LLaMA-30B     60 GB       ✓ Barely fits
LLaMA-65B     130 GB      ✗ DOES NOT FIT
LLaMA-70B     140 GB      ✗ DOES NOT FIT
Mixtral-8x7B  93 GB       ✗ DOES NOT FIT

Solution: Split across multiple GPUs!
```

**Benefits:**
1. **Larger models** - Run models that don't fit in single GPU memory
2. **Higher throughput** - Process multiple requests in parallel
3. **Lower latency** - Distribute computation (if communication is fast)
4. **Cost efficiency** - Multiple cheaper GPUs vs. expensive H100

### Parallelism Strategies

**1. Data Parallelism** (NOT for single inference)
```
GPU 0: Process batch[0:8]   with full model copy
GPU 1: Process batch[8:16]  with full model copy
GPU 2: Process batch[16:24] with full model copy
GPU 3: Process batch[24:32] with full model copy

Use case: Training, high-throughput serving
NOT useful for: Single-request latency
```

**2. Tensor Parallelism** (Split layers horizontally)
```
GPU 0: Process first half of each layer
GPU 1: Process second half of each layer

Use case: Large models, low latency
Downside: High communication overhead
```

**3. Pipeline Parallelism** (Split layers vertically)
```
GPU 0: Layers 0-7
GPU 1: Layers 8-15
GPU 2: Layers 16-23
GPU 3: Layers 24-31

Use case: Very deep models, moderate batches
Downside: Pipeline bubbles, load imbalance
```

---

## Tensor Parallelism

### Concept: Column Parallelism

**Single GPU:**
```
X: [batch, seq, 4096]
W: [4096, 4096]
Y = X @ W  →  [batch, seq, 4096]
```

**Two GPUs (Column Split):**
```
GPU 0:
  W_0: [4096, 2048]  (left half)
  Y_0 = X @ W_0  →  [batch, seq, 2048]

GPU 1:
  W_1: [4096, 2048]  (right half)
  Y_1 = X @ W_1  →  [batch, seq, 2048]

Concatenate: Y = [Y_0 | Y_1]  →  [batch, seq, 4096]
```

**Communication:** None until concatenation (can be fused with next op)

### Concept: Row Parallelism

**Single GPU:**
```
X: [batch, seq, 4096]
W: [4096, 11008]  (FFN up-projection)
Y = X @ W  →  [batch, seq, 11008]
```

**Two GPUs (Row Split):**
```
GPU 0:
  X_0: [batch, seq, 2048]  (broadcast needed)
  W_0: [2048, 11008]
  Y_0 = X_0 @ W_0  →  [batch, seq, 11008]

GPU 1:
  X_1: [batch, seq, 2048]
  W_1: [2048, 11008]
  Y_1 = X_1 @ W_1  →  [batch, seq, 11008]

Sum: Y = Y_0 + Y_1  →  [batch, seq, 11008]
```

**Communication:** All-reduce (sum) after computation

### Transformer Layer with Tensor Parallelism

**Attention Block:**
```python
# Column parallelism for Q, K, V projections
def attention_forward(x, W_q, W_k, W_v, W_o, gpu_id, num_gpus):
    # Split W_q, W_k, W_v columnwise across GPUs
    head_start = gpu_id * (num_heads // num_gpus)
    head_end = (gpu_id + 1) * (num_heads // num_gpus)

    # Each GPU computes subset of heads
    q_local = x @ W_q[:, head_start:head_end]
    k_local = x @ W_k[:, head_start:head_end]
    v_local = x @ W_v[:, head_start:head_end]

    # Attention (no communication needed)
    attn_local = scaled_dot_product_attention(q_local, k_local, v_local)

    # Row parallelism for output projection
    out_local = attn_local @ W_o[head_start:head_end, :]

    # All-reduce across GPUs
    out = all_reduce_sum(out_local)  # Communication!

    return out
```

**FFN Block:**
```python
def ffn_forward(x, W_up, W_down, gpu_id, num_gpus):
    # Column parallelism for up-projection
    hidden_start = gpu_id * (hidden_dim // num_gpus)
    hidden_end = (gpu_id + 1) * (hidden_dim // num_gpus)

    # Each GPU computes slice of hidden dim
    up_local = x @ W_up[:, hidden_start:hidden_end]
    up_local = gelu(up_local)

    # Row parallelism for down-projection
    out_local = up_local @ W_down[hidden_start:hidden_end, :]

    # All-reduce
    out = all_reduce_sum(out_local)  # Communication!

    return out
```

### Communication Patterns

**Required Operations:**
1. **Broadcast** - Send same data to all GPUs (e.g., input tokens)
2. **All-Reduce** - Sum results from all GPUs, broadcast back
3. **All-Gather** - Collect slices from all GPUs, concatenate
4. **Reduce-Scatter** - Sum and split result across GPUs

**NCCL (NVIDIA Collective Communications Library):**
```cpp
#include <nccl.h>

// Initialize NCCL
ncclComm_t comms[4];
ncclCommInitRank(&comms[0], 4, ncclId, 0);  // GPU 0
ncclCommInitRank(&comms[1], 4, ncclId, 1);  // GPU 1
ncclCommInitRank(&comms[2], 4, ncclId, 2);  // GPU 2
ncclCommInitRank(&comms[3], 4, ncclId, 3);  // GPU 3

// All-reduce (sum)
ncclAllReduce(
    send_buf,           // Input
    recv_buf,           // Output (all GPUs get same result)
    count,              // Number of elements
    ncclFloat,          // Data type
    ncclSum,            // Operation
    comms[gpu_id],      // Communicator
    stream              // CUDA stream
);
```

**Performance:**
```
NVLink (300 GB/s):  4 GB all-reduce in 13 ms
PCIe (32 GB/s):     4 GB all-reduce in 125 ms

NVLink is 10x faster!
```

---

## Pipeline Parallelism

### Concept: Layer-wise Distribution

**4 GPUs, 32 Layers:**
```
GPU 0: Layers  0-7   (8 layers)
GPU 1: Layers  8-15  (8 layers)
GPU 2: Layers 16-23  (8 layers)
GPU 3: Layers 24-31  (8 layers)
```

**Forward Pass (Sequential):**
```
Time →
  GPU 0: [Process token 0]────────→ Send to GPU 1
  GPU 1:                   [Process token 0]────→ Send to GPU 2
  GPU 2:                                [Process token 0]───→ Send to GPU 3
  GPU 3:                                             [Process token 0]──→ Output

Problem: Only 1 GPU active at a time! 75% idle!
```

### Micro-Batching for Pipeline

**Solution: Overlap processing with micro-batches**
```
Time →
  GPU 0: [Tok 0][Tok 1][Tok 2][Tok 3]────────────→
  GPU 1:       [Tok 0][Tok 1][Tok 2][Tok 3]──────→
  GPU 2:              [Tok 0][Tok 1][Tok 2][Tok 3]→
  GPU 3:                     [Tok 0][Tok 1][Tok 2][Tok 3]

Now: Higher utilization! But still "bubble" at start/end.
```

**Throughput Analysis:**
```
Single GPU: 1 token every 100 ms

Pipeline (4 GPUs, perfect):
  Theoretical: 4 tokens every 100 ms (4x speedup)
  Actual: ~3.2 tokens every 100 ms (3.2x speedup)
  Efficiency: 80% (pipeline bubbles)

Latency:
  Single GPU: 100 ms
  Pipeline: 100 ms (same! just for first token)
  Pipeline: 125 ms (with communication overhead)

Best for: Throughput, not latency
```

### Pipeline Parallelism Implementation

```python
class PipelineParallel:
    def __init__(self, num_stages, layers_per_stage):
        self.num_stages = num_stages
        self.layers_per_stage = layers_per_stage

    def forward(self, tokens, micro_batch_size):
        # Split into micro-batches
        micro_batches = split_batch(tokens, micro_batch_size)

        outputs = []
        for mb_idx, micro_batch in enumerate(micro_batches):
            # Stage 0: Process first layers
            if mb_idx == 0:
                hidden = self.stage_0.forward(micro_batch)
                send_to_gpu(hidden, gpu=1)

            # Stage 1: Process second layers
            if mb_idx >= 1:
                hidden = recv_from_gpu(gpu=0)
                hidden = self.stage_1.forward(hidden)
                send_to_gpu(hidden, gpu=2)

            # ... and so on for all stages

        return outputs
```

---

## NVLink and GPU Interconnects

### Interconnect Technologies

**PCIe (Peripheral Component Interconnect Express):**
```
Generation   Bandwidth per x16   Latency
───────────────────────────────────────
PCIe 3.0     16 GB/s             ~10 μs
PCIe 4.0     32 GB/s             ~8 μs
PCIe 5.0     64 GB/s             ~6 μs
PCIe 6.0     128 GB/s            ~5 μs

Pros: Universal, cheap
Cons: Limited bandwidth, high latency, shared bus
```

**NVLink (NVIDIA proprietary):**
```
Generation   Bandwidth (bidirectional)   Latency
────────────────────────────────────────────────
NVLink 2.0   300 GB/s (V100)             ~3 μs
NVLink 3.0   600 GB/s (A100)             ~2 μs
NVLink 4.0   900 GB/s (H100)             ~1.5 μs

Pros: 10-20x faster than PCIe, lower latency
Cons: NVIDIA only, expensive, limited connectivity
```

**NVSwitch (For full NVLink mesh):**
```
A100 DGX:  8 GPUs, all-to-all NVLink
           Total fabric: 4.8 TB/s

H100 DGX:  8 GPUs, NVLink 4.0
           Total fabric: 7.2 TB/s

Without NVSwitch: Only 2-4 GPUs can connect via NVLink
With NVSwitch: All GPUs can communicate at full speed
```

### Topology Matters!

**Bad Topology (PCIe only, 4 GPUs):**
```
CPU
 ├── GPU 0  \
 ├── GPU 1   } PCIe x16 each
 ├── GPU 2  /
 └── GPU 3

GPU 0 → GPU 3: Through CPU! (~10 GB/s, high latency)
```

**Good Topology (NVLink, 4 GPUs):**
```
GPU 0 ←→ GPU 1
  ↕        ↕
GPU 2 ←→ GPU 3

GPU 0 → GPU 3: Direct NVLink (300 GB/s)
Or 1 hop through GPU 2 (still fast)
```

**Check Topology:**
```bash
nvidia-smi topo -m

# Output (example):
        GPU0  GPU1  GPU2  GPU3
GPU0     X    NV12  NV12  NV12
GPU1    NV12   X    NV12  NV12
GPU2    NV12  NV12   X    NV12
GPU3    NV12  NV12  NV12   X

Legend:
  X = Self
  NV# = NVLink (higher number = more links)
  SYS = Connection traverses PCIe + CPU
  PHB = Connection traverses PCIe host bridge
```

---

## llama.cpp Multi-GPU Implementation

### Tensor Split Configuration

**Automatic (Equal Split):**
```cpp
// llama.cpp automatically detects GPUs and splits equally
llama_model_params params = llama_model_default_params();
params.n_gpu_layers = 32;  // All layers to GPU(s)

llama_model * model = llama_load_model_from_file("model.gguf", params);
// Automatically uses all available GPUs
```

**Manual Split:**
```cpp
// Custom split ratios
float tensor_split[4] = {0.4, 0.3, 0.2, 0.1};  // GPU0 gets 40%, GPU1 30%, etc.

// Set environment variable
setenv("LLAMA_ARG_TENSOR_SPLIT", "0.4,0.3,0.2,0.1", 1);

// Or via API:
params.tensor_split = tensor_split;
```

**When to use unequal split:**
- **Different GPU sizes:** A100 40GB + A100 80GB → split 0.33, 0.67
- **Shared GPUs:** GPU 0 also rendering UI → split 0.2, 0.8
- **Memory constraints:** One GPU has less free VRAM

### Split Buffer Implementation

**From `ggml-cuda.h`:**
```cpp
// Split tensor buffer type
ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(
    int main_device,
    const float * tensor_split
);

// Usage:
auto split_buf_type = ggml_backend_cuda_split_buffer_type(0, splits);

// When tensor is allocated with this buffer type:
// 1. Tensor rows are distributed across GPUs according to split ratios
// 2. Each GPU stores its portion in local VRAM
// 3. Kernels automatically operate on local data
// 4. Cross-GPU ops use NCCL for communication
```

**How it works:**
```cpp
// Simplified from ggml-cuda.cu
void split_tensor_across_gpus(
    ggml_tensor * tensor,
    const float * split_ratios,
    int num_gpus
) {
    const int64_t total_rows = tensor->ne[1];

    int64_t row_offset = 0;
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        // Calculate rows for this GPU
        int64_t rows_this_gpu = total_rows * split_ratios[gpu];

        // Allocate on GPU
        ggml_cuda_set_device(gpu);
        cudaMalloc(&tensor->data_split[gpu], rows_this_gpu * row_bytes);

        // Copy rows
        cudaMemcpy(
            tensor->data_split[gpu],
            tensor->data + row_offset * row_bytes,
            rows_this_gpu * row_bytes,
            cudaMemcpyHostToDevice
        );

        row_offset += rows_this_gpu;
    }
}
```

### Multi-GPU Matrix Multiplication

**From `ggml-cuda/mmf.cu` (simplified):**
```cpp
void ggml_cuda_mul_mat_f16_split(
    const ggml_tensor * src0,  // Weights (split across GPUs)
    const ggml_tensor * src1,  // Activations (replicated)
    ggml_tensor * dst,         // Output (split across GPUs)
    int main_device
) {
    const int num_gpus = ggml_cuda_info().device_count;

    for (int gpu = 0; gpu < num_gpus; gpu++) {
        ggml_cuda_set_device(gpu);

        // Each GPU computes its portion of output
        // No inter-GPU communication needed for GEMM!
        cublasSgemm(
            handles[gpu],
            CUBLAS_OP_T, CUBLAS_OP_N,
            n, m_split[gpu], k,  // m_split varies by GPU
            &alpha,
            src1_d[gpu], k,
            src0_d[gpu], k,
            &beta,
            dst_d[gpu], n
        );
    }

    // Synchronize all GPUs
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaDeviceSynchronize();
    }
}
```

**Key Insight:** Row-wise split of weight matrix allows **independent computation** on each GPU!

### Communication Overhead

**Ops requiring communication:**
1. **Attention:** KV cache may need gathering (if split)
2. **Softmax:** Needs max/sum across GPUs (if split)
3. **Layer Norm:** Needs mean/variance across GPUs (if split)

**Ops NOT requiring communication:**
1. **Matrix multiply:** If weights are row-split and activations replicated
2. **Element-wise ops:** Independent per GPU
3. **Embeddings:** If split by vocabulary

**llama.cpp optimization:**
```cpp
// Replicate small tensors (activations) to avoid communication
if (tensor_size < REPLICATION_THRESHOLD) {
    // Copy to all GPUs
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaMemcpy(tensor_d[gpu], tensor_h, size, H2D);
    }
} else {
    // Split large tensors (weights)
    split_tensor(tensor, split_ratios, num_gpus);
}
```

---

## Scaling Performance Analysis

### Ideal vs Actual Scaling

**Ideal Speedup (Amdahl's Law):**
```
Speedup = 1 / ((1 - P) + P/N)

Where:
  P = Fraction of work that can be parallelized
  N = Number of GPUs

For LLM inference, P ≈ 0.95 (95% parallelizable)

Ideal speedup with 4 GPUs:
  Speedup = 1 / ((1 - 0.95) + 0.95/4)
          = 1 / (0.05 + 0.2375)
          = 3.48x
```

**Actual Speedup (with communication):**
```
Model        1 GPU    2 GPU    4 GPU    8 GPU
───────────────────────────────────────────────
LLaMA-7B     50 ms    28 ms    18 ms    14 ms
Speedup       1.0x     1.79x    2.78x    3.57x
Efficiency    100%     90%      70%      45%

LLaMA-70B    N/A      320 ms   170 ms   95 ms
Speedup       N/A      1.0x     1.88x    3.37x
Efficiency    N/A      100%     94%      84%
```

**Why does efficiency drop?**
1. **Communication overhead** - NVLink has finite bandwidth
2. **Load imbalance** - Not all GPUs have equal work
3. **Synchronization** - GPUs wait for slowest
4. **Memory bandwidth** - Shared memory bus contention

### Batch Size Impact

**Throughput Scaling:**
```
LLaMA-7B, 4× A100, Tensor Parallelism

Batch Size   Tokens/sec   GPU Util   Speedup vs 1 GPU
──────────────────────────────────────────────────────
1            20           35%        2.1x
4            72           58%        2.5x
16           256          78%        3.1x
64           892          92%        3.7x

Larger batch → Better GPU utilization → Better speedup
```

**Why?**
- Communication cost is amortized over more work
- Higher arithmetic intensity (compute vs memory ratio)
- Better occupancy on each GPU

### When to Use Multi-GPU

**Use 1 GPU when:**
- Model fits in single GPU VRAM
- Latency is critical (avoid communication overhead)
- Serving single user

**Use 2-4 GPUs when:**
- Model is 2-4x too large for single GPU
- Have NVLink between GPUs
- Moderate batch sizes (4-16)

**Use 8+ GPUs when:**
- Very large models (70B+, 175B+)
- High throughput serving (batch 32-64)
- Training (data parallelism)

**Avoid Multi-GPU when:**
- Only connected via PCIe (slow, <2x speedup)
- Very small batch sizes (overhead dominates)
- Different GPU generations (slowest GPU is bottleneck)

---

## Key Takeaways

1. **Tensor parallelism splits layers horizontally** - each GPU computes subset of heads/neurons
2. **Pipeline parallelism splits layers vertically** - each GPU computes subset of layers
3. **NVLink is essential for good multi-GPU performance** - 10-20x faster than PCIe
4. **llama.cpp uses row-wise tensor splitting** - minimizes communication in GEMM
5. **Scaling efficiency decreases with more GPUs** - communication overhead increases
6. **Larger batches improve multi-GPU efficiency** - amortize communication cost

---

## Interview Questions

1. **Q:** What's the difference between tensor and pipeline parallelism?
   **A:** Tensor parallelism splits each layer across GPUs (horizontal split), requiring communication per layer. Pipeline parallelism splits layers across GPUs (vertical split), requiring communication between stages but not within layers.

2. **Q:** Why is NVLink important for multi-GPU LLM inference?
   **A:** LLM inference requires frequent inter-GPU communication (all-reduce, all-gather). NVLink provides 300-900 GB/s bidirectional bandwidth vs PCIe's 16-64 GB/s, reducing communication overhead from bottleneck to <10% of compute time.

3. **Q:** How does llama.cpp split tensors across GPUs?
   **A:** Row-wise split of weight matrices according to user-specified ratios. Activations are replicated to all GPUs. This allows independent GEMM on each GPU without communication until results are needed.

4. **Q:** Why does multi-GPU scaling efficiency decrease with more GPUs?
   **A:** (1) Communication overhead increases, (2) synchronization costs grow, (3) load imbalance becomes harder to avoid, (4) Amdahl's Law limits parallelizable fraction.

5. **Q:** When is pipeline parallelism better than tensor parallelism?
   **A:** When: (1) model has many layers, (2) high throughput with batching is needed, (3) limited inter-GPU bandwidth (PCIe instead of NVLink), (4) layers have different compute requirements (load balancing).

---

**Next Lesson:** [05-alternative-backends.md](05-alternative-backends.md)

**Related Labs:**
- [Lab 3: Multi-GPU Setup](../labs/lab3-multi-gpu-setup.md)
