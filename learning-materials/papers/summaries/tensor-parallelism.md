# Tensor Parallelism and Model Parallelism for Large Language Models

**Topic**: Distributed Training and Inference Strategies
**Key Papers**: Megatron-LM, GPipe, PipeDream
**Module**: 4 - GPU Acceleration & Performance
**Impact**: ⭐⭐⭐⭐

---

## Executive Summary

Tensor parallelism splits individual layers across multiple GPUs, enabling training and inference of models too large for a single GPU. Combined with pipeline parallelism and data parallelism, it forms the foundation of distributed LLM systems.

**Key Concept**: Split tensors along specific dimensions to parallelize computation while minimizing communication overhead.

---

## 1. Types of Parallelism

### 1.1 Data Parallelism (DP)
```
Each GPU has full model copy, processes different data batches

GPU 0: Model copy, Batch 0
GPU 1: Model copy, Batch 1
GPU 2: Model copy, Batch 2
GPU 3: Model copy, Batch 3

Sync: Gradient averaging after each step

Pros: Simple, scales well for small models
Cons: Memory redundancy, doesn't help with large models
```

### 1.2 Tensor Parallelism (TP)
```
Split individual layers across GPUs

GPU 0: Left half of weight matrix
GPU 1: Right half of weight matrix

Computation: Parallel matmul, then reduce
Communication: All-reduce after each layer

Pros: Enables large models, high GPU utilization
Cons: High communication overhead
```

### 1.3 Pipeline Parallelism (PP)
```
Split layers across GPUs

GPU 0: Layers 0-7
GPU 1: Layers 8-15
GPU 2: Layers 16-23
GPU 3: Layers 24-31

Computation: Sequential through pipeline
Communication: Activations between stages

Pros: Low communication, simple
Cons: GPU bubbles (idle time), poor parallelism for small batches
```

---

## 2. Tensor Parallelism in Detail

### 2.1 Column-wise Parallel Matrix Multiply

```python
# Standard matmul: Y = XW
# W: [d_in, d_out]
# X: [batch, seq_len, d_in]

# Split W along columns across N GPUs
# Each GPU computes partial output

def column_parallel_linear(X, W_splits, rank, world_size):
    """
    W_splits[i]: [d_in, d_out/world_size] on GPU i

    Example: d_out=4096, world_size=4
    GPU 0: W[:, 0:1024]
    GPU 1: W[:, 1024:2048]
    GPU 2: W[:, 2048:3072]
    GPU 3: W[:, 3072:4096]
    """
    # Each GPU computes its partition
    Y_local = X @ W_splits[rank]  # [batch, seq_len, d_out/world_size]

    # No communication needed! Outputs are independent
    return Y_local

# Later: Concatenate outputs if needed
# Y = concat([Y_0, Y_1, Y_2, Y_3], dim=-1)
```

### 2.2 Row-wise Parallel Matrix Multiply

```python
def row_parallel_linear(X, W_splits, rank, world_size):
    """
    W_splits[i]: [d_in/world_size, d_out] on GPU i

    Each GPU processes part of input dimension
    """
    # Split input across GPUs (requires communication)
    X_local = X[:, :, rank * (d_in // world_size):(rank+1) * (d_in // world_size)]

    # Compute partial output
    Y_local = X_local @ W_splits[rank]  # [batch, seq_len, d_out]

    # All-reduce to sum partial outputs
    Y = all_reduce(Y_local, op=SUM)  # Communication!

    return Y
```

---

## 3. Megatron-LM Tensor Parallelism

### 3.1 Transformer Block Parallelization

```python
class ParallelTransformerBlock(nn.Module):
    """
    Megatron-LM tensor parallel transformer block
    """
    def __init__(self, d_model, num_heads, d_ff, tp_size):
        super().__init__()
        self.tp_size = tp_size

        # Attention: Column-parallel for QKV, Row-parallel for output
        self.qkv = ColumnParallelLinear(d_model, 3 * d_model)
        self.attn_out = RowParallelLinear(d_model, d_model)

        # FFN: Column-parallel for up, Row-parallel for down
        self.ffn_up = ColumnParallelLinear(d_model, d_ff)
        self.ffn_down = RowParallelLinear(d_ff, d_model)

    def forward(self, x):
        # Attention
        qkv = self.qkv(x)  # Column parallel, no communication
        # Split QKV (each GPU has subset of heads)
        q, k, v = split_heads(qkv, self.num_heads // self.tp_size)

        # Attention computation (independent on each GPU)
        attn_out = scaled_dot_product_attention(q, k, v)

        # All-reduce attention output
        x = x + self.attn_out(attn_out)  # Row parallel, all-reduce

        # FFN
        ffn_out = self.ffn_up(x)  # Column parallel, no communication
        ffn_out = gelu(ffn_out)
        x = x + self.ffn_down(ffn_out)  # Row parallel, all-reduce

        return x

# Communication: 2 all-reduces per layer (attention out + FFN out)
```

### 3.2 Communication Pattern

```
Forward Pass (per layer):
1. Attention: Column-parallel QKV (no comm)
2. Compute attention (independent)
3. Row-parallel output (all-reduce) ← Communication
4. FFN up projection (column-parallel, no comm)
5. FFN down projection (row-parallel, all-reduce) ← Communication

Total: 2 all-reduces per layer per forward pass
Backward: 2 all-reduces per layer (symmetric)
```

---

## 4. Sequence Parallelism

### Extension for Long Sequences

```python
# Problem: Activation memory grows with sequence length
# LayerNorm, Dropout operate on full sequence

# Solution: Also parallelize sequence dimension

def sequence_parallel_layernorm(x, weight, bias, rank, world_size):
    """
    Split sequence dimension across GPUs
    """
    seq_len = x.shape[1]
    local_seq_len = seq_len // world_size

    # Each GPU processes part of sequence
    x_local = x[:, rank * local_seq_len:(rank+1) * local_seq_len, :]

    # Local layernorm
    x_normalized = layernorm(x_local, weight, bias)

    return x_normalized

# Combine with tensor parallelism for maximum memory efficiency
```

---

## 5. Performance Analysis

### 5.1 Communication Overhead

```python
# All-reduce bandwidth requirement
def compute_communication_cost(d_model, seq_len, batch_size, tp_size):
    """
    Per layer communication
    """
    # Attention output: [batch, seq_len, d_model]
    attention_bytes = batch_size * seq_len * d_model * 2  # FP16

    # FFN output: same size
    ffn_bytes = attention_bytes

    total_bytes = attention_bytes + ffn_bytes

    # All-reduce requires 2× data transfer (ring all-reduce)
    bandwidth = total_bytes * 2

    print(f"Bandwidth per layer: {bandwidth / 1e9:.2f} GB")
    return bandwidth

# Example: LLaMA 7B, batch=32, seq=2048, tp=4
# ~1 GB per layer × 32 layers = 32 GB total
# On 800 Gbps NVLink: 32 GB / 100 GB/s = 0.32s communication overhead
```

### 5.2 Optimal TP Size

```
Trade-off:
- Larger TP size: Fit bigger models
- Smaller TP size: Less communication overhead

Rule of thumb:
- TP size = number of GPUs per node (NVLink fast)
- 4-8 GPUs typical (A100 node)
- Avoid TP across nodes (network slower than NVLink)

Example:
- 32 GPUs total → TP=8 (intra-node), DP=4 (across nodes)
```

---

## 6. llama.cpp and Tensor Parallelism

### 6.1 Current Status

```
llama.cpp tensor parallelism support:
- Not natively supported (CPU-focused)
- Single-GPU inference only
- Multi-GPU via layer split (simple parallelism)
```

### 6.2 Layer Split (Simple Parallelism)

```bash
# Distribute layers across GPUs
./llama-cli \
  -m model.gguf \
  --tensor-split 0.5,0.3,0.2  # GPU 0: 50%, GPU 1: 30%, GPU 2: 20%

# Example: 32-layer model, 3 GPUs
# GPU 0: Layers 0-15 (16 layers)
# GPU 1: Layers 16-25 (10 layers)
# GPU 2: Layers 26-31 (6 layers)
```

**Limitations**:
- Not true tensor parallelism (layers, not tensors)
- Sequential execution (no parallelism within layer)
- Works for multi-GPU systems with limited VRAM per GPU

---

## 7. Advanced: 3D Parallelism

### Combining TP, PP, DP

```
For very large models (100B-1T params):
- Data Parallel: 4 ways
- Tensor Parallel: 8 ways (per node)
- Pipeline Parallel: 4 ways

Total: 4 × 8 × 4 = 128 GPUs

Example partition (GPT-3 175B):
- 96 layers
- TP = 8 (split within layer)
- PP = 12 (8 layers per stage)
- DP = 4 (4 data replicas)

Total: 8 × 12 = 96 GPUs (1 per pipeline stage)
      × 4 = 384 GPUs total
```

---

## 8. Practical Recommendations

### For Large-Scale Inference

✅ **When to use Tensor Parallelism**:
- Model doesn't fit on single GPU
- High-throughput serving (multiple GPUs available)
- NVLink available (fast inter-GPU communication)

✅ **Alternatives for llama.cpp users**:
- **Quantization**: Q4_K_M fits 70B models on single A100
- **Layer offloading**: Split layers across GPUs (simpler than TP)
- **vLLM/TGI**: If TP needed, use specialized serving frameworks

### Configuration Guidelines

```python
def recommend_parallelism(model_size_gb, num_gpus, gpu_memory_gb):
    """
    Recommend parallelism strategy
    """
    single_gpu_fits = model_size_gb < gpu_memory_gb * 0.8

    if single_gpu_fits:
        return "No parallelism needed, use single GPU"

    elif num_gpus <= 8 and model_size_gb < num_gpus * gpu_memory_gb * 0.6:
        return f"Use TP={num_gpus} (model fits with tensor parallelism)"

    else:
        tp_size = min(8, num_gpus)  # Max 8 for NVLink
        pp_size = (num_gpus + tp_size - 1) // tp_size
        return f"Use TP={tp_size}, PP={pp_size} (hybrid parallelism)"

# Example:
recommend_parallelism(140, 8, 80)  # 70B FP16, 8× A100
# → "Use TP=4 (model fits with tensor parallelism)"
```

---

## 9. Key Takeaways

### Tensor Parallelism Fundamentals
- Split layers across GPUs (column/row parallelism)
- Minimize communication (2 all-reduces per layer)
- Best with fast interconnect (NVLink > PCIe > Network)

### For llama.cpp Users
- Native TP not supported (CPU-first design)
- Use quantization to fit models on fewer GPUs
- Layer split for multi-GPU when needed
- Consider vLLM/TGI for production TP serving

---

## Further Reading

- **Megatron-LM**: https://arxiv.org/abs/1909.08053
- **GPipe**: https://arxiv.org/abs/1811.06965
- **DeepSpeed**: https://www.deepspeed.ai/
- **PyTorch FSDP**: Fully Sharded Data Parallel

---

**Status**: Complete | Module 4 Complete (3/3) papers
