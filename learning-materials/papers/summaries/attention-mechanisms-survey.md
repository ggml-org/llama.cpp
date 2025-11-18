# Survey of Attention Mechanisms in Deep Learning

**Paper Type**: Survey/Review Paper
**Key Papers Reviewed**: Multiple foundational works (2014-2023)
**Relevance**: Module 2 - Understanding LLM Architecture
**Reading Time**: 45-60 minutes
**Practical Impact**: ⭐⭐⭐⭐⭐

---

## Executive Summary

Attention mechanisms have revolutionized deep learning, particularly in sequence modeling tasks. This survey covers the evolution from basic additive attention (Bahdanau, 2014) to modern variants like Flash Attention and Multi-Query Attention. Understanding these mechanisms is crucial for optimizing LLM inference and implementing efficient serving systems.

**Key Takeaway**: Attention allows models to dynamically focus on relevant parts of input sequences, but at O(n²) computational cost. Modern variants optimize this trade-off through algorithmic and hardware-aware improvements.

---

## 1. Foundational Attention Mechanisms

### 1.1 Bahdanau Attention (2014)

**Core Innovation**: Neural Machine Translation by Jointly Learning to Align and Translate

**Mechanism**:
```python
# Additive (concat) attention
def bahdanau_attention(query, keys, values):
    # query: [batch, d_model]
    # keys: [batch, seq_len, d_model]
    # values: [batch, seq_len, d_model]

    # Compute alignment scores
    scores = tanh(W1 @ query + W2 @ keys)  # [batch, seq_len, d_hidden]
    scores = v @ scores  # [batch, seq_len]

    # Apply softmax to get attention weights
    weights = softmax(scores)  # [batch, seq_len]

    # Weighted sum of values
    context = weights @ values  # [batch, d_model]
    return context, weights
```

**Key Insights**:
- Solves fixed-length encoding bottleneck in seq2seq models
- Learns alignment between source and target sequences
- Computational complexity: O(n × m) where n=source, m=target length

**Limitations**:
- Requires learned parameters (W1, W2, v)
- Not parallelizable across positions during training
- Slower than scaled dot-product attention

---

### 1.2 Scaled Dot-Product Attention (Vaswani, 2017)

**Core Innovation**: "Attention Is All You Need" - Transformer architecture

**Mechanism**:
```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [batch, num_heads, seq_len, d_k]
    K: [batch, num_heads, seq_len, d_k]
    V: [batch, num_heads, seq_len, d_v]
    """
    d_k = Q.size(-1)

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, heads, seq_len, seq_len]
    scores = scores / math.sqrt(d_k)  # Scaling prevents softmax saturation

    # Apply mask (for causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Attention weights
    attn_weights = F.softmax(scores, dim=-1)  # [batch, heads, seq_len, seq_len]

    # Weighted sum of values
    output = torch.matmul(attn_weights, V)  # [batch, heads, seq_len, d_v]
    return output, attn_weights
```

**Why Scaling by √d_k?**
- Without scaling, dot products grow large in magnitude
- Large values push softmax into saturation regions (tiny gradients)
- Scaling maintains variance ≈ 1 for random Q, K

**Mathematical Derivation**:
```
Var(Q · K) = d_k × Var(Q) × Var(K)
If Var(Q) = Var(K) = 1, then Var(Q · K) = d_k
Dividing by √d_k gives: Var((Q · K) / √d_k) = 1
```

**Computational Complexity**:
- Time: O(n² × d) where n=sequence length, d=dimension
- Space: O(n²) to materialize attention matrix
- **This is the key bottleneck for long sequences!**

---

### 1.3 Multi-Head Attention

**Core Insight**: Multiple attention heads learn different representation subspaces

**Implementation**:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections and reshape to [batch, heads, seq_len, d_k]
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and apply final linear
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)

        return output
```

**Why Multiple Heads?**
- Different heads can attend to different positions
- Some heads focus on syntax, others on semantics
- Ensemble effect improves representation power
- Empirically: 8-16 heads work well for most tasks

---

## 2. Modern Attention Variants for Efficiency

### 2.1 Multi-Query Attention (MQA) - Shazeer 2019

**Problem**: KV-cache memory dominates inference cost for long sequences

**Solution**: Share single K, V across all query heads

**Architecture Change**:
```python
# Standard Multi-Head Attention
Q: [batch, num_heads, seq_len, d_k]  # 32 heads × 128 dim = 4096
K: [batch, num_heads, seq_len, d_k]  # 32 heads × 128 dim = 4096
V: [batch, num_heads, seq_len, d_k]  # 32 heads × 128 dim = 4096

# Multi-Query Attention
Q: [batch, num_heads, seq_len, d_k]  # 32 heads × 128 dim = 4096
K: [batch, 1, seq_len, d_k]           # 1 head × 128 dim = 128
V: [batch, 1, seq_len, d_k]           # 1 head × 128 dim = 128
```

**Memory Savings**:
```
LLaMA-7B with 2048 context:
- MHA: 32 heads × 128 dim × 2048 tokens × 2 bytes = 16 MB per layer × 32 layers = 512 MB
- MQA: 1 head × 128 dim × 2048 tokens × 2 bytes = 0.5 MB per layer × 32 layers = 16 MB
Reduction: 32× smaller KV-cache!
```

**Trade-offs**:
- ✅ Massive memory reduction
- ✅ Faster inference (less data movement)
- ❌ Quality degradation (~1-2% on benchmarks)
- ❌ Requires training from scratch or fine-tuning

**Use Cases**:
- Real-time inference with strict latency requirements
- Edge deployment with limited memory
- Serving many concurrent requests (larger batch sizes possible)

---

### 2.2 Grouped-Query Attention (GQA) - LLaMA 2

**Problem**: MQA trades too much quality for memory savings

**Solution**: Middle ground - multiple KV heads (groups), but fewer than query heads

**Architecture**:
```python
# Grouped-Query Attention (GQA)
num_query_heads = 32
num_kv_heads = 8  # 4 query heads per KV head

Q: [batch, 32, seq_len, 128]  # 32 query heads
K: [batch, 8, seq_len, 128]   # 8 KV heads
V: [batch, 8, seq_len, 128]   # 8 KV heads

# Each KV head is shared by 4 query heads
```

**Implementation**:
```python
def grouped_query_attention(Q, K, V, num_kv_heads):
    """
    Q: [batch, num_q_heads, seq_len, d_k]
    K: [batch, num_kv_heads, seq_len, d_k]
    V: [batch, num_kv_heads, seq_len, d_k]
    """
    batch, num_q_heads, seq_len, d_k = Q.shape
    num_kv_heads = K.shape[1]
    group_size = num_q_heads // num_kv_heads

    # Repeat K, V to match Q heads
    K = K.repeat_interleave(group_size, dim=1)  # [batch, num_q_heads, seq_len, d_k]
    V = V.repeat_interleave(group_size, dim=1)

    # Standard scaled dot-product attention
    return scaled_dot_product_attention(Q, K, V)
```

**GQA Spectrum**:
```
MHA (32 KV heads) ←→ GQA-8 (8 KV heads) ←→ MQA (1 KV head)
       Best quality              Balanced           Most efficient
```

**LLaMA 2 Results**:
- GQA-8 vs MHA: 4× KV-cache reduction, <1% quality loss
- GQA-8 vs MQA: Better quality, only 8× larger KV-cache

**Practical Recommendation**:
- **LLaMA 2 7B**: Uses GQA with 8 KV heads (32 query heads)
- **LLaMA 2 70B**: Uses GQA with 8 KV heads (64 query heads)
- Sweet spot: num_kv_heads = num_query_heads / 4

---

### 2.3 Flash Attention - Dao et al. 2022

**Problem**: Standard attention has excessive memory I/O between HBM (GPU memory) and SRAM (on-chip cache)

**Key Insight**: Attention doesn't need to materialize full N×N matrix

**Algorithm**:
1. Tile Q, K, V into blocks that fit in SRAM
2. Compute attention incrementally using online softmax
3. Fuse operations to minimize HBM reads/writes

**Pseudocode**:
```python
def flash_attention_forward(Q, K, V, block_size=128):
    """
    Memory-efficient attention via tiling and recomputation
    """
    N, d = Q.shape
    num_blocks = (N + block_size - 1) // block_size

    O = torch.zeros_like(Q)  # Output
    l = torch.zeros(N)        # Softmax normalization (log-sum-exp)
    m = torch.full((N,), float('-inf'))  # Running max for numerical stability

    for j in range(num_blocks):
        # Load K, V blocks into SRAM
        K_j = K[j*block_size:(j+1)*block_size]  # [block_size, d]
        V_j = V[j*block_size:(j+1)*block_size]

        for i in range(num_blocks):
            # Load Q block into SRAM
            Q_i = Q[i*block_size:(i+1)*block_size]  # [block_size, d]

            # Compute attention scores for this block
            S_ij = Q_i @ K_j.T / sqrt(d)  # [block_size, block_size]

            # Online softmax (numerically stable)
            m_new = torch.maximum(m[i*block_size:(i+1)*block_size], S_ij.max(dim=1))
            l_new = torch.exp(m - m_new) * l + torch.exp(S_ij - m_new).sum(dim=1)

            # Update output
            O_i = O[i*block_size:(i+1)*block_size]
            O_i = O_i * torch.exp(m - m_new) + torch.exp(S_ij - m_new) @ V_j

            m[i*block_size:(i+1)*block_size] = m_new
            l[i*block_size:(i+1)*block_size] = l_new

    # Final normalization
    O = O / l.unsqueeze(-1)
    return O
```

**Performance Gains**:
- **Speed**: 2-4× faster than standard attention
- **Memory**: O(N) instead of O(N²) for intermediate activations
- **Quality**: Exact attention (not approximate!)

**Hardware Requirements**:
- Works best on modern GPUs (A100, H100) with fast HBM
- Less effective on older GPUs with slower memory
- Not applicable to CPU inference (different memory hierarchy)

**llama.cpp Integration**:
```bash
# Enable Flash Attention in llama.cpp (CUDA backend)
cmake -B build -DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON
cmake --build build

# Usage automatically enabled for CUDA inference
./llama-cli -m model.gguf --n-gpu-layers 32
```

---

### 2.4 Flash Attention 2 (2023)

**Further Optimizations**:
1. Better work partitioning across thread blocks
2. Reduced non-matmul FLOPs
3. Improved parallelism for different sequence lengths

**Speedup**: 2× faster than FlashAttention-1 (4-8× vs standard)

---

## 3. Efficient Attention Variants

### 3.1 Linear Attention (Katharopoulos et al. 2020)

**Idea**: Approximate attention with kernel feature maps

**Standard Attention**:
```
Attention(Q, K, V) = softmax(QK^T)V
Complexity: O(n²d)
```

**Linear Attention**:
```
Attention(Q, K, V) = φ(Q)(φ(K)^T V)
Complexity: O(nd²) - linear in sequence length!
```

**Feature Map φ**:
```python
def feature_map(x):
    # ELU + 1 (ensures non-negativity)
    return F.elu(x) + 1
```

**Trade-offs**:
- ✅ O(n) complexity - enables million-token contexts
- ✅ Recurrent formulation possible
- ❌ Quality loss compared to softmax attention
- ❌ Less expressive for long-range dependencies

---

### 3.2 Sparse Attention Patterns

**Motivation**: Most attention weights are near-zero for long sequences

**Patterns**:

1. **Local Attention** (windowed):
```
Each token attends to k neighbors
Complexity: O(nk) where k << n
```

2. **Strided Attention**:
```
Attend to every s-th token
Useful for hierarchical patterns
```

3. **Block-Sparse Attention** (BigBird, Longformer):
```
Combination of:
- Local attention (sliding window)
- Global attention (few tokens attend to all)
- Random attention (sparse random connections)
```

**Implementation Example** (Local Attention):
```python
def local_attention(Q, K, V, window_size=256):
    """Attend only to nearby tokens"""
    seq_len = Q.size(2)

    # Create local attention mask
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 1

    # Apply masked attention
    return scaled_dot_product_attention(Q, K, V, mask)
```

---

## 4. Practical Implications for llama.cpp

### 4.1 Attention Mechanism Selection

**Standard Transformer Attention**:
- Used in: LLaMA 1, GPT-3, most models
- llama.cpp implementation: Optimized BLAS/cuBLAS matmul
- Best for: CPU inference, moderate sequence lengths (<4K tokens)

**Grouped-Query Attention (GQA)**:
- Used in: LLaMA 2, LLaMA 3
- llama.cpp: Automatic detection from model architecture
- Memory savings: Proportional to num_query_heads / num_kv_heads
- Best for: Serving multiple users, long contexts

**Flash Attention**:
- Used in: CUDA backend when available
- llama.cpp: Compile-time flag `-DGGML_FLASH_ATTN=ON`
- Speedup: 2-3× for long contexts (>2K tokens)
- Best for: GPU inference, long documents

---

### 4.2 Memory Calculation with Different Attention Types

**Example: LLaMA 2 7B (4096 context)**

```python
def calculate_kv_cache_size(
    num_layers=32,
    num_kv_heads=8,  # GQA
    head_dim=128,
    context_length=4096,
    bytes_per_element=2  # FP16
):
    kv_cache_per_layer = num_kv_heads * head_dim * context_length * 2  # K and V
    total_kv_cache = kv_cache_per_layer * num_layers * bytes_per_element
    return total_kv_cache / (1024**2)  # Convert to MB

# LLaMA 2 7B with GQA-8
gqa_size = calculate_kv_cache_size(num_kv_heads=8)
print(f"GQA-8 KV-cache: {gqa_size:.1f} MB")  # ~128 MB

# Hypothetical MHA version
mha_size = calculate_kv_cache_size(num_kv_heads=32)
print(f"MHA KV-cache: {mha_size:.1f} MB")    # ~512 MB

# Hypothetical MQA version
mqa_size = calculate_kv_cache_size(num_kv_heads=1)
print(f"MQA KV-cache: {mqa_size:.1f} MB")    # ~16 MB
```

---

### 4.3 Choosing Attention Configuration

**Decision Tree**:

```
┌─ Inference Hardware?
│
├─ CPU
│  ├─ Context < 2K: Standard attention (best compatibility)
│  └─ Context > 2K: GQA if available (memory savings)
│
└─ GPU
   ├─ Modern GPU (A100, H100)
   │  └─ Use Flash Attention (2-4× speedup)
   │
   └─ Older GPU
      ├─ Memory limited: GQA or MQA
      └─ Compute limited: Standard attention with batch optimization
```

---

## 5. Code Examples for llama.cpp Users

### 5.1 Checking Model Attention Configuration

```python
from gguf import GGUFReader

def inspect_attention_config(model_path):
    reader = GGUFReader(model_path)

    # Extract attention-related metadata
    n_heads = reader.fields.get('llama.attention.head_count')
    n_kv_heads = reader.fields.get('llama.attention.head_count_kv')

    if n_kv_heads is None or n_kv_heads == n_heads:
        attention_type = "Multi-Head Attention (MHA)"
        kv_heads = n_heads
    elif n_kv_heads == 1:
        attention_type = "Multi-Query Attention (MQA)"
        kv_heads = 1
    else:
        attention_type = f"Grouped-Query Attention (GQA-{n_kv_heads})"
        kv_heads = n_kv_heads

    print(f"Attention Type: {attention_type}")
    print(f"Query Heads: {n_heads}")
    print(f"KV Heads: {kv_heads}")
    print(f"Sharing Ratio: {n_heads / kv_heads}:1")

    return attention_type, n_heads, kv_heads

# Usage
inspect_attention_config("llama-2-7b.Q4_K_M.gguf")
```

---

### 5.2 Estimating KV-Cache Requirements

```python
def estimate_memory_requirements(
    model_path,
    context_length=4096,
    batch_size=1
):
    reader = GGUFReader(model_path)

    n_layers = reader.fields['llama.block_count']
    n_kv_heads = reader.fields.get('llama.attention.head_count_kv',
                                     reader.fields['llama.attention.head_count'])
    n_embd = reader.fields['llama.embedding_length']
    head_dim = n_embd // reader.fields['llama.attention.head_count']

    # KV-cache size (FP16)
    kv_cache_bytes = (
        2 *  # K and V
        n_layers *
        n_kv_heads *
        head_dim *
        context_length *
        batch_size *
        2  # FP16 bytes
    )

    kv_cache_mb = kv_cache_bytes / (1024**2)

    print(f"KV-Cache Memory: {kv_cache_mb:.1f} MB")
    print(f"Per layer: {kv_cache_mb / n_layers:.2f} MB")
    print(f"Per token: {kv_cache_bytes / context_length / 1024:.2f} KB")

    return kv_cache_mb

# Usage
estimate_memory_requirements("llama-2-7b.Q4_K_M.gguf", context_length=8192)
```

---

## 6. Key Takeaways for Module 2

### 6.1 Attention Evolution Summary

| Mechanism | Complexity | Memory | Quality | Use Case |
|-----------|-----------|---------|---------|----------|
| Bahdanau | O(nm) | O(n) | Good | Legacy seq2seq |
| Scaled Dot-Product | O(n²d) | O(n²) | Excellent | Standard transformer |
| Multi-Query (MQA) | O(n²d) | O(n) | Good | Memory-constrained |
| Grouped-Query (GQA) | O(n²d) | O(n×G) | Very Good | Balanced (LLaMA 2) |
| Flash Attention | O(n²d) | O(n) | Excellent | GPU, long context |
| Linear Attention | O(nd²) | O(d²) | Fair | Ultra-long context |

### 6.2 Must-Know Concepts

✅ **For Inference**:
- Scaled dot-product attention is O(n²) → bottleneck for long sequences
- KV-cache trades computation for memory
- GQA provides best quality/memory trade-off
- Flash Attention is essential for GPU long-context inference

✅ **For Architecture Selection**:
- Check model's attention type in GGUF metadata
- GQA models have better serving characteristics
- MQA models are rare but most memory-efficient

✅ **For Optimization**:
- Enable Flash Attention for CUDA builds
- Monitor KV-cache usage (grows with context length)
- Batch multiple requests to amortize attention overhead

---

## 7. Further Reading

### Essential Papers
1. **Attention Is All You Need** (Vaswani et al., 2017)
   - https://arxiv.org/abs/1706.03762
   - THE foundational paper

2. **Flash Attention** (Dao et al., 2022)
   - https://arxiv.org/abs/2205.14135
   - Must-read for GPU optimization

3. **GQA in LLaMA 2** (Meta, 2023)
   - https://arxiv.org/abs/2307.09288
   - Section 2.2 on architecture

### Implementation References
- PyTorch Scaled Dot-Product Attention: `torch.nn.functional.scaled_dot_product_attention`
- Flash Attention GitHub: https://github.com/Dao-AILab/flash-attention
- llama.cpp attention kernels: `ggml-cuda/attention.cu`

---

**Document Information**
- Created: 2025-11-18
- Module: 2 - Understanding LLM Architecture
- Author: Research Coordinator
- Status: Complete
- Next: Read transformer-architecture-papers.md
