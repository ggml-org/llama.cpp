# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Paper**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
**Authors**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford)
**Published**: June 2022 (NeurIPS 2022)
**Link**: https://arxiv.org/abs/2205.14135
**Module**: 4 - GPU Acceleration & Performance
**Impact**: ⭐⭐⭐⭐⭐

---

## Executive Summary

FlashAttention achieves 2-4× speedup and reduces memory from O(n²) to O(n) for attention computation through IO-aware algorithm design. By carefully orchestrating data movement between GPU memory hierarchy levels, it enables training and inference on sequences 4-16× longer than standard attention.

**Key Innovation**: Tiling + kernel fusion + online softmax = no materialization of full attention matrix

---

## 1. Problem: Standard Attention is Memory-Bound

### Standard Attention Algorithm
```python
def standard_attention(Q, K, V, mask=None):
    # Q, K, V: [batch, heads, seq_len, head_dim]
    N, d = Q.shape[-2:]

    # Step 1: Compute S = QK^T [N×N matrix]
    S = Q @ K.transpose(-2, -1) / sqrt(d)  # Write N×N to HBM

    # Step 2: Apply mask
    if mask is not None:
        S = S + mask  # Read + write N×N from HBM

    # Step 3: Softmax
    P = softmax(S, dim=-1)  # Read + write N×N from HBM

    # Step 4: Multiply by V
    O = P @ V  # Read N×N from HBM, write N×d

    return O

# HBM accesses:
# - Write S: N²
# - Read/write mask: 2N²
# - Read/write softmax: 2N²
# - Read P, write O: N² + Nd
# Total: ~6N² + Nd operations on slow HBM!
```

**For seq_len=4096, head_dim=64**:
- Attention matrix: 4096² × 2 bytes = 32 MB per head
- LLaMA 32 heads × 32 layers = 32 GB memory traffic!
- Bottleneck: Moving data between HBM and compute units

---

## 2. FlashAttention Algorithm

### Key Ideas
1. **Tiling**: Break Q, K, V into blocks that fit in SRAM
2. **Recomputation**: Don't store attention matrix, recompute in backward pass
3. **Online softmax**: Incremental softmax without full matrix
4. **Kernel fusion**: Fuse all attention ops into single CUDA kernel

### Forward Pass

```python
def flash_attention_forward(Q, K, V, block_size=128):
    """
    FlashAttention forward pass

    Memory: O(N) instead of O(N²)
    Speed: 2-4× faster
    """
    N, d = Q.shape  # seq_len, head_dim
    N_blocks = (N + block_size - 1) // block_size

    # Output and softmax statistics
    O = torch.zeros_like(Q)  # Output
    l = torch.zeros(N)        # Softmax denominator (logsumexp)
    m = torch.full((N,), float('-inf'))  # Softmax max (for numerical stability)

    # Iterate over K, V blocks (outer loop)
    for j in range(N_blocks):
        # Load K, V block into SRAM (on-chip memory)
        K_j = K[j * block_size:(j+1) * block_size]
        V_j = V[j * block_size:(j+1) * block_size]

        # Iterate over Q blocks (inner loop)
        for i in range(N_blocks):
            # Load Q block into SRAM
            Q_i = Q[i * block_size:(i+1) * block_size]

            # Compute attention scores for this block (in SRAM)
            S_ij = (Q_i @ K_j.T) / sqrt(d)  # [block_size × block_size]

            # Online softmax update (incremental computation)
            m_new = torch.maximum(
                m[i * block_size:(i+1) * block_size],
                S_ij.max(dim=1)[0]
            )

            # Update normalization factor
            l_new = (
                torch.exp(m[i * block_size:(i+1) * block_size] - m_new) *
                l[i * block_size:(i+1) * block_size] +
                torch.exp(S_ij - m_new.unsqueeze(1)).sum(dim=1)
            )

            # Update output (rescale previous output + add new contribution)
            O_i = O[i * block_size:(i+1) * block_size]
            O_i = (
                O_i * torch.exp(m[i * block_size:(i+1) * block_size] - m_new).unsqueeze(1) +
                torch.exp(S_ij - m_new.unsqueeze(1)) @ V_j
            )

            O[i * block_size:(i+1) * block_size] = O_i
            m[i * block_size:(i+1) * block_size] = m_new
            l[i * block_size:(i+1) * block_size] = l_new

    # Final normalization
    O = O / l.unsqueeze(1)
    return O

# Key: Never materialize full N×N attention matrix!
# Only compute block_size × block_size tiles in fast SRAM
```

### IO Complexity Analysis

**Standard Attention**:
- HBM reads/writes: O(Nd + N²)
- For large N: O(N²) dominates

**FlashAttention**:
- HBM reads: O(N²d²M⁻¹) where M = SRAM size
- HBM writes: O(Nd)
- For typical M (100KB): O(Nd) dominates
- **Asymptotically better!**

---

## 3. Implementation Details

### CUDA Kernel Structure

```cuda
__global__ void flash_attention_kernel(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int N, int d,
    int block_size
) {
    // Shared memory for blocks
    extern __shared__ half smem[];
    half* Q_smem = smem;
    half* K_smem = smem + block_size * d;
    half* V_smem = K_smem + block_size * d;

    // Thread block processes one Q block across all K blocks
    int q_block_idx = blockIdx.x;
    int q_start = q_block_idx * block_size;

    // Load Q block to shared memory (collaborative load)
    for (int i = threadIdx.x; i < block_size * d; i += blockDim.x) {
        int row = i / d;
        int col = i % d;
        if (q_start + row < N) {
            Q_smem[i] = Q[(q_start + row) * d + col];
        }
    }
    __syncthreads();

    // Initialize output and softmax stats
    float O_local[d] = {0};
    float l_local = 0.0f;
    float m_local = -INFINITY;

    // Loop over K, V blocks
    for (int k_block = 0; k_block < (N + block_size - 1) / block_size; k_block++) {
        int k_start = k_block * block_size;

        // Load K, V blocks to shared memory
        // ... (collaborative loading similar to Q) ...

        // Compute attention for this block
        float S_block[block_size];  // Per-thread scores
        for (int j = 0; j < block_size; j++) {
            float score = 0.0f;
            for (int k = 0; k < d; k++) {
                score += Q_smem[threadIdx.x * d + k] * K_smem[j * d + k];
            }
            S_block[j] = score / sqrtf(d);
        }

        // Online softmax update
        float m_new = m_local;
        for (int j = 0; j < block_size; j++) {
            m_new = fmaxf(m_new, S_block[j]);
        }

        float l_new = 0.0f;
        for (int j = 0; j < block_size; j++) {
            l_new += expf(S_block[j] - m_new);
        }
        l_new += expf(m_local - m_new) * l_local;

        // Update output
        float scale_old = expf(m_local - m_new);
        for (int k = 0; k < d; k++) {
            O_local[k] *= scale_old;
            for (int j = 0; j < block_size; j++) {
                O_local[k] += expf(S_block[j] - m_new) * V_smem[j * d + k];
            }
        }

        m_local = m_new;
        l_local = l_new;
        __syncthreads();
    }

    // Write output (normalized)
    for (int k = 0; k < d; k++) {
        O[(q_start + threadIdx.x) * d + k] = (half)(O_local[k] / l_local);
    }
}
```

---

## 4. Performance Results

### Speed Comparison (A100 GPU)

| Sequence Length | Standard Attention | FlashAttention | Speedup |
|-----------------|-------------------|----------------|---------|
| 512 | 0.8 ms | 0.6 ms | 1.3× |
| 1024 | 2.1 ms | 1.1 ms | 1.9× |
| 2048 | 7.4 ms | 2.8 ms | 2.6× |
| 4096 | 28.1 ms | 8.3 ms | **3.4×** |
| 8192 | OOM | 29.2 ms | **∞** |

**Observation**: Speedup increases with sequence length!

### Memory Comparison

```
Standard Attention (seq_len=4096, head_dim=64):
- Attention matrix: 4096² × 2 bytes = 32 MB per head
- Gradients: 32 MB per head
- Total: 64 MB per head
- 32 heads: 2 GB

FlashAttention:
- No attention matrix storage
- Recompute in backward pass
- Total: ~0.5 MB per head (100× reduction!)
- Enables 16× longer sequences
```

---

## 5. FlashAttention-2 (2023)

### Additional Optimizations

1. **Better parallelism**: Different work partitioning across thread blocks
2. **Reduced non-matmul FLOPs**: Fewer synchronizations
3. **Optimized for different sequence lengths**

**Results**: 2× faster than FlashAttention-1 (4-8× vs standard)

---

## 6. Integration with llama.cpp

### Compilation

```bash
# Enable Flash Attention in CUDA build
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_FLASH_ATTN=ON

cmake --build build --config Release
```

### Usage

```bash
# Flash Attention automatically enabled for long contexts
./llama-cli \
  -m model.gguf \
  -ngl 32 \
  -c 8192 \          # Long context
  -fa \              # Force Flash Attention
  -p "Long prompt..."

# Monitor GPU memory usage
nvidia-smi dmon -s mu
```

### When FlashAttention Helps in llama.cpp

✅ **Best for**:
- Long contexts (>2048 tokens)
- GPU inference (CUDA required)
- Batch processing multiple requests
- Memory-constrained GPUs

❌ **Not needed for**:
- Short contexts (<512 tokens)
- CPU inference
- Single-token generation (KV-cache dominates)

---

## 7. Practical Implications

### For Developers

**Key Lessons**:
1. **IO-awareness matters**: Algorithm design should consider memory hierarchy
2. **Kernel fusion**: Combine operations to reduce memory traffic
3. **Online algorithms**: Incremental computation can avoid materialization
4. **Recomputation trade-offs**: Sometimes cheaper to recompute than to store

### For llama.cpp Users

**When to enable**:
```bash
# Automatic decision based on context length
if context_length > 2048:
    use_flash_attention = True
```

**Performance tips**:
- FlashAttention shines with long contexts (4K-128K tokens)
- Combine with GQA models (LLaMA 2/3) for maximum efficiency
- Monitor GPU memory—FlashAttention enables larger batch sizes

---

## 8. Key Takeaways

### Innovations
1. **Tiling**: Process attention in blocks that fit in SRAM
2. **Online softmax**: Incremental computation without full matrix
3. **IO-optimal**: Asymptotically better memory complexity
4. **Exact**: Not approximate—produces identical results

### Impact
- **2-4× faster** than standard attention
- **O(n) memory** instead of O(n²)
- **Enables 16× longer sequences** on same hardware
- **Widely adopted**: PyTorch, JAX, HuggingFace, llama.cpp

---

## Further Reading

- **Paper**: https://arxiv.org/abs/2205.14135
- **FlashAttention-2**: https://arxiv.org/abs/2307.08691
- **Code**: https://github.com/Dao-AILab/flash-attention
- **Tutorial**: https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad

---

**Status**: Complete | Module 4 (2/3) papers
