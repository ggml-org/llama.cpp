# Transformer Architecture: From Original to Modern Variants

**Paper Collection**: Architecture evolution (2017-2024)
**Key Papers**: Attention Is All You Need, GPT series, LLaMA series
**Relevance**: Module 2 - Understanding LLM Architecture
**Reading Time**: 90-120 minutes (all papers)
**Practical Impact**: ⭐⭐⭐⭐⭐

---

## Executive Summary

The Transformer architecture has evolved from the original encoder-decoder design (2017) to modern decoder-only models optimized for inference. This document traces the architectural innovations from Vaswani et al.'s "Attention Is All You Need" through GPT, BERT, to LLaMA and beyond, focusing on changes that improve inference efficiency.

**Key Evolution**: Encoder-Decoder → Decoder-Only → Optimized Components (RMSNorm, RoPE, SwiGLU) → Efficient Variants (GQA, FlashAttention)

---

## 1. The Original Transformer (2017)

### 1.1 "Attention Is All You Need" - Vaswani et al.

**Paper**: https://arxiv.org/abs/1706.03762
**Impact**: Foundation of all modern LLMs
**Key Innovation**: Self-attention replaces recurrence

### Architecture Overview

```
┌─────────────────────────────────────┐
│         Transformer Block            │
├─────────────────────────────────────┤
│                                     │
│  Input Embeddings                   │
│       ↓                             │
│  Positional Encoding (sinusoidal)   │
│       ↓                             │
│  ┌───────────────────┐              │
│  │   ENCODER (N×)    │              │
│  ├───────────────────┤              │
│  │ Multi-Head Attn   │              │
│  │      ↓            │              │
│  │ Add & Norm        │              │
│  │      ↓            │              │
│  │ Feed Forward      │              │
│  │      ↓            │              │
│  │ Add & Norm        │              │
│  └───────────────────┘              │
│       ↓                             │
│  ┌───────────────────┐              │
│  │   DECODER (N×)    │              │
│  ├───────────────────┤              │
│  │ Masked Self-Attn  │              │
│  │      ↓            │              │
│  │ Add & Norm        │              │
│  │      ↓            │              │
│  │ Cross-Attention   │ ← Encoder    │
│  │      ↓            │              │
│  │ Add & Norm        │              │
│  │      ↓            │              │
│  │ Feed Forward      │              │
│  │      ↓            │              │
│  │ Add & Norm        │              │
│  └───────────────────┘              │
│       ↓                             │
│  Linear + Softmax                   │
│       ↓                             │
│  Output Probabilities               │
└─────────────────────────────────────┘
```

### Core Components

#### 1. Multi-Head Self-Attention

```python
class TransformerSelfAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
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

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # Linear projections and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads and apply final linear
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        output = self.W_o(attn_output)

        return output
```

#### 2. Position-wise Feed-Forward Networks

```python
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

**Key Properties**:
- Applied position-wise (same transformation to each position)
- Two linear transformations with ReLU
- Expansion ratio: d_ff / d_model = 4× (typically)

#### 3. Layer Normalization

```python
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

**Why LayerNorm?**
- Stabilizes training (reduces internal covariate shift)
- Allows deeper networks
- Faster convergence than BatchNorm for sequential data

#### 4. Positional Encoding (Sinusoidal)

```python
def sinusoidal_positional_encoding(max_len, d_model):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe
```

**Properties**:
- Deterministic (no learned parameters)
- Allows relative position attention (PE(pos+k) is linear function of PE(pos))
- Generalizes to longer sequences than seen during training

---

### 1.2 Complete Transformer Block

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = TransformerSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x
```

**Design Patterns**:
- Residual connections: Enable training deep networks (100+ layers)
- Pre-normalization (modern): Move LayerNorm before sub-layer (more stable)
- Post-normalization (original): LayerNorm after sub-layer

---

## 2. Decoder-Only Transformers (GPT Series)

### 2.1 GPT-1 (2018) - "Improving Language Understanding by Generative Pre-Training"

**Paper**: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
**Key Innovation**: Unsupervised pre-training + supervised fine-tuning

**Architecture Change**: Remove encoder, use only decoder

```python
class GPTDecoderLayer(nn.Module):
    def __init__(self, d_model=768, num_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.masked_self_attn = TransformerSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Causal (masked) self-attention
        batch_size, seq_len = x.shape[:2]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

        attn_output = self.masked_self_attn(x, mask=causal_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x
```

**Causal Masking**:
```python
# Prevents attending to future tokens
causal_mask = [
    [1, 0, 0, 0],  # Token 0 sees only token 0
    [1, 1, 0, 0],  # Token 1 sees tokens 0-1
    [1, 1, 1, 0],  # Token 2 sees tokens 0-2
    [1, 1, 1, 1],  # Token 3 sees tokens 0-3
]
```

**Training Objective**:
```python
def gpt_loss(model, tokens):
    """Next-token prediction loss"""
    logits = model(tokens[:, :-1])  # Input: all but last token
    targets = tokens[:, 1:]          # Target: all but first token
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    return loss
```

---

### 2.2 GPT-2 (2019) - Scaling and Few-Shot Learning

**Paper**: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
**Key Innovations**:
- Scaled to 1.5B parameters
- Demonstrated in-context learning
- Pre-normalization (move LayerNorm before attention)

**Pre-Normalization Architecture**:
```python
class GPT2DecoderLayer(nn.Module):
    def __init__(self, d_model=768, num_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.self_attn = TransformerSelfAttention(d_model, num_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-normalization: norm before attention
        x = x + self.dropout1(self.self_attn(self.norm1(x), mask))
        x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        return x
```

**Why Pre-Normalization?**
- More stable training for deep networks
- Better gradient flow
- Allows learning rate warm-up to be optional

---

### 2.3 GPT-3 (2020) - Massive Scale

**Paper**: https://arxiv.org/abs/2005.14165
**Key Innovation**: Scale to 175B parameters, emergent abilities

**Architecture**: Same as GPT-2, but scaled:
- Layers: 96
- Hidden size: 12,288
- Attention heads: 96
- Context window: 2048 tokens

**Scaling Laws** (Kaplan et al., 2020):
```
Loss ∝ N^(-α) × D^(-β) × C^(-γ)

Where:
N = number of parameters
D = dataset size
C = compute budget

Optimal: N ∝ C^0.73, D ∝ C^0.27
```

**Implication**: Model size should grow faster than data

---

## 3. LLaMA: Optimized for Inference

### 3.1 LLaMA 1 (2023) - Meta's Open Foundation

**Paper**: https://arxiv.org/abs/2302.13971
**Key Innovations**:
1. **Pre-normalization** (from GPT-3)
2. **RMSNorm** instead of LayerNorm
3. **SwiGLU activation** instead of ReLU
4. **RoPE** instead of absolute positional embeddings

### LLaMA Architecture

```python
class LLaMADecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=4096,
        num_heads=32,
        d_ff=11008,  # SwiGLU requires 2/3 × 4 × d_model
        rope_theta=10000.0,
        norm_eps=1e-5
    ):
        super().__init__()
        self.attention_norm = RMSNorm(d_model, eps=norm_eps)
        self.attention = LLaMAAttention(d_model, num_heads, rope_theta)
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.feed_forward = SwiGLU(d_model, d_ff)

    def forward(self, x, freqs_cos, freqs_sin, mask=None):
        # Attention block with RMSNorm
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cos, freqs_sin,
            mask
        )

        # Feed-forward block with RMSNorm
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
```

#### Innovation 1: RMSNorm

**Motivation**: LayerNorm computes mean AND variance, but mean is often unnecessary

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # LayerNorm: normalize by mean and std
        # RMSNorm: normalize by RMS only
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed

# Comparison
def layer_norm(x):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True)
    return (x - mean) / sqrt(var + eps)  # 3 ops: mean, var, divide

def rms_norm(x):
    rms = sqrt(mean(x**2) + eps)
    return x / rms  # 2 ops: mean of squares, divide
```

**Speedup**: 7-64% faster depending on hardware
**Quality**: Comparable or slightly better than LayerNorm

#### Innovation 2: SwiGLU Activation

**Standard FFN** (GPT-style):
```python
def ffn_relu(x):
    return W2(relu(W1(x)))
```

**SwiGLU FFN** (LLaMA):
```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Up projection

    def forward(self, x):
        # SwiGLU(x) = (Swish(xW1) ⊙ xW3)W2
        # Swish(x) = x * sigmoid(x)
        gate = F.silu(self.w1(x))  # SiLU = Swish
        x = gate * self.w3(x)       # Element-wise multiplication (gating)
        x = self.w2(x)
        return x
```

**Why Gating?**
- Allows network to dynamically control information flow
- Swish activation is smooth (better gradients than ReLU)
- Empirically improves quality

**Parameter Cost**:
- Standard FFN: 2 × d_model × d_ff
- SwiGLU: 3 × d_model × d_ff (but d_ff is smaller: 8/3 × d_model vs 4 × d_model)
- Net: ~Same parameter count, better quality

#### Innovation 3: RoPE (Rotary Position Embedding)

**Problem with Sinusoidal PE**: Added to input, position info can be lost in deep layers

**RoPE Solution**: Apply rotary matrix to Q, K directly in attention

```python
def apply_rope(q, k, cos, sin):
    """
    Apply rotary position embedding

    q, k: [batch, heads, seq_len, head_dim]
    cos, sin: [seq_len, head_dim] - precomputed rotation frequencies
    """
    # Rotate q and k
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def rotate_half(x):
    """Rotate half the hidden dims"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

# Precompute rotation frequencies
def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin
```

**Advantages**:
- Encodes absolute position with relative distance properties
- Decays naturally with distance (long-range attention weakens)
- Extrapolates to longer contexts than training length
- No learned parameters

**llama.cpp Implementation**:
```cpp
// ggml-quant.c - RoPE implementation
void ggml_rope_inplace(
    struct ggml_tensor * a,
    int n_past,
    int n_dims,
    int mode,
    int n_ctx
) {
    // Apply rotary position embeddings in-place
    // Optimized for CPU/GPU backends
}
```

---

### 3.2 LLaMA 2 (2023) - Production Ready

**Paper**: https://arxiv.org/abs/2307.09288
**Key Innovations**:
1. **Grouped-Query Attention (GQA)** - reduces KV-cache by 4×
2. **Context length**: 2048 → 4096 tokens
3. **Improved training** (2T tokens vs 1.4T)

**GQA Architecture**:
```python
class LLaMA2Attention(nn.Module):
    def __init__(
        self,
        d_model=4096,
        num_q_heads=32,
        num_kv_heads=8,  # GQA: fewer KV heads
        head_dim=128
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_q_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin, mask=None, cache=None):
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q, k = apply_rope(q, k, freqs_cos, freqs_sin)

        # Repeat K, V to match Q heads (for GQA)
        k = k.repeat_interleave(self.num_kv_groups, dim=2)
        v = v.repeat_interleave(self.num_kv_groups, dim=2)

        # Standard attention
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)
```

**Memory Savings**:
```
LLaMA 2 7B (4096 context):
- MHA (32 KV heads): 512 MB KV-cache
- GQA (8 KV heads):  128 MB KV-cache (4× reduction)
- Quality loss: <1% on benchmarks
```

---

### 3.3 LLaMA 3 (2024) - Extended Context

**Paper**: https://arxiv.org/abs/2407.21783
**Key Innovations**:
1. **128K context** with RoPE scaling
2. **Improved tokenizer** (128K vocabulary)
3. **Better data quality**

**RoPE Scaling** for long context:
```python
def apply_scaled_rope(q, k, cos, sin, scale=1.0):
    """
    NTK-aware RoPE scaling for extended context
    """
    # Adjust rotation frequencies
    cos_scaled = cos * scale
    sin_scaled = sin * scale
    return apply_rope(q, k, cos_scaled, sin_scaled)

# llama.cpp RoPE scaling
--rope-freq-base 10000    # Original
--rope-freq-base 1000000  # For 128K context (100× scaling)
```

---

## 4. Architecture Comparison Table

| Model | Layers | d_model | Heads | Context | Norm | Activation | Positional | KV Heads |
|-------|--------|---------|-------|---------|------|------------|------------|----------|
| Transformer | 6 | 512 | 8 | 512 | LayerNorm | ReLU | Sinusoidal | 8 |
| GPT-1 | 12 | 768 | 12 | 512 | LayerNorm | GELU | Learned | 12 |
| GPT-2 | 48 | 1600 | 25 | 1024 | LayerNorm | GELU | Learned | 25 |
| GPT-3 | 96 | 12288 | 96 | 2048 | LayerNorm | GELU | Learned | 96 |
| LLaMA 1 7B | 32 | 4096 | 32 | 2048 | RMSNorm | SwiGLU | RoPE | 32 |
| LLaMA 2 7B | 32 | 4096 | 32 | 4096 | RMSNorm | SwiGLU | RoPE | 8 (GQA) |
| LLaMA 3 8B | 32 | 4096 | 32 | 128K | RMSNorm | SwiGLU | RoPE | 8 (GQA) |

---

## 5. Practical Implications for llama.cpp

### 5.1 Architecture Detection

```python
from gguf import GGUFReader

def analyze_architecture(model_path):
    reader = GGUFReader(model_path)

    arch = {
        'n_layers': reader.fields['llama.block_count'],
        'd_model': reader.fields['llama.embedding_length'],
        'n_heads': reader.fields['llama.attention.head_count'],
        'n_kv_heads': reader.fields.get('llama.attention.head_count_kv'),
        'context_length': reader.fields['llama.context_length'],
        'rope_theta': reader.fields.get('llama.rope.freq_base', 10000.0),
        'norm_type': 'RMSNorm',  # All LLaMA models
    }

    # Infer model generation
    if arch['n_kv_heads'] is None or arch['n_kv_heads'] == arch['n_heads']:
        generation = "LLaMA 1 (MHA)"
    elif arch['context_length'] >= 100000:
        generation = "LLaMA 3 (GQA, long context)"
    else:
        generation = "LLaMA 2 (GQA)"

    print(f"Architecture: {generation}")
    for key, value in arch.items():
        print(f"  {key}: {value}")

    return arch
```

### 5.2 Inference Optimization Based on Architecture

```python
def optimize_inference_params(arch):
    """Suggest optimal inference parameters based on architecture"""

    # Context length recommendations
    if arch['context_length'] >= 100000:
        print("Long-context model detected:")
        print("  - Consider using --ctx-size 8192 or higher")
        print("  - Enable RoPE scaling: --rope-freq-scale 4.0")
        print("  - Flash Attention recommended for GPU")

    # GQA models
    if arch['n_kv_heads'] and arch['n_kv_heads'] < arch['n_heads']:
        kv_ratio = arch['n_heads'] / arch['n_kv_heads']
        print(f"GQA model (KV sharing ratio: {kv_ratio}:1)")
        print("  - KV-cache is reduced, can increase batch size")
        print(f"  - Suggested -b/--batch-size: {int(8 * kv_ratio)}")

    # Memory estimates
    bytes_per_param = 0.5  # Q4_K_M quantization
    model_size_gb = arch['n_layers'] * arch['d_model']**2 * 8 * bytes_per_param / 1e9
    print(f"Estimated model size (Q4_K_M): {model_size_gb:.1f} GB")
```

---

## 6. Key Takeaways

### Evolution Summary

```
Original Transformer (2017)
    ↓
├─ Encoder-only: BERT (2018)
│  └─ Use: Classification, QA, NER
│
└─ Decoder-only: GPT (2018)
       ↓
   GPT-2 (2019) - Pre-normalization
       ↓
   GPT-3 (2020) - Scaling laws
       ↓
   LLaMA 1 (2023) - Inference optimizations
       ↓
   LLaMA 2 (2023) - GQA for efficiency
       ↓
   LLaMA 3 (2024) - Long context
```

### Must-Know Innovations

1. **Pre-normalization**: Norm before attention/FFN (not after)
2. **RMSNorm**: Faster than LayerNorm, same quality
3. **SwiGLU**: Better than ReLU/GELU for language
4. **RoPE**: Better positional encoding than learned/sinusoidal
5. **GQA**: Best quality/memory trade-off for inference

### For llama.cpp Users

✅ **Model Selection**:
- LLaMA 1: Good quality, higher memory usage
- LLaMA 2: Best balance (GQA reduces KV-cache)
- LLaMA 3: Long context capability

✅ **Optimization**:
- GQA models: Increase batch size
- Long context: Enable RoPE scaling
- GPU: Use Flash Attention

✅ **Debugging**:
- Check architecture with GGUF metadata
- Verify RoPE parameters for context extension
- Monitor KV-cache usage

---

## 7. Further Reading

### Essential Papers (in order)
1. Attention Is All You Need (2017) - https://arxiv.org/abs/1706.03762
2. GPT-1 (2018) - https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
3. GPT-3 (2020) - https://arxiv.org/abs/2005.14165
4. LLaMA 1 (2023) - https://arxiv.org/abs/2302.13971
5. LLaMA 2 (2023) - https://arxiv.org/abs/2307.09288

### Component Papers
- RMSNorm: https://arxiv.org/abs/1910.07467
- SwiGLU: https://arxiv.org/abs/2002.05202
- RoPE: https://arxiv.org/abs/2104.09864

---

**Document Information**
- Created: 2025-11-18
- Module: 2 - Understanding LLM Architecture
- Author: Research Coordinator
- Status: Complete
- Next: Read tokenization-algorithms.md
