# Model Architecture Deep Dive

**Learning Module**: Module 2 - Core Implementation
**Estimated Reading Time**: 30 minutes
**Prerequisites**: Module 1 complete, understanding of neural networks
**Related Content**:
- [Tokenization and Vocabulary](./02-tokenization-and-vocabulary.md)
- [KV Cache Implementation](./03-kv-cache-implementation.md)
- [Inference Pipeline](./04-inference-pipeline.md)

---

## Overview

This document provides a comprehensive deep dive into transformer architecture as implemented in llama.cpp, covering the core components, mathematical foundations, and code-level details of modern LLM architectures.

### Learning Objectives

After completing this lesson, you will:
- ✅ Understand the transformer architecture from first principles
- ✅ Know how LLaMA, Mistral, and other models differ architecturally
- ✅ Read and interpret architecture definitions in llama.cpp code
- ✅ Identify performance characteristics of different architectural choices
- ✅ Debug architecture-related issues

---

## Transformer Architecture Fundamentals

### The Original Transformer

The transformer architecture, introduced in "Attention is All You Need" (Vaswani et al., 2017), revolutionized NLP by replacing recurrence with self-attention.

```
┌─────────────────────────────────────────────────────┐
│            Transformer Architecture                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Input → Embedding → Positional Encoding            │
│                           ↓                          │
│  ┌─────────────────────────────────────────────┐   │
│  │  Transformer Block (repeated N times)       │   │
│  │                                              │   │
│  │  1. Multi-Head Self-Attention                │   │
│  │     - Query, Key, Value projections          │   │
│  │     - Scaled dot-product attention           │   │
│  │     - Multi-head concatenation               │   │
│  │  2. Layer Normalization                      │   │
│  │  3. Feed-Forward Network                     │   │
│  │     - 2-layer MLP with activation            │   │
│  │  4. Layer Normalization                      │   │
│  │  5. Residual Connections                     │   │
│  └─────────────────────────────────────────────┘   │
│                           ↓                          │
│  Final Layer Norm → Output Projection → Logits      │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Decoder-Only Architecture

Modern LLMs (LLaMA, GPT, Mistral) use decoder-only transformers optimized for autoregressive generation:

**Key Differences from Original Transformer**:
1. **Causal Masking**: Attention only to previous tokens (autoregressive)
2. **No Encoder**: Single stack of decoder layers
3. **Pre-normalization**: Layer norm before attention/FFN (not after)
4. **Rotary Position Embeddings (RoPE)**: Instead of absolute positional encodings

---

## Core Components

### 1. Token Embedding

Converts input token IDs to dense vectors:

```python
# Conceptual implementation
embedding_table: (vocab_size, embedding_dim)
input_tokens: (batch, seq_len)
embeddings = embedding_table[input_tokens]  # (batch, seq_len, embedding_dim)
```

**In llama.cpp**:
```cpp
// From llama.cpp
struct llama_layer {
    // Token embeddings
    struct ggml_tensor * tok_embd;  // (n_vocab, n_embd)
    // ...
};
```

**Key Parameters**:
- `n_vocab`: Vocabulary size (typically 32,000 - 128,000)
- `n_embd`: Embedding dimension (hidden size)
  - 7B models: 4096
  - 13B models: 5120
  - 70B models: 8192

### 2. Self-Attention Mechanism

The core innovation enabling transformers to capture long-range dependencies.

#### Mathematical Foundation

**Scaled Dot-Product Attention**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
  Q = Query matrix  (seq_len, d_k)
  K = Key matrix    (seq_len, d_k)
  V = Value matrix  (seq_len, d_v)
  d_k = dimension of keys (for scaling)
```

**Why Scale by √d_k?**
- Prevents dot products from becoming too large
- Keeps softmax gradients stable
- Critical for training stability

#### Multi-Head Attention (MHA)

Allows the model to attend to different representation subspaces:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

Where:
  head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

Parameters:
  W^Q_i, W^K_i, W^V_i: projection matrices for head i
  W^O: output projection matrix
  h: number of heads
```

**Example Configuration (LLaMA-7B)**:
```
n_embd = 4096      # Hidden size
n_head = 32        # Number of attention heads
head_dim = 128     # 4096 / 32 = 128 dimensions per head
```

#### Grouped Query Attention (GQA)

Modern optimization used in LLaMA-2 and Mistral:

```
┌─────────────────────────────────────────────────────┐
│         Multi-Head Attention vs GQA                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  MHA (Traditional):                                  │
│  Q: 32 heads → K: 32 heads → V: 32 heads           │
│                                                      │
│  GQA (n_kv_heads = 8):                              │
│  Q: 32 heads → K: 8 heads → V: 8 heads             │
│                 (each K/V used by 4 Q heads)        │
│                                                      │
│  Multi-Query Attention (MQA):                        │
│  Q: 32 heads → K: 1 head → V: 1 head               │
│                 (single K/V for all Q)              │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Benefits of GQA**:
- Reduced KV cache memory (critical for long contexts)
- Faster inference (less data to load)
- Minimal quality degradation vs MHA

**In llama.cpp**:
```cpp
// From llama-model.cpp
const uint32_t n_head    = hparams.n_head;       // Query heads
const uint32_t n_head_kv = hparams.n_head_kv;    // KV heads (GQA)
const uint32_t n_embd_head = hparams.n_embd_head(); // Per-head dimension
```

### 3. Rotary Position Embeddings (RoPE)

RoPE encodes position information by rotating query/key vectors:

**Key Concept**:
Instead of adding position embeddings, RoPE multiplies Q and K by rotation matrices based on position.

```python
# Simplified RoPE implementation
def apply_rope(x, position):
    """
    x: (batch, seq_len, n_heads, head_dim)
    position: token position in sequence
    """
    # Create rotation matrix based on position
    theta = 10000 ** (-2 * (torch.arange(0, head_dim, 2) / head_dim))
    m_theta = position * theta

    # Apply rotation to pairs of dimensions
    cos = torch.cos(m_theta)
    sin = torch.sin(m_theta)

    # Rotate: [x0, x1, x2, x3, ...] → [x0*cos - x1*sin, x0*sin + x1*cos, ...]
    x_rotated = rotate_half(x) * sin + x * cos
    return x_rotated
```

**Advantages**:
- Relative position encoding (distance matters, not absolute position)
- Extrapolates to longer sequences
- No learned parameters
- Efficient implementation

**RoPE Scaling** (for extended context):
- **Linear Scaling**: Divide position by scaling factor
- **NTK-Aware Scaling**: Modify base frequency (10000 → larger value)
- **YaRN**: Advanced scaling method for extreme lengths

### 4. Feed-Forward Network (FFN)

Two-layer MLP that processes each position independently:

```python
# Standard FFN
def ffn(x):
    # x: (batch, seq_len, n_embd)
    hidden = activation(x @ W1 + b1)  # (batch, seq_len, n_ff)
    output = hidden @ W2 + b2          # (batch, seq_len, n_embd)
    return output
```

**Typical Configuration**:
```
n_embd = 4096
n_ff = 4 * n_embd = 16384  # "Intermediate size"
```

**SwiGLU Activation** (used in LLaMA):
```python
def swiglu(x, gate):
    """
    SwiGLU: Swish-Gated Linear Unit
    More expressive than ReLU/GELU
    """
    return swish(gate) * x

def ffn_swiglu(x):
    gate = x @ W_gate
    up = x @ W_up
    return swiglu(up, gate) @ W_down
```

**Why SwiGLU?**
- Better performance than GELU
- Smoother gradients
- Used in most modern LLMs

### 5. Layer Normalization

Normalizes activations across the embedding dimension:

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """
    x: (batch, seq_len, n_embd)
    gamma, beta: learnable parameters (n_embd,)
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / sqrt(var + eps)
    return gamma * x_norm + beta
```

**RMSNorm** (used in LLaMA):
```python
def rms_norm(x, gamma, eps=1e-6):
    """
    Root Mean Square Normalization
    Simpler and faster than LayerNorm
    """
    rms = sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return gamma * (x / rms)
```

**Benefits of RMSNorm**:
- No mean subtraction (faster)
- No beta parameter (fewer params)
- Empirically equivalent performance

### 6. Residual Connections

Enable gradient flow through deep networks:

```python
# Pre-norm architecture (modern standard)
def transformer_block(x):
    # Attention with residual
    x = x + attention(layer_norm(x))

    # FFN with residual
    x = x + ffn(layer_norm(x))

    return x
```

**Pre-Norm vs Post-Norm**:
```
Pre-Norm (LLaMA, GPT-3):              Post-Norm (Original Transformer):
x → LN → Attn → (+)                   x → Attn → (+) → LN
     ↓         ↗                           ↓      ↗
     └─────────┘                           └──────┘

Benefits: Better training stability   Benefits: Slightly better performance
```

---

## Model Architectures in llama.cpp

### LLaMA / LLaMA-2 / LLaMA-3

**Architecture Highlights**:
- Pre-normalization with RMSNorm
- SwiGLU activation
- RoPE positional embeddings
- Multi-head attention (LLaMA-1) or GQA (LLaMA-2/3)

**Model Sizes**:
```
Model       Params   Layers  Hidden  Heads  Context
─────────────────────────────────────────────────────
LLaMA-7B    7B       32      4096    32     4096
LLaMA-13B   13B      40      5120    40     4096
LLaMA-30B   30B      60      6656    52     4096
LLaMA-65B   65B      80      8192    64     4096
LLaMA2-70B  70B      80      8192    64     4096
LLaMA3-8B   8B       32      4096    32     8192
LLaMA3-70B  70B      80      8192    64     8192
```

**Code Representation**:
```cpp
// llama-model.cpp
case LLM_ARCH_LLAMA:
    {
        ml.get_key(LLM_KV_ATTENTION_HEAD_COUNT,    hparams.n_head);
        ml.get_key(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv);
        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT,    hparams.n_rot);
        // ...

        layer.attn_norm   = ml.create_tensor("blk.%d.attn_norm", i);
        layer.attn_q      = ml.create_tensor("blk.%d.attn_q", i);
        layer.attn_k      = ml.create_tensor("blk.%d.attn_k", i);
        layer.attn_v      = ml.create_tensor("blk.%d.attn_v", i);
        layer.attn_output = ml.create_tensor("blk.%d.attn_output", i);

        layer.ffn_norm    = ml.create_tensor("blk.%d.ffn_norm", i);
        layer.ffn_gate    = ml.create_tensor("blk.%d.ffn_gate", i);
        layer.ffn_up      = ml.create_tensor("blk.%d.ffn_up", i);
        layer.ffn_down    = ml.create_tensor("blk.%d.ffn_down", i);
    }
```

### Mistral

**Architecture Innovations**:
- Sliding Window Attention (SWA): 4096 token window
- GQA: 8 KV heads for 32 query heads
- Rolling buffer for KV cache

**Sliding Window Attention**:
```
┌─────────────────────────────────────────────────────┐
│          Sliding Window Attention                    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Token positions: [0, 1, 2, 3, 4, 5, 6, 7, 8, ...]  │
│                                                      │
│  Token 8 attends to: [4, 5, 6, 7, 8]               │
│                      └─ window_size = 4 ─┘         │
│                                                      │
│  Benefits:                                           │
│  - Constant memory per layer                        │
│  - Still sees full context via stacking             │
│  - Faster attention computation                     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Mixtral (Mixture of Experts)

**MoE Architecture**:
- 8 experts per layer
- Top-2 routing (activate 2 of 8 experts)
- 46.7B total params, 12.9B active per token

```
┌─────────────────────────────────────────────────────┐
│         Mixture of Experts (MoE) Layer               │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Input → Router (learned gating)                    │
│           ↓                                          │
│     [Expert 0] [Expert 1] ... [Expert 7]           │
│         ↓           ↓              ↓                │
│     (Select top-2 experts per token)                │
│         ↓           ↓                                │
│     weight_0    weight_1                            │
│         ↓           ↓                                │
│     Weighted sum → Output                           │
│                                                      │
│  Benefits:                                           │
│  - Larger capacity without proportional compute     │
│  - Specialized experts for different patterns       │
│  - Better performance per active parameter          │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**In llama.cpp**:
```cpp
case LLM_ARCH_MIXTRAL:
    {
        // MoE-specific parameters
        ml.get_key(LLM_KV_EXPERT_COUNT,      hparams.n_expert);
        ml.get_key(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used);

        // Router weights
        layer.ffn_gate_inp = ml.create_tensor("blk.%d.ffn_gate_inp", i);

        // Expert weights (n_expert copies of FFN)
        for (int e = 0; e < hparams.n_expert; ++e) {
            layer.ffn_gate_exp[e] = ml.create_tensor("blk.%d.ffn_gate.%d", i, e);
            layer.ffn_up_exp[e]   = ml.create_tensor("blk.%d.ffn_up.%d", i, e);
            layer.ffn_down_exp[e] = ml.create_tensor("blk.%d.ffn_down.%d", i, e);
        }
    }
```

### Qwen / Phi / Gemma

Each model family has unique architectural choices:

**Qwen**:
- Uses learned absolute position embeddings (not RoPE)
- GELU activation
- Bias terms in attention projections

**Phi**:
- Partial RoPE (not applied to all dimensions)
- Dense architecture (fewer but wider layers)

**Gemma**:
- Normalizer-free architecture variants
- Specific attention masking patterns

---

## Architecture Parameters in GGUF

Model architecture is stored in GGUF metadata:

```python
# Reading architecture from GGUF
def read_architecture(gguf_file):
    metadata = {
        # Model type
        'general.architecture': 'llama',  # or 'mistral', 'mixtral', etc.
        'general.name': 'LLaMA v2',

        # Core dimensions
        'llama.embedding_length': 4096,     # n_embd
        'llama.block_count': 32,            # n_layer
        'llama.context_length': 4096,       # n_ctx_train

        # Attention configuration
        'llama.attention.head_count': 32,        # n_head
        'llama.attention.head_count_kv': 8,      # n_head_kv (GQA)
        'llama.attention.layer_norm_rms_epsilon': 1e-5,

        # FFN configuration
        'llama.feed_forward_length': 11008,  # n_ff (intermediate size)

        # RoPE configuration
        'llama.rope.dimension_count': 128,   # n_rot
        'llama.rope.freq_base': 10000.0,     # theta

        # Vocabulary
        'llama.vocab_size': 32000,
    }
    return metadata
```

**Key Metadata Fields**:

| Field | Description | Typical Values |
|-------|-------------|----------------|
| `block_count` | Number of layers | 32 (7B), 40 (13B), 80 (70B) |
| `embedding_length` | Hidden dimension | 4096, 5120, 8192 |
| `attention.head_count` | Query heads | 32, 40, 64 |
| `attention.head_count_kv` | KV heads (GQA) | 1 (MQA), 8 (GQA), 32 (MHA) |
| `feed_forward_length` | FFN intermediate | 4 × hidden_dim |
| `rope.freq_base` | RoPE base frequency | 10000.0 (standard), 1000000.0 (extended) |

---

## Memory Layout and Computation

### Tensor Shapes

Understanding tensor dimensions is critical for optimization:

```python
# Forward pass tensor shapes (batch_size=1, seq_len=1 for simplicity)

# 1. Token embedding
input_ids: (1, 1)                    # Single token
embeddings: (1, 1, n_embd)           # (batch, seq, hidden)

# 2. Attention layer
x: (1, 1, n_embd)                    # Input to attention

# Query, Key, Value projections
q: (1, 1, n_head, head_dim)          # (batch, seq, heads, dim_per_head)
k: (1, 1, n_head_kv, head_dim)       # Fewer KV heads (GQA)
v: (1, 1, n_head_kv, head_dim)

# Attention computation (with KV cache)
k_cache: (1, past_seq_len, n_head_kv, head_dim)  # Cached keys
v_cache: (1, past_seq_len, n_head_kv, head_dim)  # Cached values

# Attention scores
scores: (1, n_head, 1, past_seq_len + 1)         # Q @ K^T
attn_output: (1, 1, n_embd)                       # After weighted sum

# 3. FFN layer
ffn_input: (1, 1, n_embd)
ffn_gate: (1, 1, n_ff)               # Gate projection
ffn_up: (1, 1, n_ff)                 # Up projection
ffn_hidden: (1, 1, n_ff)             # After SwiGLU
ffn_output: (1, 1, n_embd)           # Down projection

# 4. Output
logits: (1, 1, vocab_size)           # Final predictions
```

### Computation Patterns

**Attention Computation** (most expensive):
```
FLOPs per token ≈ 2 * n_layer * (
    # QKV projections
    3 * n_embd^2 +
    # Attention scores
    n_head * seq_len * head_dim +
    # Output projection
    n_embd^2
)

For LLaMA-7B at seq_len=1 (prefill done):
  ≈ 2 * 32 * (3 * 4096^2 + 32 * 1 * 128 + 4096^2)
  ≈ 7 billion FLOPs per token
```

**FFN Computation**:
```
FLOPs per token ≈ 2 * n_layer * (
    # Gate and Up projections
    2 * n_embd * n_ff +
    # Down projection
    n_ff * n_embd
)

For LLaMA-7B:
  ≈ 2 * 32 * (2 * 4096 * 11008 + 11008 * 4096)
  ≈ 11 billion FLOPs per token
```

**Total**: ~18 billion FLOPs per token for 7B model

---

## Performance Considerations

### Architectural Choices Impact

**Multi-Head Attention vs GQA**:
```
MHA (32 KV heads):
  KV Cache Size: 2 * n_layer * seq_len * n_head * head_dim * bytes_per_element
  = 2 * 32 * 2048 * 32 * 128 * 2 bytes
  = 1 GB for 2K context (FP16)

GQA (8 KV heads):
  KV Cache Size: 2 * 32 * 2048 * 8 * 128 * 2 bytes
  = 256 MB for 2K context (FP16)

Memory Savings: 4x smaller KV cache!
```

**Sliding Window Attention** (Mistral):
- Constant memory per layer (4096 token window)
- But still sees full context through layer stacking
- Trade-off: Slightly reduced long-range attention

**Mixture of Experts**:
- Larger model capacity (46.7B total for Mixtral)
- But only 12.9B active per token
- Inference cost: Similar to 13B dense model
- Quality: Better than 13B, approaching 70B

---

## Inspecting Architecture in llama.cpp

### Using llama-cli

```bash
# View model architecture
llama-cli -m model.gguf --verbose-prompt

# Output includes:
# llm_load_print_meta: n_vocab    = 32000
# llm_load_print_meta: n_ctx      = 4096
# llm_load_print_meta: n_embd     = 4096
# llm_load_print_meta: n_head     = 32
# llm_load_print_meta: n_head_kv  = 8
# llm_load_print_meta: n_layer    = 32
# llm_load_print_meta: n_ff       = 11008
```

### Programmatic Access

```python
# Using llama-cpp-python
from llama_cpp import Llama

llm = Llama(model_path="model.gguf")

# Access model metadata
n_vocab = llm.n_vocab()
n_ctx = llm.n_ctx()
n_embd = llm.n_embd()

print(f"Vocabulary size: {n_vocab}")
print(f"Context length: {n_ctx}")
print(f"Embedding dimension: {n_embd}")
```

### Reading GGUF Metadata

```python
# Direct GGUF reading
import gguf

reader = gguf.GGUFReader("model.gguf")

# Iterate through metadata
for key, value in reader.fields.items():
    if 'attention' in key or 'embedding' in key:
        print(f"{key}: {value}")

# Example output:
# llama.attention.head_count: 32
# llama.attention.head_count_kv: 8
# llama.embedding_length: 4096
```

---

## Debugging Architecture Issues

### Common Problems

**1. Shape Mismatches**

```bash
# Error: Tensor shape mismatch
ggml_mul: incompatible dimensions (4096, 11008) and (4096, 14336)

# Cause: Model metadata doesn't match tensor dimensions
# Solution: Verify n_ff parameter in GGUF metadata
```

**2. Attention Configuration**

```bash
# Error: Invalid head configuration
n_head (32) must be divisible by n_head_kv (12)

# Cause: GQA requires n_head % n_head_kv == 0
# Solution: Check model conversion, ensure correct n_head_kv
```

**3. Context Length Issues**

```bash
# Error: Context length exceeded
requested context (8192) > model trained context (4096)

# Solution: Use RoPE scaling or model fine-tuned for longer context
```

### Validation Checks

```cpp
// llama.cpp performs these validations
bool llama_model_check(const llama_model & model) {
    const auto & hparams = model.hparams;

    // Check head configuration
    if (hparams.n_head % hparams.n_head_kv != 0) {
        return false;
    }

    // Check embedding divisibility
    if (hparams.n_embd % hparams.n_head != 0) {
        return false;
    }

    // Check FFN dimension
    if (hparams.n_ff == 0) {
        return false;
    }

    return true;
}
```

---

## Interview Questions

**Q1: Explain the difference between Multi-Head Attention (MHA) and Grouped Query Attention (GQA). Why is GQA used in modern LLMs?**

**Answer**: MHA uses separate key/value projections for each attention head, while GQA shares key/value projections across multiple query heads. For example, with 32 query heads and 8 KV heads, each KV head is used by 4 query heads.

Benefits:
- 4x smaller KV cache (critical for long contexts)
- Faster inference (less memory bandwidth)
- Minimal quality loss (~1-2% vs MHA)
- Enables longer contexts within same memory budget

**Q2: What is RoPE and why is it preferred over absolute position embeddings?**

**Answer**: Rotary Position Embedding (RoPE) encodes position by rotating Q/K vectors in complex space. Benefits:
- Encodes relative positions naturally (distance-aware)
- No learned parameters (purely algorithmic)
- Extrapolates to longer sequences
- Better long-range attention
- Used in LLaMA, Mistral, most modern LLMs

**Q3: How does Mixtral's Mixture of Experts architecture achieve better efficiency?**

**Answer**: Mixtral uses sparse activation:
- 8 experts per layer, only 2 activated per token
- Total params: 46.7B, active per token: 12.9B
- Achieves ~70B model quality at ~13B inference cost
- Each expert specializes in different patterns
- Router learns which experts to activate

---

## Further Reading

### Code References
- [llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp): Model loading
- [llama-arch.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp): Architecture definitions
- [ggml-attention.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/ggml/src/ggml-attention.cpp): Attention implementations

### Research Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): Original Transformer
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Mistral 7B](https://arxiv.org/abs/2310.06825): Sliding window attention
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088): MoE architecture
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245)

### Tutorials
- [Lab 1: Model Architecture Exploration](../labs/lab-01-architecture-exploration.ipynb)
- [Tutorial: Understanding Transformer Layers](../tutorials/tutorial-01-transformer-layers.ipynb)
- [Code Example: Architecture Inspector](../code/architecture_inspector.py)

---

**Last Updated**: 2025-11-18
**Author**: Agent 5 (Documentation Writer)
**Reviewed By**: Agent 7 (Quality Validator)
**Module**: 2 - Core Implementation
