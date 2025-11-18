# Module 2: LLM Architecture - Curated Reading List

**Curator**: Research Coordinator
**Module**: Module 2 - Understanding LLM Architecture
**Last Updated**: 2025-11-18
**Target Audience**: Engineers deepening LLM architecture knowledge
**Total Resources**: 12 curated items
**Estimated Study Time**: 10-14 hours

---

## Learning Objectives

By completing this reading list, you will:
- ✅ Understand transformer architecture evolution from 2017 to present
- ✅ Master attention mechanisms (standard, multi-query, grouped-query, Flash)
- ✅ Explain LLaMA-specific optimizations (RMSNorm, RoPE, SwiGLU, GQA)
- ✅ Comprehend tokenization algorithms (BPE, WordPiece, SentencePiece)
- ✅ Apply architectural knowledge to llama.cpp optimization

---

## Essential Papers

### 1. Attention Mechanisms Survey ⭐⭐⭐

**Our Summary**: `/learning-materials/papers/summaries/attention-mechanisms-survey.md`
**Reading Time**: 45-60 minutes
**Priority**: Critical

**What You'll Learn**:
- Evolution from Bahdanau attention to Flash Attention
- Multi-Query Attention (MQA) vs Grouped-Query Attention (GQA)
- Memory efficiency trade-offs
- KV-cache optimization techniques

**Key Takeaways**:
- GQA provides best quality/memory trade-off (LLaMA 2/3 choice)
- Flash Attention: 2-4× speedup with O(n) memory
- Attention is O(n²) → bottleneck for long sequences

**Hands-On Exercise**:
```python
# Calculate KV-cache for different attention types
def calculate_kv_cache(n_layers, n_kv_heads, context_len, head_dim):
    return n_layers * n_kv_heads * context_len * head_dim * 2 * 2  # bytes
# Compare: MHA (32 heads) vs GQA-8 vs MQA (1 head)
```

---

### 2. Transformer Architecture Papers ⭐⭐⭐

**Our Summary**: `/learning-materials/papers/summaries/transformer-architecture-papers.md`
**Reading Time**: 90-120 minutes
**Priority**: Critical

**Paper Sequence**:
1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Link: https://arxiv.org/abs/1706.03762
   - Foundation of all modern LLMs

2. **GPT-2** (Radford et al., 2019)
   - Link: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
   - Pre-normalization, scaling insights

3. **LLaMA 1** (Touvron et al., 2023)
   - Link: https://arxiv.org/abs/2302.13971
   - RMSNorm, RoPE, SwiGLU optimizations

4. **LLaMA 2** (Touvron et al., 2023)
   - Link: https://arxiv.org/abs/2307.09288
   - GQA, 4K context, production improvements

**Architecture Evolution**:
```
Original Transformer (encoder-decoder)
  ↓
GPT (decoder-only, LayerNorm, learned positions)
  ↓
LLaMA (RMSNorm, RoPE, SwiGLU) ← Inference-optimized!
  ↓
LLaMA 2 (+ GQA) ← Production-ready
```

**Practical Application**:
```bash
# Inspect model architecture in llama.cpp
python -c "
from gguf import GGUFReader
reader = GGUFReader('model.gguf')
print(f\"Layers: {reader.fields['llama.block_count']}\")
print(f\"Heads: {reader.fields['llama.attention.head_count']}\")
print(f\"KV Heads: {reader.fields.get('llama.attention.head_count_kv', 'N/A')}\")
print(f\"Context: {reader.fields['llama.context_length']}\")
"
```

---

### 3. Tokenization Algorithms ⭐⭐⭐

**Our Summary**: `/learning-materials/papers/summaries/tokenization-algorithms.md`
**Reading Time**: 60-75 minutes
**Priority**: High

**Key Papers**:
1. **BPE for NMT** (Sennrich et al., 2016)
   - Link: https://arxiv.org/abs/1508.07909
   - Subword tokenization foundation

2. **SentencePiece** (Kudo & Richardson, 2018)
   - Link: https://arxiv.org/abs/1808.06226
   - Language-agnostic, reversible tokenization

**What You'll Learn**:
- Byte Pair Encoding (BPE) algorithm
- WordPiece vs SentencePiece vs Tiktoken
- Vocabulary size impact (32K vs 128K)
- Tokenization impact on inference speed

**Tokenization Comparison**:
```python
# LLaMA 2: 32K vocab, SentencePiece BPE
text = "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
# Tokens: ~45

# LLaMA 3: 128K vocab, improved SentencePiece
# Tokens: ~35 (22% reduction → faster inference!)
```

**Interactive Tool**:
- OpenAI Tokenizer: https://platform.openai.com/tokenizer
- Try different texts, compare models

---

## Component Papers

### 4. RMSNorm Paper ⭐⭐

**Paper**: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
**Link**: https://arxiv.org/abs/1910.07467
**Reading Time**: 25 minutes

**Why It Matters**:
- LLaMA uses RMSNorm instead of LayerNorm
- 7-64% faster (removes mean computation)
- Same quality as LayerNorm

**Implementation**:
```python
import torch

def rms_norm(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x

# vs LayerNorm (slower):
def layer_norm(x, weight, bias, eps=1e-6):
    mean = x.mean(-1, keepdim=True)  # Extra computation!
    variance = x.var(-1, keepdim=True)
    return weight * (x - mean) / torch.sqrt(variance + eps) + bias
```

---

### 5. RoPE Paper ⭐⭐

**Paper**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
**Link**: https://arxiv.org/abs/2104.09864
**Reading Time**: 40 minutes

**Why It Matters**:
- Better than absolute/learned position embeddings
- Enables context extension (RoPE scaling)
- Used in LLaMA, GPT-NeoX, PaLM

**Key Property**: Encodes absolute position with relative distance

**llama.cpp RoPE Scaling**:
```bash
# Extend context beyond training length
./llama-cli -m model.gguf \
  --ctx-size 8192 \
  --rope-freq-base 10000 \    # Default
  --rope-freq-scale 2.0       # 2× context extension
```

---

### 6. SwiGLU Paper ⭐⭐

**Paper**: "GLU Variants Improve Transformer" (Shazeer, 2020)
**Link**: https://arxiv.org/abs/2002.05202
**Reading Time**: 20 minutes

**Why It Matters**:
- LLaMA uses SwiGLU activation (not ReLU/GELU)
- Gating mechanism improves quality
- Slight compute overhead, worth it for accuracy

**Formula**: SwiGLU(x, W, V) = Swish(xW) ⊙ xV

---

## Supplementary Resources

### 7. The Illustrated Transformer ⭐⭐⭐

**Type**: Blog Post (Visual Tutorial)
**Author**: Jay Alammar
**Link**: https://jalammar.github.io/illustrated-transformer/
**Reading Time**: 30 minutes

**Why Read This**:
- Best visual explanation of transformers
- Perfect introduction before reading papers
- Animations showing attention flow

**Recommended Sequence**:
1. Read this first (visual intuition)
2. Then read "Attention Is All You Need" (formal)
3. Then read LLaMA paper (modern optimizations)

---

### 8. Andrej Karpathy - Let's Build GPT ⭐⭐⭐

**Type**: Video Tutorial
**Link**: https://www.youtube.com/watch?v=kCc8FmEb1nY
**Duration**: 2 hours
**Format**: Code-along

**What You'll Build**:
- Transformer from scratch in PyTorch
- Understanding every line of code
- Training a small GPT on Shakespeare

**Value**:
- Deepest understanding of transformer mechanics
- Perfect for hands-on learners
- Complements paper reading

---

### 9. 3Blue1Brown - Attention in Transformers ⭐⭐⭐

**Type**: Video
**Link**: https://www.youtube.com/watch?v=eMlx5fFNoYc
**Duration**: 27 minutes

**Why Watch**:
- Best visual explanation of attention mechanism
- Intuitive understanding before diving into math
- High production quality

---

### 10. Flash Attention Blog Post ⭐⭐

**Type**: Blog Post
**Author**: Aleksa Gordić
**Link**: https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad
**Reading Time**: 20 minutes

**Why Read**:
- Simplified Flash Attention explanation
- Complements the formal paper
- Focus on intuition over math

---

## Hands-On Labs

### Lab 1: Inspect LLaMA Architecture

```python
# Analyze GGUF model architecture
from gguf import GGUFReader

def analyze_model(path):
    reader = GGUFReader(path)

    print("=== Model Architecture ===")
    print(f"Layers: {reader.fields['llama.block_count']}")
    print(f"Hidden dim: {reader.fields['llama.embedding_length']}")
    print(f"Query heads: {reader.fields['llama.attention.head_count']}")
    print(f"KV heads: {reader.fields.get('llama.attention.head_count_kv', 'N/A')}")
    print(f"Context: {reader.fields['llama.context_length']}")
    print(f"Vocab size: {len(reader.tensors)}")

    # Determine generation
    n_heads = reader.fields['llama.attention.head_count']
    n_kv_heads = reader.fields.get('llama.attention.head_count_kv')

    if n_kv_heads and n_kv_heads < n_heads:
        print(f"\n✓ Uses GQA (KV heads: {n_kv_heads}, ratio: {n_heads/n_kv_heads}:1)")
    else:
        print("\n✓ Uses MHA (standard multi-head attention)")

analyze_model("llama-2-7b.gguf")
```

### Lab 2: Tokenization Exploration

```python
# Compare tokenizers
from transformers import AutoTokenizer

# LLaMA 2 tokenizer (32K vocab)
tokenizer_llama2 = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# LLaMA 3 tokenizer (128K vocab)
tokenizer_llama3 = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

test_text = "The quick brown fox jumps over the lazy dog"

tokens_llama2 = tokenizer_llama2.tokenize(test_text)
tokens_llama3 = tokenizer_llama3.tokenize(test_text)

print(f"LLaMA 2 tokens: {len(tokens_llama2)} - {tokens_llama2}")
print(f"LLaMA 3 tokens: {len(tokens_llama3)} - {tokens_llama3}")
print(f"Compression improvement: {(1 - len(tokens_llama3)/len(tokens_llama2))*100:.1f}%")
```

### Lab 3: Attention Visualization

```python
# Visualize attention patterns
import matplotlib.pyplot as plt

def visualize_attention(attn_weights, tokens):
    """
    attn_weights: [seq_len, seq_len]
    tokens: List of tokens
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_weights, cmap='viridis')
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.title("Attention Weights")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.tight_layout()
    plt.show()

# Exercise: Extract attention weights from model and visualize
```

---

## Learning Path

### Week 1: Transformer Fundamentals

**Day 1-2**:
- Watch: 3Blue1Brown attention video (27 min)
- Read: The Illustrated Transformer (30 min)
- Lab: Visualize attention patterns

**Day 3-4**:
- Read: "Attention Is All You Need" (Sections 3.1-3.4)
- Read: Our transformer architecture summary
- Exercise: Implement attention in NumPy

**Day 5-7**:
- Watch: Andrej Karpathy GPT video (2 hours, code-along)
- Read: GPT-2 paper (focus on architecture section)
- Lab: Build tiny transformer

### Week 2: LLaMA Architecture

**Day 1-2**:
- Read: LLaMA 1 paper (full paper)
- Read: Our LLaMA architecture summary
- Lab: Inspect LLaMA model architecture (GGUF)

**Day 3-4**:
- Read: RMSNorm paper
- Read: RoPE paper (Sections 3-4)
- Read: SwiGLU paper
- Exercise: Implement RMSNorm, compare to LayerNorm

**Day 5-6**:
- Read: LLaMA 2 paper (focus on GQA section)
- Read: Our attention mechanisms survey
- Lab: Calculate KV-cache sizes for MHA vs GQA

**Day 7**:
- Read: Flash Attention blog post
- Review: All concepts
- Create: Architecture mind map

### Week 3: Tokenization & Advanced Topics

**Day 1-3**:
- Read: BPE for NMT paper
- Read: SentencePiece paper
- Read: Our tokenization summary
- Lab: Tokenization exploration

**Day 4-5**:
- Read: Flash Attention paper (Sections 1-3)
- Understand: IO-aware algorithm design
- Exercise: Calculate memory savings

**Day 6-7**:
- Read: LLaMA 3 updates
- Review: Module 2 complete curriculum
- Assessment: Self-test (below)

---

## Self-Assessment Questions

### Architecture Understanding

1. **Q**: Why does LLaMA use RMSNorm instead of LayerNorm?
   **A**: RMSNorm is 7-64% faster (removes mean computation), same quality

2. **Q**: How does GQA reduce memory compared to MHA?
   **A**: Shares KV projections across query heads (e.g., 8 KV heads for 32 query heads = 4× reduction)

3. **Q**: What is the computational complexity of attention?
   **A**: O(n² × d) where n = sequence length, d = dimension → bottleneck for long sequences

4. **Q**: How does RoPE enable context extension?
   **A**: Rotary embeddings extrapolate to longer sequences through frequency scaling

5. **Q**: Why use SwiGLU instead of ReLU?
   **A**: Gating mechanism (Swish(xW) ⊙ xV) improves quality, slight compute overhead

### Tokenization

6. **Q**: How does BPE balance vocabulary size and flexibility?
   **A**: Merges frequent character pairs → common words = single tokens, rare words = subunits

7. **Q**: Why does LLaMA 3 use 128K vocab vs LLaMA 2's 32K?
   **A**: Better compression (~1.3× fewer tokens), especially for code and non-English

8. **Q**: What is SentencePiece's advantage over BPE?
   **A**: Language-agnostic, reversible (lossless), no pre-tokenization needed

---

## Advanced Topics (Optional)

### Multi-Modal Extensions
- LLaVA: Vision + Language (Module 7)
- Audio transformers: Whisper architecture

### Efficient Variants
- Linear attention: O(n) complexity
- Sparse attention patterns: Local, strided, block-sparse
- State space models: Mamba (alternative to attention)

---

## Resources

### Official Documentation
- llama.cpp docs: `/home/user/llama.cpp-learn/docs/`
- GGUF spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- HuggingFace Transformers docs

### Tools
- GGUF inspector: `gguf-py` library
- Tokenizer playground: https://platform.openai.com/tokenizer
- Attention visualizer: BertViz

---

## Key Insights for Module 3

**Prepare for Quantization Module**:
- Understanding tensor shapes → quantization granularity
- KV-cache optimization → why quantization matters
- Attention memory → target for compression

**Architecture knowledge helps**:
- Choose quantization levels per tensor type
- Understand which operations are memory-bound
- Optimize for specific model architectures

---

**Document Created**: 2025-11-18
**Status**: Complete - Ready for Module 2 curriculum
**Next Module**: Module 3 - Quantization Deep Dive

---

**For Curriculum Designer (Agent 2)**:
- Focus on hands-on labs (architecture inspection, tokenization)
- Include visualization exercises (attention, embeddings)
- Connect to llama.cpp practical usage
- Build on Module 1 GGUF knowledge

**For Lab Developer (Agent 4)**:
- Lab 1: GGUF architecture inspection (expand on this)
- Lab 2: Tokenization comparison (LLaMA 2 vs 3)
- Lab 3: Attention visualization
- Lab 4: Implement RMSNorm, compare to LayerNorm
- Lab 5: Calculate memory for different attention types
