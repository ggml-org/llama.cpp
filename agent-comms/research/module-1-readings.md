# Module 1: Foundations - Curated Reading List
## Research-Backed Resources for LLM Inference Mastery

**Curator**: Agent 1 (Research Curator)
**Module**: Module 1 - Foundations
**Last Updated**: 2025-11-18
**Target Audience**: Engineers learning LLM inference with llama.cpp
**Total Resources**: 15 curated items
**Estimated Study Time**: 12-15 hours

---

## 📋 How to Use This Reading List

### Reading Path Options

**🏃 Fast Track (4-6 hours)**
- Essential papers marked with ⭐⭐⭐
- Focus on summaries and key sections
- Skip deep mathematical proofs

**🚶 Standard Path (8-10 hours)**
- All papers marked ⭐⭐⭐ and ⭐⭐
- Read technical sections thoroughly
- Complete recommended exercises

**🎓 Deep Dive (12-15 hours)**
- All resources including ⭐ items
- Study supplementary materials
- Implement key algorithms

---

## Essential Papers

### 1. LLaMA: Open and Efficient Foundation Language Models ⭐⭐⭐

**Type**: Research Paper
**Authors**: Hugo Touvron, Thibaut Lavril, et al. (Meta AI)
**Published**: February 2023
**Link**: https://arxiv.org/abs/2302.13971
**Reading Time**: 45 minutes (full paper) | 15 minutes (our summary)

**Why This Matters**:
LLaMA is THE foundational architecture for llama.cpp. Understanding its design choices (RMSNorm, SwiGLU, RoPE) directly impacts your ability to optimize inference, choose quantization strategies, and debug performance issues.

**Key Sections to Focus On**:
- Section 2: Pre-training Approach (data composition, training details)
- Section 3: Architecture (RMSNorm, SwiGLU, RoPE explained)
- Section 4: Results (performance benchmarks vs. GPT-3, PaLM)
- Section 5: Analysis (scaling laws, data efficiency)

**Learning Objectives**:
- [ ] Understand why smaller models can outperform larger ones
- [ ] Explain RMSNorm, SwiGLU, and RoPE to a colleague
- [ ] Identify architectural optimizations for inference
- [ ] Relate training decisions to inference performance

**Practical Application**:
After reading, you should be able to:
- Choose appropriate quantization levels for different tensor types
- Explain why llama.cpp is CPU-friendly
- Predict memory requirements for different model sizes

**Companion Resources**:
- Our summary: `/learning-materials/papers/summaries/llama-paper-summary.md`
- Lab 1: First Inference with LLaMA
- Code example: Basic LLaMA inference

**Discussion Questions**:
1. Why does RMSNorm improve inference speed compared to LayerNorm?
2. How does RoPE enable context extension beyond training length?
3. What trade-offs did Meta make choosing public-only training data?

---

### 2. Attention Is All You Need (Transformer Foundation) ⭐⭐⭐

**Type**: Research Paper
**Authors**: Vaswani, Shazeer, Parmar, et al. (Google)
**Published**: June 2017
**Link**: https://arxiv.org/abs/1706.03762
**Reading Time**: 60 minutes

**Why This Matters**:
The Transformer architecture is the foundation of all modern LLMs. While LLaMA makes modifications, understanding the base architecture is essential for comprehending how attention, positional encoding, and feed-forward networks work together.

**Key Sections to Focus On**:
- Section 3.1: Attention mechanisms
- Section 3.2: Multi-head attention
- Section 3.3: Position-wise feed-forward networks
- Section 3.4: Embeddings and positional encoding
- Figure 1: Model architecture diagram (study this carefully!)

**Learning Objectives**:
- [ ] Understand scaled dot-product attention formula
- [ ] Explain why multi-head attention improves model capacity
- [ ] Describe the role of position encodings
- [ ] Identify components in the transformer block

**Practical Application**:
- Understand where computational bottlenecks are (attention is O(n²))
- Recognize why KV-caching is critical for inference
- Identify opportunities for optimization (Flash Attention, etc.)

**Helpful Supplementary Materials**:
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- Annotated Transformer: http://nlp.seas.harvard.edu/annotated-transformer/
- 3Blue1Brown video on Attention: https://www.youtube.com/watch?v=eMlx5fFNoYc

**Discussion Questions**:
1. Why is attention O(n²) in sequence length? Can this be improved?
2. How does multi-head attention differ from single-head?
3. What would happen if we removed positional encodings?

**Connection to llama.cpp**:
- llama.cpp implements optimized attention kernels
- KV-cache is critical for autoregressive generation
- Flash Attention reduces memory usage dramatically

---

### 3. GGUF Format Specification (Official Docs) ⭐⭐⭐

**Type**: Technical Specification
**Authors**: GGML Team
**Published**: 2023 (updated continuously)
**Link**: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
**Reading Time**: 30 minutes

**Why This Matters**:
GGUF is the binary format that makes llama.cpp efficient. Understanding it helps you debug loading issues, choose quantization formats, and optimize model distribution.

**Key Sections to Focus On**:
- File structure overview
- Metadata system (key-value pairs)
- Tensor data types and quantization formats
- Alignment requirements for memory mapping

**Learning Objectives**:
- [ ] Describe GGUF file structure from header to tensor data
- [ ] Explain why alignment matters for mmap
- [ ] List at least 5 quantization formats and their trade-offs
- [ ] Read and interpret GGUF metadata using gguf-py

**Practical Application**:
```python
# Hands-on exercise
from gguf import GGUFReader

reader = GGUFReader("model.gguf")
# What metadata is present?
# What quantization formats are used?
# How many tensors? What are their shapes?
```

**Companion Resources**:
- Our summary: `/learning-materials/papers/summaries/gguf-format-summary.md`
- Lab 2: GGUF Format Exploration
- Tool: gguf-py for inspection

**Discussion Questions**:
1. Why use binary format instead of JSON/text?
2. How does memory mapping improve loading speed?
3. What are the trade-offs between Q4_K_M and Q8_0?

---

## Quantization and Optimization

### 4. LLM.int8(): 8-bit Matrix Multiplication for Transformers ⭐⭐⭐

**Type**: Research Paper
**Authors**: Tim Dettmers, Mike Lewis, et al.
**Published**: November 2022
**Link**: https://arxiv.org/abs/2208.07339
**Reading Time**: 40 minutes

**Why This Matters**:
Quantization is what makes LLMs runnable on consumer hardware. This paper introduces techniques that enable 8-bit inference with minimal quality loss, foundational to understanding modern quantization methods.

**Key Concepts**:
- Outlier detection and mixed-precision decomposition
- Vector-wise quantization
- Why naive quantization fails (outlier features)
- Zero-degradation 8-bit matrix multiplication

**Learning Objectives**:
- [ ] Understand why outliers cause quantization problems
- [ ] Explain mixed-precision decomposition
- [ ] Calculate memory savings from 8-bit quantization
- [ ] Identify when to use higher precision

**Practical Application**:
- Understand why certain tensors (embeddings, output layer) need higher precision
- Predict which quantization schemes will work for your use case
- Debug quantization-related quality degradation

**Connection to llama.cpp**:
- Q8_0 quantization in GGUF implements similar ideas
- Importance matrix quantization builds on these concepts
- Understanding helps choose quantization levels per tensor

---

### 5. GPTQ: Accurate Post-Training Quantization for GPT Models ⭐⭐

**Type**: Research Paper
**Authors**: Elias Frantar, Saleh Ashkboos, et al.
**Published**: October 2022
**Link**: https://arxiv.org/abs/2210.17323
**Reading Time**: 35 minutes

**Why This Matters**:
GPTQ achieves excellent 4-bit quantization quality through careful optimization. While llama.cpp uses K-quants, understanding GPTQ's approach helps you appreciate quantization trade-offs.

**Key Concepts**:
- Layer-wise quantization
- Hessian-aware quantization (importance-based)
- Group-wise quantization
- 4-bit with near-lossless quality

**Learning Objectives**:
- [ ] Understand importance-based quantization
- [ ] Explain group-wise vs. per-tensor quantization
- [ ] Compare GPTQ to uniform quantization
- [ ] Identify computational trade-offs

**Discussion Questions**:
1. Why does layer-wise quantization work better than global?
2. How does Hessian information improve quantization?
3. What are GPTQ's limitations for CPU inference?

**Comparison to llama.cpp**:
- GPTQ: GPU-optimized, excellent quality
- K-quants: CPU-friendly, good quality, more flexible
- Both achieve ~4-bit effective compression

---

### 6. AWQ: Activation-aware Weight Quantization ⭐⭐

**Type**: Research Paper
**Authors**: Ji Lin, Jiaming Tang, et al. (MIT, NVIDIA)
**Published**: June 2023
**Link**: https://arxiv.org/abs/2306.00978
**Reading Time**: 30 minutes

**Why This Matters**:
AWQ demonstrates that protecting "salient" weights (those with high activation magnitudes) enables better 4-bit quantization. This concept influences importance-matrix quantization in llama.cpp.

**Key Innovation**:
- Activation magnitude predicts weight importance
- Protect salient weights with higher precision
- Achieve better quality than GPTQ at same bit-width

**Learning Objectives**:
- [ ] Understand activation-aware quantization
- [ ] Explain salience detection
- [ ] Compare AWQ to GPTQ and uniform quantization

**Practical Application**:
- Understand llama.cpp's importance matrix (imatrix) quantization
- Choose calibration data for better quantization
- Predict which layers need higher precision

---

## Architecture Deep Dives

### 7. RoFormer: Enhanced Transformer with Rotary Position Embedding ⭐⭐

**Type**: Research Paper
**Authors**: Jianlin Su, Yu Lu, et al.
**Published**: April 2021
**Link**: https://arxiv.org/abs/2104.09864
**Reading Time**: 40 minutes

**Why This Matters**:
RoPE (Rotary Positional Embeddings) is used in LLaMA and most modern LLMs. Understanding RoPE helps you understand context extension techniques and position encoding choices.

**Key Concepts**:
- Absolute position with relative distance preservation
- Rotation matrices in complex space
- Long-range decay property
- Extrapolation to longer sequences

**Learning Objectives**:
- [ ] Explain RoPE mathematically
- [ ] Describe advantages over absolute/learned positional encodings
- [ ] Understand why RoPE enables context extension
- [ ] Implement basic RoPE in Python/NumPy

**Practical Application**:
```python
# Exercise: Implement basic RoPE
import numpy as np

def apply_rope(q, k, position, theta=10000.0):
    """Apply rotary position embedding"""
    # Your implementation here
    pass
```

**Connection to llama.cpp**:
- RoPE implementation in ggml
- RoPE scaling for extended context
- NTK-aware RoPE scaling

---

### 8. Root Mean Square Layer Normalization ⭐⭐

**Type**: Research Paper
**Authors**: Biao Zhang, Rico Sennrich
**Published**: October 2019
**Link**: https://arxiv.org/abs/1910.07467
**Reading Time**: 25 minutes

**Why This Matters**:
RMSNorm replaces LayerNorm in LLaMA, providing 7-64% speedup with comparable quality. Understanding why helps you optimize inference and appreciate architectural choices.

**Key Concepts**:
- Removing mean normalization step
- Root mean square scaling
- Training stability comparison
- Computational efficiency

**Learning Objectives**:
- [ ] Derive RMSNorm formula from LayerNorm
- [ ] Explain computational savings
- [ ] Understand when RMSNorm might underperform
- [ ] Implement RMSNorm in Python

**Code Exercise**:
```python
import torch

def rms_norm(x, weight, eps=1e-6):
    """Implement RMSNorm"""
    # Your implementation
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x
```

---

### 9. GLU Variants Improve Transformer ⭐⭐

**Type**: Research Paper
**Authors**: Noam Shazeer (Google)
**Published**: February 2020
**Link**: https://arxiv.org/abs/2002.05202
**Reading Time**: 20 minutes

**Why This Matters**:
SwiGLU (used in LLaMA) is introduced here. Understanding gated activation functions helps you understand why LLaMA chose this over ReLU/GELU.

**Key Concepts**:
- Gated Linear Units (GLU) family
- Swish activation (x * sigmoid(x))
- SwiGLU = GLU + Swish
- Performance comparison across variants

**Learning Objectives**:
- [ ] Explain gating mechanism
- [ ] Describe SwiGLU advantages
- [ ] Understand computational overhead
- [ ] Compare activation functions

**Discussion Questions**:
1. Why does gating improve model quality?
2. What's the computational cost of SwiGLU vs. ReLU?
3. How does SwiGLU affect gradient flow?

---

## Inference Optimization

### 10. FlashAttention: Fast and Memory-Efficient Exact Attention ⭐⭐⭐

**Type**: Research Paper
**Authors**: Tri Dao, Daniel Y. Fu, et al. (Stanford)
**Published**: May 2022
**Link**: https://arxiv.org/abs/2205.14135
**Reading Time**: 45 minutes

**Why This Matters**:
FlashAttention is implemented in llama.cpp and provides 2-4x speedup with reduced memory usage. Understanding it helps you optimize inference performance.

**Key Innovation**:
- IO-aware algorithm design
- Fused kernel avoiding materialization of N×N attention matrix
- Recomputation for memory efficiency
- Exact attention (not approximate)

**Learning Objectives**:
- [ ] Understand memory bottlenecks in standard attention
- [ ] Explain tiling and kernel fusion
- [ ] Calculate memory savings
- [ ] Identify when FlashAttention helps most

**Practical Impact**:
- Enables longer context lengths
- Faster training and inference
- Lower memory usage
- Better GPU utilization

**Connection to llama.cpp**:
- Flash Attention support in CUDA backend
- Reduces peak memory during attention
- Critical for long-context inference

**Follow-up**: FlashAttention-2 (https://arxiv.org/abs/2307.08691) - further optimizations

---

### 11. Grouped-Query Attention (GQA) - From LLaMA 2 Paper ⭐⭐

**Type**: Technical Report (part of LLaMA 2 paper)
**Authors**: Meta AI
**Published**: July 2023
**Link**: https://arxiv.org/abs/2307.09288 (Section 2.2)
**Reading Time**: 15 minutes (relevant section)

**Why This Matters**:
GQA reduces KV-cache memory by sharing key/value projections across query heads. This is critical for serving large models efficiently.

**Key Concepts**:
- Multi-Query Attention (MQA) extreme: 1 KV head
- Multi-Head Attention (MHA) baseline: H KV heads
- Grouped-Query Attention (GQA): G KV heads (1 < G < H)
- Trade-off: memory vs. quality

**Learning Objectives**:
- [ ] Explain GQA vs. MHA vs. MQA
- [ ] Calculate KV-cache savings with GQA
- [ ] Understand quality trade-offs
- [ ] Implement GQA attention in pseudocode

**Memory Calculation Example**:
```
LLaMA 2 7B:
- 32 query heads
- 32 KV heads (MHA) → 1.0 GB KV-cache
- 8 KV heads (GQA)  → 0.25 GB KV-cache (4x reduction!)
- 1 KV head (MQA)   → 0.03 GB KV-cache (32x, but quality loss)
```

**Connection to llama.cpp**:
- GQA support in attention kernels
- Reduced memory enables larger batch sizes
- Critical for LLaMA 2/3 models

---

## Practical Guides and Tutorials

### 12. The Illustrated Transformer (Blog Post) ⭐⭐⭐

**Type**: Blog Post / Tutorial
**Author**: Jay Alammar
**Published**: 2018 (updated)
**Link**: https://jalammar.github.io/illustrated-transformer/
**Reading Time**: 30 minutes

**Why This Matters**:
Best visual explanation of Transformers available. Perfect for understanding how components fit together before diving into code.

**What You'll Learn**:
- Visual walkthrough of transformer architecture
- Step-by-step attention mechanism
- How positional encoding works
- Encoder-decoder vs. decoder-only

**Learning Value**: ⭐⭐⭐⭐⭐
- Excellent for visual learners
- Complements formal papers perfectly
- Makes complex concepts intuitive

**Recommended Sequence**:
1. Read this first (before "Attention is All You Need")
2. Then read the original Transformer paper
3. Then read LLaMA paper

---

### 13. A Gentle Introduction to 8-bit Matrix Multiplication ⭐⭐

**Type**: Blog Post
**Author**: Hugging Face Blog / Tim Dettmers
**Published**: 2022
**Link**: https://huggingface.co/blog/hf-bitsandbytes-integration
**Reading Time**: 25 minutes

**Why This Matters**:
Practical introduction to quantization concepts with code examples. Bridges theory (papers) and practice (llama.cpp).

**Key Topics**:
- Why quantization works
- Absmax vs. zeropoint quantization
- Outlier handling strategies
- Practical integration examples

**Learning Value**:
- Hands-on code examples
- Practical tips for using quantization
- Debugging quantization issues

**Companion Exercise**:
- Try quantizing a small matrix in NumPy
- Measure quantization error
- Experiment with different bit widths

---

### 14. Understanding GGML (Blog Series) ⭐⭐

**Type**: Blog Series / Code Walkthrough
**Author**: Community / Multiple contributors
**Published**: 2023-2024
**Link**: https://github.com/ggml-org/ggml
**Reading Time**: 60 minutes (browsing repo + examples)

**Why This Matters**:
GGML is the tensor library underlying llama.cpp. Understanding its design helps you understand llama.cpp's capabilities and limitations.

**Key Components to Explore**:
- `ggml.h`: Core tensor operations
- `ggml-cuda.cu`: CUDA backend
- `ggml-metal.m`: Metal backend (Apple)
- Examples: `examples/` directory

**Learning Objectives**:
- [ ] Understand GGML's graph-based execution
- [ ] Identify available tensor operations
- [ ] Explore backend implementations
- [ ] Run simple GGML examples

**Hands-On**:
```bash
# Build GGML examples
cd ggml
mkdir build && cd build
cmake ..
make

# Run matrix multiplication example
./bin/mul-mat
```

---

### 15. llama.cpp Documentation Deep Dive ⭐⭐⭐

**Type**: Official Documentation
**Maintainer**: GGML Team
**Published**: Continuous updates
**Link**: `/home/user/llama.cpp-learn/docs/`
**Reading Time**: 90 minutes (full docs)

**Why This Matters**:
Official docs are the source of truth. Understanding build options, backend selection, and performance tuning is essential for production use.

**Essential Docs to Read**:
1. `build.md` - Build system, backends, optimization flags
2. `backend/` - CUDA, Metal, OpenCL setup
3. `development/HOWTO-add-model.md` - Model architecture details
4. `development/token_generation_performance_tips.md` - Optimization guide

**Learning Objectives**:
- [ ] Build llama.cpp with different backends
- [ ] Enable/disable features via CMake flags
- [ ] Optimize for your specific hardware
- [ ] Understand performance tuning options

**Hands-On Exercises**:
```bash
# Exercise 1: Build with CUDA support
cmake -B build -DGGML_CUDA=ON
cmake --build build

# Exercise 2: Test different quantizations
./llama-quantize model-f16.gguf model-q4.gguf Q4_K_M
./llama-quantize model-f16.gguf model-q5.gguf Q5_K_M
# Compare file sizes and perplexity

# Exercise 3: Benchmark
./llama-bench -m model.gguf -p 512 -n 128
```

---

## Supplementary Videos

### Video 1: 3Blue1Brown - Attention in Transformers

**Link**: https://www.youtube.com/watch?v=eMlx5fFNoYc
**Duration**: 27 minutes
**Why Watch**: Best visual explanation of attention mechanism
**Learning Value**: ⭐⭐⭐⭐⭐

### Video 2: Andrej Karpathy - Let's Build GPT

**Link**: https://www.youtube.com/watch?v=kCc8FmEb1nY
**Duration**: 2 hours
**Why Watch**: Build a transformer from scratch, understand every line
**Learning Value**: ⭐⭐⭐⭐⭐

### Video 3: Yannic Kilcher - LLaMA Paper Review

**Link**: https://www.youtube.com/watch?v=E5OnoYF2oAk
**Duration**: 45 minutes
**Why Watch**: Deep technical analysis of LLaMA innovations
**Learning Value**: ⭐⭐⭐⭐

---

## Learning Path Recommendations

### Week 1: Foundations
**Goal**: Understand transformer architecture and LLaMA innovations

**Day 1-2**:
- Read: "The Illustrated Transformer" (blog)
- Watch: 3Blue1Brown attention video
- Exercise: Implement basic attention in NumPy

**Day 3-4**:
- Read: "Attention is All You Need" (paper, focus on Sections 3.1-3.4)
- Read: LLaMA paper summary (ours)
- Exercise: Trace through transformer forward pass

**Day 5-7**:
- Read: RoPE paper (Section 3)
- Read: RMSNorm paper
- Read: SwiGLU paper
- Exercise: Implement these components

### Week 2: Quantization and Optimization
**Goal**: Understand how to make models efficient

**Day 1-2**:
- Read: GGUF specification
- Read: GGUF format summary (ours)
- Exercise: Inspect GGUF files with gguf-py

**Day 3-5**:
- Read: LLM.int8() paper
- Read: GPTQ paper (Sections 1-3)
- Exercise: Experiment with different quantizations

**Day 6-7**:
- Read: FlashAttention paper (Sections 1-3)
- Read: GQA section in LLaMA 2 paper
- Exercise: Benchmark attention with/without Flash

### Week 3: Practice and Integration
**Goal**: Apply knowledge to real inference scenarios

**Day 1-3**:
- Read: llama.cpp documentation (all essential docs)
- Exercise: Build llama.cpp with different backends
- Exercise: Quantize models, measure quality

**Day 4-5**:
- Explore: GGML repository
- Exercise: Run GGML examples
- Exercise: Profile llama.cpp inference

**Day 6-7**:
- Watch: Andrej Karpathy GPT video
- Exercise: Build mini-transformer from scratch
- Review: All concepts, create mind map

---

## Assessment Checkpoints

### Checkpoint 1: Architecture Understanding
**Can you explain to a colleague:**
- [ ] How attention mechanism works (with diagrams)
- [ ] Why RoPE is better than absolute position embeddings
- [ ] How RMSNorm saves compute vs. LayerNorm
- [ ] What SwiGLU does and why it improves quality

**If No**: Re-read Section 3 of Transformer paper, watch 3Blue1Brown video

### Checkpoint 2: Quantization Mastery
**Can you:**
- [ ] Explain why naive quantization fails (outliers)
- [ ] Choose appropriate quantization for your use case
- [ ] Calculate memory savings from quantization
- [ ] Debug quantization quality issues

**If No**: Re-read LLM.int8() paper, experiment with llama-quantize tool

### Checkpoint 3: Format and Tooling
**Can you:**
- [ ] Inspect GGUF files and extract metadata
- [ ] Build llama.cpp with custom flags
- [ ] Optimize for your specific hardware
- [ ] Profile and benchmark inference

**If No**: Re-read GGUF spec, work through llama.cpp build docs

---

## Common Pitfalls to Avoid

### ❌ Pitfall 1: Reading Papers Passively
**Problem**: Reading papers like novels without engagement
**Solution**:
- Take notes while reading
- Implement key algorithms
- Discuss concepts with peers
- Teach concepts back to yourself

### ❌ Pitfall 2: Skipping Math
**Problem**: Avoiding mathematical sections
**Solution**:
- Math is where understanding happens
- Work through derivations with pen and paper
- Implement equations in code to verify understanding

### ❌ Pitfall 3: Theory Without Practice
**Problem**: Reading papers but never using llama.cpp
**Solution**:
- Alternate reading with hands-on exercises
- Test concepts immediately in code
- Build intuition through experimentation

### ❌ Pitfall 4: Rushing Through
**Problem**: Trying to read everything in one week
**Solution**:
- Follow the 3-week learning path above
- Deep understanding > surface coverage
- Revisit difficult concepts multiple times

---

## Discussion Questions for Study Groups

### Week 1: Architecture
1. How would transformers change if attention was O(n) instead of O(n²)?
2. Why doesn't RoPE need learned parameters?
3. What are the limits of context length scaling with RoPE?

### Week 2: Quantization
4. When would you choose Q8_0 over Q4_K_M despite the size difference?
5. How does quantization interact with model fine-tuning?
6. What parts of the model are most sensitive to quantization?

### Week 3: Implementation
7. How would you design a serving system for 10,000 QPS with LLaMA-7B?
8. What are the trade-offs between mmap and traditional loading?
9. How does batch size affect memory usage and throughput?

---

## Advanced Reading (Post-Module 1)

Once you've mastered Module 1, consider these advanced topics:

### Model Architectures
- Mixtral (Mixture of Experts): https://arxiv.org/abs/2401.04088
- Mamba (State Space Models): https://arxiv.org/abs/2312.00752
- Retentive Networks: https://arxiv.org/abs/2307.08621

### Optimization Techniques
- Speculative Decoding: https://arxiv.org/abs/2211.17192
- PagedAttention: https://arxiv.org/abs/2309.06180
- Ring Attention: https://arxiv.org/abs/2310.01889

### Serving and Deployment
- vLLM paper: https://arxiv.org/abs/2309.06180
- Continuous Batching: https://www.anyscale.com/blog/continuous-batching-llm-inference
- Multi-GPU strategies: Various blog posts and docs

---

## Glossary of Key Terms

**Autoregressive**: Generating sequence one token at a time using previous tokens

**Attention**: Mechanism for weighing importance of different positions in sequence

**Decoder-only**: Transformer using only decoder stack (no encoder)

**GGML**: Tensor library powering llama.cpp

**GGUF**: Binary format for storing LLM models

**GQA (Grouped Query Attention)**: Sharing KV projections across query groups

**KV-cache**: Cached key/value projections for efficient generation

**Mmap**: Memory mapping - treating files as RAM

**Perplexity**: Metric measuring model prediction quality (lower = better)

**Quantization**: Reducing numerical precision to save memory

**RMSNorm**: Root Mean Square Normalization

**RoPE**: Rotary Positional Embeddings

**SwiGLU**: Swish-Gated Linear Unit activation function

**Tensor**: Multi-dimensional array of numbers

**Token**: Smallest text unit (~0.75 words in English)

---

## Citation Format

When referencing these papers in your work:

```bibtex
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and others},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}

@article{dettmers2022llm,
  title={LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale},
  author={Dettmers, Tim and Lewis, Mike and Belkada, Younes and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2208.07339},
  year={2022}
}
```

---

## Feedback and Updates

This reading list is a living document. Have suggestions?

**Feedback channels**:
- Submit PR with additions: `/agent-comms/research/`
- Discussion: Open issue tagged "research-curator"
- Contact: Agent 1 (Research Curator)

**Update schedule**:
- Monthly: New papers added
- Quarterly: Major revisions
- As needed: Broken link fixes

---

## Related Module Reading Lists

**Coming soon from Agent 1:**
- Module 2: Prompt Engineering & Context Management
- Module 3: Quantization Deep Dive
- Module 4: GPU Acceleration & CUDA Programming
- Module 5: Production Deployment Patterns
- Module 6: Fine-tuning & LoRA
- Module 7: Advanced Optimization Techniques
- Module 8: Multi-modal Models
- Module 9: Contributing to llama.cpp

---

## Key Insights for Agent 2 (Tutorial Architect)

**Module 1 should focus on:**
1. Transformer fundamentals (attention, FFN, normalization)
2. LLaMA-specific innovations (RMSNorm, RoPE, SwiGLU)
3. GGUF format and practical file handling
4. Basic quantization concepts (Q4_K_M vs. Q8_0 vs. F16)
5. Hands-on inference with llama.cpp Python bindings

**Prerequisite knowledge assumed:**
- Python programming (intermediate)
- Basic linear algebra (matrix multiplication, vectors)
- Basic machine learning concepts (training, inference, loss functions)
- Command-line comfort (bash, file operations)

**Recommended learning time**: 2-3 weeks (8-12 hours/week)

**Critical concepts that MUST be mastered:**
- ✅ Attention mechanism (with and without Flash Attention)
- ✅ RoPE and its impact on context extension
- ✅ Quantization trade-offs and selection
- ✅ GGUF file structure and loading
- ✅ Memory calculation for inference

**Common student struggles** (anticipate in tutorials):
1. Attention mechanism math (O(n²) complexity)
2. Understanding why RoPE works for position encoding
3. Choosing quantization levels (provide decision tree)
4. Memory calculations (provide formulas and examples)
5. Building llama.cpp with correct flags

---

**Document Created By**: Agent 1 (Research Curator)
**Last Updated**: 2025-11-18
**Status**: Complete - Ready for Module 1 curriculum design

**Next Actions**:
- Agent 2: Use this reading list to design Module 1 structure
- Agent 3: Identify code examples needed from reading list
- Agent 4: Design labs around key concepts from readings
- Agent 5: Create documentation linking to these resources

**Quality Check**: Agent 7 to verify all links accessible and resources appropriate
