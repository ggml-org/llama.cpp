# Module 2: Core Implementation

**Duration**: 3-4 weeks | **Difficulty**: Mid-Senior | **Prerequisites**: Module 1, C++ basics

---

## 📚 Overview

Deep dive into the core implementation of llama.cpp, understanding transformer architecture, attention mechanisms, memory management, and token generation at the implementation level.

**Learning Outcomes**:
- ✅ Understand transformer architecture in detail
- ✅ Implement attention mechanisms
- ✅ Master KV cache design and optimization
- ✅ Learn memory management strategies
- ✅ Implement sampling algorithms
- ✅ Debug inference issues

---

## 📖 Lessons

### Lesson 2.1: Transformer Architecture (6 hours)
- Transformer layer structure
- Multi-head attention
- Feed-forward networks (SwiGLU)
- Layer normalization (RMSNorm)
- Residual connections

### Lesson 2.2: Attention Mechanisms (8 hours)
- Self-attention mathematics
- Multi-Head Attention (MHA)
- Grouped-Query Attention (GQA)
- Multi-Query Attention (MQA)
- RoPE (Rotary Position Embeddings)

### Lesson 2.3: KV Cache Design (6 hours)
- Why KV cache is needed
- Memory layout and management
- Cache allocation strategies
- Multi-user scenarios
- Eviction policies

### Lesson 2.4: Token Generation & Sampling (5 hours)
- Logit processing
- Greedy decoding
- Temperature, top-k, top-p
- Mirostat sampling
- Repetition penalties

### Lesson 2.5: Memory Management (6 hours)
- GGML tensor structure
- Memory layout (row-major)
- Scratch buffers
- Memory mapping
- Efficient memory access patterns

---

## 🔬 Labs & Exercises

### Lab 2.1: Implement Attention (8 hours)
Build attention mechanism from scratch:
- Q, K, V projections
- Scaled dot-product attention
- Multi-head parallelism
- Output projection

**Deliverable**: Working Python/C++ implementation

### Lab 2.2: KV Cache Profiling (4 hours)
Analyze KV cache impact:
- Memory usage tracking
- Performance with/without cache
- Cache hit rates
- Optimization strategies

**Deliverable**: Performance report with charts

### Lab 2.3: Sampling Experiments (4 hours)
Compare sampling strategies:
- Temperature sweep
- Top-k vs top-p
- Quality vs diversity tradeoffs
- Benchmark different prompts

**Deliverable**: Analysis document

### Lab 2.4: Memory Layout Analysis (4 hours)
Understand GGML tensors:
- Inspect tensor structure
- Calculate strides
- Visualize memory layout
- Cache efficiency analysis

**Deliverable**: Technical writeup

---

## 📝 Interview Prep

**Question Count**: 20 questions
**Difficulty**: Mid to Senior level
**Location**: `/interview-prep/questions/module-02-core-implementation-questions.md`

**Topics Covered**:
- Transformer architecture
- Attention variants (MHA, GQA, MQA)
- KV cache strategies
- Memory management
- System design for inference
- Debugging techniques

**Recommended Study**: 10-15 hours

---

## 🎯 Assessments

### Knowledge Check Quizzes
- Quiz 2.1: Transformer Basics (10 questions)
- Quiz 2.2: Attention Mechanisms (10 questions)
- Quiz 2.3: Memory & Caching (10 questions)

**Passing Score**: 80%

### Coding Assignments
1. Implement simplified transformer layer
2. Build KV cache manager
3. Create sampling algorithm library

### Module Project
**Build a Mini Inference Engine**
- Load GGUF model
- Implement forward pass
- Add KV cache
- Support sampling

**Duration**: 1 week
**Evaluation**: Functionality, performance, code quality

---

## 📚 Additional Resources

### Code Examples
- `/code-examples/cpp/transformer/` - C++ implementations
- `/code-examples/python/attention/` - Python notebooks
- `/code-examples/cuda/` - GPU kernels

### Reading Materials
- "Attention Is All You Need" (Vaswani et al.)
- "LLaMA" paper (Touvron et al.)
- "FlashAttention" paper (Dao et al.)

### Video Tutorials
- Karpathy's transformer deep dive
- Architecture walkthroughs

---

## 🔗 Related Modules

**Prerequisites**: Module 1 (Foundations)
**Follows To**: 
- Module 3 (Quantization)
- Module 5 (Advanced Inference)

**Related Topics**:
- Module 4 (GPU Acceleration) - Kernel implementation
- Module 6 (Server) - Production inference

---

**Next Module**: [Module 3: Quantization](../03-quantization/README.md)
**Previous Module**: [Module 1: Foundations](../01-foundations/README.md)

**Estimated Completion**: 3-4 weeks part-time (20-25 hours total)
