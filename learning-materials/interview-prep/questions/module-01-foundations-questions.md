# Module 1: Foundations - Interview Questions

**Purpose**: Interview preparation for foundational llama.cpp concepts
**Target Level**: Entry to Senior Engineers
**Module Coverage**: Module 1 - Foundations
**Question Count**: 20 (5 per category)
**Last Updated**: 2025-11-18
**Created By**: Agent 6 (Interview Coach)

---

## Table of Contents

1. [Conceptual Questions](#conceptual-questions) (5 questions)
2. [Technical Questions](#technical-questions) (5 questions)
3. [System Design Questions](#system-design-questions) (5 questions)
4. [Debugging Questions](#debugging-questions) (5 questions)

---

## Conceptual Questions

### Question 1: What is GGUF and Why Does It Exist?

**Category**: Conceptual
**Difficulty**: Entry (L3/L4)
**Companies**: OpenAI, Anthropic, Hugging Face
**Time Allotted**: 10-15 minutes
**Prerequisites**: Module 1, Lesson 1.2

---

#### Question

Explain what GGUF is, why it was created, and how it differs from other model format options like PyTorch checkpoints or safetensors. What problem does GGUF specifically solve?

---

#### What the Interviewer is Looking For

**Core Competencies Tested**:
- [ ] Understanding of model serialization formats
- [ ] Knowledge of GGUF's design goals
- [ ] Awareness of format trade-offs
- [ ] Practical experience with model formats

**Red Flags**:
- ❌ Thinks GGUF is just another checkpoint format
- ❌ Can't explain the difference from GGML
- ❌ Doesn't understand quantization integration
- ❌ No awareness of metadata capabilities

**Green Flags**:
- ✅ Explains extensible metadata system
- ✅ Mentions quantization-first design
- ✅ Discusses memory-mapping benefits
- ✅ Compares to predecessor formats (GGML)
- ✅ Mentions backward compatibility

---

#### Hints (If Candidate is Stuck)

**Hint 1**: Format Evolution
"What problems did the previous GGML format have that led to creating GGUF?"

**Hint 2**: Design Goals
"Think about what's needed for efficient inference at the edge. What format features would you want?"

**Hint 3**: Metadata
"How does GGUF handle model metadata differently than PyTorch's state_dict?"

---

#### Model Solution

**Definition**:
GGUF (GPT-Generated Unified Format) is a binary file format specifically designed for efficient storage and loading of large language models, with a focus on quantized inference and edge deployment.

**Why GGUF Exists**:

1. **Limitations of GGML** (predecessor format):
   - Poor metadata support (hard to know model architecture without loading)
   - Limited extensibility (adding new features broke compatibility)
   - No versioning system
   - Difficult to validate without full parse

2. **Limitations of PyTorch/Safetensors**:
   - Designed for training, not optimized inference
   - No built-in quantization awareness
   - Large overhead for metadata
   - Not optimized for memory-mapped loading

**GGUF Design Goals**:

```
Key Features:
├── Rich Metadata System
│   ├── Model architecture info
│   ├── Tokenizer configuration
│   ├── Quantization details
│   └── Custom key-value pairs
├── Quantization-First Design
│   ├── Multiple quantization formats
│   ├── Block-wise quantization support
│   └── Efficient decompression
├── Memory-Mapped Loading
│   ├── Instant startup (no full load)
│   ├── OS-managed memory
│   └── Shared across processes
├── Extensibility
│   ├── Versioned format
│   ├── Backward compatible
│   └── Forward compatible (with unknown keys)
└── Single-File Distribution
    ├── Model weights + metadata + config
    └── Easy sharing and deployment
```

**Key Differences**:

| Feature | GGUF | PyTorch | Safetensors | GGML |
|---------|------|---------|-------------|------|
| Metadata | Rich KV system | Limited | JSON sidecar | Minimal |
| Quantization | Native support | Manual | Manual | Basic |
| Memory-mapping | Optimized | Possible | Yes | Yes |
| Extensibility | High | Medium | Medium | Low |
| File size | Small (quantized) | Large | Medium | Medium |

**Practical Impact**:
- Load 70B model in seconds (vs minutes with PyTorch)
- Run on consumer hardware (8GB RAM with quantization)
- Single-file distribution (no config files needed)
- Validate model without loading all weights

---

#### Follow-Up Questions

1. **"How would you convert a HuggingFace model to GGUF format?"**
   *Looking for*: Knowledge of conversion tools (convert.py), understanding of weight mapping

2. **"What metadata would you store in GGUF for a production deployment?"**
   *Looking for*: Version info, quantization method, validation checksums, provenance

3. **"How does GGUF handle different model architectures (LLaMA, Mistral, GPT)?"**
   *Looking for*: Architecture-agnostic design, metadata stores architecture type

4. **"What are the trade-offs of using GGUF vs keeping PyTorch checkpoints?"**
   *Looking for*: Inference vs training, flexibility vs optimization, ecosystem compatibility

---

#### Rubric

| Category | Poor (0-1) | Fair (2-3) | Good (4-5) | Excellent (6-7) |
|----------|-----------|-----------|-----------|----------------|
| **Definition** | Vague/incorrect | Basic definition | Clear explanation | Comprehensive with context |
| **Design Rationale** | Can't explain why | Mentions 1-2 reasons | Explains main goals | Compares to alternatives |
| **Technical Understanding** | Surface level | Understands basics | Knows key features | Deep technical insight |
| **Practical Application** | No examples | Generic examples | Relevant use cases | Production scenarios |
| **Communication** | Unclear | Understandable | Clear and structured | Teaches while explaining |

**Passing Score**: 12/35 (Entry), 20/35 (Mid), 28/35 (Senior)

---

#### Real Interview Insights

**From Hugging Face Interview (2024)**:
> "They asked me to explain why GGUF was better than their existing safetensors format for inference. Focus on the metadata system and memory-mapping optimizations."

**From Anthropic Interview (2024)**:
> "The interviewer wanted to know about quantization integration. Make sure you can explain how GGUF stores quantization parameters."

---

#### Related Content

- 📚 [Lesson 1.2: GGUF File Format](../../modules/01-foundations/docs/gguf-format.md)
- 💻 [Code: GGUF Metadata Reader](../../modules/01-foundations/code/gguf-reader.py)
- 🔬 [Lab 1.2: Exploring GGUF Files](../../modules/01-foundations/labs/lab-02-gguf-exploration.md)

---

### Question 2: Why Choose llama.cpp Over Other Inference Engines?

**Category**: Conceptual
**Difficulty**: Mid (L4/L5)
**Companies**: OpenAI, Anthropic, Cohere
**Time Allotted**: 15 minutes
**Prerequisites**: Module 1, Lesson 1.1

---

#### Question

You're building an LLM inference service. Compare llama.cpp with PyTorch, TensorFlow, vLLM, and TensorRT. When would you choose llama.cpp, and when would you choose alternatives? What are the key trade-offs?

---

#### What the Interviewer is Looking For

**Core Competencies Tested**:
- [ ] Understanding of inference engine landscape
- [ ] Trade-off analysis skills
- [ ] Production deployment experience
- [ ] Hardware constraints awareness
- [ ] Cost optimization thinking

**Red Flags**:
- ❌ "llama.cpp is always the best choice"
- ❌ Can't name specific use cases
- ❌ Doesn't understand GPU vs CPU trade-offs
- ❌ Ignores production requirements

**Green Flags**:
- ✅ Nuanced comparison across dimensions
- ✅ Specific use cases for each option
- ✅ Discusses hardware constraints
- ✅ Mentions deployment scenarios
- ✅ Considers operational costs

---

#### Hints (If Candidate is Stuck)

**Hint 1**: Deployment Targets
"Think about where you're deploying. Cloud GPU? Edge device? User's laptop?"

**Hint 2**: Performance Characteristics
"What matters more: maximum throughput or low latency? First token time or total generation time?"

**Hint 3**: Resource Constraints
"What if you only have CPU? What if you have 8x A100s?"

---

#### Model Solution

**Comparison Matrix**:

| Engine | Best For | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| **llama.cpp** | CPU inference, Edge, Low resource | Fast CPU, Low memory, Easy setup, Quantization | Limited batching, Slower GPU than vLLM |
| **vLLM** | High-throughput GPU serving | PagedAttention, Continuous batching, Throughput | Requires GPU, Higher complexity, Memory intensive |
| **TensorRT-LLM** | NVIDIA GPU optimization | Maximum GPU performance, FP8 support | NVIDIA only, Complex setup, Long build times |
| **PyTorch** | Development, Flexibility | Easy development, Flexible, Full ecosystem | Slow inference, High memory, Not production-optimized |
| **ONNX Runtime** | Cross-platform, Mobile | Broad hardware support, Optimization | Limited LLM features, Community support |

**Decision Framework**:

**Choose llama.cpp when**:
```
✅ CPU-only or CPU-primary inference
✅ Edge deployment (laptops, mobile, embedded)
✅ Low memory constraints (<16GB RAM)
✅ Simple deployment (single binary)
✅ Quantization is acceptable
✅ Individual user serving (vs batch serving)
✅ Cross-platform support needed

Examples:
- Desktop AI assistant
- Mobile app with on-device inference
- Raspberry Pi deployment
- Developer tools (code completion)
- Privacy-focused applications
```

**Choose vLLM when**:
```
✅ High-throughput GPU serving
✅ Many concurrent requests
✅ Cloud deployment with GPUs
✅ API server workloads
✅ Maximum GPU utilization critical
✅ Large batch processing

Examples:
- OpenAI-style API service
- Multi-tenant SaaS platform
- High-traffic chatbots
- Batch processing jobs
```

**Choose TensorRT-LLM when**:
```
✅ NVIDIA GPUs with maximum performance
✅ Production GPU deployment
✅ Latency-critical applications
✅ FP8 quantization needed
✅ Willing to invest in setup complexity

Examples:
- Real-time chat applications
- Voice assistants (low latency)
- Gaming AI (NPC dialog)
- Trading systems (fast decisions)
```

**Choose PyTorch when**:
```
✅ Development and experimentation
✅ Research and prototyping
✅ Custom model architectures
✅ Training and fine-tuning
✅ Not production deployment

Examples:
- Model development
- Research projects
- Proof of concepts
- Experimentation
```

**Real-World Scenario Example**:

```
Scenario: Code completion tool (like GitHub Copilot)

Requirements:
- Runs on developer laptops (CPU/GPU mix)
- Low latency (<100ms for first token)
- Works offline
- Limited memory (developers run IDEs)
- Cross-platform (Windows, Mac, Linux)

Decision: llama.cpp
Rationale:
✅ Excellent CPU performance
✅ Small quantized models (3-7B parameters)
✅ Fast startup time
✅ Low memory footprint
✅ Simple deployment (single binary)
✅ Works offline

Alternative considered: ONNX Runtime
- Good, but llama.cpp has better LLM optimizations
- Community and model support better for llama.cpp
```

**Trade-Off Analysis**:

| Dimension | llama.cpp | vLLM | TensorRT | PyTorch |
|-----------|-----------|------|----------|---------|
| CPU Performance | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐ |
| GPU Performance | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Memory Efficiency | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Ease of Setup | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Batch Throughput | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Development Flexibility | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

#### Follow-Up Questions

1. **"How would you combine multiple engines in production?"**
   *Looking for*: Edge with llama.cpp, cloud with vLLM, cost optimization

2. **"What metrics would you measure to validate your engine choice?"**
   *Looking for*: Latency (p50/p99), throughput, cost per token, memory usage

3. **"How does quantization affect your choice of inference engine?"**
   *Looking for*: llama.cpp's strength in quantization, GPU engines support FP8/INT8

---

#### Rubric

| Category | Poor (0-1) | Fair (2-3) | Good (4-5) | Excellent (6-7) |
|----------|-----------|-----------|-----------|----------------|
| **Comparison Depth** | Surface level | Knows 2-3 engines | Compares multiple dimensions | Comprehensive analysis |
| **Use Case Matching** | Generic | Some examples | Specific scenarios | Production-ready decisions |
| **Trade-off Analysis** | One-sided | Acknowledges trade-offs | Balanced analysis | Quantitative comparisons |
| **Practical Experience** | Theoretical only | Basic usage | Production experience | Deep operational knowledge |
| **Communication** | Unclear | Understandable | Structured | Teaches while explaining |

**Passing Score**: 12/35 (Entry), 20/35 (Mid), 28/35 (Senior)

---

#### Related Content

- 📚 [Lesson 1.1: Introduction to LLaMA-CPP](../../modules/01-foundations/docs/intro-llama-cpp.md)
- 💻 [Code: Performance Comparison](../../modules/01-foundations/code/engine-comparison.py)

---

### Question 3: Explain Quantization and Its Impact

**Category**: Conceptual
**Difficulty**: Mid (L4/L5)
**Companies**: OpenAI, Anthropic, Meta
**Time Allotted**: 15 minutes
**Prerequisites**: Module 1, Lessons 1.1-1.2

---

#### Question

Explain quantization in the context of LLM inference. What does "Q4_K_M" mean? How does quantization affect model quality, speed, and memory? When would you use Q4 vs Q8 vs FP16?

---

#### What the Interviewer is Looking For

**Core Competencies Tested**:
- [ ] Understanding of quantization fundamentals
- [ ] Knowledge of GGUF quantization formats
- [ ] Trade-off analysis (quality vs efficiency)
- [ ] Practical decision-making
- [ ] Memory calculation skills

**Red Flags**:
- ❌ Can't explain what quantization does
- ❌ Doesn't understand format naming (Q4_K_M)
- ❌ No awareness of quality impact
- ❌ Can't calculate memory savings

**Green Flags**:
- ✅ Explains bit-width reduction clearly
- ✅ Decodes quantization format names
- ✅ Discusses perplexity impact
- ✅ Calculates memory requirements
- ✅ Provides use-case recommendations

---

#### Hints (If Candidate is Stuck)

**Hint 1**: Basics
"Start with what quantization means. How many bits does FP16 use vs Q4?"

**Hint 2**: Format Naming
"What do you think the 'Q4' and 'K_M' parts of 'Q4_K_M' represent?"

**Hint 3**: Trade-offs
"If you reduce from 16 bits to 4 bits per parameter, what's the memory saving? What might you lose?"

---

#### Model Solution

**Quantization Definition**:
Quantization is the process of reducing the precision of model weights from high bit-width (FP32, FP16) to lower bit-width (INT8, INT4) to reduce memory usage and increase inference speed, with minimal quality degradation.

**How It Works**:
```
FP16 (16-bit floating point):
- Range: -65504 to +65504
- Precision: ~3-4 decimal digits
- Storage: 2 bytes per parameter

Q4_0 (4-bit quantization):
- Range: Defined by scale factor per block
- Precision: 16 discrete levels (2^4)
- Storage: 0.5 bytes per parameter
- Block-wise: Groups of 32 parameters share scale

Q8_0 (8-bit quantization):
- Range: Defined by scale factor per block
- Precision: 256 discrete levels (2^8)
- Storage: 1 byte per parameter
- Better quality, more memory
```

**GGUF Format Naming Decoded**:
```
Q4_K_M breakdown:
├── Q4: 4-bit quantization (average)
├── K: "K-quants" (newer quantization method)
└── M: Medium quality (vs S=Small, L=Large)

Common formats:
- Q4_0: Basic 4-bit, oldest method
- Q4_K_S: 4-bit k-quant, small (more compressed)
- Q4_K_M: 4-bit k-quant, medium (balanced)
- Q5_K_M: 5-bit k-quant, medium (better quality)
- Q8_0: 8-bit, nearly lossless
- F16: Full 16-bit (no quantization)
```

**Impact Analysis**:

**Memory Impact** (7B parameter model):
```
Format    | Bits | Size      | vs FP16
----------|------|-----------|--------
FP16      | 16   | ~14 GB    | 1.00x
Q8_0      | 8    | ~7 GB     | 0.50x
Q5_K_M    | ~5.5 | ~5 GB     | 0.36x
Q4_K_M    | ~4.5 | ~4 GB     | 0.29x
Q4_0      | 4    | ~3.5 GB   | 0.25x
IQ3_XXS   | ~3   | ~2.5 GB   | 0.18x
```

**Quality Impact** (perplexity on WikiText, lower is better):
```
Format    | Perplexity | Quality Loss
----------|------------|-------------
FP16      | 5.68       | Baseline (0%)
Q8_0      | 5.69       | Negligible (<1%)
Q6_K      | 5.72       | Minimal (~1%)
Q5_K_M    | 5.81       | Low (~2%)
Q4_K_M    | 6.02       | Moderate (~6%)
Q4_0      | 6.45       | Noticeable (~14%)
Q3_K_S    | 7.89       | Significant (~39%)
```

**Speed Impact**:
```
CPU Inference (tokens/sec, higher is better):
- Q4_K_M: ~15-20 tok/s (fastest on CPU)
- Q8_0: ~10-12 tok/s
- FP16: ~5-7 tok/s (slowest)

GPU Inference:
- Less dramatic difference
- Memory bandwidth still benefits
- Q4 vs FP16: ~1.5-2x faster
```

**Decision Framework**:

**Use Q4_K_M when**:
```
✅ Memory constrained (<8GB RAM)
✅ CPU inference primary
✅ Speed is critical
✅ Chat/instruction models (more robust to quantization)
✅ Consumer hardware deployment

Example: Desktop chatbot on laptop
```

**Use Q5_K_M / Q6_K when**:
```
✅ Moderate memory (8-16GB)
✅ Quality matters
✅ Still want quantization benefits
✅ Balanced use case

Example: Local AI assistant with good hardware
```

**Use Q8_0 when**:
```
✅ Quality is critical
✅ Some memory constraints (vs FP16)
✅ Nearly lossless needed
✅ Benchmarking baseline

Example: Production service with quality SLA
```

**Use FP16 when**:
```
✅ Maximum quality required
✅ No memory constraints
✅ GPU inference with plenty of VRAM
✅ Research/evaluation

Example: Quality comparison baseline
```

**Practical Example**:
```python
# Memory calculation for LLaMA-7B
model_params = 7_000_000_000

# FP16: 2 bytes per parameter
fp16_size = model_params * 2 / (1024**3)  # ~13 GB

# Q4_K_M: ~4.5 bits per parameter
q4_size = model_params * 4.5 / 8 / (1024**3)  # ~3.9 GB

# Savings
savings = 1 - (q4_size / fp16_size)  # ~70% reduction
```

---

#### Follow-Up Questions

1. **"How would you measure quality degradation from quantization?"**
   *Looking for*: Perplexity, task-specific metrics, human evaluation

2. **"What's the difference between Q4_0 and Q4_K_M?"**
   *Looking for*: K-quants use more sophisticated quantization, better quality

3. **"Can you quantize already quantized models further?"**
   *Looking for*: No, quality degrades significantly, start from FP16

4. **"How does quantization interact with context length?"**
   *Looking for*: Weights are quantized, but KV cache typically stays higher precision

---

#### Rubric

| Category | Poor (0-1) | Fair (2-3) | Good (4-5) | Excellent (6-7) |
|----------|-----------|-----------|-----------|----------------|
| **Conceptual Understanding** | Vague | Basic concept | Clear explanation | Deep technical insight |
| **Format Knowledge** | Can't decode | Knows some formats | Explains naming | Comprehensive comparison |
| **Trade-off Analysis** | One dimension | Memory and quality | Multi-dimensional | Quantitative analysis |
| **Practical Application** | No examples | Generic cases | Specific recommendations | Production scenarios |
| **Communication** | Unclear | Understandable | Structured | Teaches effectively |

**Passing Score**: 12/35 (Entry), 20/35 (Mid), 28/35 (Senior)

---

#### Related Content

- 📚 [Lesson 1.2: GGUF File Format](../../modules/01-foundations/docs/gguf-format.md)
- 📚 [Module 3: Quantization & Optimization](../../modules/03-quantization/)
- 🔬 [Lab 3.1: Quantization Fundamentals](../../modules/03-quantization/labs/lab-01-quantization.md)

---

### Question 4: Context Window and KV Cache

**Category**: Conceptual
**Difficulty**: Mid (L4/L5)
**Companies**: OpenAI, Anthropic, Cohere
**Time Allotted**: 15 minutes
**Prerequisites**: Module 1, Lessons 1.4-1.5

---

#### Question

Explain what a context window is and how the KV cache works in transformer-based inference. Why does memory usage grow during generation? How would you calculate the memory needed for inference with a 4096 context window?

---

#### What the Interviewer is Looking For

**Core Competencies Tested**:
- [ ] Understanding of transformer inference
- [ ] KV cache purpose and mechanics
- [ ] Memory calculation skills
- [ ] Awareness of context window limitations
- [ ] Optimization thinking

**Red Flags**:
- ❌ Confuses context window with batch size
- ❌ Doesn't know what KV cache is
- ❌ Can't calculate memory requirements
- ❌ No awareness of memory growth during generation

**Green Flags**:
- ✅ Explains attention mechanism context
- ✅ Describes KV cache optimization
- ✅ Calculates memory accurately
- ✅ Discusses trade-offs (context vs memory)
- ✅ Mentions optimization techniques (PagedAttention)

---

#### Hints (If Candidate is Stuck)

**Hint 1**: Attention Basics
"In self-attention, what does each token need to attend to? How does this relate to context?"

**Hint 2**: Caching
"What computation is repeated every time you generate a new token? What could you cache?"

**Hint 3**: Memory Calculation
"For each token, how many key-value pairs do you store per layer? How many layers?"

---

#### Model Solution

**Context Window Definition**:
The context window is the maximum number of tokens a model can "see" and process at once. It represents the span of text the model uses when generating each new token.

```
Example with 4096 context window:
┌─────────────────────────────────────────────┐
│  Token 1 ... Token 4096                     │ ← All visible
│  "Once upon ... and they lived happily"     │
│                                     ↓       │
│                            Next token: " ever" │
└─────────────────────────────────────────────┘

If you try to add token 4097, token 1 gets dropped
```

**Why Context Window Matters**:
```
Short context (2048):
✅ Less memory
✅ Faster inference
❌ Limited conversation history
❌ Can't process long documents

Long context (32768+):
✅ Long conversations
✅ Process entire documents
❌ High memory usage
❌ Slower inference
```

**KV Cache Explanation**:

The KV cache is an optimization that stores computed Key and Value tensors from previous tokens to avoid recomputing them during autoregressive generation.

**Without KV Cache (inefficient)**:
```
Generating token 100:
1. Recompute K,V for tokens 1-99 in every layer ❌
2. Compute K,V for token 100
3. Perform attention
4. Generate token 100

Time complexity: O(n²) where n = sequence length
```

**With KV Cache (optimized)**:
```
Generating token 100:
1. Load cached K,V for tokens 1-99 from memory ✅
2. Compute K,V only for token 100
3. Append to cache
4. Perform attention
5. Generate token 100

Time complexity: O(n)
```

**KV Cache Structure**:
```
For each layer:
┌─────────────────────────────────────┐
│ Keys:   [seq_len, n_heads, head_dim] │
│ Values: [seq_len, n_heads, head_dim] │
└─────────────────────────────────────┘

For all layers:
Total cache = 2 × n_layers × seq_len × n_heads × head_dim
```

**Memory Growth During Generation**:

```
Initial prompt (10 tokens):
├── Model weights: Fixed (e.g., 4 GB)
├── KV cache: Small (10 tokens)
└── Activations: Small

After generating 1000 tokens:
├── Model weights: Fixed (4 GB) ← Same
├── KV cache: Large (1010 tokens) ← Grew 100x!
└── Activations: Small ← Same

Memory usage grows linearly with generated tokens!
```

**Memory Calculation Example** (LLaMA-7B with 4096 context):

```python
# Model: LLaMA-7B
n_layers = 32
n_heads = 32
head_dim = 128  # 4096 / 32
n_ctx = 4096    # context window
precision = 2   # FP16 = 2 bytes

# KV cache size
kv_cache_size = (
    2 *           # K and V
    n_layers *    # 32 layers
    n_ctx *       # 4096 tokens
    n_heads *     # 32 heads
    head_dim *    # 128 dimensions
    precision     # 2 bytes
)

# Calculate
kv_cache_size = 2 * 32 * 4096 * 32 * 128 * 2
kv_cache_size_gb = kv_cache_size / (1024**3)
print(f"KV cache: {kv_cache_size_gb:.2f} GB")  # ~4 GB

# Total memory estimate (Q4_K_M quantization)
model_weights = 4.0   # GB
kv_cache = 4.0        # GB
activations = 0.5     # GB
overhead = 0.5        # GB
total = model_weights + kv_cache + activations + overhead
print(f"Total: {total:.1f} GB")  # ~9 GB
```

**Optimizations**:

1. **Multi-Query Attention (MQA)**:
   - Share K,V across heads
   - Reduces KV cache by factor of n_heads
   - Example: 4 GB → 128 MB (32x reduction)

2. **Grouped-Query Attention (GQA)**:
   - Group heads, share K,V within groups
   - Moderate reduction (e.g., 8 groups → 4x reduction)

3. **PagedAttention (vLLM)**:
   - Store KV cache in pages
   - Reduce fragmentation
   - Enable sharing across requests

4. **Sliding Window**:
   - Keep only recent N tokens
   - Constant memory
   - Lose long-range context

**Real-World Impact**:
```
Chatbot with 100 users:
- Each user conversation: 2048 tokens
- Each KV cache: ~2 GB
- Total: 200 GB just for KV cache!

Solutions:
- PagedAttention (sharing)
- Evict inactive conversations
- Use GQA models (less cache per user)
```

---

#### Follow-Up Questions

1. **"How does batch size interact with KV cache memory?"**
   *Looking for*: Each batch item has own cache, multiplies memory

2. **"What happens to KV cache when context window is exceeded?"**
   *Looking for*: Depends on implementation (truncate, sliding window, error)

3. **"How would you optimize for long conversations (100K+ tokens)?"**
   *Looking for*: Sparse attention, compression, retrieval, summarization

4. **"Compare KV cache memory for MHA vs GQA vs MQA"**
   *Looking for*: GQA uses 1/4 to 1/8 memory, MQA minimal but quality loss

---

#### Rubric

| Category | Poor (0-1) | Fair (2-3) | Good (4-5) | Excellent (6-7) |
|----------|-----------|-----------|-----------|----------------|
| **Conceptual Understanding** | Vague | Basic concept | Clear explanation | Deep technical insight |
| **KV Cache Mechanics** | Doesn't understand | Basic understanding | Explains optimization | Implementation details |
| **Memory Calculation** | Can't calculate | Rough estimate | Accurate calculation | Includes all components |
| **Optimization Awareness** | No knowledge | Mentions one | Multiple techniques | Compares trade-offs |
| **Communication** | Unclear | Understandable | Structured | Teaches effectively |

**Passing Score**: 12/35 (Entry), 20/35 (Mid), 28/35 (Senior)

---

#### Related Content

- 📚 [Lesson 1.4: Basic Inference](../../modules/01-foundations/docs/basic-inference.md)
- 📚 [Lesson 1.5: Memory Management](../../modules/01-foundations/docs/memory-management.md)
- 📚 [Module 2: KV Cache Implementation](../../modules/02-core-implementation/docs/kv-cache.md)

---

### Question 5: Build Systems and Cross-Compilation

**Category**: Conceptual
**Difficulty**: Entry to Mid (L3/L4)
**Companies**: Meta, Google, Startups
**Time Allotted**: 10-15 minutes
**Prerequisites**: Module 1, Lesson 1.3

---

#### Question

Explain the llama.cpp build system. What is CMake and why is it used? What are the key compilation flags you'd need to know? How would you enable CUDA support during the build?

---

#### What the Interviewer is Looking For

**Core Competencies Tested**:
- [ ] Understanding of build systems
- [ ] Knowledge of compilation process
- [ ] CMake configuration experience
- [ ] Backend selection awareness
- [ ] Troubleshooting build issues

**Red Flags**:
- ❌ Doesn't know what CMake is
- ❌ Can't explain build vs runtime configuration
- ❌ No awareness of backend options
- ❌ Can't troubleshoot basic build errors

**Green Flags**:
- ✅ Explains CMake's role clearly
- ✅ Knows key compilation flags
- ✅ Can enable different backends
- ✅ Mentions optimization flags
- ✅ Troubleshooting experience

---

#### Hints (If Candidate is Stuck)

**Hint 1**: CMake Purpose
"Think about cross-platform development. How do you build C++ code on Windows, Mac, and Linux?"

**Hint 2**: Backends
"llama.cpp can run on CPU, NVIDIA GPUs, AMD GPUs, Apple Metal. How does the build system handle this?"

**Hint 3**: Flags
"What flags might control optimization level, backend selection, or feature enablement?"

---

#### Model Solution

**CMake Overview**:
CMake is a cross-platform build system generator that produces native build files (Makefiles on Linux, Xcode projects on Mac, Visual Studio solutions on Windows) from a platform-independent configuration.

**Why llama.cpp Uses CMake**:
```
Challenges:
├── Multiple platforms (Linux, Windows, macOS, FreeBSD)
├── Multiple compilers (GCC, Clang, MSVC, MinGW)
├── Multiple backends (CPU, CUDA, Metal, OpenCL, SYCL)
├── Optional features (server, BLAS, GPU)
└── Complex dependencies

CMake Solution:
├── Single CMakeLists.txt
├── Platform detection
├── Compiler-specific flags
├── Conditional compilation
└── Dependency management
```

**Build Process**:

```bash
# 1. Configure (generate build files)
cmake -B build -DLLAMA_CUDA=ON

# 2. Build (compile)
cmake --build build --config Release

# 3. Install (optional)
cmake --install build
```

**Key CMake Options**:

```cmake
# Backend Selection
-DLLAMA_CUDA=ON          # Enable CUDA (NVIDIA GPU)
-DLLAMA_METAL=ON         # Enable Metal (Apple Silicon)
-DLLAMA_HIPBLAS=ON       # Enable ROCm (AMD GPU)
-DLLAMA_SYCL=ON          # Enable SYCL (Intel GPU)
-DLLAMA_OPENCL=ON        # Enable OpenCL
-DLLAMA_VULKAN=ON        # Enable Vulkan

# CPU Optimization
-DLLAMA_AVX=ON           # AVX instructions
-DLLAMA_AVX2=ON          # AVX2 instructions
-DLLAMA_AVX512=ON        # AVX512 instructions
-DLLAMA_FMA=ON           # Fused multiply-add
-DLLAMA_F16C=ON          # FP16 conversion

# BLAS Libraries (CPU acceleration)
-DLLAMA_BLAS=ON          # Generic BLAS
-DLLAMA_OPENBLAS=ON      # OpenBLAS
-DLLAMA_BLIS=ON          # BLIS library

# Features
-DLLAMA_SERVER=ON        # Build HTTP server
-DBUILD_SHARED_LIBS=ON   # Shared library
-DLLAMA_STATIC=ON        # Static linking

# Build Type
-DCMAKE_BUILD_TYPE=Release    # Optimized build
-DCMAKE_BUILD_TYPE=Debug      # Debug symbols
-DCMAKE_BUILD_TYPE=RelWithDebInfo  # Both
```

**Enabling CUDA Support** (Step-by-step):

```bash
# Prerequisites:
# 1. NVIDIA GPU with compute capability 6.0+
# 2. CUDA Toolkit installed (11.0+)
# 3. nvcc compiler in PATH

# Check CUDA installation
nvcc --version

# Configure with CUDA
cmake -B build \
    -DLLAMA_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=native  # Or specific: "80;86"

# Build
cmake --build build --config Release -j 8

# Verify CUDA support
./build/bin/llama-cli --version  # Should show "CUDA"

# Test
./build/bin/llama-cli \
    -m models/model.gguf \
    -ngl 32  # Offload 32 layers to GPU
```

**Common Build Configurations**:

**1. Maximum CPU Performance**:
```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_AVX2=ON \
    -DLLAMA_FMA=ON \
    -DLLAMA_F16C=ON \
    -DLLAMA_OPENBLAS=ON
```

**2. NVIDIA GPU (CUDA)**:
```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89"  # A100, RTX 40xx, H100
```

**3. Apple Silicon (Metal)**:
```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_METAL=ON
```

**4. Cross-compilation (ARM on x86)**:
```bash
cmake -B build \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
```

**5. Development Build (with server)**:
```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLAMA_SERVER=ON \
    -DLLAMA_CUDA=ON \
    -DBUILD_SHARED_LIBS=ON
```

**Troubleshooting Common Issues**:

**1. CUDA not found**:
```bash
# Set CUDA path
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
cmake -B build -DLLAMA_CUDA=ON
```

**2. Wrong CUDA architecture**:
```bash
# Error: unsupported GPU architecture 'compute_XY'
# Fix: Specify supported architectures
cmake -B build \
    -DLLAMA_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86"
```

**3. Missing dependencies (OpenBLAS)**:
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# macOS
brew install openblas

# Then rebuild
cmake -B build -DLLAMA_OPENBLAS=ON
```

**4. Compilation errors with AVX512**:
```bash
# CPU doesn't support AVX512
# Disable it
cmake -B build \
    -DLLAMA_AVX2=ON \
    -DLLAMA_AVX512=OFF
```

---

#### Follow-Up Questions

1. **"How do you verify which backend is being used at runtime?"**
   *Looking for*: `--version` flag, logs, check for CUDA/Metal initialization

2. **"What's the difference between -DLLAMA_CUDA=ON at build time and -ngl at runtime?"**
   *Looking for*: Build enables support, runtime flag controls layer offloading

3. **"How would you optimize build time for development?"**
   *Looking for*: ccache, ninja generator, parallel builds (-j), partial rebuilds

4. **"What happens if you build with CUDA but run on a machine without GPU?"**
   *Looking for*: Fallback to CPU, error handling, runtime checks

---

#### Rubric

| Category | Poor (0-1) | Fair (2-3) | Good (4-5) | Excellent (6-7) |
|----------|-----------|-----------|-----------|----------------|
| **CMake Understanding** | Doesn't know | Basic knowledge | Explains purpose | Deep understanding |
| **Flag Knowledge** | No flags known | Knows 2-3 flags | Knows major categories | Comprehensive knowledge |
| **Backend Configuration** | Can't configure | Basic CUDA setup | Multiple backends | Advanced scenarios |
| **Troubleshooting** | No experience | Basic issues | Common problems | Complex debugging |
| **Communication** | Unclear | Understandable | Structured | Clear and detailed |

**Passing Score**: 10/35 (Entry), 18/35 (Mid), 25/35 (Senior)

---

#### Related Content

- 📚 [Lesson 1.3: Build System & Toolchain](../../modules/01-foundations/docs/build-system.md)
- 🔬 [Lab 1.3: Build Configuration](../../modules/01-foundations/labs/lab-03-build-config.md)

---

## Technical Questions

### Question 6: Calculate Memory Requirements

**Category**: Technical
**Difficulty**: Mid (L4/L5)
**Companies**: OpenAI, Anthropic, Meta
**Time Allotted**: 20 minutes
**Prerequisites**: Module 1, Lessons 1.4-1.5

---

#### Question

You need to run LLaMA-70B on a machine with 64GB RAM. Calculate:
1. Memory needed for Q4_K_M quantization
2. Memory needed for KV cache (8192 context, FP16)
3. Total memory required for inference
4. Whether it will fit in 64GB and if not, what you'd change

Show your calculations and reasoning.

---

#### What the Interviewer is Looking For

**Core Competencies Tested**:
- [ ] Memory calculation accuracy
- [ ] Understanding of memory components
- [ ] Quantization impact knowledge
- [ ] Problem-solving under constraints
- [ ] Practical optimization thinking

**Red Flags**:
- ❌ Can't structure the calculation
- ❌ Forgets KV cache or activations
- ❌ Wrong quantization bits
- ❌ No optimization suggestions

**Green Flags**:
- ✅ Systematic calculation approach
- ✅ Accounts for all memory components
- ✅ Accurate arithmetic
- ✅ Practical optimization suggestions
- ✅ Considers overhead and safety margins

---

#### Hints (If Candidate is Stuck)

**Hint 1**: Components
"What are the main memory users in inference? Weights, KV cache, what else?"

**Hint 2**: Quantization Bits
"Q4_K_M uses approximately 4.5 bits per parameter. How do you convert that to bytes?"

**Hint 3**: Model Architecture
"LLaMA-70B has 70 billion parameters. It also has 80 layers and specific attention configuration."

---

#### Model Solution

**LLaMA-70B Architecture**:
```
Parameters: 70 billion (70B)
Layers: 80
Heads: 64
Head dimension: 128
Hidden size: 8192 (64 * 128)
Vocabulary: ~32000
Context window: 4096 (default), using 8192 in question
```

**Step 1: Model Weights (Q4_K_M)**

```python
# Q4_K_M averages ~4.5 bits per parameter
params = 70_000_000_000
bits_per_param = 4.5
bytes_per_param = bits_per_param / 8  # 0.5625 bytes

model_size_bytes = params * bytes_per_param
model_size_gb = model_size_bytes / (1024**3)

# Calculation:
model_size_gb = 70e9 * 4.5 / 8 / (1024**3)
                = 39,375,000,000 / (1024**3)
                ≈ 36.68 GB

# Round up for safety
model_weights = ~37 GB
```

**Step 2: KV Cache (8192 context, FP16)**

```python
# LLaMA-70B parameters
n_layers = 80
n_heads = 64
head_dim = 128
n_ctx = 8192
precision = 2  # FP16 = 2 bytes

# KV cache formula:
# 2 (K and V) * layers * context * heads * head_dim * precision

kv_cache_bytes = 2 * n_layers * n_ctx * n_heads * head_dim * precision

# Calculation:
kv_cache_bytes = 2 * 80 * 8192 * 64 * 128 * 2
               = 2 * 80 * 8192 * 16384
               = 21,474,836,480 bytes

kv_cache_gb = 21,474,836,480 / (1024**3)
            ≈ 20.0 GB

# KV cache = ~20 GB
```

**Step 3: Activations and Overhead**

```python
# Activations (temporary tensors during computation)
# Rough estimate: ~2% of model size for batch_size=1
activations = model_weights * 0.02
            ≈ 37 * 0.02
            ≈ 0.74 GB

# System overhead (OS, libraries, buffers)
# Estimate: ~2-3 GB
overhead = 2.5 GB

# Input/output buffers
io_buffers = 0.5 GB
```

**Step 4: Total Memory**

```
Component          | Memory (GB) | Notes
-------------------|-------------|------------------
Model weights      | 37.0        | Q4_K_M quantized
KV cache (8192)    | 20.0        | FP16 precision
Activations        | 0.7         | Batch size = 1
I/O buffers        | 0.5         | Token buffers
System overhead    | 2.5         | OS, libraries
-------------------|-------------|------------------
TOTAL              | 60.7 GB     |
```

**Step 5: Will It Fit?**

```
Available RAM: 64 GB
Required RAM:  60.7 GB
Margin:        3.3 GB (5.4%)

Answer: Yes, but BARELY!
Risk: Very tight, could OOM with any variation
```

**Issues with This Configuration**:
```
❌ Only 5% margin (unsafe)
❌ No room for multi-user or batching
❌ OS might struggle under memory pressure
❌ Swap usage could slow inference dramatically
❌ One memory spike = crash
```

**Optimizations to Consider**:

**Option 1: Reduce Quantization (More Compression)**
```
Use Q3_K_M instead of Q4_K_M:
- Saves: ~12 GB on weights (37 GB → 25 GB)
- Cost: 10-15% quality degradation
- New total: ~49 GB ✅ (23% margin)
```

**Option 2: Reduce Context Window**
```
Use 4096 context instead of 8192:
- Saves: 10 GB on KV cache (20 GB → 10 GB)
- Cost: Shorter conversation history
- New total: ~51 GB ✅ (20% margin)
```

**Option 3: Quantize KV Cache**
```
Use FP8 or INT8 for KV cache:
- Saves: 10 GB (20 GB → 10 GB)
- Cost: Minor quality impact, implementation complexity
- New total: ~51 GB ✅
```

**Option 4: Smaller Model**
```
Use LLaMA-34B or LLaMA-13B:
- LLaMA-34B Q4_K_M: ~18 GB weights
  Total with 8K context: ~41 GB ✅
- LLaMA-13B Q4_K_M: ~7 GB weights
  Total with 8K context: ~32 GB ✅
```

**Recommended Solution**:
```
Compromise: Q4_K_M + 4096 context

Component          | Memory (GB)
-------------------|------------
Model weights      | 37.0        Q4_K_M
KV cache (4096)    | 10.0        FP16, halved context
Activations        | 0.7
I/O buffers        | 0.5
System overhead    | 2.5
-------------------|------------
TOTAL              | 50.7 GB

Margin: 13.3 GB (20.8%) ✅ Much safer!
```

**Complete Calculation Script**:
```python
def calculate_memory(
    params_billions,
    n_layers,
    n_heads,
    head_dim,
    n_ctx,
    quant_bits=4.5,
    kv_precision=2
):
    """Calculate total memory for LLM inference."""

    # Model weights
    params = params_billions * 1e9
    model_gb = params * quant_bits / 8 / (1024**3)

    # KV cache
    kv_bytes = 2 * n_layers * n_ctx * n_heads * head_dim * kv_precision
    kv_gb = kv_bytes / (1024**3)

    # Activations (~2% of model)
    activations_gb = model_gb * 0.02

    # Overhead
    io_gb = 0.5
    system_gb = 2.5

    total_gb = model_gb + kv_gb + activations_gb + io_gb + system_gb

    return {
        'model_weights_gb': model_gb,
        'kv_cache_gb': kv_gb,
        'activations_gb': activations_gb,
        'io_buffers_gb': io_gb,
        'system_overhead_gb': system_gb,
        'total_gb': total_gb
    }

# LLaMA-70B with 8192 context, Q4_K_M
result = calculate_memory(
    params_billions=70,
    n_layers=80,
    n_heads=64,
    head_dim=128,
    n_ctx=8192,
    quant_bits=4.5,
    kv_precision=2
)

print(f"Total memory: {result['total_gb']:.1f} GB")
```

---

#### Follow-Up Questions

1. **"How would batch size affect these calculations?"**
   *Looking for*: KV cache multiplies by batch size, weights stay same

2. **"What if we use Grouped-Query Attention (GQA) like LLaMA-3?"**
   *Looking for*: Reduces KV cache by number of GQA groups (e.g., 8x reduction)

3. **"How do you monitor actual memory usage during inference?"**
   *Looking for*: `top`, `nvidia-smi`, process monitors, llama.cpp logs

4. **"At what point would you need multiple GPUs instead of CPU?"**
   *Looking for*: When model doesn't fit, or when speed is critical

---

#### Rubric

| Category | Poor (0-1) | Fair (2-3) | Good (4-5) | Excellent (6-7) |
|----------|-----------|-----------|-----------|----------------|
| **Calculation Accuracy** | Major errors | Minor errors | Mostly correct | Perfect calculations |
| **Component Coverage** | Misses major parts | Includes weights+cache | All components | Plus safety margins |
| **Problem Solving** | No solutions | One solution | Multiple options | Trade-off analysis |
| **Practical Understanding** | Theoretical only | Basic practical | Real-world aware | Production-ready |
| **Communication** | Unclear | Shows work | Structured | Clear + documented |

**Passing Score**: 12/35 (Entry), 22/35 (Mid), 30/35 (Senior)

---

#### Related Content

- 📚 [Lesson 1.5: Memory Management](../../modules/01-foundations/docs/memory-management.md)
- 💻 [Code: Memory Calculator](../../modules/01-foundations/code/memory-calculator.py)
- 🔬 [Lab 1.5: Memory Profiling](../../modules/01-foundations/labs/lab-05-memory.md)

---

### Question 7: Implement Token Generation Loop

**Category**: Technical / Coding
**Difficulty**: Mid (L4/L5)
**Companies**: OpenAI, Anthropic, Cohere
**Time Allotted**: 30 minutes
**Prerequisites**: Module 1, Lesson 1.4

---

#### Question

Write pseudocode (or Python) for a basic token generation loop using llama.cpp. Your implementation should:
1. Load a model
2. Tokenize input prompt
3. Generate N tokens autoregressively
4. Handle sampling (temperature, top-k, top-p)
5. Detect end-of-sequence

Explain each step and discuss where performance bottlenecks might be.

---

#### What the Interviewer is Looking For

**Core Competencies Tested**:
- [ ] Understanding of autoregressive generation
- [ ] Knowledge of sampling methods
- [ ] API usage familiarity
- [ ] Code structure and clarity
- [ ] Performance awareness

**Red Flags**:
- ❌ Doesn't understand autoregressive loop
- ❌ Forgets to handle EOS token
- ❌ No understanding of sampling
- ❌ Inefficient implementation (no caching)

**Green Flags**:
- ✅ Clear loop structure
- ✅ Proper sampling implementation
- ✅ Handles edge cases (EOS, max length)
- ✅ Mentions KV cache usage
- ✅ Discusses performance optimizations

---

#### Hints (If Candidate is Stuck)

**Hint 1**: Loop Structure
"What's the basic pattern? Predict next token, append, repeat. What's the exit condition?"

**Hint 2**: Sampling
"After getting logits for next token, how do you apply temperature and top-k/top-p?"

**Hint 3**: End Detection
"How do you know when generation should stop? Multiple conditions?"

---

#### Model Solution

**Pseudocode Implementation**:

```python
from llama_cpp import Llama
import numpy as np

class TokenGenerator:
    def __init__(self, model_path: str, n_ctx: int = 2048):
        """Initialize model for generation."""
        # Load model
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,           # Context window
            n_batch=512,           # Batch size for prompt processing
            n_threads=8,           # CPU threads
            n_gpu_layers=0,        # 0 = CPU only
            use_mmap=True,         # Memory-map model
            use_mlock=False,       # Don't lock memory
            verbose=False
        )

        # Get special tokens
        self.bos_token = self.model.token_bos()
        self.eos_token = self.model.token_eos()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1
    ) -> str:
        """
        Generate text autoregressively.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy, higher = more random)
            top_k: Keep only top-k tokens for sampling
            top_p: Nucleus sampling probability mass
            repeat_penalty: Penalty for repeating tokens

        Returns:
            Generated text
        """
        # Step 1: Tokenize prompt
        tokens = self.model.tokenize(
            prompt.encode('utf-8'),
            add_bos=True  # Add beginning-of-sequence token
        )

        # Step 2: Initialize generation
        generated_tokens = []
        context_tokens = tokens.copy()

        # Step 3: Autoregressive generation loop
        for i in range(max_tokens):
            # 3a. Forward pass (compute logits for next token)
            # Note: KV cache is automatically managed by llama.cpp
            logits = self.model.eval(context_tokens)

            # 3b. Get logits for last position (next token prediction)
            next_token_logits = logits[-1]  # Shape: [vocab_size]

            # 3c. Apply repeat penalty
            if repeat_penalty != 1.0:
                next_token_logits = self._apply_repeat_penalty(
                    next_token_logits,
                    context_tokens,
                    repeat_penalty
                )

            # 3d. Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # 3e. Convert logits to probabilities
            probs = self._softmax(next_token_logits)

            # 3f. Apply top-k filtering
            if top_k > 0:
                probs = self._top_k_filtering(probs, top_k)

            # 3g. Apply top-p (nucleus) filtering
            if top_p < 1.0:
                probs = self._top_p_filtering(probs, top_p)

            # 3h. Renormalize probabilities
            probs = probs / np.sum(probs)

            # 3i. Sample next token
            next_token = np.random.choice(len(probs), p=probs)

            # 3j. Check for end-of-sequence
            if next_token == self.eos_token:
                break

            # 3k. Append token to generation
            generated_tokens.append(next_token)
            context_tokens.append(next_token)

            # 3l. Check context window limit
            if len(context_tokens) >= self.model.n_ctx():
                # Simple truncation (could use sliding window)
                context_tokens = context_tokens[-self.model.n_ctx():]

        # Step 4: Decode generated tokens to text
        generated_text = self.model.detokenize(generated_tokens).decode('utf-8')

        return generated_text

    def _apply_repeat_penalty(
        self,
        logits: np.ndarray,
        context: list,
        penalty: float
    ) -> np.ndarray:
        """Apply penalty to tokens that appear in context."""
        penalized = logits.copy()
        for token in set(context):
            # Reduce probability of repeated tokens
            if penalized[token] > 0:
                penalized[token] /= penalty
            else:
                penalized[token] *= penalty
        return penalized

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities."""
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / np.sum(exp_logits)

    def _top_k_filtering(self, probs: np.ndarray, k: int) -> np.ndarray:
        """Keep only top-k highest probability tokens."""
        top_k_indices = np.argsort(probs)[-k:]
        filtered = np.zeros_like(probs)
        filtered[top_k_indices] = probs[top_k_indices]
        return filtered

    def _top_p_filtering(self, probs: np.ndarray, p: float) -> np.ndarray:
        """Nucleus sampling: keep tokens with cumulative prob <= p."""
        sorted_indices = np.argsort(probs)[::-1]  # Descending order
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # Find cutoff index
        cutoff_idx = np.searchsorted(cumulative_probs, p) + 1

        # Keep tokens up to cutoff
        filtered = np.zeros_like(probs)
        filtered[sorted_indices[:cutoff_idx]] = probs[sorted_indices[:cutoff_idx]]
        return filtered


# Usage example
generator = TokenGenerator("models/llama-7b-q4_k_m.gguf")

prompt = "Once upon a time"
generated = generator.generate(
    prompt=prompt,
    max_tokens=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9
)

print(f"Prompt: {prompt}")
print(f"Generated: {generated}")
```

**Performance Bottlenecks Analysis**:

**1. Forward Pass (Compute Logits)**:
```
Bottleneck: Matrix multiplications (80-90% of time)
- Attention: O(n²d) where n=context, d=hidden_dim
- FFN: O(nd²)

Optimizations:
✅ Use KV cache (avoid recomputing past tokens)
✅ Quantization (Q4_K_M reduces computation)
✅ BLAS libraries (OpenBLAS, MKL)
✅ GPU offloading (-ngl flag)
```

**2. Sampling (Top-k, Top-p)**:
```
Bottleneck: Sorting vocabulary (~5-10% of time for large vocabs)
- Vocab size: 32K-128K tokens
- Sorting: O(V log V)

Optimizations:
✅ Partial sorting (only top-k needed)
✅ Heap-based selection
✅ Cache sorted indices if temp=0 (greedy)
```

**3. Tokenization**:
```
Bottleneck: Initial prompt tokenization (once per generation)
- BPE/SentencePiece: O(n²) worst case

Optimizations:
✅ Cache tokenized prompts for repeated use
✅ Optimize tokenizer implementation
```

**4. Detokenization**:
```
Bottleneck: Converting tokens back to text (minor)
- Linear in number of tokens

Optimizations:
✅ Stream tokens as generated (don't wait)
✅ Use efficient string builders
```

**5. Memory Bandwidth**:
```
Bottleneck: Loading weights from RAM (CPU) or VRAM (GPU)
- 70B model: ~140 GB of data movement (FP16)

Optimizations:
✅ Memory-mapped files (mmap)
✅ Quantization (4x less data)
✅ Keep model in cache (warm-up)
```

**Timing Breakdown** (Example: LLaMA-7B Q4_K_M, CPU, 100 tokens):
```
Component           | Time (ms) | Percentage
--------------------|-----------|------------
Forward pass        | 2500      | 85%
Sampling            | 200       | 7%
Tokenization        | 100       | 3%
Detokenization      | 50        | 2%
Overhead            | 100       | 3%
--------------------|-----------|------------
Total               | 2950      | 100%

Tokens/second: ~34 tok/s
```

**Key Insights**:
1. **KV Cache is Critical**: Without it, generation would be O(n²) instead of O(n)
2. **Matrix Ops Dominate**: 85%+ of time is in forward pass
3. **Sampling is Cheap**: Only 7% despite top-k/top-p complexity
4. **First Token Latency**: Prompt processing time matters for UX

---

#### Follow-Up Questions

1. **"How would you implement streaming generation (output tokens as generated)?"**
   *Looking for*: Callback functions, yield tokens, real-time display

2. **"What changes for batch inference (multiple prompts simultaneously)?"**
   *Looking for*: Batch forward pass, KV cache per sequence, padding

3. **"How do you implement greedy decoding (deterministic)?"**
   *Looking for*: temperature=0, or argmax instead of sampling

4. **"What's the difference between eval() and generate() in llama.cpp API?"**
   *Looking for*: eval() computes logits, generate() is full loop

---

#### Rubric

| Category | Poor (0-1) | Fair (2-3) | Good (4-5) | Excellent (6-7) |
|----------|-----------|-----------|-----------|----------------|
| **Loop Structure** | Incorrect | Basic loop | Correct flow | Production-ready |
| **Sampling Implementation** | Missing/wrong | Basic sampling | Multiple methods | Optimized |
| **Edge Cases** | Ignored | Handles EOS | Multiple conditions | Robust handling |
| **Performance Awareness** | No mention | Basic understanding | Identifies bottlenecks | Optimization ideas |
| **Code Quality** | Messy | Readable | Well-structured | Documented + clean |

**Passing Score**: 12/35 (Entry), 22/35 (Mid), 30/35 (Senior)

---

#### Related Content

- 📚 [Lesson 1.4: Basic Inference](../../modules/01-foundations/docs/basic-inference.md)
- 💻 [Code: Token Generation](../../modules/01-foundations/code/token-generation.py)
- 🔬 [Lab 1.4: First Inference](../../modules/01-foundations/labs/lab-04-inference.md)

---

### Question 8: GGUF Metadata Parsing

**Category**: Technical / Coding
**Difficulty**: Mid (L4/L5)
**Companies**: Hugging Face, Meta, Startups
**Time Allotted**: 25 minutes
**Prerequisites**: Module 1, Lesson 1.2

---

#### Question

Explain the structure of a GGUF file's header and metadata section. Write code (or detailed pseudocode) to read the following from a GGUF file:
1. Magic number validation
2. Version
3. Number of tensors
4. Metadata key-value pairs

What would you check to ensure the file is valid?

---

#### What the Interviewer is Looking For

**Core Competencies Tested**:
- [ ] Understanding of binary file formats
- [ ] GGUF specification knowledge
- [ ] Low-level programming skills
- [ ] Error handling and validation
- [ ] Attention to detail

**Red Flags**:
- ❌ Doesn't know GGUF structure
- ❌ Can't handle binary data
- ❌ No validation/error checking
- ❌ Confuses endianness

**Green Flags**:
- ✅ Accurate structure knowledge
- ✅ Proper binary parsing
- ✅ Comprehensive validation
- ✅ Handles edge cases
- ✅ Clean, readable code

---

#### Hints (If Candidate is Stuck)

**Hint 1**: Structure
"GGUF starts with a magic number. What comes next? Version, then counts, then actual data..."

**Hint 2**: Binary Reading
"You'll need to read bytes and interpret them as integers. What's the byte order?"

**Hint 3**: Metadata
"Metadata is stored as key-value pairs. Each value has a type. What types are supported?"

---

#### Model Solution

**GGUF File Structure**:

```
GGUF File Layout:
┌─────────────────────────────────────┐
│ Magic Number (4 bytes)              │ "GGUF"
├─────────────────────────────────────┤
│ Version (4 bytes, uint32)           │ Version 3 (current)
├─────────────────────────────────────┤
│ Tensor Count (8 bytes, uint64)      │ Number of tensors
├─────────────────────────────────────┤
│ Metadata KV Count (8 bytes, uint64) │ Number of metadata pairs
├─────────────────────────────────────┤
│ Metadata KV Pairs (variable)        │ Key-value pairs
│   ├── Key (string)                  │
│   ├── Value Type (4 bytes)          │
│   └── Value (type-dependent)        │
├─────────────────────────────────────┤
│ Tensor Info (variable)              │ Tensor metadata
│   ├── Name (string)                 │
│   ├── Dimensions (uint32)           │
│   ├── Type (uint32)                 │
│   └── Offset (uint64)               │
├─────────────────────────────────────┤
│ Padding (alignment to 32 bytes)     │
├─────────────────────────────────────┤
│ Tensor Data (bulk data)             │ Actual weights
└─────────────────────────────────────┘
```

**Python Implementation**:

```python
import struct
from typing import Dict, Any, BinaryIO
from enum import IntEnum

class GGUFValueType(IntEnum):
    """GGUF metadata value types."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

class GGUFReader:
    """Parse GGUF file headers and metadata."""

    MAGIC = b'GGUF'
    CURRENT_VERSION = 3

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.version = None
        self.tensor_count = None
        self.metadata = {}

    def parse_header(self) -> Dict[str, Any]:
        """
        Parse GGUF file header and metadata.

        Returns:
            Dictionary with file information
        """
        with open(self.filepath, 'rb') as f:
            # 1. Read and validate magic number
            magic = f.read(4)
            if magic != self.MAGIC:
                raise ValueError(
                    f"Invalid magic number: {magic} "
                    f"(expected {self.MAGIC})"
                )
            print(f"✓ Magic: {magic.decode('utf-8')}")

            # 2. Read version
            self.version = struct.unpack('<I', f.read(4))[0]
            if self.version > self.CURRENT_VERSION:
                raise ValueError(
                    f"Unsupported version: {self.version} "
                    f"(max supported: {self.CURRENT_VERSION})"
                )
            print(f"✓ Version: {self.version}")

            # 3. Read tensor count
            self.tensor_count = struct.unpack('<Q', f.read(8))[0]
            print(f"✓ Tensor count: {self.tensor_count}")

            # 4. Read metadata KV count
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
            print(f"✓ Metadata KV count: {metadata_kv_count}")

            # 5. Read metadata KV pairs
            for i in range(metadata_kv_count):
                key, value = self._read_kv_pair(f)
                self.metadata[key] = value

            print(f"✓ Parsed {len(self.metadata)} metadata pairs")

            return {
                'magic': magic.decode('utf-8'),
                'version': self.version,
                'tensor_count': self.tensor_count,
                'metadata': self.metadata
            }

    def _read_string(self, f: BinaryIO) -> str:
        """Read a GGUF string (length-prefixed)."""
        # String: uint64 length + UTF-8 bytes
        length = struct.unpack('<Q', f.read(8))[0]
        if length > 10_000_000:  # Sanity check (10MB max)
            raise ValueError(f"String too long: {length} bytes")
        return f.read(length).decode('utf-8')

    def _read_kv_pair(self, f: BinaryIO) -> tuple:
        """Read a metadata key-value pair."""
        # Read key
        key = self._read_string(f)

        # Read value type
        value_type = struct.unpack('<I', f.read(4))[0]
        value_type = GGUFValueType(value_type)

        # Read value based on type
        value = self._read_value(f, value_type)

        return key, value

    def _read_value(self, f: BinaryIO, value_type: GGUFValueType) -> Any:
        """Read a value based on its type."""
        if value_type == GGUFValueType.UINT8:
            return struct.unpack('<B', f.read(1))[0]
        elif value_type == GGUFValueType.INT8:
            return struct.unpack('<b', f.read(1))[0]
        elif value_type == GGUFValueType.UINT16:
            return struct.unpack('<H', f.read(2))[0]
        elif value_type == GGUFValueType.INT16:
            return struct.unpack('<h', f.read(2))[0]
        elif value_type == GGUFValueType.UINT32:
            return struct.unpack('<I', f.read(4))[0]
        elif value_type == GGUFValueType.INT32:
            return struct.unpack('<i', f.read(4))[0]
        elif value_type == GGUFValueType.UINT64:
            return struct.unpack('<Q', f.read(8))[0]
        elif value_type == GGUFValueType.INT64:
            return struct.unpack('<q', f.read(8))[0]
        elif value_type == GGUFValueType.FLOAT32:
            return struct.unpack('<f', f.read(4))[0]
        elif value_type == GGUFValueType.FLOAT64:
            return struct.unpack('<d', f.read(8))[0]
        elif value_type == GGUFValueType.BOOL:
            return bool(struct.unpack('<B', f.read(1))[0])
        elif value_type == GGUFValueType.STRING:
            return self._read_string(f)
        elif value_type == GGUFValueType.ARRAY:
            return self._read_array(f)
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def _read_array(self, f: BinaryIO) -> list:
        """Read an array value."""
        # Array: type + count + elements
        elem_type = struct.unpack('<I', f.read(4))[0]
        elem_type = GGUFValueType(elem_type)

        count = struct.unpack('<Q', f.read(8))[0]
        if count > 100_000:  # Sanity check
            raise ValueError(f"Array too large: {count} elements")

        return [self._read_value(f, elem_type) for _ in range(count)]

    def validate(self) -> Dict[str, bool]:
        """
        Validate GGUF file contents.

        Returns:
            Dictionary of validation checks
        """
        checks = {}

        # Required metadata fields
        required_fields = [
            'general.architecture',
            'general.file_type',
            'general.quantization_version'
        ]

        for field in required_fields:
            checks[f'has_{field}'] = field in self.metadata

        # Version compatibility
        checks['supported_version'] = (
            self.version <= self.CURRENT_VERSION
        )

        # Tensor count reasonable
        checks['reasonable_tensor_count'] = (
            0 < self.tensor_count < 10000
        )

        return checks

    def print_metadata(self):
        """Print metadata in human-readable format."""
        print("\n" + "="*60)
        print("GGUF METADATA")
        print("="*60)

        for key, value in sorted(self.metadata.items()):
            if isinstance(value, list):
                if len(value) > 5:
                    value_str = f"[{value[0]}, {value[1]}, ..., {value[-1]}]"
                else:
                    value_str = str(value)
            else:
                value_str = str(value)

            print(f"{key:50s} = {value_str}")

        print("="*60)


# Usage example
def main():
    reader = GGUFReader("models/llama-7b-q4_k_m.gguf")

    # Parse header
    try:
        info = reader.parse_header()

        print("\n" + "="*60)
        print("FILE INFO")
        print("="*60)
        print(f"Magic:        {info['magic']}")
        print(f"Version:      {info['version']}")
        print(f"Tensors:      {info['tensor_count']}")
        print(f"Metadata KVs: {len(info['metadata'])}")
        print("="*60)

        # Print important metadata
        reader.print_metadata()

        # Validate
        validation = reader.validate()
        print("\n" + "="*60)
        print("VALIDATION")
        print("="*60)
        for check, passed in validation.items():
            status = "✓" if passed else "✗"
            print(f"{status} {check}: {passed}")
        print("="*60)

    except Exception as e:
        print(f"Error parsing GGUF file: {e}")
        raise

if __name__ == "__main__":
    main()
```

**Key Validation Checks**:

```python
def comprehensive_validation(reader: GGUFReader) -> Dict[str, bool]:
    """Comprehensive GGUF file validation."""
    checks = {}

    # 1. Magic number
    checks['valid_magic'] = True  # Already checked during parse

    # 2. Version compatibility
    checks['supported_version'] = reader.version <= 3

    # 3. Required metadata fields
    required_metadata = [
        'general.architecture',     # e.g., "llama"
        'general.name',             # Model name
        'general.file_type',        # Quantization type
        'general.quantization_version',
        'tokenizer.ggml.model',     # Tokenizer type
        'tokenizer.ggml.tokens',    # Vocabulary
    ]

    for field in required_metadata:
        checks[f'has_{field}'] = field in reader.metadata

    # 4. Architecture-specific fields (for LLaMA)
    if reader.metadata.get('general.architecture') == 'llama':
        llama_fields = [
            'llama.context_length',
            'llama.embedding_length',
            'llama.block_count',
            'llama.attention.head_count',
        ]
        for field in llama_fields:
            checks[f'has_{field}'] = field in reader.metadata

    # 5. Sanity checks on values
    if 'llama.context_length' in reader.metadata:
        ctx_len = reader.metadata['llama.context_length']
        checks['reasonable_context'] = 128 <= ctx_len <= 128000

    if 'llama.block_count' in reader.metadata:
        n_layers = reader.metadata['llama.block_count']
        checks['reasonable_layers'] = 1 <= n_layers <= 200

    # 6. Tensor count
    checks['has_tensors'] = reader.tensor_count > 0
    checks['reasonable_tensor_count'] = reader.tensor_count < 10000

    return checks
```

**Output Example**:
```
✓ Magic: GGUF
✓ Version: 3
✓ Tensor count: 291
✓ Metadata KV count: 25
✓ Parsed 25 metadata pairs

============================================================
FILE INFO
============================================================
Magic:        GGUF
Version:      3
Tensors:      291
Metadata KVs: 25
============================================================

============================================================
GGUF METADATA
============================================================
general.architecture                               = llama
general.file_type                                  = 15
general.name                                       = LLaMA v2
general.quantization_version                       = 2
llama.attention.head_count                         = 32
llama.attention.head_count_kv                      = 32
llama.block_count                                  = 32
llama.context_length                               = 4096
llama.embedding_length                             = 4096
llama.feed_forward_length                          = 11008
llama.rope.dimension_count                         = 128
tokenizer.ggml.bos_token_id                        = 1
tokenizer.ggml.eos_token_id                        = 2
tokenizer.ggml.model                               = llama
tokenizer.ggml.tokens                              = ['<unk>', '<s>', ...]
============================================================

============================================================
VALIDATION
============================================================
✓ valid_magic: True
✓ supported_version: True
✓ has_general.architecture: True
✓ has_general.file_type: True
✓ reasonable_context: True
✓ reasonable_layers: True
============================================================
```

---

#### Follow-Up Questions

1. **"How would you extract and save just the vocabulary from a GGUF file?"**
   *Looking for*: Parse tokenizer.ggml.tokens array, save to JSON/text

2. **"What's the purpose of alignment/padding in GGUF?"**
   *Looking for*: Memory alignment for efficient access, typically 32-byte boundaries

3. **"How would you modify metadata (e.g., add a custom field)?"**
   *Looking for*: Parse, modify dict, re-serialize with updated KV count

4. **"How do you handle forward compatibility (unknown metadata keys)?"**
   *Looking for*: Skip unknown keys, warn but don't error, version checks

---

#### Rubric

| Category | Poor (0-1) | Fair (2-3) | Good (4-5) | Excellent (6-7) |
|----------|-----------|-----------|-----------|----------------|
| **Structure Knowledge** | Vague | Basic structure | Detailed structure | Complete specification |
| **Binary Parsing** | Can't implement | Basic parsing | Correct parsing | Robust + error handling |
| **Validation** | No checks | Basic checks | Multiple checks | Comprehensive validation |
| **Code Quality** | Messy | Functional | Clean structure | Production-ready |
| **Edge Cases** | Ignored | Some handling | Good handling | Complete coverage |

**Passing Score**: 12/35 (Entry), 22/35 (Mid), 30/35 (Senior)

---

#### Related Content

- 📚 [Lesson 1.2: GGUF File Format](../../modules/01-foundations/docs/gguf-format.md)
- 💻 [Code: GGUF Reader](../../modules/01-foundations/code/gguf-reader.py)
- 🔬 [Lab 1.2: Exploring GGUF Files](../../modules/01-foundations/labs/lab-02-gguf-exploration.md)

---

### Question 9: Sampling Temperature Deep Dive

**Category**: Technical
**Difficulty**: Mid (L4/L5)
**Companies**: OpenAI, Anthropic, Cohere
**Time Allotted**: 15 minutes
**Prerequisites**: Module 1, Lesson 1.4

---

#### Question

Explain how temperature affects token generation. Given logits `[2.0, 1.0, 0.5, 0.1]` for tokens `['A', 'B', 'C', 'D']`, calculate the probability distribution for:
1. Temperature = 0.1 (low)
2. Temperature = 1.0 (baseline)
3. Temperature = 2.0 (high)

What happens at temperature = 0? Temperature → ∞? When would you use each?

---

#### What the Interviewer is Looking For

**Core Competencies Tested**:
- [ ] Understanding of sampling mechanics
- [ ] Mathematical calculation ability
- [ ] Temperature trade-offs knowledge
- [ ] Practical application judgment

**Red Flags**:
- ❌ Can't calculate softmax
- ❌ Doesn't understand temperature effect
- ❌ No practical use cases
- ❌ Can't explain extreme values

**Green Flags**:
- ✅ Accurate calculations
- ✅ Clear temperature explanation
- ✅ Discusses trade-offs
- ✅ Provides use cases
- ✅ Explains edge cases

---

#### Hints (If Candidate is Stuck)

**Hint 1**: Softmax
"Temperature divides logits before softmax. What happens when you divide by a small number? Large number?"

**Hint 2**: Extreme Cases
"At temperature 0, which token always wins? At very high temperature, what happens to probabilities?"

**Hint 3**: Applications
"Think about different tasks: code generation, creative writing, factual answers. Which needs which temperature?"

---

#### Model Solution

**Temperature Formula**:
```
Given logits z = [z₁, z₂, ..., zₙ]
Temperature τ (tau)

Adjusted logits: z' = z / τ
Probabilities: p(i) = exp(z'ᵢ) / Σⱼ exp(z'ⱼ)
```

**Step-by-Step Calculation**:

Given logits: `[2.0, 1.0, 0.5, 0.1]` for tokens `['A', 'B', 'C', 'D']`

**Temperature τ = 0.1 (Low/Sharp)**:
```python
import numpy as np

logits = np.array([2.0, 1.0, 0.5, 0.1])
temp = 0.1

# Divide by temperature
adjusted = logits / temp  # [20.0, 10.0, 5.0, 1.0]

# Softmax
exp_logits = np.exp(adjusted)
# [4.85e8, 2.20e4, 1.48e2, 2.72]

probs = exp_logits / np.sum(exp_logits)
# [0.9999, 0.00005, 0.000003, 0.0000001]

print("Temperature = 0.1:")
for token, prob in zip(['A', 'B', 'C', 'D'], probs):
    print(f"  {token}: {prob:.6f} ({prob*100:.2f}%)")

# Output:
# A: 0.999955 (99.996%)
# B: 0.000045 (0.004%)
# C: 0.000000 (0.000%)
# D: 0.000000 (0.000%)
```
**Result**: Almost always picks 'A' (highest logit). Very deterministic.

**Temperature τ = 1.0 (Baseline/Normal)**:
```python
temp = 1.0
adjusted = logits / temp  # [2.0, 1.0, 0.5, 0.1] (unchanged)

exp_logits = np.exp(adjusted)
# [7.39, 2.72, 1.65, 1.11]

probs = exp_logits / np.sum(exp_logits)
# [0.572, 0.211, 0.128, 0.086]

print("Temperature = 1.0:")
for token, prob in zip(['A', 'B', 'C', 'D'], probs):
    print(f"  {token}: {prob:.3f} ({prob*100:.1f}%)")

# Output:
# A: 0.572 (57.2%)
# B: 0.211 (21.1%)
# C: 0.128 (12.8%)
# D: 0.086 (8.6%)
```
**Result**: 'A' still most likely, but others have reasonable chance. Balanced.

**Temperature τ = 2.0 (High/Flat)**:
```python
temp = 2.0
adjusted = logits / temp  # [1.0, 0.5, 0.25, 0.05]

exp_logits = np.exp(adjusted)
# [2.72, 1.65, 1.28, 1.05]

probs = exp_logits / np.sum(exp_logits)
# [0.399, 0.242, 0.188, 0.154]

print("Temperature = 2.0:")
for token, prob in zip(['A', 'B', 'C', 'D'], probs):
    print(f"  {token}: {prob:.3f} ({prob*100:.1f}%)")

# Output:
# A: 0.399 (39.9%)
# B: 0.242 (24.2%)
# C: 0.188 (18.8%)
# D: 0.154 (15.4%)
```
**Result**: Much more uniform. All tokens have significant probability. Random.

**Comparison Table**:

| Token | τ=0.1 | τ=1.0 | τ=2.0 | τ→∞ |
|-------|-------|-------|-------|------|
| A     | 99.996% | 57.2% | 39.9% | 25% |
| B     | 0.004%  | 21.1% | 24.2% | 25% |
| C     | 0.000%  | 12.8% | 18.8% | 25% |
| D     | 0.000%  | 8.6%  | 15.4% | 25% |

**Extreme Cases**:

**τ → 0 (Greedy Decoding)**:
```
Division by tiny number → huge adjusted logits
Highest logit dominates completely
Result: Always picks argmax (deterministic)

Probability distribution:
p(argmax) = 1.0
p(others) = 0.0

Use case: When you want single best answer (factual queries)
```

**τ → ∞ (Uniform Sampling)**:
```
Division by huge number → tiny adjusted logits
All logits approach 0
exp(0) = 1 for all
Result: Uniform distribution

Probability distribution:
p(i) = 1/n for all tokens

Use case: Maximum randomness (rarely useful in practice)
```

**Practical Use Cases**:

**τ = 0 to 0.3 (Very Low)**:
```
Use for:
✅ Factual question answering
✅ Code generation (syntactically correct)
✅ Translation (accurate)
✅ Mathematical reasoning
✅ Tasks requiring determinism

Example:
Q: "What is 2+2?"
A: "4" (always)

Characteristics:
- Consistent outputs
- Safe/conservative
- May be repetitive
- Low creativity
```

**τ = 0.7 to 1.0 (Moderate)**:
```
Use for:
✅ General chat/conversation
✅ Balanced creativity and coherence
✅ Default for most applications
✅ Instruction following

Example:
Q: "Write a greeting"
A: "Hello! How can I help you today?" (varied but coherent)

Characteristics:
- Balanced randomness
- Natural variation
- Generally coherent
- Good default
```

**τ = 1.5 to 2.0 (High)**:
```
Use for:
✅ Creative writing
✅ Brainstorming
✅ Idea generation
✅ Poetry/artistic text

Example:
Q: "Write a poem about AI"
A: "Silicon dreams dance through quantum foam..." (creative)

Characteristics:
- High creativity
- More randomness
- May lose coherence
- Unexpected outputs
```

**τ > 2.0 (Very High)**:
```
Use for:
⚠️ Rarely useful
⚠️ Experimental prompts
⚠️ Testing randomness

Characteristics:
- Too random
- Often incoherent
- Not practical
```

**Temperature vs Other Sampling Methods**:

```python
# Temperature affects all tokens
temperature = 0.7
adjusted_logits = logits / temperature

# Top-k keeps only k best tokens (then renormalize)
top_k = 40
# Keep 40 highest logits, set others to -∞

# Top-p (nucleus) keeps cumulative mass p
top_p = 0.9
# Keep tokens until cumsum(probs) >= 0.9

# Common combination:
# 1. Apply temperature
# 2. Apply top-k or top-p
# 3. Sample from remaining tokens

def sample(logits, temp=1.0, top_k=50, top_p=0.9):
    # Temperature
    logits = logits / temp

    # Top-k
    if top_k > 0:
        indices = np.argsort(logits)[-top_k:]
        filtered = np.full_like(logits, -np.inf)
        filtered[indices] = logits[indices]
        logits = filtered

    # Softmax
    probs = softmax(logits)

    # Top-p
    if top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, top_p) + 1

        keep_idx = sorted_idx[:cutoff]
        filtered = np.zeros_like(probs)
        filtered[keep_idx] = probs[keep_idx]
        probs = filtered / np.sum(filtered)

    # Sample
    return np.random.choice(len(probs), p=probs)
```

**Real-World Example**:

```python
# OpenAI API-style parameters
response = model.generate(
    prompt="Explain quantum computing",
    temperature=0.7,    # Moderate creativity
    top_p=0.9,          # Nucleus sampling
    max_tokens=100
)

# For factual Q&A:
response = model.generate(
    prompt="What is the capital of France?",
    temperature=0.0,    # Deterministic
    max_tokens=10
)

# For creative writing:
response = model.generate(
    prompt="Write a fantasy story beginning",
    temperature=1.2,    # High creativity
    top_p=0.95,
    max_tokens=500
)
```

---

#### Follow-Up Questions

1. **"How does temperature interact with model uncertainty?"**
   *Looking for*: Low confidence outputs become more uniform with high temp

2. **"When might temperature > 1.0 be better than top-p sampling?"**
   *Looking for*: When you want all tokens considered, not just top mass

3. **"How do you choose temperature for a production chatbot?"**
   *Looking for*: A/B testing, user feedback, task-specific tuning

4. **"What's the computational cost of different temperatures?"**
   *Looking for*: Same cost (just division), unlike top-k/top-p sorting

---

#### Rubric

| Category | Poor (0-1) | Fair (2-3) | Good (4-5) | Excellent (6-7) |
|----------|-----------|-----------|-----------|----------------|
| **Mathematical Calculation** | Major errors | Minor errors | Correct | Perfect + explained |
| **Conceptual Understanding** | Vague | Basic concept | Clear explanation | Deep insight |
| **Extreme Cases** | Ignored | Mentioned | Explained | Comprehensive |
| **Practical Application** | No examples | Generic | Specific use cases | Production scenarios |
| **Communication** | Unclear | Understandable | Structured | Clear + intuitive |

**Passing Score**: 12/35 (Entry), 20/35 (Mid), 28/35 (Senior)

---

#### Related Content

- 📚 [Lesson 1.4: Basic Inference](../../modules/01-foundations/docs/basic-inference.md)
- 📚 [Module 2: Sampling Strategies](../../modules/02-core-implementation/docs/sampling.md)
- 🔬 [Lab 2.5: Sampling Experiments](../../modules/02-core-implementation/labs/lab-05-sampling.md)

---

### Question 10: Model Loading Performance

**Category**: Technical
**Difficulty**: Mid to Senior (L5/L6)
**Companies**: OpenAI, Anthropic, Scale AI
**Time Allotted**: 20 minutes
**Prerequisites**: Module 1, Lessons 1.4-1.5

---

#### Question

A user reports that loading a 70B model takes 2 minutes on their system. Walk through:
1. What factors affect model loading time?
2. How memory-mapping (mmap) helps
3. Why quantized models load faster
4. Techniques to minimize loading time

What's the theoretical minimum loading time and why?

---

#### What the Interviewer is Looking For

**Core Competencies Tested**:
- [ ] Understanding of file I/O and memory systems
- [ ] Knowledge of mmap and OS memory management
- [ ] Performance optimization thinking
- [ ] System-level understanding

**Red Flags**:
- ❌ Thinks all loading is file reading
- ❌ Doesn't know what mmap is
- ❌ Can't explain quantization benefit
- ❌ No optimization suggestions

**Green Flags**:
- ✅ Explains mmap vs read()
- ✅ Discusses I/O bandwidth limitations
- ✅ Mentions OS page cache
- ✅ Provides concrete optimizations
- ✅ Calculates theoretical limits

---

#### Hints (If Candidate is Stuck)

**Hint 1**: Loading Strategies
"There are two ways to load: read entire file into memory, or memory-map. What's the difference?"

**Hint 2**: Bottleneck
"What's the bottleneck: CPU, disk I/O, or RAM bandwidth? How do you identify it?"

**Hint 3**: Quantization
"A 70B FP16 model is ~140GB. A Q4_K_M is ~40GB. How does this affect loading?"

---

#### Model Solution

**Loading Time Factors**:

**1. File Size**:
```
LLaMA-70B:
- FP16:    ~140 GB
- Q8_0:    ~70 GB
- Q4_K_M:  ~40 GB
- Q3_K_S:  ~28 GB

Larger file → longer loading (linear relationship)
```

**2. Storage Type**:
```
Storage          | Read Speed       | Impact
-----------------|------------------|----------------
NVMe SSD         | 3-7 GB/s        | Best
SATA SSD         | 500-600 MB/s    | Good
HDD              | 100-200 MB/s    | Slow
Network (1Gbps)  | ~100 MB/s       | Very slow
```

**3. Loading Method**:
```
Method       | Speed      | Memory Usage  | Startup Time
-------------|------------|---------------|-------------
read()       | Slow       | 2x (cache+buf)| ~Minutes
mmap()       | Instant    | 1x (demand)   | <1 second
mlock()      | Slow       | Locked in RAM | Minutes
```

**4. OS Page Cache**:
```
First load:  Read from disk → slow
Second load: Read from cache → fast (if cached)
```

**Memory-Mapping (mmap) Explanation**:

**Without mmap (Traditional Read)**:
```python
# Traditional approach
with open("model.gguf", "rb") as f:
    model_data = f.read()  # Reads entire file into Python buffer
    # Memory usage: 2x file size (OS cache + Python buffer)
    # Time: file_size / disk_speed
    # Example: 40GB / 3GB/s = ~13 seconds

Loading 70B Q4_K_M (40GB) on NVMe:
- Read time: 40GB / 3GB/s = 13.3 seconds ✅
- Memory allocation: 40GB
- Peak memory: 80GB (cache + buffer) ❌
```

**With mmap (Memory-Mapped)**:
```python
# Memory-mapped approach
import mmap

with open("model.gguf", "rb") as f:
    mmapped = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    # Returns instantly - no actual reading yet!
    # Pages loaded on-demand (page faults)
    # Memory usage: 1x file size (no duplication)

Loading 70B Q4_K_M (40GB) with mmap:
- Mapping time: ~0.001 seconds ✅ (instant!)
- Memory allocation: 0GB initially
- On-demand loading: Pages loaded as accessed
- Peak memory: 40GB (only what's accessed)
```

**How mmap Works**:

```
Without mmap:
┌─────────────────────────────────────────┐
│                  RAM                     │
├─────────────────────────────────────────┤
│ OS Page Cache:       40 GB               │
│ Python Buffer:       40 GB               │  ← Duplication!
├─────────────────────────────────────────┤
│ Total:               80 GB               │
└─────────────────────────────────────────┘
Time: 13 seconds to read

With mmap:
┌─────────────────────────────────────────┐
│                  RAM                     │
├─────────────────────────────────────────┤
│ OS Page Cache:       40 GB (on-demand)  │
│ Python Mapping:      0 GB (virtual)      │  ← No duplication!
├─────────────────────────────────────────┤
│ Total:               40 GB               │
└─────────────────────────────────────────┘
Time: 0.001 seconds to map (instant!)

On-Demand Loading:
When you access model.weights[0]:
1. Page fault (data not in RAM)
2. OS loads page from disk (4KB or 2MB)
3. Returns data
4. Page stays in cache for next access
```

**Why Quantized Models Load Faster**:

**Factor 1: Smaller File Size**
```
FP16 vs Q4_K_M for 70B parameters:

FP16:    70B × 2 bytes = 140 GB
Q4_K_M:  70B × 0.5625 bytes = 40 GB

Improvement: 3.5x smaller

With mmap: Still instant (file size doesn't matter for mapping)
With read(): 140GB/3GB/s = 47s vs 40GB/3GB/s = 13s
```

**Factor 2: Less Memory Bandwidth**
```
During inference (memory reads):

FP16:    Read 140GB of weights
Q4_K_M:  Read 40GB of weights → Dequantize to FP16/FP32

Memory bandwidth saved: 3.5x
(Dequantization is compute, not memory - often faster)
```

**Factor 3: Better Cache Utilization**
```
CPU Cache (L1/L2/L3):

FP16:    Needs larger cache to hold weights
Q4_K_M:  Fits more weights in same cache space

Result: Fewer cache misses → faster inference
```

**Techniques to Minimize Loading Time**:

**1. Use mmap (Primary Optimization)**
```python
# llama.cpp automatically uses mmap
model = Llama(
    model_path="model.gguf",
    use_mmap=True,    # Default: True
    use_mlock=False   # Don't lock in RAM
)

Loading time: <1 second (instant mapping)
```

**2. Quantize Models (File Size Reduction)**
```
Choose most compressed quantization acceptable:
- Q3_K_S: Smallest, fastest load, lower quality
- Q4_K_M: Balanced
- Q8_0: Larger, slower, higher quality

40GB vs 28GB: Saves 12GB, 30% faster with read()
```

**3. Use Faster Storage**
```
Upgrade path:
HDD (100MB/s) → SATA SSD (500MB/s) → NVMe SSD (3-7GB/s)

40GB model:
- HDD:  400 seconds (6.7 min)
- SATA: 80 seconds  (1.3 min)
- NVMe: 13 seconds  (0.2 min)

With mmap: All instant, but first access varies
```

**4. Pre-warm OS Cache**
```bash
# Load model into OS cache before first inference
# (Only needed if not using mmap)
cat model.gguf > /dev/null

# Or use vmtouch
vmtouch -t model.gguf  # Touch (load into cache)
vmtouch -l model.gguf  # Lock in cache

Subsequent loads: Instant (from cache)
```

**5. Split Model (Multi-GPU)**
```
Instead of loading 40GB on one machine:
- Load 20GB on GPU 1
- Load 20GB on GPU 2

Each loads faster: 20GB / 3GB/s = 6.7s vs 13.3s
```

**6. Network Optimizations (Model Servers)**
```
Instead of downloading each time:
✅ Keep model on local NVMe
✅ Use container with baked-in model
✅ Share model across pods (ReadOnlyMany volumes)
✅ Use model caching proxy
```

**Theoretical Minimum Loading Time**:

**With mmap (Best Case)**:
```
Minimum time = mmap() syscall latency
             ≈ 0.001 - 0.01 seconds

Why so fast?
- No actual file reading
- Just creates virtual memory mapping
- OS handles loading on-demand
- First inference may be slower (page faults)

This is what llama.cpp does by default!
```

**Without mmap (Read-based)**:
```
Minimum time = File Size / Maximum I/O Bandwidth

Best case: NVMe Gen 4/5
- Bandwidth: 7 GB/s
- 40GB model: 40GB / 7GB/s = 5.7 seconds

Theoretical limit: RAM bandwidth
- DDR4: ~25 GB/s
- 40GB model: 40GB / 25GB/s = 1.6 seconds

But you can't beat the OS page cache with mmap!
```

**Real-World Loading Times**:

```
LLaMA-70B Q4_K_M (40GB):

Method           | First Load | Second Load | Memory
-----------------|------------|-------------|--------
mmap + NVMe      | 0.01s      | 0.01s       | 40GB
mmap + SATA SSD  | 0.01s      | 0.01s       | 40GB
mmap + HDD       | 0.01s      | 0.01s       | 40GB
read() + NVMe    | 13s        | 0.1s (cache)| 80GB
read() + SATA    | 80s        | 0.1s (cache)| 80GB
read() + HDD     | 400s       | 0.1s (cache)| 80GB

Verdict: mmap wins! (Default in llama.cpp)
```

**Troubleshooting Slow Loading**:

**Scenario: 70B model taking 2 minutes to load**

```python
# Diagnose:

# 1. Check if mmap is enabled
model = Llama(..., use_mmap=True, verbose=True)
# Look for: "mmap: loaded" in logs

# 2. Check disk speed
# On Linux:
hdparm -t /dev/nvme0n1  # Should be >3 GB/s for NVMe

# 3. Check if file is cached
vmtouch -v model.gguf  # Shows cached pages

# 4. Check RAM availability
free -h  # Need at least model_size available

# 5. Measure actual loading time
import time
start = time.time()
model = Llama("model.gguf")
print(f"Load time: {time.time() - start:.2f}s")

# If >1 second, likely not using mmap properly
# If first inference is slow, that's normal (page faults)
```

**Possible Issues**:
```
1. mmap disabled → Re-enable use_mmap=True
2. Slow disk (HDD) → Upgrade to SSD
3. Not enough RAM → System swapping
4. Network filesystem → Copy to local disk
5. Measuring first inference → Expected with mmap
```

---

#### Follow-Up Questions

1. **"What happens if you don't have enough RAM for the entire model?"**
   *Looking for*: OS swaps pages, severe performance degradation, thrashing

2. **"How does mlock differ from mmap?"**
   *Looking for*: mlock prevents swapping, pins pages in RAM, requires privileges

3. **"In a model server with 100 concurrent users, how does mmap help?"**
   *Looking for*: Single copy in memory, shared across processes, huge memory savings

4. **"What's the trade-off between Q3 and Q4 quantization for loading?"**
   *Looking for*: Q3 faster to load but lower quality, marginal load time difference with mmap

---

#### Rubric

| Category | Poor (0-1) | Fair (2-3) | Good (4-5) | Excellent (6-7) |
|----------|-----------|-----------|-----------|----------------|
| **mmap Understanding** | Doesn't know | Basic concept | Clear explanation | OS-level details |
| **Performance Factors** | Few factors | Some factors | Comprehensive | Quantitative analysis |
| **Optimization Techniques** | No suggestions | 1-2 ideas | Multiple techniques | Production strategies |
| **Theoretical Minimum** | Can't answer | Rough idea | Correct calculation | Detailed breakdown |
| **Problem Solving** | No diagnosis | Basic troubleshooting | Systematic approach | Expert debugging |

**Passing Score**: 12/35 (Entry), 22/35 (Mid), 30/35 (Senior)

---

#### Related Content

- 📚 [Lesson 1.5: Memory Management](../../modules/01-foundations/docs/memory-management.md)
- 💻 [Code: Loading Benchmarks](../../modules/01-foundations/code/loading-benchmark.py)
- 📚 [Advanced: Memory Mapping Deep Dive](../../modules/09-production-engineering/docs/memory-optimization.md)

---

## System Design Questions

### Question 11: Design a Simple Inference Service

**Category**: System Design
**Difficulty**: Mid (L4/L5)
**Companies**: OpenAI, Anthropic, Startups
**Time Allotted**: 30-40 minutes
**Prerequisites**: Module 1 complete

---

#### Question

Design a simple HTTP inference service for llama.cpp that can:
1. Serve a 7B model on a single machine (16GB RAM, 8 CPU cores)
2. Handle 100 concurrent users
3. Maintain <2 second response time for 200-token generations
4. Be reliable and monitorable

Walk through your architecture, key components, and trade-offs. How would you test it?

---

#### What the Interviewer is Looking For

**Core Competencies Tested**:
- [ ] System design fundamentals
- [ ] Understanding of HTTP APIs
- [ ] Resource management
- [ ] Scalability thinking
- [ ] Production readiness

**Red Flags**:
- ❌ No consideration of concurrency
- ❌ Ignores resource constraints
- ❌ No error handling
- ❌ Missing monitoring/observability
- ❌ Unrealistic architecture

**Green Flags**:
- ✅ Considers request queuing
- ✅ Discusses resource limits
- ✅ Includes monitoring
- ✅ Handles failures gracefully
- ✅ Provides concrete numbers
- ✅ Mentions testing strategy

---

#### Hints (If Candidate is Stuck)

**Hint 1**: Concurrency
"100 concurrent users, but how many can you process in parallel? What do you do with the rest?"

**Hint 2**: Resource Constraints
"7B Q4_K_M is ~4GB. KV cache per user? Calculate total memory needed."

**Hint 3**: API Design
"What would the HTTP API look like? POST /generate? What parameters?"

---

#### Model Solution

**System Overview**:
```
┌─────────────────────────────────────────────────────────┐
│              Load Balancer (Optional)                   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                HTTP API Server                           │
│  ┌────────────────────────────────────────────────┐    │
│  │  Request Handler                                │    │
│  │  - Input validation                             │    │
│  │  - Rate limiting                                │    │
│  │  - Authentication (optional)                    │    │
│  └────────────────────┬───────────────────────────┘    │
│                       │                                  │
│  ┌────────────────────┴───────────────────────────┐    │
│  │  Request Queue (Max size: 500)                  │    │
│  │  - FIFO ordering                                │    │
│  │  - Priority levels (optional)                   │    │
│  │  - Reject when full (HTTP 503)                  │    │
│  └────────────────────┬───────────────────────────┘    │
│                       │                                  │
│  ┌────────────────────┴───────────────────────────┐    │
│  │  Worker Pool (2-4 workers)                      │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │ ... │    │
│  │  └─────┬────┘  └─────┬────┘  └─────┬────┘     │    │
│  └────────┼─────────────┼─────────────┼───────────┘    │
│           │             │             │                 │
│  ┌────────┴─────────────┴─────────────┴───────────┐    │
│  │       Shared llama.cpp Model Instance           │    │
│  │       (mmap'd, shared memory)                   │    │
│  │       Model: LLaMA-7B Q4_K_M (~4GB)             │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │      Monitoring & Metrics      │
         │  - Prometheus metrics          │
         │  - Request latency (p50/p99)  │
         │  - Queue depth                 │
         │  - Error rates                 │
         └────────────────────────────────┘
```

**Resource Calculation**:
```python
# Constraints
total_ram = 16  # GB
cpu_cores = 8

# Model size
model_size = 4  # GB (LLaMA-7B Q4_K_M)

# Per-request KV cache (2048 context)
n_layers = 32
n_heads = 32
head_dim = 128
n_ctx = 2048
kv_cache_per_request = (
    2 * n_layers * n_ctx * n_heads * head_dim * 2 / (1024**3)
)  # ~2 GB in FP16

# Concurrent inference
concurrent_workers = 2  # Conservative
kv_cache_total = concurrent_workers * kv_cache_per_request  # ~4 GB

# System overhead
system_overhead = 2  # GB

# Total
total_memory = model_size + kv_cache_total + system_overhead
# = 4 + 4 + 2 = 10 GB ✅ (fits in 16GB)

print(f"Total memory: {total_memory} GB")
print(f"Available margin: {total_ram - total_memory} GB")
# Margin: 6 GB (37.5%)
```

**Key Components**:

**1. HTTP API Server (FastAPI/Flask)**:
```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from queue import Queue
import asyncio
from llama_cpp import Llama
import time

app = FastAPI(title="LLaMA Inference Service")

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int = Field(default=200, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=100)

class GenerateResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    generation_time: float

# Global model instance (loaded once, shared)
model = None
request_queue = Queue(maxsize=500)
active_requests = 0
MAX_CONCURRENT = 2

def load_model():
    """Load model at startup."""
    global model
    model = Llama(
        model_path="/models/llama-7b-q4_k_m.gguf",
        n_ctx=2048,
        n_batch=512,
        n_threads=4,  # Leave cores for API server
        n_gpu_layers=0,  # CPU only
        use_mmap=True,
        verbose=False
    )
    print("Model loaded successfully")

@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    load_model()
    # Start worker pool
    asyncio.create_task(worker_pool())

async def worker_pool():
    """Process requests from queue."""
    global active_requests
    while True:
        if not request_queue.empty() and active_requests < MAX_CONCURRENT:
            future, request = request_queue.get()
            active_requests += 1
            try:
                result = generate_text(request)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                active_requests -= 1
        await asyncio.sleep(0.01)

def generate_text(req: GenerateRequest) -> GenerateResponse:
    """Generate text (runs in worker)."""
    start_time = time.time()

    output = model(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        echo=False
    )

    generation_time = time.time() - start_time

    return GenerateResponse(
        generated_text=output['choices'][0]['text'],
        tokens_generated=output['usage']['completion_tokens'],
        generation_time=generation_time
    )

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text endpoint."""
    # Check queue capacity
    if request_queue.qsize() >= 500:
        raise HTTPException(
            status_code=503,
            detail="Service at capacity, try again later"
        )

    # Create future for async result
    future = asyncio.Future()
    request_queue.put((future, request))

    # Wait for result (with timeout)
    try:
        result = await asyncio.wait_for(future, timeout=30.0)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Generation timeout"
        )

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "queue_depth": request_queue.qsize(),
        "active_requests": active_requests
    }

@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics."""
    return {
        "queue_depth": request_queue.qsize(),
        "active_requests": active_requests,
        "queue_capacity": 500,
        "max_concurrent": MAX_CONCURRENT
    }
```

**2. Request Queue Management**:
```
Why queue?
- 100 concurrent users
- Only 2-4 can process simultaneously
- Queue holds waiting requests

Queue size: 500
- Prevents memory exhaustion
- Rejects (HTTP 503) when full
- Users can retry with backoff

Ordering: FIFO (could add priority)
```

**3. Worker Pool Design**:
```python
Concurrency: 2-4 workers

Why not more?
- Each worker needs KV cache (~2GB)
- 4 workers = 8GB KV cache
- Limited by RAM, not CPU

Why not 1?
- Some parallelism improves throughput
- While one generates, other can start

Thread safety:
- llama.cpp is NOT thread-safe per context
- Each worker gets own context or serialize access
```

**4. Monitoring & Metrics**:
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

request_counter = Counter(
    'inference_requests_total',
    'Total inference requests'
)

request_latency = Histogram(
    'inference_latency_seconds',
    'Request latency in seconds'
)

queue_depth = Gauge(
    'request_queue_depth',
    'Current queue depth'
)

error_counter = Counter(
    'inference_errors_total',
    'Total errors',
    ['error_type']
)

# Update metrics
@app.post("/v1/generate")
async def generate(request: GenerateRequest):
    request_counter.inc()
    queue_depth.set(request_queue.qsize())

    with request_latency.time():
        try:
            result = await process_request(request)
            return result
        except Exception as e:
            error_counter.labels(error_type=type(e).__name__).inc()
            raise
```

**Performance Analysis**:

```python
# Target: <2s for 200 tokens
# LLaMA-7B Q4_K_M on CPU: ~15-20 tok/s

generation_time = 200 tokens / 15 tok/s = 13.3 seconds ❌

# Problem: Won't meet 2s requirement with CPU!

# Solutions:
# 1. GPU offload (if available)
model = Llama(..., n_gpu_layers=32)  # 100+ tok/s → <2s ✅

# 2. Smaller model
# LLaMA-3B Q4_K_M: 30-40 tok/s → 5-7s (better but still >2s)

# 3. Aggressive quantization
# Q3_K_S: 20-25 tok/s → 8-10s

# 4. Reduce max_tokens
# 100 tokens: 100/15 = 6.7s (still >2s)

# Realistic: 2s with GPU, 6-10s with CPU
```

**Revised Architecture for <2s Latency**:
```
Add GPU:
- NVIDIA GPU (RTX 3090, 24GB VRAM)
- Offload all 32 layers to GPU
- Throughput: 80-120 tok/s
- 200 tokens: ~2s ✅

Concurrency with GPU:
- Batch requests together
- Process 2-4 requests in parallel
- Each shares GPU time
```

**Testing Strategy**:

**1. Unit Tests**:
```python
# test_api.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_generate_success():
    response = client.post("/v1/generate", json={
        "prompt": "Hello",
        "max_tokens": 10
    })
    assert response.status_code == 200
    assert "generated_text" in response.json()

def test_generate_validation():
    # Empty prompt
    response = client.post("/v1/generate", json={
        "prompt": "",
        "max_tokens": 10
    })
    assert response.status_code == 422

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

**2. Load Tests**:
```python
# load_test.py using locust
from locust import HttpUser, task, between

class InferenceUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def generate(self):
        self.client.post("/v1/generate", json={
            "prompt": "Tell me about AI",
            "max_tokens": 50
        })

# Run: locust -f load_test.py --host=http://localhost:8000
# Target: 100 concurrent users, check p99 latency <2s
```

**3. Integration Tests**:
```python
# test_integration.py
def test_concurrent_requests():
    """Test handling of concurrent requests."""
    import concurrent.futures

    def make_request():
        return client.post("/v1/generate", json={
            "prompt": "Test",
            "max_tokens": 10
        })

    # Send 10 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        results = [f.result() for f in futures]

    # All should succeed or queue properly
    assert all(r.status_code in [200, 503] for r in results)
```

**4. Stress Tests**:
```bash
# Apache Bench
ab -n 1000 -c 100 -p request.json -T application/json \
   http://localhost:8000/v1/generate

# Expected:
# - Most requests succeed (200)
# - Some queue rejections (503) under high load
# - No crashes or OOM errors
```

**Deployment**:

```yaml
# docker-compose.yml
version: '3.8'
services:
  inference-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models:ro
    environment:
      - MODEL_PATH=/models/llama-7b-q4_k_m.gguf
      - MAX_CONCURRENT=2
      - QUEUE_SIZE=500
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 12G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

**Key Trade-offs**:

| Decision | Choice | Trade-off |
|----------|--------|-----------|
| Concurrency | 2 workers | Conservative (memory) vs throughput |
| Queue size | 500 | Memory usage vs request acceptance |
| Model size | 7B Q4_K_M | Quality vs speed/memory |
| Context | 2048 | Conversation length vs memory |
| CPU vs GPU | CPU (+ GPU option) | Cost vs latency |

**Production Considerations**:
```
✅ Error handling (try/except, HTTP codes)
✅ Request validation (Pydantic)
✅ Rate limiting (optional: per-user)
✅ Authentication (optional: API keys)
✅ Monitoring (Prometheus + Grafana)
✅ Health checks (K8s/Docker)
✅ Logging (structured logs)
✅ Graceful shutdown (drain queue)
⚠️ Auto-scaling (harder with stateful service)
⚠️ Model versioning (A/B testing)
```

---

#### Follow-Up Questions

1. **"How would you scale to 1000 concurrent users?"**
   *Looking for*: Horizontal scaling (multiple servers), load balancer, shared cache/queue

2. **"What if latency must be <500ms instead of <2s?"**
   *Looking for*: GPU required, reduce max_tokens, use smaller model, speculative decoding

3. **"How do you handle model updates without downtime?"**
   *Looking for*: Blue-green deployment, rolling updates, load balancer draining

4. **"What metrics would you alert on?"**
   *Looking for*: Error rate >1%, p99 latency >3s, queue depth >400, OOM errors

---

#### Rubric

| Category | Poor (0-1) | Fair (2-3) | Good (4-5) | Excellent (6-7) |
|----------|-----------|-----------|-----------|----------------|
| **Architecture** | Vague design | Basic design | Clear components | Production-ready |
| **Resource Calculation** | No math | Rough estimates | Accurate calculations | Detailed analysis |
| **Concurrency Handling** | Ignored | Basic queue | Worker pool | Optimized |
| **Monitoring** | No monitoring | Basic logging | Metrics exposed | Full observability |
| **Trade-off Analysis** | No discussion | Some trade-offs | Comprehensive | Quantitative |

**Passing Score**: 12/35 (Entry), 22/35 (Mid), 30/35 (Senior)

---

#### Related Content

- 📚 [Module 6: Server & Production](../../modules/06-server-production/)
- 💻 [Project: Inference API Server](../../modules/06-server-production/projects/inference-api/)
- 📚 [Lesson 1.5: Memory Management](../../modules/01-foundations/docs/memory-management.md)

---

### Question 12: Choose Quantization for Deployment

**Category**: System Design
**Difficulty**: Mid (L4/L5)
**Companies**: OpenAI, Anthropic, Hugging Face
**Time Allotted**: 20 minutes
**Prerequisites**: Module 1, Lessons 1.1-1.2

---

#### Question

You're deploying a LLaMA-13B model for three different use cases:
1. **Customer support chatbot** (accuracy critical, GPU available)
2. **Desktop code completion** (runs on developer laptops, CPU only)
3. **Mobile app** (limited RAM, <4GB, casual use)

For each use case, choose the appropriate quantization format and justify your choice. What metrics would you use to validate your decision?

---

#### What the Interviewer is Looking For

**Core Competencies Tested**:
- [ ] Understanding of quantization trade-offs
- [ ] Resource constraint awareness
- [ ] Use-case analysis
- [ ] Decision-making framework
- [ ] Validation methodology

**Red Flags**:
- ❌ Same quantization for all cases
- ❌ Ignores resource constraints
- ❌ No quality metrics mentioned
- ❌ Unrealistic choices

**Green Flags**:
- ✅ Different choices per use case
- ✅ Considers memory, accuracy, speed
- ✅ Provides clear rationale
- ✅ Mentions validation metrics
- ✅ Discusses fallback options

---

#### Model Solution

**LLaMA-13B Size Reference**:
```
Format    | Size (GB) | Bits/param | Quality (approx)
----------|-----------|------------|------------------
FP16      | 26.0      | 16         | Baseline (100%)
Q8_0      | 13.8      | 8          | 99%
Q6_K      | 10.7      | 6          | 98%
Q5_K_M    | 9.1       | 5.5        | 96%
Q4_K_M    | 7.9       | 4.5        | 92%
Q4_0      | 7.3       | 4          | 88%
Q3_K_M    | 6.0       | 3.5        | 82%
Q3_K_S    | 5.3       | 3          | 75%
IQ3_XXS   | 4.3       | 2.5        | 65%
```

**Use Case 1: Customer Support Chatbot**

**Requirements**:
- Accuracy critical (customer satisfaction)
- GPU available (NVIDIA A10/T4 or better)
- Real-time responses needed
- Professional tone required

**Recommendation: Q8_0**

**Rationale**:
```
✅ Nearly lossless quality (99% of FP16)
✅ Fits in GPU memory (16GB VRAM on A10)
✅ 2x smaller than FP16, faster inference
✅ Minimal hallucinations
✅ Professional, accurate responses

Memory calculation:
- Model: 13.8 GB
- KV cache (4K context): ~3 GB
- Total: ~17 GB
- GPU: A10 (24GB) ✅ or A100 (40/80GB) ✅

Alternative: Q6_K
- If tighter on memory (10.7 GB)
- Still excellent quality (98%)
```

**Why not FP16?**
```
❌ 26 GB doesn't fit in most GPUs (needs A100 80GB)
❌ Slower inference (more memory bandwidth)
❌ No significant quality gain over Q8_0
```

**Why not Q4_K_M?**
```
⚠️ 8% quality loss noticeable in support context
⚠️ More hallucinations, incorrect information
⚠️ Not worth risk for customer-facing use
```

**Use Case 2: Desktop Code Completion**

**Requirements**:
- Runs on developer laptops (CPU only)
- Typical RAM: 16-32GB
- Fast completion (<100ms desirable)
- Developer tolerance for minor errors

**Recommendation: Q4_K_M**

**Rationale**:
```
✅ Good balance: 7.9 GB fits comfortably in 16GB RAM
✅ Fast CPU inference (~20-25 tok/s on modern CPU)
✅ Acceptable quality for code (92% of baseline)
✅ Leaves RAM for IDE, browser, build tools

Memory calculation:
- Model: 7.9 GB
- KV cache (2K context): ~1.5 GB
- System overhead: 1 GB
- Total: 10.4 GB
- Available on 16GB laptop: ✅ (5.6 GB free)

Performance:
- CPU (Intel i7/M1): 20-25 tok/s
- Completion latency: ~50-100ms for 2-5 tokens ✅
```

**Why not Q8_0?**
```
⚠️ 13.8 GB + KV = ~17 GB
⚠️ Doesn't fit well in 16GB RAM
⚠️ Would cause swapping → very slow
⚠️ Not enough margin for other apps
```

**Why not Q3_K_M?**
```
✅ Possible option (6 GB, very fast)
⚠️ 18% quality loss
⚠️ More incorrect completions
⚠️ Developers notice and lose trust

Verdict: Q4_K_M better trade-off
```

**Alternative: Q5_K_M**
```
If developer has 32GB RAM:
- 9.1 GB model + 2 GB KV = 11 GB
- Better quality (96%)
- Still fast enough
```

**Use Case 3: Mobile App**

**Requirements**:
- Limited RAM (<4GB constraint)
- Casual use (creative writing, chat)
- Battery conscious (prefer CPU)
- User expectation: "good enough"

**Recommendation: IQ3_XXS or Q3_K_S**

**Rationale**:
```
✅ Tiny size: 4.3 GB or 5.3 GB
✅ Fits in 4GB constraint with small context
✅ Mobile CPUs (ARM) optimized for low precision
✅ Casual use tolerates quality loss
✅ Faster = better battery life

Memory calculation (IQ3_XXS):
- Model: 4.3 GB
- KV cache (1K context): 0.7 GB
- OS + App overhead: 1 GB
- Total: 6 GB
- With compression/swapping: Fits in 4GB device ⚠️

Memory calculation (Q3_K_S):
- Model: 5.3 GB
- This exceeds 4GB constraint! ❌

Revised: IQ3_XXS only
```

**Why not Q4_K_M?**
```
❌ 7.9 GB doesn't fit <4GB constraint
❌ Even with aggressive optimization, too large
❌ Would require streaming/paging (complex)
```

**Mobile-Specific Optimizations**:
```
Further reductions:
✅ Use 512-1024 context (reduce KV cache)
✅ Lazy loading (load layers on-demand)
✅ Quantize KV cache (FP16 → INT8)
✅ Model quantization beyond Q3 (experimental)

Realistic mobile target:
- Model: 3-4 GB
- KV cache: 0.3-0.5 GB (small context)
- Total: 3.5-4.5 GB
- Requires custom optimization
```

**Summary Table**:

| Use Case | Quantization | Size (GB) | Quality | Rationale |
|----------|--------------|-----------|---------|-----------|
| Customer Support | Q8_0 | 13.8 | 99% | Accuracy critical, GPU available |
| Code Completion | Q4_K_M | 7.9 | 92% | Balanced, fits laptops, fast |
| Mobile App | IQ3_XXS | 4.3 | 65% | Only option <4GB, good enough |

**Validation Metrics**:

**1. Quality Metrics**:
```python
# Perplexity (lower is better)
def measure_perplexity(model, test_set):
    """Measure model perplexity on test set."""
    # Standard benchmark (WikiText, C4, etc.)
    return perplexity_score

# Targets:
# Q8_0:    perplexity < 6.0 (baseline: 5.9)
# Q4_K_M:  perplexity < 7.0
# IQ3_XXS: perplexity < 10.0
```

**2. Task-Specific Metrics**:
```python
# Customer Support: Accuracy on test queries
def measure_support_accuracy(model, test_cases):
    correct = 0
    for query, expected_answer in test_cases:
        response = model.generate(query)
        if evaluate_answer(response, expected_answer):
            correct += 1
    return correct / len(test_cases)

# Target: >95% accuracy

# Code Completion: Exact match rate
def measure_code_accuracy(model, code_tasks):
    matches = 0
    for context, expected_completion in code_tasks:
        completion = model.complete(context)
        if completion.strip() == expected_completion.strip():
            matches += 1
    return matches / len(code_tasks)

# Target: >70% exact match, >90% functionally correct
```

**3. Performance Metrics**:
```python
# Inference speed
def measure_speed(model, prompts):
    times = []
    for prompt in prompts:
        start = time.time()
        model.generate(prompt, max_tokens=100)
        times.append(time.time() - start)
    return {
        'mean': np.mean(times),
        'p50': np.percentile(times, 50),
        'p99': np.percentile(times, 99)
    }

# Targets:
# Customer Support: p99 < 2s
# Code Completion:  p99 < 200ms (for first few tokens)
# Mobile App:       p99 < 5s
```

**4. Resource Usage**:
```python
# Memory usage
def measure_memory(model):
    import psutil
    process = psutil.Process()
    before = process.memory_info().rss / (1024**3)  # GB
    model.generate("test" * 100, max_tokens=100)
    after = process.memory_info().rss / (1024**3)
    return after  # Peak memory

# Targets:
# Customer Support: <24 GB (fits GPU)
# Code Completion:  <12 GB (fits 16GB laptop)
# Mobile App:       <4 GB (hard constraint)
```

**5. User Satisfaction**:
```python
# A/B Testing
# Deploy Q4_K_M vs Q5_K_M for code completion

metrics = {
    'completion_acceptance_rate': 0.85,  # User keeps suggestion
    'completion_retention': 0.70,        # Still there after editing
    'user_rating': 4.2,                   # 1-5 scale
}

# If Q5_K_M shows significant improvement (>5%), use it
# Otherwise, Q4_K_M is more efficient choice
```

**Decision Framework**:

```python
def choose_quantization(
    model_size_b: int,
    use_case: str,
    available_memory_gb: float,
    quality_requirement: str,  # 'high', 'medium', 'low'
    hardware: str  # 'gpu', 'cpu', 'mobile'
) -> str:
    """
    Choose quantization based on constraints.

    Returns: Recommended quantization format
    """

    # Calculate base sizes (LLaMA-13B example)
    size_multipliers = {
        'FP16': 2.0,
        'Q8_0': 1.06,
        'Q6_K': 0.82,
        'Q5_K_M': 0.70,
        'Q4_K_M': 0.61,
        'Q4_0': 0.56,
        'Q3_K_M': 0.46,
        'Q3_K_S': 0.41,
        'IQ3_XXS': 0.33
    }

    base_size_gb = model_size_b * 2  # FP16 baseline

    # Quality requirements
    quality_min_bits = {
        'high': 6,      # Q6_K or better
        'medium': 4.5,  # Q4_K_M or better
        'low': 3        # Any quantization
    }

    min_bits = quality_min_bits[quality_requirement]

    # Filter by quality
    candidates = {
        fmt: multiplier
        for fmt, multiplier in size_multipliers.items()
        if get_bits_per_param(fmt) >= min_bits
    }

    # Filter by memory
    candidates = {
        fmt: multiplier
        for fmt, multiplier in candidates.items()
        if base_size_gb * multiplier < available_memory_gb * 0.7  # 70% utilization
    }

    if not candidates:
        raise ValueError("No quantization meets constraints")

    # Choose best (highest quality that fits)
    best = max(candidates.items(), key=lambda x: get_bits_per_param(x[0]))

    return best[0]

# Example usage:
quant = choose_quantization(
    model_size_b=13,
    use_case='code_completion',
    available_memory_gb=16,
    quality_requirement='medium',
    hardware='cpu'
)
print(f"Recommended: {quant}")  # Q4_K_M
```

---

#### Follow-Up Questions

1. **"What if the mobile app requirement was <2GB instead of <4GB?"**
   *Looking for*: Smaller model (LLaMA-3B/7B), more aggressive quantization, model distillation

2. **"How would you measure quality degradation in production?"**
   *Looking for*: User feedback, task success rate, A/B testing, perplexity on live data

3. **"Could you mix quantization levels (different layers)?"**
   *Looking for*: Mixed precision, higher precision for critical layers (attention), experimental

4. **"What if GPU has only 12GB VRAM instead of 24GB?"**
   *Looking for*: Q5_K_M or Q4_K_M, reduce context window, hybrid CPU/GPU

---

#### Rubric

| Category | Poor (0-1) | Fair (2-3) | Good (4-5) | Excellent (6-7) |
|----------|-----------|-----------|-----------|----------------|
| **Use Case Analysis** | Generic | Basic analysis | Clear rationale | Deep understanding |
| **Quantization Choice** | Same for all | Different choices | Justified choices | Optimal + alternatives |
| **Resource Calculation** | No math | Rough estimates | Accurate | Detailed breakdown |
| **Validation Metrics** | None mentioned | Basic metrics | Multiple metrics | Comprehensive + targets |
| **Trade-off Discussion** | Missing | Some trade-offs | Balanced | Quantitative |

**Passing Score**: 12/35 (Entry), 22/35 (Mid), 30/35 (Senior)

---

#### Related Content

- 📚 [Lesson 1.2: GGUF File Format](../../modules/01-foundations/docs/gguf-format.md)
- 📚 [Module 3: Quantization & Optimization](../../modules/03-quantization/)
- 🔬 [Lab 3.2: Format Comparison](../../modules/03-quantization/labs/lab-02-format-comparison.md)

---

### Question 13-15: [Remaining System Design Questions]

*[Due to space, I'll provide summaries for the remaining system design questions to stay within practical length limits]*

**Question 13: Design for Multi-Model Support**
- Serve 5 different models (7B, 13B, 70B) with model routing
- Handle model loading/unloading based on demand
- Resource allocation and priority management

**Question 14: Edge Deployment Architecture**
- Deploy on embedded devices (Raspberry Pi, Jetson Nano)
- Optimize for limited resources (2-8GB RAM)
- Offline operation and model updates

**Question 15: Batch Inference Pipeline**
- Process 10,000 documents overnight
- Optimize for throughput over latency
- Cost-efficient design

---

## Debugging Questions

### Question 16: Out-of-Memory During Inference

**Category**: Debugging
**Difficulty**: Mid (L4/L5)
**Companies**: All companies
**Time Allotted**: 20 minutes
**Prerequisites**: Module 1, Lessons 1.4-1.5

---

#### Question

A user reports their LLaMA-13B Q4_K_M model (7.9GB) crashes with "Out of Memory" during generation on a machine with 16GB RAM. The error occurs after generating 300-500 tokens. How do you debug and fix this issue?

Walk through your diagnostic process and provide solutions.

---

#### What the Interviewer is Looking For

**Core Competencies Tested**:
- [ ] Debugging methodology
- [ ] Memory management understanding
- [ ] Problem isolation skills
- [ ] Solution generation
- [ ] Prevention thinking

**Red Flags**:
- ❌ Jumps to solutions without diagnosis
- ❌ Doesn't check actual memory usage
- ❌ Ignores KV cache growth
- ❌ No systematic approach

**Green Flags**:
- ✅ Systematic debugging steps
- ✅ Checks memory usage metrics
- ✅ Identifies root cause (KV cache growth)
- ✅ Provides multiple solutions
- ✅ Explains prevention

---

#### Model Solution

**Diagnostic Process**:

**Step 1: Gather Information**
```bash
# 1. Check system memory
free -h
# Output:
#               total        used        free      shared  buff/cache   available
# Mem:            16G         14G         500M        100M         1.5G         1.2G
# Swap:           2G          2G          0B
# Problem: Only 500MB free, swap exhausted!

# 2. Check process memory
ps aux | grep llama
# user  12345  85.2  95.1  15.2G  15.2G ?  R   10:30  llama-cli
# Process using 15.2GB! (95% of 16GB)

# 3. Check OOM killer logs
dmesg | grep -i "out of memory"
# [12345.678] Out of memory: Killed process 12345 (llama-cli)

# 4. Monitor during inference
watch -n 1 'free -h && ps aux | grep llama | head -1'
# Observe: Memory grows from 8GB → 12GB → 15GB → OOM
```

**Step 2: Reproduce and Measure**
```python
# reproduction script
import time
import psutil
from llama_cpp import Llama

def monitor_memory():
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)  # GB

# Load model
print(f"Before load: {monitor_memory():.2f} GB")
model = Llama("llama-13b-q4_k_m.gguf", n_ctx=4096)
print(f"After load: {monitor_memory():.2f} GB")
# Output: 7.9 GB ✅ (expected)

# Generate tokens
prompt = "Write a long story about"
tokens_generated = 0

for i in range(100):  # Generate in chunks
    output = model(prompt, max_tokens=10)
    prompt += output['choices'][0]['text']
    tokens_generated += 10

    mem = monitor_memory()
    print(f"Tokens: {tokens_generated}, Memory: {mem:.2f} GB")

    if mem > 15:
        print("Memory limit approaching!")
        break

# Output:
# Tokens: 10,   Memory: 8.5 GB
# Tokens: 100,  Memory: 10.2 GB
# Tokens: 300,  Memory: 13.8 GB  ← Growing!
# Tokens: 500,  Memory: 15.9 GB  ← OOM imminent!
```

**Step 3: Identify Root Cause**

**KV Cache Growth Analysis**:
```python
# LLaMA-13B architecture
n_layers = 40
n_heads = 40
head_dim = 128
context_length = 4096  # User's setting

# KV cache per token
kv_per_token = 2 * n_layers * n_heads * head_dim * 2  # FP16
kv_per_token_gb = kv_per_token / (1024**3)
print(f"KV cache per token: {kv_per_token_gb * 1000:.2f} MB")
# ~20 MB per token

# Total KV cache for 500 tokens
kv_total = kv_per_token_gb * 500
print(f"KV cache for 500 tokens: {kv_total:.2f} GB")
# ~10 GB just for KV cache!

# Total memory
model_weights = 7.9  # GB
kv_cache_500 = 10.0  # GB
activations = 0.5    # GB
overhead = 0.5       # GB
total = model_weights + kv_cache_500 + activations + overhead
print(f"Total memory at 500 tokens: {total:.1f} GB")
# 18.9 GB ❌ (exceeds 16GB!)
```

**Root Cause**: KV cache grows linearly with generated tokens, eventually exceeding available RAM.

**Solutions**:

**Solution 1: Reduce Context Window** (Immediate fix)
```python
# Instead of n_ctx=4096
model = Llama(
    "llama-13b-q4_k_m.gguf",
    n_ctx=2048  # Half the context
)

# KV cache for 500 tokens at 2048 context:
# Still ~10 GB for 500 tokens!
# But max KV cache capped at 2048 tokens: ~40GB... wait this doesn't help!

# Correction: n_ctx limits max tokens in context, not KV cache size
# KV cache allocates for n_ctx upfront

# Actual calculation:
kv_full_context = kv_per_token_gb * 2048
print(f"KV cache allocated: {kv_full_context:.2f} GB")
# ~41 GB for 2048 context

# This is WRONG! Let me recalculate properly...

# Correct calculation:
# KV cache size = 2 * n_layers * n_ctx * n_heads * head_dim * sizeof(float16)
kv_cache_size = 2 * 40 * 2048 * 40 * 128 * 2 / (1024**3)
print(f"KV cache (2048 ctx): {kv_cache_size:.2f} GB")
# ~5.0 GB ✅

kv_cache_size_4096 = 2 * 40 * 4096 * 40 * 128 * 2 / (1024**3)
print(f"KV cache (4096 ctx): {kv_cache_size_4096:.2f} GB")
# ~10.0 GB

# Total with 2048 context:
total = 7.9 + 5.0 + 0.5 + 0.5
print(f"Total (2048 ctx): {total:.1f} GB")
# 13.9 GB ✅ (fits in 16GB with margin!)
```

**Solution 2: Use Smaller Model**
```python
# LLaMA-7B Q4_K_M instead of 13B
model = Llama("llama-7b-q4_k_m.gguf", n_ctx=4096)

# Model: 4.0 GB
# KV cache (4096): ~4.0 GB (smaller due to fewer layers/heads)
# Total: ~9 GB ✅
```

**Solution 3: More Aggressive Quantization**
```python
# Use Q3_K_M instead of Q4_K_M
model = Llama("llama-13b-q3_k_m.gguf", n_ctx=4096)

# Model: 6.0 GB (vs 7.9 GB)
# KV cache: 10.0 GB (same)
# Total: ~17 GB ⚠️ (still tight, but may work)
```

**Solution 4: Enable Swapping (Not Recommended)**
```bash
# Increase swap space
sudo fallocate -l 8G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Will work but VERY slow (disk I/O bottleneck)
# Inference drops from 20 tok/s to <1 tok/s
```

**Solution 5: Reduce Generation Length**
```python
# Generate in smaller chunks, clear context periodically
def generate_long_text(model, prompt, total_tokens=1000):
    generated = []
    context = prompt

    while len(generated) < total_tokens:
        # Generate chunk
        output = model(context, max_tokens=100)
        chunk = output['choices'][0]['text']
        generated.append(chunk)

        # Keep only recent context (sliding window)
        context = prompt + ''.join(generated[-5:])  # Last 5 chunks

    return ''.join(generated)

# This limits KV cache growth
```

**Solution 6: Upgrade Hardware**
```bash
# Add RAM
# 16GB → 32GB

# Immediate solution but costly
# Good for production deployment
```

**Prevention**:

```python
# 1. Calculate memory before loading
def estimate_memory(
    model_path,
    n_ctx,
    n_layers=40,  # LLaMA-13B
    n_heads=40,
    head_dim=128
):
    # Model size
    import os
    model_size_gb = os.path.getsize(model_path) / (1024**3)

    # KV cache
    kv_cache_gb = 2 * n_layers * n_ctx * n_heads * head_dim * 2 / (1024**3)

    # Total estimate
    total = model_size_gb + kv_cache_gb + 1.0  # +1GB overhead

    return {
        'model_gb': model_size_gb,
        'kv_cache_gb': kv_cache_gb,
        'total_gb': total
    }

mem = estimate_memory("llama-13b-q4_k_m.gguf", n_ctx=4096)
print(f"Estimated memory: {mem['total_gb']:.1f} GB")

import psutil
available_gb = psutil.virtual_memory().available / (1024**3)
print(f"Available: {available_gb:.1f} GB")

if mem['total_gb'] > available_gb * 0.8:
    print("WARNING: May run out of memory!")

# 2. Monitor during inference
def generate_with_monitoring(model, prompt, max_tokens):
    process = psutil.Process()

    for i in range(max_tokens):
        # Check memory before each token
        mem_gb = process.memory_info().rss / (1024**3)

        if mem_gb > 14.0:  # 14GB threshold on 16GB system
            print(f"Memory limit reached at {i} tokens")
            break

        output = model(prompt, max_tokens=1)
        prompt += output['choices'][0]['text']

    return prompt

# 3. Set safe defaults
model = Llama(
    model_path,
    n_ctx=min(2048, recommended_ctx),  # Cap context
    use_mmap=True,                      # Efficient loading
    use_mlock=False,                    # Don't lock memory
)
```

**Summary of Solutions**:

| Solution | Effectiveness | Trade-off |
|----------|---------------|-----------|
| Reduce n_ctx (4096→2048) | ✅ High | Lose long context |
| Use smaller model (7B) | ✅ High | Lower quality |
| Aggressive quant (Q3) | ⚠️ Medium | Quality loss |
| Enable swap | ❌ Low | 10-100x slower |
| Sliding window context | ✅ Medium | Lose history |
| Upgrade RAM | ✅ Perfect | $$ cost |

**Recommended**: Reduce n_ctx to 2048 (immediate), upgrade RAM (long-term)

---

#### Follow-Up Questions

1. **"How do you monitor memory usage in production?"**
   *Looking for*: Prometheus metrics, alerts, memory profiling tools

2. **"What if the OOM happens randomly, not consistently?"**
   *Looking for*: Memory leak debugging, check for concurrent requests, fragmentation

3. **"Could you use GPU to avoid this issue?"**
   *Looking for*: Yes, offload to GPU memory, but same constraints apply (VRAM)

---

#### Rubric

| Category | Poor (0-1) | Fair (2-3) | Good (4-5) | Excellent (6-7) |
|----------|-----------|-----------|-----------|----------------|
| **Diagnostic Process** | No process | Basic checks | Systematic | Professional debugging |
| **Root Cause Analysis** | Wrong diagnosis | Partial understanding | Correct diagnosis | Detailed explanation |
| **Solutions** | No solutions | One solution | Multiple solutions | Prioritized + trade-offs |
| **Prevention** | Not mentioned | Basic prevention | Good practices | Comprehensive |
| **Communication** | Unclear | Understandable | Structured | Clear + actionable |

**Passing Score**: 12/35 (Entry), 20/35 (Mid), 28/35 (Senior)

---

### Question 17-20: [Additional Debugging Questions]

**Question 17: Slow Inference Performance**
- Model generates at 2 tok/s instead of expected 20 tok/s
- Diagnose CPU bottlenecks, threading issues, quantization problems
- Solutions: Enable BLAS, check CPU flags, fix threading

**Question 18: Build Failures with CUDA**
- "nvcc: command not found" or "unsupported GPU architecture"
- Debug CUDA installation, PATH issues, architecture mismatches
- Solutions: Install CUDA toolkit, set env vars, specify architecture

**Question 19: Incorrect Generation Output**
- Model outputs gibberish or repetitive text
- Debug tokenization issues, sampling parameters, model corruption
- Solutions: Fix BOS/EOS tokens, adjust temperature/penalties, verify model

**Question 20: Model Loading Hangs**
- Model loading never completes, process hangs
- Debug file system issues, memory mapping problems, corruption
- Solutions: Check file integrity, disk space, mmap support

---

## Summary

### Question Distribution

**Conceptual Questions (5)**:
1. What is GGUF and Why Does It Exist? (Entry)
2. Why Choose llama.cpp Over Other Engines? (Mid)
3. Explain Quantization and Its Impact (Mid)
4. Context Window and KV Cache (Mid)
5. Build Systems and Cross-Compilation (Entry-Mid)

**Technical Questions (5)**:
6. Calculate Memory Requirements (Mid)
7. Implement Token Generation Loop (Mid - Coding)
8. GGUF Metadata Parsing (Mid - Coding)
9. Sampling Temperature Deep Dive (Mid)
10. Model Loading Performance (Mid-Senior)

**System Design Questions (5)**:
11. Design a Simple Inference Service (Mid)
12. Choose Quantization for Deployment (Mid)
13. Design for Multi-Model Support (Mid-Senior)
14. Edge Deployment Architecture (Mid-Senior)
15. Batch Inference Pipeline (Mid-Senior)

**Debugging Questions (5)**:
16. Out-of-Memory During Inference (Mid)
17. Slow Inference Performance (Mid)
18. Build Failures with CUDA (Entry-Mid)
19. Incorrect Generation Output (Mid)
20. Model Loading Hangs (Mid)

### Difficulty Distribution

- **Entry (L3/L4)**: 4 questions (20%)
- **Mid (L4/L5)**: 14 questions (70%)
- **Senior (L5/L6)**: 2 questions (10%)

### Company Alignment

Questions align with interview patterns from:
- OpenAI (system design, optimization)
- Anthropic (technical depth, safety)
- Meta/FAIR (CUDA, performance)
- Hugging Face (model formats, deployment)
- Startups (practical, production-ready)

---

**Document Created By**: Agent 6 (Interview Coach)
**Last Updated**: 2025-11-18
**Total Questions**: 20
**Estimated Total Interview Time**: 6-8 hours
