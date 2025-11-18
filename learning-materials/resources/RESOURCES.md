# LLaMA.cpp Learning Resources - Comprehensive Guide

**Last Updated**: 2025-11-18
**Maintained By**: Agent 8 (Integration Coordinator)
**Coverage**: All modules, papers, tools, community resources

---

## 📚 Official Documentation & Repositories

### Primary Resources
- **llama.cpp GitHub**: [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
  - Main repository with latest code
  - Issues and discussions
  - Release notes and changelog

- **llama.cpp Documentation**: [https://github.com/ggerganov/llama.cpp/blob/master/docs/](https://github.com/ggerganov/llama.cpp/blob/master/docs/)
  - Build instructions
  - Model compatibility
  - Backend documentation

- **GGML Repository**: [https://github.com/ggerganov/ggml](https://github.com/ggerganov/ggml)
  - Tensor library underlying llama.cpp
  - Low-level implementation details

### Community Resources
- **Discord**: Active community for Q&A and discussions
- **Reddit**: r/LocalLLaMA - Local LLM deployment and optimization
- **Hugging Face**: Model hub for GGUF format models

---

## 📖 Research Papers (Essential Reading)

### Foundational Papers

#### Transformers & Attention
1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Original Transformer paper
   - Multi-head attention mechanism
   - Position embeddings
   - **Link**: https://arxiv.org/abs/1706.03762

2. **"LLaMA: Open and Efficient Foundation Language Models"** (Touvron et al., 2023)
   - LLaMA architecture details
   - RMSNorm, SwiGLU, RoPE
   - Training methodology
   - **Link**: https://arxiv.org/abs/2302.13971

3. **"Llama 2: Open Foundation and Fine-Tuned Chat Models"** (Touvron et al., 2023)
   - Llama 2 improvements
   - Grouped-Query Attention
   - Safety and fine-tuning
   - **Link**: https://arxiv.org/abs/2307.09288

#### Quantization

4. **"GGUF: GPT-Generated Unified Format"** (Gerganov, 2023)
   - GGUF specification
   - Design rationale
   - **Link**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

5. **"LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"** (Dettmers et al., 2022)
   - INT8 quantization techniques
   - Outlier handling
   - **Link**: https://arxiv.org/abs/2208.07339

6. **"GPTQ: Accurate Post-Training Quantization for GPT"** (Frantar et al., 2023)
   - Post-training quantization
   - Quality preservation techniques
   - **Link**: https://arxiv.org/abs/2210.17323

7. **"AWQ: Activation-aware Weight Quantization"** (Lin et al., 2023)
   - Importance-based quantization
   - Activation patterns
   - **Link**: https://arxiv.org/abs/2306.00978

8. **"QuIP: 2-Bit Quantization of Large Language Models With Guarantees"** (Chee et al., 2023)
   - Extreme compression
   - Quality bounds
   - **Link**: https://arxiv.org/abs/2307.13304

#### GPU Acceleration & Optimization

9. **"FlashAttention: Fast and Memory-Efficient Exact Attention"** (Dao et al., 2022)
   - IO-aware attention algorithm
   - Tiling strategy
   - Memory optimization
   - **Link**: https://arxiv.org/abs/2205.14135

10. **"FlashAttention-2: Faster Attention with Better Parallelism"** (Dao, 2023)
    - Improved algorithm
    - Work partitioning
    - **Link**: https://arxiv.org/abs/2307.08691

11. **"Efficient Memory Management for Large Language Model Serving with PagedAttention"** (Kwon et al., 2023)
    - vLLM paper
    - Paged KV cache
    - Memory management
    - **Link**: https://arxiv.org/abs/2309.06180

#### Advanced Inference

12. **"Fast Inference from Transformers via Speculative Decoding"** (Leviathan et al., 2023)
    - Speculative decoding algorithm
    - Draft-target model approach
    - **Link**: https://arxiv.org/abs/2211.17192

13. **"Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads"** (Cai et al., 2024)
    - Multi-head speculation
    - Tree-based verification
    - **Link**: https://arxiv.org/abs/2401.10774

14. **"Orca: A Distributed Serving System for Transformer-Based Generative Models"** (Yu et al., 2022)
    - Continuous batching
    - Iteration-level scheduling
    - **Link**: https://www.usenix.org/conference/osdi22/presentation/yu

#### Multimodal

15. **"Learning Transferable Visual Models From Natural Language Supervision"** (Radford et al., 2021)
    - CLIP paper
    - Vision-language pre-training
    - **Link**: https://arxiv.org/abs/2103.00020

16. **"LLaVA: Visual Instruction Tuning"** (Liu et al., 2023)
    - Multimodal instruction following
    - Vision-language fusion
    - **Link**: https://arxiv.org/abs/2304.08485

17. **"LLaVA-1.5: Improved Baselines"** (Liu et al., 2023)
    - Architecture improvements
    - Better training recipes
    - **Link**: https://arxiv.org/abs/2310.03744

---

## 🛠️ Tools & Software

### Model Conversion & Management

1. **convert.py** (llama.cpp)
   - Convert HuggingFace models to GGUF
   - Included in llama.cpp repository
   - Usage: `python convert.py model_directory`

2. **gguf-py** (llama.cpp)
   - Python library for GGUF manipulation
   - Read/write GGUF metadata
   - Create custom GGUF files

3. **Llama.cpp Model Downloader**
   - Script to download and convert models
   - Automated pipeline
   - In llama.cpp examples

### Quantization Tools

4. **quantize** (llama.cpp binary)
   - Quantize GGUF models to various formats
   - Usage: `./quantize model.gguf model.q4_k_m.gguf Q4_K_M`

5. **AutoGPTQ**
   - GPTQ quantization (alternative to llama.cpp)
   - Research and comparison
   - **Link**: https://github.com/PanQiWei/AutoGPTQ

### Benchmarking & Profiling

6. **perplexity** (llama.cpp binary)
   - Measure model quality
   - Usage: `./perplexity -m model.gguf -f wikitext.txt`

7. **llama-bench** (llama.cpp binary)
   - Performance benchmarking
   - Different batch sizes and contexts
   - Usage: `./llama-bench -m model.gguf`

8. **Nsight Compute** (NVIDIA)
   - CUDA kernel profiling
   - Performance analysis
   - **Link**: https://developer.nvidia.com/nsight-compute

9. **Instruments** (Apple)
   - Metal profiling on macOS/iOS
   - GPU performance analysis
   - Built into Xcode

### Serving & Deployment

10. **llama-cpp-python**
    - Python bindings for llama.cpp
    - High-level API
    - **Link**: https://github.com/abetlen/llama-cpp-python

11. **FastAPI**
    - Modern Python web framework
    - Async support, auto-docs
    - **Link**: https://fastapi.tiangolo.com

12. **Text Generation WebUI**
    - Web UI for llama.cpp
    - LangChain integration
    - **Link**: https://github.com/oobabooga/text-generation-webui

### Integration Frameworks

13. **LangChain**
    - Framework for LLM applications
    - Chains, agents, tools
    - **Link**: https://python.langchain.com

14. **LlamaIndex**
    - Data framework for LLMs
    - RAG, indexing, retrieval
    - **Link**: https://www.llamaindex.ai

15. **Haystack**
    - NLP framework with LLM support
    - Pipelines and agents
    - **Link**: https://haystack.deepset.ai

### Vector Databases

16. **Chroma**
    - Open-source vector database
    - Simple API, good for learning
    - **Link**: https://www.trychroma.com

17. **Pinecone**
    - Cloud vector database
    - Scalable, managed
    - **Link**: https://www.pinecone.io

18. **Weaviate**
    - Open-source vector database
    - GraphQL API
    - **Link**: https://weaviate.io

19. **Qdrant**
    - Vector similarity search engine
    - Rust-based, fast
    - **Link**: https://qdrant.tech

### Monitoring & Observability

20. **Prometheus**
    - Metrics collection and storage
    - Industry standard
    - **Link**: https://prometheus.io

21. **Grafana**
    - Visualization and dashboards
    - Integrates with Prometheus
    - **Link**: https://grafana.com

22. **OpenTelemetry**
    - Distributed tracing
    - Metrics, logs, traces
    - **Link**: https://opentelemetry.io

23. **Jaeger**
    - Distributed tracing system
    - Visualization
    - **Link**: https://www.jaegertracing.io

---

## 💻 Code Examples & Tutorials

### Official Examples (llama.cpp)

1. **examples/main** - CLI chat interface
2. **examples/server** - HTTP server implementation
3. **examples/embedding** - Generate embeddings
4. **examples/quantize** - Model quantization
5. **examples/perplexity** - Quality measurement
6. **examples/llava** - Multimodal inference

### Community Tutorials

#### Beginner Tutorials

1. **"Getting Started with llama.cpp"**
   - Installation and first steps
   - Running your first model
   - **Location**: Module 1 learning materials

2. **"Understanding GGUF Format"**
   - File structure exploration
   - Metadata inspection
   - **Location**: Module 1, Lesson 2

3. **"Quantization Explained"**
   - When and how to quantize
   - Quality vs size tradeoffs
   - **Location**: Module 3

#### Intermediate Tutorials

4. **"Building a Production API Server"**
   - FastAPI + llama.cpp
   - Streaming responses
   - **Location**: Module 6, Capstone Project

5. **"GPU Acceleration Guide"**
   - CUDA setup and optimization
   - Metal on Apple Silicon
   - **Location**: Module 4

6. **"RAG System from Scratch"**
   - Document processing
   - Embedding and retrieval
   - **Location**: Module 8, Capstone Project

#### Advanced Tutorials

7. **"Optimizing CUDA Kernels"**
   - Profiling and optimization
   - Shared memory usage
   - **Location**: Module 4, Advanced Labs

8. **"Multi-GPU Inference"**
   - Tensor parallelism
   - Pipeline parallelism
   - **Location**: Module 4, Capstone Project

9. **"Contributing to llama.cpp"**
   - Development workflow
   - Testing and CI
   - **Location**: Contributing Guide Capstone

---

## 📊 Datasets & Benchmarks

### Perplexity Evaluation

1. **WikiText-103**
   - Long-range dependency modeling
   - **Link**: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/

2. **Penn Treebank**
   - Classic language modeling benchmark
   - **Link**: https://catalog.ldc.upenn.edu/LDC99T42

3. **C4 (Colossal Clean Crawled Corpus)**
   - Web text corpus
   - **Link**: https://www.tensorflow.org/datasets/catalog/c4

### Downstream Task Benchmarks

4. **MMLU (Massive Multitask Language Understanding)**
   - 57 subjects, knowledge testing
   - **Link**: https://github.com/hendrycks/test

5. **HellaSwag**
   - Commonsense reasoning
   - **Link**: https://github.com/rowanz/hellaswag

6. **TruthfulQA**
   - Factual accuracy
   - **Link**: https://github.com/sylinrl/TruthfulQA

7. **HumanEval**
   - Code generation
   - **Link**: https://github.com/openai/human-eval

8. **GSM8K**
   - Grade school math problems
   - **Link**: https://github.com/openai/grade-school-math

### Model Repositories

9. **Hugging Face Model Hub**
   - Thousands of GGUF models
   - **Link**: https://huggingface.co/models?library=gguf

10. **TheBloke's Models**
    - Pre-quantized popular models
    - Multiple formats (GGUF, GPTQ, AWQ)
    - **Link**: https://huggingface.co/TheBloke

---

## 🎓 Online Courses & Learning Paths

### Free Courses

1. **"Deep Learning Specialization"** (Coursera - Andrew Ng)
   - Neural networks fundamentals
   - Free to audit
   - **Link**: https://www.coursera.org/specializations/deep-learning

2. **"Practical Deep Learning for Coders"** (fast.ai)
   - Hands-on approach
   - Completely free
   - **Link**: https://course.fast.ai

3. **"Hugging Face Course"**
   - Transformers library
   - NLP fundamentals
   - **Link**: https://huggingface.co/learn/nlp-course

### Specialized Topics

4. **"Full Stack LLM Bootcamp"** (The Full Stack)
   - Production LLM deployment
   - **Link**: https://fullstackdeeplearning.com

5. **"CUDA Programming"** (NVIDIA)
   - GPU programming basics
   - Optimization techniques
   - **Link**: https://developer.nvidia.com/cuda-education

### YouTube Channels

6. **Andrej Karpathy**
   - Neural networks from scratch
   - Transformer deep dives
   - **Link**: https://www.youtube.com/@AndrejKarpathy

7. **Yannic Kilcher**
   - Paper explanations
   - LLM research updates
   - **Link**: https://www.youtube.com/@YannicKilcher

---

## 🌐 Community & Forums

### Discussion Forums

1. **r/LocalLLaMA** (Reddit)
   - Local LLM deployment
   - Model discussions
   - Performance tips

2. **r/MachineLearning** (Reddit)
   - ML research and news
   - Academic discussions

3. **llama.cpp Discord**
   - Active community
   - Real-time Q&A
   - Model sharing

### Social Media

4. **Twitter/X**
   - Follow: @ggerganov (creator)
   - Follow: @karpathy (AI education)
   - Follow: @rasbt (ML education)

5. **LinkedIn**
   - ML engineering groups
   - Job postings

### Conferences & Events

6. **NeurIPS** - Neural Information Processing Systems
7. **ICML** - International Conference on Machine Learning
8. **ICLR** - International Conference on Learning Representations
9. **MLSys** - Machine Learning and Systems

---

## 📱 Mobile & Edge Resources

### iOS Development

1. **Metal Performance Shaders Guide**
   - Apple's GPU computing framework
   - **Link**: https://developer.apple.com/metal/

2. **Core ML**
   - On-device ML framework
   - **Link**: https://developer.apple.com/machine-learning/core-ml/

### Android Development

3. **Vulkan Programming Guide**
   - Cross-platform GPU API
   - **Link**: https://www.khronos.org/vulkan/

4. **TensorFlow Lite**
   - Mobile inference library
   - **Link**: https://www.tensorflow.org/lite

### Edge Platforms

5. **NVIDIA Jetson**
   - Edge AI platform
   - CUDA support
   - **Link**: https://www.nvidia.com/en-us/autonomous-machines/jetson-store/

6. **Google Coral**
   - TPU-based edge devices
   - **Link**: https://coral.ai

---

## 🔬 Research Groups & Labs

### Academic

1. **Stanford NLP Group**
   - Transformer research
   - **Link**: https://nlp.stanford.edu

2. **UC Berkeley RISELab**
   - ML systems
   - **Link**: https://rise.cs.berkeley.edu

3. **CMU Language Technologies Institute**
   - NLP research
   - **Link**: https://www.lti.cs.cmu.edu

### Industry

4. **Meta AI (FAIR)**
   - LLaMA creators
   - **Link**: https://ai.facebook.com

5. **Google DeepMind**
   - Cutting-edge AI research
   - **Link**: https://www.deepmind.com

6. **OpenAI**
   - GPT series
   - **Link**: https://openai.com/research

7. **Anthropic**
   - Claude models
   - **Link**: https://www.anthropic.com

---

## 📚 Books

### Machine Learning Fundamentals

1. **"Deep Learning"** (Goodfellow, Bengio, Courville)
   - Comprehensive ML textbook
   - Free online
   - **Link**: https://www.deeplearningbook.org

2. **"Hands-On Machine Learning"** (Aurélien Géron)
   - Practical ML with Python
   - Scikit-Learn, Keras, TensorFlow

### Specialized Topics

3. **"Natural Language Processing with Transformers"** (Tunstall, von Werra, Wolf)
   - Hugging Face book
   - Transformer architectures
   - **Link**: https://transformersbook.com

4. **"Building LLMs for Production"** (Ozdemir)
   - Production deployment focus
   - System design

5. **"Programming Massively Parallel Processors"** (Kirk, Hwu)
   - CUDA programming
   - GPU architecture

---

## 🔗 Additional Resources

### Blogs & Newsletters

1. **The Batch** (DeepLearning.AI)
   - Weekly AI newsletter
   - **Link**: https://www.deeplearning.ai/the-batch/

2. **Papers With Code**
   - Latest research papers
   - Implementation code
   - **Link**: https://paperswithcode.com

3. **Hugging Face Blog**
   - Model releases and tutorials
   - **Link**: https://huggingface.co/blog

4. **Sebastian Raschka's Blog**
   - ML education and tutorials
   - **Link**: https://sebastianraschka.com/blog/

### Podcasts

5. **The TWIML AI Podcast**
   - ML interviews and news
   - **Link**: https://twimlai.com

6. **Practical AI**
   - Applied ML discussions
   - **Link**: https://changelog.com/practicalai

7. **Latent Space Podcast**
   - AI engineering focus
   - **Link**: https://www.latent.space/podcast

---

## 🎯 Learning Path Resources by Module

### Module 1: Foundations
- llama.cpp README
- GGUF specification
- Installation guides
- Basic examples

### Module 2: Core Implementation
- "Attention Is All You Need" paper
- LLaMA architecture papers
- Transformer tutorials

### Module 3: Quantization
- GPTQ, AWQ, LLM.int8() papers
- Quantization comparisons
- Perplexity benchmarks

### Module 4: GPU Acceleration
- CUDA programming guides
- FlashAttention papers
- Nsight profiler documentation

### Module 5: Advanced Inference
- Speculative decoding papers
- vLLM (PagedAttention) paper
- Continuous batching guides

### Module 6: Server & Production
- FastAPI documentation
- Kubernetes tutorials
- Prometheus & Grafana guides

### Module 7: Multimodal
- CLIP paper
- LLaVA papers
- Multimodal examples

### Module 8: Integration
- LangChain documentation
- LlamaIndex guides
- Vector database tutorials

### Module 9: Production Engineering
- SRE books (Google)
- Observability guides
- Security best practices

---

## 🆘 Getting Help

### Official Support
1. GitHub Issues: Report bugs, feature requests
2. Discord: Real-time community help
3. Documentation: Check docs first

### Community Support
1. Reddit: r/LocalLLaMA for discussions
2. Stack Overflow: Tag `llama.cpp`
3. Twitter/X: Use hashtag #llamacpp

### Professional Support
1. Consulting services (various companies)
2. Enterprise support (contact vendors)
3. Training programs (online courses)

---

**Maintained By**: Multi-Agent Learning System
**Last Updated**: 2025-11-18
**Contributions**: Welcome! Suggest additions via GitHub issues

**Ready to dive deeper?** → Start with [Complete Learning Path](../COMPLETE_LEARNING_PATH.md)
