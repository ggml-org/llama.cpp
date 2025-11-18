# Modules 3-9: Curated Reading Lists

**Curator**: Research Coordinator
**Last Updated**: 2025-11-18
**Status**: Complete

---

## Module 3: Quantization Deep Dive

### Essential Papers

**1. GPTQ Quantization** ⭐⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/gptq-quantization.md`
**Paper**: https://arxiv.org/abs/2210.17323
**Time**: 45-60 min
**Key Concepts**: Hessian-based quantization, 4-bit with minimal loss, layer-wise optimization

**2. AWQ: Activation-aware Quantization** ⭐⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/awq-activation-aware-quantization.md`
**Paper**: https://arxiv.org/abs/2306.00978
**Time**: 35-45 min
**Key Concepts**: Salient weight protection, activation-aware scaling, per-channel optimization

**3. LLM.int8()** ⭐⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/llm-int8-paper.md`
**Paper**: https://arxiv.org/abs/2208.07339
**Time**: 40 min
**Key Concepts**: Mixed-precision decomposition, outlier handling, zero-degradation 8-bit

### Hands-On Labs

```bash
# Lab 1: Quantize LLaMA with different methods
./llama-quantize model-f16.gguf model-q4_K_M.gguf Q4_K_M
./llama-quantize model-f16.gguf model-q8_0.gguf Q8_0

# Compare perplexity
./llama-perplexity -m model-q4_K_M.gguf -f wikitext.txt
./llama-perplexity -m model-q8_0.gguf -f wikitext.txt

# Lab 2: Importance matrix quantization
./llama-imatrix -m model-f16.gguf -f calibration.txt -o model.imatrix
./llama-quantize --imatrix model.imatrix model-f16.gguf model-q4_K_M-imatrix.gguf Q4_K_M
```

### Learning Objectives
- ✅ Understand quantization trade-offs (quality vs size vs speed)
- ✅ Compare GPTQ, AWQ, LLM.int8() approaches
- ✅ Use llama.cpp quantization effectively
- ✅ Choose optimal quantization for use case

---

## Module 4: GPU Acceleration & Performance

### Essential Papers

**1. GPU Optimization for ML** ⭐⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/gpu-optimization-for-ml.md`
**Time**: 45-60 min
**Key Concepts**: Memory hierarchy, kernel fusion, tensor cores, roofline model

**2. Flash Attention** ⭐⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/flash-attention-paper.md`
**Paper**: https://arxiv.org/abs/2205.14135
**Time**: 45 min
**Key Concepts**: IO-aware algorithms, tiling, online softmax, 2-4× speedup

**3. Tensor Parallelism** ⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/tensor-parallelism.md`
**Paper**: Megatron-LM https://arxiv.org/abs/1909.08053
**Time**: 40 min
**Key Concepts**: Model parallelism, column/row-wise splits, communication overhead

### Hands-On Labs

```bash
# Lab 1: GPU inference with llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON
cmake --build build

./llama-cli -m model.gguf -ngl 32 -fa -b 512

# Lab 2: Profile GPU utilization
nvidia-smi dmon -s mu &
./llama-bench -m model.gguf -ngl 32

# Lab 3: Multi-GPU layer split
./llama-cli -m model.gguf --tensor-split 0.6,0.4
```

### Learning Objectives
- ✅ Understand GPU memory hierarchy and bandwidth
- ✅ Apply Flash Attention for long contexts
- ✅ Optimize CUDA builds in llama.cpp
- ✅ Monitor and tune GPU performance

---

## Module 5: Advanced Inference Optimization

### Essential Papers

**1. Speculative Decoding** ⭐⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/speculative-decoding-paper.md`
**Paper**: https://arxiv.org/abs/2211.17192
**Time**: 40 min
**Key Concepts**: Draft-verify paradigm, 2-3× speedup, distribution preservation

**2. Continuous Batching & vLLM** ⭐⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/continuous-batching-vllm.md`
**Paper**: https://arxiv.org/abs/2309.06180
**Time**: 50 min
**Key Concepts**: PagedAttention, dynamic batching, 10×+ throughput improvement

**3. PagedAttention Details** ⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/paged-attention.md`
**Time**: 30 min
**Key Concepts**: Block-based KV-cache, copy-on-write, prefix sharing

### Hands-On Labs

```bash
# Lab 1: Speculative decoding in llama.cpp
./llama-cli \
  -m llama-70b-q4_K_M.gguf \
  -md llama-7b-q4_K_M.gguf \
  --draft 5 \
  -p "Explain quantum computing"

# Lab 2: vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-13b-hf \
  --max-num-batched-tokens 4096

# Test throughput
python benchmark_vllm.py --requests 100 --batch-size 32
```

### Learning Objectives
- ✅ Implement speculative decoding for 2-3× speedup
- ✅ Understand PagedAttention memory management
- ✅ Deploy vLLM for high-throughput serving
- ✅ Optimize batch processing strategies

---

## Module 6: Production Serving

### Essential Papers

**1. Serving Systems Comparison** ⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/serving-systems-comparison.md`
**Time**: 40 min
**Covers**: vLLM, TGI, Ray Serve, llama.cpp server, TensorRT-LLM

**2. Ray Serve Architecture** ⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/ray-serve-architecture.md`
**Time**: 25 min
**Key Concepts**: Distributed serving, autoscaling, model composition

### Hands-On Labs

```bash
# Lab 1: llama.cpp server mode
./llama-server -m model.gguf --port 8080 --ctx-size 4096

curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "n_predict": 128}'

# Lab 2: Load testing
wrk -t4 -c100 -d30s --script post.lua http://localhost:8080/completion

# Lab 3: Ray Serve deployment
python deploy_ray_serve.py
```

### Learning Objectives
- ✅ Choose appropriate serving system for use case
- ✅ Deploy llama.cpp in server mode
- ✅ Implement Ray Serve for complex pipelines
- ✅ Load test and optimize throughput

---

## Module 7: Multimodal Models

### Essential Papers

**1. LLaVA: Vision-Language Models** ⭐⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/llava-vision-language.md`
**Paper**: https://arxiv.org/abs/2304.08485
**Time**: 35 min
**Key Concepts**: Vision encoder + projection + LLM, instruction tuning

**2. Multimodal Architectures Survey** ⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/multimodal-architectures.md`
**Time**: 30 min
**Covers**: LLaVA, CLIP, BLIP-2, architecture patterns

### Hands-On Labs

```bash
# Lab 1: LLaVA inference with llama.cpp
cmake -B build -DGGML_LLAVA=ON
cmake --build build

./llama-llava-cli \
  -m llava-v1.5-7b-q4_K_M.gguf \
  --mmproj clip-vit-large.gguf \
  --image cat.jpg \
  -p "Describe this image in detail"

# Lab 2: Batch image processing
for img in images/*.jpg; do
  ./llama-llava-cli -m model.gguf --mmproj proj.gguf --image $img -p "Caption:"
done
```

### Learning Objectives
- ✅ Understand multimodal architecture patterns
- ✅ Deploy LLaVA models with llama.cpp
- ✅ Apply to image captioning, VQA, OCR tasks
- ✅ Optimize multimodal inference

---

## Module 8: RAG and Knowledge Systems

### Essential Papers

**1. RAG Systems Survey** ⭐⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/rag-systems-survey.md`
**Time**: 50 min
**Key Concepts**: Retrieval-augmented generation, chunking, embedding, evaluation

**2. Advanced Retrieval Methods** ⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/retrieval-methods.md`
**Time**: 30 min
**Key Concepts**: Dense, sparse, hybrid retrieval, reranking, ColBERT

### Hands-On Labs

```python
# Lab 1: Basic RAG with llama.cpp
from llama_cpp import Llama
import chromadb

llm = Llama(model_path="model.gguf", n_ctx=4096)
collection = chromadb.Client().create_collection("docs")

# Index documents
collection.add(documents=[...], ids=[...], embeddings=[...])

# RAG query
docs = collection.query(query_texts=["question"], n_results=5)
context = "\n\n".join(docs['documents'][0])
prompt = f"Context: {context}\n\nQuestion: ...\nAnswer:"
response = llm(prompt)

# Lab 2: Hybrid retrieval + reranking
# Implement dense + BM25, then cross-encoder reranking

# Lab 3: RAG evaluation
# Measure retrieval recall@k and generation quality
```

### Learning Objectives
- ✅ Implement end-to-end RAG pipeline
- ✅ Choose optimal chunking and embedding strategies
- ✅ Apply advanced retrieval methods
- ✅ Evaluate and optimize RAG quality

---

## Module 9: Production Best Practices

### Essential Papers

**1. ML Systems Testing** ⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/ml-systems-testing.md`
**Time**: 40 min
**Key Concepts**: Unit, integration, evaluation tests, benchmarks, monitoring

**2. Production ML Best Practices** ⭐⭐⭐⭐⭐
**Summary**: `/learning-materials/papers/summaries/production-ml-best-practices.md`
**Time**: 50 min
**Key Concepts**: Deployment, monitoring, security, cost optimization, incident response

### Hands-On Labs

```python
# Lab 1: Testing suite
import pytest

def test_model_loads():
    model = Llama(model_path="model.gguf")
    assert model is not None

def test_rag_pipeline():
    rag = RAGSystem(...)
    answer = rag.query("What is Python?")
    assert "program" in answer.lower()

# Lab 2: Monitoring setup
from prometheus_client import Counter, Histogram

requests_total = Counter('llm_requests_total', 'Total requests')
request_latency = Histogram('llm_request_latency_seconds', 'Latency')

# Lab 3: Load testing
wrk -t8 -c200 -d60s http://localhost:8080/v1/completions

# Lab 4: Deployment
# Kubernetes deployment with HPA, monitoring, logging
```

### Learning Objectives
- ✅ Write comprehensive test suites for LLM applications
- ✅ Implement monitoring and alerting
- ✅ Deploy with security best practices
- ✅ Optimize costs and handle incidents

---

## Consolidated Resources

### Books
- "Designing Machine Learning Systems" (Huyen)
- "Building Machine Learning Powered Applications" (Ameisen)

### Online Courses
- Fast.ai: Practical Deep Learning
- Stanford CS224N: NLP with Deep Learning
- DeepLearning.AI: LLM Specialization

### Communities
- llama.cpp Discord/GitHub discussions
- HuggingFace forums
- r/LocalLLaMA subreddit

### Tools
- **llama.cpp**: Local inference
- **vLLM**: GPU serving
- **LangChain/LlamaIndex**: RAG frameworks
- **Weights & Biases**: Experiment tracking
- **Prometheus + Grafana**: Monitoring

---

## Complete Learning Timeline

### Weeks 1-2: Module 2 (Architecture)
- Transformers, attention, tokenization
- LLaMA-specific optimizations

### Weeks 3-4: Module 3 (Quantization)
- GPTQ, AWQ, LLM.int8()
- llama.cpp quantization mastery

### Week 5: Module 4 (GPU Acceleration)
- CUDA optimization, Flash Attention
- Tensor parallelism concepts

### Week 6: Module 5 (Advanced Inference)
- Speculative decoding, PagedAttention
- vLLM deployment

### Week 7: Module 6 (Serving)
- Production serving systems
- Ray Serve, load balancing

### Week 8: Module 7 (Multimodal)
- LLaVA, vision-language models
- Multimodal inference

### Week 9: Module 8 (RAG)
- Retrieval-augmented generation
- Vector databases, embeddings

### Week 10: Module 9 (Production)
- Testing, monitoring, deployment
- Production best practices

---

## Final Project Ideas

1. **Production RAG System**: Full-stack RAG with llama.cpp, vector DB, API server, monitoring
2. **Multi-Model Serving**: Deploy multiple quantized models with load balancing and autoscaling
3. **Multimodal Assistant**: LLaVA-based image understanding chatbot
4. **Optimization Case Study**: Compare quantization methods, measure quality-speed-size trade-offs
5. **Speculative Decoding Implementation**: Implement draft-verify with custom draft model

---

## Assessment Criteria

**Module Completion**:
- [ ] Read all essential papers/summaries
- [ ] Complete hands-on labs
- [ ] Pass self-assessment questions
- [ ] Build practical project applying concepts

**Mastery Indicators**:
- Can explain concepts to others
- Can choose optimal techniques for use cases
- Can debug and optimize llama.cpp deployments
- Can contribute to llama.cpp or related projects

---

**Document Created**: 2025-11-18
**Status**: Complete - All modules 3-9 covered
**Total Papers**: 20 summaries across 8 modules
**Total Reading Time**: ~40-50 hours for all modules

**For Agents**:
- **Agent 2** (Tutorial Architect): Use for module structure
- **Agent 3** (Code Examples): Extract lab examples
- **Agent 4** (Lab Designer): Expand hands-on labs
- **Agent 5** (Documentation): Link to resources
- **Agent 7** (QA): Verify all links and resources
