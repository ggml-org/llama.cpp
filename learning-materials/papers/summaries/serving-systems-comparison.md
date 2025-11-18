# LLM Serving Systems: Comparative Analysis

**Topic**: Production LLM Serving Infrastructure
**Module**: 6 - Production Serving
**Impact**: ⭐⭐⭐⭐

---

## Executive Summary

This document compares major LLM serving systems (vLLM, TGI, Ray Serve, llama.cpp server, TensorRT-LLM) across architecture, performance, and use cases.

---

## 1. System Comparison Matrix

| System | Best For | Throughput | Latency | GPU Support | CPU Support | Complexity |
|--------|----------|-----------|---------|-------------|-------------|------------|
| **vLLM** | High-throughput GPU serving | ★★★★★ | ★★★★ | ✅ Excellent | ❌ No | Medium |
| **TGI** | Production HuggingFace models | ★★★★ | ★★★★ | ✅ Good | ❌ Limited | Medium |
| **Ray Serve** | Distributed, multi-model | ★★★★ | ★★★ | ✅ Good | ✅ Yes | High |
| **llama.cpp** | Local/CPU/Edge inference | ★★ | ★★★★★ | ✅ Basic | ✅ Excellent | Low |
| **TensorRT-LLM** | Maximum GPU performance | ★★★★★ | ★★★★★ | ✅ NVIDIA only | ❌ No | High |

---

## 2. vLLM (Covered in Detail)

**Architecture**: PagedAttention + Continuous Batching
**Strengths**:
- Highest throughput (12× vs naive HF)
- Excellent memory efficiency
- Easy deployment

**Use Cases**:
- API serving (OpenAI-compatible)
- High request volume
- Cost optimization

```bash
# Deployment
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-13b-hf \
  --max-model-len 4096
```

---

## 3. Text Generation Inference (TGI)

**Organization**: HuggingFace
**Architecture**: Custom CUDA kernels + dynamic batching

**Key Features**:
- Native HuggingFace model support
- Token streaming
- Distributed inference (multi-GPU)
- Safetensors loading
- Flash Attention integration

**Deployment**:
```bash
docker run -p 8080:80 \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-7b-hf \
  --max-batch-size 128
```

**Performance**: 2-4× throughput vs naive, 60-70% of vLLM

**Best For**:
- HuggingFace ecosystem integration
- Production-ready Docker images
- Managed deployment (HF Inference Endpoints)

---

## 4. Ray Serve

**Organization**: Anyscale
**Architecture**: Distributed Python framework

**Key Features**:
- Multi-model serving
- Autoscaling
- Request batching
- Model composition (pipelines)
- Framework agnostic

**Example**:
```python
from ray import serve
import ray

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class LLMDeployment:
    def __init__(self, model_name):
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt):
        return self.model.generate(prompt)

serve.run(LLMDeployment.bind("meta-llama/Llama-2-7b-hf"))
```

**Best For**:
- Complex ML pipelines (RAG, ensembles)
- Multi-model deployments
- Autoscaling requirements
- Python-first teams

**Limitations**:
- Higher overhead than specialized systems
- More complex to configure optimally

---

## 5. llama.cpp Server

**Architecture**: CPU-optimized quantized inference

**Key Features**:
- Excellent CPU performance
- Low memory usage (quantization)
- Cross-platform (Mac, Linux, Windows)
- No Python dependencies
- Simple deployment

**Deployment**:
```bash
# Build
cmake -B build && cmake --build build

# Serve
./llama-server -m model-q4_K_M.gguf \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 0  # CPU mode
```

**API**:
```bash
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "n_predict": 128}'
```

**Best For**:
- Edge deployment
- CPU-only environments
- Low-resource inference
- MacBooks (Metal acceleration)
- Embedded systems

**Limitations**:
- Lower throughput than GPU systems
- Basic batching (improving in recent versions)
- Single-model per process

---

## 6. TensorRT-LLM

**Organization**: NVIDIA
**Architecture**: Highly optimized CUDA kernels + TensorRT compiler

**Key Features**:
- Maximum GPU utilization
- Custom kernel compilation
- FP8 quantization (H100)
- Multi-GPU/multi-node
- C++ runtime (no Python overhead)

**Performance**: Best-in-class for NVIDIA GPUs
- LLaMA 7B: 1000+ tokens/sec (vs 300-400 for vLLM)
- But: Much more complex to deploy

**Deployment**:
```python
# Build TensorRT-LLM engine (one-time)
python build.py \
  --model_dir ./llama-2-7b \
  --dtype float16 \
  --use_gpt_attention_plugin \
  --use_gemm_plugin \
  --output_dir ./trt_engines

# Serve
python run.py --engine_dir ./trt_engines
```

**Best For**:
- Maximum performance on NVIDIA GPUs
- High-value use cases justifying complexity
- Teams with CUDA expertise

**Limitations**:
- NVIDIA GPUs only
- Complex build process
- Longer iteration time (engine compilation)

---

## 7. Decision Matrix

### Choose **vLLM** if:
✅ Need high throughput
✅ GPU available
✅ Want simplicity + performance
✅ OpenAI API compatibility needed

### Choose **TGI** if:
✅ Using HuggingFace models
✅ Want managed deployment (HF Endpoints)
✅ Need Docker-first deployment
✅ Moderate throughput sufficient

### Choose **Ray Serve** if:
✅ Complex ML pipelines (RAG, multi-model)
✅ Need autoscaling
✅ Python-first team
✅ Multiple models in production

### Choose **llama.cpp** if:
✅ CPU-only or edge deployment
✅ MacBook / consumer hardware
✅ Low resource constraints
✅ Quantization acceptable

### Choose **TensorRT-LLM** if:
✅ Need absolute maximum performance
✅ NVIDIA GPUs only
✅ Have CUDA expertise
✅ Can invest in complex setup

---

## 8. Hybrid Approaches

### vLLM + Ray Serve
```python
# Use Ray for orchestration, vLLM for inference
@serve.deployment
class vLLMDeployment:
    def __init__(self):
        from vllm import LLM
        self.llm = LLM("meta-llama/Llama-2-7b-hf")

    def generate(self, prompt):
        return self.llm.generate(prompt)

# Benefits: vLLM performance + Ray scalability
```

### llama.cpp + API Gateway
```
nginx → load balance → multiple llama-server instances
- Each instance: Different model or quantization
- Scale horizontally on CPU clusters
```

---

## 9. Performance Benchmarks

**LLaMA 13B on A100 (batch=32, input=512, output=128)**

| System | Throughput (tokens/s) | Latency p50 (ms) | Memory (GB) |
|--------|----------------------|------------------|-------------|
| TensorRT-LLM | 4,200 | 18 | 14 |
| vLLM | 2,800 | 22 | 16 |
| TGI | 1,900 | 28 | 18 |
| Ray Serve (HF) | 800 | 45 | 22 |

**LLaMA 7B on CPU (12-core, no GPU)**

| System | Throughput (tokens/s) | Memory (GB) |
|--------|----------------------|-------------|
| llama.cpp (Q4_K_M) | 25 | 4.5 |
| llama.cpp (Q8_0) | 18 | 7.5 |
| HF Transformers | 5 | 14 |

---

## 10. Key Takeaways

**For Production**:
- **GPU serving**: vLLM or TensorRT-LLM
- **CPU/Edge**: llama.cpp
- **Complex pipelines**: Ray Serve
- **HF ecosystem**: TGI

**Trends**:
- Continuous batching becoming standard
- PagedAttention-style memory management spreading
- Quantization + GPU (FP8, INT4) improving
- Serverless LLM offerings emerging

---

## Further Reading

- **vLLM**: https://github.com/vllm-project/vllm
- **TGI**: https://github.com/huggingface/text-generation-inference
- **Ray Serve**: https://docs.ray.io/en/latest/serve/
- **TensorRT-LLM**: https://github.com/NVIDIA/TensorRT-LLM

---

**Status**: Complete | Module 6 (1/2) papers
