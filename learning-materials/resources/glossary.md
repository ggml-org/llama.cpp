# LLaMA.cpp Learning System - Glossary

A comprehensive glossary of terms used throughout the learning materials. Terms are organized alphabetically with clear, accessible definitions and cross-references.

**Last Updated**: 2025-11-18
**Coverage**: Modules 1-9
**Total Terms**: 50+ (expanding)

---

## A

### Activation
The output values produced by neurons (or layers) in a neural network after applying an activation function. In LLMs, activations are intermediate tensor values computed during the forward pass. See also: [Forward Pass](#forward-pass), [Tensor](#tensor).

### Attention Mechanism
A core component of transformer models that allows the model to focus on different parts of the input when processing each token. The attention mechanism computes weighted relationships between all tokens in a sequence. See also: [Self-Attention](#self-attention), [Multi-Head Attention](#multi-head-attention), [KV Cache](#kv-cache).

### AWQ (Activation-aware Weight Quantization)
A quantization method that preserves important weights based on activation magnitudes, achieving better quality than naive quantization at the same bit width. Not directly used in llama.cpp but important for understanding quantization approaches. See also: [Quantization](#quantization), [GPTQ](#gptq).

---

## B

### Backend
The computational engine that executes tensor operations. llama.cpp supports multiple backends including CPU (default), CUDA (NVIDIA GPUs), Metal (Apple Silicon), ROCm (AMD GPUs), and SYCL (Intel GPUs). The backend is selected at build time. See also: [CUDA](#cuda), [Metal](#metal).

### Batching
Processing multiple inference requests simultaneously to improve throughput. Batching amortizes fixed costs (like model loading) across multiple requests. See also: [Continuous Batching](#continuous-batching), [Throughput](#throughput).

### BOS (Beginning of Sequence)
A special token inserted at the start of a sequence to indicate the beginning. Different models use different BOS tokens. In llama models, it's typically token ID 1. See also: [EOS](#eos-end-of-sequence), [Special Tokens](#special-tokens).

### BPE (Byte Pair Encoding)
A tokenization algorithm that iteratively merges the most frequent pairs of bytes or characters to build a vocabulary. BPE balances vocabulary size with the ability to represent rare words. See also: [Tokenization](#tokenization), [SentencePiece](#sentencepiece).

---

## C

### Context Window
The maximum number of tokens a model can process at once. This includes both the prompt and the generated output. Common context sizes are 2048, 4096, 8192, or 32768 tokens. Larger context windows require more memory for the KV cache. See also: [KV Cache](#kv-cache), [Token](#token).

### CUDA
NVIDIA's parallel computing platform and programming model. CUDA enables llama.cpp to run inference on NVIDIA GPUs, significantly accelerating computation. Requires NVIDIA GPU and CUDA Toolkit. See also: [Backend](#backend), [Kernel](#kernel).

### Continuous Batching
An advanced batching technique where new requests are added to a batch as soon as previous requests complete, rather than waiting for all requests in a batch to finish. Maximizes GPU utilization. See also: [Batching](#batching).

---

## D

### Decoding
The process of converting model outputs (logits) into tokens through sampling. Decoding strategies include greedy decoding, beam search, and various sampling methods. See also: [Sampling](#sampling), [Logits](#logits), [Token](#token).

### Dynamic Quantization
Quantization applied at runtime during inference, as opposed to static quantization applied once during model conversion. llama.cpp primarily uses static quantization stored in GGUF files. See also: [Quantization](#quantization), [GGUF](#gguf).

---

## E

### Embedding
A continuous vector representation of a token. The embedding layer converts discrete token IDs into dense vectors that the model can process. Embeddings capture semantic relationships between tokens. See also: [Token](#token), [Vocabulary](#vocabulary).

### EOS (End of Sequence)
A special token indicating the end of generated text. When the model generates an EOS token, generation typically stops. In llama models, it's typically token ID 2. See also: [BOS](#bos-beginning-of-sequence), [Special Tokens](#special-tokens).

---

## F

### Feed-Forward Network (FFN)
A component of each transformer layer consisting of linear transformations and non-linear activations. In llama models, the FFN uses a SwiGLU activation. FFNs introduce non-linearity and increase model capacity. See also: [Transformer](#transformer), [Layer](#layer).

### Float16 (FP16)
16-bit floating-point format, half the size of standard 32-bit floats. Often used for neural network weights and activations to reduce memory usage while maintaining reasonable precision. See also: [Quantization](#quantization), [Precision](#precision).

### Forward Pass
The process of computing model outputs from inputs by passing data through each layer sequentially. During inference, the forward pass generates logits for the next token. See also: [Inference](#inference), [Layer](#layer).

---

## G

### GBNF (GGML BNF)
A grammar format used by llama.cpp to constrain text generation. GBNF grammars ensure outputs follow specific structures (like JSON schemas). Based on Backus-Naur Form notation. See also: [Grammar](#grammar), [Constrained Generation](#constrained-generation).

### GGML
The tensor library underlying llama.cpp, providing CPU and GPU operations for inference. GGML handles memory management, tensor operations, and computational graphs. Named after creator Georgi Gerganov. See also: [Tensor](#tensor), [Backend](#backend).

### GGUF
The file format used by llama.cpp to store models. GGUF (GGML Unified Format) replaced the older GGML format and includes metadata, vocabulary, and quantized weights in a single file. See also: [Quantization](#quantization), [Metadata](#metadata).

### GPTQ (Generative Pre-trained Transformer Quantization)
A post-training quantization method that minimizes reconstruction error. While not used by llama.cpp directly, understanding GPTQ helps contextualize GGUF's quantization approach. See also: [Quantization](#quantization), [AWQ](#awq-activation-aware-weight-quantization).

### GPU (Graphics Processing Unit)
A specialized processor designed for parallel computations. GPUs dramatically accelerate LLM inference through massive parallelism. llama.cpp supports NVIDIA (CUDA), AMD (ROCm), and Apple (Metal) GPUs. See also: [CUDA](#cuda), [Backend](#backend).

### Grammar
A set of rules defining valid output structures. llama.cpp can use grammars to ensure generated text follows specific formats like JSON, XML, or custom patterns. See also: [GBNF](#gbnf-ggml-bnf), [Constrained Generation](#constrained-generation).

### Greedy Decoding
A simple sampling strategy that always selects the token with the highest probability. Greedy decoding is deterministic but can lead to repetitive or suboptimal outputs. See also: [Sampling](#sampling), [Top-K Sampling](#top-k-sampling).

---

## H

### HuggingFace
A popular platform for sharing and using ML models. Many models are distributed in HuggingFace format and must be converted to GGUF for use with llama.cpp. See also: [GGUF](#gguf), [Model Conversion](#model-conversion).

---

## I

### Inference
The process of using a trained model to generate outputs (predictions) from inputs. In LLMs, inference generates text token-by-token. Distinct from training, which adjusts model weights. See also: [Forward Pass](#forward-pass), [Token Generation](#token-generation).

### IQ (Importance Quantization)
A GGUF quantization format family (IQ1, IQ2, IQ3, IQ4) that uses importance-based methods to achieve very low bit-widths while preserving quality. Particularly effective for aggressive compression. See also: [Quantization](#quantization), [K-Quants](#k-quants).

---

## K

### Kernel
A GPU function that executes on many threads in parallel. In llama.cpp's CUDA backend, kernels implement operations like matrix multiplication and attention. See also: [CUDA](#cuda), [GPU](#gpu-graphics-processing-unit).

### K-Quants
A family of quantization formats in GGUF (Q4_K, Q5_K, Q6_K, etc.) that use sophisticated block-wise quantization strategies. The "K" denotes larger block sizes and better quality. Variants include _S (small), _M (medium), and _L (large). See also: [Quantization](#quantization), [Block-wise Quantization](#block-wise-quantization).

### KV Cache
A memory optimization that stores previously computed Key and Value tensors from the attention mechanism. The KV cache eliminates redundant computation when generating tokens sequentially, but grows with context length. See also: [Attention Mechanism](#attention-mechanism), [Memory](#memory).

---

## L

### Latency
The time delay between submitting a request and receiving a response. In LLM inference, latency includes time to process the prompt and generate tokens. Lower latency is critical for interactive applications. See also: [Throughput](#throughput), [Time to First Token](#time-to-first-token-ttft).

### Layer
A computational unit in a neural network. Transformer models consist of many identical layers stacked sequentially. Each layer contains attention and feed-forward sub-components. See also: [Transformer](#transformer), [Feed-Forward Network](#feed-forward-network-ffn).

### Layer Normalization
A normalization technique applied within each layer to stabilize training and improve performance. llama models use RMSNorm, a variant of layer normalization. See also: [Layer](#layer), [Normalization](#normalization).

### LLaMA (Large Language Model Meta AI)
A family of open-source language models released by Meta. llama.cpp was originally designed for LLaMA models but now supports many architectures. See also: [Model Architecture](#model-architecture).

### llama.cpp
A C/C++ implementation of LLM inference optimized for efficiency on consumer hardware. Supports CPU and GPU inference with various quantization formats. Created by Georgi Gerganov. See also: [GGML](#ggml), [GGUF](#gguf).

### Logits
Raw, unnormalized scores output by the model for each token in the vocabulary. Logits are converted to probabilities via softmax before sampling. Higher logits indicate higher confidence. See also: [Sampling](#sampling), [Softmax](#softmax).

---

## M

### Memory Mapping (mmap)
A technique that maps file contents directly into memory, allowing the OS to load model weights on-demand. Reduces startup time and memory usage compared to loading the entire model upfront. See also: [Memory](#memory), [GGUF](#gguf).

### Memory
The RAM (or VRAM on GPUs) required for inference. Memory usage includes model weights, KV cache, and activations. Quantization and memory mapping reduce memory requirements. See also: [KV Cache](#kv-cache), [Quantization](#quantization).

### Metal
Apple's GPU programming framework for macOS and iOS. llama.cpp's Metal backend enables GPU acceleration on Apple Silicon (M1, M2, M3) and AMD GPUs in Macs. See also: [Backend](#backend), [GPU](#gpu-graphics-processing-unit).

### Metadata
Information stored in GGUF files describing the model: architecture type, hyperparameters, vocabulary, and training details. Metadata enables llama.cpp to correctly load and run models. See also: [GGUF](#gguf).

### Mirostat
An adaptive sampling algorithm that maintains consistent perplexity by dynamically adjusting the sampling distribution. Produces more coherent long-form text than static sampling. See also: [Sampling](#sampling), [Perplexity](#perplexity).

### mmap
See [Memory Mapping](#memory-mapping-mmap).

### Model Architecture
The structure and organization of a neural network: number of layers, hidden dimensions, attention heads, etc. llama.cpp supports LLaMA, Mistral, Mixtral, GPT-2, and many other architectures. See also: [Transformer](#transformer).

### Model Conversion
The process of converting models from formats like HuggingFace/PyTorch to GGUF. llama.cpp provides Python scripts for conversion. See also: [GGUF](#gguf), [HuggingFace](#huggingface).

### Multi-Head Attention (MHA)
An attention mechanism using multiple parallel attention "heads," each learning different aspects of relationships between tokens. Outputs are concatenated and projected. See also: [Attention Mechanism](#attention-mechanism), [Grouped-Query Attention](#grouped-query-attention-gqa).

---

## N

### Normalization
Techniques that rescale values to improve training stability and performance. Layer normalization and RMSNorm are common in transformers. See also: [Layer Normalization](#layer-normalization).

---

## O

### Offloading
Splitting computation between CPU and GPU by running some layers on CPU and others on GPU. Useful when GPU memory is limited. llama.cpp supports layer offloading via the `-ngl` parameter. See also: [GPU](#gpu-graphics-processing-unit).

---

## P

### Parameter
A learnable weight in a neural network. Model size is often described by parameter count (e.g., "7B parameters" = 7 billion parameters). More parameters generally mean more capability but higher memory and compute requirements. See also: [Quantization](#quantization), [Memory](#memory).

### Perplexity
A metric measuring how well a model predicts a sequence. Lower perplexity indicates better prediction. Used to evaluate model quality and quantization impact. See also: [Quantization](#quantization).

### Precision
The numerical format used to represent values. Higher precision (e.g., float32) provides more accuracy but uses more memory. Quantization reduces precision to save memory. See also: [Float16](#float16-fp16), [Quantization](#quantization).

### Prompt
The input text provided to a language model. The model processes the prompt and generates a completion. Prompt engineering involves crafting prompts to elicit desired outputs. See also: [Token](#token), [Context Window](#context-window).

---

## Q

### Quantization
Reducing the precision of model weights from floating-point (e.g., 16-bit or 32-bit) to lower bit-widths (e.g., 4-bit or 8-bit). Quantization dramatically reduces model size and memory usage with minimal quality loss. See also: [K-Quants](#k-quants), [GGUF](#gguf).

---

## R

### Repetition Penalty
A parameter that discourages the model from repeating tokens or phrases. Higher values reduce repetition but may affect coherence. Typically ranges from 1.0 (no penalty) to 1.3. See also: [Sampling](#sampling).

### ROCm
AMD's open-source platform for GPU computing, similar to CUDA. llama.cpp's ROCm backend enables inference on AMD GPUs. See also: [Backend](#backend), [CUDA](#cuda).

### RoPE (Rotary Position Embedding)
A method for encoding token positions in the input sequence. RoPE rotates embedding vectors based on position, enabling the model to understand token order without explicit position embeddings. Used in LLaMA and many modern models. See also: [Embedding](#embedding), [Transformer](#transformer).

---

## S

### Sampling
The process of selecting the next token from the probability distribution output by the model. Sampling strategies balance randomness and quality. See also: [Greedy Decoding](#greedy-decoding), [Top-K Sampling](#top-k-sampling), [Temperature](#temperature).

### Self-Attention
An attention mechanism where each token attends to all tokens in the same sequence (including itself). Self-attention allows the model to capture relationships between words regardless of distance. See also: [Attention Mechanism](#attention-mechanism), [Transformer](#transformer).

### SentencePiece
A tokenization library that learns subword units directly from raw text. Used by LLaMA and many other models. Handles multilingual text well. See also: [Tokenization](#tokenization), [BPE](#bpe-byte-pair-encoding).

### Softmax
A function that converts logits into a probability distribution over the vocabulary. Each token's probability is proportional to the exponential of its logit. See also: [Logits](#logits), [Sampling](#sampling).

### Special Tokens
Tokens with specific meanings like BOS (beginning), EOS (end), PAD (padding), or SEP (separator). Different models use different special tokens. See also: [BOS](#bos-beginning-of-sequence), [EOS](#eos-end-of-sequence).

### Speculative Decoding
An optimization technique where a small "draft" model generates candidate tokens quickly, and a larger "target" model verifies them. Reduces overall latency when the draft model is accurate. See also: [Inference](#inference), [Latency](#latency).

---

## T

### Temperature
A sampling parameter controlling randomness. Lower temperatures (< 1.0) make outputs more deterministic and focused. Higher temperatures (> 1.0) increase randomness and diversity. Temperature of 0 is equivalent to greedy decoding. See also: [Sampling](#sampling), [Greedy Decoding](#greedy-decoding).

### Tensor
A multi-dimensional array of numbers. In deep learning, tensors represent data (inputs, weights, activations). llama.cpp uses GGML for tensor operations. See also: [GGML](#ggml), [Activation](#activation).

### Throughput
The number of tokens or requests processed per unit time. Higher throughput means more efficient resource utilization. Measured in tokens/second or requests/second. See also: [Latency](#latency), [Batching](#batching).

### Time to First Token (TTFT)
The latency between submitting a prompt and receiving the first generated token. TTFT depends on prompt length and model size. Critical for interactive applications. See also: [Latency](#latency), [Prompt](#prompt).

### Token
The basic unit of text processed by language models. A token might be a word, subword, or character depending on the tokenizer. Models have a fixed vocabulary of tokens. See also: [Tokenization](#tokenization), [Vocabulary](#vocabulary).

### Token Generation
The iterative process of producing output text one token at a time. Each token is sampled based on the probability distribution from the model, conditioned on previous tokens. See also: [Inference](#inference), [Sampling](#sampling).

### Tokenization
The process of converting text into tokens (and vice versa). Tokenization is model-specific and critical for correct inference. See also: [BPE](#bpe-byte-pair-encoding), [SentencePiece](#sentencepiece), [Token](#token).

### Top-K Sampling
A sampling strategy that considers only the K most probable tokens at each step. Reduces the chance of selecting unlikely tokens while maintaining diversity. Typical K values range from 10 to 100. See also: [Sampling](#sampling), [Top-P Sampling](#top-p-sampling).

### Top-P Sampling (Nucleus Sampling)
A sampling strategy that selects from the smallest set of tokens whose cumulative probability exceeds P. Adapts the number of candidates based on the probability distribution. Typical P values are 0.9 or 0.95. See also: [Sampling](#sampling), [Top-K Sampling](#top-k-sampling).

### Transformer
The dominant neural network architecture for LLMs, introduced in "Attention is All You Need" (2017). Transformers use self-attention and feed-forward layers stacked in sequence. See also: [Attention Mechanism](#attention-mechanism), [Layer](#layer).

---

## U

### Unified Memory
A memory model where CPU and GPU share the same memory space, eliminating explicit data transfers. Supported by some Apple Silicon devices. llama.cpp can leverage unified memory on compatible hardware. See also: [Memory](#memory), [Metal](#metal).

---

## V

### Vocabulary
The set of all tokens a model can use. Vocabulary size typically ranges from 32,000 to 128,000 tokens. Stored in GGUF metadata. See also: [Token](#token), [Tokenization](#tokenization).

---

## Cross-Reference Index

### By Module

**Module 1 (Foundations)**:
llama.cpp, GGUF, GGML, Backend, Inference, Token, Tokenization, Vocabulary, Context Window, KV Cache, Memory, Quantization, Model Architecture, Metadata, Forward Pass, Logits, Sampling

**Module 2 (Core Implementation)**:
Transformer, Attention Mechanism, Self-Attention, Multi-Head Attention, Feed-Forward Network, Layer, Layer Normalization, RoPE, BPE, SentencePiece, Special Tokens, BOS, EOS, Embedding, Grammar, GBNF

**Module 3 (Quantization)**:
Quantization, K-Quants, IQ, Precision, Float16, Perplexity, GPTQ, AWQ, Block-wise Quantization

**Module 4 (GPU Acceleration)**:
GPU, CUDA, Metal, ROCm, Backend, Kernel, Offloading, Unified Memory, Throughput, Latency

**Module 5 (Advanced Inference)**:
Speculative Decoding, Batching, Continuous Batching, Constrained Generation

### By Difficulty Level

**Beginner**: Token, Prompt, Inference, Model Architecture, Vocabulary, Sampling, Temperature, Greedy Decoding, BOS, EOS

**Intermediate**: Quantization, KV Cache, Context Window, Attention Mechanism, Layer, GGUF, Backend, Logits, Softmax, Tokenization

**Advanced**: K-Quants, Transformer, RoPE, Speculative Decoding, Kernel, Memory Mapping, Perplexity, Continuous Batching

---

## Additional Terms

### Block-wise Quantization
Quantizing weights in blocks rather than individually or per-tensor. K-quants use block-wise quantization for better quality preservation. See also: [K-Quants](#k-quants), [Quantization](#quantization).

### Constrained Generation
Generating text that conforms to specific rules or structures. Implemented via grammars in llama.cpp. See also: [Grammar](#grammar), [GBNF](#gbnf-ggml-bnf).

### Grouped-Query Attention (GQA)
An attention variant that reduces the number of KV heads compared to query heads, decreasing memory usage while maintaining quality. Used in models like LLaMA-2 70B and Mistral. See also: [Multi-Head Attention](#multi-head-attention-mha), [KV Cache](#kv-cache).

---

## Glossary Usage Guide

**For Beginners**: Start with terms marked "Beginner" in the difficulty index. Read definitions in order of the learning path.

**For Reference**: Use Ctrl+F / Cmd+F to search for specific terms. Follow cross-references (in "See also" sections) to understand related concepts.

**For Deep Learning**: Read all definitions in a category (e.g., all quantization-related terms) to build comprehensive understanding.

---

**Maintained by**: Agent 8 (Integration Coordinator)
**Contributors**: All learning system agents
**Last Updated**: 2025-11-18
**Status**: Living document - expanding with new modules

Found a term that's missing or unclear? Suggest additions to improve this glossary!

---

## W

### Weight
A learnable parameter in a neural network, typically a value in a matrix that is multiplied with input during forward pass. Model size is determined by the number of weights. See also: [Parameter](#parameter), [Quantization](#quantization).

### WMMA (Warp Matrix Multiply-Accumulate)
CUDA API for using Tensor Cores to accelerate matrix multiplication. Operates on warp-sized matrix fragments for high throughput. See also: [Tensor Cores](#tensor-cores), [CUDA](#cuda).

---

## X

### XLA (Accelerated Linear Algebra)
Google's compiler for optimizing tensor computations. Not used directly by llama.cpp but relevant for understanding compilation-based optimization. See also: [Kernel](#kernel).

---

## Z

### Zero-Copy
Memory optimization technique where data is accessed directly without copying. Enabled by memory mapping in llama.cpp for efficient model loading. See also: [Memory Mapping](#memory-mapping-mmap).

---

## Advanced Topics (Cross-Module)

### API Gateway
A server that routes requests to backend services, handles authentication, rate limiting, and monitoring. Used in production LLM deployments. See also: [Load Balancing](#load-balancing).

### Asynchronous Inference
Non-blocking inference where requests are queued and processed asynchronously, improving concurrency. Critical for high-throughput servers. See also: [Continuous Batching](#continuous-batching).

### Auto-Scaling
Automatic adjustment of compute resources based on load. Kubernetes HPA (Horizontal Pod Autoscaler) commonly used for LLM serving. See also: [Kubernetes](#kubernetes).

### Bits Per Weight (BPW)
Metric measuring average storage per weight after quantization. Q4_K_M typically achieves ~4.5 bpw. See also: [Quantization](#quantization), [K-Quants](#k-quants).

### Cache Locality
Property where data accessed together in time is stored close in memory, improving performance. Critical for matrix operations in LLMs. See also: [Memory](#memory).

### Causal Attention
Attention mechanism where each token can only attend to previous tokens, not future ones. Essential for autoregressive generation. See also: [Attention Mechanism](#attention-mechanism).

### Chat Template
Format string defining how to structure conversation history into a prompt. Model-specific (e.g., ChatML, Llama-2 format). See also: [Prompt](#prompt), [Special Tokens](#special-tokens).

### Checkpointing
Saving model state during training or for deployment. llama.cpp loads from checkpoints converted to GGUF. See also: [Model Conversion](#model-conversion).

### Classifier-Free Guidance (CFG)
Technique for controlling generation by mixing conditional and unconditional predictions. Used in image generation, applicable to text. See also: [Sampling](#sampling).

### CLIP (Contrastive Language-Image Pre-training)
Vision-language model that learns joint embeddings of images and text. Used as vision encoder in multimodal models like LLaVA. See also: [LLaVA](#llava), [Embedding](#embedding).

### Context Length
See [Context Window](#context-window).

### Continual Learning
Training models to learn new tasks without forgetting old ones. Not directly relevant to llama.cpp (inference-only) but important concept. See also: [Fine-tuning](#fine-tuning).

### CoreML
Apple's machine learning framework for iOS/macOS. llama.cpp can export to CoreML for Apple Neural Engine acceleration. See also: [Metal](#metal), [Apple Neural Engine](#apple-neural-engine).

### CORS (Cross-Origin Resource Sharing)
Web security mechanism for controlling access to APIs from different domains. Important for web-deployed LLM APIs. See also: [API](#api).

### Cosine Similarity
Metric measuring similarity between vectors, commonly used in retrieval and embedding search. See also: [Embedding](#embedding), [Vector Database](#vector-database).

### CUDA Streams
Mechanism for concurrent kernel execution on NVIDIA GPUs. Used to overlap computation and memory transfers. See also: [CUDA](#cuda), [Kernel](#kernel).

### Data Parallelism
Distributing different data samples across multiple devices, processing in parallel. Contrasts with tensor/pipeline parallelism. See also: [Multi-GPU](#multi-gpu).

### Dead Neuron
Neuron that always outputs zero or never activates, contributing nothing to the model. Can result from poor initialization or ReLU. See also: [Activation](#activation).

### Dequantization
Converting quantized (low-precision) values back to floating-point for computation. llama.cpp dequantizes on-the-fly during inference. See also: [Quantization](#quantization).

### Distributed Tracing
Tracking requests across multiple services to debug latency and failures. OpenTelemetry commonly used. See also: [Observability](#observability).

### Docker
Containerization platform for packaging applications with dependencies. Commonly used to deploy llama.cpp servers. See also: [Kubernetes](#kubernetes).

### Dropout
Regularization technique that randomly drops neurons during training. Not used during inference. See also: [Inference](#inference).

### ELK Stack (Elasticsearch, Logstash, Kibana)
Suite of tools for logging, search, and visualization. Used for centralized logging in production. See also: [Observability](#observability).

### End-to-End Latency
Total time from request submission to final response. Includes queuing, inference, and network time. See also: [Latency](#latency).

### FastAPI
Modern Python web framework for building APIs. Commonly used for llama.cpp serving due to async support and auto-docs. See also: [API](#api).

### Few-Shot Learning
Providing few examples in the prompt to guide model behavior. Effective for task specification. See also: [Prompt](#prompt), [In-Context Learning](#in-context-learning).

### Fine-tuning
Training a pre-trained model on specific data to adapt it for a task. llama.cpp is for inference; fine-tuning done in PyTorch/etc. See also: [Training](#training).

### Flash Attention
Memory-efficient attention algorithm that uses tiling to reduce memory access. Achieves significant speedup for long contexts. See also: [Attention Mechanism](#attention-mechanism).

### Floating-Point Operations (FLOPs)
Measure of computational work. LLM inference requires billions of FLOPs per token. See also: [Throughput](#throughput).

### Function Calling
Feature where LLM can output structured function invocations (like OpenAI function calling). Requires parsing and validation. See also: [Constrained Generation](#constrained-generation).

### GEMM (General Matrix Multiply)
Fundamental linear algebra operation (C = A × B) that dominates LLM inference computation. Heavily optimized in llama.cpp. See also: [Matrix Multiplication](#matrix-multiplication).

### Georgi Gerganov
Creator of llama.cpp and GGML. Also created whisper.cpp and other C/C++ ML inference tools. See also: [llama.cpp](#llamacpp), [GGML](#ggml).

### Graceful Degradation
System design principle where performance degrades gradually under load rather than failing catastrophically. See also: [Reliability](#reliability).

### Gradient
Derivative of loss with respect to weights, used in training to update parameters. Not computed during inference. See also: [Training](#training).

### Grafana
Visualization and dashboarding platform, commonly used with Prometheus for monitoring. See also: [Prometheus](#prometheus), [Observability](#observability).

### Hallucination
When an LLM generates plausible-sounding but false information. Major challenge in production deployments. See also: [Quality](#quality).

### Head (Attention)
One of multiple parallel attention computations in Multi-Head Attention. Each head learns different token relationships. See also: [Multi-Head Attention](#multi-head-attention-mha).

### Health Check
Endpoint or mechanism for monitoring service availability. Kubernetes uses liveness/readiness probes. See also: [Kubernetes](#kubernetes), [Monitoring](#monitoring).

### HuggingFace Transformers
Popular library for transformer models. Models often converted from HuggingFace to GGUF for llama.cpp. See also: [Model Conversion](#model-conversion).

### Hyperparameter
Configuration value (not learned during training) like learning rate, temperature, top-k. llama.cpp uses hyperparameters for sampling. See also: [Sampling](#sampling).

### Idempotency
Property where repeating an operation produces the same result. Important for API design. See also: [API](#api).

### Importance Quantization (IQ)
Advanced quantization method in GGUF (IQ2, IQ3, IQ4) that preserves important weights at higher precision. See also: [K-Quants](#k-quants), [Quantization](#quantization).

### In-Context Learning
LLM's ability to learn from examples provided in the prompt without weight updates. See also: [Few-Shot Learning](#few-shot-learning), [Prompt](#prompt).

### Inference Optimization
Techniques to accelerate inference: quantization, kernel fusion, batching, caching. Core focus of llama.cpp. See also: [Optimization](#optimization).

### Ingress
Kubernetes resource for managing external access to services. Provides load balancing, SSL termination. See also: [Kubernetes](#kubernetes).

### INT4 / INT8
Integer data types with 4 or 8 bits. Used for quantized weights and activations. See also: [Quantization](#quantization).

### Jaeger
Distributed tracing system for monitoring microservices. Part of observability stack. See also: [Distributed Tracing](#distributed-tracing).

### JIT (Just-In-Time) Compilation
Compiling code at runtime for optimization. PyTorch uses JIT; llama.cpp uses ahead-of-time compilation. See also: [Optimization](#optimization).

### JSON Mode
Feature to constrain LLM output to valid JSON. Can use grammars or post-processing. See also: [GBNF](#gbnf-ggml-bnf), [Constrained Generation](#constrained-generation).

### Kernel Fusion
Combining multiple GPU operations into one kernel to reduce memory transfers. Key optimization in llama.cpp. See also: [Kernel](#kernel), [CUDA](#cuda).

### Key-Value (KV) Heads
In GQA/MQA, the number of K/V projections (fewer than query heads). Reduces KV cache size. See also: [Grouped-Query Attention](#grouped-query-attention-gqa), [KV Cache](#kv-cache).

### Kubernetes
Container orchestration platform for deploying and scaling applications. Common for production LLM serving. See also: [Docker](#docker).

### LangChain
Framework for building LLM applications with chains, agents, and tools. Can integrate with llama.cpp. See also: [Integration](#integration), [RAG](#rag-retrieval-augmented-generation).

### Layer Offloading
Running some model layers on GPU and others on CPU when full model doesn't fit in VRAM. llama.cpp supports via `-ngl` parameter. See also: [Offloading](#offloading).

### LlamaIndex
Framework for connecting LLMs with external data, similar to LangChain. Supports llama.cpp backends. See also: [RAG](#rag-retrieval-augmented-generation).

### LLaVA (Large Language and Vision Assistant)
Multimodal model combining vision encoder (CLIP) with language model. Supported by llama.cpp. See also: [CLIP](#clip-contrastive-language-image-pre-training), [Multimodal](#multimodal).

### Load Balancer
System distributing traffic across multiple servers. Nginx, HAProxy commonly used for LLM APIs. See also: [API Gateway](#api-gateway).

### Local Deployment
Running models on-premises or on-device rather than cloud. Key use case for llama.cpp. See also: [Edge Deployment](#edge-deployment).

### Locally Typical Sampling
Sampling strategy based on information theory, selecting tokens with typical information content. See also: [Sampling](#sampling).

### Lora (Low-Rank Adaptation)
Parameter-efficient fine-tuning method. llama.cpp supports LoRA adapters for inference. See also: [Fine-tuning](#fine-tuning).

### Matrix Multiplication
Core operation in neural networks (y = x × W). Dominates LLM inference computation. See also: [GEMM](#gemm-general-matrix-multiply).

### Memory Bandwidth
Rate of data transfer between memory and processor. Often the bottleneck in LLM inference. See also: [Bandwidth](#bandwidth).

### Microservices
Architectural pattern with independent, loosely coupled services. LLM serving often uses microservices architecture. See also: [Architecture](#architecture).

### Mixtral
Mixture-of-Experts language model. llama.cpp supports MoE architectures. See also: [MoE](#moe-mixture-of-experts).

### Model Architecture
Structure of neural network: layer types, dimensions, connectivity. llama.cpp supports many architectures. See also: [Transformer](#transformer).

### Model Registry
System for storing and versioning ML models. Production deployments use registries for model management. See also: [MLOps](#mlops).

### MoE (Mixture of Experts)
Architecture with multiple "expert" sub-networks, routing inputs to subset of experts. More parameters, same compute. See also: [Mixtral](#mixtral).

### Multi-GPU
Using multiple GPUs for inference via tensor, pipeline, or data parallelism. See also: [Tensor Parallelism](#tensor-parallelism).

### Multimodal
Models processing multiple input types (text, image, audio). LLaVA is multimodal. See also: [LLaVA](#llava-large-language-and-vision-assistant).

### NCCL (NVIDIA Collective Communications Library)
Library for multi-GPU communication. Optimized for NVIDIA GPUs with NVLink. See also: [Multi-GPU](#multi-gpu).

### Ngrok
Tool for exposing local servers to the internet. Useful for demos and testing. See also: [API](#api).

### Nsight Compute
NVIDIA profiling tool for CUDA kernels. Essential for GPU optimization. See also: [Profiling](#profiling).

### Nucleus Sampling
See [Top-P Sampling](#top-p-sampling-nucleus-sampling).

### NVLink
High-bandwidth interconnect between NVIDIA GPUs, much faster than PCIe. Critical for multi-GPU. See also: [Multi-GPU](#multi-gpu).

### Observability
Practice of understanding system internals via metrics, logs, and traces. Critical for production. See also: [Monitoring](#monitoring).

### ONNX (Open Neural Network Exchange)
Format for representing ML models. llama.cpp uses GGUF; ONNX less common for LLMs. See also: [Model Conversion](#model-conversion).

### Optimization
Improving performance (speed, memory, cost) without changing functionality. Core focus of llama.cpp. See also: [Inference Optimization](#inference-optimization).

### Outlier
Value significantly different from the norm. Activation outliers challenge quantization. See also: [Quantization](#quantization).

### PagedAttention
Attention implementation using paged KV cache for better memory management. Used in vLLM. See also: [KV Cache](#kv-cache).

### Pipeline Parallelism
Distributing model layers across devices, processing in pipeline stages. See also: [Multi-GPU](#multi-gpu).

### Pinecone
Cloud vector database for similarity search. Common in RAG systems. See also: [Vector Database](#vector-database).

### Pre-training
Initial training of an LLM on large corpus. llama.cpp uses pre-trained models for inference. See also: [Training](#training).

### Prefix Caching
Caching KV cache for common prompt prefixes to avoid recomputation. Improves multi-turn conversation efficiency. See also: [KV Cache](#kv-cache).

### Profiling
Measuring program performance to identify bottlenecks. Essential for optimization. See also: [Nsight Compute](#nsight-compute).

### Prometheus
Open-source monitoring system for collecting and querying metrics. Standard for Kubernetes. See also: [Monitoring](#monitoring), [Grafana](#grafana).

### Prompt Engineering
Crafting effective prompts to get desired LLM behavior. Important skill for LLM applications. See also: [Prompt](#prompt).

### PyTorch
Popular deep learning framework. Models often trained in PyTorch then converted to GGUF. See also: [Model Conversion](#model-conversion).

### QLoRA
Quantized LoRA for parameter-efficient fine-tuning with reduced memory. See also: [LoRA](#lora-low-rank-adaptation).

### Quality Assurance (QA)
Testing and validation processes ensuring software quality. Important for production LLM deployments. See also: [Testing](#testing).

### RAG (Retrieval-Augmented Generation)
Technique combining retrieval from knowledge base with LLM generation. Common pattern for Q&A systems. See also: [Vector Database](#vector-database).

### Rate Limiting
Controlling request rate per user/IP to prevent abuse and ensure fairness. Essential for APIs. See also: [API](#api).

### Rectified Linear Unit (ReLU)
Activation function f(x) = max(0, x). Simple but can cause dead neurons. LLMs use alternatives like SwiGLU. See also: [Activation](#activation).

### Redis
In-memory data store, often used for caching and rate limiting in LLM systems. See also: [Caching](#caching).

### Reliability
System's ability to function correctly over time. Measured by uptime, error rate. See also: [SLA](#sla-service-level-agreement).

### Request Queue
Buffer holding pending requests before processing. Important for load management. See also: [Continuous Batching](#continuous-batching).

### Response Streaming
Sending output tokens as generated rather than waiting for completion. Improves perceived latency. See also: [Server-Sent Events](#server-sent-events-sse).

### Retrieval
Finding relevant documents/information from a corpus, typically using vector similarity. Core of RAG. See also: [RAG](#rag-retrieval-augmented-generation).

### RMSNorm (Root Mean Square Normalization)
Simpler alternative to LayerNorm used in LLaMA models. Slightly faster, similar performance. See also: [Layer Normalization](#layer-normalization).

### ROCm (Radeon Open Compute)
AMD's GPU computing platform, similar to CUDA. llama.cpp supports ROCm backend. See also: [Backend](#backend).

### Safetensors
Safe, fast serialization format for tensors. Alternative to PyTorch pickles. See also: [Model Conversion](#model-conversion).

### Scaling Law
Empirical relationships between model size, data, compute, and performance. Guide model selection. See also: [Model Architecture](#model-architecture).

### Server-Sent Events (SSE)
Web standard for server-to-client streaming. Used for streaming LLM responses. See also: [Response Streaming](#response-streaming).

### Sigmoid
Activation function f(x) = 1/(1 + e^(-x)), outputs 0 to 1. Used in gates (e.g., SwiGLU). See also: [Activation](#activation).

### SigLIP
Google's improved CLIP variant. Used in some multimodal models. See also: [CLIP](#clip-contrastive-language-image-pre-training).

### SIMD (Single Instruction Multiple Data)
Parallel processing instruction set (AVX2, AVX-512, NEON). llama.cpp uses SIMD for CPU inference. See also: [Vectorization](#vectorization).

### SLA (Service Level Agreement)
Contract defining expected service quality (uptime, latency). Critical for production. See also: [SLO](#slo-service-level-objective).

### SLO (Service Level Objective)
Specific measurable target (e.g., 99.9% uptime, p99 < 1s). See also: [SLA](#sla-service-level-agreement).

### Sparse Attention
Attention mechanism attending to subset of tokens, reducing O(n²) complexity. Research topic. See also: [Attention Mechanism](#attention-mechanism).

### Stop Sequence
String that when generated, terminates generation early. Useful for formatting control. See also: [Sampling](#sampling).

### Swish
Activation function f(x) = x * sigmoid(x). Used in SwiGLU. See also: [SwiGLU](#swiglu).

### SwiGLU
Gated activation function used in LLaMA FFN: SwiGLU(x, W, V) = Swish(xW) ⊙ (xV). Better than ReLU. See also: [Feed-Forward Network](#feed-forward-network-ffn).

### SYCL
Cross-platform abstraction for heterogeneous computing. llama.cpp supports SYCL for Intel GPUs. See also: [Backend](#backend).

### System Prompt
Initial instruction given to chat models to set behavior/role. Part of prompt engineering. See also: [Prompt](#prompt), [Chat Template](#chat-template).

### Tensor Cores
Specialized hardware in NVIDIA GPUs for fast matrix multiplication. Accelerate INT8/FP16 inference. See also: [CUDA](#cuda).

### Tensor Parallelism
Splitting individual layers across GPUs, computing parts in parallel. See also: [Multi-GPU](#multi-gpu).

### TensorRT
NVIDIA's inference optimization library. Alternative to llama.cpp, focused on NVIDIA GPUs. See also: [Optimization](#optimization).

### Testing
Validating software correctness through unit, integration, and end-to-end tests. See also: [Quality Assurance](#quality-assurance-qa).

### Tiling
Breaking computation into blocks (tiles) that fit in fast memory. Critical for cache efficiency. See also: [Cache Locality](#cache-locality).

### TLS/SSL
Encryption protocols for secure communication. HTTPS uses TLS. See also: [Security](#security).

### Token Bucket
Rate limiting algorithm allowing bursts up to capacity, refilling at constant rate. See also: [Rate Limiting](#rate-limiting).

### Token Healing
Re-tokenizing at boundaries to fix tokenization artifacts. Improves output quality. See also: [Tokenization](#tokenization).

### Training
Process of learning model weights from data. llama.cpp is for inference; training done elsewhere. See also: [Fine-tuning](#fine-tuning).

### Truncation
Cutting context or sequence to fit limits. Important for managing context window. See also: [Context Window](#context-window).

### Unified Memory
Memory architecture where CPU and GPU share address space. Simplifies programming on Apple Silicon. See also: [Metal](#metal).

### Uptime
Percentage of time service is available. 99.9% uptime = ~8.7 hours downtime/year. See also: [SLO](#slo-service-level-objective).

### Vector Database
Database optimized for storing and searching high-dimensional vectors. Used in RAG. Examples: Pinecone, Chroma, Weaviate. See also: [Embedding](#embedding), [RAG](#rag-retrieval-augmented-generation).

### Vectorization
Using SIMD instructions to process multiple data points in parallel. See also: [SIMD](#simd-single-instruction-multiple-data).

### vLLM
High-throughput LLM serving system using PagedAttention. Alternative to llama.cpp, focused on throughput. See also: [Continuous Batching](#continuous-batching).

### VRAM (Video RAM)
GPU memory. Model must fit in VRAM (or use offloading). A100 has 40/80GB. See also: [GPU](#gpu-graphics-processing-unit).

### Warmup
Initial period where system performance is lower (loading models, filling caches). Important for auto-scaling. See also: [Auto-Scaling](#auto-scaling).

### Weight Decay
Regularization technique adding penalty for large weights. Training concept, not relevant for inference. See also: [Training](#training).

### Weight Sharing
Multiple parts of model sharing same weights. Reduces parameters. See also: [Parameter](#parameter).

### Zero-Shot
Model performing task without task-specific examples, using only instruction. Contrasts with few-shot. See also: [Few-Shot Learning](#few-shot-learning).

---

## Acronyms Reference

- **API**: Application Programming Interface
- **AWS**: Amazon Web Services
- **BNF**: Backus-Naur Form
- **BOS**: Beginning of Sequence
- **BPE**: Byte Pair Encoding
- **BPW**: Bits Per Weight
- **CFG**: Classifier-Free Guidance
- **CLIP**: Contrastive Language-Image Pre-training
- **CORS**: Cross-Origin Resource Sharing
- **CPU**: Central Processing Unit
- **CUDA**: Compute Unified Device Architecture
- **DPO**: Direct Preference Optimization
- **ELK**: Elasticsearch, Logstash, Kibana
- **EOS**: End of Sequence
- **FFN**: Feed-Forward Network
- **FLOP**: Floating-Point Operation
- **FP16/FP32**: 16/32-bit Floating Point
- **GBNF**: GGML Backus-Naur Form
- **GEMM**: General Matrix Multiply
- **GGML**: Georgi Gerganov Machine Learning
- **GGUF**: GGML Unified Format
- **GPU**: Graphics Processing Unit
- **GQA**: Grouped-Query Attention
- **HPA**: Horizontal Pod Autoscaler
- **HTTP**: Hypertext Transfer Protocol
- **INT4/INT8**: 4/8-bit Integer
- **IQ**: Importance Quantization
- **JIT**: Just-In-Time
- **JSON**: JavaScript Object Notation
- **KV**: Key-Value
- **LoRA**: Low-Rank Adaptation
- **LLM**: Large Language Model
- **MHA**: Multi-Head Attention
- **MLOps**: Machine Learning Operations
- **MoE**: Mixture of Experts
- **MQA**: Multi-Query Attention
- **NCCL**: NVIDIA Collective Communications Library
- **NER**: Named Entity Recognition
- **ONNX**: Open Neural Network Exchange
- **OOM**: Out Of Memory
- **PII**: Personally Identifiable Information
- **QA**: Quality Assurance
- **QLoRA**: Quantized LoRA
- **RAG**: Retrieval-Augmented Generation
- **ReLU**: Rectified Linear Unit
- **REST**: Representational State Transfer
- **RMSE**: Root Mean Square Error
- **RMSNorm**: Root Mean Square Normalization
- **ROCm**: Radeon Open Compute
- **RoPE**: Rotary Position Embedding
- **RPO**: Recovery Point Objective
- **RTO**: Recovery Time Objective
- **SIMD**: Single Instruction Multiple Data
- **SLA**: Service Level Agreement
- **SLO**: Service Level Objective
- **SSE**: Server-Sent Events
- **SSL/TLS**: Secure Sockets Layer / Transport Layer Security
- **SYCL**: C++ abstraction for heterogeneous computing
- **TTFT**: Time To First Token
- **VRAM**: Video RAM
- **WMMA**: Warp Matrix Multiply-Accumulate

---

## Module Coverage Summary

**Module 1 (Foundations)**: llama.cpp, GGUF, GGML, Backend, Inference, Token, Tokenization, Vocabulary, Context Window, Model Conversion, Memory Mapping

**Module 2 (Core Implementation)**: Transformer, Attention, Self-Attention, Multi-Head Attention, Grouped-Query Attention, Feed-Forward Network, Layer, RMSNorm, RoPE, SwiGLU, KV Cache, Forward Pass, Embedding, Causal Attention

**Module 3 (Quantization)**: Quantization, K-Quants, IQ Formats, Bits Per Weight, Perplexity, Precision, Float16, INT4, INT8, GPTQ, AWQ, Dequantization, Outlier

**Module 4 (GPU Acceleration)**: GPU, CUDA, Metal, ROCm, SYCL, Kernel, Tensor Cores, WMMA, NCCL, NVLink, Multi-GPU, Tensor Parallelism, Pipeline Parallelism, Data Parallelism, SIMD, Vectorization, Memory Bandwidth

**Module 5 (Advanced Inference)**: Speculative Decoding, Continuous Batching, Beam Search, Constrained Generation, GBNF, Grammar, Mirostat, Locally Typical Sampling, Prefix Caching, Flash Attention

**Module 6 (Server & Production)**: API, REST, FastAPI, Server-Sent Events, Load Balancer, Health Check, Prometheus, Grafana, Docker, Kubernetes, Ingress, Auto-Scaling, Rate Limiting, Response Streaming

**Module 7 (Multimodal)**: Multimodal, LLaVA, CLIP, SigLIP, Vision Encoder, Image Preprocessing, Cross-Modal Attention

**Module 8 (Integration)**: LangChain, LlamaIndex, RAG, Vector Database, Function Calling, LoRA, Embedding Search, Retrieval

**Module 9 (Production Engineering)**: SLA, SLO, Observability, Monitoring, Distributed Tracing, Reliability, Uptime, Security, TLS, CORS, Testing, Quality Assurance, Graceful Degradation

---

**Total Terms**: 200+
**Last Expanded**: 2025-11-18
**Maintained By**: Agent 8 (Integration Coordinator)

**Usage Notes**:
- Terms are cross-referenced with "See also" links
- Organized alphabetically for easy lookup
- Module mapping helps contextualize learning
- Acronym reference for quick translation
- Living document - continues to expand

Found a missing term? Suggest additions to improve this glossary!
