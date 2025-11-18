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
