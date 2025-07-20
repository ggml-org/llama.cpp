# Diffusion Text Generation Examples

This directory contains implementations for diffusion-based text generation using two different model architectures: **Dream** and **LLaDA-8B**. Both models use iterative denoising processes to generate text, but employ different sampling strategies and algorithms.

## Supported Models

### 1. Dream Model (`llama-diffusion-dream-cli`)

- https://huggingface.co/Dream-org/Dream-v0-Base-7B
- Original PR - https://github.com/ggml-org/llama.cpp/pull/14644

The Dream model supports four different sampling algorithms controlled by the `--diffusion-algorithm` parameter:

1. **ORIGIN (0)** - Original diffusion algorithm
   - Uses probability transfer based on timestep ratios
   - Default algorithm with standard confidence-based token selection

2. **MASKGIT_PLUS (1)** - Enhanced MaskGIT sampling
   - Improved version of the MaskGIT algorithm

3. **TOPK_MARGIN (2)** - Top-K margin-based sampling
   - Confidence calculated as the margin between top-1 and top-2 probabilities

4. **ENTROPY (3)** - Entropy-based sampling (recommended)
   - Uses entropy calculation for confidence estimation

### 2. LLaDA-8B Model (`llama-diffusion-llada-cli`)

- https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct

### LLaDA Model Remasking Strategies

The LLaDA model uses two remasking approaches controlled by the `--diffusion-algorithm` parameter:

1. **REMASKING_LOW_CONFIDENCE (0)** - Default strategy
   - Remasks tokens with lowest confidence scores
   - Uses softmax probabilities to determine confidence

2. **REMASKING_RANDOM (1)** - Random remasking
