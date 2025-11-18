# Quantization Fundamentals

**Module 3, Lesson 1** | **Estimated Time**: 3 hours | **Difficulty**: Intermediate

## Table of Contents
1. [Introduction](#introduction)
2. [What is Quantization?](#what-is-quantization)
3. [Why Quantize LLMs?](#why-quantize-llms)
4. [Types of Quantization](#types-of-quantization)
5. [Quantization Mathematics](#quantization-mathematics)
6. [Post-Training Quantization (PTQ)](#post-training-quantization-ptq)
7. [Quantization-Aware Training (QAT)](#quantization-aware-training-qat)
8. [Quality vs Size Trade-offs](#quality-vs-size-trade-offs)
9. [Best Practices](#best-practices)
10. [Interview Questions](#interview-questions)

---

## Introduction

Quantization is one of the most important techniques for deploying Large Language Models (LLMs) in production. It enables running models that would otherwise require hundreds of gigabytes of memory on consumer-grade hardware, while maintaining acceptable inference quality.

**Learning Objectives:**
- Understand the theory and mathematics behind quantization
- Compare different quantization approaches
- Analyze the trade-offs between model size and accuracy
- Choose appropriate quantization for specific use cases

**Prerequisites:**
- Understanding of neural network fundamentals
- Basic linear algebra
- Familiarity with floating-point representation

---

## What is Quantization?

### Definition

**Quantization** is the process of mapping continuous or high-precision values to a smaller set of discrete values, typically using fewer bits to represent each value.

In the context of LLMs:
- **Original Models**: Use 32-bit floating point (FP32) or 16-bit floating point (FP16/BF16)
- **Quantized Models**: Use lower precision representations (8-bit, 4-bit, or even lower)

### Visual Example

```
FP32 (32 bits): 3.141592653589793
FP16 (16 bits): 3.140625
INT8 (8 bits):  3
INT4 (4 bits):  3 (with scaling)
```

### Memory Savings

| Precision | Bits per Parameter | 7B Model Size | 70B Model Size |
|-----------|-------------------|---------------|----------------|
| FP32      | 32                | 28 GB         | 280 GB         |
| FP16/BF16 | 16                | 14 GB         | 140 GB         |
| INT8      | 8                 | 7 GB          | 70 GB          |
| INT4      | 4                 | 3.5 GB        | 35 GB          |
| INT2      | 2                 | 1.75 GB       | 17.5 GB        |

---

## Why Quantize LLMs?

### 1. **Memory Reduction**

Modern LLMs have billions of parameters:
- **LLaMA-7B**: 7 billion parameters
- **LLaMA-13B**: 13 billion parameters
- **LLaMA-70B**: 70 billion parameters

Without quantization, these models require:
- LLaMA-7B FP16: ~14 GB RAM
- LLaMA-70B FP16: ~140 GB RAM

With 4-bit quantization:
- LLaMA-7B Q4: ~4 GB RAM ✅
- LLaMA-70B Q4: ~35 GB RAM ✅

### 2. **Faster Inference**

Lower precision means:
- **Faster memory transfers** (4x faster for INT8 vs FP32)
- **Better cache utilization** (more data fits in cache)
- **SIMD optimization opportunities** (process more values per instruction)

### 3. **Energy Efficiency**

- Lower precision operations consume less power
- Critical for mobile and edge deployment
- Reduced data center costs at scale

### 4. **Accessibility**

Enables running powerful models on:
- Consumer GPUs (RTX 3060, 4060)
- High-end CPUs with 32GB+ RAM
- Mobile devices
- Edge devices

---

## Types of Quantization

### 1. Symmetric Quantization

Maps values symmetrically around zero.

**Formula:**
```
quantized = round(value / scale)
dequantized = quantized * scale
```

**Characteristics:**
- Simple implementation
- Zero point is always 0
- Works well for weights with symmetric distribution
- Used in most LLaMA.cpp quantization schemes

**Example:**
```python
# Symmetric quantization to 8-bit
value = 0.745
scale = 0.01
quantized = round(0.745 / 0.01) = 75  # INT8 value
dequantized = 75 * 0.01 = 0.75       # Close to original
```

### 2. Asymmetric Quantization

Uses both scale and zero-point to handle asymmetric distributions.

**Formula:**
```
quantized = round(value / scale) + zero_point
dequantized = (quantized - zero_point) * scale
```

**Characteristics:**
- More flexible for asymmetric distributions
- Requires storing zero_point
- Better for activations
- Used in some mobile inference frameworks

**Example:**
```python
# Asymmetric quantization to 8-bit
value = 0.745
scale = 0.01
zero_point = -128
quantized = round(0.745 / 0.01) + (-128) = -53  # INT8 value
dequantized = (-53 - (-128)) * 0.01 = 0.75
```

### 3. Per-Tensor Quantization

Single scale factor for entire tensor.

**Advantages:**
- Simple and fast
- Minimal overhead
- Easy to implement

**Disadvantages:**
- Less accurate if tensor has wide value range
- Outliers can dominate scale factor

### 4. Per-Channel Quantization

Different scale factor for each channel (often per output channel in weights).

**Advantages:**
- More accurate than per-tensor
- Handles varying ranges across channels
- Still relatively efficient

**Disadvantages:**
- More scale factors to store
- Slightly more complex computation

### 5. Block-wise Quantization

Divides weights into blocks and quantizes each block independently.

**Advantages:**
- **Best accuracy for low-bit quantization**
- Handles local variations in weight distributions
- Used in GGUF k-quants

**Disadvantages:**
- More metadata (scale per block)
- More complex implementation

**Example (GGUF Q4_K_M):**
```
Block size: 256 values
Each block has: 1 scale (FP16), 128 quantized pairs (4-bit each)
```

---

## Quantization Mathematics

### Quantization Error

The difference between original and quantized values:

```
error = |original_value - dequantized_value|
```

### Quantization Noise

Quantization introduces noise with properties:
- **Mean**: Approximately 0 (for symmetric quantization)
- **Variance**: Proportional to `scale²`
- **Distribution**: Approximately uniform within [-scale/2, scale/2]

### Signal-to-Quantization-Noise Ratio (SQNR)

```
SQNR = 10 * log10(signal_power / quantization_noise_power)
```

For n-bit quantization:
```
Theoretical SQNR ≈ 6.02 * n + 1.76 dB
```

Examples:
- 8-bit: ~50 dB
- 4-bit: ~26 dB
- 2-bit: ~14 dB

### Calibration

Finding the optimal scale factor:

**Method 1: Min-Max**
```python
scale = (max_value - min_value) / (2^n - 1)
```

**Method 2: Mean Absolute Value**
```python
scale = mean(abs(values)) / (2^(n-1) - 1)
```

**Method 3: Mean Squared Error (MSE)**
```python
# Find scale that minimizes MSE between original and quantized
scale = argmin_s(MSE(original, quantize(original, s)))
```

---

## Post-Training Quantization (PTQ)

### Definition

Quantization applied to a **pre-trained** model without additional training.

### PTQ Approaches

#### 1. **Weight-Only Quantization**

- Quantize only the weights
- Keep activations in higher precision
- Used in most LLaMA.cpp quantization

**Advantages:**
- Simple to implement
- No calibration data needed
- Good quality retention

**Example: GGUF Quantization**
```python
# Pseudo-code for weight-only quantization
def quantize_weights(model):
    for layer in model.layers:
        for weight_tensor in layer.weights:
            scale = compute_scale(weight_tensor)
            quantized = round(weight_tensor / scale)
            store_quantized(quantized, scale)
```

#### 2. **Weight + Activation Quantization**

- Quantize both weights and activations
- Requires calibration dataset
- More complex but better performance

**Advantages:**
- Full integer arithmetic possible
- Maximum speed on specialized hardware

**Challenges:**
- Activations harder to quantize (more dynamic range)
- Requires representative calibration data

#### 3. **Dynamic Quantization**

- Weights quantized offline
- Activations quantized at runtime
- Compute scales dynamically

**Advantages:**
- Better quality than static activation quantization
- No calibration needed

**Disadvantages:**
- Overhead of computing scales at runtime

### Advanced PTQ Methods

#### GPTQ (Gradient-based Post-Training Quantization)

- Uses second-order information (Hessian)
- Minimizes quantization error layer by layer
- Extremely high quality at 4-bit

**Key Idea:**
```
For each layer:
  1. Compute importance of each weight (using Hessian)
  2. Quantize weights in order of importance
  3. Adjust remaining weights to compensate for error
```

**Results:**
- 4-bit GPTQ often matches 16-bit quality
- Used for creating high-quality quantized models

#### AWQ (Activation-aware Weight Quantization)

- Protects weights that are important for activations
- Uses activation statistics for quantization decisions
- Better than naive quantization at low bits

**Key Idea:**
```
For each weight:
  1. Measure impact on activations
  2. Use mixed precision: higher precision for important weights
  3. Lower precision for less important weights
```

---

## Quantization-Aware Training (QAT)

### Definition

Training (or fine-tuning) a model with quantization in mind.

### How it Works

1. **Forward Pass**: Simulate quantization
   ```
   quantized_weight = quantize(weight)
   output = matmul(input, quantized_weight)
   ```

2. **Backward Pass**: Use straight-through estimator
   ```
   gradient flows through as if no quantization
   ```

3. **Weight Update**: Update full-precision weights
   ```
   weight = weight - lr * gradient
   ```

### Advantages

- **Best quality** at very low bit widths
- Can recover from quantization errors
- Model learns robust representations

### Disadvantages

- Requires training/fine-tuning
- Computational cost
- Need training data and infrastructure

### When to Use

- Sub-4-bit quantization
- Mission-critical applications
- When PTQ quality is insufficient

---

## Quality vs Size Trade-offs

### Perplexity Impact

Typical perplexity increase for LLaMA models:

| Quantization | Size Reduction | Perplexity Increase | Use Case |
|--------------|----------------|---------------------|----------|
| FP16         | Baseline       | 0%                  | Training, reference |
| Q8_0         | 2x smaller     | <0.5%               | High quality, limited memory |
| Q6_K         | 2.67x smaller  | ~0.5-1%             | Excellent quality/size balance |
| Q5_K_M       | 3.2x smaller   | ~1-2%               | Very good quality, popular |
| Q4_K_M       | 4x smaller     | ~2-4%               | Good quality, recommended |
| Q4_0         | 4x smaller     | ~3-5%               | Acceptable quality |
| Q3_K_M       | 5.3x smaller   | ~5-10%              | Noticeable degradation |
| Q2_K         | 8x smaller     | ~10-20%             | Experimental, significant loss |

### Quality Metrics

**1. Perplexity**
- Measures how well model predicts text
- Lower is better
- Most common metric for quantization

**2. Benchmark Performance**
- MMLU (Massive Multitask Language Understanding)
- HellaSwag
- TruthfulQA
- Varies by task

**3. Subjective Quality**
- Coherence
- Factual accuracy
- Instruction following
- Often most important in practice

### Selection Guidelines

**For 7B-13B models:**
- **Q8_0**: Maximum quality, if memory allows
- **Q6_K**: Excellent balance
- **Q5_K_M**: Recommended default
- **Q4_K_M**: If memory constrained

**For 30B-70B models:**
- **Q6_K**: If possible
- **Q5_K_M**: Excellent balance
- **Q4_K_M**: Most practical for consumer hardware
- **Q3_K_M**: If desperate for memory

**For 100B+ models:**
- **Q4_K_M**: Recommended
- **Q3_K_M**: If needed
- Consider model sharding

---

## Best Practices

### 1. Start Conservative

Begin with higher precision and move down:
```
Q8_0 → Q6_K → Q5_K_M → Q4_K_M
```

Test quality at each step.

### 2. Use K-Quants for Modern Models

For most models, prefer k-quant variants:
- Better quality than legacy formats (Q4_0, Q5_0)
- Optimized block-wise quantization

```
Prefer: Q4_K_M, Q5_K_M, Q6_K
Legacy: Q4_0, Q5_0, Q5_1
```

### 3. Benchmark Your Use Case

Quality is task-dependent:
- Chat may tolerate more quantization
- Code generation is more sensitive
- Reasoning tasks vary

**Always test on your specific workload.**

### 4. Consider Mixed Precision

Some layers are more sensitive:
- Attention layers often critical
- Feed-forward layers more robust
- Embedding layers depend on vocabulary size

### 5. Monitor Memory Bandwidth

Even with small models:
- Memory bandwidth often bottleneck
- Lower precision helps even if model fits in memory
- Measure tokens/second, not just perplexity

### 6. Document Your Choices

For production:
```yaml
model: llama-2-7b
quantization: Q5_K_M
reasoning: |
  - Fits in 6GB VRAM
  - <1% perplexity increase
  - 25% faster than Q8_0
  - Tested on customer support dataset
  - MMLU: 62.3 vs 63.1 (FP16)
```

---

## Practical Example: Quantizing LLaMA-7B

### Scenario

You need to run LLaMA-7B on a machine with 16GB RAM.

### Analysis

**Memory Requirements:**
- FP16: ~14 GB (model) + 2 GB (context/OS) = 16 GB ❌ Too tight
- Q8_0: ~7 GB (model) + 2 GB (context/OS) = 9 GB ✅
- Q5_K_M: ~4.5 GB (model) + 2 GB (context/OS) = 6.5 GB ✅✅
- Q4_K_M: ~3.8 GB (model) + 2 GB (context/OS) = 5.8 GB ✅✅✅

### Decision Process

1. **Q8_0**: Safe choice, excellent quality
2. **Test Q5_K_M**:
   - Run perplexity test
   - Test on representative prompts
   - If acceptable → use Q5_K_M (more headroom)
3. **Consider Q4_K_M**:
   - If need even more context
   - Or running multiple models
   - Quality acceptable for most use cases

### Recommendation

**Use Q5_K_M** unless:
- Quality degradation unacceptable → Q8_0
- Need maximum context → Q4_K_M
- Running multiple models → Q4_K_M

---

## Interview Questions

### Conceptual Questions

1. **Q: What is quantization and why is it important for LLM deployment?**

   A: Quantization maps high-precision values to lower precision, reducing memory and compute requirements. For LLMs, it's critical because it enables running billion-parameter models on consumer hardware (e.g., 70B model from 140GB to 35GB with 4-bit quantization), while maintaining acceptable quality.

2. **Q: Explain the difference between symmetric and asymmetric quantization.**

   A: Symmetric quantization uses only a scale factor and assumes values are centered around zero (quantized = round(value/scale)). Asymmetric uses both scale and zero-point to handle arbitrary ranges (quantized = round(value/scale) + zero_point). Symmetric is simpler and faster; asymmetric is more accurate for skewed distributions.

3. **Q: What is the trade-off between per-tensor and per-channel quantization?**

   A: Per-tensor uses one scale for the entire tensor (simple, fast, less accurate). Per-channel uses different scales per channel (more accurate, especially for layers with varying channel magnitudes, slightly more overhead). Per-channel is usually worth it for weights.

4. **Q: How does block-wise quantization work and why is it used in GGUF?**

   A: Block-wise quantization divides tensors into blocks (e.g., 256 values) and quantizes each block independently with its own scale. This handles local variations in weight distributions, crucial for maintaining quality at 4-bit and below. GGUF k-quants use this for superior quality vs legacy formats.

5. **Q: What is GPTQ and how does it differ from naive quantization?**

   A: GPTQ is a layer-wise quantization method using second-order information (Hessian) to minimize reconstruction error. It quantizes weights in order of importance and adjusts remaining weights to compensate for errors. This achieves near-FP16 quality at 4-bit, much better than naive rounding.

### Practical Questions

6. **Q: You need to run a 70B model on a system with 64GB RAM. What quantization would you choose?**

   A: Q4_K_M (≈35GB model + context) or Q5_K_M (≈45GB model + context). I'd start with Q5_K_M, test perplexity and task performance, and drop to Q4_K_M only if memory pressure is an issue. Both should fit, but Q5_K_M provides better quality.

7. **Q: How would you measure the quality impact of quantization?**

   A: Multiple approaches:
   - Perplexity on validation set (quantitative)
   - Benchmark tasks (MMLU, HellaSwag, etc.)
   - A/B testing on representative prompts
   - Task-specific metrics (accuracy, ROUGE for summarization, etc.)
   - Most importantly: test on actual use case

8. **Q: When would you use QAT instead of PTQ?**

   A: QAT when:
   - Targeting very low bit-widths (<4-bit)
   - PTQ quality insufficient for use case
   - Have training infrastructure and data
   - Can afford training time
   PTQ is usually sufficient for 4-bit and above.

### Advanced Questions

9. **Q: Explain how quantization affects inference speed. Why doesn't 4-bit quantization give 8x speedup vs FP32?**

   A: Factors limiting speedup:
   - Memory bandwidth (often the bottleneck, not compute)
   - Dequantization overhead (converting back to higher precision for compute)
   - Limited INT4 hardware support (often processed as INT8)
   - Non-quantized operations (layernorm, softmax)
   - Amdahl's law (non-quantizable parts limit overall speedup)

   Typical 4-bit speedup is 2-3x vs FP16, not 4x or 8x.

10. **Q: Design a quantization strategy for a multi-tenant LLM serving system with diverse quality requirements.**

    A: Strategy:
    - Tier system with multiple quantization levels:
      - Premium: Q8_0 or FP16 (highest quality)
      - Standard: Q5_K_M (balanced)
      - Economy: Q4_K_M (cost-effective)
    - Model routing based on SLA
    - Monitoring per tier (perplexity drift, latency, user satisfaction)
    - Ability to upgrade/downgrade models based on load
    - A/B test quantization changes before deployment
    - Document quality differences for customer transparency

---

## Summary

**Key Takeaways:**

1. Quantization reduces model size by using fewer bits per parameter (8-bit, 4-bit, 2-bit)
2. Trade-off: smaller models vs. some quality loss
3. Modern techniques (GPTQ, AWQ, k-quants) minimize quality impact
4. Block-wise quantization crucial for sub-8-bit quality
5. Always test on your specific use case and metrics

**Next Steps:**
- Lesson 2: Deep dive into GGUF quantization formats
- Lesson 3: Performance optimization techniques
- Lab 1: Hands-on quantization experiments

---

**Further Reading:**

- [GPTQ Paper](https://arxiv.org/abs/2210.17323): Accurate Post-Training Quantization for GPT
- [AWQ Paper](https://arxiv.org/abs/2306.00978): Activation-aware Weight Quantization
- [LLM.int8() Paper](https://arxiv.org/abs/2208.07339): 8-bit Matrix Multiplication
- [GGUF Format Documentation](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

**Author**: Agent 5 (Documentation Specialist)
**Module**: 3 - Quantization & Optimization
**Last Updated**: 2025-11-18
