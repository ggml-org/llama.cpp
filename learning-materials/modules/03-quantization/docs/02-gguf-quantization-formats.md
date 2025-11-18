# GGUF Quantization Formats Guide

**Module 3, Lesson 2** | **Estimated Time**: 4 hours | **Difficulty**: Intermediate to Advanced

## Table of Contents
1. [Introduction](#introduction)
2. [GGUF Quantization Overview](#gguf-quantization-overview)
3. [Legacy Formats (Q4_0, Q5_0, Q8_0)](#legacy-formats)
4. [K-Quants (Modern Formats)](#k-quants-modern-formats)
5. [Importance Quantization (IQ)](#importance-quantization-iq)
6. [Format Comparison Matrix](#format-comparison-matrix)
7. [How to Choose the Right Format](#how-to-choose-the-right-format)
8. [Format Internals](#format-internals)
9. [Conversion Between Formats](#conversion-between-formats)
10. [Interview Questions](#interview-questions)

---

## Introduction

GGUF (GPT-Generated Unified Format) supports multiple quantization schemes, each with different trade-offs between model size, quality, and inference speed. Understanding these formats is crucial for deploying LLaMA models effectively.

**Learning Objectives:**
- Master all GGUF quantization formats (15+ variants)
- Understand the internal structure of each format
- Choose the optimal format for specific hardware and use cases
- Convert between formats effectively

**Prerequisites:**
- Quantization Fundamentals (Lesson 3.1)
- Understanding of GGUF file format
- Basic knowledge of memory alignment

---

## GGUF Quantization Overview

### Naming Convention

GGUF quantization formats follow a pattern:
```
Q<bits>_<variant>_<size>

Examples:
Q4_0      → 4-bit, variant 0 (legacy)
Q4_K_M    → 4-bit, K-quant, Medium
Q5_K_S    → 5-bit, K-quant, Small
```

### Format Categories

1. **Legacy Formats**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
   - Simple, straightforward
   - Block-wise with fixed block size
   - Good baseline performance

2. **K-Quants**: Q2_K, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, Q6_K
   - Modern, optimized formats
   - Better quality at same bit-width
   - More complex structure

3. **IQ (Importance-based)**: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, etc.
   - Experimental, ultra-low bit counts
   - Uses importance matrices
   - Cutting-edge research

### Bits per Weight

Actual average bits per weight (including metadata):

| Format  | Theoretical | Actual (with overhead) |
|---------|-------------|------------------------|
| Q2_K    | 2.0         | ~2.5 bpw               |
| Q3_K_S  | 3.0         | ~3.4 bpw               |
| Q3_K_M  | 3.0         | ~3.7 bpw               |
| Q4_0    | 4.0         | ~4.5 bpw               |
| Q4_K_S  | 4.0         | ~4.3 bpw               |
| Q4_K_M  | 4.0         | ~4.8 bpw               |
| Q5_0    | 5.0         | ~5.5 bpw               |
| Q5_K_S  | 5.0         | ~5.5 bpw               |
| Q5_K_M  | 5.0         | ~5.9 bpw               |
| Q6_K    | 6.0         | ~6.6 bpw               |
| Q8_0    | 8.0         | ~8.5 bpw               |

---

## Legacy Formats

### Q4_0: Basic 4-bit Quantization

**Structure:**
```c
// Block size: 32 values
struct block_q4_0 {
    ggml_fp16_t d;          // delta (scale) - 16 bits
    uint8_t qs[QK4_0/2];   // quantized values - 16 bytes (32 values, 4-bit each)
};
// Total: 18 bytes per 32 values = 4.5 bits per weight
```

**Characteristics:**
- Simplest quantization
- 32-value blocks
- Single FP16 scale per block
- 4-bit values symmetric around 0 (range: -8 to 7)

**Formula:**
```
dequantized_value = (quantized - 8) * scale
```

**Pros:**
- Fast to decode
- Minimal overhead
- Good for education/learning

**Cons:**
- Lower quality than k-quants
- Superseded by Q4_K_M

**When to Use:**
- Legacy compatibility
- Baseline comparisons
- Generally prefer Q4_K_M instead

---

### Q4_1: 4-bit with Zero-Point

**Structure:**
```c
// Block size: 32 values
struct block_q4_1 {
    ggml_fp16_t d;          // delta (scale)
    ggml_fp16_t m;          // min value (zero point)
    uint8_t qs[QK4_1/2];   // quantized values
};
// Total: 20 bytes per 32 values = 5 bits per weight
```

**Difference from Q4_0:**
- Adds minimum value (zero-point)
- Handles asymmetric distributions better
- Slightly larger (5 bpw vs 4.5 bpw)

**Formula:**
```
dequantized_value = quantized * scale + min_value
```

**When to Use:**
- Rarely - Q5_K_S usually better at similar size

---

### Q5_0: Basic 5-bit Quantization

**Structure:**
```c
// Block size: 32 values
struct block_q5_0 {
    ggml_fp16_t d;          // delta (scale)
    uint8_t qh[4];          // high bit for each value
    uint8_t qs[QK5_0/2];   // low 4 bits
};
// Total: 22 bytes per 32 values = 5.5 bits per weight
```

**Characteristics:**
- 5 bits per value (4 low + 1 high)
- Better quality than Q4_0
- Range: -16 to 15

**Pros:**
- Noticeable quality improvement over Q4
- Still relatively compact

**Cons:**
- Q5_K_M is better at similar size

**When to Use:**
- Legacy compatibility
- Prefer Q5_K_M for new quantizations

---

### Q5_1: 5-bit with Zero-Point

**Structure:**
```c
struct block_q5_1 {
    ggml_fp16_t d;          // delta (scale)
    ggml_fp16_t m;          // min value
    uint8_t qh[4];          // high bits
    uint8_t qs[QK5_1/2];   // low 4 bits
};
// Total: 24 bytes per 32 values = 6 bits per weight
```

**When to Use:**
- Rarely used
- Q6_K usually better at this size

---

### Q8_0: High-Quality 8-bit

**Structure:**
```c
// Block size: 32 values
struct block_q8_0 {
    ggml_fp16_t d;          // delta (scale)
    int8_t qs[QK8_0];      // quantized values (32 bytes)
};
// Total: 34 bytes per 32 values = 8.5 bits per weight
```

**Characteristics:**
- Nearly lossless quantization
- 8-bit signed integers (-128 to 127)
- Excellent quality retention

**Pros:**
- **Minimal quality loss** (<0.5% perplexity increase)
- Fast decoding
- Recommended when memory allows

**Cons:**
- Larger than other quantizations
- 2x size of Q4

**When to Use:**
- Maximum quality priority
- Sufficient memory available
- Baseline for quality comparisons
- When <1% perplexity increase required

**Recommended for:**
- High-end systems with ample RAM
- Quality-critical applications
- 7B-13B models where memory isn't constrained

---

## K-Quants (Modern Formats)

K-quants use a more sophisticated block structure with mixed precision within blocks, achieving better quality at the same average bit-width.

### K-Quant Philosophy

**Key Innovations:**
1. **Larger blocks** (256 values instead of 32)
2. **Super-blocks** containing multiple sub-blocks
3. **Mixed precision** within blocks
4. **Importance-based bit allocation**

### Q2_K: Ultra-Compressed

**Structure:**
```
Super-block: 256 values
- 16 sub-blocks of 16 values each
- 16 scales (4-bit each)
- 1 super-scale (FP16)
- Quantized values (2-bit)
```

**Characteristics:**
- ~2.5 bits per weight
- Significant quality loss
- Experimental

**Use Cases:**
- Extreme memory constraints
- Testing lower bounds
- Research purposes

**Quality Impact:**
- 10-20% perplexity increase
- Noticeable degradation in outputs
- Not recommended for production

---

### Q3_K: Three Variants

K-quants at 3-bit come in three sizes: Small, Medium, Large

#### Q3_K_S (Small)

**Characteristics:**
- ~3.4 bits per weight
- Minimal overhead
- Best size/quality at 3-bit

**Use Cases:**
- Large models (70B+) on limited hardware
- When 4-bit doesn't fit

**Quality:**
- 5-10% perplexity increase
- Acceptable for some applications
- Test thoroughly on use case

#### Q3_K_M (Medium)

**Characteristics:**
- ~3.7 bits per weight
- Better quality than Q3_K_S
- More metadata

**Use Cases:**
- When Q3_K_S quality insufficient
- Extra bits worth the improvement

#### Q3_K_L (Large)

**Characteristics:**
- ~4.0 bits per weight
- Highest quality 3-bit variant
- Similar size to Q4_0

**Recommendation:**
- Usually prefer Q4_K_S at this size
- Use only if 3-bit requirement is strict

---

### Q4_K: Two Variants

The most popular quantization tier.

#### Q4_K_S (Small)

**Characteristics:**
- ~4.3 bits per weight
- Excellent size/quality balance
- Slightly better than Q4_0

**Structure:**
```
Super-block: 256 values
- Scales: 6-bit quantized
- Quantized values: 4-bit
- Minimal metadata overhead
```

**Use Cases:**
- Maximizing model size in limited memory
- When every MB counts
- Batch inference (less memory for more batches)

**Quality:**
- 2-4% perplexity increase
- Usually imperceptible in practice
- Excellent for most applications

**Recommended for:**
- 70B models on consumer hardware (24GB VRAM)
- Running multiple models simultaneously
- Cloud deployments optimizing cost

#### Q4_K_M (Medium)

**Characteristics:**
- ~4.8 bits per weight
- **Best overall format** for most use cases
- Sweet spot of quality and size

**Structure:**
```
Super-block: 256 values
- 8 sub-blocks of 32 values
- Scales: 6-bit quantized
- Super-scale: FP16
- Some values at higher precision
```

**Use Cases:**
- **Default recommendation**
- Production deployments
- General-purpose inference

**Quality:**
- 1-3% perplexity increase
- Nearly identical to FP16 in practice
- Passes most quality thresholds

**Recommended for:**
- All models when possible
- Production systems
- When you want to "set and forget"

---

### Q5_K: Two Variants

Higher quality, approaching FP16.

#### Q5_K_S (Small)

**Characteristics:**
- ~5.5 bits per weight
- Minimal overhead variant
- Excellent quality

**Use Cases:**
- When Q4_K_M isn't quite enough
- Quality-sensitive applications
- 13B-30B models

**Quality:**
- 0.5-1.5% perplexity increase
- Very close to FP16
- Hard to distinguish from higher precision

#### Q5_K_M (Medium)

**Characteristics:**
- ~5.9 bits per weight
- **Nearly lossless quality**
- Recommended for quality-first scenarios

**Structure:**
```
Super-block: 256 values
- Higher precision scales
- Some values at 6-bit
- Sophisticated bit allocation
```

**Use Cases:**
- Quality-critical applications
- When memory allows but want smaller than Q8
- Professional use cases

**Quality:**
- 0.3-1% perplexity increase
- Essentially indistinguishable from FP16
- Excellent benchmark performance

**Recommended for:**
- 7B-13B models with quality priority
- Professional deployments
- When 2x compression over FP16 needed with minimal loss

---

### Q6_K: Highest K-Quant

**Characteristics:**
- ~6.6 bits per weight
- Extremely high quality
- Between Q5_K_M and Q8_0

**Structure:**
```
Super-block: 256 values
- 6-bit quantization
- FP16 scales
- Sophisticated encoding
```

**Use Cases:**
- Maximum quality with compression
- Alternative to Q8_0 with better size
- Research and evaluation

**Quality:**
- <0.5% perplexity increase
- Virtually identical to FP16
- Benchmark scores match FP16

**When to Use:**
- Need better than Q5_K_M
- Q8_0 too large
- Usually overkill - Q5_K_M sufficient

**Recommended for:**
- Specific quality requirements
- Generally prefer Q5_K_M or Q8_0

---

## Importance Quantization (IQ)

IQ formats are experimental and use importance matrices for sophisticated bit allocation.

### IQ2_XXS

- ~2.0 bpw
- Ultra-compressed
- Uses importance matrix
- Research-grade

### IQ2_XS, IQ2_S, IQ2_M

Progressive quality improvements at ~2.3-2.6 bpw

### IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M

Similar progression at ~3 bpw range

**Note:** IQ formats are cutting-edge and may not be stable. Prefer K-quants for production.

---

## Format Comparison Matrix

### For LLaMA-7B Model

| Format  | Size  | Perplexity† | Speed‡ | Quality | Use Case |
|---------|-------|-------------|--------|---------|----------|
| FP16    | 13.5G | 5.68        | 1.0x   | ★★★★★   | Reference |
| Q8_0    | 7.2G  | 5.70        | 1.8x   | ★★★★★   | Max quality |
| Q6_K    | 5.5G  | 5.72        | 2.0x   | ★★★★★   | Excellent |
| Q5_K_M  | 4.8G  | 5.75        | 2.2x   | ★★★★☆   | Recommended |
| Q5_K_S  | 4.7G  | 5.77        | 2.2x   | ★★★★☆   | Great |
| Q4_K_M  | 4.1G  | 5.82        | 2.4x   | ★★★★☆   | **DEFAULT** |
| Q4_K_S  | 3.9G  | 5.85        | 2.5x   | ★★★☆☆   | Compact |
| Q4_0    | 3.8G  | 5.90        | 2.5x   | ★★★☆☆   | Legacy |
| Q3_K_M  | 3.3G  | 6.10        | 2.6x   | ★★★☆☆   | Acceptable |
| Q3_K_S  | 3.0G  | 6.25        | 2.7x   | ★★☆☆☆   | Limited |
| Q2_K    | 2.5G  | 6.80        | 2.8x   | ★★☆☆☆   | Experimental |

† Lower is better
‡ Relative to FP16 on CPU

### Quick Selection Guide

**Need the best quality?**
→ Q8_0 (if memory allows) or Q5_K_M

**Want the best balance?**
→ Q4_K_M (universal recommendation)

**Tight on memory?**
→ Q4_K_S or Q3_K_M

**Extreme memory limits?**
→ Q3_K_S (test quality carefully)

**For 70B models?**
→ Q4_K_M (ideal) or Q4_K_S (if needed)

**For 7B-13B models?**
→ Q5_K_M (quality) or Q4_K_M (balanced)

---

## How to Choose the Right Format

### Decision Tree

```
1. What's your memory budget?
   ├─ Plenty (2x model size) → Q8_0 or Q6_K
   ├─ Comfortable (1.5x model size) → Q5_K_M
   ├─ Typical (1.2x model size) → Q4_K_M ← START HERE
   ├─ Tight (1.1x model size) → Q4_K_S
   └─ Critical (<1.1x model size) → Q3_K_M or Q3_K_S

2. How sensitive is your use case?
   ├─ Very (code, math, reasoning) → Higher precision (+1 tier)
   ├─ Moderate (general chat) → As calculated
   └─ Tolerant (creative writing) → Can go -1 tier

3. What's your hardware?
   ├─ Modern CPU with AVX2/AVX512 → K-quants work great
   ├─ ARM (Apple Silicon) → K-quants optimized
   ├─ GPU (CUDA) → All formats work, prefer K-quants
   └─ Old/limited CPU → Simpler formats (Q4_0, Q5_0) may be faster
```

### Hardware-Specific Recommendations

#### Consumer GPU (8-12GB VRAM)

**For 7B models:**
- Use Q5_K_M or Q6_K (quality)
- Or Q4_K_M (if running multiple models)

**For 13B models:**
- Use Q4_K_M or Q5_K_S

**For 30B+ models:**
- Not recommended (insufficient VRAM)
- Use CPU with Q4_K_M

#### High-End GPU (24GB VRAM)

**For 7B-13B:**
- Use Q8_0 or FP16 (maximum quality)

**For 30B:**
- Use Q5_K_M or Q6_K

**For 70B:**
- Use Q4_K_M (fits with context)
- Or Q4_K_S (maximum context)

#### CPU with 32GB RAM

**For 7B:**
- Use Q8_0 or Q6_K

**For 13B:**
- Use Q5_K_M or Q6_K

**For 30B:**
- Use Q4_K_M

**For 70B:**
- Not recommended (too slow)
- If must: Q3_K_M or Q4_K_S

#### CPU with 64GB+ RAM

**For 7B-30B:**
- Use Q8_0 (maximum quality)

**For 70B:**
- Use Q4_K_M or Q5_K_S

**For 180B:**
- Use Q3_K_M or Q4_K_S

---

## Format Internals

### Block Alignment

GGUF formats use specific block sizes for memory alignment:

```c
#define QK4_0 32    // Q4_0 block size
#define QK4_1 32    // Q4_1 block size
#define QK5_0 32    // Q5_0 block size
#define QK5_1 32    // Q5_1 block size
#define QK8_0 32    // Q8_0 block size
#define QK_K 256    // K-quants super-block size
```

### Memory Layout Example: Q4_K_M

```
Super-block (256 values):
┌─────────────────────────────────────┐
│ Super-scale (FP16): 2 bytes         │
├─────────────────────────────────────┤
│ Sub-block scales (6-bit): 12 bytes  │
├─────────────────────────────────────┤
│ Sub-block mins (6-bit): 12 bytes    │
├─────────────────────────────────────┤
│ Quantized values (4-bit): 128 bytes │
└─────────────────────────────────────┘
Total: 154 bytes for 256 values
= 4.8 bits per value
```

### Decoding Process

For Q4_K_M:

```c
float decode_q4_k_m(block_q4_k block, int index) {
    // Determine sub-block
    int sb = index / 32;
    int sb_offset = index % 32;

    // Get scale and min for this sub-block
    float scale = decode_scale(block.scales[sb]) * block.d;
    float min = decode_min(block.mins[sb]) * block.dmin;

    // Get quantized value
    int q = get_nibble(block.qs, index);

    // Dequantize
    return scale * q + min;
}
```

---

## Conversion Between Formats

### Using llama.cpp

```bash
# Convert model to different quantization
./llama-quantize \
    model.gguf \
    model-q4_k_m.gguf \
    Q4_K_M

# Available formats:
# Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
# Q2_K, Q3_K_S, Q3_K_M, Q3_K_L
# Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K
# IQ2_XXS, IQ2_XS, IQ3_XXS, etc.
```

### Batch Conversion Script

```bash
#!/bin/bash
# quantize_all.sh - Create multiple quantizations

MODEL="llama-2-7b.gguf"
FORMATS="Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K Q8_0"

for format in $FORMATS; do
    echo "Creating $format..."
    ./llama-quantize "$MODEL" "${MODEL%.gguf}-${format,,}.gguf" "$format"
done
```

### Importance Quantization

Some formats require importance matrix:

```bash
# Generate importance matrix (imatrix)
./llama-imatrix \
    -m model.gguf \
    -f calibration_data.txt \
    -o model.imatrix

# Use imatrix for quantization
./llama-quantize \
    --imatrix model.imatrix \
    model.gguf \
    model-iq3_m.gguf \
    IQ3_M
```

### Conversion Tips

1. **Always keep FP16 master**
   - Don't quantize from quantized
   - Maintain FP16 as source of truth

2. **Test before deleting**
   - Verify quantized model works
   - Check perplexity
   - Test sample prompts

3. **Storage management**
   ```bash
   # Original FP16: 13.5 GB
   # Q8_0: 7.2 GB
   # Q5_K_M: 4.8 GB
   # Q4_K_M: 4.1 GB

   # Keep only what you need
   # Total for all: ~30 GB
   ```

---

## Interview Questions

### Format Knowledge

1. **Q: What's the difference between Q4_0 and Q4_K_M?**

   A: Q4_0 is a legacy format with 32-value blocks, each having a single FP16 scale (4.5 bpw). Q4_K_M is a k-quant with 256-value super-blocks, sub-blocks with individual scales, and mixed precision within blocks (4.8 bpw). Q4_K_M achieves significantly better quality at slightly larger size through sophisticated bit allocation.

2. **Q: Why do k-quants have better quality than legacy formats at the same bit-width?**

   A: K-quants use larger blocks (256 vs 32 values) with sub-block structure, allowing:
   - Finer-grained scale factors
   - Mixed precision within blocks
   - Importance-based bit allocation
   - Better adaptation to local weight distributions

   This reduces quantization error versus uniform quantization.

3. **Q: Explain the S/M/L variants in Q3_K and Q4_K formats.**

   A:
   - S (Small): Minimal metadata, smallest size, lower quality
   - M (Medium): More metadata, better quality, moderate size
   - L (Large): Maximum metadata, best quality, largest size

   They trade size for quality within the same nominal bit-width. Example: Q3_K_S (3.4 bpw), Q3_K_M (3.7 bpw), Q3_K_L (4.0 bpw).

4. **Q: When would you use Q8_0 instead of FP16?**

   A: Q8_0 when:
   - Need 2x memory reduction with minimal quality loss
   - Storage/bandwidth constrained
   - Running on CPU (faster due to less data movement)
   - Quality loss <0.5% acceptable

   FP16 when:
   - Absolute maximum quality required
   - Memory abundant
   - Using as training reference
   - GPU with FP16 acceleration

### Practical Application

5. **Q: You need to deploy LLaMA-70B on a system with 48GB RAM. What quantization do you choose and why?**

   A: **Q4_K_M** (≈37-40GB including overhead):
   - Fits comfortably with headroom for context/OS
   - Excellent quality (2-3% perplexity increase)
   - Industry standard for 70B
   - Well-tested and optimized

   Alternative: Q4_K_S if need more context length (≈35GB)
   Avoid: Q3_K (unnecessary quality loss), Q5_K (won't fit with adequate context)

6. **Q: A user reports quality degradation after quantization. How would you diagnose which format is acceptable?**

   A: Systematic testing:

   ```bash
   # 1. Measure perplexity
   ./llama-perplexity -m model-q8_0.gguf -f wikitext.txt
   ./llama-perplexity -m model-q5_k_m.gguf -f wikitext.txt
   ./llama-perplexity -m model-q4_k_m.gguf -f wikitext.txt

   # 2. Benchmark on standard tasks
   # Run MMLU, HellaSwag, etc.

   # 3. A/B test on user's specific prompts
   # Generate outputs from different quantizations
   # Compare quality subjectively

   # 4. Find acceptable threshold
   # Start high (Q8_0), go down until quality unacceptable
   ```

   Set threshold: <5% perplexity increase, <2% benchmark drop

7. **Q: Explain why Q4_K_M is recommended as the default format.**

   A: Q4_K_M balances multiple factors:
   - **Quality**: 1-3% perplexity increase (acceptable for most uses)
   - **Size**: 4x compression vs FP16 (enables larger models/longer context)
   - **Speed**: Good inference performance
   - **Compatibility**: Well-supported across all backends
   - **Tested**: Extensively validated in production
   - **Versatile**: Works well for 7B to 70B+ models

   It's the "safe default" - rarely wrong choice.

### Advanced Topics

8. **Q: How would you implement a custom quantization format for a specific use case?**

   A: Steps:

   ```c
   // 1. Define block structure
   struct block_custom {
       ggml_fp16_t scale;      // Scale factor
       uint8_t metadata[N];    // Custom metadata
       uint8_t qs[M];         // Quantized values
   };

   // 2. Implement quantization function
   void quantize_custom(const float* src, block_custom* dst, int n);

   // 3. Implement dequantization
   void dequantize_custom(const block_custom* src, float* dst, int n);

   // 4. Register with GGML
   ggml_register_quantization(GGML_TYPE_CUSTOM, ...);

   // 5. Optimize with SIMD
   // Implement AVX2/NEON versions

   // 6. Test extensively
   // Perplexity, benchmarks, production load
   ```

9. **Q: Why does Q5_K_M sometimes have better quality than Q6_K despite fewer nominal bits?**

   A: This can occur due to:
   - Different block structures and algorithms
   - Optimization focus (Q5_K_M highly optimized)
   - Specific weight distributions in the model
   - Rounding behavior in specific layers

   However, typically Q6_K ≥ Q5_K_M in quality. If Q5_K_M appears better:
   - Statistical variance in measurement
   - Task-specific behavior
   - Implementation differences

   Always measure on multiple metrics and test sets.

10. **Q: Design a quantization strategy for a model serving system handling 1000 QPS with diverse model sizes.**

    A: Multi-tier strategy:

    ```yaml
    Small Models (7B-13B):
      Quantization: Q5_K_M or Q6_K
      Reason: Quality priority, memory not limiting factor
      Deployment: Multiple instances for QPS

    Medium Models (30B-34B):
      Quantization: Q4_K_M
      Reason: Balance for multi-instance deployment
      Deployment: Fewer instances, load balanced

    Large Models (70B+):
      Quantization: Q4_K_M or Q4_K_S
      Reason: Memory constraints, single instance per GPU
      Deployment: Queue-based serving

    Strategy:
    - A/B test quantizations before rollout
    - Monitor perplexity drift
    - Gradual rollout with canary deployments
    - Per-model quality SLAs
    - Auto-scaling based on load
    - Fallback to higher precision if quality issues detected
    ```

---

## Summary

**Key Takeaways:**

1. **K-quants are superior** to legacy formats at same bit-width
2. **Q4_K_M is the default recommendation** for most use cases
3. **Q5_K_M for quality-first**, Q4_K_S for size-first
4. **Q8_0 for minimal loss**, Q3_K for extreme compression
5. Always test on your specific use case and metrics

**Format Selection Shorthand:**
- Quality priority: Q8_0 → Q6_K → Q5_K_M
- Balanced: Q5_K_M → Q4_K_M → Q4_K_S
- Size priority: Q4_K_S → Q3_K_M → Q3_K_S

**Next Steps:**
- Lesson 3: Performance optimization techniques
- Lab 2: Hands-on format comparison
- Tutorial: Building a quantization pipeline

---

**Further Reading:**

- [GGML Quantization Source](https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c)
- [K-Quants PR Discussion](https://github.com/ggerganov/llama.cpp/pull/1684)
- [Quantization Benchmarks](https://github.com/ggerganov/llama.cpp/discussions/2094)

**Author**: Agent 5 (Documentation Specialist)
**Module**: 3 - Quantization & Optimization
**Last Updated**: 2025-11-18
