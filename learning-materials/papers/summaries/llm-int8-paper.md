# LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale

**Paper**: "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"
**Authors**: Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
**Published**: November 2022 (NeurIPS)
**Link**: https://arxiv.org/abs/2208.07339
**Module**: 3 - Quantization Deep Dive
**Impact**: ⭐⭐⭐⭐⭐

---

## Executive Summary

LLM.int8() enables zero-degradation 8-bit inference for large language models through mixed-precision decomposition. By identifying and preserving outlier features in FP16 while quantizing the rest to INT8, it achieves 2× memory reduction with no quality loss.

**Key Innovation**: Vector-wise quantization + mixed-precision decomposition for outliers

---

## 1. The Outlier Problem

### Emergence of Outliers at Scale

```python
# Observation: Outliers emerge at ~6.7B parameters
def analyze_outliers(activations):
    """
    At small scale (<1B params): activations are roughly normal
    At large scale (>6B params): extreme outliers appear (>100× median)
    """
    median = activations.median()
    max_val = activations.max()
    outlier_ratio = max_val / median

    print(f"Outlier ratio: {outlier_ratio:.1f}×")
    # Small models: 5-10×
    # Large models (OPT-175B, BLOOM-176B): 100-1000×!

# These outliers destroy naive quantization
```

**Why Outliers Matter**:
```
Quantization error ∝ (max - min) / (2^bits - 1)

With outliers:
- max = 1000, min = -1000
- 8-bit: error = 2000 / 255 ≈ 7.8 per step

Without outliers:
- max = 10, min = -10
- 8-bit: error = 20 / 255 ≈ 0.08 per step

Outliers cause 100× larger errors!
```

---

## 2. LLM.int8() Solution

### 2.1 Vector-wise Quantization

```python
def vector_wise_quantization(X):
    """
    Quantize each row independently
    (vs tensor-wise: single scale for entire tensor)
    """
    n_rows = X.shape[0]
    X_quant = torch.zeros_like(X, dtype=torch.int8)
    scales = torch.zeros(n_rows)

    for i in range(n_rows):
        row = X[i, :]

        # Compute scale for this row
        absmax = row.abs().max()
        scale = absmax / 127.0

        # Quantize
        X_quant[i, :] = (row / scale).round().clamp(-127, 127).to(torch.int8)
        scales[i] = scale

    return X_quant, scales

# Dequantization
def dequantize(X_quant, scales):
    return X_quant.float() * scales.unsqueeze(1)
```

**Benefit**: Adapts to per-row magnitude variations

---

### 2.2 Mixed-Precision Decomposition

**Core Algorithm**:
```python
def llm_int8_matmul(X, W, threshold=6.0):
    """
    LLM.int8() matrix multiplication

    Args:
        X: Activations [batch, seq_len, hidden]
        W: Weights [hidden_out, hidden_in]
        threshold: Outlier threshold (default 6.0 std devs)

    Returns:
        Y: Output [batch, seq_len, hidden_out]
    """
    hidden_dim = X.shape[-1]

    # 1. Identify outlier features (columns of X, rows of W)
    X_absmax = X.abs().max(dim=[0, 1])[0]  # [hidden_dim]
    outlier_mask = X_absmax > threshold * X_absmax.mean()

    outlier_idx = torch.where(outlier_mask)[0]
    normal_idx = torch.where(~outlier_mask)[0]

    # 2. Decompose X and W into outlier and normal parts
    X_outlier = X[:, :, outlier_idx]  # FP16
    X_normal = X[:, :, normal_idx]    # Will be INT8

    W_outlier = W[:, outlier_idx]     # FP16
    W_normal = W[:, normal_idx]       # Will be INT8

    # 3. FP16 matmul for outlier features (high precision needed)
    Y_outlier = torch.matmul(X_outlier, W_outlier.T)

    # 4. INT8 matmul for normal features (quantize both X and W)
    X_normal_int8, X_scales = vector_wise_quantization(X_normal)
    W_normal_int8, W_scales = vector_wise_quantization(W_normal.T)

    # INT8 matmul (fast!)
    Y_normal_int8 = torch.matmul(X_normal_int8.to(torch.int32),
                                  W_normal_int8.T.to(torch.int32))

    # Dequantize
    Y_normal = Y_normal_int8.float() * X_scales.unsqueeze(-1) * W_scales.unsqueeze(0)

    # 5. Combine results
    Y = Y_outlier + Y_normal

    return Y

# Typical outlier ratio: 0.1-1% of features
# So 99%+ of computation is INT8 (fast), 1% is FP16 (accurate)
```

---

## 3. Key Results

### Performance (OPT-175B, BLOOM-176B)

| Model | Method | Memory | Perplexity | Quality Loss |
|-------|--------|--------|------------|--------------|
| OPT-175B | FP16 | 350 GB | 10.13 | Baseline |
| OPT-175B | Naive INT8 | 175 GB | **>1000** | ❌ Catastrophic |
| OPT-175B | LLM.int8() | 175 GB | **10.13** | ✅ **Zero!** |
|  |  |  |  |  |
| BLOOM-176B | FP16 | 352 GB | 10.94 | Baseline |
| BLOOM-176B | LLM.int8() | 176 GB | **10.94** | ✅ **Zero!** |

**Conclusion**: 2× memory reduction with NO quality loss

---

### Speed Comparison

```
A100 GPU (80GB), OPT-66B:
- FP16: Doesn't fit (need 132 GB)
- LLM.int8(): 66 GB, 75 tokens/sec
- Speedup: Infinite (enables inference that wasn't possible!)

RTX 3090 (24GB), OPT-13B:
- FP16: 26 GB (doesn't fit)
- LLM.int8(): 13 GB (fits!), 42 tokens/sec
```

**Trade-off**: ~15-20% slower than FP16 (when both fit) due to dequantization overhead, but enables much larger models on same hardware.

---

## 4. Implementation

### 4.1 bitsandbytes Library

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb

# Load model with LLM.int8()
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,          # Enable LLM.int8()
    device_map="auto",           # Automatic device placement
    llm_int8_threshold=6.0       # Outlier threshold
)

# Inference (automatic int8 matmul)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))

# Memory usage: ~7 GB (vs 14 GB for FP16)
```

### 4.2 Custom LLM.int8() Layer

```python
import bitsandbytes.functional as F

class Int8Linear(nn.Module):
    def __init__(self, in_features, out_features, threshold=6.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        # Weight stored in INT8 (except outliers)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.int8))
        self.weight_scale = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        # LLM.int8() matmul
        return F.linear_8bit_mixed_precision(
            x, self.weight, self.weight_scale,
            threshold=self.threshold
        )
```

---

## 5. Comparison to Other Methods

| Method | Bits | Quality | Memory | Speed | Complexity |
|--------|------|---------|--------|-------|------------|
| FP16 | 16 | ★★★★★ | 1.0× | 1.0× | Simple |
| LLM.int8() | 8 | ★★★★★ | 0.5× | 0.8× | Medium |
| GPTQ | 4 | ★★★★ | 0.25× | 1.2× (GPU) | High |
| AWQ | 4 | ★★★★★ | 0.25× | 1.3× (GPU) | High |

**LLM.int8() Niche**:
- When quality is paramount (zero loss)
- When 4-bit is too aggressive
- When you want simple integration (HuggingFace native support)

---

## 6. For llama.cpp Users

### Relationship to Q8_0 Quantization

```bash
# llama.cpp Q8_0 is similar concept to LLM.int8()
./llama-quantize model-f16.gguf model-q8.gguf Q8_0

# Differences:
# - llama.cpp Q8_0: Per-block quantization (32-element blocks)
# - LLM.int8(): Per-row quantization + outlier handling
# - llama.cpp: Optimized for CPU (SIMD, cache)
# - LLM.int8(): Optimized for GPU (CUDA kernels)
```

**When to use Q8_0 in llama.cpp**:
- Need high quality (better than Q4/Q5)
- Have enough RAM/VRAM
- CPU inference (Q8_0 is fast on CPU)

---

## 7. Key Takeaways

### Core Innovations
1. **Outlier identification**: Extreme values emerge at scale (>6B params)
2. **Mixed precision**: FP16 for outliers, INT8 for rest
3. **Vector-wise quantization**: Per-row scales handle magnitude variations
4. **Zero degradation**: No quality loss (vs 4-bit methods)

### Practical Use
✅ **Best for**: Quality-critical applications, enabling large models on limited GPUs
✅ **Limitations**: 8-bit (not 4-bit), GPU-only, 15-20% slower than FP16

---

## Further Reading

**Paper**: https://arxiv.org/abs/2208.07339
**Code**: https://github.com/TimDettmers/bitsandbytes
**HuggingFace Integration**: `load_in_8bit=True` parameter

---

**Status**: Complete | Module 3 (3/3) papers done
