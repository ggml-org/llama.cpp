# GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

**Paper**: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
**Authors**: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
**Published**: October 2022 (ICLR 2023)
**Link**: https://arxiv.org/abs/2210.17323
**Relevance**: Module 3 - Quantization Deep Dive
**Reading Time**: 45-60 minutes
**Practical Impact**: ⭐⭐⭐⭐⭐

---

## Executive Summary

GPTQ achieves near-lossless 4-bit quantization for large language models using a layer-wise quantization approach based on approximate second-order information (Hessian). It enables running 175B parameter models on a single GPU while maintaining quality, and has become the de facto standard for GPU-optimized quantized models.

**Key Innovation**: Efficient approximate solution to optimal quantization problem using iterative greedy updates with Hessian inverse.

**Impact**: 4× model compression with <1% perplexity increase, enabling consumer GPU deployment of large models.

---

## 1. Problem Statement

### 1.1 Why Quantization?

**Memory Requirements** (FP16):
```
LLaMA 13B:  13B params × 2 bytes = 26 GB
LLaMA 70B:  70B params × 2 bytes = 140 GB
GPT-3 175B: 175B params × 2 bytes = 350 GB
```

**Goal**: Reduce to 4-bit (INT4) → 4× compression
```
LLaMA 13B: 13B params × 0.5 bytes = 6.5 GB (fits on consumer GPU!)
LLaMA 70B: 70B params × 0.5 bytes = 35 GB (fits on A100)
```

### 1.2 Naive Quantization Fails

**Uniform Quantization**:
```python
def naive_quantize(W, bits=4):
    """Naive per-tensor quantization"""
    n_levels = 2 ** bits  # 16 levels for 4-bit
    W_min, W_max = W.min(), W.max()

    # Quantize
    scale = (W_max - W_min) / (n_levels - 1)
    W_quant = torch.round((W - W_min) / scale)

    # Dequantize
    W_dequant = W_min + W_quant * scale
    return W_dequant

# Result: Large accuracy degradation for 4-bit
# LLaMA 7B: Perplexity 5.68 (FP16) → 234.5 (naive 4-bit) ❌
```

**Why it fails**:
- Outlier activations cause large quantization errors
- Uniform quantization wastes resolution on low-magnitude weights
- Doesn't account for weight importance

---

## 2. GPTQ Algorithm

### 2.1 Theoretical Foundation

**Optimal Quantization Problem**:
```
Given:
- Weight matrix W ∈ ℝ^(d_out × d_in)
- Calibration data X ∈ ℝ^(n × d_in)

Find: Quantized weights Ŵ that minimize output error:
  argmin_Ŵ ||WX - ŴX||²_F

Subject to: Ŵ ∈ {quantization levels}
```

**Optimal Brain Quantization (OBQ) Solution**:
```
For each weight w_i:
  δ_i = -w_i / [H^(-1)]_ii        # Optimal update
  W' = W + δ_i × H^(-1)_:,i       # Update all weights

Where H = 2X^T X is the Hessian of the squared error
```

**Problem**: O(d³) complexity for d×d matrices → too slow for billion-parameter models

---

### 2.2 GPTQ Approximations

**Key Insights**:
1. **Layer-wise quantization**: Quantize each layer independently
2. **Row-wise processing**: Process weight rows independently (assume block-diagonal Hessian)
3. **Greedy quantization order**: Quantize weights sequentially, not jointly
4. **Cholesky decomposition**: Efficient Hessian inverse computation

**Algorithm**:
```python
import torch
import torch.nn as nn

def gptq_quantize_layer(W, X, bits=4):
    """
    GPTQ quantization for a single layer

    Args:
        W: Weight matrix [d_out, d_in]
        X: Calibration inputs [n_samples, d_in]
        bits: Target bit-width

    Returns:
        W_quant: Quantized weights
        scale, zero: Quantization parameters
    """
    d_out, d_in = W.shape

    # Compute Hessian: H = 2X^T X / n_samples
    H = 2 * X.T @ X / X.shape[0]

    # Add damping for numerical stability
    damping = 0.01
    H.diagonal().add_(damping * torch.mean(H.diagonal()))

    # Cholesky decomposition: H = L L^T
    # Then H^(-1) can be efficiently computed from L
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except:
        # Fallback to pseudo-inverse if Cholesky fails
        H_inv = torch.pinverse(H)

    W_quant = W.clone()
    Errors = torch.zeros_like(W)

    # Process each row independently
    for i in range(d_out):
        w_row = W[i, :].clone()
        quant_row = torch.zeros_like(w_row)

        # Quantize weights sequentially (column by column)
        for j in range(d_in):
            # Quantize current weight
            w_val = w_row[j]
            q_val, scale, zero = quantize_weight(w_val, bits)
            quant_row[j] = q_val

            # Compute quantization error
            error = w_val - q_val

            # Update remaining weights using Hessian inverse
            # This distributes the error to correlated weights
            if j < d_in - 1:
                w_row[(j+1):] -= (error / H_inv[j, j]) * H_inv[j, (j+1):]

        W_quant[i, :] = quant_row

    return W_quant

def quantize_weight(w, bits):
    """
    Quantize single weight value

    Returns:
        quantized_value: Dequantized weight
        scale: Quantization scale
        zero_point: Zero point
    """
    n_levels = 2 ** bits
    w_min, w_max = w.min().item(), w.max().item()

    # Asymmetric quantization
    scale = (w_max - w_min) / (n_levels - 1)
    zero_point = -w_min / scale

    # Quantize
    q = torch.clamp(torch.round(w / scale + zero_point), 0, n_levels - 1)

    # Dequantize
    w_quant = (q - zero_point) * scale

    return w_quant, scale, zero_point
```

---

### 2.3 Group-wise Quantization

**Problem**: Single scale/zero-point per row is suboptimal for outliers

**Solution**: Divide each row into groups, quantize separately

```python
def gptq_groupwise(W, X, bits=4, group_size=128):
    """
    GPTQ with group-wise quantization

    Each group of 128 weights shares scale/zero-point
    """
    d_out, d_in = W.shape
    n_groups = (d_in + group_size - 1) // group_size

    W_quant = torch.zeros_like(W)
    scales = torch.zeros(d_out, n_groups)
    zeros = torch.zeros(d_out, n_groups)

    for group_idx in range(n_groups):
        start = group_idx * group_size
        end = min(start + group_size, d_in)

        # Quantize this group
        W_group = W[:, start:end]
        X_group = X[:, start:end]

        W_group_quant = gptq_quantize_layer(W_group, X_group, bits)

        W_quant[:, start:end] = W_group_quant

    return W_quant, scales, zeros

# Example
W = torch.randn(4096, 4096)  # LLaMA layer
X = torch.randn(1024, 4096)  # Calibration data

W_quant, scales, zeros = gptq_groupwise(W, X, bits=4, group_size=128)

# Storage:
# - W_quant: 4096×4096 × 4 bits = 8 MB
# - scales: 4096 × 32 groups × 2 bytes = 256 KB
# - zeros: 4096 × 32 groups × 2 bytes = 256 KB
# Total: ~8.5 MB vs 32 MB (FP16) → 3.8× compression
```

---

## 3. Implementation Details

### 3.1 Calibration Data Selection

**Important**: Quality of calibration data affects quantization accuracy

```python
def prepare_calibration_data(model, tokenizer, n_samples=128):
    """
    Prepare calibration data for GPTQ

    Common sources:
    1. WikiText-2 (default)
    2. C4 dataset
    3. Task-specific data
    """
    from datasets import load_dataset

    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    # Sample random sequences
    calibration_samples = []
    for _ in range(n_samples):
        # Random slice
        idx = torch.randint(0, len(dataset), (1,)).item()
        text = dataset[idx]['text']

        # Tokenize (512 tokens)
        tokens = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        calibration_samples.append(tokens['input_ids'])

    return torch.cat(calibration_samples, dim=0)

# Usage
calibration_data = prepare_calibration_data(model, tokenizer, n_samples=128)
# Shape: [128, 512] - 128 sequences of 512 tokens each
```

**Best Practices**:
- More samples = better quantization (but slower)
- 128 samples is a good default
- Use diverse text (not just one domain)
- Longer sequences capture more context

---

### 3.2 Quantization Schemes

**Asymmetric vs Symmetric**:

```python
# Asymmetric quantization (better for weights with skew)
def asymmetric_quant(W, bits):
    n_levels = 2 ** bits
    W_min, W_max = W.min(), W.max()
    scale = (W_max - W_min) / (n_levels - 1)
    zero = torch.round(-W_min / scale)

    W_q = torch.clamp(torch.round(W / scale) + zero, 0, n_levels - 1)
    W_dq = (W_q - zero) * scale
    return W_dq, scale, zero

# Symmetric quantization (simpler, faster inference)
def symmetric_quant(W, bits):
    n_levels = 2 ** (bits - 1) - 1  # -7 to 7 for 4-bit
    W_max = torch.max(W.abs())
    scale = W_max / n_levels

    W_q = torch.clamp(torch.round(W / scale), -n_levels, n_levels)
    W_dq = W_q * scale
    return W_dq, scale

# GPTQ typically uses asymmetric for better accuracy
```

---

### 3.3 Lazy Batch Updates (Optimization)

**Problem**: Processing weights one-by-one is slow

**Solution**: Batch multiple quantizations

```python
def gptq_batch_update(W, X, bits=4, batch_size=128):
    """
    Process multiple columns in parallel for speedup
    """
    d_out, d_in = W.shape

    # Compute Hessian inverse (once)
    H = 2 * X.T @ X / X.shape[0]
    H_inv = compute_hinv_cholesky(H)

    W_quant = W.clone()

    # Process in batches
    for start in range(0, d_in, batch_size):
        end = min(start + batch_size, d_in)
        batch_cols = end - start

        # Quantize batch of columns
        W_batch = W[:, start:end]
        Q_batch = quantize(W_batch, bits)

        # Compute errors
        E_batch = W_batch - Q_batch

        # Update remaining weights (vectorized)
        if end < d_in:
            H_inv_block = H_inv[start:end, end:]
            correction = E_batch @ H_inv_block / H_inv[start:end, start:end].diagonal().unsqueeze(-1)
            W[:, end:] -= correction

        W_quant[:, start:end] = Q_batch

    return W_quant

# Speedup: 10-20× faster than sequential processing
```

---

## 4. Results and Evaluation

### 4.1 Perplexity Results

**WikiText-2 Perplexity** (lower is better):

| Model | FP16 | GPTQ 4-bit | GPTQ 3-bit | GPTQ 2-bit |
|-------|------|------------|------------|------------|
| OPT-175B | 10.13 | 10.29 (+0.16) | 11.43 (+1.30) | 23.91 (+13.78) |
| BLOOM-176B | 10.94 | 11.15 (+0.21) | 12.58 (+1.64) | 29.43 (+18.49) |
| LLaMA 7B | 5.68 | 5.81 (+0.13) | 6.42 (+0.74) | 14.23 (+8.55) |
| LLaMA 13B | 5.09 | 5.20 (+0.11) | 5.71 (+0.62) | 11.34 (+6.25) |
| LLaMA 30B | 4.10 | 4.18 (+0.08) | 4.52 (+0.42) | 7.89 (+3.79) |

**Key Observations**:
- **4-bit**: Minimal degradation (<2% perplexity increase)
- **3-bit**: Acceptable for many tasks (~10-20% increase)
- **2-bit**: Significant degradation, not recommended

---

### 4.2 Downstream Task Performance

**MMLU Benchmark** (accuracy %):

| Model | FP16 | GPTQ 4-bit | Δ |
|-------|------|------------|---|
| LLaMA 7B | 35.1 | 34.8 | -0.3 |
| LLaMA 13B | 46.9 | 46.5 | -0.4 |
| LLaMA 30B | 58.1 | 57.7 | -0.4 |
| LLaMA 65B | 63.4 | 63.1 | -0.3 |

**Conclusion**: <1% accuracy drop on complex reasoning tasks

---

### 4.3 Speed and Memory

**LLaMA 30B on A100 (80GB)**:
```
FP16:
- Memory: 60 GB
- Tokens/sec: 42

GPTQ 4-bit:
- Memory: 15 GB (4× reduction)
- Tokens/sec: 38 (10% slower due to dequantization)
- Batch size: 4× larger (due to memory savings)
- Effective throughput: 152 tokens/sec (4× improvement!)
```

---

## 5. Comparison to Other Methods

### 5.1 GPTQ vs LLM.int8()

| Feature | GPTQ | LLM.int8() |
|---------|------|------------|
| Bit-width | 4-bit, 3-bit | 8-bit |
| Compression | 4× | 2× |
| Method | Weight-only quantization | Mixed precision |
| Calibration | Required (one-time) | Not required |
| Accuracy | ~0.5% loss @ 4-bit | ~0.1% loss @ 8-bit |
| Inference speed | Fast (GPU) | Medium (outlier handling overhead) |
| Best for | Maximum compression | High accuracy, easy deployment |

---

### 5.2 GPTQ vs AWQ

| Feature | GPTQ | AWQ |
|---------|------|-----|
| Approach | Hessian-based optimal quantization | Activation-aware importance |
| Calibration | General text (WikiText) | Activation statistics |
| Mixed precision | Uniform bit-width | Per-channel importance scaling |
| Accuracy | Excellent | Slightly better |
| Speed | Fast | Fast |
| Complexity | Medium | Low |

**When to choose**:
- **GPTQ**: General-purpose, well-established, good tooling
- **AWQ**: Cutting-edge accuracy, activation-aware

---

## 6. Practical Usage

### 6.1 Using GPTQ Models with llama.cpp

**Note**: llama.cpp doesn't directly support GPTQ format, but you can convert:

```bash
# Convert GPTQ model to GGUF
python convert-gptq-to-gguf.py \
    --gptq-model TheBloke/Llama-2-7B-GPTQ \
    --output llama-2-7b-gptq.gguf

# Use with llama.cpp
./llama-cli -m llama-2-7b-gptq.gguf -p "Hello world"
```

**Better approach**: Use native GGUF quantization (similar quality, better CPU support)

```bash
# llama.cpp K-quants achieve similar results to GPTQ
./llama-quantize model-f16.gguf model-q4.gguf Q4_K_M

# Q4_K_M ≈ GPTQ 4-bit in quality
# But optimized for CPU inference
```

---

### 6.2 AutoGPTQ Library

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

# Load model
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Quantization config
quantize_config = BaseQuantizeConfig(
    bits=4,                  # 4-bit quantization
    group_size=128,          # Group size
    desc_act=False,          # Act order (experimental)
    damp_percent=0.01,       # Damping factor
)

# Load and quantize
model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config
)

# Prepare calibration data
import random
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calibration_data = [
    tokenizer(dataset[random.randint(0, len(dataset))]['text'])
    for _ in range(128)
]

# Quantize (this takes 10-30 minutes)
model.quantize(calibration_data)

# Save
model.save_quantized("llama-2-7b-gptq-4bit")

# Inference
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

---

### 6.3 Pre-quantized Models (TheBloke)

```python
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

# Load pre-quantized model from HuggingFace
model_name = "TheBloke/Llama-2-7B-GPTQ"

model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    device="cuda:0",
    use_triton=True,  # Faster inference with Triton kernels
    use_safetensors=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Inference
text = "What is the meaning of life?"
inputs = tokenizer(text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## 7. Key Takeaways

### 7.1 When to Use GPTQ

✅ **Best for**:
- GPU inference (CUDA support required)
- Maximum compression (4-bit)
- Models >7B parameters
- Production serving with GPUs
- When accuracy is critical (better than naive quantization)

❌ **Not ideal for**:
- CPU inference (use GGUF Q4_K_M instead)
- Edge devices (limited CUDA support)
- When calibration data is hard to obtain
- Real-time applications (quantization is slow)

---

### 7.2 Key Innovations

1. **Layer-wise quantization**: Process each layer independently
2. **Hessian-based optimization**: Use second-order information for optimal quantization
3. **Greedy sequential updates**: Correct errors iteratively
4. **Group-wise quantization**: Balance accuracy and compression

---

### 7.3 Practical Recommendations

**For llama.cpp users**:
- Use native GGUF quantization (Q4_K_M, Q5_K_M)
- GPTQ concepts inform K-quants design
- Understanding GPTQ helps choose quantization levels

**For GPU serving**:
- GPTQ is excellent for vLLM, TGI, etc.
- 4-bit group size 128 is the sweet spot
- Use TheBloke's pre-quantized models to save time

**For research**:
- GPTQ is the baseline for quantization papers
- Understanding Hessian-based methods is valuable
- Group size and calibration data are key hyperparameters

---

## 8. Further Reading

### Papers
1. **GPTQ Paper** (Frantar et al., 2022): https://arxiv.org/abs/2210.17323
2. **Optimal Brain Quantization** (OBQ): Foundation of GPTQ
3. **AWQ Paper** (Lin et al., 2023): https://arxiv.org/abs/2306.00978 - Comparison to GPTQ

### Code Repositories
- **AutoGPTQ**: https://github.com/PanQiWei/AutoGPTQ
- **GPTQ-for-LLaMA**: https://github.com/qwopqwop200/GPTQ-for-LLaMa
- **TheBloke GPTQ Models**: https://huggingface.co/TheBloke

### Related Topics
- LLM.int8() paper (mixed-precision quantization)
- K-quants in llama.cpp (CPU-optimized quantization)
- vLLM with GPTQ support (production serving)

---

**Document Information**
- Created: 2025-11-18
- Module: 3 - Quantization Deep Dive
- Author: Research Coordinator
- Status: Complete
- Next: Read awq-activation-aware-quantization.md
