# AWQ: Activation-aware Weight Quantization

**Paper**: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
**Authors**: Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Song Han (MIT, NVIDIA)
**Published**: June 2023
**Link**: https://arxiv.org/abs/2306.00978
**Relevance**: Module 3 - Quantization Deep Dive
**Reading Time**: 35-45 minutes
**Practical Impact**: ⭐⭐⭐⭐⭐

---

## Executive Summary

AWQ introduces activation-aware weight quantization, protecting "salient" weights (those with high activation magnitudes) during quantization. By analyzing activation patterns and applying per-channel scaling, AWQ achieves superior 4-bit quantization quality compared to GPTQ while maintaining fast inference speed.

**Key Insight**: Not all weights are equally important—protect the 1% most salient weights to preserve 100% of the model's capability.

**Impact**: State-of-the-art 4-bit quantization with minimal accuracy loss, 3× faster than FP16 on GPUs.

---

## 1. Core Concept: Weight Salience

### 1.1 Observation: Activation Patterns Reveal Importance

```python
import torch

# Collect activation statistics during forward pass
def analyze_activations(model, calibration_data):
    """
    Key finding: Some channels have consistently large activations
    These correspond to important weights
    """
    activation_stats = {}

    def hook_fn(module, input, output):
        # Record activation magnitudes
        act = input[0].detach()
        activation_stats[module] = act.abs().mean(dim=0)  # [hidden_dim]

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass with calibration data
    with torch.no_grad():
        for batch in calibration_data:
            model(batch)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activation_stats

# Example: LLaMA layer activations
activations = analyze_activations(model, calibration_data)

# Observation: 1% of channels have 10-100× larger activations!
# These are "salient" weights that need protection
```

### 1.2 Salient Weight Protection

**Principle**: Quantization error δ in weight translates to output error based on activation magnitude

```
Output error = |X · δW|
             = |X| · |δW|

For same weight error δW:
- Large |X| → Large output error (CRITICAL)
- Small |X| → Small output error (acceptable)

Solution: Scale up salient weights before quantization
```

**Algorithm**:
```python
def awq_quantize(W, X, bits=4):
    """
    AWQ: Activation-aware weight quantization

    Args:
        W: Weight matrix [d_out, d_in]
        X: Calibration activations [n_samples, d_in]
        bits: Target bit-width

    Returns:
        W_quant: Quantized weights
        scales: Per-channel scaling factors
    """
    # 1. Compute activation statistics (per input channel)
    s_x = X.abs().mean(dim=0)  # [d_in]

    # 2. Identify salient channels (top 1%)
    threshold = torch.quantile(s_x, 0.99)
    salient_mask = s_x > threshold

    # 3. Compute per-channel scaling factors
    # Salient channels: scale up (reduces relative quantization error)
    # Non-salient: scale down or keep same
    alpha = 0.5  # Hyperparameter (typically 0.5)
    scales = torch.ones_like(s_x)
    scales[salient_mask] = s_x[salient_mask].pow(alpha)

    # 4. Scale weights before quantization
    W_scaled = W / scales.unsqueeze(0)  # Broadcasting

    # 5. Quantize scaled weights (standard symmetric quantization)
    n_levels = 2 ** (bits - 1)
    W_max = W_scaled.abs().max(dim=1, keepdim=True)[0]
    scale = W_max / n_levels

    W_quant_scaled = torch.clamp(
        torch.round(W_scaled / scale), -n_levels, n_levels - 1
    )
    W_quant_scaled = W_quant_scaled * scale

    # 6. Unscale for inference
    # During inference: W_quant = W_quant_scaled * scales
    # This is absorbed into input: (W_quant * scales) @ X = W_quant @ (scales * X)
    return W_quant_scaled, scales

# Inference
def awq_inference(W_quant, scales, X):
    """
    Efficient AWQ inference

    Fuse scaling into activation (no extra ops!)
    """
    X_scaled = X * scales  # Element-wise scaling
    return W_quant @ X_scaled
```

---

## 2. Search-based Scaling Factor Optimization

### 2.1 Optimal Scaling Search

**Problem**: What scaling factor α minimizes quantization error?

```python
def search_optimal_alpha(W, X, bits=4, n_trials=20):
    """
    Grid search for optimal alpha (AWQ paper uses α ∈ [0, 1])

    Objective: Minimize ||WX - Ŵ_α X||
    """
    best_alpha = 0.5
    best_error = float('inf')

    # Try different alpha values
    for alpha in torch.linspace(0, 1, n_trials):
        # Compute scaling
        s_x = X.abs().mean(dim=0)
        scales = s_x.pow(alpha)

        # Quantize with this scaling
        W_scaled = W / scales.unsqueeze(0)
        W_quant_scaled = quantize(W_scaled, bits)

        # Compute output error
        output_fp16 = W @ X
        output_quant = (W_quant_scaled * scales.unsqueeze(0)) @ X
        error = (output_fp16 - output_quant).pow(2).mean()

        if error < best_error:
            best_error = error
            best_alpha = alpha

    print(f"Optimal alpha: {best_alpha:.3f}, Error: {best_error:.6f}")
    return best_alpha

# Empirical finding: α ≈ 0.5 works well across most layers
```

### 2.2 Per-Group Scaling

**Extension**: Apply different scaling factors to groups of channels

```python
def awq_per_group_quantize(W, X, bits=4, group_size=128):
    """
    AWQ with group-wise quantization
    Combines AWQ salience protection + GPTQ group quantization
    """
    d_out, d_in = W.shape
    n_groups = (d_in + group_size - 1) // group_size

    W_quant = torch.zeros_like(W)
    scales_list = []

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, d_in)

        # Extract group
        W_group = W[:, start:end]
        X_group = X[:, start:end]

        # AWQ quantization for this group
        W_group_quant, group_scales = awq_quantize(W_group, X_group, bits)

        W_quant[:, start:end] = W_group_quant
        scales_list.append(group_scales)

    return W_quant, scales_list

# Result: Better quality than pure group quantization
```

---

## 3. Implementation Optimizations

### 3.1 Efficient Kernel Fusion

**Observation**: Scaling can be fused with matrix multiplication

```python
# Naive (slow - 2 operations)
def slow_inference(W_quant, scales, X):
    X_scaled = X * scales              # Element-wise multiply
    output = torch.matmul(W_quant, X_scaled)  # Matrix multiply
    return output

# Optimized (fast - fused kernel)
def fast_inference(W_quant, scales, X):
    """
    Fuse scaling into GEMM kernel
    No separate scaling operation!
    """
    # Pseudo-CUDA code:
    # output[i, j] = sum_k W_quant[i, k] * (X[j, k] * scales[k])
    #              = sum_k (W_quant[i, k] * scales[k]) * X[j, k]
    # Pre-scale weights offline, or scale on-the-fly in GEMM kernel
    return custom_awq_gemm(W_quant, scales, X)

# Speedup: 15-20% faster than naive approach
```

### 3.2 INT4 Packing

```python
def pack_int4_weights(W_quant):
    """
    Pack two 4-bit weights into one byte
    """
    W_int4 = W_quant.to(torch.int8)  # -7 to 7 fits in 4 bits

    # Pack two weights per byte
    W_packed = torch.zeros(W_int4.shape[0], W_int4.shape[1] // 2, dtype=torch.uint8)

    for i in range(W_int4.shape[1] // 2):
        # Low nibble: weight i*2
        # High nibble: weight i*2+1
        low = (W_int4[:, i*2] + 7).to(torch.uint8)      # Offset to 0-15
        high = (W_int4[:, i*2+1] + 7).to(torch.uint8)
        W_packed[:, i] = (high << 4) | low

    return W_packed

# Storage: 4096×4096 matrix
# FP16: 32 MB
# AWQ 4-bit packed: 8 MB (4× compression)
```

---

## 4. Results and Comparisons

### 4.1 Perplexity Comparison (WikiText-2)

| Model | FP16 | RTN 4-bit | GPTQ 4-bit | AWQ 4-bit |
|-------|------|-----------|------------|-----------|
| LLaMA-7B | 5.68 | 6.29 | 5.81 | **5.73** |
| LLaMA-13B | 5.09 | 5.63 | 5.20 | **5.12** |
| LLaMA-30B | 4.10 | 4.54 | 4.18 | **4.12** |
| LLaMA-65B | 3.53 | 3.88 | 3.61 | **3.55** |

**RTN**: Round-to-nearest (naive quantization)
**GPTQ**: Hessian-based quantization
**AWQ**: Activation-aware quantization (best!)

**Key Observation**: AWQ achieves near-FP16 quality, outperforming GPTQ

---

### 4.2 Downstream Tasks (Accuracy %)

**MMLU Benchmark**:
| Model | FP16 | GPTQ 4-bit | AWQ 4-bit |
|-------|------|------------|-----------|
| LLaMA-7B | 35.1 | 34.8 | **35.0** |
| LLaMA-13B | 46.9 | 46.5 | **46.8** |
| LLaMA-30B | 58.1 | 57.7 | **58.0** |

**Result**: <0.2% accuracy drop (vs 0.4% for GPTQ)

---

### 4.3 Inference Speed (LLaMA-7B on A100)

| Precision | Latency (ms/token) | Throughput (tokens/s) | Speedup |
|-----------|--------------------|-----------------------|---------|
| FP16 | 12.3 | 81 | 1.0× |
| GPTQ 4-bit | 5.8 | 172 | 2.1× |
| AWQ 4-bit | 4.2 | **238** | **2.9×** |

**Why faster than GPTQ?**
- Simpler dequantization (no complex Hessian corrections)
- Better kernel fusion opportunities
- Optimized CUDA kernels (TinyChat, AutoAWQ)

---

## 5. AWQ vs GPTQ: Technical Comparison

| Aspect | AWQ | GPTQ |
|--------|-----|------|
| **Core Idea** | Protect salient weights via activation-aware scaling | Minimize reconstruction error via Hessian |
| **Calibration** | Activation statistics (simpler) | Full Hessian computation (expensive) |
| **Quantization** | Per-channel scaling + standard quantization | Layer-wise with Hessian-based corrections |
| **Complexity** | O(d²) - linear passes | O(d³) - Hessian inverse |
| **Quality** | Slightly better (empirically) | Excellent |
| **Speed** | Faster (simpler operations) | Good |
| **Memory** | Same (both 4-bit) | Same |
| **Best for** | GPU inference, especially with fused kernels | General-purpose, well-established |

---

## 6. Practical Usage

### 6.1 AutoAWQ Library

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Quantization config
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Load and quantize
model = AutoAWQForCausalLM.from_pretrained(model_name)
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized("llama-2-7b-awq-4bit")

# Inference
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_quantized("llama-2-7b-awq-4bit")
inputs = tokenizer("The future of AI is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 6.2 Pre-quantized Models

```python
# Load from HuggingFace
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-AWQ",
    fuse_layers=True,  # Fuse layers for speed
    use_triton=True    # Use Triton kernels (faster)
)

# Inference is 3× faster than FP16!
```

---

## 7. AWQ in llama.cpp Context

### 7.1 Conceptual Similarities to K-quants

llama.cpp's K-quants incorporate similar ideas:

```
AWQ Concepts in K-quants:
1. Importance-based quantization (imatrix)
   - Similar to AWQ's salience protection
   - Uses activation statistics

2. Mixed-precision within groups
   - K_M, K_L variants use different precision for important weights
   - Analogous to AWQ's scaling

3. Per-channel scaling
   - Both use per-channel quantization parameters
   - Optimizes for activation patterns
```

### 7.2 Using Importance Matrix (imatrix) in llama.cpp

```bash
# Generate importance matrix (similar to AWQ activation statistics)
./llama-imatrix \
  -m model-f16.gguf \
  -f calibration.txt \
  -o model.imatrix \
  --chunks 100

# Quantize with importance matrix
./llama-quantize \
  --imatrix model.imatrix \
  model-f16.gguf \
  model-q4_k_m.gguf \
  Q4_K_M

# Result: Better quality than without imatrix (AWQ-inspired!)
```

---

## 8. Key Takeaways

### 8.1 Core Insights

✅ **Weight Importance**: Not all weights are equal—protect the 1% most important
✅ **Activation-Aware**: Activation magnitude predicts weight importance
✅ **Simple but Effective**: Per-channel scaling is simpler than Hessian methods
✅ **Fast Inference**: Fused kernels make AWQ faster than GPTQ

### 8.2 When to Use AWQ

**Best for**:
- GPU inference (requires CUDA)
- Maximum quality at 4-bit
- Fast inference is critical
- Modern serving systems (vLLM supports AWQ)

**Not ideal for**:
- CPU inference (use GGUF K-quants instead)
- When GPTQ ecosystem is more mature for your use case

### 8.3 For llama.cpp Users

**Lessons from AWQ**:
1. Use importance matrix (imatrix) for better quantization
2. Collect diverse calibration data (activations vary by task)
3. Higher quantization for important tensors (K_L vs K_S)

---

## 9. Further Reading

**Papers**:
- AWQ Paper: https://arxiv.org/abs/2306.00978
- GPTQ Comparison: See GPTQ paper summary
- SmoothQuant: Related activation smoothing technique

**Code**:
- AutoAWQ: https://github.com/casper-hansen/AutoAWQ
- TinyChat (AWQ inference): https://github.com/mit-han-lab/llm-awq
- TheBloke AWQ Models: https://huggingface.co/TheBloke

**Related**:
- vLLM with AWQ: Production serving
- llama.cpp imatrix: Importance-based quantization

---

**Document Information**
- Created: 2025-11-18
- Module: 3 - Quantization Deep Dive
- Author: Research Coordinator
- Status: Complete
- Next: Read llm-int8-paper.md
