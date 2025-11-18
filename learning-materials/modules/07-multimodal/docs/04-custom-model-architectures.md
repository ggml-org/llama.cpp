# Adding Custom Model Architectures to LLaMA.cpp

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture System Overview](#architecture-system-overview)
3. [Understanding Existing Architectures](#understanding-existing-architectures)
4. [Adding a New Architecture](#adding-a-new-architecture)
5. [Implementing Custom Layers](#implementing-custom-layers)
6. [Testing and Validation](#testing-and-validation)
7. [Performance Optimization](#performance-optimization)
8. [Case Studies](#case-studies)

---

## Introduction

llama.cpp's architecture system allows adding support for new model types beyond the original LLaMA. This enables running cutting-edge models like Mistral, Mixtral, Qwen, and custom architectures.

### Why Add Custom Architectures?

1. **New Model Support**: Run latest research models
2. **Proprietary Models**: Deploy custom company models
3. **Research**: Experiment with novel architectures
4. **Optimization**: Architecture-specific optimizations
5. **Community**: Contribute to llama.cpp ecosystem

### Architecture Examples in llama.cpp

| Architecture | Key Features | Use Case |
|--------------|--------------|----------|
| LLaMA | Standard transformer | Base model |
| Mistral | Sliding window attention | Long context |
| Mixtral | Mixture of Experts (MoE) | Efficient scaling |
| Qwen | Multilingual, grouped query attention | Chinese/multilingual |
| Phi | Small, efficient | Edge deployment |
| Falcon | Multi-query attention | Fast inference |
| GPT-2 | Classic transformer | Compatibility |

---

## Architecture System Overview

### Key Components

```cpp
// src/llama.cpp

// 1. Model Architecture Enumeration
enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_MISTRAL,
    LLM_ARCH_MIXTRAL,
    LLM_ARCH_QWEN,
    LLM_ARCH_PHI2,
    // ... add your architecture here
    LLM_ARCH_UNKNOWN,
};

// 2. Architecture Definition
struct llm_arch_def {
    const char *name;                    // Architecture name
    std::map<llm_tensor, const char*> tensor_names;  // Tensor mappings
    std::map<llm_kv, const char*> kv_names;          // Metadata keys
};

// 3. Model Loading
static void llm_load_arch(llama_model & model, llama_model_loader & ml);

// 4. Model Building (computation graph)
static struct ggml_cgraph * llm_build_llama(
    llama_context & lctx,
    const llama_batch & batch
);
```

### Architecture Definition Flow

```
1. Register Architecture
   ↓
2. Define Tensor Names
   ↓
3. Define Hyperparameters
   ↓
4. Implement Model Loader
   ↓
5. Implement Forward Pass
   ↓
6. Add Tokenizer Support
   ↓
7. Test and Validate
```

---

## Understanding Existing Architectures

### LLaMA Architecture (Base)

```cpp
// Standard LLaMA architecture
static const std::map<llm_tensor, const char*> LLM_TENSOR_NAMES_LLAMA = {
    { LLM_TENSOR_TOKEN_EMBD,     "token_embd" },
    { LLM_TENSOR_OUTPUT_NORM,    "output_norm" },
    { LLM_TENSOR_OUTPUT,         "output" },
    { LLM_TENSOR_ROPE_FREQS,     "rope_freqs" },
    { LLM_TENSOR_ATTN_NORM,      "blk.%d.attn_norm" },
    { LLM_TENSOR_ATTN_Q,         "blk.%d.attn_q" },
    { LLM_TENSOR_ATTN_K,         "blk.%d.attn_k" },
    { LLM_TENSOR_ATTN_V,         "blk.%d.attn_v" },
    { LLM_TENSOR_ATTN_OUT,       "blk.%d.attn_output" },
    { LLM_TENSOR_FFN_NORM,       "blk.%d.ffn_norm" },
    { LLM_TENSOR_FFN_GATE,       "blk.%d.ffn_gate" },
    { LLM_TENSOR_FFN_DOWN,       "blk.%d.ffn_down" },
    { LLM_TENSOR_FFN_UP,         "blk.%d.ffn_up" },
};

// Hyperparameters
struct llama_hparams {
    uint32_t n_vocab = 32000;     // Vocabulary size
    uint32_t n_ctx   = 512;       // Context length
    uint32_t n_embd  = 4096;      // Embedding dimension
    uint32_t n_head  = 32;        // Attention heads
    uint32_t n_head_kv = 32;      // KV heads (for GQA)
    uint32_t n_layer = 32;        // Number of layers
    uint32_t n_ff    = 11008;     // FFN intermediate size

    float f_norm_rms_eps = 1e-5;  // RMSNorm epsilon
    float rope_freq_base = 10000.0;
    float rope_freq_scale = 1.0;
};
```

### Mistral Architecture (Sliding Window)

```cpp
// Mistral adds sliding window attention
static const std::map<llm_kv, const char*> LLM_KV_MISTRAL = {
    // ... standard keys ...
    { LLM_KV_ATTENTION_WINDOW, "mistral.attention_window" },
};

// In forward pass:
if (model.arch == LLM_ARCH_MISTRAL) {
    // Apply sliding window mask
    Qcur = ggml_rope_custom(
        ctx,
        ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head, n_tokens),
        inp_pos,
        n_rot,
        rope_type,
        0,
        n_orig_ctx,
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        beta_fast,
        beta_slow
    );

    // Apply sliding window in attention
    if (n_past >= sliding_window) {
        // Mask out tokens outside window
    }
}
```

### Mixtral Architecture (MoE)

```cpp
// Mixtral uses Mixture of Experts
struct mixtral_hparams : llama_hparams {
    uint32_t n_expert = 8;           // Number of experts
    uint32_t n_expert_used = 2;      // Active experts per token
};

// Expert tensors
{ LLM_TENSOR_FFN_GATE_EXP,  "blk.%d.ffn_gate.%d" },
{ LLM_TENSOR_FFN_DOWN_EXP,  "blk.%d.ffn_down.%d" },
{ LLM_TENSOR_FFN_UP_EXP,    "blk.%d.ffn_up.%d" },
{ LLM_TENSOR_FFN_GATE_INP,  "blk.%d.ffn_gate_inp" },  // Expert router

// Forward pass with expert routing
ggml_tensor * cur = ...;  // Input to FFN

// Route to experts
ggml_tensor * logits = ggml_mul_mat(ctx, model.layers[il].ffn_gate_inp, cur);
ggml_tensor * probs = ggml_soft_max(ctx, logits);

// Select top-k experts
// ... expert selection logic ...

// Combine expert outputs
ggml_tensor * expert_outputs[n_expert];
for (int i = 0; i < n_expert_used; i++) {
    int expert_id = selected_experts[i];
    expert_outputs[i] = apply_expert(cur, expert_id);
}

cur = weighted_sum(expert_outputs, expert_weights);
```

---

## Adding a New Architecture

### Step 1: Define Architecture Enum

```cpp
// src/llama.cpp

enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_MISTRAL,
    // ... existing architectures ...

    // Add your architecture
    LLM_ARCH_MYMODEL,

    LLM_ARCH_UNKNOWN,
};

// Add architecture name mapping
static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,    "llama"    },
    { LLM_ARCH_MISTRAL,  "mistral"  },
    // ...
    { LLM_ARCH_MYMODEL,  "mymodel"  },
};
```

### Step 2: Define Tensor Names

```cpp
// Define tensor name mappings for your architecture
static const std::map<llm_tensor, const char*> LLM_TENSOR_NAMES_MYMODEL = {
    // Token embeddings
    { LLM_TENSOR_TOKEN_EMBD,     "token_embd" },

    // Output layer
    { LLM_TENSOR_OUTPUT_NORM,    "output_norm" },
    { LLM_TENSOR_OUTPUT,         "output" },

    // Per-layer tensors (use %d for layer index)
    { LLM_TENSOR_ATTN_NORM,      "blk.%d.attn_norm" },
    { LLM_TENSOR_ATTN_Q,         "blk.%d.attn_q" },
    { LLM_TENSOR_ATTN_K,         "blk.%d.attn_k" },
    { LLM_TENSOR_ATTN_V,         "blk.%d.attn_v" },
    { LLM_TENSOR_ATTN_OUT,       "blk.%d.attn_output" },

    // Feed-forward
    { LLM_TENSOR_FFN_NORM,       "blk.%d.ffn_norm" },
    { LLM_TENSOR_FFN_GATE,       "blk.%d.ffn_gate" },
    { LLM_TENSOR_FFN_DOWN,       "blk.%d.ffn_down" },
    { LLM_TENSOR_FFN_UP,         "blk.%d.ffn_up" },

    // Custom tensors for your architecture
    { LLM_TENSOR_CUSTOM_1,       "blk.%d.custom_layer" },
};

// Register in the tensor name map
LLM_TENSOR_NAMES[LLM_ARCH_MYMODEL] = LLM_TENSOR_NAMES_MYMODEL;
```

### Step 3: Define Hyperparameters

```cpp
// Add architecture-specific metadata keys
static const std::map<llm_kv, const char*> LLM_KV_MYMODEL = {
    { LLM_KV_GENERAL_ARCHITECTURE,        "general.architecture"        },
    { LLM_KV_GENERAL_NAME,                "general.name"                },

    { LLM_KV_CONTEXT_LENGTH,              "mymodel.context_length"      },
    { LLM_KV_EMBEDDING_LENGTH,            "mymodel.embedding_length"    },
    { LLM_KV_BLOCK_COUNT,                 "mymodel.block_count"         },
    { LLM_KV_FEED_FORWARD_LENGTH,         "mymodel.feed_forward_length" },
    { LLM_KV_ATTENTION_HEAD_COUNT,        "mymodel.attention.head_count"},
    { LLM_KV_ATTENTION_HEAD_COUNT_KV,     "mymodel.attention.head_count_kv"},
    { LLM_KV_ROPE_DIMENSION_COUNT,        "mymodel.rope.dimension_count"},

    // Custom hyperparameters
    { LLM_KV_CUSTOM_PARAM_1,              "mymodel.custom_param_1"      },
};

// Register KV names
LLM_KV_NAMES[LLM_ARCH_MYMODEL] = LLM_KV_MYMODEL;
```

### Step 4: Implement Model Loader

```cpp
// src/llama.cpp

static void llm_load_hparams_mymodel(
    llama_model_loader & ml,
    llama_model & model
) {
    auto & hparams = model.hparams;
    const gguf_context * ctx = ml.ctx_gguf;

    // Read hyperparameters from GGUF
    GGUF_GET_KEY(ctx, hparams.n_vocab,    gguf_get_val_u32,  GGUF_TYPE_UINT32,  true);
    GGUF_GET_KEY(ctx, hparams.n_embd,     gguf_get_val_u32,  GGUF_TYPE_UINT32,  true);
    GGUF_GET_KEY(ctx, hparams.n_layer,    gguf_get_val_u32,  GGUF_TYPE_UINT32,  true);
    GGUF_GET_KEY(ctx, hparams.n_head,     gguf_get_val_u32,  GGUF_TYPE_UINT32,  true);

    // Read custom parameters
    GGUF_GET_KEY(ctx, hparams.custom_param_1, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false);

    // Set defaults for optional parameters
    if (hparams.n_head_kv == 0) {
        hparams.n_head_kv = hparams.n_head;
    }

    // Calculate derived parameters
    hparams.n_embd_head = hparams.n_embd / hparams.n_head;
    hparams.n_embd_gqa  = hparams.n_embd / hparams.n_head_kv;

    // Validate parameters
    GGML_ASSERT(hparams.n_embd % hparams.n_head == 0);
}

// Add to architecture loading switch
static void llm_load_arch(llama_model & model, llama_model_loader & ml) {
    // ...
    switch (model.arch) {
        case LLM_ARCH_LLAMA:
            llm_load_hparams_llama(ml, model);
            break;
        // ...
        case LLM_ARCH_MYMODEL:
            llm_load_hparams_mymodel(ml, model);
            break;
        default:
            throw std::runtime_error("unknown architecture");
    }
}
```

### Step 5: Implement Forward Pass

```cpp
static struct ggml_cgraph * llm_build_mymodel(
    llama_context & lctx,
    const llama_batch & batch
) {
    const auto & model = lctx.model;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    const int64_t n_embd     = hparams.n_embd;
    const int64_t n_layer    = hparams.n_layer;
    const int64_t n_head     = hparams.n_head;
    const int64_t n_head_kv  = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head;

    auto & buf_compute = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // Input tokens
    struct ggml_tensor * inpL = ggml_get_rows(ctx0,
        model.tok_embd,
        ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch.n_tokens)
    );

    // Process through layers
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

        // Attention norm
        cur = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
        cur = ggml_mul(ctx0, cur, model.layers[il].attn_norm);

        // Self-attention
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);

        // Apply rotary embeddings
        Qcur = ggml_rope_custom(ctx0, Qcur, /* ... */);
        Kcur = ggml_rope_custom(ctx0, Kcur, /* ... */);

        // Attention computation
        struct ggml_tensor * attn = ggml_flash_attn(ctx0, Qcur, Kcur, Vcur, /* ... */);

        // Output projection
        cur = ggml_mul_mat(ctx0, model.layers[il].wo, attn);

        // Residual connection
        cur = ggml_add(ctx0, cur, inpL);

        // FFN
        struct ggml_tensor * ffn_inp = cur;

        // FFN norm
        cur = ggml_rms_norm(ctx0, ffn_inp, hparams.f_norm_rms_eps);
        cur = ggml_mul(ctx0, cur, model.layers[il].ffn_norm);

        // FFN computation (SwiGLU)
        struct ggml_tensor * tmp = ggml_mul_mat(ctx0, model.layers[il].ffn_gate, cur);
        tmp = ggml_silu(ctx0, tmp);

        cur = ggml_mul_mat(ctx0, model.layers[il].ffn_up, cur);
        cur = ggml_mul(ctx0, cur, tmp);
        cur = ggml_mul_mat(ctx0, model.layers[il].ffn_down, cur);

        // Residual
        cur = ggml_add(ctx0, cur, ffn_inp);

        // Custom layer (if applicable)
        if (model.layers[il].custom_layer) {
            cur = apply_custom_layer(ctx0, cur, model.layers[il]);
        }

        // Update input for next layer
        inpL = cur;
    }

    // Output layer
    inpL = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
    inpL = ggml_mul(ctx0, inpL, model.output_norm);

    // Final projection to vocabulary
    inpL = ggml_mul_mat(ctx0, model.output, inpL);

    // Build forward graph
    ggml_build_forward_expand(gf, inpL);

    return gf;
}

// Register in build switch
static struct ggml_cgraph * llama_build_graph(
    llama_context & lctx,
    const llama_batch & batch
) {
    switch (lctx.model.arch) {
        case LLM_ARCH_LLAMA:
            return llm_build_llama(lctx, batch);
        // ...
        case LLM_ARCH_MYMODEL:
            return llm_build_mymodel(lctx, batch);
        default:
            GGML_ASSERT(false);
    }
}
```

---

## Implementing Custom Layers

### Example: Custom Attention Mechanism

```cpp
// Implement a custom attention variant
static struct ggml_tensor * custom_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    const llama_layer & layer,
    const llama_hparams & hparams
) {
    const int64_t n_embd = hparams.n_embd;
    const int64_t n_head = hparams.n_head;

    // Custom Q, K, V projections
    struct ggml_tensor * Q = ggml_mul_mat(ctx, layer.custom_wq, input);
    struct ggml_tensor * K = ggml_mul_mat(ctx, layer.custom_wk, input);
    struct ggml_tensor * V = ggml_mul_mat(ctx, layer.custom_wv, input);

    // Apply custom transformation
    Q = custom_transform(ctx, Q);

    // Compute attention with custom mask
    struct ggml_tensor * attn_weights = ggml_mul_mat(ctx, K, Q);
    attn_weights = ggml_scale(ctx, attn_weights, 1.0f / sqrtf(n_embd / n_head));

    // Custom attention pattern (e.g., local + global)
    attn_weights = apply_custom_mask(ctx, attn_weights);

    attn_weights = ggml_soft_max(ctx, attn_weights);

    // Attention output
    struct ggml_tensor * output = ggml_mul_mat(ctx, attn_weights, V);

    return output;
}
```

### Example: Normalization Variants

```cpp
// LayerNorm (vs RMSNorm in LLaMA)
static struct ggml_tensor * layer_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * gamma,
    struct ggml_tensor * beta,
    float eps
) {
    // Compute mean
    struct ggml_tensor * mean = ggml_mean(ctx, input);

    // Compute variance
    struct ggml_tensor * variance = ggml_sqr(ctx, ggml_sub(ctx, input, mean));
    variance = ggml_mean(ctx, variance);

    // Normalize
    struct ggml_tensor * norm = ggml_div(ctx,
        ggml_sub(ctx, input, mean),
        ggml_sqrt(ctx, ggml_add(ctx, variance, ggml_new_f32(ctx, eps)))
    );

    // Scale and shift
    norm = ggml_mul(ctx, norm, gamma);
    norm = ggml_add(ctx, norm, beta);

    return norm;
}
```

---

## Testing and Validation

### Step 1: Unit Tests

```python
# tests/test_mymodel.py

import pytest
from llama_cpp import Llama
import numpy as np

def test_model_loading():
    """Test that model loads correctly"""
    model = Llama(model_path="mymodel.gguf")
    assert model is not None

def test_inference():
    """Test basic inference"""
    model = Llama(model_path="mymodel.gguf", n_ctx=512)

    output = model.create_completion(
        "Hello, how are you?",
        max_tokens=50
    )

    assert len(output['choices']) > 0
    assert len(output['choices'][0]['text']) > 0

def test_hyperparameters():
    """Verify hyperparameters loaded correctly"""
    model = Llama(model_path="mymodel.gguf")

    # Check architecture
    # Note: This requires exposing metadata through Python API
    metadata = model.metadata

    assert metadata['general.architecture'] == 'mymodel'
    assert metadata['mymodel.block_count'] > 0

def test_output_consistency():
    """Test that outputs are consistent"""
    model = Llama(model_path="mymodel.gguf", seed=42)

    output1 = model.create_completion("Test", max_tokens=10, temperature=0)
    output2 = model.create_completion("Test", max_tokens=10, temperature=0)

    assert output1['choices'][0]['text'] == output2['choices'][0]['text']
```

### Step 2: Comparison with Reference

```python
def test_against_reference():
    """
    Compare outputs with reference implementation (e.g., PyTorch)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load reference model
    ref_model = AutoModelForCausalLM.from_pretrained("mymodel-hf")
    tokenizer = AutoTokenizer.from_pretrained("mymodel-hf")

    # Load llama.cpp model
    llama_model = Llama(model_path="mymodel.gguf")

    # Test prompts
    prompts = [
        "Hello world",
        "The capital of France is",
        "In the year 2024,"
    ]

    for prompt in prompts:
        # Reference output
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            ref_logits = ref_model(**inputs).logits[0, -1, :]
        ref_top_tokens = torch.topk(ref_logits, k=10).indices.tolist()

        # llama.cpp output (need to get logits, not text)
        # This requires extending the API to return logits
        llama_logits = llama_model.get_logits(prompt)
        llama_top_tokens = np.argsort(llama_logits)[-10:][::-1]

        # Check that top tokens overlap significantly
        overlap = len(set(ref_top_tokens) & set(llama_top_tokens))
        assert overlap >= 7, f"Insufficient overlap: {overlap}/10"
```

### Step 3: Performance Benchmarking

```python
import time

def benchmark_model():
    """Benchmark inference performance"""
    model = Llama(
        model_path="mymodel.gguf",
        n_ctx=2048,
        n_gpu_layers=32
    )

    prompt = "Write a story about " * 10  # ~100 tokens

    # Warmup
    for _ in range(3):
        model.create_completion(prompt, max_tokens=10)

    # Benchmark
    latencies = []
    token_counts = []

    for _ in range(20):
        start = time.time()
        output = model.create_completion(prompt, max_tokens=100)
        latency = time.time() - start

        latencies.append(latency)
        token_counts.append(output['usage']['completion_tokens'])

    # Report
    print(f"Average latency: {np.mean(latencies):.3f}s")
    print(f"Average tokens/sec: {np.mean(token_counts) / np.mean(latencies):.1f}")
    print(f"P95 latency: {np.percentile(latencies, 95):.3f}s")
```

---

## Performance Optimization

### Operator Fusion

```cpp
// Fuse operations for better performance
static struct ggml_tensor * fused_norm_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    const llama_layer & layer
) {
    // Fuse RMSNorm + Q/K/V projection
    // Instead of: norm -> mul -> Q/K/V
    // Do: fused_norm_qkv

    struct ggml_tensor * cur = input;

    // Custom GGML operation that fuses norm + projection
    cur = ggml_rms_norm_mul_mat(ctx, cur,
                                layer.attn_norm,
                                layer.wq);  // Fused!

    return cur;
}
```

### Custom CUDA Kernels

```cpp
// src/ggml-cuda/custom-ops.cu

__global__ void custom_attention_kernel(
    const float * Q,
    const float * K,
    const float * V,
    float * output,
    int n_head,
    int n_embd_head,
    int n_tokens
) {
    // Implement custom attention pattern optimized for your architecture
    // ...
}

// Register custom kernel
void ggml_cuda_op_custom_attention(
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst
) {
    // Launch custom kernel
    const dim3 block_dims(32, 4, 1);
    const dim3 grid_dims(/*...*/);

    custom_attention_kernel<<<grid_dims, block_dims>>>(
        (const float *)src0->data,
        (const float *)src1->data,
        /*...*/
    );
}
```

---

## Case Studies

### Case Study 1: Adding Mistral

Mistral's main novelty: **Sliding Window Attention**

```cpp
// Key differences from LLaMA:
1. Sliding window size (default: 4096)
2. Rolling buffer KV cache
3. Grouped query attention

// Implementation:
static struct ggml_cgraph * llm_build_mistral(...) {
    // Same as LLaMA, but with sliding window mask

    // In attention computation:
    int32_t sliding_window = hparams.n_sliding_window;

    // Create mask that only attends to last N tokens
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                   n_past + n_tokens, n_tokens);

    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < n_past + n_tokens; j++) {
            int distance = (n_past + i) - j;
            float value = (distance <= sliding_window) ? 0.0f : -INFINITY;
            ggml_set_f32_1d(mask, i * (n_past + n_tokens) + j, value);
        }
    }

    attn = ggml_soft_max(ctx, ggml_add(ctx, attn, mask));
    // ...
}
```

### Case Study 2: Adding Phi-2

Phi-2's key features: **Parallel attention and FFN, CodeGen tokenizer**

```cpp
// Parallel blocks instead of sequential
static struct ggml_cgraph * llm_build_phi2(...) {
    for (int il = 0; il < n_layer; ++il) {
        // Both attention and FFN use same input (parallel)
        struct ggml_tensor * attn_inp = inpL;
        struct ggml_tensor * ffn_inp = inpL;

        // Attention branch
        struct ggml_tensor * attn_out = build_attention(ctx, attn_inp, ...);

        // FFN branch (parallel, not sequential)
        struct ggml_tensor * ffn_out = build_ffn(ctx, ffn_inp, ...);

        // Combine outputs
        inpL = ggml_add(ctx, attn_out, ffn_out);
    }
}
```

---

## Summary

Adding custom architectures to llama.cpp:

✅ **System**: Modular architecture registration system
✅ **Components**: Tensor definitions, hyperparameters, forward pass
✅ **Testing**: Unit tests, reference comparison, benchmarking
✅ **Optimization**: Operator fusion, custom kernels
✅ **Examples**: Mistral, Mixtral, Phi-2 case studies

**Next Steps**:
- Study existing architecture implementations
- Complete Lab 7.3 on custom architectures
- Implement a simple architecture variant
- Explore Lesson 7.5 on model conversion

---

**References**:
- llama.cpp architecture guide: https://github.com/ggerganov/llama.cpp/wiki/Adding-a-new-model
- Mistral paper: "Mistral 7B" (Jiang et al., 2023)
- Mixtral paper: "Mixtral of Experts" (Jiang et al., 2024)
- GGML documentation: https://github.com/ggerganov/ggml
