# Lab 7.3: Custom Architecture Integration

**Estimated Time**: 120+ minutes
**Difficulty**: Expert
**Prerequisites**: Strong C++ knowledge, Modules 1-4 complete

## Learning Objectives

- Understand llama.cpp architecture system
- Add support for a custom model architecture
- Implement custom layers and operations
- Test and validate the implementation

## Overview

In this advanced lab, you'll add support for a custom architecture variant to llama.cpp. We'll implement a simplified "MyLLaMA" architecture with a custom attention mechanism as a learning exercise.

---

## Part 1: Understanding the Architecture System (30 minutes)

### Task 1: Explore Existing Architectures

```bash
cd llama.cpp
grep -n "LLM_ARCH_" src/llama.cpp | head -20
```

Study how Mistral is implemented:
```bash
# Find Mistral-specific code
grep -A 10 "LLM_ARCH_MISTRAL" src/llama.cpp
```

**Questions**:
1. Where are architectures registered?
2. How are tensor names mapped?
3. Where is the forward pass implemented?

### Task 2: Analyze Model Structure

Find and study:
- `enum llm_arch` - Architecture enumeration
- `LLM_TENSOR_NAMES` - Tensor name mappings
- `llm_load_arch` - Model loading
- `llm_build_*` - Forward pass functions

**Create a diagram** showing the data flow from GGUF file to inference output.

---

## Part 2: Define Custom Architecture (30 minutes)

### Task 3: Register Architecture

Edit `src/llama.cpp`:

**Step 1**: Add to enum (around line 150):
```cpp
enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_MISTRAL,
    // ... existing architectures ...
    LLM_ARCH_MYLLAMA,  // ADD THIS
    LLM_ARCH_UNKNOWN,
};
```

**Step 2**: Add name mapping (around line 200):
```cpp
static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,    "llama"    },
    // ...
    { LLM_ARCH_MYLLAMA,  "myllama"  },  // ADD THIS
};
```

**Step 3**: Define tensor names (around line 300):
```cpp
static const std::map<llm_tensor, const char*> LLM_TENSOR_NAMES_MYLLAMA = {
    { LLM_TENSOR_TOKEN_EMBD,     "token_embd" },
    { LLM_TENSOR_OUTPUT_NORM,    "output_norm" },
    { LLM_TENSOR_OUTPUT,         "output" },

    // Per-layer tensors
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
```

**Step 4**: Register tensor names (around line 500):
```cpp
// In LLM_TENSOR_NAMES initialization
{ LLM_ARCH_MYLLAMA, LLM_TENSOR_NAMES_MYLLAMA },
```

**Verify**: Recompile and check for errors
```bash
make clean
make -j$(nproc)
```

---

## Part 3: Implement Model Loading (30 minutes)

### Task 4: Hyperparameter Loading

Add metadata keys (around line 800):
```cpp
static const std::map<llm_kv, const char*> LLM_KV_MYLLAMA = {
    { LLM_KV_GENERAL_ARCHITECTURE,        "general.architecture"        },
    { LLM_KV_CONTEXT_LENGTH,              "myllama.context_length"      },
    { LLM_KV_EMBEDDING_LENGTH,            "myllama.embedding_length"    },
    { LLM_KV_BLOCK_COUNT,                 "myllama.block_count"         },
    { LLM_KV_FEED_FORWARD_LENGTH,         "myllama.feed_forward_length" },
    { LLM_KV_ATTENTION_HEAD_COUNT,        "myllama.attention.head_count"},
    { LLM_KV_ATTENTION_HEAD_COUNT_KV,     "myllama.attention.head_count_kv"},
    { LLM_KV_ROPE_DIMENSION_COUNT,        "myllama.rope.dimension_count"},
    { LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, "myllama.attention.layer_norm_rms_epsilon"},
};
```

Register KV names:
```cpp
{ LLM_ARCH_MYLLAMA, LLM_KV_MYLLAMA },
```

### Task 5: Load Function

Add loading function (around line 3000):
```cpp
static void llm_load_hparams_myllama(
    llama_model_loader & ml,
    llama_model & model
) {
    auto & hparams = model.hparams;
    const gguf_context * ctx = ml.ctx_gguf;

    // Load hyperparameters
    GGUF_GET_KEY(ctx, hparams.n_vocab,    gguf_get_val_u32, GGUF_TYPE_UINT32, true);
    GGUF_GET_KEY(ctx, hparams.n_embd,     gguf_get_val_u32, GGUF_TYPE_UINT32, true);
    GGUF_GET_KEY(ctx, hparams.n_layer,    gguf_get_val_u32, GGUF_TYPE_UINT32, true);
    GGUF_GET_KEY(ctx, hparams.n_head,     gguf_get_val_u32, GGUF_TYPE_UINT32, true);
    GGUF_GET_KEY(ctx, hparams.n_ff,       gguf_get_val_u32, GGUF_TYPE_UINT32, true);

    // Optional parameters
    hparams.n_head_kv = hparams.n_head;  // Default: same as n_head
    GGUF_GET_KEY(ctx, hparams.n_head_kv, gguf_get_val_u32, GGUF_TYPE_UINT32, false);

    // Derived parameters
    hparams.n_embd_head = hparams.n_embd / hparams.n_head;
    hparams.n_embd_gqa  = hparams.n_embd / hparams.n_head_kv;

    // Validate
    GGML_ASSERT(hparams.n_embd % hparams.n_head == 0);
    GGML_ASSERT(hparams.n_head % hparams.n_head_kv == 0);
}
```

Add to switch statement in `llm_load_arch`:
```cpp
case LLM_ARCH_MYLLAMA:
    llm_load_hparams_myllama(ml, model);
    break;
```

---

## Part 4: Implement Forward Pass (45 minutes)

### Task 6: Build Graph Function

Add forward pass implementation (around line 10000):
```cpp
static struct ggml_cgraph * llm_build_myllama(
    llama_context & lctx,
    const llama_batch & batch
) {
    const auto & model = lctx.model;
    const auto & hparams = model.hparams;

    const int64_t n_embd     = hparams.n_embd;
    const int64_t n_layer    = hparams.n_layer;
    const int64_t n_head     = hparams.n_head;
    const int64_t n_head_kv  = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head;
    const int64_t n_ff       = hparams.n_ff;

    auto & buf_compute = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    const int n_tokens = batch.n_tokens;
    const int n_past   = lctx.kv_self.n;

    // Input: token embeddings
    struct ggml_tensor * inpL = ggml_get_rows(ctx0,
        model.tok_embd,
        // Token IDs tensor
        ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens)
    );

    // Process through layers
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

        // --- Self-Attention ---

        // Norm
        cur = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
        cur = ggml_mul(ctx0, cur, model.layers[il].attn_norm);

        // Q, K, V projections
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);

        // Reshape for attention
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

        // Apply RoPE
        Qcur = ggml_rope_custom(ctx0, Qcur, /* RoPE params */);
        Kcur = ggml_rope_custom(ctx0, Kcur, /* RoPE params */);

        // Store K, V in cache
        // ... KV cache logic ...

        // Attention computation
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, 1.0f / sqrtf(n_embd_head));
        KQ = ggml_soft_max(ctx0, KQ);

        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        cur = ggml_mul_mat(ctx0, model.layers[il].wo, KQV);

        // Residual connection
        cur = ggml_add(ctx0, cur, inpL);

        // --- Feed-Forward Network ---

        struct ggml_tensor * ffn_inp = cur;

        // Norm
        cur = ggml_rms_norm(ctx0, ffn_inp, hparams.f_norm_rms_eps);
        cur = ggml_mul(ctx0, cur, model.layers[il].ffn_norm);

        // SwiGLU
        struct ggml_tensor * tmp = ggml_mul_mat(ctx0, model.layers[il].ffn_gate, cur);
        tmp = ggml_silu(ctx0, tmp);

        cur = ggml_mul_mat(ctx0, model.layers[il].ffn_up, cur);
        cur = ggml_mul(ctx0, cur, tmp);
        cur = ggml_mul_mat(ctx0, model.layers[il].ffn_down, cur);

        // Residual
        cur = ggml_add(ctx0, cur, ffn_inp);

        // Next layer input
        inpL = cur;
    }

    // Output layer
    inpL = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
    inpL = ggml_mul(ctx0, inpL, model.output_norm);

    // Project to vocabulary
    inpL = ggml_mul_mat(ctx0, model.output, inpL);

    // Build graph
    ggml_build_forward_expand(gf, inpL);

    ggml_free(ctx0);
    return gf;
}
```

Register in build switch (around line 12000):
```cpp
case LLM_ARCH_MYLLAMA:
    return llm_build_myllama(lctx, batch);
```

---

## Part 5: Testing (30 minutes)

### Task 7: Create Test Model

Convert a small LLaMA model and modify metadata:

```python
# modify_arch.py
import gguf

# Load existing model
reader = gguf.GGUFReader("tiny-llama.gguf")

# Create writer
writer = gguf.GGUFWriter("tiny-myllama.gguf", "myllama")

# Copy all tensors
for tensor in reader.tensors:
    writer.add_tensor(tensor.name, tensor.data, tensor.tensor_type)

# Set architecture
writer.add_string("general.architecture", "myllama")

# Copy other metadata, changing keys
for key, value in reader.metadata.items():
    if "llama." in key:
        new_key = key.replace("llama.", "myllama.")
        writer.add_value(new_key, value)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
```

### Task 8: Run Inference

Test your implementation:
```bash
./main -m tiny-myllama.gguf -p "Hello, world!" -n 50
```

**Debug checklist**:
- [ ] Model loads without errors
- [ ] Hyperparameters read correctly
- [ ] All tensors found
- [ ] Forward pass executes
- [ ] Output is coherent

---

## Deliverables

1. **Code Changes**:
   - Modified `src/llama.cpp` with MyLLaMA implementation
   - Git diff showing all changes

2. **Test Results**:
   - Successful compilation
   - Model loading output
   - Inference examples

3. **Documentation**:
   - Architecture design document
   - Implementation notes
   - Comparison with LLaMA base

4. **Validation**:
   - Output matches reference (if applicable)
   - Performance benchmarks

---

## Bonus Challenges

### Challenge 1: Custom Attention

Implement a novel attention mechanism:
- Sliding window attention (like Mistral)
- Local + global attention
- Sparse attention patterns

### Challenge 2: Mixture of Experts

Add MoE support:
- Expert routing layer
- Top-k expert selection
- Load balancing loss

### Challenge 3: Architecture-Specific Optimization

Optimize for your architecture:
- Custom CUDA kernels
- Operator fusion
- Memory layout optimization

---

## Reflection

1. What are the challenges in adding new architectures?
2. How does the architecture affect performance?
3. What testing is needed for production use?
4. How would you contribute this to llama.cpp?

---

**Lab Complete!** You've added a custom architecture to llama.cpp - an advanced achievement!
