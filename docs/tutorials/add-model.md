# Guide: Adding new model architectures to llama.cpp

This guide walks through the steps required to add support for a new model architecture in llama.cpp. For the full technical reference, see [HOWTO-add-model.md](../development/HOWTO-add-model.md).

## Overview

Adding a new model architecture involves 4 steps:

1. Write a conversion script (Python) to convert the model to GGUF format
2. Define the model architecture metadata in C/C++
3. Implement the GGML inference graph
4. Optionally add multimodal encoder support

## Step 1: Convert the model to GGUF

The GGUF format is llama.cpp's model storage format. Conversion is done in Python.

### Create a model class

In one of the files under the `conversion/` directory, register a new model class:

```python
from gguf.model import TextModel, ModelBase

@ModelBase.register("MyModelForCausalLM")
class MyModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MYMODEL
```

### Define GGUF tensor layout

In `gguf-py/gguf/constants.py`, add:

```python
class MODEL_ARCH(IntEnum):
    MYMODEL = auto()  # add your arch here

MODEL_ARCH_NAMES = {
    MODEL_ARCH.MYMODEL: "mymodel",
}

MODEL_TENSORS = {
    MODEL_ARCH.MYMODEL: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ]
}
```

### Map original tensor names to GGUF

In `gguf-py/gguf/tensor_mapping.py`, map the HuggingFace tensor names to GGUF standard names:

```python
block_mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
    MODEL_TENSOR.ATTN_Q: (
        "model.layers.{bid}.self_attn.q_proj",  # your mapping
    ),
    # ... more mappings
}
```

Override methods as needed:
- `TextModel#set_gguf_parameters` -- set model hyperparameters
- `ModelBase#set_vocab` -- configure the tokenizer
- `ModelBase#modify_tensors` -- custom tensor modifications

### Run the conversion

```sh
python convert_hf_to_gguf.py /path/to/huggingface-model --outfile model.gguf
```

## Step 2: Define the architecture in C++

### Add the architecture enum

In `src/llama-arch.h`:

```cpp
enum llm_arch {
    // ... existing arches ...
    LLM_ARCH_MYMODEL,
};
```

### Register architecture metadata

In `src/llama-arch.cpp`:

- Add to `LLM_ARCH_NAMES`: `{ LLM_ARCH_MYMODEL, "mymodel" }`
- Add to `LLM_TENSOR_NAMES` if new tensors are needed
- Add to `LLM_KV_NAMES` for any model-specific metadata keys

### Handle metadata loading

In `src/llama-model-loader.cpp`, add any non-standard metadata loading for your model in the `llama_model_loader` constructor.

### Map RoPE type

If your model uses a specific RoPE type, add a case in `llama_model_rope_type()` in `src/llama-model.cpp`.

## Step 3: Implement the inference graph

This is the core implementation step. Create a struct that builds the computation graph.

In `src/llama-model.cpp`:

```cpp
struct llama_model_mymodel : llama_model_base {
    llm_graph_context_ptr build_arch_graph(
        const llm_graph_params & params) const override {

        auto gf = std::make_unique<llm_graph_context>(params);

        // Build the computation graph here using ggml operations:
        // - ggml_mul_mat (matrix multiply)
        // - ggml_rms_norm (RMS normalization)
        // - ggml_rope_ext (positional encoding)
        // - ggml_silu, ggml_gelu (activations)
        // - ggml_soft_max (attention)
        // - ggml_add, ggml_mul, ggml_reshape, etc.

        return gf;
    }
};
```

### Register the model

In the `llama_model_mapping()` function, add your model:

```cpp
case LLM_ARCH_MYMODEL:
    return std::make_unique<llama_model_mymodel>();
```

### Study existing implementations

Look at these reference implementations in `src/llama-model.cpp`:

| Model | Complexity | Notes |
|-------|-----------|-------|
| `llama_model_llama` | Simple | Standard decoder-only transformer |
| `llama_model_bert` | Simple | Encoder-only, different pooling |
| `llama_model_dbrx` | Medium | Fine-grained MoE |
| `llama_model_mixtral` | Medium | Sparse MoE |

### Test your implementation

Run the basic tools to verify correctness:

```sh
# Command line inference
llama-cli -m model.gguf -p "Hello world"

# Test with the server
llama-server -m model.gguf

# Verify perplexity matches reference
llama-perplexity -m model.gguf -f test.txt

# Check quantization works
llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M
```

### Debugging the graph

Use `llama-eval-callback` to inspect intermediate tensor values:

```sh
llama-eval-callback -m model.gguf -p "test"
```

## Step 4: Multimodal encoder (optional)

For models that process images, audio, or video:

1. Extend `MmprojModel` in the conversion script (instead of `TextModel`)
2. Add encoder definition in `tools/mtmd/models/`
3. Implement the preprocessor in `tools/mtmd/`
4. Build the encoder GGML graph

See `docs/multimodal.md` and `tools/mtmd/` for examples.

## Common pitfalls

- **Tensor name suffixes**: All tensor names must end with `.weight` or `.bias`
- **Dimension ordering**: GGML uses row-major order; PyTorch typically uses column-major. Dimensions are reversed.
- **Matrix multiplication**: `C = ggml_mul_mat(A, B)` computes C^T = A * B^T
- **Backend support**: Not all ggml backends support all operations. CPU-only is acceptable for initial PRs; add GPU backends in follow-up PRs
- **New quantization types**: Adding a new `ggml_type` requires extensive validation data (perplexity, KL divergence, performance benchmarks)

## Submission checklist

Before opening a PR:
- [ ] The model converts correctly from HuggingFace format
- [ ] `llama-cli` runs without errors
- [ ] `llama-perplexity` shows expected scores
- [ ] `llama-quantize` can quantize the model
- [ ] `llama-server` serves correctly via `/v1/chat/completions`
- [ ] CI passes locally ([ci/README.md](../ci/README.md))
- [ ] Only CPU support in the initial PR; add GPU backends later
