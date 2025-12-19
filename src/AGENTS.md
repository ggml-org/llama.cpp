# src/AGENTS.md (llama layer: new architecture support)

This directory owns: GGUF metadata parsing, tensor mapping, graph building, and runtime behavior. When adding a new model architecture, prefer keeping changes in `src/` unless a new ggml op/type is strictly required.

## Tooling / Search (fast path)

- Narrow by filename first: `fd -t f -a '(arch|model|loader|graph|context|kv).*\\.(h|cpp)$' src`
- Then search content: `rg -n "LLM_ARCH_|llm_arch|llm_kv|general\\.architecture|build_graph|load_tensors" src`

Avoid searching `build*`, `models`, `tmp` unless you are explicitly debugging those outputs.

## Architecture Entry Points (real anchors in this repo)

### Architecture enum / string mapping
- `src/llama-arch.h`: `enum llm_arch`, `enum llm_kv`, `struct LLM_KV`
- `src/llama-arch.cpp`: `LLM_ARCH_NAMES`, `llm_arch_from_string()`, `llm_arch_name()`

### GGUF KV reading / hyperparams
- `src/llama-model-loader.h`: `struct llama_model_loader`, `LLM_KV llm_kv`
- `src/llama-model-loader.cpp`: `llama_model_loader::get_arch_name()`, `llama_model_loader::get_arch()`, `get_key*()`, `get_arr*()`
- GGUF runtime: `ggml/include/gguf.h`, `ggml/src/gguf.cpp`

### Tensor mapping / shape checks
- `src/llama-model.cpp`: `llama_model::load_tensors()`
- `src/llama-model-loader.cpp`: `check_tensor_dims()`, `require_tensor_meta()`, `require_weight()`

### Graph build (prefill/decode)
- `src/llama-model.h`: `llama_model::build_graph(const llm_graph_params &)`
- `src/llama-model.cpp`: `llama_model::build_graph(...)`
- `src/llama-graph.h` / `src/llama-graph.cpp`: graph helpers and node construction
- `src/llama-context.cpp`: calls `model.build_graph(...)` for prefill/decode paths

## Guardrails

- Validate tensor shapes early and fail with actionable error messages (include expected dims and GGUF tensor name).
- Do not silently “accept and reinterpret” incompatible GGUF layouts; prefer a clear error + fallback instructions.
- Avoid cross-arch semantic changes unless explicitly intended; keep the default path stable for existing models.
- If you need new GGUF keys/tensors, also update conversion tooling (`gguf-py/`, scripts) or document required keys.
