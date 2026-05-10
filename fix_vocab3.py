with open("llama.cpp-PoC/src/llama-model-loader.cpp", "r") as f:
    content = f.read()

# We need to change the logic:
# if (split_info.type == SPLIT_TYPE_ATTENTION && is_ffn_tensor(tn.str().c_str())) {
#     return nullptr;
# }
# However, if `flags & TENSOR_NOT_REQUIRED` is NOT set, returning nullptr from `buft_for_tensor` is actually a bug if it is returning nullptr to `create_tensor`, which then checks `if (!buft_for_tensor(t_meta))`
# No, `buft_for_tensor` returns `ggml_backend_buffer_type_t`, which is `nullptr` if we skip it.
# Then `create_tensor` handles it:
#         ggml_backend_buffer_type_t buft = buft_for_tensor(t_meta);
#         if (!buft) { return nullptr; }
# Wait, `llama_model_base::load_tensors` might throw if `layer.wq = create_tensor(..., flags)` returns nullptr and `flags` does not include `TENSOR_NOT_REQUIRED`!
# Ah! For FFN split, it skips `token_embd.weight`, but `token_embd.weight` is REQUIRED.
# So `load_tensors` crashes because it tries to load a required tensor!
# So we need to automatically add `TENSOR_NOT_REQUIRED` if it's being skipped by split logic!

import re

# let's change `create_tensor` to handle split logic properly
# Actually, inside `create_tensor`, if we skip it due to split logic, we should just return nullptr, BUT we must bypass the check in `llama_model_base::create_tensor`?
# No, `llama_model_loader::create_tensor` just returns nullptr.
# But wait, where is it throwing the error?
# "llama_model_load: error loading model: missing tensor 'token_embd.weight'"
# This happens at:
#         if (!t_meta) {
#             if (flags & TENSOR_NOT_REQUIRED) {
#                 return nullptr;
#             }
#             throw std::runtime_error(format("missing tensor '%s'", tn.str().c_str()));
#         }
# So it's not `buft_for_tensor` returning nullptr that throws.
# It is `ggml_tensor * t_meta = get_tensor_meta(tn.str().c_str());` being nullptr, and then `buft_for_tensor(t_meta)` throws because `t_meta` is nullptr!
# So `get_tensor_meta` is returning nullptr because `token_embd.weight` is literally missing from `ffn.gguf`!

# So we should modify `buft_for_tensor` (or the beginning of `create_tensor`) to check if the tensor should be skipped BEFORE `!t_meta`.

old_logic = """
    auto buft_for_tensor = [&](ggml_tensor * t_meta) -> ggml_backend_buffer_type_t {
        if (!t_meta) {
            if (flags & TENSOR_NOT_REQUIRED) {
                return nullptr;
            }
            throw std::runtime_error(format("missing tensor '%s'", tn.str().c_str()));
        }


        // TENSOR ROUTING
        if (split_info.type == SPLIT_TYPE_ATTENTION && is_ffn_tensor(tn.str().c_str())) {
            LLAMA_LOG_INFO("%s: skipping FFN tensor %s\\n", __func__, tn.str().c_str());
            return nullptr;
        }
        if (split_info.type == SPLIT_TYPE_FFN && !is_ffn_tensor(tn.str().c_str())) {
            LLAMA_LOG_INFO("%s: skipping attention tensor %s\\n", __func__, tn.str().c_str());
            return nullptr;
        }
"""

new_logic = """
    auto buft_for_tensor = [&](ggml_tensor * t_meta) -> ggml_backend_buffer_type_t {
        // TENSOR ROUTING
        if (split_info.type == SPLIT_TYPE_ATTENTION && is_ffn_tensor(tn.str().c_str())) {
            LLAMA_LOG_INFO("%s: skipping FFN tensor %s\\n", __func__, tn.str().c_str());
            return nullptr;
        }
        if (split_info.type == SPLIT_TYPE_FFN && !is_ffn_tensor(tn.str().c_str())) {
            LLAMA_LOG_INFO("%s: skipping attention tensor %s\\n", __func__, tn.str().c_str());
            return nullptr;
        }

        if (!t_meta) {
            if (flags & TENSOR_NOT_REQUIRED) {
                return nullptr;
            }
            throw std::runtime_error(format("missing tensor '%s'", tn.str().c_str()));
        }
"""
content = content.replace(old_logic, new_logic)

with open("llama.cpp-PoC/src/llama-model-loader.cpp", "w") as f:
    f.write(content)
