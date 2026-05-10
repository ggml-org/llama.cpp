import re

with open("llama.cpp-PoC/src/llama-model-loader.cpp", "r") as f:
    content = f.read()

# I need to modify `create_tensor` to take `flags` by reference, or to just ignore the missing tensor if split logic skips it without changing `flags` locally (since flags is passed by value and changing it doesn't affect the caller, but the caller checks `if (!tensor)` and it's fine as long as `create_tensor` returns `nullptr`).
# BUT wait! `create_tensor` is returning `nullptr`, which is correct.
# Then the CALLER (in `llama_model_base::load_tensors`) throws if it's missing AND `TENSOR_NOT_REQUIRED` was NOT in the caller's `flags`.
# So we need to modify `llama_model_base::create_tensor` (or `create_tensor_qkv` etc.)? No, we don't want to touch all that.
# The caller checks `if (!tensor)` only if it's required?
# Actually, the caller doesn't check it directly, it's `get_tensor_meta` that throws, or `check_tensor_dims`?
# In `llama_model_loader::create_tensor`, the code is:
#     if (!t_meta) {
#         if (flags & TENSOR_NOT_REQUIRED) { return nullptr; }
#         throw std::runtime_error(format("missing tensor '%s'", tn.str().c_str()));
#     }
# If `split_info.type == SPLIT_TYPE_FFN`, it skips `token_embd.weight`.
# `token_embd.weight` is missing from the GGUF file? NO, it IS in the GGUF file!
# Wait, if it IS in the GGUF file, `get_tensor_meta` would return a valid `t_meta`.
# Then my `buft_for_tensor` check comes. If it returns `nullptr`, then what?
#     ggml_backend_buffer_type_t buft = buft_for_tensor(t_meta);
#     if (!buft) {
#         return nullptr; // Wait! If it returns `nullptr`, `create_tensor` returns `nullptr`.
#     }
# And then `llama_model_base::load_tensors` receives `nullptr`.
# And then `llama_model_base::load_tensors` continues. Does it crash?
# Wait! "missing tensor 'token_embd.weight'"
# This error string "missing tensor '%s'" is thrown by `buft_for_tensor`!
# Ah!
#         if (!t_meta) {
#             if (flags & TENSOR_NOT_REQUIRED) {
#                 return nullptr;
#             }
#             throw std::runtime_error(format("missing tensor '%s'", tn.str().c_str()));
#         }
# So `t_meta` IS NULL! Which means it is NOT in the GGUF file!
# Wait! In my python script, I copied the entire vocab model. `token_embd.weight` SHOULD be in the file.
# Let's check if it is in the file.
