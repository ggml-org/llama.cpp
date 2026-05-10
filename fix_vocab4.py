with open("llama.cpp-PoC/src/llama-model-loader.cpp", "r") as f:
    content = f.read()

# wait, in my new_logic for `buft_for_tensor`:
old = """
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
"""

new = """
        // TENSOR ROUTING
        if (split_info.type == SPLIT_TYPE_ATTENTION && is_ffn_tensor(tn.str().c_str())) {
            LLAMA_LOG_INFO("%s: skipping FFN tensor %s\\n", __func__, tn.str().c_str());
            flags |= TENSOR_NOT_REQUIRED;
            return nullptr;
        }
        if (split_info.type == SPLIT_TYPE_FFN && !is_ffn_tensor(tn.str().c_str())) {
            LLAMA_LOG_INFO("%s: skipping attention tensor %s\\n", __func__, tn.str().c_str());
            flags |= TENSOR_NOT_REQUIRED;
            return nullptr;
        }

        if (!t_meta) {
"""
content = content.replace(old, new)

with open("llama.cpp-PoC/src/llama-model-loader.cpp", "w") as f:
    f.write(content)
