with open("llama.cpp-PoC/src/llama-model-loader.cpp", "r") as f:
    content = f.read()

# We need to find the tensor routing logic
# Search for `// some models use the token embedding tensor as the output`
# We'll inject our logic before that.

if 'if (split_info.type == SPLIT_TYPE_ATTENTION && is_ffn_tensor(tn.str().c_str()))' not in content:
    idx = content.find('// some models use the token embedding tensor as the output')
    if idx != -1:
        insert = """
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
        content = content[:idx] + insert + content[idx:]
        with open("llama.cpp-PoC/src/llama-model-loader.cpp", "w") as f:
            f.write(content)
