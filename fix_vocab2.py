with open("llama.cpp-PoC/src/llama-model-loader.cpp", "r") as f:
    content = f.read()

content = content.replace(
    'LLAMA_LOG_INFO("%s: split.type: %s\\n", __func__, split_type.c_str());\n            fprintf(stderr, "split.type: %s\\n", split_type.c_str());',
    'LLAMA_LOG_INFO("%s: split.type: %s\\n", __func__, split_type.c_str());\n            fprintf(stderr, "split.type: %s\\n", split_type.c_str());'
)

# wait I need to find the print of ffn.
# It seems "split.type: ffn" is not printed for models/ffn.gguf.
# wait, for models/ffn.gguf, token_embd.weight is not skipped?
# wait, token_embd.weight is skipped when split_info.type == SPLIT_TYPE_FFN && !is_ffn_tensor("token_embd.weight") -> Yes, it returns nullptr.
# then `llama_model_loader::create_tensor` returns nullptr.
# However, if `TENSOR_NOT_REQUIRED` is not set, `llama_model_loader::create_tensor` might throw an error or handle it.
# Let's check `llama_model_loader::create_tensor`

import re
out = re.search(r'auto buft_for_tensor = \[\&\]\(ggml_tensor \* t_meta\).*?return buft;', content, re.DOTALL)
if out:
    print(out.group(0))
