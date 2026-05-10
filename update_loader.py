import re

with open("llama.cpp-PoC/src/llama-model-loader.h", "r") as f:
    content = f.read()

if '#include "llama-split-info.h"' not in content:
    content = content.replace(
        '#include "ggml-cpp.h"',
        '#include "ggml-cpp.h"\n#include "llama-split-info.h"'
    )

if 'split_info_t split_info;' not in content:
    content = content.replace(
        'int n_tensors = 0;',
        'int n_tensors = 0;\n    split_info_t split_info;'
    )

with open("llama.cpp-PoC/src/llama-model-loader.h", "w") as f:
    f.write(content)

with open("llama.cpp-PoC/src/llama-model-loader.cpp", "r") as f:
    content = f.read()

# Try to match where to put the split info parsing
# We can put it at the end of the constructor
if 'split.type' not in content:
    # let's find the end of llama_model_loader::llama_model_loader
    # it ends with something like:
    #         n_type[cur.type]++;
    #     }
    #     // print info
    # }

    # Actually, we can add it right after:
    # get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
    # llm_kv = LLM_KV(llm_arch_from_string(arch_name));

    split_logic = """
    {
        std::string split_type;
        if (get_key("split.type", split_type, false)) {
            if (split_type == "attention") {
                split_info.type = SPLIT_TYPE_ATTENTION;
            } else if (split_type == "ffn") {
                split_info.type = SPLIT_TYPE_FFN;
            }
            get_key("split.source_sha256", split_info.source_sha256, false);
            get_key("split.layer_first", split_info.layer_first, false);
            get_key("split.layer_last", split_info.layer_last, false);
            get_key("split.n_embd", split_info.n_embd, false);
            LLAMA_LOG_INFO("%s: split.type: %s\\n", __func__, split_type.c_str());
        }
    }
"""

    # insert before fver = (enum llama_fver) gguf_get_version(metadata);
    content = content.replace(
        'fver = (enum llama_fver) gguf_get_version(metadata);',
        split_logic + '\n    fver = (enum llama_fver) gguf_get_version(metadata);'
    )

with open("llama.cpp-PoC/src/llama-model-loader.cpp", "w") as f:
    f.write(content)
