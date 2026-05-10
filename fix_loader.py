import re

with open("llama.cpp-PoC/src/llama-model-loader.cpp", "r") as f:
    content = f.read()

# fix `llama_model_ftype_name(ftype)` error -> `llama_model_ftype_name(this->ftype)` doesn't work since `ftype` doesn't exist? Oh, it's just `fver` perhaps, or we need to add `ftype` to `llama_model_loader`. Let's check `llama_model_loader.h`.
with open("llama.cpp-PoC/src/llama-model-loader.h", "r") as f:
    hcontent = f.read()

if "llama_ftype ftype" not in hcontent:
    hcontent = hcontent.replace(
        "llama_fver fver = GGUF_FILE_VERSION_V3;",
        "llama_fver fver = GGUF_FILE_VERSION_V3;\n    llama_ftype ftype;"
    )

with open("llama.cpp-PoC/src/llama-model-loader.h", "w") as f:
    f.write(hcontent)
