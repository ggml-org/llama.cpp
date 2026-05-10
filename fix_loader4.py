with open("llama.cpp-PoC/src/llama-model-loader.h", "r") as f:
    hcontent = f.read()

if "size_t  size_data  = 0;" not in hcontent:
    hcontent = hcontent.replace(
        "size_t  n_bytes    = 0;",
        "size_t  n_bytes    = 0;\n    size_t  size_data  = 0;\n    size_t  size_done  = 0;"
    )

with open("llama.cpp-PoC/src/llama-model-loader.h", "w") as f:
    f.write(hcontent)
