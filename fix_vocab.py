with open("llama.cpp-PoC/src/llama-model-loader.cpp", "r") as f:
    content = f.read()

# Make it print the log even if it fails to load vocab
# we'll find the line: LLAMA_LOG_INFO("%s: split.type: %s\\n", __func__, split_type.c_str());
# and we'll change it to use stderr so it always prints.
content = content.replace(
    'LLAMA_LOG_INFO("%s: split.type: %s\\n", __func__, split_type.c_str());',
    'LLAMA_LOG_INFO("%s: split.type: %s\\n", __func__, split_type.c_str());\n            fprintf(stderr, "split.type: %s\\n", split_type.c_str());'
)

with open("llama.cpp-PoC/src/llama-model-loader.cpp", "w") as f:
    f.write(content)
