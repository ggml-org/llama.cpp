with open("llama.cpp-PoC/src/CMakeLists.txt", "r") as f:
    content = f.read()

# Only add it if not already there to avoid dupes
if "llama-ffn-local.cpp" not in content:
    content = content.replace("llama-model-loader.cpp", "llama-model-loader.cpp\n            llama-ffn-local.cpp")
    with open("llama.cpp-PoC/src/CMakeLists.txt", "w") as f:
        f.write(content)
