with open("llama.cpp-PoC/src/CMakeLists.txt", "r") as f:
    content = f.read()

content = content.replace("llama-model-loader.cpp", "llama-model-loader.cpp\n            llama-ffn-local.cpp")

with open("llama.cpp-PoC/src/CMakeLists.txt", "w") as f:
    f.write(content)
