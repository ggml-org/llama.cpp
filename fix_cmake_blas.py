with open("llama.cpp-PoC/src/CMakeLists.txt", "r") as f:
    content = f.read()

# Since we use OpenBLAS, we need to link it
# In CMakeLists.txt for llama library
if "openblas" not in content:
    content = content.replace("target_link_libraries(llama PUBLIC ggml)", "target_link_libraries(llama PUBLIC ggml openblas)")
    with open("llama.cpp-PoC/src/CMakeLists.txt", "w") as f:
        f.write(content)
