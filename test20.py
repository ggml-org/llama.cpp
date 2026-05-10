with open("llama.cpp-PoC/src/llama-model-loader.cpp", 'r') as f:
    content = f.read()

lines = content.split('\n')
for j in range(1250, 1310):
    if "ggml_dup_tensor" in lines[j]:
        print(f"Around ggml_dup_tensor: {j+1}: {lines[j]}")
