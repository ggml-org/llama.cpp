with open("llama.cpp-PoC/src/llama-model-loader.h", 'r') as f:
    content = f.read()

lines = content.split('\n')
for j in range(80, 130):
    print(f"{j+1}: {lines[j]}")
