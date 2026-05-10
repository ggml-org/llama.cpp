with open("llama.cpp-PoC/src/llama-model-loader.cpp", 'r') as f:
    content = f.read()

lines = content.split('\n')
for j in range(500, 520):
    print(f"{j+1}: {lines[j]}")
