with open("llama.cpp-PoC/src/llama-model.cpp", 'r') as f:
    content = f.read()

lines = content.split('\n')
for j in range(1415, 1465):
    print(f"{j+1}: {lines[j]}")
