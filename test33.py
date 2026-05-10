with open("llama.cpp-PoC/src/llama-context.h", 'r') as f:
    content = f.read()

lines = content.split('\n')
for j in range(40, 100):
    print(f"{j+1}: {lines[j]}")
