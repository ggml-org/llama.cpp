with open("llama.cpp-PoC/src/llama-context.cpp", 'r') as f:
    content = f.read()

lines = content.split('\n')
for j in range(1650, 1690):
    print(f"{j+1}: {lines[j]}")
