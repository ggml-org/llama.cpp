with open("llama.cpp-PoC/src/llama-graph.cpp", 'r') as f:
    content = f.read()

lines = content.split('\n')
for j in range(1140, 1160):
    if j < len(lines):
        print(f"{j+1}: {lines[j]}")
