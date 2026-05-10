with open("llama.cpp-PoC/src/llama-graph.h", 'r') as f:
    content = f.read()

lines = content.split('\n')
for j in range(810, 830):
    if j < len(lines):
        print(f"{j+1}: {lines[j]}")
