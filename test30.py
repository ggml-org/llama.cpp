with open("llama.cpp-PoC/src/llama-context.cpp", 'r') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if "int llama_context::decode(" in line:
        for j in range(i, min(len(lines), i+300)):
            if "build_" in lines[j] or "graph" in lines[j] or "eval_" in lines[j] or "compute" in lines[j]:
                print(f"{j+1}: {lines[j]}")
