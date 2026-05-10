with open("llama.cpp-PoC/src/llama-context.cpp", 'r') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if "ggml_backend_graph_compute" in line:
        print(f"{i+1}: {line}")
