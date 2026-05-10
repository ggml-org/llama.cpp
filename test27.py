with open("llama.cpp-PoC/src/llama.cpp", 'r') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if "ggml_backend_graph_compute" in line or "llama_decode_internal" in line:
        print(f"{i+1}: {line}")
