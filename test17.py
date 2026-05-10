with open("llama.cpp-PoC/src/llama-model.cpp", 'r') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if "ggml_backend_sched_alloc_graph" in line or "llama_model_loader" in line and "create_tensor" in line:
        print(f"{i+1}: {line}")
