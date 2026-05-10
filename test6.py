with open("llama.cpp-PoC/src/llama-model.cpp", 'r') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if "ggml_backend_alloc_ctx_tensors_from_buft" in line:
        print(f"--- Context around line {i+1} ---")
        for j in range(max(0, i-10), min(len(lines), i+15)):
            print(f"{j+1}: {lines[j]}")
