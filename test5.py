import os
for root, dirs, files in os.walk("llama.cpp-PoC/src"):
    for file in files:
        if file.endswith(".cpp") or file.endswith(".h"):
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                content = f.read()
                if "gguf_get_n_tensors" in content:
                    print(f"Found gguf_get_n_tensors in {file}")
