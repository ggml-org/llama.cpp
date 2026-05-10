with open("llama.cpp-PoC/src/llama-model.h", 'r') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if "ffn_mode" in line or "llama_model" in line:
        print(f"{i+1}: {line}")
