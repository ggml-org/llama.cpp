with open("llama.cpp-PoC/src/llama-model-loader.cpp", 'r') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if "void llama_model_loader::init" in line or "llama_model_loader::llama_model_loader" in line:
        print(f"Found at line {i+1}: {line}")
