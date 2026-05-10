with open("llama.cpp-PoC/src/llama-context.h", 'r') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if "ffn_mode" in line or "cpu_hidden" in line or "struct llama_context" in line or "ffn_local" in line:
        print(f"{i+1}: {line}")
