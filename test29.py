with open("llama.cpp-PoC/src/llama-context.cpp", 'r') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if "int llama_context::decode(" in line:
        print(f"--- decode starts at {i+1} ---")
        for j in range(i, min(len(lines), i+150)):
            if "build_ffn" in lines[j] or "graph_compute" in lines[j] or "llama_decode_internal" in lines[j]:
                print(f"{j+1}: {lines[j]}")
