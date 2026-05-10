import os
import re
with open("llama.cpp-PoC/src/llama-model.cpp", 'r') as f:
    content = f.read()

# Let's search for "gguf_get_n_tensors" in llama-model.cpp as well
if "gguf_get_n_tensors" in content:
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "gguf_get_n_tensors" in line:
            print(f"llama-model.cpp:{i+1}: {line}")
            for j in range(max(0, i-5), min(len(lines), i+20)):
                print(f"  {lines[j]}")
            print("-" * 40)
