import sys
import re

content = open("llama.cpp-PoC/src/llama-model.cpp").read()

match = re.search(r"void\s+llama_model_load_weights\s*\(", content)
if match:
    print(f"llama_model_load_weights found at index {match.start()}")
    print(content[match.start():match.start()+500])
else:
    print("llama_model_load_weights not found")
