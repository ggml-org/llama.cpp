with open("llama.cpp-PoC/src/llama-model.cpp", 'r') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    # try to find the start of the function containing line 1474
    if i == 1474-1:
        # traverse backwards to find the function signature
        for j in range(i, -1, -1):
            if "{" in lines[j] and ("llama_model" in lines[j] or "void" in lines[j] or "::" in lines[j]):
                print(f"Function starts somewhere around line {j+1}: {lines[j]}")
                break
