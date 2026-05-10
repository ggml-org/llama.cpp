with open("llama.cpp-PoC/tools/cli/cli.cpp", 'r') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if "main" in line:
        print(f"{i+1}: {line}")
