import sys

with open('cmake/triton_aot_compile.py', 'r') as f:
    content = f.read()

content = content.replace("import triton", "import triton  # type: ignore")
content = content.replace("import triton.language as tl", "import triton.language as tl  # type: ignore\ntl = tl  # type: ignore")

with open('cmake/triton_aot_compile.py', 'w') as f:
    f.write(content)
