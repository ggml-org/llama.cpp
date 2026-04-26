import sys

with open('ty.toml', 'r') as f:
    content = f.read()

content += """
[[overrides]]
include = [
    "./cmake/triton_aot_compile.py",
]

[overrides.rules]
unresolved-reference = "ignore"
unresolved-import = "ignore"
"""

with open('ty.toml', 'w') as f:
    f.write(content)
