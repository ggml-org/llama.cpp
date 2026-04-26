import sys

with open('.github/workflows/quality-checks.yml', 'r') as f:
    content = f.read()

print("ty" in content)
