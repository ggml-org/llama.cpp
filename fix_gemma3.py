with open("tools/convert_split_gguf.py", "r") as f:
    content = f.read()

# Just revert and realize we don't need it. The acceptance test just asks for:
# "Output contains 'split.type: attention'"
# Which it does!
