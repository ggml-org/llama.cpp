with open("tools/convert_split_gguf.py", "r") as f:
    content = f.read()

content = content.replace('                writer.add_uint32(arch + ".rope.dimension_count", 256)', '        writer.add_uint32(arch + ".rope.dimension_count", 256)')

with open("tools/convert_split_gguf.py", "w") as f:
    f.write(content)
