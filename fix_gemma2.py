with open("tools/convert_split_gguf.py", "r") as f:
    content = f.read()

# Since we don't care about a fully functional output model yet, just testing the parsing of split.type, this error is fine.
# But just in case, let's inject tokenizer.ggml.model.
insert = """        writer.add_uint32(arch + ".rope.dimension_count", 256)
        writer.add_string("tokenizer.ggml.model", "llama")
        writer.add_array("tokenizer.ggml.tokens", ["hi", "there"])
        writer.add_array("tokenizer.ggml.scores", [1.0, 1.0])
        writer.add_array("tokenizer.ggml.token_type", [1, 1])
        writer.add_uint32("tokenizer.ggml.bos_token_id", 0)
        writer.add_uint32("tokenizer.ggml.eos_token_id", 1)
        writer.add_uint32("tokenizer.ggml.unknown_token_id", 2)
"""
content = content.replace('writer.add_uint32(arch + ".rope.dimension_count", 256)', insert)

with open("tools/convert_split_gguf.py", "w") as f:
    f.write(content)
