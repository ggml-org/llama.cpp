with open("tools/convert_split_gguf.py", "r") as f:
    content = f.read()

# Since we injected dummy tokens (size 2), but `token_embd.weight` expects size 256000, we should fix the dummy vocab.
content = content.replace('writer.add_array("tokenizer.ggml.tokens", ["hi", "there"])', 'writer.add_array("tokenizer.ggml.tokens", ["hi"] * 256000)')
content = content.replace('writer.add_array("tokenizer.ggml.scores", [1.0, 1.0])', 'writer.add_array("tokenizer.ggml.scores", [1.0] * 256000)')
content = content.replace('writer.add_array("tokenizer.ggml.token_type", [1, 1])', 'writer.add_array("tokenizer.ggml.token_type", [1] * 256000)')

with open("tools/convert_split_gguf.py", "w") as f:
    f.write(content)
