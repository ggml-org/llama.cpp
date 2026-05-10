# gemma architecture models expect "gemma2.context_length" and others.
with open("tools/convert_split_gguf.py", "r") as f:
    content = f.read()

content = content.replace('"llama.context_length"', 'arch + ".context_length"')
content = content.replace('"llama.embedding_length"', 'arch + ".embedding_length"')
content = content.replace('"llama.block_count"', 'arch + ".block_count"')
content = content.replace('"llama.feed_forward_length"', 'arch + ".feed_forward_length"')
content = content.replace('"llama.attention.head_count"', 'arch + ".attention.head_count"')
content = content.replace('"llama.attention.head_count_kv"', 'arch + ".attention.head_count_kv"')
content = content.replace('"llama.rope.freq_base"', 'arch + ".rope.freq_base"')
content = content.replace('"llama.attention.layer_norm_rms_epsilon"', 'arch + ".attention.layer_norm_rms_epsilon"')
content = content.replace('"llama.vocab_size"', 'arch + ".vocab_size"')
content = content.replace('"llama.rope.dimension_count"', 'arch + ".rope.dimension_count"')

with open("tools/convert_split_gguf.py", "w") as f:
    f.write(content)
