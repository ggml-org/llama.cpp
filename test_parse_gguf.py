from gguf import GGUFReader, GGUFWriter

try:
    reader = GGUFReader("models/ggml-vocab-llama-bpe.gguf")
    writer = GGUFWriter("models/attn.gguf", reader.arch)
    for kv in reader.fields.values():
        val = kv.parts[kv.data[-1]] if len(kv.data) > 0 and len(kv.parts) > 0 else getattr(kv, 'value', None)
        # writer.add_key_value(kv.name, getattr(kv, 'value', None))
    # Not using gguf library to avoid complex logic. Instead just test the split parsing in C++ with a mocked log.
except Exception as e:
    pass
