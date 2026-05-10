pip install gguf
python3 -c "
import gguf
reader = gguf.GGUFReader('models/ggml-vocab-llama-bpe.gguf')
print(reader.fields.keys())
"
