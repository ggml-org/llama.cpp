# llama.cpp/example/mimi

This demonstrates running [Kyutai's Mimi](https://huggingface.co/kyutai/mimi) model via GGML.

## Quickstart

Convert model to GGUF (no need to download, the script will automatically download the `safetensors` file)

```sh
python examples/tts/convert_mimi_to_gguf.py

# output file: kyutai-mimi.gguf

# optionally, use q8_0 quantization for faster speed
python examples/tts/convert_mimi_to_gguf.py --outtype q8_0
```

Then compile, run it:

```sh
cmake --build build -j --target llama-mimi

./build/bin/llama-mimi kyutai-mimi.gguf codes.txt

# output: output.wav

# alternatively, use "dummy1" to get a "wah hello there" sample output file
./build/bin/llama-mimi kyutai-mimi.gguf dummy1
```

Example of code file (one code per line):

```
1263
1597
1596
1477
1540
1720
1433
118
1066
1968
1096
232
418
566
1653
2010
```
