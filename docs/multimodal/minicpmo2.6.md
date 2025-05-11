## MiniCPM-o 2.6
Currently, this readme only supports minicpm-omni's image capabilities, and we will update the full-mode support as soon as possible.

### Prepare models and code

Download [MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6) PyTorch model from huggingface to "MiniCPM-o-2_6" folder.


### Build llama.cpp
Readme modification time: 20250206

If there are differences in usage, please refer to the official build [documentation](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)

Clone llama.cpp:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

Build llama.cpp using `CMake`:
```bash
cmake -B build
cmake --build build --config Release
```


### Usage of MiniCPM-o 2.6

Convert PyTorch model to gguf files (You can also download the converted [gguf](https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf) by us)

```bash
python ./tools/mtmd/minicpmv-surgery.py -m ../MiniCPM-o-2_6
python ./tools/mtmd/minicpmv-convert-image-encoder-to-gguf.py -m ../MiniCPM-o-2_6 --minicpmv-projector ../MiniCPM-o-2_6/minicpmv.projector --output-dir ../MiniCPM-o-2_6/ --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5 --minicpmv_version 4
python ./convert_hf_to_gguf.py ../MiniCPM-o-2_6/model

# quantize int4 version
./build/bin/llama-quantize ../MiniCPM-o-2_6/model/ggml-model-f16.gguf ../MiniCPM-o-2_6/model/ggml-model-Q4_K_M.gguf Q4_K_M
```


Inference on Linux or Mac
```bash
# run in single-turn mode
./build/bin/llama-mtmd-cli -m ../MiniCPM-o-2_6/model/ggml-model-f16.gguf --mmproj ../MiniCPM-o-2_6/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -p "What is in the image?"

# run in conversation mode
./build/bin/llama-mtmd-cli -m ../MiniCPM-o-2_6/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-o-2_6/mmproj-model-f16.gguf
```
