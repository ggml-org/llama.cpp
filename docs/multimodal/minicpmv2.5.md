## MiniCPM-Llama3-V 2.5

### Prepare models and code

Download [MiniCPM-Llama3-V-2_5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5) PyTorch model from huggingface to "MiniCPM-Llama3-V-2_5" folder.


### Build llama.cpp
Readme modification time: 20250206

If there are differences in usage, please refer to the official build [documentation](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)

Clone llama.cpp:
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

Build llama.cpp using `CMake`:
```bash
cmake -B build
cmake --build build --config Release
```


### Usage of MiniCPM-Llama3-V 2.5

Convert PyTorch model to gguf files (You can also download the converted [gguf](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf) by us)

```bash
python ./tools/mtmd/minicpmv-surgery.py -m ../MiniCPM-Llama3-V-2_5
python ./tools/mtmd/minicpmv-convert-image-encoder-to-gguf.py -m ../MiniCPM-Llama3-V-2_5 --minicpmv-projector ../MiniCPM-Llama3-V-2_5/minicpmv.projector --output-dir ../MiniCPM-Llama3-V-2_5/ --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5 --minicpmv_version 2
python ./convert_hf_to_gguf.py ../MiniCPM-Llama3-V-2_5/model

# quantize int4 version
./build/bin/llama-quantize ../MiniCPM-Llama3-V-2_5/model/model-8B-F16.gguf ../MiniCPM-Llama3-V-2_5/model/ggml-model-Q4_K_M.gguf Q4_K_M
```


Inference on Linux or Mac
```bash
# run in single-turn mode
./build/bin/llama-mtmd-cli -m ../MiniCPM-Llama3-V-2_5/model/model-8B-F16.gguf --mmproj ../MiniCPM-Llama3-V-2_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -p "What is in the image?"

# run in conversation mode
./build/bin/llama-mtmd-cli -m ../MiniCPM-Llama3-V-2_5/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-Llama3-V-2_5/mmproj-model-f16.gguf
```
