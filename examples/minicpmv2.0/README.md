# 所有命令在 llama.cpp 根目录执行，模型位于根目录上级目录处
# All command should be executed under the root path of llama.cpp repo. We assume the MiniCPM-V-2 model are put in its parent folder.

```bash
make
make minicpmv-cli

python ./examples/minicpmv2.0/minicpm-surgery.py -m ../MiniCPM-V-2
python ./examples/minicpmv2.0/convert-image-encoder-to-gguf.py -m ../MiniCPM-V-2 --llava-projector ../MiniCPM-V-2/llava.projector --output-dir ../MiniCPM-V-2 --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5
python ./convert-hf-to-gguf.py ../MiniCPM-V-2/model
./minicpmv-cli -m ../MiniCPM-V-2/model/ggml-model-f16.gguf --mmproj ../MiniCPM-V-2/mmproj-model-f16.gguf -c 4096 --temp 0.6 --top-p 0.8 --top-k 100 --repeat-penalty 1.0 --image ../test.jpg -p "这张图里有什么?"

# or run quantize int4 version
./quantize ../MiniCPM-V-2/model/ggml-model-f16.gguf ../MiniCPM-V-2/model/ggml-model-Q4_K_M.gguf Q4_K_M
./minicpmv-cli -m ../MiniCPM-V-2/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-2/mmproj-model-f16.gguf -c 4096 --temp 0.6 --top-p 0.8 --top-k 100 --repeat-penalty 1.0 --image ../test.jpg -p "这张图里有什么?"
```
