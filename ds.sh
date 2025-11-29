chmod -R +x build/bin/

cd build/bin/

wget https://www.modelscope.cn/models/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/master/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf && mv DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf ds1.5b.gguf

cd ../..

./build/bin/llama-cli -m build/bin/ds1.5b.gguf --color -t 4 -ngl 99
