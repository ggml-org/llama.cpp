chmod +x build/bin*

./build/bin/llama-cli -m ./build/bin/ds1.5b.gguf -n 256 -c 2048 --color -t $(nproc)
