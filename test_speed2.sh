#!/bin/bash
./build/bin/llama-cli -m /mnt/p5/models/Qwen3.5-35B-A3B-abliterated-Q4_K_M.gguf -n 16 -p "List the first 10 numbers" -ngl 4 -ncmoe 60 -t 12 > out2.log 2>&1
grep -A 10 "eval time" out2.log
