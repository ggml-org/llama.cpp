#!/bin/bash
./build/bin/llama-cli -m /mnt/p5/models/Qwen3.5-35B-A3B-abliterated-Q4_K_M.gguf -n 32 -p "List the first 10 numbers" -ngl 10 -ncmoe 54 -t 12 > out.log 2>&1
grep "eval time" out.log
