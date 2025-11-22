cd /data/local/tmp/llama.cpp
export LD_LIBRARY_PATH=$PWD/lib
export ADSP_LIBRARY_PATH=$PWD/lib

GGML_HEXAGON_NDEV=2 ./bin/llama-server  --no-mmap -m ../gguf/LFM2-8B-A1B-Q4_0.gguf   \
         --poll 1000 -t 6 --cpu-mask 0xfc --cpu-strict 1             \
         --ctx-size 8192 --batch-size 128 -ctk q8_0 -ctv q8_0 -fa on \
         -ngl 99 --device HTP0,HTP1 --host 0.0.0.0 --verbose &
