#!/bin/bash
#
# Use this script only on fresh pods (runpod.io)!
# Otherwise, it can break your environment!
#

if [ -z "$1" ]; then
    echo "Usage: $0 <data>"
    echo "  0: no models"
    echo "  1: tinyllama-1b"
    echo "  2: codellama-7b"
    echo "  3: codellama-13b"
    echo "  4: codellama-34b"
    echo "  5: codellama-7b-instruct"
    echo "  6: codellama-13b-instruct"
    echo "  7: codellama-34b-instruct"

    exit 1
fi

set -x

# setup deps
apt-get update
apt-get install -y git-lfs cmake cmake-curses-gui vim ruby
git-lfs install

if [ ! -d "/workspace" ]; then
    ln -sfn $(pwd) /workspace
fi

# download data
cd /workspace

# this is useful to git clone repos without doubling the disk size due to .git
git clone https://github.com/iboB/git-lfs-download
ln -sfn /workspace/git-lfs-download/git-lfs-download /usr/local/bin/git-lfs-download

# llama.cpp
cd /workspace
git clone https://github.com/ggerganov/llama.cpp

cd llama.cpp

GGML_CUDA=1 make -j

ln -sfn /workspace/TinyLlama-1.1B-Chat-v0.3  ./models/tinyllama-1b
ln -sfn /workspace/CodeLlama-7b-hf           ./models/codellama-7b
ln -sfn /workspace/CodeLlama-13b-hf          ./models/codellama-13b
ln -sfn /workspace/CodeLlama-34b-hf          ./models/codellama-34b
ln -sfn /workspace/CodeLlama-7b-Instruct-hf  ./models/codellama-7b-instruct
ln -sfn /workspace/CodeLlama-13b-Instruct-hf ./models/codellama-13b-instruct
ln -sfn /workspace/CodeLlama-34b-Instruct-hf ./models/codellama-34b-instruct

pip install -r requirements.txt

# cmake
cd /workspace/llama.cpp

mkdir build-cublas
cd build-cublas

cmake -DGGML_CUDA=1 ../
make -j

if [ "$1" -eq "0" ]; then
    exit 0
fi

# more models
if [ "$1" -eq "1" ]; then
    cd /workspace

    git-lfs-download https://huggingface.co/PY007/TinyLlama-1.1B-Chat-v0.3

    cd /workspace/llama.cpp

    python3 examples/convert_legacy_llama.py ./models/tinyllama-1b  --outfile ./models/tinyllama-1b/ggml-model-f16.gguf  --outtype f16

    ./llama-quantize ./models/tinyllama-1b/ggml-model-f16.gguf ./models/tinyllama-1b/ggml-model-q4_0.gguf q4_0
    ./llama-quantize ./models/tinyllama-1b/ggml-model-f16.gguf ./models/tinyllama-1b/ggml-model-q4_k.gguf q4_k
    ./llama-quantize ./models/tinyllama-1b/ggml-model-f16.gguf ./models/tinyllama-1b/ggml-model-q8_0.gguf q8_0
fi

if [ "$1" -eq "2" ]; then
    cd /workspace

    git-lfs-download https://huggingface.co/codellama/CodeLlama-7b-hf  --without *safetensors*
    rm -v ./CodeLlama-7b-hf/*safetensors*

    cd /workspace/llama.cpp

    python3 examples/convert_legacy_llama.py ./models/codellama-7b  --outfile ./models/codellama-7b/ggml-model-f16.gguf  --outtype f16

    ./llama-quantize ./models/codellama-7b/ggml-model-f16.gguf ./models/codellama-7b/ggml-model-q4_0.gguf q4_0
    ./llama-quantize ./models/codellama-7b/ggml-model-f16.gguf ./models/codellama-7b/ggml-model-q4_k.gguf q4_k
    ./llama-quantize ./models/codellama-7b/ggml-model-f16.gguf ./models/codellama-7b/ggml-model-q8_0.gguf q8_0
fi

if [ "$1" -eq "3" ]; then
    cd /workspace

    git-lfs-download https://huggingface.co/codellama/CodeLlama-13b-hf --without *safetensors*
    rm -v ./CodeLlama-13b-hf/*safetensors*

    cd /workspace/llama.cpp

    python3 examples/convert_legacy_llama.py ./models/codellama-13b --outfile ./models/codellama-13b/ggml-model-f16.gguf --outtype f16

    ./llama-quantize ./models/codellama-13b/ggml-model-f16.gguf ./models/codellama-13b/ggml-model-q4_0.gguf q4_0
    ./llama-quantize ./models/codellama-13b/ggml-model-f16.gguf ./models/codellama-13b/ggml-model-q4_k.gguf q4_k
    ./llama-quantize ./models/codellama-13b/ggml-model-f16.gguf ./models/codellama-13b/ggml-model-q8_0.gguf q8_0
fi

if [ "$1" -eq "4" ]; then
    cd /workspace

    git-lfs-download https://huggingface.co/codellama/CodeLlama-34b-hf --without *safetensors*
    rm -v ./CodeLlama-34b-hf/*safetensors*

    cd /workspace/llama.cpp

    python3 examples/convert_legacy_llama.py ./models/codellama-34b --outfile ./models/codellama-34b/ggml-model-f16.gguf --outtype f16

    ./llama-quantize ./models/codellama-34b/ggml-model-f16.gguf ./models/codellama-34b/ggml-model-q4_0.gguf q4_0
    ./llama-quantize ./models/codellama-34b/ggml-model-f16.gguf ./models/codellama-34b/ggml-model-q4_k.gguf q4_k
    ./llama-quantize ./models/codellama-34b/ggml-model-f16.gguf ./models/codellama-34b/ggml-model-q8_0.gguf q8_0
fi

if [ "$1" -eq "5" ]; then
    cd /workspace

    git-lfs-download https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf  --without *safetensors*
    rm -v ./CodeLlama-7b-Instruct-hf/*safetensors*

    cd /workspace/llama.cpp

    python3 examples/convert_legacy_llama.py ./models/codellama-7b-instruct  --outfile ./models/codellama-7b-instruct/ggml-model-f16.gguf  --outtype f16

    ./llama-quantize ./models/codellama-7b-instruct/ggml-model-f16.gguf ./models/codellama-7b-instruct/ggml-model-q4_0.gguf q4_0
    ./llama-quantize ./models/codellama-7b-instruct/ggml-model-f16.gguf ./models/codellama-7b-instruct/ggml-model-q4_k.gguf q4_k
    ./llama-quantize ./models/codellama-7b-instruct/ggml-model-f16.gguf ./models/codellama-7b-instruct/ggml-model-q8_0.gguf q8_0
fi

if [ "$1" -eq "6" ]; then
    cd /workspace

    git-lfs-download https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf --without *safetensors*
    rm -v ./CodeLlama-13b-Instruct-hf/*safetensors*

    cd /workspace/llama.cpp

    python3 examples/convert_legacy_llama.py ./models/codellama-13b-instruct --outfile ./models/codellama-13b-instruct/ggml-model-f16.gguf --outtype f16

    ./llama-quantize ./models/codellama-13b-instruct/ggml-model-f16.gguf ./models/codellama-13b-instruct/ggml-model-q4_0.gguf q4_0
    ./llama-quantize ./models/codellama-13b-instruct/ggml-model-f16.gguf ./models/codellama-13b-instruct/ggml-model-q4_k.gguf q4_k
    ./llama-quantize ./models/codellama-13b-instruct/ggml-model-f16.gguf ./models/codellama-13b-instruct/ggml-model-q8_0.gguf q8_0
fi

if [ "$1" -eq "7" ]; then
    cd /workspace

    git-lfs-download https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf --without *safetensors*
    rm -v ./CodeLlama-34b-Instruct-hf/*safetensors*

    cd /workspace/llama.cpp

    python3 examples/convert_legacy_llama.py ./models/codellama-34b-instruct --outfile ./models/codellama-34b-instruct/ggml-model-f16.gguf --outtype f16

    ./llama-quantize ./models/codellama-34b-instruct/ggml-model-f16.gguf ./models/codellama-34b-instruct/ggml-model-q4_0.gguf q4_0
    ./llama-quantize ./models/codellama-34b-instruct/ggml-model-f16.gguf ./models/codellama-34b-instruct/ggml-model-q4_k.gguf q4_k
    ./llama-quantize ./models/codellama-34b-instruct/ggml-model-f16.gguf ./models/codellama-34b-instruct/ggml-model-q8_0.gguf q8_0
fi

if [ "$1" -eq "1" ]; then
    # perf + perplexity
    cd /workspace/llama.cpp/build-cublas

    make -j && ../scripts/run-all-perf.sh tinyllama-1b "f16" "-ngl 99 -t 1 -p 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512,1024,2048 -n 128"

    ../scripts/get-wikitext-2.sh
    unzip wikitext-2-raw-v1.zip

    make -j && ./bin/llama-perplexity -m ../models/tinyllama-1b/ggml-model-f16.gguf -f ./wikitext-2-raw/wiki.test.raw -ngl 100 --chunks 32

    # batched
    cd /workspace/llama.cpp

    GGML_CUDA=1 make -j && ./llama-batched ./models/tinyllama-1b/ggml-model-f16.gguf "Hello, my name is" 8 128 999

    # batched-bench
    cd /workspace/llama.cpp

    GGML_CUDA=1 make -j && ./llama-batched-bench ./models/tinyllama-1b/ggml-model-f16.gguf 4608 1 99 0 512 128 1,2,3,4,5,6,7,8,16,32

    # parallel
    cd /workspace/llama.cpp

    GGML_CUDA=1 make -j && ./llama-parallel -m ./models/tinyllama-1b/ggml-model-f16.gguf -t 1 -ngl 100 -c 4096 -b 512 -s 1 -np 8 -ns 128 -n 100 -cb

fi

# speculative
#if [ "$1" -eq "7" ]; then
#    cd /workspace/llama.cpp
#
#    GGML_CUDA=1 make -j && ./llama-speculative -m ./models/codellama-34b-instruct/ggml-model-f16.gguf -md ./models/codellama-7b-instruct/ggml-model-q4_0.gguf -p "# Dijkstra's shortest path algorithm in Python (4 spaces indentation) + complexity analysis:\n\n" -e -ngl 999 -ngld 999 -t 4 -n 512 -c 4096 -s 21 --draft 16 -np 1 --temp 0.0
#fi

# more benches
#GGML_CUDA=1 make -j && ./llama-batched-bench ./models/codellama-7b/ggml-model-q4_k.gguf  4096 1 99 1 512,3200 128,128,800 1
#GGML_CUDA=1 make -j && ./llama-batched-bench ./models/codellama-13b/ggml-model-q4_k.gguf 4096 1 99 1 512,3200 128,128,800 1
