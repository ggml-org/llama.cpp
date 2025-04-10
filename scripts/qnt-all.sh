#!/bin/bash

qnt=(q8_0 q6_k q5_k q5_1 q5_0 q4_k q4_1 q4_0 q3_k q2_k)
args=""

if [ -z "$1" ]; then
    echo "usage: $0 <model> [qnt] [args]"
    echo "default: $0 <model> \"${qnt[@]}\" \"${args}\""
    exit 1
fi

if [ ! -z "$2" ]; then
    qnt=($2)
fi

if [ ! -z "$3" ]; then
    args="$3"
fi

model="$1"
out="../tmp/results-${model}"

set -o pipefail
set -e

mkdir -p ${out}

for q in ${qnt[@]}; do
    time ./bin/llama-quantize ../models/${model}/ggml-model-f16.gguf ../models/${model}/ggml-model-${q}.gguf ${q} 2>&1 ${args} | tee ${out}/qnt-${q}.txt
done
