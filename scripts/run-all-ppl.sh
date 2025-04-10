#!/bin/bash

qnt=(f16 q8_0 q6_k q5_k q5_1 q5_0 q4_k q4_1 q4_0 q3_k q2_k)
args="-ngl 999 -t 8"

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

set -o pipefail
set -e

model="$1"
out="../tmp/results-${model}"

mkdir -p ${out}

for q in ${qnt[@]}; do
    time ./bin/llama-perplexity -m ../models/${model}/ggml-model-f16.gguf -f ./wiki.test.raw ${args} 2>&1 | tee ${out}/ppl-${q}.txt
done
