#!/usr/bin/env bash

qnt=(f16 q8_0 q6_k q5_k q5_1 q5_0 q4_k q4_1 q4_0 q3_k q2_k)
args="-ngl 999 -n 64 -p 512"

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

mstr=""

for q in ${qnt[@]}; do
    mstr="${mstr} -m ../models/${model}/ggml-model-${q}.gguf"
done

./bin/llama-bench ${mstr} ${args} 2> /dev/null
