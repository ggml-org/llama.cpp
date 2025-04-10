#!/bin/bash

if [ $# -lt 2 ]; then
    echo "usage: ./scripts/compare-commits.sh <commit1> <commit2> [additional llama-bench arguments]"
    exit 1
fi

set -e
set -x

# verify at the start that the compare script has all the necessary dependencies installed
./scripts/compare-llama-bench.py --check

bench_args="${@:3}"

rm -f llama-bench.sqlite > /dev/null

# to test a backend, call the script with the corresponding environment variable (e.g. GGML_CUDA=1 ./scripts/compare-commits.sh ...)
if [ -n "$GGML_CUDA" ]; then
    cmake_opts="-DGGML_CUDA=ON"
fi

dir="build-bench"

function run {
    rm -fr ${dir} > /dev/null
    cmake -B ${dir} -S . $cmake_opts > /dev/null
    cmake --build ${dir} -t llama-bench > /dev/null
    ${dir}/bin/llama-bench -o sql -oe md $bench_args | sqlite3 llama-bench.sqlite
}

git checkout $1 > /dev/null
run

git checkout $2 > /dev/null
run

./scripts/compare-llama-bench.py -b $1 -c $2
