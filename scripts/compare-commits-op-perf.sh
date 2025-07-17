#!/bin/bash

if [ $# -lt 2 ]; then
    echo "usage: ./scripts/compare-commits-op-perf.sh <commit1> <commit2> [additional test-backend-ops arguments]"
    exit 1
fi

set -e
set -x

test_backend_ops_args="${@:3}"

# Extract short form of commits (first 7 characters)
commit1_short=$(echo $1 | cut -c1-7)
commit2_short=$(echo $2 | cut -c1-7)

rm -f test-backend-ops-perf-*.log

# to test a backend, call the script with the corresponding environment variable (e.g. GGML_CUDA=1 ./scripts/compare-commits.sh ...)
if [ -n "$GGML_CUDA" ]; then
    CMAKE_OPTS="${CMAKE_OPTS} -DGGML_CUDA=ON"
fi

dir="build-test-backend-ops"

function run {
    commit_short=$1
    rm -fr ${dir} > /dev/null
    cmake -B ${dir} -S . ${CMAKE_OPTS} > /dev/null
    cmake --build ${dir} -t test-backend-ops > /dev/null
    ${dir}/bin/test-backend-ops $test_backend_ops_args perf 2>&1 | tee test-backend-ops-perf-${commit_short}.log
}

git checkout $1 > /dev/null
run $commit1_short

git checkout $2 > /dev/null
run $commit2_short

./scripts/compare-test-backend-ops-perf.py -b test-backend-ops-perf-$commit1_short.log -c test-backend-ops-perf-$commit2_short.log
