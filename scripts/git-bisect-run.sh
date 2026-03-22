#!/usr/bin/env bash

cmake_args=()
llama_results_args=()

for arg in "${@}"; do
    if [[ "$arg" == -D* ]]; then
        cmake_args+=("$arg")
    else
        llama_results_args+=("$arg")
    fi
done

dir="build-bisect"
rm -rf ${dir} > /dev/null
cmake -B ${dir} -S . ${cmake_args} > /dev/null
cmake --build ${dir} -t llama-results -j $(nproc) > /dev/null
${dir}/bin/llama-results "${llama_results_args[@]}"
