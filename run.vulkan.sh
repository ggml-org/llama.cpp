#! /bin/bash

if [[ ${1:-} == "gdb" ]]; then
    prefix="gdb --args"
else
    prefix=""
fi

export VN_DEBUG=init
$prefix ../build.vulkan/bin/llama-run --ngl 99 --verbose ~/models/llama3.2 "say nothing"
