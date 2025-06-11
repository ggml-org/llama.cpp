#! /bin/bash
#clear
if [[ ${1:-} == "strace" ]]; then
    prefix="strace"
elif [[ ${1:-} == "gdb" ]]; then
    prefix="gdb --args"
else
    prefix=""
fi

MODEL=${MODEL:-llama3.2}

LLAMA_BUILD_DIR=../build.remoting-frontend$FLAVOR

MODEL_HOME="$HOME/models"

set -x
if [[ "${BENCH_MODE:-}" == "bench" ]]; then
    cat <<EOF
###
### Running llama-bench
###

EOF
    $prefix \
        $LLAMA_BUILD_DIR/bin/llama-bench \
        --model "$MODEL_HOME/$MODEL" \
        --n-gpu-layers 99
elif [[ "${BENCH_MODE:-}" == "perf" ]]; then
    cat <<EOF
###
### Running test-backend-ops perf
###

EOF
    $prefix \
        $LLAMA_BUILD_DIR/bin/test-backend-ops perf

else
    PROMPT="say nothing"
    #PROMPT="tell what's Apple metal API"
    $prefix \
        $LLAMA_BUILD_DIR/bin/llama-run \
        --ngl 99 \
        --verbose \
        --context-size 4096 \
        "$MODEL_HOME/$MODEL" \
        "$PROMPT"
fi
