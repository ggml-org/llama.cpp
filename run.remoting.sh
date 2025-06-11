#! /bin/bash
#clear
if [[ ${1:-} == "strace" ]]; then
    prefix="strace"
elif [[ ${1:-} == "gdb" ]]; then
    prefix="gdb --args"
else
    prefix=""
fi

if [[ "${PERF_MODE:-}" ]]; then
    FLAVOR="-prod"
else
    FLAVOR=""
fi

MODEL=${MODEL:-llama3.2}

if [[ "$FLAVOR" == "-prod" ]]; then
    cat <<EOF
###
### Running the prod flavor
###

EOF
fi

if [[ "${BENCH_MODE:-}" ]]; then
    bench=yes
else
    bench=no
fi

LLAMA_BUILD_DIR=../build.remoting-frontend$FLAVOR

MODEL_HOME="$HOME/models"

set -x
if [[ "$bench" == yes ]]; then
    $prefix \
        $LLAMA_BUILD_DIR/bin/llama-bench \
        --model "$MODEL_HOME/$MODEL" \
        --n-gpu-layers 99
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
