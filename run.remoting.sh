#! /bin/bash
clear
if [[ ${1:-} == "gdb" ]]; then
    prefix="gdb --args"
else
    prefix=""
fi

MODEL="$HOME/models/llama3.2"
#PROMPT="say nothing"
PROMPT="tell what's Apple metal API"
$prefix \
    ../build.remoting-frontend/bin/llama-run \
    --ngl 99 \
    --verbose \
    "$MODEL" \
    "$PROMPT"
