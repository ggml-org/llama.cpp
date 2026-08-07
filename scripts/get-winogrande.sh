#!/bin/sh
# vim: set ts=4 sw=4 et:

FILE="winogrande-debiased-eval.csv"
URL="https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp/raw/main/$FILE"

die() {
    printf "%s\n" "$@" >&2
    exit 1
}

have_cmd() {
    for cmd; do
        command -v "$cmd" >/dev/null || return
    done
}

dl() {
    [ -f "$2" ] && return
    if have_cmd wget; then
        wget "$1" -O "$2"
    elif have_cmd curl; then
        curl -L "$1" -o "$2"
    else
        die "Please install wget or curl"
    fi
}

if [ ! -f "$FILE" ]; then
    dl "$URL" "$FILE" || exit
fi

cat <<EOF
Usage:

  llama-perplexity -m model.gguf -f $FILE --winogrande [--winogrande-tasks N] [other params]

EOF
