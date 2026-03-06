#!/bin/sh
# vim: set ts=4 sw=4 et:

ZIP="wikitext-2-raw-v1.zip"
FILE="wikitext-2-raw/wiki.test.raw"
URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/$ZIP"

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

have_cmd unzip || die "Please install unzip"

if [ ! -f "$FILE" ]; then
    dl "$URL" "$ZIP" || exit
    unzip -o "$ZIP"  || exit
    rm -f -- "$ZIP"
fi

cat <<EOF
Usage:

  llama-perplexity -m model.gguf -f $FILE [other params]

EOF
