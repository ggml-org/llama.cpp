#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    printf "Usage: $0 <git-repo> <target-folder> [<test-exe>]\n"
    exit 1
fi

if [ $# -eq 3 ]; then
    toktest=$3
else
    toktest="./test-tokenizer-0"
fi

if [ ! -x $toktest ]; then
    printf "Test executable \"$toktest\" not found!\n"
    exit 1
fi

repo=$1
folder=$2

# det note: ensure any large tokenizer artifacts are available locally. If git-lfs
# is installed, pull LFS objects after updating/cloning; otherwise skip gracefully.
if [ -d $folder ] && [ -d $folder/.git ]; then
    (cd $folder; git pull && command -v git-lfs >/dev/null 2>&1 && git lfs pull || true)
else
    git clone $repo $folder
    (cd $folder; command -v git-lfs >/dev/null 2>&1 && git lfs pull || true)
fi

shopt -s globstar
for gguf in $folder/**/*.gguf; do
    if [ -f $gguf.inp ] && [ -f $gguf.out ]; then
        $toktest $gguf
    else
        printf "Found \"$gguf\" without matching inp/out files, ignoring...\n"
    fi
done
