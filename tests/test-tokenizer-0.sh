#!/bin/bash
#
# Usage:
#
#   test-tokenizer-0.sh <name> <input>
#

if [ $# -ne 2 ]; then
    printf "Usage: $0 <name> <input>\n"
    exit 1
fi

name=$1
input=$2

make -j tests/test-tokenizer-0

printf "Testing %s on %s ...\n" $name $input

set -e

printf "Tokenizing using (py)  Python AutoTokenizer ...\n"
python3 ./tests/test-tokenizer-0.py ./models/tokenizers/$name --fname-tok $input > /tmp/test-tokenizer-0-$name-py.log 2>&1

printf "Tokenizing using (cpp) llama.cpp ...\n"
./tests/test-tokenizer-0 ./models/ggml-vocab-$name.gguf $input > /tmp/test-tokenizer-0-$name-cpp.log 2>&1

cat /tmp/test-tokenizer-0-$name-py.log | grep "tokenized in"
cat /tmp/test-tokenizer-0-$name-cpp.log | grep "tokenized in"

set +e

diff $input.tok $input.tokcpp > /dev/null 2>&1

if [ $? -eq 0 ]; then
    printf "Tokenization is correct!\n"
else
    diff $input.tok $input.tokcpp | head -n 32

    printf "Tokenization differs!\n"
fi
