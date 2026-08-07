#!/usr/bin/env bash

cd `dirname $0`
cd ..

# get -m model parameter otherwise defer to default
if [ "$1" == "-m" ]; then
  MODEL="-m $2 "
fi

./llama-cli $MODEL --color \
    -f ./prompts/reason-act.txt \
    -i --interactive-first \
    --top_k 10000 --temp 0.2 --repeat_penalty 1 -t 7 -c 2048 \
    -r "Question:" -r "Observation:" --in-prefix " " \
    -n -1
