#!/usr/bin/env bash
set -e

MODEL=./models/ggml-vicuna-13b-1.1-q4_0.bin
MODEL_NAME=Vicuna

# exec options
prefix="Human: " # Ex. Vicuna uses "Human: "
opts="--temp 0 -n 80" # additional flags
nl='
'
introduction="You will be playing a game of Jeopardy. Simply answer the question in the correct format (Ex. What is Paris, or Who is George Washington)."

# file options
question_file=./examples/jeopardy/questions.txt
touch ./examples/jeopardy/results/$MODEL_NAME.txt
output_file=./examples/jeopardy/results/$MODEL_NAME.txt

counter=1

echo 'Running'
while IFS= read -r question
do
  exe_cmd="./llama-cli -p "\"$prefix$introduction$nl$prefix$question\"" "$opts" -m ""\"$MODEL\""" >> ""\"$output_file\""
  echo $counter
  echo "Current Question: $question"
  eval "$exe_cmd"
  echo -e "\n------" >> $output_file
  counter=$((counter+1))
done < "$question_file"
