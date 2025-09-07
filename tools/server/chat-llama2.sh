#!/usr/bin/env bash

API_URL="${API_URL:-http://127.0.0.1:8080}"

CHAT=(
    "Hello, Assistant."
    "Hello. How may I help you today?"
)

INSTRUCTION="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

trim() {
    shopt -s extglob
    set -- "${1##+([[:space:]])}"
    printf "%s" "${1%%+([[:space:]])}"
}

trim_trailing() {
    shopt -s extglob
    printf "%s" "${1%%+([[:space:]])}"
}

format_prompt() {
    if [[ "${#CHAT[@]}" -eq 0 ]]; then
        echo -n "[INST] <<SYS>>\n${INSTRUCTION}\n<</SYS>>"
    else
        LAST_INDEX=$(( ${#CHAT[@]} - 1 ))
        echo -n "${CHAT[$LAST_INDEX]}\n[INST] $1 [/INST]"
    fi
}

tokenize() {
    curl \
        --silent \
        --request POST \
        --url "${API_URL}/tokenize" \
        --header "Content-Type: application/json" \
        --data-raw "$(jq -ns --arg content "$1" '{content:$content}')" \
    | jq '.tokens[]'
}

N_KEEP=$(tokenize "[INST] <<SYS>>\n${INSTRUCTION}\n<</SYS>>" | wc -l)

chat_completion() {
    PROMPT="$(trim_trailing "$(format_prompt "$1")")"
    DATA="$(echo -n "$PROMPT" | jq -Rs --argjson n_keep $N_KEEP '{
        prompt: .,
        temperature: 0.2,
        top_k: 40,
        top_p: 0.9,
        n_keep: $n_keep,
        n_predict: 1024,
        stop: ["[INST]"],
        stream: true
    }')"

    # Create a temporary file to hold the Python output
    TEMPFILE=$(mktemp)

    exec 3< <(curl \
        --silent \
        --no-buffer \
        --request POST \
        --url "${API_URL}/completion" \
        --header "Content-Type: application/json" \
        --data-raw "${DATA}")

    python -c "
import json
import sys

answer = ''
while True:
    line = sys.stdin.readline()
    if not line:
        break
    if line.startswith('data: '):
        json_content = line[6:].strip()
        content = json.loads(json_content)['content']
        sys.stdout.write(content)
        sys.stdout.flush()
        answer += content

answer = answer.rstrip('\n')

# Write the answer to the temporary file
with open('$TEMPFILE', 'w') as f:
    f.write(answer)
    " <&3

    exec 3<&-

    # Read the answer from the temporary file
    ANSWER=$(cat $TEMPFILE)

    # Clean up the temporary file
    rm $TEMPFILE

    printf "\n"

    CHAT+=("$1" "$(trim "$ANSWER")")
}

while true; do
    echo -en "\033[0;32m"  # Green color
    read -r -e -p "> " QUESTION
    echo -en "\033[0m"  # Reset color
    chat_completion "${QUESTION}"
done
