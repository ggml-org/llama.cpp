#!/usr/bin/env bash

API_URL="${API_URL:-http://127.0.0.1:8080}"

CHAT=(
    "Hello, Assistant."
    "Hello. How may I help you today?"
    "Please tell me the largest city in Europe."
    "Sure. The largest city in Europe is Moscow, the capital of Russia."
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
    echo -n "${INSTRUCTION}"
    printf "\n### Human: %s\n### Assistant: %s" "${CHAT[@]}" "$1"
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

N_KEEP=$(tokenize "${INSTRUCTION}" | wc -l)

chat_completion() {
    PROMPT="$(trim_trailing "$(format_prompt "$1")")"
    DATA="$(echo -n "$PROMPT" | jq -Rs --argjson n_keep $N_KEEP '{
        prompt: .,
        temperature: 0.2,
        top_k: 40,
        top_p: 0.9,
        n_keep: $n_keep,
        n_predict: 256,
        cache_prompt: true,
        stop: ["\n### Human:"],
        stream: true
    }')"

    ANSWER=''

    while IFS= read -r LINE; do
        if [[ $LINE = data:* ]]; then
            CONTENT="$(echo "${LINE:5}" | jq -r '.content')"
            printf "%s" "${CONTENT}"
            ANSWER+="${CONTENT}"
        fi
    done < <(curl \
        --silent \
        --no-buffer \
        --request POST \
        --url "${API_URL}/completion" \
        --header "Content-Type: application/json" \
        --data-raw "${DATA}")

    printf "\n"

    CHAT+=("$1" "$(trim "$ANSWER")")
}

while true; do
    read -r -e -p "> " QUESTION
    chat_completion "${QUESTION}"
done
