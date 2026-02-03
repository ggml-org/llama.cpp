#!/bin/bash

API_URL="http://localhost:8080/v1/chat/completions"
AUTH_TOKEN="Bearer no-key"

request_oai(){
    curl -s --no-buffer "$API_URL" \
    -H "Content-Type: application/json" \
    -H "Authorization: $AUTH_TOKEN" \
    -d '{
        "model": "gpt-3.5-turbo",
        "max_tokens": 64,
        "temperature": 1,
        "top_k": 0,
        "top_p": 1,
        "min_p": 0.5,
        "stream": true,
        "repeat_penalty": 1,
        "repeat_last_n": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "ignore_eos": false,
        "n_probs": 0,
        "seed": -1,
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant. Your top priority is to fulfill the requests of the user by helping them with their needs."
            },
            {
                "role": "user",
                "content": "'"$USER_INPUT"'"
            }
        ]
    }' | \
        while IFS= read -r LINE; do
            if [[ $LINE = data:* ]]; then
                CONTENT="$(echo "${LINE:5}" | jq -r '.choices[].delta.content // empty')"
                printf "%s" "${CONTENT}"
            fi
        done
    echo
    echo
}

while true; do
    echo -e "\033[1;94mUser\033[0m"
    read -r USER_INPUT
    echo
    echo -e "\033[1mAssistant\033[0m"
    request_oai
done
