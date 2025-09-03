#!/usr/bin/env bash
curl --request POST \
    --url http://localhost:8080/embedding \
    --header "Content-Type: application/json" \
    --data '{"input": "Hello world today"}' \
    --silent
