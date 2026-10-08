#!/usr/bin/env bash
curl --request POST \
    --url http://localhost:9931/embedding \
    --header "Content-Type: application/json" \
    --data '{"input": "Hello world today"}' \
    --silent
