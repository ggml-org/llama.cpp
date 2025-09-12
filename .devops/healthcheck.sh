#!/bin/sh

curl -f "http://localhost:${LLAMA_ARG_PORT:-8080}/health"
