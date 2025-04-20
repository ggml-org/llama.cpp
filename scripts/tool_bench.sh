#!/bin/bash
set -euo pipefail

cmake --build build -j

export LLAMA_CACHE=${LLAMA_CACHE:-$HOME/Library/Caches/llama.cpp}
export LLAMA_SERVER_BIN_PATH=$PWD/build/bin/llama-server

if [ ! -x "$LLAMA_SERVER_BIN_PATH" ]; then
    echo "Could not find llama-server binary at $LLAMA_SERVER_BIN_PATH"
    exit 1
fi
if [ ! -d "$LLAMA_CACHE" ]; then
    echo "Could not find llama cache at $LLAMA_CACHE, please set LLAMA_CACHE explicitly."
    exit 1
fi

export ARGS=(
    --llama-baseline="$(which llama-server)"
    --n 30
    --temp -1  # Leaves temperature parameter unset (use the server's default, e.g. 0.6 for ollama)
    --temp 0
    --temp 0.5
    --temp 0.75
    --temp 1
    --temp 1.5
    --temp 2
    --temp 5
    "$@"
)

./scripts/tool_bench.py run ${ARGS[@]} --model "Qwen 2.5 Coder 0.5B Q4_K_M"           --output ../qwenc0.5b.jsonl --hf bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF:Q4_K_M --ollama qwen2.5-coder:0.5b-instruct-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Qwen 2.5 Coder 1.5B Q4_K_M"           --output ../qwenc1.5b.jsonl --hf bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF:Q4_K_M --ollama qwen2.5-coder:1.5b-instruct-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Qwen 2.5 Coder 3B Q4_K_M"             --output ../qwenc3b.jsonl   --hf bartowski/Qwen2.5-Coder-3B-Instruct-GGUF:Q4_K_M   --ollama qwen2.5-coder:3b-instruct-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Qwen 2.5 Coder 7B Q4_K_M"             --output ../qwenc7b.jsonl   --hf bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M   --ollama qwen2.5-coder:7b-instruct-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Qwen 2.5 Coder 32B Q4_K_M"            --output ../qwenc32b.jsonl  --hf bartowski/Qwen2.5-Coder-32B-Instruct-GGUF:Q4_K_M  --ollama qwen2.5-coder:32B-instruct-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Qwen 2.5 1.5B Q4_K_M"                 --output ../qwen1.5b.jsonl  --hf bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M       --ollama qwen2.5:1.5b-instruct-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Qwen 2.5 3B Q4_K_M"                   --output ../qwen3b.jsonl    --hf bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M         --ollama qwen2.5:3b-instruct-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Qwen 2.5 7B Q4_K_M"                   --output ../qwen7b.jsonl    --hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M         --ollama qwen2.5:7b-instruct-q4_K_M

./scripts/tool_bench.py run ${ARGS[@]} --model "Llama 3.2 Instruct 1B Q4_K_M"         --output ../llama1b.jsonl   --hf bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M       --ollama llama3.2:1b-instruct-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Llama 3.2 Instruct 3B Q4_K_M"         --output ../llama3b.jsonl   --hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M       --ollama llama3.2:3b-instruct-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Llama 3.1 Instruct 8B Q4_K_M"         --output ../llama8b.jsonl   --hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M  --ollama llama3.1:8b-instruct-q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "Llama 3.3 70B Q4_K_M"                 --output ../llama70b.jsonl  --hf bartowski/Llama-3.3-70B-Instruct-GGUF:Q4_K_M

./scripts/tool_bench.py run ${ARGS[@]} --model "Mistral Nemo Q4_K_M"                  --output ../nemo.jsonl      --hf bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M  --ollama mistral-nemo:12b-instruct-2407-q4_K_M

./scripts/tool_bench.py run ${ARGS[@]} --model "Hermes 3 Llama 3.1 8B Q4_K_M"         --output ../hermes3.jsonl   --hf bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M       --ollama hermes3:8b-llama3.1-q4_K_M  --chat-template-file <( python scripts/get_chat_template.py NousResearch/Hermes-3-Llama-3.1-8B tool_use )
./scripts/tool_bench.py run ${ARGS[@]} --model "Hermes 2 Pro Llama 3 8B Q4_K_M"       --output ../hermes2.jsonl   --hf bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M     --ollama hermes2:8b-llama3-q4_K_M    --chat-template-file <( python scripts/get_chat_template.py NousResearch/Hermes-2-Pro-Llama-3-8B tool_use )

./scripts/tool_bench.py run ${ARGS[@]} --model "Functionary Small V3.2 Q4_K_M"        --output ../funct3.2.jsonl  --hf bartowski/functionary-small-v3.2-GGUF:Q4_K_M
./scripts/tool_bench.py run ${ARGS[@]} --model "FireFunction V2 IQ1_M"                --output ../firef2.jsonl    --hf bartowski/firefunction-v2-GGUF:IQ1_M                                                   --chat-template-file <( python scripts/get_chat_template.py fireworks-ai/llama-3-firefunction-v2 tool_use )

./scripts/tool_bench.py run ${ARGS[@]} --model "Command R7B 12-2024 Q6_K_L"           --output ../c4ai.jsonl      --hf bartowski/c4ai-command-r7b-12-2024-GGUF:Q6_K_L                                         --chat-template-file <( python scripts/get_chat_template.py CohereForAI/c4ai-command-r7b-12-2024 tool_use )

./scripts/tool_bench.py run ${ARGS[@]} --model "Gemma 2 2B Q8_0"                      --output ../gemma2.jsonl    --hf bartowski/gemma-2-2b-it-GGUF:Q8_0
./scripts/tool_bench.py run ${ARGS[@]} --model "Phi 4 Instruct Q4_K_M"                --output ../phi4.jsonl      --hf bartowski/phi-4-GGUF:Q4_K_M                       # --ollama phi4
./scripts/tool_bench.py run ${ARGS[@]} --model "Phi 3.5 Mini Instruct Q4_K_M"         --output ../phi3.5.jsonl    --hf bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M       # --ollama phi3.5:3.8b-mini-instruct-q4_K_M

# ./scripts/tool_bench.py run ${ARGS[@]} --model "DeepSeek R1 Distill Qwen 7B Q6_K_L"   --output ../dsqw7.jsonl     --hf bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q6_K_L --chat-template-file <( python scripts/get_chat_template.py NousResearch/DeepSeek-R1-Distill-Qwen-7B tool_use )
# ./scripts/tool_bench.py run ${ARGS[@]} --model "DeepSeek R1 Distill Qwen 32B Q4_K_M"  --output ../dsqw32.jsonl    --hf bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q4_K_M --chat-template-file <( python scripts/get_chat_template.py NousResearch/DeepSeek-R1-Distill-Qwen-32B tool_use )


for f in ../*.jsonl; do
    ./scripts/tool_bench.py plot "$f" --output ${f%.jsonl}.png || true
done
