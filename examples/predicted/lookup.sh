# test lookahead sampler

# take input code from stdin and use prompt specified as argument

QUERY="$1"
CODE=$(cat)

BIN_DIR="../../build/bin"
BIN="$BIN_DIR/llama-lookup"

MODEL_DIR="../../fast_models"
MODEL="$MODEL_DIR/gemma-3-12b-it-q8_0.gguf"

SYSTEM="You are an assistant that makes changes to code. You are given a code snippet and a prompt. You need to make the changes to the code snippet to satisfy the prompt. You need to return the modified code snippet. Do not include other text or code block markers in your response."

PROMPT="${SYSTEM}\n\nPROMPT: ${QUERY}\n\nCODE:\n${CODE}\n\n"

$BIN -m $MODEL -c 4096 -ngl 99 -fa --color --prompt "${PROMPT}" --draft-min 5 --draft-max 32
