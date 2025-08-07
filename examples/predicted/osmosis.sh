# run osmosis query

BIN_DIR="../../build/bin"
BIN="$BIN_DIR/llama-predicted"

MODEL_DIR="../../fast_models"
MODEL="$MODEL_DIR/osmosis-apply-1.7b-bf16.gguf"

SYSTEM=$(cat data/osmosis_system.txt)
CODE=$(cat data/osmosis_code.txt)
EDIT=$(cat data/osmosis_edit.txt)

NL=$'\n'
PROMPT="<code>${NL}${CODE}${NL}</code>${NL}${NL}<edit>${NL}${EDIT}${NL}</edit>"
DRAFT="<code>${NL}${CODE}${NL}</code>"

$BIN -m "${MODEL}" -c 0 -ngl 99 -fa --color --system-prompt "${SYSTEM}" --prompt "${PROMPT}" --draft-text "${DRAFT}" --draft-min 5 --draft-max 32
