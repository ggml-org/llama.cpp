# test speculative sampler

# take input code from stdin and use prompt specified as argument

MODEL_DIR="../../fast_models"
MODEL="$MODEL_DIR/gemma-3-12b-it-q8_0.gguf"
MODEL_DRAFT="$MODEL_DIR/gemma-3-1b-it-q8_0.gguf"
PROMPT="$1"
CODE=$(cat)

../../build/bin/llama-speculative -m $MODEL -md $MODEL_DRAFT -c 0 -ngl 99 -ngld 99 -fa --color -p "$PROMPT\n\n$CODE"
