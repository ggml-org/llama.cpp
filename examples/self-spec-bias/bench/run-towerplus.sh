#!/usr/bin/env bash
# TowerPlus 2B, FLORES-200 devtest cut every 3 words.
# Reproduces the table in ../README.md.
#
#   bash run-towerplus.sh           # all 1012 sentences, ~2.5 h
#   QUICK=1 bash run-towerplus.sh   # 20 sentences, a few minutes

PRESET_NAME=towerplus
PRESET_MODEL=towerplus-2b-Q4_K_M.gguf

# TowerPlus is Gemma based, so it uses the start_of_turn chat format
PRESET_PRE=$'<start_of_turn>user\nTranslate the {src} source text to {tgt}. Return only the translation, without any additional explanations or commentary.\n{src}: '
PRESET_SUF=$'\n{tgt}:<end_of_turn>\n<start_of_turn>model\n'

PRESET_MODEL_HELP="The TowerPlus weights are at https://huggingface.co/Unbabel/Tower-Plus-2B, convert with convert_hf_to_gguf.py then llama-quantize."

source "$(dirname "$0")/preset-common.sh"
