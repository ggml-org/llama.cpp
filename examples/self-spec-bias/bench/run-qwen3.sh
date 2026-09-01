#!/usr/bin/env bash
# Qwen3 4B, FLORES-200 devtest cut every 3 words.
#
#   bash run-qwen3.sh           # all 1012 sentences
#   QUICK=1 bash run-qwen3.sh   # 20 sentences, a few minutes

PRESET_NAME=qwen3
PRESET_MODEL=Qwen3-4B-Q4_K_M.gguf

# Qwen uses the im_start chat format. The empty think block keeps a thinking
# capable Qwen3 from reasoning out loud before answering, and costs nothing on
# a non thinking one. Priming with the target language name keeps the reply short.
PRESET_PRE=$'<|im_start|>system\nTranslate the {src} source text to {tgt}. Return only the translation, without any additional explanations or commentary.<|im_end|>\n<|im_start|>user\n{src}: '
PRESET_SUF=$'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{tgt}: '

PRESET_MODEL_HELP="Measured with Qwen3-4B-Instruct-2507, https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507, converted with convert_hf_to_gguf.py then llama-quantize."

source "$(dirname "$0")/preset-common.sh"
