#!/usr/bin/env bash

RESULTS="bench-models-results.txt"
: > "$RESULTS"

ARGS_BB="-c 270336 -npp 512,4096,8192 -npl 1,2,4,8,16,32 -ntg 32"
ARGS_B="-d 0,4096,8192,16384,32768 -p 2048 -n 32"

QUICK=0
DIO=0
while (( "$#" )); do
    case "$1" in
        --quick) QUICK=1; shift ;;
        --dio) DIO=1; shift ;;
        *) shift ;;
    esac
done

if (( QUICK )); then
    ARGS_BB="-c 20480 -npp 512,4096 -npl 1,2,4 -ntg 32"
    ARGS_B="-d 0 -p 2048 -n 32"
fi

if (( DIO )); then
    ARGS_BB="${ARGS_BB} --no-mmap --direct-io"
    ARGS_B="${ARGS_B} -mmp 0 -dio 1"
fi

run_model() {
    local HFR=$1
    local HFF=$2

    printf "## ${HFR}\n" | tee -a "$RESULTS"
    printf "\n" | tee -a "$RESULTS"
    printf "Model: https://huggingface.co/${HFR}\n" | tee -a "$RESULTS"
    printf "\n" | tee -a "$RESULTS"

    printf -- "- \`llama-batched-bench\`\n" | tee -a "$RESULTS"
    printf "\n" | tee -a "$RESULTS"

    ./bin/llama-batched-bench \
        -hfr "${HFR}" -hff "${HFF}" \
        -m "${HFF}" -fa 1 -ub 2048 \
        ${ARGS_BB} | tee -a "$RESULTS"

    printf "\n" | tee -a "$RESULTS"

    printf -- "- \`llama-bench\`\n" | tee -a "$RESULTS"
    printf "\n" | tee -a "$RESULTS"

    ./bin/llama-bench \
        -m "${HFF}" -fa 1 -ub 2048 \
        ${ARGS_B} | tee -a "$RESULTS"

    printf "\n" | tee -a "$RESULTS"

    printf "\n"
}

run_model "ggml-org/gpt-oss-20b-GGUF"                       "gpt-oss-20b-mxfp4.gguf"
run_model "ggml-org/gpt-oss-120b-GGUF"                      "gpt-oss-120b-mxfp4-00001-of-00003.gguf"
run_model "ggml-org/Qwen3-Coder-30B-A3B-Instruct-Q8_0-GGUF" "qwen3-coder-30b-a3b-instruct-q8_0.gguf"
run_model "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF"             "qwen2.5-coder-7b-q8_0.gguf"
run_model "ggml-org/gemma-3-4b-it-qat-GGUF"                 "gemma-3-4b-it-qat-Q4_0.gguf"
run_model "ggml-org/GLM-4.7-Flash-GGUF"                     "GLM-4.7-Flash-Q8_0.gguf"

if [[ -f models-extra.txt ]]; then
    while read -r HFR HFF; do
        [[ -z "$HFR" ]] && continue
        run_model "$HFR" "$HFF"
    done < models-extra.txt
fi

printf "\n=====================================\n"
printf "\n"

cat "$RESULTS"

printf "\n"
printf "Done! Results are written to $RESULTS\n"
printf "\n"

