#!/bin/bash

# make sure we are in the right directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

#export LLAMA_CACHE="$SCRIPT_DIR/tmp"

set -eux

mkdir -p $SCRIPT_DIR/output

PROJ_ROOT="$SCRIPT_DIR/../.."
cd $PROJ_ROOT

# Check if the first argument is "big", then run test with big models
# This is useful if we're running the script on a larger machine, so we can test the big models
RUN_BIG_TESTS=false
if [ "${1:-}" = "big" ]; then
    RUN_BIG_TESTS=true
    echo "Include BIG models..."
fi

###############

arr_bin=()
arr_hf=()
arr_tmpl=() # chat template

add_test() {
    local bin=$1
    local hf=$2
    local tmpl=${3:-""} # default to empty string if not provided
    arr_bin+=("$bin")
    arr_hf+=("$hf")
    arr_tmpl+=("$tmpl")
}

add_test "llama-mtmd-cli"  "ggml-org/SmolVLM-500M-Instruct-GGUF:Q8_0"
add_test "llama-mtmd-cli"  "ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M"
add_test "llama-mtmd-cli"  "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF:Q8_0"
add_test "llama-mtmd-cli"  "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"
add_test "llama-mtmd-cli"  "THUDM/glm-edge-v-5b-gguf:Q4_K_M"
add_test "llama-mtmd-cli"  "second-state/Llava-v1.5-7B-GGUF:Q2_K"            "vicuna"
add_test "llama-mtmd-cli"  "cjpais/llava-1.6-mistral-7b-gguf:Q3_K"           "vicuna"
add_test "llama-mtmd-cli"  "ibm-research/granite-vision-3.2-2b-GGUF:Q4_K_M"
add_test "llama-mtmd-cli"  "second-state/MiniCPM-Llama3-V-2_5-GGUF:Q2_K"  # model from openbmb is corrupted
add_test "llama-mtmd-cli"  "openbmb/MiniCPM-V-2_6-gguf:Q2_K"
add_test "llama-mtmd-cli"  "openbmb/MiniCPM-o-2_6-gguf:Q4_0"
add_test "llama-mtmd-cli"  "bartowski/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M"
add_test "llama-mtmd-cli"  "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M"
add_test "llama-mtmd-cli"  "ggml-org/InternVL2_5-1B-GGUF:Q8_0"
add_test "llama-mtmd-cli"  "ggml-org/InternVL3-1B-Instruct-GGUF:Q8_0"

# to test the big models, run: ./tests.sh big
if [ "$RUN_BIG_TESTS" = true ]; then
    add_test "llama-mtmd-cli" "ggml-org/pixtral-12b-GGUF:Q4_K_M"
    add_test "llama-mtmd-cli" "ggml-org/Mistral-Small-3.1-24B-Instruct-2503-GGUF" "mistral-v7"
    add_test "llama-mtmd-cli" "ggml-org/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M"
    add_test "llama-mtmd-cli" "ggml-org/Qwen2-VL-7B-Instruct-GGUF:Q4_K_M"
    add_test "llama-mtmd-cli" "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M"
    add_test "llama-mtmd-cli" "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M"
    add_test "llama-mtmd-cli"  "ggml-org/InternVL3-8B-Instruct-GGUF:Q4_K_M"
    add_test "llama-mtmd-cli"  "ggml-org/InternVL3-14B-Instruct-GGUF:Q4_K_M"
    # add_test "llama-mtmd-cli" "ggml-org/Qwen2.5-VL-32B-Instruct-GGUF:Q4_K_M" # does not work on my mac M3 Ultra
    # add_test "llama-mtmd-cli" "ggml-org/Qwen2.5-VL-72B-Instruct-GGUF:Q4_K_M" # too big
fi

# these models always give the wrong answer, not sure why
# add_test "llama-mtmd-cli"  "ggml-org/SmolVLM-Instruct-GGUF:Q4_K_M"
# add_test "llama-mtmd-cli"  "ggml-org/SmolVLM-256M-Instruct-GGUF:Q8_0"
# add_test "llama-mtmd-cli"  "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q8_0"

# this model has broken chat template, not usable
# add_test "llama-mtmd-cli"  "cmp-nct/Yi-VL-6B-GGUF:Q5_K"
# add_test "llama-mtmd-cli"  "guinmoon/MobileVLM-3B-GGUF:Q4_K_M" "deepseek"

###############

cmake --build build -j --target "${arr_bin[@]}"

arr_res=()

for i in "${!arr_bin[@]}"; do
    bin="${arr_bin[$i]}"
    hf="${arr_hf[$i]}"
    tmpl="${arr_tmpl[$i]}"

    echo "Running test with binary: $bin and HF model: $hf"
    echo ""
    echo ""

    output=$(\
        "$PROJ_ROOT/build/bin/$bin" \
        -hf "$hf" \
        --image $SCRIPT_DIR/test-1.jpeg \
        -p "what is the publisher name of the newspaper?" \
        --temp 0 -n 128 \
        ${tmpl:+--chat-template "$tmpl"} \
        2>&1 | tee /dev/tty)

    echo "$output" > $SCRIPT_DIR/output/$bin-$(echo "$hf" | tr '/' '-').log

    if echo "$output" | grep -iq "new york"; then
        result="\033[32mOK\033[0m:   $bin $hf"
    else
        result="\033[31mFAIL\033[0m: $bin $hf"
    fi
    echo -e "$result"
    arr_res+=("$result")

    echo ""
    echo ""
    echo ""
    echo "#################################################"
    echo "#################################################"
    echo ""
    echo ""
done

set +x

for i in "${!arr_res[@]}"; do
    echo -e "${arr_res[$i]}"
done
echo ""
echo "Output logs are saved in $SCRIPT_DIR/output"
