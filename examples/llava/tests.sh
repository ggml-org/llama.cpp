#!/bin/bash

# make sure we are in the right directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

#export LLAMA_CACHE="$SCRIPT_DIR/tmp"

set -eux

mkdir -p $SCRIPT_DIR/output

PROJ_ROOT="$SCRIPT_DIR/../.."
cd $PROJ_ROOT

###############

arr_bin=()
arr_hf=()

add_test() {
    local bin=$1
    local hf=$2
    arr_bin+=("$bin")
    arr_hf+=("$hf")
}

add_test "llama-gemma3-cli"   "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"
add_test "llama-llava-cli"    "cmp-nct/Yi-VL-6B-GGUF:Q5_K"
add_test "llama-llava-cli"    "guinmoon/MobileVLM-3B-GGUF:Q4_K_M"
add_test "llama-llava-cli"    "THUDM/glm-edge-v-5b-gguf:Q4_K_M"
add_test "llama-llava-cli"    "second-state/Llava-v1.5-7B-GGUF:Q2_K"
add_test "llama-llava-cli"    "cjpais/llava-1.6-mistral-7b-gguf:Q3_K"
add_test "llama-llava-cli"    "ibm-research/granite-vision-3.2-2b-GGUF:Q4_K_M"
add_test "llama-minicpmv-cli" "second-state/MiniCPM-Llama3-V-2_5-GGUF:Q2_K" # model from openbmb is corrupted
add_test "llama-minicpmv-cli" "openbmb/MiniCPM-V-2_6-gguf:Q2_K"
add_test "llama-minicpmv-cli" "openbmb/MiniCPM-o-2_6-gguf:Q4_0"
add_test "llama-qwen2vl-cli"  "bartowski/Qwen2-VL-2B-Instruct-GGUF:Q4_K_M"

###############

cmake --build build -j --target "${arr_bin[@]}"

arr_res=()

for i in "${!arr_bin[@]}"; do
    bin="${arr_bin[$i]}"
    hf="${arr_hf[$i]}"

    echo "Running test with binary: $bin and HF model: $hf"
    echo ""
    echo ""

    output=$("$PROJ_ROOT/build/bin/$bin" -hf "$hf" --image $SCRIPT_DIR/test-1.jpeg -p "what is the publisher name of the newspaper?" --temp 0 2>&1 | tee /dev/tty)

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
