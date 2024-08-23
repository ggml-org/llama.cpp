#!/bin/bash
set -e

# Array of models to iterate over
declare -a params=(
    "Gemma2ForCausalLM 64"
    "LlamaForCausalLM 64"
    "Phi3ForCausalLM 64"
)

MODELS_REPO=lora-tests
MODELS_REPO_URL=https://huggingface.co/ggml-org/$MODELS_REPO

# Clone the Hugging Face repository if the directory does not exist
if [ ! -d "$MODELS_REPO" ]; then
    echo "Cloning the Hugging Face repository..."
    git clone $MODELS_REPO_URL --depth 1
else
    echo "Repository already exists. Skipping clone."
fi

# Array to store results to print
results=()

trim_leading_whitespace() {
    local input_string="$1"
    echo "${input_string#"${input_string%%[![:space:]]*}"}"
}

extract_starting_substring() {
    local reference_string="$1"
    local target_string="$2"

    local target_length=${#target_string}
    echo "${reference_string:0:$target_length}"
}

get_first_word() {
    local input_string="$1"
    read -r first_word _ <<< "$input_string"
    echo "$first_word"
}

# Load the expected strings
EXPECTED_BASE_FULL=$(cat $MODELS_REPO/data/pale_blue_dot.txt)
EXPECTED_LORA_FULL=$(cat $MODELS_REPO/data/bohemian_rhapsody.txt)
EXPECTED_BASE_FIRST_WORD=$(get_first_word "$EXPECTED_BASE_FULL")
EXPECTED_LORA_FIRST_WORD=$(get_first_word "$EXPECTED_LORA_FULL")

run_conversion_and_inference_lora() {
    local model_name=$1
    local hidden_size=$2

    echo -e "\n\n-------- RUNNING TEST FOR MODEL $model_name --------\n\n"

    # Convert safetensors to gguf
    echo "Running convert_hf_to_gguf.py for $model_name with hidden_size $hidden_size..."
    python convert_hf_to_gguf.py $MODELS_REPO/$model_name/hidden_size=$hidden_size/base \
        --outfile $MODELS_REPO/$model_name/hidden_size=$hidden_size/base/Base-F32.gguf \
        --outtype f32

    echo -e "\n\n---------------------------\n\n"
    echo "Running convert_lora_to_gguf.py for $model_name with hidden_size $hidden_size..."
    python3 convert_lora_to_gguf.py $MODELS_REPO/$model_name/hidden_size=$hidden_size/lora \
        --base $MODELS_REPO/$model_name/hidden_size=$hidden_size/base \
        --outtype f32

    echo -e "\n\n---------------------------\n\n"
    echo "Running llama-export-lora with lora for $model_name with hidden_size $hidden_size..."
    ./llama-export-lora \
        -m $MODELS_REPO/$model_name/hidden_size=$hidden_size/base/Base-F32.gguf \
        -o $MODELS_REPO/$model_name/hidden_size=$hidden_size/base/Base-F32-lora-merged.gguf \
        --lora $MODELS_REPO/$model_name/hidden_size=$hidden_size/lora/Lora-F32-LoRA.gguf

    # Run inference
    echo -e "\n\n---------------------------\n\n"
    echo "Running llama-cli without lora for $model_name with hidden_size $hidden_size..."
    OUTPUT_BASE=$(./llama-cli -m $MODELS_REPO/$model_name/hidden_size=$hidden_size/base/Base-F32.gguf \
        -p "$EXPECTED_BASE_FIRST_WORD" -n 50 --seed 42 --temp 0)

    echo -e "\n\n---------------------------\n\n"
    echo "Running llama-cli with hot lora for $model_name with hidden_size $hidden_size..."
    OUTPUT_LORA_HOT=$(./llama-cli -m $MODELS_REPO/$model_name/hidden_size=$hidden_size/base/Base-F32.gguf \
        --lora $MODELS_REPO/$model_name/hidden_size=$hidden_size/lora/Lora-F32-LoRA.gguf \
        -p "$EXPECTED_LORA_FIRST_WORD" -n 50 --seed 42 --temp 0)

    echo -e "\n\n---------------------------\n\n"
    echo "Running llama-cli with merged lora for $model_name with hidden_size $hidden_size..."
    OUTPUT_LORA_MERGED=$(./llama-cli -m $MODELS_REPO/$model_name/hidden_size=$hidden_size/base/Base-F32-lora-merged.gguf \
        -p "$EXPECTED_LORA_FIRST_WORD" -n 50 --seed 42 --temp 0)

    # Remove any initial white space
    OUTPUT_BASE=$(trim_leading_whitespace "$OUTPUT_BASE")
    OUTPUT_LORA_HOT=$(trim_leading_whitespace "$OUTPUT_LORA_HOT")
    OUTPUT_LORA_MERGED=$(trim_leading_whitespace "$OUTPUT_LORA_MERGED")
    # Extract the corresponding substring from full string
    EXPECTED_BASE=$(extract_starting_substring "$EXPECTED_BASE_FULL" "$OUTPUT_BASE")
    EXPECTED_LORA=$(extract_starting_substring "$EXPECTED_LORA_FULL" "$OUTPUT_LORA_HOT")

    # Assert output equals the expected output
    if [[ "$OUTPUT_BASE" != "$EXPECTED_BASE" ]]; then
        echo "Error: $model_name OUTPUT_BASE does not start with the expected string."
        echo -e "Out=$OUTPUT_BASE\n\nExp=$EXPECTED_BASE"
        exit 1
    fi
    if [[ "$OUTPUT_LORA_HOT" != "$EXPECTED_LORA" ]]; then
        echo "Error: $model_name OUTPUT_LORA_HOT does not start with the expected string."
        echo -e "Out=$OUTPUT_LORA_HOT\n\nExp=$EXPECTED_LORA"
        exit 1
    fi
    if [[ "$OUTPUT_LORA_MERGED" != "$EXPECTED_LORA" ]]; then
        echo "Error: $model_name OUTPUT_LORA_MERGED does not start with the expected string."
        echo -e "Out=$OUTPUT_LORA_MERGED\n\nExp=$EXPECTED_LORA"
        exit 1
    fi

    # Store the results
    results+=("
    \n\033[1mResults for $model_name with hidden_size $hidden_size:\033[0m
    \n\033[32m  • Base:\n$OUTPUT_BASE
    \n\033[34m  • Lora hot:\n$OUTPUT_LORA_HOT
    \n\033[36m  • Lora merged:\n$OUTPUT_LORA_MERGED
    \n \033[0m
    ")

    echo "All tests passed for $model_name with hidden_size $hidden_size!"
}

# Run test for each model
for param in "${params[@]}"; do
    run_conversion_and_inference_lora $param
done

# Print results
echo -e "\n\n---------------------------\n\n"
echo -e "\n\033[1mSummary of All Results:\033[0m"
for result in "${results[@]}"; do
    echo -e "$result"
done
