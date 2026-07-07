#!/usr/bin/env bash
set -e

# This test exercises hot-swapping of LoRA adapters through llama-server:
#   - GET  /lora-adapters : Get list of all LoRA adapters
#   - POST /lora-adapters : Set list of LoRA adapters

# Array of models to iterate over
declare -a params=(
    "Gemma2ForCausalLM 64"
    "LlamaForCausalLM 64"
    "Phi3ForCausalLM 64"
)

MODELS_REPO="lora-tests"
MODELS_REPO_URL=https://huggingface.co/ggml-org/$MODELS_REPO
COMMIT=c26d5fb85b4070a9e9c4e65d132c783b98086890

LLAMA_BIN_DIR=${LLAMA_BIN_DIR:-build/bin}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
SERVER_PORT=${SERVER_PORT:-8080}
SERVER_URL=http://$SERVER_HOST:$SERVER_PORT

# Clone the Hugging Face repository if the directory does not exist
if [ ! -d "$MODELS_REPO" ]; then
    echo "Cloning the Hugging Face repository..."
    git clone $MODELS_REPO_URL --depth 1
    cd $MODELS_REPO
    git fetch --depth=1 origin $COMMIT
    git reset --hard $COMMIT
    cd -
else
    echo "Repository already exists. Skipping clone."
fi

# Array to store results to print
results=()

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

SERVER_PID=""

stop_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}

# Make sure the server is stopped when the script exits for any reason
trap stop_server EXIT

wait_for_server() {
    echo "Waiting for server to become healthy..."
    for _ in $(seq 1 120); do
        if curl -s -f "$SERVER_URL/health" >/dev/null 2>&1; then
            echo "Server is ready."
            return 0
        fi
        if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Error: server process exited before becoming healthy."
            exit 1
        fi
        sleep 1
    done
    echo "Error: timed out waiting for server."
    exit 1
}

# Run a completion and return the generated content
run_completion() {
    local prompt="$1"
    local lora="$2" # JSON array for the "lora" field, may be empty

    local payload
    if [ -n "$lora" ]; then
        payload=$(jq -n --arg p "$prompt" --argjson l "$lora" \
            '{prompt: $p, n_predict: 50, seed: 42, temperature: 0, lora: $l, cache_prompt: false}')
    else
        payload=$(jq -n --arg p "$prompt" \
            '{prompt: $p, n_predict: 50, seed: 42, temperature: 0, cache_prompt: false}')
    fi

    curl -s -X POST "$SERVER_URL/completion" \
        -H "Content-Type: application/json" \
        -d "$payload" | jq -r '.content'
}

run_server_lora_test() {
    local model_name=$1
    local hidden_size=$2

    local model_dir=$MODELS_REPO/$model_name/hidden_size=$hidden_size
    local base_gguf=$model_dir/base/Base-F32.gguf
    local lora_gguf=$model_dir/lora/Lora-F32-LoRA.gguf

    echo -e "\n\n-------- RUNNING SERVER LORA TEST FOR MODEL $model_name --------\n\n"

    # Convert safetensors to gguf
    echo "Running convert_hf_to_gguf.py for $model_name with hidden_size $hidden_size..."
    python3 convert_hf_to_gguf.py "$model_dir"/base \
        --outfile "$base_gguf" \
        --outtype f32

    echo -e "\n\n---------------------------\n\n"
    echo "Running convert_lora_to_gguf.py for $model_name with hidden_size $hidden_size..."
    python3 convert_lora_to_gguf.py "$model_dir"/lora \
        --base "$model_dir"/base \
        --outtype f32

    # Start the server with the adapter loaded (default scale 1)
    echo -e "\n\n---------------------------\n\n"
    echo "Starting llama-server for $model_name with hidden_size $hidden_size..."
    "$LLAMA_BIN_DIR"/llama-server \
        -m "$base_gguf" \
        --lora "$lora_gguf" \
        --host "$SERVER_HOST" --port "$SERVER_PORT" \
        --seed 42 &
    SERVER_PID=$!
    wait_for_server

    # GET /lora-adapters : the adapter must be listed
    echo -e "\n\n---------------------------\n\n"
    echo "Checking GET /lora-adapters (initial state)..."
    LORA_LIST=$(curl -s -f "$SERVER_URL/lora-adapters")
    echo "Adapters: $LORA_LIST"

    NUM_ADAPTERS=$(echo "$LORA_LIST" | jq 'length')
    if [[ "$NUM_ADAPTERS" -lt 1 ]]; then
        echo "Error: $model_name GET /lora-adapters returned no adapters."
        exit 1
    fi

    # POST /lora-adapters : hot-swap the adapter off (scale 0)
    echo -e "\n\n---------------------------\n\n"
    echo "Disabling adapter via POST /lora-adapters (scale 0)..."
    POST_RESP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$SERVER_URL/lora-adapters" \
        -H "Content-Type: application/json" \
        -d '[{"id": 0, "scale": 0.0}]')
    if [[ "$POST_RESP_CODE" != "200" ]]; then
        echo "Error: $model_name POST /lora-adapters returned HTTP $POST_RESP_CODE."
        exit 1
    fi

    # GET /lora-adapters : the scale must now be 0
    OFF_SCALE=$(curl -s -f "$SERVER_URL/lora-adapters" | jq -r '.[0].scale')
    if [[ "$OFF_SCALE" != "0" && "$OFF_SCALE" != "0.0" ]]; then
        echo "Error: $model_name adapter scale should be 0 after POST, got $OFF_SCALE."
        exit 1
    fi

    # Completion with the adapter disabled should behave like the base model
    echo -e "\n\n---------------------------\n\n"
    echo "Running completion with adapter disabled (base behavior)..."
    # The server returns only the generated content, so prepend the prompt word
    OUTPUT_BASE=$EXPECTED_BASE_FIRST_WORD$(run_completion "$EXPECTED_BASE_FIRST_WORD" "")

    # POST /lora-adapters : hot-swap the adapter on (scale 1)
    echo -e "\n\n---------------------------\n\n"
    echo "Enabling adapter via POST /lora-adapters (scale 1)..."
    POST_RESP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$SERVER_URL/lora-adapters" \
        -H "Content-Type: application/json" \
        -d '[{"id": 0, "scale": 1.0}]')
    if [[ "$POST_RESP_CODE" != "200" ]]; then
        echo "Error: $model_name POST /lora-adapters returned HTTP $POST_RESP_CODE."
        exit 1
    fi

    # GET /lora-adapters : the scale must now be 1
    HOT_SCALE=$(curl -s -f "$SERVER_URL/lora-adapters" | jq -r '.[0].scale')
    if [[ "$HOT_SCALE" != "1" && "$HOT_SCALE" != "1.0" ]]; then
        echo "Error: $model_name adapter scale should be 1 after POST, got $HOT_SCALE."
        exit 1
    fi

    # Completion with the adapter enabled should follow the lora behavior
    echo -e "\n\n---------------------------\n\n"
    echo "Running completion with adapter enabled (lora behavior)..."
    OUTPUT_LORA_HOT=$EXPECTED_LORA_FIRST_WORD$(run_completion "$EXPECTED_LORA_FIRST_WORD" "")

    stop_server

    # Assert output matches the expected prefixes
    EXPECTED_BASE=${EXPECTED_BASE_FULL:0:${#OUTPUT_BASE}}
    EXPECTED_LORA=${EXPECTED_LORA_FULL:0:${#OUTPUT_LORA_HOT}}

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

    # Store the results
    results+=("
    \n\033[1mResults for $model_name with hidden_size $hidden_size:\033[0m
    \n\033[32m  • Base (adapter disabled):\n$OUTPUT_BASE
    \n\033[34m  • Lora hot (adapter enabled via POST):\n$OUTPUT_LORA_HOT
    \n \033[0m
    ")

    echo "All server lora tests passed for $model_name with hidden_size $hidden_size!"
}

# Run test for each model
for param in "${params[@]}"; do
    read -r model_name hidden_size <<< "$param"
    run_server_lora_test "$model_name" "$hidden_size"
done

# Print results
echo -e "\n\n---------------------------\n\n"
echo -e "\n\033[1mSummary of All Results:\033[0m"
for result in "${results[@]}"; do
    echo -e "$result"
done
