#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
    printf "Usage: %s <llama-quantize> <llama-perplexity> [<scratch-dir>] [--no-check]\n" "$0"
    printf "  <scratch-dir>  dir for the model and outputs, default /tmp/test-quantize-be\n"
    printf "  --no-check     print the perplexity of each output without checking it\n"
    printf "Example: %s ./build/bin/llama-quantize ./build/bin/llama-perplexity\n" "$0"
    exit 1
fi

quantize=$1
perplexity=$2
shift 2

scratch_dir=/tmp/test-quantize-be
no_check=
for arg in "$@"; do
    case $arg in
        --no-check) no_check=--no-check ;;
        *)          scratch_dir=$arg ;;
    esac
done

for bin in "$quantize" "$perplexity"; do
    if [ ! -x "$bin" ]; then
        echo "error: $bin not found, build it first" >&2
        exit 1
    fi
done

if [ "$(uname -m)" != s390x ]; then
    echo "error: this test needs a big-endian s390x host" >&2
    exit 1
fi

MODEL_URL=https://huggingface.co/ggml-org/models/resolve/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories15M.gguf
MODEL_SHA256=61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a
F32_SHA256=55174d7a7d46f29fc6be163ccb040330c357aa727b22ec13e2c5d713e5ba5b76

# perplexity ceiling of each big-endian output on the text below, about 10% above the measured value
# a wrong byte order gives nan or a perplexity in the hundreds
# F32 is the converted input, all other types come from llama-quantize --pure
declare -a outputs=(
    "F32       3.60"
    "F16       3.60"
    "BF16      3.59"
    "Q4_0      4.14"
    "Q4_1      4.01"
    "Q5_0      3.82"
    "Q5_1      3.65"
    "Q8_0      3.60"
    "Q2_K      5.24"
    "Q3_K      4.25"
    "Q4_K      3.83"
    "Q5_K      3.67"
    "Q6_K      3.60"
    "IQ4_NL    3.86"
    "IQ4_XS    3.85"
    "MXFP4_MOE 4.27"
)

check_sha256() {
    local name=$1
    local file=$2
    local expected=$3

    local actual
    actual=$(sha256sum "$file" | cut -d ' ' -f 1)

    if [ "$actual" != "$expected" ]; then
        echo "error: sha256 mismatch for $name: expected '$expected', got '$actual'" >&2
        exit 1
    fi
}

# keep the report readable, only show a command's output when it fails
run_logged() {
    local log=$1
    shift

    if ! "$@" > "$log" 2>&1; then
        cat "$log" >&2
        echo "error: command failed: $*" >&2
        return 1
    fi
}

# quantize one type and print its perplexity, print nothing if a step fails
measure_ppl() {
    local type=$1
    local out=$2
    local log=$scratch_dir/stories15M-be-$type-ppl.log

    if [ "$type" != F32 ]; then
        run_logged "$scratch_dir/stories15M-be-$type.log" "$quantize" --pure "$f32" "$out" "$type" || return 1
    fi
    run_logged "$log" "$perplexity" -m "$out" -f "$text" -c 128 || return 1

    # llama-perplexity prints no final estimate when the result is nan
    sed -n 's/.*Final estimate: PPL = \([0-9.]*\).*/\1/p' "$log"
}

mkdir -p "$scratch_dir"

model=$scratch_dir/stories15M.gguf
if [ ! -f "$model" ]; then
    wget -q -O "$model" "$MODEL_URL"
fi

check_sha256 stories15M.gguf "$model" "$MODEL_SHA256"

f32=$scratch_dir/stories15M-be-F32.gguf
cp "$model" "$f32"
echo YES | run_logged "$scratch_dir/stories15M-be-F32.log" python3 "$(dirname "$0")/../gguf-py/gguf/scripts/gguf_convert_endian.py" "$f32" big || exit 1
check_sha256 F32 "$f32" "$F32_SHA256"

# 588 tokens, enough for 4 chunks of 128
text=$scratch_dir/stories.txt
cat > "$text" << 'EOF'
Once upon a time, there was a little girl named Lily. She lived in a small house with her mom and dad. Lily loved to play outside in the garden. One sunny day, she saw a big red ball under a tree. She ran to the ball and picked it up. It was soft and round.
Lily wanted to play with her friend Tom. She went to his house and knocked on the door. Tom opened the door and smiled. "Hi, Lily! Do you want to play?" he asked. "Yes! Look at my new ball," said Lily. They went to the park together.
At the park, Lily and Tom threw the ball to each other. They laughed and ran on the green grass. Then the ball went too high. It flew over a fence and into a pond. Lily was sad. "Oh no, my ball is in the water," she said.
Tom had an idea. He found a long stick near the pond. He held the stick out over the water and pushed the ball to the side. Lily helped him. Soon the ball came close and Lily picked it up. It was wet, but it was not broken.
"Thank you, Tom," said Lily. "You are a good friend." Tom smiled and said, "You are welcome. Let's play again, but not near the pond." They both laughed and went back to the grass.
When the sun went down, Lily's mom called her home. Lily said goodbye to Tom and walked home with her ball. She told her mom about the pond and the stick. Her mom gave her a big hug. "I am happy you and Tom helped each other," she said.
That night, Lily put the ball next to her bed. She thought about the fun day at the park. She closed her eyes and smiled. Lily knew that sharing and helping made every day better.
The next morning, Lily woke up early. She saw a little bird outside her window. The bird was singing a happy song. Lily wanted to give the bird some food. She asked her dad for some bread. Her dad gave her a small piece, and Lily put it on the window.
The bird hopped over and ate the bread. Then it sang again, louder than before. Lily clapped her hands. "Thank you for the song, little bird," she said. The bird flew away, but it came back every morning after that.
Lily and the bird became good friends. Every day she gave it bread, and every day it sang for her. Lily learned that being kind to others, even small birds, can make the world a happy place.
EOF

rows=()
n_ok=0
n_bad=0
for entry in "${outputs[@]}"; do
    read -r type max <<< "$entry"
    out=$scratch_dir/stories15M-be-$type.gguf

    if ! ppl=$(measure_ppl "$type" "$out"); then
        ppl=-
        status=FAIL
    elif [ "$no_check" = --no-check ]; then
        status=-
    elif [[ $ppl =~ ^[0-9.]+$ ]] && awk -v p="$ppl" -v m="$max" 'BEGIN { exit !(p <= m) }'; then
        status=OK
    else
        status=FAIL
    fi

    case $status in
        OK)   n_ok=$((n_ok + 1)) ;;
        FAIL) n_bad=$((n_bad + 1)) ;;
    esac

    rows+=("$type ${ppl:-nan} $max $status")
    printf "%s: %-10s (%s) done\n" "$(basename "$0")" "$type" "$out"
done

printf "\n%-10s %10s %10s %s\n" type ppl max status
for row in "${rows[@]}"; do
    read -r type ppl max status <<< "$row"
    case $status in
        OK)   status="\033[32mOK\033[0m" ;;
        FAIL) status="\033[31mFAIL\033[0m" ;;
    esac
    printf "%-10s %10s %10s %b\n" "$type" "$ppl" "$max" "$status"
done

if [ "$no_check" != --no-check ]; then
    printf "summary: %d ok, %d failed\n" "$n_ok" "$n_bad"
fi

if [ "$n_bad" -ne 0 ]; then
    printf "\nif the perplexity is expected to change, run with --no-check and update the max values in %s:\n\n" "$0"
    printf "%s %s %s %s --no-check\n" "$0" "$quantize" "$perplexity" "$scratch_dir"
    exit 1
fi
