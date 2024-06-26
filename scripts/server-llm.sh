#!/bin/bash
#
# Helper script for deploying llama.cpp server with a single Bash command
#
# - Works on Linux and macOS
# - Supports: CPU, CUDA, Metal
# - Can run all GGUF models from HuggingFace
# - Can serve requests in parallel
# - Always builds latest llama.cpp from GitHub
#
# Limitations
#
# - Chat templates are poorly supported (base models recommended)
# - Might be unstable!
#
# Usage:
#   ./server-llm.sh [--port] [--repo] [--wtype] [--backend] [--gpu-id] [--n-parallel] [--n-kv] [--verbose] [-non-interactive]
#
#   --port:            port number, default is 8888
#   --repo:            path to a repo containing GGUF model files
#   --wtype:           weights type (f16, q8_0, q4_0, q4_1), default is user-input
#   --backend:         cpu, cuda, metal, depends on the OS
#   --gpu-id:          gpu id, default is 0
#   --n-parallel:      number of parallel requests, default is 8
#   --n-kv:            KV cache size, default is 4096
#   --verbose:         verbose output
#   --non-interactive: run without asking a permission to run
#
# Example:
#
#   bash -c "$(curl -s https://ggml.ai/server-llm.sh)"
#

set -e

# required utils: curl, git, make
if ! command -v curl &> /dev/null; then
    printf "[-] curl not found\n"
    exit 1
fi
if ! command -v git &> /dev/null; then
    printf "[-] git not found\n"
    exit 1
fi
if ! command -v make &> /dev/null; then
    printf "[-] make not found\n"
    exit 1
fi

# parse arguments
is_interactive=1
port=8888
repo=""
wtype=""
backend="cpu"

# if macOS, use metal backend by default
if [[ "$OSTYPE" == "darwin"* ]]; then
    backend="metal"
elif command -v nvcc &> /dev/null; then
    backend="cuda"
fi

gpu_id=0
n_parallel=8
n_kv=4096
verbose=0

function print_usage {
    printf "Usage:\n"
    printf "  ./server-llm.sh [--port] [--repo] [--wtype] [--backend] [--gpu-id] [--n-parallel] [--n-kv] [--verbose] [-non-interactive]\n\n"
    printf "  --port:             port number, default is 8888\n"
    printf "  --repo:             path to a repo containing GGUF model files\n"
    printf "  --wtype:            weights type (f16, q8_0, q4_0, q4_1), default is user-input\n"
    printf "  --backend:          cpu, cuda, metal, depends on the OS\n"
    printf "  --gpu-id:           gpu id, default is 0\n"
    printf "  --n-parallel:       number of parallel requests, default is 8\n"
    printf "  --n-kv:             KV cache size, default is 4096\n"
    printf "  --verbose:          verbose output\n\n"
    printf "  --non-interactive:  run without asking a permission to run\n"
    printf "Example:\n\n"
    printf '  bash -c "$(curl -s https://ggml.ai/server-llm.sh)"\n\n'
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --non-interactive)
            is_interactive=0
            shift
            ;;
        --port)
            port="$2"
            shift
            shift
            ;;
        --repo)
            repo="$2"
            shift
            shift
            ;;
        --wtype)
            wtype="$2"
            shift
            shift
            ;;
        --backend)
            backend="$2"
            shift
            shift
            ;;
        --gpu-id)
            gpu_id="$2"
            shift
            shift
            ;;
        --n-parallel)
            n_parallel="$2"
            shift
            shift
            ;;
        --n-kv)
            n_kv="$2"
            shift
            shift
            ;;
        --verbose)
            verbose=1
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $key"
            print_usage
            exit 1
            ;;
    esac
done

# available weights types
wtypes=("F16" "Q8_0" "Q4_0" "Q4_1" "Q5_0" "Q5_1" "Q6_K" "Q5_K_M" "Q5_K_S" "Q4_K_M" "Q4_K_S" "Q3_K_L" "Q3_K_M" "Q3_K_S" "Q2_K")

wfiles=()
for wt in "${wtypes[@]}"; do
    wfiles+=("")
done

# map wtype input to index
if [[ ! -z "$wtype" ]]; then
    iw=-1
    is=0
    for wt in "${wtypes[@]}"; do
        # uppercase
        uwt=$(echo "$wt" | tr '[:lower:]' '[:upper:]')
        if [[ "$uwt" == "$wtype" ]]; then
            iw=$is
            break
        fi
        is=$((is+1))
    done

    if [[ $iw -eq -1 ]]; then
        printf "[-] Invalid weight type: %s\n" "$wtype"
        exit 1
    fi

    wtype="$iw"
fi

# sample repos
repos=(
    "https://huggingface.co/TheBloke/Llama-2-7B-GGUF"
    "https://huggingface.co/TheBloke/Llama-2-13B-GGUF"
    "https://huggingface.co/TheBloke/Llama-2-70B-GGUF"
    "https://huggingface.co/TheBloke/CodeLlama-7B-GGUF"
    "https://huggingface.co/TheBloke/CodeLlama-13B-GGUF"
    "https://huggingface.co/TheBloke/CodeLlama-34B-GGUF"
    "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF"
    "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF"
    "https://huggingface.co/TheBloke/OpenHermes-2-Mistral-7B-GGUF"
    "https://huggingface.co/TheBloke/CausalLM-7B-GGUF"
)
if [ $is_interactive -eq 1 ]; then
    printf "\n"
    printf "[I] This is a helper script for deploying llama.cpp's server on this machine.\n\n"
    printf "    Based on the options that follow, the script might download a model file\n"
    printf "    from the internet, which can be a few GBs in size. The script will also\n"
    printf "    build the latest llama.cpp source code from GitHub, which can be unstable.\n"
    printf "\n"
    printf "    Upon success, an HTTP server will be started and it will serve the selected\n"
    printf "    model using llama.cpp for demonstration purposes.\n"
    printf "\n"
    printf "    Please note:\n"
    printf "\n"
    printf "    - All new data will be stored in the current folder\n"
    printf "    - The server will be listening on all network interfaces\n"
    printf "    - The server will run with default settings which are not always optimal\n"
    printf "    - Do not judge the quality of a model based on the results from this script\n"
    printf "    - Do not use this script to benchmark llama.cpp\n"
    printf "    - Do not use this script in production\n"
    printf "    - This script is only for demonstration purposes\n"
    printf "\n"
    printf "    If you don't know what you are doing, please press Ctrl-C to abort now\n"
    printf "\n"
    printf "    Press Enter to continue ...\n\n"

    read
fi

if [[ -z "$repo" ]]; then
    printf "[+] No repo provided from the command line\n"
    printf "    Please select a number from the list below or enter an URL:\n\n"

    is=0
    for r in "${repos[@]}"; do
        printf "    %2d) %s\n" $is "$r"
        is=$((is+1))
    done

    # ask for repo until index of sample repo is provided or an URL
    while [[ -z "$repo" ]]; do
        printf "\n    Or choose one from: https://huggingface.co/models?sort=trending&search=gguf\n\n"
        read -p "[+] Select repo: " repo

        # check if the input is a number
        if [[ "$repo" =~ ^[0-9]+$ ]]; then
            if [[ "$repo" -ge 0 && "$repo" -lt ${#repos[@]} ]]; then
                repo="${repos[$repo]}"
            else
                printf "[-] Invalid repo index: %s\n" "$repo"
                repo=""
            fi
        elif [[ "$repo" =~ ^https?:// ]]; then
            repo="$repo"
        else
            printf "[-] Invalid repo URL: %s\n" "$repo"
            repo=""
        fi
    done
fi

# remove suffix
repo=$(echo "$repo" | sed -E 's/\/tree\/main$//g')

printf "[+] Checking for GGUF model files in %s\n" "$repo"

# find GGUF files in the source
# TODO: better logic
model_tree="${repo%/}/tree/main"
model_files=$(curl -s "$model_tree" | grep -i "\\.gguf</span>" | sed -E 's/.*<span class="truncate group-hover:underline">(.*)<\/span><\/a>/\1/g')

# list all files in the provided git repo
printf "[+] Model files:\n\n"
for file in $model_files; do
    # determine iw by grepping the filename with wtypes
    iw=-1
    is=0
    for wt in "${wtypes[@]}"; do
        # uppercase
        ufile=$(echo "$file" | tr '[:lower:]' '[:upper:]')
        if [[ "$ufile" =~ "$wt" ]]; then
            iw=$is
            break
        fi
        is=$((is+1))
    done

    if [[ $iw -eq -1 ]]; then
        continue
    fi

    wfiles[$iw]="$file"

    have=" "
    if [[ -f "$file" ]]; then
        have="*"
    fi

    printf "    %2d) %s %s\n" $iw "$have" "$file"
done

wfile="${wfiles[$wtype]}"

# ask for weights type until provided and available
while [[ -z "$wfile" ]]; do
    printf "\n"
    read -p "[+] Select weight type: " wtype
    wfile="${wfiles[$wtype]}"

    if [[ -z "$wfile" ]]; then
        printf "[-] Invalid weight type: %s\n" "$wtype"
        wtype=""
    fi
done

printf "[+] Selected weight type: %s (%s)\n" "$wtype" "$wfile"

url="${repo%/}/resolve/main/$wfile"

# check file if the model has been downloaded before
chk="$wfile.chk"

# check if we should download the file
# - if $wfile does not exist
# - if $wfile exists but $chk does not exist
# - if $wfile exists and $chk exists but $wfile is newer than $chk
# TODO: better logic using git lfs info

do_download=0

if [[ ! -f "$wfile" ]]; then
    do_download=1
elif [[ ! -f "$chk" ]]; then
    do_download=1
elif [[ "$wfile" -nt "$chk" ]]; then
    do_download=1
fi

if [[ $do_download -eq 1 ]]; then
    printf "[+] Downloading weights from %s\n" "$url"

    # download the weights file
    curl -o "$wfile" -# -L "$url"

    # create a check file if successful
    if [[ $? -eq 0 ]]; then
        printf "[+] Creating check file %s\n" "$chk"
        touch "$chk"
    fi
else
    printf "[+] Using cached weights %s\n" "$wfile"
fi

# get latest llama.cpp and build

printf "[+] Downloading latest llama.cpp\n"

llama_cpp_dir="__llama_cpp_port_${port}__"

if [[ -d "$llama_cpp_dir" && ! -f "$llama_cpp_dir/__ggml_script__" ]]; then
    # if the dir exists and there isn't a file "__ggml_script__" in it, abort
    printf "[-] Directory %s already exists\n" "$llama_cpp_dir"
    printf "[-] Please remove it and try again\n"
    exit 1
elif [[ -d "$llama_cpp_dir" ]]; then
    printf "[+] Directory %s already exists\n" "$llama_cpp_dir"
    printf "[+] Using cached llama.cpp\n"

    cd "$llama_cpp_dir"
    git reset --hard
    git fetch
    git checkout origin/master

    cd ..
else
    printf "[+] Cloning llama.cpp\n"

    git clone https://github.com/ggerganov/llama.cpp "$llama_cpp_dir"
fi

# mark that that the directory is made by this script
touch "$llama_cpp_dir/__ggml_script__"

if [[ $verbose -eq 1 ]]; then
    set -x
fi

# build
cd "$llama_cpp_dir"

make clean

log="--silent"
if [[ $verbose -eq 1 ]]; then
    log=""
fi

if [[ "$backend" == "cuda" ]]; then
    printf "[+] Building with CUDA backend\n"
    GGML_CUDA=1 make -j llama-server $log
elif [[ "$backend" == "cpu" ]]; then
    printf "[+] Building with CPU backend\n"
    make -j llama-server $log
elif [[ "$backend" == "metal" ]]; then
    printf "[+] Building with Metal backend\n"
    make -j llama-server $log
else
    printf "[-] Unknown backend: %s\n" "$backend"
    exit 1
fi

# run the server

printf "[+] Running server\n"

args=""
if [[ "$backend" == "cuda" ]]; then
    export CUDA_VISIBLE_DEVICES=$gpu_id
    args="-ngl 999"
elif [[ "$backend" == "cpu" ]]; then
    args="-ngl 0"
elif [[ "$backend" == "metal" ]]; then
    args="-ngl 999"
else
    printf "[-] Unknown backend: %s\n" "$backend"
    exit 1
fi

if [[ $verbose -eq 1 ]]; then
    args="$args --verbose"
fi

./llama-server -m "../$wfile" --host 0.0.0.0 --port "$port" -c $n_kv -np "$n_parallel" $args

exit 0
