#!/usr/bin/env bash
#
# Shortcut for downloading HF models
#
# Usage:
#   ./llama-cli -m $(./scripts/hf.sh https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/resolve/main/mixtral-8x7b-v0.1.Q4_K_M.gguf)
#   ./llama-cli -m $(./scripts/hf.sh --url https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/blob/main/mixtral-8x7b-v0.1.Q4_K_M.gguf)
#   ./llama-cli -m $(./scripts/hf.sh --repo TheBloke/Mixtral-8x7B-v0.1-GGUF --file mixtral-8x7b-v0.1.Q4_K_M.gguf)
#

# all logs go to stderr
function log {
    echo "$@" 1>&2
}

function usage {
    log "Usage: $0 [[--url] <url>] [--repo <repo>] [--file <file>] [--outdir <dir> [-h|--help]"
    exit 1
}

# check for curl or wget
function has_cmd {
    if ! [ -x "$(command -v $1)" ]; then
        return 1
    fi
}

if has_cmd wget; then
    cmd="wget -q -c -O %s/%s %s"
elif has_cmd curl; then
    cmd="curl -C - -f --output-dir %s -o %s -L %s"
else
    log "[E] curl or wget not found"
    exit 1
fi

url=""
repo=""
file=""
outdir="."

# parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --url)
            url="$2"
            shift 2
            ;;
        --repo)
            repo="$2"
            shift 2
            ;;
        --file)
            file="$2"
            shift 2
            ;;
        --outdir)
            outdir="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            url="$1"
            shift
            ;;
    esac
done

if [ -n "$repo" ] && [ -n "$file" ]; then
    url="https://huggingface.co/$repo/resolve/main/$file"
fi

if [ -z "$url" ]; then
    log "[E] missing --url"
    usage
fi

# check if the URL is a HuggingFace model, and if so, try to download it
is_url=false

if [[ ${#url} -gt 22 ]]; then
    if [[ ${url:0:22} == "https://huggingface.co" ]]; then
        is_url=true
    fi
fi

if [ "$is_url" = false ]; then
    log "[E] invalid URL, must start with https://huggingface.co"
    exit 0
fi

# replace "blob/main" with "resolve/main"
url=${url/blob\/main/resolve\/main}

basename=$(basename $url)

log "[+] attempting to download $basename"

if [ -n "$cmd" ]; then
    cmd=$(printf "$cmd" "$outdir" "$basename" "$url")
    log "[+] $cmd"
    if $cmd; then
        echo $outdir/$basename
        exit 0
    fi
fi

log "[-] failed to download"

exit 1
