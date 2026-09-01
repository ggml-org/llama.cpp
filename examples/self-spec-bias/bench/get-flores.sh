#!/usr/bin/env bash
# Fetch FLORES-200 devtest files.
#
#   bash get-flores.sh eng_Latn zho_Hans
#   FLORES_DIR=/tmp/flores bash get-flores.sh eng_Latn deu_Latn jpn_Jpan
#
# Language codes are the FLORES-200 ones, <iso639-3>_<script>. Files already
# present are left alone, so this is safe to call on every run.
#
# FLORES-200 is CC-BY-SA 4.0, https://github.com/facebookresearch/flores
set -u

FLORES_DIR="${FLORES_DIR:-$(cd "$(dirname "$0")" && pwd)/data/flores200}"
FLORES_URL="${FLORES_URL:-https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz}"

# devtest is a fixed size, anything else means a bad extract
FLORES_LINES="${FLORES_LINES:-1012}"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <lang_Script> [lang_Script ...]" >&2
    echo "  e.g. $0 eng_Latn zho_Hans" >&2
    exit 1
fi

mkdir -p "$FLORES_DIR"

missing=()
for lang in "$@"; do
    if [ ! -f "$FLORES_DIR/$lang.devtest" ]; then
        missing+=("$lang")
    fi
done

if [ "${#missing[@]}" -eq 0 ]; then
    echo "flores: all requested files already in $FLORES_DIR"
    exit 0
fi

echo "flores: fetching ${missing[*]} (25 MB archive)"
echo "  from $FLORES_URL"
echo "  CC-BY-SA 4.0, https://github.com/facebookresearch/flores"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if ! curl -fL --max-time 600 -o "$tmp/flores.tar.gz" "$FLORES_URL"; then
    echo "flores: download failed" >&2
    exit 1
fi

members=()
for lang in "${missing[@]}"; do
    members+=("./flores200_dataset/devtest/$lang.devtest")
done

if ! tar -xzf "$tmp/flores.tar.gz" -C "$tmp" "${members[@]}"; then
    echo "flores: extract failed, check the language codes" >&2
    exit 1
fi

for lang in "${missing[@]}"; do
    src=$tmp/flores200_dataset/devtest/$lang.devtest
    n=$(wc -l < "$src")
    if [ "$n" -ne "$FLORES_LINES" ]; then
        echo "flores: $lang.devtest has $n lines, expected $FLORES_LINES" >&2
        exit 1
    fi
    mv "$src" "$FLORES_DIR/$lang.devtest"
    echo "flores: $FLORES_DIR/$lang.devtest ($n lines)"
done
