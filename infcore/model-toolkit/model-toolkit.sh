#!/usr/bin/env bash
# infcore model-toolkit — офлайн-обёртки над инструментами llama.cpp (build-time).
# Не редактирует апстрим; вызывает штатные бинари из нашей сборки. Всё локально, без сети.
set -euo pipefail

BUILD="${INFCORE_BUILD:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/build}"
BIN="${BUILD}/bin"

die() { echo "model-toolkit: $*" >&2; exit 1; }

need() {
  local tool="$1"
  [ -x "${BIN}/${tool}" ] || die "не найден ${BIN}/${tool} — соберите профилем (LLAMA_BUILD_TOOLS=ON) или задайте INFCORE_BUILD"
}

usage() {
  cat >&2 <<EOF
Использование: model-toolkit.sh <команда> [аргументы...]
  quantize <in.gguf> <out.gguf> <type> [--imatrix f.imatrix] [прочие флаги llama-quantize]
  split    <in.gguf> <out-prefix> [--split-max-size 20G | --split-max-tensors N]
  merge    <shard-00001-of-000NN.gguf> <out.gguf>
  imatrix  [флаги llama-imatrix: -m ... -f calib.txt -o out.imatrix]
  export-lora [флаги llama-export-lora: -m base.gguf --lora a.gguf -o out.gguf]
INFCORE_BUILD=${BUILD}
EOF
  exit 2
}

[ $# -ge 1 ] || usage
cmd="$1"; shift

case "${cmd}" in
  quantize)
    need llama-quantize
    # llama-quantize принимает [--imatrix file] перед позиционными; прокидываем всё как есть.
    exec "${BIN}/llama-quantize" "$@"
    ;;
  split)
    need llama-gguf-split
    exec "${BIN}/llama-gguf-split" --split "$@"
    ;;
  merge)
    need llama-gguf-split
    exec "${BIN}/llama-gguf-split" --merge "$@"
    ;;
  imatrix)
    need llama-imatrix
    exec "${BIN}/llama-imatrix" "$@"
    ;;
  export-lora)
    need llama-export-lora
    exec "${BIN}/llama-export-lora" "$@"
    ;;
  -h|--help|help) usage ;;
  *) die "неизвестная команда '${cmd}' (см. --help)" ;;
esac
