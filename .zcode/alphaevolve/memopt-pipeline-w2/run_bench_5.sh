#!/bin/zsh
# 5-run bench, prints all 5 RSS values + median
set -e
BUILD=$1; LABEL=$2; shift 2
MODEL="/Volumes/Julian T7/models/gemma-4-12B-it-qat-unified-mtp-Q5_K_M-telemetry.gguf"
BENCH=$BUILD/bin/llama-bench
ENVV=(); EXTRA=(); seen_dd=false
for a in "$@"; do
  if [ "$a" = "--" ]; then seen_dd=true; continue; fi
  if $seen_dd; then EXTRA+=("$a"); else ENVV+=("$a"); fi
done
RSS=()
for i in 1 2 3 4 5; do
  OUT=$(env "${ENVV[@]}" /usr/bin/time -l "$BENCH" -m "$MODEL" -p 512 -n 32 "${EXTRA[@]}" 2>&1 || true)
  rss=$(echo "$OUT" | grep -iE "maximum resident set size" | awk '{print $1}')
  RSS+=("${rss:-0}")
done
medr=$(printf '%s\n' "${RSS[@]}" | sort -n | awk 'NR==3{print; exit}')
echo "${LABEL}|${RSS[*]}|${medr}"
