#!/bin/zsh
# Usage: run_bench_env.sh <build_dir> <label> <env assignments...> -- <extra llama-bench args...>
# Prints: <label>|rss1 rss2 rss3|median_rss|pp_tps|tg_tps
set -e
BUILD=$1; LABEL=$2; shift 2
MODEL="/Volumes/Julian T7/models/gemma-4-12B-it-qat-unified-mtp-Q5_K_M-telemetry.gguf"
BENCH=$BUILD/bin/llama-bench
ENVV=()
EXTRA=()
seen_dd=false
for a in "$@"; do
  if [ "$a" = "--" ]; then seen_dd=true; continue; fi
  if $seen_dd; then EXTRA+=("$a"); else ENVV+=("$a"); fi
done
RSS=(); PPS=(); TGS=()
for i in 1 2 3; do
  OUT=$(env "${ENVV[@]}" /usr/bin/time -l "$BENCH" -m "$MODEL" -p 512 -n 32 "${EXTRA[@]}" 2>&1 || true)
  rss=$(echo "$OUT" | grep -iE "maximum resident set size" | awk '{print $1}')
  pp=$(echo "$OUT"  | awk -F'|' '/pp512/ {gsub(/ ±.*/,"",$(NF-1)); gsub(/ /,"",$(NF-1)); print $(NF-1); exit}')
  tg=$(echo "$OUT"  | awk -F'|' '/tg32/  {gsub(/ ±.*/,"",$(NF-1)); gsub(/ /,"",$(NF-1)); print $(NF-1); exit}')
  RSS+=("${rss:-0}"); PPS+=("${pp:-0}"); TGS+=("${tg:-0}")
done
medr=$(printf '%s\n' "${RSS[@]}" | sort -n | awk 'NR==2{print; exit}')
[ -z "$medr" ] && medr=$(printf '%s\n' "${RSS[@]}" | sort -n | head -1)
medp=$(printf '%s\n' "${PPS[@]}" | sort -g | awk 'NR==2{print; exit}')
medt=$(printf '%s\n' "${TGS[@]}" | sort -g | awk 'NR==2{print; exit}')
echo "${LABEL}|${RSS[1]} ${RSS[2]} ${RSS[3]}|${medr}|${medp}|${medt}"
