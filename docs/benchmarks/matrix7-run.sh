#!/usr/bin/env bash
# matrix7-run.sh — output check for GGML_SYCL_MAX_WG_PER_CU=32 vs 16: 8B perplexity (deterministic) and Ornith long-context greedy generations.
source /mnt/nvme1/oneapi-ab/matrix-lib.sh
init; setccs 1
export QUIET_IDLE=80 QUIET_MAX=600

sec_ppl8b() {
  log "== sec ppl8b"
  local r v txt="$ROOT/correct/text.txt"
  for r in 0 1; do for v in $(rotate_arms $r 16 32); do
    cell wgcheck "ppl8b.wg$v.r$r" "GGML_SYCL_MAX_WG_PER_CU=$v" /usr/bin/llama-perplexity -m "$M8B" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 8192 -b 2048 -ub 512 --chunks 4 -f "$txt"
  done; done
}

sec_long() {
  log "== sec long"
  local d="$ROOT/wgcheck" n v tag sp up i h
  mkdir -p "$d"
  for n in 0 1 2 3; do
    case $n in 0|3) v=16;; *) v=32;; esac
    tag="wg$v.l$n"
    wait_quiet
    h=$(fuser /dev/dri/renderD128 2>/dev/null | tr -s " " "\n" | grep -E "^[0-9]+$" | while read -r q; do [[ "$(cat /proc/$q/comm 2>/dev/null)" =~ ^llama- ]] && echo $q; done)
    [ -n "$h" ] && { log "long $tag: stray GPU holder $h, skipping"; continue; }
    ( clean_env
      exec env GGML_SYCL_MAX_WG_PER_CU=$v timeout 5400 /usr/bin/llama-server --model "$MORN" --alias bench --parallel 1 --flash-attn on --threads 12 --threads-batch 12 \
        --jinja --host 127.0.0.1 --port 8096 -ctk q8_0 -ctv q8_0 --ctx-size 131072 --fit on --fit-target 1024 --moe-cache off --load-mode none --spec-type none \
        > "$d/$tag.server.log" 2>&1 ) &
    sp=$!; up=0
    for i in $(seq 1 240); do curl -sf -m 3 http://127.0.0.1:8096/health >/dev/null && { up=1; break; }; kill -0 $sp 2>/dev/null || break; sleep 5; done
    if [ $up = 1 ]; then
      python3 /mnt/nvme1/oneapi-ab/longcheck.py 8096 "$tag" "$d/raw" >> "$d/long-results.jsonl" 2> "$d/$tag.client.err"
      log "long $tag done faults=$(faults)"
    else log "long $tag server FAILED"; fi
    pkill -P $sp 2>/dev/null; kill $sp 2>/dev/null; sleep 3; kill -9 $(pgrep -P $sp) 2>/dev/null; wait $sp 2>/dev/null; sleep 6
  done
}
for s in "$@"; do "sec_$s"; done
log "PHASE7 DONE: $*"
