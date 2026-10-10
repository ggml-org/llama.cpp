#!/usr/bin/env bash
# matrix6-run.sh — phase 6: clean-host reruns (cell_clean: retry until mean idle >= CLEAN_IDLE) of everything that ran contaminated or only once.
source /mnt/nvme1/oneapi-ab/matrix-lib.sh
init; setccs 1
export QUIET_IDLE=${QUIET_IDLE:-90} QUIET_MAX=${QUIET_MAX:-900} CLEAN_IDLE=${CLEAN_IDLE:-92}

sec_wgclean() { # MAX_WG_PER_CU 16/24/32 on Ornith at depth, 4 rotated rounds
  log "== sec wgclean"; RUN_TIMEOUT_S=5400
  local r v
  for r in 0 1 2; do
    for v in $(rotate_arms $r 16 24 32 64); do
      cell_clean wgclean "wg$v.r$r" "GGML_SYCL_MAX_WG_PER_CU=$v" /usr/bin/llama-bench -m "$MORN" -fitt 1024 -fitc 131072 -ctk q8_0 -ctv q8_0 -fa 1 \
        --moe-cache off --load-mode none -p 0 -n 64 -d 0,16384,32768,65536 -r 2 -t 12 -o md
    done
  done
}

sec_mkl32kclean() {
  log "== sec mkl32kclean"; RUN_TIMEOUT_S=3600
  local r v
  for r in 0 1; do for v in $(rotate_arms $r 1 0); do
    cell_clean mkl32kclean "llama31-8b.mkl$v.r$r" "GGML_SYCL_ENABLE_MKL_FA=$v" /usr/bin/llama-bench -m "$M8B" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 0 -d 32768 -r 2 -t 12 -o md
    cell_clean mkl32kclean "ornith35.mkl$v.r$r" "GGML_SYCL_ENABLE_MKL_FA=$v" /usr/bin/llama-bench -m "$MORN" -fitt 1024 -fitc 131072 -ctk q8_0 -ctv q8_0 -fa 1 --moe-cache off --load-mode none -p 512 -n 0 -d 32768 -r 2 -t 12 -o md
  done; done
}

sec_q8clean() {
  log "== sec q8clean"
  local MQ8=/mnt/mrgr/models/ornith-1.5-35b-a3b-uncensored-q8_0.gguf/Ornith-1.5-35B-A3B-uncensored-Q8_0.gguf
  local arms=( "q8_base:" "q8_mkl0:GGML_SYCL_ENABLE_MKL_FA=0" "q8_grf0:GGML_SYCL_FA_LARGE_GRF=0" "q8_wg32:GGML_SYCL_MAX_WG_PER_CU=32" ) a r
  for r in 0 1 2; do
    while read -r a; do
      cell_clean q8clean "${a%%:*}.r$r" "${a#*:}" /usr/bin/llama-bench -m "$MQ8" -fitt 1024 -fitc 131072 -ctk q8_0 -ctv q8_0 -fa 1 --moe-cache off --load-mode none -p 512 -n 64 -d 0,8192 -r 3 -t 12 -o md
    done < <(rotate_arms $r "${arms[@]}")
  done
}

sec_wg8bclean() { # phase-5 round 1 of the 8B WG cells ran at 75% idle
  log "== sec wg8bclean"
  local r v
  for r in 0 1; do for v in $(rotate_arms $r 16 32 64); do
    cell_clean wg8bclean "wg$v.r$r" "GGML_SYCL_MAX_WG_PER_CU=$v" /usr/bin/llama-bench -m "$M8B" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 64 -d 0,8192,16384,32768 -r 3 -t 12 -o md
  done; done
}

# srv6 NAME MODEL ENV ARGS — server on :8095; prompt set at temp 0 and 0.6; retry once if mean idle during the launch < CLEAN_IDLE
srv6() {
  local name=$1 model=$2 envs=$3 args=$4 d="$ROOT/spec6" up=0 sp i vs idle attempt
  mkdir -p "$d"
  for attempt in 1 2; do
    wait_quiet; up=0
    # no process may hold the A770 render node (a stray server keeps its whole VRAM allocation)
    for i in $(seq 1 12); do h=$(fuser /dev/dri/renderD128 2>/dev/null | tr -s " " "\n" | grep -E "^[0-9]+$" | while read -r q; do [[ "$(cat /proc/$q/comm 2>/dev/null)" =~ ^llama- ]] && echo $q; done); [ -z "$h" ] && break; sleep 5; done
    [ -n "$h" ] && log "srv6 $name: stray GPU holder(s) $h still present, aborting launch" && return 0
    ( clean_env
      # shellcheck disable=SC2086
      exec env $envs timeout 3600 /usr/bin/llama-server --model "$model" --alias bench --parallel 1 --flash-attn on --threads 12 --threads-batch 12 \
        --jinja --host 127.0.0.1 --port 8095 -ctk q8_0 -ctv q8_0 $args > "$d/$name.a$attempt.server.log" 2>&1 ) &
    sp=$!
    for i in $(seq 1 240); do curl -sf -m 3 http://127.0.0.1:8095/health >/dev/null && { up=1; break; }; kill -0 $sp 2>/dev/null || break; sleep 5; done
    if [ $up != 1 ]; then log "srv6 $name FAILED: $(grep -iE 'error|abort|Failed' "$d/$name.a$attempt.server.log" | head -2 | cut -c1-160 | tr '\\n' ' ')"; pkill -P $sp 2>/dev/null; kill $sp 2>/dev/null; sleep 6; return 0; fi
    ( vmstat 5 > "$d/$name.a$attempt.vmstat" ) & vs=$!
    svpid=$(pgrep -P $sp | head -1); g0=$(awk '/^cpu / {t=0; for(i=2;i<=NF;i++) t+=$i; printf "%d %d", t, t-$5-$6}' /proc/stat); o0=$(sed 's/.*) //' /proc/$svpid/stat | awk '{print $12+$13}')
    rm -f "$d/$name.a$attempt.jsonl"
    for t in 0 0.6; do
      SPEC_TEMP=$t SPEC_TOP_P=$([ "$t" = 0 ] && echo 1.0 || echo 0.95) python3 /mnt/nvme1/oneapi-ab/matrix2-spec.py 8095 "$name.t$t" "$d/raw" >> "$d/$name.a$attempt.jsonl" 2>> "$d/$name.a$attempt.client.err"
    done
    g1=$(awk '/^cpu / {t=0; for(i=2;i<=NF;i++) t+=$i; printf "%d %d", t, t-$5-$6}' /proc/stat); o1=$(sed 's/.*) //' /proc/$svpid/stat | awk '{print $12+$13}')
    fpct=$(echo "$g0 $g1 $o0 $o1" | awk '{dt=$3-$1; db=$4-$2; dc=$6-$5; if(dt>0) printf "%.1f", (db-dc)*100/dt; else print -1}')
    kill $vs 2>/dev/null; pkill -P $vs 2>/dev/null
    pkill -P $sp 2>/dev/null; kill $sp 2>/dev/null; sleep 3; kill -9 $(pgrep -P $sp) 2>/dev/null; wait $sp 2>/dev/null; sleep 6
    idle=$(awk 'NR>3 && $15 ~ /^[0-9]+$/ {s+=$15;n++} END{if(n) printf "%.0f", s/n; else print -1}' "$d/$name.a$attempt.vmstat")
    log "srv6 $name attempt $attempt idle=$idle foreign=${fpct}% faults=$(faults)"
    if awk -v f="$fpct" -v m="${CLEAN_FOREIGN:-18}" 'BEGIN{exit !(f>=0 && f<=m)}' || [ $attempt = 2 ]; then
      cat "$d/$name.a$attempt.jsonl" | sed "s/\$/ /" | python3 -c "import sys,json;[print(json.dumps({**json.loads(l),'idle_run':$idle,'foreign_pct':$fpct,'attempt':$attempt})) for l in sys.stdin if l.strip()]" >> "$d/results.jsonl"
      return 0
    fi
    sleep 120
  done
}

sec_specclean() { # none / ngram-mod / ngram-simple, 3 launches each, rotated order, production flags
  log "== sec specclean"
  local K="--ctx-size 131072 --fit on --fit-target 1024 --moe-cache off --load-mode none" r c
  for r in 0 1 2; do
    for c in $(rotate_arms $r none ngmod ngsimple); do
      case $c in none) a="--spec-type none";; ngmod) a="--spec-type ngram-mod";; ngsimple) a="--spec-type ngram-simple";; esac
      srv6 "$c.r$r" "$MORN" "" "$K $a"
    done
  done
}

for s in "$@"; do "sec_$s"; rt "dmesg" > "$ROOT/klog6-$s.txt" 2>&1; log "klog6 saved ($s) faults=$(faults)"; done
log "PHASE6 DONE: $*"
