#!/usr/bin/env bash
# Waits for matrix-run.sh to finish, then benchmarks spec-type / moe-cache combos on the production server config.
while kill -0 72813 2>/dev/null; do sleep 30; done
source /mnt/nvme1/oneapi-ab/matrix-lib.sh
init
setccs 1
PORT=8091; D=$ROOT/ngram; mkdir -p "$D"
cfgs=(
 "none|--spec-type none --moe-cache off --load-mode none"
 "ngmod|--spec-type ngram-mod --moe-cache off --load-mode none"
 "ngmod_small|--spec-type ngram-mod --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-min 16 --spec-ngram-mod-n-max 32 --moe-cache off --load-mode none"
 "ngsimple|--spec-type ngram-simple --moe-cache off --load-mode none"
 "ngmapk|--spec-type ngram-map-k --moe-cache off --load-mode none"
 "ngmapk4v|--spec-type ngram-map-k4v --moe-cache off --load-mode none"
 "ngcache|--spec-type ngram-cache --moe-cache off --load-mode none"
 "none_mcauto|--spec-type none --moe-cache auto --load-mode none"
 "ngmod_mcauto|--spec-type ngram-mod --moe-cache auto --load-mode none"
 "ngmod_lmauto|--spec-type ngram-mod --moe-cache off --load-mode auto"
 "ngmod_prefetch4|--spec-type ngram-mod --moe-cache off --load-mode none --prefetch-experts-slots 4"
)
for r in 0 1; do
 for c in "${cfgs[@]}"; do
  n=${c%%|*}; a=${c#*|}
  ( clean_env
    exec timeout 1500 /usr/bin/llama-server --model "$MORN" --alias bench --ctx-size 131072 --parallel 1 --fit on --fit-target 1024 \
      --flash-attn on --threads 12 --threads-batch 12 --jinja --temp 0.6 --top-p 0.95 --host 127.0.0.1 --port $PORT \
      -ctk q8_0 -ctv q8_0 $a > "$D/$n.r$r.server.log" 2>&1 ) &
  sp=$!
  up=0; for i in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null && { up=1; break; }; kill -0 $sp 2>/dev/null || break; sleep 5; done
  if [ $up = 1 ]; then
    f0=$(faults)
    python3 /mnt/nvme1/oneapi-ab/matrix-ngram.py $PORT "$n.r$r" >> "$D/results.jsonl" 2> "$D/$n.r$r.client.err"
    log "ngram $n.r$r done faults+$(( $(faults)-f0 ))"
  else log "ngram $n.r$r server failed to start"; fi
  kill $sp 2>/dev/null; sleep 2; pkill -x llama-server 2>/dev/null; sleep 5
 done
done
log "NGRAM DONE"
