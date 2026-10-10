#!/usr/bin/env bash
# Wait (max ~22 min, service up) for a genuinely quiet host: instantaneous idle >= 92% AND loadavg1 < 3
# for 9 consecutive 20 s samples. Then stop the service and run NEW/OLD plain decode interleaved x2.
set -uo pipefail
readonly UNIT=llama-gpu@Ornith-1.5-35B-Q4_K_M.service D=/mnt/nvme1/oneapi-ab/decode
readonly MODEL=/mnt/ssd2/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf OLD=$D/old-4e7400c3a
ok=0; stopped=0
trap '[[ $stopped -eq 1 ]] && { ssh -o BatchMode=yes vinbonesjr "systemctl start $UNIT"; echo "service: $(systemctl is-active $UNIT) $(date +%T)"; }' EXIT
for ((n = 0; n < 66; n++)); do
  idle=$(vmstat 5 2 | tail -1 | awk '{print $15}'); l1=$(cut -d' ' -f1 /proc/loadavg)
  if [[ $idle -ge 92 ]] && awk -v l="$l1" 'BEGIN{exit !(l < 3)}'; then ok=$((ok + 1)); else ok=0; fi
  (( n % 9 == 0 )) && echo "wait $(date +%T) idle=${idle}% load1=$l1 consecutive_ok=$ok"
  [[ $ok -ge 9 ]] && break
  sleep 15
done
if [[ $ok -lt 9 ]]; then echo "NO QUIET WINDOW in ~22 min (last idle=${idle}% load1=$l1); nothing run"; exit 3; fi
echo "quiet window found $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg)"
ssh -o BatchMode=yes vinbonesjr "systemctl stop $UNIT"; stopped=1; sleep 5
run() { local tag=$1 build=$2; mkdir -p "$D/$tag"
  echo "== $tag build=$build start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) idle=$(vmstat 3 2 | tail -1 | awk '{print $15}')% "
  local common=(ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 GGML_SYCL_GRAPH_EVICTION_TIMEOUT=300)
  local args=(-m "$MODEL" -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 -t 12 -r 5 -p 0 -n 64 -d 0 -o md)
  if [[ $build == new ]]; then env "${common[@]}" GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0 GGML_SYCL_FA_LARGE_GRF=1 timeout 600 /usr/bin/llama-bench "${args[@]}" > "$D/$tag/bench.md" 2> "$D/$tag/bench.err"
  else env "${common[@]}" LD_LIBRARY_PATH=$OLD/usr/lib timeout 600 $OLD/usr/bin/llama-bench "${args[@]}" > "$D/$tag/bench.md" 2> "$D/$tag/bench.err"; fi
  echo "   exit=$? end $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) idle=$(vmstat 3 2 | tail -1 | awk '{print $15}')% :: $(rg -o 'tg64[^|]*\| *[0-9.]+ ± [0-9.]+' "$D/$tag/bench.md" | tr -s ' ' | head -1)"; }
run Q-warm-new new
run Q-new1 new; run Q-old1 old; run Q-new2 new; run Q-old2 old
echo "quiet set done $(date +%T)"
