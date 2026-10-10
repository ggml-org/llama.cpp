#!/usr/bin/env bash
# Phase 1: wait for a quiet host, stop the service, warm up, run off1,on1; leave the service stopped.
# Phase 2: run off2,on2, restart the service. Any failure/kill path restarts the service.
set -uo pipefail
readonly PHASE=${1:?phase 1|2} UNIT=llama-gpu@Ornith-1.5-35B-Q4_K_M.service OUT=/mnt/nvme1/oneapi-ab/grfbench
readonly MODEL=/mnt/ssd2/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf
keep_stopped=0
trap '[[ $keep_stopped -eq 1 ]] || { ssh -o BatchMode=yes vinbonesjr "systemctl start $UNIT"; echo "service: $(systemctl is-active $UNIT) $(date +%T)"; }' EXIT
bench() { # tag knob reps tests...
  local tag=$1 knob=$2 reps=$3; shift 3
  mkdir -p "$OUT/$tag"
  echo "== $tag knob=$knob start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) idle=$(vmstat 3 2 | tail -1 | awk '{print $15}')% top: $(ps -eo pcpu,comm --sort=-pcpu | sed -n 2,3p | tr -s ' ' | tr '\n' ';')"
  ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 GGML_SYCL_GRAPH_EVICTION_TIMEOUT=300 \
  GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0 GGML_SYCL_FA_LARGE_GRF=$knob \
    timeout 1500 /usr/bin/llama-bench -m "$MODEL" -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 -t 12 -r "$reps" -o md "$@" \
    > "$OUT/$tag/bench.md" 2> "$OUT/$tag/bench.err"
  echo "   exit=$? end $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) idle=$(vmstat 3 2 | tail -1 | awk '{print $15}')% :: $(rg -o 'FA_LARGE_GRF: [0-9]+ \([^)]*\)' "$OUT/$tag/bench.err" | head -1)"
}
if [[ $PHASE == 1 ]]; then
  quiet=0
  for ((m = 0; m < 20; m++)); do
    idle=$(vmstat 5 2 | tail -1 | awk '{print $15}')
    if [[ $idle -ge 60 ]]; then quiet=$((quiet + 1)); else quiet=0; fi
    [[ $quiet -ge 3 ]] && break
  done
  echo "idle-cpu gate done $(date +%T) consecutive_ok=$quiet idle=${idle}% load=$(cut -d' ' -f1-3 /proc/loadavg)"
  [[ $quiet -ge 3 ]] || { echo "host never quiet; not starting"; exit 3; }
  ssh -o BatchMode=yes vinbonesjr "systemctl stop $UNIT"; sleep 5
  echo "gpu users after stop: $(fuser /dev/dri/renderD129 2>/dev/null | wc -w)"
  bench warmup 0 1 -p 512 -n 0 -d 0
  bench off1 0 5 -p 128,512 -n 64 -d 0,8192
  bench on1  1 5 -p 128,512 -n 64 -d 0,8192
  keep_stopped=1; echo "phase 1 done $(date +%T); service left stopped for phase 2"
else
  echo "gpu users: $(fuser /dev/dri/renderD129 2>/dev/null | wc -w); service $(systemctl is-active $UNIT)"
  bench off2 0 5 -p 128,512 -n 64 -d 0,8192
  bench on2  1 5 -p 128,512 -n 64 -d 0,8192
fi
