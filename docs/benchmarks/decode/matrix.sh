#!/usr/bin/env bash
# Plain-decode matrix: NEW (installed b12321, production env) vs OLD (b12305 extracted) under the host's
# current load, then with injected busy cores. Service stopped throughout; EXIT trap restarts it.
set -uo pipefail
readonly UNIT=llama-gpu@Ornith-1.5-35B-Q4_K_M.service D=/mnt/nvme1/oneapi-ab/decode
readonly MODEL=/mnt/ssd2/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf OLD=$D/old-4e7400c3a
readonly DEADLINE=1350   # seconds; skip remaining runs past this (background tasks die at ~30 min)
busy=()
load_on()  { local n=$1; for ((i = 0; i < n; i++)); do bash -c 'while :; do :; done' & busy+=($!); done; }
load_off() { for p in "${busy[@]:-}"; do [[ -n $p ]] && kill "$p" 2>/dev/null; done; busy=(); wait 2>/dev/null; }
trap 'load_off; ssh -o BatchMode=yes vinbonesjr "systemctl start $UNIT"; echo "service: $(systemctl is-active $UNIT) $(date +%T)"' EXIT
run() { # tag build injected_cores depth reps
  local tag=$1 build=$2 inj=$3 depth=$4 reps=$5
  if (( SECONDS > DEADLINE )); then echo "== $tag SKIPPED (deadline)"; return; fi
  mkdir -p "$D/$tag"
  load_on "$inj"; sleep 3
  local idle0; idle0=$(vmstat 3 2 | tail -1 | awk '{print $15}')
  echo "== $tag build=$build injected=$inj depth=$depth start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) idle=${idle0}%"
  local common=(ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 GGML_SYCL_GRAPH_EVICTION_TIMEOUT=300)
  local args=(-m "$MODEL" -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 -t 12 -r "$reps" -p 0 -n 64 -d "$depth" -o md)
  if [[ $build == new ]]; then
    env "${common[@]}" GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0 GGML_SYCL_FA_LARGE_GRF=1 timeout 900 /usr/bin/llama-bench "${args[@]}" > "$D/$tag/bench.md" 2> "$D/$tag/bench.err"
  else
    env "${common[@]}" LD_LIBRARY_PATH=$OLD/usr/lib timeout 900 $OLD/usr/bin/llama-bench "${args[@]}" > "$D/$tag/bench.md" 2> "$D/$tag/bench.err"
  fi
  local rc=$?; load_off
  echo "   exit=$rc end $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) idle=$(vmstat 3 2 | tail -1 | awk '{print $15}')% :: $(rg -o 'tg64[^|]*\| *[0-9.]+ ± [0-9.]+' "$D/$tag/bench.md" | tr -s ' ' | head -2 | tr '\n' ' ')"
}
ssh -o BatchMode=yes vinbonesjr "systemctl stop $UNIT"; sleep 5
echo "gpu users after stop: $(fuser /dev/dri/renderD129 2>/dev/null | wc -w)"
run warm-new new 0 0 1
run warm-old old 0 0 1
run A-new1 new 0 0 5
run A-old1 old 0 0 5
run A-new2 new 0 0 5
run A-old2 old 0 0 5
run B-new-inj4 new 4 0 5
run B-old-inj4 old 4 0 5
run C-new-inj10 new 10 0 5
run A-new3 new 0 0 5
run D-new-d8k new 0 8192 3
run D-old-d8k old 0 8192 3
echo "matrix done $(date +%T) elapsed ${SECONDS}s"
