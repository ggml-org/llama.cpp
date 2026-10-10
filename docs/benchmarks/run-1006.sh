#!/usr/bin/env bash
# run-1006.sh PHASE ORDER... — A=b12321 (old), B=b12327 (new); service stopped, restored by EXIT trap; deadline 25 min.
set -uo pipefail
readonly PHASE="$1"; shift
readonly UNIT="llama-gpu@Ornith-1.5-35B-Q4_K_M.service"
readonly MODEL="/mnt/ssd2/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf"
readonly OUT="/mnt/nvme1/oneapi-ab/b12327-${PHASE}"
readonly OLD=/mnt/nvme1/oneapi-ab/old-b12321/usr
readonly DEADLINE=$(( $(date +%s) + 1500 ))
mkdir -p "$OUT"
ssh -o BatchMode=yes vinbonesjr "systemctl stop ${UNIT}"
trap 'ssh -o BatchMode=yes vinbonesjr "systemctl start ${UNIT}"; echo "service: $(systemctl is-active ${UNIT})" >> "$OUT/done"' EXIT
echo "ccs_mode=$(cat /sys/bus/pci/devices/0000:03:00.0/tile0/gt0/ccs_mode)" > "$OUT/meta"
i=0
for v in "$@"; do
  i=$((i+1)); [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "deadline before run $i" >> "$OUT/meta"; break; }
  if [ "$v" = A ]; then BIN=$OLD/bin; LDP=$OLD/lib; else BIN=/usr/bin; LDP=/usr/lib; fi
  vmstat 1 5 | tail -1 | awk -v r=$i -v v=$v '{print "run",r,v,"idle",$15,"load",""}' >> "$OUT/meta"; cut -d' ' -f1 /proc/loadavg >> "$OUT/meta"
  env LD_LIBRARY_PATH="$LDP" ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 GGML_SYCL_GRAPH_EVICTION_TIMEOUT=300 \
    GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0 GGML_SYCL_FA_LARGE_GRF=1 \
    timeout 900 "$BIN/llama-bench" -m "$MODEL" -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 --moe-cache off --load-mode none \
    -p 512 -n 64 -d 0 -r 3 -t 12 -o md > "$OUT/run$i-$v.md" 2> "$OUT/run$i-$v.err"
  echo "run $i $v exit $?" >> "$OUT/meta"
done
