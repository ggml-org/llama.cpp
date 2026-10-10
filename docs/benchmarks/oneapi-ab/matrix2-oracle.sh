#!/usr/bin/env bash
# Runs after phase 2: CPU-oracle gate with the MKL FA route on and off (the in-phase sec_oracle died: setvars.sh under set -u).
P=$(cat /mnt/nvme1/oneapi-ab/matrix2.pid); while kill -0 "$P" 2>/dev/null; do sleep 30; done
source /mnt/nvme1/oneapi-ab/matrix-lib.sh
init; setccs 1
GATE=/mnt/nvme1/llama-sycl-build/build/llama.cpp-sycl-f16-git/src/build/bin/test-sycl-turbo-correctness
mkdir -p "$ROOT/oracle"
for v in 1 0; do
  ( set +u; clean_env; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    GGML_SYCL_ENABLE_MKL_FA=$v SYCL_CACHE_PERSISTENT=1 GGML_SYCL_ENABLE_GRAPH=1 timeout 1500 "$GATE" > "$ROOT/oracle/gate.mklfa$v.log" 2>&1 )
  log "oracle MKL_FA=$v: $(grep -E 'summary|GATE-FAIL' "$ROOT/oracle/gate.mklfa$v.log" | tail -2 | tr '\n' ' ')"
done
rt "dmesg" > "$ROOT/klog-oracle2.txt" 2>&1
log "ORACLE DONE"
