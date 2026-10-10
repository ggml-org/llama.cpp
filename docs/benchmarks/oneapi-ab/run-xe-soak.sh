#!/usr/bin/env bash
# Soak: three Ornith rounds on xe with UR_L0_USE_COPY_ENGINE=0 (2026-09-29 16:40).
set -uo pipefail
export BENCH_TIMEOUT=420
for r in 2 3 4; do
  echo "== N-xe-copyeng0-r$r start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  env UR_L0_USE_COPY_ENGINE=0 /mnt/nvme1/oneapi-ab/bench-xe.sh "N-xe-copyeng0-r$r"
  echo "== N-xe-copyeng0-r$r end $(date +%T)"
done
echo "SOAK DONE"
