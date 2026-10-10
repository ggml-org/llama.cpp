#!/usr/bin/env bash
# UR Level Zero debug trace of the failing regime (tg64 @ 8k, default env), for an
# upstream report. Log goes to a file, not the bench stderr.
set -uo pipefail
export BENCH_TIMEOUT=600
OUT=/mnt/nvme1/oneapi-ab/N-xe-trace; mkdir -p "$OUT"
echo "== N-xe-trace start $(date +%T)"
env UR_LOG_LEVEL_ZERO="level:debug;flush:debug;output:file,${OUT}/ur-l0-debug.log" UR_L0_DEBUG=1 \
  /mnt/nvme1/oneapi-ab/bench-xe.sh N-xe-trace
echo "== N-xe-trace end $(date +%T) log=$(du -h ${OUT}/ur-l0-debug.log 2>/dev/null | cut -f1)"
