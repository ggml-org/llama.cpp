#!/usr/bin/env bash
# --moe-cache variants with the copy engine off (2026-09-29 17:50). llama-bench default is auto.
set -uo pipefail
export BENCH_TIMEOUT=600 UR_L0_USE_COPY_ENGINE=0
run() { local tag="$1"; shift; echo "== $tag start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) extra: $*"; BENCH_EXTRA="$*" /mnt/nvme1/oneapi-ab/bench-xe2.sh "$tag"; echo "== $tag end $(date +%T)"; }
run N-xe-copyeng0-moesoft --moe-cache soft
run N-xe-copyeng0-moeoff  --moe-cache off
echo "MOECACHE DONE"
