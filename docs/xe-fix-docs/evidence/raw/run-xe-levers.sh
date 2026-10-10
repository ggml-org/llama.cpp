#!/usr/bin/env bash
# xe decode-regression levers on Ornith, one bench.sh round each (2026-09-29).
# Baseline is N-xe-r1/r2 (same flags). Env passes through bench-xe.sh.
set -uo pipefail
run() { local tag="$1"; shift; export BENCH_TIMEOUT=420; echo "== $tag start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) env: $*"; env "$@" /mnt/nvme1/oneapi-ab/bench-xe.sh "$tag"; echo "== $tag end $(date +%T)"; }
run N-xe-copyeng0   UR_L0_USE_COPY_ENGINE=0
run N-xe-pinned0    GGML_SYCL_ENABLE_HOST_PINNED_MEM=0
run N-xe-graph0     GRAPH=0
