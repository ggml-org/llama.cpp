#!/usr/bin/env bash
# Second lever rung (2026-09-29): submission-model knobs, default copy engine.
# Goal: localize the hang/reset (UR in-order lists, counter-based events, immediate
# cmdlists, NEO direct submission) and see which restores decode. One Ornith round each.
set -uo pipefail
export BENCH_TIMEOUT=420
run() { local tag="$1"; shift; echo "== $tag start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) env: $*"; env "$@" /mnt/nvme1/oneapi-ab/bench-xe.sh "$tag"; echo "== $tag end $(date +%T)"; }
run N-xe-inorder0     UR_L0_USE_DRIVER_INORDER_LISTS=0
run N-xe-cbevents0    UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS=0
run N-xe-immcl0       UR_L0_USE_IMMEDIATE_COMMANDLISTS=0
run N-xe-nodirectsub  NEOReadDebugKeys=1 EnableDirectSubmission=0
run N-xe-norelaxed    NEOReadDebugKeys=1 DirectSubmissionRelaxedOrdering=0
echo "LEVERS2 DONE"
