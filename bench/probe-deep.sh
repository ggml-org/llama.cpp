#!/usr/bin/env bash
# Opt-in probe of the depth the main sweep deliberately excludes.
#
# 131072 has never completed on this hardware. At ubatch 2048 depth 65536 already hangs the compute
# ring, and a full unbounded ladder starved the Wayland compositor until the machine was rebooted.
# So this is separate, single-cell, watchdogged, and run deliberately with nothing else on the GPU.
# It writes to the same log under suite "deep-probe" so it never mixes into the matrix.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
DEPTH=${DEPTH:-131072} BUDGET=${BUDGET:-5400} UBATCH=${UBATCH:-512} FA=${FA:-1} REPS=${REPS:-1}

SUITE=deep-probe DEPTHS="$DEPTH" ROUNDS=1 \
  REPS="$REPS" UBATCH="$UBATCH" FA="$FA" \
  OUT=${OUT:-$HERE/results.jsonl} \
  "$HERE/run-depth-sweep.sh"
