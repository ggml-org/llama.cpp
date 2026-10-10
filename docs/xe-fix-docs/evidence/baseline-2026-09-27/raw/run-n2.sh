#!/usr/bin/env bash
# Two rounds of the installed build (N, 4e7400c3a), each after a quiet-host wait.
set -uo pipefail
wait_quiet() {
  local quiet=0
  for ((m = 0; m < 180; m++)); do
    if awk -v l="$(cut -d' ' -f1 /proc/loadavg)" 'BEGIN{exit !(l < 4)}'; then quiet=$((quiet + 1)); else quiet=0; fi
    [[ $quiet -ge 3 ]] && return 0
    sleep 60
  done
  return 1
}
for r in 1 2; do
  wait_quiet; echo "round $r start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) others=$(pgrep -c -f 'llama-(server|bench)')"
  /mnt/nvme1/oneapi-ab/bench.sh "N-clean-r$r"
done
