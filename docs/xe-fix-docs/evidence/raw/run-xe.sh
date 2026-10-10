#!/usr/bin/env bash
# Two rounds of the installed build (N, 4e7400c3a) on the xe KMD, 2026-09-29.
# Same bench.sh / flags as run-n2.sh; quiet-host wait kept (load < 4 for 3 min).
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
echo "kmd=$(basename "$(readlink /sys/bus/pci/devices/0000:03:00.0/driver)") ccs_preempt_us=$(cat /sys/bus/pci/devices/0000:03:00.0/tile0/gt0/engines/ccs/preempt_timeout_us) ccs_job_ms=$(cat /sys/bus/pci/devices/0000:03:00.0/tile0/gt0/engines/ccs/job_timeout_ms) min_freq=$(cat /sys/class/drm/card0/device/tile0/gt0/freq0/min_freq)"
for r in 1 2; do
  wait_quiet; echo "round $r start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg) others=$(pgrep -c -f 'llama-(server|bench)') swap_used=$(free -g | awk '/Swap/{print $3}')G"
  /mnt/nvme1/oneapi-ab/bench.sh "N-xe-r$r"
  echo "round $r end $(date +%T)"
done
