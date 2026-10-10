#!/usr/bin/env bash
# Waits for a quiet host (1-min load < 4 for 3 consecutive minutes, max 3 h),
# then benchmarks A (2026.0, no DNN), C (2026.1, no DNN), D (2026.1, DNN) back to back.
set -uo pipefail
quiet=0
for ((m = 0; m < 180; m++)); do
  load=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l="$load" 'BEGIN{exit !(l < 4)}'; then quiet=$((quiet + 1)); else quiet=0; fi
  [[ $quiet -ge 3 ]] && break
  sleep 60
done
echo "start $(date +%T) quiet=$quiet load=$(cut -d' ' -f1-3 /proc/loadavg) swap_used=$(free -g | awk '/Swap/{print $3}')G"
for v in "D:/mnt/nvme1/llama-sycl-build/D/build/llama.cpp-sycl-f16-git/src/build/bin:/mnt/nvme1/llama-sycl-build/D/build/llama.cpp-sycl-f16-git/src/build/bin" "C:/mnt/nvme1/llama-sycl-build/C/build/llama.cpp-sycl-f16-git/src/build/bin:/mnt/nvme1/llama-sycl-build/C/build/llama.cpp-sycl-f16-git/src/build/bin" "A:$(cat /mnt/nvme1/oneapi-ab/A.ldpath):/mnt/nvme1/oneapi-ab/A-pkg/usr/bin"; do
  IFS=: read -r tag rest <<< "$v"
  bin="${v##*:}"; ldp="${v#*:}"; ldp="${ldp%:*}"
  echo "== $tag  load before: $(cut -d' ' -f1-3 /proc/loadavg)"
  /mnt/nvme1/oneapi-ab/bench.sh "${tag}-final2" "$ldp" "$bin"
done
