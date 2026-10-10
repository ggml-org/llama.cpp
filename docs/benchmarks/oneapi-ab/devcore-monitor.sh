#!/usr/bin/env bash
# Poll /sys/class/devcoredump every 3 s; copy any dump immediately (unread dumps are freed after ~5 min).
R=/mnt/nvme1/oneapi-ab/matrix-1006/monitor; mkdir -p "$R"
while true; do
  for d in /sys/class/devcoredump/devcd*; do
    [ -e "$d" ] || continue
    t=$(date +%s); echo "$t $d" >> "$R/devcoredump.log"
    ssh -n -o BatchMode=yes vinbonesjr "head -c 20000000 $d/data" > "$R/devcoredump-$t.bin" 2>>"$R/devcoredump.log"
    ssh -n -o BatchMode=yes vinbonesjr "echo 1 > $d/data" 2>/dev/null
  done
  sleep 3
done
