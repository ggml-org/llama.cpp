#!/usr/bin/env bash
# matrix5-run.sh — phase 5: GGML_SYCL_MAX_WG_PER_CU sweep on Ornith at depth (decode gain at 16k-64k seen with 32).
source /mnt/nvme1/oneapi-ab/matrix-lib.sh
init; setccs 1
sec_wg() {
  log "== sec wg"; RUN_TIMEOUT_S=5400
  local r v
  for r in 0 1; do
    for v in $(rotate_arms $r 16 32 64 24 48 128); do
      [ $r = 1 ] && case $v in 24|48|128) continue;; esac
      cell wg "wg$v.r$r" "GGML_SYCL_MAX_WG_PER_CU=$v" /usr/bin/llama-bench -m "$MORN" -fitt 1024 -fitc 131072 -ctk q8_0 -ctv q8_0 -fa 1 --moe-cache off --load-mode none \
        -p 0 -n 64 -d 0,8192,16384,32768,65536 -r 2 -t 12 -o md
    done
  done
}
sec_wg8b() { # same knob on the 8B (head dim 128) at depth
  log "== sec wg8b"
  local r v
  for r in 0 1; do for v in $(rotate_arms $r 16 32 64); do
    cell wg8b "wg$v.r$r" "GGML_SYCL_MAX_WG_PER_CU=$v" /usr/bin/llama-bench -m "$M8B" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 64 -d 0,8192,16384,32768 -r 3 -t 12 -o md
  done; done
}
for s in wg wg8b; do "sec_$s"; rt "dmesg" > "$ROOT/klog5-$s.txt" 2>&1; log "klog5 saved ($s) faults=$(faults)"; done
log "PHASE5 DONE"
