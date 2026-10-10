#!/usr/bin/env bash
# matrix-lib.sh — shared helpers for the 2026-10-06 comprehensive A770 matrix (b12327).
# Source it, then call a sec_* function. Service stays stopped for the whole run; the EXIT trap
# restores sysfs knobs, ccs_mode and the service. A SIGKILL skips the trap: check `systemctl is-active` after.
set -uo pipefail
readonly ROOT=/mnt/nvme1/oneapi-ab/matrix-1006
readonly UNIT="llama-gpu@Ornith-1.5-35B-Q4_K_M.service"
readonly G=/sys/bus/pci/devices/0000:03:00.0/tile0/gt0
readonly M8B=/mnt/ssd2/models/llama31-8b-q4km/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
readonly MORN=/mnt/ssd2/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf
readonly CCS_ORIG=1
RUN_TIMEOUT_S=${RUN_TIMEOUT_S:-1200}
readonly FAULT_RE='Engine reset|timedout|Timedout job|wedged|banned|page ?fault|CAT error|GPU HANG|job timeout'
mkdir -p "$ROOT"

rt() { ssh -n -o BatchMode=yes vinbonesjr "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ROOT/master.log"; }

restore() {
  rt "echo 600 > $G/freq0/min_freq; echo 2400 > $G/freq0/max_freq; echo base > $G/freq0/power_profile;
      echo 1000 > $G/engines/ccs/timeslice_duration_us; echo ${CCS_ORIG} > $G/ccs_mode;
      systemctl start ${UNIT}" >/dev/null 2>&1
  log "restore done; service=$(systemctl is-active ${UNIT}) ccs_mode=$(cat $G/ccs_mode)"
}
init() {
  trap restore EXIT
  rt "systemctl stop ${UNIT}" >/dev/null 2>&1
  sleep 3
  log "init: service=$(systemctl is-active ${UNIT}) ccs_mode=$(cat $G/ccs_mode) min/max=$(cat $G/freq0/min_freq)/$(cat $G/freq0/max_freq)"
}

# Scrub tuning env so every cell starts from the same documented baseline, then apply the production unit env.
clean_env() {
  local v
  for v in $(compgen -e | grep -E '^(GGML_SYCL|SYCL_|UR_|ZE_|ONEAPI_|LLAMA_ARG)'); do unset "$v"; done
  export ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 GGML_SYCL_GRAPH_EVICTION_TIMEOUT=300 \
         GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0 GGML_SYCL_FA_LARGE_GRF=1
}

setccs() { # mode
  local m=$1
  [ "$(cat $G/ccs_mode)" = "$m" ] && return 0
  rt "echo $m > $G/ccs_mode" || { log "ccs_mode=$m write FAILED (EBUSY?)"; return 1; }
  sleep 4
  log "ccs_mode now $(cat $G/ccs_mode)"
}
setfreq() { # min max
  rt "echo 600 > $G/freq0/min_freq; echo $2 > $G/freq0/max_freq; echo $1 > $G/freq0/min_freq"
}
faults() { local n; n=$(rt "dmesg | grep -ciE '$FAULT_RE' || true" 2>/dev/null | tail -1); echo "${n:--1}"; }
# tenancy: non-llama processes using >30% CPU or holding the A770 render node (foreign load shows up in index.tsv)
tenancy() { { ps -eo pcpu,comm --sort=-pcpu | awk 'NR>1 && $1>30 {print $2":"$1}' | grep -vE "^(llama-bench|llama-server|timeout|ps|awk|ssh):"; fuser /dev/dri/renderD128 2>/dev/null | tr -s " " "\n" | grep -E "^[0-9]+$" | while read -r q; do c=$(cat /proc/$q/comm 2>/dev/null); [[ "$c" =~ ^llama- ]] || echo "gpu:$c"; done; } | sort -u | paste -sd, - ; }

# cpu_snap — "total_jiffies busy_jiffies own_child_jiffies": whole-host CPU from /proc/stat and this shell's reaped-children CPU time
cpu_snap() {
  awk '/^cpu / {t=0; for(i=2;i<=NF;i++) t+=$i; b=t-$5-$6; printf "%d %d ", t, b}' /proc/stat
  sed 's/.*) //' /proc/self/stat | awk '{print $14+$15}'
}
# foreign_pct S0 S1 — share of host CPU (percent) used by processes other than this shell's measured children
foreign_pct() { echo "$1 $2" | awk '{dt=$4-$1; db=$5-$2; dc=$6-$3; if(dt>0) printf "%.1f", (db-dc)*100/dt; else print -1}'; }

# wait_quiet — block until CPU idle >= 85% for 3 consecutive 3 s samples (foreign sessions share this host); give up after 25 min and flag.
wait_quiet() {
  local t0 ok=0 idle waited
  t0=$(date +%s)
  while [ $(( $(date +%s)-t0 )) -lt "${QUIET_MAX:-1500}" ]; do
    idle=$(vmstat 3 2 | tail -1 | awk '{print $15}')
    if [ "${idle:-0}" -ge "${QUIET_IDLE:-85}" ]; then ok=$((ok+1)); [ $ok -ge 3 ] && break; else ok=0; fi
    sleep 5
  done
  waited=$(( $(date +%s)-t0 ))
  [ $waited -gt 15 ] && log "wait_quiet: waited ${waited}s (idle=${idle:-?}) foreign=$(tenancy)"
  return 0
}

# cell DIR TAG "ENV=.. ENV=.." CMD... — run one measurement with sampling and a fault check.
cell() {
  local dir=$1 tag=$2 envs=$3; shift 3
  mkdir -p "$ROOT/$dir"
  local out="$ROOT/$dir/$tag" f0 f1 idle load t0 rc pid
  if pgrep -x 'llama-server|llama-bench' >/dev/null; then log "SKIP $tag: foreign llama process"; return 1; fi
  wait_quiet
  f0=$(faults); ten0=$(tenancy)
  idle=$(vmstat 1 4 | tail -1 | awk '{print $15}'); load=$(cut -d' ' -f1 /proc/loadavg)
  t0=$(date +%s); local snap0 snap1; snap0=$(cpu_snap)
  (
    clean_env
    # shellcheck disable=SC2086
    exec env $envs timeout "$RUN_TIMEOUT_S" "$@" > "$out.md" 2> "$out.err"
  ) &
  pid=$!
  ( # sampler: freq / throttle every 2 s, fdinfo once after 25 s
    sleep 25
    for p in $(pgrep -x 'llama-bench|llama-server'); do
      for fd in /proc/$p/fdinfo/*; do grep -qs 'drm-driver:.*xe' "$fd" && grep -E 'drm-(engine|cycles|total-cycles|engine-capacity)-(ccs|rcs|bcs)' "$fd"; done
    done > "$out.fdinfo" 2>/dev/null
  ) &
  local samp=$!
  ( while kill -0 $pid 2>/dev/null; do
      echo "$(date +%s) act=$(cat $G/freq0/act_freq) cur=$(cat $G/freq0/cur_freq) thr=$(cat $G/freq0/throttle/status 2>/dev/null) why=$(cat $G/freq0/throttle/reasons 2>/dev/null | tr ' ' ',')" >> "$out.freq"
      sleep 2
    done ) &
  local fs=$!
  ( vmstat 5 > "$out.vmstat" ) &
  local vs=$!
  wait $pid; rc=$?
  kill $samp $fs $vs 2>/dev/null; pkill -P $vs 2>/dev/null; wait $samp $fs $vs 2>/dev/null
  snap1=$(cpu_snap); LAST_FOREIGN=$(foreign_pct "$snap0" "$snap1")
  LAST_IDLE_RUN=$(awk 'NR>3 && $15 ~ /^[0-9]+$/ {s+=$15; n++} END {if(n) printf "%.0f", s/n; else print -1}' "$out.vmstat" 2>/dev/null)
  f1=$(faults); ten1=$(tenancy)
  printf '%s\t%s\t%s\tidle=%s\tload=%s\tfaults+%s\trc=%s\t%ss\tforeign=%s|%s\tt0=%s\tt1=%s\tavail=%s\tcache=%s\ttemp=%s\tidlerun=%s\tforeignpct=%s\n' "$dir" "$tag" "$envs" "$idle" "$load" "$((f1-f0))" "$rc" "$(( $(date +%s)-t0 ))" "$ten0" "$ten1" "$t0" "$(date +%s)" "$(free -m | awk '/Mem:/{print $7}')" "$(free -m | awk '/Mem:/{print $6}')" "$(cat /sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon0/temp3_input)" "$LAST_IDLE_RUN" "$LAST_FOREIGN" >> "$ROOT/index.tsv"
  log "cell $dir/$tag rc=$rc faults+$((f1-f0)) idle=$idle $(grep -E 'pp512 |tg64 ' "$out.md" | awk -F'|' '{printf "%s=%s ", $(NF-2), $(NF-1)}' | tr -s ' ')"
  [ "$((f1-f0))" -gt 0 ] && { log "FAULT after $tag, pausing 30 s"; sleep 30; }
  return 0
}

B8=( /usr/bin/llama-bench -m "$M8B" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 64 -d 0,8192 -r 3 -t 12 -o md )
BORN=( /usr/bin/llama-bench -m "$MORN" -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 -p 512 -n 64 -d 0,8192 -r 3 -t 12 -o md )

# rotate_arms ROUND arms... — echo arms rotated left by ROUND so no arm is always first.
rotate_arms() {
  local r=$1; shift; local n=$#; local a=("$@"); local i
  for ((i=0;i<n;i++)); do echo "${a[$(( (i+r) % n ))]}"; done
}

# cell_clean DIR TAG ENVS CMD... — rerun a cell (up to 3 attempts) until mean CPU idle during the run >= CLEAN_IDLE (default 92); bad attempts kept as .badN
cell_clean() {
  local dir=$1 tag=$2 a ext
  for a in 1 2 3; do
    cell "$@"
    if awk -v f="${LAST_FOREIGN:--1}" -v m="${CLEAN_FOREIGN:-4}" 'BEGIN{exit !(f>=0 && f<=m)}'; then return 0; fi
    log "cell_clean $dir/$tag attempt $a: foreign CPU ${LAST_FOREIGN}% > ${CLEAN_FOREIGN:-4}% (idle ${LAST_IDLE_RUN}%)"
    [ $a = 3 ] && { log "cell_clean $dir/$tag kept DIRTY (foreign ${LAST_FOREIGN}%)"; return 0; }
    for ext in md err freq fdinfo vmstat; do [ -e "$ROOT/$dir/$tag.$ext" ] && mv "$ROOT/$dir/$tag.$ext" "$ROOT/$dir/$tag.bad$a.$ext"; done
    sleep 120
  done
}
