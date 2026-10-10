#!/usr/bin/env bash
# Whole-campaign monitors: kernel log stream, fault marker file, 5 s time series (GPU temp, fan, energy, freq, throttle reasons, mem, load, foreign CPU).
R=/mnt/nvme1/oneapi-ab/matrix-1006/monitor; mkdir -p "$R"
G=/sys/bus/pci/devices/0000:03:00.0/tile0/gt0; H=/sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon0
RE='Engine reset|timedout|Timedout job|wedged|banned|page ?fault|CAT error|GPU HANG|job timeout|GT[0-9]: reset|i915.*(hang|timeout)'
( ssh -n -o BatchMode=yes vinbonesjr "dmesg -w --time-format iso" 2>&1 | tee "$R/dmesg-stream.log" | grep --line-buffered -iE "$RE" | grep -v --line-buffered 'ccs_mode_store\|Setting compute' | while read -r l; do echo "$(date +%s) $l" >> "$R/FAULTS.txt"; done ) &
while true; do
  printf '%s temp2=%s temp3=%s fan1=%s energy2=%s act=%s cur=%s min=%s max=%s prof=%s why=%s avail=%s cache=%s load=%s foreign=%s\n' \
    "$(date +%s)" "$(cat $H/temp2_input)" "$(cat $H/temp3_input)" "$(cat $H/fan1_input)" "$(cat $H/energy2_input)" \
    "$(cat $G/freq0/act_freq)" "$(cat $G/freq0/cur_freq)" "$(cat $G/freq0/min_freq)" "$(cat $G/freq0/max_freq)" "$(cat $G/freq0/power_profile | tr -d ' ')" \
    "$(cat $G/freq0/throttle/reasons | tr ' ' ,)" "$(free -m | awk '/Mem:/{print $7}')" "$(free -m | awk '/Mem:/{print $6}')" "$(cut -d' ' -f1 /proc/loadavg)" \
    "$(ps -eo pcpu,comm --sort=-pcpu | awk 'NR>1 && $1>30 {print $2":"$1}' | grep -vE '^(llama-bench|llama-server|timeout|ps|awk|ssh|python3|bash|curl):' | paste -sd, -)" >> "$R/series.log"
  sleep 5
done
