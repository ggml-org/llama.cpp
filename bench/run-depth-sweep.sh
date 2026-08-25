#!/usr/bin/env bash
# Depth sweep: this fork vs pinned upstream, prefill and generation, appended to results.jsonl.
#
# Execution model. Three earlier attempts were lost to the harness rather than the measurement:
# one bad output line killed a whole matrix, a deep cell hung the GPU compute ring, and a 46 minute
# run produced zero rows before the compositor starved and the machine was rebooted. So:
#
#  * One depth per process. llama-bench caches the depth context and reuses it across repetitions
#    and across the pp/tg pair at one depth, so the expensive prefill is still paid once - but rows
#    now land after every depth, the display gets a gap between processes, and a failure costs one
#    depth instead of the run.
#  * Resume. Cells already in the log are skipped, so stopping at any point is cheap.
#  * Per-cell timeout, so we kill a stuck cell before the driver's ring watchdog or the display does.
#  * Pre-flight: nothing else may hold the GPU. GPU-mapping containers are stopped and restored.
#
# Measurement controls, each of which cost a wrong conclusion before:
#
#  * Palindrome arm order (fork mainline mainline fork) so linear clock drift cancels between arms.
#  * A discarded warmup per process; the first run of a set carries a 15-20% boost clock.
#  * Upstream pinned at the commit this fork merged, so the delta is ours and not upstream drift.
#  * ubatch 512 and flash attention on, matching models.ini. ub 2048 hangs the ring at depth >=
#    65536 on this hardware and is slower than 512 at depth anyway.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
OUT=${OUT:-$HERE/results.jsonl}
SUITE=${SUITE:-depth-sweep}

FORK_BIN=${FORK_BIN:-$REPO/build/bin/llama-bench}
# Disk-backed on purpose. This lived under /tmp once, which is tmpfs here: the reboot destroyed
# the reference build, and while it existed it was holding ~10 GB of the 30 GB of system RAM that
# the benchmarks themselves needed.
MAIN_BIN=${MAIN_BIN:-$REPO/../llama.cpp-upstream/build/bin/llama-bench}

# 131072 is deliberately absent: it is unvalidated and belongs to probe-deep.sh.
DEPTHS=${DEPTHS:-0 4096 16384 32768 65536}
NPROMPT=${NPROMPT:-2048}
NGEN=${NGEN:-64}
UBATCH=${UBATCH:-512}
FA=${FA:-1}
REPS=${REPS:-3}       # llama-bench internal repetitions -> stddev_ts within a cell
ROUNDS=${ROUNDS:-1}   # palindromes; each contributes 2 fork and 2 mainline samples per cell
VRAM_IDLE_MB=${VRAM_IDLE_MB:-4096}
GAP_SECONDS=${GAP_SECONDS:-10}          # quiet GPU between processes, so the compositor can breathe
MAX_NODES_PER_SUBMIT=${MAX_NODES_PER_SUBMIT:-}   # unset = driver default (100)
STOP_GPU_CONTAINERS=${STOP_GPU_CONTAINERS:-1}

HF=/home/laurent/.cache/huggingface/hub
MODEL_LABELS=(qwen38-27b-q4kxl ornith-35b-a3b-q4km)
MODEL_PATHS=(
  "$HF/models--unsloth--Qwen3.8-27B-GGUF/snapshots/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/Qwen3.8-27B-UD-Q4_K_XL.gguf"
  "$HF/models--bartowski--Ornith-1.5-35B-A3B-GGUF/snapshots/64b0493d34a5ca4c1b4ad67bb99b41d74b4f07d6/Ornith-1.5-35B-A3B-Q4_K_M.gguf"
)

vram_mb() { echo $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null || echo 0) / 1048576 )); }

# Deeper cells legitimately take longer; anything past this is stuck, and we would rather kill it
# ourselves than let the ring watchdog or the display timeout do it.
budget_for() {
  local d=$1
  if   [ "$d" -le 32768 ]; then echo 600
  elif [ "$d" -le 65536 ]; then echo 1500
  else                          echo 3600
  fi
}

settle() {
  for _ in $(seq 1 120); do
    [ "$(vram_mb)" -le "$VRAM_IDLE_MB" ] && { sleep "$GAP_SECONDS"; return 0; }
    sleep 2
  done
  echo "  warning: VRAM still at $(vram_mb)MB after 240s" >&2
  sleep "$GAP_SECONDS"
}

STOPPED_CONTAINERS=""
restore_containers() {
  [ -n "$STOPPED_CONTAINERS" ] || return 0
  echo "restarting containers: $STOPPED_CONTAINERS"
  for c in $STOPPED_CONTAINERS; do docker start "$c" >/dev/null 2>&1 || echo "  failed to restart $c" >&2; done
  STOPPED_CONTAINERS=""
}
trap restore_containers EXIT INT TERM

preflight() {
  local busy
  busy=$(pgrep -af "bin/llama-server|bin/llama-bench|bin/llama-cli" | grep -v "run-depth-sweep" || true)
  if [ -n "$busy" ]; then
    echo "refusing to start, something already holds the GPU:" >&2
    echo "$busy" | sed 's/^/  /' >&2
    exit 1
  fi

  if [ "$STOP_GPU_CONTAINERS" = 1 ] && command -v docker >/dev/null 2>&1; then
    for c in $(docker ps --format '{{.Names}}' 2>/dev/null); do
      if docker inspect "$c" --format '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' 2>/dev/null \
           | grep -qE "/dev/dri|/dev/kfd"; then
        echo "stopping GPU container: $c"
        docker stop "$c" >/dev/null 2>&1 && STOPPED_CONTAINERS="$STOPPED_CONTAINERS $c"
      fi
    done
  fi

  for bin in "$FORK_BIN" "$MAIN_BIN"; do
    [ -x "$bin" ] || { echo "missing binary: $bin" >&2; exit 1; }
  done

  PREFLIGHT_VRAM=$(vram_mb)
  # Power profile changes throughput by more than most effects measured here, and it can be changed
  # out from under a run. Stamp it on every row so two profiles can never be silently averaged.
  PWR_LEVEL=$(cat /sys/class/drm/card1/device/power_dpm_force_performance_level 2>/dev/null || echo unknown)
  PWR_STATE=$(cat /sys/class/drm/card1/device/power_dpm_state 2>/dev/null || echo unknown)
  echo "preflight ok: vram=${PREFLIGHT_VRAM}MB stopped=[${STOPPED_CONTAINERS:-none}]"
}

# A cell is (model, build, depth, round, slot). The slot matters: a palindrome runs fork twice per
# round, so keying without it would treat the second fork cell as already done and quietly halve the
# samples, taking the drift cancellation with it.
already_done() {
  local label=$1 build=$2 depth=$3 round=$4 slot=$5
  [ -f "$OUT" ] || return 1
  python3 - "$OUT" "$SUITE" "$label" "$build" "$depth" "$round" "$slot" <<'PY'
import json, sys
path, suite, label, build, depth, rnd, slot = sys.argv[1:8]
want = (suite, label, build, int(depth), int(rnd), int(slot))
try:
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r.get("suite"), r.get("model_label"), r.get("build"),
                r.get("n_depth"), r.get("round"), r.get("slot")) == want:
            sys.exit(0)
except FileNotFoundError:
    pass
sys.exit(1)
PY
}

record() {  # record <build> <bin> <label> <path> <depth> <round> <slot>
  local build=$1 bin=$2 label=$3 path=$4 depth=$5 round=$6 slot=$7
  local budget; budget=$(budget_for "$depth")

  settle
  timeout 300 "$bin" -m "$path" -p 512 -n 16 -d 0 -r 1 -fa "$FA" -o jsonl >/dev/null 2>&1 || true

  set +e
  timeout "$budget" "$bin" -m "$path" -p "$NPROMPT" -n "$NGEN" -d "$depth" \
      -ub "$UBATCH" -r "$REPS" -fa "$FA" -o jsonl 2>/dev/null \
    | python3 -c "
import json, sys, datetime
extra = {'build': '$build', 'model_label': '$label', 'round': $round, 'slot': $slot, 'suite': '$SUITE',
         'fa_requested': $FA, 'preflight_vram_mb': ${PREFLIGHT_VRAM:-0},
         'max_nodes_per_submit': '${MAX_NODES_PER_SUBMIT:-default}',
         'power_dpm_level': '${PWR_LEVEL:-unknown}', 'power_dpm_state': '${PWR_STATE:-unknown}',
         'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        print('skipped non-json output line', file=sys.stderr)
        continue
    row.update(extra)
    print(json.dumps(row))
" >> "$OUT"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    [ "$rc" -eq 124 ] && echo "  cell TIMED OUT after ${budget}s: $build/$label d=$depth" >&2 \
                      || echo "  cell failed (rc=$rc): $build/$label d=$depth" >&2
  fi
}

preflight
[ -n "$MAX_NODES_PER_SUBMIT" ] && export GGML_VK_MAX_NODES_PER_SUBMIT="$MAX_NODES_PER_SUBMIT"

for i in "${!MODEL_LABELS[@]}"; do
  label=${MODEL_LABELS[$i]}; path=${MODEL_PATHS[$i]}
  [ -f "$path" ] || { echo "skipping $label, model not found" >&2; continue; }
  for depth in $DEPTHS; do
    for round in $(seq 1 "$ROUNDS"); do
      slot=0
      for build in fork mainline mainline fork; do
        slot=$((slot + 1))
        bin=$FORK_BIN; [ "$build" = mainline ] && bin=$MAIN_BIN
        if already_done "$label" "$build" "$depth" "$round" "$slot"; then
          echo "[$(date +%H:%M:%S)] skip (done) $label d=$depth round=$round slot=$slot $build"
          continue
        fi
        echo "[$(date +%H:%M:%S)] $label d=$depth round=$round slot=$slot $build"
        record "$build" "$bin" "$label" "$path" "$depth" "$round" "$slot"
      done
    done
  done
done
echo "DEPTH_SWEEP_DONE -> $OUT"
