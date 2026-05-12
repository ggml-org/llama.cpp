#!/bin/bash
# llama.cpp Hexagon test entry script for QDC Linux IoT (BASH framework).
#
# Placeholders substituted by run_qdc_jobs.py (--platform linux) before upload:
#   {MODEL_URL}   direct URL to a .gguf model file
#   {TEST_MODE}   bench | backend-ops | all
#
# QDC extracts the artifact zip to /data/local/tmp/TestContent/ and invokes
# this script via: /bin/bash /data/local/tmp/TestContent/run_linux.sh
# Any files written under /data/local/tmp/QDC_logs/ are auto-uploaded.

set +e
umask 022

LOG_DIR=/data/local/tmp/QDC_logs
BUNDLE_DIR=/data/local/tmp/TestContent/llama_cpp_bundle
MODEL_DIR=/data/local/tmp/gguf
MODEL_PATH="$MODEL_DIR/model.gguf"
RESULTS_XML="$LOG_DIR/results.xml"

mkdir -p "$LOG_DIR" "$MODEL_DIR"
# Redirect all parent-shell output to script.log so QDC auto-uploads it;
# per-case runs still capture their own stdout/stderr into dedicated logs.
exec > "$LOG_DIR/script.log" 2>&1

echo "=== env ==="
date -u
uname -a
pwd

mount -o rw,remount / 2>/dev/null || true

cd "$BUNDLE_DIR" || { echo "FATAL: bundle missing at $BUNDLE_DIR"; exit 1; }
chmod +x bin/* 2>/dev/null
export LD_LIBRARY_PATH="$BUNDLE_DIR/lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$BUNDLE_DIR/lib"
export GGML_HEXAGON_EXPERIMENTAL=1

echo "=== download model ==="
MODEL_URL="{MODEL_URL}"
if [ -z "$MODEL_URL" ]; then
  echo "No model URL provided, skipping download"
elif [ ! -f "$MODEL_PATH" ]; then
  curl -L -fS --retry 3 --retry-delay 5 -o "$MODEL_PATH" "$MODEL_URL"
  curl_rc=$?
  if [ $curl_rc -ne 0 ]; then
    echo "FATAL: model download failed (rc=$curl_rc)"
    exit 1
  fi
  ls -la "$MODEL_PATH"
fi

# ---------------------------------------------------------------------------
# JUnit XML helpers
# ---------------------------------------------------------------------------

xml_open() {
  printf '%s\n' \
    '<?xml version="1.0" encoding="utf-8"?>' \
    "<testsuites>" \
    "<testsuite name=\"llama_cpp_linux\">" \
    > "$RESULTS_XML"
}

xml_close() {
  printf '%s\n' '</testsuite>' '</testsuites>' >> "$RESULTS_XML"
}

xml_case_pass() {
  local classname=$1 name=$2
  printf '<testcase classname="%s" name="%s"/>\n' "$classname" "$name" >> "$RESULTS_XML"
}

xml_case_fail() {
  local classname=$1 name=$2 rc=$3 logfile=$4
  {
    printf '<testcase classname="%s" name="%s">\n' "$classname" "$name"
    printf '<failure message="exit %s"><![CDATA[\n' "$rc"
    tail -c 4096 "$logfile" 2>/dev/null | sed 's/]]>/]] >/g'
    printf '\n]]></failure>\n</testcase>\n'
  } >> "$RESULTS_XML"
}

# Map backend name -> "NDEV --device" pair. "none" means no offload (CPU).
backend_env() {
  case "$1" in
    cpu) echo "0 none" ;;
    gpu) echo "0 GPUOpenCL" ;;
    npu) echo "1 HTP0" ;;
  esac
}

# Append a diagnostic block when a per-case `timeout N` fires (rc=124). The
# naked log file at that point usually just ends mid-OpenCL-init with no
# stderr, which is hard to read in CI summaries.
note_timeout_if_triggered() {
  local rc=$1 budget=$2 log=$3
  [ "$rc" -eq 124 ] || return 0
  {
    printf '\n'
    printf '=== TIMEOUT after %ss ===\n' "$budget"
    printf 'uptime: '; uptime 2>/dev/null
    printf 'free -m:\n'; free -m 2>/dev/null
    printf 'loadavg: '; cat /proc/loadavg 2>/dev/null
  } >> "$log"
}

run_bench_case() {
  local name=$1
  local parts=($(backend_env "$name"))
  local ndev=${parts[0]} device=${parts[1]}
  local log="$LOG_DIR/llama_bench_${name}.log"
  echo "=== [bench:$name] llama-bench --device $device (NDEV=$ndev) ==="
  timeout 600 env GGML_HEXAGON_NDEV=$ndev ./bin/llama-bench \
      -m "$MODEL_PATH" \
      --device "$device" \
      -ngl 99 \
      --batch-size 128 \
      -t 4 \
      -p 128 \
      -n 32 \
      > "$log" 2>&1
  local rc=$?
  note_timeout_if_triggered "$rc" 600 "$log"
  if [ $rc -eq 0 ]; then
    xml_case_pass bench "llama_bench_${name}"
  else
    xml_case_fail bench "llama_bench_${name}" "$rc" "$log"
  fi
}

run_backend_ops_case() {
  local dtype=$1
  local log="$LOG_DIR/backend_ops_${dtype}.log"
  local pattern
  case "$dtype" in
    q4_0)
      # Matches Android: exclude a known-broken shape on NPU.
      pattern='^(?=.*type_a=q4_0)(?!.*type_b=f32,m=576,n=512,k=576).*$'
      ;;
    *)
      pattern="type_a=${dtype}"
      ;;
  esac
  echo "=== [backend-ops:$dtype] test-backend-ops -b HTP0 -o MUL_MAT ==="
  timeout 600 env GGML_HEXAGON_NDEV=1 GGML_HEXAGON_HOSTBUF=0 ./bin/test-backend-ops \
      -b HTP0 -o MUL_MAT -p "$pattern" \
      > "$log" 2>&1
  local rc=$?
  note_timeout_if_triggered "$rc" 600 "$log"
  if [ $rc -eq 0 ]; then
    xml_case_pass backend_ops "$dtype"
  else
    xml_case_fail backend_ops "$dtype" "$rc" "$log"
  fi
}

xml_open

case "{TEST_MODE}" in
  bench)
    for b in cpu gpu npu; do run_bench_case "$b"; done
    ;;
  backend-ops)
    for d in mxfp4 fp16 q4_0; do run_backend_ops_case "$d"; done
    ;;
  all)
    for b in cpu gpu npu; do run_bench_case "$b"; done
    for d in mxfp4 fp16 q4_0; do run_backend_ops_case "$d"; done
    ;;
  *)
    echo "FATAL: unsupported TEST_MODE={TEST_MODE}"
    ;;
esac

xml_close
echo "=== done ==="
# Host parses results.xml to decide pass/fail.
exit 0
