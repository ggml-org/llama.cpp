#!/usr/bin/env bash
# bench.sh TAG [LD_LIBRARY_PATH] [BINDIR] — offline llama-bench on Ornith, service stopped.
set -uo pipefail
readonly TAG="$1" LDP="${2:-}" BIN="${3:-/usr/bin}"
readonly UNIT="llama-gpu@Ornith-1.5-35B-Q4_K_M.service"
readonly MODEL="/mnt/ssd2/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf"
readonly OUT="/mnt/nvme1/oneapi-ab/${TAG}"
mkdir -p "${OUT}"
ssh -o BatchMode=yes vinbonesjr "systemctl stop ${UNIT}"
trap 'ssh -o BatchMode=yes vinbonesjr "systemctl start ${UNIT}"; echo "service: $(systemctl is-active ${UNIT})"' EXIT
env ${LDP:+LD_LIBRARY_PATH="${LDP}"} ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH="${GRAPH:-1}" \
  timeout -k 15 "${BENCH_TIMEOUT:-3600}" "${BIN}/llama-bench" -m "${MODEL}" -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 \
  -p 512 -n 64 -d 0,8192 -r 5 -t 12 -o md > "${OUT}/bench.md" 2> "${OUT}/bench.err"
echo "llama-bench exit $?"
cat "${OUT}/bench.md"
rg -n 'GGML_SYCL_DNNL|GGML_SYCL_GRAPH|ENABLE_GRAPH|compiler|IntelLLVM' "${OUT}/bench.err" | head -6
