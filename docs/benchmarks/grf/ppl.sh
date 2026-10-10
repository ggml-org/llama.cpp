#!/usr/bin/env bash
# PPL with the 256-GRF FA kernels off vs on (runtime knob, same binary). Service stopped meanwhile.
set -uo pipefail
readonly UNIT=llama-gpu@Ornith-1.5-35B-Q4_K_M.service B=/mnt/nvme1/llama-sycl-build/build/llama.cpp-sycl-f16-git/src/build/bin OUT=/mnt/nvme1/oneapi-ab/grf
readonly MODEL=/mnt/ssd2/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf
readonly WIKI=/mnt/mrgr/home/svnbjrn/.cache/pr51-integration/wikitext-2-raw/wiki.test.raw
trap 'ssh -o BatchMode=yes vinbonesjr "systemctl start $UNIT"; echo "service: $(systemctl is-active $UNIT)"' EXIT
ssh -o BatchMode=yes vinbonesjr "systemctl stop $UNIT"
for grf in 0 1; do
  echo "== GGML_SYCL_FA_LARGE_GRF=$grf start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  LD_LIBRARY_PATH=$B ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 GGML_SYCL_FA_LARGE_GRF=$grf \
    timeout 2400 $B/llama-perplexity -m $MODEL -f $WIKI -c 4096 -b 512 -ub 512 --chunks 10 \
    --fit on --fit-target 2048 -fa on -ctk q8_0 -ctv q8_0 -t 12 > $OUT/ppl-grf$grf.log 2>&1
  echo "exit=$? $(rg -o 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $OUT/ppl-grf$grf.log)"
  rg -m1 'FA_LARGE_GRF' $OUT/ppl-grf$grf.log | cut -c1-160
done
