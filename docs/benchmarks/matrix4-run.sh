#!/usr/bin/env bash
# matrix4-run.sh SECTION... — phase 4: reconcile bench vs real-text decode, spec-type at production sampling, MKL at 32k, verify-cost path.
source /mnt/nvme1/oneapi-ab/matrix-lib.sh
init; setccs 1
PROMPT="$ROOT/realtext-prompt.txt"
man ls 2>/dev/null | col -bx | head -c 2300 > "$PROMPT"

sec_realtext() { # llama-completion on a ~560-token man-page prompt, temp 0, production flags: reconcile with bench tg64 and the server rows
  log "== sec realtext"; mkdir -p "$ROOT/realtext"
  local r m out
  for r in 0 1 2; do
    for m in $(rotate_arms $r 1 2); do
      setccs "$m" || continue
      out="$ROOT/realtext/ccs$m.r$r"
      cell realtext "ccs$m.r$r" "" /usr/bin/llama-completion -m "$MORN" -fitt 1024 -fitc 131072 -c 16384 -ctk q8_0 -ctv q8_0 -fa on --moe-cache off --load-mode none \
        -t 12 -f "$PROMPT" -n 256 --temp 0 --top-k 1 -no-cnv --no-display-prompt --perf
      # llama-completion prints llama_perf lines to stderr (.err); extract t/s for the table
      grep -E 'prompt eval time|eval time' "$ROOT/realtext/ccs$m.r$r.err" | grep -v 'prompt' | tail -1 >> "$ROOT/realtext/summary.txt"
      grep -E 'prompt eval time' "$ROOT/realtext/ccs$m.r$r.err" | tail -1 >> "$ROOT/realtext/summary.txt"
    done
  done
  setccs 1
}

srv4() { # NAME MODEL ENV ARGS — server on :8094, prompt set at temp 0 and at production sampling (0.6 / 0.95)
  local name=$1 model=$2 envs=$3 args=$4 d="$ROOT/spec4" up=0 sp i f0 t
  wait_quiet; mkdir -p "$d"
  ( clean_env
    # shellcheck disable=SC2086
    exec env $envs timeout 3600 /usr/bin/llama-server --model "$model" --alias bench --parallel 1 --flash-attn on --threads 12 --threads-batch 12 \
      --jinja --host 127.0.0.1 --port 8094 -ctk q8_0 -ctv q8_0 $args > "$d/$name.server.log" 2>&1 ) &
  sp=$!
  for i in $(seq 1 200); do curl -sf -m 3 http://127.0.0.1:8094/health >/dev/null && { up=1; break; }; kill -0 $sp 2>/dev/null || break; sleep 5; done
  if [ $up = 1 ]; then
    f0=$(faults)
    for t in 0 0.6; do
      SPEC_TEMP=$t SPEC_TOP_P=$([ "$t" = 0 ] && echo 1.0 || echo 0.95) python3 /mnt/nvme1/oneapi-ab/matrix2-spec.py 8094 "$name.t$t" "$d/raw" >> "$d/results.jsonl" 2>> "$d/$name.client.err"
    done
    log "srv4 $name done faults+$(( $(faults)-f0 ))"
  else log "srv4 $name FAILED: $(grep -iE 'error|abort|Failed' "$d/$name.server.log" | head -2 | cut -c1-160 | tr '\n' ' ')"; fi
  pkill -P $sp 2>/dev/null; kill $sp 2>/dev/null; sleep 3; kill -9 $(pgrep -P $sp) 2>/dev/null; wait $sp 2>/dev/null; sleep 6
}

sec_specreal() { # spec-type under production sampling, ccs_mode 1 vs 2 for the none baseline
  log "== sec specreal"
  local K="--ctx-size 131072 --fit on --fit-target 1024 --moe-cache off --load-mode none" r
  for r in 0; do
    setccs 1
    srv4 "ccs1_none.r$r"     "$MORN" "" "$K --spec-type none"
    srv4 "ccs1_ngsimple.r$r" "$MORN" "" "$K --spec-type ngram-simple"
    srv4 "ccs1_ngmod.r$r"    "$MORN" "" "$K --spec-type ngram-mod"
    srv4 "ccs1_mcauto_none.r$r" "$MORN" "" "--ctx-size 131072 --fit on --fit-target 1024 --moe-cache auto --load-mode none --spec-type none"
    srv4 "ccs1_mcauto_ngmod.r$r" "$MORN" "" "--ctx-size 131072 --fit on --fit-target 1024 --moe-cache auto --load-mode none --spec-type ngram-mod"
    setccs 2
    srv4 "ccs2_none.r$r"     "$MORN" "" "$K --spec-type none"
  done
  setccs 1
}

sec_mkl32k() { # MKL FA at 32k depth, 8B and Ornith, q8_0
  log "== sec mkl32k"
  local r v
  for r in 0 1; do for v in 1 0; do
    cell mkl32k "llama31-8b.mkl$v.r$r" "GGML_SYCL_ENABLE_MKL_FA=$v" /usr/bin/llama-bench -m "$M8B" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 0 -d 32768 -r 2 -t 12 -o md
    cell mkl32k "ornith35.mkl$v.r$r" "GGML_SYCL_ENABLE_MKL_FA=$v" /usr/bin/llama-bench -m "$MORN" -fitt 1024 -fitc 131072 -ctk q8_0 -ctv q8_0 -fa 1 --moe-cache off --load-mode none -p 512 -n 0 -d 32768 -r 2 -t 12 -o md
  done; done
}

sec_costpath() { # which MoE path the verify batches hit: fusion / graph arms with profile counters
  log "== sec costpath"
  local arms=( "base:" "FUSION0:GGML_SYCL_ENABLE_FUSION=0" "FFN0:GGML_SYCL_FFN_FUSION=0" "GRAPH0:GGML_SYCL_ENABLE_GRAPH=0" "PROFILE:GGML_SYCL_FFN_FUSION_PROFILE=1 GGML_SYCL_GRAPH_PROFILE=1" )
  local r a
  for r in 0 1 2; do
    while read -r a; do
      cell costpath "${a%%:*}.r$r" "${a#*:}" /usr/bin/llama-bench -m "$MORN" -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 --moe-cache off --load-mode none \
        -p 1,2,3,4,6,8,12,17,32,64 -n 0 -r 3 -t 12 -o md
    done < <(rotate_arms $r "${arms[@]}")
  done
}

sec_q8() { # the Q8_0 uncensored model (the phase-2 fitc section pointed at an empty directory)
  log "== sec q8"
  local MQ8=/mnt/mrgr/models/ornith-1.5-35b-a3b-uncensored-q8_0.gguf/Ornith-1.5-35B-A3B-uncensored-Q8_0.gguf
  local arms=( "q8_base:" "q8_mkl0:GGML_SYCL_ENABLE_MKL_FA=0" "q8_grf0:GGML_SYCL_FA_LARGE_GRF=0" "q8_mc_auto:" ) a r
  for r in 0 1; do
    while read -r a; do
      local mc="--moe-cache off"; [ "${a%%:*}" = q8_mc_auto ] && mc="--moe-cache auto"
      # shellcheck disable=SC2086
      cell q8 "${a%%:*}.r$r" "${a#*:}" /usr/bin/llama-bench -m "$MQ8" -fitt 1024 -fitc 131072 -ctk q8_0 -ctv q8_0 -fa 1 $mc --load-mode none -p 512 -n 64 -d 0,8192 -r 3 -t 12 -o md
    done < <(rotate_arms $r "${arms[@]}")
  done
}

sec_correct3() { # Ornith MKL FA on/off perplexity with VRAM headroom (phase-3 attempt aborted in ggml_sycl_pool_vmm::alloc)
  log "== sec correct3"
  local txt="$ROOT/correct/text.txt" r kv v
  for r in 0 1; do for kv in q8_0 f16; do for v in 1 0; do
    cell correct3 "ornith35.$kv.mkl$v.r$r" "GGML_SYCL_ENABLE_MKL_FA=$v" /usr/bin/llama-perplexity -m "$MORN" -fitt 3072 -fitc 32768 --moe-cache off --load-mode none \
      -fa on -ctk $kv -ctv $kv -c 8192 -b 512 -ub 512 --chunks 3 -f "$txt"
  done; done; done
}

sec_prefetchng() { # the original crash config: prefetch + ngram-mod (earlier ngmod_prefetch4 arm); vary fit-target and VMM
  log "== sec prefetchng"
  local K="--ctx-size 131072 --fit on --moe-cache off --load-mode none --spec-type ngram-mod --prefetch-experts-slots 4" r
  for r in 0; do
    srv4 "pfng_fit1024.r$r"       "$MORN" "" "--fit-target 1024 $K"
    srv4 "pfng_fit3072.r$r"       "$MORN" "" "--fit-target 3072 $K"
    srv4 "pfng_vmm0.r$r"          "$MORN" "GGML_SYCL_ENABLE_VMM=0" "--fit-target 1024 $K"
    srv4 "pfng_ur1.r$r"           "$MORN" "UR_L0_USE_COPY_ENGINE=1" "--fit-target 1024 $K"
    srv4 "pfng_noprefetch.r$r"    "$MORN" "" "--fit-target 1024 --ctx-size 131072 --fit on --moe-cache off --load-mode none --spec-type ngram-mod"
  done
}

sec_mtpfit() { # Qwen4Exp + ngram specs aborted in ggml_sycl_pool_vmm::alloc at --fit-target 1024; does headroom fix it?
  log "== sec mtpfit"
  local T=/mnt/ssd1/gguf/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
  local t
  for t in 2048 3072; do
    srv4 "q4exp_ngmod_fit$t"    "$T" "" "--ctx-size 16384 --fit on --fit-target $t --spec-type ngram-mod"
    srv4 "q4exp_ngsimple_fit$t" "$T" "" "--ctx-size 16384 --fit on --fit-target $t --spec-type ngram-simple"
  done
}

for s in "$@" correct3 prefetchng mtpfit; do "sec_$s"; rt "dmesg" > "$ROOT/klog4-$s.txt" 2>&1; log "klog4 saved ($s) faults=$(faults)"; done
log "PHASE4 DONE: $*"
