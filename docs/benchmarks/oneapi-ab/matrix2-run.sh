#!/usr/bin/env bash
# matrix2-run.sh SECTION... — phase 2: Ornith env axes, mode A/B, MKL FA sign sweep, depth, spec mechanism, prefetch, oracle.
source /mnt/nvme1/oneapi-ab/matrix-lib.sh
ROOT=/mnt/nvme1/oneapi-ab/matrix-1006
init
setccs 1
BO=( /usr/bin/llama-bench -m "$MORN" -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 --moe-cache off --load-mode none -p 512 -n 64 -d 0,8192 -r 3 -t 12 -o md )
GATE=/mnt/nvme1/llama-sycl-build/build/llama.cpp-sycl-f16-git/src/build/bin/test-sycl-turbo-correctness

sec_oracle() { # CPU-oracle gate on the MKL FA route vs the fallback route (section [4c])
  log "== sec oracle"; mkdir -p "$ROOT/oracle"
  local v
  for v in 1 0; do
    ( clean_env; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
      GGML_SYCL_ENABLE_MKL_FA=$v SYCL_CACHE_PERSISTENT=1 timeout 1500 "$GATE" > "$ROOT/oracle/gate.mklfa$v.log" 2>&1 )
    log "oracle MKL_FA=$v: $(grep -E 'summary|GATE-FAIL' "$ROOT/oracle/gate.mklfa$v.log" | tail -2 | tr '\n' ' ')"
  done
}

sec_cost() { # verify-batch cost: pp at small batch sizes, dense 8B vs Ornith MoE (mechanism for ngram-mod slowdown)
  log "== sec cost"
  local r
  for r in 0 1 2; do
    cell cost "8b.r$r" "" /usr/bin/llama-bench -m "$M8B" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 1,2,4,8,16,32,64,128 -n 0 -r 3 -t 12 -o md
    cell cost "ornith.r$r" "" /usr/bin/llama-bench -m "$MORN" -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 --moe-cache off --load-mode none -p 1,2,4,8,16,32,64,128 -n 0 -r 3 -t 12 -o md
  done
}

sec_mode() { # paired ccs_mode A/B on Ornith production, one session, one build
  log "== sec mode"
  local r m
  for r in 0 1 2 3; do
    for m in $(rotate_arms $r 1 2 4); do setccs "$m" || continue; cell mode "ccs$m.r$r" "" "${BO[@]}"; done
  done
  setccs 1
}

sec_ornenv() { # env axes on Ornith production, with 8k columns; ILCL unset vs 1 vs 0
  log "== sec ornenv"
  local arms=( "base:" "GRAPH0:GGML_SYCL_ENABLE_GRAPH=0" "Q8QF0:GGML_SYCL_Q8_KV_QUANTS_FIRST=0" "Q8GQATILE:GGML_SYCL_FA_Q8_GQA_TILE=1"
    "OPT0:GGML_SYCL_ENABLE_OPT=0" "ILCL0:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0" "ILCL1:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1"
    "WG8:GGML_SYCL_MAX_WG_PER_CU=8" "WG32:GGML_SYCL_MAX_WG_PER_CU=32" "MKLFA0:GGML_SYCL_ENABLE_MKL_FA=0" "GRF0:GGML_SYCL_FA_LARGE_GRF=0"
    "FUSION0:GGML_SYCL_ENABLE_FUSION=0" "ESIMD0:GGML_SYCL_ENABLE_ESIMD=0" "VMM0:GGML_SYCL_ENABLE_VMM=0" "PINNED0:GGML_SYCL_ENABLE_HOST_PINNED_MEM=0"
    "VECSTD:GGML_SYCL_FA_FORCE_VEC_STANDARD=1" "ASYNC0:GGML_SYCL_USE_ASYNC_MEM_OP=0" "DMMV1:GGML_SYCL_PRIORITIZE_DMMV=1" )
  local r a
  for r in 0 1 2; do
    while read -r a; do cell ornenv "${a%%:*}.r$r" "${a#*:}" "${BO[@]}"; done < <(rotate_arms $r "${arms[@]}")
  done
}

sec_8bnoise() { # more rounds on the sub-2% 8B effects
  log "== sec 8bnoise"
  local arms=( "base:" "WG32:GGML_SYCL_MAX_WG_PER_CU=32" "Q8QF0:GGML_SYCL_Q8_KV_QUANTS_FIRST=0" "VMM0:GGML_SYCL_ENABLE_VMM=0"
    "ILCL0:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0" "ILCL1:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1" )
  local r a
  for r in 0 1 2 3 4 5; do
    while read -r a; do cell 8bnoise "${a%%:*}.r$r" "${a#*:}" "${B8[@]}"; done < <(rotate_arms $r "${arms[@]}")
  done
}

sec_mkl() { # MKL FA on/off vs model, KV type, depth
  log "== sec mkl"
  local models=( "llama31-8b|$M8B|-ngl 99" "qwen25-7b|/mnt/mrgr/models/qwen2.5-coder-7b-instruct-q6_k.gguf|-ngl 99"
                 "qwen35-9b|/mnt/mrgr/models/Qwen3.5-9B.Q4_K_M.gguf|-ngl 99"
                 "ornith35|$MORN|-fitt 1024 -fitc 32768 --moe-cache off --load-mode none" )
  local r m kv v name path args
  for r in 0; do
    for m in "${models[@]}"; do
      IFS='|' read -r name path args <<<"$m"
      for kv in q8_0 f16; do
        for v in 1 0; do
          # shellcheck disable=SC2086
          cell mkl "$name.$kv.mkl$v.r$r" "GGML_SYCL_ENABLE_MKL_FA=$v" /usr/bin/llama-bench -m "$path" $args -fa 1 -ctk $kv -ctv $kv \
            -p 512 -n 0 -d 0,2048,4096,8192,16384 -r 2 -t 12 -o md
        done
      done
    done
  done
}

sec_deep() { # context depth on Ornith at the production fit reservation (ctx 131072)
  log "== sec deep"
  RUN_TIMEOUT_S=5400
  local arms=( "base:" "MKLFA0:GGML_SYCL_ENABLE_MKL_FA=0" "Q8QF0:GGML_SYCL_Q8_KV_QUANTS_FIRST=0" "Q8GQATILE:GGML_SYCL_FA_Q8_GQA_TILE=1"
               "WG32:GGML_SYCL_MAX_WG_PER_CU=32" "GRF0:GGML_SYCL_FA_LARGE_GRF=0" )
  local r a
  for r in 0; do
    while read -r a; do
      cell deep "${a%%:*}.r$r" "${a#*:}" /usr/bin/llama-bench -m "$MORN" -fitt 1024 -fitc 131072 -ctk q8_0 -ctv q8_0 -fa 1 --moe-cache off --load-mode none \
        -p 512 -n 64 -d 0,16384,32768,65536 -r 2 -t 12 -o md
    done < <(rotate_arms $r "${arms[@]}")
  done
  RUN_TIMEOUT_S=1200
}


sec_fitc() { # production reservation: -fitc 131072 vs 32768 on the Ornith Q4_K_M, and the Q8_0 unit model
  log "== sec fitc"
  local r a
  local MQ8=/mnt/ssd2/models/ornith-1.5-35b-a3b-uncensored-q8_0.gguf
  local arms=( "q4_fitc32k|$MORN|32768|" "q4_fitc128k|$MORN|131072|" "q4_fitc128k_mkl0|$MORN|131072|GGML_SYCL_ENABLE_MKL_FA=0"
               "q8_fitc128k|$MQ8|131072|" "q8_fitc128k_mkl0|$MQ8|131072|GGML_SYCL_ENABLE_MKL_FA=0" "q8_fitc128k_grf0|$MQ8|131072|GGML_SYCL_FA_LARGE_GRF=0" )
  for r in 0 1 2; do
    while read -r a; do
      IFS='|' read -r n path fc envs <<<"$a"
      cell fitc "$n.r$r" "$envs" /usr/bin/llama-bench -m "$path" -fitt 1024 -fitc "$fc" -ctk q8_0 -ctv q8_0 -fa 1 --moe-cache off --load-mode none \
        -p 512 -n 64 -d 0,8192 -r 3 -t 12 -o md
    done < <(rotate_arms $r "${arms[@]}")
  done
}

sec_two2() { # two concurrent 8B with 8k columns, launch overlap recorded
  log "== sec two2"
  mkdir -p "$ROOT/two2"
  local r m out
  local B2=( /usr/bin/llama-bench -m "$M8B" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 64 -d 0,8192 -r 3 -t 6 -o md )
  for r in 0 1 2; do
    for m in $(rotate_arms $r 1 2); do
      setccs "$m" || continue
      out="$ROOT/two2/ccs$m.r$r"
      ( clean_env; s=$(date +%s.%N); timeout 1500 "${B2[@]}" > "$out.a.md" 2>"$out.a.err"; echo "A $s $(date +%s.%N)" >> "$out.span" ) &
      pa=$!
      ( clean_env; s=$(date +%s.%N); timeout 1500 "${B2[@]}" > "$out.b.md" 2>"$out.b.err"; echo "B $s $(date +%s.%N)" >> "$out.span" ) &
      pb=$!
      wait $pa $pb
      log "two2 ccs$m r$r span: $(tr '\n' ' ' < "$out.span") A: $(grep -E 'pp512|tg64' "$out.a.md" | awk -F'|' '{printf "%s=%s ", $(NF-2),$(NF-1)}' | tr -s ' ')"
    done
  done
  setccs 1
}

sec_correct() { # output-equivalence: perplexity at ctx 8192 (exercises MKL FA / q8 KV routes) per kernel-changing arm, 8B in VRAM
  log "== sec correct"
  mkdir -p "$ROOT/correct"
  local txt="$ROOT/correct/text.txt"
  cat /home/svnbjrn/.docs/sessions/*llama-cpp-sycl-f16-git-Raudbjorn*.md /mnt/nvme1/llama-sycl-build/build/llama.cpp-sycl-f16-git/src/llama.cpp/docs/backend/SYCL.md > "$txt"
  local arms=( "base:" "MKLFA0:GGML_SYCL_ENABLE_MKL_FA=0" "Q8QF0:GGML_SYCL_Q8_KV_QUANTS_FIRST=0" "OPT0:GGML_SYCL_ENABLE_OPT=0" "ESIMD0:GGML_SYCL_ENABLE_ESIMD=0"
    "Q8GQATILE:GGML_SYCL_FA_Q8_GQA_TILE=1" "GRF0:GGML_SYCL_FA_LARGE_GRF=0" "FUSION0:GGML_SYCL_ENABLE_FUSION=0" "WG8:GGML_SYCL_MAX_WG_PER_CU=8"
    "VECSTD:GGML_SYCL_FA_FORCE_VEC_STANDARD=1" "GRAPH0:GGML_SYCL_ENABLE_GRAPH=0" "DMMV1:GGML_SYCL_PRIORITIZE_DMMV=1" "base2:" )
  local r a n
  for r in 0 1; do
    while read -r a; do
      n=${a%%:*}
      cell correct "$n.r$r" "${a#*:}" /usr/bin/llama-perplexity -m "$M8B" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 8192 -b 2048 -ub 512 --chunks 4 -f "$txt"
    done < <(rotate_arms $r "${arms[@]}")
  done
  # ccs_mode must not change numerics either
  for m in 2 4 1; do setccs $m && cell correct "ccs$m.r0" "" /usr/bin/llama-perplexity -m "$M8B" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 8192 -b 2048 -ub 512 --chunks 4 -f "$txt"; done
}

sec_host() { # host-side knobs for the host-bound Ornith placement: threads, CCD affinity, batch sizes
  log "== sec host"
  local W=( 0-5,12-17 6-11,18-23 ) # CCD0 (V-cache) and CCD1 of the 7900X3D, with SMT siblings
  local base=( /usr/bin/llama-bench -m "$MORN" -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 --moe-cache off --load-mode none -p 512 -n 64 -d 0,8192 -r 3 -o md )
  local arms=( "t12||-t 12" "t6||-t 6" "t8||-t 8" "t16||-t 16" "t24||-t 24"
               "ccd0_t12|taskset -c ${W[0]}|-t 12" "ccd0_t6|taskset -c ${W[0]}|-t 6" "ccd1_t12|taskset -c ${W[1]}|-t 12" "ccd1_t6|taskset -c ${W[1]}|-t 6"
               "ub256||-t 12 -b 2048 -ub 256" "ub1024||-t 12 -b 2048 -ub 1024" "ub2048||-t 12 -b 2048 -ub 2048" )
  local r a n pre args
  for r in 0 1; do
    while read -r a; do
      IFS='|' read -r n pre args <<<"$a"
      # shellcheck disable=SC2086
      cell host "$n.r$r" "" $pre "${base[@]}" $args
    done < <(rotate_arms $r "${arms[@]}")
  done
}

# serve NAME MODEL EXTRA_ENV "server args..." — start llama-server on :8092, run the heterogeneous prompt set, stop it.
serve_run() {
  local name=$1 model=$2 envs=$3 args=$4 d="$ROOT/spec2" up=0 sp i f0
  mkdir -p "$d"
  ( clean_env
    # shellcheck disable=SC2086
    exec env $envs timeout 2400 /usr/bin/llama-server --model "$model" --alias bench --parallel 1 --flash-attn on --threads 12 --threads-batch 12 \
      --jinja --temp 0.6 --top-p 0.95 --host 127.0.0.1 --port 8092 -ctk q8_0 -ctv q8_0 $args > "$d/$name.server.log" 2>&1 ) &
  sp=$!
  for i in $(seq 1 150); do curl -sf -m 3 http://127.0.0.1:8092/health >/dev/null && { up=1; break; }; kill -0 $sp 2>/dev/null || break; sleep 5; done
  if [ $up = 1 ]; then
    f0=$(faults); python3 /mnt/nvme1/oneapi-ab/matrix2-spec.py 8092 "$name" "$d/raw" >> "$d/results.jsonl" 2> "$d/$name.client.err"
    log "spec2 $name done faults+$(( $(faults)-f0 )) tenancy=$(tenancy)"
  else log "spec2 $name server failed: $(grep -iE 'error|abort' "$d/$name.server.log" | head -2 | cut -c1-160 | tr '\n' ' ')"; fi
  kill $sp 2>/dev/null; sleep 2; pkill -x llama-server 2>/dev/null; sleep 6
}

sec_spec2() {
  log "== sec spec2"
  local ORN="--model-placeholder" r c n a
  local cfgs=( "none|--spec-type none" "ngsimple|--spec-type ngram-simple" "ngmod|--spec-type ngram-mod" "ngmapk4v|--spec-type ngram-map-k4v" )
  for r in 0 1; do
    for c in "${cfgs[@]}"; do   # Ornith production placement
      n=${c%%|*}; a=${c#*|}
      serve_run "ornith.$n.r$r" "$MORN" "" "--ctx-size 131072 --fit on --fit-target 1024 --moe-cache off --load-mode none $a"
    done
    for c in "${cfgs[@]}"; do   # dense 8B fully in VRAM: is the ngram-mod slowdown expert related?
      n=${c%%|*}; a=${c#*|}
      serve_run "llama8b.$n.r$r" "$M8B" "" "--ctx-size 16384 -ngl 99 $a"
    done
  done
}

sec_prefetch() { # needs the copy engines back on (UR_L0_USE_COPY_ENGINE=1); blitter hang risk is mitigated by NEO patch 050; dmesg checked
  log "== sec prefetch"
  local r
  for r in 0 1; do
    serve_run "prefetch0.r$r" "$MORN" "UR_L0_USE_COPY_ENGINE=1" "--ctx-size 131072 --fit on --fit-target 1024 --moe-cache off --load-mode none --spec-type none"
    serve_run "prefetch4.r$r" "$MORN" "UR_L0_USE_COPY_ENGINE=1" "--ctx-size 131072 --fit on --fit-target 1024 --moe-cache off --load-mode none --spec-type none --prefetch-experts-slots 4"
    serve_run "prefetch2.r$r" "$MORN" "UR_L0_USE_COPY_ENGINE=1" "--ctx-size 131072 --fit on --fit-target 1024 --moe-cache off --load-mode none --spec-type none --prefetch-experts-slots 2"
    serve_run "prefetch4_fit3072.r$r" "$MORN" "UR_L0_USE_COPY_ENGINE=1" "--ctx-size 131072 --fit on --fit-target 3072 --moe-cache off --load-mode none --spec-type none --prefetch-experts-slots 4"
    serve_run "prefetch0_fit3072.r$r" "$MORN" "UR_L0_USE_COPY_ENGINE=1" "--ctx-size 131072 --fit on --fit-target 3072 --moe-cache off --load-mode none --spec-type none"
    serve_run "prefetch0_nocopy.r$r" "$MORN" "" "--ctx-size 131072 --fit on --fit-target 1024 --moe-cache off --load-mode none --spec-type none"
  done
}

for s in "$@"; do "sec_$s"; rt "dmesg" > "$ROOT/klog-$s.txt" 2>&1; log "klog saved klog-$s.txt ($(wc -l < "$ROOT/klog-$s.txt") lines, faults=$(faults))"; done
log "PHASE2 DONE: $*"
