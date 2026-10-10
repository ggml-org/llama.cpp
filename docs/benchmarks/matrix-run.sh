#!/usr/bin/env bash
# matrix-run.sh SECTION... — sections: ccs envs il freq ornith two ngram
source /mnt/nvme1/oneapi-ab/matrix-lib.sh
init

ROUNDS=3

sec_ccs() {  # ccs_mode 1/2/4, single process, 8B in VRAM
  log "== sec ccs"
  local r m
  for ((r=0;r<ROUNDS;r++)); do
    for m in $(rotate_arms $r 1 2 4); do
      setccs "$m" || continue
      cell ccs "ccs$m.r$r" "" "${B8[@]}"
    done
  done
}

sec_envs() { # environment axes at ccs_mode=1
  log "== sec envs"
  setccs 1 || return
  local arms=(
    "base:"
    "MKLFA0:GGML_SYCL_ENABLE_MKL_FA=0"
    "MKLQT2048:GGML_SYCL_MKL_FA_Q_TILE=2048"
    "GRAPH0:GGML_SYCL_ENABLE_GRAPH=0"
    "GRF0:GGML_SYCL_FA_LARGE_GRF=0"
    "Q8GQATILE:GGML_SYCL_FA_Q8_GQA_TILE=1"
    "VECSTD:GGML_SYCL_FA_FORCE_VEC_STANDARD=1"
    "Q8QF0:GGML_SYCL_Q8_KV_QUANTS_FIRST=0"
    "FUSION0:GGML_SYCL_ENABLE_FUSION=0"
    "FFN0:GGML_SYCL_FFN_FUSION=0"
    "DMMV1:GGML_SYCL_PRIORITIZE_DMMV=1"
    "ESIMD0:GGML_SYCL_ENABLE_ESIMD=0"
    "OPT0:GGML_SYCL_ENABLE_OPT=0"
    "WG8:GGML_SYCL_MAX_WG_PER_CU=8"
    "WG32:GGML_SYCL_MAX_WG_PER_CU=32"
    "ASYNC0:GGML_SYCL_USE_ASYNC_MEM_OP=0"
    "PINNED0:GGML_SYCL_ENABLE_HOST_PINNED_MEM=0"
    "VMM0:GGML_SYCL_ENABLE_VMM=0"
    "ILCL0:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0"
    "ILCL1:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1"
    "COPYENG1:UR_L0_USE_COPY_ENGINE=1"
  )
  local r a
  for ((r=0;r<ROUNDS;r++)); do
    while read -r a; do
      cell envs "${a%%:*}.r$r" "${a#*:}" "${B8[@]}"
    done < <(rotate_arms $r "${arms[@]}")
  done
}

sec_il() { # ccs_mode x immediate command lists
  log "== sec il"
  local r m il
  for ((r=0;r<ROUNDS;r++)); do
    for m in $(rotate_arms $r 1 2 4); do
      setccs "$m" || continue
      for il in 0 1; do
        cell il "ccs$m.il$il.r$r" "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=$il" "${B8[@]}"
      done
    done
  done
}

sec_freq() { # gt0 frequency / power profile at ccs_mode=1
  log "== sec freq"
  setccs 1 || return
  local arms=( "def:600:2400:base" "pin2400:2400:2400:base" "pin2000:2000:2000:base" "pin1500:1500:1500:base" "powersave:600:2400:power_saving" "floor1500:1500:2400:base" )
  local r a n mn mx pp
  for ((r=0;r<ROUNDS;r++)); do
    while read -r a; do
      IFS=: read -r n mn mx pp <<<"$a"
      setfreq "$mn" "$mx"; rt "echo $pp > $G/freq0/power_profile"
      cell freq "$n.r$r" "" "${B8[@]}"
    done < <(rotate_arms $r "${arms[@]}")
  done
  setfreq 600 2400; rt "echo base > $G/freq0/power_profile"
}

sec_ornith() { # Ornith Q4_K_M production placement: moe-cache / load-mode / env
  log "== sec ornith"
  setccs 1 || return
  local P=( "${BORN[@]}" )
  local arms=(
    "prod:--moe-cache off --load-mode none:"
    "mcauto_none:--moe-cache auto --load-mode none:"
    "mcauto_auto:--moe-cache auto --load-mode auto:"
    "mcoff_auto:--moe-cache off --load-mode auto:"
    "mcsoft_none:--moe-cache soft --load-mode none:"
    "mcon_none:--moe-cache on --load-mode none:"
    "mc4096_none:--moe-cache 4096 --load-mode none:"
    "prod_pinned0:--moe-cache off --load-mode none:GGML_SYCL_ENABLE_HOST_PINNED_MEM=0"
    "prod_ilcl1:--moe-cache off --load-mode none:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1"
    "prod_mkl0:--moe-cache off --load-mode none:GGML_SYCL_ENABLE_MKL_FA=0"
    "prod_grf0:--moe-cache off --load-mode none:GGML_SYCL_FA_LARGE_GRF=0"
    "prod_graph0:--moe-cache off --load-mode none:GGML_SYCL_ENABLE_GRAPH=0"
  )
  local r a n args envs
  for ((r=0;r<ROUNDS;r++)); do
    while read -r a; do
      n=${a%%:*}; a=${a#*:}; args=${a%%:*}; envs=${a#*:}
      # shellcheck disable=SC2086
      cell ornith "$n.r$r" "$envs" "${P[@]}" $args
    done < <(rotate_arms $r "${arms[@]}")
  done
}

sec_two() { # two concurrent 8B processes (tier 2), ccs_mode x timeslice
  log "== sec two"
  local r m ts out
  local B2=( /usr/bin/llama-bench -m "$M8B" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 64 -d 0 -r 3 -t 6 -o md )
  for ((r=0;r<ROUNDS;r++)); do
    for m in $(rotate_arms $r 1 2 4); do
      setccs "$m" || continue
      for ts in 1000 20000; do
        rt "echo $ts > $G/engines/ccs/timeslice_duration_us"
        out="$ROOT/two/ccs$m.ts$ts.r$r"; mkdir -p "$ROOT/two"
        (clean_env; timeout $RUN_TIMEOUT_S "${B2[@]}" > "$out.a.md" 2>"$out.a.err") &
        local pa=$!
        (clean_env; timeout $RUN_TIMEOUT_S "${B2[@]}" > "$out.b.md" 2>"$out.b.err") &
        local pb=$!
        wait $pa $pb
        log "two ccs$m ts$ts r$r: A $(grep -E 'pp512|tg64' $out.a.md | awk -F'|' '{printf "%s ",$(NF-1)}') B $(grep -E 'pp512|tg64' $out.b.md | awk -F'|' '{printf "%s ",$(NF-1)}')"
        printf 'two\tccs%s.ts%s.r%s\tfaults+%s\n' "$m" "$ts" "$r" "$(faults)" >> "$ROOT/index.tsv"
      done
    done
  done
  rt "echo 1000 > $G/engines/ccs/timeslice_duration_us"
}

for s in "$@"; do "sec_$s"; done
log "ALL DONE: $*"
