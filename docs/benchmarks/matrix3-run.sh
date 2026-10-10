#!/usr/bin/env bash
# matrix3-run.sh SECTION... — phase 3: gaps closed after review: sysfs readback, throttle reasons, -t 6 single control, hd256/GQA7 correctness,
# Qwen4Exp MTP (PR #90), prefetch discriminators. PID-targeted server teardown (no pkill of other sessions' servers).
source /mnt/nvme1/oneapi-ab/matrix-lib.sh
init; setccs 1

rt_set() { # path value — write via ssh and read back; log mismatch
  local path=$1 val=$2 got
  rt "echo $val > $path" ; got=$(cat "$path" | tr -d '[]')
  [ "$got" = "$(echo $val | tr -d '[]')" ] || log "READBACK MISMATCH $path want=$val got=$got"
  echo "$(date +%s) $path want=$val got=$got" >> "$ROOT/sysfs-writes.log"
}

sec_freq2() { # frequency / profile with verified writes, 2 rounds, throttle reasons in .freq
  log "== sec freq2"
  local arms=( "def:600:2400:base" "pin2400:2400:2400:base" "pin2000:2000:2000:base" "powersave:600:2400:power_saving" "powersave_pin2000:2000:2000:power_saving" )
  local r a n mn mx pp
  for r in 0 1; do
    while read -r a; do
      IFS=: read -r n mn mx pp <<<"$a"
      rt_set $G/freq0/min_freq 600; rt_set $G/freq0/max_freq $mx; rt_set $G/freq0/min_freq $mn; rt_set $G/freq0/power_profile $pp
      log "freq2 $n applied: min=$(cat $G/freq0/min_freq) max=$(cat $G/freq0/max_freq) prof=$(cat $G/freq0/power_profile)"
      cell freq2 "$n.r$r" "" "${B8[@]}"
    done < <(rotate_arms $r "${arms[@]}")
  done
  rt_set $G/freq0/min_freq 600; rt_set $G/freq0/max_freq 2400; rt_set $G/freq0/power_profile base
}

sec_single6() { # single-process controls at the two-process thread count (-t 6), 8k columns
  log "== sec single6"
  local r
  for r in 0 1 2; do
    cell single6 "t6.r$r" "" /usr/bin/llama-bench -m "$M8B" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 64 -d 0,8192 -r 3 -t 6 -o md
    cell single6 "t12.r$r" "" "${B8[@]}"
  done
}

sec_correct2() { # perplexity with the MKL FA route on/off for shapes the oracle [4c] does not cover (hd256, GQA 7:1, hybrid)
  log "== sec correct2"
  mkdir -p "$ROOT/correct2"; local txt="$ROOT/correct/text.txt"
  local models=( "ornith35|$MORN|-fitt 1024 -fitc 32768 --moe-cache off --load-mode none" "qwen35-9b|/mnt/mrgr/models/Qwen3.5-9B.Q4_K_M.gguf|-ngl 99"
                 "qwen25-7b|/mnt/mrgr/models/qwen2.5-coder-7b-instruct-q6_k.gguf|-ngl 99" )
  local r m name path args v kv
  for r in 0 1; do
    for m in "${models[@]}"; do
      IFS='|' read -r name path args <<<"$m"
      for kv in q8_0 f16; do for v in 1 0; do
        # shellcheck disable=SC2086
        cell correct2 "$name.$kv.mkl$v.r$r" "GGML_SYCL_ENABLE_MKL_FA=$v" /usr/bin/llama-perplexity -m "$path" $args -fa on -ctk $kv -ctv $kv -c 8192 -b 2048 -ub 512 --chunks 3 -f "$txt"
      done; done
    done
  done
}

# srv NAME MODEL "ENV" "ARGS" [WAIT_ITER] — llama-server on :8093, prompt set, PID-targeted teardown.
srv() {
  local name=$1 model=$2 envs=$3 args=$4 iters=${5:-150} d="$ROOT/spec3" up=0 sp i f0 t0
  wait_quiet; mkdir -p "$d"; t0=$(date +%s)
  ( clean_env
    # shellcheck disable=SC2086
    exec env $envs timeout 3600 /usr/bin/llama-server --model "$model" --alias bench --parallel 1 --flash-attn on --threads 12 --threads-batch 12 \
      --jinja --temp 0.6 --top-p 0.95 --host 127.0.0.1 --port 8093 -ctk q8_0 -ctv q8_0 $args > "$d/$name.server.log" 2>&1 ) &
  sp=$!
  for i in $(seq 1 "$iters"); do curl -sf -m 3 http://127.0.0.1:8093/health >/dev/null && { up=1; break; }; kill -0 $sp 2>/dev/null || break; sleep 5; done
  if [ $up = 1 ]; then
    f0=$(faults); python3 /mnt/nvme1/oneapi-ab/matrix2-spec.py 8093 "$name" "$d/raw" >> "$d/results.jsonl" 2> "$d/$name.client.err"
    log "srv $name up in $(( $(date +%s)-t0 ))s done faults+$(( $(faults)-f0 )) avail=$(free -m | awk '/Mem:/{print $7}')MB"
  else log "srv $name FAILED: $(grep -iE 'error|abort|Failed' "$d/$name.server.log" | head -3 | cut -c1-200 | tr '\n' ' ')"; fi
  pkill -P $sp 2>/dev/null; kill $sp 2>/dev/null; sleep 3; kill -9 $(pgrep -P $sp) 2>/dev/null; wait $sp 2>/dev/null; sleep 6
}

sec_prefetch2() { # discriminate the prefetch crash: copy-engine env, VMM, VRAM budget (fit-target), hook default
  log "== sec prefetch2"
  local P="--ctx-size 131072 --fit on --fit-target"; local K="--moe-cache off --load-mode none --spec-type none"
  local r
  for r in 0 1; do
    srv "p0_base.r$r"           "$MORN" "" "$P 1024 $K"
    srv "p4_base.r$r"           "$MORN" "" "$P 1024 $K --prefetch-experts-slots 4"
    srv "p4_ur1.r$r"            "$MORN" "UR_L0_USE_COPY_ENGINE=1" "$P 1024 $K --prefetch-experts-slots 4"
    srv "p4_vmm0.r$r"           "$MORN" "GGML_SYCL_ENABLE_VMM=0" "$P 1024 $K --prefetch-experts-slots 4"
    srv "p4_fit2048.r$r"        "$MORN" "" "$P 2048 $K --prefetch-experts-slots 4"
    srv "p4_fit3072.r$r"        "$MORN" "" "$P 3072 $K --prefetch-experts-slots 4"
    srv "p4_ur1_fit3072.r$r"    "$MORN" "UR_L0_USE_COPY_ENGINE=1" "$P 3072 $K --prefetch-experts-slots 4"
    srv "p2_base.r$r"           "$MORN" "" "$P 1024 $K --prefetch-experts-slots 2"
    srv "p4_hookdefault.r$r"    "$MORN" "-u GGML_SYCL_XE_COPY_ENGINE_DEFAULT" "$P 1024 $K --prefetch-experts-slots 4"
  done
}

sec_mtp() { # PR #90 Qwen4Exp MTP / chained drafting on the real trunk + head; run alone (trunk ~54 GiB vs 58 GiB RAM)
  log "== sec mtp"
  local T=/mnt/ssd1/gguf/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
  local H=/mnt/ssd2/gguf/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M/mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
  local B="--ctx-size 16384 --fit on --fit-target 1024"
  local r
  for r in 0 1; do
    srv "mtp_none.r$r"      "$T" "" "$B --spec-type none" 400
    srv "mtp_n6.r$r"        "$T" "" "$B -md $H --spec-type draft-mtp --spec-draft-n-max 6" 400
    srv "mtp_n3.r$r"        "$T" "" "$B -md $H --spec-type draft-mtp --spec-draft-n-max 3" 400
    srv "mtp_n6_chain4.r$r" "$T" "" "$B -md $H --spec-type draft-mtp --spec-draft-n-max 6 --spec-chain 4" 400
    srv "mtp_n6_fit3072.r$r" "$T" "" "--ctx-size 16384 --fit on --fit-target 3072 -md $H --spec-type draft-mtp --spec-draft-n-max 6" 400
    srv "mtp_ngmod.r$r"     "$T" "" "$B --spec-type ngram-mod" 400
    srv "mtp_ngsimple.r$r"  "$T" "" "$B --spec-type ngram-simple" 400
  done
}

for s in "$@"; do "sec_$s"; rt "dmesg" > "$ROOT/klog3-$s.txt" 2>&1; log "klog3 saved ($s) faults=$(faults)"; done
log "PHASE3 DONE: $*"
