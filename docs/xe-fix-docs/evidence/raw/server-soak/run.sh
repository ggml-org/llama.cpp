#!/usr/bin/env bash
# Server-level soak through the running llama-gpu@Ornith unit (xe + UR_L0_USE_COPY_ENGINE=0).
set -uo pipefail
D=/mnt/nvme1/oneapi-ab/server-soak; KEY=$(sed -n 2p ~/.config/llama-server/api-key | tr -d '[:space:]')
for r in r1 r2 r3 r4 r5 r6; do
  s=$(date +%T); code=$(curl -s -m 900 -o "$D/resp-$r.json" -w '%{http_code}' -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d @"$D/req-$r.json" 127.0.0.1:8089/completion)
  t=$(python3 -c "import json,sys
try:
  d=json.load(open('$D/resp-$r.json')); t=d['timings']; print('prompt_n=%s pp=%.1f t/s predicted_n=%s tg=%.2f t/s'%(t['prompt_n'],t['prompt_per_second'],t['predicted_n'],t['predicted_per_second']))
except Exception as e: print('no timings:',e)")
  echo "== $r start=$s http=$code $t"
  [ "$code" = 200 ] || { echo "FAIL $r http=$code"; ssh vinbonesjr "journalctl -u llama-gpu@Ornith-1.5-35B-Q4_K_M.service -q --no-pager --since -3min | grep -iE 'error|abort|exited|Failed|Started' | tail -5; journalctl -k -q --no-pager --since -3min | grep -E 'xe 0000:03:00.0' | tail -3"; }
done
echo "SERVER SOAK DONE"
