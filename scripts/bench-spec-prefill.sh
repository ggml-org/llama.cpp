#!/bin/bash
# Measure TTFT and decode rate for speculative prefill against a llama-server arm.
#
#   scripts/bench-spec-prefill.sh <label> <prompt-file> -m target.gguf [server args...]
#
# Runs one warmup plus 3 timed completions with prompt caching off, then reports the median
# TTFT. Pass -mpd/--spec-prefill-percentage in the server args to measure a speculative
# prefill arm, or leave them out for the baseline.

set -u

LABEL="$1"; shift
PROMPT_FILE="$1"; shift

PORT="${PORT:-8899}"
NEEDLE="${NEEDLE:-}"
N_PREDICT="${N_PREDICT:-128}"
BIN="${BIN:-./build/bin/llama-server}"
LOG="/tmp/bench-spf-$LABEL.log"

"$BIN" --port "$PORT" --no-webui "$@" > "$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo "$LABEL: server died"; tail -8 "$LOG"; exit 1; }
    sleep 2
done

PROMPT=$(python3 -c "import json,sys;print(json.dumps(open(sys.argv[1]).read()))" "$PROMPT_FILE")
REQ="{\"prompt\":$PROMPT,\"n_predict\":$N_PREDICT,\"temperature\":0,\"cache_prompt\":false,\"ignore_eos\":true}"

curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "$REQ" > /dev/null

for _ in 1 2 3; do
    curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "$REQ"
    echo
done | LABEL="$LABEL" NEEDLE="$NEEDLE" python3 -c "
import sys, json, os
label  = os.environ['LABEL']
needle = os.environ.get('NEEDLE', '')
ttft = []; tg = []; n = 0; hits = 0; runs = 0; acc = 0; drafted = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    d = json.loads(line); t = d['timings']
    ttft.append(t['prompt_ms']); tg.append(t['predicted_per_second']); n = t['prompt_n']
    acc += t.get('draft_n_accepted', 0); drafted += t.get('draft_n', 0)
    runs += 1
    hits += bool(needle) and needle in d['content']
ttft.sort()
med = ttft[len(ttft)//2]
out = f'{label}  N={n}  ttft_med={med:.0f}ms  eff_pp={1000*n/med:.1f} tok/s  tg_avg={sum(tg)/len(tg):.2f} tok/s'
if drafted:
    out += f'  accept={100*acc/drafted:.0f}%'
if needle:
    out += f'  needle={hits}/{runs}'
print(out)
"

grep -oE "kept [0-9]+ / [0-9]+ tokens \([0-9.]+%\)" "$LOG" | sort -u
