#!/usr/bin/env bash
# infcore — ручной hardening-smoke шлюза. Проверяет прод-фиксы против ФЕЙКОВОГО
# бэкенда (реальные модели не нужны):
#   M5  — лимит тела запроса -> 413
#   B3  — mu_ не держится во время SIGTERM->SIGKILL (шлюз отзывчив при остановке)
#   F1  — disable во время старта бэкенда не теряется (502 + бэкенд погашен)
#   F2  — рантайм-сбой аудита -> fail-closed (503) при audit.require=true
#
# Использование:
#   infcore/tests/manual/hardening_smoke.sh [путь_к_infcore_gateway]
# Если бинарь не передан — берётся ./build/bin/infcore_gateway (собери:
#   cmake -S infcore -B build -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --target infcore_gateway -j)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
GW="${1:-./build/bin/infcore_gateway}"
FAKE="$HERE/fake_llama_server.py"
RLIMIT="$HERE/rlimit_exec.py"
KEY="real-key"
PASS=0; FAIL=0
TMP="$(mktemp -d)"; PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done
            pkill -9 -f "fake_llama_server.py" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

if [ ! -x "$GW" ]; then echo "НЕ найден infcore_gateway: $GW"; exit 2; fi
ok(){ echo "  PASS: $1"; PASS=$((PASS+1)); }
no(){ echo "  FAIL: $1"; FAIL=$((FAIL+1)); }
now(){ python3 -c 'import time;print(time.time())'; }

# wrapper'ы llama_server_bin: супервайзер зовёт их с --host/--port/--model...,
# мы доклеиваем нужные для теста флаги фейка.
mk_wrap(){ # $1=имя $2=доп.флаги
  local f="$TMP/llama_$1.sh"
  printf '#!/bin/sh\nexec python3 %q %s "$@"\n' "$FAKE" "$2" > "$f"; chmod +x "$f"; echo "$f"
}
mk_cfg(){ # stdin=json -> файл
  local f="$TMP/$1.json"; cat > "$f"; echo "$f"
}

echo "=== infcore hardening smoke ==="; echo "gateway: $GW"; echo

# ---------- M5: лимит тела -> 413 ----------
echo "[M5] лимит размера тела запроса"
CFG=$(mk_cfg m5 <<EOF
{"server":{"host":"127.0.0.1","port":18090,"max_body_bytes":1024},
 "security":{"rbac_enabled":true,
   "principals":[{"api_key":"$KEY","subject":"a","role":"admin"}],
   "roles":[{"name":"admin","allow_models":["*"],"allow_endpoints":["*"]}],
   "audit":{"sink":"none","path":"/tmp/x","require":false}},
 "runtime":{"llama_server_bin":"/bin/true"},
 "models":[{"logical_name":"a","gguf_path":"/tmp/a.gguf"}]}
EOF
)
"$GW" "$CFG" >/dev/null 2>&1 & GWPID=$!; PIDS+=($GWPID); sleep 1
BIG=$(python3 -c 'print("x"*5000)')
c=$(curl -s -o /dev/null -w "%{http_code}" -m5 http://127.0.0.1:18090/v1/chat/completions \
      -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
      -d "{\"model\":\"a\",\"pad\":\"$BIG\"}")
[ "$c" = "413" ] && ok "тело >1КБ -> 413" || no "ожидал 413, получил $c"
kill "$GWPID" 2>/dev/null; sleep 1; echo

# ---------- B3: шлюз отзывчив во время остановки бэкенда ----------
echo "[B3] mu_ не держится во время SIGTERM->SIGKILL"
W=$(mk_wrap b3 "--ignore-sigterm")   # бэкенд игнорит SIGTERM -> 5с до SIGKILL
CFG=$(mk_cfg b3 <<EOF
{"server":{"host":"127.0.0.1","port":18091,"max_concurrent_requests":16},
 "security":{"rbac_enabled":true,
   "principals":[{"api_key":"$KEY","subject":"a","role":"admin"}],
   "roles":[{"name":"admin","allow_models":["*"],"allow_endpoints":["*"]}],
   "audit":{"sink":"none","path":"/tmp/x","require":false}},
 "runtime":{"llama_server_bin":"$W","port_range_start":18200,"idle_timeout_ms":600000,"startup_timeout_ms":15000},
 "models":[{"logical_name":"a","gguf_path":"/tmp/a.gguf"},
           {"logical_name":"b","gguf_path":"/tmp/b.gguf"}]}
EOF
)
"$GW" "$CFG" >/dev/null 2>&1 & GWPID=$!; PIDS+=($GWPID); sleep 1
curl -s -m20 http://127.0.0.1:18091/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" -d '{"model":"a","messages":[]}' >/dev/null   # поднять backendA
( curl -s -m10 -X POST http://127.0.0.1:18091/admin/models/a/disable -H "Authorization: Bearer $KEY" >/dev/null ) & DPID=$!
sleep 0.3
t0=$(now); curl -s -m8 http://127.0.0.1:18091/health >/dev/null; t1=$(now)
hl=$(python3 -c "print(f'{$t1-$t0:.2f}')")
python3 -c "import sys;sys.exit(0 if $hl < 2.0 else 1)" \
  && ok "/health отвечает за ${hl}s во время 5с-остановки (не блокируется)" \
  || no "/health ждал ${hl}s — mu_ держится во время kill"
wait "$DPID"; kill "$GWPID" 2>/dev/null; sleep 6; echo   # дать шлюзу дожать SIGKILL

# ---------- F1: disable во время Starting не теряется ----------
echo "[F1] disable во время старта бэкенда"
W=$(mk_wrap f1 "--ready-delay 3")    # бэкенд «грузится» 3с -> широкое окно Starting
CFG=$(mk_cfg f1 <<EOF
{"server":{"host":"127.0.0.1","port":18092},
 "security":{"rbac_enabled":true,
   "principals":[{"api_key":"$KEY","subject":"a","role":"admin"}],
   "roles":[{"name":"admin","allow_models":["*"],"allow_endpoints":["*"]}],
   "audit":{"sink":"none","path":"/tmp/x","require":false}},
 "runtime":{"llama_server_bin":"$W","port_range_start":18210,"idle_timeout_ms":600000,"startup_timeout_ms":15000},
 "models":[{"logical_name":"a","gguf_path":"/tmp/a.gguf"}]}
EOF
)
"$GW" "$CFG" >/dev/null 2>&1 & GWPID=$!; PIDS+=($GWPID); sleep 1
( curl -s -m20 http://127.0.0.1:18092/v1/chat/completions -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" -d '{"model":"a","messages":[]}' > "$TMP/f1.out" 2>&1 ) & RPID=$!
sleep 1.5   # backendA в Starting
curl -s -m5 -X POST http://127.0.0.1:18092/admin/models/a/disable -H "Authorization: Bearer $KEY" >/dev/null
wait "$RPID"
grep -q "отключена во время старта" "$TMP/f1.out" \
  && ok "запрос-инициатор получил 502 (disable учтён)" \
  || no "ожидал 502 'отключена во время старта', получил: $(cat "$TMP/f1.out")"
sleep 1
if pgrep -f "fake_llama_server.py" >/dev/null; then no "бэкенд жив — disable потерян"; else ok "бэкенд погашен (не дожил до idle-таймаута)"; fi
kill "$GWPID" 2>/dev/null; sleep 1; echo

# ---------- F2: рантайм-сбой аудита -> fail-closed ----------
echo "[F2] fail-closed при сбое аудита (audit.require=true)"
AUD="$TMP/audit.log"
CFG=$(mk_cfg f2 <<EOF
{"server":{"host":"127.0.0.1","port":18093},
 "security":{"rbac_enabled":true,
   "principals":[{"api_key":"$KEY","subject":"a","role":"admin"}],
   "roles":[{"name":"admin","allow_models":["*"],"allow_endpoints":["*"]}],
   "audit":{"sink":"file","path":"$AUD","require":true}},
 "runtime":{"llama_server_bin":"/bin/true"},
 "models":[{"logical_name":"a","gguf_path":"/tmp/a.gguf"}]}
EOF
)
python3 "$RLIMIT" 3000 "$GW" "$CFG" >/dev/null 2>"$TMP/f2.err" & GWPID=$!; PIDS+=($GWPID); sleep 1
prev=""; saw503=""
for i in $(seq 1 80); do
  c=$(curl -s -o /dev/null -w "%{http_code}" -m3 http://127.0.0.1:18093/v1/models -H "Authorization: Bearer WRONG")
  [ "$c" = "503" ] && { saw503=1; break; }
done
[ -n "$saw503" ] && ok "после сбоя аудита запросы -> 503 (fail-closed)" || no "503 так и не наступил"
h=$(curl -s -m3 http://127.0.0.1:18093/health)
echo "$h" | grep -q '"audit":"failed"' && ok "/health отражает деградацию: $h" || no "/health не показал failed: $h"
grep -qa "КРИТИЧНО" "$TMP/f2.err" && ok "громкий stderr о сбое аудита" || no "нет КРИТИЧНО в stderr"
kill "$GWPID" 2>/dev/null; echo

echo "=== ИТОГ: PASS=$PASS FAIL=$FAIL ==="
[ "$FAIL" -eq 0 ]
