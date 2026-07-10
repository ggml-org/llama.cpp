# Архитектура infcore

## Принцип
Слой над llama.cpp без правок движка. Граница «чужое/своё» — C-API `include/llama.h`.
Движок (ggml + libllama + tools/server) — апстрим, обновляется drop-in. Всё «своё» —
в каталоге `infcore/` (новый каталог → не конфликтует при слиянии с апстримом).

## Слои
```
┌──────────────────────────────────────────────────────────────┐
│ infcore/sdk (клиентский SDK) · infcore-cli                     │  клиенты
├──────────────────────────────────────────────────────────────┤
│ infcore/gateway: OpenAI-surface + routing + policy             │  API (своё)
│   authn/rbac (infcore/security) · audit · metrics (observ.)    │
│   registry (multi-model) · lazy-supervisor                     │
├───────────────── граница: include/llama.h ───────────────────-┤
│ tools/server (OpenAI-совм., SSE) · libllama (src/) · ggml      │  движок (апстрим)
│   бэкенды: cpu + cuda + vulkan; слушает только 127.0.0.1       │
└──────────────────────────────────────────────────────────────┘
```

## Топология процессов
- **Движок-сервер:** `llama-server` (собирается из форка, профиль cpu+cuda+vulkan),
  отдаёт OpenAI-совместимый API. Управляемые бэкенды всегда слушают только 127.0.0.1
  и стартуют с per-boot случайным `--api-key`; прямой доступ к их портам (8100+) без
  ключа -> 401.
- **Gateway (`infcore_gateway`):** наша надстройка — единая точка входа: authn/RBAC,
  routing по multi-model registry, политики/квоты, audit, pull-метрики, healthcheck.
  Работает как фронт (proxy-front) перед подпроцессами `llama-server`; при
  проксировании добавляет per-boot ключ бэкенда на `Authorization`.

## Ленивый супервайзер
Управляемые модели (без `backend_url`) поднимаются по первому запросу и гасятся по
простою. Порт назначается под локом (без гонки), при неудачном старте — backoff и
сброс порта; liveness через `waitpid(WNOHANG)` (без зомби, детект упавших бэкендов);
active-token на RAII (аборт клиента не течёт в счётчик).

## Поток запроса (chat, OpenAI)
1. REST принимает `/v1/chat/completions` → `authn` (api_key, constant-time сравнение).
2. `rbac` + `policy`: роль → разрешена ли модель/эндпоинт (allow_endpoints); квоты.
3. `routing`: `model` → `registry` → бэкенд (llama-server нужной модели, при
   необходимости поднимается супервайзером).
4. Проксирование с SSE при `stream:true`. Статус апстрима проверяется ДО коммита
   стрима: не-2xx возвращается обычной JSON-ошибкой (OpenAI shape), а не SSE внутри
   200; синтетические ошибки завершают стрим `data: [DONE]`.
5. `audit` (кто/модель/токены/реальный статус бэкенда, в т.ч. отказы) + `metrics`.

## Offline-инварианты
- Рантайм без исходящих соединений (`offline.enforce_no_egress`, `tests/egress/`).
- Только локальные GGUF. Важно: движок llama.cpp (апстрим) **содержит** сетевой код
  загрузки (`common/download.cpp`, `-hf`/`--model-url`, fetch удалённых картинок в
  `server-common.cpp`). По правилу wrap-not-touch мы движок не редактируем, поэтому
  этот код **не удалён, а нейтрализован**: супервайзер никогда не передаёт
  download-триггерящих аргументов, а egress блокируется на инфра-уровне (systemd
  `IPAddressDeny`, docker `internal: true`) и проверкой `enforce_no_egress` для
  внешних `backend_url` (только loopback/RFC1918). См. `../AUDIT.md`.
- Зависимости — из внутренних зеркал.

## Модели и модальности
Поддерживаются любые локальные GGUF, как у llama.cpp: text, embeddings, vision (VLM
через `mmproj_path` -> `--mmproj`). Audio (ASR/TTS) — движок (`tools/mtmd`/`tools/tts`)
сохранён профилем сборки, но эндпоинты `/v1/audio/*` пока не реализованы (roadmap).
Новые архитектуры приезжают drop-in вместе с обновлением движка.
