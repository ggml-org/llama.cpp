# Архитектура infcore

## Принцип
Слой над llama.cpp без правок движка. Граница «чужое/своё» — C-API `include/llama.h`.
Движок (ggml + libllama + tools/server) — апстрим, обновляется drop-in. Всё «своё» —
в каталоге `infcore/` (новый каталог → не конфликтует при слиянии с апстримом).

## Слои
```
┌──────────────────────────────────────────────────────────────┐
│ infcore/sdk · infcore/extensions/{agents,rag}                  │  клиенты
├──────────────────────────────────────────────────────────────┤
│ infcore/gateway: OpenAI-surface + routing + policy             │  API (своё)
│   authn/rbac (infcore/security) · audit · metrics (observ.)    │
│   registry (multi-model)                                       │
├───────────────── граница: include/llama.h ───────────────────-┤
│ tools/server (OpenAI-совм., SSE) · libllama (src/) · ggml      │  движок (апстрим)
│   бэкенды: cpu + cuda + vulkan                                 │
└──────────────────────────────────────────────────────────────┘
```

## Топология процессов
- **Движок-сервер:** `llama-server` (собирается из форка, профиль cpu+cuda+vulkan),
  отдаёт OpenAI-совместимый API на loopback.
- **Gateway (`infcore_gateway`):** наша надстройка — единая точка входа: authn/RBAC,
  routing по multi-model registry, политики/квоты, audit, pull-метрики, healthcheck.
  Может работать как фронт перед `llama-server` или как самостоятельный процесс,
  встраивающий libllama (оба варианта возможны; на старте — фронт перед сервером).

## Поток запроса (chat, OpenAI)
1. REST принимает `/v1/chat/completions` → `authn` (api_key/mTLS/OIDC внутр. IdP).
2. `rbac` + `policy`: роль → разрешена ли модель/эндпоинт; квоты/лимиты.
3. `routing`: `model` → `registry` → бэкенд (llama-server нужной модели).
4. Проксирование с **passthrough SSE** при `stream:true`.
5. `audit` (кто/модель/токены) + `metrics`.

## Offline-инварианты
- Рантайм без исходящих соединений (`offline.enforce_no_egress`, `tests/egress/`).
- Только локальные GGUF; сетевые загрузчики удалены (см. `../AUDIT.md` §2).
- Зависимости — из внутренних зеркал.

## Модели и модальности
Поддерживаются любые локальные GGUF, как у llama.cpp: text, embeddings, vision (VLM),
audio (ASR/TTS — через `tools/mtmd`/`tools/tts`). Новые архитектуры приезжают
drop-in вместе с обновлением движка.
