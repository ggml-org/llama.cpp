# infcore — внутренний слой над llama.cpp

Слой **gateway/SDK** для локального инференса LLM (offline РФ-контур), построенный
**на базе open-source llama.cpp (ggml authors, MIT)** — см. `NOTICE`.

## Модель сопровождения: «обернуть, не трогая ядро»
- **Движок llama.cpp не редактируется.** Весь каталог форка остаётся структурно
  идентичным апстриму → обновления забираются **drop-in** (см. `scripts/update-upstream.sh`).
- **Своё** живёт только здесь, в `infcore/` (каталога нет в апстриме → нулевой
  конфликт при слиянии). Общение с движком — через C-API `include/llama.h` и
  хелперы `llama-common`. Файлы `tools/server` не правим — gateway строится рядом.
- Подробности и карта «ядро/периферия/своё» — в `../AUDIT.md` (в корне форка).

## Возможности (целевые)
- OpenAI-совместимый API (chat/completions, completions, embeddings, models) + SSE.
- Любые локальные GGUF-модели (text / embeddings / vision / audio) — как у llama.cpp.
- Multi-model registry, authn/RBAC, audit, observability (pull-метрики), RAG/agents, SDK.
- Полностью offline: нулевой исходящий трафик в рантайме (`tests/egress/`).

## Сборка
```
./infcore/scripts/build.sh           # из корня форка; профиль cpu+cuda+vulkan
# или вручную:
cmake -S infcore -B build -C infcore/cmake/profile-rf.cmake
cmake --build build -j
```
Бинари в `build/bin`: `llama-server` (движок-сервер), `infcore_gateway`, `mtmd` и т.д.

## Обновление движка из апстрима
```
./infcore/scripts/update-upstream.sh b1234   # release-тег ggml-org/llama.cpp
```

## Структура
```
cmake/profile-rf.cmake   профиль бэкендов/состава (cpu+cuda+vulkan, server+mtmd)
CMakeLists.txt           супер-проект: add_subdirectory(.. ) движка + слой infcore
gateway/                 надстройка над tools/server: OpenAI-surface, routing, policy
security/                authn / rbac / audit / secrets
observability/           pull-метрики (VictoriaMetrics/Prometheus)
registry/                реестр моделей (multi-model, идея из llm_gateway)
extensions/              agents, rag
sdk/python/              клиентский SDK
config/                  конфиги + JSON-Schema
deploy/                  docker / compose / systemd (РФ-образы)
model-toolkit/           convert/quantize/bench — offline, не рантайм
tests/egress/            проверка нулевого egress
docs/                    ARCHITECTURE, COMPLIANCE
NOTICE / sbom.cdx.json / THIRD_PARTY_LICENSES   compliance
```
