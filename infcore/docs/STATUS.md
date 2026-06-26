# STATUS — отчёт по проекту infcore (форк llama.cpp)

**Дата:** 2026-06-26 · **Статус:** ⏸ ПАУЗА · **Ветка:** `infcore` (в origin)

> Дубликат ключевого контекста живёт в авто-памяти ассистента
> (`llama-cpp-internal-fork.md`). Этот файл — версионируемая копия в самом репо.

---

## 1. Что это за проект
Форк `llama.cpp` → внутренняя **gateway/SDK-библиотека** для локального инференса
LLM (offline РФ-контур, реестр Минцифры).
- **Модели:** любые локальные GGUF (цель — Qwen3-MoE `qwen3moe`, поддержана из
  коробки; на Qwen не завязываемся).
- **Модальности:** текст, embeddings, vision (VLM), audio (ASR/TTS).
- **Железо:** NVIDIA CUDA + Vulkan + CPU.
- **Жёстко:** полностью offline (нулевой egress в рантайме), требования рос. ПО,
  open-source compliance без сокрытия происхождения.

## 2. Стратегия: «обернуть, не трогая ядро» (wrap-not-touch fork)
- Движок llama.cpp **не редактируем** → обновления из апстрима забираем drop-in
  (release-теги `b####`, не master).
- Всё своё — только в каталоге `infcore/` (апстрим его не создаёт → 0 конфликтов).
- Граница «чужое/своё» = C-API `include/llama.h`.
- Лишнее железо/бэкенды — гасим **флагами** CMake, исходники в ядре **не удаляем**.
- Физически удаляем только короткий compliance/branding-набор.
- Из Go-репо `llm_gateway` берём **идеи**, не код.

## 3. Состояние репозитория
```
origin/infcore  a63da4f  compliance-cleanup (143 файла удалено)
                4f8326f  каркас infcore/ + прототип OpenAI-gateway
base            e8ecce5  апстрим ggml-org/llama.cpp (2026-06-25)
```
remotes: `origin`=Nasferatuss/llama.cpp (запушено), `upstream`=ggml-org/llama.cpp.

## 4. Что СДЕЛАНО
- **Git/структура:** ветка `infcore`, upstream добавлен, слой `infcore/` внутри форка, всё в origin.
- **Сборка:** `infcore/CMakeLists.txt` встраивает движок через `add_subdirectory(..)`
  без правок апстрима (форсит `LLAMA_BUILD_COMMON=ON`); профиль
  `infcore/cmake/profile-rf.cmake` (cpu+cuda+vulkan ON; metal/sycl/opencl/cann/musa/
  hexagon/openvino/webgpu/zdnn/zendnn/virtgpu/hip/rpc=OFF; server+mtmd ON;
  UI/app/examples/tests=OFF). **Проверено на macOS:** сборка `llama-server` и
  `infcore_gateway` проходит (CUDA/Vulkan на mac не проверить — нужно целевое железо).
- **Прототип gateway** (`infcore/gateway/`, proxy-front): OpenAI-совместимый
  control-plane перед бэкендами `llama-server`. Работает (curl + mock):
  `/health`, `/v1/models`, Bearer-auth (401), `/v1/chat/completions`,
  `/v1/completions`, `/v1/embeddings`, маршрутизация по registry,
  **SSE passthrough** при `stream:true`, OpenAI-формат ошибок, `/metrics`.
- **Compliance-cleanup** (коммит a63da4f, 143 файла): удалены `app/`, `media/`,
  `.github/`, `.devops/`, `.gemini/`, `.pi/`, `ci/`, `flake.nix`,
  `build-xcframework.sh`, сетевые скрипты. Движок не тронут. `tools/ui`
  нейтрализован флагами (UI=OFF) — удалять нельзя (server-http.cpp `#include "ui.h"`).

## 5. Технические «грабли» (важно)
- `tools/ui` физически удалять НЕЛЬЗЯ — ломает сборку сервера; только флаги UI=OFF.
- Супер-проект обязан форсить `LLAMA_BUILD_COMMON=ON` (при встраивании standalone=OFF).
- `scripts/ui-assets.cmake` не удалять (нужен сборке tools/ui даже при UI=OFF).
- gateway = подпроцессы llama-server (proxy-front), не in-process — ради drop-in + изоляции падений.
- CUDA/Vulkan проверяются только на целевом железе.

## 6. Следующие шаги (по приоритету)
- 🔴 **RBAC + audit** вместо заглушек (`infcore/security/`): доступ роль→модель/
  эндпоинт + лимиты; неизменяемый локальный журнал. Главная ценность gateway.
- 🔴 **Авто-подъём `llama-server`** из registry: gateway сам стартует/health-check/
  рестарт процессов моделей (сейчас `backend_url` вручную).
- 🟡 Валидация конфига по JSON-Schema при старте (fail-fast).
- 🟡 RAG/agents (`extensions/`), реальный Python-SDK, реальный egress-тест.
- ⚪ platform-pass (убрать `scripts/apple`,`hip`), `model-toolkit` (convert/quantize offline),
  целевая сборка с CUDA/Vulkan, обкатка РФ-деплоя.

## 7. Как возобновить
Рабочий каталог `/Users/nasferatus/projects/llama/fork` (ветка `infcore`). Правки —
через worktree → commit → `git -C fork merge --ff-only <wt-branch>` → push origin
infcore → ExitWorktree remove. Сборка: `./infcore/scripts/build.sh` (целевое железо).
Обновление движка: `./infcore/scripts/update-upstream.sh b####`.
