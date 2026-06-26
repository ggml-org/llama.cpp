# Аудит форка llama.cpp → внутренний продукт (offline РФ-контур)

**Источник:** `https://github.com/Nasferatuss/llama.cpp`
**Аудируемый коммит (UPSTREAM_COMMIT):** `e8ecce53b872257e173d9d491a57e7a09c7cd8cb` (2026-06-25)
**Объём:** 189 MB, 2995 файлов под git.

## Стратегия (зафиксирована с заказчиком)
**Модель: «обернуть, не трогая ядро» (wrap-not-touch fork).** Цель — максимально
собственный продукт с теми же функциями работы с моделями, что у llama.cpp,
с прицелом на разные локальные модели (не только Qwen). Из этого следует:

1. **Движок остаётся функционально полным и неизменным** — это и есть «функции
   llama.cpp по моделям». Содержимое ядра руками не редактируем.
2. **«Своё» = слой-обвязка в новых каталогах** поверх C-API `llama.h`
   (gateway, security, observability, registry, RAG/agents, SDK, deploy).
3. **Удаляем только то, что не относится к моделям**: чужое железо, периферию,
   сетевые/онлайн-пути, брендинг/CI апстрима.
4. **Обновления из оригинала остаются возможны (drop-in)** — поэтому ненужное
   предпочтительно **гасить флагами CMake**, а не удалять файлы (удаление создаёт
   merge-конфликты при синке с апстримом). Физическое удаление — только для
   небольшого compliance-набора (см. §4).

### Параметры контура (зафиксированы)
- **Модальности:** текст (chat/LLM), embeddings/reranking, vision (VLM), audio (ASR/TTS) — **все**.
- **Железо:** NVIDIA CUDA + Vulkan + CPU (обязателен).
- **Владение ядром:** обёртка без правок внутри; drop-in апдейты сохраняем.

---

## 1. Карта репозитория: ЯДРО / ПЕРИФЕРИЯ / СВОЁ

### 1.1 ЯДРО — НЕ редактировать; обновляется заменой на новый release-тег
| Путь | Роль |
|---|---|
| `ggml/` (компьют + кванты, 40 типов) | тензорный движок и бэкенды |
| `src/` libllama: `llama-model.cpp`, `llama-arch.*` (134 арки), `llama-vocab.*` (6 токенизаторов), `llama-chat.cpp` (56 шаблонов), `src/models/*` (134 граф-билдера, вкл. qwen3moe/qwen3vlmoe), `llama-kv-cache*`, `llama-context.*`, `llama-graph.*`, `llama-batch.*`, `llama-memory*` (вкл. recurrent/hybrid для Mamba/RWKV), `llama-sampler.*` | вся model-механика |
| `include/llama.h`, `include/llama-cpp.h` | **C-API = граница «чужое/своё»** |
| `common/` | общая обвязка апстрима (нужна серверу) |
| `vendor/` (cpp-httplib, nlohmann, **stb, miniaudio**, sheredom) | бандл-зависимости ядра/сервера/мультимодальности |
| `gguf-py/`, `convert_hf_to_gguf.py`, `conversion/` | конвертер HF→GGUF (offline, под любые арки) |
| `tools/mtmd/`, `tools/tts/` | мультимодальность (vision/audio) — **нужны по требованиям** |
| `tools/server/` | in-tree OpenAI-совместимый сервер (chat/completions/embeddings + SSE) — **основа gateway** |
| `cmake/`, корневой `CMakeLists.txt`, `ggml/CMakeLists.txt`, `Makefile` | сборка (минимальная сверка при апдейте) |

> Полнота ядра — осознанное требование: поддержка «любых будущих моделей» = сохранение
> всех арок, токенизаторов, шаблонов, типов квантов и типов памяти.

### 1.2 ПЕРИФЕРИЯ — гасим флагами (файлы можно оставить ради чистых апдейтов)
| Путь | Действие |
|---|---|
| `examples/`, `pocs/`, `benches/` | `LLAMA_BUILD_EXAMPLES=OFF`; не собирать. Удаление по желанию (даёт merge-шум). |
| `tools/cli`, `tools/batched-bench`, `tools/llama-bench`, `tools/perplexity`, `tools/imatrix`, `tools/cvector-generator`, `tools/export-lora`, `tools/gguf-split`, `tools/quantize`, `tools/tokenize`, `tools/parser`, `tools/completion`, `tools/fit-params`, `tools/results`, `tools/rpc` | не включать в рантайм-сборку/образ. `quantize`/`gguf-split`/`export-lora`/`imatrix` → перенести в offline `model-toolkit` (build-time, не runtime). |
| ggml-бэкенды не под наше железо: `metal`, `sycl`, `opencl`, `cann`, `musa`, `hexagon`, `openvino`, `webgpu`, `zdnn`, `zendnn`, `virtgpu`, `hip`, `rpc` | выключить флагами (см. §3) |

### 1.3 СВОЁ — новые каталоги (апстрим их не создаёт → нулевой конфликт при апдейте)
`gateway/` (надстройка над tools/server: routing, policy, OpenAI-surface), `security/`
(authn/RBAC/audit/secrets), `observability/`, `registry/` (multi-model), `extensions/`
(agents, rag), `sdk/`, `deploy/`, `model-toolkit/`, `docs/`, compliance-артефакты.

---

## 2. УДАЛИТЬ физически (минимальный compliance/branding-набор)
Только то, что нельзя оставлять в offline-контуре или что чисто брендинг апстрима.
Принимаем, что после merge с апстримом часть вернётся и потребует повторного удаления
(список держим коротким).

| Путь | Причина |
|---|---|
| `app/` (особенно `download.cpp`) | загрузка моделей из сети — нарушает offline |
| `tools/ui/` (npm/PWA веб-фронт) | заменяется нашим gateway; тянет node-зависимости из сети |
| сетевые скрипты в `scripts/` | скачивание моделей/хэшей — аудировать пофайлово и удалить сетевые |
| `media/` | брендинг/логотипы апстрима |
| `.github/`, `.devops/`, `.gemini/`, `.pi/`, `ci/`, `flake.nix` | CI/боты/Nix апстрима → заменить на GitLab CI/Jenkins РФ-контура |
| `models/*.gguf` (vocab-фикстуры) | удалить вместе с соответствующими тестами апстрима, если тесты не сохраняем |

---

## 3. ggml-бэкенды: профиль сборки (флаги, без удаления файлов)
Оставляем **cpu + cuda + vulkan**. CMake-профиль рантайма:
```
-DGGML_CUDA=ON
-DGGML_VULKAN=ON
-DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DGGML_CANN=OFF \
-DGGML_MUSA=OFF -DGGML_HEXAGON=OFF -DGGML_OPENVINO=OFF -DGGML_WEBGPU=OFF \
-DGGML_ZDNN=OFF -DGGML_ZENDNN=OFF -DGGML_VIRTGPU=OFF -DGGML_HIP=OFF -DGGML_RPC=OFF
-DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON   # tools: только нужные (server, mtmd, tts)
```
> `ggml-cpu` собирается всегда (референс/fallback). `GGML_RPC=OFF` обязателен для offline
> (сетевой распределённый бэкенд). `blas` — опционально для CPU-ускорения.
> Невкомпилированные бэкенды не входят в бинарь → attack-surface рантайма уже минимальна
> без физического удаления исходников.

---

## 4. Поддержка моделей и модальностей (проверка под требования)
- **Текст:** базово; целевая Qwen3-MoE (`qwen3moe`) поддержана из коробки
  (`src/models/qwen3moe.cpp`), патч ядра не нужен.
- **Embeddings/reranking:** ядро поддерживает pooling; сервер отдаёт `/v1/embeddings`.
- **Vision (VLM):** через `tools/mtmd` (+ `vendor/stb`). **Сохраняем.**
- **Audio (ASR/TTS):** через `tools/mtmd` + `tools/tts` (+ `vendor/miniaudio`). **Сохраняем.**
- Новые арки добавляются апстримом в `src/models/*` + `conversion/*` — приедут с drop-in
  апдейтом ядра. Это и есть «как у llama.cpp по моделям».

---

## 5. OpenAI-совместимость (обязательное требование)
База — `tools/server/` (уже реализует OpenAI-совместимые `/v1/chat/completions` со
стримингом SSE, `/v1/completions`, `/v1/embeddings`, `/v1/models`).
**Принцип:** не править `server.cpp` (иначе теряем дешёвые апдейты), а строить gateway
рядом — отдельный target/слой, переиспользующий серверные хелперы и `libllama`+`common`.
Из `llm_gateway` (Go) берём **идеи, не код**: multi-model registry с метаданными,
абстракция провайдеров, `/admin/models`, RBAC, audit, формат ошибок OpenAI.

---

## 6. Лицензии / compliance
- `LICENSE` (MIT, ggml authors) — сохранить. `licenses/LICENSE-jsonhpp` — сохранить.
- Добавить: `NOTICE`, `THIRD_PARTY_LICENSES/`, `SBOM` (CycloneDX), фиксация SHA апстрима.
- Происхождение раскрываем: запрещено «полностью собственная разработка / с нуля»;
  корректно — «на базе open-source llama.cpp (ggml authors, MIT); собственная разработка —
  слой gateway/security/SDK/расширений и сборка/поставка под РФ-контур».
- РФ-замены: образы Astra/РЕД ОС/ALT, reverse-proxy Angie, метрики VictoriaMetrics,
  RAG vector store Postgres Pro + pgvector, внутреннее PyPI-зеркало, GitLab CI/Jenkins.

---

## 7. Процедура обновления из апстрима (drop-in)
1. `git remote add upstream https://github.com/ggml-org/llama.cpp` (на машине сборки с сетью).
2. Выбрать стабильный **release-тег** `b####` (не master).
3. Влить/наложить тег; конфликты возможны только в `cmake`/корневом `CMakeLists` и в
   физически удалённом compliance-наборе (§2) — список короткий, разрешается быстро.
4. Пересобрать профиль (§3), прогнать тесты (загрузка + инференс целевых моделей,
   мультимодальность, embeddings) и **egress-тест** (ноль исходящих).
5. Обновить `UPSTREAM_COMMIT`, `NOTICE`, `SBOM`.

---

## 8. Следующий шаг
Каркас слоя «своё» (gateway/security/registry/...) разместить **внутри форка** в новых
каталогах (не отдельным vendor-репозиторием — это противоречило бы drop-in модели).
Ранее сгенерированный `infcore/`-каркас переиспользуется частично (заготовки security/
observability/sdk/deploy + compliance-шаблоны), но БЕЗ вендоринга `third_party/` и БЕЗ
C++ gateway-over-C-API — gateway строится на `tools/server`.
