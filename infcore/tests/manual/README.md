# infcore — ручные hardening-тесты шлюза

Проверяют прод-фиксы против **фейкового** бэкенда (реальные модели/GPU не нужны).
Дополняют автоматические `tests/unit` (ctest) и `tests/egress`.

## Состав
- `hardening_smoke.sh` — прогоняет 4 проверки, печатает PASS/FAIL, ненулевой код при провале:
  - **M5** — тело запроса сверх лимита → `413`;
  - **B3** — `mu_` супервайзера не держится во время `SIGTERM→SIGKILL`
    (`/health` отвечает мгновенно, пока «неубиваемый» бэкенд дожёвывается);
  - **F1** — `disable` во время старта бэкенда не теряется (инициатор → `502`,
    бэкенд погашен, не доживает до idle-таймаута);
  - **F2** — рантайм-сбой аудита (диск полон) → fail-closed `503` при `audit.require=true`,
    `/health` = `degraded`, громкий stderr.
- `fake_llama_server.py` — фейковый llama-server (флаги `--ready-delay`, `--ignore-sigterm`).
- `rlimit_exec.py` — запуск процесса с `RLIMIT_FSIZE` (детерминированная имитация «диск полон»).

## Запуск
```sh
# 1. Собрать шлюз (CPU-only достаточно):
cmake -S infcore -B build -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target infcore_gateway -j

# 2. Прогнать:
infcore/tests/manual/hardening_smoke.sh ./build/bin/infcore_gateway
```
Требуется `python3`, `curl`, `bash`. Ожидаемый итог: `PASS=7 FAIL=0`.

Прочие блокеры проверяются иначе: **M1** (обход egress) — в `tests/unit`
(`ctest -R infcore_unit`); **M3** (durability аудита) — стресс-нагрузкой
(параллельные запросы к работающему шлюзу, сверка числа строк в журнале).
