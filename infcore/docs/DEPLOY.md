# infcore - руководство по развёртыванию (для DevOps)

Шлюз к локальным LLM в offline-контуре. Документ описывает сборку, настройку и
запуск. Рантайм строго offline (нулевой egress); интернет нужен только на сборке.

## 1. Состав

После сборки в `build/bin/` появляются три бинарника:

| Бинарь | Назначение |
|---|---|
| `infcore_gateway` | основной сервис: OpenAI-совместимый шлюз, auth, RBAC, audit, авто-подъём моделей |
| `llama-server` | движок инференса (от llama.cpp); поднимается шлюзом по требованию |
| `infcore-cli` | терминальный клиент (диагностика, управление моделями) |

## 2. Сборка (на целевом сервере)

Профиль сборки фиксирует состав и бэкенды (CUDA+Vulkan вкл., лишнее выкл.):

```sh
cmake -S infcore -B build -C infcore/cmake/profile-rf.cmake
cmake --build build -j"$(nproc)"
# артефакты: build/bin/{infcore_gateway,llama-server,infcore-cli}
```

Требования стадии сборки: CMake >= 3.21, C++17-компилятор, CUDA toolkit и/или
Vulkan SDK из внутреннего зеркала пакетов. GPU на стадии сборки не нужен.

Образ Docker: `infcore/deploy/docker/Dockerfile` (контекст = корень форка).
Базовые образы - из внутреннего реестра (Astra/РЕД ОС).

## 3. Конфигурация

Единственный конфиг - `gateway.yaml` (формат JSON). При старте он **проверяется по
встроенной JSON-Schema**; при ошибке сервис не запускается и печатает все проблемы.

Ключевые секции:
- `server` - host/port (по умолчанию 127.0.0.1:8080), таймауты.
- `security` - principals (ключ -> subject/role), roles (allowlists), audit.
- `runtime` - `llama_server_bin` (путь к llama-server), `port_range_start` (порты
  под управляемые модели), таймауты простоя/старта.
- `models` - каталог моделей (logical_name, gguf_path, modality, n_ctx, n_gpu_layers).

### Секреты (API-ключи)
Не храните ключи открытым текстом. Поле `api_key` поддерживает:
- `"env:INFCORE_KEY_ADMIN"` - значение из переменной окружения;
- `"file:/run/secrets/admin_key"` - значение из файла.

Для systemd переменные задаются в `/etc/infcore/gateway.env`, для compose - в
`gateway.env` (см. `deploy/compose/gateway.env.example`). Отсутствие заданной
переменной/файла - фатальная ошибка при старте (fail-fast).

## 4. Модели

- **Управляемые** (без `backend_url`): шлюз сам поднимает `llama-server` по первому
  запросу и гасит по простою. Требуют `runtime.llama_server_bin` и `gguf_path`.
- **Внешние** (`backend_url` задан): шлюз только проксирует на уже запущенный сервер.

Веса (`.gguf`) в образ не зашиваются - монтируются томом (`/opt/infcore/models`,
read-only). Путь в `gguf_path` должен совпадать с точкой монтирования.

## 5. Порты

| Порт | Назначение | Внешний доступ |
|---|---|---|
| 8080 | HTTP API (`/v1/*`, `/admin/*`) | через reverse-proxy (Angie/TLS) |
| 9090 | зарезервирован под метрики | внутренний |
| 8100+ | управляемые `llama-server` (по числу моделей) | **только localhost, должны быть свободны** |

`/health` и `/metrics` доступны без авторизации (для оркестратора и сбора метрик).
`/metrics` - формат Prometheus, забирается VictoriaMetrics со стороны DevOps.

## 6. Запуск

### systemd (рекомендуется для bare-metal/VM)
```sh
cp infcore/deploy/systemd/infcore-gateway.service /etc/systemd/system/
# положить бинарники в /opt/infcore/bin, конфиг в /opt/infcore/config/gateway.yaml
systemctl daemon-reload
systemctl enable --now infcore-gateway
```
Юнит использует `KillMode=control-group` - при остановке/перезапуске гасятся ВСЕ
дочерние `llama-server` (без осиротевших процессов). Это обязательно: не меняйте.

### Docker Compose
```sh
cd infcore/deploy/compose
cp gateway.env.example gateway.env     # заполнить секреты
docker compose up -d
```

## 7. Аудит-журнал

Пишется append-only в `audit.sink=file` по пути `audit.path` (по умолчанию
`/var/log/infcore/audit.log`), формат JSONL, с `fsync` на каждую запись. Каталог
должен быть на запись пользователю сервиса.

Для неизменяемости (требование контура) на стороне ОС:
```sh
chattr +a /var/log/infcore/audit.log     # только дозапись, без перезаписи/удаления
```
Ротацию настроить через logrotate с `copytruncate` (или штатным сбросом сервиса).

## 8. Дымовой тест

```sh
export INFCORE_URL=http://127.0.0.1:8080 INFCORE_KEY=<admin-ключ>
/opt/infcore/bin/infcore-cli models                 # список моделей
/opt/infcore/bin/infcore-cli chat -m <model> "тест" # проверка инференса
curl -s $INFCORE_URL/health                          # {"status":"ok",...}
```

## 9. Обновление движка

Обновления оригинального llama.cpp забираются скриптом
`infcore/scripts/update-upstream.sh` (по release-тегам). Слой `infcore/` при этом не
конфликтует. После обновления - пересобрать и прогнать дымовой тест.

## 10. Offline-инвариант

В рантайме сервис не делает исходящих соединений. Сеть контейнера - `internal: true`.
Любой выход в интернет - нарушение требований; проверяется при приёмке.
