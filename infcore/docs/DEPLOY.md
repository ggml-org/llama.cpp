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

### GPU в рантайме
Профиль по умолчанию собирает `llama-server` под CUDA+Vulkan, поэтому в рантайме нужны:
- **NVIDIA:** драйвер хоста + `nvidia-container-toolkit` (для Docker проброс `--gpus`/
  `deploy.resources.devices`), а в base-образе рантайма - CUDA runtime-библиотеки
  (`libcudart` и т.п.) из внутреннего зеркала.
- **Vulkan:** пакет Vulkan ICD/loader в base-образе рантайма.

Base-образ рантайма ОБЯЗАН содержать эти библиотеки, иначе `llama-server` не стартует
(шлюз вернёт 502 backend start failed). Состав base-образов и скелеты-рецепты:
`infcore/deploy/docker/base/`.

Для **CPU-only** контура пересоберите без GPU. ВАЖНО: при встраивании через
`add_subdirectory` `llama-server` по умолчанию НЕ собирается — нужно явно включить
сервер и инструменты (профиль `profile-rf.cmake` это делает, «голая» команда — нет):
```sh
cmake -S infcore -B build -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
      -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```
Без `-DLLAMA_BUILD_SERVER=ON` соберётся только `infcore_gateway`, а `llama-server`
не будет — шлюз стартует, но каждый запрос к управляемой модели даст 502.

> Примечание: `cmake -S .` (конфигурация КОРНЯ форка, не `infcore/`) требует
> `-DLLAMA_BUILD_APP=OFF` — каталог `app/` удалён при compliance-cleanup, а апстрим
> включает его при standalone-сборке. Штатные пути (`-S infcore` / профиль) не задеты.

## 3. Конфигурация

Единственный конфиг - `gateway.yaml` (формат JSON). При старте он **проверяется по
встроенной JSON-Schema**; при ошибке сервис не запускается и печатает все проблемы.

Ключевые секции:
- `server` - host/port (по умолчанию 127.0.0.1:8080), `max_concurrent_requests`,
  `request_timeout_ms` (по умолчанию 120000). ВНИМАНИЕ: этот таймаут применяется и как
  read-timeout к бэкенду. Для НЕ-стриминговых запросов ответ приходит одним куском в
  конце генерации, поэтому длинные генерации (>120 c) на слабом железе оборвутся 502 -
  поднимите `request_timeout_ms` под ваш worst-case или используйте `stream:true`.
  - `max_concurrent_requests` (по умолчанию 64) = размер пула воркеров и потолок
    одновременных запросов. SSE-стрим держит воркер весь стрим, поэтому значение
    должно превышать ожидаемое число одновременных стримов (+ запас на /health,
    /metrics, /admin).
  - **Периметровые лимиты** (защита публичного/полу-доверенного периметра):
    `read_timeout_ms` (30000) рвёт медленную отправку запроса (slowloris);
    `write_timeout_ms` (120000) освобождает воркер, если потребитель SSE «умер» и не
    читает; `max_body_bytes` (8 МиБ) режет гигантские тела до буферизации -> без OOM
    на `json::parse`. Тело сверх лимита получает `413`. Для сугубо-loopback-деплоя
    значения по умолчанию можно оставить как есть.
- `security` - principals (ключ -> subject/role), roles (allowlists), audit.
- `runtime` - `llama_server_bin` (путь к llama-server), `port_range_start` (порты
  под управляемые модели), таймауты простоя/старта.
- `models` - каталог моделей (logical_name, gguf_path, modality, n_ctx, n_gpu_layers,
  `mmproj_path`). Для управляемых моделей модальности `vision` поле
  `mmproj_path` (проектор mtmd) ОБЯЗАТЕЛЬНО - иначе fail-fast при старте.

В Docker шлюз должен слушать `0.0.0.0` (наружу публикуется только `127.0.0.1:8080`).
Не редактируйте смонтированный read-only конфиг — задайте переменные окружения
**`INFCORE_HOST`** / **`INFCORE_PORT`** (имеют приоритет над `server.host`/`server.port`).
В `docker-compose.yml` уже прописано `INFCORE_HOST=0.0.0.0`. Управляемые `llama-server`
всё равно остаются на loopback и под per-boot ключом, так что это безопасно.

### Аудит обязателен (fail-fast)
При `security.audit.sink="file"` и `audit.require=true` (по умолчанию) шлюз **не
стартует**, если журнал не открылся (нет каталога/прав) — чтобы не отдавать трафик с
молча выключенным аудитом. Убедитесь, что `audit.path` (по умолчанию
`/var/log/infcore/audit.log`) доступен на запись пользователю сервиса:
- **systemd** создаёт каталог сам (`LogsDirectory=infcore`, владелец `User=`);
- **Docker/compose**: смонтированный `/var/log/infcore` должен принадлежать uid
  пользователя `infcore` из образа — `chown` на хосте перед первым запуском, иначе
  контейнер упадёт на старте (громко, а не тихо).

Осознанно снять требование (не для контура): `audit.require=false`.

### Секреты (API-ключи)
Не храните ключи открытым текстом. Поле `api_key` поддерживает:
- `"env:INFCORE_KEY_ADMIN"` - значение из переменной окружения;
- `"file:/run/secrets/admin_key"` - значение из файла.

Для systemd переменные задаются в `/etc/infcore/gateway.env`, для compose - в
`gateway.env` (см. `deploy/compose/gateway.env.example`). Отсутствие заданной
переменной/файла - фатальная ошибка при старте (fail-fast). Ключи-заглушки вида
`change-me*` также отвергаются на старте.

## 4. Модели

- **Управляемые** (без `backend_url`): шлюз сам поднимает `llama-server` по первому
  запросу и гасит по простою. Требуют `runtime.llama_server_bin` и `gguf_path`.
- **Внешние** (`backend_url` задан): шлюз только проксирует на уже запущенный сервер.

Веса (`.gguf`) в образ не зашиваются - монтируются томом (`/opt/infcore/models`,
read-only). Путь в `gguf_path` должен совпадать с точкой монтирования.

## 5. Порты

| Порт | Назначение | Внешний доступ |
|---|---|---|
| 8080 | HTTP API (`/v1/*`, `/admin/*`, `/health`, `/metrics`) | через reverse-proxy (Angie/TLS) |
| 8100+ | управляемые `llama-server` (по числу моделей) | **только localhost + per-boot api-key** |

`/health` и `/metrics` доступны без авторизации (для оркестратора и сбора метрик).
`/metrics` - формат Prometheus на том же порту 8080, забирается VictoriaMetrics со
стороны DevOps. Отдельного порта метрик нет.

Управляемые `llama-server` слушают только `127.0.0.1` и требуют per-boot `--api-key`
(генерируется шлюзом при старте). Достучаться до них в обход RBAC/audit нельзя, даже
если шлюз слушает `0.0.0.0`.

## 6. Запуск

### systemd (рекомендуется для bare-metal/VM)
```sh
# 1. сервисный пользователь и каталоги (один раз, от root):
useradd -r -s /usr/sbin/nologin infcore
install -d -o infcore -g infcore /opt/infcore/bin /opt/infcore/config
# 2. положить бинарники в /opt/infcore/bin, конфиг в /opt/infcore/config/gateway.yaml
#    (каталог журнала /var/log/infcore создаст сам systemd — LogsDirectory=)
# 3. секреты-ключи — в /etc/infcore/gateway.env (см. «Секреты» выше)
cp infcore/deploy/systemd/infcore-gateway.service /etc/systemd/system/
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
`/var/log/infcore/audit.log`), формат JSONL, каждое событие фиксируется на диск
(`fsync`) прежде, чем запрос завершится — потери событий нет даже при отключении
питания. Запись вынесена в отдельный поток-писатель с **group-commit**: параллельные
запросы делят один `fsync`, поэтому флуд (в т.ч. отказы 401/403) не сериализуется на
диске и не становится вектором DoS. Каталог должен быть на запись пользователю сервиса.

Для неизменяемости (требование контура) на стороне ОС:
```sh
chattr +a /var/log/infcore/audit.log     # только дозапись, без перезаписи/удаления
```

**Ротация несовместима с `chattr +a` напрямую** (нельзя ни `copytruncate`, ни
переименовать/удалить append-only файл), и `copytruncate` в любом случае некорректен:
шлюз держит открытый fd с `O_APPEND` и продолжит писать по нему. Выберите одну схему:

- **С неизменяемостью (рекомендуется для контура):** ротация со снятием флага и
  перезапуском сервиса (по control-group корректно закрывает fd):
  ```
  /var/log/infcore/audit.log {
      weekly
      rotate 52
      missingok
      prerotate  /usr/bin/chattr -a /var/log/infcore/audit.log
      postrotate systemctl restart infcore-gateway   # переоткрывает журнал; chattr +a ставится сервисом-обёрткой/через prerotate следующего цикла
      endscript
  }
  ```
  Либо не ротируйте штатно, а архивируйте по дате внешним сборщиком журналов (auditd/SIEM).
- **Без неизменяемости:** обычный logrotate с `postrotate systemctl restart
  infcore-gateway` (НЕ `copytruncate`).

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

В рантайме сервис не делает исходящих соединений за пределы контура. Важно: движок
`llama-server` (апстрим, не редактируется по стратегии wrap-not-touch) СОДЕРЖИТ код
сетевой загрузки (флаги `-hf`/`--model-url`, догрузка картинок по URL). Шлюз никогда
не передаёт таких аргументов, но инвариант обеспечивается на уровне инфраструктуры:

- **systemd:** `IPAddressDeny=any` + `IPAddressAllow=localhost` в юните (добавьте
  подсеть клиентов в allow);
- **Docker:** сеть `internal: true` (без выхода наружу);
- **config:** `offline.enforce_no_egress: true` - шлюз при старте отвергает внешние
  `backend_url`, не указывающие в loopback/RFC1918;
- при жёстких требованиях - nftables/firewall egress-deny на хосте.

Механизм проверяется тестом `infcore/tests/egress` (netns) и на приёмке. Любой выход
в интернет - нарушение требований.
