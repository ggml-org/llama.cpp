# Базовые образы infcore (build + runtime)

`infcore/deploy/docker/Dockerfile` ссылается на два образа из внутреннего реестра:

| ARG | По умолчанию | Назначение |
|---|---|---|
| `BASE_BUILD` | `registry.internal/astra/base-devel:1.7` | стадия сборки (toolchain, SDK) |
| `BASE_RUNTIME` | `registry.internal/astra/base:1.7` | стадия рантайма (только рантайм-либы) |

Этих образов **нет в апстриме** — их собирает и публикует DevOps контура один раз.
Здесь лежат **скелеты-рецепты** (`Dockerfile.base-devel`, `Dockerfile.base-runtime`):
адаптируйте `FROM` под вашу отечественную ОС (Astra / РЕД ОС / ALT) и имена пакетов
её репозитория, соберите и запушьте во внутренний реестр под теми же тегами (или
переопределите `--build-arg BASE_BUILD=... BASE_RUNTIME=...` при сборке основного образа).

## Что ОБЯЗАНО быть в build-образе
- C++17-компилятор (gcc/g++ >= 11), `make`, `ninja` (опц.), `git` (для номера сборки);
- CMake >= 3.21;
- **CUDA toolkit** (nvcc) — версия под ваш парк GPU (arch пинуется в `profile-rf.cmake`:
  `75;80;86;89;90`); GPU на стадии сборки НЕ нужен, т.к. `native` не используется;
- **Vulkan SDK** — заголовки + `glslc`/`glslangValidator` (нужны для компиляции шейдеров
  ggml-vulkan на этапе сборки).
- Всё — из внутреннего зеркала пакетов (интернет доступен ТОЛЬКО на стадии build-образа).

## Что ОБЯЗАНО быть в runtime-образе
- glibc + libstdc++ (совместимые с build-образом), `libgomp` (OpenMP);
- **CUDA runtime** (`libcudart`, `libcublas`) — при `GGML_CUDA=ON`;
- **Vulkan loader** (`libvulkan.so` + ICD вашего драйвера) — при `GGML_VULKAN=ON`;
- НИКАКИХ toolchain/SDK (меньше attack surface). Драйвер NVIDIA пробрасывается
  `nvidia-container-toolkit` с хоста, в образ не кладётся.
- Для **CPU-only** контура ни CUDA, ни Vulkan рантайм не нужны — соберите основной
  образ из профиля с `-DGGML_CUDA=OFF -DGGML_VULKAN=OFF`.

## Проверка совместимости
Версия CUDA runtime в base-runtime должна быть >= CUDA toolkit в base-devel; Vulkan
loader — не старше SDK. Иначе `llama-server` не стартует, gateway вернёт `502 backend
start failed` (при `audit.require=true` это громкая ошибка, не тихий простой).
