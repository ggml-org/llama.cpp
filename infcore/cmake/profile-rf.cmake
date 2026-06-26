# infcore — профиль сборки под РФ-контур (cache-init).
# Применяется так:  cmake -S infcore -B build -C infcore/cmake/profile-rf.cmake
# Задаёт флаги движка ДО конфигурации llama.cpp. Сами файлы апстрима не меняются.

# --- Бэкенды ggml: оставляем cpu (всегда) + cuda + vulkan ---------------------
set(GGML_CUDA      ON  CACHE BOOL "" FORCE)
set(GGML_VULKAN    ON  CACHE BOOL "" FORCE)

# Выключаем всё, что не под наше железо / нарушает offline:
set(GGML_METAL     OFF CACHE BOOL "" FORCE)
set(GGML_SYCL      OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL    OFF CACHE BOOL "" FORCE)
set(GGML_CANN      OFF CACHE BOOL "" FORCE)
set(GGML_MUSA      OFF CACHE BOOL "" FORCE)
set(GGML_HEXAGON   OFF CACHE BOOL "" FORCE)
set(GGML_OPENVINO  OFF CACHE BOOL "" FORCE)
set(GGML_WEBGPU    OFF CACHE BOOL "" FORCE)
set(GGML_ZDNN      OFF CACHE BOOL "" FORCE)
set(GGML_ZENDNN    OFF CACHE BOOL "" FORCE)
set(GGML_VIRTGPU   OFF CACHE BOOL "" FORCE)
set(GGML_HIP       OFF CACHE BOOL "" FORCE)
set(GGML_RPC       OFF CACHE BOOL "" FORCE)   # offline: без сетевого RPC-бэкенда
# BLAS — опционально для CPU-ускорения; по умолчанию off
set(GGML_BLAS      OFF CACHE BOOL "" FORCE)

# --- Состав сборки llama.cpp --------------------------------------------------
# Сервер и mtmd НУЖНЫ (gateway строится на tools/server; mtmd — мультимодальность).
set(LLAMA_BUILD_SERVER   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
# Примеры/тесты апстрима в рантайм-сборке не нужны:
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)

set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
