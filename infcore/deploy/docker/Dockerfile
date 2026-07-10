# infcore — образ для offline РФ-контура. Контекст сборки = КОРЕНЬ форка.
# Интернет допустим ТОЛЬКО на стадии build; рантайм — строго offline.
# Базовые образы — из внутреннего реестра (Astra Linux / РЕД ОС / ALT).

# --- build stage ---
ARG BASE_BUILD=registry.internal/astra/base-devel:1.7
FROM ${BASE_BUILD} AS build
WORKDIR /src
COPY . /src
# CUDA-toolkit / Vulkan SDK — из внутреннего зеркала пакетов.
RUN cmake -S infcore -B build -C infcore/cmake/profile-rf.cmake \
 && cmake --build build -j"$(nproc)"

# --- runtime stage (offline) ---
# ВНИМАНИЕ: base-образ рантайма должен содержать РАНТАЙМ-библиотеки GPU под профиль
# сборки (при GGML_CUDA=ON - libcuda/CUDA runtime, при GGML_VULKAN=ON - Vulkan ICD),
# иначе llama-server не стартует. Драйвер NVIDIA пробрасывается nvidia-container-toolkit;
# CUDA/Vulkan runtime ставится из внутреннего зеркала в base-образ. Для CPU-only контура
# соберите с -DGGML_CUDA=OFF -DGGML_VULKAN=OFF (см. DEPLOY.md, раздел GPU).
ARG BASE_RUNTIME=registry.internal/astra/base:1.7
FROM ${BASE_RUNTIME}
WORKDIR /opt/infcore
# В рантайм только нужные бинари (gateway, cli, llama-server); прочие tools апстрима
# (bench/quantize/rpc-client и т.п.) в образ не тащим - меньше attack surface.
COPY --from=build /src/build/bin/infcore_gateway /src/build/bin/infcore-cli /src/build/bin/llama-server /opt/infcore/bin/
# Конфиг-ШАБЛОН (env: ключи, без секретов). В бою монтируйте свой (host=0.0.0.0).
COPY infcore/config/gateway.yaml /opt/infcore/config/gateway.yaml
COPY infcore/docs/DEPLOY.md /opt/infcore/docs/DEPLOY.md
COPY infcore/NOTICE infcore/sbom.cdx.json /opt/infcore/
# Непривилегированный пользователь + каталог под append-only audit-журнал.
RUN useradd -r -s /usr/sbin/nologin infcore \
 && mkdir -p /var/log/infcore \
 && chown infcore:infcore /var/log/infcore
USER infcore
# Модели монтируются томом, в образ не зашиваются. Управляемые llama-server gateway
# поднимает сам (config.runtime.llama_server_bin=/opt/infcore/bin/llama-server).
EXPOSE 8080
ENTRYPOINT ["/opt/infcore/bin/infcore_gateway", "/opt/infcore/config/gateway.yaml"]
