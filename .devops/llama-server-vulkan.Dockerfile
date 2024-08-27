ARG UBUNTU_VERSION=jammy

FROM ubuntu:$UBUNTU_VERSION AS build

# Install build tools
RUN apt update && apt install -y git build-essential cmake wget

# Install Vulkan SDK and cURL
RUN wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list && \
    apt update -y && \
    apt-get install -y vulkan-sdk libcurl4-openssl-dev curl

# Build it
WORKDIR /app
COPY . .
RUN cmake -B build -DGGML_VULKAN=1 -DLLAMA_CURL=1 && \
    cmake --build build --config Release --target llama-server

# Clean up
WORKDIR /
RUN cp /app/build/bin/llama-server /llama-server && \
    rm -rf /app

ENV LC_ALL=C.utf8
# Must be set to 0.0.0.0 so it can listen to requests from host machine
ENV LLAMA_ARG_HOST=0.0.0.0

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/llama-server" ]
