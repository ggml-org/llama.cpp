FROM ubuntu:22.04

# Zarur kutubxonalar
RUN apt-get update && apt-get install -y \
    build-essential cmake git python3 python3-pip

# Klon qilish
RUN git clone https://github.com/xoleric512/llama.cpp.git /llama.cpp
WORKDIR /llama.cpp

# Qurish
RUN mkdir build && cd build && cmake .. && make -j$(nproc)

# Models papkasini konteyner ichiga qo‘shish
VOLUME ["/models"]

# Port ochish (llama-server odatda port 8080)
EXPOSE 8080

# Entrypoint
CMD ["./build/bin/llama-server", "--port", "8080", "-m", "/models/model.gguf"]
