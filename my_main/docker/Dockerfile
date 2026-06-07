# Gemma 4 UA Phase 0 – streaming audio inference environment
#
# Base: CUDA 12.2 devel (matches host driver 535.x / CUDA 12.2 support)
# llama-cpp-python is built from source with GGML_CUDA=on.
# silero-vad + sounddevice are included for VAD and mic support.
#
# Build:
#   docker build -t gemma4-ua-phase0 my_main/docker/
#
# Run (WAV file):
#   ./my_main/docker/run_phase0.sh --audio my_main/sample/gemma4_audio_qa_input.wav
#
# NOTE: --mic is not supported on a headless server (no audio device).
#       Use --mic on a local desktop with an audio device attached.

FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages
#   python3.11-dev : needed to build llama-cpp-python C extension
#   cmake / build-essential : CMake build for llama.cpp
#   libsndfile1 : runtime dep for sounddevice
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        build-essential \
        cmake \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

WORKDIR /workspace

# Build llama-cpp-python with CUDA backend.
# Separated into its own layer so a requirements.txt change doesn't trigger a rebuild.
RUN CMAKE_ARGS="-DGGML_CUDA=on" \
    pip install --no-cache-dir llama-cpp-python

# Remaining Python dependencies (numpy, scipy, silero-vad, sounddevice)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The script and model files are provided at runtime via volume mounts;
# nothing large is baked into the image.
ENTRYPOINT ["python3", "/workspace/my_main/scripts/phase0_streaming.py"]
