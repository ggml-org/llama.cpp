FROM ubuntu:24.04

# Build arguments for optional components (default: disabled)
ARG INSTALL_CUDA=false
ARG INSTALL_ROCM=false
ARG INSTALL_PYTHON_DEPS=false

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Copy in a zscaler.crt if one exists
# This allows the container to access the internet on corporate laptops
COPY zscaler.cr[t] /usr/local/share/ca-certificates/

# This tells various tools to use the system CA certificates
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV NODE_OPTIONS=--use-openssl-ca

# Update and install system dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        ca-certificates \
        cmake \
        git \
        curl \
        wget \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
        libcurl4-openssl-dev \
        libnuma-dev \
        numactl \
        hwloc-nox \
        libhwloc-dev \
        ccache \
        ninja-build \
        gdb \
        valgrind \
        sudo \
        gh && \
    update-ca-certificates && \
    apt-get autoremove -y && \
    apt-get clean
    
# Install CUDA 12.9 (conditional)
RUN if [ "$INSTALL_CUDA" = "true" ]; then \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O cuda-keyring.deb && \
        dpkg -i cuda-keyring.deb && \
        apt-get update && \
        apt-get -y install cuda-toolkit-12-9 && \
        rm cuda-keyring.deb; \
    else \
        echo "Skipping CUDA installation"; \
    fi

# Install ROCm 6.4 (conditional)
RUN if [ "$INSTALL_ROCM" = "true" ]; then \
        mkdir -p --mode=0755 /etc/apt/keyrings && \
        wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
            gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null && \
        echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.4.2 noble main" \
            | tee /etc/apt/sources.list.d/rocm.list && \
        echo 'Package: *' \
            | tee /etc/apt/preferences.d/rocm-pin-600 && \
        echo 'Pin: release o=repo.radeon.com' \
            | tee -a /etc/apt/preferences.d/rocm-pin-600 && \
        echo 'Pin-Priority: 600' \
            | tee -a /etc/apt/preferences.d/rocm-pin-600 && \
        apt-get update && \
        apt-get install -y rocm && \
        apt-get autoremove -y && \
        apt-get clean; \
    else \
        echo "Skipping ROCm installation"; \
    fi

# Install Python dependencies for gguf conversion tools (conditional)
RUN if [ "$INSTALL_PYTHON_DEPS" = "true" ]; then \
        python3 -m pip install --break-system-packages \
            numpy \
            torch \
            transformers \
            sentencepiece \
            protobuf \
            gguf; \
    else \
        echo "Skipping Python dependencies installation"; \
    fi

# Set up ccache for faster compilation
ENV PATH="/usr/lib/ccache:${PATH}"
ENV CCACHE_DIR="/tmp/ccache"
RUN mkdir -p /tmp/ccache

# Create a non-root user
RUN useradd -m -s /bin/bash developer && \
    usermod -aG sudo developer && \
    echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Fix ownership of ccache directory for developer user
RUN chown -R developer:developer /tmp/ccache

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER developer

# Set up shell environment
RUN echo 'export PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> ~/.bashrc && \
    echo 'alias ll="ls -alF"' >> ~/.bashrc && \
    echo 'alias la="ls -A"' >> ~/.bashrc && \
    echo 'alias l="ls -CF"' >> ~/.bashrc

# Expose common ports
EXPOSE 8080 8081

CMD ["/bin/bash"]
