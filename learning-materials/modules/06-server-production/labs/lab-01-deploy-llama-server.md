# Lab 6.1: Deploy llama-server

**Difficulty**: Beginner
**Estimated Time**: 45-60 minutes
**Prerequisites**:
- Module 1 complete
- Docker installed (optional)
- Access to a model file (.gguf)

---

## Learning Objectives

By completing this lab, you will:
1. Deploy llama-server in different configurations
2. Test the OpenAI-compatible API
3. Monitor server health and metrics
4. Understand configuration parameters
5. Troubleshoot common deployment issues

---

## Part 1: Basic Deployment (15 min)

### Step 1: Prepare the Environment

```bash
# Create project directory
mkdir -p ~/llm-server-lab
cd ~/llm-server-lab

# Create directories
mkdir -p models logs

# Download a small model (or use your own)
# Example: Download a 7B quantized model
# wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf \
#   -O models/llama-2-7b-chat.Q4_K_M.gguf
```

### Step 2: Build llama.cpp (if not already done)

```bash
# Clone repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with appropriate backend
# CPU only:
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# With CUDA:
# cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
# cmake --build build --config Release

# Verify build
./build/bin/llama-server --version
```

### Step 3: Start the Server

```bash
# Basic server launch
./build/bin/llama-server \
  -m ../models/llama-2-7b-chat.Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 2048 \
  --log-format text

# Server should start and display:
# - Model loading progress
# - Server listening address
# - Available endpoints
```

**Expected Output**:
```
llama_model_loader: loaded meta data with 19 key-value pairs
llama_model_loader: - type  : llama
llama_model_loader: - arch  : llama
llama server listening at http://0.0.0.0:8080
```

### Step 4: Test the Server

Open a new terminal and test the API:

```bash
# Health check
curl http://localhost:8080/health

# List models
curl http://localhost:8080/v1/models

# Test chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

**✅ Checkpoint**: You should receive a JSON response with the model's answer.

---

## Part 2: GPU-Accelerated Deployment (15 min)

### Step 1: Configure GPU Offloading

```bash
# Check available GPUs
nvidia-smi

# Launch with GPU acceleration
./build/bin/llama-server \
  -m ../models/llama-2-7b-chat.Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 4096 \
  -ngl 35 \
  --parallel 4 \
  --log-format json

# Explanation of parameters:
# -ngl 35      : Offload 35 layers to GPU (adjust based on VRAM)
# --parallel 4 : Handle 4 requests simultaneously
# -c 4096      : Context size of 4096 tokens
```

### Step 2: Monitor GPU Usage

```bash
# In another terminal, watch GPU utilization
watch -n 1 nvidia-smi

# Make several concurrent requests to see parallel processing
for i in {1..4}; do
  curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "llama-2-7b-chat",
      "messages": [{"role": "user", "content": "Tell me a short story"}],
      "max_tokens": 200
    }' &
done
wait
```

**✅ Checkpoint**: nvidia-smi should show GPU utilization and memory usage.

---

## Part 3: Production Configuration (15 min)

### Step 1: Enable Metrics and Advanced Features

Create a startup script:

```bash
cat > start-server.sh <<'EOF'
#!/bin/bash

MODEL_PATH="$1"
PORT="${2:-8080}"
GPU_LAYERS="${3:-35}"

./build/bin/llama-server \
  -m "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  -c 4096 \
  -ngl "$GPU_LAYERS" \
  --parallel 8 \
  --cont-batching \
  --metrics \
  --log-format json \
  --api-key "${LLAMA_API_KEY:-secret-key}" \
  2>&1 | tee logs/server-$(date +%Y%m%d-%H%M%S).log
EOF

chmod +x start-server.sh

# Run server
export LLAMA_API_KEY="your-secret-key"
./start-server.sh ../models/llama-2-7b-chat.Q4_K_M.gguf 8080 35
```

### Step 2: Test Authentication

```bash
# Request without API key (should fail)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-2-7b-chat", "messages": [{"role": "user", "content": "Hello"}]}'

# Request with API key (should succeed)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{"model": "llama-2-7b-chat", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Step 3: Check Metrics

```bash
# View Prometheus metrics
curl http://localhost:8080/metrics | grep llama

# Key metrics to observe:
# - llama_requests_total
# - llama_tokens_generated_total
# - llama_slots_available
# - llama_slots_processing
```

**✅ Checkpoint**: Metrics endpoint returns Prometheus-formatted data.

---

## Part 4: Docker Deployment (15 min)

### Step 1: Create Dockerfile

```dockerfile
cat > Dockerfile <<'EOF'
FROM ubuntu:22.04 as builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /build/llama.cpp
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --config Release

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/

RUN useradd -m -u 1000 llama && \
    mkdir -p /models /data && \
    chown -R llama:llama /models /data

USER llama
WORKDIR /home/llama

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
EOF
```

### Step 2: Build and Run

```bash
# Build image
docker build -t llama-server:latest .

# Run container
docker run -d \
  --name llama-server \
  -p 8080:8080 \
  -v $(pwd)/models:/models:ro \
  llama-server:latest \
  -m /models/llama-2-7b-chat.Q4_K_M.gguf \
  -c 2048

# Check logs
docker logs -f llama-server

# Test
curl http://localhost:8080/health
```

**✅ Checkpoint**: Server runs in Docker container and responds to requests.

---

## Part 5: Performance Testing (15 min)

### Step 1: Install Testing Tools

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Or use hey
wget https://hey-release.s3.us-east-2.amazonaws.com/hey_linux_amd64
chmod +x hey_linux_amd64
sudo mv hey_linux_amd64 /usr/local/bin/hey
```

### Step 2: Create Test Payload

```bash
cat > test-request.json <<'EOF'
{
  "model": "llama-2-7b-chat",
  "messages": [
    {"role": "user", "content": "Explain quantum computing in one sentence."}
  ],
  "max_tokens": 50
}
EOF
```

### Step 3: Run Load Tests

```bash
# Sequential requests
echo "=== Sequential Test ==="
time for i in {1..10}; do
  curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @test-request.json > /dev/null
done

# Concurrent requests
echo "=== Concurrent Test ==="
hey -n 100 -c 10 -m POST \
  -H "Content-Type: application/json" \
  -D test-request.json \
  http://localhost:8080/v1/chat/completions

# Observe metrics during load
curl -s http://localhost:8080/metrics | grep -E "llama_(requests|slots)"
```

### Step 4: Analyze Results

**Questions to answer**:
1. What is the average response time?
2. How many requests per second can the server handle?
3. What is the slot utilization during peak load?
4. Did any requests fail?

---

## Troubleshooting

### Issue: Server won't start

**Symptoms**: Server crashes or fails to load model

**Solutions**:
```bash
# Check model file
ls -lh models/

# Verify model format
file models/llama-2-7b-chat.Q4_K_M.gguf

# Check available memory
free -h

# Reduce context size
./build/bin/llama-server -m models/llama-2-7b-chat.Q4_K_M.gguf -c 1024
```

### Issue: Out of memory

**Solutions**:
```bash
# Reduce GPU layers
./build/bin/llama-server -m models/... -ngl 20

# Reduce parallel slots
./build/bin/llama-server -m models/... --parallel 2

# Use smaller quantization
# Download Q4_0 instead of Q4_K_M
```

### Issue: Slow inference

**Diagnosis**:
```bash
# Check GPU usage
nvidia-smi

# Check CPU usage
top

# Verify GPU layers are being used
./build/bin/llama-server -m models/... -ngl 35 --verbose
```

---

## Deliverables

Submit the following:

1. **Screenshot**: Server startup logs showing model loading
2. **Screenshot**: Successful API response from `/v1/chat/completions`
3. **Metrics Output**: Output from `/metrics` endpoint
4. **Load Test Results**: Output from hey command
5. **Answers**:
   - How many tokens/second did you achieve?
   - What was the slot utilization under load?
   - What configuration gave the best performance?

---

## Challenge Exercises

1. **Multi-Model Setup**: Run two different models on different ports
2. **Streaming**: Implement a client that handles streaming responses
3. **Benchmarking**: Compare performance with different quantization levels
4. **Monitoring**: Set up Prometheus to scrape metrics every 15 seconds

---

## Key Takeaways

- llama-server provides an OpenAI-compatible API
- GPU acceleration significantly improves performance
- Metrics are essential for monitoring production services
- Docker provides portable deployment
- Performance depends on: model size, quantization, GPU layers, context size

**Next Lab**: [Lab 6.2 - Build Custom API Wrapper](./lab-02-custom-api-wrapper.md)
