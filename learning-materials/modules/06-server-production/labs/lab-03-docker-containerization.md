# Lab 6.3: Docker Containerization

**Difficulty**: Intermediate
**Estimated Time**: 45 minutes
**Prerequisites**: Lab 6.1-6.2, Docker basics

---

## Objectives

1. Containerize llama-server
2. Create multi-stage Docker builds
3. Use Docker Compose for full stack
4. Implement health checks
5. Optimize image size

---

## Part 1: Create Optimized Dockerfile (15 min)

**Dockerfile.llama** (Multi-stage build):
```dockerfile
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 as builder

RUN apt-get update && apt-get install -y build-essential cmake git wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /build/llama.cpp
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
RUN cmake --build build --config Release

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/

RUN useradd -m llama && mkdir -p /models
USER llama

EXPOSE 8080
HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["llama-server"]
```

**Build and test**:
```bash
docker build -f Dockerfile.llama -t llama-server:v1 .
docker run -d --gpus all -p 8080:8080 -v ./models:/models llama-server:v1 \
  -m /models/llama-2-7b-chat.Q4_K_M.gguf -ngl 35
```

---

## Part 2: Docker Compose Stack (20 min)

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  llama-server:
    build:
      context: .
      dockerfile: Dockerfile.llama
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    command: -m /models/llama-2-7b-chat.Q4_K_M.gguf -ngl 35 --metrics

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

**prometheus.yml**:
```yaml
scrape_configs:
  - job_name: 'llama'
    static_configs:
      - targets: ['llama-server:8080']
```

**Launch stack**:
```bash
docker-compose up -d
docker-compose ps
docker-compose logs -f llama-server
```

---

## Part 3: Optimization & Best Practices (10 min)

**Tasks**:
1. Reduce image size (use alpine base where possible)
2. Add .dockerignore
3. Use BuildKit for faster builds
4. Implement graceful shutdown
5. Add resource limits

**.dockerignore**:
```
.git
*.md
tests/
venv/
__pycache__/
```

**Build with BuildKit**:
```bash
DOCKER_BUILDKIT=1 docker build -t llama-server:optimized .
```

**Resource limits in docker-compose.yml**:
```yaml
services:
  llama-server:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
        reservations:
          devices:
            - capabilities: [gpu]
              count: 1
```

---

## Deliverables

1. Multi-stage Dockerfile
2. Working docker-compose stack
3. Screenshot of running containers
4. Image size comparison

---

## Challenge

Create a CI/CD pipeline to build and push images

**Next Lab**: [Lab 6.4 - Kubernetes Deployment](./lab-04-kubernetes-deployment.md)
