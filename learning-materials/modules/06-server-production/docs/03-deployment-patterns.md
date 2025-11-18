# Deployment Patterns for LLM Services

**Learning Module**: Module 6 - Server & Production
**Estimated Reading Time**: 35 minutes
**Prerequisites**: Docker basics, cloud platform familiarity, Module 6.1-6.2
**Related Content**:
- [LLaMA Server Architecture](./01-llama-server-architecture.md)
- [Load Balancing & Scaling](./04-load-balancing-scaling.md)
- [Production Best Practices](./06-production-best-practices.md)

---

## Overview

This guide covers production deployment patterns for llama.cpp-based inference services, from simple single-server deployments to complex multi-region architectures.

### Deployment Tiers

| Tier | Scale | Infrastructure | Use Case |
|------|-------|----------------|----------|
| **Development** | 1 server | Local/VM | Testing, prototyping |
| **Small Production** | 1-3 servers | Single cloud region | Startups, MVPs |
| **Medium Production** | 4-20 servers | Multi-AZ | Growing businesses |
| **Large Production** | 20-100+ servers | Multi-region | Enterprise, high scale |

---

## Docker Deployment

### 1. Dockerfile for llama-server

**CPU-only Dockerfile**:
```dockerfile
FROM ubuntu:22.04 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp
WORKDIR /build
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /build/llama.cpp
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --config Release -j$(nproc)

# Runtime stage
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/

# Create non-root user
RUN useradd -m -u 1000 llama && \
    mkdir -p /models /data && \
    chown -R llama:llama /models /data

USER llama
WORKDIR /home/llama

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
```

**GPU (CUDA) Dockerfile**:
```dockerfile
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp with CUDA
WORKDIR /build
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /build/llama.cpp
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
RUN cmake --build build --config Release -j$(nproc)

# Runtime stage
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

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
```

### 2. Docker Compose Setup

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  llama-server:
    build:
      context: .
      dockerfile: Dockerfile
    image: llama-server:latest
    container_name: llama-inference
    restart: unless-stopped

    # GPU support (uncomment for NVIDIA GPUs)
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

    ports:
      - "8080:8080"

    volumes:
      - ./models:/models:ro
      - ./data:/data

    command: >
      --host 0.0.0.0
      --port 8080
      -m /models/llama-2-7b-chat.Q4_K_M.gguf
      -c 4096
      --parallel 4
      --cont-batching
      --metrics

    environment:
      - LLAMA_API_KEY=${LLAMA_API_KEY:-secret}

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 60s

  # Optional: Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  # Optional: Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}

volumes:
  prometheus-data:
  grafana-data:
```

**prometheus.yml**:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'llama-server'
    static_configs:
      - targets: ['llama-server:8080']
    metrics_path: '/metrics'
```

### 3. Running with Docker

```bash
# Build image
docker build -t llama-server:latest .

# Run CPU-only
docker run -d \
  --name llama-server \
  -p 8080:8080 \
  -v $(pwd)/models:/models:ro \
  llama-server:latest \
  -m /models/llama-2-7b.gguf

# Run with GPU
docker run -d \
  --gpus all \
  --name llama-server-gpu \
  -p 8080:8080 \
  -v $(pwd)/models:/models:ro \
  llama-server:latest \
  -m /models/llama-2-7b.gguf \
  -ngl 35

# Using docker-compose
docker-compose up -d

# View logs
docker-compose logs -f llama-server

# Scale up
docker-compose up -d --scale llama-server=3
```

---

## Kubernetes Deployment

### 1. Basic Deployment

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-server
  namespace: inference
  labels:
    app: llama-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llama-server
  template:
    metadata:
      labels:
        app: llama-server
    spec:
      containers:
      - name: llama-server
        image: llama-server:v1.0.0
        imagePullPolicy: IfNotPresent

        command:
          - /usr/local/bin/llama-server
        args:
          - --host
          - "0.0.0.0"
          - --port
          - "8080"
          - -m
          - /models/llama-2-7b-chat.Q4_K_M.gguf
          - -c
          - "4096"
          - --parallel
          - "4"
          - --cont-batching
          - --metrics

        ports:
        - containerPort: 8080
          name: http
          protocol: TCP

        env:
        - name: LLAMA_API_KEY
          valueFrom:
            secretKeyRef:
              name: llama-secrets
              key: api-key

        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"

        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true

        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
          readOnly: true

      # Anti-affinity to spread pods across nodes
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - llama-server
              topologyKey: kubernetes.io/hostname
```

### 2. GPU Node Deployment

**gpu-deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-server-gpu
  namespace: inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llama-server-gpu
  template:
    metadata:
      labels:
        app: llama-server-gpu
    spec:
      containers:
      - name: llama-server
        image: llama-server:v1.0.0-cuda
        args:
          - --host
          - "0.0.0.0"
          - -m
          - /models/llama-2-13b-chat.Q4_K_M.gguf
          - -ngl
          - "43"  # Offload all layers to GPU
          - -c
          - "4096"
          - --parallel
          - "8"

        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"

        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true

      # Node selector for GPU nodes
      nodeSelector:
        nvidia.com/gpu: "true"
        node.kubernetes.io/instance-type: "g5.2xlarge"

      # Tolerations for GPU taints
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

### 3. Service & Ingress

**service.yaml**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: llama-server
  namespace: inference
  labels:
    app: llama-server
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  selector:
    app: llama-server
```

**ingress.yaml**:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llama-server-ingress
  namespace: inference
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: llama-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: llama-server
            port:
              number: 80
```

### 4. Persistent Volume for Models

**pvc.yaml**:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
  namespace: inference
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: efs-sc  # AWS EFS for shared access
```

**ConfigMap for models**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: inference
data:
  models.json: |
    {
      "models": [
        {
          "name": "llama-2-7b-chat",
          "path": "/models/llama-2-7b-chat.Q4_K_M.gguf",
          "context": 4096,
          "gpu_layers": 35
        },
        {
          "name": "llama-2-13b-chat",
          "path": "/models/llama-2-13b-chat.Q4_K_M.gguf",
          "context": 4096,
          "gpu_layers": 43
        }
      ]
    }
```

---

## Cloud Platform Deployments

### 1. AWS Deployment

#### EC2 Single Instance

**Launch Configuration**:
```bash
#!/bin/bash
# EC2 User Data script

# Update system
apt-get update && apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install NVIDIA driver (for GPU instances)
apt-get install -y nvidia-driver-535
apt-get install -y nvidia-docker2

# Download model
aws s3 cp s3://my-models/llama-2-7b-chat.gguf /models/

# Run llama-server
docker run -d \
  --gpus all \
  --name llama-server \
  -p 8080:8080 \
  -v /models:/models:ro \
  --restart unless-stopped \
  llama-server:latest \
  -m /models/llama-2-7b-chat.gguf \
  -ngl 35

# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb
```

**Terraform Configuration**:
```hcl
# main.tf
provider "aws" {
  region = "us-east-1"
}

resource "aws_instance" "llama_server" {
  ami           = "ami-0c55b159cbfafe1f0"  # Ubuntu 22.04
  instance_type = "g5.xlarge"  # GPU instance

  vpc_security_group_ids = [aws_security_group.llama_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.llama_profile.name

  user_data = file("user-data.sh")

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  tags = {
    Name = "llama-server"
    Environment = "production"
  }
}

resource "aws_security_group" "llama_sg" {
  name        = "llama-server-sg"
  description = "Security group for LLaMA server"

  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]  # Internal only
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_lb" "llama_alb" {
  name               = "llama-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = data.aws_subnets.public.ids

  tags = {
    Name = "llama-alb"
  }
}

resource "aws_lb_target_group" "llama_tg" {
  name     = "llama-tg"
  port     = 8080
  protocol = "HTTP"
  vpc_id   = data.aws_vpc.main.id

  health_check {
    enabled             = true
    path                = "/health"
    port                = "8080"
    protocol            = "HTTP"
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
  }
}
```

#### ECS Deployment

**task-definition.json**:
```json
{
  "family": "llama-server",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "llama-server",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/llama-server:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/models/llama-2-7b-chat.gguf"
        }
      ],
      "secrets": [
        {
          "name": "API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:llama-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/llama-server",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### 2. Google Cloud Platform (GCP)

**GKE Deployment**:
```yaml
# gke-cluster.yaml
apiVersion: container.gke.io/v1
kind: NodePool
metadata:
  name: gpu-pool
spec:
  cluster: llama-cluster
  initialNodeCount: 2
  config:
    machineType: n1-standard-4
    guestAccelerator:
      - type: nvidia-tesla-t4
        count: 1
    oauthScopes:
      - https://www.googleapis.com/auth/cloud-platform
  autoscaling:
    enabled: true
    minNodeCount: 1
    maxNodeCount: 10
```

**Cloud Run Deployment**:
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: llama-server
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "10"
    spec:
      containerConcurrency: 4
      containers:
      - image: gcr.io/project/llama-server:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            memory: 16Gi
            cpu: "4"
        env:
        - name: MODEL_PATH
          value: /models/llama-2-7b-chat.gguf
```

### 3. Azure Deployment

**AKS with GPU**:
```bash
# Create resource group
az group create --name llama-rg --location eastus

# Create AKS cluster with GPU node pool
az aks create \
  --resource-group llama-rg \
  --name llama-cluster \
  --node-count 1 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Add GPU node pool
az aks nodepool add \
  --resource-group llama-rg \
  --cluster-name llama-cluster \
  --name gpupool \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --node-taints sku=gpu:NoSchedule

# Get credentials
az aks get-credentials --resource-group llama-rg --name llama-cluster
```

**Azure Container Instances**:
```bash
az container create \
  --resource-group llama-rg \
  --name llama-server \
  --image llama-server:latest \
  --cpu 4 \
  --memory 16 \
  --ports 8080 \
  --environment-variables \
    MODEL_PATH=/models/llama-2-7b-chat.gguf \
  --secure-environment-variables \
    API_KEY=secret123 \
  --command-line \
    "llama-server -m /models/llama-2-7b-chat.gguf --host 0.0.0.0"
```

---

## Edge Deployment Patterns

### 1. CDN Edge Computing

**Cloudflare Workers**:
```javascript
// Proxy to nearest llama-server instance
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  // Route to nearest server based on geolocation
  const geo = request.cf
  const serverUrl = getClosestServer(geo.country)

  // Forward request
  return fetch(serverUrl + new URL(request.url).pathname, {
    method: request.method,
    headers: request.headers,
    body: request.body
  })
}

function getClosestServer(country) {
  const servers = {
    'US': 'https://us-east.llama.example.com',
    'EU': 'https://eu-west.llama.example.com',
    'ASIA': 'https://asia-east.llama.example.com'
  }
  return servers[country] || servers['US']
}
```

### 2. On-Premises Deployment

**Bare Metal Setup**:
```bash
#!/bin/bash
# Production server setup

# Install dependencies
apt-get update
apt-get install -y build-essential cmake nvidia-driver-535

# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Install as systemd service
cat > /etc/systemd/system/llama-server.service <<EOF
[Unit]
Description=LLaMA Inference Server
After=network.target

[Service]
Type=simple
User=llama
Group=llama
WorkingDirectory=/opt/llama
ExecStart=/opt/llama/llama-server \\
  -m /models/llama-2-7b-chat.gguf \\
  --host 0.0.0.0 \\
  --port 8080 \\
  -ngl 35 \\
  --parallel 8 \\
  --cont-batching
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
systemctl daemon-reload
systemctl enable llama-server
systemctl start llama-server
```

---

## Multi-Region Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Global Load Balancer                        │
│                  (Route 53 / CloudFlare)                      │
└─────────┬───────────────────────┬────────────────────────────┘
          │                       │
          ▼                       ▼
┌──────────────────────┐  ┌──────────────────────┐
│   us-east-1          │  │   eu-west-1          │
│ ┌──────────────────┐ │  │ ┌──────────────────┐ │
│ │  Regional ALB    │ │  │ │  Regional ALB    │ │
│ └────────┬─────────┘ │  │ └────────┬─────────┘ │
│          │           │  │          │           │
│    ┌─────┴─────┐     │  │    ┌─────┴─────┐     │
│    │           │     │  │    │           │     │
│    ▼           ▼     │  │    ▼           ▼     │
│  ┌───┐       ┌───┐  │  │  ┌───┐       ┌───┐  │
│  │Pod│       │Pod│  │  │  │Pod│       │Pod│  │
│  └───┘       └───┘  │  │  └───┘       └───┘  │
│                      │  │                      │
└──────────────────────┘  └──────────────────────┘
```

---

## Summary

**Key Deployment Patterns**:
1. **Containerization**: Docker for consistency and portability
2. **Orchestration**: Kubernetes for scaling and management
3. **Cloud Platforms**: AWS, GCP, Azure for managed infrastructure
4. **Multi-Region**: Deploy across regions for low latency
5. **Edge Computing**: CDN integration for global reach

**Next Steps**:
- [Load Balancing & Scaling](./04-load-balancing-scaling.md)
- [Monitoring & Observability](./05-monitoring-observability.md)
- Lab 6.3: Deploy to Kubernetes

---

**Interview Topics**:
- Container orchestration strategies
- Cloud platform selection criteria
- Multi-region architecture design
- Infrastructure as Code (Terraform)
- Cost optimization techniques
