# Tutorial: From Development to Production

**Duration**: 90 minutes
**Level**: Intermediate to Advanced

---

## Overview

This tutorial walks you through taking an LLM inference service from local development to production deployment, covering every step of the journey.

---

## Phase 1: Local Development (20 min)

### Step 1: Set Up Development Environment

```bash
# Create project
mkdir llm-service && cd llm-service
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn httpx prometheus-client

# Create basic structure
mkdir -p app tests deployment
```

### Step 2: Build Basic API

**app/main.py**:
```python
from fastapi import FastAPI
import httpx

app = FastAPI()
client = httpx.AsyncClient(base_url="http://localhost:8080")

@app.post("/v1/chat/completions")
async def chat(messages: list):
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": messages}
    )
    return response.json()
```

### Step 3: Test Locally

```bash
# Terminal 1: Start llama-server
llama-server -m model.gguf

# Terminal 2: Start API
uvicorn app.main:app --reload

# Terminal 3: Test
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

---

## Phase 2: Add Production Features (30 min)

### Step 1: Add Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer
import jwt

security = HTTPBearer()
SECRET_KEY = "your-secret-key"

async def verify_token(credentials = Security(security)):
    try:
        jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
    except:
        raise HTTPException(status_code=401)

@app.post("/v1/chat/completions")
async def chat(messages: list, user = Security(verify_token)):
    # ... implementation
```

### Step 2: Add Metrics

```python
from prometheus_client import Counter, Histogram

request_counter = Counter('requests_total', 'Total requests')
duration_histogram = Histogram('request_duration_seconds', 'Duration')

@app.middleware("http")
async def track_metrics(request, call_next):
    with duration_histogram.time():
        response = await call_next(request)
    request_counter.inc()
    return response
```

### Step 3: Add Health Checks

```python
@app.get("/health")
async def health():
    try:
        # Check backend
        response = await client.get("/health", timeout=5.0)
        if response.status_code == 200:
            return {"status": "healthy"}
    except:
        pass
    return {"status": "unhealthy"}, 503

@app.get("/ready")
async def ready():
    # Check if ready to serve traffic
    return {"status": "ready"}
```

### Step 4: Add Error Handling

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": str(exc),
            "request_id": request.state.request_id
        }
    )
```

---

## Phase 3: Containerization (20 min)

### Step 1: Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2: Create Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLAMA_SERVER_URL=http://llama-server:8080
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      - llama-server

  llama-server:
    image: llama-server:latest
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
    command: -m /models/model.gguf --metrics
```

### Step 3: Test Containers

```bash
docker-compose up -d
docker-compose logs -f
curl http://localhost:8000/health
```

---

## Phase 4: Kubernetes Deployment (20 min)

### Step 1: Create Kubernetes Manifests

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-api
  template:
    metadata:
      labels:
        app: llm-api
    spec:
      containers:
      - name: api
        image: llm-api:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: jwt-secret
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
```

**service.yaml**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-api
spec:
  selector:
    app: llm-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

**hpa.yaml**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Step 2: Deploy

```bash
# Create namespace
kubectl create namespace llm-production

# Create secrets
kubectl create secret generic llm-secrets \
  --from-literal=jwt-secret=your-secret \
  -n llm-production

# Deploy
kubectl apply -f deployment.yaml -n llm-production
kubectl apply -f service.yaml -n llm-production
kubectl apply -f hpa.yaml -n llm-production

# Verify
kubectl get pods -n llm-production
kubectl get svc -n llm-production
```

---

## Phase 5: Monitoring & Observability (20 min)

### Step 1: Set Up Prometheus

**prometheus-config.yaml**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    scrape_configs:
      - job_name: 'llm-api'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - llm-production
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            action: keep
            regex: llm-api
```

### Step 2: Create Grafana Dashboard

**Key Panels**:
1. Request Rate: `rate(requests_total[5m])`
2. Latency: `histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m]))`
3. Error Rate: `rate(requests_total{status="500"}[5m])`
4. Active Replicas: `kube_deployment_status_replicas{deployment="llm-api"}`

### Step 3: Set Up Alerts

**alerts.yaml**:
```yaml
groups:
  - name: production_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(requests_total{status="500"}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate in production"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m])) > 2
        for: 10m
        annotations:
          summary: "P95 latency above 2s"
```

---

## Phase 6: CI/CD Pipeline (20 min)

### GitHub Actions Workflow

**.github/workflows/deploy.yml**:
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: |
          docker build -t llm-api:${{ github.sha }} .
          docker tag llm-api:${{ github.sha }} llm-api:latest

      - name: Push to registry
        run: |
          docker push llm-api:${{ github.sha }}
          docker push llm-api:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/llm-api \
            api=llm-api:${{ github.sha }} \
            -n llm-production

      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/llm-api \
            -n llm-production

      - name: Run smoke tests
        run: |
          curl -f https://api.example.com/health
```

---

## Summary Checklist

**Development**:
- ✅ Local API working
- ✅ Tests passing
- ✅ Error handling implemented

**Production Features**:
- ✅ Authentication
- ✅ Metrics
- ✅ Health checks
- ✅ Logging

**Deployment**:
- ✅ Containerized
- ✅ Kubernetes manifests
- ✅ Auto-scaling configured
- ✅ Load balancer set up

**Monitoring**:
- ✅ Prometheus collecting metrics
- ✅ Grafana dashboard
- ✅ Alerts configured

**CI/CD**:
- ✅ Automated testing
- ✅ Automated deployment
- ✅ Rollback capability

---

## Next Steps

1. Implement blue-green deployments
2. Add distributed tracing
3. Set up disaster recovery
4. Implement cost monitoring
5. Add regional redundancy

**Congratulations!** Your LLM service is now production-ready!
