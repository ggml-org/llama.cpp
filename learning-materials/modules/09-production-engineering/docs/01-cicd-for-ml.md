# CI/CD for ML Inference Systems

## Introduction

Continuous Integration and Continuous Deployment (CI/CD) for ML inference systems requires special considerations beyond traditional software development. Unlike conventional applications, ML systems involve models, data, and inference code that must all be versioned, tested, and deployed together.

This lesson covers how to build robust CI/CD pipelines specifically designed for llama.cpp-based inference systems.

## Why CI/CD for ML is Different

### Traditional Software CI/CD
- Code → Build → Test → Deploy
- Binary artifacts (executables, libraries)
- Deterministic behavior
- Version control: Git

### ML Inference CI/CD
- Code + Model + Data → Build → Test → Deploy
- Multiple artifact types (code, models, configs)
- Non-deterministic behavior (model outputs)
- Version control: Git + Model registry
- Additional testing: quality, performance, regression

## Core Principles

### 1. Everything as Code

**Infrastructure as Code (IaC)**
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-inference
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: llama-server
        image: llama-cpp:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

**Configuration as Code**
```yaml
# config/model-config.yaml
model:
  name: llama-2-7b-chat
  format: gguf
  quantization: Q4_K_M
  context_size: 4096

inference:
  temperature: 0.7
  top_p: 0.9
  max_tokens: 512
```

### 2. Automated Testing

All changes must pass automated tests before deployment:
- Unit tests (code correctness)
- Integration tests (system behavior)
- Performance tests (latency, throughput)
- Quality tests (model output quality)

### 3. Reproducibility

Every build must be reproducible:
- Pin all dependencies (exact versions)
- Use containerization (Docker)
- Version all artifacts (code, models, data)
- Document build environment

### 4. Gradual Rollout

Deploy changes incrementally to minimize risk:
- Canary deployment (5% → 50% → 100%)
- Blue-green deployment
- Feature flags
- Easy rollback

## CI/CD Pipeline Stages

### Stage 1: Code Integration (CI)

```yaml
# .github/workflows/ci.yml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3
      with:
        submodules: recursive

    - name: Set up build environment
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake build-essential

    - name: Build llama.cpp
      run: |
        mkdir build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        cmake --build . --config Release -j$(nproc)

    - name: Run unit tests
      run: |
        cd build
        ctest --output-on-failure

    - name: Run integration tests
      run: |
        python -m pytest tests/integration/ -v

    - name: Build artifacts
      run: |
        tar -czf llama-cpp-build.tar.gz build/

    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: llama-cpp-binaries
        path: llama-cpp-build.tar.gz
```

### Stage 2: Model Testing

```yaml
  model-quality-test:
    runs-on: ubuntu-latest
    needs: build-and-test

    steps:
    - name: Download artifacts
      uses: actions/download-artifact@v3

    - name: Download test model
      run: |
        wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

    - name: Run perplexity test
      run: |
        ./build/bin/llama-perplexity \
          -m llama-2-7b-chat.Q4_K_M.gguf \
          -f tests/data/validation.txt \
          --perplexity

    - name: Run quality tests
      run: |
        python tests/quality/test_outputs.py \
          --model llama-2-7b-chat.Q4_K_M.gguf \
          --benchmark tests/data/qa_pairs.json
```

### Stage 3: Performance Testing

```yaml
  performance-test:
    runs-on: ubuntu-latest
    needs: build-and-test

    steps:
    - name: Run performance benchmarks
      run: |
        ./build/bin/llama-bench \
          -m models/llama-2-7b-chat.Q4_K_M.gguf \
          -p 512 -n 128 -r 10

    - name: Check performance regression
      run: |
        python scripts/check_performance.py \
          --current results/current-bench.json \
          --baseline results/baseline-bench.json \
          --threshold 0.05  # 5% regression tolerance
```

### Stage 4: Container Build

```yaml
  build-container:
    runs-on: ubuntu-latest
    needs: [build-and-test, model-quality-test, performance-test]

    steps:
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ghcr.io/${{ github.repository }}/llama-inference:${{ github.sha }}
          ghcr.io/${{ github.repository }}/llama-inference:latest
        cache-from: type=registry,ref=ghcr.io/${{ github.repository }}/llama-inference:buildcache
        cache-to: type=registry,ref=ghcr.io/${{ github.repository }}/llama-inference:buildcache,mode=max
```

### Stage 5: Deployment (CD)

```yaml
# .github/workflows/deploy.yml
name: Continuous Deployment

on:
  workflow_run:
    workflows: ["Continuous Integration"]
    types: [completed]
    branches: [main]

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}

    steps:
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG_STAGING }}

    - name: Deploy to staging
      run: |
        kubectl set image deployment/llama-inference \
          llama-server=ghcr.io/${{ github.repository }}/llama-inference:${{ github.sha }} \
          -n staging

        kubectl rollout status deployment/llama-inference -n staging

    - name: Run smoke tests
      run: |
        python tests/smoke/test_endpoints.py \
          --url https://staging.inference.example.com

  deploy-production:
    runs-on: ubuntu-latest
    needs: deploy-staging
    environment: production

    steps:
    - name: Deploy to production (canary)
      run: |
        # Deploy to 10% of traffic
        kubectl set image deployment/llama-inference-canary \
          llama-server=ghcr.io/${{ github.repository }}/llama-inference:${{ github.sha }} \
          -n production

        # Wait and monitor
        sleep 300

        # Check metrics
        python scripts/check_canary_metrics.py

    - name: Promote to full production
      if: success()
      run: |
        kubectl set image deployment/llama-inference \
          llama-server=ghcr.io/${{ github.repository }}/llama-inference:${{ github.sha }} \
          -n production
```

## Multi-Stage Dockerfile for Inference

```dockerfile
# Dockerfile
# Stage 1: Build environment
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source code
COPY . .

# Build llama.cpp with CUDA support
RUN mkdir build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CUDA=ON \
    -DLLAMA_CUDA_F16=ON && \
    cmake --build . --config Release -j$(nproc)

# Stage 2: Runtime environment
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 llama && \
    mkdir -p /models /app && \
    chown -R llama:llama /models /app

USER llama
WORKDIR /app

# Copy binaries from builder
COPY --from=builder /build/build/bin/llama-server /app/
COPY --from=builder /build/build/bin/llama-cli /app/

# Health check script
COPY scripts/healthcheck.sh /app/
RUN chmod +x /app/healthcheck.sh

# Expose server port
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD ["/app/healthcheck.sh"]

ENTRYPOINT ["/app/llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
```

## Model Versioning and Artifact Management

### Model Registry Structure

```
models/
├── llama-2-7b-chat/
│   ├── v1.0/
│   │   ├── model.gguf
│   │   ├── metadata.json
│   │   └── performance_report.json
│   ├── v1.1/
│   │   ├── model.gguf
│   │   ├── metadata.json
│   │   └── performance_report.json
│   └── latest -> v1.1
```

### Metadata Format

```json
{
  "model_name": "llama-2-7b-chat",
  "version": "1.1",
  "format": "GGUF",
  "quantization": "Q4_K_M",
  "base_model": "meta-llama/Llama-2-7b-chat-hf",
  "created_at": "2025-11-18T10:00:00Z",
  "created_by": "ml-team",
  "parameters": {
    "num_parameters": "7B",
    "context_length": 4096,
    "vocab_size": 32000
  },
  "performance": {
    "perplexity": 5.67,
    "tokens_per_second": 45.3,
    "memory_gb": 4.2
  },
  "tests_passed": {
    "quality": true,
    "performance": true,
    "security": true
  },
  "checksum": "sha256:abc123...",
  "download_url": "https://models.example.com/llama-2-7b-chat/v1.1/model.gguf"
}
```

## Deployment Strategies

### 1. Blue-Green Deployment

```yaml
# Two identical environments: blue (current) and green (new)
apiVersion: v1
kind: Service
metadata:
  name: llama-inference
spec:
  selector:
    app: llama-inference
    version: blue  # Switch to 'green' after validation
  ports:
  - port: 8080
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-inference-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llama-inference
      version: blue
  template:
    metadata:
      labels:
        app: llama-inference
        version: blue
    spec:
      containers:
      - name: llama-server
        image: llama-inference:v1.0
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-inference-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llama-inference
      version: green
  template:
    metadata:
      labels:
        app: llama-inference
        version: green
    spec:
      containers:
      - name: llama-server
        image: llama-inference:v1.1  # New version
```

**Deployment Process:**
1. Deploy new version to green environment
2. Test green environment thoroughly
3. Switch traffic from blue to green (update service selector)
4. Monitor for issues
5. Keep blue environment for quick rollback if needed

### 2. Canary Deployment

```yaml
# Primary deployment (90% traffic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-inference-stable
spec:
  replicas: 9
  selector:
    matchLabels:
      app: llama-inference
      track: stable
  template:
    metadata:
      labels:
        app: llama-inference
        track: stable
    spec:
      containers:
      - name: llama-server
        image: llama-inference:v1.0
---
# Canary deployment (10% traffic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-inference-canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llama-inference
      track: canary
  template:
    metadata:
      labels:
        app: llama-inference
        track: canary
    spec:
      containers:
      - name: llama-server
        image: llama-inference:v1.1  # New version
---
# Service routes to both
apiVersion: v1
kind: Service
metadata:
  name: llama-inference
spec:
  selector:
    app: llama-inference  # Matches both stable and canary
  ports:
  - port: 8080
```

**Canary Process:**
1. Deploy canary with new version (10% traffic)
2. Monitor metrics (latency, errors, quality)
3. Gradually increase canary traffic (10% → 25% → 50% → 100%)
4. Promote canary to stable once validated
5. Rollback if issues detected

### 3. Rolling Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-inference
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2        # Max 2 extra pods during update
      maxUnavailable: 1  # Max 1 pod down during update
  selector:
    matchLabels:
      app: llama-inference
  template:
    metadata:
      labels:
        app: llama-inference
    spec:
      containers:
      - name: llama-server
        image: llama-inference:v1.1
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
```

## GitLab CI Alternative

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - package
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

build:
  stage: build
  image: ubuntu:22.04
  script:
    - apt-get update && apt-get install -y cmake build-essential
    - mkdir build && cd build
    - cmake .. -DCMAKE_BUILD_TYPE=Release
    - cmake --build . --config Release -j$(nproc)
  artifacts:
    paths:
      - build/
    expire_in: 1 day

test:unit:
  stage: test
  image: ubuntu:22.04
  dependencies:
    - build
  script:
    - cd build && ctest --output-on-failure

test:integration:
  stage: test
  image: python:3.11
  dependencies:
    - build
  script:
    - pip install pytest
    - pytest tests/integration/ -v

test:performance:
  stage: test
  dependencies:
    - build
  script:
    - ./scripts/run_benchmarks.sh
    - python scripts/check_performance.py
  artifacts:
    reports:
      performance: performance.json

package:
  stage: package
  image: docker:24.0
  services:
    - docker:24.0-dind
  dependencies:
    - build
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main
    - tags

deploy:staging:
  stage: deploy
  image: alpine/kubectl:latest
  script:
    - kubectl config use-context staging
    - kubectl set image deployment/llama-inference llama-server=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - kubectl rollout status deployment/llama-inference
  environment:
    name: staging
    url: https://staging.inference.example.com
  only:
    - main

deploy:production:
  stage: deploy
  image: alpine/kubectl:latest
  script:
    - kubectl config use-context production
    - kubectl set image deployment/llama-inference llama-server=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - kubectl rollout status deployment/llama-inference
  environment:
    name: production
    url: https://inference.example.com
  when: manual
  only:
    - tags
```

## Monitoring CI/CD Pipeline

### Pipeline Metrics to Track

```python
# scripts/track_pipeline_metrics.py
import datetime
from prometheus_client import Counter, Histogram, Gauge

# Counters
pipeline_runs = Counter('cicd_pipeline_runs_total', 'Total pipeline runs', ['status', 'branch'])
test_failures = Counter('cicd_test_failures_total', 'Test failures', ['test_type'])
deployments = Counter('cicd_deployments_total', 'Deployments', ['environment'])

# Histograms
build_duration = Histogram('cicd_build_duration_seconds', 'Build duration')
test_duration = Histogram('cicd_test_duration_seconds', 'Test duration', ['test_type'])
deployment_duration = Histogram('cicd_deployment_duration_seconds', 'Deployment duration')

# Gauges
model_size_mb = Gauge('model_artifact_size_mb', 'Model size in MB')
container_size_mb = Gauge('container_image_size_mb', 'Container image size in MB')

def record_pipeline_run(status, branch, duration):
    pipeline_runs.labels(status=status, branch=branch).inc()
    build_duration.observe(duration)

def record_deployment(environment, duration):
    deployments.labels(environment=environment).inc()
    deployment_duration.observe(duration)
```

## Rollback Procedures

### Automated Rollback

```yaml
# scripts/automated_rollback.yml
- name: Deploy with automatic rollback
  hosts: production
  tasks:
    - name: Deploy new version
      kubernetes.core.k8s:
        state: present
        definition: "{{ lookup('file', 'deployment.yaml') }}"
      register: deployment

    - name: Wait for rollout
      kubernetes.core.k8s_info:
        kind: Deployment
        name: llama-inference
        namespace: production
      register: deployment_status
      until: deployment_status.resources[0].status.updatedReplicas == deployment_status.resources[0].spec.replicas
      retries: 10
      delay: 30

    - name: Check error rate
      uri:
        url: http://prometheus:9090/api/v1/query
        method: POST
        body_format: form-urlencoded
        body:
          query: "rate(http_requests_total{status=~'5..'}[5m])"
      register: error_rate

    - name: Rollback if error rate too high
      kubernetes.core.k8s:
        state: present
        definition:
          apiVersion: apps/v1
          kind: Deployment
          metadata:
            name: llama-inference
            namespace: production
          spec:
            rollbackTo:
              revision: 0  # Previous revision
      when: error_rate.json.data.result[0].value[1] | float > 0.01
```

### Manual Rollback

```bash
#!/bin/bash
# scripts/rollback.sh

# List deployment history
kubectl rollout history deployment/llama-inference -n production

# Rollback to previous version
kubectl rollout undo deployment/llama-inference -n production

# Rollback to specific revision
kubectl rollout undo deployment/llama-inference -n production --to-revision=5

# Verify rollback
kubectl rollout status deployment/llama-inference -n production
```

## Best Practices

### 1. Fast Feedback
- Keep pipeline fast (< 15 minutes for full pipeline)
- Run quick tests first, slow tests later
- Parallelize independent stages
- Use caching extensively

### 2. Clear Responsibilities
- Define code owners for different components
- Require reviews before merging
- Automate what can be automated
- Clear escalation paths for failures

### 3. Security
- Scan for vulnerabilities in CI
- Sign artifacts and images
- Use secrets management (not env vars in code)
- Least privilege for service accounts

### 4. Observability
- Log all pipeline steps
- Track metrics (duration, success rate)
- Alert on failures
- Dashboard for pipeline health

### 5. Documentation
- Document pipeline architecture
- Explain why each stage exists
- Troubleshooting guides
- Runbooks for common issues

## Common Pitfalls

1. **Flaky Tests**: Tests that pass/fail randomly break confidence
   - Solution: Identify and fix or quarantine flaky tests

2. **Slow Pipelines**: Long pipelines delay feedback
   - Solution: Optimize, parallelize, cache

3. **No Rollback Plan**: Deployments fail without recovery path
   - Solution: Always have automated rollback

4. **Missing Monitoring**: Can't tell if deployment succeeded
   - Solution: Comprehensive post-deployment validation

5. **Poor Secret Management**: Secrets leaked in logs or code
   - Solution: Use proper secret management tools

## Summary

Key takeaways for ML inference CI/CD:

- **Automate Everything**: Testing, building, deployment, rollback
- **Version Everything**: Code, models, data, configs
- **Test Thoroughly**: Unit, integration, performance, quality
- **Deploy Gradually**: Canary, blue-green, feature flags
- **Monitor Constantly**: Metrics, logs, alerts
- **Rollback Quickly**: Automated detection and rollback
- **Learn Continuously**: Post-mortems, metrics, iteration

## Further Reading

- [Google SRE Book - Release Engineering](https://sre.google/sre-book/release-engineering/)
- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Kubernetes Deployment Strategies](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)

## Next Steps

1. **Hands-On**: Complete Lab 9.1 to build your first CI/CD pipeline
2. **Practice**: Set up GitHub Actions for a personal project
3. **Explore**: Try different deployment strategies
4. **Advanced**: Implement automated rollback based on metrics

---

**Authors**: Agent 5 (Documentation Specialist)
**Last Updated**: 2025-11-18
**Estimated Reading Time**: 60 minutes
