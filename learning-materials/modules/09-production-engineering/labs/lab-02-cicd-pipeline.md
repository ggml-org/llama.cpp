# Lab 2: CI/CD Pipeline Setup

## Objectives

- ✅ Configure GitHub Actions for automated testing
- ✅ Build Docker images in CI
- ✅ Implement automated deployment
- ✅ Set up rollback procedures
- ✅ Configure notifications

**Estimated Time**: 3-4 hours

## Part 1: GitHub Actions Workflow

### Task 1.1: Create Basic CI Workflow

Create `.github/workflows/ci.yml`:

```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3
      with:
        submodules: recursive

    - name: Build llama.cpp
      run: |
        mkdir build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        cmake --build . -j$(nproc)

    - name: Run tests
      run: |
        cd build && ctest --output-on-failure

    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: llama-binaries
        path: build/bin/
```

**✏️ Task**: Add caching to speed up builds

### Task 1.2: Add Multi-Platform Testing

Extend workflow to test on multiple OS:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest]
    build_type: [Release, Debug]
```

### Task 1.3: Integration Tests in CI

Add integration testing step:

```yaml
integration-test:
  needs: build-and-test
  runs-on: ubuntu-latest

  steps:
    - uses: actions/checkout@v3

    - name: Download artifacts
      uses: actions/download-artifact@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Run integration tests
      run: |
        pip install pytest requests
        pytest tests/integration/ -v
```

## Part 2: Docker Build Pipeline

### Task 2.1: Dockerfile for CI

Create optimized Dockerfile:

```dockerfile
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y cmake build-essential

COPY . /build
WORKDIR /build

RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . -j$(nproc)

FROM ubuntu:22.04

COPY --from=builder /build/build/bin/llama-server /app/
WORKDIR /app

EXPOSE 8080
CMD ["./llama-server"]
```

### Task 2.2: Add Docker Build to Workflow

```yaml
build-docker:
  needs: integration-test
  runs-on: ubuntu-latest

  steps:
    - uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
```

**✏️ Task**: Add registry login and multi-arch builds

## Part 3: Deployment Pipeline

### Task 3.1: Deploy to Staging

```yaml
deploy-staging:
  needs: build-docker
  if: github.ref == 'refs/heads/main'
  runs-on: ubuntu-latest
  environment: staging

  steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/llama-inference \
          llama-server=ghcr.io/${{ github.repository }}:${{ github.sha }} \
          -n staging

    - name: Wait for rollout
      run: |
        kubectl rollout status deployment/llama-inference -n staging

    - name: Run smoke tests
      run: |
        curl https://staging.example.com/health
```

### Task 3.2: Production Deployment with Approval

```yaml
deploy-production:
  needs: deploy-staging
  runs-on: ubuntu-latest
  environment:
    name: production
    url: https://example.com

  steps:
    - name: Deploy canary
      run: |
        kubectl set image deployment/llama-canary \
          llama-server=ghcr.io/${{ github.repository }}:${{ github.sha }}

    - name: Monitor metrics
      run: |
        python scripts/check_canary.py --duration 300

    - name: Promote to production
      if: success()
      run: |
        kubectl set image deployment/llama-inference \
          llama-server=ghcr.io/${{ github.repository }}:${{ github.sha }}
```

## Part 4: Rollback Procedures

### Task 4.1: Automated Rollback

Create `scripts/rollback.sh`:

```bash
#!/bin/bash

DEPLOYMENT="llama-inference"
NAMESPACE="production"

# Get previous revision
CURRENT_REV=$(kubectl rollout history deployment/$DEPLOYMENT -n $NAMESPACE | tail -n 1 | awk '{print $1}')
PREVIOUS_REV=$((CURRENT_REV - 1))

echo "Rolling back from revision $CURRENT_REV to $PREVIOUS_REV"

# Rollback
kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE

# Wait for rollback
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE

echo "Rollback complete"
```

### Task 4.2: Health Check and Auto-Rollback

```yaml
- name: Health check
  id: health
  run: |
    for i in {1..10}; do
      if curl -f https://example.com/health; then
        echo "healthy=true" >> $GITHUB_OUTPUT
        exit 0
      fi
      sleep 10
    done
    echo "healthy=false" >> $GITHUB_OUTPUT
    exit 1

- name: Rollback on failure
  if: steps.health.outputs.healthy == 'false'
  run: ./scripts/rollback.sh
```

## Part 5: Notifications

### Task 5.1: Slack Notifications

```yaml
- name: Notify Slack
  if: always()
  uses: slackapi/slack-github-action@v1
  with:
    payload: |
      {
        "text": "Deployment ${{ job.status }}",
        "status": "${{ job.status }}"
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### Task 5.2: Email on Failure

```yaml
- name: Send email on failure
  if: failure()
  uses: dawidd6/action-send-mail@v3
  with:
    server_address: smtp.gmail.com
    server_port: 465
    username: ${{ secrets.EMAIL_USERNAME }}
    password: ${{ secrets.EMAIL_PASSWORD }}
    subject: CI/CD Pipeline Failed
    body: Pipeline failed for ${{ github.sha }}
```

## Verification

1. Push code and observe workflow execution
2. Check that all stages complete successfully
3. Verify deployment to staging
4. Test rollback procedure
5. Confirm notifications are received

## Deliverables

- ✅ Working CI/CD pipeline
- ✅ Automated testing and deployment
- ✅ Rollback procedures
- ✅ Notification system
- ✅ Documentation of pipeline

## Challenge Tasks

1. Add security scanning (Trivy, Snyk)
2. Implement blue-green deployment
3. Add performance testing gate
4. Set up GitLab CI alternative
5. Create dashboard for pipeline metrics

---

**Next**: Lab 3: Performance Testing
