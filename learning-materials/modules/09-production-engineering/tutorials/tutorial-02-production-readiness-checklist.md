# Tutorial: Production Readiness Checklist

## Introduction

This tutorial provides a comprehensive checklist and walkthrough for deploying llama.cpp inference systems to production.

**Time Required**: 2-3 hours to complete all steps

## The Production Readiness Checklist

### Phase 1: Code Quality ✅

#### 1.1 Testing

- [ ] **Unit tests** with >80% coverage
- [ ] **Integration tests** for all endpoints
- [ ] **Performance benchmarks** established
- [ ] **Regression tests** in place

**Verify**:
```bash
pytest tests/ --cov=. --cov-report=term
# Coverage should be >80%
```

#### 1.2 Code Review

- [ ] Code follows project style guide
- [ ] No TODO or FIXME in production code
- [ ] Error handling comprehensive
- [ ] Logging appropriate (not excessive)

**Verify**:
```bash
# Check for TODOs
grep -r "TODO\|FIXME" src/ --exclude-dir=tests

# Run linter
pylint src/ --rcfile=.pylintrc
```

### Phase 2: Security 🔒

#### 2.1 Vulnerability Scanning

- [ ] **No HIGH/CRITICAL vulnerabilities**
- [ ] Dependencies up to date
- [ ] Security headers configured
- [ ] No secrets in code

**Check**:
```bash
# Scan Docker image
trivy image llama-inference:latest --severity HIGH,CRITICAL

# Check Python dependencies
safety check

# Scan for secrets
trufflehog filesystem .
```

#### 2.2 Input Validation

- [ ] Prompt validation implemented
- [ ] Rate limiting configured
- [ ] Authentication required
- [ ] Authorization checked

**Implementation**:
```python
from fastapi import HTTPException, Security

async def validate_input(prompt: str, max_length=4096):
    if len(prompt) > max_length:
        raise HTTPException(400, "Prompt too long")

    # Check for injection
    if "ignore previous" in prompt.lower():
        raise HTTPException(400, "Invalid input")

    return prompt
```

#### 2.3 Secure Deployment

- [ ] Running as non-root user
- [ ] TLS/HTTPS enabled
- [ ] Secrets in vault/env vars
- [ ] Resource limits set

**Kubernetes Example**:
```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: llama
        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
```

### Phase 3: Performance 🚀

#### 3.1 Benchmarks

- [ ] **Latency targets** met (P95 < 2s)
- [ ] **Throughput targets** met (>100 req/s)
- [ ] Load testing completed
- [ ] Bottlenecks identified

**Measure**:
```python
import time
import statistics

def benchmark(url, num_requests=100):
    latencies = []
    for _ in range(num_requests):
        start = time.time()
        requests.post(f"{url}/completion", json={"prompt": "Test", "max_tokens": 10})
        latencies.append(time.time() - start)

    return {
        'p50': statistics.median(latencies),
        'p95': sorted(latencies)[int(len(latencies) * 0.95)],
        'throughput': num_requests / sum(latencies)
    }

results = benchmark("http://localhost:8080")
assert results['p95'] < 2.0, "P95 latency too high"
```

#### 3.2 Optimization

- [ ] Model quantization optimal (Q4_K_M or Q5_K_M)
- [ ] Batching implemented
- [ ] Caching configured
- [ ] GPU utilization >70%

**Monitor**:
```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Should see >70% utilization under load
```

### Phase 4: Reliability 💪

#### 4.1 Error Handling

- [ ] All errors logged
- [ ] Graceful degradation
- [ ] Circuit breakers configured
- [ ] Retry logic implemented

**Example**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def call_model(prompt):
    try:
        return model(prompt)
    except Exception as e:
        logger.error(f"Model call failed: {e}")
        raise
```

#### 4.2 Health Checks

- [ ] Liveness probe configured
- [ ] Readiness probe configured
- [ ] Startup probe configured

**Implementation**:
```python
@app.get("/health/live")
async def liveness():
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness():
    # Check if model loaded
    if model is None:
        raise HTTPException(503, "Model not loaded")
    return {"status": "ready"}
```

### Phase 5: Observability 👁️

#### 5.1 Logging

- [ ] Structured logging (JSON)
- [ ] Log levels appropriate
- [ ] Correlation IDs used
- [ ] Sensitive data not logged

**Setup**:
```python
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def log_request(request_id, prompt, latency):
    logging.info(json.dumps({
        'timestamp': datetime.utcnow().isoformat(),
        'request_id': request_id,
        'prompt_length': len(prompt),
        'latency_ms': latency * 1000
    }))
```

#### 5.2 Metrics

- [ ] **Prometheus metrics** exported
- [ ] **Key metrics** tracked (latency, throughput, errors)
- [ ] **Alerts** configured
- [ ] **Dashboards** created

**Expose Metrics**:
```python
from prometheus_client import Counter, Histogram

requests_total = Counter('requests_total', 'Total requests', ['status'])
request_duration = Histogram('request_duration_seconds', 'Request duration')

@request_duration.time()
def process_request():
    # Process...
    requests_total.labels(status='success').inc()
```

#### 5.3 Tracing

- [ ] Distributed tracing enabled
- [ ] Spans capture key operations
- [ ] Traces exported to backend

### Phase 6: Deployment 🚢

#### 6.1 Infrastructure

- [ ] **Kubernetes manifests** tested
- [ ] **Auto-scaling** configured
- [ ] **Load balancing** set up
- [ ] **Rollback** procedure documented

**Auto-scaling**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### 6.2 CI/CD

- [ ] **Automated testing** in pipeline
- [ ] **Security scanning** in pipeline
- [ ] **Staging deployment** automated
- [ ] **Production deployment** requires approval

**GitHub Actions**:
```yaml
jobs:
  deploy-staging:
    needs: [test, security-scan]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        run: kubectl apply -f k8s/staging/

  deploy-production:
    needs: deploy-staging
    environment: production
    # Requires manual approval
```

#### 6.3 Monitoring

- [ ] **Alerts** firing correctly
- [ ] **On-call** rotation set up
- [ ] **Runbooks** created
- [ ] **Dashboards** accessible

### Phase 7: Documentation 📚

#### 7.1 Technical Documentation

- [ ] API documentation complete
- [ ] Deployment guide available
- [ ] Architecture diagram created
- [ ] Troubleshooting guide written

#### 7.2 Operational Documentation

- [ ] Runbooks for common issues
- [ ] Rollback procedures documented
- [ ] Scaling guide available
- [ ] Incident response plan ready

**Example Runbook**:
```markdown
# Runbook: High Memory Usage

## Symptoms
- Memory alerts firing
- OOM kills occurring

## Investigation
1. Check current memory: `kubectl top pods`
2. Review logs: `kubectl logs -f pod-name`
3. Check for memory leaks

## Resolution
- Scale up: `kubectl scale deployment llama --replicas=5`
- Restart pods: `kubectl rollout restart deployment llama`

## Prevention
- Enable memory limits
- Set up memory-based autoscaling
```

### Phase 8: Compliance 📋

#### 8.1 Data Privacy

- [ ] **PII handling** documented
- [ ] **Data retention** policy set
- [ ] **Audit logging** enabled
- [ ] **GDPR compliance** verified

#### 8.2 Legal

- [ ] **Terms of service** accepted
- [ ] **Model license** reviewed
- [ ] **Usage limits** enforced

## Final Verification

### Pre-Launch Checklist

```bash
# 1. Run all tests
pytest tests/ -v

# 2. Security scan
trivy image llama-inference:latest

# 3. Performance test
python performance_test.py

# 4. Smoke test deployment
curl https://staging.example.com/health

# 5. Review dashboards
open https://grafana.example.com

# 6. Test alerts
python trigger_test_alert.py

# 7. Verify rollback
kubectl rollout undo deployment/llama-inference --dry-run

# 8. Review documentation
grep -r "TODO" docs/
```

### Launch Day

```markdown
# Launch Day Checklist

- [ ] All team members notified
- [ ] Monitoring dashboards open
- [ ] On-call engineer available
- [ ] Rollback plan ready
- [ ] Deploy to production
- [ ] Monitor for 1 hour
- [ ] Validate metrics
- [ ] Announce success
```

## Post-Launch

### Week 1

- Monitor error rates daily
- Review performance metrics
- Collect user feedback
- Address any issues

### Week 2-4

- Optimize based on real usage
- Update documentation
- Conduct retrospective
- Plan improvements

## Summary

Production readiness requires:

1. ✅ **Thorough testing** (unit, integration, performance)
2. ✅ **Strong security** (scanning, validation, authentication)
3. ✅ **Good performance** (benchmarks, optimization)
4. ✅ **High reliability** (error handling, health checks)
5. ✅ **Full observability** (logging, metrics, tracing)
6. ✅ **Solid deployment** (CI/CD, auto-scaling)
7. ✅ **Complete documentation** (technical, operational)
8. ✅ **Compliance** (privacy, legal)

Use this checklist before every production deployment!

## Resources

- [Google SRE Book](https://sre.google/sre-book/)
- [Production Readiness Review](https://gruntwork.io/devops-checklist/)
- Module 9 Documentation
