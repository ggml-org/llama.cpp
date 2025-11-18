# Incident Response for LLM Systems

## Introduction

Production systems will experience incidents. The key is to detect, respond, and recover quickly while learning from each incident. This lesson covers incident response procedures, debugging techniques, and building resilient systems.

## Observability: The Three Pillars

### 1. Metrics

```python
# monitoring/prometheus_metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary
import time

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'request_duration_seconds',
    'Request duration',
    ['endpoint'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

# Model metrics
model_inference_duration = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_name'],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

tokens_generated_total = Counter(
    'tokens_generated_total',
    'Total tokens generated',
    ['model_name']
)

# System metrics
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization', ['gpu_id'])
gpu_memory_used = Gauge('gpu_memory_used_bytes', 'GPU memory used', ['gpu_id'])
active_requests = Gauge('active_requests', 'Currently active requests')
queue_depth = Gauge('request_queue_depth', 'Requests waiting in queue')

# Error metrics
errors_total = Counter(
    'errors_total',
    'Total errors',
    ['error_type', 'endpoint']
)

class MetricsMiddleware:
    async def __call__(self, request, call_next):
        active_requests.inc()
        start = time.time()

        try:
            response = await call_next(request)
            status = response.status_code

            # Record metrics
            http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=status
            ).inc()

            duration = time.time() - start
            request_duration.labels(endpoint=request.url.path).observe(duration)

            return response

        except Exception as e:
            errors_total.labels(
                error_type=type(e).__name__,
                endpoint=request.url.path
            ).inc()
            raise

        finally:
            active_requests.dec()

# Export metrics
from prometheus_client import make_asgi_app
metrics_app = make_asgi_app()
```

### 2. Logging

```python
# monitoring/structured_logging.py
import logging
import json
from datetime import datetime
from typing import Any, Dict

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(self.JSONFormatter())
        self.logger.addHandler(handler)

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
            }

            # Add extra fields
            if hasattr(record, 'extra'):
                log_data.update(record.extra)

            # Add exception info
            if record.exc_info:
                log_data['exception'] = self.formatException(record.exc_info)

            return json.dumps(log_data)

    def log(self, level: str, message: str, **kwargs):
        """Log with structured data"""
        extra = {'extra': kwargs}

        if level == 'debug':
            self.logger.debug(message, extra=extra)
        elif level == 'info':
            self.logger.info(message, extra=extra)
        elif level == 'warning':
            self.logger.warning(message, extra=extra)
        elif level == 'error':
            self.logger.error(message, extra=extra)
        elif level == 'critical':
            self.logger.critical(message, extra=extra)

# Usage
logger = StructuredLogger('llama-server')

logger.log('info', 'Request received',
    request_id='abc123',
    user_id='user456',
    endpoint='/v1/completions',
    prompt_length=50
)

logger.log('error', 'Model inference failed',
    request_id='abc123',
    model='llama-2-7b',
    error_type='OOM',
    gpu_memory_gb=24
)
```

### 3. Distributed Tracing

```python
# monitoring/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Initialize tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name='localhost',
    agent_port=6831,
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Custom tracing
async def process_request(prompt: str, max_tokens: int):
    with tracer.start_as_current_span("process_request") as span:
        span.set_attribute("prompt.length", len(prompt))
        span.set_attribute("max_tokens", max_tokens)

        # Tokenization span
        with tracer.start_as_current_span("tokenize"):
            tokens = tokenize(prompt)
            span.set_attribute("tokens.count", len(tokens))

        # Inference span
        with tracer.start_as_current_span("inference") as inference_span:
            inference_span.set_attribute("model", "llama-2-7b")

            try:
                result = await model.generate(tokens, max_tokens)
                inference_span.set_attribute("tokens.generated", len(result))
            except Exception as e:
                inference_span.record_exception(e)
                inference_span.set_status(trace.Status(trace.StatusCode.ERROR))
                raise

        # Detokenization span
        with tracer.start_as_current_span("detokenize"):
            text = detokenize(result)

        return text
```

## Alerting

### Alert Rules (Prometheus)

```yaml
# prometheus/alerts.yml
groups:
  - name: llama_inference
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High request latency"
          description: "P95 latency is {{ $value }}s (threshold: 10s)"

      # GPU memory pressure
      - alert: GPUMemoryPressure
        expr: |
          (gpu_memory_used_bytes / gpu_memory_total_bytes) > 0.9
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage high"
          description: "GPU {{ $labels.gpu_id }} memory is {{ $value | humanizePercentage }} full"

      # Service down
      - alert: ServiceDown
        expr: up{job="llama-server"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "llama-server instance {{ $labels.instance }} is down"

      # Queue depth growing
      - alert: QueueDepthGrowing
        expr: |
          deriv(request_queue_depth[5m]) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Request queue is growing"
          description: "Queue depth increasing at {{ $value }} requests/second"

      # Model inference slow
      - alert: SlowInference
        expr: |
          histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Model inference is slow"
          description: "P95 inference time is {{ $value }}s"
```

### Alert Manager Configuration

```yaml
# alertmanager/alertmanager.yml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'

route:
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default'

  routes:
    # Critical alerts go to PagerDuty
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true

    # All alerts go to Slack
    - receiver: 'slack'

receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://alertmanager-webhook:5001/'

  - name: 'slack'
    slack_configs:
      - channel: '#llama-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
        send_resolved: true

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: '{{ .GroupLabels.alertname }}: {{ .Annotations.summary }}'
```

## Incident Response Procedures

### 1. Detection

```python
# monitoring/anomaly_detection.py
import numpy as np
from typing import List
from scipy import stats

class AnomalyDetector:
    def __init__(self, window_size: int = 100, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold  # Standard deviations
        self.history: List[float] = []

    def add_sample(self, value: float) -> bool:
        """Add sample and detect anomaly"""
        self.history.append(value)

        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) < 10:  # Need enough samples
            return False

        # Calculate z-score
        mean = np.mean(self.history)
        std = np.std(self.history)

        if std == 0:
            return False

        z_score = abs((value - mean) / std)

        return z_score > self.threshold

# Usage
latency_detector = AnomalyDetector(window_size=100, threshold=3.0)

async def monitor_latency():
    while True:
        latency = await get_current_latency()

        if latency_detector.add_sample(latency):
            alert(f"Anomalous latency detected: {latency}s")

        await asyncio.sleep(10)
```

### 2. Triage

```python
# incident/triage.py
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class Severity(Enum):
    SEV1 = "Critical - Service down"
    SEV2 = "High - Major functionality impaired"
    SEV3 = "Medium - Degraded performance"
    SEV4 = "Low - Minor issue"

@dataclass
class Incident:
    id: str
    severity: Severity
    title: str
    description: str
    started_at: datetime
    detected_by: str
    assigned_to: str = None
    resolved_at: datetime = None

    def time_to_detect(self) -> float:
        """Minutes from start to detection"""
        return (self.detected_at - self.started_at).total_seconds() / 60

    def time_to_resolve(self) -> float:
        """Minutes from detection to resolution"""
        if self.resolved_at:
            return (self.resolved_at - self.detected_at).total_seconds() / 60
        return None

class IncidentManager:
    def __init__(self):
        self.incidents: Dict[str, Incident] = {}

    def create_incident(self, severity: Severity, title: str, description: str) -> Incident:
        """Create new incident"""
        incident = Incident(
            id=generate_id(),
            severity=severity,
            title=title,
            description=description,
            started_at=datetime.utcnow(),
            detected_by="monitoring"
        )

        self.incidents[incident.id] = incident

        # Notify on-call
        self.notify_oncall(incident)

        return incident

    def notify_oncall(self, incident: Incident):
        """Notify on-call engineer"""
        if incident.severity in [Severity.SEV1, Severity.SEV2]:
            pagerduty.trigger_incident(
                title=incident.title,
                description=incident.description,
                severity=incident.severity.value
            )

        slack.send_message(
            channel='#incidents',
            message=f"🚨 {incident.severity.value}: {incident.title}"
        )
```

### 3. Debugging

```python
# debugging/diagnostic_tools.py
import psutil
import GPUtil
from typing import Dict, Any

class DiagnosticTool:
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system diagnostic info"""
        return {
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
                'swap': psutil.swap_memory()._asdict()
            },
            'disk': {
                'usage': psutil.disk_usage('/')._asdict(),
                'io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None
            },
            'network': {
                'io': psutil.net_io_counters()._asdict(),
                'connections': len(psutil.net_connections())
            }
        }

    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """Get GPU diagnostic info"""
        gpus = GPUtil.getGPUs()
        return [
            {
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                'temperature': gpu.temperature
            }
            for gpu in gpus
        ]

    @staticmethod
    def get_process_info() -> Dict[str, Any]:
        """Get process diagnostic info"""
        process = psutil.Process()

        return {
            'pid': process.pid,
            'cpu_percent': process.cpu_percent(),
            'memory_info': process.memory_info()._asdict(),
            'num_threads': process.num_threads(),
            'open_files': len(process.open_files()),
            'connections': len(process.connections())
        }

# Diagnostic endpoint
@app.get("/debug/diagnostics")
async def diagnostics():
    return {
        'system': DiagnosticTool.get_system_info(),
        'gpus': DiagnosticTool.get_gpu_info(),
        'process': DiagnosticTool.get_process_info()
    }
```

### 4. Mitigation

```python
# incident/mitigation.py
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open

    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker"""
        if self.state == 'open':
            # Check if should try again
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)

            # Success - reset or close
            if self.state == 'half-open':
                self.state = 'closed'
                self.failures = 0

            return result

        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold:
                self.state = 'open'

            raise

# Graceful degradation
class GracefulDegradation:
    def __init__(self):
        self.fallback_model = None

    async def generate(self, prompt: str, **kwargs):
        """Generate with fallback"""
        try:
            # Try primary model
            return await primary_model.generate(prompt, **kwargs)

        except Exception as e:
            logger.error(f"Primary model failed: {e}")

            # Try fallback model
            if self.fallback_model:
                logger.info("Using fallback model")
                return await self.fallback_model.generate(prompt, **kwargs)

            # Return cached/default response
            return get_cached_response(prompt) or "I apologize, but I'm experiencing technical difficulties."
```

## Root Cause Analysis (RCA)

### Post-Incident Review Template

```markdown
# Incident Post-Mortem: [Incident Title]

## Summary
- **Date**: 2025-11-18
- **Duration**: 45 minutes
- **Severity**: SEV2
- **Impact**: 15% of requests failed, 200 users affected
- **Root Cause**: GPU memory leak in batch processing code

## Timeline
- **14:30 UTC**: Alert triggered - high error rate
- **14:32 UTC**: On-call engineer paged
- **14:35 UTC**: Investigation began
- **14:45 UTC**: Identified GPU memory leak
- **14:50 UTC**: Rolled back to previous version
- **14:55 UTC**: Service recovered
- **15:00 UTC**: Monitoring confirmed stability
- **15:15 UTC**: Incident closed

## Root Cause
GPU memory was not being freed after batch processing completed. Over time,
available GPU memory decreased until OOM errors occurred.

## Detection
- Prometheus alert: HighErrorRate triggered at 14:30 UTC
- Time to detect: ~5 minutes (leak started at 14:25 UTC)

## Impact
- 450 failed requests over 25 minutes
- ~200 users affected
- No data loss
- Automated rollback prevented extended outage

## Resolution
1. Immediate: Rolled back to v1.2.3 (previous stable version)
2. Root cause: Fixed memory leak in v1.2.4-beta
3. Prevention: Added memory leak detection tests

## Action Items
1. [ ] Add GPU memory usage monitoring (Owner: @alice, Due: 2025-11-20)
2. [ ] Implement automated memory leak detection (Owner: @bob, Due: 2025-11-22)
3. [ ] Add canary deployment for better testing (Owner: @charlie, Due: 2025-11-25)
4. [ ] Update runbook with GPU OOM procedures (Owner: @diana, Due: 2025-11-21)
5. [ ] Review all batch processing code for similar issues (Owner: @eve, Due: 2025-11-27)

## Lessons Learned
### What went well
- Alert fired within 5 minutes
- On-call responded quickly
- Automated rollback worked correctly
- Communication was clear

### What can be improved
- Need better memory profiling in CI/CD
- Should have caught this in canary deployment
- Need GPU-specific health checks

## Prevention
- Memory leak tests added to CI/CD
- Enhanced monitoring for GPU resources
- Canary deployment process improved
```

## Chaos Engineering

```python
# testing/chaos.py
import random
import asyncio

class ChaosMonkey:
    def __init__(self, failure_rate: float = 0.01):
        self.failure_rate = failure_rate
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    async def inject_failure(self, request):
        """Randomly inject failures"""
        if not self.enabled:
            return

        if random.random() < self.failure_rate:
            failure_type = random.choice([
                'timeout',
                'error',
                'slow'
            ])

            if failure_type == 'timeout':
                raise TimeoutError("Chaos: Simulated timeout")
            elif failure_type == 'error':
                raise Exception("Chaos: Simulated error")
            elif failure_type == 'slow':
                await asyncio.sleep(random.uniform(5, 10))

# Usage
chaos = ChaosMonkey(failure_rate=0.05)  # 5% failure rate

@app.middleware("http")
async def chaos_middleware(request, call_next):
    await chaos.inject_failure(request)
    return await call_next(request)
```

## Runbooks

### Example: High Memory Usage Runbook

```markdown
# Runbook: High GPU Memory Usage

## Symptoms
- Alert: GPUMemoryPressure fired
- GPU memory usage > 90%
- Increased OOM errors
- Request failures

## Investigation
1. Check current GPU memory usage:
   ```bash
   nvidia-smi
   ```

2. Check active requests:
   ```bash
   curl localhost:8080/debug/active-requests
   ```

3. Review recent logs:
   ```bash
   kubectl logs -n production deployment/llama-inference --tail=100
   ```

4. Check metrics dashboard:
   - Navigate to Grafana
   - Open "GPU Metrics" dashboard
   - Review memory usage trends

## Resolution

### Option 1: Restart Service (Quick fix)
```bash
kubectl rollout restart deployment/llama-inference -n production
```

### Option 2: Scale Up (More capacity)
```bash
kubectl scale deployment/llama-inference --replicas=6 -n production
```

### Option 3: Reduce Load (Emergency)
- Enable rate limiting
- Reject long prompts temporarily
- Reduce max_tokens limit

## Prevention
- Monitor memory trends
- Set up auto-scaling based on GPU memory
- Implement request size limits
- Add memory leak detection

## Escalation
- If issue persists after restart: Page @ml-infra-team
- If multiple nodes affected: Page @sre-lead
- If customer-facing impact: Notify @support-team
```

## Summary

Key components of incident response:
- **Observability**: Metrics, logs, traces
- **Alerting**: Early detection and notification
- **Procedures**: Clear response workflows
- **Debugging**: Diagnostic tools and techniques
- **Learning**: Post-incident reviews and improvement

---

**Authors**: Agent 5 (Documentation Specialist)
**Last Updated**: 2025-11-18
**Estimated Reading Time**: 40 minutes
