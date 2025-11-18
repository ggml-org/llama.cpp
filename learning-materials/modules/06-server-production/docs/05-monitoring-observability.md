# Monitoring & Observability for LLM Services

**Learning Module**: Module 6 - Server & Production
**Estimated Reading Time**: 30 minutes
**Prerequisites**: Prometheus, Grafana basics, Module 6.1-6.4
**Related Content**:
- [LLaMA Server Architecture](./01-llama-server-architecture.md)
- [Load Balancing & Scaling](./04-load-balancing-scaling.md)
- [Production Best Practices](./06-production-best-practices.md)

---

## Overview

Observability is critical for production LLM services. The three pillars are:

1. **Metrics**: Quantitative measurements (request rate, latency, errors)
2. **Logs**: Discrete events (requests, errors, debug info)
3. **Traces**: Request flow through distributed system

---

## Metrics Collection

### 1. LLaMA Server Native Metrics

**Enabling Metrics**:
```bash
llama-server -m model.gguf --metrics
```

**Prometheus Endpoint**: `http://localhost:8080/metrics`

**Key Metrics**:
```prometheus
# Request metrics
llama_requests_total{endpoint="/v1/chat/completions"} 1234
llama_request_duration_seconds_bucket{le="0.5"} 120
llama_request_errors_total{error_type="timeout"} 5

# Token metrics
llama_tokens_generated_total 567890
llama_prompt_tokens_total 234567
llama_tokens_per_second 45.2

# Resource metrics
llama_slots_processing 3
llama_slots_available 5
llama_kv_cache_usage_ratio 0.75

# Model metrics
llama_model_load_duration_seconds 12.5
llama_context_size 4096
```

### 2. Custom Application Metrics

**Python Implementation (Prometheus Client)**:
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Counters
requests_total = Counter(
    'llm_requests_total',
    'Total requests',
    ['model', 'endpoint', 'status']
)

tokens_generated = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated',
    ['model']
)

# Histograms
request_duration = Histogram(
    'llm_request_duration_seconds',
    'Request duration',
    ['model', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

token_latency = Histogram(
    'llm_token_latency_seconds',
    'Time to first token',
    ['model'],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

# Gauges
active_requests = Gauge(
    'llm_active_requests',
    'Currently active requests',
    ['model']
)

queue_depth = Gauge(
    'llm_queue_depth',
    'Requests waiting in queue'
)

# Usage
@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    model = request.model
    start_time = time.time()

    # Track active requests
    active_requests.labels(model=model).inc()

    try:
        # Make request
        response = await process_request(request)

        # Record success
        requests_total.labels(
            model=model,
            endpoint='/v1/chat/completions',
            status='success'
        ).inc()

        # Record tokens
        usage = response.get('usage', {})
        tokens_generated.labels(model=model).inc(usage.get('completion_tokens', 0))

        return response

    except Exception as e:
        # Record error
        requests_total.labels(
            model=model,
            endpoint='/v1/chat/completions',
            status='error'
        ).inc()
        raise

    finally:
        # Record duration
        duration = time.time() - start_time
        request_duration.labels(
            model=model,
            endpoint='/v1/chat/completions'
        ).observe(duration)

        # Decrement active
        active_requests.labels(model=model).dec()

# Start metrics server
start_http_server(9090)
```

### 3. System Metrics

**Node Exporter** (for host metrics):
```bash
# Install node_exporter
wget https://github.com/prometheus/node_exporter/releases/download/v1.7.0/node_exporter-1.7.0.linux-amd64.tar.gz
tar xvfz node_exporter-*.tar.gz
cd node_exporter-*
./node_exporter &

# Metrics available at :9100/metrics
# - CPU usage
# - Memory usage
# - Disk I/O
# - Network I/O
```

**NVIDIA GPU Metrics** (DCGM Exporter):
```bash
# Run DCGM exporter
docker run -d \
  --gpus all \
  --rm \
  -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04

# Metrics available at :9400/metrics
# - GPU utilization
# - GPU memory usage
# - GPU temperature
# - Power consumption
```

### 4. Prometheus Configuration

**prometheus.yml**:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'production'
    region: 'us-east-1'

scrape_configs:
  # LLaMA server metrics
  - job_name: 'llama-server'
    static_configs:
      - targets:
        - 'server1:8080'
        - 'server2:8080'
        - 'server3:8080'
    metrics_path: '/metrics'
    scrape_interval: 10s

  # Application metrics
  - job_name: 'llm-api'
    static_configs:
      - targets: ['api-server:9090']

  # System metrics
  - job_name: 'node'
    static_configs:
      - targets:
        - 'server1:9100'
        - 'server2:9100'
        - 'server3:9100'

  # GPU metrics
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets:
        - 'gpu-server1:9400'
        - 'gpu-server2:9400'

  # Kubernetes metrics
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__

# Alerting rules
rule_files:
  - 'alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

---

## Logging

### 1. Structured Logging

**JSON Logging**:
```python
import logging
import json
import sys
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'model'):
            log_data['model'] = record.model
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms

        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)

# Setup logger
logger = logging.getLogger('llm-api')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info(
    'Request completed',
    extra={
        'request_id': 'req_123',
        'user_id': 'user_456',
        'model': 'llama-2-7b',
        'duration_ms': 1234,
        'tokens': 156
    }
)

# Output:
# {
#   "timestamp": "2024-01-15T10:30:45.123456",
#   "level": "INFO",
#   "message": "Request completed",
#   "module": "api",
#   "function": "chat_completion",
#   "line": 123,
#   "request_id": "req_123",
#   "user_id": "user_456",
#   "model": "llama-2-7b",
#   "duration_ms": 1234,
#   "tokens": 156
# }
```

### 2. Log Aggregation

**Loki Configuration** (Grafana Loki):

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yaml:/etc/loki/local-config.yaml
      - loki-data:/loki
    command: -config.file=/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - ./promtail-config.yaml:/etc/promtail/config.yaml
    command: -config.file=/etc/promtail/config.yaml

volumes:
  loki-data:
```

**promtail-config.yaml**:
```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  # Docker logs
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        target_label: 'container'
      - source_labels: ['__meta_docker_container_log_stream']
        target_label: 'stream'

  # Application logs
  - job_name: llm-api
    static_configs:
      - targets:
          - localhost
        labels:
          job: llm-api
          __path__: /var/log/llm-api/*.log
```

### 3. Log Queries (LogQL)

**Example Queries**:
```logql
# All errors in last hour
{job="llm-api"} |= "ERROR" | json

# Slow requests (>5 seconds)
{job="llm-api"} | json | duration_ms > 5000

# Requests by user
{job="llm-api"} | json | user_id="user_123"

# Token usage by model
sum by (model) (
  rate({job="llm-api"} | json | unwrap tokens [5m])
)

# Error rate
sum(rate({job="llm-api"} |= "ERROR" [5m])) /
sum(rate({job="llm-api"} [5m]))
```

---

## Distributed Tracing

### 1. OpenTelemetry Setup

**Python Implementation**:
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# Setup tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Export to Jaeger
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Auto-instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Auto-instrument HTTP client
HTTPXClientInstrumentor().instrument()

# Manual tracing
@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    with tracer.start_as_current_span("chat_completion") as span:
        # Add attributes
        span.set_attribute("model", request.model)
        span.set_attribute("max_tokens", request.max_tokens)
        span.set_attribute("user_id", get_user_id())

        # Tokenization span
        with tracer.start_as_current_span("tokenize"):
            tokens = tokenize(request.messages)
            span.set_attribute("prompt_tokens", len(tokens))

        # Inference span
        with tracer.start_as_current_span("inference"):
            response = await call_llama_server(request)
            span.set_attribute("completion_tokens", response["usage"]["completion_tokens"])

        # Post-processing span
        with tracer.start_as_current_span("post_process"):
            result = process_response(response)

        return result
```

### 2. Jaeger Deployment

**docker-compose.yml**:
```yaml
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "16686:16686"  # UI
      - "14268:14268"
      - "14250:14250"
      - "9411:9411"
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
```

---

## Dashboards

### 1. Grafana Dashboard Configuration

**LLM Service Dashboard**:
```json
{
  "dashboard": {
    "title": "LLM Inference Service",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(llama_requests_total[5m])"
        }],
        "type": "graph"
      },
      {
        "title": "Request Duration (p50, p95, p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(llm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(llm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p99"
          }
        ]
      },
      {
        "title": "Tokens per Second",
        "targets": [{
          "expr": "rate(llama_tokens_generated_total[5m])"
        }]
      },
      {
        "title": "Active Slots",
        "targets": [
          {
            "expr": "llama_slots_processing",
            "legendFormat": "Processing"
          },
          {
            "expr": "llama_slots_available",
            "legendFormat": "Available"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(llama_request_errors_total[5m])"
        }]
      },
      {
        "title": "GPU Utilization",
        "targets": [{
          "expr": "DCGM_FI_DEV_GPU_UTIL"
        }]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [{
          "expr": "DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE * 100"
        }]
      }
    ]
  }
}
```

### 2. Key Performance Indicators (KPIs)

**Service Health Scorecard**:
```python
from dataclasses import dataclass
from typing import List

@dataclass
class SLO:
    name: str
    target: float
    actual: float

    @property
    def is_met(self) -> bool:
        return self.actual >= self.target

    @property
    def error_budget_remaining(self) -> float:
        return (self.actual - self.target) / (100 - self.target)

class ServiceHealthMonitor:
    def __init__(self, prometheus_client):
        self.prom = prometheus_client

    async def calculate_slos(self) -> List[SLO]:
        """Calculate Service Level Objectives"""

        # Availability SLO (99.9% uptime)
        availability_query = """
            sum(rate(llama_requests_total{status="success"}[7d])) /
            sum(rate(llama_requests_total[7d])) * 100
        """
        availability = await self.prom.query(availability_query)

        # Latency SLO (95% of requests < 2s)
        latency_query = """
            histogram_quantile(0.95,
                rate(llm_request_duration_seconds_bucket[7d])
            )
        """
        p95_latency = await self.prom.query(latency_query)
        latency_slo_met = p95_latency < 2.0

        # Throughput SLO (> 100 req/s)
        throughput_query = "rate(llama_requests_total[5m])"
        throughput = await self.prom.query(throughput_query)

        return [
            SLO("Availability", 99.9, availability),
            SLO("Latency P95 < 2s", 95.0, 100 if latency_slo_met else 90),
            SLO("Throughput > 100 req/s", 100, min(throughput, 100))
        ]
```

---

## Alerting

### 1. Alert Rules

**alerts.yml**:
```yaml
groups:
  - name: llm_service_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(llama_request_errors_total[5m])) /
            sum(rate(llama_requests_total[5m]))
          ) > 0.05
        for: 5m
        labels:
          severity: critical
          service: llm-inference
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            rate(llm_request_duration_seconds_bucket[5m])
          ) > 5
        for: 10m
        labels:
          severity: warning
          service: llm-inference
        annotations:
          summary: "High request latency"
          description: "P95 latency is {{ $value }}s (threshold: 5s)"

      # No available slots
      - alert: NoAvailableSlots
        expr: llama_slots_available == 0
        for: 5m
        labels:
          severity: warning
          service: llm-inference
        annotations:
          summary: "All inference slots occupied"
          description: "No available slots for {{ $labels.instance }}"

      # GPU out of memory
      - alert: GPUMemoryHigh
        expr: |
          (
            DCGM_FI_DEV_FB_USED /
            DCGM_FI_DEV_FB_FREE
          ) > 0.95
        for: 5m
        labels:
          severity: critical
          service: llm-inference
        annotations:
          summary: "GPU memory usage critical"
          description: "GPU {{ $labels.gpu }} memory at {{ $value | humanizePercentage }}"

      # Service down
      - alert: ServiceDown
        expr: up{job="llama-server"} == 0
        for: 1m
        labels:
          severity: critical
          service: llm-inference
        annotations:
          summary: "Service is down"
          description: "{{ $labels.instance }} has been down for 1 minute"

      # High queue depth
      - alert: HighQueueDepth
        expr: llm_queue_depth > 100
        for: 10m
        labels:
          severity: warning
          service: llm-inference
        annotations:
          summary: "High request queue depth"
          description: "Queue depth is {{ $value }} requests"

      # SLO burn rate
      - alert: SLOBurnRateHigh
        expr: |
          (
            1 - (
              sum(rate(llama_requests_total{status="success"}[1h])) /
              sum(rate(llama_requests_total[1h]))
            )
          ) > 0.001 * 14.4  # 1.44% error budget burn in 1 hour
        labels:
          severity: critical
          service: llm-inference
        annotations:
          summary: "High SLO error budget burn rate"
          description: "Error budget burning too fast, will exhaust in {{ $value }} hours"
```

### 2. Alert Manager Configuration

**alertmanager.yml**:
```yaml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'

route:
  receiver: 'default'
  group_by: ['alertname', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h

  routes:
    # Critical alerts to PagerDuty
    - match:
        severity: critical
      receiver: pagerduty
      continue: true

    # Warnings to Slack
    - match:
        severity: warning
      receiver: slack

    # Default to email
    - receiver: email

receivers:
  - name: 'default'
    email_configs:
      - to: 'team@example.com'
        from: 'alerts@example.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'alerts@example.com'
        auth_password: 'password'

  - name: 'slack'
    slack_configs:
      - channel: '#llm-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
        description: '{{ .GroupLabels.alertname }}'
```

---

## Cost Monitoring

### 1. Resource Cost Tracking

**Cost Calculation**:
```python
import asyncio
from datetime import datetime, timedelta

class CostTracker:
    def __init__(self, db):
        self.db = db

        # Pricing per hour
        self.instance_costs = {
            "g5.xlarge": 1.006,  # AWS GPU instance
            "g5.2xlarge": 1.212,
            "p3.2xlarge": 3.06
        }

        # Token processing costs
        self.token_costs = {
            "llama-2-7b": 0.0001,  # per 1K tokens
            "llama-2-13b": 0.0002,
            "llama-2-70b": 0.001
        }

    async def calculate_hourly_cost(self) -> dict:
        """Calculate infrastructure cost"""
        # Get active instances
        instances = await self.get_active_instances()

        infrastructure_cost = sum(
            self.instance_costs.get(inst.type, 0)
            for inst in instances
        )

        return {
            "infrastructure_cost_per_hour": infrastructure_cost,
            "infrastructure_cost_per_month": infrastructure_cost * 730,
            "active_instances": len(instances)
        }

    async def calculate_usage_cost(self, model: str, tokens: int) -> float:
        """Calculate token processing cost"""
        cost_per_1k = self.token_costs.get(model, 0.0001)
        return (tokens / 1000) * cost_per_1k

    async def get_daily_report(self) -> dict:
        """Generate daily cost report"""
        today = datetime.now().date()

        # Get token usage
        usage_query = """
            SELECT model, SUM(total_tokens) as tokens
            FROM usage_logs
            WHERE DATE(created_at) = $1
            GROUP BY model
        """
        usage = await self.db.fetch(usage_query, today)

        total_usage_cost = sum(
            await self.calculate_usage_cost(row['model'], row['tokens'])
            for row in usage
        )

        infra_cost = await self.calculate_hourly_cost()
        daily_infra_cost = infra_cost['infrastructure_cost_per_hour'] * 24

        return {
            "date": str(today),
            "infrastructure_cost": daily_infra_cost,
            "usage_cost": total_usage_cost,
            "total_cost": daily_infra_cost + total_usage_cost,
            "usage_by_model": usage
        }
```

---

## Summary

**Observability Stack**:
1. **Metrics**: Prometheus + Grafana
2. **Logs**: Loki + Promtail
3. **Traces**: Jaeger + OpenTelemetry
4. **Alerts**: AlertManager

**Key Metrics to Monitor**:
- Request rate, latency, errors (RED method)
- Slot utilization
- Token generation rate
- GPU utilization and memory
- Cost per request

**Best Practices**:
- ✅ Use structured logging (JSON)
- ✅ Implement distributed tracing
- ✅ Set up comprehensive dashboards
- ✅ Define and monitor SLOs
- ✅ Alert on SLO violations, not symptoms
- ✅ Track costs in real-time

**Next Steps**:
- [Production Best Practices](./06-production-best-practices.md)
- Lab 6.5: Set up monitoring stack

---

**Interview Topics**:
- Three pillars of observability
- SLO vs SLA vs SLI
- Alert design principles
- Metrics aggregation strategies
- Cost optimization techniques
