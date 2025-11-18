# Lab 6.5: Complete Monitoring Setup

**Difficulty**: Advanced
**Estimated Time**: 90 minutes
**Prerequisites**: Lab 6.1-6.4, Prometheus/Grafana basics

---

## Objectives

1. Set up complete observability stack
2. Create custom Grafana dashboards
3. Configure alerting rules
4. Implement log aggregation
5. Set up distributed tracing

---

## Part 1: Prometheus & Grafana Setup (20 min)

**docker-compose.monitoring.yml**:
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alerts.yml:/etc/prometheus/alerts.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  alertmanager:
    image: prom/alertmanager:latest
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    ports:
      - "9093:9093"

volumes:
  prometheus-data:
  grafana-data:
```

**Start monitoring stack**:
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

---

## Part 2: Create Grafana Dashboard (25 min)

**grafana/datasources/prometheus.yml**:
```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

**Create dashboard via UI**:
1. Visit http://localhost:3000 (admin/admin)
2. Create new dashboard
3. Add panels:

**Panel 1 - Request Rate**:
```promql
rate(llama_requests_total[5m])
```

**Panel 2 - Latency Percentiles**:
```promql
histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate(llm_request_duration_seconds_bucket[5m]))
```

**Panel 3 - Error Rate**:
```promql
rate(llama_request_errors_total[5m]) / rate(llama_requests_total[5m])
```

**Panel 4 - Slot Utilization**:
```promql
llama_slots_processing / (llama_slots_processing + llama_slots_available)
```

**Panel 5 - Tokens per Second**:
```promql
rate(llama_tokens_generated_total[5m])
```

**Panel 6 - GPU Metrics** (if available):
```promql
DCGM_FI_DEV_GPU_UTIL
DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE) * 100
```

**Export dashboard**:
```bash
# Export dashboard JSON
curl http://admin:admin@localhost:3000/api/dashboards/uid/llm-dashboard \
  | jq > grafana/dashboards/llm-dashboard.json
```

---

## Part 3: Configure Alerting (20 min)

**alerts.yml**:
```yaml
groups:
  - name: llm_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          rate(llama_request_errors_total[5m]) /
          rate(llama_requests_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate: {{ $value }}"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            rate(llm_request_duration_seconds_bucket[5m])) > 5
        for: 10m
        annotations:
          summary: "P95 latency: {{ $value }}s"

      - alert: NoAvailableSlots
        expr: llama_slots_available == 0
        for: 5m
        annotations:
          summary: "All slots occupied"

      - alert: ServiceDown
        expr: up{job="llama-server"} == 0
        for: 1m
        annotations:
          summary: "Service is down"
```

**alertmanager.yml**:
```yaml
global:
  slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'

route:
  receiver: 'default'
  group_by: ['alertname']
  group_wait: 10s
  repeat_interval: 12h

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

**Test alerts**:
```bash
# Trigger high error rate
# Stop llama-server to trigger ServiceDown alert

# View alerts in Prometheus
# http://localhost:9090/alerts

# View in Alertmanager
# http://localhost:9093
```

---

## Part 4: Log Aggregation with Loki (25 min)

**Add to docker-compose**:
```yaml
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yaml:/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - ./promtail-config.yaml:/etc/promtail/config.yaml
```

**loki-config.yaml**:
```yaml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
  filesystem:
    directory: /loki/chunks
```

**promtail-config.yaml**:
```yaml
server:
  http_listen_port: 9080

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        target_label: 'container'
```

**Add Loki datasource in Grafana**:
```yaml
# grafana/datasources/loki.yml
apiVersion: 1
datasources:
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
```

**Query logs in Grafana**:
```logql
{container="llama-server"} |= "error"
{container="llama-server"} | json | duration_ms > 1000
```

---

## Deliverables

1. **Screenshots**:
   - Grafana dashboard showing all panels
   - Prometheus alerts page
   - Loki log viewer
2. **Configuration Files**:
   - All YAML configs
   - Dashboard JSON export
3. **Report**:
   - Summary of key metrics observed
   - Any alerts triggered
   - Log analysis findings

---

## Challenge Exercises

1. **Advanced Dashboard**: Add business metrics (cost per request, users, etc.)
2. **Custom Alerts**: Create SLO-based alerts
3. **Distributed Tracing**: Add Jaeger for request tracing
4. **Anomaly Detection**: Implement basic anomaly detection

---

## Key Takeaways

- Complete observability requires metrics, logs, and traces
- Grafana dashboards provide real-time visibility
- Alerting prevents issues from becoming incidents
- Log aggregation enables debugging
- Monitoring is essential for production services

---

**Module 6 Complete!**
Next: Review interview questions and complete capstone project
