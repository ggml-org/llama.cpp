# Multi-model llama-server Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a companion Grafana dashboard `deploy/grafana/llama-server-models.json` that shows all llama-server models at once (ladder-activity timeline + per-model KV/latency/throughput grouped `by (model)`), with no `$model` selector.

**Architecture:** New dashboard JSON sibling to `llama-server-rainbow.json`. Drops the `$model` template variable; every llama panel groups `by (model)` and colors with `palette-classic`. Host/GPU panels stay host-level (single lines), copied verbatim from rainbow. A README paragraph documents when to use which dashboard.

**Tech Stack:** Grafana dashboard JSON (schemaVersion 39), PromQL against Mimir (Prometheus API).

## Global Constraints

- Datasource pattern: `{ "type": "prometheus", "uid": "${datasource}" }` on every panel and template query — copied verbatim from rainbow.
- Metric names are exact and verified against live Mimir: `llamacpp:kv_cache_k_bytes`, `llamacpp:kv_cache_v_bytes`, `llamacpp:prompt_tokens_size_bucket`, `llamacpp:context_used_tokens_bucket`, `llamacpp:time_to_first_token_seconds_bucket`, `llamacpp:generation_latency_seconds_bucket`, `llamacpp:predicted_tokens_seconds`, `llamacpp:prompt_tokens_seconds`, `llamacpp:draft_tokens_accepted_total`, `llamacpp:draft_tokens_total`, `llamacpp:n_ctx_shift_total`, `nvidia_smi_memory_used_bytes`, `nvidia_smi_memory_total_bytes`, `nvidia_smi_utilization_gpu_ratio`, `nvidia_smi_temperature_gpu`, `nvidia_smi_power_draw_watts`, `system_cpu_load_average_1m`, `system_memory_usage_bytes`.
- Series labels available: `host`, `model`. Every llama panel filters `{host="$host"}` and groups `by (model)`.
- No `$model` variable. No changes to `llama-server-rainbow.json`, the collector config, or metric names.
- Every multi-series panel carries a legend (`{{model}}`) so identity is never color-alone.

---

### Task 1: Create the multi-model dashboard JSON

**Files:**
- Create: `deploy/grafana/llama-server-models.json`

**Interfaces:**
- Consumes: nothing (standalone JSON artifact).
- Produces: an importable Grafana dashboard with uid `llama-models`.

- [ ] **Step 1: Write the dashboard file**

Create `deploy/grafana/llama-server-models.json` with exactly this content:

```json
{
  "__inputs": [],
  "__requires": [],
  "title": "llama-server — models (rainbow)",
  "uid": "llama-models",
  "tags": ["llama.cpp", "llm", "gpu"],
  "schemaVersion": 39,
  "version": 1,
  "editable": true,
  "time": { "from": "now-6h", "to": "now" },
  "refresh": "30s",
  "timezone": "",
  "templating": {
    "list": [
      {
        "name": "datasource",
        "label": "Data source",
        "type": "datasource",
        "query": "prometheus",
        "current": {},
        "hide": 0
      },
      {
        "name": "host",
        "label": "Host",
        "type": "query",
        "datasource": { "type": "prometheus", "uid": "${datasource}" },
        "query": "label_values(up{job=\"llama-server\"}, host)",
        "refresh": 2,
        "current": {},
        "includeAll": false,
        "sort": 1
      }
    ]
  },
  "panels": [
    {
      "id": 1,
      "type": "row",
      "title": "Ladder activity",
      "gridPos": { "h": 1, "w": 24, "x": 0, "y": 0 },
      "collapsed": false
    },
    {
      "id": 2,
      "type": "state-timeline",
      "title": "Model loaded (which model was emitting metrics)",
      "description": "One band per model; a band is present only while that model was the loaded child emitting metrics. Gaps mean the model was not loaded. Reads ladder switches across the window at a glance.",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 7, "w": 24, "x": 0, "y": 1 },
      "fieldConfig": {
        "defaults": {
          "color": { "mode": "palette-classic" },
          "custom": { "fillOpacity": 80, "lineWidth": 0 },
          "mappings": [
            { "type": "value", "options": { "1": { "text": "loaded", "index": 0 } } }
          ]
        },
        "overrides": []
      },
      "options": {
        "mergeValues": true,
        "showValue": "never",
        "rowHeight": 0.9,
        "legend": { "displayMode": "list", "placement": "bottom", "showLegend": true },
        "tooltip": { "mode": "single" }
      },
      "targets": [
        {
          "refId": "A",
          "expr": "sum by (model) (present_over_time(llamacpp:kv_cache_k_bytes{host=\"$host\"}[5m]))",
          "legendFormat": "{{model}}"
        }
      ]
    },
    {
      "id": 10,
      "type": "row",
      "title": "Memory & KV cache",
      "gridPos": { "h": 1, "w": 24, "x": 0, "y": 8 },
      "collapsed": false
    },
    {
      "id": 11,
      "type": "timeseries",
      "title": "KV total (K+V) per model",
      "description": "K+V cache footprint, one line per model. Only the loaded model updates; others go flat then stop.",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 8, "w": 16, "x": 0, "y": 9 },
      "fieldConfig": {
        "defaults": {
          "unit": "bytes",
          "color": { "mode": "palette-classic" },
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 8, "showPoints": "never" }
        },
        "overrides": []
      },
      "options": { "legend": { "displayMode": "list", "placement": "bottom", "showLegend": true }, "tooltip": { "mode": "multi", "sort": "desc" } },
      "targets": [
        { "refId": "A", "expr": "sum by (model) (llamacpp:kv_cache_k_bytes{host=\"$host\"} + llamacpp:kv_cache_v_bytes{host=\"$host\"})", "legendFormat": "{{model}}" }
      ]
    },
    {
      "id": 12,
      "type": "timeseries",
      "title": "VRAM used vs free (host)",
      "description": "Host-level shared GPU memory — not per-model. VRAM is shared across whichever model is loaded.",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 8, "w": 8, "x": 16, "y": 9 },
      "fieldConfig": {
        "defaults": {
          "unit": "bytes",
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 8, "showPoints": "never" }
        },
        "overrides": [
          { "matcher": { "id": "byName", "options": "Used" }, "properties": [ { "id": "color", "value": { "mode": "fixed", "fixedColor": "#eb6834" } } ] },
          { "matcher": { "id": "byName", "options": "Free" }, "properties": [ { "id": "color", "value": { "mode": "fixed", "fixedColor": "#1baf7a" } } ] }
        ]
      },
      "options": { "legend": { "displayMode": "list", "placement": "bottom", "showLegend": true }, "tooltip": { "mode": "multi", "sort": "desc" } },
      "targets": [
        { "refId": "A", "expr": "nvidia_smi_memory_used_bytes{host=\"$host\"}", "legendFormat": "Used" },
        { "refId": "B", "expr": "nvidia_smi_memory_total_bytes{host=\"$host\"} - nvidia_smi_memory_used_bytes{host=\"$host\"}", "legendFormat": "Free" }
      ]
    },
    {
      "id": 20,
      "type": "row",
      "title": "GPU & host",
      "gridPos": { "h": 1, "w": 24, "x": 0, "y": 17 },
      "collapsed": false
    },
    {
      "id": 21,
      "type": "stat",
      "title": "VRAM used",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 4, "w": 6, "x": 0, "y": 18 },
      "fieldConfig": {
        "defaults": { "unit": "bytes", "color": { "mode": "fixed", "fixedColor": "#2a78d6" } },
        "overrides": []
      },
      "options": { "reduceOptions": { "calcs": ["lastNotNull"] }, "textMode": "value", "colorMode": "value", "graphMode": "area" },
      "targets": [ { "refId": "A", "expr": "nvidia_smi_memory_used_bytes{host=\"$host\"}" } ]
    },
    {
      "id": 22,
      "type": "gauge",
      "title": "VRAM utilisation",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 4, "w": 6, "x": 6, "y": 18 },
      "fieldConfig": {
        "defaults": {
          "unit": "percent",
          "min": 0,
          "max": 100,
          "thresholds": {
            "mode": "absolute",
            "steps": [
              { "color": "green", "value": null },
              { "color": "orange", "value": 85 },
              { "color": "red", "value": 95 }
            ]
          }
        },
        "overrides": []
      },
      "options": { "reduceOptions": { "calcs": ["lastNotNull"] }, "showThresholdMarkers": true },
      "targets": [ { "refId": "A", "expr": "100 * nvidia_smi_memory_used_bytes{host=\"$host\"} / nvidia_smi_memory_total_bytes{host=\"$host\"}" } ]
    },
    {
      "id": 23,
      "type": "timeseries",
      "title": "GPU utilisation",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 4, "w": 12, "x": 12, "y": 18 },
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit",
          "min": 0,
          "max": 1,
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 12, "showPoints": "never" },
          "color": { "mode": "fixed", "fixedColor": "#2a78d6" }
        },
        "overrides": []
      },
      "options": { "legend": { "showLegend": true, "placement": "bottom" }, "tooltip": { "mode": "single" } },
      "targets": [ { "refId": "A", "expr": "nvidia_smi_utilization_gpu_ratio{host=\"$host\"}", "legendFormat": "GPU util" } ]
    },
    {
      "id": 24,
      "type": "timeseries",
      "title": "GPU temperature",
      "description": "Two different scales (temp vs power) are intentionally split into two panels — never a dual y-axis.",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 6, "w": 6, "x": 0, "y": 22 },
      "fieldConfig": {
        "defaults": {
          "unit": "celsius",
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 8, "showPoints": "never" },
          "color": { "mode": "fixed", "fixedColor": "#eb6834" }
        },
        "overrides": []
      },
      "options": { "legend": { "showLegend": true, "placement": "bottom" }, "tooltip": { "mode": "single" } },
      "targets": [ { "refId": "A", "expr": "nvidia_smi_temperature_gpu{host=\"$host\"}", "legendFormat": "Temp" } ]
    },
    {
      "id": 25,
      "type": "timeseries",
      "title": "GPU power draw",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 6, "w": 6, "x": 6, "y": 22 },
      "fieldConfig": {
        "defaults": {
          "unit": "watt",
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 8, "showPoints": "never" },
          "color": { "mode": "fixed", "fixedColor": "#1baf7a" }
        },
        "overrides": []
      },
      "options": { "legend": { "showLegend": true, "placement": "bottom" }, "tooltip": { "mode": "single" } },
      "targets": [ { "refId": "A", "expr": "nvidia_smi_power_draw_watts{host=\"$host\"}", "legendFormat": "Power" } ]
    },
    {
      "id": 26,
      "type": "timeseries",
      "title": "Host CPU load (1m)",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 6, "w": 6, "x": 12, "y": 22 },
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 8, "showPoints": "never" },
          "color": { "mode": "fixed", "fixedColor": "#2a78d6" }
        },
        "overrides": []
      },
      "options": { "legend": { "showLegend": true, "placement": "bottom" }, "tooltip": { "mode": "single" } },
      "targets": [ { "refId": "A", "expr": "system_cpu_load_average_1m{host=\"$host\"}", "legendFormat": "Load 1m" } ]
    },
    {
      "id": 27,
      "type": "timeseries",
      "title": "Host memory used",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 6, "w": 6, "x": 18, "y": 22 },
      "fieldConfig": {
        "defaults": {
          "unit": "bytes",
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 8, "showPoints": "never" },
          "color": { "mode": "fixed", "fixedColor": "#eb6834" }
        },
        "overrides": []
      },
      "options": { "legend": { "showLegend": true, "placement": "bottom" }, "tooltip": { "mode": "single" } },
      "targets": [ { "refId": "A", "expr": "system_memory_usage_bytes{host=\"$host\",state=\"used\"}", "legendFormat": "Used" } ]
    },
    {
      "id": 30,
      "type": "row",
      "title": "Latency & distribution (p95 per model)",
      "gridPos": { "h": 1, "w": 24, "x": 0, "y": 28 },
      "collapsed": false
    },
    {
      "id": 31,
      "type": "timeseries",
      "title": "Prompt size (tokens) — p95",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 29 },
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "color": { "mode": "palette-classic" },
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 0, "showPoints": "never" }
        },
        "overrides": []
      },
      "options": { "legend": { "displayMode": "list", "placement": "bottom", "showLegend": true }, "tooltip": { "mode": "multi", "sort": "desc" } },
      "targets": [
        { "refId": "A", "expr": "histogram_quantile(0.95, sum by (le, model) (rate(llamacpp:prompt_tokens_size_bucket{host=\"$host\"}[$__rate_interval])))", "legendFormat": "{{model}}" }
      ]
    },
    {
      "id": 32,
      "type": "timeseries",
      "title": "Context used (tokens) — p95",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 29 },
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "color": { "mode": "palette-classic" },
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 0, "showPoints": "never" }
        },
        "overrides": []
      },
      "options": { "legend": { "displayMode": "list", "placement": "bottom", "showLegend": true }, "tooltip": { "mode": "multi", "sort": "desc" } },
      "targets": [
        { "refId": "A", "expr": "histogram_quantile(0.95, sum by (le, model) (rate(llamacpp:context_used_tokens_bucket{host=\"$host\"}[$__rate_interval])))", "legendFormat": "{{model}}" }
      ]
    },
    {
      "id": 33,
      "type": "timeseries",
      "title": "Time to first token (s) — p95",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 37 },
      "fieldConfig": {
        "defaults": {
          "unit": "s",
          "color": { "mode": "palette-classic" },
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 0, "showPoints": "never" }
        },
        "overrides": []
      },
      "options": { "legend": { "displayMode": "list", "placement": "bottom", "showLegend": true }, "tooltip": { "mode": "multi", "sort": "desc" } },
      "targets": [
        { "refId": "A", "expr": "histogram_quantile(0.95, sum by (le, model) (rate(llamacpp:time_to_first_token_seconds_bucket{host=\"$host\"}[$__rate_interval])))", "legendFormat": "{{model}}" }
      ]
    },
    {
      "id": 34,
      "type": "timeseries",
      "title": "Generation latency (s) — p95",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 37 },
      "fieldConfig": {
        "defaults": {
          "unit": "s",
          "color": { "mode": "palette-classic" },
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 0, "showPoints": "never" }
        },
        "overrides": []
      },
      "options": { "legend": { "displayMode": "list", "placement": "bottom", "showLegend": true }, "tooltip": { "mode": "multi", "sort": "desc" } },
      "targets": [
        { "refId": "A", "expr": "histogram_quantile(0.95, sum by (le, model) (rate(llamacpp:generation_latency_seconds_bucket{host=\"$host\"}[$__rate_interval])))", "legendFormat": "{{model}}" }
      ]
    },
    {
      "id": 40,
      "type": "row",
      "title": "Throughput & speculative decoding (per model)",
      "gridPos": { "h": 1, "w": 24, "x": 0, "y": 45 },
      "collapsed": false
    },
    {
      "id": 41,
      "type": "timeseries",
      "title": "Generation throughput (tokens/s)",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 6, "w": 6, "x": 0, "y": 46 },
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "color": { "mode": "palette-classic" },
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 8, "showPoints": "never" }
        },
        "overrides": []
      },
      "options": { "legend": { "displayMode": "list", "placement": "bottom", "showLegend": true }, "tooltip": { "mode": "multi" } },
      "targets": [
        { "refId": "A", "expr": "sum by (model) (llamacpp:predicted_tokens_seconds{host=\"$host\"})", "legendFormat": "{{model}}" }
      ]
    },
    {
      "id": 42,
      "type": "timeseries",
      "title": "Prompt throughput (tokens/s)",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 6, "w": 6, "x": 6, "y": 46 },
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "color": { "mode": "palette-classic" },
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 8, "showPoints": "never" }
        },
        "overrides": []
      },
      "options": { "legend": { "displayMode": "list", "placement": "bottom", "showLegend": true }, "tooltip": { "mode": "multi" } },
      "targets": [
        { "refId": "A", "expr": "sum by (model) (llamacpp:prompt_tokens_seconds{host=\"$host\"})", "legendFormat": "{{model}}" }
      ]
    },
    {
      "id": 43,
      "type": "timeseries",
      "title": "Spec-decode acceptance rate",
      "description": "accepted / drafted, windowed, per model. Higher = the draft/MTP is earning its keep.",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 6, "w": 6, "x": 12, "y": 46 },
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit",
          "min": 0,
          "max": 1,
          "color": { "mode": "palette-classic" },
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 12, "showPoints": "never" }
        },
        "overrides": []
      },
      "options": { "legend": { "displayMode": "list", "placement": "bottom", "showLegend": true }, "tooltip": { "mode": "multi" } },
      "targets": [
        { "refId": "A", "expr": "sum by (model) (rate(llamacpp:draft_tokens_accepted_total{host=\"$host\"}[$__rate_interval])) / clamp_min(sum by (model) (rate(llamacpp:draft_tokens_total{host=\"$host\"}[$__rate_interval])), 1)", "legendFormat": "{{model}}" }
      ]
    },
    {
      "id": 44,
      "type": "timeseries",
      "title": "Context-shift rate (per s)",
      "description": "Rate of context shifts (oldest tokens discarded) per model. Sustained non-zero means a rung is undersized for its traffic.",
      "datasource": { "type": "prometheus", "uid": "${datasource}" },
      "gridPos": { "h": 6, "w": 6, "x": 18, "y": 46 },
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "color": { "mode": "palette-classic" },
          "custom": { "drawStyle": "line", "lineWidth": 2, "fillOpacity": 12, "showPoints": "never" }
        },
        "overrides": []
      },
      "options": { "legend": { "displayMode": "list", "placement": "bottom", "showLegend": true }, "tooltip": { "mode": "multi" } },
      "targets": [
        { "refId": "A", "expr": "sum by (model) (rate(llamacpp:n_ctx_shift_total{host=\"$host\"}[$__rate_interval]))", "legendFormat": "{{model}}" }
      ]
    }
  ]
}
```

- [ ] **Step 2: Validate the JSON parses**

Run: `python3 -m json.tool deploy/grafana/llama-server-models.json > /dev/null && echo VALID`
Expected: `VALID`

- [ ] **Step 3: Assert no `$model` variable leaked in**

Run: `grep -c '\$model' deploy/grafana/llama-server-models.json || true`
Expected: `0`

- [ ] **Step 4: Assert every llama panel groups by model**

Run: `python3 -c "import json; d=json.load(open('deploy/grafana/llama-server-models.json')); exprs=[t['expr'] for p in d['panels'] for t in p.get('targets',[])]; llama=[e for e in exprs if 'llamacpp:' in e]; bad=[e for e in llama if 'by (model)' not in e and 'by (le, model)' not in e]; print('BAD:', bad); assert not bad"`
Expected: `BAD: []`

- [ ] **Step 5: Validate every metric name against live Mimir**

Run:
```bash
MIMIR=http://192.168.0.42:9009/prometheus/api/v1
for m in llamacpp:kv_cache_k_bytes llamacpp:kv_cache_v_bytes llamacpp:prompt_tokens_size_bucket llamacpp:context_used_tokens_bucket llamacpp:time_to_first_token_seconds_bucket llamacpp:generation_latency_seconds_bucket llamacpp:predicted_tokens_seconds llamacpp:prompt_tokens_seconds llamacpp:draft_tokens_accepted_total llamacpp:draft_tokens_total llamacpp:n_ctx_shift_total nvidia_smi_memory_used_bytes nvidia_smi_memory_total_bytes nvidia_smi_utilization_gpu_ratio nvidia_smi_temperature_gpu nvidia_smi_power_draw_watts system_cpu_load_average_1m system_memory_usage_bytes; do
  n=$(curl -s -G "$MIMIR/series" --data-urlencode "match[]=$m" --data-urlencode 'start=0' | python3 -c "import sys,json;print(len(json.load(sys.stdin)['data']))")
  echo "$n  $m"
done
```
Expected: a nonzero count for each metric name (they exist in Mimir retention even when the server is currently down). If any line shows `0`, that metric name is wrong — fix it before committing.

- [ ] **Step 6: Commit**

```bash
git add deploy/grafana/llama-server-models.json
git commit -m "deploy: add multi-model Grafana dashboard for llama-server

Companion to llama-server-rainbow.json showing all models at once: a
ladder-activity state-timeline (which model was loaded, via present_over_time),
plus per-model KV / p95 latency / throughput / spec-decode panels grouped
by (model) with palette-classic coloring. GPU & host panels stay host-level.
No \$model selector. Metric names validated against live Mimir.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Document the second dashboard

**Files:**
- Modify: `deploy/README-metrics.md` (the `## Dashboard` section near the end)

**Interfaces:**
- Consumes: `deploy/grafana/llama-server-models.json` from Task 1.
- Produces: nothing (docs only).

- [ ] **Step 1: Add a subsection after the existing dashboard description**

In `deploy/README-metrics.md`, the `## Dashboard` section currently ends with a paragraph about the colorblind-safe palette. Immediately after that paragraph (before end of file), append:

```markdown

### Multi-model view

`deploy/grafana/llama-server-models.json` (uid `llama-models`) is a companion
dashboard that shows **all models at once** instead of one selected `$model`.
Import it the same way. It has no `$model` variable — every llama panel groups
`by (model)` and uses Grafana's colorblind-safe `palette-classic`, so each model
gets its own stable color and legend entry. A top **Ladder activity**
state-timeline shows which model was the loaded child over the window (built from
`present_over_time(llamacpp:kv_cache_k_bytes[5m])`), making client laddering
visible at a glance. GPU & host panels stay host-level (VRAM, GPU, CPU are shared
across whichever model is loaded).

Use `llama-server-rainbow.json` to drill into a single model in full detail; use
`llama-server-models.json` for the whole-ladder overview and per-model
comparison. Same idle-state caveat applies: inactive models show gaps/NaN, and
latency/throughput lines fill in only while a model is the loaded child.
```

- [ ] **Step 2: Verify both dashboards are referenced**

Run: `grep -c 'llama-server-models.json' deploy/README-metrics.md`
Expected: `1` (or more)

- [ ] **Step 3: Commit**

```bash
git add deploy/README-metrics.md
git commit -m "deploy: document the multi-model dashboard in README-metrics

Explain when to use llama-server-models.json (whole-ladder overview, no \$model
selector, per-model panels) vs llama-server-rainbow.json (single-model drilldown).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- File `deploy/grafana/llama-server-models.json`, uid/title/tags → Task 1 Step 1. ✓
- Templating: `$datasource` + `$host`, no `$model` → Task 1 Step 1; asserted Step 3. ✓
- `palette-classic` on per-model panels → all llama panels in Step 1. ✓
- Row 1 Ladder activity (state-timeline, `present_over_time`) → panel id 2. ✓
- Row 2 Memory & KV (KV by model + host VRAM used/free) → panels 11, 12. ✓
- Row 3 GPU & host (VRAM stat/gauge, GPU util/temp/power, CPU load, host mem) → panels 21–27. ✓
- Row 4 Latency p95 by model (prompt/context/TTFT/gen) → panels 31–34. ✓
- Row 5 Throughput & spec-decode by model (gen/prompt tok/s, accept rate, ctx-shift) → panels 41–44. ✓
- README paragraph, when-to-use-which → Task 2. ✓

**Placeholder scan:** No TBD/TODO; all JSON and commands are literal. ✓

**Type consistency:** All `expr` group by `(model)` or `(le, model)` (asserted in Step 4); legends all `{{model}}`; datasource block identical everywhere; metric names match Global Constraints and are validated in Step 5. ✓
