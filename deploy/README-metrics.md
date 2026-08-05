# llama-server metrics → Mimir

## Topology
An OTel Collector agent on the Fedora GPU host scrapes the local llama-server
router, host metrics, and a local GPU exporter, then remote_writes directly to
Mimir. Grafana reads Mimir. See
`docs/superpowers/specs/2026-07-21-llama-server-metrics-design.md`.

This procedure was validated end-to-end on a real deployment (host `rainbow`,
RTX 4070 SUPER, otelcol-contrib v0.156.0, Mimir on a separate box) — the notes
below record what actually worked, not just the intended design.

## Prerequisites

### 1. otelcol-contrib (NOT in Fedora repos)
`dnf search otelcol` finds nothing — OpenTelemetry ships RPMs as GitHub release
assets, and you need the **contrib** distribution (core lacks `hostmetrics`).
```bash
# resolve the latest version, then install the amd64 contrib RPM
VER=$(curl -sI https://github.com/open-telemetry/opentelemetry-collector-releases/releases/latest \
      | grep -i '^location:' | grep -oP 'tag/v\K[0-9.]+')
curl -LO "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${VER}/otelcol-contrib_${VER}_linux_amd64.rpm"
sudo dnf install "./otelcol-contrib_${VER}_linux_amd64.rpm"   # installs + enables the service
```
Confirm the receivers/exporter this config needs are present:
```bash
otelcol-contrib components | grep -E "hostmetrics|prometheus|prometheusremotewrite|resourcedetection|batch"
```
Note: the contrib build has **no `dcgm` receiver** — that is why GPU metrics go
via the exporter below, not a dcgm receiver.

### 2. llama-server with --metrics
The router must be started with `--metrics`. In **router mode**, children
inherit the flag, and the router's `/metrics` proxies the active child's series
(labelled by `model`). Scrape ONLY the router port — see the config comment.
Verify: `curl -s -o /dev/null -w '%{http_code}\n' 127.0.0.1:8080/metrics` → `200`
(a `501` means that endpoint was started without `--metrics`).

### 3. GPU: nvidia_gpu_exporter (also a GitHub RPM)
A single Go binary that wraps `nvidia-smi` and exposes `:9835`. Preferred over
DCGM on consumer GeForce cards.
```bash
VER=$(curl -sI https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/latest \
      | grep -i '^location:' | grep -oP 'tag/v\K[0-9.]+')
curl -LO "https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/download/v${VER}/nvidia_gpu_exporter_${VER}_linux_x86_64.rpm"
sudo dnf install "./nvidia_gpu_exporter_${VER}_linux_x86_64.rpm"
sudo systemctl enable --now nvidia_gpu_exporter
```
Verify: `curl -s 127.0.0.1:9835/metrics | grep nvidia_smi_utilization_gpu_ratio`.
This exporter version emits `nvidia_smi_*` metric names (older builds used
`nvidia_gpu_*` — check yours and adjust the Grafana queries accordingly).

## Fill in before deploying
- `scrape_configs` targets: the router port (`127.0.0.1:8080` by default) and
  the GPU exporter (`127.0.0.1:9835`).
- `<mimir-host>`: Mimir host (default push path `/api/v1/push`, port 9009).
- `<gpuhost>`: `external_labels.host` value identifying this box (e.g. `rainbow`);
  it tags every series — llama, GPU, and host — so panels join on it.
- `X-Scope-OrgID`: uncomment ONLY if Mimir has `multitenancy_enabled: true`
  (with multitenancy off, omit it).

## Install (systemd)
1. `sudo cp deploy/otel-collector-llama.yaml /etc/otelcol-contrib/config.yaml`
2. Edit the placeholders above.
3. Validate before restarting:
   `otelcol-contrib validate --config=/etc/otelcol-contrib/config.yaml`
4. `sudo systemctl restart otelcol-contrib` (the RPM already enabled it at boot).
5. Confirm data lands in Mimir (run from anywhere that can reach it; note the
   `-G --data-urlencode` — raw `{}`/`"` in a URL fail to parse):
   ```bash
   curl -s -G 'http://<mimir-host>:9009/prometheus/api/v1/query' \
     --data-urlencode 'query=up{job="llama-server"}'         # expect value "1"
   curl -s -G 'http://<mimir-host>:9009/prometheus/api/v1/query' \
     --data-urlencode 'query=nvidia_smi_utilization_gpu_ratio'
   ```

## Avoid double-scraping
If a pre-existing collector/Prometheus already scrapes this router (e.g. a
central agent on the Grafana box), remove that job once this local agent is
live — otherwise the same `/metrics` is ingested twice under different
`job`/`instance` labels, doubling series and risking double-counts in
un-pinned queries. Stale series age out of Mimir on their own (~5 min for `up`).

## Grafana panels (PromQL)
- KV vs VRAM (headline): `llamacpp:kv_cache_k_bytes + llamacpp:kv_cache_v_bytes`
  overlaid with `nvidia_smi_memory_used_bytes` and `nvidia_smi_memory_total_bytes`
  (join on `host`).
- GPU utilisation: `nvidia_smi_utilization_gpu_ratio` (0–1); temp
  `nvidia_smi_temperature_gpu`; power `nvidia_smi_power_draw_watts`.
- Prompt-size p95: `histogram_quantile(0.95, sum by (le,model) (rate(llamacpp:prompt_tokens_size_bucket[$__rate_interval])))`
- TTFT p95: same over `llamacpp:time_to_first_token_seconds_bucket`.
- Spec-decode accept rate: `rate(llamacpp:draft_tokens_accepted_total[$__rate_interval]) / rate(llamacpp:draft_tokens_total[$__rate_interval])`
- Context-shift rate: `rate(llamacpp:n_ctx_shift_total[$__rate_interval])`
- Host CPU busy: hostmetrics emits `system_cpu_time_seconds_total` (counter),
  memory `system_memory_usage_bytes{state="used"}`, load `system_cpu_load_average_1m`.

## Dashboard
A ready-made dashboard is at `deploy/grafana/llama-server-rainbow.json`. Import
it in Grafana (Dashboards → New → Import → Upload JSON), then pick the Prometheus
(Mimir) data source; the `$host` and `$model` variables auto-populate from label
values. Rows: **Memory & KV cache** (KV bytes vs free VRAM, live cache type),
**GPU & host**, **Latency & distribution** (prompt/context/TTFT/gen quantiles),
**Throughput & speculative decoding**. Latency/quantile panels read `NaN` while
the server is idle (no `rate()` samples) — they fill in once requests flow.
The palette is colorblind-safe (validated blue/orange/aqua); every multi-series
panel carries a legend so identity is never color-alone.

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
