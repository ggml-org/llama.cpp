# llama-server metrics → Mimir

## Topology
An OTel Collector agent on the Fedora GPU host scrapes the local llama-server
router, host metrics, and a local GPU exporter, then remote_writes directly to
Mimir. Grafana reads Mimir. See
`docs/superpowers/specs/2026-07-21-llama-server-metrics-design.md`.

This procedure was validated end-to-end on a real deployment (host `rainbow`,
Intel Arc Pro B70 32 GB on the `xe` driver, SYCL build in a podman container,
otelcol-contrib v0.156.0, Mimir on a separate box) — the notes below record what
actually worked, not just the intended design.

> **GPU vendor note.** This was originally built against an RTX 4070 SUPER and
> `nvidia_gpu_exporter`. It now targets Intel Arc. The lesson from the swap:
> `nvidia_gpu_exporter` kept running and kept answering `:9835` after the card
> was gone, serving only `nvidia_smi_command_exit_code 9` and
> `nvidia_smi_failed_scrapes_total`, so every GPU panel went silently blank. The
> Intel bridge below always publishes `intel_gpu_scrape_success`; alert on it
> (`intel_gpu_scrape_success == 0`, or `node_textfile_mtime_seconds` going
> stale) rather than trusting a panel to look wrong.

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
Note: the contrib build has **no GPU receiver we can use** — no `dcgm`/`nvml`,
and nothing for Intel — which is why GPU metrics come in over a scraped
exporter (section 3), not a receiver.

### 2. llama-server with --metrics
The router must be started with `--metrics`. In **router mode**, children
inherit the flag, and the router's `/metrics` proxies the active child's series
(labelled by `model`). Scrape ONLY the router port — see the config comment.
Verify: `curl -s -o /dev/null -w '%{http_code}\n' 127.0.0.1:8080/metrics` → `200`
(a `501` means that endpoint was started without `--metrics`).

### 3. GPU (Intel Arc): node_exporter + a metrics bridge

Three signals; VRAM comes free from llama-server, the other two from one small
bridge script:

| Signal | Source | Notes |
| ------ | ------ | ----- |
| VRAM used / free / total | **llama-server itself** (`llamacpp:vram_*`) | Already there, per SYCL device. Nothing to install. |
| Engine busy | **the bridge**, from `/proc/<pid>/fdinfo` | No perf, no capabilities, no igt-gpu-tools. |
| Temperature, energy, fan | **the bridge**, from the GPU's hwmon sysfs | Not node_exporter's hwmon collector — see below. |

node_exporter is present only to host its **textfile collector**; both of the
bridge's outputs land in one `.prom` file. Its own hwmon collector is *not*
used, for two reasons: `xe` exposes no instantaneous `power1_average` (only a
cumulative `energy1_input`, so power needs `rate()`), and reading the sysfs
directly lets the bridge use each sensor's own `*_label` — the Arc Pro B70
publishes 20 temperature sensors (`pkg`, `vram`, `mctrl`, `pcie` and 16
`vram_ch_N`), and `sensor="temp5"` would tell you nothing. It also
avoids a `group_left` join to separate the Arc from the host's other hwmon
devices; rainbow also carries `amdgpu`, `nvme`, `k10temp` and two `spd5118`.

> **Do not reach for `intel_gpu_top`.** It is i915-only and refuses outright on
> the `xe` driver used by Arc B-series: *"Detected Xe device which is not
> supported by intel_gpu_top. Please use 'gputop' tool instead."* And `gputop`
> has no machine-readable output in igt-gpu-tools 2.4 — only `-d`, `-n`, `-h`
> — so there is nothing to parse but a TUI table. The bridge below reads the
> kernel's `drm-usage-stats` interface instead, which is both more portable and
> less privileged.

`llamacpp:vram_*` replaces what `nvidia_smi_memory_*` used to provide, and it is
strictly better here: it reports per-`device` (`SYCL0`, …) exactly what the
backend llama.cpp is actually allocating from. The one caveat is that it only
exists while llama-server is running, so it tracks the server's view of the GPU,
not the card's absolute state.

On a host with more than one GPU, confirm which card a `device` label refers to
before trusting the number — both of rainbow's GPUs are passed into the
container, and system RAM (60 GiB) is close enough to the Arc's 32 GB that a
shared-memory device would not obviously look wrong:
```bash
podman exec <container> sycl-ls    # names each device, in enumeration order
```
On rainbow this reports one device, `Intel(R) Arc(TM) Pro B70 Graphics`, because
`ONEAPI_DEVICE_SELECTOR=level_zero:gpu` restricts the runtime to Level Zero
GPUs — the AMD integrated GPU can never be selected. So `SYCL0` is the Arc.

**node_exporter** — packaged in Fedora, and the only thing to install:
```bash
sudo dnf install golang-github-prometheus-node-exporter
sudo mkdir -p /var/lib/node_exporter/textfile_collector
```
Run it with defaults **disabled**, so it only serves the textfile —
CPU/memory/filesystem series already come from the `host_metrics` receiver and
would otherwise be ingested twice:
```bash
# /etc/sysconfig/node_exporter  (Fedora's unit sources this as $OPTIONS)
OPTIONS="--collector.disable-defaults --collector.textfile \
  --collector.textfile.directory=/var/lib/node_exporter/textfile_collector"
```
```bash
sudo systemctl enable --now node_exporter
curl -s 127.0.0.1:9100/metrics | grep intel_gpu_
```
Only the textfile collector is enabled, so node_exporter itself contributes no
sensor series — everything GPU comes from the bridge below.

**The bridge** — reads per-engine busy counters out of `/proc/<pid>/fdinfo` and
sensors out of the GPU's hwmon, into a `.prom` file for the textfile collector:
```bash
sudo install -m 0755 deploy/intel-gpu-textfile.py /usr/local/bin/
sudo install -m 0644 deploy/systemd/intel-gpu-metrics.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now intel-gpu-metrics.timer
```
Verify — and note the failure mode is explicit, not blank:
```bash
sudo systemctl start intel-gpu-metrics.service
cat /var/lib/node_exporter/textfile_collector/intel_gpu.prom
# intel_gpu_scrape_success 1  -> good
# intel_gpu_scrape_success 0  -> read the reason: journalctl -u intel-gpu-metrics -n 20
```
The two drivers report the same thing in different forms, and the bridge handles
both, normalising to identical `engine=` labels (`render`, `copy`, `compute`,
`video`, `video-enhance`):

| Driver | fdinfo keys | Unit |
| ------ | ----------- | ---- |
| `xe` | `drm-cycles-<class>` / `drm-total-cycles-<class>` | driver ticks |
| `i915` | `drm-engine-<class>` (plus a synthesised `CLOCK_MONOTONIC` reference) | nanoseconds |

Both are cumulative, so they are exported as **counters** and utilisation is
`rate(busy) / rate(elapsed)` — dimensionless, so one query works on either
driver, and it averages over the whole scrape interval instead of a sampled
slice. `drm-engine-capacity-<class>` is an engine *count* (2 for VCS on Arc), so
the bridge divides by it to keep the ratio in 0..1.

Every series carries a `driver="xe"` (or `i915`) label. Hosts commonly have more
than one GPU — rainbow pairs the Arc with an AMD integrated one — so which
device a number came from is visible in the data rather than implied by a
filter. Non-Intel GPUs are skipped entirely.

Three consequences worth knowing:
- It runs as root only to read fdinfo of processes owned by `llama`. If you
  harden the unit further, do **not** add `ProtectProc=` or `ProcSubset=` —
  they hide other processes' fdinfo and you get a silent zero, not an error.
- llama.cpp runs in a podman container here, which does not matter: container
  processes are visible in the host PID namespace, so the host-side bridge reads
  their fdinfo normally, and it keeps working while the container is restarting.
- fdinfo is per-DRM-client, so engine busy is llama-server's GPU usage, not the
  whole card's. In router mode the counters reset when a model swap replaces the
  child process; `rate()` absorbs the reset at the cost of one interval.

## Fill in before deploying
- `scrape_configs` targets: the router port (`127.0.0.1:8080` by default) and
  node_exporter (`127.0.0.1:9100`).
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
     --data-urlencode 'query=llamacpp:vram_free_bytes'          # VRAM, from llama-server
   curl -s -G 'http://<mimir-host>:9009/prometheus/api/v1/query' \
     --data-urlencode 'query=intel_gpu_scrape_success'          # expect value "1"
   curl -s -G 'http://<mimir-host>:9009/prometheus/api/v1/query' \
     --data-urlencode 'query=rate(intel_gpu_engine_busy_total[5m])'
   ```

## Avoid double-scraping
If a pre-existing collector/Prometheus already scrapes this router (e.g. a
central agent on the Grafana box), remove that job once this local agent is
live — otherwise the same `/metrics` is ingested twice under different
`job`/`instance` labels, doubling series and risking double-counts in
un-pinned queries. Stale series age out of Mimir on their own (~5 min for `up`).

## Grafana panels (PromQL)
- KV vs VRAM (headline): `llamacpp:kv_cache_k_bytes + llamacpp:kv_cache_v_bytes`
  overlaid with free VRAM,
  `sum by (host) (max by (host, device) (llamacpp:vram_free_bytes))`.
  The `max by (host, device)` is load-bearing: it collapses the `model` label.
  The router only exposes the active child's series, but stale ones linger in
  Mimir for ~5 min, so a plain `sum` double-counts across model labels during a
  ladder switch. Used = `total - free` per device, then summed.
- GPU engine utilisation:
  `rate(intel_gpu_engine_busy_total[$__rate_interval]) / rate(intel_gpu_engine_elapsed_total[$__rate_interval])`
  (0–1, one series per engine; watch `compute` and `render` for inference).
- GPU temperature: `intel_gpu_temp_celsius{sensor!~"vram_ch_.*"}` — the Arc Pro
  B70 publishes 20 sensors, 16 of them per-VRAM-channel; the useful four are
  `pkg`, `vram` (tracks the hottest channel), `mctrl` and `pcie`. Drop the
  matcher for per-channel detail.
- GPU power: `rate(intel_gpu_energy_joules_total[$__rate_interval])` — watts,
  since `xe` publishes only a cumulative energy counter. Two domains: `card`
  (whole board) and `pkg` (the GPU package). `intel_gpu_power_cap_watts` is the
  configured limit (275 W on the B70), for reference.
- GPU telemetry health: `intel_gpu_scrape_success` (alert on `== 0`), and
  `intel_gpu_hwmon_sensors` / `intel_gpu_clients` to see what was found.
- Prompt-size p95: `histogram_quantile(0.95, sum by (le,model) (rate(llamacpp:prompt_tokens_size_bucket[$__rate_interval])))`
- TTFT p95: same over `llamacpp:time_to_first_token_seconds_bucket`.
- Spec-decode accept rate: `rate(llamacpp:spec_decode_num_accepted_tokens_total[$__rate_interval]) / clamp_min(rate(llamacpp:spec_decode_num_draft_tokens_total[$__rate_interval]), 1e-9)`
  (the `clamp_min` only guards the 0/0 case when spec-decode is off; do not clamp
  to `1`, that understates the ratio whenever the draft rate is below 1 token/s)
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
