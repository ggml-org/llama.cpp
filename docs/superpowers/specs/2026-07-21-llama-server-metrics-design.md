# llama-server metrics expansion — design

**Date:** 2026-07-21
**Status:** approved (design), pending implementation plan
**Author:** Nick Farrell (with Claude)

## Purpose

Expand observability of `llama-server` to support **tuning decisions** on a
memory-constrained deployment (single RTX 4070, 12 GB VRAM, running a ~35B MoE
model with context "laddering" across rungs 8k/16k/32k/64k/128k/256k).

The metrics must answer concrete operational questions:

1. **Memory headroom / OOM risk** — how much VRAM is KV cache vs weights, and
   how much would a different KV-cache quant (e.g. f16 → q8_0) save?
2. **Laddering thresholds** — what is the real distribution of prompt/context
   sizes, so rung boundaries can be placed correctly?
3. **Spec-decode effectiveness** — is MTP/draft speculation earning its keep?
4. **Latency breakdown** — time-to-first-token (prompt eval) vs generation
   latency, including tail behaviour and the cost of re-prefill on a rung switch.
5. **CPU / GPU load** — host and GPU utilisation, temperature, power.

**This work is local to the deployment fork.** It is not intended for upstream,
so it may be pragmatic and llama.cpp-specific. It should still be minimal and
follow existing conventions to remain maintainable on the fork.

## Guiding principle: produce each metric where its data lives

- **llama-server** exposes only what *it alone knows*: KV-cache byte sizes, the
  live cache-type, prompt/context-size distributions, prompt-eval and generation
  latency distributions, and spec-decode counters. These cannot be derived from
  outside the process.
- **Hardware metrics** (GPU util/mem/temp/power, host CPU/RAM/disk) come from
  standard collectors, **not** from llama.cpp. ggml exposes no compute-utilisation
  API (no SM occupancy / GPU-load call); only `ggml_backend_dev_memory()` for VRAM
  free/total. True utilisation requires NVML/DCGM or `/proc`, which belong in a
  collector, not the inference engine.

## Architecture / collection topology

A single **OpenTelemetry Collector agent runs on the Fedora GPU host** and ships
directly to Mimir. This is required (not merely convenient): `hostmetrics` and the
NVIDIA GPU receivers can only observe the machine they run on, so a local agent is
mandatory for the hardware half. Once present, it also scrapes the llama-server
rungs on `localhost`, which keeps the metrics port off the network.

```
Fedora GPU host                                     Mimir/Grafana host
┌───────────────────────────────────────┐          ┌─────────────────┐
│ llama-server rungs ──/metrics(local)──▶│          │                 │
│                        OTel Collector  │ remote_  │      Mimir      │
│ hostmetrics ──────────▶ (agent, NEW)   │ write    │        ▲        │
│ GPU (DCGM/nvml) ──────▶                │─────────▶│────────┘        │
└───────────────────────────────────────┘          │      Grafana    │
                                                    └─────────────────┘
```

- Existing central OTel Collector on the Mimir box is **not** in this path.
- Agent writes **directly to Mimir** via `prometheusremotewrite` (one less hop
  than forwarding through the central Collector).
- Correlation in Grafana: llama-server series carry `{model="…"}` (already
  emitted today); hardware series carry `host`/`instance`. Panels join on these.

### Division of labor

| Concern | Source | Mechanism |
|---|---|---|
| KV-cache K/V **bytes**, live cache **type**, prompt/context-size & latency **histograms**, spec-decode counters, VRAM gauges | **llama-server `/metrics`** (new) | Collector `prometheus` receiver scrapes each rung on localhost |
| Host CPU / RAM / disk / load | **OTel Collector** | `hostmetrics` receiver (built-in) |
| GPU util / mem / temp / power | **OTel Collector** | `dcgm` receiver (contrib) or scrape `nvidia_gpu_exporter` |

**Out of scope:** no custom exporter/sidecar binary is built. Hardware metrics use
off-the-shelf receivers/exporters only.

## llama-server `/metrics` additions

All new fields are added to `server_metrics` (`tools/server/server-context.cpp:787`),
copied into `server_task_result_metrics`, and emitted in the exposition block
(`tools/server/server-context.cpp:4404`). The `llamacpp:` namespace and the
existing `{model="…"}` label are preserved on every new series.

### 2a. KV-cache bytes and type — headline metric for quant decisions

| Metric | Type | Source |
|---|---|---|
| `kv_cache_k_bytes` | gauge | `Σ_layers ggml_row_size(type_k, n_embd_k_gqa(il)) × n_ctx` |
| `kv_cache_v_bytes` | gauge | same with `type_v` / `n_embd_v_gqa(il)` |
| `kv_cache_type_k` | gauge, labelled `{type="…"}` | value `1` for the live K cache type (e.g. `f16`, `q8_0`) |
| `kv_cache_type_v` | gauge, labelled `{type="…"}` | value `1` for the live V cache type |

This is precisely what DCGM/hostmetrics **cannot** report: how the 12 GB budget
splits between KV cache, weights, and compute buffers, and how much a KV-quant
change would reclaim. Cache geometry is fixed at load, so these are computed once
at startup — near-zero runtime cost.

### 2b. Prompt-size and context histograms — laddering thresholds

| Metric | Type |
|---|---|
| `prompt_tokens` (histogram: `_bucket`/`_sum`/`_count`) | histogram |
| `context_used_tokens` (histogram) | histogram |

Bucket boundaries (aligned to the 8k→256k power-of-2 ladder, with a sub-bucket
just below each rung so headroom before the next rung is visible):

```
512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768,
49152, 65536, 98304, 131072, 196608, 262144
```

Prometheus appends `+Inf` automatically. `histogram_quantile()` then answers
"what prompt size is my p95?" → whether a rung is overflowing or over-provisioned.

### 2c. Latency histograms — rung-switch cost and tails

| Metric | Type | Source |
|---|---|---|
| `time_to_first_token_seconds` (histogram) | histogram | `t_prompt_processing` per request (`server-context.cpp:504`) |
| `generation_latency_seconds` (histogram) | histogram | `t_token_generation` per request |

Bucket boundaries (wide — a 128k/256k prefill on an offloaded 12 GB card can take
seconds):

```
time_to_first_token_seconds: 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64
generation_latency_seconds:  0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 80
```

TTFT tail is where the re-prefill cost of a laddering rung-switch shows up.

### 2d. Spec-decode & context-shift — Grafana-only, no new server metric

`draft_tokens_total` and `draft_tokens_accepted_total` already exist. Acceptance
**rate** is computed in Grafana as `rate(accepted)/rate(total)` over the chosen
window — correct for any interval, unlike a server-side lifetime-averaged ratio
gauge which would be misleading. `n_ctx_shift_total` already exists and needs no
change. **No new server metric here** — only Grafana panels.

### 2e. VRAM gauges — included

| Metric | Type | Source |
|---|---|---|
| `vram_free_bytes` | gauge, per device | `ggml_backend_dev_memory(dev, &free, &total)` |
| `vram_total_bytes` | gauge, per device | same |

Overlaps DCGM/hostmetrics, but carries the `{model="…"}` label natively, making
per-rung correlation trivial (e.g. KV-cache bytes vs free VRAM on one panel).
Cheap to query; included deliberately.

### New exposition machinery — cumulative histograms

The current renderer emits scalar counters/gauges only. A small helper is added to
the exposition loop to emit histogram series (`_bucket{le="…"}`, `_sum`, `_count`).

**Important semantic divergence:** these histograms are **cumulative (monotonic)**
— Prometheus/Mimir deltas them via `rate()` / `histogram_quantile()`. They do
**not** follow the reset-per-scrape behaviour introduced in commit `87c14ba1d`
(which resets bucket-scoped gauges like `n_tokens_max` on each scrape). This is the
correct and intended semantics for Prometheus histograms; the divergence from the
existing bucket-reset pattern is deliberate and documented here so it is not
mistaken for an inconsistency. Existing bucket-reset gauges are left unchanged.

## OTel Collector agent configuration

Single agent on the GPU host; three receivers; direct remote_write to Mimir.

```yaml
receivers:
  prometheus:                       # 3a. llama-server rungs, localhost only
    config:
      scrape_configs:
        - job_name: llama-server
          scrape_interval: 15s
          static_configs:
            - targets: ['127.0.0.1:8081', '127.0.0.1:8082']   # one per rung (confirm ports)
  hostmetrics:                      # 3b. host CPU/RAM/disk/load (local-only observable)
    collection_interval: 15s
    scrapers: { cpu: {}, memory: {}, load: {}, filesystem: {}, network: {} }
  dcgm:                             # 3c. GPU util/mem/temp/power (needs nvidia-dcgm)
    collection_interval: 15s
    # fallback if Collector lacks contrib: scrape nvidia_gpu_exporter via a prometheus job

processors:
  resourcedetection: { detectors: [system], system: { hostname_sources: [os] } }
  batch: {}

exporters:
  prometheusremotewrite:
    endpoint: "http://<mimir-host>:9009/api/v1/push"
    external_labels: { host: "<gpuhost>" }
    # headers: { X-Scope-OrgID: "<tenant>" }   # required iff Mimir is multi-tenant

service:
  pipelines:
    metrics:
      receivers: [prometheus, hostmetrics, dcgm]
      processors: [resourcedetection, batch]
      exporters: [prometheusremotewrite]
```

### Environment facts to confirm at implementation time (not design decisions)

1. **GPU receiver:** `dcgm` ships in `otelcol-contrib` and needs the `nvidia-dcgm`
   service. If the installed Collector is the core distro, fall back to
   `nvidia_gpu_exporter` + a `prometheus` scrape job. Verify which is installed.
2. **Mimir tenancy:** if multi-tenant, `X-Scope-OrgID` is required or writes 401.
3. **Rung ports** and **Mimir endpoint** are deployment specifics.
4. **Label hygiene:** `model` (from llama-server) + `host`/`instance` (from
   `resourcedetection`/`external_labels`) must be sufficient and non-colliding for
   Grafana joins.

## Testing

- **Unit-ish:** verify histogram exposition format is valid Prometheus (buckets
  monotonic, `le="+Inf"` present, `_sum`/`_count` emitted) by scraping `/metrics`
  from a locally-running server against a local gguf. (Note: server pytest cannot
  download models in this sandbox — verify against a local gguf, see memory
  `llamacpp-server-test-sandbox`.)
- **KV-byte correctness:** cross-check `kv_cache_k_bytes + kv_cache_v_bytes`
  against the `KV self size = … MiB` startup log line for a known model + cache
  type; and confirm it changes as expected when switching `--cache-type-k`.
- **Histogram sanity:** drive prompts of known sizes, confirm they land in the
  expected buckets.
- **End-to-end:** confirm the Collector agent scrapes all rungs, hostmetrics and
  GPU, and that series land in Mimir with correct `model`/`host` labels and render
  in Grafana.

## Success criteria

- Grafana can plot, per rung (`model`): KV-cache bytes vs free VRAM; p50/p95/p99 of
  prompt size and context used; p95 TTFT and generation latency; spec-decode accept
  rate; context-shift rate.
- Grafana can plot, per host: GPU util/mem/temp/power and CPU/RAM.
- The KV-cache-bytes metric lets a KV-quant change (f16 ↔ q8_0) be evaluated for
  VRAM savings before committing.
```
