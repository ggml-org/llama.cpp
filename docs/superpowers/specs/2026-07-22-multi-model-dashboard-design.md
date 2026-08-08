# Multi-model llama-server Grafana dashboard

**Date:** 2026-07-22
**Status:** approved design

## Problem

The existing dashboard `deploy/grafana/llama-server-rainbow.json` pins every
llama panel to a single `$model` template variable, so it can only show one
model at a time. In router mode the server hosts many models and a client
*ladders* between them (switching model per request). We want a companion
dashboard that shows **all models at once** — a whole-ladder overview plus
per-model comparison — without having to re-select `$model`.

This is most useful precisely because only one model is loaded at a time: the
dashboard makes the ladder's switching behaviour and each rung's per-model
metrics legible in one view.

## Constraints & verified facts (against live Mimir, 2026-07-22)

- Mimir at `http://192.168.0.42:9009`, `multitenancy_enabled: false` (no
  `X-Scope-OrgID`). Grafana reads Mimir. Single host `rainbow`, single scrape
  target (the router on `127.0.0.1:8080`, `job="llama-server"`).
- The `model` label has **50 distinct values** across retention, but only a
  handful emit in any recent window (4 in the last 24h at spec time). Grouping
  `by (model)` over a live gauge/counter naturally scopes to models with samples
  in the panel's time window, so the default 6h view stays uncluttered — no
  explicit model filter is required to keep it readable.
- Series carry both `host` and `model` labels (plus `instance`, `job`,
  `otel_scope_*`). Confirmed on `llamacpp:kv_cache_k_bytes`.
- `present_over_time(llamacpp:kv_cache_k_bytes[5m])` is supported and returns a
  per-model series that is present (value 1) only while that model was emitting.
  This is the liveness signal for the ladder-activity timeline: gaps = the model
  was not the loaded child.
- Router mode: only the **active** child emits fresh metrics; inactive models go
  stale (~5 min) then stop. So idle models show gaps/flat/NaN — same idle-state
  caveat already documented for the rainbow dashboard.

## Design

New file `deploy/grafana/llama-server-models.json`, a sibling to the rainbow
dashboard. Do not modify the rainbow dashboard.

- `uid`: `llama-models`
- `title`: `llama-server — models (rainbow)`
- `tags`: `["llama.cpp", "llm", "gpu"]`
- schemaVersion 39, `refresh: "30s"`, `time: now-6h → now` — matches rainbow.

### Templating

- `$datasource` — Prometheus datasource picker. Identical to rainbow.
- `$host` — `label_values(up{job="llama-server"}, host)`. Identical to rainbow.
- **No `$model` variable.** Every llama panel groups `by (model)` across all
  models on `$host`.

### Coloring

All per-model panels use `color.mode: "palette-classic"` so each model gets a
stable, distinct, colorblind-safe color assigned automatically — no hardcoded
model names (there are 50 and the set changes). Every multi-series panel keeps a
legend so model identity is never color-alone. Host-level panels (GPU/host row)
keep fixed single colors as in rainbow.

### Rows & panels

**Row 1 — Ladder activity** (new; the headline of this dashboard)
- `state-timeline` panel, full width (`w: 24`), showing which model was emitting
  over the window — one horizontal band per model, gaps where it wasn't loaded.
- Query: `sum by (model) (present_over_time(llamacpp:kv_cache_k_bytes{host="$host"}[5m]))`
- Value mapping: value `1` → text "loaded"; legend/row label = `{{model}}`.
  `color.mode: palette-classic`. This reads ladder switches across the window at
  a glance.

**Row 2 — Memory & KV cache**
- KV total per model — `timeseries`, one line per model, `palette-classic`:
  `sum by (model) (llamacpp:kv_cache_k_bytes{host="$host"} + llamacpp:kv_cache_v_bytes{host="$host"})`
- Host VRAM used + free (host-level, shared resource), adjacent:
  `nvidia_smi_memory_used_bytes{host="$host"}` and
  `nvidia_smi_memory_total_bytes{host="$host"} - nvidia_smi_memory_used_bytes{host="$host"}`
  — keeps the KV-vs-VRAM headline meaning. VRAM is shared across whichever model
  is loaded, so a single host-level line is correct.

**Row 3 — GPU & host** (kept as-is, host-level, single lines)
- VRAM used (stat), VRAM utilisation (gauge), GPU utilisation, GPU temperature,
  GPU power draw, host CPU load 1m, host memory used — identical queries and
  fixed colors to the rainbow dashboard, all `{host="$host"}`.

**Row 4 — Latency & distribution** (p95 only, one line per model)
Series explosion is avoided by showing a single quantile (p95 — the tail that
matters for rung tuning) with `series = model`. Four `timeseries` panels:
- Prompt size p95:
  `histogram_quantile(0.95, sum by (le, model) (rate(llamacpp:prompt_tokens_size_bucket{host="$host"}[$__rate_interval])))`
- Context used p95: same over `llamacpp:context_used_tokens_bucket`
- TTFT p95: same over `llamacpp:time_to_first_token_seconds_bucket` (unit `s`)
- Generation latency p95: same over `llamacpp:generation_latency_seconds_bucket` (unit `s`)

All `palette-classic`, legend `{{model}}`, panel titles say "p95".

**Row 5 — Throughput & speculative decoding** (one line per model)
- Generation tok/s: `sum by (model) (llamacpp:predicted_tokens_seconds{host="$host"})`
- Prompt tok/s: `sum by (model) (llamacpp:prompt_tokens_seconds{host="$host"})`
- Spec-decode accept rate:
  `sum by (model) (rate(llamacpp:draft_tokens_accepted_total{host="$host"}[$__rate_interval])) / clamp_min(sum by (model) (rate(llamacpp:draft_tokens_total{host="$host"}[$__rate_interval])), 1)`
- Context-shift rate:
  `sum by (model) (rate(llamacpp:n_ctx_shift_total{host="$host"}[$__rate_interval]))`

All `palette-classic`, legend `{{model}}`.

### Documentation

Add a short paragraph to `deploy/README-metrics.md` describing the second
dashboard and when to use which:
- `llama-server-rainbow.json` — drill into one model (`$model` selector), full
  host detail.
- `llama-server-models.json` — whole-ladder overview + per-model comparison; no
  model selector, everything grouped `by (model)`.
Note the idle-state caveat (inactive models show gaps/NaN; latency/throughput
fill in only while a model is the loaded child).

## Non-goals (YAGNI)

- No `$model` multi-select filter — grouping `by (model)` over live windows keeps
  it readable; a 50-value selector adds clutter for no gain.
- No p50/p99 on latency panels — p95 per model is enough; more quantiles ×
  models is unreadable.
- No "current active model" stat/table — the activity timeline covers the
  now-state and the history in one panel.
- No changes to the rainbow dashboard, the collector config, or metric names.

## Testing / verification

- `python3 -m json.tool` the new file (valid JSON).
- Cross-check every metric name and label against the rainbow dashboard and the
  live Mimir queries recorded above.
- Import-time: `$host` auto-populates; per-model panels render one series per
  recently-active model. (Full visual verification requires the server up and
  laddering traffic; server was down at spec time — `up=0`.)
