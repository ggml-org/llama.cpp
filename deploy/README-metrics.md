# llama-server metrics → Mimir

## Topology
OTel Collector agent on the Fedora GPU host scrapes localhost llama-server rungs,
host metrics, and GPU metrics, then remote_writes directly to Mimir. Grafana reads
Mimir. See `docs/superpowers/specs/2026-07-21-llama-server-metrics-design.md`.

## Prerequisites
- llama-server started with `--metrics` (bind metrics port to 127.0.0.1).
- `otelcol-contrib` (for the `dcgm` and `hostmetrics` receivers). Verify:
  `otelcol-contrib components | grep -E "dcgm|hostmetrics|prometheusremotewrite"`
- For GPU: either the `nvidia-dcgm` service running, OR `nvidia_gpu_exporter`
  (then swap the `dcgm` receiver for a prometheus scrape job — see config comment).

## Fill in before deploying
- `static_configs.targets`: one `127.0.0.1:<port>` per running rung.
- `<mimir-host>`: Mimir distributor host (default push path `/api/v1/push`, port 9009).
- `<gpuhost>`: label value identifying this host.
- `X-Scope-OrgID`: uncomment iff Mimir enforces multi-tenancy (else writes 401).

## Install (systemd sketch)
1. `cp deploy/otel-collector-llama.yaml /etc/otelcol-contrib/config.yaml`
2. Edit placeholders.
3. `systemctl restart otelcol-contrib && systemctl enable otelcol-contrib`
4. Verify: `curl -s localhost:8888/metrics | grep otelcol_exporter_sent` climbs.

## Grafana panels (PromQL)
- KV vs VRAM: `llamacpp:kv_cache_k_bytes + llamacpp:kv_cache_v_bytes` overlaid with
  `llamacpp:vram_free_bytes` (and DCGM `DCGM_FI_DEV_FB_USED`).
- Prompt p95: `histogram_quantile(0.95, sum by (le,model) (rate(llamacpp:prompt_tokens_size_bucket[$__rate_interval])))`
- TTFT p95: same over `llamacpp:time_to_first_token_seconds_bucket`.
- Spec-decode accept rate: `rate(llamacpp:draft_tokens_accepted_total[$__rate_interval]) / rate(llamacpp:draft_tokens_total[$__rate_interval])`
- Context-shift rate: `rate(llamacpp:n_ctx_shift_total[$__rate_interval])`
