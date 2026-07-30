# llama-tweak TODO

Future work and design notes (not implemented unless stated otherwise in README).

## Metrics and selection strategy

- Today: maximize cached `mean_tps` (`avg_ts` from llama-bench JSONL) per `(pp, tg)` bucket.
- Add optional objectives: **TPOT**, **TTFT**, and other llama-bench fields; support **weighted** scores (for example latency vs throughput).
- Pluggable **selection strategy** (config or cache metadata): best metric, Pareto, or user-defined weights.
- **First-working-backend** mode: ordered backend list; use the first entry that completes a probe (no crash / OOM) instead of highest tok/s.

## Cache quality and failures

- Allow marking a backend as **broken for this model** in the JSON (failed probe, crash, OOM) so runtime and `explain` skip it without re-benchmarking every time.
- Distinguish "missing data" vs "known bad" vs "stale fingerprint".

## Backend matrix configuration

- Stop hardcoding the Intel MVP matrix in `record.cpp`; load case list from JSON/YAML or CLI flags.
- **Filter** backends (include/exclude tags, vendors, devices).
- Per-case env overrides beyond today's OpenVINO/SYCL/Vulkan knobs.

## SYCL Graphs backend variant

- SYCL Graph execution already exists when built with `GGML_SYCL_GRAPH` and enabled via **`GGML_SYCL_ENABLE_GRAPH=1`** (see [SYCL backend docs](../../docs/backend/SYCL.md)); it is experimental and **does not work reliably** on all models/drivers/shapes.
- Add optional **record** cases (or flags on existing SYCL cases) that bench with graphs on vs off.
- **Probe / capability detection**: short test graph; if recording or replay fails, mark the variant **unsupported** in the cache (same idea as NPU skip) instead of treating a failed run as zero tok/s.
- At runtime, only select graph-backed plans when the cache says graphs worked for that model and `(pp, tg)` bucket.

## Multi-vendor architecture (NVIDIA / AMD / others)

- Split **vendor plugins** or registration tables: Intel code owns Intel case generation and may **ignore** cache entries tagged for other vendors on Intel-only builds.
- Reserve JSON fields: `vendor`, `platform_tags`, `min_driver` so external contributors can add cases without forking core selection logic.
- Document extension points in README when the hook API exists.

## Power and mobility

- Detect **AC vs battery** (platform-specific); store power state in cache key or separate cache sections.
- **Power-cost** strategy: optimize energy (joules per token) or cap power when on battery, even if tok/s is lower on AC-tuned entries.
- Optional bench pass that records power samples alongside tok/s (needs stable OS APIs).

## Tooling

- `llama-bench --device tweak` parity checks vs `record` (leader + expected tok/s).
- Harden in-process `llama_bench` capture in `record` (backend reset between cases, stdout isolation).
- Schema version bumps and a small migration helper for cache files.
- CI smoke: one model, one pp, mocked or skipped backends when hardware absent.
