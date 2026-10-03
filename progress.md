# RX 7900 XTX Vulkan MTP progress

Status: initial benchmark baseline complete. No Vulkan optimization code has been added.

## Confirmed environment

- Source revision: `0f0796f9076f3ebd3f49fa9be74598b33c42aada`
- Server: `0.5.0-dev` build `11380`, GCC `16.2.1`
- GPU: `Vulkan0`, AMD Radeon RX 7900 XTX, RADV NAVI31
- Model: `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`
- Model SHA-256: `9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5`
- Requested context: 200000 tokens
- Effective context: 200192 tokens after upstream 256-token alignment

## Initial MTP baseline

Run directory: `bench-results/20261003T182056Z-iq3s-quick-baseline/` (local and ignored)

- Workload: project-authored chat fixture, 4096 prompt tokens, 512 output tokens
- Mode: fresh requests with `--no-cache-prompt`
- Server: one loopback slot, `Vulkan0`, all GPU layers, Q8_0 K/V cache, Flash Attention, `draft-mtp`, `n-max=4`
- Warmup: 1 request, excluded from scores
- Measured requests: 5, all completed with 512 final output tokens
- MTP: active; startup created the MTP draft context and each request reported draft acceptance

| Metric | Result |
|---|---:|
| Client end-to-end output TPS, mean | 40.87 |
| Client end-to-end output TPS, median | 41.97 |
| Client delivery TPS, mean | 67.06 |
| Client delivery TPS, median | 69.54 |
| Client TTFT, mean | 4829.09 ms |
| Server predicted TPS, mean | 66.94 |
| Server predicted TPS, median | 69.41 |
| Draft proposed | 3634 |
| Draft accepted | 1644 |
| Weighted draft acceptance | 45.24% |

`server_predicted_tps_unvalidated` is retained as raw server timing. Its exact coverage of the complete MTP draft/verify/commit loop still needs source validation before it becomes the authoritative score.

The slot-selection log can report LCP similarity on later requests. This is not prefix reuse in this run: `--no-cache-prompt` disables reuse of prior prompt evaluation, and every measured request evaluated all 4096 prompt tokens.

## Benchmark runner status

Implemented and tested:

- owned loopback server launch, readiness, log capture, and owned-process cleanup
- pinned IQ3_S server configuration and resolved argv dry run
- model hash, device inventory, model path, context, slot count, and MTP startup checks
- chat-template prompt construction with exact token depth and token hash
- warmup, five measured fresh requests, raw responses, manifests, JSONL, CSV, JSON, and Markdown report
- authoritative final output token count, client timings, raw server timings, and speculative counter deltas
- unit tests for configuration, metrics, process cleanup, report generation, context alignment, SSE parsing, and prompt suffix handling

## Next work

1. Validate the server timing field against the pinned source and decide which final-output TPS is authoritative.
2. Run the MTP `n-max` sweep at 4096 and 16384 tokens, then freeze the selected configuration.
3. Profile the measured serving path before proposing any Vulkan kernel or graph change.
4. Add the smallest measured optimization and compare same-commit control and optimized builds.
