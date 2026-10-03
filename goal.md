# Project goal: RX 7900 XTX Vulkan MTP optimization

Status: initial benchmark runner and IQ3_S MTP baseline are complete; Vulkan optimization work has not started.
Specification date: 2026-10-01.

## 1. Read this first

Build a maintainable optimization patchset on top of `ggml-org/llama.cpp`, focused on faster **single-user MTP inference through llama-server on one AMD Radeon RX 7900 XTX** using the user's primary IQ3_S GGUF. Improve actual final output throughput and interactive latency while preserving inference correctness.

This is an MTP-only performance project. Target-only generation is not a performance goal, acceptance gate, or required benchmark suite. It may be used temporarily to diagnose a specific issue when necessary. Accelerating shared target-model operations is in scope when it improves MTP serving performance.

Keep reusable device, quantization, and shape optimizations independent of model names. Isolate architecture-specific graph changes and fused operations so a future model with a different architecture can be added without redesigning the project. Preserve upstream fallback for unsupported operations and configurations.

## 2. Fixed targets

| Item | Agreed target |
|---|---|
| Upstream | `https://github.com/ggml-org/llama.cpp`, tracking `master` |
| GPU | One Radeon RX 7900 XTX, Navi 31 / RDNA 3, nominal 24 GB VRAM |
| Backend | Vulkan |
| OS and driver | Fedora Linux 44 KDE, kernel `7.2.8-200.fc44.x86_64`, Mesa RADV `26.2.3` |
| Serving | `llama-server` |
| Concurrency | One user, one active request, one server slot |
| Speculation | Embedded MTP, verified on each real GGUF and pinned upstream revision |
| Primary model | `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` |
| Primary allocated context | `200000` tokens |
| Benchmark populated depths | `4096`, `16384`, `32768`, `65536`, `98304` tokens |

The primary model filename is an exact user-supplied identifier. Preserve its spelling. Do not silently substitute a differently named Qwen model. The secondary Q4_K_M model is out of scope and must not be used as a validation substitute. Do not infer layer count, hidden size, attention layout, training context limit, or tensor quantization from a filename or earlier conversational descriptions.

The primary file occupies `12120016896` bytes and has SHA-256 `9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5`. Its verified GGUF architecture key is `qwen35`, it contains 866 tensors, and its metadata declares `qwen35.nextn_predict_layers`. This establishes that the artifact contains MTP metadata; real draft/verify activation must still be verified in the baseline. Memory telemetry is useful, but fitting this model is not a current problem to solve. Do not make speculative memory constraints the center of the project.

Fedora Linux/RADV is the confirmed reference environment. Windows Vulkan remains possible, but driver-dependent tuning and benchmark results must be validated separately. Do not change the backend to HIP, CUDA, or another runtime without a new decision.

## 3. Real usage and priorities

The user normally has short chat conversations, sometimes uses longer contexts for coding, and wants large context capacity available. The server is launched with the large context setting even when only a small part is populated.

Therefore keep the allocated context fixed at the real deployment value in every scored test. A 4k benchmark means approximately 4k actual prompt tokens inside a server requested with `-c 200000`, which upstream aligns to an effective 200192-token context. It does not mean launching with `-c 4096`.

Priority order:

1. MTP final output speed and responsiveness at 4k and 16k populated context.
2. MTP performance at 32k.
3. Validation of behavior and performance at 64k and 96k.
4. Prompt ingestion and end-to-end request latency throughout this range.

No benchmark in the agreed suite should populate context beyond 98304 tokens. A gain below 96k might carry over to a larger context, but that is an expectation, not a proven result. Testing 128k or 200k populated context is outside the current task. Startup at the configured capacity remains part of normal operation.

How the user starts this model currently:
```
  /home/tmatkovic/llama.cpp/vulkan/llama-0.5.0-dev-11380/llama-server \
  -m /home/tmatkovic/.lmstudio/models/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  --mmproj /home/tmatkovic/.lmstudio/models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/mmproj-Qwen3.8-27B-BF16.gguf \
  --alias ukisaiswift1.5-27b-GSQ-RCO \
  --host 0.0.0.0 \
  --port 8080 \
  --api-key-file /home/tmatkovic/.config/llama.cpp/api-keys \
  --device Vulkan0 \
  --split-mode none \
  --n-gpu-layers all \
  --parallel 1 \
  --ctx-size 200000 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --spec-draft-n-min 0 \
  --threads 4 \
  --jinja \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --reasoning-effort xhigh
```

## 4. What success means

The main metric is **actual final output tokens per second with MTP enabled**, measured through the serving path. Draft tokens that are rejected are not output tokens. Acceptance percentage is explanatory, not the objective to maximize.

Success also includes:

- A manually runnable benchmark executable with documented presets, repetitions, depth selection, MTP sweeps, baseline comparison, and durable machine-readable results.
- A reproducible baseline on the real machine before kernel experimentation.
- A measured MTP configuration appropriate to each model and the hardware, selected for throughput and latency rather than acceptance alone.
- Correctness checks for affected kernels, recurrent/attention state where applicable, speculative acceptance, rollback, and output behavior.
- A small patchset that can be regularly rebased onto upstream master.
- A single optimized binary that automatically falls back for unsupported devices, tensor layouts, types, operations, and architectures.
- Evidence for every retained optimization: raw results, exact source state, runtime configuration, and a concise explanation of the gain.

Do not promise a fixed percentage improvement before profiling. Small repeatable gains are valuable. Model-specific optimizations are explicitly welcome, including specialized kernels and graph fusion, when measurements justify them.

## 5. Scope

In scope:

- Vulkan kernels, launch geometry, quantization paths, fusion, transfers, synchronization, command scheduling, and buffer reuse.
- Operations in both the target and MTP draft/verification paths.
- Runtime tuning of MTP draft length and supported related controls.
- Architecture-specific graph transformations that preserve the inference contract.
- Memory optimizations when they improve serving performance or enable a demonstrably better configuration.
- Profiling and small diagnostic tools that explain an observed bottleneck.
- Mixed tensor quantizations actually present in the two GGUF files.

Out of current scope:

- Standalone target-only performance optimization or a required target-only score.
- DFlash/DFlash2 or other speculative methods; keep extension points but do not implement them now.
- Multiple simultaneous users, batching across users, multi-GPU scaling, or serving fleets.
- The secondary Q4_K_M GGUF, including its inventory, benchmark, validation, and optimization.
- Requantizing weights, changing the model, or accepting output-quality loss to raise a score.
- Populated-context benchmarks above 96k.
- Implementing speculative details of an unreleased/future model architecture.
- A separate inference engine or a broad fork of llama.cpp internals.

## 6. Correctness and acceptance rules

Use upstream MTP on the same pinned revision as the semantic reference. Performance changes must not weaken verification, force draft acceptance, replace real acceptance with synthetic values, or alter sampling settings in a supposedly identical A/B comparison.

Floating-point kernels need justified numerical tolerances rather than universal bit equality. Compare affected operation outputs and relevant state against a reference. For deterministic end-to-end tests, compare tokens where reproducible and investigate divergences; token matching by itself is not sufficient proof. For stochastic modes, validate the algorithm and targeted numerical behavior without pretending one matching sample proves distribution equivalence.

No fixed minimum gain is required. Retain a change when it shows a repeatable useful improvement, passes relevant correctness checks, has no unexplained material regression in the agreed workloads, and has reasonable maintenance cost. Restrict dispatch to the beneficial regime when another model, shape, or depth would otherwise regress. Record unresolved tradeoffs for the user rather than hiding them in an average.

## 7. Initial work sequence

1. Confirm OS, driver, compiler, CPU/RAM, Vulkan features, model paths, and current serving arguments.
2. Pin a compatible upstream commit; verify the primary GGUF identity, metadata, tensor types, and embedded MTP support. Completed for the initial IQ3_S baseline.
3. Implement the benchmark runner and establish the IQ3_S MTP baseline. Completed at 4096 prompt tokens; see `progress.md`.
4. Sweep MTP settings using representative prompts, then select and freeze a configuration for kernel A/B tests.
5. Profile the actual MTP serving workload.
6. Add the optimization gates and the smallest necessary dispatch integration.
7. Implement one measured optimization at a time; validate and retain only supported gains.
8. Rebase regularly and refresh baselines after upstream or driver changes.

## 8. Instructions for a coding agent starting a session

Read `goal.md` and `architecture.md` first. Read `benchmark.md` before benchmark work or accepting performance changes. Read `howtobuild.md` for setup and repository maintenance. Read `progress.md` to see progress. Take it with a grain of salt...maybe it is outdated and not updated. Only the code is apsolute truth.

Inspect the real checkout, its instruction files, current branch, diff, and existing implementation before creating files. The companion documents describe the desired architecture and interface; they do not assert those components already exist. Preserve completed work and update documentation when implementation settles an open detail.

Use the smallest upstream hooks that permit a sound implementation. Keep new logic in project-owned files where practical; do not create empty future modules or large generic frameworks. Do not promote an experiment based on visual impressions of streaming speed. Do not relabel target-only results as MTP results.

## 9. Remaining facts to discover

The valid context handling and real draft/verify activation are confirmed for the initial 4096-token baseline. Optimal MTP draft length, cache types, batch settings, server timing semantics, and measured bottlenecks remain unknown. Discover and record these facts; they are not permission to alter the agreed goals.

## References

- [Upstream repository](https://github.com/ggml-org/llama.cpp)
- [Upstream server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

Upstream master changes. The checkout's source and binary help at the recorded commit govern implementation details.
