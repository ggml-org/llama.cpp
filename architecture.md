# Project architecture

Status: implementation specification, not a description of existing code.
Specification date: 2026-10-01. Read together with `goal.md`.

## 1. Architectural decisions

Maintain one llama.cpp fork and one production patch branch: `rx7900xtx-vulkan`. Track upstream `master` without adding project commits to the fork's `master`. Use short-lived experiment branches or worktrees. Both supported GGUFs use the same optimized binary.

The architecture has five responsibilities:

| Component | Responsibility | Must not own |
|---|---|---|
| Device capability layer | Recognize supported GPU/driver capabilities and legal kernel requirements | Model identification or speculation policy |
| Vulkan optimization layer | Kernel implementations, pipeline variants, legal shape/layout dispatch | Token sampling or GGUF filename matching |
| Model profile layer | Verified architecture descriptors and measured shape tuning | Duplicated device discovery |
| Model graph integration | Architecture-specific fusions and the minimal backend hint bridge when needed | General Vulkan backend behavior |
| MTP serving integration | Draft/verify scheduling optimizations and measurement counters | Quantization arithmetic |

The benchmark is an external client/process orchestrator. It does not participate in inference decisions.

Generic kernels must not depend on Qwen names or filenames. A model-specific kernel is allowed and belongs in its model module. A profile is a measured selection table, not a copy of an existing kernel.

## 2. Repository layout

The following tree is the agreed destination layout. Paths marked `[existing]` are upstream integration areas; inspect their actual contents at the pinned revision. Other paths are project-owned additions. Create them only when a real implementation needs them.

```text
llama.cpp/
  goal.md
  architecture.md
  benchmark.md
  howtobuild.md
  docs/rx7900xtx/
    status.md
    upstream-hooks.md
    decisions.md
    experiments/
      <experiment-id>.md
    models/
      swift-iq3s.inventory.json
      swift-q4km.inventory.json
  ggml/src/ggml-vulkan/
    ggml-vulkan.cpp                         [existing]
    CMakeLists.txt                          [existing]
    opt/
      CMakeLists.txt
      config.hpp
      config.cpp
      types.hpp
      integration.hpp
      integration.cpp
      device/
        capabilities.hpp
        capabilities.cpp
        rx7900xtx.hpp
        rx7900xtx.cpp
      dispatch/
        selector.hpp
        selector.cpp
        shape_rules.hpp
        shape_rules.cpp
      kernels/
        registry.hpp
        registry.cpp
        iq3s.hpp
        iq3s.cpp
        q4k.hpp
        q4k.cpp
      models/
        registry.hpp
        registry.cpp
        qwen38_27b/
          profile.hpp
          profile.cpp
          fused_ops.hpp
          fused_ops.cpp
      telemetry/
        counters.hpp
        counters.cpp
    vulkan-shaders/                         [existing shader integration area]
      rxopt/
        common/
          rdna3.glsl
          quant_helpers.glsl
        iq3s_gemv.comp
        q4k_gemv.comp
        models/qwen38_27b/
          <verified-fusion>.comp
  src/
    <upstream-model-graph-files>             [existing; identify in checkout]
    rxopt/
      model_descriptor.hpp
      model_descriptor.cpp
      graph_hints.hpp
      graph_hints.cpp
      models/qwen38_27b/
        graph_rules.hpp
        graph_rules.cpp
  common/
    <upstream-speculative-files>            [existing; identify in checkout]
    rxopt/
      mtp_policy.hpp
      mtp_policy.cpp
      mtp_metrics.hpp
      mtp_metrics.cpp
  tools/server/
    <upstream-server-files>                 [existing; identify in checkout]
    rxopt/
      metrics_adapter.hpp
      metrics_adapter.cpp
  tools/rx7900xtx-bench/
    pyproject.toml
    rx7900xtx_bench/
      __init__.py
      __main__.py
      cli.py
      config.py
      capabilities.py
      process.py
      server_client.py
      workloads.py
      tokenization.py
      metrics.py
      environment.py
      inventory.py
      runner.py
      statistics.py
      compare.py
      report.py
    presets/
      qwen38-iq3s.json
      qwen38-q4km.json
    prompts/
      manifest.json
      chat/
      code/
      long-context/
    schemas/
      preset-v1.schema.json
      result-v1.schema.json
    tests/
      test_metrics.py
      test_tokenization.py
      test_comparison.py
      test_process_lifecycle.py
    packaging/
      build_executable.py
  tests/rxopt/
    CMakeLists.txt
    test_dispatch.cpp
    test_quant_kernels.cpp
    test_fused_ops.cpp
    test_mtp_state.cpp
  bench-results/                            [local, ignored]
  build-control/                           [local, ignored]
  build-rx7900xtx/                          [local, ignored]
```

`qwen38_27b` is a provisional module identifier taken from the supplied model names. Confirm the actual architecture from GGUF metadata before implementing profile guards or graph rules. If the identifier is inaccurate, rename the module and presets deliberately and document the mapping; preserve user-supplied filenames in model identity records.

`<verified-fusion>` and other angle-bracket entries are placeholders, not literal filenames. Do not implement a presumed DeltaNet fusion just because an earlier conversation mentioned DeltaNet.

## 3. File ownership and change routing

| File or directory | What belongs here | When an agent should edit it |
|---|---|---|
| Root four documents | Goals, architecture contract, benchmark contract, operator instructions | A confirmed implementation detail changes the specification |
| `status.md` | Current base SHA, accepted patches, known limitations, next concrete work | End of meaningful development milestones |
| `upstream-hooks.md` | Every upstream edit, its purpose, owner, and rebase checks | A new integration hook is added or changed |
| `decisions.md` | Short dated design decisions with evidence and alternatives | A consequential design choice is resolved |
| `experiments/` | Hypothesis, exact patch/configuration, result paths, keep/drop decision | An experiment concludes |
| `docs/.../models/*.inventory.json` | Metadata, tensor shape/type/byte summaries, hashes, MTP tensors | A model is inspected or replaced |
| `opt/config.*` | Parse runtime gates once; immutable configuration | Adding or changing a project option |
| `opt/types.hpp` | Backend-local request descriptors and launch plans | A dispatch input or result contract changes |
| `opt/integration.*` | Narrow adapters around upstream pipeline/context structures | Upstream Vulkan integration changes |
| `opt/device/` | Device and driver identification, feature queries, legal capabilities | Device requirements or supported driver conditions change |
| `opt/dispatch/` | Validate layouts, match rules, choose safe variants | A new optimized operation/shape regime is supported |
| `opt/kernels/` | Pipeline creation/lookup and launch descriptions for generic kernels | Kernel variants or resource requirements change |
| `opt/models/.../profile.*` | Model guard and measured tuning entries | Measured model/shape tuning changes |
| `opt/models/.../fused_ops.*` | Launch logic for model-specific fused shaders | A verified architecture-specific backend operation is added |
| `vulkan-shaders/rxopt/` | Project-owned GLSL kernels and includes | Shader arithmetic or workgroup behavior changes |
| `src/rxopt/` | High-level identity extraction, graph patterns, metadata bridge | Model graph semantics must be specialized |
| `common/rxopt/` | MTP-only policy additions, bookkeeping, counters | Profiling justifies a speculation/scheduling change |
| `tools/server/rxopt/` | Translate counters into request-scoped output | Existing server output cannot supply required metrics |
| Benchmark package | External orchestration and reporting | Benchmark behavior changes |
| `tests/rxopt/` | Targeted checks of project dispatch, kernels, fusion, state | A performance change needs corresponding correctness coverage |

Do not place all C++ optimization code under the benchmark tool or all model logic inside `ggml-vulkan.cpp`. Do not move broad upstream implementations into project directories merely to make a diff look isolated.

## 4. Build and runtime gates

Introduce **project-owned** CMake option `GGML_VULKAN_RX7900XTX_OPT`, default `OFF`. It is not an existing upstream option. `ON` compiles and registers optimization modules; `OFF` excludes project optimization shaders, graph transforms, MTP policy changes, and hot-path telemetry hooks.

The flag must reach every modified target that needs it, including high-level graph or speculative targets. Guard headers and references consistently. The build should work with Vulkan disabled and with the project option disabled. Do not let a backend-only preprocessor definition leave high-level project code unguarded.

Planned environment controls, parsed once at initialization:

| Control | Values and default | Effect |
|---|---|---|
| `GGML_VULKAN_RX7900XTX_OPT` | `auto` / `off`; default `auto` in an ON build | `off` bypasses all project inference changes |
| `GGML_VULKAN_RX7900XTX_MODEL_OPT` | `auto` / `off`; default `auto` | Disable model profiles and graph specialization while retaining eligible generic optimizations |
| `GGML_VULKAN_RX7900XTX_TRACE` | `0` / `1`; default `0` | Diagnostic selection/fallback information outside benchmark scoring |

These controls are planned interfaces, not currently usable upstream settings. Runtime-off must also reach project graph and MTP policy hooks before contexts are constructed. Avoid changing gates halfway through a request. No force-on option should bypass correctness guards.

MTP remains enabled in the OFF control build: this gate disables **our patchset**, not upstream speculation. Same-commit ON/OFF builds are the primary performance comparison. A separate clean upstream build checks that project-OFF behavior actually preserves upstream semantics.

## 5. Dispatch contract

`opt/types.hpp` should define small backend-local structures with equivalent information to:

- `device_caps`: vendor/device identifiers, driver identity/version, relevant Vulkan extensions/features, subgroup limits, alignment and storage limits.
- `op_request`: operation, tensor type, dimensions, strides, alignment, buffer bounds, batch/ubatch, operation phase when safely available, and optional validated model/profile ID.
- `launch_plan`: kernel variant ID, specialization constants, workgroup dimensions, scratch requirements, and a reason/identifier useful for diagnostics.

The selector returns either a fully legal plan or “no optimized plan.” Integration then invokes the existing upstream path unchanged.

Selection order:

1. Check compile/runtime enablement.
2. Validate supported GPU and required capabilities. Use Vulkan device identifiers and features, not only a display-name substring. Start eligibility with the verified RX 7900 XTX; extend to other RDNA3 devices only after validation.
3. Validate operation, type, dimensions, strides, alignment, and memory requirements.
4. Apply a validated model-specific rule when enabled and relevant.
5. Otherwise apply a measured generic shape/type rule.
6. If no legal rule matches, return upstream fallback.

Cache reusable pipeline/selection state in the backend context, using owned lifetime. No per-token GGUF parsing, environment parsing, driver discovery, or string matching. New shader or allocation failure must not leave a half-mutated operation that then attempts fallback. Select and validate first; launch after all required resources exist.

## 6. Model identity and the graph/backend boundary

GGML backend operations do not automatically know a high-level llama model architecture. Do not assume a convenient `model` argument exists at a Vulkan dispatch call.

Prefer shape/type-only optimizations when their semantics are generic. A shape match alone must never authorize an architecture-specific fusion.

When specialization needs model identity:

1. Extract a descriptor in `src/rxopt/model_descriptor.*` from verified loader metadata and relevant graph properties. Include architecture/version traits and tensor signatures necessary for that optimization.
2. Validate graph patterns in `src/rxopt/models/.../graph_rules.*`.
3. First inspect upstream facilities for stable graph annotations or context metadata. Reuse an appropriate facility if available.
4. If none is sufficient, introduce one narrow private bridge in `graph_hints.*` and `opt/integration.*`, documenting any required upstream type/API changes. Keep it lifetime-safe, scoped to a model/context, and usable with multiple loaded contexts even though serving concurrency is one.
5. Communicate an opaque validated profile/operation hint, not a public llama model object inside generic ggml headers. Require the backend to revalidate layout and capabilities.

Never use tensor display names, pointer-global maps without lifecycle handling, filename substring checks, or dimensions alone as proof of a model-specific semantic pattern. File SHA-256 identifies a benchmark artifact; it need not lock a generic architecture profile to one quantization if its guards validate other compatible files.

If the bridge cannot be implemented safely with a small patch, keep the first version shape-based and defer that fusion. Record the unresolved integration decision rather than guessing an upstream API.

## 7. Model-specific optimization rules

Profiles select generic variants using measured tuples such as tensor type, matrix dimensions, batch regime, layout, and operation role when verified. Every entry needs a benchmark reference and complete applicability conditions.

Specialized fused shaders belong under `vulkan-shaders/rxopt/models/<model-id>/`; launch code belongs under `opt/models/<model-id>/`. High-level graph recognition belongs under `src/rxopt/models/<model-id>/`. Shared mathematical helpers move to generic directories only when semantics are actually shared.

Fusion must account for tensor aliasing, graph dependencies, masks, numerical precision, and state updates. Recurrent state and speculative rollback are particularly important if the inspected model has those features. Never skip state restoration to improve timings.

For a future Qwen4 or other architecture, add a new validated descriptor/profile and graph module only after inspecting the real model. Reuse generic device/quant kernels when legal. No current module should require every future model to have the same layers, state, or MTP implementation.

## 8. MTP integration

First use upstream MTP unchanged. Do not create a custom drafter merely to organize files. Add `common/rxopt/` files only if measured scheduling or bookkeeping work needs them.

Optimize drafting, verification, state handling, and shared target computations as one serving pipeline. Preserve upstream acceptance semantics, sampling, draft rejection, and state rollback. `n-max` tuning is a configuration experiment, not a kernel implementation change.

Counters must be request-scoped or obtained by reliable deltas. Drafted and accepted counts need documented definitions. Prefer existing structured timings/metrics. Add a narrow adapter only when needed; make it negligible overhead or optional. GPU phase timing requires proper GPU timestamps; CPU wall time around an asynchronous dispatch is not GPU execution time.

## 9. Shader build integration

Integrate into the existing Vulkan shader generator and CMake dependency chain. Upstream currently uses `ggml/src/ggml-vulkan/vulkan-shaders/` and generated shader headers. Nested project shader directories therefore require explicit discovery/registration and dependency tracking; simply placing a `.comp` file in `rxopt/` is insufficient.

Extend the smallest necessary generator/CMake hooks to register project shader sources only when the gate is ON. Track included GLSL files so edits cause regeneration. Keep generated SPIR-V and headers in the build directory. Use unique `rxopt_` variant identifiers and avoid collisions with upstream symbols. Do not implement a second independent shader compilation system.

## 10. Verification, maintenance, and change discipline

For each accepted change, cover the affected operation/types/shapes and edge conditions, unsupported-case fallback, numerical tolerances, and relevant MTP state behavior. Use upstream backend tests where practical and project tests for gaps. Add meaningful benchmark-client tests for token counts, partial SSE frames, error handling, metric semantics, and owned-process cleanup.

Integration hooks must stay small. Record their exact locations and rebase validation in `upstream-hooks.md`. Reuse upstream buffer and pipeline ownership. Avoid global state, per-token allocation, public API churn, and broad refactors unrelated to a measured gain.

Recommended patch sequence: benchmark/inventory; opt gates and dispatch adapter; first generic kernel; model profile; model graph fusion only if justified; MTP scheduling only if justified. Each patch should build and be reviewable independently.

Keep raw benchmark artifacts outside normal source commits. Commit small result summaries, reproducible configurations, experiment notes, and selected baselines when useful. Never commit model weights or user chat data.

## References

- [Vulkan build integration](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/CMakeLists.txt)
- [Upstream speculative implementation](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp)
- [Upstream server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

References identify integration areas. Inspect the pinned checkout before relying on a signature or filename; moving upstream files does not change component ownership in this design.
