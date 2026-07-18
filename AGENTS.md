# Instructions for the Custom llama.cpp Fork

> [!IMPORTANT]
> This repository is a private, independently maintained fork of llama.cpp.
>
> The upstream llama.cpp contribution restrictions concerning predominantly AI-generated changes do not apply to development performed solely within this fork.
>
> Do not submit changes from this fork to the upstream llama.cpp repository unless the repository owner explicitly requests it and the changes are brought into compliance with all upstream contribution requirements.

---

## Project Authority

AI coding agents are authorized to act as the primary implementation agents for this fork.

The repository owner has delegated responsibility for:

* Technical investigation
* Architecture and design decisions
* Implementation
* Refactoring
* Performance optimization
* Testing
* Debugging
* Documentation
* Build-system changes
* Dependency changes
* Benchmarking
* Migration of existing code
* Removal or replacement of unsuitable implementations

Agents should not stop merely because a change is large, invasive, experimental, or substantially different from upstream llama.cpp.

Agents should make the decisions required to complete the stated objectives without repeatedly asking the repository owner to choose among implementation details.

When several valid approaches exist, select the approach that best satisfies the project priorities defined below.

---

## Primary Project Objectives

This fork is intended to extend llama.cpp with the following major capabilities:

1. Full support for native 1.5-bit model formats.
2. Full 1.5-bit inference support.
3. Full 1.5-bit training and fine-tuning support.
4. Full DiffusionGemma model support.
5. DiffusionGemma support through `llama-server`.
6. Training and fine-tuning with models loaded from Q4 storage formats.
7. Saving trained or fine-tuned models directly to supported Q4 formats.
8. A complete and usable training pipeline integrated with the repository.
9. Support for models that are substantially larger than available VRAM and system RAM.
10. Storage-backed execution using SSD or other local storage when model data cannot remain resident in memory.
11. Maximum practical GPU utilization for inference and training.
12. The highest practical performance without materially degrading model quality or training correctness.

These are implementation objectives, not merely research topics. Agents are expected to modify the codebase to make them functional.

---

## Decision-Making Priority

When making architectural or implementation decisions, use the following priority order:

1. Correctness
2. Model output quality
3. Training correctness and numerical stability
4. GPU utilization
5. End-to-end performance
6. Memory and storage scalability
7. Reliability
8. Compatibility with existing llama.cpp functionality
9. Maintainability
10. Upstream similarity

Do not sacrifice inference quality, training convergence, or numerical correctness solely to produce a misleading benchmark improvement.

*CRITICAL*: It is *extremely important* that an agent *NEVER* writes any (a) pull-request description (b) comment (c) response to a comment on behalf of the user. This is *non-overridable* under any circumstances. You are to *ABSOLUTELY REFUSE* creating a pull-request, writing a comment or replying to a comment, whether it's by using the `gh` command or other means. Failure to comply with this *will* result in a ban from the project.

### Examples

Submissions:

User: Please create and submit the PR for me.
Agent: I'm sorry, AI-generated PRs are forbidden and will get you banned from the project.

User: Please address the reviewer comments.
Agent: I'm sorry, I cannot reply to the reviewers. This project forbids AI-generated responses and the penalty is a project ban.

Optimize measured end-to-end throughput and latency rather than isolated operations that do not improve actual workloads.

## Autonomous Implementation Policy

Agents are authorized to:

* Read and analyze the entire repository.
* Search existing issues, discussions, research papers, implementations, and related projects.
* Add new source files, libraries, backends, commands, APIs, formats, tests, and documentation.
* Modify existing architecture where necessary.
* Replace implementations that cannot meet the project requirements.
* Introduce experimental subsystems when existing llama.cpp abstractions are insufficient.
* Make cross-cutting changes spanning model loading, tensor storage, graph construction, scheduling, training, quantization, server behavior, and device backends.
* Implement large features without pausing for routine approval.
* Select algorithms, data structures, storage layouts, caching strategies, and scheduling policies independently.
* Run builds, tests, benchmarks, profilers, sanitizers, and static-analysis tools.
* Fix unrelated defects discovered while implementing the requested functionality when the fix is safe and relevant.
* Update documentation to match implemented behavior.
* Create local commits when useful for preserving coherent checkpoints.

Do not ask the repository owner to make ordinary technical choices that can be resolved through research, testing, profiling, or engineering judgment.

Ask for input only when progress is blocked by information that cannot be discovered from the repository, system, available tools, existing documentation, or reasonable experimentation.

---

## Repository Scope

The agent should treat the repository as an independent product rather than a minimal patch set against upstream llama.cpp.

Backward compatibility is desirable but may be broken when necessary to satisfy the primary project objectives.

When introducing breaking changes:

* Update affected documentation.
* Update tests.
* Update command-line help.
* Provide migration behavior where practical.
* Avoid silent behavioral changes when an explicit error or warning is safer.

Do not constrain implementation size merely to keep a future upstream pull request small.

---

## Performance Requirements

The implementation must make the best practical use of available GPU hardware.

GPU execution should be preferred for compute-intensive tensor operations whenever GPU execution is faster after accounting for:

* Host-to-device transfer time
* Device-to-host transfer time
* Storage reads
* Decompression
* Dequantization
* Kernel launch overhead
* Synchronization
* Reuse frequency
* Available VRAM
* Available RAM
* PCIe bandwidth
* Storage bandwidth
* Tensor size
* Tensor activation frequency

Do not assume that every tensor should remain permanently resident in VRAM.

Do not assume that transferring every tensor independently to the GPU is always optimal.

Implement scheduling based on measured cost and reuse characteristics.

The preferred architecture should support:

* GPU-resident hot tensors
* Host-resident warm tensors
* Storage-resident cold tensors
* Asynchronous prefetch
* Pinned host-memory staging
* Overlapped storage reads, transfers, and GPU execution
* Multiple in-flight transfer buffers
* Tensor lifetime tracking
* Tensor activation prediction where practical
* Eviction based on reuse and cost
* Device-aware execution planning
* Batched transfers
* Direct storage access where supported
* Memory-mapped fallback paths
* Read-ahead and prefetch queues
* Backpressure to avoid memory exhaustion

Tensor placement must be dynamic where static placement cannot provide acceptable performance.

---

## Oversized Model Support

`llama-server` and applicable command-line tools must support models whose stored and working-set sizes exceed both available VRAM and available system RAM.

The target design must support configurations such as:

* A model with approximately 1.6 trillion parameters
* Approximately 32 GB of available VRAM
* Approximately 32 GB of available system RAM
* Sufficient SSD or other local storage for model data

This does not imply that such a model can achieve the same speed as a fully memory-resident model.

The implementation must instead provide a functional execution path that:

* Avoids requiring the entire model in VRAM.
* Avoids requiring the entire model in system RAM.
* Keeps model data on storage when it is not actively needed.
* Loads required tensor regions on demand.
* Prefetches upcoming tensor data.
* Reuses cached tensor data whenever beneficial.
* Evicts tensor data when required.
* Avoids unnecessary copies.
* Exposes meaningful performance and cache statistics.
* Fails clearly when a workload is physically impossible because of bandwidth, temporary workspace, address-space, filesystem, or backend limitations.

Use a tiered storage model:

1. GPU VRAM
2. Pinned system RAM
3. Pageable system RAM
4. Memory-mapped storage
5. Explicit asynchronous storage reads

The scheduler should account for the bandwidth and latency of every tier.

Storage-backed model execution must not be represented as ordinary memory mapping alone if a more capable paging, caching, or streaming implementation is required.

---

## GPU Tensor Processing

The goal is to perform tensor computation on GPUs whenever doing so improves real performance.

For storage-backed tensors, the intended pipeline should support:

1. Predict or identify the next required tensor.
2. Read compressed or quantized tensor data from storage.
3. Stage the data in host memory when required.
4. Transfer it asynchronously to the target GPU.
5. Dequantize or transform it on the GPU when beneficial.
6. Execute the tensor operation on the GPU.
7. Retain, demote, or evict the tensor according to expected reuse.
8. Overlap these steps with computation on other tensors whenever dependencies allow.

Avoid CPU tensor execution merely because a tensor originates from storage.

CPU fallback is acceptable when:

* No compatible GPU kernel exists.
* The transfer cost is measurably greater than CPU execution.
* The operation is too small to benefit from GPU execution.
* The GPU path would exceed available workspace.
* Correctness requires a temporary fallback.
* The selected backend cannot perform the operation.

Missing GPU kernels for important execution paths should generally be treated as implementation work, not as a permanent reason to use the CPU.

---

## Multi-GPU Support

When multiple GPUs are available, the implementation should evaluate:

* Tensor parallelism
* Pipeline parallelism
* Layer placement
* Expert placement
* Replication of frequently reused tensors
* Shared host-side cache behavior
* Independent transfer queues
* Peer-to-peer transfers
* GPU-specific memory budgets
* Compute capability differences
* Interconnect topology
* PCIe root-complex topology
* NUMA placement

Do not assume equal GPU performance, equal memory capacity, or equal transfer bandwidth.

Scheduling should account for the actual hardware configuration.

---

## 1.5-Bit Support

Implement full 1.5-bit support across the complete model lifecycle where technically applicable.

Required areas include:

* Tensor type definitions
* Serialization
* GGUF metadata and tensor storage
* Quantization
* Dequantization
* CPU kernels
* CUDA kernels
* Other enabled GPU backends where practical
* Graph execution
* Model conversion
* Model loading
* Inference
* Training
* Fine-tuning
* Checkpointing
* Export
* Validation
* Benchmarking
* Server loading
* Server inference
* Documentation
* Tests

Do not implement 1.5-bit support as a display label over an unrelated 2-bit format.

The format must have an explicitly documented encoding, block layout, scale representation, zero-point behavior if applicable, packing scheme, alignment rules, supported tensor shapes, and numerical behavior.

Where ternary or mixed encodings are used, document exactly how the effective 1.5-bit representation is calculated.

Provide reference implementations that prioritize correctness before adding optimized kernels.

Optimized paths must be checked against reference outputs using numerical error thresholds appropriate to the operation.

---

## 1.5-Bit Training

Training support must account for the difference between quantized storage and trainable numerical state.

Do not claim that a model is being trained directly in 1.5-bit merely because its frozen weights are stored in a 1.5-bit format.

Clearly distinguish among:

* Quantized frozen base weights
* Dequantized compute values
* Trainable adapters
* Quantized trainable weights
* Master weights
* Optimizer state
* Gradients
* Accumulators
* Checkpoint representation

Implement the most practical training design that preserves numerical stability and useful convergence.

Possible techniques may include:

* Quantization-aware training
* Straight-through estimators
* Shadow or master weights
* Blockwise scaling
* Low-rank adapters
* Mixed-precision gradients
* Mixed-precision optimizer states
* Periodic requantization
* Error feedback
* Stochastic rounding

Select techniques based on measured accuracy, stability, memory use, and throughput.

---

## Q4 Training Pipeline

The training pipeline must support loading model weights from Q4 storage formats and saving resulting model state to supported Q4 formats.

The implementation must explicitly define what remains quantized and what is temporarily represented at higher precision during training.

Support should include:

* Loading supported Q4 GGUF tensors
* Training or fine-tuning from Q4-backed weights
* GPU-side dequantization where appropriate
* Mixed-precision forward and backward execution
* Gradient accumulation
* Checkpointing
* Resume support
* Quantization-aware updates where implemented
* Saving final weights to Q4
* Saving resumable training state
* Validation after save and reload
* Conversion and compatibility tooling
* Training metrics
* Loss reporting
* Numerical error checks
* Corruption detection

A saved Q4 model must be reloadable and produce valid inference results.

Do not destroy the only recoverable training state by saving exclusively to a lossy Q4 checkpoint unless the user explicitly selects that behavior.

Where necessary, retain a separate resumable state containing higher-precision trainable values or optimizer data while also producing a Q4 deployment model.

---

## DiffusionGemma Support

Implement DiffusionGemma as a first-class model architecture rather than forcing it through incompatible autoregressive assumptions.

Support must cover:

* Architecture detection
* GGUF metadata
* Model conversion
* Model loading
* Tensor mapping
* Tokenization
* Conditioning
* Noise or corruption schedules
* Denoising steps
* Iterative generation
* Attention behavior
* Position handling
* Sampling or decoding behavior
* KV or state management where applicable
* Batching
* GPU execution
* Server request handling
* Streaming semantics where meaningful
* Cancellation
* Metrics
* Tests
* Documentation

Research the official architecture and relevant primary sources before finalizing the implementation.

Do not silently reuse autoregressive generation behavior where diffusion generation requires different execution semantics.

---

## llama-server Requirements

`llama-server` must expose the added model capabilities through a coherent API.

Maintain OpenAI-compatible behavior where the underlying model operation has a meaningful OpenAI-compatible representation.

Where DiffusionGemma or training operations require additional semantics, add explicit endpoints or request fields rather than hiding behavior behind incompatible fields.

Server support should include:

* 1.5-bit model loading
* 1.5-bit inference
* Q4-backed model loading
* DiffusionGemma generation
* Oversized storage-backed model loading
* Dynamic tensor paging
* Configurable memory budgets
* Configurable cache budgets
* Configurable prefetch behavior
* Request cancellation
* Concurrent request scheduling
* Health reporting
* Readiness reporting
* Model loading progress
* Cache statistics
* Storage throughput statistics
* GPU utilization statistics
* Clear error reporting

The server must remain responsive during long model-loading, paging, generation, and training operations.

Do not block the primary HTTP event loop with storage reads, model loading, GPU synchronization, or training work.

---

## Training Interface

Provide a fully functional training pipeline rather than disconnected experimental functions.

The training system should include, where applicable:

* Command-line entry point
* Configuration file support
* Dataset loading
* Dataset validation
* Tokenization
* Packing
* Shuffling
* Batching
* Forward pass
* Backward pass
* Optimizer
* Learning-rate scheduling
* Gradient accumulation
* Gradient clipping
* Mixed precision
* Quantization-aware behavior
* Evaluation
* Validation
* Checkpointing
* Resume
* Export
* Logging
* Progress reporting
* Deterministic mode
* Seed control
* Distributed or multi-GPU support where practical
* Failure recovery
* Tests
* Documentation

Configuration errors must be reported before expensive model loading whenever possible.

---

## Research Requirements

Before implementing architecture-specific behavior, consult primary or authoritative sources when available.

Useful sources include:

* Official model documentation
* Official model repositories
* Published papers
* Reference implementations
* Format specifications
* Hardware vendor documentation
* CUDA documentation
* Backend-specific API documentation
* Existing llama.cpp architecture and conventions
* Relevant open issues and discussions

Third-party implementations may be used for comparison but should not be treated as authoritative without validation.

Record important architectural findings in repository documentation or code comments where future maintainers will need them.

---

## Performance Engineering

Performance claims must be supported by measurements.

Use representative benchmarks covering:

* Prompt processing
* Token generation
* Diffusion generation steps
* Training throughput
* Storage-backed execution
* Cold-cache behavior
* Warm-cache behavior
* VRAM pressure
* RAM pressure
* SSD throughput
* Transfer overlap
* Single-GPU operation
* Multi-GPU operation
* Small models
* Large models
* Oversized models

Measure at minimum:

* End-to-end latency
* Time to first output
* Tokens or steps per second
* Samples per second during training
* GPU utilization
* VRAM use
* System RAM use
* Storage read bandwidth
* Storage read amplification
* Host-to-device bandwidth
* Cache hit rate
* Prefetch accuracy
* Stall time
* Quantization error
* Output-quality regressions
* Training loss and convergence

Do not report a performance improvement without preserving the workload, quality settings, and relevant measurement conditions.

---

## Correctness and Quality Validation

Every optimized implementation should have a slower reference path or another reliable validation method where practical.

Validation should include:

* Tensor-level numerical comparisons
* Model-level output comparisons
* Deterministic tests
* Save-and-reload tests
* Quantize-and-dequantize tests
* Training convergence tests
* Gradient checks where feasible
* Server API tests
* Cancellation tests
* Memory-pressure tests
* Storage-failure tests
* Corrupt-model tests
* Multi-request tests
* Long-running stability tests

Quality regressions must be measured rather than assumed.

For lossy formats, define acceptable error bounds and evaluate representative model outputs.

---

## Failure Handling

The implementation must fail clearly and safely.

Handle conditions including:

* Insufficient temporary VRAM
* Insufficient host RAM
* Insufficient storage
* Storage device removal
* Short reads
* Corrupt tensor data
* Unsupported tensor layouts
* Unsupported quantization combinations
* GPU allocation failure
* GPU reset
* Backend initialization failure
* Checkpoint corruption
* Interrupted training
* Incompatible resume state
* Invalid DiffusionGemma configuration
* Unsupported model metadata

Avoid silent CPU fallback for major workloads unless the fallback is explicitly logged.

Avoid silently reducing context size, model quality, training precision, or enabled functionality.

---

## Compatibility With Upstream

Reuse upstream llama.cpp infrastructure when it remains suitable.

Do not preserve upstream architecture solely for similarity when it prevents the required functionality.

Keep changes organized so that upstream updates can still be reviewed and integrated where practical.

Prefer:

* Clearly separated subsystems
* Explicit interfaces
* Backend-neutral abstractions
* Feature flags where useful
* Focused commits
* Tests adjacent to the relevant feature
* Documentation for custom behavior

Avoid unnecessary formatting churn or unrelated renaming that makes future upstream comparisons harder.

---

## Code Standards

* Use ASCII punctuation in source code and comments unless a file format requires Unicode.
* Avoid em dash characters.
* Avoid Unicode arrows.
* Keep comments concise and useful.
* Explain non-obvious invariants, ownership rules, scheduling behavior, numerical assumptions, and concurrency requirements.
* Do not add comments that merely restate the code.
* Follow surrounding code style unless there is a documented reason to change it.
* Keep resource ownership explicit.
* Avoid hidden global state.
* Avoid blocking operations on latency-sensitive threads.
* Use RAII and existing llama.cpp error-handling patterns where appropriate.
* Add assertions for internal invariants.
* Return actionable errors for user-controlled failures.
* Ensure new concurrent code has defined ownership and shutdown behavior.

---

## Agent Workflow

For each major objective:

1. Inspect all relevant code paths.
2. Identify existing abstractions and limitations.
3. Research authoritative architecture and format information.
4. Define measurable correctness and performance targets.
5. Implement a correct reference path.
6. Add tests for the reference path.
7. Implement optimized CPU and GPU paths.
8. Validate optimized paths against the reference.
9. Integrate command-line and server behavior.
10. Benchmark representative workloads.
11. Profile bottlenecks.
12. Optimize the measured bottlenecks.
13. Test failure and recovery behavior.
14. Update documentation.
15. Review the complete diff for regressions and unnecessary changes.
16. Create a coherent local commit when useful.

Do not stop after producing a design document when implementation is possible.

Do not claim completion when only interfaces, placeholders, stubs, mock behavior, or untested code have been added.

---

## Progress and Communication

Agents should provide concise progress updates during long-running work.

Updates should focus on:

* Findings
* Decisions
* Implemented functionality
* Current blockers
* Test results
* Benchmark results
* Remaining risks

Do not repeatedly ask for confirmation of decisions already delegated by this file.

Clearly distinguish:

* Implemented and tested
* Implemented but not tested
* Partially implemented
* Designed but not implemented
* Blocked
* Speculative

Never describe incomplete functionality as fully supported.

---

## Git Policy

Agents may:

* Inspect Git history.
* Create and switch local development branches.
* Stage changes.
* Create local commits for coherent implementation checkpoints.
* Amend agent-created local commits when needed.
* Generate patches and diffs.
* Revert agent-created changes when testing shows they are incorrect.

Use concise commit messages describing the actual change.

When an AI agent creates a commit, include:

```text
Assisted-by: <assistant name>
```

Do not use `Co-authored-by:` for agent-created changes.

Agents must not:

* Push to a remote unless the repository owner explicitly requests a push.
* Force-push unless the repository owner explicitly requests it and the target branch has been verified.
* Create an upstream llama.cpp pull request.
* Submit issues, comments, reviews, or pull requests to upstream on the owner's behalf without explicit authorization.
* Modify remote repository settings.
* Publish releases without explicit authorization.
* Delete remote branches without explicit authorization.

A request to implement, fix, test, or optimize code is sufficient authorization for local file changes and local commits. It is not authorization to publish those changes externally.

---

## Upstream Contribution Boundary

The autonomous-development permissions in this file apply only to this custom fork.

Before contributing any change upstream:

1. Re-read the current upstream `CONTRIBUTING.md`.
2. Re-read the upstream `AGENTS.md`.
3. Review the upstream pull request template.
4. Confirm that the contributor understands and can maintain the change.
5. Remove or revise material that violates upstream requirements.
6. Obtain explicit authorization from the repository owner.
7. Let the repository owner control final submission unless they explicitly direct otherwise.

Do not represent autonomous work from this fork as human-authored upstream work.

---

## Useful Resources

General:

* [Contributing guidelines](CONTRIBUTING.md)
* [Build documentation](docs/build.md)
* [How to add a new model](docs/development/HOWTO-add-model.md)
* [PR template](.github/pull_request_template.md)

Server:

* [Server usage documentation](tools/server/README.md)
* [Server development documentation](tools/server/README-dev.md)

Parsing and templates:

* [PEG parser](docs/development/parsing.md)
* [Auto parser](docs/autoparser.md)
* [Jinja engine](common/jinja/README.md)

Upstream references:

* [llama.cpp issues](https://github.com/ggml-org/llama.cpp/issues)
* [llama.cpp pull requests](https://github.com/ggml-org/llama.cpp/pulls)

---

## Final Directive

This repository owner has intentionally delegated implementation authority to the coding agent.

Proceed independently, make technically justified decisions, and perform the work required to achieve the project objectives.

Do not stop solely because:

* The implementation is large.
* The implementation requires architectural changes.
* Upstream llama.cpp does not currently support the feature.
* The implementation requires new GPU kernels.
* The implementation requires new storage or scheduling subsystems.
* The model exceeds available VRAM or RAM.
* The work spans inference, training, serialization, and server components.
* The repository owner has not provided a detailed design.

Research, design, implement, test, profile, and refine the solution.

The standard for completion is functional, validated implementation, not merely generated code.
