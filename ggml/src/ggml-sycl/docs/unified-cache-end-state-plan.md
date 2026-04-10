# Unified Cache End-State Plan

Status: Proposed
Date: 2026-04-08
Epic: `llama.cpp-mubmt`

## Summary

Yes, this requires shared `ggml` changes.

The current SYCL backend has enough pieces to prove the direction, but not enough to satisfy the intended contract:

- `unified_cache` is not yet the only owner of SYCL memory
- `ggml` still assumes many buffers have a stable contiguous base address
- model load still allocates pinned host buffers directly in some paths
- long-lived tensor state still mixes raw pointers and partial handle indirection
- temporary buffers still allocate outside unified-cache in hot paths
- the scheduler and loader do not treat event chaining as the primary synchronization model

The target architecture is:

- plan first
- allocate once
- load directly into final placement
- resolve storage through handles or leases
- route execution based on resolved residency
- chain work with events rather than blocking waits
- reuse pre-allocated memory during PP/TG instead of allocating or freeing

This plan is optimized for:

- automatic memory placement under limited VRAM
- predictable PP/TG speed
- large-model load success, including GPT-OSS 120B class models
- multi-GPU / TP correctness without unnecessary synchronizations

## Design Invariants

These invariants define the end state.

### 1. Single memory owner

`unified_cache` owns all SYCL-facing memory used by inference:

- VRAM allocations
- pinned host allocations
- persistent scratch
- staging buffers
- KV allocations
- TP / tensor-split / multi-GPU helper buffers
- upload rings and host loader staging
- optional shared-USM debug or interoperability buffers, if any remain

Direct `sycl::malloc_*`, `sycl::free`, or ad hoc host/device pools are not allowed outside unified-cache internals.

### 2. Planner-authoritative placement

The planner decides placement before model load:

- which weights live in VRAM
- which weights live in pinned host memory
- per-device budgets for multi-GPU / TP
- KV reservation
- persistent scratch reservation
- loader and staging budgets

Model load then materializes the plan directly into its final tier. Runtime does not invent a second placement policy.

### 3. Relocatable storage

Long-lived tensor state does not own physical pointers as the source of truth.

Instead, code stores:

- a handle for long-lived storage
- a lease for temporary storage
- metadata describing tier, layout, readiness event, and device ownership

Raw pointers are resolved only at the edge where a kernel or memcpy needs an address.

### 4. No runtime malloc/free in steady-state inference

After model load and warmup:

- PP and TG do not allocate new SYCL memory
- PP and TG do not free SYCL memory back to the driver
- temporary release means "return to unified-cache-managed pool or zone"
- runtime may move or evict data between pre-allocated pools, but it does not create new pools

### 5. Residency-driven execution

Execution follows residency:

- device-resident data executes on GPU
- host-planned data executes on CPU
- cross-tier transitions happen at explicit boundaries, not by surprise inside a hot kernel path

The system must not revive dense runtime weight streaming as a fallback.

### 6. Event chaining over waits

The default synchronization rule is:

- queue/stream work emits an event
- downstream work consumes that event through `depends_on` or backend event wait APIs
- full blocking `wait()` or `synchronize()` is only allowed at explicit graph / API boundaries, shutdown, or fatal recovery

This matters for:

- CPU/GPU split execution
- multi-GPU / TP staging
- load-time upload rings
- graph replay / graph reuse

## Why Shared `ggml` Changes Are Required

The current shared backend contract is still biased toward stable contiguous storage.

### Current core assumptions that conflict with the target

1. `ggml_backend_tensor_alloc()` stores a concrete address in `tensor->data` and asserts it lies inside one stable buffer base.

Source:
- [ggml/src/ggml-backend.cpp](/Apps/llama.cpp/ggml/src/ggml-backend.cpp#L2284)

2. View tensors derive their address from `view_src->data + view_offs`, which assumes stable physical backing.

Source:
- [ggml/src/ggml-backend.cpp](/Apps/llama.cpp/ggml/src/ggml-backend.cpp#L2279)

3. The backend buffer interface treats `get_base()` as a fundamental access path for normal buffers.

Source:
- [ggml/src/ggml-backend-impl.h](/Apps/llama.cpp/ggml/src/ggml-backend-impl.h#L41)

4. The model loader allocates explicit host buffers for async uploads through the device host buffer type and expects a base pointer from each buffer.

Source:
- [src/llama-model-loader.cpp](/Apps/llama.cpp/src/llama-model-loader.cpp#L1225)

These assumptions are acceptable for simple backends with fixed physical storage. They are not sufficient for relocatable residency with unified-cache-owned host and device arenas.

### Required shared-core outcomes

Shared `ggml` must support the following:

1. Backend-managed relocatable tensor storage
- A backend must be able to allocate a tensor into logical storage without making `tensor->data` the long-lived source of truth.
- Views must be expressible relative to logical storage, not only a stable physical pointer.

2. Buffer capabilities that describe storage behavior
- Whether a buffer has a stable contiguous base
- Whether storage is relocatable
- Whether host access is direct or must go through backend callbacks

3. Loader-owned staging must become backend-owned staging
- The loader should request upload staging from the backend or device, not allocate raw host buffers by itself.
- Async uploads should use backend-provided events and staging leases.

4. Scheduler split execution must favor event chaining
- Shared scheduler logic should not force more synchronizations than required between split backends.
- Existing backend event APIs should be used aggressively before falling back to blocking synchronization.

`ggml` already has async and event support in the backend API:
- [ggml/include/ggml-backend.h](/Apps/llama.cpp/ggml/include/ggml-backend.h#L87)
- [ggml/src/ggml-backend-impl.h](/Apps/llama.cpp/ggml/src/ggml-backend-impl.h#L112)

The work is to make the scheduler and allocation path actually rely on those capabilities instead of falling back to stable-pointer assumptions.

## End-State Architecture

### 1. Unified-cache memory classes

Unified-cache should expose explicit allocation roles:

- device weight arena
- host weight arena
- KV arena
- compute scratch arena
- oneDNN scratch arena
- staging arena / staging ring
- persistent TG helper arena
- TP / multi-GPU helper arena
- debug / profiling arena

Each allocation returns either:

- a long-lived handle, or
- a scoped lease

Both must answer:

- current pointer
- current tier
- current layout
- owning device
- readiness event

### 2. Per-device VRAM arenas

Each GPU gets a pre-allocated VRAM arena with fixed or planned zones:

- compute scratch
- KV
- oneDNN scratch
- reusable temporary storage
- weights

The planner decides zone sizes and placement order.

### 3. Shared host-pinned arena

Pinned host memory is also unified-cache-owned and planned.

It should support:

- weight storage
- host KV spill if enabled
- upload staging
- CPU-dispatch output staging
- TP / peer staging
- scratch reuse

The host arena must not devolve into one giant ad hoc `sycl::malloc_host` request for a single logical buffer.

### 4. Residency handles and leases

Long-lived tensors use handles.
Temporary storage uses leases.

Both are resolved through one source of truth. The current partial handle system in the SYCL backend already points in this direction:
- [ggml/src/ggml-sycl/mem-handle.cpp](/Apps/llama.cpp/ggml/src/ggml-sycl/mem-handle.cpp#L28)
- [ggml/src/ggml-sycl/common.hpp](/Apps/llama.cpp/ggml/src/ggml-sycl/common.hpp#L1841)

The migration is incomplete because raw pointer fields still exist as compatibility shims.

### 5. Event graph

Every cross-stage transfer or async producer should surface a readiness event:

- load-time upload completion
- host-to-device promotion
- device-to-host spill
- TP peer transfer completion
- CPU-dispatch host result readiness
- multi-GPU reduce / merge staging completion

Downstream work consumes those events through:

- `queue.depends_on(...)` inside SYCL
- `ggml_backend_event_record()` / `ggml_backend_event_wait()` across backends

## Research-Based Design Choices

This section records the design choices that are worth making explicit before implementation.

### 1. Exact long-context support should use dynamic KV placement, not a monolithic KV slab

For exact inference at very large context lengths, the baseline design should be a dynamic KV system:

- paged or block-based KV allocation, or
- virtual-memory-backed contiguous KV if Level Zero interop proves practical

Reasoning:

- PagedAttention showed that fixed-size blocks solve the internal-fragmentation problem that comes from reserving maximum KV capacity up front.
- vAttention argues that keeping KV contiguous in virtual memory while dynamically mapping physical pages avoids the kernel and software overheads of explicit paging.
- FlexGen reinforces that in resource-constrained settings, exact inference should treat memory as a hierarchy and compute where data lives.

Current repository relevance:

- the SYCL backend already has paged-attention-aware kernels
- the SYCL backend already has a `kv_tier_manager`
- the SYCL backend already has a `kv_offload_manager`

Sources:

- vAttention: https://www.microsoft.com/en-us/research/publication/vattention-dynamic-memory-management-for-serving-llms-without-pagedattention/
- FlexGen: https://arxiv.org/abs/2303.06865

Local code:

- [ggml/src/ggml-sycl/kv-tier-manager.hpp](/Apps/llama.cpp/ggml/src/ggml-sycl/kv-tier-manager.hpp#L16)
- [ggml/src/ggml-sycl/kv-offload.hpp](/Apps/llama.cpp/ggml/src/ggml-sycl/kv-offload.hpp#L24)
- paged attention support appears in:
  [fattn-esimd-f16.hpp](/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-esimd-f16.hpp) and [fattn-v2-esimd.hpp](/Apps/llama.cpp/ggml/src/ggml-sycl/fattn-v2-esimd.hpp)

### 2. Baseline algorithm for this codebase: exact block-based KV with planner-driven tiering

The best baseline for this repo is not a fresh virtual-memory design first.
It is:

- exact block-based KV allocation
- planner-driven per-layer and per-device placement
- host spill for cold blocks
- event-chained prefetch and reuse

Why this is the right baseline:

- it fits the current codebase better because paged-attention kernels and KV offload structures already exist
- it preserves exact model behavior
- it gives a clear path to massive context lengths
- it works with limited VRAM and host-pinned spill
- it lets us integrate the planner and unified-cache before attempting Level Zero virtual-memory interop

### 3. Advanced algorithm to spike: Level Zero virtual-memory-backed KV

The most promising advanced path is a vAttention-style KV design backed by Level Zero virtual memory APIs.

Why it is worth spiking:

- it preserves contiguous virtual KV layout
- it avoids explicit block-table logic in kernels
- it may reduce kernel and framework complexity
- vAttention reports significant speedups over paged-attention variants in some workloads

Why it should be a spike first, not the immediate baseline:

- it requires Level Zero interop beneath SYCL
- it changes how the backend thinks about buffer ownership and mapping
- it must be validated against Intel GPU driver behavior, multi-GPU behavior, and graph replay interactions

Official API basis:

- Level Zero exposes virtual address reservation and mapping APIs in the core memory model
  https://oneapi-src.github.io/level-zero-spec/level-zero/1.9.0/index.html

### 4. Approximate long-context algorithms should stay optional, not baseline

Algorithms such as StreamingLLM, H2O, SnapKV, or KV quantization/compression can be valuable, but they should not define the baseline architecture for this epic.

Reasoning:

- some are approximate and can change model behavior
- some require model- or workload-specific tuning
- they complicate correctness and benchmarking

They should be treated as explicit opt-in features after the exact memory architecture is stable.

Relevant source:

- StreamingLLM: https://arxiv.org/abs/2309.17453

### 5. KV/weight co-location should be explicit planner policy

The user requirement to co-allocate KV alongside the weights that use it is directionally consistent with the existing SYCL code and the literature on heterogeneous LLM serving.

For this plan, interpret co-location as:

- the planner assigns each layer a residency affinity
- weights and the hot KV working set for that layer should default to the same tier and device
- if a layer is host-planned, its KV working set should default to host too
- if a layer is device-planned, its hot KV blocks should get first claim in that device's KV arena
- TP-sharded layers should shard KV according to the same rank/device policy used by the corresponding attention work

This is partly an inference from the target architecture, not a direct copy of one paper.
It should be implemented as explicit planner policy, not as an accidental byproduct of allocation order.

There is already a local hint in the codebase:

- [ggml/src/ggml-sycl/kv-tier-manager.hpp](/Apps/llama.cpp/ggml/src/ggml-sycl/kv-tier-manager.hpp#L24)

### 6. Heterogeneous memory allocation should support layer-specific policies

Recent research reinforces that a single global allocator policy is not enough for modern heterogeneous LLM memory patterns.

In particular, Jenga argues that heterogeneous layer shapes and access patterns need layer-specific allocation and eviction APIs.

That aligns with this plan:

- different layers can have different weight/KV locality needs
- MoE, dense FFN, and attention layers should not all share the same allocator policy
- planner outputs should be able to express layer-class-specific caching rules

Source:

- Jenga: https://arxiv.org/abs/2503.18292

## What To Prove Before Full Implementation

Some parts of the plan are too central or too risky to implement blind.
These should be proved out first with small spikes or microbenchmarks.

### 1. Shared `ggml` relocatable-buffer proof

We should first prove that a backend can:

- allocate logical storage without relying on a stable physical `tensor->data`
- support views without baking in a single base pointer forever
- survive scheduler allocation, graph duplication, and tensor copies

Small test:

- add a minimal opt-in relocatable buffer mode to a contained backend path
- allocate a toy graph with views and copies
- confirm `set/get/copy` still work without depending on one permanent base pointer

Why spike first:

- this is the largest shared-core contract change
- if this path is wrong, every downstream SYCL plan inherits the wrong assumptions

### 2. Long-context KV algorithm selection spike

We should compare:

- exact paged/block KV on current SYCL kernels
- Level Zero virtual-memory-backed KV prototype, if feasible

Decision criteria:

- TTFT and TG latency
- allocator complexity
- graph replay compatibility
- host spill integration
- multi-GPU behavior

Why spike first:

- this determines whether the long-context architecture is block-table-first or virtual-memory-first
- it also decides how invasive the shared `ggml` changes need to be

### 3. Event-chained scheduler and loader proof

We should prove that the shared scheduler and model loader can reduce waits materially before rewriting broad parts of the backend.

Current evidence that this matters:

- the model loader currently blocks on upload events
  [src/llama-model-loader.cpp](/Apps/llama.cpp/src/llama-model-loader.cpp#L1406)
- the shared scheduler still synchronizes in many split and copy paths
  [ggml/src/ggml-backend.cpp](/Apps/llama.cpp/ggml/src/ggml-backend.cpp#L1640)

Small test:

- measure a loader upload ring with event chaining vs eager synchronize
- measure a split-copy microbenchmark with backend event wait vs full backend synchronize

Why spike first:

- this directly impacts PP/TG speed
- the result should shape the final scheduler work rather than following it

### 4. Host segmented-storage compatibility proof

We should prove that large logical host storage can be backed by unified-cache-owned segmented or zone-backed memory without forcing one giant pinned allocation.

Small test:

- build a mock large host-backed tensor allocation path
- verify loader, checksum, and tensor set/get paths can operate without requiring one single direct pinned allocation

Why spike first:

- this is the concrete 120B blocker today

## Massive Context Plan

Massive context support should not be treated as an add-on after weights and scratch are done.
It has to shape the allocator and planner from the start.

### 1. Exact KV should be block-managed

For the baseline architecture:

- KV is divided into fixed-size blocks or pages
- allocation and release operate at block granularity
- old blocks can spill to host-pinned memory
- hot blocks remain on the same device as the layer that will consume them

### 2. KV planning should be layer-affine

The planner should emit per-layer KV policy, not just one global KV budget:

- hot KV blocks per layer
- host spill allowance per layer
- device affinity per layer
- TP shard placement for KV

This is how the user requirement of placing KV alongside the weights that use it becomes actionable.

### 3. KV and weights should share planner affinity groups

Define planner affinity groups such as:

- attention layer weights + its hot KV working set
- MoE router weights + routing scratch
- FFN weights + any persistent helper buffers

These groups are then ranked for VRAM placement by frequency, latency sensitivity, and expected reuse.

### 4. Block size must be benchmarked, not assumed

Paged-attention systems are sensitive to block size.
vAttention explicitly calls out that block size can materially affect runtime.

So block size should be chosen by benchmark, likely from a small set such as:

- 32 tokens
- 64 tokens
- 128 tokens
- 256 tokens

The chosen size can be architecture- and model-dependent.

### 5. Approximate long-context support is a follow-on

If the project later wants:

- token dropping
- sink-token retention
- KV compression
- low-bit KV

that work should be layered on top of the exact block-managed design and gated behind explicit opt-in behavior.

## Required Changes By Area

## A. Shared `ggml` backend and buffer contract

### A1. Add relocatable buffer semantics

Required result:

- `ggml` can represent backend-managed tensor storage without requiring every tensor to have a stable physical `tensor->data` address for its full lifetime.

Likely changes:

- extend backend buffer capabilities
- allow backend buffer types to opt into relocatable storage semantics
- update tensor allocation / view handling so relocatable backends can store logical storage metadata

This is the most important shared-core change because it unlocks direct model-load materialization into unified-cache-managed storage.

### A2. Make buffer data access callback-first for relocatable buffers

Required result:

- set/get/memset paths for relocatable backend buffers go through backend buffer callbacks or backend resolve paths instead of assuming direct pointer arithmetic is always valid

### A3. Upgrade model loader async upload integration

Required result:

- loader obtains upload staging through backend/device APIs
- the upload ring is unified-cache-owned for SYCL
- events are chained between file read, host staging, device upload, and tensor readiness
- no direct host buffer allocation outside backend-owned staging on the planned SYCL path

### A4. Reduce shared scheduler synchronizations

Required result:

- split execution prefers event chaining
- TP and multi-backend copies wait on backend events rather than forcing unnecessary global synchronization

This is a shared `ggml` / scheduler quality-of-implementation issue, not only a SYCL issue.

## B. Unified-cache API and memory ownership

### B1. Create one authoritative allocation API

Required result:

- all SYCL-facing allocations flow through unified-cache
- the API expresses role, lifetime, device, size, alignment, and priority
- return type is a handle or lease, never "raw ownership pointer"

### B2. Unify device arenas

Required result:

- compute scratch
- oneDNN scratch
- reorder temp
- runtime temp
- weight storage
- KV storage

all sub-allocate from pre-reserved device arenas instead of separate `sycl::malloc_device` call sites.

### B3. Unify host-pinned arenas

Required result:

- weight, KV, staging, scratch, and helper buffers all sub-allocate from planned host zones or reusable pools
- no runtime fallback to ad hoc host-pinned allocation in PP/TG

### B4. Centralize deallocation policy

Required result:

- freeing a handle or lease returns memory to unified-cache
- direct `sycl::free` is limited to unified-cache teardown or arena destruction

## C. Planner and placement policy

### C1. Plan all material memory classes

Required result:

The planner produces authoritative placement for:

- dense weights
- MoE weights
- KV reservation
- persistent scratch
- oneDNN scratch
- staging budgets
- TP / multi-GPU helper budgets

### C2. Add explicit priority classes

Required result:

- hot attention weights and latency-critical storage get first claim on VRAM
- colder FFN / experts / helper buffers spill first
- priorities are explicit and consumed by both allocation and eviction

### C3. Make the plan multi-GPU aware

Required result:

- per-device VRAM budgets
- peer access capabilities
- TP shard placement
- shared host budget for cross-device staging
- event lanes for peer or host-mediated movement

The planner must produce one coherent residency plan across all active devices.

## D. Model-load materialization

### D1. Load directly into final storage

Required result:

- model load creates final storage in unified-cache-managed VRAM or host zones
- tensors are populated directly into that storage
- later runtime promotion or streaming is not required just because the initial load used the wrong buffer path

### D2. Replace flat `SYCL_Host` model buffers

Required result:

- no large one-shot pinned host allocation outside unified-cache
- the 120B path uses unified-cache-owned host storage
- any caller that currently requires one flat host pointer is updated to use backend-managed storage or a planned contiguous host zone

### D3. Keep load asynchronous through events

Required result:

- no forced waits between upload chunks unless correctness requires it
- use event chaining from staging ring to upload queue to readiness state

## E. Long-lived tensor-state migration

### E1. Finish handle migration in tensor extras

Required result:

- raw pointer fields stop being the source of truth
- weights, KV, and persistent helper state resolve through handles first

### E2. Convert MoE and TP metadata

Required result:

- expert pointer tables
- shard metadata
- split buffers
- any related residency metadata

all use handle-backed ownership instead of raw long-lived pointer caches.

## F. Temporary-buffer leasing

### F1. Convert all active-path temporaries to leases

Required result:

- getrows staging
- reorder staging
- CPU-dispatch staging
- tensor-split / TP staging
- persistent TG helper buffers
- graph scratch that participates in normal execution

all use unified-cache leases.

### F2. Eliminate hot-path waits

Required result:

- temp buffer reuse is gated by events, not by blocking waits unless the API boundary truly requires a completed result
- multi-GPU helpers use `depends_on` and backend events instead of host-side polling or eager synchronization

## G. Execution routing and scheduling

### G1. Route by resolved residency

Required result:

- host-planned layers execute on CPU
- device-planned layers execute on GPU
- mixed graphs are partitioned intentionally

### G2. Remove runtime dense weight streaming

Required result:

- dense weights do not get opportunistically uploaded during inference
- a planned host decision remains a host decision

### G3. Make multi-GPU / TP execution event-driven

Required result:

- cross-device copies, staging, and reductions are chained by events
- the default path avoids global `wait()` or `synchronize()` at each step

## H. Enforcement and verification

### H1. Add phase-based allocation counters

Required result:

- load
- warmup
- PP
- TG

all report allocation and free counts by role and by tier.

### H2. Add fail-fast debug guards

Required result:

- debug mode can fail if PP/TG allocate outside allowed windows
- debug mode can fail if a non-unified-cache allocator is hit on the planned path

### H3. Add regression and benchmark coverage

Required result:

- GPT-OSS 20B low-VRAM PP/TG run with zero new allocations after warmup
- GPT-OSS 120B load path without direct giant `SYCL_Host` allocation failure
- multi-GPU / TP smoke path with event-chained execution
- PP/TG benchmark comparison showing no regression from excessive synchronization

## Suggested Implementation Order

1. Shared contract and debug guards
2. Shared `ggml` relocatable-buffer and loader changes
3. Unified-cache allocation API and device/host arena unification
4. Planner expansion to all classes and all devices
5. Model-load materialization into final storage
6. Long-lived handle migration
7. Temporary lease migration and event-chained helper paths
8. Residency-driven execution routing and graph partitioning
9. Delete remaining direct alloc/free paths
10. Lock in zero-allocation and performance regressions

## Performance Rules

These rules exist specifically to protect PP/TG speed.

1. No runtime weight streaming on the planned path.
2. No new SYCL allocations in steady-state PP/TG.
3. No full-stream or full-backend waits in hot loops unless a public API boundary demands completed output.
4. Prefer event chaining to host polling.
5. Prefer direct execution where the data lives over transfer-heavy fallback paths.
6. Reserve oneDNN and persistent scratch up front so large prompt batches do not thrash device allocation.

## Risks To Manage

1. Shared `ggml` API changes can affect other backends.
Mitigation: capability-gated behavior, backend opt-in, and narrow core API deltas.

2. Host segmented storage can conflict with code that still expects one flat pointer.
Mitigation: convert those consumers deliberately; do not hide this behind unsafe compatibility hacks.

3. Event-chained execution can expose latent lifetime bugs.
Mitigation: leases must carry readiness information, and reuse must honor outstanding events.

4. Multi-GPU planning can overfit one topology.
Mitigation: planner outputs must depend on detected peer access and bandwidth traits.

## Open Policy Choices

These should be treated as explicit policy, not accidental behavior:

1. `malloc_shared` is not a first-class residency tier for planned inference.
If it remains anywhere, unified-cache must still own it and the planner must treat it as an explicit exception.

2. CPU execution is the default for host-planned dense layers.
Zero-copy GPU execution against host-pinned weights is not the primary model.

3. Large-model load failure is preferable to silent drift from the plan.
If the plan does not fit, fail early and explain why.

## Mapping To Beads Work

This document is the implementation reference for epic `llama.cpp-mubmt`.

Recommended execution order:

1. `llama.cpp-mubmt.1` define relocatable-memory contract, buffer capabilities, and allocation guards
2. `llama.cpp-mubmt.14` prove relocatable `ggml` tensor/view storage before broad backend changes
3. `llama.cpp-mubmt.18` benchmark event-chained loader and split execution before scheduler rewrite
4. `llama.cpp-mubmt.15` choose exact long-context KV allocator: block-paged baseline vs Level Zero virtual memory
5. `llama.cpp-mubmt.2` support relocatable backend buffers and logical tensor storage
6. `llama.cpp-mubmt.3` make model-loader upload staging backend-owned and event-driven
7. `llama.cpp-mubmt.4` unify device-side arenas under unified-cache
8. `llama.cpp-mubmt.5` unify host-pinned arenas and segmented host weight storage
9. `llama.cpp-mubmt.6` expand the planner to authoritative per-device budgets, priorities, and TP placement
10. `llama.cpp-mubmt.16` implement exact long-context KV block manager in unified-cache
11. `llama.cpp-mubmt.17` planner: co-locate per-layer KV with weight residency and TP shards
12. `llama.cpp-mubmt.7` materialize model load directly into planned unified-cache storage
13. `llama.cpp-mubmt.8` finish handle migration for long-lived tensor, KV, MoE, and TP state
14. `llama.cpp-mubmt.9` route temporary, TP, and multi-GPU helper buffers through unified-cache leases
15. `llama.cpp-mubmt.10` reduce scheduler and split-copy synchronization using backend events
16. `llama.cpp-mubmt.11` make execution routing follow planner residency and event readiness
17. `llama.cpp-mubmt.12` remove direct alloc/free call sites and ban raw fallback allocators on the planned path
18. `llama.cpp-mubmt.13` add zero-allocation and 20B/120B/multi-GPU performance regression coverage
19. `llama.cpp-mubmt.19` evaluate approximate long-context KV retention/compression as explicit opt-in modes

Area-to-task map:

- shared `ggml` contract and enforcement:
  `llama.cpp-mubmt.1`
- proof-first relocatable storage spike:
  `llama.cpp-mubmt.14`
- proof-first loader/scheduler event-chaining spike:
  `llama.cpp-mubmt.18`
- proof-first long-context KV algorithm selection:
  `llama.cpp-mubmt.15`
- shared `ggml` relocatable buffer support:
  `llama.cpp-mubmt.2`
- shared model-loader backend-owned staging:
  `llama.cpp-mubmt.3`
- unified-cache device arena unification:
  `llama.cpp-mubmt.4`
- unified-cache host arena and 120B host-storage fix:
  `llama.cpp-mubmt.5`
- authoritative multi-GPU / TP-aware planning:
  `llama.cpp-mubmt.6`
- exact long-context KV block management:
  `llama.cpp-mubmt.16`
- planner weight/KV co-location:
  `llama.cpp-mubmt.17`
- load-time direct materialization into final storage:
  `llama.cpp-mubmt.7`
- long-lived handle migration:
  `llama.cpp-mubmt.8`
- temporary lease migration and reuse:
  `llama.cpp-mubmt.9`
- scheduler event chaining:
  `llama.cpp-mubmt.10`
- residency-driven execution routing:
  `llama.cpp-mubmt.11`
- raw allocator deletion and hardening:
  `llama.cpp-mubmt.12`
- regression and PP/TG proof:
  `llama.cpp-mubmt.13`
- optional approximate long-context follow-on:
  `llama.cpp-mubmt.19`
