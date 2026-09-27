# Device Memory Infrastructure

This document describes the device-memory infrastructure added by the `feature/device-memory-infra` branch. The changes let multiple GGML components use bounded regions of one backend allocation without confusing physical storage ownership, logical buffer state, or graph workspace lifetime.

The implementation is infrastructure. It does not yet choose how much memory weights, KV cache, or compute graphs receive. It provides the contracts needed for a later arena manager to make those decisions safely.

## Contents

- [Motivation](#motivation)
- [Compatibility](#compatibility)
- [Architecture overview](#architecture-overview)
- [Core concepts](#core-concepts)
- [Public API](#public-api)
- [Placement and lifetime](#placement-flow)
- [Backend implementations](#backend-implementations)
- [Validation and safety](#validation-and-failure-behavior)
- [Testing strategy](#testing-strategy)
- [Commit structure](#commit-structure)
- [Future work](#what-this-enables-next)
- [Backend integration checklist](#adding-another-backend)

## Status

The branch adds:

- bounded tensor allocation inside a caller-selected buffer range;
- graph allocation inside retained external workspace;
- atomic backend-buffer reference counting;
- independently classified backend buffer views;
- reset-safe scheduler integration for stateful backends;
- view implementations for CPU, CUDA/HIP, OpenCL, and SYCL.

The implementation deliberately does not add:

- automatic eviction or migration policy;
- resizable buffer views;
- virtual-to-physical page remapping in GGML;
- overlap detection between independent clients;
- phase-aware prefill/decode memory policy;
- support for every backend that exists in llama.cpp.

## Compatibility

- Existing single-owner buffers still allocate with one reference and are destroyed by one call to `ggml_backend_buffer_free`.
- Existing unbounded tallocr keeps its abort-on-capacity-failure behavior.
- Default gallocr still allocates and owns backend buffers when no external workspace is attached.
- Backends opt in to views per buffer. Unsupported backends return null without changing their existing allocation path.
- Tensor operations continue through each backend's normal buffer interface, preserving existing backend recognition and optimized copy paths.
- The public backend-buffer handle remains opaque. The new reference count, parent link, and view factory are internal object state.

## Motivation

Before this work, a graph allocator normally measured a graph and allocated its own backend buffers. This is simple, but it prevents a higher-level memory manager from reserving one large device allocation and lending different regions to persistent tensors, prefill graphs, decode graphs, or KV storage.

A raw pointer and size are not sufficient for that design:

1. Tensor placement must never exceed the assigned range.
2. A graph allocator must not free storage owned by an arena.
3. The arena storage must not disappear while a scheduler still uses it.
4. Stateful backends must reset graph-local tensor metadata without resetting metadata owned by another component.
5. Each backend must translate a logical view into its native address or offset model.
6. Existing callers that own complete buffers must keep their old behavior.

The new design separates these concerns.

## Architecture overview

~~~mermaid
flowchart TB
    Policy["Future policy layer<br/>weights, KV, prefill, decode"] --> Scheduler
    Scheduler["GGML backend scheduler"] --> Gallocr
    Gallocr["Graph allocator<br/>placement and reuse"] --> View
    Tallocr["Tensor allocator<br/>sequential bounded placement"] --> View
    View["Logical backend buffer view<br/>independent state and lifetime"] --> Parent
    Parent["Parent backend buffer<br/>physical allocation owner"] --> Backend
    Backend["CPU, CUDA/HIP, OpenCL, SYCL"]

    Policy -. "not implemented yet" .-> VMM["Future VMM and eviction policy"]
~~~

There are four layers:

1. **Physical allocation** - A backend buffer owns CPU memory, device memory, an OpenCL object, or SYCL USM.
2. **Logical view** - A child buffer exposes a bounded range and independent usage/reset state while retaining its parent.
3. **Placement** - Tallocr and gallocr assign tensor addresses only inside an approved range.
4. **Scheduling** - A scheduler retains the workspace it receives and reuses it across graph executions.

## Core concepts

### Parent buffer

A parent buffer is the object returned by a backend buffer-type allocation. It normally owns the native allocation and releases it after its final reference is released.

### Buffer view

A buffer view is a new `ggml_backend_buffer_t` representing a non-empty, aligned subrange of another buffer. It has:

- its own GGML buffer object;
- its own size and base address;
- its own usage classification;
- backend-specific context;
- a retained reference to its immediate parent.

A view does not allocate or copy tensor storage. Nested views compose offsets.

### Borrowed range

A borrowed range is an offset and size inside an existing buffer. A bounded tallocr can place persistent tensors directly in such a range.

A raw borrowed range keeps the parent buffer's metadata scope. It is suitable when the caller owns reset and lifetime coordination. It is not equivalent to a buffer view.

### External graph workspace

An external graph workspace is a retained buffer or view attached to gallocr. Gallocr plans tensor reuse inside the supplied capacity instead of allocating a new backend buffer.

### Stateful backend buffer

Some backends attach temporary metadata to a buffer while tensors are initialized:

- OpenCL pools tensor-extra objects;
- SYCL records tensor-extra objects used by optimized layouts.

Resetting the parent buffer could invalidate metadata belonging to persistent tensors or another graph. A complete buffer view gives gallocr an isolated metadata scope that it may reset.

## Memory layout

A future arena can divide one physical allocation without creating another device allocation for each component.

~~~mermaid
flowchart LR
    P["Parent device allocation"] --> A["Persistent tensors<br/>raw bounded range"]
    P --> B["Prefill workspace view<br/>independent reset scope"]
    P --> C["Decode workspace view<br/>independent reset scope"]
    P --> D["KV or other reserved region<br/>future consumer"]

    B --> BG["Prefill gallocr"]
    C --> CG["Decode gallocr"]
~~~

A representative layout is:

~~~text
parent base
|
+---------------- persistent ----------------+
|                                            |
+---------------- prefill view --------------+
|                                            |
+---------------- decode view ---------------+
|                                            |
+---------------- reserved ------------------+
                                             parent end
~~~

This branch does not create that policy. It makes each region enforceable.

## Public API

### Buffer lifetime

~~~c
ggml_backend_buffer_t ggml_backend_buffer_retain(
    ggml_backend_buffer_t buffer);

void ggml_backend_buffer_free(
    ggml_backend_buffer_t buffer);
~~~

`ggml_backend_buffer_free` now releases one reference. The backend destructor runs only after the final reference is released. Existing single-owner code still behaves as before because a newly allocated buffer starts with one reference.

Retain requires an already-live reference. It cannot safely race with release of the last reference or resurrect a destroyed object.

### Buffer views

~~~c
ggml_backend_buffer_t ggml_backend_buffer_view(
    ggml_backend_buffer_t buffer,
    size_t offset,
    size_t size);

bool ggml_backend_buffer_is_view(
    ggml_backend_buffer_t buffer);
~~~

View creation returns null when:

- the parent is null;
- the size is zero;
- the backend did not provide a view factory;
- the range exceeds the parent;
- pointer arithmetic would overflow;
- the view start is not backend-aligned;
- the backend returns an inconsistent child.

The common layer verifies that a backend-created view has:

- the same buffer type as its parent;
- the requested size;
- a base equal to parent base plus offset;
- no existing parent relationship.

Only after validation does the common layer retain and attach the parent.

### Bounded tensor allocation

~~~c
bool ggml_tallocr_new_range(
    struct ggml_tallocr * talloc,
    ggml_backend_buffer_t buffer,
    size_t offset,
    size_t size);
~~~

The bounded allocator aligns the first usable address upward. Alignment padding counts against the supplied size.

Unlike the legacy unbounded path, a capacity failure returns `GGML_STATUS_ALLOC_FAILED`. It does not change the tensor or advance the allocation cursor.

The call rejects null, empty, multi-buffer, meta-buffer, out-of-range, invalid alignment, and arithmetic-overflow cases. Invalid input leaves the destination allocator unchanged.

### External gallocr workspace

~~~c
bool ggml_gallocr_set_buffer_range(
    ggml_gallocr_t galloc,
    int buffer_id,
    ggml_backend_buffer_t buffer,
    size_t offset,
    size_t size);
~~~

Gallocr retains the supplied buffer. The caller may release its own reference after a successful call.

The workspace must be attached after optional size measurement and before reserve. Attaching it invalidates cached node and leaf placements so the next reserve uses the supplied range.

For a stateful backend:

- a raw parent buffer is rejected;
- the buffer must be a view;
- offset must be zero;
- size must equal the complete view size.

This prevents gallocr from resetting metadata outside its assigned logical workspace.

### Scheduler integration

~~~c
bool ggml_backend_sched_set_buffer_range(
    ggml_backend_sched_t sched,
    ggml_backend_t backend,
    ggml_backend_buffer_t buffer,
    size_t offset,
    size_t size);
~~~

The scheduler maps a backend to its gallocr slot and delegates the retained workspace contract to gallocr.

## Placement flow

~~~mermaid
flowchart TD
    M["Measure graph size"] --> V["Create or select workspace view"]
    V --> S["Attach view to gallocr"]
    S --> I["Invalidate cached placements"]
    I --> R["Reset dynamic allocator to<br/>workspace offset and size"]
    R --> P["Plan tensor lifetimes and reuse"]
    P --> C{"Plan fits one bounded chunk?"}
    C -- yes --> K["Keep external storage<br/>no backend allocation"]
    C -- no --> F["Reserve fails cleanly"]
    K --> E["Initialize tensors in view"]
    E --> X["Execute graph"]
    X --> Z["Reset view metadata for reuse"]
~~~

For external workspace, the dynamic allocator begins with one free block at the configured offset. After graph planning, gallocr verifies:

- exactly one chunk was used;
- the maximum planned address is no greater than range end.

Gallocr never grows or replaces external storage. A graph that does not fit returns failure.

If multiple scheduler slots share the same buffer type, they share one dynamic allocator and one virtual buffer wrapper. Capacity reporting counts the shared workspace once.

## Lifetime model

The parent-child relationship forms a reference chain.

~~~mermaid
sequenceDiagram
    participant C as Caller
    participant P as Parent buffer
    participant V as Buffer view
    participant S as Scheduler/gallocr

    C->>P: allocate, refcount = 1
    C->>V: create view
    V->>P: retain parent
    C->>S: attach view
    S->>V: retain view
    C->>P: release caller parent reference
    C->>V: release caller view reference
    Note over P,V: Storage remains alive through scheduler -> view -> parent
    S->>V: reset graph-local metadata
    S->>V: release during scheduler destruction
    V->>V: destroy child backend context
    V->>P: release retained parent
    P->>P: destroy physical allocation after final release
~~~

A child backend context is destroyed before its retained parent is released. This lets a backend finish child cleanup while parent storage is still valid.

The reference counter uses atomic operations:

- retain uses compare-and-exchange while a live reference is held;
- release uses an acquire-release decrement;
- only the thread observing the transition from one to zero destroys the buffer.

The atomic counter protects lifetime only. It does not make all buffer operations concurrently thread-safe.

## Ownership and reset are separate

The graph allocator's virtual buffer wrapper previously used one `owns_chunks` flag for two different questions:

1. Should this wrapper release the buffer?
2. May this wrapper reset backend tensor metadata?

The new model separates them:

- every stored chunk is either owned or explicitly retained and is always released by the wrapper;
- `reset_chunks` controls whether gallocr calls the backend reset callback.

~~~mermaid
flowchart LR
    Owned["Gallocr-allocated buffer"] --> OR["Release: yes<br/>Reset: yes"]
    Raw["Retained raw external buffer"] --> RR["Release: yes<br/>Reset: no"]
    View["Retained external view"] --> VR["Release: yes<br/>Reset: yes"]
~~~

This is the key rule that permits stateful OpenCL and SYCL workspaces without resetting their parent.

## Buffer usage classification

A view starts with its own `GGML_BACKEND_BUFFER_USAGE_ANY` state. Changing a view to COMPUTE or WEIGHTS does not change its parent.

This matters for backends that specialize initialization or memory behavior based on usage. A future arena can lend one physical allocation to components with different logical roles.

## Backend implementations

### CPU

A CPU view stores parent base plus offset as its context and uses the existing non-owning CPU buffer interface. The child never frees the pointer. The common parent reference keeps an owning CPU parent alive.

Nested views add offsets naturally.

For a buffer created from an external pointer, retaining the GGML parent keeps the GGML wrapper alive but cannot extend the lifetime of memory owned outside GGML. The external memory owner must still keep that pointer valid.

### CUDA and HIP

CUDA and HIP share the same GGML backend source. The native buffer context now contains an `owns_data` flag.

- Parent context: native pointer, owns_data = true.
- View context: parent pointer plus offset, owns_data = false.

The view has a separate backend context but never calls cudaFree or hipFree on parent-owned storage. Tensor operations continue through the normal interface, so existing CUDA/HIP fast-path detection still recognizes views.

The same design works for cudaMalloc and managed-memory allocations. HIP shares the implementation but requires AMD hardware for runtime validation.

### OpenCL

OpenCL uses a synthetic GGML base address and stores actual memory in a `cl_mem` object. Its view context therefore contains:

- the shared cl_mem;
- accumulated base offset;
- independent tensor-extra pools;
- an owns_storage flag.

Tensor initialization converts a local view offset into an absolute cl_mem offset. Clear operations begin at the view's base offset instead of zero.

A child does not release the shared cl_mem, while its independent tensor-extra objects are released normally. A view is rejected if the source context no longer represents one contiguous cl_mem object.

### SYCL

An ordinary SYCL buffer owns a USM pointer, queue reference, and a collection of tensor-extra objects. The view context contains:

- parent USM pointer plus offset;
- the same device and queue;
- the same USM-system classification;
- an independent tensor-extra collection;
- owns_data = false.

Reset and destruction affect only the view's tensor-extra collection. The view does not free parent USM.

SYCL split buffers remain unsupported. Their storage is allocated separately per tensor and their GGML base is only a sentinel, so they do not represent one contiguous byte range.

## Backend capability matrix

| Backend | View address model | Independent reset state | Physical storage owner | Runtime tested |
| --- | --- | --- | --- | --- |
| CPU | Host pointer plus offset | No state required | Parent CPU buffer | Yes |
| CUDA | Device pointer plus offset | No state required | Parent CUDA context | Yes |
| HIP | Device pointer plus offset | No state required | Parent HIP context | Source-shared with CUDA |
| OpenCL | Shared cl_mem plus absolute base offset | Independent extra pools | Parent OpenCL context | Yes |
| SYCL | USM pointer plus offset | Independent tensor extras | Parent SYCL context | Yes |
| SYCL split | Not contiguous | Not applicable | Per-tensor allocations | Unsupported |

Other backends return null from `ggml_backend_buffer_view` until they provide a backend view factory.

## Example: shared persistent and graph workspace

~~~c
ggml_backend_buffer_t arena =
    ggml_backend_buft_alloc_buffer(buft, arena_size);

struct ggml_tallocr persistent = {0};
bool persistent_ok = ggml_tallocr_new_range(
    &persistent, arena, persistent_offset, persistent_size);

ggml_backend_buffer_t graph_view =
    ggml_backend_buffer_view(arena, graph_offset, graph_size);

bool workspace_ok = ggml_backend_sched_set_buffer_range(
    sched, backend, graph_view, 0, graph_size);

// The scheduler retained graph_view, and graph_view retained arena.
ggml_backend_buffer_free(graph_view);

// Keep the arena reference until persistent tensors are no longer in use.
// ggml_backend_buffer_free(arena);
~~~

The real owner must check every return value. The caller is also responsible for ensuring that persistent and graph ranges do not overlap while both are live. Tallocr does not retain the arena, so the arena owner must keep its reference until all persistent tensors are dead.

## Example: phase-oriented arena

A later policy layer can use the infrastructure like this:

~~~mermaid
stateDiagram-v2
    [*] --> Load
    Load --> Prefill
    Prefill --> Decode
    Decode --> Prefill: new request
    Decode --> [*]

    state Load {
        [*] --> ReserveArena
        ReserveArena --> PlacePersistentData
    }

    state Prefill {
        [*] --> CreatePrefillView
        CreatePrefillView --> AttachPrefillGallocr
        AttachPrefillGallocr --> ExecutePrefill
    }

    state Decode {
        [*] --> ReleasePrefillView
        ReleasePrefillView --> RepartitionArena
        RepartitionArena --> CreateDecodeView
        CreateDecodeView --> ExecuteDecode
    }
~~~

The missing policy layer must synchronize outstanding work before releasing or repartitioning a region. Reference counting prevents premature destruction; it does not prove that device commands have completed.

## Validation and failure behavior

### Tallocr validation

Bounded tallocr rejects:

- null output or buffer;
- zero size;
- multi-buffer and meta-buffer storage;
- range overflow;
- null or invalid base;
- zero or non-power-of-two alignment;
- pointer arithmetic overflow;
- a range consumed entirely by initial alignment padding.

Allocation failure leaves tensor state and cursor unchanged.

### View validation

The common view API rejects:

- unsupported backend;
- empty or out-of-bounds view;
- address arithmetic overflow;
- unaligned starting address;
- backend child with wrong type, size, base, or existing parent.

The backend-created child is released if common validation fails.

### Gallocr validation

External workspace attachment rejects:

- null or empty input;
- a slot that already has storage;
- mismatched buffer type;
- raw stateful parent;
- partial stateful view;
- invalid bounds or alignment;
- another attached slot sharing the same allocator.

Reserve rejects plans that need more than the supplied range. It does not fall back to an implicit allocation.

## Safety invariants

The implementation relies on these invariants:

1. Every successful retain has exactly one later release.
2. View storage is owned by an ancestor, never by the child view.
3. A view's base is parent base plus its requested offset.
4. Tensor addresses remain inside the buffer object recorded on the tensor.
5. Gallocr resets only buffers whose metadata scope it owns.
6. External workspaces are never silently reallocated.
7. The caller prevents overlap among simultaneously live independent ranges.
8. Device work completes before a region is repurposed.
9. A backend without a correct view implementation reports unsupported.

## Testing strategy

The implementation was developed test-first. The focused tests cover:

### Lifetime tests

- destruction occurs only after the final release;
- concurrent retain/release pairs are safe while one reference remains live;
- a view keeps its parent alive;
- nested views keep the complete ancestor chain alive;
- gallocr keeps caller-supplied workspace alive after caller release.

### Range tests

- alignment padding counts against capacity;
- exact-boundary allocation succeeds;
- overflow and undersized ranges fail;
- failure does not mutate tensor or allocator state;
- graph plans cannot grow beyond external capacity;
- guard bytes outside a workspace remain unchanged.

### State isolation tests

- raw stateful parent is rejected by gallocr;
- complete stateful view is accepted;
- partial stateful view is rejected;
- repeated scheduler reset affects view state rather than persistent parent state.

### Backend runtime tests

- CPU normal builds;
- CPU AddressSanitizer;
- isolated ThreadSanitizer lifetime test;
- CUDA on an RTX 5070 Ti with ordinary device allocation;
- CUDA with managed-memory allocation;
- OpenCL on Intel UHD Graphics 770;
- SYCL on Intel UHD Graphics 770 through the OpenCL SYCL adapter.

The accelerator tests create nested views, release parent handles first, clear only the leaf range, transfer tensors, execute graphs, and reuse scheduler workspace.

HIP shares the CUDA code path but was not runtime-tested on AMD hardware in this development environment.

## Commit structure

Each change is a separate review unit:

| Commit | Purpose |
| --- | --- |
| 5d2f35c3a | Support borrowed graph workspace ranges |
| d8e767886 | Support bounded tensor allocation ranges |
| ecc32c25f | Retain backend buffers across shared owners |
| 9f2a1fd18 | Add generic and CPU backend buffer views |
| 9512abd3e | Retain borrowed graph workspace and isolate reset behavior |
| 3dd1a680b | Add CUDA/HIP backend views |
| de1f7298b | Add OpenCL backend views |
| fb8adb5fd | Add SYCL backend views |

## What this enables next

### Common device-memory arena

A future arena can own one or more parent allocations and lend views to named components:

- model weights;
- persistent KV storage;
- graph workspace;
- transfer staging;
- vision workspace;
- speculative decode state.

The arena should track leases, alignment, synchronization, and phase transitions. It should use backend views rather than exposing raw pointers.

### Prefill-to-decode repartition

After prefill completes and its commands synchronize, a policy can release the prefill view and make that space available to a larger decode KV pool or decode workspace.

This requires a policy and synchronization layer. The current infrastructure only supplies safe storage and allocator primitives.

### VMM-backed growth and shrink

A later virtual-memory layer can reserve stable address ranges and map physical pages according to demand. Buffer views can remain the logical ownership and classification boundary above VMM.

This would address fragmentation and stable-address requirements for weights, KV cache, and expert paging. It is not implemented by this branch.

### Dynamic residency policy

The same foundation can support:

- LRU or frequency-based MoE expert residency;
- KV page residency and streaming;
- temporary graph unload/reload;
- vision and text phase multiplexing;
- memory pressure response.

Those policies must remain outside tallocr and gallocr. Allocators enforce placement; they should not decide model-level eviction policy.

## Adding another backend

A backend view factory must:

1. Create a new backend buffer object.
2. Report base equal to parent base plus offset.
3. Report exactly the requested size and the same buffer type.
4. Avoid freeing parent-owned native storage.
5. Preserve the native device or allocation identity.
6. Maintain an independent reset scope if the backend reset callback is stateful.
7. Support nested views or explicitly reject them.
8. Translate local tensor offsets to native backing offsets.
9. Pass the common CPU and accelerator view tests.

If native APIs use explicit `buffer + offset + length` bindings, the backend can share the native handle and store a base offset in the child context. If native APIs expose sub-buffer handles, the backend may use those handles as long as lifetime and base-address validation remain correct.

## Review checklist

When changing this infrastructure, verify:

- Is ownership explicit for every success and failure path?
- Does a child destructor run before its parent is released?
- Can reset affect tensors outside the logical view?
- Are all range calculations overflow-safe?
- Does alignment padding reduce usable capacity?
- Can an external workspace be silently replaced?
- Are shared allocator slots counted once?
- Does the backend preserve fast-path buffer recognition?
- Are host-visible and device-local ownership rules both correct?
- Does the test release caller references before using retained storage?
- Does the test check guard regions outside the view?
- Has the backend been tested on real hardware, not only compiled?

## Source map

The main implementation is located in:

- `ggml/include/ggml-alloc.h`
- `ggml/include/ggml-backend.h`
- `ggml/src/ggml-alloc.c`
- `ggml/src/ggml-backend-impl.h`
- `ggml/src/ggml-backend.cpp`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-opencl/ggml-opencl.cpp`
- `ggml/src/ggml-sycl/ggml-sycl.cpp`

Focused tests are located in:

- `tests/test-alloc.cpp`
- `tests/test-backend-buffer.cpp`

