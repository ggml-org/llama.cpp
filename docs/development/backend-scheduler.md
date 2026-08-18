# The backend scheduler

`ggml_backend_sched` lets one graph run across several backend devices. It assigns every node to
a backend, allocates the compute buffers, inserts the copies needed where a node's operands live
on another device, and executes the result.

Callers interact with it through `ggml-backend.h`; this document describes what it does and the
contracts it places on callers and on backends.

## Contents

- [Assigning nodes to backends](#assigning-nodes-to-backends)
- [Splits](#splits)
- [Allocation and graph reuse](#allocation-and-graph-reuse)
- [Graph input ring buffer](#graph-input-ring-buffer)
- [Memory read in place by another backend](#memory-read-in-place-by-another-backend)
- [Ops that alias their source](#ops-that-alias-their-source)
- [The scheduler sanitizer](#the-scheduler-sanitizer)
- [Environment variables](#environment-variables)

## Assigning nodes to backends

Backends are passed in priority order, highest first, and the last one must be the CPU. A node is
assigned by, in order:

1. **Where its data already is.** A node whose tensor is already allocated, or which is a view of
   an allocated tensor, must run on the backend owning that buffer - it cannot be moved.
2. **Op support and operand location.** Otherwise the highest priority backend that supports the
   op is used, preferring the backend holding the operands. Ops reading tensors in a buffer marked
   `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` prefer that buffer's backend, so that weights are not
   copied.
3. **Expansion.** Assignments are then expanded along the graph, so that runs of adjacent nodes
   end up on the same backend rather than alternating and forcing a copy at every step.
4. **Graph inputs** (`GGML_TENSOR_FLAG_INPUT`) are assigned to the last backend, which is the CPU.
   The caller writes them, so they have to live somewhere the host can write directly.

`ggml_backend_sched_set_tensor_backend()` overrides the choice for a specific tensor.

## Splits

Once every node has a backend, the graph is cut into **splits**: maximal runs of consecutive nodes
sharing one backend. Each split is submitted to its backend as a single graph, in order.

Where a split reads a tensor produced on another backend, the scheduler checks whether the
consuming backend can read that buffer directly (`ggml_backend_supports_buft`). If it can, the
tensor is read where it is. If it cannot, a copy is created in the consuming backend's buffer and
the node's operand is repointed at the copy - these are the tensors named `<backend>#<name>#<n>`.

Copies are issued asynchronously where the backends support it, with events ordering the consumer
behind the producer.

## Allocation and graph reuse

Compute buffers are sized by `ggml_backend_sched_reserve()` with a worst case graph, so that later
graphs of the same shape need no reallocation. Within a graph the allocator reuses memory: a
tensor's memory is handed to a later tensor once its last consumer *in graph order* has run.

A caller that rebuilds an identical graph can skip the split and the allocation entirely and reuse
the previous one - llama.cpp does this whenever the graph topology is unchanged. This is fast, but
it means neither the split nor the allocator runs, so anything that normally happens there does not
happen on that path. The mechanisms below exist because of that.

## Graph input ring buffer

Graph inputs are assigned to the CPU backend, but the caller may pass a **device host buffer type**
for that slot - pinned memory owned by a GPU. Some devices, integrated GPUs in particular, accept
that buffer type for compute. When that happens no input copy is created and the device reads
exactly the memory the host thread writes.

Nothing then stops the host writing the next iteration's inputs while the device is still reading
the previous ones. The writes are a plain `memcpy` on the calling thread; they are not ordered
against an in-flight `ggml_backend_graph_compute_async`.

The scheduler detects this in `ggml_backend_sched_new()` by testing the condition that elides the
copy - a host buffer type in the last slot that some other backend accepts - rather than by
identifying particular devices, and requiring that backend to be asynchronous. Backends that
deliberately refuse to compute on pinned host memory are unaffected, and so are synchronous ones
such as BLAS - their work is finished before `graph_compute` returns, so nothing of theirs can
still be reading.

When it holds, each graph input gets a **ring of buffers** instead of one, and the scheduler moves
the inputs onto the next slot rather than waiting for the device. The slot being stepped onto was
last read `n_copies - 1` iterations ago, so the wait for it is normally already satisfied.

The extra buffers are not free: one additional copy of every graph input per extra slot, which for
attention masks scales with context and batch size. The default depth is 2 - enough to remove the
wait at the smallest cost that does.

**Rotation only happens when a compute may still be in flight.** Anything that waits for the
backends clears that state, so a caller that synchronizes between iterations - reading logits in a
sampling loop, say - keeps its inputs at fixed addresses and never pays for the ring. Callers that
issue several computes without synchronizing rotate between them.

Two obligations come with this:

- **Callers** must call `ggml_backend_sched_prepare_inputs()` before writing the inputs of a graph
  that is being reused. That path allocates nothing and so cannot rotate on its own. After
  `ggml_backend_sched_alloc_graph()` it is a no-op - allocating already rotated.

- **Backends** that cache work against a graph, such as the CUDA graph cache, may treat an
  unchanged `ggml_cgraph::uid` as a promise that nothing the cached work captured has moved, and
  skip re-reading the addresses. Rotating moves tensor addresses without re-splitting, so the
  scheduler re-stamps the split uids whenever it re-points inputs. A backend keeping such a cache
  must key it on the uid or re-check the addresses itself. Violating this does not degrade
  gracefully: the cached work replays against stale addresses and the results are quietly wrong.

## Memory read in place by another backend

The allocator frees a tensor's memory once its last consumer in graph order has run. That is only
sound if the consumers have finished. When a split reads a tensor in place out of a buffer another
backend owns, the reading split is asynchronous and may still be reading well after graph order
says the memory is dead - so a later split on the owning backend can be given the same memory and
overwrite it.

Such tensors are pinned for the lifetime of the graph with `ggml_gallocr_pin_tensor()`, keeping
them out of the reuse pool. The pin is keyed on the **view root**, since that is the tensor that
owns the memory and the one the allocator frees; keying it on the accessed tensor would miss every
access that goes through a view.

## Ops that alias their source

Some ops write through to one of their sources rather than to fresh memory: `ggml_set()` and
friends, where the result is a view of `src0`. Such a node must run on the backend where that
source lives.

If it does not, the split logic sees an operand on another backend, substitutes a copy for it, and
the op writes into the copy. The copy is never written back, so the update is silently discarded -
the graph still computes, every op still runs, and the result is wrong. On a recurrent model this
shows up as state that stops being carried between tokens, and generation degenerates into a
repeated token.

The scheduler therefore assigns any node that aliases its source, and is not a pure view op, to
the backend of the tensor it aliases.

That move is only made when the target backend supports the op. Support can be conditional on
the tensor types - CUDA runs `GGML_OP_SET` only for F32 and I32 - so the backend owning the
aliased memory is not guaranteed to be able to run the op writing into it. There is no correct
placement in that case: the scheduler copies operands into a split, never results out of one, so
whichever backend runs the op, the write cannot reach the aliased memory. The node is left where
the earlier passes put it, which is what the scheduler did before this rule existed, and the
reason is logged under `GGML_SCHED_DEBUG`.

## The scheduler sanitizer

The above are ordering rules, and ordering bugs do not announce themselves - they produce wrong
numbers on some runs and not others. The sanitizer is a happens-before checker for the backend
API, built into `ggml-base` and enabled at runtime.

It maintains a vector clock per actor - the host thread and each backend - and a shadow map of
which actor last read or wrote every byte of every buffer. Synchronization points (backend
synchronize, event record, event wait, event synchronize, async copies) advance those clocks. When
an access conflicts with a previous one and no happens-before edge connects them, it reports:

```
ggml-sched-sanitize: RACE (write-after-read) on ROCm_Host[0, 2048)
ggml-sched-sanitize:   read  ROCm0#2  @3  split 0   inp_tokens (compute)
ggml-sched-sanitize:   write HOST     @15 split -1  inp_tokens (tensor_set)
ggml-sched-sanitize:   no happens-before edge: HOST knows ROCm0#2@2, needs >=3
```

Where the access is through a view, the report names both the accessed tensor and the allocation
that owns the memory, as `accessed <- root`. The two are frequently unrelated, and the root is the
one that matters.

**It only sees what goes through the backend API.** A caller that writes a host-visible tensor by
taking `tensor->data` and memcpy'ing into it is invisible to it, and races against such writes are
missed entirely. Callers doing that must announce it with `ggml_backend_tensor_set_direct()`,
which is a no-op unless the sanitizer is enabled. In llama.cpp every such site goes through
`llama_host_write()`.

By default the first race aborts. `GGML_SCHED_SANITIZE_NONFATAL=1` reports each one and continues,
so a single run enumerates a whole workload; a count is printed at exit.

## Environment variables

| variable | effect |
|---|---|
| `GGML_SCHED_DEBUG` | `1` prints assignments and scheduler decisions, `2` adds per-node detail |
| `GGML_SCHED_DEBUG_REALLOC` | report, or abort on, unexpected graph reallocations |
| `GGML_SCHED_UMA_RING` | when the ring is in use: `0`/`1` disables it, larger sets the depth. cannot enable it where it was not detected |
| `GGML_SCHED_PIN_ASYNC_READS` | `0` disables pinning of memory read in place by another backend |
| `GGML_SCHED_SANITIZE` | `1` enables the sanitizer, `2` also traces synchronization edges |
| `GGML_SCHED_SANITIZE_NONFATAL` | `1` reports every race instead of aborting on the first |

## Example usage

```c
// operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be assigned
// preferably to run on the same backend as the buffer
ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

sched = ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends,
                               GGML_DEFAULT_GRAPH_SIZE, false, true);

// initialize buffers from a max size graph (optional)
reserve_graph = build_graph(sched, max_batch_size);

// manually assign nodes to a backend (optional, should not be needed in most cases)
struct ggml_tensor * node = ggml_mul_mat(ctx, ...);
ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu);

ggml_backend_sched_reserve(sched, reserve_graph);

// compute
// the graph and its tensors are single-use in terms of allocation, multi-use in terms of computation
graph = build_graph(sched);
for (int i = 0; i < 10; ++i) {
    ggml_backend_sched_graph_compute(sched, graph); // on the first iteration the graph is allocated automatically
}

// if there are graph inputs:
graph = build_graph(sched);           // a new graph that is not allocated
ggml_backend_sched_reset(sched);      // clear the allocation of the previous graph
ggml_backend_sched_alloc_graph(sched, graph);
ggml_backend_tensor_set(input_tensor, ...);
ggml_backend_sched_graph_compute(sched, graph);

// when reusing an already allocated graph rather than allocating a new one, the inputs must be
// made safe to write first - see "Graph input ring buffer"
ggml_backend_sched_prepare_inputs(sched);
ggml_backend_tensor_set(input_tensor, ...);
ggml_backend_sched_graph_compute(sched, graph);
```

An alternative to the above is to assign the inputs to a dedicated context and allocate them
statically with `ggml_backend_alloc_ctx_tensors()`.
