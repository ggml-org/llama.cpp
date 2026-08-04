# vLLM Concurrency Deep Study: What vLLM Does Better Than `llama-server`

Read-only design study. No code changes. No commits.

> Purpose: scope a future Tessera server rewrite. Tessera is a llama.cpp fork
> built on the ggml substrate. This document dissects the architectural
> decisions that let vLLM dominate high-concurrency serving, and assesses -
> honestly - which of those decisions are worth porting onto ggml and which
> are not. Every claim about llama.cpp / Tessera is verified by reading the
> working tree at `/Users/user/Developer/GitHub/tessera/`; every claim about
> vLLM cites the vLLM source path, the vLLM docs, or the originating paper.
>
> ASCII only. No em-dashes, no unicode arrows.

## Table of Contents

1. Executive summary
2. PagedAttention (the headline feature)
3. Continuous batching (iteration-level scheduling)
4. Chunked prefill
5. Disaggregated prefill/decode
6. Speculative decoding in a serving context
7. Prefix caching / automatic prompt caching
8. Scheduler internals
9. Distributed serving
10. LoRA / adapter serving
11. Quantization serving tradeoffs
12. Observability and ops
13. What llama.cpp / `llama-server` does BETTER than vLLM
14. Concrete recommendations for a Tessera server
15. Appendix: verified source-path index

---

## 1. Executive summary

vLLM and `llama-server` solve different primary problems. vLLM is a
throughput-first serving stack for GPU datacenters; `llama-server` is a
portability-first single-binary server that also works well for one or a
handful of users. The gap under high concurrency is not a single feature - it
is a stack of cooperating mechanisms (paged KV + iteration-level batching +
chunked prefill + APC + async multiprocessing) that compound.

### 1.1 Headline comparison

| Dimension | vLLM (V1) | `llama-server` (Tessera tree) | Gap multiple |
|---|---|---|---|
| Max throughput (aggregate tok/s, 64 users) | ~12,000 tok/s | ~4,500 tok/s | ~2.7x |
| Max throughput (RPS, peak concurrency) | 35x baseline | 1x baseline | ~35x |
| Throughput at low concurrency (1-4 users) | Similar or lower | Often higher on CPU/Mac | - |
| Max concurrent requests | 100s-1000s (scheduler-bound) | `n_parallel` slots (auto=4) | - |
| P99 TTFT under load | Stable at 64 users | Rises exponentially (queue-bound) | Large |
| P99 ITL under load | Slightly higher with big batches | Extremely low (small batches) | llama wins |
| Memory efficiency (KV cache waste) | <4% (paged) | Slot-based, higher waste unless `kv_unified` | - |
| Cold start (load + first token) | Seconds to tens of seconds (Python, torch compile) | Sub-second to seconds | llama wins |
| Hardware breadth | NVIDIA, AMD, Intel, TPU, XPU, CPU (slow) | CUDA, Metal, Vulkan, SYCL, CPU, RPC, ANE | llama wins |
| Binary portability | Python wheel + heavy deps | Single static binary | llama wins |
| Async / multiprocessing engine | EngineCore in own proc + ZeroMQ IPC | Single event loop, httplib | vLLM wins |

Benchmark sources:
- Red Hat, "vLLM or llama.cpp: choosing the right LLM inference engine" (Sept
  2025), H200 PCIe 141GB, Llama-3.1-8B-Instruct, vLLM v0.10.0 vs llama.cpp
  b6100, GuideLLM v0.2.1, 1-64 concurrency, 300s per run:
  https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case
  - vLLM delivered >35x the request throughput (RPS) and >44x the total
    output tokens/s at peak concurrency.
  - `llama.cpp` throughput was "almost perfectly flat" across concurrency
    (it does not exploit concurrency to raise aggregate tok/s the way vLLM
    does).
  - P99 TTFT: vLLM "remarkably low and stable" at 64 users; `llama.cpp` rose
    exponentially because of its queue model.
  - P99 ITL: roles reversed - `llama.cpp` kept ITL extremely low; vLLM ITL
    rose slightly under large batches.
- GigaGPU benchmark: vLLM ~12,000 tok/s vs `llama.cpp` ~4,500 tok/s at 64
  concurrent users: https://gigagpu.com/vllm-vs-llama-cpp-gpu-servers/
- Original vLLM paper (Kwon et al., SOSP 2023): 2-4x throughput vs
  FasterTransformer / TGI at comparable latency:
  https://arxiv.org/abs/2309.06180

### 1.2 The compounding stack

The 35x RPS figure is not the product of one technique. It is roughly:

```
paged KV (less waste -> bigger batches)
   x continuous batching (no idle decode slots)
   x chunked prefill (long prompts do not stall decodes)
   x APC prefix caching (chat/agent prefixes skipped)
   x async multiprocessing (CPU work overlaps GPU work)
```

Pull any one out and the multiplier shrinks. This matters for the
recommendations in Section 14: there is no single "magic" port.

### 1.3 Important context for Tessera

The Tessera tree already contains several mechanisms that upstream llama.cpp
does not, and that narrow the gap materially:

- `kv_unified` mode: a single shared KV stream across slots instead of one
  per sequence (`common/arg.cpp:1833`, `tools/server/server.cpp:147`).
- `tessera_paged_attn`: a real paged-attention kernel on Metal and CPU with a
  logical-to-physical page map (`src/llama-kv-cache.cpp:1307`,
  `ggml/src/ggml-metal/ggml-metal.metal:11240`).
- Radix-tree prompt cache with cross-slot KV sharing via `llama_memory_seq_cp`
  (`tools/server/server-task.cpp:1638`, `tools/server/server-context.cpp:1729`).
- A cache-aware prefill scheduler with a real cost model
  (`tools/server/server-context.cpp:1692`).
- Speculative decoding (draft model, Eagle3, MTP, n-gram, DFlash/DSpark)
  wired into the same iteration loop (`common/common.h:170`,
  `src/speculative.cpp`).

So this is NOT a comparison of "vLLM vs naive llama.cpp." Several of vLLM's
headline ideas already exist in this tree in a different form. Section 14 is
written accordingly.

---

## 2. PagedAttention (the headline feature)

### 2.1 The problem it solves

In a naive KV cache, each sequence is allocated a *contiguous* buffer sized
for the worst case (`max_seq_len * num_layers * 2 * num_kv_heads *
head_dim`). Two pathologies follow:

1. **Internal fragmentation**: a request that ends at 200 tokens in a 2048
   token slot wastes 90% of its KV reservation.
2. **External fragmentation**: as sequences come and go, the cache becomes a
   Swiss cheese of holes that cannot fit a new request even when the total
   free bytes are sufficient.

The vLLM paper measured existing systems (FasterTransformer, TGI) wasting
**60-80%** of KV cache memory to these two effects plus redundant storage of
shared prefixes
(https://arxiv.org/abs/2309.06180). That waste is what caps the achievable
batch size.

### 2.2 How the paged KV cache works

PagedAttention treats the KV cache like OS virtual memory:

| OS concept | PagedAttention analogue |
|---|---|
| Physical page (4 KiB) | KV block (`BLOCK_SIZE` tokens, default 16) |
| Page table (per process) | Per-sequence block table |
| Virtual address = page index * page_size + offset | Logical token position -> block_id * BLOCK_SIZE + offset |
| Free page list | Block pool (free block queue) |
| Copy-on-write | Reference-counted block sharing for prefixes |

Each sequence holds a **block table** - a list of physical block IDs - that
maps its logical token positions to scattered physical KV cells. Attention
is rewritten to read K/V through this indirection: for each query token, the
kernel looks up `block_table[pos // BLOCK_SIZE]`, then reads
`block[offset]` where `offset = pos % BLOCK_SIZE`.

Because blocks are fixed-size and allocated out of a global pool, there is
no external fragmentation and the only internal waste is the last
partially-filled block of each sequence (at most `BLOCK_SIZE-1` tokens,
i.e. <4% at `BLOCK_SIZE=16`). That is the "near-zero waste" claim.

### 2.3 Block size choice

Default `BLOCK_SIZE = 16`. The paper's docs note that block sizes from 16 to
128 yield similar end-to-end performance
(https://docs.vllm.ai/en/latest/design/paged_attention/):

- Smaller block -> less internal fragmentation, more block-table indirection
  overhead, more pointer chasing in the kernel.
- Larger block -> better memory access locality, more waste in the last
  block.

16 is the empirically sweet spot for GPU memory access patterns.

### 2.4 Where the block manager lives in vLLm

The V1 engine (the current default; see Section 8) split the old
`vllm/core/block_manager.py` into a cleaner set:

| File | Role |
|---|---|
| `vllm/v1/core/kv_cache_manager.py` | Per-request block allocation, ref counting, hash-based reuse |
| `vllm/v1/core/block_pool.py` | Global free-block queue (LRU), alloc/free, eviction |
| `vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock` dataclass, block hashing for APC, `FreeKVCacheBlockQueue` |
| `vllm/v1/core/kv_cache_coordinator.py` | Multi-layer / multi-type coordination |
| `vllm/v1/core/kv_cache_metrics.py` | Hit/miss counters |
| `csrc/attention/attention_dtypes.h`, `csrc/attention/attention_generic.cuh` | CUDA paged-attention kernels (decode + reshape-and-flash prefill) |

The block pool uses a custom doubly-linked list (`FreeKVCacheBlockQueue`)
rather than Python's `collections.deque` because it must remove blocks from
the *middle* of the free list when they are re-matched as prefix-cache hits
(https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py).

### 2.5 Quantitative impact (from the paper)

Kwon et al. 2023 (https://arxiv.org/abs/2309.06180):

- Memory waste: **60-80%** (existing systems) -> **<4%** (vLLM).
- Throughput: **2-4x** vs FasterTransformer and TGI (the state-of-the-art at
  the time), at *comparable* latency.
- The improvements came overwhelmingly from raising the achievable batch
  size on a fixed GPU: when KV waste drops from 80% to 4%, roughly 5x more
  concurrent sequences fit, and decode throughput scales roughly linearly
  with batch size in the memory-bound regime.
- Prefix sharing (parallel sampling and beam search) gave additional
  2-4x on workloads that share prefixes, because shared blocks are
  reference-counted rather than copied.

Caveat: 2-4x was measured against 2023 baselines. Against a modern
tuned `llama-server` (Section 1.1), the gap on a single H200 is closer to
the GigaGPU ~2.7x aggregate-tok/s figure than to 35x; the 35x RPS figure is
amplified by `llama-server`'s flat-throughput queue model at high
concurrency, not by raw per-token efficiency.

### 2.6 Tessera's existing paged KV (verified)

This tree already implements paged KV, with a different surface than vLLM:

```
src/llama-kv-cache.cpp:1307  tessera_kv_block_table::make_page_map()
src/llama-kv-cache.cpp:1324  build_tessera_block_table(sinfo, n_tokens, block_size)
src/llama-graph.cpp:35       tessera_paged_attn_enabled()
src/llama-graph.cpp:3071     ggml_tessera_paged_attn(ctx0, q, k, v, page_map, kq_scale)
ggml/include/ggml.h:2440     ggml_tessera_paged_attn() op declaration
ggml/src/ggml-metal/ggml-metal.metal:11240  kernel_tessera_paged_attn<f16|f32>
ggml/src/ggml-cpu/ops.cpp:9244            ggml_compute_forward_tessera_paged_attn()
```

The mechanism mirrors PagedAttention: `build_tessera_block_table` walks each
stream's cell-index list, groups contiguous runs of cells into spans, and
never crosses a logical block boundary "to keep per-block quantization
metadata and future Metal page tables stable" (verbatim comment at
`src/llama-kv-cache.cpp:1349`). The resulting `page_map[logical_position] =
physical_cell` is fed into the attention op as a tensor.

There is also a higher-level cross-slot radix cache
(`server_kv_block_radix`, `tools/server/server-context.cpp:1729`) that
publishes sealed 32-token KV prefixes and reattaches them with
`llama_memory_seq_cp` so a new slot gets a real block-reference handoff
instead of re-evaluation. This is Tessera's analogue of vLLM's reference-
counted prefix sharing.

So: PagedAttention, as a kernel-level idea, is **already in this tree**. The
remaining gap is in the *scheduler* (Section 3) and the *async engine*
(Section 8), not in attention itself.

---

## 3. Continuous batching (iteration-level scheduling)

### 3.1 Static vs continuous batching

**Static batching** (a.k.a. request-level batching) builds a batch of N
requests, runs the whole batch until *every* request in it finishes, then
starts the next batch. If one request generates 500 tokens and the rest
generate 5, the batch runs at the slowest request's lifetime and the GPU
sits idle on finished requests for hundreds of iterations.

**Continuous batching** (a.k.a. dynamic batching, iteration-level batching,
"in-flight batching") makes the batch membership a *per-iteration* decision.
At every decode step the scheduler may: admit a new request, drop a finished
request, or preempt one. The batch always contains live requests, so GPU
utilization stays high.

This is the single highest-leverage serving optimization. ORCA (OSDI 2022,
https://www.usenix.org/conference/osdi22/presentation/yu) introduced
iteration-level scheduling; vLLM popularized it for LLMs.

### 3.2 `llama-server`'s batching (verified)

`llama-server` does **iteration-level batching across slots** and has for
some time. The relevant code:

```
tools/server/server-context.cpp:3060  update_slots()
tools/server/server-context.cpp:3170  pre_decode()   // builds the batch
tools/server/server-queue.cpp:125     server_queue::start_loop()
```

The loop in `server_queue::start_loop` (the single event loop) calls
`callback_update_slots()` -> `update_slots()` once per iteration, which:

1. Calls `pre_decode()` to build a `llama_batch` from every slot that is
   `SLOT_STATE_GENERATING` and `can_batch_with()` the running batch
   (`server-context.cpp:3250`). New prefill tokens and ongoing decode
   tokens are mixed in the same batch.
2. Slices the batch into `n_batch`-sized views and calls `decode()` on each
   view (`server-context.cpp:3131`).
3. Returns control to the event loop, which processes any new tasks that
   arrived during inference and immediately re-enters `update_slots()`.

Continuous batching is gated by `--cont-batching` / `-cb` and **default
enabled** (`tools/server/README.md:178`). The number of parallel slots is
`n_parallel` (auto = 4, or 32 in the "reasoning" preset, or 2 in long-context
presets; `common/arg.cpp:5246`).

So the gap is NOT "llama-server has static batching." It does not. The gaps
are subtler:

- **Slot count is fixed at startup.** vLLM admits requests up to a running
  cap (default 1024 in V1) and dynamically decides per-step who runs; the
  "slots" are virtual. `llama-server` preallocates `n_parallel` slots and
  cannot exceed that without a restart. Under bursty load, requests queue
  even when KV memory could fit more concurrent short sequences.
- **Preemption.** When KV memory is exhausted, vLLM preempts (V1: recompute)
  the lowest-priority running request and reschedules it. `llama-server`
  has no preemption; if all slots are busy, new tasks defer, and if a slot's
  KV is full it triggers context-shift (`server-context.cpp:3173`) rather
  than swapping the request out.
- **No chunked prefill by default** (see Section 4 - this is the real
  divergence).

### 3.3 vLLM V1 scheduler loop

V1 merged prefill and decode into one scheduling pass. The `schedule()`
method in `vllm/v1/core/sched/scheduler.py` runs every step:

1. **Running phase**: iterate `self.running`. For each request compute
   `num_new_tokens = num_scheduled_tokens - num_computed_tokens` (plus spec
   tokens if speculation is on). Try `kv_cache_manager.allocate_slots()`. If
   allocation fails -> preempt.
2. **Waiting phase**: pull from `self.waiting` (FCFS) or `self.skipped_waiting`.
   Check APC hits, schedule encoder inputs if multimodal, allocate blocks.
   If a remote KV transfer is in flight, set status to
   `WAITING_FOR_REMOTE_KVS` and skip this step.
3. **Output phase**: build a `SchedulerOutput` (new/cached/finished request
   data, encoder/spec metadata).
4. **Post-schedule**: advance `num_computed_tokens` for each scheduled
   request, snapshot routed-expert block IDs for MoE.

### 3.4 Joining, leaving, and preemption

| Event | vLLM V1 behavior |
|---|---|
| Request joins running | Admitted from `waiting` when a slot opens AND KV blocks are available for its first chunk |
| Request leaves | Moved out of `running` the step after its stop condition (EOS, max tokens, abort) |
| KV exhausted | `_preempt_request()`: free the request's blocks, set `num_computed_tokens = 0`, `status = PREEMPTED`, prepend to `waiting` front. **Recompute only - V1 dropped SWAP** (https://docs.vllm.ai/en/stable/configuration/optimization/, https://github.com/vllm-project/vllm/issues/18115) |
| Priority policy | `PRIORITY` preempts `max(running, key=(priority, arrival_time))` - the lowest-priority request, not the newest |

### 3.5 Fairness and starvation

V1 supports two policies in `vllm/v1/core/sched/request_queue.py`:

- `FCFS` (default): strict arrival order.
- `PRIORITY`: requests carry a `priority` field; the scheduler admits the
  highest priority waiting request first and preempts the lowest priority
  running request under pressure.

Starvation prevention relies on two mechanisms: (a) chunked prefill (Section
4) guarantees that decode of already-running requests keeps progressing
during long prefills, and (b) preemption with front-of-queue reinsertion
guarantees a preempted request will be the next one admitted. There is no
explicit aging/lottery scheduler.

### 3.6 Mixing prefill and decode

V1 does **not** separate prefill and decode iterations. A single step's
batch may contain:

- One chunk of a new request's prefill (e.g. 8192 tokens).
- Decode tokens (1 each, plus spec tokens) for every running request.

This is the Sarathi-Serve "decode-maximal batching" pattern (Section 4).
The unified token view (`{request_id: num_tokens}`) is what makes it
natural: prefill and decode are just requests with different token counts
this step.

---

## 4. Chunked prefill

### 4.1 The problem

Without chunked prefill, a long prompt (e.g. 32k tokens) must be processed
in a single forward pass. During that pass:

- No decode token from any other request is generated (the GPU is busy
  prefilling).
- TTFT for every queued request jumps by the prefill time of the in-flight
  long prompt (often 1+ seconds).
- The prefill step is compute-bound; the immediately following decode-only
  steps are memory-bound, so GPU compute is alternately saturated and
  starved.

This is the "prefill stalls decode" pathology. Sarathi-Serve (Agrawal et
al., OSDI 2024 / https://arxiv.org/abs/2308.16369) characterized it
precisely: prefill is compute-bound at even small batch sizes; decode is
memory-bound (1 token per request per step, low arithmetic intensity).

### 4.2 The chunked prefill mechanism

Split the prompt into equal-sized chunks (vLLM V1 default **8192 tokens**
for online serving; configurable via `--long-prefill-token-threshold`).
Each step, build a "decode-maximal batch": one prefill chunk plus as many
running decode requests as fit in the budget. The compute-bound prefill
chunk saturates the SMs; the memory-bound decodes "piggyback" at near-zero
marginal cost (Sarathi reports up to 10x cheaper than a decode-only batch).

This simultaneously:

- Keeps decode flowing (no multi-second stalls while a 64k prompt preflights).
- Smooths compute utilization (every step is a similar compute profile).
- Reduces pipeline-parallel bubbles (uniform micro-batches).

### 4.3 Sarathi-Serve numbers

From https://arxiv.org/abs/2308.16369:

| Model / hardware | Decode throughput | End-to-end throughput | Pipeline bubbles |
|---|---|---|---|
| LLaMA-13B / A6000 | up to 10x | up to 1.33x | - |
| LLaMA-33B / A100 | up to 4.25x | 1.25x | - |
| GPT-3 (pipeline parallel) | - | 1.91x | 6.29x reduction |

### 4.4 V1 specifics

In `vllm/v1/core/sched/scheduler.py`:

- Chunked prefill is **on by default**. (In V0 it was opt-in.)
- `long_prefill_token_threshold` caps the chunk size for unusually long
  prompts; chunks beyond it are split across steps.
- Pooling requests (embeddings) must opt in explicitly because pooling needs
  all tokens in one ubatch.
- The unified `num_new_tokens` budget per step covers prefill chunks plus
  decode tokens plus speculative tokens.

Sources:
- V1 default 8192 chunk size and "enabled by default":
  https://docs.vllm.ai/en/stable/configuration/optimization/
- V1 blog: https://openlm.ai/vllm-v1/

### 4.5 How it interacts with PagedAttention

Chunked prefill needs paged KV: each chunk writes K/V for a contiguous range
of new logical positions, which is exactly the append-a-block operation a
paged cache supports in O(1). With a contiguous per-sequence KV buffer,
chunking would still work but you would lose the fragmentation benefits
that compound with chunking (many concurrent requests at varying lengths).

### 4.6 Tessera's current state (verified)

`llama-server` does **not** do chunked prefill. Prefill of a slot's prompt
is unrolled inside `pre_decode()` and pushed into `batch` until the slot's
prompt is fully ingested or `n_batch` is hit. There is no mechanism to cap
a single request's per-step prefill to a fixed token budget while leaving
room for decodes in the same step.

What `llama-server` does have that partially mitigates the same pathology:

- Prefix caching (radix prompt cache + `kv_unified` block sharing) means
  most chat requests only prefill the *suffix*, so the "long prefill" case
  is far less common in agent workloads than in synthetic benchmarks.
- `n_batch` caps the batch view fed to `decode()` (`server-context.cpp:3131`)
  but this caps *total* tokens in a step, not *per-request prefill* tokens,
  so a single big prompt can still monopolize a step.

**This is the single biggest serving-feature gap.** See Section 14.1.

---

## 5. Disaggregated prefill/decode (prefill-decode separation)

### 5.1 The idea

Even with chunked prefill, prefill and decode share a GPU. DistServe (Zhong
et al., 2024, https://arxiv.org/abs/2401.09670) and Splitwise (Patel et
al., 2024, https://arxiv.org/abs/2311.18677) argue for a harder split: run
prefill and decode on **different GPU pools**, each tuned for its phase:

- Prefill pool: high FLOPS GPUs (prefill is compute-bound).
- Decode pool: high-bandwidth GPUs (decode is memory-bound).

The KV cache produced during prefill is shipped to the decode pool over the
interconnect.

### 5.2 Why it helps

DistServe (https://arxiv.org/abs/2401.09670) measured "strong
prefill-decoding interference" when phases are colocated: a single in-flight
prefill can spike decode tail latency (TPOT) by an order of magnitude. By
eliminating that interference, DistServe reported serving **7.4x more
requests** or enforcing **12.6x tighter SLOs** vs the colocated
state-of-the-art, with >90% of requests within latency limits.

Splitwise (https://arxiv.org/abs/2311.18677) reported cluster-level gains:
**1.4x higher throughput at 20% lower cost**, or **2.35x more throughput at
the same cost and power budget**, by matching hardware to phase.

### 5.3 When it helps and when it hurts

| Helps | Hurts |
|---|---|
| Tight P99 TTFT and P99 ITL SLOs simultaneously | Throughput-first workloads with loose latency SLOs |
| Cluster-scale (many GPUs) where you can afford two pools | Single-GPU or single-node deployments (no second pool to fill) |
| Skewed workloads (very long prompts, short decodes, or vice versa) | Balanced workloads (prefill/decode ratio near the system optimum) |
| When KV transfer fits in interconnect bandwidth | Low-bandwidth interconnects (KV transfer dominates) |

The vLLM docs are explicit on the throughput point
(https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_prefill.md):

> "Disaggregated prefill DOES NOT improve throughput."

It is a latency-SLO tool. You give up some aggregate throughput (extra KV
transfer, two pools to keep warm) to buy predictable tail latency.

### 5.4 vLLM's implementation

V1 ships a connector framework for moving KV between prefill and decode
instances. Layout (paths verified via GitHub):

```
vllm/distributed/kv_transfer/kv_transfer_state.py
vllm/distributed/kv_transfer/kv_connector/base.py            # KVConnectorBase_V1
vllm/distributed/kv_transfer/kv_connector/factory.py
vllm/distributed/kv_transfer/kv_connector/v1/nixl/           # NIXL connector (UCX/GDS)
vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py
vllm/distributed/processor/getpprintpp_processor.py
vllm/config/kv_transfer.py                                   # KVTransferConfig
docs/features/disagg_prefill.md
docs/features/disagg_encoder.md                             # encoder-decode disagg
docs/design/nixl_kv_cache_lease.md
examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py
examples/disaggregated/lmcache/                             # LMCache connector (NIXL-backed)
examples/disaggregated/mooncake_connector/                  # Mooncake (RDMA) connector
```

The architecture has two halves:

- **Scheduler connector** (in the scheduler process): decides when to
  schedule KV cache transfer operations.
- **Worker connectors** (in each worker process): execute the layer-by-layer
  KV store/load against the attention module.

Supported transport connectors (from `docs/features/disagg_prefill.md`):
NIXL (UCX/GDS), Mooncake (RDMA), LMCache (NIXL-backed or its own shared
server), plus custom example connectors.

Request routing is API-driven via `kv_transfer_params`:

1. Client sends prefill request with `do_remote_decode: True` and
   `return_token_ids`.
2. Prefill instance runs the prefill, ships KV, returns the generated
   prompt_token_ids.
3. Client sends decode request to the decode replica with
   `do_remote_prefill: True` and the token IDs, so the decode instance
   skips re-tokenization.

There is also a newer **encoder-prefill-decode** (EPD) split for multimodal
models: `examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py`
and `disagg_1e1p1d_example.sh` (1 encoder, 1 prefill, 1 decode).

### 5.5 Relevance to Tessera

Low. Tessera targets edge and single-node server, not multi-node GPU
clusters. The connector framework is meaningful only when you can dedicate
separate GPUs (or separate nodes) to each phase. On a single Metal/Vulkan
device or a single CUDA card, disaggregation is pure overhead with no
benefit. File under "do not port" (Section 14.2).

---

## 6. Speculative decoding integration in a serving context

### 6.1 Why serving changes spec decoding

Speculative decoding (a cheap draft proposes K tokens, the target verifies
them in one forward pass) is well understood for single-stream inference.
In a *serving* context two new questions dominate:

1. **Does it compose with batching?** Verifying K draft tokens costs ~1
   token of compute but K tokens of KV-cache memory. At high batch sizes,
   the GPU is already compute-saturated (the decode batch is memory-bound
   but big batches approach compute-bound), so the speedup shrinks.
2. **Does it compose with continuous batching and paged KV?** Draft tokens
   must reserve KV blocks speculatively, then release them on rejection.
   Tree-style speculation (multiple candidate continuations) needs tree
   attention, which interacts with paged KV.

### 6.2 vLLM's spec decoding (V1)

V1 rewrote spec decoding around the same unified-token abstraction as the
scheduler. The proposer modules live in `vllm/v1/spec_decode/`:

| File | Method |
|---|---|
| `eagle.py` | EAGLE / EAGLE2 / EAGLE3 |
| `medusa.py` | Medusa (multiple heads) |
| `draft_model.py` | Standalone draft-model speculation |
| `ngram_proposer.py`, `ngram_proposer_gpu.py` | NGram lookup (prompt lookup decoding) |
| `suffix_decoding.py` | Suffix-decoding (retrieval-based draft) |
| `dflash.py` | DFlash |
| `step3p5.py` | "3.5-step" speculation |
| `gemma4.py` | Gemma-specific |
| `custom_class_proposer.py` | User-supplied proposer |
| `dynamic/` | Dynamic speculation-rate scheduling |

MTP (multi-token prediction, as in DeepSeek-V3) is supported via the
draft-model mechanism with embedded MTP heads.

The integration surface:

- The scheduler pads new decode requests with `pad_spec_decode` to keep
  CUDA-graph capture shapes stable.
- A request's `num_tokens_with_spec` feeds the unified scheduling budget.
- Rejected draft tokens trigger KV rollback (cheap with paged KV: just free
  the blocks).
- V1 supports **batched** speculation - multiple concurrent requests can
  speculate in the same step - because the spec tokens are just more
  `num_tokens` per request in the unified scheduler view.

Reported speedups (production numbers, 2025-2026):

| Source | Speedup | Regime |
|---|---|---|
| Snowflake Arctic Inference on vLLM V1 (https://www.snowflake.com/en/blog/engineering/fast-speculative-decoding-vllm-arctic/) | 2.05-2.45x | V1, optimized baseline |
| Red Hat, EAGLE3 on vLLM (https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding) | up to 2.5x | Single-stream |
| AWS P-EAGLE (https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) | 2-3x (EAGLE), +1.69x (P-EAGLE over EAGLE-3) | Production |
| Jarvis Labs (https://jarvislabs.ai/blog/speculative-decoding-vllm-faster-llm-inference) | 1.4-1.6x | Large concurrent batches |
| arXiv 2508.08192 (https://arxiv.org/html/2508.08192v1) | ~1.4x | Large batches |

The pattern is consistent: spec decoding gives 2-3x for single-stream /
low-concurrency workloads and **converges toward ~1.4x at large batch
sizes** because the GPU is already saturated and there is no spare compute
to absorb verification work for free.

### 6.3 Batched tree attention

EAGLE2/3 propose a *tree* of candidate continuations (not a linear chain)
to raise acceptance rate. Verifying a tree needs attention that masks out
non-tree edges. vLLM supports this via the unified attention backend;
tree-attention as a first-class kernel was tracked in
https://github.com/vllm-project/vllm/issues/18327.

### 6.4 Tessera's spec decoding (verified)

This tree has a substantial spec-decoding stack that is **already wired
into the iteration loop** alongside continuous batching:

```
common/common.h:170           common_speculative_type enum
common/common.h:329           common_params_speculative_draft
common/common.h:377           common_params_speculative (ngram_mod, map_k, map_k4v, cache)
src/speculative.cpp           common_speculative_*
tools/server/server-context.cpp:3264  drafting/generating slot split
tools/server/server-context.cpp:3273  n_draft_max from slot state
tools/server/server-context.cpp:446   slot.generate_draft()      // emits sampled + spec_draft into batch
```

Supported types (`common/common.h:170`):

- `DRAFT_SIMPLE` - standalone draft model.
- `DRAFT_EAGLE3` - EAGLE3.
- `DRAFT_DFLASH`, `DRAFT_DSPARK` - DFlash / DSpark.
- `DRAFT_MTP` - multi-token prediction (default when a model has MTP heads
  and no explicit type is set: `server-context.cpp:1072`). Includes an
  **ANE MTP program** path (`common_params_speculative_draft::ane_mtp_program`,
  `common_speculative_init_result_ptr`), which is Tessera-specific.
- `NGRAM_SIMPLE`, `NGRAM_MAP_K`, `NGRAM_MAP_K4V`, `NGRAM_CACHE` - self-
  speculative n-gram methods (no draft model needed).

Slot draft metrics are tracked (`n_draft_total`, `n_draft_accepted`,
`n_draft_verif_steps`, per-position acceptance) and exposed via the
per-slot API and timings (`server-context.cpp:292`, `:613`).

The gap vs vLLM here is narrow. Tessera already:

- Mixes draft and verification into the same batched iteration loop.
- Supports per-slot drafting decisions (`slot.get_n_draft_max()`,
  `server-context.cpp:425`).
- Supports tree-style multi-position drafts via per-position acceptance
  arrays (`n_accepted_per_pos`, `server-context.cpp:295`).

What vLLM has that Tessera does not:

- **Suffix decoding** and the **dynamic spec-rate scheduler**
  (`vllm/v1/spec_decode/dynamic/`) that tunes the speculation depth based
  on observed acceptance.
- Batched tree attention as a single fused kernel (Tessera does it per-
  position via the slot loop).

These are second-order. The first-order capability (batched spec decoding
integrated with continuous batching and paged KV) is present.

---

## 7. Prefix caching / automatic prompt caching

### 7.1 vLLM APC: how blocks are hashed and reused

vLLM's Automatic Prefix Caching (APC) exploits the paged KV cache directly.
Because KV is stored in fixed-size blocks identified by physical block ID,
a prefix cache hit is just "these blocks already contain valid K/V for my
prefix - share them by reference."

The hash chain (verified in
`vllm/v1/core/kv_cache_utils.py`):

- `init_none_hash` seeds the chain (random unless `PYTHONHASHSEED` set).
- `hash_block_tokens(parent_block_hash, curr_block_token_ids, extra_keys)`
  produces a block's fingerprint. `extra_keys` carries multimodal hash,
  LoRA id, or prompt-embedding id, so the cache key includes everything
  that affects K/V.
- `get_request_block_hasher` walks the request's tokens at `hash_block_size`
  granularity, chaining each block's hash off the previous one.

Reuse mechanics:

- `KVCacheBlock` carries a `_block_hash` attribute set via
  `set_block_hash`.
- A `ref_cnt` field tracks how many requests share the block. Multiple
  requests sharing a prefix share the same physical blocks by reference -
  no copy.
- The free queue is `FreeKVCacheBlockQueue`, a custom doubly-linked list
  (not `collections.deque`) so a block can be removed from the middle when
  it transitions from free to cached.
- Eviction is LRU: when `ref_cnt` hits 0 the block moves to the tail of the
  free queue; the next allocation pops from the head. `reset_hash` clears
  the cached hash on eviction so it is not re-matched.

In V1, APC is **enabled by default** because the hash-chain data structures
are constant-time and avoid Python object churn, so the no-hit case has
near-zero overhead (https://openlm.ai/vllm-v1/).

### 7.2 Hit-rate implications

APC helps the workloads that dominate production:

- **Multi-turn chat**: each turn's prompt is `system + history + new turn`.
  The shared prefix grows monotonically; APC turns each turn's prefill
  cost from O(history) into O(1) suffix tokens.
- **Agent / RAG**: many requests share a system prompt, tool list, or
  retrieved-context preamble. APC serves the preamble once.
- **Few-shot batch**: same instruction prefix, different inputs.

APC does **not** help decode (it is a prefill optimization only) and does
not help when requests have no shared prefix
(https://github.com/vllm-project/vllm/blob/main/docs/features/automatic_prefix_caching.md).
vLLM also warns that logits are not guaranteed bit-identical across batch
sizes, so enabling APC can introduce small numerical drift vs re-evaluating
every time.

### 7.3 vLLM prefix-cache metrics

V1 exposes `vllm:prefix_cache_queries` (counter) and `vllm:prefix_cache_hits`
(counter) so hit rate is observable directly.

### 7.4 Tessera's prefix caching (verified)

This tree has a layered prefix-cache design that is in some ways richer
than vLLM's, though less tightly coupled to the paged KV:

```
tools/server/server-task.cpp:1638    server_prompt_cache (radix tree)
tools/server/server-task.cpp:1663    radix_rebuild()
tools/server/server-task.cpp:1692    find_longest_prefix()
tools/server/server-task.cpp:1714    alloc()
tools/server/server-task.cpp:1790    load()
tools/server/server-context.cpp:975  std::unique_ptr<server_prompt_cache> prompt_cache
tools/server/server-context.cpp:1729 publish_slot_kv_blocks()   # block-radix publish
tools/server/server-context.cpp:1746 get_available_slot()        # cache-aware admission
```

Three layers, in increasing order of efficiency:

1. **Serialized prompt snapshots** (`server_prompt_cache_state`): full
   prompt + KV state saved to a RAM-Limited (`cache_ram_mib`, default
   8192 MiB, `tools/server/README.md:168`) radix tree. On a slot miss, the
   radix tree is walked to find the longest common prefix and the snapshot
   is restored. This is the slowest path - involves serialization.
2. **Cross-slot `seq_cp` sharing** (`server-context.cpp:1909`): when a new
   slot's prompt shares a prefix with an idle slot, `llama_memory_seq_cp`
   copies the KV *cell references* (not the data) within the unified
   stream. This is the equivalent of vLLM's reference-counted block sharing.
3. **Block radix** (`server_kv_block_radix`, `kv_block_radix`): when
   `kv_unified` is on, sealed 32-token KV prefixes are published into a
   block-keyed radix. A new request attaches to a matching source with one
   `seq_cp` call, "without serializing or copying K/V"
   (`server-context.cpp:1863-1866`).

Admission is cache-aware: `get_available_slot()` chooses which slot to
assign a new request to based on (a) `slot_prompt_similarity` (LCP
threshold), (b) radix-hit preference for the shortest resident context
(preserves large idle contexts), (c) LRU fallback
(`server-context.cpp:1746-1853`).

There is also a **cache-aware prefill scheduler** that scores which
admitted prompt gets the next shared batch
(`server-context.cpp:1692`):

```
score_milli = 1000 * prefix_positions
            +    8 * min(age_ms, 60000)
            +   16 * acceptance_milli        // spec-decode rolling acceptance
            -    4 * estimated_cost_tokens   // attention-weighted suffix cost
```

This is more sophisticated than vLLM's FCFS+prefix matching in some
respects (it factors in spec-decode acceptance and attention-weighted
suffix cost). It is explicitly described as a "local policy" that does not
reorder the task queue (`server-context.cpp:1687`).

Gaps vs vLLM:

- vLLM's prefix cache is **on by default and near-zero overhead** in V1.
  Tessera's is on by default (`--cache-prompt` defaults enabled,
  `tools/server/README.md:214`) but the serialized-snapshot path is not
  zero-cost; only the `kv_unified` block-radix path approaches vLLM's
  characteristics.
- Tessera's block radix is **gated behind `kv_unified`** (which defaults
  off except when `n_parallel < 0`, `server.cpp:147`). vLLM's paged +
  APC is the unconditional default.

---

## 8. Scheduler internals

### 8.1 vLLM V1 async architecture

V1's headline architectural change is multiprocessing the engine. From
`vllm/v1/engine/async_llm.py` and the V1 announcement:

```
HTTP request
   |
   v
AsyncLLM (API server process, asyncio)
   |   tokenization, streaming, detokenization, tool parsing
   |   ZeroMQ IPC  <---  EngineCorePoller
   v
EngineCore (separate process)
   |   scheduler + model executor + worker(s)
   |   GPU work
   v
GPU
```

- **AsyncLLM** runs in the API server process. It owns the asyncio event
  loop, tokenization, request streaming, and output processing. It talks
  to EngineCore over ZeroMQ (https://openlm.ai/vllm-v1/).
- **EngineCore** (`vllm/v1/engine/core.py`) is an isolated process that
  runs the scheduler and the model executor back-to-back in a tight loop.
  Because it is its own process and Python GIL, CPU-heavy work in the API
  server (tokenization of a 100k-token prompt, JSON tool-call parsing of a
  long stream) overlaps with GPU inference rather than blocking it.
- **EngineCoreClient** (`vllm/v1/engine/core_client.py`) is the IPC
  abstraction (in-process, multiprocess, or ray).
- For tensor parallelism, the EngineCore in turn drives multiple workers
  over either multiprocessing or Ray (`vllm/v1/executor/`:
  `uniproc_executor.py`, `multiproc_executor.py`, `ray_executor.py`,
  `ray_executor_v2.py`).

The payoff: **up to 1.7x throughput vs V0** (https://openlm.ai/vllm-v1/)
purely from removing CPU overhead from the GPU critical path. This is the
same insight as NVIDIA's "CPU offload the host work" but applied to the
Python orchestrator.

### 8.2 Request flow: HTTP to GPU

1. HTTP hits `vllm/entrypoints/openai/api_server.py` (uvicorn + FastAPI).
2. Request is parsed, validated, and handed to AsyncLLM.
3. AsyncLLM tokenizes (async, off the hot path), builds a
   `EngineCoreRequest`, and pushes it over ZeroMQ.
4. EngineCore's scheduler (`vllm/v1/core/sched/scheduler.py`) picks it up
   on the next `schedule()` call, allocates blocks via
   `KVCacheManager`, and emits a `SchedulerOutput`.
5. The output goes to the model executor (`vllm/v1/worker/`), which builds
   a GPU batch and launches the kernel(s).
6. Outputs come back to EngineCore, get pushed over ZeroMQ to AsyncLLM,
   which detokenizes and streams to the client.

### 8.3 Where vLLM is still bottlenecked

- **Python overhead.** Even with V1's work, the scheduler is Python. At
  very high QPS or with very small models, Python scheduling per step
  becomes a measurable fraction of step time. CUDA graphs mitigate the GPU
  side; the host side is still CPython.
- **Tokenizer latency.** Tokenizing a long prompt is single-threaded CPU
  work; V1 overlaps it with GPU work but a cold huge prompt still has to
  be tokenized before its first chunk can be scheduled.
- **Single-stream spec verification.** Spec decoding adds host-side
  bookkeeping per step (draft proposal, target verification, acceptance
  mask) which raises per-step CPU cost.
- **Prefix-cache hash computation.** Hashing every block of every request
  is cheap but not free; V1's LRU caching of block hashes
  (`vllm/v1/core/kv_cache_utils.py`) is the mitigation.

### 8.4 Tessera's scheduler (verified)

Single-threaded event loop. No multiprocessing, no async I/O separate from
inference:

```
tools/server/server.cpp:161          server_context ctx_server
tools/server/server-queue.cpp:125    server_queue::start_loop()
tools/server/server-queue.h:76       start_loop(idle_sleep_ms)
tools/server/server-context.cpp:3060 update_slots()
tools/server/server-http.cpp         httplib-based synchronous HTTP
```

The flow:

1. httplib receives a request on its worker pool, parses JSON, posts a
   `server_task` to `server_queue`.
2. `server_queue::start_loop()` is the single inference-driving thread.
   Each iteration it drains the task queue (calling `callback_new_task`),
   then calls `callback_update_slots` -> `update_slots()` to run one
   inference step.
3. `update_slots()` builds the batch from all eligible slots, decodes,
   posts results back to `server_response`, which the HTTP worker is
   blocking on (`server_response::recv`).
4. If no tasks arrive for `idle_sleep_ms`, the loop enters a sleeping
   state (configurable; intended to let the OS idle the CPU / GPU for
   power saving). A new task wakes it via `req_stop_sleeping`.

Strengths of this design:

- No IPC, no serialization between HTTP and inference - they share memory
  directly through `server_task`.
- Zero Python in the hot path. The entire scheduler and execution are C++.
- Deterministic ordering.

Weaknesses for high concurrency:

- **HTTP work and inference share a thread boundary.** Although httplib
  uses a thread pool for connections, every task that needs to mutate
  slot state is funneled through `server_queue`'s single mutex and
  processed serially before the next `update_slots()`. Tokenization of a
  huge prompt, JSON parsing of a giant tool array, or multimodal chunking
  blocks the queue's drain.
- **No preemption and no dynamic admission.** Slot count is fixed; the
  scheduler cannot over-subscribe and then prune.
- **`idle_sleep_ms` adds wake latency.** For battery-constrained edge
  deployment this is a feature; for a loaded server it is a tax.

---

## 9. Distributed serving

### 9.1 vLLM

vLLM supports all three classical parallelism strategies, plus expert
parallelism for MoE:

| Strategy | What | Where | Config |
|---|---|---|---|
| Tensor parallel (TP) | Shard each layer's weights across GPUs (Megatron column/row parallel) | Within a node (NVLink) | `tensor_parallel_size` |
| Pipeline parallel (PP) | Split layers across GPUs/nodes in stages | Across nodes (often) | `pipeline_parallel_size`, virtual stages |
| Data parallel (DP) | Replicate the full model; LB across instances | Across nodes | https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/ |
| Expert parallel (EP) | Shard experts across ranks for MoE | Within/ across nodes | mixed with TP |

Source paths:
- `vllm/distributed/` (the whole directory; parallel state, comms, custom
  all-reduce).
- Execution backends in `vllm/v1/executor/`: `uniproc_executor.py`,
  `multiproc_executor.py`, `ray_executor.py`, `ray_executor_v2.py`.
- KV transfer for disagg in `vllm/distributed/kv_transfer/`.
- TP uses Megatron-LM's algorithm (column parallel for `qkv_proj`, row
  parallel for `o_proj` / `down_proj`); only the inputs/outputs of
  parallel regions need all-reduce, so communication is amortized over the
  layer's compute.

Docs:
- Parallelism overview: https://docs.vllm.ai/en/stable/serving/parallelism_scaling/
- TP/PP/DP guide for MoE (AMD): https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html
- Heuristic: TP size = GPUs per node, PP size = number of nodes
  (https://docs.vllm.ai/en/v0.5.2/serving/distributed_serving.html).

### 9.2 `llama-server` (verified)

llama.cpp has a **single distributed mechanism**: the RPC backend
(`tools/rpc/`):

```
tools/rpc/rpc-server.cpp        ggml-rpc-server: expose local ggml devices over TCP
tools/rpc/README.md             "currently in a proof-of-concept development stage"
```

The RPC backend is a *device offload* mechanism, not tensor or pipeline
parallelism. The main process owns the model graph and offloads individual
ggml tensors/ops to remote `ggml-rpc-server` instances, which expose their
local CUDA / Metal / CPU devices. Topology (from `tools/rpc/README.md`):

```
host (main) ---- TCP ---- host A: ggml-rpc-server <-> CUDA0
            \---- TCP ---- host B: ggml-rpc-server <-> Metal
            \---- TCP ---- host N: ggml-rpc-server <-> CPU/CUDA
```

What this gives you:

- Heterogeneous compute (mix Metal + CUDA + CPU across hosts).
- Layer-level or tensor-level placement of weights and ops on remote
  devices (`--rpc` flag, override-tensor buffer types).

What it does NOT give you:

- No tensor parallelism (no Megatron-style sharded matmul with all-reduce).
- No pipeline parallelism (no micro-batched stage scheduling).
- No data parallelism with load balancing at the server layer (you would
  have to put N `llama-server` instances behind your own LB).
- The README explicitly warns it is "fragile and insecure" and not for
  production network exposure.

The gap here is large in absolute terms but **low-relevance for Tessera's
target**. Tensor parallelism matters when a single model does not fit on
one GPU; that is a datacenter problem. For edge + single-card server,
the RPC backend's heterogeneous offload is arguably more useful than TP.

---

## 10. LoRA / adapter serving

### 10.1 The multi-LoRA problem

Serving many LoRA adapters concurrently means: in a single batch, request A
uses adapter 1, request B uses adapter 2, etc. Naively you would run N
forward passes (one per adapter). The state of the art runs **one** batched
forward pass with the LoRA delta computed on-the-fly per token using
segmented/grouped GEMM kernels.

### 10.2 vLLM's multi-LoRA

Source paths:

```
vllm/lora/lora_model.py
vllm/lora/lora_weights.py
vllm/lora/model_manager.py
vllm/lora/worker_manager.py
vllm/lora/punica_wrapper/punica_gpu.py        # GPU batched LoRA kernels
vllm/lora/punica_wrapper/punica_cpu.py
vllm/lora/punica_wrapper/punica_xpu.py
vllm/lora/ops/triton_ops/                     # SGMV / BGMV Triton kernels
vllm/lora/ops/torch_ops/
vllm/lora/ops/xpu_ops/
vllm/lora/layers/                             # LoRA-aware layer impls
```

The Punica wrapper (from the Punica paper, Wang et al. 2023) implements
the segmented/grouped matrix-vector multiply that lets one batched kernel
serve many adapters. S-LoRA (Chen et al., 2023, https://arxiv.org/abs/2311.03285)
extended this with CUDA kernels for batching without sacrificing
throughput; vLLM's implementation draws from both lines of work.

Configuration (`docs/features/lora.md`):

- `max_loras`: number of adapters the engine processes concurrently.
- `max_cpu_loras`: number staged in host memory for swapping to/from GPU.
- `max_lora_rank`: max rank of supported adapters (over-provisioning wastes
  memory).
- Per-request adapter selection via the `model` field (server) or
  `LoRARequest` (offline).
- Runtime adapter loading/unloading via `/v1/load_lora_adapter` and
  `/v1/unload_lora_adapter` (gated on `VLLM_ALLOW_RUNTIME_LORA_UPDATING`).
- In-place LoRA reload for asynchronous RL training (`load_inplace`).
- LoRAResolver plugins for pulling adapters from local dir, HF Hub, S3, etc.

S-LoRA's headline result (https://arxiv.org/abs/2311.03285): serve
**thousands** of LoRA adapters on a single GPU with **2x-4x** higher
throughput than naive batching, at the cost of one extra grouped GEMM per
LoRA-applied layer.

### 10.3 `llama-server` LoRA (verified)

llama.cpp supports LoRA but the multi-LoRA-in-one-batch story is limited:

- Multiple adapters can be loaded (`--lora FNAME,FNAME2`, `--lora-scaled
  FNAME:SCALE`, `tools/server/README.md:95`).
- Adapters can be applied/changed at runtime via `/lora-adapters`
  (`tools/server/README.md:236`, `--lora-init-without-apply`).
- **Per-slot adapter selection exists**: each `server_slot` carries a
  `lora` map (`server-context.cpp:3116` calls `common_set_adapter_lora`
  once per batch with `slot_batched->lora`). However, the current batching
  requires all slots in a batch to share a single active LoRA configuration
  for the batched forward pass - the comment at `server-context.cpp:3114`
  is explicit: "TODO @ngxson : alora handling is too messy."

What is missing vs vLLM:

- No Punica/SGMV-style batched kernel that applies *different* adapters to
  *different* requests in the same forward pass.
- No `max_cpu_loras` style adapter swapping for serving hundreds of
  adapters.
- No S-LoRA-style uniform-memory multi-adapter throughput.

For Tessera's likely workloads (a small number of adapters, often zero or
one), the current state is adequate. For a multi-tenant inference service
where every tenant has their own fine-tune, vLLM's Punica integration is a
hard requirement.

---

## 11. Quantization serving tradeoffs

### 11.1 vLLM quantization support

vLLM exposes a uniform `--quantization` flag across many backends
(https://docs.vllm.ai/en/latest/features/quantization/). Categories:

| Family | Methods | Notes |
|---|---|---|
| Weight-only INT4 | `gptq`, `awq`, GPTQ-Marlin, AWQ-Marlin | Marlin is a *kernel*, not a format; it accelerates already-quantized GPTQ/AWQ |
| FP8 | `fp8` (E4M3, E5M2) | Hopper+ (H100/H200), Blackwell; recommended for serving |
| INT8 | `int8` | |
| Other | `bitsandbytes`, `gguf`, `compressed-tensors`, `aqlm`, `fpaq`, `exllama2` | |

The crucial insight (benchmark-verified, https://www.reddit.com/r/LocalLLaMA/comments/1q7ysj2):

- **Marlin turns quantization from a loss into a win.** GPTQ *without*
  Marlin was actually slower than FP16 (276 tok/s vs 461 tok/s FP16) on
  H200. GPTQ *with* Marlin hit 712 tok/s - quantized AND faster.
- AWQ via Marlin: 741 tok/s vs AWQ's default kernel at 68 tok/s (~10x).
- For Hopper+/Blackwell, FP8 is the recommended serving format (best
  speed/quality tradeoff).
- Avoid serving GPTQ/AWQ without an optimized kernel - throughput can
  regress below FP16.

### 11.2 Tessera quantization

Tessera/llama.cpp uses the k-quant family (Q4_K_M, Q5_K_M, Q8_0, etc.) and
the Tessera-specific T640 format. The relevant tradeoffs:

- **k-quants** are designed for CPU/Mobile/GPU portability, not peak GPU
  throughput. They dequantize inline during the matmul rather than using a
  fused INT4 GEMM kernel.
- **T640** is Tessera's block-quant format; `constexpr_lut_to_dense`-style
  LUT dequant is a candidate mapping (per `docs/ane-backend-deep-study.md`).
- llama.cpp's CUDA backend uses Marlin-derived fused kernels for some
  formats but not the full GPTQ/AWQ-Marlin surface that vLLM exposes.

The serving implication: at high batch size, a fused low-precision GEMM
(FP8 or INT4-Marlin) gives substantially more decode throughput per GPU
than k-quant dequant-then-matmul. This is one place where vLLM's CUDA-
specific kernel investment pays off and ggml's portability-first design
costs throughput.

For Tessera's stated targets (edge + Apple Silicon + heterogeneous
backends), the k-quant family remains the right default; the gap matters
mainly on NVIDIA datacenter GPUs where switching to a Marlin/FP8 path
would be a win.

---

## 12. Observability and ops

### 12.1 vLLM

V1 exposes a comprehensive Prometheus metrics surface and OpenTelemetry
tracing (https://docs.vllm.ai/en/stable/design/metrics/):

**Engine/core metrics (gauges):**
- `vllm:num_requests_running` - currently running requests.
- `vllm:num_requests_waiting` - queued.
- `vllm:kv_cache_usage_perc` - fraction of KV blocks in use.

**Request metrics (Prometheus histograms, bucketed for quantile
estimation):**
- `vllm:time_to_first_token_seconds` (TTFT).
- `vllm:e2e_request_latency_seconds`.
- `vllm:request_prefill_time_seconds`.
- `vllm:request_decode_time_seconds`.
- `vllm:inter_token_latency_seconds` (ITL).
- `vllm:request_prompt_tokens`, `vllm:request_generation_tokens`.

**Prefix-cache metrics (counters):**
- `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits`.

**Spec decoding metrics:** listed as future work in V1 (V0 had
`vllm:spec_decode_draft_acceptance_rate`, `..._efficiency`,
`..._num_accepted_tokens`, `..._num_draft_tokens`).

**Tracing:** OpenTelemetry via `--otlp-traces-endpoint`
(https://docs.vllm.ai/en/latest/api/vllm/config/observability/). Per-
request spans cover tokenization, scheduling, prefill, decode.

**Ops:** vLLM has no first-class hot-reload or model swap (you redeploy),
but does support graceful shutdown signal handling. Multi-model serving is
typically done by deploying multiple vLLM instances behind an LB.

### 12.2 `llama-server` (verified)

- **Metrics endpoint**: `/metrics` with Prometheus-compatible output
  (`tools/server/README.md:1056`), gated on `--metrics`. Includes counters
  for `tokens_predicted_total`, `prompt_tokens_total`, slots in use,
  `requests_processing`, draft acceptance stats
  (`server-context.cpp:4827`), and Tessera-specific scheduler counters
  (`scheduler_prefill_selections_total`,
  `scheduler_prefix_positions_total`,
  `scheduler_estimated_cost_tokens_total`,
  `scheduler_acceptance_milli_total`).
- **Slots endpoint**: `/slots` exposes per-slot state (speed, processed
  tokens, sampling params, draft stats).
- **Router mode**: multi-model serving via the router server
  (`server.cpp:130`, `--models-dir`, `--models-max`) with autoload and
  runtime load/unload (`/models/load`, `/models/unload`,
  `/models/sse`). This is more capable than vLLM's typical "one model per
  process" deployment.
- **Graceful shutdown**: SIGINT/SIGTERM handler
  (`server.cpp:399`, `:446`); double-Ctrl+C force-terminates
  (`server.cpp:30`). Session GC finalizes live sessions and wakes pending
  readers before tearing down (`server.cpp:383`, `:411`).
- **Resumable streaming / sessions**: conversation-id-based session
  identity end-to-end through the router (`server.cpp:267`).
- **Hot model swap**: via router mode at runtime; a single-model server
  requires restart.
- **OpenTelemetry / distributed tracing**: **not present**. No `otlp`,
  no tracing symbols in `tools/server/` or `common/`.
- **Structured logs**: log macros (`SRV_INF`, `SRV_DBG`, etc.) with
  component prefixes, but not JSON-structured by default.

The metrics surface is decent; the gaps are tracing (no OTel) and the
absence of P50/P99 *histogram* outputs (llama-server exposes counters and
instantaneous values, not bucketed latency distributions).

---

## 13. What llama.cpp / `llama-server` does BETTER than vLLM

Honesty in both directions. Tessera targets edge + server; vLLM is
datacenter-only. The contrast matters because porting vLLM's design
wholesale would regress the things llama-server does well.

### 13.1 Single-user / low-concurrency latency

vLLM's per-step Python overhead, async IPC, and large minimum batch
economics mean that at concurrency 1, `llama-server` is often *faster*
on the same GPU (Red Hat benchmark: llama.cpp kept P99 ITL extremely low
even at high load). On CPU and Apple Silicon, vLLM is not competitive at
all.

### 13.2 Cold start

`llama-server` is a single static binary. Cold start is sub-second to a
few seconds (model load + KV alloc). vLLM cold start includes Python
interpreter startup, torch import, CUDA context, optional torch.compile
tracing - often tens of seconds. For edge, function-as-a-service, or
autoscaling deployments, this is decisive.

### 13.3 Binary portability and dependencies

One binary, no Python, no pip, no wheel conflicts. `llama-server` runs
on stock macOS / Linux / Windows with no container. vLLM effectively
requires a container or a curated environment.

### 13.4 Hardware breadth

`llama-server` runs on CUDA, Metal, Vulkan, SYCL, CPU, RPC, and (in
Tessera) ANE. vLLM is CUDA-first; ROCm, Intel GPU, TPU, CPU, and Apple
backends exist but are second-class. For an Apple-Silicon-focused fork
like Tessera, Metal + ANE parity with CUDA is a structural advantage.

### 13.5 Memory footprint

llama.cpp's resident size is dominated by the model weights; the runtime
itself is small (~1.8GB cited for some configs). vLLM carries torch,
CUDA, NCCL, and a Python stack - several GB of overhead before the first
weight loads.

### 13.6 Simplicity and auditability

The entire `llama-server` inference path is readable C++ in
`tools/server/`. The scheduler is ~200 lines in `server-queue.cpp`. The
vLLM V1 scheduler is a few thousand lines of Python across many files.
For a team that needs to deeply understand and modify the scheduler
(which is exactly Tessera's situation), the smaller codebase is a real
advantage.

### 13.7 Heterogeneous / RPC compute

llama.cpp's RPC backend lets you mix a Metal Mac, a CUDA box, and a CPU
node into one inference target. vLLM has no analogue - it assumes a
homogeneous CUDA or ROCm cluster.

### 13.8 Things Tessera already does that vLLM does (or does differently)

For completeness, this tree already matches vLLM on:

- Paged attention kernel (Metal + CPU): `tessera_paged_attn`.
- Continuous batching: `update_slots()` iteration loop.
- Prefix caching: radix prompt cache + `kv_unified` block radix.
- Speculative decoding integrated with batching: draft/EAGLE3/MTP/ngram.
- Cache-aware prefill scheduler with a richer cost model than vLLM's FCFS.
- Prometheus metrics endpoint.

---

## 14. Concrete recommendations for a Tessera server

Ordered by impact-to-complexity ratio. The order assumes Tessera's stated
targets (edge + server on ggml) and the existing codebase, not a greenfield
rewrite.

### 14.1 Do these first (high impact, fits ggml)

**Priority 1: Chunked prefill.**

This is the largest serving gap with the cleanest fit. The pathology
(long prompt stalls all decodes for seconds) is real and hurts TTFT
directly. The fix is localized to `pre_decode()` / `update_slots()` in
`tools/server/server-context.cpp`:

- Add a per-iteration prefill token budget (configurable, default ~2048-
  8192 tokens).
- In `pre_decode()`, cap each slot's prefill contribution to the budget;
  leave the remainder for the next iteration.
- Continue admitting decode tokens from other slots up to the budget.

Why this fits ggml: chunked prefill is purely a host-side scheduling
decision. It does not require new kernels, does not touch the graph
executor, and composes naturally with the existing iteration loop and the
`kv_unified` paged KV (each chunk just appends blocks).

Expected impact: removes multi-second TTFT spikes under mixed prefill/
decode load; smooths GPU utilization; modest aggregate throughput gain
(Sarathi's 1.25-1.33x end-to-end). Tessera's existing cache-aware
scheduler score (`server-context.cpp:1692`) already has the right hooks
(prefix positions, attention-weighted suffix cost) to decide *which*
prefill to chunk first.

**Priority 2: Make `kv_unified` + paged KV + block-radix the unconditional
default, with APC-style hash-chain prefix matching.**

The pieces exist (`tessera_paged_attn`, `kv_block_radix`,
`publish_slot_kv_blocks`). What is missing vs vLLM:

- `kv_unified` defaults off (`server.cpp:147` only enables it when
  `n_parallel < 0`). Flipping the default on (with a fallback for
  backends where `tessera_paged_attn` is not implemented) gets the
  memory-efficiency and prefix-sharing benefits in the common case.
- Replace / augment the serialized prompt-snapshot path
  (`server_prompt_cache`) with the block-radix path as the primary
  mechanism, so prefix hits are O(1) reference handoffs rather than O(n)
  snapshot restores. The block radix already does this
  (`server-context.cpp:1863`); promoting it to default removes the slow
  path.
- Add APC-style per-block hash chaining (`parent_hash + token_ids`) so
  prefix matching is constant-time and does not require walking the full
  radix tree on every admission. Tessera's `get_stable_cache_keys` already
  produces stable keys (`server-context.cpp:1734`); the work is to chain
  them rather than re-derive.

Why this fits ggml: it is already half-built. The remaining work is
defaults + a hashing tweak, not new infrastructure.

Expected impact: closes most of the prefix-cache gap with vLLM; raises
effective batch size by removing KV waste (the 60-80% -> <4% number, in
proportion to how much of the workload is prefix-sharing chat/agent
traffic).

**Priority 3: Dynamic slot count / over-subscription with recompute
preemption.**

Today the slot count is fixed at startup. Under bursty load this leaves
KV memory unused (when slots are lightly loaded but all occupied by short
sequences) or queues requests (when slots are full). Two changes:

- Decouple "admitted requests" from "slots." Admit up to a configurable
  cap (e.g. 256), and let `update_slots()` decide per-step which admitted
  requests actually run, based on KV budget.
- Add recompute preemption: when KV is exhausted, free the blocks of the
  lowest-scoring running request (Tessera's scheduler score already
  ranks candidates), reset its computed-token count, prepend it to the
  queue. This is vLLM V1's policy and is simpler than swap.

Why this fits ggml: the scheduler score (`server-context.cpp:1692`) and
the block pool are already there. The work is to (a) lift the slot-count
cap into a soft admission limit and (b) add the preempt path. The
`server_kv_block_radix.release()` call already exists for the block-
release half.

Expected impact: better tail latency under bursts; higher achieved
concurrency on the same hardware. Smaller than Priorities 1-2 but
compound with them.

### 14.2 Do NOT port (low impact-to-complexity for Tessera's targets)

**Disaggregated prefill/decode (Section 5).** Explicitly a latency-SLO
tool for multi-GPU clusters. vLLM's own docs say it does not improve
throughput. On single-node / edge / Apple Silicon targets there is no
second pool to disaggregate onto. The connector framework (NIXL,
Mooncake, LMCache) is large, RDMA-oriented, and not meaningful without
fast interconnect. Skip entirely.

**Tensor / pipeline parallelism (Section 9).** Megatron-style TP needs
all-reduce comms woven into every layer's matmul - a deep change to
ggml's graph executor that contradicts its portability-first design.
For models that fit on one GPU/Metal device, the RPC backend already
covers heterogeneous offload. revisit only if Tessera explicitly targets
multi-GPU datacenter deployment, which contradicts its current
positioning.

**Punica / S-LoRA batched multi-LoRA kernels (Section 10).** Worth it
only for multi-tenant serving with hundreds of adapters. Tessera's
typical workload is zero or one LoRA. The current per-slot adapter
selection is adequate; the cleanup TODO at `server-context.cpp:3114`
(alora handling) is worth doing but does not require new kernels.

**Wholesale async multiprocessing engine (Section 8).** vLLM's
EngineCore-in-separate-process design buys 1.7x throughput by overlapping
Python CPU work with GPU work. Tessera's hot path is already C++ with no
GIL, so the *reason* for the split does not apply. What is worth porting
is the *spirit*: keep tokenization, multimodal chunking, and JSON tool
parsing off the inference-driving thread. That can be done with a worker
thread pool feeding `server_queue`, without ZeroMQ IPC or a second
process.

### 14.3 Order of attack

1. **Chunked prefill** (Section 14.1 P1). Highest single-feature impact,
   most localized change. Ship first.
2. **Defaults flip + APC hash chain** (P2). Make `kv_unified` and the
   block-radix prefix cache the default. Removes the slow serialized-
   snapshot path from the hot path.
3. **Dynamic admission + recompute preemption** (P3). Lifts the slot
   cap. Build on the scheduler score that already exists.
4. **Async host work offload** (Section 14.2 last paragraph). Move
   tokenization / multimodal / tool parsing off the inference thread
   without going multiprocess. Optional; do this if profiling shows
   host-side stalls on big prompts.
5. **Histogram metrics + OpenTelemetry tracing** (Section 12). Add P50/
   P99 buckets for TTFT/ITL and per-request OTel spans. Low-risk,
   high-observability payoff; do once the above are stable.
6. **Do not pursue** disagg, TP/PP, or Punica multi-LoRA unless the
   target deployment changes.

### 14.4 What the resulting Tessera server would look like

After steps 1-3, Tessera would have: paged KV (already present) + APC-by-
default + chunked prefill + dynamic admission with recompute preemption +
continuous batching (already present) + integrated spec decoding (already
present). That is the vLLM V1 feature set, in C++, on the ggml substrate,
without Python, without ZeroMQ, without Ray. The expected throughput
outcome on a single GPU is to close most of the GigaGPU-style 2.7x gap;
the residual gap would be GPU-specific fused-quant kernels (Section 11),
which is a kernel-investment problem, not an architecture problem.

---

## 15. Appendix: verified source-path index

### 15.1 vLLM (V1, current main)

| Component | Path |
|---|---|
| Scheduler | `vllm/v1/core/sched/scheduler.py` |
| Scheduler policies | `vllm/v1/core/sched/request_queue.py` |
| Async scheduler | `vllm/v1/core/sched/async_scheduler.py` |
| KV cache manager | `vllm/v1/core/kv_cache_manager.py` |
| Block pool | `vllm/v1/core/block_pool.py` |
| Block / hash utils | `vllm/v1/core/kv_cache_utils.py` |
| KV coordinator | `vllm/v1/core/kv_cache_coordinator.py` |
| AsyncLLM (API server side) | `vllm/v1/engine/async_llm.py` |
| EngineCore | `vllm/v1/engine/core.py` |
| EngineCore IPC client | `vllm/v1/engine/core_client.py` |
| Executors (uniproc/multiproc/ray) | `vllm/v1/executor/` |
| Spec decode proposers | `vllm/v1/spec_decode/` (eagle, medusa, draft_model, ngram, suffix, dflash, dynamic/) |
| LoRA + Punica | `vllm/lora/`, `vllm/lora/punica_wrapper/`, `vllm/lora/ops/triton_ops/` |
| Disagg KV transfer | `vllm/distributed/kv_transfer/kv_connector/v1/{nixl,offloading_connector}.py` |
| Disagg docs | `docs/features/disagg_prefill.md`, `docs/features/disagg_encoder.md` |
| CUDA attention kernels | `csrc/attention/` |
| Metrics docs | `docs/design/metrics.md` |
| Quantization docs | `docs/features/quantization/` |

### 15.2 Tessera / llama.cpp (this tree, verified by reading)

| Component | Path |
|---|---|
| Server entrypoint | `tools/server/server.cpp` |
| Inference event loop | `tools/server/server-queue.cpp:125` (`server_queue::start_loop`) |
| Slot/batch update | `tools/server/server-context.cpp:3060` (`update_slots`) |
| Batch builder | `tools/server/server-context.cpp:3170` (`pre_decode`) |
| Cache-aware scheduler score | `tools/server/server-context.cpp:1692` (`score_prefill_slot`) |
| Slot admission + radix sharing | `tools/server/server-context.cpp:1746` (`get_available_slot`) |
| KV block publish | `tools/server/server-context.cpp:1729` (`publish_slot_kv_blocks`) |
| Radix prompt cache | `tools/server/server-task.cpp:1638` (`server_prompt_cache`) |
| Continuous batching flag | `tools/server/README.md:178` (`-cb`, default on) |
| Metrics endpoint | `tools/server/server-http.cpp`, `tools/server/server-context.cpp:4827` |
| Speculative types enum | `common/common.h:170` |
| Speculative params | `common/common.h:329-388` |
| Speculative runtime | `src/speculative.cpp`, `tools/server/server-context.cpp:446` (`generate_draft`) |
| Paged attention kernel (Metal) | `ggml/src/ggml-metal/ggml-metal.metal:11240` (`kernel_tessera_paged_attn`) |
| Paged attention op | `ggml/include/ggml.h:2440` (`ggml_tessera_paged_attn`) |
| Paged attention graph use | `src/llama-graph.cpp:3071`, `:3330`, `:3474` |
| Page-map builder | `src/llama-kv-cache.cpp:1307` (`make_page_map`), `:1324` (`build_tessera_block_table`) |
| KV cache unified flag | `common/arg.cpp:1833`, `tools/server/server.cpp:147` (auto-default) |
| RPC backend | `tools/rpc/rpc-server.cpp`, `tools/rpc/README.md` |
| Graceful shutdown | `tools/server/server.cpp:399`, `:446` |
| Server README | `tools/server/README.md` |

### 15.3 Papers cited

| Paper | URL | Key number |
|---|---|---|
| Kwon 2023, PagedAttention (SOSP) | https://arxiv.org/abs/2309.06180 | 60-80% -> <4% KV waste; 2-4x throughput |
| Agrawal 2024, Sarathi-Serve (OSDI) | https://arxiv.org/abs/2308.16369 | up to 10x decode, 1.33x e2e; 6.29x PP bubble reduction |
| Zhong 2024, DistServe | https://arxiv.org/abs/2401.09670 | 7.4x more requests / 12.6x tighter SLO |
| Patel 2024, Splitwise | https://arxiv.org/abs/2311.18677 | 1.4x throughput at 20% lower cost; 2.35x at same cost |
| Chen 2023, S-LoRA | https://arxiv.org/abs/2311.03285 | thousands of adapters; 2-4x throughput |

### 15.4 Benchmarks cited

| Source | URL | Setup / result |
|---|---|---|
| Red Hat, vLLM vs llama.cpp | https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case | H200, Llama-3.1-8B, vLLM v0.10.0 vs llama.cpp b6100; >35x RPS, >44x TPS at peak |
| GigaGPU | https://gigagpu.com/vllm-vs-llama-cpp-gpu-servers/ | 64 users: vLLM ~12k tok/s, llama.cpp ~4.5k tok/s |
| Red Hat, EAGLE3 on vLLM | https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding | up to 2.5x speedup |
| Snowflake Arctic Inference | https://www.snowflake.com/en/blog/engineering/fast-speculative-decoding-vllm-arctic/ | 2.05-2.45x on vLLM V1 |
| AWS P-EAGLE | https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/ | 2-3x (EAGLE); +1.69x (P-EAGLE) |
| Reddit 4-bit benchmark | https://www.reddit.com/r/LocalLLaMA/comments/1q7ysj2 | Marlin 712 tok/s vs FP16 461 vs GPTQ-no-Marlin 276 |
| vLLM V1 blog | https://vllm.ai/blog/2025-01-27-v1-alpha-release | 1.7x throughput vs V0 |
| openlm.ai V1 announce | https://openlm.ai/vllm-v1/ | APC default; unified scheduler; ZeroMQ IPC |

---

End of document.
