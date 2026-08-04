# Inference Engines Comparison Study: What Non-vLLM Servers Do That llama.cpp Does Not

Read-only research study. No code changes. No commits.

> Companion to `docs/ane-backend-deep-study.md`. vLLM is covered by a separate
> study; this document treats it only as a point of comparison. All claims about
> llama.cpp / `llama-server` are verified against the source tree at
> `/Users/user/Developer/GitHub/tessera/tools/server/` (commit at time of study).
> A surprising finding, repeated throughout: **llama.cpp is far less backward
> than the inference-engine literature assumes.** It already ships a block-level
> radix KV cache, continuous batching, chunked prefill with an adaptive quantum
> scheduler, speculative decoding (including MTP), and zero-copy `seq_cp` prefix
> sharing. The real gaps are in (a) C++ scheduler sophistication vs the Python
> engines, (b) fused CUDA kernels, (c) distributed / disaggregated serving, and
> (d) the production "ergonomics" layer (router, autoscaler, KV-cache connector).

## Table of Contents

1. Executive comparison table
2. SGLang - RadixAttention
3. SGLang - speculative decoding and compressed FSMs
4. TensorRT-LLM - inflight batching
5. TGI / NVIDIA Dynamo
6. LMDeploy / TurboMind
7. DeepSpeed-FastGen / Dynamic Splitfuse
8. MLC-LLM compiler approach
9. Cross-cutting techniques and their originators
10. Hardware specialization
11. What llama.cpp / llama-server does that none of these do
12. Concrete recommendations for Tessera
13. Sources

---

## 1. Executive Comparison Table

Engines as rows. Where a cell is uncertain or version-dependent, it is marked.
Throughput numbers are from third-party benchmarks (artificialanalysis.ai,
SemiAnalysis InferenceMAX, aimultiple.com, marktechpost.com); they shift every
quarter and should be treated as order-of-magnitude.

| Engine | Primary differentiator | Batching model | KV cache mgmt | Prefix caching | Spec decoding | Quant formats | Distributed | Language / runtime | License | Hardware targets | Production maturity |
|--------|------------------------|----------------|---------------|----------------|---------------|---------------|-------------|--------------------|---------|------------------|---------------------|
| **SGLang** | RadixAttention (auto prefix reuse) | Continuous + chunked prefill | Paged KV + radix tree on CPU | Radix tree, automatic, ref-counted | EAGLE-2/3, MTP, n-gram, draft models | FP8 (W8A8), INT8, INT4 (AWQ/GPTQ), FP16/BF16 | TP, PP, prefill/decode disaggregation, expert parallel | Python (torch) + C++/CUDA kernels | Apache 2.0 | NVIDIA H100/H200/B100/B200, AMD ROCm (partial) | High; deployed at scale by Anthropic, Meta, DeepSeek-style stacks |
| **TensorRT-LLM** | Inflight batching + NVIDIA-fused plugins | Inflight (continuous + chunked context) | Paged KV cache pools, KV cache reuse manager, KV cache connector API | Block reuse (`enableBlockReuse`), LRU, host offload | EAGLE, Medusa, speculative via plugins | FP8, INT4 AWQ, INT4 GPTQ, SmoothQuant, FP4 (Blackwell), BF16, FP16, NVFP4 | TP, PP, CP, EP, disaggregated (Dynamo) | C++ runtime + Python builder | Apache 2.0 | NVIDIA only (Hopper/Blackwell, GB200/NVL72) | Highest at NVIDIA rack-scale; MLPerf winner |
| **TGI** (Hugging Face) | Rust router + Python shards (now multi-backend) | Continuous batching (engine-dependent) | Delegates to backend (vLLM/TRT-LLM/native) | Native engine had hash-based APC; now backend-dependent | Delegates to backend | BF16, FP8, GPTQ, AWQ (backend-dependent) | TP (shards), RR across shards | Rust (router) + Python (shard) | HFOIL | NVIDIA, AMD (via backends) | **Maintenance mode Dec 2025; repo archived Mar 2026**; HF recommends vLLM/SGLang |
| **NVIDIA Dynamo** | Datacenter-scale orchestrator + KV-aware router | Orchestration layer over engines (TRT-LLM, vLLM, SGLang) | KV Block Manager (KVBM), multi-tier (GPU/CPU/SSD/remote) | KV-cache-aware routing across workers | Delegates to backend | Delegates to backend | Native: prefill/decode disaggregation, NIXL fabric, autoscale | Rust + Python + etcd | Apache 2.0 | NVIDIA (designed for NVL72/GB200) | Production; Baseten reports 2x, AWS/Azure K8s deployments |
| **LMDeploy / TurboMind** | C++/CUDA-native TurboMind engine | Persistent (continuous) batch + dynamic split-and-fuse | Blocked (paged) KV cache | Blocked KV reuse | EAGLE, Medusa, draft models | W4A16, W8A8, FP8, INT4 (AWQ, GPTQ) | TP | C++/CUDA (TurboMind) + Python (PyTorch backend) | Apache 2.0 | NVIDIA (primary), AMD (partial) | High; ~1.8x vLLM throughput in third-party tests |
| **DeepSpeed-FastGen** | Dynamic Splitfuse (Sarathi-style chunked prefill) | Continuous + Splitfuse interleaving | Blocked KV cache | Limited (not radix) | Limited | BF16, FP8 (via DeepSpeed quant) | TP, PP | Python (PyTorch) | MIT | NVIDIA | Niche; technique absorbed by vLLM/SGLang/TRT-LLM |
| **MLC-LLM** | TVM Unity compiler, universal deployment | Single-stream / simple batching (not a high-concurrency server) | Standard KV | Limited | Limited | INT4 (group quant), INT3, FP16, BF16 | Single-device (TP not primary) | Python + TVM (C++/CUDA/Metal/Vulkan/WebGPU) | Apache 2.0 | Apple Metal, AMD/NVIDIA CUDA, Vulkan, WebGPU/WASM, iOS, Android | Edge/mobile focus; not a datacenter server |
| **Ollama** | UX layer over llama.cpp | Inherits llama-server | Inherits llama-server | Inherits llama-server | Inherits llama-server | GGUF quants (Q4_K_M etc.) | Single-node | Go (CLI) + llama.cpp | MIT | Whatever llama.cpp supports | High (consumer); not a perf leader |
| **llamafile** | Cosmopolitan single-file binary over llama.cpp | Inherits llama-server | Inherits llama-server | Inherits llama-server | Inherits llama-server | GGUF quants | Single-node | C/C++ (Cosmopolitan) | Apache 2.0 | Whatever llama.cpp supports (x86/ARM, CPU-focused) | Distribution tool, not a server rewrite |
| **llama.cpp / llama-server (Tessera upstream)** | ggml substrate, every backend, tiny binary | Continuous batching (gated by `cont_batching`), chunked prefill with adaptive quantum | Per-seq KV cells, unified KV mode + block radix | **Yes: serialized prompt-cache radix AND native unified-KV block radix with ref-counted zero-copy `seq_cp`** | **Yes: draft-model, n-gram, prompt-schema, MTP (incl. ANE-MTP)** | GGUF Qx_K_M, Q8_0, Q4_0, F16, BF16, IQ- quants, (FP8 partial) | Single-node tensor split (`LLAMA_SPLIT_MODE_TENSOR`), RPC backend | C/C++ | MIT | **Everything**: CUDA, Metal, Vulkan, SYCL, ROCm (partial), RPC, CPU | High (edge); medium (server) |

Sources: SGLang repo + NeurIPS 2024 paper; NVIDIA TensorRT-LLM docs + blog; HF
TGI architecture docs + maintenance-mode announcement; NVIDIA Dynamo docs;
LMDeploy docs + arXiv 2508.15601; DeepSpeed-FastGen arXiv 2401.08671; MLC-LLM
repo; llama.cpp `tools/server/server-context.cpp`, `server-task.cpp`,
`server-task.h`; SemiAnalysis InferenceMAX; artificialanalysis.ai.

### Key surprise vs the literature

Most external comparisons list llama.cpp as having "no prefix caching" or "no
continuous batching." Both are wrong as of the Tessera tree:

- Prefix caching: two layers. A serialized `server_prompt_cache` with a
  token-keyed radix index (`tools/server/server-task.h:631-690`,
  `find_longest_prefix` at `server-task.cpp:1692`), and a native
  `server_kv_block_radix` for the `kv_unified` mode
  (`server-task.h:696-737`, `server-task.cpp:1920-2029`) that publishes sealed
  32-token blocks, reference-counts owners, and hands them off via
  `common_context_seq_cp` with no K/V copy (`server-context.cpp:1860-1927`).
- Continuous batching: gated by `params_base.cont_batching`
  (`server-context.cpp:3373`); prefill and decode slots are mixed in the same
  `llama_batch`.
- Chunked prefill: `prefill_quantum_for` at `server-context.cpp:1648-1678`
  adaptively shrinks the prefill quantum (divisor 1/2/4/8 by context length
  4096/16384/65536) when interactive decode work is pending, quantized to
  32-token boundaries "to preserve Metal workgroup/page shapes and make a
  future block-KV cache use the same boundary."

The folklore is outdated because these features landed across 2024-2025 and
most comparison posts were written against early-2024 llama.cpp.

---

## 2. SGLang - RadixAttention

### 2.1 The radix tree and why prefix reuse is automatic

RadixAttention (Zheng et al., NeurIPS 2024; arXiv 2312.07104) stores the
mapping from token sequences to KV cache tensors in a **radix tree** (a
compressed/patricia trie). The edges are labeled with token sequences rather
than single tokens, so a long shared prefix collapses to a single edge.

- The tree lives on the **CPU** (cheap pointer chasing, no GPU occupancy).
- The K/V tensors it points to live on the GPU in a **paged layout**, one page
  per token (block size 1 in the original paper; some deployments use larger).
- Lookup walks the tree once per request. For a prompt of length k, the walk
  is **O(k)** in the average case (each token is one hash-table probe at the
  current node) and the worst-case tree maintenance (split, insert) is
  O(k log k) under the rebalancing/child-merge rules. The "O(k log k)" claim in
  the prompt refers to the amortized maintenance cost across a sequence of
  inserts that share prefixes, not a single lookup.
- When a new prompt shares a prefix with an existing node but then diverges,
  the tree **splits** that node so the shared part stays cached and both
  children hang off the split point. This is what makes reuse *automatic*:
  no caller has to declare "cache this prefix."

Source: `python/sglang/srt/mem_cache/radix_cache.py` in sgl-project/sglang.
Key methods (verified via the GitHub source and the dev.to walkthrough):

```
match_prefix(params)   # walk tree, split node on partial match, return KV indices + terminal node
insert(params)         # add tokens + KV indices, splitting existing nodes on overlap
evict(params)          # LRU heap over evictable_leaves; free, delete, recheck parent
inc_lock_ref / dec_lock_ref  # refcount along the path to root; protected_size vs evictable_size
```

The `TreeNode` carries `lock_ref` (active requests using it) and
`host_ref_counter` (protects from eviction), plus `last_access_time` and
`priority` for the eviction strategy. A node is an evictable leaf only if it
is unlocked, not already evicted, and has no live children (`_update_leaf_status`).

Related files: `python/sglang/srt/mem_cache/memory_pool.py` (the two-level
KV tensor pool the radix tree points into), `hiradix_cache.py` (hierarchical
radix for GPU/CPU tiering), `python/sglang/srt/managers/schedule_batch.py`
(scheduling references `_cache.radix_cache`).

### 2.2 RadixAttention vs vLLM hash-based APC

vLLM's original Automatic Prefix Caching (APC) is **hash-based**: each token
position in a sequence hashes to a block key, and a hash map points the key to
a physical KV block. The differences:

| Aspect | vLLM hash APC (original) | SGLang RadixAttention |
|--------|--------------------------|-----------------------|
| Data structure | Hash chain over token positions | Radix tree over token sequences |
| Longest-prefix query | Walk hash chain, stop on first miss | Tree walk, O(k) |
| Insertion of a divergent suffix | New hash entries | Node split, shared prefix retained |
| Tree-shaped reuse (e.g. tree-of-thought, ReAct) | Awkward; manual configuration often needed | Native (the tree is the structure) |
| Eviction | Per-block LRU | Per-leaf LRU with refcount protection |
| Overhead when no hits | Small (hash probes) | Small (tree walk); paper claims "no noticeable overhead" |

When vLLM wins: workloads with near-zero prefix overlap and very high request
rates where the radix tree's pointer chasing costs more than a hash lookup,
and where PagedAttention's mature block allocator is the bottleneck. When
SGLang wins: multi-turn chat, agents with shared system/tool prompts, few-shot
batches, structured generation with shared schemas - i.e. most modern
workloads. Empirically SGLang has led LLMPerf and artificialanalysis on
single-node throughput for these patterns.

Note: vLLM has since adopted a hierarchical prefix cache and the gap has
narrowed, but the radix tree is still the cleaner abstraction for non-linear
reuse.

### 2.3 The structured-generation frontend

SGLang ships a frontend DSL (`sgl.front_end`) where a program is expressed as
`gen()`, `select()`, `fork()`, etc. The runtime maps these onto primitives:
- **Regex / JSON / choices constrained decoding** compiled to a finite state
  machine that masks logits at each step.
- **Fork**: spawn parallel continuations that share the prefix KV - this is
  where RadixAttention pays off, because the forked branches literally share
  tree nodes.
- **Variable-speed gen**: the frontend can ask for `max_tokens`, `stop`,
  `regex`, etc. and the server compiles a sampling+masking plan.

The server integration is what makes this fast: the FSM mask is computed
server-side, in the same process as the KV cache, so there is no per-token
RPC to a separate "constrained decoder" service.

### 2.4 Why SGLang wins benchmarks

Concrete techniques, each tied to a measurable win:

1. **RadixAttention** - 2-5x on prefix-sharing workloads (the paper reports up
   to 6.4x over TGI/vLLM on MMLU/ReAct/ToT/Chat with Llama-7B and
   Mixtral-8x7B).
2. **Compressed FSM** (see section 3) - up to 3x on JSON decoding.
3. **Zero-overhead CPU scheduler** - the scheduling loop is pinned and does
   not contend with GPU work; the scheduler thread does not hold the GIL
   during inference.
4. **Overlap scheduler / CUDA-graph-friendly path** - similar in spirit to
   TRT-LLM's overlap scheduler; hides host-side scheduling behind GPU streams.
5. **FP8 inference path** on Hopper/Blackwell.
6. **Disaggregated prefill/decode** for datacenter scale.

Third-party: aimultiple.com reports SGLang ~16,215 tok/s vs vLLM ~12,553 tok/s
on H100 (about 29% ahead). SemiAnalysis InferenceMAX notes TRT-LLM still wins
at GB200 NVL72 rack scale for FP4 DeepSeek 670B MoE.

Source paths in sgl-project/sglang (main branch):
- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/hiradix_cache.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler.py` (scheduler loop)
- `python/sglang/srt/constrained/` (regex/JSON FSM, xgrammar integration)

---

## 3. SGLang - Speculative and Compressed FSMs

### 3.1 EAGLE-2/3 and MTP

SGLang's spec-decoding roster (as of 2025):
- **EAGLE-2 / EAGLE-3** - autoregressive draft head; reported 2.7-3.5x speedup,
  and roughly double throughput at fixed latency. EAGLE-3 also does
  hidden-state-only drafting (no extra token embedding round-trip).
- **Multi-Token Prediction (MTP)** - the DeepSeek-style multi-token head.
- **Standalone draft models** - load a smaller model as the drafter.
- **N-gram / prompt lookup** - free, works well for RAG and code edit.

SGLang + LMSYS also released **SpecForge** (July 2025) - a training framework
for EAGLE-3 draft heads.

### 3.2 Compressed FSM for structured decoding

The compressed-FSM optimization (lmsys.org blog 2024-02-05) collapses
**deterministic paths** in the regex FSM into a single decoding step. A JSON
schema often forces long runs where only one token is legal (e.g. the literal
`{"type":` prefix). Instead of emitting one token per step through that run,
the compressed FSM lets the engine accept the whole deterministic run in one
forward pass and only spend a real sampling step at the first actual branch.

Result: up to ~3x faster JSON decoding on structured tasks.

Known caveat (GitHub issue #9187): combining **speculative decoding with
xgrammar / compressed-FSM constraints** can hang in some versions - the two
optimizations are largely orthogonal but the integration is fiddly. vLLM has
the same class of problem.

### 3.3 FP8 inference path

SGLang supports W8A8 FP8 (weights and activations) on H100/H200/B100/B200 via
scaled FP8 GEMMs, with online or offline calibration. This roughly doubles
throughput vs FP16 on the same GPU (see section 9.7 for SOTA numbers).

---

## 4. TensorRT-LLM - Inflight Batching

### 4.1 Inflight batching vs continuous batching

The terms are often conflated. The precise distinction (from the TRT-LLM
batch_manager docs and the SqueezeBits scheduler comparison):

- **Continuous batching** (vLLM, the general technique): the engine admits new
  requests into the running batch as soon as KV cache slots free, iteration by
  iteration. Prefill and decode can be mixed (especially with chunked prefill).
- **Inflight batching** (TRT-LLM's term): a specific implementation of
  continuous batching driven by a **C++ scheduler** that also runs an
  **overlap scheduler** - it prepares the next iteration's batch (host-side
  work: token list, KV slot assignment, attention metadata) while the GPU is
  still computing the current iteration. The "inflight" name comes from the
  fact that requests are *in flight* (mid-generation) when they are rescheduled.

The overlap is the real win. vLLM also has overlap scheduling now, so the
practical gap has narrowed, but TRT-LLM's C++ implementation has historically
had lower per-iteration host overhead, which matters at small batch token
counts and short decode lengths.

### 4.2 C++ scheduler vs Python scheduler

TRT-LLM's runtime is C++ (`tensorrt_llm/runtime/`, the `batch_manager`
component). The builder is Python (`tensorrt_llm/`), which emits an engine
(serialized TensorRT network + custom plugins) that the C++ Executor loads.

Tradeoffs:
- C++ scheduler: lower host overhead per iteration, deterministic memory,
  easier to hit hard real-time SLOs, harder to extend.
- Python scheduler (vLLM, SGLang): faster to develop, easier plugin ecosystem,
  torch-native debugging, but the GIL and Python dispatch cost show up at
  small batch sizes / short sequences.

For Tessera: llama-server is already C++, so the "C++ scheduler" advantage is
not a structural gap - the gap is in *what the scheduler does* (see section 12).

### 4.3 Plugin system and hand-fused ops

TRT-LLM is built on TensorRT plugins. The important fused/custom ops:

- **Attention plugin** - the fused multi-head / multi-query / grouped-query
  attention kernel, including paged-context attention and the inflight-batched
  decode kernel. This is where KV cache layout (paged, 128 tokens/block by
  default) meets the scheduler.
- **GEMM plugins** - INT4/INT8/FP8 GEMMs with cuBLAS-style autotuning per
  shape. SmoothQuant folds the activation scale into the GEMM.
- **RMSNorm / LayerNorm plugins** - fused with bias/epsilon.
- **Fused MLP** - gated SiLU + bias + matmul fusion for SwiGLU/Llama-style FFN.
- **MoE plugins** - fused gating + top-k + expert dispatch (matters a lot for
  DeepSeek/Qwen MoE).
- **Custom All-Reduce / TP plugins** overlapped with compute.

This is the layer llama.cpp does not have on CUDA: ggml kernels are general
and correct but not autotuned per shape the way TRT-LLM plugins are.

### 4.4 KV cache reuse manager and KV cache connector

TRT-LLM's KV cache reuse (`enableBlockReuse`, default false) is **hash-based
block sharing**:

- KV cache organized into 128-token blocks (configurable at `trtllm-build`).
- Only **full blocks** can be shared - this is a coarser granularity than
  SGLang's per-token radix tree.
- LRU eviction. Frequent system prompts tend to stay resident.
- **Host memory offload** of reusable blocks to pinned CPU to survive LRU
  pressure, at the cost of PCIe transfer.
- Limitation: a block becomes reusable only after the request that generated
  it **terminates**. If many requests sharing a system prompt are scheduled
  simultaneously, no reuse happens until one finishes.

The newer **KV Cache Connector API** (RFC #14918, v0.20 era) is the plugin
mechanism that separates scheduler-side orchestration from worker-side KV
load/save. External stores (LMCache, MoonCake, NIXL) plug in here - this is
how disaggregated KV transfer across nodes is wired.

### 4.5 Quantization toolkit

TRT-LLM ships the most complete quantization suite in the ecosystem:
- **SmoothQuant** (activation migration, INT8).
- **FP8** (H100/H200/B100/B200), with per-tensor or per-channel scales.
- **INT4 AWQ** and **INT4 GPTQ**.
- **NVFP4 / FP4** on Blackwell.
- **INT8 / INT4 weight-only**.
- BF16, FP16.

This is the reference implementation; everyone else's INT4/FP8 story is
measured against TRT-LLM.

Source paths in NVIDIA/TensorRT-LLM:
- `tensorrt_llm/runtime/` (C++ executor, batch manager, KV cache manager)
- `tensorrt_llm/plugins/` (attention, GEMM, norm, MoE plugins)
- `cpp/tensorrt_llm/kernels/` (CUDA kernels)
- `tensorrt_llm/quantization/` (quant toolkit)
- `docs/source/batch_manager.md`, `docs/source/advanced/kv-cache-reuse.html`

---

## 5. TGI / NVIDIA Dynamo

### 5.1 Naming correction

The prompt conflates two projects. There is **no "Hugging Face dynamo."** The
"dynamo" name in 2025 LLM serving refers to **NVIDIA Dynamo**
(github.com/ai-dynamo/dynamo), a datacenter-scale orchestration framework.
TGI's eventual answer to disaggregation was to wrap other engines
(vLLM, TRT-LLM) rather than build its own orchestrator.

### 5.2 TGI - what the router actually does

TGI is a **Rust router + Python shards** architecture (HF architecture docs):

- **Launcher** - orchestrates startup.
- **Router** (Rust binary) - accepts HTTP (custom API + OpenAI Messages API),
  does **request scheduling, batching, queue management**, and routes over
  gRPC to shards. Its job is to keep shards busy without OOM. Implements
  continuous-batching admission.
- **Shards / Server** (Python) - one per GPU, runs the actual model; results
  stream back over gRPC. Enables tensor-parallel sharding.

The router's cleverness is admission control: it tracks per-shard KV pressure
and queues requests so the engine rarely has to preempt.

### 5.3 Flash-attention integration

TGI was an early production shipper of FlashAttention v2 and v3.
FlashAttention-3 (Tri Dao, 2024) is Hopper-specific and delivers ~1.5-1.75x
over FA2 on H100/H200. TGI integrates FA3 in its shards; vLLM and SGLang have
since caught up (vLLM via neuralmagic/vllm-flash-attention, SGLang via its
own FA paths).

### 5.4 Why TGI lost share - and entered maintenance mode

The trajectory:
1. vLLM's PagedAttention (2023) gave it a large throughput lead; academic
   benchmarks cite up to 24x over TGI on high-concurrency workloads.
2. TGI's Rust router was good but the Python engine underneath fell behind on
   fused kernels, prefix caching, and spec decoding.
3. Late-2024 pivot: TGI v3.0 introduced **multi-backend support** - the Rust
   router could sit on top of vLLM or TRT-LLM instead of TGI's native engine.
4. **December 11, 2025**: Hugging Face announced TGI in **maintenance mode**
   (bug fixes and docs only, no new features).
5. **March 21, 2026**: the GitHub repo was archived read-only.
6. HF now officially recommends **vLLM or SGLang** for new deployments.

Lesson: a router without a competitive engine underneath is a wrapper. The
multi-backend pivot conceded the engine war.

### 5.5 NVIDIA Dynamo (the actual "dynamo")

NVIDIA Dynamo (github.com/ai-dynamo/dynamo) is a **datacenter-scale inference
framework** - an orchestration layer above engines (TRT-LLM, vLLM, SGLang).
Core components:
- **KV-aware Smart Router** - routes requests by evaluating decode cost (active
  blocks) and prefill cost (KV cache overlap) across workers. Baseten reports
  **2x** speedup in production from KV-aware routing alone.
- **KV Block Manager (KVBM)** - multi-tier KV cache (GPU/CPU/SSD/remote).
- **NIXL** - NVIDIA Inference Transfer Library, unifying InfiniBand, NVLink,
  and PCIe fabrics for KV cache transfer between prefill and decode workers.
- **Prefill/decode disaggregation** as a first-class pattern.
- **Operator + autoscaler** built on etcd service discovery.
- **KV Cache Connector** bridges to external stores (LMCache, MoonCake).

Deployments: AWS EKS, Azure AKS, IBM Storage Scale. Production at scale.

For Tessera: Dynamo is too large a system to clone, but the **KV-aware routing
idea is portable** - a router that prefers the worker whose resident KV prefix
maximally overlaps the incoming prompt. This composes naturally with the
block-radix already in llama-server.

---

## 6. LMDeploy / TurboMind

### 6.1 TurboMind's custom CUDA kernels

TurboMind is LMDeploy's C++/CUDA engine (lmdeploy.readthedocs.io; arXiv
2508.15601 for the mixed-precision evaluation). The distinguishing design
choice vs vLLM is implementing **continuous batching, paged KV cache, and
quantization kernels directly in C++/CUDA** rather than going through Python
+ PyTorch. The fused ops include:
- Fused **batched attention** with paged KV (FlashAttention-style).
- Fused **linear** with INT4/INT8/FP8 weight-only or weight-activation quant.
- Fused **norm + activation + bias**.
- Fused **MoE gating + dispatch**.
- Custom **sampling** kernels.

The evaluation paper (arXiv 2508.15601) benchmarks TurboMind across 16 dense
and MoE LLMs and documents the kernel-level design.

### 6.2 Dynamic batching strategy

LMDeploy runs a **persistent (continuous) batch** with **dynamic split-and-fuse**
- i.e. chunked prefill interleaved with decode, the same family as Sarathi /
  DeepSpeed Splitfuse. Combined with the C++ kernels, this is what produces
  the reported ~1.8x over vLLM.

### 6.3 Where LMDeploy beats vLLM, and where it doesn't

Beats vLLM: single-node H100 throughput on the workloads they tune for
(~16,132 tok/s in aimultiple's test, neck-and-neck with SGLang at ~16,215);
C++ kernel efficiency at small batch sizes; tight Triton Inference Server
integration for production.

Doesn't beat vLLM: ecosystem breadth (vLLM has more community-contributed
model implementations and backends); AMD/Intel support; spec decoding variety
(vLLM/SGLang ship more spec strategies); English-language documentation depth
(LMDeploy is OpenMMLab/InternLM, primarily Chinese-community driven).

For Tessera: TurboMind is the closest architectural cousin - a C++ engine
doing its own fused kernels. The lesson is that a C++ server can absolutely
compete with Python engines on throughput *if* the kernel layer is fused and
autotuned. llama.cpp's CUDA kernels are not currently at that level.

---

## 7. DeepSpeed-FastGen / Dynamic Splitfuse

### 7.1 The mechanism

Dynamic SplitFuse (arXiv 2401.08671) is DeepSpeed-FastGen's rebranding of the
**Sarathi-Serve chunked-prefill** idea (arXiv 2403.02310; the Sarathi authors
confirmed the lineage). The core:

- **Long prompts are decomposed into small chunks** and processed across
  multiple forward passes.
- **Only the final chunk performs generation** for that prompt.
- Each iteration targets a fixed **token budget**: short prompts are composed
  to exactly fill the budget; long prompts are sliced to fit.
- The budget keeps the GPU in the **throughput-saturating region** (large
  enough batch) while **never preempting ongoing decodes** - this is the
  "stall-free" / "no preemption" property.

Contrast with early vLLM (pre-chunked-prefill), which did **either** prefill
**or** decode in a forward pass. When a new prompt arrived, decode was
preempted to run prefill, causing latency spikes for in-flight requests.
Splitfuse eliminates that preemption.

### 7.2 Comparison to vLLM chunked prefill

As of 2025, **vLLM ships native chunked prefill** (merged after issues #1562
and #1569). SGLang ships chunked prefill. TRT-LLM ships "chunked context."
So Splitfuse's unique advantage has **dissolved** - it has been universally
absorbed. DeepSpeed-FastGen remains a niche option; the technique outlived the
framework.

### 7.3 Has it been absorbed?

Yes, completely. The Sarathi-Serve chunked-prefill + stall-free-scheduling
idea is now table stakes. Even llama.cpp has it (see section 11.3). What is
**not** universally absorbed: the *quality* of the chunk-size policy (the
budget function) - that is where engines still differ.

---

## 8. MLC-LLM Compiler Approach

### 8.1 TVM / Unity compilation

MLC-LLM (github.com/mlc-ai/mlc-llm) is built on **TVM Unity**, the Apache TVM
compiler stack. The model definition (often in Relax / TVM's IR) is compiled
to platform-native code: Metal, CUDA, Vulkan, WebGPU/WASM, ROCm, Android
(Vulkan/NNAPI), iOS (Metal).

The compilation story:
- Quantization (group-quant INT4/INT3, FP16) is folded into the compiled
  artifact.
- The compiler fuses ops and applies platform-specific scheduling (e.g. iOS
  Metal workgroup tuning, WebGPU bind-group layout).
- Output is a single deployable artifact (`MLCEngine`).

### 8.2 Universal deployment

| Platform | Backend | Status |
|----------|---------|--------|
| iOS / iPadOS | Metal (Apple A/M-series) | Yes |
| Android | Vulkan / NNAPI | Yes |
| Browser | WebGPU + WASM | Yes |
| macOS / Linux / Windows | Metal / CUDA / Vulkan | Yes |

This is the strongest "compile once, run on anything" story in the ecosystem.
For edge/mobile, MLC-LLM is the reference.

### 8.3 Where compilation helps and hurts for serving

Helps: edge deployment (small binary, no Python, tuned kernels); consistent
numerics across platforms; offline optimization of fixed model+shape.

Hurts for *server* serving: compilation is **shape-specific** - a server with
variable batch sizes needs either recompilation per shape bucket or a generic
fallback. Dynamic shapes (variable sequence length, variable batch) are the
norm for high-concurrency serving, and JIT cost is real. MLC-LLM is not a
serious datacenter server; it is an edge/mobile framework that happens to be
able to serve.

For Tessera: the MLC-LLM lesson is "the compiler story matters for edge." If
Tessera's ANE backend (`docs/ane-backend-deep-study.md`) ever ships, it will
look more like MLC-LLM's iOS story than like SGLang's server story.

---

## 9. Cross-Cutting Techniques - Originators and Current State

| Technique | Originator | Current state (2025-2026) | Adopted by |
|-----------|-----------|---------------------------|------------|
| **PagedAttention** | vLLM (Kwon et al., SOSP 2023) | Universal; the default KV memory model | vLLM, SGLang, TRT-LLM, LMDeploy, TGI backends, llama.cpp (ggml cells, different impl) |
| **Continuous batching** | FasterTransformer / Orca (OSDI 2022) pre-vLLM; vLLM popularized it | Universal; every serious server | All |
| **RadixAttention (radix-tree prefix cache)** | SGLang (Zheng et al., NeurIPS 2024) | SGLang-native; vLLM added hierarchical APC; TRT-LLM uses hash block reuse; llama.cpp has its own block radix | SGLang (primary); partial equivalents everywhere |
| **Chunked prefill / stall-free** | Sarathi-Serve (Agrawal et al., OSDI 2024; arXiv 2403.02310) | Universal; rebranded as "Dynamic Splitfuse" by DeepSpeed | vLLM, SGLang, TRT-LLM, LMDeploy, DeepSpeed, llama.cpp |
| **Disaggregated prefill/decode** | DistServe (Zhong et al., OSDI 2024; arXiv 2401.09670) + Splitwise (arXiv 2311.18677) | **Default at datacenter scale** per DistServe retrospective; deployed by DeepSeek (3 prefill + 9 decode nodes), Meta, Amazon | NVIDIA Dynamo, vLLM, SGLang, TRT-LLM, Ray Serve, custom hyperscaler stacks |
| **Speculative decoding** | Medusa (Cai et al. 2024), EAGLE (Li et al. 2024), EAGLE-2/3, MTP (DeepSeek) | EAGLE-3 / MTP ship in vLLM, SGLang, TRT-LLM (plugin), llama.cpp | All major servers |
| **Disaggregated KV cache transfer (RDMA/NVLink)** | DistServe (same-node to minimize transfer); production pushed it cross-node | Practical and heavily optimized: NIXL (NVIDIA, IB+NVLink+PCIe), 3FS (DeepSeek), LMCache, MoonCake | NVIDIA Dynamo, DeepSeek, LMCache integrations |
| **FP8 inference (Hopper/Blackwell)** | H100 hardware (NVIDIA 2022); productionized by TRT-LLM first | Standard; ~2x FP16 throughput where accuracy holds | TRT-LLM, vLLM, SGLang, LMDeploy, TGI backends |
| **Inflight / overlap scheduling** | TRT-LLM (C++ scheduler) | Universalized; vLLM and SGLang now overlap too | TRT-LLM (reference), vLLM, SGLang |
| **Compressed FSM (structured decoding)** | SGLang (LMSYS 2024) | SGLang-native; xgrammar / LLGuidance are the cross-engine grammar backends | SGLang (compressed FSM), vLLM/SGLang/llama.cpp via xgrammar/LLGuidance |
| **Multi-token prediction (MTP) draft head** | DeepSeek-V3 (2024) | Adopted as a spec-decoding strategy | vLLM, SGLang, llama.cpp, TRT-LLM (plugin) |

### 9.1 PagedAttention adoption

The original vLLM paper introduced paged KV memory inspired by OS paging.
Every server now has some form of paged KV. Important nuance: **llama.cpp's
ggml KV cells are not a clone of PagedAttention** - they predate the vLLM
paper and use a different (per-sequence, cell-based) layout. The user-visible
behavior (variable sequence lengths sharing a memory pool) is similar.

### 9.2 RadixAttention adoption

This is the technique most often claimed to be "SGLang-only." Reality:
- SGLang: the canonical radix tree.
- vLLM: hierarchical prefix cache (a different data structure, similar effect).
- TRT-LLM: hash-based block reuse (`enableBlockReuse`), coarser (128-token
  blocks, full-block-only sharing).
- **llama.cpp (Tessera)**: a `server_kv_block_radix` with 32-token blocks,
  ref-counted owners, zero-copy `seq_cp` handoff (`tools/server/server-task.h:696`).
  This is architecturally closer to SGLang than to vLLM's hash APC, though it
  is younger and less battle-tested.

### 9.3 Chunked prefill adoption

Sarathi-Serve's idea is now universal. llama.cpp's implementation is at
`server-context.cpp:1648` (`prefill_quantum_for`) and `:3373` (cont_batching
gate). The adaptive divisor by context length (1/2/4/8 at 4096/16384/65536)
and 32-token quantization is a respectable implementation choice.

### 9.4 Disaggregated prefill/decode - production status

From the DistServe authors' 18-month retrospective (haoailab.com/blogs/distserve-retro):
- The thesis held up: prefill and decode on separate pools, each scaled
  independently, treating TTFT and TPOT as separate SLOs.
- It is now the **default at large scale**. DeepSeek deployed it at
  3 prefill + 9 decode nodes (24 + 72 H100s) for DeepSeek-R1.
- Cross-node KV transfer is practical via NIXL (NVIDIA), 3FS (DeepSeek),
  LMCache, MoonCake.
- Surprise: hardware vendors are now **co-designing chips for P/D
  disaggregation** (decode-specialized ASICs; NVIDIA Rubin CPX rack fabric).
- Attention-FFN disaggregation (AFD), long impractical for dense models,
  becomes near-free for large MoE because the all-to-all communication
  patterns fuse with AFD transfers.

For Tessera: this is firmly "multi-node, later" territory. Single-node P/D
disaggregation is also meaningful (one GPU for prefill, one for decode on the
same machine) and is a plausible post-MVP target.

### 9.5 Speculative decoding - which engines ship it

| Method | vLLM | SGLang | TRT-LLM | LMDeploy | llama.cpp |
|--------|------|--------|---------|----------|-----------|
| EAGLE-2/3 | Yes | Yes | Plugin | Yes | Via draft model |
| Medusa | Yes | Yes | Plugin | Yes | Via draft model |
| MTP (DeepSeek-style) | Yes | Yes | Plugin | - | Yes (incl. ANE-MTP, Tessera-specific) |
| N-gram / prompt lookup | Yes | Yes | - | - | Yes |
| Standalone draft model | Yes | Yes | Yes | Yes | Yes (`common_speculative`) |

llama.cpp's speculative layer (`common/speculative.cpp`, `common/speculative.h`)
supports multiple draft sources and chains drafts through implementations. The
Tessera tree adds `ane-mtp` (Apple Neural Engine MTP head) - see
`tools/ane-mtp/`. This is a unique advantage on Apple Silicon.

### 9.6 Disaggregated KV cache transfer

Status: production at hyperscalers via NIXL / 3FS / LMCache / MoonCake.
Not something a single-node server needs, but the **API shape** (a KV-cache
connector that abstracts load/save) is a good design - TRT-LLM adopted it
(RFC #14918). llama.cpp has `llama_state_seq_get_data_ext` /
`llama_state_seq_set_data_ext` which are the serialization primitives; a
connector layer on top is missing.

### 9.7 FP8 state of the art numbers

Independent 8-GPU benchmark numbers (2025, third-party):

| GPU | Relative LLM inference throughput (FP8) |
|-----|------------------------------------------|
| H100 | Baseline (1x) |
| H200 | 1.83-2.14x H100 (long context) |
| B100 | Between H200 and B200 |
| B200 | up to ~4.87x vs lower-tier GPUs; NVIDIA claims 11-15x vs Hopper per-GPU on favorable workloads |

FP8 roughly doubles throughput vs FP16 where accuracy holds. NVFP4 on
Blackwell pushes further at some accuracy cost. SemiAnalysis InferenceMAX
notes that at GB200 NVL72 rack scale, TRT-LLM on FP4 DeepSeek 670B MoE beats
single-node SGLang by a wide margin - i.e. at the very top end, NVIDIA's
vertical stack (TRT-LLM + NVL72 + FP4) is ahead.

---

## 10. Hardware Specialization

### 10.1 NVIDIA (H100 / H200 / B100 / B200 / GB200 / NVL72)

| Engine | H100/H200 | B100/B200 | GB200 / NVL72 rack |
|--------|-----------|-----------|---------------------|
| TRT-LLM | First-class (reference) | First-class (FP4, NVFP4) | First-class (MLPerf winner) |
| vLLM | First-class | Good | Partial (community) |
| SGLang | First-class | Good | Single-node strong; rack-scale trails TRT-LLM |
| LMDeploy | Good | Partial | Limited |
| DeepSpeed | Good | Partial | Limited |
| llama.cpp | Works (CUDA backend) | Works (not first-class) | Not targeted |

### 10.2 AMD MI300X (ROCm)

- **vLLM**: first-class ROCm support, official AMD docs and blog.
- **SGLang**: partial ROCm support as of 2026.
- **TGI**: via backends.
- **TRT-LLM**: limited ROCm (NVIDIA product).
- **llama.cpp**: ROCm/HIP backend exists, maintained but not the headline
  target.
- Third-party (Moreh vLLM) claims up to 2x over baseline vLLM on AMD.

### 10.3 Intel Gaudi (Habana)

- **vLLM**: via `HabanaAI/vllm-fork` (based on vLLM 0.9.0.1 at time of study),
  Intel-supported.
- Others: limited. Gaudi is a niche accelerator for LLM inference; the
  ecosystem is vLLM-centric.

### 10.4 Apple Silicon (Metal / unified memory) - the llama.cpp opening

This is where the landscape is most lopsided:

| Engine | Apple Silicon support |
|--------|----------------------|
| TRT-LLM | None |
| vLLM | Experimental/none (CPU fallback only; no Metal) |
| SGLang | None (CUDA/ROCm focused) |
| TGI | None |
| LMDeploy | None |
| DeepSpeed | None |
| MLC-LLM | Yes (Metal) - the only non-llama.cpp option with real Apple support |
| Ollama / llamafile | Yes (via llama.cpp) |
| **llama.cpp / llama-server** | **First-class Metal, unified memory, best-in-class on M-series Max** |

For an M2/M3/M4 Max with 38+ GPU cores and 64-192 GB unified memory, llama.cpp
is the only serious high-throughput option. MLC-LLM is the only competitor and
is edge/mobile focused rather than a server. **This is the structural opening
for Tessera**: a Metal-first server that actually competes on concurrency.

Practical Apple Silicon serving concerns:
- **Unified memory**: no PCIe copy between CPU and GPU. Models stay GPU-resident
  for free (no `--n-gpu-layers` cost the way there is on discrete GPUs).
- **Swap avoidance**: macOS will swap under memory pressure and destroy
  inference throughput. The relevant `sysctl` (wired memory clamp) matters more
  than any llama.cpp flag. A server should pin its working set and never exceed
  it.
- **GPU residency sets**: on M-series, the GPU has a fixed share of unified
  memory; going over triggers contention. The KV cache budget must be sized to
  the GPU's share, not total system RAM.
- **Metal workgroup shapes**: the prefill scheduler in llama.cpp already
  quantizes to 32-token boundaries partly for this reason
  (`server-context.cpp:1674`).

### 10.5 Cerebras, Groq, SambaNova (dedicated silicon)

All three are wafer-scale / dedicated inference silicon plays:
- **Cerebras**: WSE, highest tokens/watt in some benchmarks; CS3 systems.
- **Groq**: LPU, deterministic ultra-low-latency, often lowest latency per
  token for streaming; acquired for ~$20B in 2025 headlines.
- **SambaNova**: SN40L, positions as fastest overall in some benchmarks.

These run their own closed stacks. They are not relevant to a ggml-based
Tessera except as proof that **specialized decode silicon** is a real
trajectory - the DistServe retrospective notes that disaggregation is
incentivizing vendors to co-design decode-specialized ASICs.

---

## 11. What llama.cpp / llama-server Does That None of These Do

Verified against `tools/server/` in the Tessera tree.

### 11.1 Runs on essentially everything (the unifying layer)

llama.cpp's backends: CUDA, Metal, Vulkan, SYCL (Intel), ROCm/HIP (AMD, partial),
CPU (everywhere), RPC (distributed), and (Tessera-targeted) Core ML / ANE.
No other engine covers this matrix. vLLM is CUDA + ROCm + Gaudi-fork + CPU.
TRT-LLM is NVIDIA only. SGLang is CUDA + partial ROCm. MLC-LLM is broad but
edge-focused.

For Tessera, the substrate is **ggml** - the same graph executes across all
backends. This is the moat.

### 11.2 Tiny binary, no Python runtime

`llama-server` is a single C++ binary with no Python dependency. TRT-LLM's
builder is Python (runtime is C++). vLLM/SGLang/LMDeploy/DeepSpeed all ship a
Python process with torch, CUDA, etc. - gigabytes of dependencies. This
matters for: edge deployment, containers with small attack surface, embedded
integrations, and startup time.

### 11.3 First-class Apple Silicon / Metal

See section 10.4. llama.cpp is the only server that is genuinely fast on
M-series Max. The unified-memory advantage (no PCIe copy, full GPU residency
for free) is unique to this hardware and llama.cpp is the only engine that
exploits it at the server level.

### 11.4 Edge deployment

Combined with the tiny binary and broad backends, llama.cpp is the default for
on-device inference (laptops, phones via the gguf ecosystem, embedded Linux,
RISC-V, s390x - see `docs/build-riscv64-spacemit.md`, `docs/build-s390x.md`).

### 11.5 AGENTS.md context, MCP, function calling in server

Verified in `tools/server/`:
- **Function / tool calling**: `server-tools.cpp` (1245 lines), `server-tools.h`.
  The server parses tool definitions and emits tool-call grammars.
- **JSON schema / GBNF grammar constrained generation**:
  `server-schema.cpp:251-276` accepts `json_schema` (compiled via
  `json_schema_to_grammar`) or raw `grammar` (GBNF). The `grammar_type` can be
  `tool_calls`, `user`, or output-format.
- **MCP (Model Context Protocol)**: `server-http.cpp:313` reserves threads for
  "MCP and other tasks in the future."
- **AGENTS.md**: Tessera's own context-management layer (see repo docs).

No other inference server ships function calling and structured generation
this tightly integrated at the engine layer. vLLM/SGLang do it via separate
tool-calling parsers; the grammar work (xgrammar, LLGuidance) is shared but
the server integration differs.

### 11.6 The honest tradeoffs

llama-server's real weaknesses (not the folklore ones):

1. **Single-node, single-server-process by default.** Multi-node is via the
   RPC backend (`tools/rpc/`) and tensor split (`LLAMA_SPLIT_MODE_TENSOR`),
   which is single-node multi-GPU. There is no built-in prefill/decode
   disaggregation, no router, no autoscaler.
2. **CUDA kernels are general, not autotuned per-shape.** ggml CUDA kernels
   are correct and reasonably fast but are not the hand-fused,
   shape-autotuned kernels TRT-LLM/TurboMind ship. This is the biggest
   throughput gap on NVIDIA.
3. **The scheduler is single-threaded and local.** `server_queue::start_loop`
   (`server-queue.cpp:125`) is one thread pulling tasks and calling
   `callback_update_slots`. It is a good policy but it is not the multi-worker
   architecture Dynamo/TRT-LLM offer.
4. **No FP8 / FP4 path on Hopper/Blackwell** at the level of TRT-LLM. ggml
   has some FP8 support but not the production W8A8 calibrated path.
5. **Quantization is GGUF-centric.** No native AWQ/GPTQ-on-GPU story at the
   level of the Python engines (Tessera has IQ- quants and the T640/septq
   work, which is its own direction).
6. **The block radix (`kv_block_radix`) is gated behind `--kv-unified`** and
   is not the default. The serialized prompt cache is the default and involves
   a real copy (`llama_state_seq_get_data_ext`), unlike the zero-copy
   `seq_cp` of the unified mode.

The folklore claim that llama.cpp has "no prefix caching, no continuous
batching, no chunked prefill, no speculative decoding" is **wrong** on all
four counts as of this tree. The real gaps are distributed serving, fused
CUDA kernels, and the production control plane.

---

## 12. Concrete Recommendations for Tessera

Tessera constraints recap: ggml substrate; Apple Silicon (Metal + future ANE
via ggml-ane) first-class; Linux/NVIDIA (CUDA) for server deployments;
single-node first, multi-node later.

### 12.1 Which 2-3 techniques to adopt

In priority order, ranked by impact-per-engineering-hour on Tessera's stated
targets:

**Priority 1 - Make the block radix the default (`--kv-unified` on by default).**
The `server_kv_block_radix` already exists
(`tools/server/server-task.h:696`, `server-task.cpp:1920`) and does
zero-copy `seq_cp` prefix sharing. It is currently opt-in. The serialized
prompt cache (`server_prompt_cache`, the default) does a real K/V copy via
`llama_state_seq_get_data_ext` - that is the waste. Flipping the default costs
engineering time in testing, not new code. This is the single highest-leverage
change because RadixAttention-style prefix reuse is the dominant win in
agent/multi-turn workloads, and the mechanism is already written.

**Priority 2 - A real overlap scheduler (TRT-LLM-style).** Today
`server_queue::start_loop` does `pre_decode()` (CPU) then `decode()` (GPU)
then `post_decode()` (CPU) serially. The GPU sits idle during host-side batch
construction. TRT-LLM's win is preparing iteration N+1's batch while iteration
N runs on the GPU. On CUDA this is achievable with a separate "prep" thread +
CUDA streams; on Metal it is achievable with shared event handles (the ANE
study documents `IOSurface + Metal Event` patterns). This is the biggest
throughput win after prefix caching.

**Priority 3 - Fused CUDA kernels for the hot path.** The hot path on NVIDIA
is: fused attention (with paged KV), fused RMSNorm+rotary, fused SwiGLU MLP.
llama.cpp's CUDA backend has these as separate ggml ops; the win from fusing
is real (TurboMind's whole advantage is this). This is the most engineering
effort of the three but it is what closes the gap to TRT-LLM/TurboMind on
NVIDIA. For Apple Silicon, the equivalent is the parametric-kernel /
interleaved-kernel work Tessera is already designing
(`docs/parametric-kernel-design.md`, `docs/interleaved-kernel-design.md`).

Lower priority (do not do first):
- Disaggregated prefill/decode: real win but multi-node; defer.
- EAGLE-3 spec decoding: already supported via draft models; the marginal win
  from native EAGLE-3 vs the existing draft-model path is modest vs the top 3.
- FP8 path: matters on Hopper/Blackwell but is a large kernel project.

### 12.2 RadixAttention (SGLang) vs PagedAttention (vLLM) on ggml

**RadixAttention is a better fit for Tessera than PagedAttention.** Reasons:

1. **The substrate already does per-sequence KV cells**, not OS-style pages.
   `llama_kv_cache` manages cells per sequence with sequence-id sets. The
   block radix Tessera already has (`server_kv_block_radix`, 32-token blocks)
   is a natural fit because it layers on top of the existing cell machinery
   via `common_context_seq_cp`. PagedAttention would require reworking the KV
   memory model to physical blocks + logical-to-physical page tables, which is
   a much larger surgery on ggml.
2. **Implementation complexity.** SGLang's `radix_cache.py` is ~200 lines of
   Python. Tessera's `server_kv_block_radix` is ~110 lines of C++
   (`server-task.cpp:1920-2029`) plus the publish/attach glue in
   `server-context.cpp:1729-1950`. PagedAttention's block allocator +
   attention kernel rewrite is thousands of lines and requires a custom
   attention kernel on every backend - a huge cost on Metal/Vulkan where
   llama.cpp leans on the backend's own attention path.
3. **The workload fit.** Tessera targets agents (AGENTS.md, MCP, function
   calling) - exactly the multi-turn, shared-system-prompt, fork-heavy
   workload RadixAttention was designed for.
4. **The cost is in the attention kernel, not the tree.** The one place
   PagedAttention wins is when the attention kernel itself reads paged KV
   directly (no gather). Tessera can get this benefit incrementally by making
   the Metal/CUDA attention kernels block-radix-aware without adopting the
   full PagedAttention memory model.

**Caveat:** the existing `server_kv_block_radix` is less mature than SGLang's.
It needs: (a) becoming the default, (b) better eviction (current `evict()`
does a linear scan, `server-task.cpp:2004-2029` - fine for thousands of
blocks, not millions), (c) hierarchical tiering (GPU/CPU) like SGLang's
`hiradix_cache.py` and TRT-LLM's host offload.

### 12.3 Apple Silicon server techniques that matter most

For a Metal-first server on M-series Max / Ultra / Studio:

1. **Memory-pressure-aware KV sizing.** Size the KV cache to the GPU's share
   of unified memory, not total system RAM. Exceeding it triggers macOS swap,
   which destroys throughput (orders of magnitude, not percent). A server
   should query the GPU memory budget (`MTLDevice.maxRecommendedWorkingSetSize`
   or the wired-memory limit) and never allocate KV beyond a safety margin.
2. **Swap avoidance via residency pinning.** Use `sysctl vm.compressor_mode`
   and the wired-memory clamps; on the server side, treat the model + KV as a
   hard working set that must stay wired. Tessera's `cache_ram_mib` flag is
   the right hook; it needs a Metal-aware default.
3. **GPU residency sets (fixed working set).** Apple's Metal encourages a
   fixed set of long-lived buffers (weights + persistent KV) that are always
   resident. Dynamic allocation that grows/shrinks the working set causes
   pressure. The block radix should pre-allocate its pool up front (the way
   vLLM/TRT-LLM pre-allocate paged pools) rather than grow on demand.
4. **Overlap via Metal shared events.** The ANE study
   (`docs/ane-backend-deep-study.md` section 4.3) already describes
   `IOSurface + Metal Event` for cross-engine coordination. The same primitive
   lets the prep thread build the next batch while the GPU runs the current
   one. This is the Apple-Silicon analog of the CUDA overlap scheduler.
5. **32-token block alignment.** llama.cpp's prefill quantum already targets
   32-token boundaries (`server-context.cpp:1674`) "to preserve Metal
   workgroup/page shapes." Keep this; it matters for threadgroup efficiency
   on Metal.
6. **ANE offload of the draft head (Tessera-specific).** The `ane-mtp` tool
   (`tools/ane-mtp/`) targets running the MTP draft head on the ANE while the
   main model runs on Metal - effectively free speculation. No other engine
   can do this because no other engine has a serious Apple Silicon path.

### 12.4 Smallest viable server rewrite for 5-10x concurrency

The claim is achievable without rewriting the engine, by rewriting the
*server's scheduling and caching policy*. Concretely:

**Slice 1 (the 80% win, weeks not months):**
- Flip `--kv-unified` to default-on. Wire the block radix into the default
  path so every completion request publishes its sealed blocks and attaches
  on prefix hits via zero-copy `seq_cp`.
- Make the prefill quantum policy (already present) the default even without
  `cont_batching` explicitly set, when more than one slot is active.
- Add an LRU fix to `server_kv_block_radix::evict` (currently O(n) scan).

Expected: 3-5x on agent/multi-turn workloads from prefix reuse alone,
because the serialized-copy path is eliminated.

**Slice 2 (the overlap win, months):**
- Introduce a prep thread that builds iteration N+1's `llama_batch` while
  iteration N runs. On CUDA use streams; on Metal use shared events.
- Move sampling off the inference thread.

Expected: additional 1.5-2x on top of Slice 1, bringing the total into the
5-10x range vs current llama-server on concurrency-bound workloads.

**Slice 3 (the kernel win, longer):**
- Fuse the attention + rotary + RMSNorm path on CUDA and Metal.

This is where throughput gains beyond 10x come from, but it is not required
to hit the 5-10x target.

**What NOT to do first:**
- Do not build a distributed router / autoscaler. Single-node first.
- Do not adopt PagedAttention's memory model. Use the existing cell + block
  radix.
- Do not write a Python control plane. The C++ server is an asset (no GIL, no
  torch dependency, tiny binary); keep it.

The minimal rewrite is therefore a *policy and threading* change on top of
the existing `tools/server/server-context.cpp` and `server-task.cpp`, not a
new engine.

---

## 13. Sources

Primary source code and papers:
- SGLang: github.com/sgl-project/sglang (`python/sglang/srt/mem_cache/radix_cache.py`, `memory_pool.py`, `hiradix_cache.py`, `managers/schedule_batch.py`)
- SGLang paper: Zheng et al., NeurIPS 2024 (arXiv 2312.07104)
- SGLang compressed FSM blog: lmsys.org/blog/2024-02-05-compressed-fsm
- SGLang intro blog: lmsys.org/blog/2024-01-17-sglang
- TensorRT-LLM: github.com/NVIDIA/TensorRT-LLM, docs nvidia.github.io/TensorRT-LLM
- TRT-LLM KV cache reuse: nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html
- TRT-LLM memory usage: nvidia.github.io/TensorRT-LLM/reference/memory.html
- TRT-LLM KV Cache Connector RFC: github.com/NVIDIA/TensorRT-LLM/issues/14918
- TRT-LLM KV reuse blog: developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm
- TGI architecture: huggingface.co/docs/text-generation-inference/en/architecture
- TGI multi-backend: huggingface.co/blog/tgi-multi-backend
- TGI maintenance mode: huggingface.co/docs/inference-endpoints/en/engines/tgi
- NVIDIA Dynamo: github.com/ai-dynamo/dynamo, docs.nvidia.com/dynamo
- Dynamo KV-aware router: docs.nvidia.com/dynamo/user-guides/kv-cache-aware-routing
- LMDeploy: github.com/InternLM/lmdeploy, lmdeploy.readthedocs.io
- LMDeploy TurboMind eval: arXiv 2508.15601
- DeepSpeed-FastGen: arXiv 2401.08671
- Sarathi-Serve: arXiv 2403.02310
- DistServe: arXiv 2401.09670, OSDI 2024 paper usenix.org/system/files/osdi24-zhong-yinmin.pdf
- Splitwise: arXiv 2311.18677
- DistServe retrospective: haoailab.com/blogs/distserve-retro
- MLC-LLM: github.com/mlc-ai/mlc-llm, blog.mlc.ai/2023/05/22/bringing-open-large-language-models-to-consumer-devices

Benchmarks and comparisons:
- artificialanalysis.ai
- SemiAnalysis InferenceMAX: newsletter.semianalysis.com/p/inferencemax-open-source-inference
- aimultiple.com/inference-engines
- marktechpost.com vLLM vs TRT-LLM vs TGI vs LMDeploy (Nov 2025)
- premai.io vLLM vs SGLang vs LMDeploy
- SqueezeBits vLLM vs TRT-LLM scheduler: blog.squeezebits.com/vllm-vs-tensorrtllm-4-which-scheduler-wins--33083
- vLLM vs DeepSpeed notes: vllm.ai/blog/2023-11-14-notes-vllm-vs-deepspeed
- Baseten Dynamo 2x: baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo

Hardware:
- vLLM on AMD MI300X: vllm.ai/blog/2024-10-23-vllm-serving-amd
- AMD ROCm vLLM optimization: rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- Intel Gaudi vLLM fork: github.com/HabanaAI/vllm-fork
- Blackwell vs Hopper: exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus
- Cerebras/SambaNova/Groq: intuitionlabs.ai/articles/cerebras-vs-sambanova-vs-groq-ai-chips

llama.cpp / llama-server source (verified in this study):
- tools/server/server-context.cpp (slot state machine, update_slots, prefill_quantum_for, kv_unified publish/attach, cont_batching)
- tools/server/server-task.cpp (server_prompt_cache, server_kv_block_radix)
- tools/server/server-task.h (structs: server_prompt_cache, server_kv_block_radix)
- tools/server/server-queue.cpp (server_queue::start_loop, single-threaded scheduler)
- tools/server/server-schema.cpp (json_schema / grammar / tool_calls)
- tools/server/server-tools.cpp (function/tool calling)
- tools/server/server-http.cpp (MCP thread reservation)
- common/speculative.h, common/speculative.cpp (speculative decoding layer)
- tools/ane-mtp/ (Tessera ANE MTP draft head)
- src/llama.cpp (LLAMA_SPLIT_MODE_TENSOR, multi-GPU tensor split)

Adjacent Tessera design docs referenced:
- docs/ane-backend-deep-study.md (format reference; ANE + Metal event architecture)
- docs/parametric-kernel-design.md
- docs/interleaved-kernel-design.md
- docs/speculative.md
- docs/multi-gpu.md
