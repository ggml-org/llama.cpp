# CPU Tensor Parallelism

This is an **experimental** backend feature that shards a model across multiple CPU *ranks* — separate
`llama.cpp` processes, one per NUMA domain (socket) and/or per node — and combines their partial results
with an all-reduce. It targets large models on CPU clusters and multi-socket servers, where a single
process is bottlenecked by **cross-socket memory traffic (UPI/NUMA)** or simply doesn't fit in one
node's RAM.

It is built around two ideas:

1. **rank = NUMA domain.** Each rank is pinned (`numactl --cpunodebind=N --membind=N`) so its weights and
   threads live in the *same* socket's local memory. No model weights cross the inter-socket link; only
   small activation tensors do (via the all-reduce). This alone recovers the large decode throughput
   that a naive 2-socket process loses to UPI.
2. **shard the weights, all-reduce the activations.** Output-/row-parallel matmuls produce *partial*
   results on each rank that are summed across ranks with one all-reduce per layer.

> [!IMPORTANT]
> This is a CPU-only, process-per-rank feature configured through environment variables and built with an
> optional UCX transport. It is independent of the GPU `--split-mode`/`--tensor-split` options described in
> [multi-gpu.md](multi-gpu.md).

## Contents

* [When to use it](#when-to-use-it)
* [Concepts](#concepts)
* [MoE parallel modes](#moe-parallel-modes)
* [Building](#building)
* [Running](#running)
* [Choosing a configuration](#choosing-a-configuration)
* [Limitations](#limitations)
* [Glossary](#glossary)

## When to use it

CPU tensor parallelism helps in two situations:

* **Capacity** — the model is too large for one node's RAM. Sharding the weights across nodes is the only
  way to run it at all.
* **Throughput on multi-socket / multi-node hardware** — decode on CPU is dominated by memory latency and
  (for quantized MoE) by per-expert compute. Keeping each rank NUMA-local and splitting the work across
  ranks can be substantially faster than a single process that spans multiple sockets.

It does **not** help a model that already fits and runs well in a single NUMA domain — there the all-reduce
is pure overhead. See [Choosing a configuration](#choosing-a-configuration).

## Concepts

### Ranks and the all-reduce

A *rank* is one `llama.cpp` process. The world size and this process's id are set with `LLAMA_TP_SIZE` and
`LLAMA_TP_RANK`. Every rank builds the *same* compute graph; the difference is that each rank loads only
its **shard** of the sharded weights and, after the row-parallel matmuls, the ranks exchange and sum their
partial activations:

```
            rank 0 (socket 0)                 rank 1 (socket 1)
          ┌──────────────────┐              ┌──────────────────┐
   x ───▶ │ shard of weights │       x ───▶ │ shard of weights │     (x is replicated)
          │   (NUMA-local)   │              │   (NUMA-local)   │
          └────────┬─────────┘              └────────┬─────────┘
                   │ partial y0                       │ partial y1
                   └───────────────┬──────────────────┘
                                   ▼
                         all-reduce: y = y0 + y1        (small: one activation tensor / layer)
                                   ▼
                       full y on every rank (graph continues)
```

The all-reduce is the only inter-rank communication. Because it moves activations, not weights, its volume
is tiny compared to the weights each rank streams from local memory.

### Transports

The all-reduce uses [UCX](https://openucx.org/). Two transports matter:

* `sm` (shared memory) — ranks on the **same node** (e.g. two sockets of one server).
* `rc` (RDMA, InfiniBand) — ranks on **different nodes**.

Select them with the standard `UCX_TLS` / `UCX_NET_DEVICES` environment variables (see [Running](#running)).
Rank 0 acts as a TCP rendezvous point for exchanging UCX addresses at startup (`LLAMA_TP_PEER` /
`LLAMA_TP_PORT`); the actual data path then uses `sm`/`rc`.

### What gets sharded

| Component | Env flag | How it is split |
|---|---|---|
| Dense FFN (`ffn_gate`/`ffn_up` column-parallel, `ffn_down` row-parallel) | always (when TP enabled) | intermediate dim `n_ff` |
| Attention (`wq`/`wk`/`wv` column, `wo` row) | `LLAMA_TP_ATTN=1` | heads |
| MLA attention (DeepSeek: `wq_b` column, `wk_b`/`wv_b` per-head, `wo` row) | `LLAMA_TP_ATTN=1` | query heads |
| Recurrent SSM / Mamba-2 mixer (`ssm_in` column, scan per-head, `ssm_out` row) | `LLAMA_TP_SSM=1` | SSM heads / `d_inner` |
| MoE routed experts | `LLAMA_TP_MOE=ep\|tp` | see [MoE parallel modes](#moe-parallel-modes) |

The MoE router (`gate_inp`), shared expert, embeddings, and (unless `LLAMA_TP_ATTN=1`) attention are
**replicated** on every rank, so every rank routes identically.

## MoE parallel modes

A Mixture-of-Experts FFN can be parallelized two ways. The right choice depends on the model's **expert
geometry**, and the mode is selectable with `LLAMA_TP_MOE`:

### `ep` — expert parallel

Each rank owns a contiguous slice of the **expert set** (`n_expert / size` whole experts). The router is
replicated, so every rank knows the global top-k; each rank computes only its locally-owned selected
experts and the partials are all-reduced.

* **Scales to many ranks** (up to `n_expert`) → best for **capacity** (huge MoEs that don't fit one node).
* Routing is data-dependent, so the per-rank load is **imbalanced**, and at batch=1 the expert compute is
  not evenly split. EP's win is memory/capacity, not single-stream decode latency.

### `tp` — tensor parallel (over experts)

Each rank holds a `1/size` slice of **every** expert's intermediate dimension `n_ff` — exactly like dense
FFN tensor parallelism (`gate`/`up` column-split, `down` row-split). Every rank computes its `n_ff` slice
of every selected expert; the partials are all-reduced. No index remap is needed.

* **Balanced** and splits the expert compute **even at batch=1** → best for **decode throughput** on
  multi-socket / multi-node hardware.
* Limited by `n_ff` and the quantization block size: the row-parallel `down` split must stay block-aligned
  (e.g. Q4_K's 256-element super-blocks), so `n_ff = 512` caps at 2-way, `n_ff = 2048` allows up to 8-way,
  etc. Models with **large experts** (e.g. Mixtral `n_ff = 14336`) split many ways; **fine-grained** models
  with many tiny experts (e.g. `n_ff = 512`) do not.

| Model style | Example | Experts × `n_ff` | Recommended mode |
|---|---|---|---|
| Coarse (few large experts) | Mixtral 8x7B | 8 × 14336 | `tp` |
| Medium | DeepSeek-V3 | 256 × 2048 | `tp` (≤8-way) or `ep` (capacity) |
| Fine-grained (many tiny experts) | Qwen3 A3B | 256 × 512 | `ep` (capacity); `tp` only 2-way |

The difference, for a MoE layer with the selected experts laid out as `[n_ff × n_expert_used]` per token:

```
        EXPERT parallel (ep)                    TENSOR parallel (tp)
   split the expert SET across ranks       split each expert's n_ff across ranks

      experts ─────────────►                    experts ─────────────►
   n_ff │ r0 r0 │ r1 r1 │ r2 r2 │            n_ff │ r0 r0 r0 r0 r0 r0 │
        │ r0 r0 │ r1 r1 │ r2 r2 │                 │ r1 r1 r1 r1 r1 r1 │
        ▼ r0 r0 │ r1 r1 │ r2 r2 │                 ▼ r2 r2 r2 r2 r2 r2 │
   each rank owns whole experts             every rank owns an n_ff slice of every expert
   (routing decides which are used →        (balanced; splits compute even at batch=1;
    imbalanced; scales to many ranks)         capped by n_ff / quant-block size)
```

## Building

Build with the native CPU backend and point the build at a UCX install (the inter-node transport is
optional; without `TP_UCX_LIB` the all-reduce compiles to a no-op and only single-rank runs are possible):

```bash
cmake -B build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
  -DTP_UCX_INC=/path/to/ucx/include -DTP_UCX_LIB=/path/to/ucx/lib
cmake --build build -j --target llama-cli llama-bench
```

> [!NOTE]
> Tensor-parallel sharding currently requires the model to be loaded **without** repacking
> (`-DGGML_CPU_REPACK=OFF`); repacked weight buffers bypass the shard loader.

## Running

Launch one process per rank, each pinned to its NUMA domain. Tensor parallelism is configured with the
options below (CLI flags, which also accept the matching `LLAMA_ARG_*` environment variables):

| CLI flag | `llama_model_params` field | Meaning |
|---|---|---|
| `--tp-size N` | `tp_size` | number of ranks (power of two; 1 = disabled) |
| `--tp-rank N` | `tp_rank` | this process's rank (`0 … size-1`); rank 0 is the rendezvous |
| `--moe-parallel {none,expert,tensor}` | `moe_parallel` | MoE parallel mode (see above) |
| `--tp-attn` | `tp_attn` | also shard attention (heads) |
| `--tp-peer ADDR` | `tp_peer` | rank 0's address (TCP rendezvous), set on every non-zero rank |
| `--tp-port PORT` | `tp_port` | rendezvous port |
| `UCX_TLS`, `UCX_NET_DEVICES` | — | UCX transport (`sm` intra-node, `rc` + device for IB) |

The transport (`UCX_*`) is selected with UCX's own environment variables. For backward compatibility the
legacy `LLAMA_TP_SIZE` / `LLAMA_TP_RANK` / `LLAMA_TP_MOE` / `LLAMA_TP_ATTN` / `LLAMA_TP_PEER` / `LLAMA_TP_PORT`
variables are still honored when the corresponding option is left at its default.

### One node, two sockets (no InfiniBand)

```bash
M=model.gguf
ARGS="-m $M --tp-size 2 --moe-parallel tensor --tp-port 13900 -t 24 -p ..."
# rank 0 — socket 0
UCX_TLS=sm,self numactl --cpunodebind=0 --membind=0 ./llama-cli $ARGS --tp-rank 0 &
# rank 1 — socket 1
UCX_TLS=sm,self numactl --cpunodebind=1 --membind=1 ./llama-cli $ARGS --tp-rank 1 &
wait
```

### Two nodes over InfiniBand (4 ranks, one per socket)

Rank 0 listens; non-zero ranks connect to rank 0's address. The bootstrap can use the management network
(e.g. ethernet); the data path uses the IB device named in `UCX_NET_DEVICES`.

```bash
# nodeA: ranks 0,1   nodeB: ranks 2,3   (--tp-peer = nodeA's address)
export UCX_TLS=rc,sm,self UCX_NET_DEVICES=mlx5_0:1
A="-m $M --tp-size 4 --moe-parallel tensor --tp-peer 10.0.0.1 --tp-port 13900 -t 24"
# on nodeA:
numactl --cpunodebind=0 --membind=0 ./llama-cli $A --tp-rank 0 &
numactl --cpunodebind=1 --membind=1 ./llama-cli $A --tp-rank 1 &
# on nodeB:
numactl --cpunodebind=0 --membind=0 ./llama-cli $A --tp-rank 2 &
numactl --cpunodebind=1 --membind=1 ./llama-cli $A --tp-rank 3 &
```

For the recursive-doubling all-reduce to keep its first step intra-node, place consecutive ranks on the
same node (ranks 0,1 on nodeA; 2,3 on nodeB).

### SLURM batch job

One task per rank (= per socket). SLURM sets `SLURM_PROCID` (global rank) and `SLURM_LOCALID` (the task's
index on its node, which maps to the socket); rank 0's node is the bootstrap peer.

```bash
#!/bin/bash
#SBATCH --job-name=llama-tp
#SBATCH --nodes=2            # nodes
#SBATCH --ntasks-per-node=2  # ranks per node (one per socket)
#SBATCH --cpus-per-task=24   # cores per socket
#SBATCH --exclusive

MODEL=/path/to/model.gguf
PEER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export UCX_TLS=rc,sm,self UCX_NET_DEVICES=mlx5_0:1

# one process per rank; bind each to its socket for NUMA-local weights
srun --cpu-bind=none bash -c "
  numactl --cpunodebind=\$SLURM_LOCALID --membind=\$SLURM_LOCALID \
    ./llama-cli -m $MODEL \
      --tp-size \$SLURM_NTASKS --tp-rank \$SLURM_PROCID \
      --tp-peer $PEER --tp-port 13900 \
      --moe-parallel tensor --no-mmap -t \$SLURM_CPUS_PER_TASK \
      -p 'The capital of France is'
"
```

> [!TIP]
> Use `-t <cores-per-socket>` (not the whole machine) on each rank, and load with `--no-mmap` (sharded
> reads need it; it is also far faster than mmap when the model sits on a network filesystem).

## Choosing a configuration

* **Pin to NUMA domains.** A single process spanning two sockets pays a large decode penalty to
  cross-socket memory traffic; rank-per-socket avoids it. This is the single most important setting.
* **Pick the MoE mode by expert size.** Large experts → `tp` (splits decode compute, balanced). Many tiny
  experts or a model that must span nodes for capacity → `ep`.
* **Dense models** shard cleanly with the FFN/attention split and scale close to linearly per added NUMA
  domain (every token uses all weights).
* **A model that fits and runs well in one NUMA domain** gains nothing from TP for single-stream latency —
  the all-reduce is overhead. Use TP for capacity or aggregate throughput.

## Indicative results

Decode (`tg`) and prefill (`pp`) throughput in tokens/s, CPU only, `rank = socket`, on a pair of
dual-socket Xeon Skylake-SP nodes (no AVX-512-VNNI) over 100 Gb EDR InfiniBand. **S1** = one socket
(NUMA-local baseline); **S2** = one process spanning both sockets (naive, cross-socket/UPI); **tp-N**/**ep-N**
= tensor/expert parallel over *N* sockets. These illustrate the geometry dependence — they are not peak
numbers (older CPUs, network-filesystem weights).

Qwen3 A3B (256 experts × `n_ff` 512 — *fine-grained*):

| config | sockets | pp | tg |
|---|---|---|---|
| S1 | 1 | 133 | **12.2** |
| S2 (naive 2-socket) | 2 | 172 | 5.8 |
| tp (tensor) | 2 | 151 | 12.0 |
| ep (expert) | 2 | 128 | 10.3 |

DeepSeek-V3.1 (256 experts × `n_ff` 2048 — *medium*):

| config | sockets | pp | tg |
|---|---|---|---|
| S1 | 1 | 11.3 | 2.61 |
| tp (tensor) | 2 | 15.5 | **3.32** |
| ep (expert) | 2 | 11.9 | 2.70 |
| tp (tensor) | 4 | **18.9** | 3.00 |

Takeaways:

* **Pin to NUMA domains.** The naive 2-socket process (S2) is *less than half* the decode of a single
  NUMA-local socket (S1) — cross-socket weight traffic dominates.
* **Tensor mode scales with expert size.** On the fine-grained model its experts are too small to beat one
  socket (tensor ≈ S1, expert < S1). On the larger-expert model tensor mode is the clear winner — **+27%
  decode and +36% prefill** at 2 sockets over S1, and **+66% prefill** at 4 — while expert mode barely
  moves. Expert mode's value is **capacity** (running models that don't fit one node), not single-stream
  speed.

## Limitations

* CPU only; process-per-rank; configured via environment variables (experimental — not yet a stable API).
* `LLAMA_TP_SIZE` must be a power of two (recursive-doubling all-reduce).
* Tensor mode (`tp`) is bounded by `n_ff` / quant block alignment (see [MoE parallel modes](#moe-parallel-modes)).
* Requires `-DGGML_CPU_REPACK=OFF` and `--no-mmap`.
* Attention sharding (`LLAMA_TP_ATTN=1`) requires separate Q/K/V tensors (not a fused QKV). MLA
  models (DeepSeek) are supported: only the query heads are sharded — the single compressed latent
  KV head (`n_head_kv == 1`) and its cache stay replicated on every rank.
* **Hybrid models** (e.g. Mamba/SSM + MoE such as `nemotron_h_moe`) only have their MoE-FFN layers
  sharded; the SSM and attention layers are replicated on every rank. TP runs correctly but only pays off
  if the MoE-FFN is the bottleneck — for SSM/attention-dominated hybrids those layers must also be sharded.
* Both gated (SwiGLU, `gate`+`up`+`down`) and non-gated (`up`+`down`, e.g. relu²) MoEs are supported.

## Glossary

| Term | Meaning |
|---|---|
| **rank** | one `llama.cpp` process participating in tensor parallelism |
| **rank = socket** | placing one rank per NUMA domain, pinned with `numactl`, so weights stay NUMA-local |
| **NUMA domain** | a CPU socket plus its directly-attached memory; cross-domain access goes over UPI/Infinity-Fabric |
| **all-reduce** | summing each rank's partial activation tensor so every rank has the full result |
| **column / row parallel** | splitting a matmul by output rows (`column`) or by the contracted dimension (`row`, requires a final all-reduce) |
| **EP (expert parallel)** | `LLAMA_TP_MOE=ep`: shard the expert *set* across ranks |
| **tensor parallel (experts)** | `LLAMA_TP_MOE=tp`: shard each expert's `n_ff` across ranks |
| **`sm` / `rc`** | UCX shared-memory (intra-node) / RDMA-RC (inter-node, InfiniBand) transports |
