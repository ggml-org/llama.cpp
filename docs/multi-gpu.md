# Using multiple GPUs with llama.cpp

This guide explains how to run [llama.cpp](https://github.com/ggml-org/llama.cpp) across more than one GPU. It covers the split modes, the command-line flags that control them, the limitations you need to know about, and ready-to-use recipes for `llama-cli` and `llama-server`.

The CLI arguments listed here are the same for both tools - or most llama.cpp binaries for that matter.

---

## When you need multi-GPU

Reach for multi-GPU when one of these is true:

- **The model doesn't fit in a single GPU's VRAM.** By spreading the weights across two or more GPUs the whole model can stay on accelerators. Otherwise part of the model will need to be run off of the comparatively slower system RAM.
- **You want more throughput.** By distributing the computation across multiple GPUs, each individual GPU has to do less work. This can result in better prefill and/or token generation performance, depending on the split mode and interconnect speed vs. the speed of an individual GPU.

---

## The split modes

Set with `--split-mode` / `-sm`.

| Mode | What it does | When to use |
|---|---|---|
| `none` | Use a single GPU only. Pick which one with `--main-gpu`. | You explicitly want to confine the model to one GPU even though more are visible. |
| `layer` (**default**) | Pipeline parallelism. Each GPU holds a contiguous slice of layers. The KV cache for layer *l* lives on the GPU that owns layer *l*. | Default and most compatible multi-GPU choice. You want more memory than a single GPU provides and your priority is a fast prefill. Can tolerate slow interconnect speeds between GPUs. |
| `row` | **Deprecated.** Older row-split tensor-parallel path with comparatively poor performance. Splits only dense weights across GPUs. Superseded by `tensor` which should be universally superior if it can be used. | Avoid in new deployments. |
| `tensor` | **EXPERIMENTAL.** Tensor parallelism that splits both weights *and* KV across the participating GPUs via a "meta device" abstraction. | You want more memory than a single GPU provides and your priority is fast token generation. Prefill speeds approach pipeline parallel speeds for large, dense models and fast GPU interconnect speeds. Treat as experimental as the code is less mature than pipeline parallelism. Performance should be good for multiple NVIDIA GPUs using the CUDA backend, no guarantees otherwise. |

> Pipeline parallel (`layer`) vs. tensor parallel (`tensor`): pipeline-parallel runs different layers on different GPUs and processes tokens sequentially through the pipeline. This minimizes data transfers between GPUs but requires many tokens to scale well. Tensor-parallel splits each layer across GPUs and does multiple cross-GPU reductions per layer. This enables parallelizing any workload but is much more bottlenecked by the GPU interconnect speed. Pipeline-parallel maximizes batch throughput; tensor-parallel minimizes latency.

---

## Command-line arguments reference

| Short | Long | Value | Default | Notes |
|---|---|---|---|---|
| `-sm` | `--split-mode` | `none` \| `layer` \| `tensor` | `layer` | See modes above. |
| `-ts` | `--tensor-split` | comma-separated proportions, e.g. `3,1` | mode-dependent | How much of the model goes to each GPU. If omitted, `layer`/`row` use automatic splitting proportional to memory, while `tensor` splits tensor segments evenly. With `3,1` on two GPUs, GPU 0 gets 75 %, GPU 1 gets 25 %. The values follow the order in `--device`. |
| `-mg` | `--main-gpu` | integer device index | `0` | The single GPU used in `--split-mode none`. |
| `-ngl` | `--n-gpu-layers` / `--gpu-layers` | integer \| `auto` \| `all` | `auto` | Maximum number of layers to keep in VRAM. Use `999` or `all` to push everything possible to the GPUs. |
| `-dev` | `--device` | comma-separated device names, or `none` | auto | Restrict which devices llama.cpp may use. See `--list-devices` for names. |
| | `--list-devices` | - | - | Print the available devices and their memory. Run this first to learn the names you'd pass to `--device`. |
| `-fa` | `--flash-attn` | `on` \| `off` \| `auto` | `auto` | Required when using `--split-mode tensor` and/or quantized V cache. Supported (and therefore enabled by default) for most combinations of models and backends. |
| `-ctk` | `--cache-type-k` | `f32` \| `f16` \| `bf16` \| `q8_0` \| `q4_0` \| ... | `f16` | KV cache type for K. |
| `-ctv` | `--cache-type-v` | same as `-ctk` | `f16` | KV cache type for V. |
| `-fit` | `--fit` | `on` \| `off` | `on` | Auto-fit unset args to device memory. **Not supported with `tensor`. You may need to manually set the `--ctx-size` to make the model fit.**  |

As for any CUDA program, the environment variable `CUDA_VISIBLE_DEVICES` can be used to control which GPUs to use for the CUDA backend: if you set it, llama.cpp only sees the specified GPUs. Use `--device` for selecting GPUs from among those visible to llama.cpp, this works for any backend.

---

## Recipes

### 1. Default - pipeline parallel across all visible GPUs

```bash
llama-cli -m model.gguf
llama-server -m model.gguf
```

Easiest configuration. KV cache spreads across the GPUs along with the layers. `--fit` (on by default) sizes things automatically.

### 2. Pipeline parallel with a custom split ratio

```bash
llama-cli -m model.gguf -ts 3,1
```

Useful when GPUs have different memory: GPU 0 (3 parts) and GPU 1 (1 part). Proportions are normalized so `-ts 3,1` is the same as e.g. `-ts 75,25`.

### 3. Single-GPU mode, picking a specific GPU

```bash
llama-cli --list-devices
llama-cli -m model.gguf -dev CUDA1
```

Use only the device listed as `CUDA1` when calling with `--list-devices`.

### 4. Tensor parallelism (experimental)

```bash
llama-cli -m model.gguf -sm tensor -ctk f16 -ctv f16
```

- `--flash-attn off` or (`--flash-attn auto` resolving to `off` when it isn't supported) is a hard error.
- KV cache types must be non-quantized: `f32`, `f16`, or `bf16`. Support for quantized KV cache is not implemented and trying to use it will result in an error.
- Mark this configuration as experimental in your tooling: validate output quality before deploying.
- `--split-mode tensor`is not implemented for all architectures. The following will fail with *"LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '...'"*:

  - **MoE / hybrid:** Grok, MPT, OLMoE, DeepSeek2, GLM-DSA, Nemotron-H, Nemotron-H-MoE, Granite-Hybrid, LFM2-MoE, Minimax-M2, Mistral4, Kimi-Linear, Jamba, Falcon-H1
  - **State-space / RWKV-style:** Mamba, Mamba2 (and the hybrid Mamba-attention models above)
  - **Other:** PLAMO2, MiniCPM3, Gemma-3n, OLMo2, BitNet, T5

### 5. Tensor parallelism over USB4 RDMA for Windows

Distributed tensor-parallel LLM inference across 2 or 3 Windows machines joined by USB4 cables.
Every layer of the model is split across the machines, and the partial results are summed with an
all-reduce over Microsoft NetworkDirect (RDMA) on the USB4 links.

This repository holds:

| Path | What |
|---|---|
| `patches/` | 21 patches for llama.cpp's `dist-tp` branch: a NetworkDirect transport for the tensor-parallel all-reduce, full-duplex sends, a 3-rank full-mesh all-reduce, multi-chunk RDMA messages, profiling, removal of the GitHub Actions workflows inherited from llama.cpp, fixes for gpt-oss, Qwen3-Next and Qwen3.5, and Vulkan decode kernels and GPU submission changes for Intel Xe2/Xe3 integrated GPUs |
| `build.cmd` | builds `llama-cli.exe` and its DLLs from scratch with the patches applied |
| `scripts/` | PowerShell scripts, run from a controller machine, that prepare the USB4 links, check the nodes, launch every rank and verify what transport each one actually used |
| `prompts/` | a sample prompt |

The patches apply to `https://github.com/hli25/llama.cpp`, branch `dist-tp`, commit
  (`ea753a291370cf0f4aead6cdf0c52ac5b5515495`),

#### 1. Requirements

**Every node**

- Windows 11 x64 with an Intel GPU and its current graphics driver (the Vulkan backend is used).
- A USB4 host-to-host networking driver that exposes a NetworkDirect provider: the provider DLL
  (`ndprov.dll`) registered with Winsock, and its `ndping.exe` test tool. Not included here.
- OpenSSH Server, and an account in the local Administrators group (the scripts change USB4 adapter
  settings and reboot nodes).
- The Microsoft Visual C++ 2015-2022 x64 Redistributable, or copy `vcruntime140.dll`,
  `vcruntime140_1.dll`, `msvcp140.dll` and `vcomp140.dll` (OpenMP) next to `llama-cli.exe`.
- The model (GGUF) and the prompt at the same path on every node.

**Cables**: one USB4 cable between every pair of nodes that are adjacent in the ring. 2 nodes need
1 cable, 3 nodes need 3.

**Controller**: a separate Windows machine with PowerShell 5.1 and the OpenSSH client that can reach
each node's normal network address on TCP 22. It must not be one of the nodes, because
`prep_fabric.ps1` reboots them.

**Build machine** (any Windows x64 machine, a node is fine):

- Git for Windows
- Visual Studio 2022 or Build Tools for Visual Studio 2022, workload "Desktop development with C++"
  (provides `cl`, `cmake`, `ninja` and `mc`)
- The Vulkan SDK (sets `VULKAN_SDK`)

#### 2. Build

```bat
build.cmd
```

or `build.cmd D:\somewhere` to put the work folder elsewhere. Behind a proxy, set `HTTPS_PROXY`
first. The script:

1. clones `https://github.com/microsoft/NetworkDirect` at `043f1bf05e41343cc88dda619ddd86aee60a02fe`
   and generates `ndstatus.h` from `src\ndutil\ndstatus.mc` with `mc` (the build needs only the
   `ndutil` sources and that header),
2. clones the `dist-tp` source with `core.longpaths=true` (some paths exceed `MAX_PATH`), checks out
   the base commit and applies `patches\*.patch` in order,
3. checks the patched source tree against the expected hash
   (`7708e932312dd107fecfc91849b6e87a4ca57919`),
4. configures and builds:

```bat
cmake -S work\llama.cpp-disttp -B work\build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=ON -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_DIST_TP_ND=ON -DLLAMA_DIST_TP_ND_SDK=<path to NetworkDirect>
cmake --build work\build
```

The configure step must print `llama: NetworkDirect dist-TP transport enabled (...)`. A binary built
without it still runs, over TCP only.

Copy the contents of `work\build\bin` to the same folder on every node (`RunDir` in `nodes.ps1`,
e.g. `C:\disttp`). Every node must run identical binaries; `verify_binaries.ps1` checks this.

#### 3. Set up the nodes

##### 3.1 Address the USB4 legs, one /24 per cable

Give each end of each cable a static IPv4 address on a subnet used by that cable only. With two
cables on one subnet, both ends can claim the same address while the link still reports Up.

```powershell
# on each node: list the USB4 adapters
Get-NetAdapter | Where-Object InterfaceDescription -match 'USB4' | Format-Table Name, ifIndex, Status
# unplug one cable to see which adapter goes down, then address that adapter for that cable
New-NetIPAddress -InterfaceIndex <idx> -IPAddress 192.168.77.1 -PrefixLength 24
```

Only ever change the USB4 adapters. Do not touch the adapter that carries the node's normal network
connection.

##### 3.2 Firewall

On every node, allow inbound TCP 29500-29502 (the rendezvous on rank 0 at `MasterPort`, 29500 by
default, and each rank's listener at `MasterPort + rank`) and TCP 23517-23518 (the NetworkDirect
connection manager, used when a rank accepts an RDMA connection; some providers accept on 23518, the
port after it):

```powershell
New-NetFirewallRule -DisplayName 'dist-tp rendezvous' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 29500-29502 -Profile Private,Domain
New-NetFirewallRule -DisplayName 'dist-tp ND connect' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 23517-23518 -Profile Private
```

With another `MasterPort`, open `MasterPort` to `MasterPort + N - 1` for N ranks. A program rule for
`llama-cli.exe` can replace the first rule, but it is bound to that exact path: a binary run from
any other folder is not admitted, and the run waits at the rendezvous with no error. A blocked
NetworkDirect port only moves the edges it serves to TCP, so check every edge in the transport line
(section 5). `link_throughput.ps1` additionally needs its port (8902 by default) open on the
server's USB4 leg.

##### 3.3 SSH from the controller

```powershell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\llama_cluster_ed25519
```

On each node, append the `.pub` line to `C:\ProgramData\ssh\administrators_authorized_keys` (an
Administrators account does not read `%USERPROFILE%\.ssh\authorized_keys`), then:

```powershell
icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant "*S-1-5-32-544:F" /grant "*S-1-5-18:F"
```

##### 3.4 Describe the cluster

```powershell
Copy-Item scripts\nodes.ps1.template scripts\nodes.ps1
```

Edit `scripts\nodes.ps1` (gitignored). It is the only file a site edits:

| Table | Contents |
|---|---|
| `$CLUSTER_NODES` | each node's name, the address the controller ssh's to, the user; `Readdress` for a node whose USB4 adapters change between boots |
| `$CLUSTER_P2P` | one entry per cable: each end's USB4 address |
| `$CLUSTER_RINGS` | the ring order per rank count, e.g. `2 = 'A,B'`, `3 = 'A,B,C'` |
| `$CLUSTER_CFG` | binary folder, ND tools folder, model, prompt, device, threads, rendezvous port, USB4 adapter match, driver files and their known-good hashes |

Test it:

```powershell
. .\scripts\nodes.ps1; Invoke-Node -N A -Ps -Cmd 'hostname'
```

##### 3.5 Choose the ring order

On each ring edge the lower rank accepts the NetworkDirect connection and the higher rank connects,
and from 3 ranks up each edge carries data one way, rank r to r+1. Test every direction of every
cable:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\nd_all_directions.ps1
```

Each direction must show `Provider returned N addresses` with N > 0 and pass. Put a ring order in
`$CLUSTER_RINGS` in which every edge's connecting node passed against its accepting node, then
confirm it with a real run (a connect test does not stream data).

#### 4. Run

Before every run (it reboots the nodes, one at a time):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify_binaries.ps1 -RankNodes A,B
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\prep_fabric.ps1 -RankNodes A,B
```

Two nodes over RDMA:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_disttp.ps1 -Transport rdma -RankNodes A,B -ExtraArgs '-ub 16'
```

Three nodes, ring all-reduce. Each all-reduce goes round the ring, every rank sending only to the
next (rank r to r+1, the last rank back to rank 0), so each cable carries data in one direction.
Order `-RankNodes` so that every direction the ring uses passed the test in section 3.5:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_disttp.ps1 -Transport rdma -RankNodes A,B,C -ExtraArgs '-ub 16'
```

Three nodes, full-mesh all-reduce. Each all-reduce up to `LLAMA_DIST_TP_MESH_MAX_BYTES` goes to
both peers at once, so the mesh uses both directions of every cable:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_disttp.ps1 -Transport rdma -RankNodes A,B,C -Mesh -ExtraArgs '-ub 16'
```

If one direction of one cable cannot carry ND data, add `-MeshRelay u`: the reverse direction of
ring edge u (rank u+1 to rank u) is then relayed through the third rank instead, and every byte
still goes over RDMA, but each relayed message takes two hops. The run must report
`ranks with a mesh direction on TCP: 0 of 3`.

`-Transport tcp` runs the same thing over TCP on the same cables, and `-Transport none -RankNodes A`
runs a single node.

**Keep the prompt on RDMA.** One all-reduce carries `ubatch x n_embd x 4` bytes. By default
NetworkDirect carries payloads up to 262144 bytes (one ND chunk) and larger ones go over TCP, so
with the default ubatch the prompt goes over TCP and only decode uses RDMA. Either:

- add `-RdmaMultiChunk` (sets `LLAMA_DIST_TP_RDMA_MULTICHUNK=1` and `LLAMA_DIST_TP_RDMA_MAX_BYTES=0`):
  a message may then span any number of ND chunks, so every payload goes over RDMA at the default
  ubatch, or
- keep every prompt all-reduce within one chunk with

      -ub N    with    N = floor(262144 / (4 x n_embd))

  `n_embd` is printed in the model load log (`print_info: n_embd = ...`): 2880 gives 22, 4096
  gives 16, 5120 gives 12, 8192 gives 8. A smaller ubatch processes the prompt in more, smaller GPU
  batches.

The run prints the largest payload it saw against the ND limit (`none` with `-RdmaMultiChunk`).

Three nodes, full mesh, every payload on RDMA, with the Vulkan settings for Intel integrated GPUs
from section 6 passed through `-ExtraEnv`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_disttp.ps1 -Transport rdma -RankNodes A,B,C -Mesh -RdmaMultiChunk -ExtraArgs '-fa on -ctk q8_0 -ctv q8_0' -ExtraEnv 'LLAMA_DIST_TP_RECORD_AHEAD=1;GGML_VK_FENCE_POLL_PAUSES=20;GGML_VK_DEFER_GATE=1;GGML_VK_FWHT=1;GGML_VK_ADD_RMS_FUSION=1'
```

Each run writes every rank's log to `scripts\logs\` and appends a JSON line to
`scripts\logs\results.jsonl`. Add `-Rank0Manual` to start ranks 1..N-1 and print the command for
rank 0, to paste into `cmd.exe` on that node (not PowerShell: there `set X=1` sets no environment
variable). That is how to watch tokens stream; detached ranks buffer their output.

##### Three nodes by hand: direct mesh, all on RDMA

The same run as the `-Mesh -RdmaMultiChunk` example above, without the scripts. "Direct" means no
relay: every rank sends straight to both peers over the two cables it has. Example cabling, one /24
per cable:

| Cable | Addresses |
|---|---|
| rank 0 - rank 1 | rank 0 `192.168.77.1`, rank 1 `192.168.77.2` |
| rank 1 - rank 2 | rank 1 `192.168.78.1`, rank 2 `192.168.78.2` |
| rank 2 - rank 0 | rank 0 `192.168.79.1`, rank 2 `192.168.79.2` |

Rank r dials rank r+1 (mod 3) over the USB4 cable between them, so `LLAMA_DIST_TP_RING_NEXT_HOST`
is rank r+1's address on that cable: `192.168.77.2` on rank 0, `192.168.78.2` on rank 1 and
`192.168.79.1` on rank 2. `--dist-master` is rank 0's own address on a network every rank reaches,
e.g. its normal network address `192.168.1.10`, and the rendezvous port.

On each node, in `cmd.exe` (not PowerShell) in the binary folder:

```bat
set LLAMA_DIST_TP_RDMA=1
set LLAMA_DIST_TP_FULL_DUPLEX=1
set LLAMA_DIST_TP_MESH=1
set LLAMA_DIST_TP_RDMA_MULTICHUNK=1
set LLAMA_DIST_TP_RDMA_MAX_BYTES=0
set LLAMA_DIST_TP_RING_NEXT_HOST=<rank r+1's address on the cable to it>
set LLAMA_DIST_TP_RECORD_AHEAD=1
set GGML_VK_FENCE_POLL_PAUSES=20
set GGML_VK_DEFER_GATE=1
set GGML_VK_FWHT=1
set GGML_VK_ADD_RMS_FUSION=1
llama-cli -m <model> -f <prompt> -n 64 -c 2048 -ngl 99 -t 8 --temp 0 --seed 1234 ^
  --device Vulkan0 -fa on -ctk q8_0 -ctv q8_0 ^
  --split-mode tensor --world-size 3 --world-rank <r> --dist-master 192.168.1.10:29500
```

Add `--no-jinja` for gpt-oss (section 8). Start rank 2, then rank 1, then rank 0: rank 0 must not
wait on peers that have not started. Every variable must be the same on every rank except
`LLAMA_DIST_TP_RING_NEXT_HOST`, and the model, prompt and binaries must be identical. Then check
every rank's log as in section 5: the transport line lists `ND` for both edges and says
`(all payloads over ND)`, and `dist-tp all-reduce: full mesh` is printed.

#### 5. Is it really RDMA, and is it correct?

A run that silently fell back to TCP still completes and prints a plausible rate. `run_disttp.ps1`
checks every rank's log, but know what it checks:

1. Every rank logs `dist-tp transport: RDMA (ND)`. At 3 ranks the line lists each edge, e.g.
   `(->1 ND, 2-> ND)`; any `TCP` there is an edge that fell back.
2. No rank logs `NetworkDirect not established on ring edge`, `falling back to TCP`,
   `no usable NetworkDirect` or `staying on TCP`. `nd_edges_up` equals `nd_edges_expected`.
3. Every `ring edge i->j dials <ip>` is a USB4 address.
4. No payload exceeded the ND limit (see `-RdmaMultiChunk` and `-ub` above).
5. No rank logs `WARN skipping required allreduce`.
6. The generated text is the same as a single-node run of the same model, prompt and seed (the
   `out_sha256` field). Not every model architecture splits correctly in tensor-parallel mode, so
   compare against a single node before trusting a new model.

#### 6. Environment variables

`run_disttp.ps1` sets or clears the `LLAMA_DIST_TP_*` switches below on every rank; pass the others,
and the opt-in ones it only clears, with `-ExtraEnv 'NAME=value;NAME=value'`. For a manual run:

| Variable | Effect |
|---|---|
| `LLAMA_DIST_TP_RDMA=1` | use NetworkDirect for the ring edges; TCP remains for the rendezvous |
| `LLAMA_DIST_TP_RING_NEXT_HOST=<ip>` | the address this rank dials for its ring-next peer: the far end of the USB4 cable to rank+1 |
| `LLAMA_DIST_TP_RDMA_MAX_BYTES` | largest payload sent over ND, default 262144 (one ND chunk); larger ones go over TCP. 0 = no ceiling. Above 262144, or 0, only with `LLAMA_DIST_TP_RDMA_MULTICHUNK=1`: without it multi-chunk ND transfers can deadlock |
| `LLAMA_DIST_TP_RDMA_MULTICHUNK=1` | a message may span any number of ND chunks: a sender blocked at its ND window keeps receiving, so two ranks sending to each other cannot deadlock. With `LLAMA_DIST_TP_RDMA_MAX_BYTES=0` every payload, the prompt included, goes over ND. The first time a sender waits at its window it prints `[dist-tp][nd] multi-chunk messages: a blocked sender keeps receiving` (`-RdmaMultiChunk`) |
| `LLAMA_DIST_TP_RDMA_ADDR=<ip>` | the local ND address, when it must differ from the ring socket's |
| `LLAMA_DIST_TP_RDMA_DEBUG=1` | log each ND listener, connection request, connect and teardown |
| `LLAMA_DIST_TP_FULL_DUPLEX=1` | post each step's send before waiting on its receive |
| `LLAMA_DIST_TP_MESH=1` | 3 ranks: full-mesh all-reduce for payloads up to `LLAMA_DIST_TP_MESH_MAX_BYTES` (default 65536), the ring above. Same sum order as the ring |
| `LLAMA_DIST_TP_MESH_RELAY=u` | with the mesh: relay the reverse direction of ring edge u through the third rank |
| `LLAMA_DIST_TP_MESH_REVERSE_TCP=r,...` | with the mesh: send the reverse direction of the listed ranks' edges over TCP |
| `LLAMA_DIST_TP_MESH_LARGE=1` | with the mesh: also mesh payloads above the ND limit, over TCP |
| `LLAMA_DIST_TP_MESH_ND_LARGE=1` | **experimental, not yet benchmarked.** With the mesh: payloads above `LLAMA_DIST_TP_MESH_MAX_BYTES` (the prompt) also go through the mesh, on ND over both links at once, instead of the ring. Needs `LLAMA_DIST_TP_RDMA=1`, `LLAMA_DIST_TP_RDMA_MULTICHUNK=1` and `LLAMA_DIST_TP_RDMA_MAX_BYTES=0`, and no relay, reverse-TCP or `MESH_LARGE`; otherwise it prints why and the ring is kept. When on, the all-reduce line says `mesh above too (ND, both links at once)`. These payloads have no TCP fallback: a rank whose ND connection is down fails the collective. Same sum order as the ring |
| `LLAMA_DIST_TP_SPLIT_RECURRENT=1` | Qwen3.5 and Qwen3.5-MoE: split the gated DeltaNet (recurrent) layers across the ranks. Unset, they run whole on every rank, which is always the case for Qwen3-Next. Use the same value on every rank |
| `LLAMA_DIST_TP_RECORD_AHEAD=1` | with one GPU per rank (`--device Vulkan0`): a helper thread records the next piece of the graph while the all-reduce runs, and prints `dist-tp record-ahead on`. Needs the Vulkan backend's deferred submit; any other backend prints that the switch is ignored. `LLAMA_DIST_TP_RECORD_AHEAD_SPIN_US` (default 2000) sets how long the helper polls for work before it sleeps |
| `LLAMA_DISTPROF=1` | print one `[distprof]` cost line per graph (GPU wait, time in the collective, bytes) |
| `GGML_TRACE=1` | write an HTML trace of the graph steps to `GGML_TRACE_HTML` (default `llama-trace.html`), bounded by `GGML_TRACE_MAX_STEPS` and `GGML_TRACE_MAX_EVENTS` |

The command line takes `--split-mode tensor --world-size N --world-rank r --dist-master <rank 0
address>:<port>` on every rank.

**Vulkan, Intel Xe2/Xe3 integrated GPUs.** The patches add two decode kernels that are on by default
on the Intel devices the Vulkan backend classifies as Xe2 that have integer dot products and 16-lane
subgroups, and switches that are off unless set. The settings used with the 3-node mesh above are
`GGML_VK_FENCE_POLL_PAUSES=20 GGML_VK_DEFER_GATE=1 GGML_VK_FWHT=1 GGML_VK_ADD_RMS_FUSION=1`, with
`LLAMA_DIST_TP_RECORD_AHEAD=1`.

| Variable | Effect |
|---|---|
| `GGML_VK_XE_MMVQ=0` | turn off the Q2_K/Q3_K decode matrix-vector kernel for Xe2/Xe3 (`GGML_VK_XE_MMVQ_ROWS`, default 8, sets its rows per workgroup) |
| `GGML_VK_FA_DEC_XE=0` | turn off the single-token Q8_0 flash-attention kernel for Xe2/Xe3, used for head size 128, Q8_0 K and V and 16 query heads per KV head (`-fa on -ctk q8_0 -ctv q8_0`). `GGML_VK_FA_DEC_XE_CHUNK=32/64` changes its KV chunk from 128 |
| `GGML_VK_DEFER_GATE=1` | with `LLAMA_DIST_TP_RECORD_AHEAD=1`: the piece recorded ahead goes to the GPU as soon as it is recorded and waits there on an event the CPU sets once the all-reduced data is written back, instead of being submitted only at that point |
| `GGML_VK_FENCE_POLL_PAUSES=N` | check the GPU fence every N CPU pause instructions while waiting (default 1000) |
| `GGML_VK_FWHT=1` | run the Hadamard transform (`MUL_MAT_HADAMARD`) as the subgroup kernel on Intel too; `2` the shared-memory kernel, `0` neither; unset keeps llama.cpp's choice, which leaves Intel out. Compare the output against a run without it on a new GPU or driver |
| `GGML_VK_ADD_RMS_FUSION=1` | fuse the residual add and the RMS norm that follows it on Intel too (llama.cpp leaves it off there); `0` turns it off everywhere |
| `GGML_VK_GRAPH_PROF=N` | diagnostic: every N synchronizes, print a `[vkgprof]` line with where the time from the first graph computed after a synchronize to the end of the next synchronize went (CPU recording before the GPU had work, GPU busy, GPU idle) |

#### 7. Scripts

| Script | What it does |
|---|---|
| `nodes.ps1.template` | the site configuration; copy to `nodes.ps1` |
| `prep_fabric.ps1` | reboot each node and prove it rebooted, MTU 1500 and Private profile on the USB4 legs, re-address `Readdress` nodes, prove every edge connects and carries full-size frames |
| `verify_binaries.ps1` | same `llama-cli`, DLLs, model and prompt on every node; reports or checks the loaded USB4/RDMA driver images |
| `run_disttp.ps1` | one run at 1..N ranks over RDMA or TCP (`-Mesh`, `-RdmaMultiChunk`, `-ExtraEnv` for further switches); analyses every rank's log |
| `distprof.ps1` | log analysis, dot-sourced by `run_disttp.ps1` |
| `nd_pair.ps1` | NetworkDirect test between two nodes in one direction |
| `nd_all_directions.ps1` | `nd_pair.ps1` for both directions of every cable |
| `fix_p2p_legs.ps1` | address a node's USB4 legs by probing the far ends (called by prep) |
| `link_throughput.ps1` | steady-state bulk TCP throughput over one cable |
| `postmortem_fabric.ps1` | after a failed run, before any reboot: USB4 adapter state and counters, TCP and full-size pings on every leg, ND on every cable |

#### 8. Pitfalls

- **USB4 legs can come up at MTU 62000, where bulk TCP stalls after the handshake** while ping,
  ARP, a TCP connect and an ndping test all pass. `prep_fabric.ps1` sets 1500.
- **Network profiles can revert to Public after a boot** and block the rendezvous. prep sets Private.
- **A leg can pass small pings but drop full-size frames.** A run then fails at its first exchange.
  prep pings every leg at 1472 bytes with Don't Fragment and reboots the sender if needed.
- **An OS update can replace the USB4/RDMA driver.** Every run then goes over TCP. Record the
  known-good hashes in `DriverSha256` and run `verify_binaries.ps1`.
- **One large model load per boot.** On integrated GPUs a second load of a large model in the same
  boot can run far slower. prep reboots before every run.
- **Detached processes can be throttled** by Windows EcoQoS on hybrid CPUs. The scripts raise every
  rank to High priority; do the same for anything you start yourself.
- **`--dist-master` must be rank 0's own address**, or every rank waits forever at the rendezvous.
- **A USB4 link can degrade under sustained load** while still reporting Up. Check each cable with
  `link_throughput.ps1` before and after a session; a reboot or re-plug restores it.
- **A run that aborts or stalls does not say which cable failed.** A NetworkDirect receive abort
  (0xC0000241) names only the rank that saw it. Run `postmortem_fabric.ps1 -RankNodes ...` before
  rebooting anything. If pings or TCP fail on a leg, that cable stopped carrying traffic (often only
  full-size frames). If every leg passes them but ND fails on every cable of one node, that node's
  RDMA stack is stuck while its links are up. Rebooting the node clears either case.
- **gpt-oss needs `--no-jinja`.** With the model's own chat template, a reply cut off by `-n` makes
  the chat parser throw, and llama-cli exits with 0xC0000409 before it prints its timings.
  `--no-jinja` uses the built-in template, which does not parse the reply.

### 5. With NCCL

There's no runtime flag for NCCL - it's selected at build time (`-DGGML_CUDA_NCCL=ON`, this is the default). Note that NCCL is **not** automatically distributed with CUDA and you may need to install it manually - when in doubt check the CMake log to see whether or not it can find the package. When llama.cpp is compiled with NCCL support it uses it automatically for cross-GPU reductions in `tensor` mode. When NCCL is missing on a multi-GPU build, you'll see this one-time warning and performance will be lower:

```
NVIDIA Collective Communications Library (NCCL) is unavailable, multi GPU performance will be suboptimal
```

When using the "ROCm" backend (which is the ggml CUDA code translated for AMD via HIP), the AMD equivalent RCCL can be used by compiling with `-DGGML_HIP_RCCL=ON`. Note that RCCL is by default *disabled* because (unlike NCCL) it was not universally beneficial during testing.
### 6. With CUDA peer-to-peer access (`GGML_CUDA_P2P`)

CUDA peer-to-peer (P2P) lets GPUs transfer data directly between each other instead of going through system memory, which generally improves multi-GPU performance. It is **opt-in** at runtime - set the environment variable `GGML_CUDA_P2P` to any value to enable it:

```bash
GGML_CUDA_P2P=1 llama-cli -m model.gguf -sm tensor
```

P2P requires driver support (usually restricted to workstation/datacenter GPUs) and **may cause crashes or corrupted outputs on some motherboards or BIOS configurations** (e.g. when IOMMU is enabled). If you see instability after enabling it, unset the variable.

---

## Troubleshooting

| Symptom | How to fix |
|---|---|
| Startup error *"SPLIT_MODE_TENSOR requires flash_attn to be enabled"* | Add `-fa on` or remove `-fa off`. |
| Startup error *"simultaneous use of SPLIT_MODE_TENSOR and KV cache quantization not implemented"* | Use `-ctk f16 -ctv f16` (or `bf16`/`f32`) with `--split-mode tensor`. |
| Startup error *"LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'X'"* | Architecture not on the TENSOR allow-list. Use `--split-mode layer`. |
| Warning *"NCCL is unavailable, multi GPU performance will be suboptimal"* | llama.cpp wasn't built with NCCL. Either accept the lower performance or install NCCL and rebuild. |
| CUDA OOM at startup or during prefill in `--split-mode tensor` | Auto-fit is disabled in this mode, so reduce memory pressure yourself. In order from least to most disruptive: lower `--ctx-size` (`-c`) (KV cache is roughly proportional to `n_ctx`); for `llama-server`, lower `--parallel` (`-np`) (a slot KV cache is allocated per concurrent sequence); as a last resort, reduce `--n-gpu-layers` (`-ngl`) (the remaining layers run on CPU and inference will be much slower). |
| Performance is worse with multi-GPU than single-GPU | The performance is bottlenecked by GPU interconnect speed. For `--split-mode tensor`, verify that NCCL is being used. Try `--split-mode layer` (less communication than `tensor`). Increase GPU interconnect speed via more PCIe lanes or e.g. NVLink (if available). |
| GPU not used at all | `--n-gpu-layers` is `0` or too low - try explicitly setting `-ngl all`. Or you are accidentally hiding the GPUs via an environment variable like `CUDA_VISIBLE_DEVICES=-1`. Or your build doesn't include support for the relevant backend. |
| Crashes or corrupted outputs after setting `GGML_CUDA_P2P=1` | Some motherboards and BIOS settings (e.g. with IOMMU enabled) don't support CUDA peer-to-peer reliably. Unset `GGML_CUDA_P2P`. |
