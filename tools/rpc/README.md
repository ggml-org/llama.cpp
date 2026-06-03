## Overview

> [!IMPORTANT]
> This example and the RPC backend are currently in a proof-of-concept development stage. As such, the functionality is fragile and
> insecure. **Never run the RPC server on an open network or in a sensitive environment!**

The `rpc-server` allows exposing `ggml` devices on a remote host.
The RPC backend communicates with one or several instances of `rpc-server` and offloads computations to them.
This can be used for distributed LLM inference with `llama.cpp` in the following way:

```mermaid
flowchart TD
    rpcb<-->|TCP|srva
    rpcb<-->|TCP|srvb
    rpcb<-.->|TCP|srvn
    subgraph hostn[Host N]
    srvn[rpc-server]<-.->dev4["CUDA0"]
    srvn[rpc-server]<-.->dev5["CPU"]
    end
    subgraph hostb[Host B]
    srvb[rpc-server]<-->dev3["Metal"]
    end
    subgraph hosta[Host A]
    srva[rpc-server]<-->dev["CUDA0"]
    srva[rpc-server]<-->dev2["CUDA1"]
    end
    subgraph host[Main Host]
    local["Local devices"]<-->ggml[llama-cli]
    ggml[llama-cli]<-->rpcb[RPC backend]
    end
    style hostn stroke:#66,stroke-width:2px,stroke-dasharray: 5 5
    classDef devcls fill:#5B9BD5
    class local,dev,dev2,dev3,dev4,dev5 devcls
```

By default, `rpc-server` exposes all available accelerator devices on the host.
If there are no accelerators, it exposes a single `CPU` device.

## Usage

### Remote hosts

On each remote host, build the backends for each accelerator by adding `-DGGML_RPC=ON` to the build options.
For example, to build the `rpc-server` with support for CUDA accelerators:

```bash
mkdir build-rpc-cuda
cd build-rpc-cuda
cmake .. -DGGML_CUDA=ON -DGGML_RPC=ON
cmake --build . --config Release
```

When started, the `rpc-server` will detect and expose all available `CUDA` devices:

```bash
$ bin/rpc-server
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
Starting RPC server v3.0.0
  endpoint       : 127.0.0.1:50052
  local cache    : n/a
Devices:
  CUDA0: NVIDIA GeForce RTX 5090 (32109 MiB, 31588 MiB free)
```

You can control the set of exposed CUDA devices with the `CUDA_VISIBLE_DEVICES` environment variable or the `--device` command line option. The following two commands have the same effect:
```bash
$ CUDA_VISIBLE_DEVICES=0 bin/rpc-server -p 50052
$ bin/rpc-server --device CUDA0 -p 50052
```

### Main host

On the main host build `llama.cpp` with the backends for the local devices and add `-DGGML_RPC=ON` to the build options.
Finally, when running `llama-cli` or `llama-server`, use the `--rpc` option to specify the host and port of each `rpc-server`:

```bash
$ llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -ngl 99 --rpc 192.168.88.10:50052,192.168.88.11:50052
```

By default, llama.cpp distributes model weights and the KV cache across all available devices -- both local and remote -- in proportion to each device's available memory.
You can override this behavior with the `--tensor-split` option and set custom proportions when splitting tensor data across devices.

### Local cache

The RPC server can use a local cache to store large tensors and avoid transferring them over the network.
This can speed up model loading significantly, especially when using large models.
To enable the cache, use the `-c` option:

```bash
$ bin/rpc-server -c
```

By default, the cache is stored in the `$HOME/.cache/llama.cpp/rpc` directory and can be controlled via the `LLAMA_CACHE` environment variable.

### RDMA transport

On Linux systems with RoCEv2-capable NICs (e.g. Mellanox ConnectX), the RPC backend can use RDMA instead of TCP for lower latency and higher throughput. The transport is negotiated automatically -- no changes to command-line usage are required.

RDMA is enabled by default when `libibverbs` is found at build time.

### TCP buffer tuning

For high-latency or overlay-network links, you can request larger TCP send and
receive buffers on both the RPC client and server:

```bash
export GGML_RPC_TCP_BUFFER_SIZE=4194304
```

The value is in bytes and is applied to accepted client sockets and outbound RPC
connections. Leave it unset to use the operating-system defaults. Larger buffers
can improve bulk model/tensor transfer on some networks, but they should be
benchmarked with your model, split, and network because they do not remove
per-token synchronization costs.

### Benchmarking changes

The `tools/rpc/bench_rpc_compare.py` helper can compare two local build trees
with the same model and RPC settings:

```bash
python3 tools/rpc/bench_rpc_compare.py \
  --base-bin /path/to/base/build/bin \
  --patch-bin /path/to/patch/build/bin \
  --model /path/to/model.gguf \
  --host 127.0.0.1 \
  --server-device CUDA0 \
  --device RPC0
```

This helper starts both RPC servers locally. Use it to isolate protocol overhead
before drawing deployment conclusions, then repeat the same `llama-bench`
parameters against manually started remote `rpc-server` processes on the real
network. Loopback runs do not model overlay routing, retransmits, slow remote
disks, or accelerator imbalance.

### Troubleshooting

Use the `GGML_RPC_DEBUG` environment variable to enable debug messages from `rpc-server`:
```bash
$ GGML_RPC_DEBUG=1 bin/rpc-server
```
