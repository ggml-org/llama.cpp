## Overview

> [!IMPORTANT]
> This example and the RPC backend are currently in a proof-of-concept development stage. As such, the functionality is fragile and
> insecure. **Never run the RPC server on an open network or in a sensitive environment!**

The `rpc-server` allows  running `ggml` backend on a remote host.
The RPC backend communicates with one or several instances of `rpc-server` and offloads computations to them.
This can be used for distributed LLM inference with `llama.cpp` in the following way:

```mermaid
flowchart TD
    rpcb<-->|TCP|srva
    rpcb<-->|TCP|srvb
    rpcb<-.->|TCP|srvn
    subgraph hostn[Host N]
    srvn[rpc-server]<-.->backend3["Backend (CUDA,Metal,etc.)"]
    end
    subgraph hostb[Host B]
    srvb[rpc-server]<-->backend2["Backend (CUDA,Metal,etc.)"]
    end
    subgraph hosta[Host A]
    srva[rpc-server]<-->backend["Backend (CUDA,Metal,etc.)"]
    end
    subgraph host[Main Host]
    local["Backend (CUDA,Metal,etc.)"]<-->ggml[llama-cli]
    ggml[llama-cli]<-->rpcb[RPC backend]
    end
    style hostn stroke:#66,stroke-width:2px,stroke-dasharray: 5 5
```

Each host can run a different backend, e.g. one with CUDA and another with Metal.
You can also run multiple `rpc-server` instances on the same host, each with a different backend.

## Usage

On each host, build the corresponding backend with `cmake` and add `-DGGML_RPC=ON` to the build options.
For example, to build the CUDA backend with RPC support:

```bash
mkdir build-rpc-cuda
cd build-rpc-cuda
cmake .. -DGGML_CUDA=ON -DGGML_RPC=ON
cmake --build . --config Release
```

Then, start the `rpc-server` with the backend:

```bash
$ bin/rpc-server -p 50052
create_backend: using CUDA backend
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:   no
ggml_cuda_init: CUDA_USE_TENSOR_CORES: yes
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA T1200 Laptop GPU, compute capability 7.5, VMM: yes
Starting RPC server on 0.0.0.0:50052
```

When using the CUDA backend, you can specify the device with the `CUDA_VISIBLE_DEVICES` environment variable, e.g.:
```bash
$ CUDA_VISIBLE_DEVICES=0 bin/rpc-server -p 50052
```
This way you can run multiple `rpc-server` instances on the same host, each with a different CUDA device.


On the main host build `llama.cpp` for the local backend and add `-DGGML_RPC=ON` to the build options.
Finally, when running `llama-cli`, use the `--rpc` option to specify the host and port of each `rpc-server`:

```bash
$ bin/llama-cli -m ../models/tinyllama-1b/ggml-model-f16.gguf -p "Hello, my name is" --repeat-penalty 1.0 -n 64 --rpc 192.168.88.10:50052,192.168.88.11:50052 -ngl 99
```

This way you can offload model layers to both local and remote devices.

