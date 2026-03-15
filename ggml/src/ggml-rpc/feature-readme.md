# Native RDMA Transport for llama.cpp RPC

## Overview

This patch adds native RDMA (Remote Direct Memory Access) transport to llama.cpp's RPC backend, enabling two-node GPU inference clusters to communicate over RoCEv2 (RDMA over Converged Ethernet v2) instead of TCP. RDMA bypasses the kernel network stack entirely, delivering lower latency and higher throughput for the frequent small messages that dominate token generation.

The transport auto-negotiates during the existing RPC HELLO handshake. No special URI scheme or command-line flags are needed -- if both client and server have RDMA-capable NICs, the connection upgrades transparently. If either side lacks RDMA, it falls back to TCP silently.

## Performance

Tested on a two-node cluster with AMD Radeon 8060S (gfx1151) iGPUs connected via Mellanox ConnectX 25GbE NICs running RoCEv2. Model: Qwen3-Coder-Next 80B Q8_K_XL.

| Metric | TCP | RDMA | Improvement |
| ------ | --: | ---: | ----------: |
| Prompt processing (pp2048) | 651.48 t/s | 678.42 t/s | **+4.1%** |
| Token generation (tg256) | 30.19 t/s | 32.16 t/s | **+6.5%** |

The token generation improvement is particularly significant because tg is latency-sensitive -- each token requires a full round-trip between the two nodes. RDMA's 1.5μs latency (vs TCP's 31μs) directly reduces per-token overhead.

## Architecture

### Connection lifecycle

```
Client                              Server
  |                                    |
  |---- TCP connect (host:port) ------>|
  |<--- TCP accept --------------------|
  |                                    |
  | [rdma_probe: find RDMA device,     | [waiting for HELLO]
  |  create QP, get local QPN/GID]     |
  |                                    |
  |---- RPC_CMD_HELLO + RDMA req ----->|
  |     (QPN, PSN, GID or zeros)       |
  |                                    | [rdma_probe: find RDMA device,
  |                                    |  create QP, get local QPN/GID]
  |<--- HELLO rsp + RDMA rsp ---------|
  |     (version + QPN, PSN, GID)      |
  |                                    |
  | [rdma_activate: INIT→RTR→RTS]      | [rdma_activate: INIT→RTR→RTS]
  | [swap fn_send/fn_recv to RDMA]     | [swap fn_send/fn_recv to RDMA]
  |                                    |
  |==== All subsequent data via RDMA ==|
  |     (TCP socket stays open but     |
  |      idle for lifetime mgmt)       |
```

### Key design decisions

**1. HELLO-embedded negotiation**

RDMA parameters (QPN, PSN, GID) are exchanged inside the standard RPC HELLO handshake rather than using a separate pre-HELLO TCP exchange. This means:
- No new wire protocol messages
- The server can distinguish extended HELLO (`input_size == 24`) from legacy (`input_size == 0`)
- Legacy clients/servers work unchanged -- they simply don't send/receive RDMA fields

**2. Function-pointer transport dispatch**

```cpp
struct socket_t {
    sockfd_t fd;
    bool (*fn_send)(socket_t *, const void *, size_t) = tcp_send_impl;
    bool (*fn_recv)(socket_t *, void *, size_t)       = tcp_recv_impl;
    rdma_conn * rdma = nullptr;  // only when compiled with GGML_RPC_RDMA
};
```

Transport is selected once at connection time by swapping function pointers. All call sites use `sock->fn_send(sock, data, size)` -- zero `#ifdef` guards or `if (sock->rdma)` checks on the hot path.

**3. Two-phase RDMA setup**

- `rdma_probe()` -- Before HELLO: opens RDMA device, creates QP (stays in RESET state), allocates and registers buffers. Returns local QPN/GID for the HELLO exchange.
- `rdma_activate()` -- After HELLO: given remote QPN/GID, transitions QP through INIT→RTR→RTS and pre-posts the receive ring.

This split is necessary because both sides need the other's QPN before they can complete the QP state machine.

**4. Auto-detect with override**

`rdma_probe()` uses `getsockname()` on the TCP socket to find the local IP, then scans all RDMA devices' GID tables for a matching IPv4-mapped entry. This provides zero-config operation on most setups. The `GGML_RDMA_DEV` and `GGML_RDMA_GID` environment variables override auto-detection when the network topology is complex (e.g., Linux bridges where the IP may not appear in the expected GID slot).

### RDMA transport internals

The data transport layer uses several optimizations developed through iterative benchmarking:

| Optimization | What it does | Why it matters |
| ------------ | ------------ | -------------- |
| Pre-posted receive ring | 24 receive buffers (256 KiB each) are posted before any data flows | Eliminates RNR (Receiver Not Ready) retries. Without pre-posted buffers, the sender fires faster than the receiver can post, causing 640μs RNR retry delays per event. |
| Separate send/recv CQs | Send completions go to `scq`, receive completions go to `rcq` | Simplifies polling -- `rdma_send` only polls `scq`, `rdma_recv` only polls `rcq`. No need to filter completion types. |
| Inline sends | Messages ≤316 bytes are sent inline (no DMA from registered memory) | RPC command headers and small responses bypass the memcpy-to-registered-buffer step. Most token generation messages are <100 bytes. |
| min_rnr_timer=1 | Sets the RNR retry delay to 0.01ms (the minimum) | Even if an RNR does occur, the retry happens in 10μs instead of the default 640μs. This was the single largest tg improvement. |
| 256 KiB chunk size | Data is sent/received in 256 KiB chunks | Fits within the default Linux locked memory limit (8 MiB). The 24-slot × 256 KiB = 6 MiB receive ring stays under `ulimit -l`. |
| Time-based CQ poll timeout | Uses `clock_gettime(CLOCK_MONOTONIC_COARSE)` with 30s timeout | Replaces spin-loop iteration counting which was inaccurate on fast hardware (Mellanox `ibv_poll_cq` returns in ~10ns). |

### Backwards compatibility

| Scenario | Behavior |
| -------- | -------- |
| New client → old server | Client sends extended HELLO (24 bytes input). Old server treats extra bytes as unknown input, responds with standard 3-byte HELLO. Client sees no RDMA fields, stays on TCP. |
| Old client → new server | Client sends standard HELLO (0 bytes input). Server detects `input_size == 0`, responds with standard HELLO. No RDMA negotiation attempted. |
| New client → new server, no RDMA hardware | `rdma_probe()` returns nullptr. Client sends standard HELLO. Normal TCP operation. |
| New client → new server, RDMA available | Full auto-negotiation. RDMA transport activated after HELLO. |

## Files changed

| File | Changes |
| ---- | ------- |
| `ggml/include/ggml-rpc.h` | Bumped `RPC_PROTO_MINOR_VERSION` from 6 to 7 |
| `ggml/src/ggml-rpc/CMakeLists.txt` | Added `GGML_RPC_RDMA` option with `libibverbs` linking |
| `ggml/src/ggml-rpc/ggml-rpc.cpp` | All transport logic (~550 lines added) |

## Building

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_RPC_RDMA=ON \
  -DAMDGPU_TARGETS=gfx1151

cmake --build build --target rpc-server llama-bench -j$(nproc)
```

Requires `libibverbs-dev` (Ubuntu: `apt install libibverbs-dev`).

Without `-DGGML_RPC_RDMA=ON`, the build produces a standard TCP-only binary with no RDMA code compiled in.
