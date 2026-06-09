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

### TCP I/O timeout

By default, RPC TCP sockets use the operating-system blocking I/O behavior. In
long-running distributed experiments this can hide sick or stale endpoints until
the whole run appears to hang. To make outbound client send and receive
operations fail after a fixed interval, set `GGML_RPC_TIMEOUT` on the RPC
client:

```bash
export GGML_RPC_TIMEOUT=30
```

The value is in seconds and is applied after the outbound socket is created; it
does not make the TCP connection attempt itself non-blocking. Leave it unset to
preserve the default blocking behavior.

Servers can also apply send and receive timeouts to accepted client sockets with
`GGML_RPC_SERVER_TIMEOUT`, but keep this unset or much larger than the expected
cold model-load interval for multi-server runs. During a large load, one server
can be idle while the client uploads or hashes tensors for another server, and a
short server-side timeout can close an otherwise healthy connection.

### Cache threshold tuning

When `rpc-server` is started with `-c`, tensors larger than 10 MiB are probed by
hash before upload and saved in the server cache. This avoids resending large
unchanged tensors, but small-model or many-small-tensor workloads can still spend
most wall-clock time uploading weights on every fresh client run. For repeated
experiments on trusted machines, `GGML_RPC_CACHE_MIN_SIZE` can lower the client
probe threshold:

```bash
export GGML_RPC_CACHE_MIN_SIZE=1048576
./build/bin/llama-bench --rpc 192.0.2.10:50052 ...
```

You may also set the same variable on selected `rpc-server` processes when you
want those servers to save smaller tensors. Lower server thresholds can create
many more cache files and increase hashing and disk churn, so change them per
node only after measuring that node. Leaving the server unset keeps the default
10 MiB save threshold while still allowing an experimental client to probe
smaller tensors; cache misses simply fall back to uploading the tensor. Lower
values trade less transfer after the cache is warm for more hashing, cache files,
and small hash-probe RPCs, so keep the default unless real network measurements
show a wall-clock win. Setting the threshold to `0` is the most aggressive option
and can hurt per-token throughput on workloads with many tiny tensors.

RPC buffer types cache successful `GET_ALLOC_SIZE` responses for identical
serialized requests on the client. This avoids repeating thousands of small
synchronous alloc-size round trips during graph and buffer planning while still
querying the server for the first instance of each distinct tensor/source
layout. The cache is capped at 4096 distinct requests per RPC buffer type and
clears itself at the cap. Use `GGML_RPC_TRACE=1` to confirm `GET_ALLOC_SIZE`
call counts before and after a change; this optimization affects setup/load
time, not the remote backend's allocation rules.

### Benchmarking changes

The `llama-rpc-copy-bench` tool measures one RPC tensor-copy path directly,
without running a full model. This is useful when a model trace shows many
layer-boundary or output tensor copies and you need to separate raw copy-path
cost from placement, cache, or compute effects:

```bash
GGML_RPC_TRACE=1 ./build/bin/llama-rpc-copy-bench \
  --src 192.0.2.10:50052/0 \
  --dst 192.0.2.11:50052/0 \
  --bytes 4096,16384,73728,262144,1048576 \
  --pattern random \
  --repetitions 9 \
  --warmup 2
```

Each output line is a JSON object for one tensor size with latency samples,
summary timing, and average MiB/s. The `--src` and `--dst` values use
`ENDPOINT/DEVICE` syntax, where `DEVICE` is the device index exposed by that
`rpc-server`. When both devices are on the same endpoint, the copy can use the
server-side `COPY_TENSOR` path. If the server backend cannot copy directly
between those two local devices, `rpc-server` keeps the fallback local by
reading from the source tensor and writing to the destination tensor on the
server. When endpoints differ, the current RPC backend falls back through the
coordinator. Use `GGML_RPC_TRACE=1` on the client to confirm the actual commands
observed for the benchmarked path.

The optional `--pattern` argument selects the payload used to initialize the
source tensor before copy timing. `random` is the default and uses a
pseudo-random `f32` pattern. `zero` and `ramp` are useful when testing
compression, cache, or transport behavior against predictable payloads. The
selected pattern is included in each JSON output row.

The optional `--verify` flag reads the destination tensor back after each byte
size and checks that it matches the source pattern. This adds extra
`GET_TENSOR` traffic, so leave it off for timing runs and use it only as a
sanity check.

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

For Tailscale or LAN runs where the base and patch `rpc-server` processes are
already running on remote hosts, use `tools/rpc/bench_rpc_remote.py` from the
main host. It runs the local `llama-bench` client against both endpoints and
writes each JSONL stream plus a `summary.json` with the command, selected
environment, endpoint hosts and ports, model, prompt, generation, repetition,
`-ngl`, device, elapsed wall-clock time, and prompt/decode tokens/sec averages,
medians, and standard deviations:

```bash
python3 tools/rpc/bench_rpc_remote.py \
  --base-llama-bench /path/to/base/build/bin/llama-bench \
  --patch-llama-bench /path/to/patch/build/bin/llama-bench \
  --model /path/to/model.gguf \
  --base-rpc 100.64.0.10:50052 \
  --patch-rpc 100.64.0.11:50052 \
  --prompt 512 \
  --gen 128 \
  --repetitions 7 \
  --ngl 99 \
  --device RPC0 \
  --patch-env GGML_RPC_CACHE_MIN_SIZE=1048576
```

Use `--env KEY=VALUE` for settings that should apply to both cases and
`--base-env` or `--patch-env` for one-sided experiments. The summary records the
effective environment for each case, with sensitive-looking values redacted.
Use `--split-mode`, `--flash-attn`, `--tensor-split`,
`--base-tensor-split`, and `--patch-tensor-split` to compare placement
variants without hand-editing the generated `llama-bench` commands. For
options not modeled directly by the helper, use `--bench-extra`,
`--base-bench-extra`, or `--patch-bench-extra` with a shell-style argument
string. If the extra argument string starts with `-`, pass it with the equals
form, for example `--bench-extra=--no-mmap`, so `argparse` does not treat it as
an option for the helper.

The remote helper only connects to endpoints passed with `--base-rpc` and
`--patch-rpc`; it does not SSH to hosts or start/stop servers. Use `--dry-run`
to write the planned commands without connecting. Use `--env KEY=VALUE` to set
and record client environment variables that apply to both cases, such as
`GGML_RPC_TCP_BUFFER_SIZE`; use `--base-env` or `--patch-env` for one-sided
knobs such as `GGML_RPC_CACHE_MIN_SIZE`. Use `--llama-bench` when both cases
should use the same local client binary, or `--base-llama-bench` and
`--patch-llama-bench` when you want an end-to-end base-client/base-server versus
patch-client/patch-server comparison. For warmed-cache or one-sided experiments,
use `--only base` or `--only patch` with the matching endpoint and binary.

Layer split placement can have a large effect on RPC decode throughput. With
`--split-mode layer`, the output tensor is placed like one additional layer
after the repeating transformer layers. If the final split point lands on a
slow endpoint, every decoded token may read the final logits from that endpoint
and may also add an extra cross-endpoint layer-boundary copy. Use
`LLAMA_LOG_MODEL_PLACEMENT=1` to confirm the output device and
`GGML_RPC_TRACE=1` to check whether `result_output` or layer-boundary tensors
dominate wait time.

For example, this compares a proportional split with a variant that gives the
last endpoint zero split weight so the output slot lands on the previous
endpoint:

```bash
RPC=192.0.2.10:50052,192.0.2.11:50052,192.0.2.12:50052
BASE_TS=24576/24576/24576
PATCH_TS=24576/24576/0

python3 tools/rpc/bench_rpc_remote.py \
  --llama-bench /path/to/build/bin/llama-bench \
  --model /path/to/model.gguf \
  --base-rpc "$RPC" \
  --patch-rpc "$RPC" \
  --prompt 128 \
  --gen 128 \
  --repetitions 5 \
  --ngl 99 \
  --split-mode layer \
  --base-tensor-split "$BASE_TS" \
  --patch-tensor-split "$PATCH_TS" \
  --env LLAMA_LOG_MODEL_PLACEMENT=1
```

Treat this as a measured tuning step, not a universal rule. Removing a slow
tail endpoint can improve decode latency, but it also changes model placement
and may reduce prompt throughput or fail to fit on another model or cluster.
Always compare prompt and decode tokens/sec on the real network after the RPC
server caches are warm enough for your workload.

For generation workloads that only need the next token, backend sampling can
avoid reading the full final logits tensor back to the coordinator each token.
`llama-bench` can measure this path with `-bs` / `--backend-sampling`, which
attaches a bounded top-k/top-p/temp/dist backend sampler for tg and pg tests:

```bash
GGML_RPC_TRACE=1 ./build/bin/llama-bench \
  --rpc "$RPC" \
  --device RPC0/RPC1/RPC2 \
  --tensor-split "$TS" \
  --n-prompt 0 \
  --n-gen 128 \
  --backend-sampling
```

Use this only when the application does not need host-side full-vocabulary
logits for every token. If logits or logprobs are part of the workload, compare
against the default raw-logits path as a separate benchmark.

For larger RPC setups, `tools/rpc/suggest_rpc_placement.py` can read a
`GGML_RPC_TRACE=1` stderr file and emit candidate `--tensor-split` values for
`tools/rpc/bench_rpc_remote.py`. The helper is intentionally conservative: it
does not claim a split is faster, it only ranks benchmark candidates from
observed `result_output`, `l_out-*`, and cross-endpoint copy timing.

```bash
python3 tools/rpc/suggest_rpc_placement.py \
  --trace-stderr baseline.stderr.txt \
  --rpc "$RPC" \
  --device RPC0/RPC1/RPC2 \
  --tensor-split "$BASE_TS" \
  --llama-bench /path/to/build/bin/llama-bench \
  --model /path/to/model.gguf \
  --flash-attn on \
  --env LLAMA_LOG_MODEL_PLACEMENT=1
```

When a single `rpc-server` exposes multiple devices, pass
`--device-endpoints RPC0=host-a:50052,RPC1=host-a:50052,...` so the report can
group candidate risk and trace cost by endpoint. The generated commands should
still be run and compared with `bench_rpc_remote.py`; the trace-derived ranking
is only a way to reduce the number of manual placement experiments.

Pass one or more previous sweep summaries with `--fit-summary` to avoid
placements that already failed to load the model or allocate memory. The
rejected candidates are still reported in `rejected_candidates` so the planner's
decision can be audited:

```bash
python3 tools/rpc/suggest_rpc_placement.py \
  --trace-stderr baseline.stderr.txt \
  --rpc "$RPC" \
  --device RPC6/RPC1/RPC3/RPC5 \
  --device-endpoints "$DEVICE_ENDPOINTS" \
  --tensor-split 1/8/8/8 \
  --fit-summary outputs/trace-suggested-zero-suffix/summary.json \
  --fit-summary outputs/tail-taper/summary.json
```

Use `--spill-device TARGET=SPILL_DEVICE` to ask the planner for same-endpoint
spill candidates. This inserts `SPILL_DEVICE` immediately after `TARGET` and
moves a small split weight from the next downstream device onto it. The intended
case is a single `rpc-server` exposing both a GPU and a CPU, where a small CPU
spill may change a costly downstream boundary without adding another network
endpoint:

```bash
python3 tools/rpc/suggest_rpc_placement.py \
  --trace-stderr baseline.stderr.txt \
  --rpc "$RPC" \
  --device RPC6/RPC1/RPC3/RPC5 \
  --device-endpoints "$DEVICE_ENDPOINTS" \
  --tensor-split 1/8/8/8 \
  --fit-summary outputs/trace-suggested-zero-suffix/summary.json \
  --spill-device RPC3=RPC4 \
  --spill-weight 1/2
```

Spill candidates are emitted with `bench_rpc_sweep_arg` values because they can
change both `--device` and `--tensor-split`. Run them with
`tools/rpc/bench_rpc_sweep.py` on the real network before adopting them; a spill
that wins in one noisy discovery run can regress once cache state, prompt size,
or node load changes.

Use `--sweep-device-weight DEVICE=WEIGHTS` to generate candidates that keep the
same device order but change one existing positive split weight. This is useful
when one RPC endpoint exposes a GPU and a CPU and the question is not whether to
add a new spill device, but how much split weight the already-listed devices
should receive:

```bash
python3 tools/rpc/suggest_rpc_placement.py \
  --trace-stderr baseline.stderr.txt \
  --rpc "$RPC" \
  --device RPC0/RPC1 \
  --device-endpoints RPC0=host-a:50052,RPC1=host-a:50052 \
  --tensor-split 1/1 \
  --sweep-device-weight RPC0=1/2/4/8/16
```

The generated candidates only propose what to benchmark. A heavier GPU-side
split can improve decode throughput on one model while hurting another model's
fit, prompt throughput, or output placement, so keep the real-network
`bench_rpc_sweep.py` comparison as the source of truth. Weight-sweep candidates
are marked as not trace-estimated because the trace cannot predict whether a
different positive split weight will improve the real decode path.

To run several suggested splits in one pass, use
`tools/rpc/bench_rpc_sweep.py`. It writes one JSONL/stderr pair per candidate,
a `summary.json`, and grep-friendly `ranked.csv` and `ranked.jsonl` files:

```bash
python3 tools/rpc/bench_rpc_sweep.py \
  --llama-bench /path/to/build/bin/llama-bench \
  --model /path/to/model.gguf \
  --rpc "$RPC" \
  --device RPC0/RPC1/RPC2 \
  --split-mode layer \
  --flash-attn on \
  --candidate baseline="$BASE_TS" \
  --candidate tail_on_rpc1=24576/24576/0 \
  --candidate tail_on_rpc0=24576/0/0 \
  --prompt 128 \
  --gen 128 \
  --repetitions 5 \
  --env LLAMA_LOG_MODEL_PLACEMENT=1
```

Each `--candidate` is a single benchmark case. Use `NAME=auto` to omit `-ts`
for that case. Comma-separated `-ts` grids are rejected so each result row maps
to one placement candidate.

Use `--candidate-device NAME=DEVICE:TENSOR_SPLIT` when the benchmark should
also compare device order or CPU/GPU spillover choices. This is useful when one
`rpc-server` exposes both a GPU and a CPU, because `--device` order changes
where the first, middle, and output-side layer groups land:

```bash
python3 tools/rpc/bench_rpc_sweep.py \
  --llama-bench /path/to/build/bin/llama-bench \
  --model /path/to/model.gguf \
  --rpc "$RPC" \
  --split-mode layer \
  --prompt 0 \
  --gen 128 \
  --repetitions 3 \
  --candidate-device cpu_only=RPC6/RPC0/RPC2/RPC4:1/8/8/8 \
  --candidate-device gpu_spill_head=RPC6/RPC1/RPC3/RPC5:1/8/8/8 \
  --bench-extra=--backend-sampling
```

Use `--candidate-device-rotate NAME=DEVICE:TENSOR_SPLIT` to generate one case
per rotation of the device list. The tensor split rotates with the devices, so
each listed device gets tested as the output-side owner while keeping the same
per-device split weight. Add `--keep-going` for fragile sweeps where some
placements may fail to fit:

```bash
python3 tools/rpc/bench_rpc_sweep.py \
  --llama-bench /path/to/build/bin/llama-bench \
  --model /path/to/model.gguf \
  --rpc "$RPC" \
  --split-mode layer \
  --prompt 0 \
  --gen 128 \
  --repetitions 3 \
  --candidate-device-rotate gpu_output=RPC6/RPC1/RPC3/RPC5:1/8/8/8 \
  --bench-extra=--backend-sampling \
  --keep-going
```

When benchmarking CUDA RPC servers across different GPU generations, build for
the remote GPUs as well as the local client GPU. For example, an RTX 3070 server
and RTX 4060 client need both architectures:

```bash
cmake -B build-cuda -DGGML_CUDA=ON -DGGML_RPC=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86-real;89-real"
```

If the server log reports `no kernel image is available for execution on the
device`, rebuild with the missing remote architecture before comparing RPC
throughput.

### Async same-server copies

Set `GGML_RPC_COPY_TENSOR_ASYNC=1` on the RPC client to send same-server
RPC-to-RPC tensor copies as one-way `COPY_TENSOR_ASYNC` commands. This avoids a
client-side response wait at layer or backend split boundaries while preserving
server-side command order on the same RPC connection.

The optimization is used only when source and destination tensors are RPC
buffers on the same server and the server advertises support during `HELLO`.
Other copy paths keep the synchronous `COPY_TENSOR` behavior.

Use `GGML_RPC_TRACE=1` to confirm `COPY_TENSOR_ASYNC` commands and see which
response-bearing command drains the queued work.

### RPC tracing

Set `GGML_RPC_TRACE=1` on the RPC client to print a summary of RPC command
counts, input bytes, output bytes, one-way send time, and request/response wait
time when the RPC backend is freed:

```bash
GGML_RPC_TRACE=1 ./build/bin/llama-bench --rpc 192.0.2.10:50052 ...
```

The client summary also reports command timing by endpoint, the slowest traced
tensor operations by endpoint and tensor name, and cross-endpoint tensor-copy
fallbacks. The tensor table reports total, average, p50, p95, and max client-side
elapsed time, so one-way tensor uploads include local send time but not remote
handler completion. Fallbacks are copies that cannot use the same-server
`COPY_TENSOR` RPC path and must fall back through the main host. These tables are
useful for separating raw transport cost from placement effects such as a final
output tensor being read back from a slow endpoint or many layer-boundary copies
being routed through the coordinator.

The client trace also prints `sync waits after one-way commands` when a
synchronous command such as `GET_TENSOR` or `GET_ALLOC_SIZE` waits after earlier
one-way commands on the same connection. This table attributes the response wait
to pending `SET_TENSOR`, `SET_TENSOR_ZLIB`, `GRAPH_COMPUTE`, and
`GRAPH_RECOMPUTE` calls, and to `COPY_TENSOR_ASYNC` when that experimental path
is enabled, so a long `GET_TENSOR` wait is easier to distinguish from pure
readback time. The `wait_ms` values are diagnostic hints rather than additive
totals: if one synchronous response drains multiple pending command types, the
same wait can appear on more than one row. Failed response waits are reported
separately as `fail_syncs` and `fail_wait_ms`, so a closed socket or server
crash does not look like successful queue-drain evidence. Rows remain grouped by
endpoint for readability.

Latency percentiles use bounded per-row sampling to avoid unbounded trace memory
growth during long benchmarks. Set `GGML_RPC_TRACE_LATENCY_SAMPLE_LIMIT=N` to
change the default cap of 8192 samples per row, or set it to `0` to keep only
totals and maxima.

Setting `GGML_RPC_TRACE=1` on `rpc-server` prints per-connection server command
counts and handler times to that server's stderr. One-way commands such as
`SET_TENSOR` and `GRAPH_RECOMPUTE` only show local send time in the client
table; use server tracing to measure remote handler time. Tracing is disabled by
default and should be used for diagnostics because it adds timing and logging
overhead.

### Compression probing

Set `GGML_RPC_COMPRESSION_PROBE=1` on the RPC client to inspect whether tensor
payload bytes look compressible without changing the RPC wire format:

```bash
GGML_RPC_COMPRESSION_PROBE=1 ./build/bin/llama-bench --rpc 192.0.2.10:50052 ...
```

The report is printed when RPC backends are freed and covers `SET_TENSOR` and
`GET_TENSOR` payloads. It includes sampled bytes, sample coverage, Shannon
entropy in bits per byte, an ideal entropy lower-bound ratio, a simple byte-run
estimate, zero-byte percentage, maximum repeated-byte run, and local probe time.
High entropy near 8 bits per byte and byte-run ratios above 1.0 mean the simple
probe did not find an easy compression win for that payload. Lower entropy or
many repeated bytes identify candidates for a real codec experiment with encode
time, decode time, compressed size, and end-to-end tokens-per-second checks.

By default, the probe samples up to 256 KiB per payload so cold model uploads do
not spend excessive time in diagnostics. Large payloads are sampled from the
beginning, middle, and end rather than only from the prefix. Use
`GGML_RPC_COMPRESSION_PROBE_MAX_SAMPLE` to change that limit; values above
16 MiB are clamped. This is a diagnostic tool only; it does not compress,
decompress, or alter network traffic.

To test real codecs against the sampled payload bytes, set
`GGML_RPC_COMPRESSION_PROBE_DUMP_DIR`:

```bash
GGML_RPC_COMPRESSION_PROBE_DUMP_DIR=/tmp/rpc-payloads \
GGML_RPC_COMPRESSION_PROBE_MAX_SAMPLE=65536 \
./build/bin/llama-bench --rpc 192.0.2.10:50052 ...

python3 tools/rpc/bench_rpc_payload_codecs.py /tmp/rpc-payloads
```

The dump directory contains bounded `.bin` samples plus `samples.jsonl`
metadata. `GGML_RPC_COMPRESSION_PROBE_DUMP_MAX_FILES` limits the number of
files written and defaults to 64. The codec bench reports JSONL rows for each
sample and summary rows per codec, including compressed ratio, encode time,
decode time, round-trip verification, and the effective ratio if expansion is
skipped. It uses Python `zlib` by default and also tests `zstd` or `lz4`
command-line tools when they are installed.

### Tensor upload compression

When both the RPC client and server are built with zlib support, the client can
selectively compress cold/cache-miss `SET_TENSOR` uploads:

```bash
GGML_RPC_SET_TENSOR_COMPRESS=1 ./build/bin/llama-bench --rpc 192.0.2.10:50052 ...
```

The feature is off by default. It is negotiated during `HELLO`, so clients only
send the compressed `SET_TENSOR_ZLIB` command to servers that advertise support.
If zlib development headers are available at configure time, `GGML_RPC_ZLIB` is
enabled automatically; pass `-DGGML_RPC_ZLIB=ON` or `-DGGML_RPC_ZLIB=OFF` to
force the build setting.

Compression is attempted only after the normal cache hash probe misses. Warm
server caches still avoid the upload through `SET_TENSOR_HASH`, which is better
than compressing. Before compressing a full tensor, the client samples the
beginning, middle, and end of the payload. By default, tensors smaller than
1 MiB are skipped, up to 64 KiB is sampled, zlib level 1 is used, and both the
sample and full payload must compress to 70% or less of the original size.

Useful knobs:

```bash
export GGML_RPC_SET_TENSOR_COMPRESS=1
export GGML_RPC_SET_TENSOR_COMPRESS_MIN_SIZE=1048576
export GGML_RPC_SET_TENSOR_COMPRESS_SAMPLE_SIZE=65536
export GGML_RPC_SET_TENSOR_COMPRESS_SAMPLE_RATIO=0.70
export GGML_RPC_SET_TENSOR_COMPRESS_LEVEL=1
```

Use `GGML_RPC_TRACE=1` to compare `SET_TENSOR` and `SET_TENSOR_ZLIB` wire bytes
and client-side elapsed time. This path can reduce cold model-load or cache-miss
upload time on slower links with highly compressible tensors, but it can hurt if
the ratio threshold is relaxed enough to compress payloads that shrink only a
little. It is not expected to improve steady tokens per second once large model
weights are already resident on the RPC servers.

Real model weights can be close to the cutoff. If traces show the client using
CPU time to compress payloads that shrink only slightly, lower
`GGML_RPC_SET_TENSOR_COMPRESS_SAMPLE_RATIO` or leave upload compression off for
that model.

Transparent compression below RPC is usually not enough for this path. Btrfs,
ZFS, NTFS, and other filesystem compression can reduce stored model size or disk
I/O, but RPC sends tensor bytes that are already in memory. SMB compression and
SSH `Compression=yes` can compress their own file-transfer or tunnel protocols,
but raw RPC over TCP/Tailscale does not pass through those layers unless you
explicitly wrap the RPC connection in such a tunnel. Tailscale/WireGuard encrypts
IP packets and does not provide a transparent per-TCP-stream compression stage
for RPC payloads.

### Model placement reporting

Set `LLAMA_LOG_MODEL_PLACEMENT=1` on the RPC client to print the model tensor
placement decision made during load:

```bash
LLAMA_LOG_MODEL_PLACEMENT=1 ./build/bin/llama-bench --rpc 192.0.2.10:50052 ...
```

The report includes the split mode, split source, raw split weights, normalized
split points, whether a split weight came from user tensor-split input, backend
reported memory, or the client CPU fallback, the input and output devices, and
compressed repeating-layer ranges. For RPC, `client_backend` and
`client_device_type` are local client/RPC-wrapper metadata; RPC devices keep the
historical GPU classification so they remain selectable as offload devices.
When both the client and server support RPC device-type reporting, the RPC
device description includes the underlying remote backend type, for example
`remote type: CPU`. Older `rpc-server` processes omit that description suffix.
RPC memory is whatever the remote backend reports and may be remote CPU RAM; use
the remote type suffix or the corresponding `rpc-server` startup device banner
to confirm each remote backend. The `RPC<index>[host:port]` value in
`default_buft` and `buft_candidate` identifies the remote RPC device index to
match against that server banner. The report is useful when RPC throughput looks
network-bound but the real issue may be layer placement, output placement, or a
device that reports no memory and therefore receives a host-memory-based split
weight.

### Troubleshooting

Use the `GGML_RPC_DEBUG` environment variable to enable debug messages from `rpc-server`:
```bash
$ GGML_RPC_DEBUG=1 bin/rpc-server
```
