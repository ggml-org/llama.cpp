# Server tests

Python based server tests scenario using [pytest](https://docs.pytest.org/en/stable/).

Tests target GitHub workflows job runners with 4 vCPU.

Note: If the host architecture inference speed is faster than GitHub runners one, parallel scenario may randomly fail.
To mitigate it, you can increase values in `n_predict`, `kv_size`.

### Install dependencies

`pip install -r requirements.txt`

### Run tests

1. Build the server

```shell
cd ../../..
cmake -B build
cmake --build build --target llama-server
```

2. Start the test: `./tests.sh`

It's possible to override some scenario steps values with environment variables:

| variable                 | description                                                                                    |
|--------------------------|------------------------------------------------------------------------------------------------|
| `PORT`                   | `context.server_port` to set the listening port of the server during scenario, default: `8080` |
| `LLAMA_SERVER_BIN_PATH`  | to change the server binary path, default: `../../../build/bin/llama-server`                         |
| `DEBUG`                  | to enable steps and server verbose mode `--verbose`                                       |
| `N_GPU_LAYERS`           | number of model layers to offload to VRAM `-ngl --n-gpu-layers`                                |
| `LLAMA_CACHE`            | by default server tests re-download models to the `tmp` subfolder. Set this to your cache (e.g. `$HOME/Library/Caches/llama.cpp` on Mac or `$HOME/.cache/llama.cpp` on Unix) to avoid this |

To run slow tests (will download many models, make sure to set `LLAMA_CACHE` if needed):

```shell
SLOW_TESTS=1 ./tests.sh
```

To run with stdout/stderr display in real time (verbose output, but useful for debugging):

```shell
DEBUG=1 ./tests.sh -s -v -x
```

To run all the tests in a file:

```shell
./tests.sh unit/test_chat_completion.py -v -x
```

To run a single test:

```shell
./tests.sh unit/test_chat_completion.py::test_invalid_chat_completion_req
```

Hint: You can compile and run test in single command, useful for local development:

```shell
cmake --build build -j --target llama-server && ./tools/server/tests/tests.sh
```

To see all available arguments, please refer to [pytest documentation](https://docs.pytest.org/en/stable/how-to/usage.html)

### Debugging external llama-server
It can sometimes be useful to run the server in a debugger when invesigating test
failures. To do this, the environment variable `DEBUG_EXTERNAL=1` can be set
which will cause the test to skip starting a llama-server itself. Instead, the
server can be started in a debugger.

Example using `gdb`:
```console
$ gdb --args ../../../build/bin/llama-server \
    --host 127.0.0.1 --port 8080 \
    --temp 0.8 --seed 42 \
    --hf-repo ggml-org/models --hf-file tinyllamas/stories260K.gguf \
    --batch-size 32 --no-slots --alias tinyllama-2 --ctx-size 512 \
    --parallel 2 --n-predict 64
```
And a break point can be set in before running:
```console
(gdb) br server.cpp:4604
(gdb) r
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  update_slots: all slots are idle
```

And then the test in question can be run in another terminal:
```console
(venv) $ env DEBUG_EXTERNAL=1 ./tests.sh unit/test_chat_completion.py -v -x
```
And this should trigger the breakpoint and allow inspection of the server state
in the debugger terminal.

### Adaptive KV vision qualification

This standalone harness needs only Python's standard library and local model/projector files. It starts temporary loopback servers and stops them in cleanup, generates deterministic PNG inputs, and disables UVM in those test processes. Free the GPU beforehand; it does not manage production containers.

```sh
python3 tools/server/tests/test_adaptive_vision.py \
  --server ./build-v2/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --mmproj /path/to/mmproj-Qwen3.8-27B-F16.gguf \
  --arena-mib 1024 --context 8192 --check-rejections \
  --output /path/to/test-results
```

The default comparison uses 64/64 batching, 16 generated tokens per native request and a 2 GiB RAM prompt cache in the temporary servers. `--background-tokens 6000` checks a longer prefill. Model/backend arithmetic is compared under matched request and cache execution modes, not claimed universally byte-identical. Chat-completion smoke, disconnect cancellation and error recovery are also checked.

Add `--mtp-length 1`, `2` or `3` to qualify embedded MTP with matching Q8_0/Q4_0 draft KV. The eager control uses this binary's image-aware input plumbing but ordinary native kernels, not an unmodified upstream server. For example, `--context 40960 --background-tokens 39000 --arena-mib 1024 --batch-size 256 --ubatch-size 256 --mtp-length 3 --decode 128 --cache-ram-mib 8192` exercises streaming and image-cache restoration. Long snapshots need enough host cache capacity; otherwise the RAM-restore assertion correctly fails. If the eager side cannot fit its separate allocations, `--stock-uvm` permits UVM only for that correctness control; the arena side still disables it. This is not a throughput comparison.

For native context-capacity admission with 256/256, use `--mode arena --context 262144 --arena-mib 2240 --batch-size 256 --ubatch-size 256 --skip-budget-rejection`. The skip flag omits the assertion that a particular 1536x1536 image must exceed a small arena; that image can legitimately fit a larger arena. This is not a full 262K-token prompt benchmark.

`--mode stock`/`--mode arena` run one side. `--cache-ram-mib` and `--decode` are configurable. Logs and accepted mode results are stored in the output directory; `--check-rejections` verifies unsupported startup settings without loading the target. No default production cache/checkpoint configuration is changed by this harness.

### Vision memory and handoff measurements

The measurement companion runs repeated uncached requests, checks unchanged token IDs and post-warmup idle memory, and samples NVIDIA device-wide memory every 50 ms. It needs Linux, `nvidia-smi`, Python's standard library and the same local model files. Free the selected GPU first; the script does not stop containers. UVM is disabled and `--gpu` selects both the server's visible physical CUDA device and the device sampler.

```sh
python3 tools/server/tests/measure_adaptive_vision.py \
  --server ./build-v2/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --mmproj /path/to/mmproj-Qwen3.8-27B-F16.gguf \
  --arena-mib 2240 --context 262144 --batches 64,256 \
  --image-size 1024 --background-tokens 6000 --repeats 3 \
  --output /path/to/vision-measurements
```

`--images 2` measures multiple images, including separate encoder batches when required by the projector. The script groups all completed vision phases into their owning request rather than assuming one encoding per request. `--context 49152 --arena-mib 1024 --batches 64 --background-tokens 40000` exercises a streamed text history. The arena is explicit; this companion does not auto-probe its maximum. `--sample-ms`, `--decode` and `--idle-tolerance-mib` are configurable. At least three repetitions are required so idle stability is checked after the first request's native graph setup.

Use `--mtp-length 1`, `2` or `3` for image-aware embedded MTP memory measurements. Other settings and the no-UVM policy are unchanged.

The report includes raw samples, request/vision-window peaks, idle baselines, actual weight/compute grants, phase timings and the first KV refill after each vision phase. `vision_phase` and `KV_reload` records are available at server verbosity 3. The C++ diagnostics use backend-neutral ownership and existing synchronous-copy boundaries; only this external device sampler is NVIDIA/Linux-specific.

The temporary servers use image-token limits 64-4096 to exercise different encoder sizes. These are test settings, not a visual-quality recommendation; Qwen recommends a higher minimum for grounding. Production image limits are not changed. Like the existing `memory_phase` diagnostics, these low-frequency backend records use warning-level visibility so the server's backend callback keeps them at verbosity 3; a record itself is not an inference failure.

- `projector_reload_us` includes file access, host upload preparation and completed weight uploads; it is not pure PCIe DMA time.
- `text_resume_us` rebuilds text grants and graph reservations; it does not upload the full KV history.
- `KV_reload.upload_us` measures the first-use resident mirror flush, including host planning and its existing synchronous copies. `drain_us` separately measures its existing backend synchronization. Later ring transfers and attention computation are not part of this measurement.
- `simultaneous_buffer_estimate_mib` is the ready text allocation plus separately measured vision weight/compute buffers. It is a counterfactual for retaining this configured text arena, not a measured stock server or a total driver-memory bound.
- Sampled peaks are lower bounds on the true instantaneous peak. They include all activity on the selected device and can miss brief allocations. The fixed parent intentionally remains allocated after vision; successful cleanup returns grants, not the entire parent to the driver.

For host-only report tests, run `python3 -B tools/server/tests/test_adaptive_vision_measurements.py`. These tests require neither a model nor a GPU and are registered with CTest when server tests and Python are available.
