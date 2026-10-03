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

For native context-capacity admission with 256/256, use `--mode arena --context 262144 --arena-mib 2240 --batch-size 256 --ubatch-size 256 --skip-budget-rejection`. The skip flag omits the assertion that a particular 1536x1536 image must exceed a small arena; that image can legitimately fit a larger arena. This is not a full 262K-token prompt benchmark.

`--mode stock`/`--mode arena` run one side. `--cache-ram-mib` and `--decode` are configurable. Logs and accepted mode results are stored in the output directory; `--check-rejections` verifies unsupported startup settings without loading the target. No default production cache/checkpoint configuration is changed by this harness.
