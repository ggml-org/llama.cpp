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

The speculative-decoding method tests are opt-in and marked `heavy`. The compact suite covers MTP,
DFlash, DSpark, and EAGLE-3 with five pinned files totaling 6.271 GiB:

```shell
SPEC_METHODS=compact SPEC_MODEL_DIR=/path/to/model-cache PYTEST_WORKERS=1 ./tests.sh -s -v unit/test_spec_methods.py
```

- MTP target: `Qwen3.5-4B-MTP-Q4_K_M.gguf`, 2.904 GiB.
- Shared DFlash/DSpark/EAGLE-3 target: `Qwen3-4B-Q4_K_M.gguf`, 2.326 GiB.
- DFlash draft: `dflash-Qwen3-4B-Q4_K_M.gguf`, 294.065 MiB.
- DSpark draft: `dspark-Qwen3-4B-Q4_K_M.gguf`, 544.453 MiB.
- EAGLE-3 draft: `eagle3-Qwen3-4B-Q8_0.gguf`, 227.075 MiB.

All files come from `ggml-org/spec-for-testing-GGUF`, pinned to revision
`a0736c8e98d3fe580352dacffa3fbd5e9c22a437` in `unit/test_spec_methods.py`.
The consolidated repository retains the licenses of its source models; review its model card and
the source model cards before use.

Use `SPEC_METHODS=nightly` for DFlash2 and LiLiCorr, or a comma-separated subset such as
`SPEC_METHODS=mtp,dflash`. The nightly pair requires 41.422 GiB and remains separate because no
verified compact DFlash2 GGUF or compact LiLiCorr checkpoint is available. `SPEC_METHODS=all` runs both
tiers and downloads about 47.693 GiB.

The `Server (speculative decoding)` workflow runs only by explicit manual dispatch on the dedicated
DGX Spark runner. It does not use the GitHub Actions cache. Set the `SPEC_METHOD_MODEL_DIR` repository
variable to the runner's persistent model directory when its mount path is confirmed. Heavyweight
DFlash2 and LiLiCorr measurements belong in the internal nightly performance/quality runs.

The acceptance floors use one-sided regression checks: improvements do not fail. Three identical
runs on the GB10 runner against the PR merged with current master produced MTP `0.6640 / 2.9766`,
DFlash `0.2004 / 2.3687`, DSpark `0.3367 / 3.3246`, and EAGLE-3 `0.2606 / 1.7757` for acceptance
rate and length. The compact floors subtract six percentage points and 0.2 accepted tokens per
verification step. The six-point rate margin is wider than the largest observed cross-environment
shift of 0.0506. Warm compact runs completed in about 40 seconds. The DFlash2 and LiLiCorr nightly
floors retain their earlier three-point rate and 0.2 length margins.

DFlash's token-acceptance rate is lower because the denominator counts every token proposed in each
seven-token draft block. It accepted 219 of 1,093 proposed tokens over 160 verification steps,
equivalent to 1.369 accepted draft tokens per step and a mean acceptance length of 2.3687. The floor
is intended to detect regressions from that deterministic baseline, not to define a quality target.

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
