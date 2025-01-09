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
cmake -B build -DLLAMA_CURL=ON
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

To run slow tests:

```shell
SLOW_TESTS=1 ./tests.sh
```

To run with stdout/stderr display in real time (verbose output, but useful for debugging):

```shell
DEBUG=1 ./tests.sh -s -v -x
```

To run single test unit:

```shell
./tests.sh unit/test_{name of test case here}.py -v -x
```

Hint: You can compile and run test in single command, useful for local developement:

```shell
cmake --build build -j --target llama-server && ./examples/server/tests/tests.sh
```

To see all available arguments, please refer to [pytest documentation](https://docs.pytest.org/en/stable/how-to/usage.html)
