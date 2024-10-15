# Server tests

Python based server tests scenario using [BDD](https://en.wikipedia.org/wiki/Behavior-driven_development)
and [behave](https://behave.readthedocs.io/en/latest/):

* [issues.feature](./features/issues.feature) Pending issues scenario
* [parallel.feature](./features/parallel.feature) Scenario involving multi slots and concurrent requests
* [security.feature](./features/security.feature) Security, CORS and API Key
* [server.feature](./features/server.feature) Server base scenario: completion, embedding, tokenization, etc...

Tests target GitHub workflows job runners with 4 vCPU.

Requests are
using [aiohttp](https://docs.aiohttp.org/en/stable/client_reference.html), [asyncio](https://docs.python.org/fr/3/library/asyncio.html)
based http client.

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
| `DEBUG`                  | "ON" to enable steps and server verbose mode `--verbose`                                       |
| `N_GPU_LAYERS`           | number of model layers to offload to VRAM `-ngl --n-gpu-layers`                                |

### Run @bug, @wip or @wrong_usage annotated scenario

Feature or Scenario must be annotated with `@llama.cpp` to be included in the default scope.

- `@bug` annotation aims to link a scenario with a GitHub issue.
- `@wrong_usage` are meant to show user issue that are actually an expected behavior
- `@wip` to focus on a scenario working in progress
- `@slow` heavy test, disabled by default

To run a scenario annotated with `@bug`, start:

```shell
DEBUG=ON ./tests.sh --no-skipped --tags bug --stop
```

After changing logic in `steps.py`, ensure that `@bug` and `@wrong_usage` scenario are updated.

```shell
./tests.sh --no-skipped --tags bug,wrong_usage || echo "should failed but compile"
```
