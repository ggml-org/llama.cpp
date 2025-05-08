### Server benchmark tools

Benchmark is using [k6](https://k6.io/).

##### Install k6 and sse extension

SSE is not supported by default in k6, you have to build k6 with the [xk6-sse](https://github.com/phymbert/xk6-sse) extension.

Example (assuming golang >= 1.21 is installed):
```shell
go install go.k6.io/xk6/cmd/xk6@latest
$GOPATH/bin/xk6 build master \
--with github.com/phymbert/xk6-sse
```

#### Download a dataset

This dataset was originally proposed in [vLLM benchmarks](https://github.com/vllm-project/vllm/blob/main/benchmarks/README.md).

```shell
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

#### Download a model
Example for PHI-2

```shell
../../../scripts/hf.sh --repo ggml-org/models --file phi-2/ggml-model-q4_0.gguf
```

#### Start the server
The server must answer OAI Chat completion requests on `http://localhost:8080/v1` or according to the environment variable `SERVER_BENCH_URL`.

Example:
```shell
llama-server --host localhost --port 8080 \
  --model ggml-model-q4_0.gguf \
  --cont-batching \
  --metrics \
  --parallel 8 \
  --batch-size 512 \
  --ctx-size 4096 \
  -ngl 33
```

#### Run the benchmark

For 500 chat completions request with 8 concurrent users during maximum 10 minutes, run:
```shell
./k6 run script.js --duration 10m --iterations 500 --vus 8
```

The benchmark values can be overridden with:
- `SERVER_BENCH_URL` server url prefix for chat completions, default `http://localhost:8080/v1`
- `SERVER_BENCH_N_PROMPTS` total prompts to randomly select in the benchmark, default `480`
- `SERVER_BENCH_MODEL_ALIAS` model alias to pass in the completion request, default `my-model`
- `SERVER_BENCH_MAX_TOKENS` max tokens to predict, default: `512`
- `SERVER_BENCH_DATASET` path to the benchmark dataset file
- `SERVER_BENCH_MAX_PROMPT_TOKENS` maximum prompt tokens to filter out in the dataset: default `1024`
- `SERVER_BENCH_MAX_CONTEXT` maximum context size of the completions request to filter out in the dataset: prompt + predicted tokens, default `2048`

Note: the local tokenizer is just a string space split, real number of tokens will differ.

Or with [k6 options](https://k6.io/docs/using-k6/k6-options/reference/):

```shell
SERVER_BENCH_N_PROMPTS=500 k6 run script.js --duration 10m --iterations 500 --vus 8
```

To [debug http request](https://k6.io/docs/using-k6/http-debugging/) use `--http-debug="full"`.

#### Metrics

Following metrics are available computed from the OAI chat completions response `usage`:
- `llamacpp_tokens_second` Trend of `usage.total_tokens / request duration`
- `llamacpp_prompt_tokens` Trend of `usage.prompt_tokens`
- `llamacpp_prompt_tokens_total_counter` Counter of `usage.prompt_tokens`
- `llamacpp_completion_tokens` Trend of `usage.completion_tokens`
- `llamacpp_completion_tokens_total_counter` Counter of `usage.completion_tokens`
- `llamacpp_completions_truncated_rate` Rate of completions truncated, i.e. if `finish_reason === 'length'`
- `llamacpp_completions_stop_rate` Rate of completions stopped by the model, i.e. if `finish_reason === 'stop'`

The script will fail if too many completions are truncated, see `llamacpp_completions_truncated_rate`.

K6 metrics might be compared against [server metrics](../README.md), with:

```shell
curl http://localhost:8080/metrics
```

### Using the CI python script
The `bench.py` script does several steps:
- start the server
- define good variable for k6
- run k6 script
- extract metrics from prometheus

It aims to be used in the CI, but you can run it manually:

```shell
LLAMA_SERVER_BIN_PATH=../../../cmake-build-release/bin/llama-server python bench.py \
              --runner-label local \
              --name local \
              --branch `git rev-parse --abbrev-ref HEAD` \
              --commit `git rev-parse HEAD` \
              --scenario script.js \
              --duration 5m \
              --hf-repo ggml-org/models	 \
              --hf-file phi-2/ggml-model-q4_0.gguf \
              --model-path-prefix models \
              --parallel 4 \
              -ngl 33 \
              --batch-size 2048 \
              --ubatch-size	256 \
              --ctx-size 4096 \
              --n-prompts 200 \
              --max-prompt-tokens 256 \
              --max-tokens 256
```
