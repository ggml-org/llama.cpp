# LLaMA.cpp HTTP Server

Fast, lightweight, pure C/C++ HTTP server based on [httplib](https://github.com/yhirose/cpp-httplib), [nlohmann::json](https://github.com/nlohmann/json) and **llama.cpp**.

Set of LLM REST APIs and a simple web front end to interact with llama.cpp.

**Features:**
 * LLM inference of F16 and quantum models on GPU and CPU
 * [OpenAI API](https://github.com/openai/openai-openapi) compatible chat completions and embeddings routes
 * Parallel decoding with multi-user support
 * Continuous batching
 * Multimodal (wip)
 * Monitoring endpoints
 * Schema-constrained JSON response format

The project is under active development, and we are [looking for feedback and contributors](https://github.com/ggerganov/llama.cpp/issues/4216).

**Command line options:**

- `-v`, `--verbose`: Enable verbose server output. When using the `/completion` endpoint, this includes the tokenized prompt, the full request and the full response.
- `-t N`, `--threads N`: Set the number of threads to use by CPU layers during generation. Not used by model layers that are offloaded to GPU. This option has no effect when using the maximum number of GPU layers. Default: `std::thread::hardware_concurrency()` (number of CPU cores).
- `-tb N, --threads-batch N`: Set the number of threads to use by CPU layers during batch and prompt processing (>= 32 tokens). This option has no effect if a GPU is available. Default: `--threads`.
- `--threads-http N`: Number of threads in the http server pool to process requests. Default: `max(std::thread::hardware_concurrency() - 1, --parallel N + 2)`
- `-m FNAME`, `--model FNAME`: Specify the path to the LLaMA model file (e.g., `models/7B/ggml-model.gguf`).
- `-mu MODEL_URL --model-url MODEL_URL`: Specify a remote http url to download the file. Default: unused
- `-hfr REPO, --hf-repo REPO`: Hugging Face model repository. Default: unused
- `-hff FILE, --hf-file FILE`: Hugging Face model file. Default: unused
- `-a ALIAS`, `--alias ALIAS`: Set an alias for the model. The alias will be returned in API responses.
- `-c N`, `--ctx-size N`: Set the size of the prompt context. The default is `512`, but LLaMA models were built with a context of `2048`, which will provide better results for longer input/inference. The size may differ in other models, for example, baichuan models were build with a context of `4096`.
- `-ngl N`, `--n-gpu-layers N`: When compiled with GPU support, this option allows offloading some layers to the GPU for computation. Generally results in increased performance.
- `-mg i, --main-gpu i`: When using multiple GPUs, this option controls which GPU is used for small tensors for which the overhead of splitting the computation across all GPUs is not worthwhile. The GPU in question will use slightly more VRAM to store a scratch buffer for temporary results. By default, GPU `0` is used.
- `-ts SPLIT, --tensor-split SPLIT`: When using multiple GPUs, this option controls how large tensors should be split across all GPUs. `SPLIT` is a comma-separated list of non-negative values that assigns the proportion of data that each GPU should get in order. For example, "3,2" will assign 60% of the data to GPU 0 and 40% to GPU 1. By default, the data is split in proportion to VRAM, but this may not be optimal for performance.
- `-b N`, `--batch-size N`: Set the batch size for prompt processing. Default: `2048`
- `-ub N`, `--ubatch-size N`: Physical maximum batch size. Default: `512`
- `--mlock`: Lock the model in memory, preventing it from being swapped out when memory-mapped.
- `--no-mmap`: Do not memory-map the model. By default, models are mapped into memory, which allows the system to load only the necessary parts of the model as needed.
- `--numa STRATEGY`: Attempt one of the below optimization strategies that may help on some NUMA systems
- `--numa distribute`: Spread execution evenly over all nodes
- `--numa isolate`: Only spawn threads on CPUs on the node that execution started on
- `--numa numactl`: Use the CPU map provided by numactl. If run without this previously, it is recommended to drop the system page cache before using this. See https://github.com/ggerganov/llama.cpp/issues/1437
- `--numa`: Attempt optimizations that may help on some NUMA systems.
- `--lora FNAME`: Apply a LoRA (Low-Rank Adaptation) adapter to the model (implies --no-mmap). This allows you to adapt the pretrained model to specific tasks or domains.
- `--lora-base FNAME`: Optional model to use as a base for the layers modified by the LoRA adapter. This flag is used in conjunction with the `--lora` flag, and specifies the base model for the adaptation.
- `-to N`, `--timeout N`: Server read/write timeout in seconds. Default `600`
- `--host`: Set the hostname or ip address to listen. Default `127.0.0.1`
- `--port`: Set the port to listen. Default: `8080`
- `--path`: Path from which to serve static files. Default: disabled
- `--api-key`: Set an api key for request authorization. By default, the server responds to every request. With an api key set, the requests must have the Authorization header set with the api key as Bearer token. May be used multiple times to enable multiple valid keys.
- `--api-key-file`: Path to file containing api keys delimited by new lines. If set, requests must include one of the keys for access. May be used in conjunction with `--api-key`s.
- `--embeddings`: Enable embedding vector output and the OAI compatible endpoint /v1/embeddings. Physical batch size (`--ubatch-size`) must be carefully defined. Default: disabled
- `-np N`, `--parallel N`: Set the number of slots for process requests. Default: `1`. Values > 1 will allow for higher throughput with multiple parallel requests but the results will **not** be deterministic due to differences in rounding error.
- `-cb`, `--cont-batching`: Enable continuous batching (a.k.a dynamic batching).  Default: disabled
- `-spf FNAME`, `--system-prompt-file FNAME` Set a file to load a system prompt (initial prompt of all slots). This is useful for chat applications. [See more](#change-system-prompt-on-runtime)
- `--mmproj MMPROJ_FILE`: Path to a multimodal projector file for LLaVA.
- `--grp-attn-n`: Set the group attention factor to extend context size through self-extend. Used together with group attention width `--grp-attn-w`. Default: `1`, which is disabled.
- `--grp-attn-w`: Set the group attention width to extend context size through self-extend.  Used together with group attention factor `--grp-attn-n`. Default: `512`
- `-n N, --n-predict N`: Set the maximum tokens to predict. Default: `-1`
- `--slots-endpoint-disable`: To disable slots state monitoring endpoint. Slots state may contain user data, prompts included.
- `--metrics`: enable prometheus `/metrics` compatible endpoint. Default: disabled
- `--slot-save-path PATH`: Specifies the path where the state of slots (the prompt cache) can be stored. If not provided, the slot management endpoints will be disabled.
- `--chat-template JINJA_TEMPLATE`: Set custom jinja chat template. This parameter accepts a string, not a file name.  Default: template taken from model's metadata. We only support [some pre-defined templates](https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template)
- `--log-disable`: Output logs to stdout only, not to `llama.log`. Default: enabled
- `--log-format FORMAT`: Define the log output to FORMAT: json or text Default: `json`
- `--rope-scaling` : RoPE scaling method. Defaults to linear unless otherwise specified by the model. Options are `none`, `linear`, `yarn`
- `--rope-freq-base N` : RoPE frequency base (default: loaded from model)
- `--rope-freq-scale N`: RoPE frequency scaling factor, expands context by a factor of 1/N (e.g. 0.25)
- `--yarn-ext-factor N` : YaRN: extrapolation mix factor (Default: 1.0, 0.0 = full interpolation)
- `--yarn-attn-factor N` : YaRN: scale sqrt(t) or attention magnitude (default: 1.0)
- `--yarn-beta-slow N`: YaRN: High correction dim or alpha (default: 1.0)
- `--yarn-beta-fast N`: YaRN: low correction dim or beta (default: 32.0)
- `--pooling` : Pooling type for embeddings, use model default if unspecified. Options are `none`, `mean`, `cls`
- `-dt N`, `--defrag-thold N`: KV cache defragmentation threshold (default: -1.0, < 0 = disabled)
- `-fa`, `--flash-attn` : enable flash attention (default: disabled).
- `-ctk TYPE`, `--cache-type-k TYPE` : KV cache data type for K (default: `f16`, options `f32`, `f16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, or `q5_1`)
- `-ctv TYPE`, `--cache-type-v TYPE` : KV cache type for V (default `f16`, see `-ctk` for options)
- `--spm-infill` : Use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this.

**If compiled with `LLAMA_SERVER_SSL=ON`**
- `--ssl-key-file FNAME`: path to file a PEM-encoded SSL private key
- `--ssl-cert-file FNAME`: path to file a PEM-encoded SSL certificate

## Build

`llama-server` is built alongside everything else from the root of the project

- Using `make`:

  ```bash
  make llama-server
  ```

- Using `CMake`:

  ```bash
  cmake -B build
  cmake --build build --config Release -t llama-server
  ```

  Binary is at `./build/bin/llama-server`

## Build with SSL

`llama-server` can also be built with SSL support using OpenSSL 3

- Using `make`:

  ```bash
  # NOTE: For non-system openssl, use the following:
  #   CXXFLAGS="-I /path/to/openssl/include"
  #   LDFLAGS="-L /path/to/openssl/lib"
  make LLAMA_SERVER_SSL=true llama-server
  ```

- Using `CMake`:

  ```bash
  cmake -B build -DLLAMA_SERVER_SSL=ON
  cmake --build build --config Release -t llama-server
  ```

## Quick Start

To get started right away, run the following command, making sure to use the correct path for the model you have:

### Unix-based systems (Linux, macOS, etc.)

```bash
./llama-server -m models/7B/ggml-model.gguf -c 2048
```

### Windows

```powershell
llama-server.exe -m models\7B\ggml-model.gguf -c 2048
```

The above command will start a server that by default listens on `127.0.0.1:8080`.
You can consume the endpoints with Postman or NodeJS with axios library. You can visit the web front end at the same url.

### Docker

```bash
docker run -p 8080:8080 -v /path/to/models:/models ghcr.io/ggerganov/llama.cpp:server -m models/7B/ggml-model.gguf -c 512 --host 0.0.0.0 --port 8080

# or, with CUDA:
docker run -p 8080:8080 -v /path/to/models:/models --gpus all ghcr.io/ggerganov/llama.cpp:server-cuda -m models/7B/ggml-model.gguf -c 512 --host 0.0.0.0 --port 8080 --n-gpu-layers 99
```

## Testing with CURL

Using [curl](https://curl.se/). On Windows, `curl.exe` should be available in the base OS.

```sh
curl --request POST \
    --url http://localhost:8080/completion \
    --header "Content-Type: application/json" \
    --data '{"prompt": "Building a website can be done in 10 simple steps:","n_predict": 128}'
```

## Advanced testing

We implemented a [server test framework](./tests/README.md) using human-readable scenario.

*Before submitting an issue, please try to reproduce it with this format.*

## Node JS Test

You need to have [Node.js](https://nodejs.org/en) installed.

```bash
mkdir llama-client
cd llama-client
```

Create a index.js file and put this inside:

```javascript
const prompt = `Building a website can be done in 10 simple steps:`;

async function Test() {
    let response = await fetch("http://127.0.0.1:8080/completion", {
        method: 'POST',
        body: JSON.stringify({
            prompt,
            n_predict: 512,
        })
    })
    console.log((await response.json()).content)
}

Test()
```

And run it:

```bash
node index.js
```

## API Endpoints

- **GET** `/health`: Returns the current state of the server:
  - 503 -> `{"status": "loading model"}` if the model is still being loaded.
  - 500 -> `{"status": "error"}` if the model failed to load.
  - 200 -> `{"status": "ok", "slots_idle": 1, "slots_processing": 2 }` if the model is successfully loaded and the server is ready for further requests mentioned below.
  - 200 -> `{"status": "no slot available", "slots_idle": 0, "slots_processing": 32}` if no slots are currently available.
  - 503 -> `{"status": "no slot available", "slots_idle": 0, "slots_processing": 32}` if the query parameter `fail_on_no_slot` is provided and no slots are currently available.

  If the query parameter `include_slots` is passed, `slots` field will contain internal slots data except if `--slots-endpoint-disable` is set.

- **POST** `/completion`: Given a `prompt`, it returns the predicted completion.

    *Options:*

    `prompt`: Provide the prompt for this completion as a string or as an array of strings or numbers representing tokens. Internally, if `cache_prompt` is `true`, the prompt is compared to the previous completion and only the "unseen" suffix is evaluated. A `BOS` token is inserted at the start, if all of the following conditions are true:

      - The prompt is a string or an array with the first element given as a string
      - The model's `tokenizer.ggml.add_bos_token` metadata is `true`
      - The system prompt is empty

    `temperature`: Adjust the randomness of the generated text. Default: `0.8`

    `dynatemp_range`: Dynamic temperature range. The final temperature will be in the range of `[temperature - dynatemp_range; temperature + dynatemp_range]` Default: `0.0`, which is disabled.

    `dynatemp_exponent`: Dynamic temperature exponent. Default: `1.0`

    `top_k`: Limit the next token selection to the K most probable tokens.  Default: `40`

    `top_p`: Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. Default: `0.95`

    `min_p`: The minimum probability for a token to be considered, relative to the probability of the most likely token. Default: `0.05`

    `n_predict`: Set the maximum number of tokens to predict when generating text. **Note:** May exceed the set limit slightly if the last token is a partial multibyte character. When 0, no tokens will be generated but the prompt is evaluated into the cache. Default: `-1`, where `-1` is infinity.

    `n_keep`: Specify the number of tokens from the prompt to retain when the context size is exceeded and tokens need to be discarded.
    By default, this value is set to `0`, meaning no tokens are kept. Use `-1` to retain all tokens from the prompt.

    `stream`: It allows receiving each predicted token in real-time instead of waiting for the completion to finish. To enable this, set to `true`.

    `stop`: Specify a JSON array of stopping strings.
    These words will not be included in the completion, so make sure to add them to the prompt for the next iteration. Default: `[]`

    `tfs_z`: Enable tail free sampling with parameter z. Default: `1.0`, which is disabled.

    `typical_p`: Enable locally typical sampling with parameter p. Default: `1.0`, which is disabled.

    `repeat_penalty`: Control the repetition of token sequences in the generated text. Default: `1.1`

    `repeat_last_n`: Last n tokens to consider for penalizing repetition. Default: `64`, where `0` is disabled and `-1` is ctx-size.

    `penalize_nl`: Penalize newline tokens when applying the repeat penalty. Default: `true`

    `presence_penalty`: Repeat alpha presence penalty. Default: `0.0`, which is disabled.

    `frequency_penalty`: Repeat alpha frequency penalty. Default: `0.0`, which is disabled.

    `penalty_prompt`: This will replace the `prompt` for the purpose of the penalty evaluation. Can be either `null`, a string or an array of numbers representing tokens. Default: `null`, which is to use the original `prompt`.

    `mirostat`: Enable Mirostat sampling, controlling perplexity during text generation. Default: `0`, where `0` is disabled, `1` is Mirostat, and `2` is Mirostat 2.0.

    `mirostat_tau`: Set the Mirostat target entropy, parameter tau. Default: `5.0`

    `mirostat_eta`: Set the Mirostat learning rate, parameter eta.  Default: `0.1`

    `grammar`: Set grammar for grammar-based sampling.  Default: no grammar

    `json_schema`: Set a JSON schema for grammar-based sampling (e.g. `{"items": {"type": "string"}, "minItems": 10, "maxItems": 100}` of a list of strings, or `{}` for any JSON). See [tests](../../tests/test-json-schema-to-grammar.cpp) for supported features.  Default: no JSON schema.

    `seed`: Set the random number generator (RNG) seed.  Default: `-1`, which is a random seed.

    `ignore_eos`: Ignore end of stream token and continue generating.  Default: `false`

    `logit_bias`: Modify the likelihood of a token appearing in the generated text completion. For example, use `"logit_bias": [[15043,1.0]]` to increase the likelihood of the token 'Hello', or `"logit_bias": [[15043,-1.0]]` to decrease its likelihood. Setting the value to false, `"logit_bias": [[15043,false]]` ensures that the token `Hello` is never produced. The tokens can also be represented as strings, e.g. `[["Hello, World!",-0.5]]` will reduce the likelihood of all the individual tokens that represent the string `Hello, World!`, just like the `presence_penalty` does. Default: `[]`

    `n_probs`: If greater than 0, the response also contains the probabilities of top N tokens for each generated token given the sampling settings. Note that for temperature < 0 the tokens are sampled greedily but token probabilities are still being calculated via a simple softmax of the logits without considering any other sampler settings. Default: `0`

    `min_keep`: If greater than 0, force samplers to return N possible tokens at minimum. Default: `0`

    `image_data`: An array of objects to hold base64-encoded image `data` and its `id`s to be reference in `prompt`. You can determine the place of the image in the prompt as in the following: `USER:[img-12]Describe the image in detail.\nASSISTANT:`. In this case, `[img-12]` will be replaced by the embeddings of the image with id `12` in the following `image_data` array: `{..., "image_data": [{"data": "<BASE64_STRING>", "id": 12}]}`. Use `image_data` only with multimodal models, e.g., LLaVA.

    `id_slot`: Assign the completion task to an specific slot. If is -1 the task will be assigned to a Idle slot.  Default: `-1`

    `cache_prompt`: Re-use KV cache from a previous request if possible. This way the common prefix does not have to be re-processed, only the suffix that differs between the requests. Because (depending on the backend) the logits are **not** guaranteed to be bit-for-bit identical for different batch sizes (prompt processing vs. token generation) enabling this option can cause nondeterministic results. Default: `false`

    `system_prompt`: Change the system prompt (initial prompt of all slots), this is useful for chat applications. [See more](#change-system-prompt-on-runtime)

    `samplers`: The order the samplers should be applied in. An array of strings representing sampler type names. If a sampler is not set, it will not be used. If a sampler is specified more than once, it will be applied multiple times. Default: `["top_k", "tfs_z", "typical_p", "top_p", "min_p", "temperature"]` - these are all the available values.

### Result JSON

- Note: When using streaming mode (`stream`), only `content` and `stop` will be returned until end of completion.

- `completion_probabilities`: An array of token probabilities for each completion. The array's length is `n_predict`. Each item in the array has the following structure:

```json
{
  "content": "<the token selected by the model>",
  "probs": [
    {
      "prob": float,
      "tok_str": "<most likely token>"
    },
    {
      "prob": float,
      "tok_str": "<second most likely token>"
    },
    ...
  ]
},
```

Notice that each `probs` is an array of length `n_probs`.

- `content`: Completion result as a string (excluding `stopping_word` if any). In case of streaming mode, will contain the next token as a string.
- `stop`: Boolean for use with `stream` to check whether the generation has stopped (Note: This is not related to stopping words array `stop` from input options)
- `generation_settings`: The provided options above excluding `prompt` but including `n_ctx`, `model`. These options may differ from the original ones in some way (e.g. bad values filtered out, strings converted to tokens, etc.).
- `model`: The path to the model loaded with `-m`
- `prompt`: The provided `prompt`
- `stopped_eos`: Indicating whether the completion has stopped because it encountered the EOS token
- `stopped_limit`: Indicating whether the completion stopped because `n_predict` tokens were generated before stop words or EOS was encountered
- `stopped_word`: Indicating whether the completion stopped due to encountering a stopping word from `stop` JSON array provided
- `stopping_word`: The stopping word encountered which stopped the generation (or "" if not stopped due to a stopping word)
- `timings`: Hash of timing information about the completion such as the number of tokens `predicted_per_second`
- `tokens_cached`: Number of tokens from the prompt which could be re-used from previous completion (`n_past`)
- `tokens_evaluated`: Number of tokens evaluated in total from the prompt
- `truncated`: Boolean indicating if the context size was exceeded during generation, i.e. the number of tokens provided in the prompt (`tokens_evaluated`) plus tokens generated (`tokens predicted`) exceeded the context size (`n_ctx`)

- **POST** `/tokenize`: Tokenize a given text.

    *Options:*

    `content`: Set the text to tokenize.

    `add_special`: Boolean indicating if special tokens, i.e. `BOS`, should be inserted.  Default: `false`

- **POST** `/detokenize`: Convert tokens to text.

    *Options:*

    `tokens`: Set the tokens to detokenize.

- **POST** `/embedding`: Generate embedding of a given text just as [the embedding example](../embedding) does.

    *Options:*

    `content`: Set the text to process.

    `image_data`: An array of objects to hold base64-encoded image `data` and its `id`s to be reference in `content`. You can determine the place of the image in the content as in the following: `Image: [img-21].\nCaption: This is a picture of a house`. In this case, `[img-21]` will be replaced by the embeddings of the image with id `21` in the following `image_data` array: `{..., "image_data": [{"data": "<BASE64_STRING>", "id": 21}]}`. Use `image_data` only with multimodal models, e.g., LLaVA.

- **POST** `/infill`: For code infilling. Takes a prefix and a suffix and returns the predicted completion as stream.

    *Options:*

    `input_prefix`: Set the prefix of the code to infill.

    `input_suffix`: Set the suffix of the code to infill.

    It also accepts all the options of `/completion` except `stream` and `prompt`.

- **GET** `/props`: Return current server settings.

### Result JSON

```json
{
  "assistant_name": "",
  "user_name": "",
  "default_generation_settings": { ... },
  "total_slots": 1,
  "chat_template": ""
}
```

- `assistant_name` - the required assistant name to generate the prompt in case you have specified a system prompt for all slots.
- `user_name` - the required anti-prompt to generate the prompt in case you have specified a system prompt for all slots.
- `default_generation_settings` - the default generation settings for the `/completion` endpoint, which has the same fields as the `generation_settings` response object from the `/completion` endpoint.
- `total_slots` - the total number of slots for process requests (defined by `--parallel` option)
- `chat_template` - the model's original Jinja2 prompt template

- **POST** `/v1/chat/completions`: OpenAI-compatible Chat Completions API. Given a ChatML-formatted json description in `messages`, it returns the predicted completion. Both synchronous and streaming mode are supported, so scripted and interactive applications work fine. While no strong claims of compatibility with OpenAI API spec is being made, in our experience it suffices to support many apps. Only models with a [supported chat template](https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template) can be used optimally with this endpoint. By default, the ChatML template will be used.

    *Options:*

    See [OpenAI Chat Completions API documentation](https://platform.openai.com/docs/api-reference/chat). While some OpenAI-specific features such as function calling aren't supported, llama.cpp `/completion`-specific features such as `mirostat` are supported.

    The `response_format` parameter supports both plain JSON output (e.g. `{"type": "json_object"}`) and schema-constrained JSON (e.g. `{"type": "json_object", "schema": {"type": "string", "minLength": 10, "maxLength": 100}}`), similar to other OpenAI-inspired API providers.

    *Examples:*

    You can use either Python `openai` library with appropriate checkpoints:

    ```python
    import openai

    client = openai.OpenAI(
        base_url="http://localhost:8080/v1", # "http://<Your api-server IP>:port"
        api_key = "sk-no-key-required"
    )

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
        {"role": "user", "content": "Write a limerick about python exceptions"}
    ]
    )

    print(completion.choices[0].message)
    ```

    ... or raw HTTP requests:

    ```shell
    curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer no-key" \
    -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
    {
        "role": "system",
        "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
    },
    {
        "role": "user",
        "content": "Write a limerick about python exceptions"
    }
    ]
    }'
    ```

- **POST** `/v1/embeddings`: OpenAI-compatible embeddings API.

    *Options:*

    See [OpenAI Embeddings API documentation](https://platform.openai.com/docs/api-reference/embeddings).

    *Examples:*

  - input as string

    ```shell
    curl http://localhost:8080/v1/embeddings \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer no-key" \
    -d '{
            "input": "hello",
            "model":"GPT-4",
            "encoding_format": "float"
    }'
    ```

  - `input` as string array

    ```shell
    curl http://localhost:8080/v1/embeddings \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer no-key" \
    -d '{
            "input": ["hello", "world"],
            "model":"GPT-4",
            "encoding_format": "float"
    }'
    ```

- **GET** `/slots`: Returns the current slots processing state. Can be disabled with `--slots-endpoint-disable`.

### Result JSON

```json
[
    {
        "dynatemp_exponent": 1.0,
        "dynatemp_range": 0.0,
        "frequency_penalty": 0.0,
        "grammar": "",
        "id": 0,
        "ignore_eos": false,
        "logit_bias": [],
        "min_p": 0.05000000074505806,
        "mirostat": 0,
        "mirostat_eta": 0.10000000149011612,
        "mirostat_tau": 5.0,
        "model": "llama-2-7b-32k-instruct.Q2_K.gguf",
        "n_ctx": 2048,
        "n_keep": 0,
        "n_predict": 100000,
        "n_probs": 0,
        "next_token": {
            "has_next_token": true,
            "n_remain": -1,
            "n_decoded": 0,
            "stopped_eos": false,
            "stopped_limit": false,
            "stopped_word": false,
            "stopping_word": ""
        },
        "penalize_nl": true,
        "penalty_prompt_tokens": [],
        "presence_penalty": 0.0,
        "prompt": "Say hello to llama.cpp",
        "repeat_last_n": 64,
        "repeat_penalty": 1.100000023841858,
        "samplers": [
            "top_k",
            "tfs_z",
            "typical_p",
            "top_p",
            "min_p",
            "temperature"
        ],
        "seed": 42,
        "state": 1,
        "stop": [
            "\n"
        ],
        "stream": false,
        "task_id": 0,
        "temperature": 0.0,
        "tfs_z": 1.0,
        "top_k": 40,
        "top_p": 0.949999988079071,
        "typical_p": 1.0,
        "use_penalty_prompt_tokens": false
    }
]
```

- **GET** `/metrics`: [Prometheus](https://prometheus.io/) compatible metrics exporter endpoint if `--metrics` is enabled:

Available metrics:
- `llamacpp:prompt_tokens_total`: Number of prompt tokens processed.
- `llamacpp:tokens_predicted_total`: Number of generation tokens processed.
- `llamacpp:prompt_tokens_seconds`: Average prompt throughput in tokens/s.
- `llamacpp:predicted_tokens_seconds`: Average generation throughput in tokens/s.
- `llamacpp:kv_cache_usage_ratio`: KV-cache usage. `1` means 100 percent usage.
- `llamacpp:kv_cache_tokens`: KV-cache tokens.
- `llamacpp:requests_processing`: Number of requests processing.
- `llamacpp:requests_deferred`: Number of requests deferred.

- **POST** `/slots/{id_slot}?action=save`: Save the prompt cache of the specified slot to a file.

    *Options:*

    `filename`: Name of the file to save the slot's prompt cache. The file will be saved in the directory specified by the `--slot-save-path` server parameter.

### Result JSON

```json
{
    "id_slot": 0,
    "filename": "slot_save_file.bin",
    "n_saved": 1745,
    "n_written": 14309796,
    "timings": {
        "save_ms": 49.865
    }
}
```

- **POST** `/slots/{id_slot}?action=restore`: Restore the prompt cache of the specified slot from a file.

    *Options:*

    `filename`: Name of the file to restore the slot's prompt cache from. The file should be located in the directory specified by the `--slot-save-path` server parameter.

### Result JSON

```json
{
    "id_slot": 0,
    "filename": "slot_save_file.bin",
    "n_restored": 1745,
    "n_read": 14309796,
    "timings": {
        "restore_ms": 42.937
    }
}
```

- **POST** `/slots/{id_slot}?action=erase`: Erase the prompt cache of the specified slot.

### Result JSON

```json
{
    "id_slot": 0,
    "n_erased": 1745
}
```

## More examples

### Change system prompt on runtime

To use the server example to serve multiple chat-type clients while keeping the same system prompt, you can utilize the option `system_prompt`. This only needs to be used once.

`prompt`: Specify a context that you want all connecting clients to respect.

`anti_prompt`: Specify the word you want to use to instruct the model to stop. This must be sent to each client through the `/props` endpoint.

`assistant_name`: The bot's name is necessary for each customer to generate the prompt. This must be sent to each client through the `/props` endpoint.

```json
{
    "system_prompt": {
        "prompt": "Transcript of a never ending dialog, where the User interacts with an Assistant.\nThe Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.\nUser: Recommend a nice restaurant in the area.\nAssistant: I recommend the restaurant \"The Golden Duck\". It is a 5 star restaurant with a great view of the city. The food is delicious and the service is excellent. The prices are reasonable and the portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm. The restaurant is closed on Saturdays and Sundays.\nUser: Who is Richard Feynman?\nAssistant: Richard Feynman was an American physicist who is best known for his work in quantum mechanics and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions to the development of quantum electrodynamics. He was a popular lecturer and author, and he wrote several books, including \"Surely You're Joking, Mr. Feynman!\" and \"What Do You Care What Other People Think?\".\nUser:",
        "anti_prompt": "User:",
        "assistant_name": "Assistant:"
    }
}
```

**NOTE**: You can do this automatically when starting the server by simply creating a .json file with these options and using the CLI option `-spf FNAME` or `--system-prompt-file FNAME`.

### Interactive mode

Check the sample in [chat.mjs](chat.mjs).
Run with NodeJS version 16 or later:

```sh
node chat.mjs
```

Another sample in [chat.sh](chat.sh).
Requires [bash](https://www.gnu.org/software/bash/), [curl](https://curl.se) and [jq](https://jqlang.github.io/jq/).
Run with bash:

```sh
bash chat.sh
```

### OAI-like API

The HTTP `llama-server` supports an OAI-like API: https://github.com/openai/openai-openapi

### API errors

`llama-server` returns errors in the same format as OAI: https://github.com/openai/openai-openapi

Example of an error:

```json
{
    "error": {
        "code": 401,
        "message": "Invalid API Key",
        "type": "authentication_error"
    }
}
```

Apart from error types supported by OAI, we also have custom types that are specific to functionalities of llama.cpp:

**When /metrics or /slots endpoint is disabled**

```json
{
    "error": {
        "code": 501,
        "message": "This server does not support metrics endpoint.",
        "type": "not_supported_error"
    }
}
```

**When the server receives invalid grammar via */completions endpoint**

```json
{
    "error": {
        "code": 400,
        "message": "Failed to parse grammar",
        "type": "invalid_request_error"
    }
}
```

### Extending or building alternative Web Front End

You can extend the front end by running the server binary with `--path` set to `./your-directory` and importing `/completion.js` to get access to the llamaComplete() method.

Read the documentation in `/completion.js` to see convenient ways to access llama.

A simple example is below:

```html
<html>
  <body>
    <pre>
      <script type="module">
        import { llama } from '/completion.js'

        const prompt = `### Instruction:
Write dad jokes, each one paragraph.
You can use html formatting if needed.

### Response:`

        for await (const chunk of llama(prompt)) {
          document.write(chunk.data.content)
        }
      </script>
    </pre>
  </body>
</html>
```
