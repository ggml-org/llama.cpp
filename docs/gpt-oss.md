# Running gpt-oss with llama.cpp

This guide shows how to download and serve OpenAI's gpt-oss models with
`llama-server`, including a workaround for a filename mismatch that
currently breaks the built-in convenience flags.

## Quick start (convenience flags)

llama.cpp ships convenience flags that are meant to download and run
gpt-oss with no extra arguments:

```sh
llama-server --gpt-oss-20b-default
# or for the larger model:
llama-server --gpt-oss-120b-default
```

> **Known issue:** at the time of writing, `--gpt-oss-20b-default` fails
> with:
>
> ```
> E common_download_get_hf_plan: file 'gpt-oss-20b-mxfp4.gguf' not found in repository
> I Available GGUF files:
> I  - gpt-oss-20b-MXFP4.gguf
> ```
>
> The flag requests a lowercase filename (`gpt-oss-20b-mxfp4.gguf`), but
> the actual file in the `ggml-org/gpt-oss-20b-GGUF` repo is uppercase
> (`gpt-oss-20b-MXFP4.gguf`). Since Hugging Face repo lookups are
> case-sensitive, the download fails before the server can start. Use the
> workaround below until this is fixed upstream.

## Workaround: specify repo and file separately

Rather than the `-default` flag or the `repo:file` colon syntax (which hits
the same case-sensitivity issue), pass the repo and filename as two
separate flags:

```sh
llama-server -hf ggml-org/gpt-oss-20b-GGUF -hff gpt-oss-20b-MXFP4.gguf --port 8012
```

- `-hf` (`--hf-repo`) selects the Hugging Face repository
- `-hff` (`--hf-file`) pins the exact filename, bypassing the broken quant
  auto-selection in the `-default` flag

This downloads the model on first run and starts the server. Expect the
initial download to take a while — gpt-oss-20b is a multi-gigabyte model,
noticeably larger than a typical embedding model.

Once it's ready, the log ends with:

```
I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 131072, kv_unified = 'true'
I srv  llama_server: model loaded
I srv  llama_server: listening on http://127.0.0.1:8012
```

The same `-hf` / `-hff` pattern works for `gpt-oss-120b` if you have the
hardware for it — just swap the repo and filename accordingly.

## Querying the server

`llama-server` exposes an OpenAI-compatible chat endpoint at
`/v1/chat/completions`:

```sh
curl http://127.0.0.1:8012/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}]}'
```

> **Windows PowerShell note:** as with other endpoints, inline
> double-quote escaping can get mangled in PowerShell. Write the body to a
> file first for reliability:
>
> ```powershell
> '{"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}]}' | Out-File -Encoding utf8 chat_body.json
> curl.exe http://127.0.0.1:8012/v1/chat/completions -H "Content-Type: application/json" --data-binary "@chat_body.json"
> ```

### Example response

```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello, how are you doing?",
        "reasoning_content": "The user requests: \"Say hello in exactly 5 words.\" ..."
      }
    }
  ],
  "model": "ggml-org/gpt-oss-20b-GGUF",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 276,
    "prompt_tokens": 75,
    "total_tokens": 351
  }
}
```

Note the `reasoning_content` field: gpt-oss is a reasoning model, so the
server returns its chain-of-thought separately from the final `content`.
In this example, the model explicitly counted words to make sure the
response was exactly five, then produced its final answer. If you only
want the final answer in your application, read `message.content` and
treat `reasoning_content` as optional debug/trace information.

## See also

- [server README](../tools/server/README.md) for the full list of server
  flags and endpoints
- [Computing embeddings with llama.cpp](./embeddings.md) for running
  llama.cpp in embedding mode instead of chat mode
