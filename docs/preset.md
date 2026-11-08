# llama.cpp INI Presets

## Introduction

The INI preset feature, introduced in [PR#17859](https://github.com/ggml-org/llama.cpp/pull/17859), allows users to create reusable and shareable parameter configurations for llama.cpp.

## Using Presets with the Server

When running multiple models on the server (router mode), INI preset files can be used to configure model-specific parameters. Please refer to the [server documentation](../tools/server/README.md) for more details.

### Using a Hugging Face Preset

> [!IMPORTANT]
>
> Please only use presets that you can trust! Unknown presets may be unsafe

You can push your preset to Hugging Face Hub and share with other users by:
1. Creating an empty model repository on Hugging Face
2. Creating a `preset.ini` file in the root directory of the repository

Example of a `preset.ini`:

```ini
[*]
ctx-size             = 0
mmap                 = 1
kv-unified           = 1
parallel             = 4
spec-default         = 1

[Qwen3.5-4B]
hf                   = unsloth/Qwen3.5-4B-GGUF:Q4_K_M
ctx-size             = 262144
batch-size           = 2048
ubatch-size          = 2048
top-p                = 1.0
top-k                = 0
min-p                = 0.01
temp                 = 1.0

[gpt-oss-120b-hf]
hf                   = ggml-org/gpt-oss-120b-GGUF
ctx-size             = 262144
batch-size           = 2048
ubatch-size          = 2048
top-p                = 1.0
top-k                = 0
min-p                = 0.01
temp                 = 1.0
chat-template-kwargs = {"reasoning_effort": "high"}
```

The preset will be loaded similarly to the `--models-preset` option. Therefore, you can also override certain params via CLI arguments:

```sh
# Force temp = 0.1, overriding the preset value
llama-cli -hf username/my-preset --temp 0.1
```

### Named presets

If you want to define multiple preset configurations for one or more GGUF models, you can create a blank HF repo containing a single `preset.ini` file that references the actual model(s):

```ini
[*]
mmap = 1

[gpt-oss-20b-hf]
hf          = ggml-org/gpt-oss-20b-GGUF
batch-size  = 2048
ubatch-size = 2048
top-p       = 1.0
top-k       = 0
min-p       = 0.01
temp        = 1.0
chat-template-kwargs = {"reasoning_effort": "high"}

[gpt-oss-120b-hf]
hf          = ggml-org/gpt-oss-120b-GGUF
batch-size  = 2048
ubatch-size = 2048
top-p       = 1.0
top-k       = 0
min-p       = 0.01
temp        = 1.0
chat-template-kwargs = {"reasoning_effort": "high"}
```

You can then use it via `llama-cli` or `llama-server`, example:

```sh
llama-server -hf user/repo:gpt-oss-120b-hf
```

Please make sure to provide the correct `hf-repo` for each child preset. Otherwise, you may get error: `The specified tag is not a valid quantization scheme.`

## System-level config

The system-level config, added in PR [#26118](https://github.com/ggml-org/llama.cpp/pull/26118), allows sharing the same set of options among multiple tools and examples. Unlike the sections above, it is not limited to the server.

These files are loaded on startup if present. A later file overrides an earlier one:
1. System-wide: `/etc/llama.cpp/config.ini` (or `%PROGRAMDATA%\llama.cpp\config.ini` on Windows)
2. User-level: `$XDG_CONFIG_HOME/llama.cpp/config.ini`, `~/.config/llama.cpp/config.ini` by default (or `%APPDATA%\llama.cpp\config.ini` on Windows)

The config file is applied first, then its options are overridden by ENV variables, CLI arguments and model presets (in router mode).

Note:
- Only the `[*]` and default sections are used; options written before any section header belong to "default. Named sections are ignored
- Tool-specific options can be specified, but will be ignored (with a warning) if the example doesn't support it<br/>Example: if you specify `port = 1234`, only `llama-server` will use it, other examples will ignore it
- `model` or `hf-repo` are not recommended to be configured system-level, because it may introduce conflicts<br/>Example: a `hf-repo` in the config file still takes effect when you pass `-m` on the command line, so you may load a different model than expected
