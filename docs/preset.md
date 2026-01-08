# llama.cpp INI Presets

## Introduction

The INI preset feature, introduced in [PR#17859](https://github.com/ggml-org/llama.cpp/pull/17859), allows users to create reusable and shareable parameter configurations for llama.cpp.

### Using Presets with the Server

When running multiple models on the server (router mode), INI preset files can be used to configure model-specific parameters. Please refer to the [server documentation](../tools/server/README.md) for more details.

### Using a Remote Preset

> [!NOTE]
>
> This feature is currently only supported via the `-hf` option.

For GGUF models hosted on Hugging Face, you can include a `preset.ini` file in the root directory of the repository to define specific configurations for that model.

Example:

```ini
hf-repo-draft = username/my-draft-model-GGUF
temp = 0.5
top-k = 20
top-p = 0.95
```

For security reasons, only certain options are allowed. Please refer to [preset.cpp](../common/preset.cpp) for the complete list of permitted options.

Example usage:

Assuming your repository `username/my-model-with-preset` contains a `preset.ini` with the configuration above:

```sh
llama-cli -hf username/my-model-with-preset

# This is equivalent to:
llama-cli -hf username/my-model-with-preset \
  --hf-repo-draft username/my-draft-model-GGUF \
  --temp 0.5 \
  --top-k 20 \
  --top-p 0.95
```

You can also override preset arguments by specifying them on the command line:

```sh
# Force temp = 0.1, overriding the preset value
llama-cli -hf username/my-model-with-preset --temp 0.1
```

If you want to define multiple preset configurations for one or more GGUF models, you can create a blank HF repo for each preset. Each HF repo should contain a `preset.ini` file that references the actual model(s):

```ini
hf-repo = user/my-model-main
hf-repo-draft = user/my-model-draft
temp = 0.8
ctx-size = 1024
; (and other configurations)
```
