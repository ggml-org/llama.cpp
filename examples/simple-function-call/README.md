# Simple Function Call Example

A standalone executable that demonstrates function calling ***from scratch*** with llama.cpp, allowing natural language to shell command execution.

## What This Is

- Users input natural language requests
- The LLM generates appropriate shell commands using function calling
- Commands are executed and results returned
- The conversation continues with real command output

## How It Works

### Architecture
- **Standalone executable** - no server architecture needed
- **Direct function calling** within the same process
- **Real shell command execution** via `popen()`
- **JSON parsing** of LLM tool calls

##### Tool Schema
The LLM has access to a single tool:
```json
{
  "name": "shell_command",
  "description": "Execute a shell command and return the output",
  "parameters": {
    "type": "object",
    "properties": {
      "command": {
        "type": "string",
        "description": "The shell command to execute"
      }
    },
    "required": ["command"]
  }
}
```

### Basic Usage
```
./simple-function-call -m model.gguf -p "your request here"
```
### Command Line Arguments
- `--jinja` Is enabled by default
- `-m model.gguf` - Model file path (REQUIRED)
- `-p "prompt"` - User's request/command (REQUIRED)
- `--chat-template-file template.jinja` - Optional chat template override
- `--grammar "grammar"` - Optional grammar constraint
- `-ngl N` - Number of GPU layers
- `-n N` - Maximum number of tokens to generate
- `--confirm` - Ask for confirmation before executing commands

### Examples

```
./build/bin/llama-simple-function-call -m ~/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf -p "Make a directory llamacppchaos"
Simple Function Call Example
Model: /home/user/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf
Prompt: Make a directory llamacppchaos
GPU layers: 99
Max tokens: 256

llama_model_loader: loaded meta data with 36 key-value pairs and 147 tensors from /home/user/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf (version GGUF V3 (latest))
...
...
...
llama_context:        CPU compute buffer size =   254.50 MiB
llama_context: graph nodes  = 582
llama_context: graph splits = 1
{"type": "function", "name": "shell_command", "parameters": {"command": "mkdir llamacppchaos"}}

Function calls detected:
  Function: shell_command
  Arguments: {"command":"mkdir llamacppchaos"}
  Command: mkdir llamacppchaos
  Result:

```

```
./build/bin/llama-simple-function-call -m ~/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf -p "List all the files in the current directory"
Simple Function Call Example
Model: /home/user/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf
Prompt: List all the files in the current directory
GPU layers: 99
Max tokens: 256

llama_model_loader: loaded meta data with 36 key-value pairs and 147 tensors from /home/user/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf (version GGUF V3 (latest))
...
...
...
llama_context:        CPU compute buffer size =   254.50 MiB
llama_context: graph nodes  = 582
llama_context: graph splits = 1

{"name": "shell_command", "parameters": {"command": "ls -l"}}

Function calls detected:
  Function: shell_command
  Arguments: {"command":"ls -l"}
  Command: ls -l
  Result:
total 840
-rw-r--r--  1 user user  47860 Jul 14 19:12 AUTHORS
drwxr-xr-x 12 user user   4096 Jul 14 19:13 build
-rwxr-xr-x  1 user user  21760 Jul 14 19:12 build-xcframework.sh
drwxr-xr-x  2 user user   4096 Jul 14 19:12 ci
drwxr-xr-x  2 user user   4096 Jul 14 19:12 cmake
-rw-r--r--  1 user user   7973 Jul 14 19:12 CMakeLists.txt
-rw-r--r--  1 user user   4570 Jul 14 19:12 CMakePresets.json
-rw-r--r--  1 user user    434 Jul 14 19:12 CODEOWNERS
drwxr-xr-x  2 user user   4096 Jul 14 19:12 common
-rw-r--r--  1 user user   6510 Jul 14 19:12 CONTRIBUTING.md
-rwxr-xr-x  1 user user 344837 Jul 14 19:12 convert_hf_to_gguf.py
-rwxr-xr-x  1 user user  22622 Jul 14 19:12 convert_hf_to_gguf_update.py
-rwxr-xr-x  1 user user  19106 Jul 14 19:12 convert_llama_ggml_to_gguf.py
-rwxr-xr-x  1 user user  18624 Jul 14 19:12 convert_lora_to_gguf.py
drwxr-xr-x  6 user user   4096 Jul 14 19:12 docs
drwxr-xr-x 29 user user   4096 Jul 14 19:12 examples
-rw-r--r--  1 user user   1556 Jul 14 19:12 flake.lock
-rw-r--r--  1 user user   7465 Jul 14 19:12 flake.nix
drwxr-xr-x  5 user user   4096 Jul 14 19:12 ggml
drwxr-xr-x  5 user user   4096 Jul 14 19:12 gguf-py
drwxr-xr-x  2 user user   4096 Jul 14 19:12 grammars
drwxr-xr-x  2 user user   4096 Jul 14 19:12 include
-rw-r--r--  1 user user   1078 Jul 14 19:12 LICENSE
drwxr-xr-x  2 user user   4096 Jul 14 19:12 licenses
drwxr-xr-x  2 user user   4096 Jul 14 19:15 llamacppchaos
-rw-r--r--  1 user user  50442 Jul 14 19:12 Makefile
drwxr-xr-x  2 user user   4096 Jul 14 19:12 media
drwxr-xr-x  3 user user   4096 Jul 14 19:12 models
-rw-r--r--  1 user user    163 Jul 14 19:12 mypy.ini
drwxr-xr-x  3 user user   4096 Jul 14 19:12 pocs
-rw-r--r--  1 user user 124786 Jul 14 19:12 poetry.lock
drwxr-xr-x  2 user user   4096 Jul 14 19:12 prompts
-rw-r--r--  1 user user   1336 Jul 14 19:12 pyproject.toml
-rw-r--r--  1 user user    616 Jul 14 19:12 pyrightconfig.json
-rw-r--r--  1 user user  29598 Jul 14 19:12 README.md
drwxr-xr-x  2 user user   4096 Jul 14 19:12 requirements
-rw-r--r--  1 user user    551 Jul 14 19:12 requirements.txt
drwxr-xr-x  3 user user   4096 Jul 14 19:12 scripts
-rw-r--r--  1 user user   5347 Jul 14 19:12 SECURITY.md
drwxr-xr-x  2 user user   4096 Jul 14 19:12 src
drwxr-xr-x  2 user user   4096 Jul 14 19:12 tests
drwxr-xr-x 17 user user   4096 Jul 14 19:12 tools
drwxr-xr-x  7 user user   4096 Jul 14 19:12 vendor
```

### Where it might fail

```
 ./build/bin/llama-simple-function-call -m ~/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf -p "Check the time"
Simple Function Call Example
Model: /home/user/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf
Prompt: Check the time
GPU layers: 99
Max tokens: 256

llama_model_loader: loaded meta data with 36 key-value pairs and 147 tensors from /home/user/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf (version GGUF V3 (latest))

...
...
...
llama_context:        CPU compute buffer size =   254.50 MiB
llama_context: graph nodes  = 582
llama_context: graph splits = 1
{
    "type": "function",
    "function": {
        "name": "shell_command",
        "description": "Check the time",
        "parameters": {
            "type": "string",
            "description": "The shell command to execute"
        }
    },
    "result": {
        "type": "object",
        "value": {
            "time": "Current time"
        }
    }
}

Response: {
    "type": "function",
    "function": {
        "name": "shell_command",
        "description": "Check the time",
        "parameters": {
            "type": "string",
            "description": "The shell command to execute"
        }
    },
    "result": {
        "type": "object",
        "value": {
            "time": "Current time"
        }
    }
}
```
### You'll have less of these types of problems with bigger models but you can't still nudge the smaller ones, if you have some knowledge you can always tell it to run commands directly but it defeats the purpose of it a bit, it would be good with some Speech to Text system where you wouldn't have to type but just say what you want the computer to do
```
./build/bin/llama-simple-function-call -m ~/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf -p "Check the current date"
Simple Function Call Example
Model: /home/user/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf
Prompt: Check the current date
GPU layers: 99
Max tokens: 256

llama_model_loader: loaded meta data with 36 key-value pairs and 147 tensors from /home/user/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf (version GGUF V3 (latest))
...
...
...
llama_context:        CPU compute buffer size =   254.50 MiB
llama_context: graph nodes  = 582
llama_context: graph splits = 1
{"type": "function", "name": "shell_command", "parameters": {"command": "date"}}

Function calls detected:
  Function: shell_command
  Arguments: {"command":"date"}
  Command: date
  Result:
Mon Jul 14 07:26:14 PM CEST 2025
```
# With confirmation ( For safety )
```
./build/bin/llama-simple-function-call -m ~/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf -p "Remove directory llamacppchaos" --confirm
Simple Function Call Example
Model: /home/user/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf
Prompt: Remove directory llamacppchaos
GPU layers: 99
Max tokens: 256
Command confirmation: enabled

llama_model_loader: loaded meta data with 36 key-value pairs and 147 tensors from /home/user/Downloads/Llama-3.2-1B-Instruct-Q6_K.gguf (version GGUF V3 (latest))
...
...
...
llama_context:        CPU compute buffer size =   254.50 MiB
llama_context: graph nodes  = 582
llama_context: graph splits = 1
{"type": "function", "name": "shell_command", "parameters": {"command": "rm -rf llamacppchaos"}}

Function calls detected:
  Function: shell_command
  Arguments: {"command":"rm -rf llamacppchaos"}
  Command: rm -rf llamacppchaos
  Execute this command? (y/N): y
  Result:

```

### Using a specific chat template for better function calling
```
./simple-function-call -m qwen2.5-7b-instruct.gguf -p "list files in current directory" \
    --chat-template-file models/templates/Qwen-Qwen2.5-7B-Instruct.jinja
```
