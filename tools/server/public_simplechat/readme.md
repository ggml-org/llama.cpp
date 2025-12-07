# SimpleChat / AnveshikaSallap

by Humans for All.

A lightweight simple minded ai chat client, which runs in a browser environment, with a web front-end that supports multiple chat sessions, vision models, reasoning and tool calling (including bundled tool calls - some browser native and some bundled simplemcp based).


## Quickstart

### Server

From the root directory of llama.cpp source code repo containing build / tools / ... sub directories

Start ai engine / server using

```bash
build/bin/llama-server -m <path/to/model.gguf> \
  --path tools/server/public_simplechat --jinja
```

- `--jinja` enables tool‑calling support
- `--mmproj <path/to/mmproj.gguf>` enables vision support
- `--port <port number>` use if a custom port is needed
  - default is 8080 wrt llama-server

If one needs web related access / tool calls dont forget to run

```bash
cd tools/server/public_simplechat/local.tools; python3 ./simplemcp.py --config simplemcp.json
```

- `--debug True` enables debug mode which captures internet handshake data
- port defaults to 3128, can be changed from simplemcp.json, if needed
- add sec.keyFile and sec.certFile to simplemcp.json, for https mode;
  - also dont forget to change mcpServerUrl to mention https scheme

### Client

1. Open `http://127.0.0.1:8080/index.html` in a browser
   - assuming one is running the llama-server locally with its default port

2. Select / Create a chat session
   - set a suitable system prompt, if needed
   - modify **settings**, if needed
     - modifying mcpServerUrl wont reload supported tool calls list, till next app page refresh
   - **Restore** loads last autosaved session with same name

3. Enter query/response into user input area at the bottom, press **Enter**
   - use **Shift‑Enter** for newline
   - include images if required (ai vision models)

4. View any streamed ai response (if enabled and supported)

5. If a tool call is requested
   - verify / edit the tool call details before triggering the same
     - one can even ask ai to rethink on the tool call requested,
       by sending a appropriate user response instead of a tool call response
   - tool call is executed using Browser's web worker or included SimpleMCP.py
   - tool call response is placed in user input area
     - the user input area is color coded to distinguish between user and tool responses
   - verify / edit the tool call response, before submit same back to ai
     - tool response initially assigned `TOOL-TEMP` role, promoted to `TOOL` upon submit
   - based on got response, if needed one can rerun tool call with modified arguments
   - at any time there can be one pending tool call wrt a chat session

6. **Delete & Copy** available via popover menu for each message

7. **Clear / + New** chat with provided buttons, as needed


## Overview

A lightweight simple minded ai chat client, which runs in a browser environment, with a web front-end that supports multiple chat sessions, vision models, reasoning and tool calling (including bundled tool calls - some browser native and some bundled simplemcp based).

- Supports multiple independent chat sessions with
  - One‑shot or Streamed (default) responses
  - Custom settings and system prompts per session
  - Automatic local autosave (restorable on next load)
  - can handshake with `/completions` or `/chat/completions` (default) endpoints

- Supports peeking at model's reasoning live
  - if model streams the same and
  - streaming mode is enabled in settings (default)

- Supports vision / image / multimodal ai models
  - attach image files as part of user chat messages
    - handshaked as `image_url`s in chat message content array along with text
    - supports multiple image uploads per message
    - images displayed inline in the chat history
  - specify `mmproj` file via `-mmproj` or using `-hf`
  - specify `-batch-size` and `-ubatch-size` if needed

- Built-in support for GenAI/LLM models that support tool calling

  - includes a bunch of useful builtin tool calls, without needing any additional setup

  - building on modern browsers' flexibility, following tool calls are directly supported by default
    - `sys_date_time`, `simple_calculator`, `run_javascript_function_code`, `data_store_*`, `external_ai`
    - except for external_ai, these are run from within a web worker context to isolate main context from them
    - data_store brings in browser IndexedDB based persistant key/value storage across sessions

  - in collaboration with included python based simplemcp.py, these additional tool calls are supported
    - `search_web_text`, `fetch_url_raw`, `fetch_html_text`, `fetch_pdf_as_text`, `fetch_xml_filtered`
      - these built‑in tool calls (via SimpleMCP) help fetch PDFs, HTML, XML or perform web search
      - PDF tool also returns an outline with numbering, if available
      - result is truncated to `iResultMaxDataLength` (default 128 kB)
    - helps isolate core of these functionality into a separate vm running locally or otherwise, if needed
    - supports whitelisting of `acl.schemes` and `acl.domains` through `simplemcp.json`
    - supports a bearer token shared between server and client for auth
      - needs https mode to be enabled, for better security wrt this flow
      - by default simplemcp.py runs in http mode,
        however if sec.keyFile and sec.certFile are specified, the logic switches to https mode
    - this handshake is loosely based on MCP standard, doesnt stick to the standard fully

  - follows a safety first design and lets the user
    - verify and optionally edit the tool call requests, before executing the same
    - verify and optionally edit the tool call response, before submitting the same
    - user can update the settings for auto executing these actions, if needed

  - external_ai allows invoking a separate optionally fresh by default ai instance
    - by default in such a instance
      - tool calling is kept disabled along with
      - client side sliding window of 1,
        ie only system prompt and latest user message is sent to ai server.
    - TCExternalAI is the special chat session used internally for this,
      and the default behaviour will get impacted if you modify the settings of this special chat session.
      - Restarting this chat client logic will force reset things to the default behaviour,
        how ever any other settings wrt TCExternalAi, that where changed, will persist across restarts.
      - this instance maps to the current ai server itself by default, but can be changed by user if needed.
    - could help with handling specific tasks using targetted personas or models
      - ai could run self modified targeted versions of itself/... using custom system prompts and user messages as needed
      - user can setup ai instance with additional compute, which should be used only if needed, to keep costs in control
      - can enable a modular pipeline with task type and or job instance specific decoupling, if needed
    - tasks offloaded could include
      - summarising, data extraction, formatted output, translation, ...
      - creative writing, task breakdown, ...

- Client side Sliding window Context control, using `iRecentUserMsgCnt`, helps limit context sent to ai model

- Optional
  - simple minded markdown parsing of chat message text contents (default wrt assistant messages/responses)
    - user can override, if needed globally or at a individual message level
  - auto trimming of trailing garbage from model outputs

- Follows responsive design to try adapt to any screen size

- built using plain html + css + javascript and python
  - no additional dependencies that one needs to worry about and inturn keep track of
    - except for pypdf, if pdf support needed. automaticaly drops pdf tool call support, if pypdf missing
  - fits within ~50KB compressed source or ~300KB in uncompressed source form (both including simplemcp.py)
  - easily extend with additional tool calls using either javascript or python, for additional functionality
    as you see fit

Start exploring / experimenting with your favorite ai models and thier capabilities.


## Configuration / Settings

One can modify the session configuration using Settings UI. All the settings and more are also exposed in the browser console via `document['gMe']`.

### Settings Groups

| Group | Purpose |
|---------|---------|
| `chatProps` | ApiEndpoint, streaming, sliding window, markdown, ... |
| `tools` | `enabled`, `mcpServerUrl`, `mcpServerAuth`, search URL/template & drop rules, max data length, timeouts |
| `apiRequestOptions` | `temperature`, `max_tokens`, `frequency_penalty`, `presence_penalty`, `cache_prompt`, ... |
| `headers` | `Content-Type`, `Authorization`, ... |

### Some specific settings

- **Ai Server** (`baseURL`)
  - ai server (llama-server) address
  - default is `http://127.0.0.1:8080`
- **SimpleMCP Server** (`mcpServerUrl`)
  - the simplemcp.py server address
  - default is `http://127.0.0.1:3128`
- **Stream** (`stream`)
  - `true` for live streaming, `false` for oneshot
- **Client side Sliding Window** (`iRecentUserMsgCnt`)
  - `-1` : send full history
  - `0`  : only system prompt
  - `>0` : last N user messages after the most recent system prompt
- **Cache Prompt** (`cache_prompt`)
  - enables server‑side caching of system prompt and history to an extent
- **Tool Call Timeout** (`toolCallResponseTimeoutMS`)
  - 200s by default
- **Tool call Auto** (`autoSecs`)
  - seconds to wait before auto-triggering tool calls and auto-submitting tool responses
  - default is 0 ie manual
- **Trim Garbage** (`bTrimGarbage`)
  - tries to remove repeating trailing text


## Debugging Tips

- **Local TCPdump**
  - `sudo tcpdump -i lo -s 0 -vvv -A host 127.0.0.1 and port 8080`
- **Browser DevTools**
  - inspect `document['gMe']` for session state
- **Reset Tool Call**
  - delete any assistant response after the tool call handshake
  - next wrt the last tool message
    - set role back to `TOOL-TEMP`
      - edit the response as needed
    - or delete the same
      - user will be given option to edit and retrigger the tool call
    - submit the new response


## At the end

A thank you to all open source and open model developers, who strive for the common good.
