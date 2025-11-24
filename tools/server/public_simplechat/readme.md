# SimpleChat / AnveshikaSallap

by Humans for All.

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

If one needs web related access / tool calls dont forget to run

```bash
cd tools/server/public_simplechat/local.tools; python3 ./simpleproxy.py --config simpleproxy.json
```

- `--debug True` enables debug mode which captures internet handshake data.

### Client

1. Open `http://127.0.0.1:PORT/index.html` in a browser

2. Select / Create a chat session
   - set a suitable system prompt, if needed
   - modify **settings**, if needed
   - **Restore** loads last autosaved session with same name

3. Enter the query, press **Enter**
   - use **Shift‑Enter** for newline
   - include images if required (ai vision models)

4. View any streamed ai response (if enabled and supported)

5. If a tool call is requested
   - verify / edit the tool call details before triggering the same
     - one can even ask ai to rethink on the tool call requested,
       by sending a appropriate user response instead of a tool call response.
   - tool call is executed using browser's web worker or simpleproxy
   - tool call response is placed in user input area (with color coding)
   - verify / edit the tool call response, before submit same back to ai
     - tool response initially assigned `TOOL-TEMP` role, promoted to `TOOL` upon submit
   - based on got response, if needed one can rerun tool call with modified arguments
   - *at any time there can be one pending tool call wrt a chat session*

6. **Delete / Copy** available via popover menu for each message

7. **Clear** / **+ New** chat with provided buttons, as needed


## Overview

A lightweight ai chat client with a web front-end that supports multiple chat sessions, vision, reasoning and tool calling.

- Supports multiple independent chat sessions with
  - One‑shot or streamed (default) responses
  - Custom settings and system prompts per session
  - Automatic local autosave (restorable on next load)
  - can handshake with `/completions` or `/chat/completions` (default) endpoints

- Supports peeking at model's reasoning live
  - if model streams the same and
  - streaming mode is enabled in settings (default)

- Supports vision / image / multimodal ai models
  - attach image files as part of user chat messages
    - handshaked as `image_url`s in chat message content array along with text
    - supports multi-image uploads per message
    - images displayed inline in the chat history
  - specify `mmproj` file via `-mmproj` or using `-hf`
  - specify `-batch-size` and `-ubatch-size` if needed

- Built-in support for GenAI models that expose tool calling

  - includes a bunch of useful builtin tool calls, without needing any additional setup

  - direct browser based tool calls include
    - `sys_date_time`, `simple_calculator`, `run_javascript_function_code`, `data_store_*`, `external_ai`
    - except for external_ai, these are run from within a web worker context to isolate main context from them
    - data_store brings in browser IndexedDB based persistant key/value storage across sessions

  - along with included python based simpleproxy.py
    - `search_web_text`, `fetch_web_url_raw`, `fetch_html_text`, `fetch_pdf_as_text`, `fetch_xml_filtered`
      - these built‑in tool calls (via SimpleProxy) help fetch PDFs, HTML, XML or perform web search
      - PDF tool also returns an outline with numbering
      - result is truncated to `iResultMaxDataLength` (default 128 kB)
    - helps isolate these functionality into a separate vm running locally or otherwise, if needed
    - supports whitelisting of `allowed.schemes` and `allowed.domains` through `simpleproxy.json`
    - supports a bearer token shared between server and client for auth
      - needs https support, for better security wrt this flow, avoided now given mostly local use

  - follows a safety first design and lets the user
    - verify and optionally edit the tool call requests, before executing the same
    - verify and optionally edit the tool call response, before submitting the same
    - user can update the settings for auto executing these actions, if needed

  - external_ai allows invoking a separate fresh ai instance
    - ai could run self modified targeted versions of itself/... with custom system prompts and user messages as needed
    - user can bring in an ai instance with additional compute access, which should be used only if needed
    - tool calling is currently kept disabled in such a instance

- Client side Sliding window Context control, using `iRecentUserMsgCnt`, helps limit context sent to ai model

- Optional auto trimming of trailing garbage from model outputs

- Follows responsive design to try adapt to any screen size

- built using plain html + css + javascript and python
  - no additional dependencies that one needs to keep track of
    - except for pypdf, if pdf support needed. automaticaly drops pdf tool call if pypdf missing
  - fits within ~260KB even in uncompressed source form (including simpleproxy.py)
  - easily extend with additional tool calls using either javascript or python, for additional functionality

Start exploring / experimenting with your favorite ai models and thier capabilities.


## Configuration / Settings

One can modify the session configuration using Settings UI. All the settings and more are also exposed in the browser console via `document['gMe']`.

### Settings Groups

| Group | Purpose |
|---------|---------|
| `chatProps` | ApiEndpoint, streaming, sliding window, ... |
| `tools` | `enabled`, `proxyUrl`, `proxyAuthInsecure`, search URL/template & drop rules, max data length, timeouts |
| `apiRequestOptions` | `temperature`, `max_tokens`, `frequency_penalty`, `presence_penalty`, `cache_prompt`, ... |
| `headers` | `Content-Type`, `Authorization`, ... |

### Some specific settings

- **Ai Server** (`baseURL`)
  - ai server (llama-server) address
  - default is `http://127.0.0.1:PORT`
- **Stream** (`stream`)
  - `true` for live streaming, `false` for oneshot
- **Client side Sliding Window** (`iRecentUserMsgCnt`)
  - `-1` : send full history
  - `0`  : only system prompt
  - `>0` : last N user messages after the most recent system prompt
- **Cache Prompt** (`cache_prompt`)
  - enables server‑side caching of system prompt and history
- **Tool Call Timeout** (`toolCallResponseTimeoutMS`)
  - 200s by default
- **Tool call Auto** (`autoSecs`)
  - seconds to wait before auto-triggering tool calls and auto-submitting tool responses
  - default is 0 ie manual
- **Trim Garbage** (`bTrimGarbage`)
  - Removes repeated trailing text


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
    - delete the same
      - user will be given option to edit and retrigger the tool call
    - submit the new response


## At the end

A thank you to all open source and open model developers, who strive for the common good.
