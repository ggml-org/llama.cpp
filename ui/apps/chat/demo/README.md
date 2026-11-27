# Llama Chat demo videos

Llama Chat's fixtures for the demo harness in [`ui/lib/demo`](../../../lib/demo/README.md), which records, renders and publishes the videos. Setup, PR and docs recording, the `demo.*` API and the render are described there; this folder sets the scene for Chat.

```text
demo/
├── cli.ts                demo:pr, demo:docs and demo:traces
├── helpers.ts            `test` with Chat's fixtures and the `demo` fixture
├── playwright.config.ts  the servers: fixture MCP servers, llama-server, vite and Storybook
├── fixtures/             fixture MCP servers, seeded settings, demo conversations, a mocked llama-server
└── videos/               output (git-ignored)
```

| Command                              | Does                                                              |
| ------------------------------------ | ----------------------------------------------------------------- |
| `npm run demo:record -- demo/<name>` | records `demo/<name>.demo.ts` (git-ignored) into `demo/videos/`   |
| `npm run demo:pr -- <number>`        | records the PR's `ts demo` block and writes its `## Demo` section |
| `npm run demo:docs`                  | records the docs scripts from `DEMO_STORAGE`                      |
| `npm run demo:traces`                | writes the built-in traces to `demo/traces/`, and pushes them     |
| `npm run demo:codegen`               | opens Playwright codegen against `:5180` for a first draft        |

Settings come from `ui/apps/chat/.env` (see `.env.example`).

## Setting the scene

Scripts get `test` and `expect` from `helpers.ts` without importing them. Use `test.use({ settings, startPath })` to seed other settings or start on another route, and `chats: <n>` to add empty conversations to the library.

### A mocked llama-server

`test.use({ mockServer: { models: ['org/Model-GGUF:Q4_K_M'], reply: 'A **Markdown** reply' } })` answers llama-server's endpoints in the browser (`fixtures/llama-server.ts`): the model list, props, model loading, tools and a streamed reply. Nothing reaches the server and no model loads, so a demo that sends a message looks the same on every machine. `tests/e2e` checks call `mockLlamaServer(page)` directly, since the e2e build runs without a server.

### Storybook stories

`test.use({ story: 'ui-forms-switch--default' })` records a Storybook story (the id from its `?path=/story/<id>` URL) instead of Llama Chat, from the Storybook that `DEMO_STORYBOOK` adds to the servers: `ui` for Llama UI (`ui/lib/ui`), `chat` for Llama Chat's. It runs on `DEMO_STORYBOOK_PORT` (default `:6007`). A small component reads better at a smaller `viewport` than zoomed in.

```bash
DEMO_STORYBOOK=ui npm run demo:record -- demo/<name>
```

### Conversations

Demos that need existing chats import them with `test.use({ traces: ['long-chat'] })`, through Settings > Data before the video starts. Traces use the export format (llama.app JSONL), so the Hub shows them as Llama traces:

- `fixtures/traces.ts` generates the built-in ones: `long-chat` (36 tall exchanges) and `agentic-debugging` (long reasoning, shell output and code blocks).
- Any other name is downloaded as `<name>.jsonl` from `DEMO_TRACES_DATASET`, e.g. an exported conversation uploaded there.

The dataset is only needed for other conversations, or to publish the built-in ones with `npm run demo:traces`. It can be private: it is read with your `hf` login.

```bash
hf repos create <owner>/llama-ui-demo-traces --repo-type dataset --private
```

```bash
DEMO_TRACES_DATASET='<owner>/llama-ui-demo-traces'
```

## Servers

The config starts, or reuses when already running, the fixture MCP servers on `:8765`, `llama-server` on `:9931` (`build/bin/llama-server` in router mode with `--tools all`, so demos can pick any model in the local cache), the UI's vite dev server and, with `DEMO_STORYBOOK`, a Storybook. Scripts that send messages to the real server need at least one model; `mockServer` needs none.
