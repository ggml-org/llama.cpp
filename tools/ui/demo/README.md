# Demo videos

Scripted Playwright walkthroughs of the UI, exported as mp4 with an animated GIF preview, for pull requests and documentation. The harness lives here; the demo scripts don't:

- **PR demos** live in the PR description, as a `ts demo` code block.
- **Docs demos** live in the storage target, under `scripts/docs/`.

```
demo/
├── cli.ts      demo:pr, demo:docs and demo:traces
├── storage.ts  storage targets (hf://, file://)
├── helpers.ts  `test` with a `demo` fixture: cursor, captions, human-like timing
├── recorder.ts screencast recording and mp4 export
├── fixtures/   fixture MCP servers, seeded settings and demo conversations
└── videos/     output (git-ignored)
```

## Setup

### Tools

| Tool                                                 | Used for                                                                                                                                                                                                          |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Node 22.18+                                          | runs `demo/cli.ts` directly (TypeScript type stripping)                                                                                                                                                           |
| `npx playwright install chromium`                    | the browser demos record in                                                                                                                                                                                       |
| `ffmpeg`                                             | mp4 encoding and the GIF preview                                                                                                                                                                                  |
| `gh`, logged in                                      | reads the PR description and writes the `## Demo` section, so you need edit rights on the PR                                                                                                                      |
| `hf`, logged in with a write token (`hf auth login`) | uploads to the bucket and the traces dataset                                                                                                                                                                      |
| `llama-server`                                       | `build/bin/llama-server`, or one already running at `DEMO_SERVER_ORIGIN` (default `:9931`). It runs in router mode, so demos can pick any model in the local cache; scripts that send messages need at least one. |

`gh` is only needed for `demo:pr`, and `hf` only with the bucket or dataset below. Without them, the video stays in `demo/videos/`.

Videos are recorded through Chromium's screencast at `DEMO_SCALE` (2x by default, so 2880x1800) and 60 fps, starting once the page has settled.

### Hugging Face bucket (videos, previews, scripts)

The bucket must be public, since GitHub loads the preview from it:

```bash
hf buckets create <owner>/llama-ui-demos
```

Then set it in `tools/ui/.env` (see `.env.example`):

```bash
DEMO_STORAGE='hf://buckets/<owner>/llama-ui-demos'
DEMO_STORAGE_PUBLIC_URL='https://huggingface.co/buckets/<owner>/llama-ui-demos/resolve'
```

Layout: `videos/pr/<number>/<sha>.{mp4,gif,demo.ts}`, `videos/docs/<name>.{mp4,gif}` and `scripts/docs/*.demo.ts` (the docs scripts). Without `DEMO_STORAGE_PUBLIC_URL`, videos are uploaded but not linked in the PR. `DEMO_STORAGE='file:///some/dir'` stores them in a local folder instead.

### Hugging Face dataset (conversations, optional)

Only needed for demos that import conversations other than the built-in ones, or to publish the built-in ones with `npm run demo:traces`. It can be private: it is read with your `hf` login.

```bash
hf repos create <owner>/llama-ui-demo-traces --repo-type dataset --private
```

```bash
DEMO_TRACES_DATASET='<owner>/llama-ui-demo-traces'
```

### Other storage

Only Hugging Face buckets and `file://` folders are supported for now. S3, R2 or similar would be one more class in `storage.ts` (`upload` and `downloadDir`) plus the matching `DEMO_STORAGE_PUBLIC_URL`.

## Recording a pull request

Add the script to the PR description:

````md
<details>
<summary>Demo script</summary>

```ts demo
test('demo', async ({ demo, page }) => {
	await demo.click(page.getByRole('button', { exact: true, name: 'Settings' }));
	await demo.caption('What this shows');
});
```

</details>
````

Then run:

```bash
npm run demo:pr -- 31303
```

It checks out the PR head in a temporary worktree, records the script against it, uploads `videos/pr/<number>/<sha>.{mp4,gif,demo.ts}` and adds or updates a `## Demo` section in the PR description with the preview linking to the video. GitHub has no API for attachments, so for an inline player drag `demo/videos/pr-<number>.mp4` into the description instead.

## Recording docs

```bash
npm run demo:docs
```

Records every `scripts/docs/*.demo.ts` from the storage target and uploads `videos/docs/<name>.{mp4,gif}`.

## Writing a demo

Scripts get `test` and `expect` from `helpers.ts` without importing them. `demo.click`, `demo.hover` and `demo.type` glide a visible cursor to the target before acting, with timing varied like a person's (seeded per demo, so re-takes match); `demo.scroll` turns the mouse wheel over a target; `demo.caption` shows a caption at the top of the frame. Use `test.use({ settings, startPath })` to seed other settings or start on another route.

## Conversations

Demos that need existing chats import them with `test.use({ traces: ['long-chat'] })`, through Settings > Import/Export before the video starts. Traces use the export format (llama.app JSONL), so the Hub shows them as Llama traces:

- `fixtures/traces.ts` generates the built-in ones: `long-chat` (36 tall exchanges) and `agentic-debugging` (long reasoning, shell output and code blocks).
- Any other name is downloaded as `<name>.jsonl` from `DEMO_TRACES_DATASET`, e.g. an exported conversation uploaded there.

`npm run demo:traces` writes the built-in traces to `demo/traces/` and pushes them to `DEMO_TRACES_DATASET`.

## Trying a script locally

To try a script locally, put it in `demo/` as `<name>.demo.ts`, importing from `./helpers.ts`, and run `npm run demo:record -- demo/<name>`. `npm run demo:codegen` opens Playwright codegen against `:5180` for a first draft.

The config starts, or reuses when already running, the fixture MCP servers on `:8765`, `llama-server` on `:9931` (router mode, `--tools all`) and the UI's vite dev server.
