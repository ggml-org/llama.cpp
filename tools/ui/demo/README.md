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

Set the storage target in `tools/ui/.env` (see `.env.example`):

```bash
DEMO_STORAGE='hf://buckets/<owner>/<bucket>'   # or file:///some/dir
DEMO_STORAGE_PUBLIC_URL='https://...'          # public base URL of the same location
```

`hf://` uses the `hf` CLI login. Without `DEMO_STORAGE`, videos stay in `demo/videos/`; without `DEMO_STORAGE_PUBLIC_URL`, they are uploaded but not linked in the PR.

`ffmpeg` is required for the mp4 and GIF exports. Videos are recorded through Chromium's screencast at `DEMO_SCALE` (2x by default, so 2880x1800) and 60 fps, starting once the page has settled.

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
