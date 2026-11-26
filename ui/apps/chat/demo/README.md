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
├── polish.ts   the render: window, ground, camera, captions
├── fixtures/   fixture MCP servers, seeded settings and demo conversations
└── videos/     output (git-ignored)
```

## Setup

### Tools

| Tool                                                 | Used for                                                                                                                                                                                                          |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Node 22.18+                                          | runs `demo/cli.ts` directly (TypeScript type stripping)                                                                                                                                                           |
| `npx playwright install chromium`                    | the browser demos record in, with its sandbox on                                                                                                                                                                  |
| `ffmpeg`                                             | mp4 encoding and the GIF preview                                                                                                                                                                                  |
| `gh`, logged in                                      | reads the PR description and writes the `## Demo` section, so you need edit rights on the PR                                                                                                                      |
| `hf`, logged in with a write token (`hf auth login`) | uploads to the bucket and the traces dataset                                                                                                                                                                      |
| `llama-server`                                       | `build/bin/llama-server`, or one already running at `DEMO_SERVER_ORIGIN` (default `:9931`). It runs in router mode, so demos can pick any model in the local cache; scripts that send messages need at least one. |

`gh` is only needed for `demo:pr`, and `hf` only with the bucket or dataset below. Without them, the video stays in `demo/videos/`.

The page is recorded through Chromium's screencast at `DEMO_SCALE` (2x by default, so 2880x1800) and 60 fps, starting once the page has settled. The video is then rendered from those frames at 1920x1080, or 1080x1920 for a portrait viewport (see [The render](#the-render)).

### Hugging Face bucket (videos, previews, scripts)

The bucket must be public, since GitHub loads the preview from it:

```bash
hf buckets create <owner>/llama-ui-demos
```

Then set it in `ui/apps/chat/.env` (see `.env.example`):

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

A PR by someone else runs its code and demo script on your machine, with your `gh` and `hf` logins, so `demo:pr` refuses it until you've read its diff and demo block and pass the head commit you read: `npm run demo:pr -- 31303 --trust <sha>`. If the PR gets new commits, it refuses again.

It checks out the PR head in a temporary worktree, records the script against it, uploads `videos/pr/<number>/<sha>.{mp4,gif,demo.ts}` and adds or updates a `## Demo` section in the PR description with the preview linking to the video.

A large PR can show its changes in several shorter videos: each `test()` in the block is recorded as its own video (`<sha>-1.mp4`, `<sha>-2.mp4`, ...), and the `## Demo` section shows their previews in a table, each headed by its test's title: up to four portrait (phone) videos to a row, or two landscape ones. Helpers and `test.use` at the top of the block are shared by all of them.

````ts
test('Header and safe areas', async ({ demo, page }) => {
	// ...
});

test('Sidebar gestures', async ({ demo, page }) => {
	// ...
});
``` GitHub has no API for attachments, so for an inline player drag `demo/videos/pr-<number>.mp4` into the description instead.

## Recording docs

```bash
npm run demo:docs
````

Records every `scripts/docs/*.demo.ts` from the storage target and uploads `videos/docs/<name>.{mp4,gif}`.

## Writing a demo

Scripts get `test` and `expect` from `helpers.ts` without importing them. `demo.click`, `demo.hover` and `demo.type` glide a visible cursor to the target before acting, with timing varied like a person's (seeded per demo, so re-takes match); on a touch device (`hasTouch`) there is no cursor, and a click shows as a tap of a fingertip; `demo.scroll` turns the mouse wheel over a target; `demo.zoom` moves the camera in on a control or a point to show a detail too small to read in the frame, holds, and moves back out; `demo.zoomIn` and `demo.zoomOut` do the two halves, to keep the camera on a detail while the demo goes on; `demo.caption` shows a caption under the window. For voice input, `demo.microphone()` gives the page a microphone that hears nothing but `demo.speak(text)`, which reads the text in a synthetic voice (macOS `say`) and shows it as a caption, since the video has no sound. Use `test.use({ settings, startPath })` to seed other settings or start on another route.

## The render

A video is not the raw screencast. The app sits in a rounded window on a ground of the Llama brand (the day sky, or the night sky for a dark recording), a phone's viewport gets a phone's rounder corners, and captions are chips under the window, in Llama Sans. `demo.zoom` moves a camera over the whole stage, with eased moves and motion blur: the window itself comes closer, with its corners and the ground round it, and the app is not cropped inside a fixed window. The grounds, the fonts and the lockup come from `@ggml-org/llama-brand`, which the app already depends on, and sizes follow the brand's grid (u = the short side / 24, a margin of 2u).

To keep the camera on a detail across several actions:

```ts
await demo.zoomIn(page.getByRole('dialog'), 1.5);
await demo.type(search, 'Inkling-2-Mini');
await demo.zoomOut();
```

| Variable             | Values                                    | Default                                |
| -------------------- | ----------------------------------------- | -------------------------------------- |
| `DEMO_POLISH`        | `0` for the plain screencast              | on                                     |
| `DEMO_POLISH_GROUND` | `day`, `night`, `light`, `dark`, `sand`   | `day`, or `night` for a dark recording |
| `DEMO_POLISH_LOGO`   | `start`, `end`, `both`: a Llama logo card | none                                   |

With `DEMO_POLISH=0` the video is the screencast as recorded: `demo.zoom` pinches the page (with the fingertips drawn on a touch device), captions show over the app, and `demo.zoomIn` and `demo.zoomOut` do nothing, as the cursor cannot act on a pinched page.

The render needs no extra tool: a page in the demo's Chromium lays out each frame from the recorded frames and the script's timeline, and `ffmpeg` encodes them. It takes a few times as long as the video, and longer with many camera moves (about 2 minutes for 18 s of video with six moves).

## A mocked llama-server

`test.use({ mockServer: { models: ['org/Model-GGUF:Q4_K_M'], reply: 'A **Markdown** reply' } })` answers llama-server's endpoints in the browser (`fixtures/llama-server.ts`): the model list, props, model loading, tools and a streamed reply. Nothing reaches the server and no model loads, so a demo that sends a message looks the same on every machine. `tests/e2e` checks call `mockLlamaServer(page)` directly, since the e2e build runs without a server.

## Storybook stories

`test.use({ story: 'ui-forms-switch--default' })` records a Storybook story (the id from its `?path=/story/<id>` URL) instead of Llama Chat, from the Storybook that `DEMO_STORYBOOK` adds to the servers: `ui` for Llama UI (`ui/lib/ui`), `chat` for Llama Chat's. It runs on `DEMO_STORYBOOK_PORT` (default `:6007`). A small component reads better at a smaller `viewport` than zoomed in.

```bash
DEMO_STORYBOOK=ui npm run demo:record -- demo/<name>
```

## Conversations

Demos that need existing chats import them with `test.use({ traces: ['long-chat'] })`, through Settings > Import/Export before the video starts. Traces use the export format (llama.app JSONL), so the Hub shows them as Llama traces:

- `fixtures/traces.ts` generates the built-in ones: `long-chat` (36 tall exchanges) and `agentic-debugging` (long reasoning, shell output and code blocks).
- Any other name is downloaded as `<name>.jsonl` from `DEMO_TRACES_DATASET`, e.g. an exported conversation uploaded there.

`npm run demo:traces` writes the built-in traces to `demo/traces/` and pushes them to `DEMO_TRACES_DATASET`.

## Trying a script locally

To try a script locally, put it in `demo/` as `<name>.demo.ts`, importing from `./helpers.ts`, and run `npm run demo:record -- demo/<name>`. `npm run demo:codegen` opens Playwright codegen against `:5180` for a first draft.

The config starts, or reuses when already running, the fixture MCP servers on `:8765`, `llama-server` on `:9931` (router mode, `--tools all`), the UI's vite dev server and, with `DEMO_STORYBOOK`, a Storybook.

For agents, [skills/ui-demo](../../../../skills/ui-demo/SKILL.md) walks through a demo, an e2e check and docs media for one change.
