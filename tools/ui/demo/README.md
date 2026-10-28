# Demo videos

Scripted Playwright walkthroughs of the UI, exported as mp4 for pull requests and documentation.

```
demo/
├── docs/       demos for documentation
├── pr/         demos for pull requests, named <pr-number>-<slug>.demo.ts
├── fixtures/   fixture MCP servers and seeded settings
├── helpers.ts  `test` with a `demo` fixture: cursor, captions, mp4 export
└── videos/     output (git-ignored)
```

## Recording

```bash
npm run demo:record                      # all demos
npm run demo:record -- demo/pr/31303     # one demo
DEMO_COLOR_SCHEME=dark npm run demo:record -- demo/docs
```

The config starts or reuses, if already running:

- fixture MCP servers on `:8765` (Weather, Notes), seeded into the UI's `mcpServers` setting
- `llama-server` on `:9931` in router mode with `--tools all` (from `build/bin`)
- the UI's vite dev server on `:5180`

Each demo is written to `videos/<file-name>.mp4` (H.264, page load trimmed). `ffmpeg` is required for the mp4 export; without it the raw `.webm` is kept.

## Recording a pull request

Point `DEMO_UI_DIR` at a checkout of the PR so the demo records its UI:

```bash
git fetch origin pull/31303/head:pr-31303
git worktree add ../pr-31303 pr-31303
ln -s "$PWD/node_modules" ../pr-31303/tools/ui/node_modules
DEMO_UI_DIR=../pr-31303/tools/ui npm run demo:record -- demo/pr/31303
```

Stop any vite server already on `:5180` first, or set `DEMO_UI_PORT`, since an existing server is reused.

## Writing a demo

```ts
import { test } from '../helpers.ts';

test('demo', async ({ demo, page }) => {
	await demo.click(page.getByRole('button', { exact: true, name: 'Settings' }));
	await demo.caption('What this shows');
});
```

`demo.click`, `demo.hover` and `demo.type` glide a visible cursor to the target before acting; `demo.caption` shows a caption at the bottom of the frame. Use `test.use({ settings, startPath })` to seed other settings or start on another route.

`npm run demo:codegen` opens Playwright codegen against `:5180` to record a first draft; then switch it to the `demo` helpers.
