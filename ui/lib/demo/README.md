# @ggml-org/llama-demo

Scripted demo videos of the llama.cpp frontend apps, for pull requests and docs: a Playwright script drives the app like a person, and the harness records it and renders an mp4 with an animated GIF preview. It lives in llama.cpp at `ui/lib/demo`. It is a private package, not published to npm: an app in this repo adds it as a local dependency and imports it by name.

- **PR demos** live in the PR description, as a `ts demo` code block.
- **Docs demos** live in the storage target, under `scripts/docs/`.

```text
src/
├── index.ts      the entry an app's demo fixtures import
├── demo.ts       Demo: the cursor or fingertip, human timing, captions, voice, camera zooms
├── recording.ts  prepareDemoPage and recordDemo, the recording round a demo
├── recorder.ts   screencast recording and the plain mp4
├── polish.ts     the render: window, ground, camera, captions
├── cli.ts        runDemoCli: the pr and docs commands
└── storage.ts    storage targets (hf://, file://)
```

Playwright is only a type here: the app passes in its page and test, so the harness loads no Playwright of its own, and runs with nothing installed in this folder. `npm ci` here is only for `npm run check`.

## In an app

Add it to the app's `devDependencies` as a local package, which npm links to this folder, so no copy is installed and nothing is published:

```json
"@ggml-org/llama-demo": "file:../../lib/demo"
```

Then import it by name: `@ggml-org/llama-demo` for the fixtures' parts (`Demo`, `prepareDemoPage`, `recordDemo`) and `@ggml-org/llama-demo/cli` for `runDemoCli`. Node runs its TypeScript sources directly, through the link.

An app has a `demo/` folder with three files ([Llama Chat's](../../apps/chat/demo/README.md) is the example):

- **`helpers.ts`**: a Playwright `test` with a `demo` fixture and a `page` fixture that sets the scene. The page fixture calls `prepareDemoPage(page, testInfo)` before the page loads, then `recordDemo(page, testInfo, options, () => use(page))` once the app is ready. `options` names `@ggml-org/llama-brand`'s folder, whether the page is dark, and where the videos go.
- **`playwright.config.ts`**: the servers the app needs. `demo:pr` passes `DEMO_UI_DIR` (the PR's checkout of the app), `DEMO_UI_PORT` and `DEMO_STORYBOOK_PORT` to it.
- **`cli.ts`**: `runDemoCli({ demoDir, dir, commands, linkNodeModules })`, with the app's own commands beside `pr` and `docs`.

## Setup

| Tool                                                 | Used for                                                                                     |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Node 22.18+                                          | runs the TypeScript directly (type stripping)                                                |
| `npx playwright install chromium`                    | the browser demos record in, with its sandbox on                                             |
| `ffmpeg`                                             | mp4 encoding and the GIF preview                                                             |
| `gh`, logged in                                      | reads the PR description and writes the `## Demo` section, so you need edit rights on the PR |
| `hf`, logged in with a write token (`hf auth login`) | uploads to the bucket                                                                        |

`gh` is only needed for `demo:pr`, and `hf` only with the bucket below. Without them, the video stays in the app's `demo/videos/`.

The page is recorded through Chromium's screencast at the device pixel ratio the app's config sets (2x in Llama Chat) and 60 fps, starting once the page has settled. The video is then rendered from those frames at 1920x1080, or 1080x1920 for a portrait viewport (see [The render](#the-render)).

### Hugging Face bucket (videos, previews, scripts)

The bucket must be public, since GitHub loads the preview from it:

```bash
hf buckets create <owner>/llama-ui-demos
```

Then set it in the app's `.env`:

```bash
DEMO_STORAGE='hf://buckets/<owner>/llama-ui-demos'
DEMO_STORAGE_PUBLIC_URL='https://huggingface.co/buckets/<owner>/llama-ui-demos/resolve'
```

Layout: `videos/pr/<number>/<sha>.{mp4,gif,demo.ts}`, `videos/docs/<name>.{mp4,gif}` and `scripts/docs/*.demo.ts` (the docs scripts). Without `DEMO_STORAGE_PUBLIC_URL`, videos are uploaded but not linked in the PR. `DEMO_STORAGE='file:///some/dir'` stores them in a local folder instead.

Only Hugging Face buckets and `file://` folders are supported for now. S3, R2 or similar would be one more class in `storage.ts` (`upload` and `downloadDir`) plus the matching `DEMO_STORAGE_PUBLIC_URL`.

## Recording a pull request

Add the script to the PR description:

````md
<details>
<summary>Demo script</summary>

```ts demo
test("demo", async ({ demo, page }) => {
  await demo.click(page.getByRole("button", { exact: true, name: "Settings" }));
  await demo.caption("What this shows");
});
```

</details>
````

Then run `npm run demo:pr -- <number>` in the app. It checks out the PR head in a temporary worktree, records the script against it, uploads `videos/pr/<number>/<sha>.{mp4,gif,demo.ts}` and adds or updates a `## Demo` section in the PR description with the preview linking to the video. `DEMO_GITHUB_REPO` picks another repository, such as a fork.

A PR by someone else runs its code and demo script on your machine, with your `gh` and `hf` logins, so `demo:pr` refuses it until you've read its diff and demo block and pass the head commit you read: `npm run demo:pr -- <number> --trust <sha>`. If the PR gets new commits, it refuses again.

A large PR can show its changes in several shorter videos: each `test()` in the block is recorded as its own video (`<sha>-1.mp4`, `<sha>-2.mp4`, ...), and the `## Demo` section shows their previews in a table, each headed by its test's title: up to four portrait (phone) videos to a row, or two landscape ones. Helpers and `test.use` at the top of the block are shared by all of them.

To say what each video shows, give its test a `description` annotation. When the tests have one, the table puts one video to a row, with its title and description beside it:

```ts
test(
  "Taps instead of a cursor on a phone",
  {
    annotation: {
      description: "A click shows as a fingertip tap.",
      type: "description",
    },
  },
  async ({ demo, page }) => {
    // ...
  },
);
```

GitHub has no API for attachments, so for an inline player drag the mp4 from `demo/videos/` into the description instead.

## Recording docs

`npm run demo:docs` in the app records every `scripts/docs/*.demo.ts` from the storage target and uploads `videos/docs/<name>.{mp4,gif}`.

## Writing a demo

Scripts get `test` and `expect` from the app's `helpers.ts`. `demo.click`, `demo.hover` and `demo.type` glide a visible cursor to the target before acting, with timing varied like a person's (seeded per demo, so re-takes match); on a touch device (`hasTouch`) there is no cursor, and a click shows as a tap of a fingertip; `demo.scroll` turns the mouse wheel over a target; `demo.zoom` moves the camera in on a control or a point to show a detail too small to read in the frame, holds, and moves back out; `demo.zoomIn` and `demo.zoomOut` do the two halves, to keep the camera on a detail while the demo goes on (a phone demo rarely needs either, as a phone's UI already fills a portrait video); `demo.caption` shows a caption under the window. For voice input, `demo.microphone()` gives the page a microphone that hears nothing but `demo.speak(text)`, which reads the text in a synthetic voice (macOS `say`) and shows it as a caption, since the video has no sound.

```ts
await demo.zoomIn(page.getByRole("dialog"), 1.5);
await demo.type(search, "Inkling-2-Mini");
await demo.zoomOut();
```

## The render

A video is not the raw screencast. The app sits in a rounded window on a ground of the Llama brand (the day sky, or the night sky for a dark recording), a phone's viewport gets a phone's rounder corners, and captions are chips under the window, in Llama Sans. `demo.zoom` moves a camera over the whole stage, with eased moves and motion blur: the window itself comes closer, with its corners and the ground round it, and the app is not cropped inside a fixed window. The grounds, the fonts and the lockup come from `@ggml-org/llama-brand`, which the app depends on, and sizes follow the brand's grid (u = the short side / 24, a margin of 2u).

| Variable             | Values                                    | Default                                |
| -------------------- | ----------------------------------------- | -------------------------------------- |
| `DEMO_POLISH`        | `0` for the plain screencast              | on                                     |
| `DEMO_POLISH_GROUND` | `day`, `night`, `light`, `dark`, `sand`   | `day`, or `night` for a dark recording |
| `DEMO_POLISH_LOGO`   | `start`, `end`, `both`: a Llama logo card | none                                   |

With `DEMO_POLISH=0` the video is the screencast as recorded: `demo.zoom` pinches the page (with the fingertips drawn on a touch device), captions show over the app, and `demo.zoomIn` and `demo.zoomOut` do nothing, as the cursor cannot act on a pinched page.

The render needs no extra tool: a page in a Chromium of its own lays out each frame on one canvas from the recorded frames and the script's timeline, and `ffmpeg` encodes them. It takes a few times as long as the video, and longer with many camera moves.

For agents, [skills/ui-demo](../../../skills/ui-demo/SKILL.md) walks through a demo, an e2e check and docs media for one change.
