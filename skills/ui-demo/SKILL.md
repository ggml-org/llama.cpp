---
name: ui-demo
description: Demo, test and document a change in Llama Chat (ui/apps/chat) or Llama UI (ui/lib/ui) with the Playwright demo harness - record a video for the PR, turn the same flow into an e2e check, and capture docs videos and screenshots. Use when a UI change needs a demo video, a Playwright check, or docs media.
---

# Demo, test and document a UI change

One scripted flow serves three ends: a **demo** video in the PR, a Playwright **check** in `tests/e2e`, and **docs** media. The harness is `ui/apps/chat/demo/` (read its `README.md` once). Everything below runs from `ui/apps/chat`.

Use it when a change in `ui/apps/chat` or `ui/lib/ui` is visible: a new or changed screen, flow or component. Skip it for refactors with no visible effect.

AGENTS.md still applies: the contributor writes the PR description and owns the script. `demo:pr` only fills the managed `## Demo` section.

## Pick the target

| Change in                         | Record                 | How                                           |
| --------------------------------- | ---------------------- | --------------------------------------------- |
| Llama Chat flow or screen         | the app, on vite dev   | default                                       |
| Llama UI component (`ui/lib/ui`)  | its Storybook story    | `test.use({ story })` + `DEMO_STORYBOOK=ui`   |
| Llama Chat component in isolation | Chat's Storybook story | `test.use({ story })` + `DEMO_STORYBOOK=chat` |

A Storybook story is the harness page for a component: if none shows the state you need, add a story next to the others (`tests/stories/` in that package) rather than a new page.

## Setup (once)

```bash
cd ui/apps/chat && npm ci
```

```bash
npx playwright install chromium
```

Also `ffmpeg`, and `npm ci` in `ui/lib/ui` for Llama UI stories. `demo:pr` needs `gh` (edit rights on the PR) and, to link the video, an `hf` login plus `DEMO_STORAGE` and `DEMO_STORAGE_PUBLIC_URL` in `ui/apps/chat/.env` (see `.env.example`).

## 1. Demo

### Write the script

Scripts get `test` and `expect` from `helpers.ts`. Drive the page with `demo.*` (visible cursor, human timing), assert with `expect`, set the scene with `test.use`:

- `demo.click`, `demo.hover`, `demo.type`, `demo.press`, `demo.scroll`, `demo.pause`, `demo.caption(text)` (a chip under the window), `demo.zoom(target, scale)` (the camera moves onto a small detail and back), `demo.zoomIn(target, scale)` ... `demo.zoomOut()` (the camera stays on it across the actions between). Zoom on desktop demos, where a detail is small in the frame; a phone's UI already fills a portrait video, so leave phone demos unzoomed unless a detail is too small to read even there.
- `test.use({ settings: {...} })` seeds UI settings, `startPath: '/#/...'` starts on a route.
- Fixtures, so no real data or third-party server is needed:
    - **MCP servers:** two fixture servers (Weather, Notes) are always configured; see `demo/fixtures/demo-config.ts`.
    - **Conversations:** `traces: ['long-chat', 'agentic-debugging']` imports built-in chats (`demo/fixtures/traces.ts`); other names come from `DEMO_TRACES_DATASET`.
    - **Models and replies:** `mockServer: { models: ['org/Model-GGUF:Q4_K_M'], reply: 'Markdown **reply**' }` answers llama-server's endpoints in the browser (`demo/fixtures/llama-server.ts`): the model list, props, load, tools and a streamed reply. Nothing reaches a server and no model loads. Use it for any demo that sends a message.
    - **Stories:** `story: 'ui-forms-switch--default'`, the id from the Storybook URL (`?path=/story/<id>`).

Without `mockServer` the app talks to the llama-server at `DEMO_SERVER_ORIGIN` (default `:9931`, started from `build/bin/llama-server` when none runs). Sending a message there loads a model: do not do that on someone's running server without asking.

The video is rendered from the recording, not the raw screencast: the app in a window on a Llama brand ground, camera moves over the whole stage with motion blur, captions under the window, and on a touch viewport taps in place of the cursor (see "The render" in the harness README). Write captions in the brand's voice: plain words, real numbers, no emoji. `DEMO_POLISH_LOGO=end` adds a Llama logo card, for a video that goes out on its own; `DEMO_POLISH=0` gives the plain screencast.

### Try it locally

Save it as `demo/<name>.demo.ts` (git-ignored), importing from `./helpers.ts`, and run:

```bash
npm run demo:record -- demo/<name>
```

Add `DEMO_STORYBOOK=ui` (or `chat`) when the script has a `story`. `DEMO_POLISH=0` records faster while iterating (the render takes a few times the video's length). The mp4 lands in `demo/videos/`; check a few frames before going further:

```bash
ffmpeg -loglevel error -ss 3 -i demo/videos/<name>.mp4 -frames:v 1 demo/videos/<name>-3s.png
```

`npm run demo:codegen` opens Playwright codegen on `:5180` for a first draft of the selectors.

### Put it in the PR

Paste the script into the PR description as a `ts demo` block, inside the last collapsible block of Additional information:

````md
<details>
<summary>Demo script</summary>

```ts demo
test("demo", async ({ demo, page }) => {
    // ...
});
```

</details>
````

Then record it at the PR head and write the `## Demo` section (between `<!-- demo:start -->` and `<!-- demo:end -->`):

```bash
npm run demo:pr -- <number>
```

- Each `test()` becomes its own video, shown in a table under its title. Group tests with their own `test.use` in `test.describe(() => { ... })`.
- In a PR block, the helpers import is replaced with `test` and `expect`, so reach fixtures through `test.use` options only.
- A PR on a fork: `DEMO_GITHUB_REPO=<owner>/llama.cpp`. A PR by someone else: read its diff and demo block, then pass `--trust <sha>`.

## 2. Test

Turn the same flow into a check in `tests/e2e/<feature>.e2e.ts`, which CI runs (`npm run test:e2e`) against the built app on `:8181`, with no llama-server:

- `@playwright/test` instead of the helpers: `locator.click()` / `fill()` for `demo.*`, keep every `expect`, drop captions and pauses.
- `mockLlamaServer(page, {...})` before `page.goto`, since nothing serves the API.
- `importTraces(page, file)` and `generateTrace(name)` seed conversations, as in a demo.

```bash
npx playwright test tests/e2e/<feature>.e2e.ts
```

Locally it uses the installed Chrome with its sandbox on; in CI, Playwright's Chromium.

For a component on its own, a `play` function on its story is the check instead (as in `ui/apps/chat/tests/stories/app/PwaRefreshAlert.stories.svelte`): both packages run their stories as vitest browser tests, `npm run test:ui` in `ui/apps/chat` and `npm test` in `ui/lib/ui`.

## 3. Document

- **Docs videos:** put the script in the storage target as `scripts/docs/<name>.demo.ts` (import `'../helpers.ts'`), then `npm run demo:docs` records every docs script and uploads `videos/docs/<name>.{mp4,gif}`. `DEMO_STORAGE=file:///some/dir` tries it without a bucket.
- **Screenshots:** in any demo, `await page.screenshot({ path: 'demo/videos/<name>.png' })` for the frame, or `await locator.screenshot({ path })` for one component, rendered at `DEMO_SCALE` (2x by default). Paths are relative to `ui/apps/chat`.
- **Component docs** belong in the package's Storybook: a story per state, prose in a `.svx` page (see `ui/apps/chat/AGENTS.md`).

## Worked example

A demo of chat search, then a reply from the mocked server, then a Llama UI story, in one PR block (record with `DEMO_STORYBOOK=ui npm run demo:pr -- <number>`):

```ts
test.describe(() => {
    test.use({ traces: ["long-chat", "agentic-debugging"] });

    test("Search finds a chat by its title", async ({ demo, page }) => {
        await demo.click(page.getByLabel("Expand navigation"));
        await demo.click(page.getByRole("button", { name: /^Search/ }));

        const search = page.getByRole("dialog", { name: "Search" });

        await demo.type(
            search.getByRole("combobox", { name: "Search" }),
            "flaky",
        );

        const result = search.getByRole("option", {
            name: /Fix the flaky cache tests/,
        });

        await expect(result).toBeVisible();
        await demo.click(result, 1200);
        await expect(page.getByRole("main")).toContainText(
            "All 48 cache tests pass now",
        );
    });
});

test.describe(() => {
    test.use({
        mockServer: {
            reply: "Rust is a systems language focused on **safety** and *speed*.",
        },
    });

    test("A reply from a mocked server, no model loaded", async ({
        demo,
        page,
    }) => {
        await demo.type(
            page.getByRole("textbox", { name: "Type a message..." }),
            "What is Rust?",
        );
        await demo.press("Enter", 1500);
        await expect(page.getByRole("main")).toContainText("focused on safety");
    });
});

test.describe(() => {
    // a small viewport frames a small component, recorded at the same scale
    test.use({
        story: "ui-forms-switch--default",
        viewport: { height: 450, width: 720 },
    });

    test("A Llama UI story: Switch", async ({ demo, page }) => {
        const off = page.getByRole("switch").first();

        await demo.click(off, 900);
        await expect(off).toBeChecked();
    });
});
```

The same search flow as a check is `tests/e2e/search-chats.e2e.ts`.

## Pitfalls

- **Ports:** vite dev `:5180` (`DEMO_UI_PORT`), fixture MCP `:8765`, llama-server `:9931`, Storybook `:6007` (`DEMO_STORYBOOK_PORT`), e2e `:8181`. Every server is reused when something already answers there, even another checkout's: stop stale servers, or pick other ports. `demo:pr` picks free ports itself.
- **Chrome sandbox:** demos record in Playwright's bundled Chromium with the sandbox on (headless installed Chrome ignores the viewport in screencasts); local e2e runs in the installed Chrome with the sandbox on. Do not turn the sandbox off to make a run pass: a PR demo runs that PR's code.
- **Waits:** wait on state, never on time. `expect(locator).toBeVisible()` / `toContainText()` before acting on something that appears late; `demo.pause` is pacing for the viewer only. The first load of a cold dev server reloads the page once; the harness waits for it before recording, so do not add sleeps for it.
- **Selectors:** use roles and accessible names. Some names match twice (`Expand navigation` is both the logo and the button: use `page.getByLabel('Expand navigation')`); scope to a dialog with `page.getByRole('dialog', { name })`. A strict-mode error names both matches.
- **Real server:** without `mockServer`, the model list and any reply come from the real llama-server and differ per machine. Mock them for anything a viewer should see the same way twice.
- **Preview size:** keep a demo short (under ~20 s per test): the GIF preview must stay under 5 MB, so a long one is shrunk until it is hard to read.
