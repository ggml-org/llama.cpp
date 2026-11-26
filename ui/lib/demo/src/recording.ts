/**
 * The recording around a demo: the page is prepared before it loads, recorded while the demo drives
 * it, and rendered into an mp4 that the app's manifest lists for the CLI.
 */

import { installCursorOverlay, timelines } from "./demo.ts";
import {
  POLISH,
  POLISH_RENDER_TIMEOUT_MS,
  type PolishEvent,
  renderPolished,
} from "./polish.ts";
import { ScreencastRecorder } from "./recorder.ts";
import type { Page, TestInfo } from "@playwright/test";
import { appendFileSync, mkdirSync } from "node:fs";
import { basename, resolve } from "node:path";

/** How long the page settles (connections, fonts) before the recording starts. */
const SETTLE_MS = 1500;

export interface RecordingOptions {
  /** @ggml-org/llama-brand's folder: the render takes its CSS, fonts and lockup from there. */
  brandDir: string;
  /** Whether the page is in its dark scheme, which gets the night sky. */
  dark: boolean;
  /** Where the videos go, with the manifest of each script that the CLI reads. */
  videosDir: string;
}

/** Draws the cursor in `page` and starts the timeline the render plays. Call it before the page loads. */
export async function prepareDemoPage(
  page: Page,
  testInfo: TestInfo,
): Promise<void> {
  if (POLISH) {
    timelines.set(page, []);
    testInfo.setTimeout(testInfo.timeout + POLISH_RENDER_TIMEOUT_MS);
  }

  await page.addInitScript(installCursorOverlay);
}

/** Records `page` while `run` drives it, then renders the video and lists it in the script's manifest. */
export async function recordDemo(
  page: Page,
  testInfo: TestInfo,
  options: RecordingOptions,
  run: () => Promise<void>,
): Promise<void> {
  const timeline: PolishEvent[] | undefined = timelines.get(page);

  const viewport = page.viewportSize();

  if (viewport) await page.mouse.move(viewport.width / 2, viewport.height / 2);

  await page.waitForTimeout(SETTLE_MS);

  const browserType = page.context().browser()?.browserType();
  const recorder = new ScreencastRecorder(testInfo.outputPath("frames"));

  await recorder.start(page);
  await run();
  await recorder.stop();
  await page.close();

  if (testInfo.status !== testInfo.expectedStatus) return;

  const file = basename(testInfo.file).replace(/\.demo\.ts$/, "");
  const name =
    testInfo.title === "demo" ? file : `${file}--${slugify(testInfo.title)}`;
  const mp4 = resolve(options.videosDir, `${name}.mp4`);

  mkdirSync(options.videosDir, { recursive: true });

  if (timeline && viewport && browserType) {
    await renderPolished(
      browserType,
      options.brandDir,
      recorder.footage,
      timeline,
      viewport,
      options.dark,
      mp4,
    );
    recorder.discard();
  } else {
    recorder.encode(mp4);
  }

  // the CLI reads which videos a script made, in order, with their titles
  appendFileSync(
    resolve(options.videosDir, `${file}.videos.jsonl`),
    `${JSON.stringify({
      description: testInfo.annotations.find((a) => a.type === "description")
        ?.description,
      mp4,
      title: testInfo.title,
    })}\n`,
  );
  console.log(`demo video: ${mp4}`);
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}
