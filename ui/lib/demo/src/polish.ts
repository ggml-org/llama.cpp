/**
 * Polish mode, on unless DEMO_POLISH=0: renders a demo as a branded video instead of the plain screencast.
 * The app sits in a rounded window on a ground of the Llama brand, `demo.zoom` moves a camera over
 * the whole stage with eased, motion-blurred moves, and captions sit under the window. A Chromium page styled by
 * @ggml-org/llama-brand lays out each frame, and ffmpeg encodes its screenshots.
 *
 * Env:
 *   DEMO_POLISH_GROUND  day | night | light | dark | sand (default: day, or night for a dark recording)
 *   DEMO_POLISH_LOGO    start | end | both: a card with the Llama lockup (default: none)
 */

import { type Footage, FPS, H264_ARGS } from "./recorder.ts";
import type { BrowserType } from "@playwright/test";
import { type ChildProcessByStdio, spawn } from "node:child_process";
import { rmSync } from "node:fs";
import { once } from "node:events";
import type { Writable } from "node:stream";
import { join, resolve, sep } from "node:path";

export const POLISH = process.env.DEMO_POLISH !== "0";
export const CAMERA_MOVE_MS = 650;
/** Added to a test's timeout, as the render runs in the test's teardown. */
export const POLISH_RENDER_TIMEOUT_MS = 600_000;

/** The grounds of the brand, and whether each takes the light mark of the lockup. */
const GROUNDS = {
  dark: true,
  day: false,
  light: false,
  night: true,
  sand: false,
};
const LOGO_CARDS = ["start", "end", "both"];
const LANDSCAPE = { height: 1080, width: 1920 };
const PORTRAIT = { height: 1920, width: 1080 };
/** The brand's grid: every size is a multiple of u, this share of the format's short side. */
const UNITS_PER_SHORT_SIDE = 24;
const MARGIN_U = 2;
const WINDOW_RADIUS_U = 0.55;
/** A phone's screen has rounder corners than a window, and they are squircles. */
const PHONE_RADIUS_U = 2.4;
const CAPTION_HEIGHT_U = 1.6;
const CAPTION_PAD_U = 0.8;
const CAPTION_FONT_U = 0.84;
/** The smallest type the brand allows, which a long caption shrinks to before it is cut. */
const CAPTION_MIN_FONT_U = 0.58;
const CAPTION_GAP_U = 0.5;
const CAPTION_RISE_U = 0.3;
const CAPTION_FADE_S = 0.35;
const LOGO_HEIGHT_U = 3;
const LOGO_CARD_S = 1.6;
const LOGO_FADE_S = 0.5;
/** Camera positions averaged into one frame while the camera moves. */
const BLUR_SAMPLES = 12;
/** The share of a frame's time those positions span: a film camera's 180 degree shutter. */
const SHUTTER = 0.5;
const SHADOW_Y_U = 0.5;
const SHADOW_BLUR_U = 1.6;
/** The shadow stops growing with the zoom here, so a close-up keeps a soft edge, not a cloud. */
const SHADOW_MAX_BLUR_PX = 120;
const SHADOW_COLOR = "rgb(0 0 0 / 0.18)";
/** Points along each squircle corner. */
const SQUIRCLE_STEPS = 24;
const JPEG_QUALITY = 95;
/** The stage page, the brand's files and the recorded frames are served from here. */
const STAGE_ORIGIN = "http://demo.invalid";

/** What a demo did, with the time (seconds, on the clock of Date.now()) it did it at. */
export type PolishEvent =
  | { scale: number; time: number; type: "camera"; x: number; y: number }
  | { text: null | string; time: number; type: "caption" };

interface Size {
  height: number;
  width: number;
}

/** A camera move: soft at both ends and fast through the middle, so it is quick without a jolt. */
const CAMERA_EASE_POWER = 5;
const cameraEase = (t: number) => {
  const x = Math.min(Math.max(t, 0), 1);

  return x < 0.5
    ? 2 ** (CAMERA_EASE_POWER - 1) * x ** CAMERA_EASE_POWER
    : 1 - (2 - 2 * x) ** CAMERA_EASE_POWER / 2;
};

/** Where the stage is drawn in a frame: scaled about its top left corner, then moved. */
interface Camera {
  scale: number;
  x: number;
  y: number;
}

/** Where the window sits on the stage, and how much larger it is than the page. */
interface Layout {
  fit: number;
  height: number;
  left: number;
  maxWidth: number;
  size: Size;
  top: number;
  u: number;
  width: number;
}

interface Shot {
  caption: null | { opacity: number; text: string };
  cameras: Camera[];
  frame: string;
  /** How much of the logo card shows over the scene, 0 to 1. */
  logo: number;
}

const smootherstep = (t: number) => {
  const x = Math.min(Math.max(t, 0), 1);

  return x * x * x * (x * (x * 6 - 15) + 10);
};

/**
 * The camera at any time: each move eases from where the camera was to its target. It looks at the
 * whole stage, so a zoom brings the window itself closer, with its corners and the ground round it.
 */
function cameraPath(
  events: PolishEvent[],
  layout: Layout,
): (time: number) => Camera {
  /** The point of the stage at the middle of the frame, and the natural log of the scale. */
  interface View {
    zoom: number;
    x: number;
    y: number;
  }

  const { height, width } = layout.size;
  const rest: View = { x: width / 2, y: height / 2, zoom: 0 };
  const moves: { from: View; start: number; to: View }[] = [];
  const at = (time: number) => {
    let view = rest;

    for (const { from, start, to } of moves) {
      if (time < start) break;

      const k = cameraEase(((time - start) * 1000) / CAMERA_MOVE_MS);

      view = {
        x: from.x + (to.x - from.x) * k,
        y: from.y + (to.y - from.y) * k,
        zoom: from.zoom + (to.zoom - from.zoom) * k,
      };
    }

    return view;
  };

  for (const event of events) {
    if (event.type !== "camera") continue;

    // the frame stays inside the stage
    const halfWidth = width / 2 / event.scale;
    const halfHeight = height / 2 / event.scale;
    const x = layout.left + event.x * layout.fit;
    const y = layout.top + event.y * layout.fit;

    moves.push({
      from: at(event.time),
      start: event.time,
      to: {
        x: Math.min(Math.max(x, halfWidth), width - halfWidth),
        y: Math.min(Math.max(y, halfHeight), height - halfHeight),
        zoom: Math.log(event.scale),
      },
    });
  }

  return (time) => {
    const { x, y, zoom } = at(time);
    const scale = Math.exp(zoom);

    return { scale, x: width / 2 - x * scale, y: height / 2 - y * scale };
  };
}

/** The caption at any time, fading in and out. */
function captionTrack(
  events: PolishEvent[],
  end: number,
): (time: number) => Shot["caption"] {
  const spans: { end: number; start: number; text: string }[] = [];

  for (const event of events) {
    if (event.type !== "caption") continue;

    const last = spans.at(-1);

    if (last && last.end > event.time) last.end = event.time;

    if (event.text !== null)
      spans.push({ end, start: event.time, text: event.text });
  }

  return (time) => {
    const span = spans.find((s) => s.start < time && time < s.end);

    if (!span) return null;

    const opacity = smootherstep(
      Math.min(time - span.start, span.end - time) / CAPTION_FADE_S,
    );

    return { opacity, text: span.text };
  };
}

/** The window is the page's viewport, as large as fits inside the margin. */
function stageLayout(viewport: Size, captions: boolean): Layout {
  const size = viewport.height > viewport.width ? PORTRAIT : LANDSCAPE;
  const u = Math.min(size.height, size.width) / UNITS_PER_SHORT_SIDE;
  // the caption row is kept free under the window only when the demo has captions
  const maxWidth = size.width - 2 * MARGIN_U * u;
  const maxHeight =
    size.height -
    2 * MARGIN_U * u -
    (captions ? (CAPTION_HEIGHT_U + CAPTION_GAP_U) * u : 0);
  const fit = Math.min(maxWidth / viewport.width, maxHeight / viewport.height);
  const width = Math.round(viewport.width * fit);
  const height = Math.round(viewport.height * fit);

  return {
    fit,
    height,
    left: Math.round((size.width - width) / 2),
    maxWidth,
    size,
    top: Math.round(MARGIN_U * u + (maxHeight - height) / 2),
    u,
    width,
  };
}

/** The stage: a ground of the brand with the window, the caption chip and the logo card on it. */
function stageHtml(layout: Layout, dark: boolean): string {
  const { height, left, maxWidth, size, top, u, width } = layout;
  const ground = process.env.DEMO_POLISH_GROUND ?? (dark ? "night" : "day");

  if (!(ground in GROUNDS))
    throw new Error(`demo: no ground "${ground}" in the brand`);

  const phone = size.height > size.width;
  const radius = (phone ? PHONE_RADIUS_U : WINDOW_RADIUS_U) * u;
  const mark = GROUNDS[ground as keyof typeof GROUNDS] ? "dark" : "light";

  return `<!doctype html>
		<link rel="stylesheet" href="/brand/css/colors.css" />
		<link rel="stylesheet" href="/brand/css/fonts.css" />
		<link rel="stylesheet" href="/brand/css/communications.css" />
		<style>
			body { position: relative; overflow: hidden; margin: 0; width: ${size.width}px; height: ${size.height}px; --u: ${u}px; }
			#scene, #logo { position: absolute; inset: 0; }
			#window { position: absolute; inset: 0; }
			#caption { position: absolute; left: 50%; bottom: calc(var(--u) * ${MARGIN_U}); display: flex; align-items: center;
				box-sizing: border-box; height: calc(var(--u) * ${CAPTION_HEIGHT_U}); padding: 0 calc(var(--u) * ${CAPTION_PAD_U});
				max-width: ${maxWidth}px; overflow: hidden; border-radius: 999px; background: var(--p-solid);
				color: var(--p-solid-ink); white-space: nowrap; opacity: 0; }
			#logo { opacity: 0; }
			#logo img { position: absolute; left: 50%; top: 50%; height: calc(var(--u) * ${LOGO_HEIGHT_U}); translate: -50% -50%; }
		</style>
		<body class="ground ground-${ground}">
			<div id="scene">
				<canvas id="window" width="${size.width}" height="${size.height}" data-left="${left}" data-top="${top}"
					data-width="${width}" data-height="${height}" data-radius="${radius}" data-squircle="${phone}"
					data-shadow-y="${SHADOW_Y_U * u}" data-shadow-blur="${SHADOW_BLUR_U * u}"
					data-shadow-max-blur="${SHADOW_MAX_BLUR_PX}" data-shadow-color="${SHADOW_COLOR}"
					data-squircle-steps="${SQUIRCLE_STEPS}"></canvas>
				<div id="caption" data-font="${CAPTION_FONT_U * u}" data-min-font="${CAPTION_MIN_FONT_U * u}"
					data-rise="${CAPTION_RISE_U * u}"></div>
			</div>
			<div id="logo"><img src="/brand/logo/svg/lockup/llama/logo-llama-${mark}.svg" /></div>
		</body>`;
}

/** Lays out one output frame on the stage. Runs in the page. */
async function drawFrame(shot: Shot): Promise<void> {
  const stage = window as unknown as {
    demoFrame?: string;
    demoImage?: ImageBitmap;
  };
  const caption = document.querySelector<HTMLElement>("#caption")!;

  if (stage.demoFrame !== shot.frame) {
    const response = await fetch(shot.frame);

    if (!response.ok)
      throw new Error(`demo: no frame at ${shot.frame} (${response.status})`);

    stage.demoImage?.close();
    stage.demoImage = await createImageBitmap(await response.blob());
    stage.demoFrame = shot.frame;
  }

  const image = stage.demoImage!;
  const canvas = document.querySelector<HTMLCanvasElement>("#window")!;
  const box = { ...canvas.dataset } as Record<string, string>;
  const [
    left,
    top,
    width,
    height,
    radius,
    shadowY,
    shadowBlur,
    shadowMaxBlur,
    steps,
  ] = [
    "left",
    "top",
    "width",
    "height",
    "radius",
    "shadowY",
    "shadowBlur",
    "shadowMaxBlur",
    "squircleSteps",
  ].map((key) => Number(box[key]));
  const squircle = box.squircle === "true";
  // a phone's corners are squircles, |x|^4 + |y|^4 = r^4, as CSS draws corner-shape: squircle
  const outline = (x: number, y: number, w: number, h: number, r: number) => {
    const path = new Path2D();

    if (!squircle) {
      path.roundRect(x, y, w, h, r);

      return path;
    }

    const corners = [
      [x + w - r, y + r, -Math.PI / 2],
      [x + w - r, y + h - r, 0],
      [x + r, y + h - r, Math.PI / 2],
      [x + r, y + r, Math.PI],
    ];

    for (const [cx, cy, from] of corners) {
      for (let step = 0; step <= steps; step++) {
        const angle = from + (step / steps) * (Math.PI / 2);
        const [cos, sin] = [Math.cos(angle), Math.sin(angle)];

        path.lineTo(
          cx + r * Math.sign(cos) * Math.sqrt(Math.abs(cos)),
          cy + r * Math.sign(sin) * Math.sqrt(Math.abs(sin)),
        );
      }
    }

    path.closePath();

    return path;
  };
  const output = canvas.getContext("2d")!;
  const copy = new OffscreenCanvas(canvas.width, canvas.height).getContext(
    "2d",
  )!;

  output.clearRect(0, 0, canvas.width, canvas.height);

  // one copy of the window per camera position, each drawn at 1/n over the ones before, which
  // leaves their mean: the motion blur. A still frame draws one copy. All of it goes on one canvas:
  // a layer per copy overruns Chromium's tile memory when the copies are large, and drops tiles
  shot.cameras.forEach(({ scale, x: cameraX, y: cameraY }, i) => {
    const shadow = Math.min(scale, shadowMaxBlur / shadowBlur);
    const [x, y, w, h] = [
      cameraX + left * scale,
      cameraY + top * scale,
      width * scale,
      height * scale,
    ];
    const path = outline(x, y, w, h, radius * scale);

    copy.clearRect(0, 0, canvas.width, canvas.height);
    copy.save();
    Object.assign(copy, {
      fillStyle: "#fff",
      shadowBlur: shadowBlur * shadow,
      shadowColor: box.shadowColor,
      shadowOffsetY: shadowY * shadow,
    });
    copy.fill(path);
    copy.restore();
    copy.save();
    copy.clip(path);
    copy.imageSmoothingQuality = "high";
    copy.drawImage(image, x, y, w, h);
    copy.restore();
    output.globalAlpha = 1 / (i + 1);
    output.drawImage(copy.canvas, 0, 0);
  });
  output.globalAlpha = 1;

  if (shot.caption && caption.textContent !== shot.caption.text) {
    const font = Number(caption.dataset.font);

    caption.textContent = shot.caption.text;
    caption.style.fontSize = `${font}px`;

    // a long caption gets smaller type, down to the smallest the brand allows
    const fit = caption.clientWidth / caption.scrollWidth;

    if (fit < 1) {
      caption.style.fontSize = `${Math.max(font * fit, Number(caption.dataset.minFont))}px`;
    }
  }

  const opacity = shot.caption?.opacity ?? 0;

  caption.style.opacity = String(opacity);
  caption.style.translate = `-50% ${(1 - opacity) * Number(caption.dataset.rise)}px`;
  document.getElementById("scene")!.style.opacity = String(1 - shot.logo);
  document.getElementById("logo")!.style.opacity = String(shot.logo);
}

/** Renders the recorded frames and the demo's timeline into `mp4`. */
export async function renderPolished(
  browserType: BrowserType,
  brandDir: string,
  footage: Footage,
  timeline: PolishEvent[],
  viewport: Size,
  dark: boolean,
  mp4: string,
): Promise<void> {
  const card = process.env.DEMO_POLISH_LOGO;

  if (card && !LOGO_CARDS.includes(card))
    throw new Error(`demo: DEMO_POLISH_LOGO is "${card}"`);

  const layout = stageLayout(
    viewport,
    timeline.some((event) => event.type === "caption"),
  );
  const html = stageHtml(layout, dark);
  const { size } = layout;
  // a browser of its own: the demo's is launched at the recording's device scale, and draws
  // every frame of this page that many times larger
  const browser = await browserType.launch({
    args: ["--force-device-scale-factor=1"],
    chromiumSandbox: true,
  });
  let ffmpeg: ChildProcessByStdio<Writable, null, null> | undefined;

  try {
    const page = await browser.newPage({
      deviceScaleFactor: 1,
      viewport: size,
    });

    await page.route(`${STAGE_ORIGIN}/**`, (route) => {
      const path = decodeURIComponent(new URL(route.request().url()).pathname);
      const file = join(brandDir, path.replace(/^\/brand\//, ""));

      if (path.startsWith("/frames/")) {
        return route.fulfill({
          path: footage.frames[Number(path.split("/")[2])].file,
        });
      }

      return path.startsWith("/brand/") && file.startsWith(brandDir + sep)
        ? route.fulfill({ path: file })
        : route.fulfill({ body: html, contentType: "text/html" });
    });
    await page.goto(STAGE_ORIGIN);

    const cdp = await page.context().newCDPSession(page);

    await page.evaluate(async () => {
      await Promise.all([...document.fonts].map((font) => font.load()));
      await document.querySelector("img")!.decode();
    });
    ffmpeg = spawn(
      "ffmpeg",
      [
        "-y",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-c:v",
        "mjpeg",
        "-framerate",
        String(FPS),
        "-i",
        "-",
        ...H264_ARGS,
        mp4,
      ],
      { stdio: ["pipe", "inherit", "inherit"] },
    );
    const start = footage.frames[0].time;
    const seconds = footage.end - start;
    const intro = card === "start" || card === "both" ? LOGO_CARD_S : 0;
    const outro = card === "end" || card === "both" ? LOGO_CARD_S : 0;
    const cameraAt = cameraPath(timeline, layout);
    const captionAt = captionTrack(timeline, footage.end);
    const moved = (a: Camera, b: Camera) =>
      Math.max(
        Math.abs(a.x - b.x),
        Math.abs(a.y - b.y),
        Math.abs(a.scale - b.scale),
      ) > 1e-3;

    let source = 0;
    let drawn = "";
    let jpeg = Buffer.alloc(0);

    for (let i = 0; i < Math.round((intro + seconds + outro) * FPS); i++) {
      // the footage holds its first frame under the opening card and its last under the closing one
      const elapsed = i / FPS - intro;
      const time = start + Math.min(Math.max(elapsed, 0), seconds);

      while ((footage.frames[source + 1]?.time ?? Infinity) <= time) source++;

      const shot: Shot = {
        cameras: moved(cameraAt(time), cameraAt(time + 1 / FPS))
          ? Array.from({ length: BLUR_SAMPLES }, (_, s) =>
              cameraAt(time + (s / BLUR_SAMPLES) * (SHUTTER / FPS)),
            )
          : [cameraAt(time)],
        caption: captionAt(time),
        frame: `${STAGE_ORIGIN}/frames/${source}`,
        logo:
          elapsed < 0
            ? 1 - smootherstep(1 + elapsed / LOGO_FADE_S)
            : smootherstep((elapsed - seconds) / LOGO_FADE_S),
      };
      const key = JSON.stringify(shot);

      // a still frame is drawn once
      if (key !== drawn) {
        await page.evaluate(drawFrame, shot);

        // straight from Chromium: Playwright's own screenshot is several times slower
        const { data } = await cdp.send("Page.captureScreenshot", {
          format: "jpeg",
          optimizeForSpeed: true,
          quality: JPEG_QUALITY,
        });

        jpeg = Buffer.from(data, "base64");
        drawn = key;
      }

      if (!ffmpeg.stdin.write(jpeg)) await once(ffmpeg.stdin, "drain");
    }

    ffmpeg.stdin.end();

    const [code] = await once(ffmpeg, "close");

    if (code !== 0)
      throw new Error(
        `demo: ffmpeg exited with ${code} on the polished render`,
      );
  } catch (error) {
    // a part-rendered mp4 would play as a shorter, valid video
    ffmpeg?.kill("SIGKILL");
    rmSync(mp4, { force: true });

    throw error;
  } finally {
    await browser.close();
  }
}
