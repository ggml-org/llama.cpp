/**
 * Polish mode, on unless DEMO_POLISH=0: renders a demo as a branded video instead of the plain screencast.
 * The app sits in a rounded window on a ground of the Llama brand, `demo.zoom` moves a camera with
 * eased, motion-blurred moves, and captions sit under the window. A Chromium page styled by
 * @ggml-org/llama-brand lays out each frame, and ffmpeg encodes its screenshots.
 *
 * Env:
 *   DEMO_POLISH_GROUND  day | night | light | dark | sand (default: day, or night for a dark recording)
 *   DEMO_POLISH_LOGO    start | end | both: a card with the Llama lockup (default: none)
 */

import { type Footage, FPS, H264_ARGS } from './recorder.ts';
import type { Browser } from '@playwright/test';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { join, resolve } from 'node:path';

export const POLISH = process.env.DEMO_POLISH !== '0';
export const CAMERA_MOVE_MS = 1050;
/** Added to a test's timeout, as the render runs in the test's teardown. */
export const POLISH_RENDER_TIMEOUT_MS = 600_000;

/** The grounds of the brand, and whether each takes the light mark of the lockup. */
const GROUNDS = { dark: true, day: false, light: false, night: true, sand: false };
const LOGO_CARDS = ['start', 'end', 'both'];
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
const JPEG_QUALITY = 95;
/** Headless Chromium captures at the device scale it was launched with, whatever the page is set to. */
const CAPTURE_SCALE = Number(process.env.DEMO_SCALE ?? 2);
/** The stage page, the brand's files and the recorded frames are served from here. */
const STAGE_ORIGIN = 'http://demo.invalid';
const BRAND_DIR = resolve(import.meta.dirname, '../node_modules/@ggml-org/llama-brand');

/** What a demo did, with the time (seconds, on the clock of Date.now()) it did it at. */
export type PolishEvent =
	| { scale: number; time: number; type: 'camera'; x: number; y: number }
	| { text: null | string; time: number; type: 'caption' };

interface Size {
	height: number;
	width: number;
}

/** The part of a recorded frame that fills the window, in fractions of the frame. */
interface Crop extends Size {
	x: number;
	y: number;
}

interface Shot {
	caption: null | { opacity: number; text: string };
	crops: Crop[];
	frame: string;
	/** How much of the logo card shows over the scene, 0 to 1. */
	logo: number;
}

const smootherstep = (t: number) => {
	const x = Math.min(Math.max(t, 0), 1);

	return x * x * x * (x * (x * 6 - 15) + 10);
};

/** The camera's crop at any time: each move eases from where the camera was to its target. */
function cameraPath(events: PolishEvent[], viewport: Size): (time: number) => Crop {
	interface Camera {
		/** Natural log of the scale, so a zoom looks as fast at 2x as at 1x. */
		zoom: number;
		x: number;
		y: number;
	}

	const rest: Camera = { x: viewport.width / 2, y: viewport.height / 2, zoom: 0 };
	const moves: { from: Camera; start: number; to: Camera }[] = [];
	const at = (time: number) => {
		let camera = rest;

		for (const { from, start, to } of moves) {
			if (time < start) break;

			const k = smootherstep(((time - start) * 1000) / CAMERA_MOVE_MS);

			camera = {
				x: from.x + (to.x - from.x) * k,
				y: from.y + (to.y - from.y) * k,
				zoom: from.zoom + (to.zoom - from.zoom) * k
			};
		}

		return camera;
	};

	for (const event of events) {
		if (event.type !== 'camera') continue;

		// the camera stays inside the page
		const halfWidth = viewport.width / 2 / event.scale;
		const halfHeight = viewport.height / 2 / event.scale;

		moves.push({
			from: at(event.time),
			start: event.time,
			to: {
				x: Math.min(Math.max(event.x, halfWidth), viewport.width - halfWidth),
				y: Math.min(Math.max(event.y, halfHeight), viewport.height - halfHeight),
				zoom: Math.log(event.scale)
			}
		});
	}

	return (time) => {
		const { x, y, zoom } = at(time);
		const size = 1 / Math.exp(zoom);

		return {
			height: size,
			width: size,
			x: x / viewport.width - size / 2,
			y: y / viewport.height - size / 2
		};
	};
}

/** The caption at any time, fading in and out. */
function captionTrack(events: PolishEvent[], end: number): (time: number) => Shot['caption'] {
	const spans: { end: number; start: number; text: string }[] = [];

	for (const event of events) {
		if (event.type !== 'caption') continue;

		const last = spans.at(-1);

		if (last && last.end > event.time) last.end = event.time;

		if (event.text !== null) spans.push({ end, start: event.time, text: event.text });
	}

	return (time) => {
		const span = spans.find((s) => s.start < time && time < s.end);

		if (!span) return null;

		const opacity = smootherstep(Math.min(time - span.start, span.end - time) / CAPTION_FADE_S);

		return { opacity, text: span.text };
	};
}

/** The stage: a ground of the brand with the window, the caption chip and the logo card on it. */
function stageHtml(viewport: Size, dark: boolean, captions: boolean): string {
	const size = viewport.height > viewport.width ? PORTRAIT : LANDSCAPE;
	const u = Math.min(size.height, size.width) / UNITS_PER_SHORT_SIDE;
	const ground = process.env.DEMO_POLISH_GROUND ?? (dark ? 'night' : 'day');

	if (!(ground in GROUNDS)) throw new Error(`demo: no ground "${ground}" in the brand`);

	// the caption row is kept free under the window only when the demo has captions
	const maxWidth = size.width - 2 * MARGIN_U * u;
	const maxHeight =
		size.height - 2 * MARGIN_U * u - (captions ? (CAPTION_HEIGHT_U + CAPTION_GAP_U) * u : 0);
	const fit = Math.min(maxWidth / viewport.width, maxHeight / viewport.height);
	const width = Math.round(viewport.width * fit);
	const height = Math.round(viewport.height * fit);
	const left = Math.round((size.width - width) / 2);
	const top = Math.round(MARGIN_U * u + (maxHeight - height) / 2);
	const phone = viewport.height > viewport.width;
	const corners = phone
		? `border-radius: calc(var(--u) * ${PHONE_RADIUS_U}); corner-shape: squircle;`
		: `border-radius: calc(var(--u) * ${WINDOW_RADIUS_U});`;
	const mark = GROUNDS[ground as keyof typeof GROUNDS] ? 'dark' : 'light';

	return `<!doctype html>
		<link rel="stylesheet" href="/brand/css/colors.css" />
		<link rel="stylesheet" href="/brand/css/fonts.css" />
		<link rel="stylesheet" href="/brand/css/communications.css" />
		<style>
			body { position: relative; overflow: hidden; margin: 0; width: ${size.width}px; height: ${size.height}px; --u: ${u}px; }
			#scene, #logo { position: absolute; inset: 0; }
			#view { position: absolute; left: ${left}px; top: ${top}px; width: ${width}px; height: ${height}px; ${corners}
				box-shadow: 0 calc(var(--u) * 0.5) calc(var(--u) * 1.6) rgb(0 0 0 / 0.18); }
			#caption { position: absolute; left: 50%; bottom: calc(var(--u) * ${MARGIN_U}); display: flex; align-items: center;
				box-sizing: border-box; height: calc(var(--u) * ${CAPTION_HEIGHT_U}); padding: 0 calc(var(--u) * ${CAPTION_PAD_U});
				max-width: ${maxWidth}px; overflow: hidden; border-radius: 999px; background: var(--p-solid);
				color: var(--p-solid-ink); white-space: nowrap; opacity: 0; }
			#logo { opacity: 0; }
			#logo img { position: absolute; left: 50%; top: 50%; height: calc(var(--u) * ${LOGO_HEIGHT_U}); translate: -50% -50%; }
		</style>
		<body class="ground ground-${ground}">
			<div id="scene">
				<canvas id="view" width="${width}" height="${height}"></canvas>
				<div id="caption" data-font="${CAPTION_FONT_U * u}" data-min-font="${CAPTION_MIN_FONT_U * u}"
					data-rise="${CAPTION_RISE_U * u}"></div>
			</div>
			<div id="logo"><img src="/brand/logo/svg/lockup/llama/logo-llama-${mark}.svg" /></div>
		</body>`;
}

/** Lays out one output frame on the stage. Runs in the page. */
async function drawFrame(shot: Shot): Promise<void> {
	const stage = window as unknown as { demoFrame?: string; demoImage?: ImageBitmap };
	const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
	const caption = document.querySelector<HTMLElement>('#caption')!;

	if (stage.demoFrame !== shot.frame) {
		const response = await fetch(shot.frame);

		if (!response.ok) throw new Error(`demo: no frame at ${shot.frame} (${response.status})`);

		stage.demoImage?.close();
		stage.demoImage = await createImageBitmap(await response.blob());
		stage.demoFrame = shot.frame;
	}

	const image = stage.demoImage!;
	const view = canvas.getContext('2d')!;

	view.imageSmoothingQuality = 'high';
	// each crop drawn at 1/n over the ones before leaves their mean: the motion blur
	shot.crops.forEach((crop, i) => {
		view.globalAlpha = 1 / (i + 1);
		view.drawImage(
			image,
			crop.x * image.width,
			crop.y * image.height,
			crop.width * image.width,
			crop.height * image.height,
			0,
			0,
			canvas.width,
			canvas.height
		);
	});

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
	document.getElementById('scene')!.style.opacity = String(1 - shot.logo);
	document.getElementById('logo')!.style.opacity = String(shot.logo);
}

/** Renders the recorded frames and the demo's timeline into `mp4`. */
export async function renderPolished(
	browser: Browser,
	footage: Footage,
	timeline: PolishEvent[],
	viewport: Size,
	dark: boolean,
	mp4: string
): Promise<void> {
	const card = process.env.DEMO_POLISH_LOGO;

	if (card && !LOGO_CARDS.includes(card)) throw new Error(`demo: DEMO_POLISH_LOGO is "${card}"`);

	const html = stageHtml(
		viewport,
		dark,
		timeline.some((event) => event.type === 'caption')
	);
	const size = viewport.height > viewport.width ? PORTRAIT : LANDSCAPE;
	const page = await browser.newPage({ deviceScaleFactor: 1, viewport: size });

	await page.route(`${STAGE_ORIGIN}/**`, (route) => {
		const path = decodeURIComponent(new URL(route.request().url()).pathname);
		const file = join(BRAND_DIR, path.replace(/^\/brand\//, ''));

		if (path.startsWith('/frames/')) {
			return route.fulfill({ path: footage.frames[Number(path.split('/')[2])].file });
		}

		return path.startsWith('/brand/') && file.startsWith(BRAND_DIR)
			? route.fulfill({ path: file })
			: route.fulfill({ body: html, contentType: 'text/html' });
	});
	await page.goto(STAGE_ORIGIN);

	const cdp = await page.context().newCDPSession(page);

	await page.evaluate(async () => {
		await Promise.all([...document.fonts].map((font) => font.load()));
		await document.querySelector('img')!.decode();
	});
	const ffmpeg = spawn(
		'ffmpeg',
		[
			'-y',
			'-loglevel',
			'error',
			'-f',
			'image2pipe',
			'-c:v',
			'mjpeg',
			'-framerate',
			String(FPS),
			'-i',
			'-',
			'-vf',
			`scale=${size.width}:${size.height}:flags=lanczos`,
			...H264_ARGS,
			mp4
		],
		{ stdio: ['pipe', 'inherit', 'inherit'] }
	);
	const start = footage.frames[0].time;
	const seconds = footage.end - start;
	const intro = card === 'start' || card === 'both' ? LOGO_CARD_S : 0;
	const outro = card === 'end' || card === 'both' ? LOGO_CARD_S : 0;
	const cropAt = cameraPath(timeline, viewport);
	const captionAt = captionTrack(timeline, footage.end);
	const moved = (a: Crop, b: Crop) =>
		Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y), Math.abs(a.width - b.width)) > 1e-6;

	let source = 0;
	let drawn = '';
	let jpeg = Buffer.alloc(0);

	for (let i = 0; i < Math.round((intro + seconds + outro) * FPS); i++) {
		// the footage holds its first frame under the opening card and its last under the closing one
		const elapsed = i / FPS - intro;
		const time = start + Math.min(Math.max(elapsed, 0), seconds);

		while ((footage.frames[source + 1]?.time ?? Infinity) <= time) source++;

		const shot: Shot = {
			caption: captionAt(time),
			crops: moved(cropAt(time), cropAt(time + 1 / FPS))
				? Array.from({ length: BLUR_SAMPLES }, (_, s) => cropAt(time + s / BLUR_SAMPLES / FPS))
				: [cropAt(time)],
			frame: `${STAGE_ORIGIN}/frames/${source}`,
			logo:
				elapsed < 0
					? 1 - smootherstep(1 + elapsed / LOGO_FADE_S)
					: smootherstep((elapsed - seconds) / LOGO_FADE_S)
		};
		const key = JSON.stringify(shot);

		// a still frame is drawn once
		if (key !== drawn) {
			await page.evaluate(drawFrame, shot);

			// straight from Chromium: Playwright's own screenshot is several times slower
			const { data } = await cdp.send('Page.captureScreenshot', {
				clip: { ...size, scale: 1 / CAPTURE_SCALE, x: 0, y: 0 },
				format: 'jpeg',
				optimizeForSpeed: true,
				quality: JPEG_QUALITY
			});

			jpeg = Buffer.from(data, 'base64');
			drawn = key;
		}

		if (!ffmpeg.stdin.write(jpeg)) await once(ffmpeg.stdin, 'drain');
	}

	ffmpeg.stdin.end();

	const [code] = await once(ffmpeg, 'close');

	await page.context().close();

	if (code !== 0) throw new Error(`demo: ffmpeg exited with ${code} on the polished render`);
}
