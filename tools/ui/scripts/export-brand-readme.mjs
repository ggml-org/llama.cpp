#!/usr/bin/env node

/**
 * Renders the artwork of ggml-org/llama-brand's README from this Storybook, so the README shows the design system's
 * own stories rather than a drawing of them: the fonts, the colours and the logo in use.
 *
 *   npm run export:brand-readme -- --out ../../../llama-brand
 *   npm run export:brand-readme -- --out ../../../llama-brand --url http://localhost:6006
 *   npm run export:brand-readme -- --out ../../../llama-brand --storybook storybook-static
 *
 * Each piece is an element (or a group of them) the stories mark with `data-readme="<mark>"`, captured at twice its
 * size on a transparent ground, in light and dark: <out>/readme/<file>-light.png and -dark.png. Pieces that keep
 * their own colours in either mode (the banners, the desktop mockups) are captured once, as <file>.png. The colour
 * table's swatches are written as SVGs from the palette's own swatches (`data-swatch="<token>"`), in sRGB. Without
 * --url it serves a static build, building it first if there is none.
 */

import { execSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const flag = (name, fallback) => {
	const i = args.indexOf(`--${name}`);

	return i >= 0 ? args[i + 1] : fallback;
};

if (!flag('out')) {
	console.error(
		'Usage: export-brand-readme --out <llama-brand checkout> [--url <storybook> | --storybook <dir>]'
	);
	process.exit(1);
}

const OUT = path.resolve(process.cwd(), flag('out'), 'readme');
// the stories and the pieces each marks, as the mark and the file under readme/ it is written to; `fixed` pieces
// look the same in light and dark
const piece = (mark, file = `logo/${mark}`, fixed = false) => ({ file, fixed, mark });
const STORIES = [
	{
		id: 'brand-typography-llama-sans--overview',
		pieces: [
			piece('font-hero', 'fonts/llama-sans'),
			piece('font-weights', 'fonts/llama-sans-weights')
		]
	},
	{
		id: 'brand-typography-llama-mono--overview',
		pieces: [
			piece('font-hero', 'fonts/llama-mono'),
			piece('font-in-code', 'fonts/llama-mono-in-code')
		]
	},
	{
		id: 'brand-colors-palette--palette',
		pieces: [piece('colours', 'colours/palette')],
		swatches: true,
		// the palette's own column, so its tiles sit four to a row
		width: 1240
	},
	{
		id: 'brand-logo-lockup--lockup',
		pieces: [
			piece('lockup-construction'),
			piece('lockup-typed'),
			piece('lockup-sizes'),
			piece('lockup-grounds'),
			piece('lockup-never'),
			piece('lockup-in-context'),
			piece('banner-llama-cpp', undefined, true),
			piece('banner-llama', undefined, true)
		]
	},
	{ id: 'brand-logo-usage--glyphs', pieces: [piece('icon-glyphs')] },
	{ id: 'brand-logo-usage--buttons', pieces: [piece('icon-buttons')] },
	{ id: 'brand-logo-usage--download-badge', pieces: [piece('icon-badges')] },
	{ id: 'brand-logo-usage--running-text', pieces: [piece('icon-inline')] },
	{
		id: 'brand-logo-usage--app-icons-and-avatars',
		pieces: [piece('icon-in-place', undefined, true)]
	},
	{ id: 'ui-foundations-brand--splash-screens', pieces: [piece('icon-splash')] }
];
// wide enough for both lockups side by side at 64 px
const VIEWPORT = { height: 1000, width: 1000 };
const MIME = {
	'.css': 'text/css',
	'.html': 'text/html',
	'.js': 'text/javascript',
	'.json': 'application/json',
	'.mjs': 'text/javascript',
	'.png': 'image/png',
	'.svg': 'image/svg+xml',
	'.webp': 'image/webp',
	'.woff2': 'font/woff2'
};

function serve(dir) {
	return new Promise((resolve) => {
		const server = http.createServer((req, res) => {
			const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);

			let file = path.join(dir, url);

			if (!file.startsWith(dir)) return res.writeHead(403).end();

			if (fs.existsSync(file) && fs.statSync(file).isDirectory())
				file = path.join(file, 'index.html');

			if (!fs.existsSync(file)) return res.writeHead(404).end();

			res.writeHead(200, {
				'content-type': MIME[path.extname(file)] ?? 'application/octet-stream'
			});
			fs.createReadStream(file).pipe(res);
		});

		server.listen(0, '127.0.0.1', () => resolve(server));
	});
}

// each flat swatch of the palette as the colour it renders, in sRGB hex: a canvas converts whatever the CSS declares
async function swatches(page) {
	const found = await page.evaluate(() => {
		const ctx = document.createElement('canvas').getContext('2d');

		return [...document.querySelectorAll('[data-swatch]')].map((el) => {
			ctx.fillStyle = getComputedStyle(el).backgroundColor;
			ctx.fillRect(0, 0, 1, 1);

			const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;

			return {
				hex: `#${[r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('')}`,
				token: el.dataset.swatch
			};
		});
	});

	for (const { hex, token } of found) {
		const out = path.join(OUT, `swatches/${token}.svg`);

		fs.mkdirSync(path.dirname(out), { recursive: true });
		fs.writeFileSync(
			out,
			`<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40"><rect x="0.5" y="0.5" width="39" height="39" rx="8" fill="${hex}" stroke="rgb(128 128 128 / 0.4)"/></svg>\n`
		);
		console.log(`${path.relative(process.cwd(), out)} ${hex}`);
	}
}

async function capture(base) {
	const { chromium } = await import('playwright');
	const browser = await chromium.launch();

	for (const tone of ['light', 'dark']) {
		const page = await browser.newPage({
			colorScheme: tone,
			deviceScaleFactor: 2,
			viewport: VIEWPORT
		});

		// the pieces on a transparent ground, so they sit on GitHub's own
		await page.addInitScript(() => {
			addEventListener('DOMContentLoaded', () => {
				const style = document.createElement('style');

				// the desktop mockups show light and dark side by side, so their captions, which follow the mode, go
				style.textContent =
					'html,body,.logo,.specimen{background:transparent!important}[data-readme="font-in-code"]{width:max-content!important;overflow:visible!important}[data-readme="icon-in-place"] figcaption{display:none!important}';
				document.head.append(style);
			});
		});

		for (const story of STORIES) {
			await page.setViewportSize({ ...VIEWPORT, width: story.width ?? VIEWPORT.width });
			await page.goto(`${base}/iframe.html?id=${story.id}&viewMode=story`);
			await page.waitForSelector('[data-readme]');
			await page.evaluate(() => document.fonts.ready);
			await page.waitForTimeout(400);

			for (const { file, fixed, mark: name } of story.pieces) {
				if (fixed && tone === 'dark') continue;

				// a piece marked on more than one element is captured as the box round all of them
				const clip = await page.evaluate((name) => {
					const boxes = [...document.querySelectorAll(`[data-readme="${name}"]`)].map((el) =>
						el.getBoundingClientRect()
					);

					if (!boxes.length) return null;

					const x = Math.min(...boxes.map((b) => b.left)) + scrollX;
					const y = Math.min(...boxes.map((b) => b.top)) + scrollY;

					return {
						height: Math.max(...boxes.map((b) => b.bottom)) + scrollY - y,
						width: Math.max(...boxes.map((b) => b.right)) + scrollX - x,
						x,
						y
					};
				}, name);

				if (!clip) throw new Error(`${story.id}: nothing marked data-readme="${name}"`);

				const out = path.join(OUT, fixed ? `${file}.png` : `${file}-${tone}.png`);

				fs.mkdirSync(path.dirname(out), { recursive: true });
				await page.screenshot({ clip, fullPage: true, omitBackground: true, path: out });
				console.log(path.relative(process.cwd(), out));
			}

			if (story.swatches && tone === 'light') await swatches(page);
		}

		await page.close();
	}

	await browser.close();
}

const url = flag('url');

if (url) {
	await capture(url.replace(/\/$/, ''));
} else {
	const storybook = path.resolve(ROOT, flag('storybook', 'storybook-static'));

	if (!fs.existsSync(path.join(storybook, 'iframe.html')))
		execSync(`npx storybook build -o ${storybook}`, { cwd: ROOT, stdio: 'inherit' });

	const server = await serve(storybook);

	await capture(`http://127.0.0.1:${server.address().port}`);
	server.close();
}
