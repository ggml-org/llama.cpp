#!/usr/bin/env node

/**
 * Renders the logo's usage for ggml-org/llama-brand's README from this Storybook, so the README shows the design
 * system's own stories rather than a drawing of them:
 *
 *   npm run export:brand-readme -- --out ../../../llama-brand
 *   npm run export:brand-readme -- --out ../../../llama-brand --url http://localhost:6006
 *   npm run export:brand-readme -- --out ../../../llama-brand --storybook storybook-static
 *
 * Each piece is an element (or a group of them) the stories mark with `data-readme="<name>"`, captured at twice its
 * size on a transparent ground, in light and dark: <out>/readme/logo/<name>-light.png and -dark.png. The banners keep
 * their own colours in either mode, so they are captured once, as <name>.png. Without --url it serves a static build,
 * building it first if there is none.
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

const OUT = path.resolve(process.cwd(), flag('out'), 'readme/logo');
// the stories and the pieces each marks; `fixed` pieces look the same in light and dark
const STORIES = [
	{
		fixed: ['banner-llama-cpp', 'banner-llama'],
		id: 'brand-logo-lockup--lockup',
		pieces: [
			'lockup-construction',
			'lockup-typed',
			'lockup-sizes',
			'lockup-grounds',
			'lockup-never',
			'lockup-in-context'
		]
	},
	{ id: 'brand-logo-usage--glyphs', pieces: ['icon-glyphs'] },
	{ id: 'brand-logo-usage--buttons', pieces: ['icon-buttons'] },
	{ id: 'brand-logo-usage--download-badge', pieces: ['icon-badges'] },
	{ id: 'brand-logo-usage--running-text', pieces: ['icon-inline'] },
	{ fixed: ['icon-in-place'], id: 'brand-logo-usage--app-icons-and-avatars', pieces: [] },
	{ id: 'ui-foundations-brand--splash-screens', pieces: ['icon-splash'] }
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

async function capture(base) {
	const { chromium } = await import('playwright');
	const browser = await chromium.launch();

	fs.mkdirSync(OUT, { recursive: true });

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
					'html,body,.logo{background:transparent!important}[data-readme="icon-in-place"] figcaption{display:none!important}';
				document.head.append(style);
			});
		});

		for (const story of STORIES) {
			await page.goto(`${base}/iframe.html?id=${story.id}&viewMode=story`);
			await page.waitForSelector('[data-readme]');
			await page.evaluate(() => document.fonts.ready);
			await page.waitForTimeout(400);

			const pieces = [...story.pieces, ...(tone === 'light' ? (story.fixed ?? []) : [])];

			for (const name of pieces) {
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

				const file = story.fixed?.includes(name) ? `${name}.png` : `${name}-${tone}.png`;

				await page.screenshot({
					clip,
					fullPage: true,
					omitBackground: true,
					path: path.join(OUT, file)
				});
				console.log(path.relative(process.cwd(), path.join(OUT, file)));
			}
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
