import { demoMcpServerSettings } from './fixtures/demo-config.ts';
import { generateTrace } from './fixtures/traces.ts';
import { type Locator, type Page, test as base } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';

const VIDEOS_DIR = resolve(import.meta.dirname, 'videos');

/**
 * Draws a fake cursor and click ripples, since recorded videos don't include
 * the OS pointer. Runs in the page via addInitScript.
 */
function installCursorOverlay(): void {
	const mount = () => {
		const style = document.createElement('style');

		style.textContent = `
			#demo-cursor { position: fixed; top: 0; left: 0; z-index: 2147483647; pointer-events: none;
				width: 22px; height: 22px; margin: -3px 0 0 -3px; transition: transform 80ms ease-out; }
			#demo-cursor.down { transform: scale(0.85); }
			.demo-ripple { position: fixed; z-index: 2147483646; pointer-events: none; width: 36px; height: 36px;
				margin: -18px 0 0 -18px; border-radius: 50%; background: rgba(59, 130, 246, 0.35);
				animation: demo-ripple 450ms ease-out forwards; }
			@keyframes demo-ripple { from { transform: scale(0.3); opacity: 1; } to { transform: scale(1.4); opacity: 0; } }
			#demo-caption { position: fixed; left: 50%; bottom: 48px; z-index: 2147483647; pointer-events: none;
				transform: translate(-50%, 8px); opacity: 0; transition: opacity 200ms ease, transform 200ms ease;
				max-width: 80%; padding: 10px 18px; border-radius: 10px; background: rgba(17, 17, 17, 0.88);
				color: #fff; font: 500 18px/1.4 system-ui, sans-serif; text-align: center; }
			#demo-caption.visible { opacity: 1; transform: translate(-50%, 0); }
		`;
		document.head.appendChild(style);

		const cursor = document.createElement('div');

		cursor.id = 'demo-cursor';
		cursor.innerHTML = `<svg viewBox="0 0 24 24" width="22" height="22"><path d="M4 2l16 10.5-7 1.5-3.5 6.5z"
			fill="#111" stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/></svg>`;
		document.documentElement.appendChild(cursor);

		const caption = document.createElement('div');

		caption.id = 'demo-caption';
		document.documentElement.appendChild(caption);

		document.addEventListener(
			'mousemove',
			(e) => (cursor.style.translate = `${e.clientX}px ${e.clientY}px`),
			true
		);
		document.addEventListener(
			'mousedown',
			(e) => {
				cursor.classList.add('down');
				const ripple = document.createElement('div');

				ripple.className = 'demo-ripple';
				ripple.style.left = `${e.clientX}px`;
				ripple.style.top = `${e.clientY}px`;
				document.documentElement.appendChild(ripple);
				setTimeout(() => ripple.remove(), 500);
			},
			true
		);
		document.addEventListener('mouseup', () => cursor.classList.remove('down'), true);
	};

	if (document.readyState === 'loading') {
		document.addEventListener('DOMContentLoaded', mount);
	} else {
		mount();
	}
}

/** Seeds UI settings once per demo, so later reloads keep what the demo changed. */
function seedSettings(config: Record<string, unknown>): void {
	const key = 'LlamaUi.config';

	if (localStorage.getItem(key) === null) {
		localStorage.setItem(key, JSON.stringify(config));
	}
}

export class Demo {
	/** Shows a caption at the bottom of the frame for `ms`. */
	async caption(text: string, ms = 2200): Promise<void> {
		await this.page.evaluate((t) => {
			const el = document.getElementById('demo-caption');

			if (!el) return;

			el.textContent = t;
			el.classList.add('visible');
		}, text);
		await this.pause(ms);
		await this.page.evaluate(() =>
			document.getElementById('demo-caption')?.classList.remove('visible')
		);
		await this.pause(250);
	}

	async click(target: Locator, ms = 400): Promise<void> {
		await this.moveTo(target);
		await this.pause(200);
		await this.page.mouse.down();
		await this.page.mouse.up();
		await this.pause(ms);
	}

	constructor(private readonly page: Page) {}

	async hover(target: Locator, ms = 350): Promise<void> {
		await this.moveTo(target);
		await this.pause(ms);
	}

	/** Glides the cursor to the center of `target`. */
	async moveTo(target: Locator, steps = 12): Promise<void> {
		await target.scrollIntoViewIfNeeded();
		const box = await target.boundingBox();

		if (!box) throw new Error(`demo: ${target} has no bounding box`);

		const x = box.x + box.width / 2;
		const y = box.y + box.height / 2;

		await this.page.mouse.move(x, y, { steps });
	}

	async pause(ms = 800): Promise<void> {
		await this.page.waitForTimeout(ms);
	}

	async press(key: string, ms = 600): Promise<void> {
		await this.page.keyboard.press(key);
		await this.pause(ms);
	}

	/** Scrolls the wheel over `target` by `dy` pixels (negative scrolls up), in small steps like a trackpad. */
	async scroll(target: Locator, dy: number, ms = 500): Promise<void> {
		await this.moveTo(target);

		const step = Math.sign(dy) * 40;

		for (let done = 0; Math.abs(done) < Math.abs(dy); done += step) {
			await this.page.mouse.wheel(0, step);
			await this.page.waitForTimeout(16);
		}

		await this.pause(ms);
	}

	async type(target: Locator, text: string, ms = 500): Promise<void> {
		await this.click(target, 200);
		await target.pressSequentially(text, { delay: 40 });
		await this.pause(ms);
	}
}

interface DemoOptions {
	/** Path the demo starts from. */
	startPath: string;
	/** Extra UI settings merged into the seeded config. */
	settings: Record<string, unknown>;
	/**
	 * Conversations imported before the demo starts: built-in traces from
	 * fixtures/traces.ts, or `<name>.jsonl` files from DEMO_TRACES_DATASET.
	 */
	traces: string[];
}

/** Reads a trace by name, generating built-in ones and downloading the rest. */
function loadTrace(name: string, dir: string): string {
	const generated = generateTrace(name);

	if (generated) return generated;

	const dataset = process.env.DEMO_TRACES_DATASET;

	if (!dataset)
		throw new Error(`demo: no built-in trace "${name}" and DEMO_TRACES_DATASET is not set`);

	execFileSync(
		'hf',
		['download', dataset, `${name}.jsonl`, '--repo-type', 'dataset', '--local-dir', dir],
		{ stdio: 'inherit' }
	);

	return readFileSync(join(dir, `${name}.jsonl`), 'utf8');
}

/** Imports conversations through Settings > Import/Export, like a user would. */
async function importTraces(page: Page, file: string): Promise<void> {
	// the expanded sidebar adds the shortcut to the name
	await page
		.getByRole('button', { name: /^Settings(\s|$)/ })
		.first()
		.click();

	const dialog = page.getByRole('dialog');

	await dialog.getByText('Import/Export', { exact: true }).filter({ visible: true }).click();

	const chooser = page.waitForEvent('filechooser');

	await dialog.getByRole('button', { name: 'Import conversations' }).click();
	await (await chooser).setFiles(file);
	await page.getByRole('button', { name: /^Import \(\d+\)$/ }).click();
	await page.getByRole('heading', { name: /^Imported \d+ conversation/ }).waitFor();
}

export const test = base.extend<DemoOptions & { demo: Demo }>({
	demo: async ({ page }, use) => {
		await use(new Demo(page));
	},
	page: async ({ page, settings, startPath, traces }, use, testInfo) => {
		const createdAt = Date.now();

		await page.addInitScript(installCursorOverlay);
		await page.addInitScript(seedSettings, {
			mcpServers: JSON.stringify(demoMcpServerSettings()),
			...settings
		});

		if (traces.length > 0) {
			const dir = testInfo.outputPath('traces');
			const file = join(dir, 'import.jsonl');

			mkdirSync(dir, { recursive: true });
			writeFileSync(file, traces.map((name) => loadTrace(name, dir)).join('\n'));
			await page.goto('/');
			await importTraces(page, file);
		}

		await page.goto(startPath);

		// a hash route after the import doesn't reload, which the conversations list needs
		if (traces.length > 0) await page.reload();

		await page.getByRole('textbox', { name: 'Type a message...' }).waitFor();

		const viewport = page.viewportSize();

		if (viewport) await page.mouse.move(viewport.width / 2, viewport.height / 2);

		// let MCP connections and fonts settle before the visible part starts
		await page.waitForTimeout(1500);

		const trimStart = (Date.now() - createdAt) / 1000;

		await use(page);

		const video = page.video();

		await page.close();

		if (!video || testInfo.status !== testInfo.expectedStatus) return;

		const file = basename(testInfo.file).replace(/\.demo\.ts$/, '');
		const name = testInfo.title === 'demo' ? file : `${file}--${slugify(testInfo.title)}`;
		const webm = resolve(VIDEOS_DIR, `${name}.webm`);

		mkdirSync(VIDEOS_DIR, { recursive: true });
		await video.saveAs(webm);
		exportMp4(webm, resolve(VIDEOS_DIR, `${name}.mp4`), trimStart);
	},

	settings: [{}, { option: true }],

	startPath: ['/', { option: true }],

	traces: [[], { option: true }]
});

export { expect } from '@playwright/test';

function slugify(text: string): string {
	return text
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, '-')
		.replace(/^-|-$/g, '');
}

/** Converts to H.264 mp4 (what GitHub accepts in PR descriptions), cutting the page load. */
function exportMp4(webm: string, mp4: string, trimStart: number): void {
	try {
		execFileSync(
			'ffmpeg',
			[
				'-y',
				'-loglevel',
				'error',
				'-ss',
				trimStart.toFixed(2),
				'-i',
				webm,
				'-c:v',
				'libx264',
				'-preset',
				'slow',
				'-crf',
				'18',
				'-pix_fmt',
				'yuv420p',
				'-movflags',
				'+faststart',
				mp4
			],
			{ stdio: 'inherit' }
		);
		rmSync(webm);
		console.log(`demo video: ${mp4}`);
	} catch {
		console.warn(`demo video: ffmpeg not available, kept ${webm}`);
	}
}
