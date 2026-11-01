import { demoMcpServerSettings } from './fixtures/demo-config.ts';
import { generateTrace } from './fixtures/traces.ts';
import { ScreencastRecorder } from './recorder.ts';
import { type Locator, type Page, test as base } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
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
			#demo-caption { position: fixed; left: 50%; top: 72px; z-index: 2147483647; pointer-events: none;
				transform: translate(-50%, -8px); opacity: 0; transition: opacity 200ms ease, transform 200ms ease;
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

/** Seeded randomness, so a demo moves the same way on every recording. */
function createRandom(seed: string): () => number {
	let state = [...seed].reduce((h, c) => Math.imul(h ^ c.charCodeAt(0), 2654435761), 1);

	return () => {
		state = (state + 0x6d2b79f5) | 0;

		let t = Math.imul(state ^ (state >>> 15), 1 | state);

		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;

		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/**
 * Drives the page like a person: the cursor glides along a slight arc with
 * Fitts-like durations, settles briefly before pressing, and typing and pauses
 * vary a little.
 */
export class Demo {
	private readonly page: Page;
	private readonly random: () => number;
	private x = 0;
	private y = 0;

	/** Shows a caption at the top of the frame for `ms`, clear of the chat form. */
	async caption(text: string, ms = 2200): Promise<void> {
		await this.page.evaluate((t) => {
			const el = document.getElementById('demo-caption');

			if (!el) return;

			el.textContent = t;
			el.classList.add('visible');
		}, text);
		await this.page.waitForTimeout(ms);
		await this.page.evaluate(() =>
			document.getElementById('demo-caption')?.classList.remove('visible')
		);
		await this.page.waitForTimeout(250);
	}

	async click(target: Locator, ms = 400): Promise<void> {
		await this.moveTo(target);
		await this.page.waitForTimeout(this.between(80, 160));
		await this.page.mouse.down();
		await this.page.waitForTimeout(this.between(50, 90));
		await this.page.mouse.up();
		await this.pause(ms);
	}

	constructor(page: Page, seed: string) {
		this.page = page;
		this.random = createRandom(seed);

		const viewport = page.viewportSize();

		this.x = (viewport?.width ?? 0) / 2;
		this.y = (viewport?.height ?? 0) / 2;
	}

	async hover(target: Locator, ms = 350): Promise<void> {
		await this.moveTo(target);
		await this.pause(ms);
	}

	/** Glides the cursor to a point near the center of `target`. */
	async moveTo(target: Locator): Promise<void> {
		await target.scrollIntoViewIfNeeded();
		const box = await target.boundingBox();

		if (!box) throw new Error(`demo: ${target} has no bounding box`);

		const x = box.x + box.width * this.between(0.4, 0.6);
		const y = box.y + box.height * this.between(0.4, 0.6);
		const dx = x - this.x;
		const dy = y - this.y;
		const distance = Math.hypot(dx, dy);
		const duration = 110 + 55 * Math.log2(distance / 40 + 1);
		// bow the path a little to one side, like a wrist turning
		const bow = distance * this.between(-0.08, 0.08);
		const fromX = this.x;
		const fromY = this.y;
		const started = Date.now();

		for (;;) {
			const t = Math.min((Date.now() - started) / duration, 1);
			const p = easeInOut(t);
			const arc = Math.sin(Math.PI * p) * bow;

			this.x = fromX + dx * p - (dy / (distance || 1)) * arc;
			this.y = fromY + dy * p + (dx / (distance || 1)) * arc;
			await this.page.mouse.move(this.x, this.y);

			if (t === 1) break;

			await this.page.waitForTimeout(8);
		}
	}

	/** Waits about `ms`, varied by up to 15% so pauses don't feel metronomic. */
	async pause(ms = 800): Promise<void> {
		await this.page.waitForTimeout(ms * this.between(0.85, 1.15));
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

	/** Types quickly with uneven key timing, a little slower after spaces and punctuation. */
	async type(target: Locator, text: string, ms = 500): Promise<void> {
		await this.click(target, 200);

		for (const char of text) {
			await this.page.keyboard.type(char);
			await this.page.waitForTimeout(
				this.between(15, 45) + (/[\s.,!?]/.test(char) ? this.between(20, 60) : 0)
			);
		}

		await this.pause(ms);
	}

	private between(min: number, max: number): number {
		return min + (max - min) * this.random();
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
	demo: async ({ page }, use, testInfo) => {
		await use(new Demo(page, `${testInfo.file}:${testInfo.title}`));
	},
	page: async ({ page, settings, startPath, traces }, use, testInfo) => {
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

		// let MCP connections and fonts settle before the recording starts
		await page.waitForTimeout(1500);

		const recorder = new ScreencastRecorder(testInfo.outputPath('frames'));

		await recorder.start(page);
		await use(page);
		await recorder.stop();
		await page.close();

		if (testInfo.status !== testInfo.expectedStatus) return;

		const file = basename(testInfo.file).replace(/\.demo\.ts$/, '');
		const name = testInfo.title === 'demo' ? file : `${file}--${slugify(testInfo.title)}`;
		const mp4 = resolve(VIDEOS_DIR, `${name}.mp4`);

		mkdirSync(VIDEOS_DIR, { recursive: true });
		recorder.encode(mp4);
		console.log(`demo video: ${mp4}`);
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
