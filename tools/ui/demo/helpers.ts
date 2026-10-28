import { demoMcpServerSettings } from './fixtures/demo-config.ts';
import { type Locator, type Page, test as base } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import { basename, resolve } from 'node:path';

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
}

export const test = base.extend<DemoOptions & { demo: Demo }>({
	demo: async ({ page }, use) => {
		await use(new Demo(page));
	},
	page: async ({ page, settings, startPath }, use, testInfo) => {
		const createdAt = Date.now();

		await page.addInitScript(installCursorOverlay);
		await page.addInitScript(seedSettings, {
			mcpServers: JSON.stringify(demoMcpServerSettings()),
			...settings
		});

		await page.goto(startPath);
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

	startPath: ['/', { option: true }]
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
