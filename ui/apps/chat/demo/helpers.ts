/**
 * Llama Chat's demo fixtures on the harness in ui/lib/demo: settings, conversations and a mocked
 * llama-server set the scene, or a Storybook story stands in for the app.
 */

import { demoMcpServerSettings } from './fixtures/demo-config.ts';
import { mockLlamaServer, type MockServerOptions } from './fixtures/llama-server.ts';
import { generateTrace } from './fixtures/traces.ts';
import { Demo, prepareDemoPage, recordDemo } from '@ggml-org/llama-demo';
import { type Locator, type Page, test as base, type TestInfo } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const VIDEOS_DIR = resolve(import.meta.dirname, 'videos');
const BRAND_DIR = resolve(import.meta.dirname, '../node_modules/@ggml-org/llama-brand');
/** How long a demo waits for the app to be ready before it starts recording. */
const APP_READY_TIMEOUT_MS = 90000;

/** Seeds UI settings once per demo, so later reloads keep what the demo changed. */
function seedSettings(config: Record<string, unknown>): void {
	const key = 'LlamaUi.config';

	if (localStorage.getItem(key) === null) {
		localStorage.setItem(key, JSON.stringify(config));
	}
}

interface DemoOptions {
	/** Empty conversations added to the library before the demo starts, e.g. to fill a list. */
	chats: number;
	/** Answers llama-server's endpoints in the browser instead (fixtures/llama-server.ts), so no model loads. */
	mockServer: MockServerOptions | null;
	/** Path the demo starts from. */
	startPath: string;
	/** Extra UI settings merged into the seeded config. */
	settings: Record<string, unknown>;
	/**
	 * Storybook story id (`?path=/story/<id>`) recorded instead of Llama Chat, from the Storybook
	 * that DEMO_STORYBOOK starts: `ui` (Llama UI) or `chat` (Llama Chat).
	 */
	story: string;
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

/** Imports conversations through Settings > Data, like a user would. */
export async function importTraces(page: Page, file: string): Promise<void> {
	// the expanded sidebar adds the shortcut to the name
	const settings = page.getByRole('button', { name: /^Settings(\s|$)/ }).first();
	const reachable = (target: Locator) =>
		target.click({ timeout: 2000, trial: true }).then(
			() => true,
			() => false
		);
	// a collapsed sidebar can sit under the page: on a phone the page's menu key slides it aside,
	// on a desktop its logo expands it
	const keys = [
		page.locator('xpath=//button[@aria-label="Expand navigation"][not(ancestor::aside)]'),
		page.locator('aside button[aria-label="Expand navigation"]')
	];

	// a few rounds, as a cold dev server reloads the page once it has optimized its dependencies,
	// which can land right after a key opened the sidebar and close it again
	for (let round = 0; round < 3 && !(await reachable(settings)); round++) {
		for (const key of keys) {
			if (await reachable(settings)) break;

			if ((await key.count()) > 0 && (await reachable(key.first()))) await key.first().click();
		}
	}

	await settings.click();

	// a dialog on a desktop or tablet, a page of its own on a phone
	const settingsView = page
		.getByRole('dialog')
		.or(page.getByRole('heading', { exact: true, level: 1, name: 'Settings' }));

	await settingsView.first().waitFor();

	const scope = (await page.getByRole('dialog').count()) > 0 ? page.getByRole('dialog') : page;

	// the section is Data in the regrouped settings, Import/Export before
	await scope
		.getByText(/^(Data|Import\/Export)$/)
		.filter({ visible: true })
		.first()
		.click();

	const chooser = page.waitForEvent('filechooser');

	await scope.getByRole('button', { name: /^Drop files here/ }).click();
	await (await chooser).setFiles(file);

	// one conversation gets a preview, then a prompt to open it; several get a picker, then a list
	const notNow = page.getByRole('button', { exact: true, name: 'Not now' });
	const importedList = page.getByRole('heading', { name: 'Imported Conversations' });

	await page
		.getByRole('button', { name: /^Import \(\d+\)$/ })
		.or(page.getByRole('alertdialog').getByRole('button', { exact: true, name: 'Import' }))
		.click();
	await notNow.or(importedList).waitFor();

	if (await notNow.isVisible()) await notNow.click();
	else await page.keyboard.press('Escape');

	// the app remembers the sidebar the import expanded; a demo starts with it collapsed, as on a
	// first visit
	await page.evaluate(() => {
		for (const key of Object.keys(localStorage)) {
			if (key.endsWith('.sidebarExpanded')) localStorage.removeItem(key);
		}
	});
}

/** The app's IndexedDB database and its conversations store, which a demo seeds. */
const DEMO_DATABASE = 'LlamaUi';
const DEMO_CONVERSATIONS_STORE = 'conversations';
const HOUR_MS = 3_600_000;
/** Names of the seeded conversations, repeated when a demo asks for more. */
const DEMO_CHAT_NAMES = [
	'Plan a trip to Lisbon',
	'Explain tail recursion',
	'Draft a polite follow-up',
	'Summarize this changelog',
	'Regex for ISO dates',
	'Ideas for a weekend project',
	'Debug a flaky test',
	'Meal plan for the week'
];

/** Adds `count` empty conversations to the library, so a list has enough chats to fold. */
async function seedChats(page: Page, count: number): Promise<void> {
	await page.goto('/');
	await page
		.getByRole('textbox', { name: 'Type a message...' })
		.waitFor({ timeout: APP_READY_TIMEOUT_MS });
	await page.evaluate(
		async ({ database, hourMs, names, store, total }) => {
			const request = indexedDB.open(database);
			const db = await new Promise<IDBDatabase>((resolve, reject) => {
				request.onsuccess = () => resolve(request.result);
				request.onerror = () => reject(request.error);
			});
			const tx = db.transaction(store, 'readwrite');

			for (let i = 0; i < total; i++) {
				tx.objectStore(store).put({
					currNode: null,
					id: `demo-chat-${i}`,
					lastModified: Date.now() - i * hourMs,
					name: names[i % names.length]
				});
			}

			await new Promise((resolve) => (tx.oncomplete = resolve));
			db.close();
		},
		{
			database: DEMO_DATABASE,
			hourMs: HOUR_MS,
			names: DEMO_CHAT_NAMES,
			store: DEMO_CONVERSATIONS_STORE,
			total: count
		}
	);
}

/** Seeds settings, imports the traces, adds the chats and opens `startPath` in Llama Chat. */
async function openChat(
	page: Page,
	settings: Record<string, unknown>,
	startPath: string,
	traces: string[],
	chats: number,
	testInfo: TestInfo
): Promise<void> {
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
		await page
			.getByRole('textbox', { name: 'Type a message...' })
			.waitFor({ timeout: APP_READY_TIMEOUT_MS });
		await importTraces(page, file);
	}

	if (chats > 0) await seedChats(page, chats);

	await page.goto(startPath);

	// a hash route after the import doesn't reload, which the conversations list needs
	if (traces.length > 0 || chats > 0) await page.reload();

	// a fresh checkout's dev server optimizes its dependencies on the first load and reloads the
	// page, which can take well past an action's timeout; the recording has not started yet
	await page
		.getByRole('textbox', { name: 'Type a message...' })
		.waitFor({ timeout: APP_READY_TIMEOUT_MS });
}

export const test = base.extend<DemoOptions & { demo: Demo }>({
	chats: [0, { option: true }],
	demo: async ({ page }, use, testInfo) => {
		await use(new Demo(page, `${testInfo.file}:${testInfo.title}`));
	},
	mockServer: [null, { option: true }],

	page: async (
		{ chats, colorScheme, mockServer, page, settings, startPath, story, traces },
		use,
		testInfo
	) => {
		await prepareDemoPage(page, testInfo);

		if (story && !process.env.DEMO_STORYBOOK) {
			throw new Error('demo: a story demo needs DEMO_STORYBOOK=ui (Llama UI) or chat (Llama Chat)');
		}

		if (story) {
			await page.goto(
				`http://localhost:${process.env.DEMO_STORYBOOK_PORT}/iframe.html?id=${story}&viewMode=story`
			);
			await page.locator('#storybook-root > *').first().waitFor({ timeout: APP_READY_TIMEOUT_MS });
		} else {
			if (mockServer) await mockLlamaServer(page, mockServer);

			await openChat(page, settings, startPath, traces, chats, testInfo);
		}

		await recordDemo(
			page,
			testInfo,
			{ brandDir: BRAND_DIR, dark: colorScheme === 'dark', videosDir: VIDEOS_DIR },
			() => use(page)
		);
	},

	settings: [{}, { option: true }],

	startPath: ['/', { option: true }],

	story: ['', { option: true }],

	traces: [[], { option: true }]
});

export { expect } from '@playwright/test';
