// Guards the read-only /skills route presentation: distinct loading / error /
// unavailable / empty / success states, safe catalog fields only (no host
// paths, opaque XML never rendered), estimate labels, truncated resource
// lower bounds, the zero-budget note (distinct from a server-empty catalog),
// retry wiring, and persisted maxSkillBudget validation through the settings
// store.

import SkillsPage from '../../src/routes/skills/+page.svelte';
import SkillsPageWrapper from './components/SkillsPageWrapper.svelte';
import { page } from 'vitest/browser';
import { CONFIG_LOCALSTORAGE_KEY, SETTINGS_KEYS } from '$lib/constants';
import { serializeSkillCatalogEnvelope } from '$lib/services/skills-packing.service';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { modelsStore } from '$lib/stores/models.svelte';
import { skillsStore } from '$lib/stores/skills.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { isMobile } from '$lib/stores';
import type { SkillCatalogEntry, SkillCatalogResponse, SkillReadResult } from '$lib/types';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		headers: { 'content-type': 'application/json' },
		status
	});
}

function makeEntry(name: string, overrides: Partial<SkillCatalogEntry> = {}): SkillCatalogEntry {
	return {
		id: `opaque-${name}`,
		name,
		description: `description of ${name}`,
		scope: 'project',
		provider: 'agents',
		instruction: { bytes: 16, lines: 1, tokens: 4, tokens_estimated: true, modified_at: null },
		resources: { count: 0, truncated: false },
		// Server-owned opaque XML; the UI must never render it (it may hold
		// host paths the catalog presentation is forbidden from showing).
		catalog_xml: `<skill><name>${name}</name></skill>`,
		...overrides
	};
}

function makeCatalog(...entries: SkillCatalogEntry[]): SkillCatalogResponse {
	return {
		skills: entries,
		catalog_instruction_xml: '<available_skills>Call read_skill(name) when matching.</available_skills>',
		diagnostics: []
	};
}

function bodyText(): string {
	return document.body.textContent ?? '';
}

function mockFetchOnce(body: unknown, status = 200) {
	vi.mocked(fetch).mockImplementation(async () => jsonResponse(body, status));
}

/** Entries with catalog_xml padded to one identical byte length. */
function makePaddedEntries(count: number, entryBytes: number): SkillCatalogEntry[] {
	return Array.from({ length: count }, (_, i) => {
		const base = makeEntry(`skill-${i}`);
		const pad = Math.max(0, entryBytes - base.catalog_xml.length);

		return { ...base, catalog_xml: `${base.catalog_xml}${' '.repeat(pad)}` };
	});
}

/** Catalog whose serialized complete envelope is exactly targetBytes long. */
function makePaddedCatalog(
	entries: SkillCatalogEntry[],
	instructionXml: string,
	targetBytes: number
): SkillCatalogResponse {
	const base = { skills: entries, catalog_instruction_xml: instructionXml, diagnostics: [] };
	const pad = targetBytes - serializeSkillCatalogEnvelope(base).length;

	if (pad < 0) {
		throw new Error(`targetBytes ${targetBytes} is below the unpadded envelope length`);
	}

	return { ...base, catalog_instruction_xml: `${instructionXml}${' '.repeat(pad)}` };
}

/** Byte length of the budgeted envelope keeping `included` entries. */
function packedPrefixLength(catalog: SkillCatalogResponse, included: number): number {
	const { catalog_instruction_xml, skills } = catalog;

	return `<skills_catalog total="${skills.length}" included="${included}">${catalog_instruction_xml}${skills
		.slice(0, included)
		.map((entry) => entry.catalog_xml)
		.join('')}</skills_catalog>`.length;
}

/** Fetch mock serving the catalog and a one-token-per-character tokenizer. */
function mockCharCountingFetch(catalog: SkillCatalogResponse) {
	vi.mocked(fetch).mockImplementation(async (url, init) => {
		if (String(url).includes('/tokenize')) {
			const body = JSON.parse((init as RequestInit).body as string) as { content: string };

			return jsonResponse({ tokens: Array.from({ length: body.content.length }, (_, i) => i) });
		}

		return jsonResponse(catalog);
	});
}

describe('maxSkillBudget persisted validation', () => {
	beforeEach(() => {
		localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
		settingsStore.initialize();
	});

	it('defaults to 2000 when nothing is persisted', () => {
		expect(settingsStore.config[SETTINGS_KEYS.MAX_SKILL_BUDGET]).toBe(2000);
	});

	it('keeps a persisted zero: valid, not a fallback to the default', () => {
		localStorage.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify({ maxSkillBudget: 0 }));
		settingsStore.initialize();

		expect(settingsStore.config[SETTINGS_KEYS.MAX_SKILL_BUDGET]).toBe(0);
	});

	it('clamps a persisted negative value to zero', () => {
		localStorage.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify({ maxSkillBudget: -5 }));
		settingsStore.initialize();

		expect(settingsStore.config[SETTINGS_KEYS.MAX_SKILL_BUDGET]).toBe(0);
	});

	it('rounds a persisted fractional value to an integer', () => {
		localStorage.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify({ maxSkillBudget: 3.7 }));
		settingsStore.initialize();

		expect(settingsStore.config[SETTINGS_KEYS.MAX_SKILL_BUDGET]).toBe(4);
	});

	it('falls back to the default for a persisted non-numeric value', () => {
		localStorage.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify({ maxSkillBudget: 'huge' }));
		settingsStore.initialize();

		expect(settingsStore.config[SETTINGS_KEYS.MAX_SKILL_BUDGET]).toBe(2000);
	});
});

describe('/skills route presentation', () => {
	beforeEach(() => {
		localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
		settingsStore.initialize();
		skillsStore.invalidate(undefined);
		modelsStore.selectedModelName = null;
	});

	it('shows a loading state while the catalog request is in flight', async () => {
		vi.mocked(fetch).mockImplementation(() => new Promise(() => {}));

		render(SkillsPage);

		await vi.waitFor(() => expect(bodyText()).toContain('Loading catalog'));
	});

	it('renders only safe catalog fields with estimate, timestamp, and resource bounds', async () => {
		const timestamp = '2024-01-02T03:04:05Z';
		const catalog = makeCatalog(
			makeEntry('demo-skill', {
				description: 'A skill that does things.',
				instruction: {
					bytes: 1024,
					lines: 42,
					modified_at: timestamp,
					tokens: 512,
					tokens_estimated: true
				},
				resources: { count: 3, truncated: true },
				scope: 'global'
			}),
			makeEntry('second-skill', {
				instruction: { bytes: 8, lines: 1, modified_at: null, tokens: 2, tokens_estimated: false },
				resources: { count: 2, truncated: false }
			})
		);

		mockFetchOnce(catalog);
		render(SkillsPage);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		const text = bodyText();

		// Safe fields: name, description, scope, provider, instruction facts,
		// estimate label, timestamp, resource count / lower bound.
		expect(text).toContain('demo-skill');
		expect(text).toContain('A skill that does things.');
		expect(text).toContain('global');
		expect(text).toContain('agents');
		expect(text).toContain('512');
		expect(text).toContain('42');
		expect(text).toContain('estimated');
		expect(text).toContain('2024');
		// `Resources:` and the count are sibling text nodes, so the browser
		// serializes the inter-node whitespace into textContent; assert the
		// label/count contract tolerantly of that whitespace.
		// Truncated resource listing renders as a lower bound.
		expect(text).toMatch(/Resources:\s*3\+/);
		// A complete resource listing renders the exact count.
		expect(text).toMatch(/Resources:\s*2\b/);
		expect(text).toContain('exact');
	});

	it('never renders opaque catalog XML or host paths', async () => {
		const catalog = makeCatalog(
			makeEntry('demo-skill', {
				catalog_xml:
					'<skill><name>demo-skill</name><path>/srv/secret/skills/demo/SKILL.md</path></skill>'
			})
		);

		mockFetchOnce(catalog);
		render(SkillsPage);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		expect(bodyText()).not.toContain('/srv/secret/skills/demo/SKILL.md');
		expect(bodyText()).not.toContain('<skill>');
		expect(bodyText()).not.toContain('catalog_instruction_xml');
	});

	it('shows a distinct empty state for a server-empty catalog', async () => {
		mockFetchOnce(makeCatalog());
		render(SkillsPage);

		await vi.waitFor(() => expect(bodyText()).toContain('No skills found'));
	});

	it('keeps a zero budget distinct from a server-empty catalog', async () => {
		settingsStore.updateConfig('maxSkillBudget', 0);

		mockFetchOnce(makeCatalog(makeEntry('demo-skill')));
		render(SkillsPage);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));
		expect(bodyText()).toContain(
			'Skills tools are disabled because the catalog budget is 0 tokens'
		);
		expect(bodyText()).not.toContain('No skills found');
	});

	it('shows the generic error state with a retry action', async () => {
		mockFetchOnce({ error: { code: 503, message: 'catalog temporarily unavailable' } }, 503);
		render(SkillsPage);

		await vi.waitFor(() => expect(bodyText()).toContain('catalog temporarily unavailable'));
		expect(bodyText()).toContain('Retry');
	});

	it('distinguishes a missing skills route (unavailable) from a request error', async () => {
		mockFetchOnce('Not Found', 404);
		render(SkillsPage);

		await vi.waitFor(() => expect(bodyText()).toContain('not enabled'));
		expect(bodyText()).not.toContain('Retry');
	});

	it('renders inside the shared standalone page shell with the visible Skills title', async () => {
		mockFetchOnce(makeCatalog(makeEntry('demo-skill')));
		const screen = await render(SkillsPage);

		await expect.element(screen.getByTestId('standalone-page-shell')).toBeVisible();
		await expect.element(screen.getByRole('heading', { name: 'Skills' })).toBeVisible();
		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));
	});

	it('renders complete budget copy from the measured full token count and drops the old Budget line', async () => {
		modelsStore.selectedModelName = 'test-model';
		const catalog = makePaddedCatalog(
			[makeEntry('demo-skill', { catalog_xml: '<s/>' }), makeEntry('second-skill', { catalog_xml: '<s/>' })],
			'<inst/>',
			120
		);

		mockCharCountingFetch(catalog);
		vi.mocked(fetch).mockClear();
		render(SkillsPage);

		await vi.waitFor(() =>
			expect(bodyText()).toContain('The full Skills catalog uses 120 of 2,000 budget tokens')
		);
		const text = bodyText();

		expect(text).toContain('list_skill() is not registered');
		expect(text).toContain('demo-skill');
		expect(text).not.toContain('Budget:');
		// One tokenizer request for the complete envelope, never remeasured.
		expect(
			vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('/tokenize'))
		).toHaveLength(1);
	});

	it('renders partial budget copy with the full token requirement and the included count', async () => {
		modelsStore.selectedModelName = 'test-model';
		const catalog = makePaddedCatalog(makePaddedEntries(8, 80), '<inst/>', 2400);

		// The default 2,000 budget keeps exactly the first 3 of the 8 entries.
		expect(packedPrefixLength(catalog, 3)).toBeLessThanOrEqual(2000);
		expect(packedPrefixLength(catalog, 4)).toBeGreaterThan(2000);

		mockCharCountingFetch(catalog);
		vi.mocked(fetch).mockClear();
		render(SkillsPage);

		await vi.waitFor(() =>
			expect(bodyText()).toContain('The full Skills catalog requires 2,400 tokens')
		);
		const text = bodyText();

		expect(text).toContain('3 of 8 skills are included');
		expect(text).toContain('list_skill() is available');
		// One tokenizer request for the complete envelope plus the four
		// budget-boundary probes (the fourth exceeds the budget); the
		// envelope itself is never remeasured.
		expect(
			vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('/tokenize'))
		).toHaveLength(5);
	});

	it('labels the budget status as estimated when no direct tokenizer is available', async () => {
		// No selected model -> estimated packing, no tokenizer request.
		mockFetchOnce(makeCatalog(makeEntry('demo-skill')));
		vi.mocked(fetch).mockClear();
		render(SkillsPage);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));
		await vi.waitFor(() => expect(bodyText()).toMatch(/budget tokens \(estimated\)/));
		expect(
			vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('/tokenize'))
		).toHaveLength(0);
	});

	it('aborts a stale pack when the budget changes while tokenization is pending', async () => {
		modelsStore.selectedModelName = 'test-model';
		const catalog = makePaddedCatalog([makeEntry('demo-skill', { catalog_xml: '<s/>' })], '<inst/>', 80);
		const tokenizeResolvers: Array<(response: Response) => void> = [];

		vi.mocked(fetch).mockImplementation(async (url, init) => {
			if (String(url).includes('/tokenize')) {
				const signal = (init as RequestInit).signal;
				const { promise, resolve, reject } = Promise.withResolvers<Response>();

				tokenizeResolvers.push(resolve);
				// Like a real fetch, reject the in-flight request when its
				// signal aborts. SkillsPackingService.pack swallows that
				// rejection and resolves through the estimate fallback, so
				// the superseded pack settles instead of hanging: the
				// regression must be visible, not masked by a never-settling
				// promise that ignores the signal.
				signal?.addEventListener(
					'abort',
					() => reject(new DOMException('The operation was aborted.', 'AbortError')),
					{ once: true }
				);

				return promise;
			}

			return jsonResponse(catalog);
		});
		vi.mocked(fetch).mockClear();

		render(SkillsPage);

		// The first tokenization stays pending: no budget copy renders yet.
		await vi.waitFor(() => expect(tokenizeResolvers).toHaveLength(1));
		expect(bodyText()).not.toContain('The full Skills catalog');

		// A budget change aborts the pending pack and starts a new one. The
		// aborted first direct request rejects, pack() falls back to the
		// labeled deterministic estimate, and that stale result settles while
		// the replacement request is still pending. It must never render.
		settingsStore.updateConfig('maxSkillBudget', 500);

		await vi.waitFor(() => expect(tokenizeResolvers).toHaveLength(2));
		// The superseded pack settles through microtasks only (fetch
		// rejection -> apiFetch wrap -> pack estimate fallback -> success
		// handler); drain the queue deterministically rather than sleeping on
		// a wall clock.
		for (let i = 0; i < 25; i++) await Promise.resolve();

		expect(bodyText()).toContain('Calculating the Skills prompt budget...');
		expect(bodyText()).not.toContain('budget tokens');

		tokenizeResolvers[1](jsonResponse({ tokens: Array.from({ length: 80 }, (_, i) => i) }));

		await vi.waitFor(() =>
			expect(bodyText()).toContain('The full Skills catalog uses 80 of 500 budget tokens')
		);
		expect(bodyText()).not.toContain('uses 80 of 2,000');
	});
});

describe('/skills catalog preview', () => {
	/** Base read result whose rendered body and raw source carry the entry name. */
	function previewResult(name: string): SkillReadResult {
		return {
			kind: 'skill',
			skill: {
				id: `opaque-${name}`,
				name,
				scope: 'project',
				provider: 'agents',
				metadata: { description: `Structured description of ${name}` }
			},
			resources: { paths: [], truncated: false },
			source: `---\nname: ${name}\ndescription: raw frontmatter\n---\n# Content of ${name}\n\nBody text.\n`,
			body_markdown: `# Content of ${name}\n\nBody text.\n`,
			content_xml: `<skill_content name="${name}">body</skill_content>`,
			diagnostics: []
		};
	}

	/** Catalog GET plus a read that resolves per requested name. */
	function mockCatalogWithReads(catalog: SkillCatalogResponse) {
		vi.mocked(fetch).mockImplementation(async (url, init) => {
			if (String(url).includes('/skills/read')) {
				const body = JSON.parse((init as RequestInit).body as string) as { name: string };

				return jsonResponse(previewResult(body.name));
			}

			return jsonResponse(catalog);
		});
	}

	/**
	 * A controllable /skills/read fetch: records the request signal and, like
	 * a real fetch, rejects the in-flight request when that signal aborts.
	 */
	function deferredRead() {
		const state: { signal?: AbortSignal } = {};
		const { promise, resolve, reject } = Promise.withResolvers<Response>();

		return {
			promise,
			resolve,
			reject,
			attach(init?: RequestInit) {
				state.signal = init?.signal ?? undefined;
				init?.signal?.addEventListener(
					'abort',
					() => reject(new DOMException('The operation was aborted.', 'AbortError')),
					{ once: true }
				);
			},
			get signal() {
				return state.signal;
			}
		};
	}

	async function useDesktopViewport() {
		await page.viewport(1024, 768);
		await vi.waitFor(() => expect(isMobile.current).toBe(false));
	}

	function panes(): HTMLElement[] {
		return Array.from(document.querySelectorAll<HTMLElement>('[data-pane]'));
	}

	beforeEach(() => {
		localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
		settingsStore.initialize();
		skillsStore.invalidate(undefined);
		modelsStore.selectedModelName = null;
		conversationsStore.pendingCwd = null;
		vi.mocked(fetch).mockClear();
	});

	afterEach(async () => {
		await page.viewport(414, 896);
	});

	it('selects a focused card with Enter and shows the detail', async () => {
		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		const card = screen.getByRole('button', { name: /demo-skill/ });
		const element = card.element();

		element.focus();
		element.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));

		await vi.waitFor(() => expect(bodyText()).toContain('Content of demo-skill'));
		expect(screen.getByRole('button', { name: 'Back' }).query()).not.toBeNull();
	});

	it('selects a focused card with Space', async () => {
		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		const card = screen.getByRole('button', { name: /demo-skill/ });
		const element = card.element();

		element.focus();
		element.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));

		await vi.waitFor(() => expect(bodyText()).toContain('Content of demo-skill'));
		expect(screen.getByRole('button', { name: 'Back' }).query()).not.toBeNull();
	});

	it('marks the selected card with aria-pressed', async () => {
		await useDesktopViewport();
		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		await screen.getByRole('button', { name: /demo-skill/ }).click();

		await expect
			.element(screen.getByRole('button', { name: /demo-skill/ }))
			.toHaveAttribute('aria-pressed', 'true');
		await expect
			.element(screen.getByRole('button', { name: /second-skill/ }))
			.toHaveAttribute('aria-pressed', 'false');
	});

	it('opens a horizontal 55/45 split with 35/30 minimums on desktop selection', async () => {
		await useDesktopViewport();
		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		await screen.getByRole('button', { name: /demo-skill/ }).click();

		await vi.waitFor(() => expect(panes()).toHaveLength(2));

		expect(document.querySelector('[data-pane-group]')?.getAttribute('data-direction')).toBe(
			'horizontal'
		);
		await vi.waitFor(() => {
			const [listPane, detailPane] = panes();

			expect(parseFloat(listPane.style.flexGrow)).toBeCloseTo(55, 0);
			expect(parseFloat(detailPane.style.flexGrow)).toBeCloseTo(45, 0);
		});

		// Keyboard-resize the handle left until the list pane clamps at its 35
		// minimum; the detail pane takes the remainder.
		const handle = document.querySelector<HTMLElement>('[data-pane-resizer]');

		expect(handle).not.toBeNull();
		handle!.focus();
		for (let i = 0; i < 6; i++) {
			handle!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
		}

		await vi.waitFor(() => expect(parseFloat(panes()[0].style.flexGrow)).toBeCloseTo(35, 0));
		expect(parseFloat(panes()[1].style.flexGrow)).toBeCloseTo(65, 0);

		// Now the other way: the detail pane clamps at its 30 minimum.
		for (let i = 0; i < 10; i++) {
			handle!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
		}

		await vi.waitFor(() => expect(parseFloat(panes()[1].style.flexGrow)).toBeCloseTo(30, 0));
		expect(parseFloat(panes()[0].style.flexGrow)).toBeCloseTo(70, 0);
	});

	it('renders the desktop split boundary as a neutral 12px gutter with one centered divider', async () => {
		await useDesktopViewport();
		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		await screen.getByRole('button', { name: /demo-skill/ }).click();

		await vi.waitFor(() => expect(panes()).toHaveLength(2));

		// The existing draggable handle is the gutter: a fixed 12px (w-3)
		// neutral band carrying a single centered 1px border divider. The
		// class contract pins width and treatment (no layout measurement).
		const handle = document.querySelector<HTMLElement>('[data-pane-resizer]');

		expect(handle).not.toBeNull();
		expect(handle!.classList.contains('w-3')).toBe(true);
		expect(handle!.classList.contains('bg-muted')).toBe(true);
		expect(handle!.classList.contains('after:bg-border')).toBe(true);
		expect(handle!.classList.contains('after:w-px')).toBe(true);

		// The gutter keeps the pane minimums: the list clamps at 35 and the
		// detail at 30.
		handle!.focus();
		for (let i = 0; i < 6; i++) {
			handle!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
		}

		await vi.waitFor(() => expect(parseFloat(panes()[0].style.flexGrow)).toBeCloseTo(35, 0));
		expect(parseFloat(panes()[1].style.flexGrow)).toBeCloseTo(65, 0);

		for (let i = 0; i < 10; i++) {
			handle!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
		}

		await vi.waitFor(() => expect(parseFloat(panes()[1].style.flexGrow)).toBeCloseTo(30, 0));
		expect(parseFloat(panes()[0].style.flexGrow)).toBeCloseTo(70, 0);
	});

	it('moves the selected desktop workspace outside the catalog reading column', async () => {
		await useDesktopViewport();
		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));
		await screen.getByRole('button', { name: /demo-skill/ }).click();
		await vi.waitFor(() => expect(panes()).toHaveLength(2));

		const shell = document.querySelector<HTMLElement>('[data-testid="standalone-page-shell"]')!;
		const header = document.querySelector<HTMLElement>(
			'[data-testid="standalone-page-shell-header"]'
		)!;
		const content = document.querySelector<HTMLElement>('[data-testid="skills-catalog-content"]')!;
		const detailBody = document.querySelector<HTMLElement>('[data-testid="skill-detail-body"]')!;
		const list = document.querySelector<HTMLElement>('[data-pane] > div')!;

		expect(shell.classList.contains('max-w-4xl')).toBe(false);
		expect(header.classList.contains('mx-auto')).toBe(true);
		expect(header.classList.contains('max-w-4xl')).toBe(true);
		expect(content.classList.contains('mx-auto')).toBe(false);
		expect(content.classList.contains('max-w-4xl')).toBe(false);
		expect(list.classList.contains('overflow-y-auto')).toBe(true);
		// Task 1 moved the preview's scroll container to skill-detail-body.
		expect(detailBody.classList.contains('overflow-y-auto')).toBe(true);
	});

	it('preserves the resized session split when selecting another skill', async () => {
		await useDesktopViewport();
		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		await screen.getByRole('button', { name: /demo-skill/ }).click();

		await vi.waitFor(() => expect(panes()).toHaveLength(2));

		const handle = document.querySelector<HTMLElement>('[data-pane-resizer]')!;

		for (let i = 0; i < 4; i++) {
			handle.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
		}

		await vi.waitFor(() => expect(parseFloat(panes()[0].style.flexGrow)).toBeCloseTo(35, 0));

		await screen.getByRole('button', { name: /second-skill/ }).click();

		await vi.waitFor(() => expect(bodyText()).toContain('Content of second-skill'));
		expect(parseFloat(panes()[0].style.flexGrow)).toBeCloseTo(35, 0);
		expect(parseFloat(panes()[1].style.flexGrow)).toBeCloseTo(65, 0);
	});

	it('restores the full-width list on Close and reopens with the session split', async () => {
		await useDesktopViewport();
		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		await screen.getByRole('button', { name: /demo-skill/ }).click();

		await vi.waitFor(() => expect(panes()).toHaveLength(2));

		const handle = document.querySelector<HTMLElement>('[data-pane-resizer]')!;

		for (let i = 0; i < 4; i++) {
			handle.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
		}

		await vi.waitFor(() => expect(parseFloat(panes()[0].style.flexGrow)).toBeCloseTo(35, 0));

		await screen.getByTestId('skill-detail').getByRole('button', { name: 'Close' }).click();

		await vi.waitFor(() => expect(panes()).toHaveLength(0));
		// Full-width catalog: the other card is visible again, no split.
		expect(screen.getByRole('button', { name: /second-skill/ }).query()).not.toBeNull();

		await screen.getByRole('button', { name: /demo-skill/ }).click();

		await vi.waitFor(() => expect(panes()).toHaveLength(2));
		await vi.waitFor(() => expect(parseFloat(panes()[0].style.flexGrow)).toBeCloseTo(35, 0));
		expect(parseFloat(panes()[1].style.flexGrow)).toBeCloseTo(65, 0);
	});

	it('shows a full-screen detail on mobile and Back restores the list', async () => {
		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		await screen.getByRole('button', { name: /demo-skill/ }).click();

		await vi.waitFor(() => expect(bodyText()).toContain('Content of demo-skill'));
		// Mobile: no split, the detail replaces the list.
		expect(panes()).toHaveLength(0);
		expect(screen.getByRole('button', { name: 'Back' }).query()).not.toBeNull();
		expect(screen.getByRole('button', { name: /second-skill/ }).query()).toBeNull();

		await screen.getByRole('button', { name: 'Back' }).click();

		await vi.waitFor(() =>
			expect(screen.getByRole('button', { name: /second-skill/ }).query()).not.toBeNull()
		);
		expect(screen.getByRole('button', { name: 'Back' }).query()).toBeNull();
	});

	it('aborts the detail read and clears the selection when the CWD changes', async () => {
		conversationsStore.pendingCwd = '/srv/project-a';
		const read = deferredRead();
		const readCwdHeaders: string[] = [];

		vi.mocked(fetch).mockImplementation(async (url, init) => {
			if (String(url).includes('/skills/read')) {
				readCwdHeaders.push(
					new Headers((init as RequestInit).headers).get('x-skill-cwd') ?? ''
				);
				read.attach(init);
				return read.promise;
			}

			return jsonResponse(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));
		});

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		await screen.getByRole('button', { name: /demo-skill/ }).click();

		await vi.waitFor(() => expect(read.signal).toBeDefined());
		expect(readCwdHeaders).toEqual(['/srv/project-a']);

		conversationsStore.pendingCwd = '/srv/project-b';

		await vi.waitFor(() => expect(read.signal?.aborted).toBe(true));
		// The old detail is cleared: the list is back and the mobile Back
		// action is gone.
		await vi.waitFor(() =>
			expect(screen.getByRole('button', { name: /second-skill/ }).query()).not.toBeNull()
		);
		expect(screen.getByRole('button', { name: 'Back' }).query()).toBeNull();
	});
});
