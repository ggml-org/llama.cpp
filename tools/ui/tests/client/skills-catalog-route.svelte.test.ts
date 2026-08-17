// Guards the read-only /skills route presentation: distinct loading / error /
// unavailable / empty / success states, safe catalog fields only (no host
// paths, opaque XML never rendered), estimate labels, truncated resource
// lower bounds, the zero-budget note (distinct from a server-empty catalog),
// retry wiring, and persisted maxSkillBudget validation through the settings
// store.

import SkillsPage from '../../src/routes/skills/+page.svelte';
import SkillsPageWrapper from './components/SkillsPageWrapper.svelte';
import SkillCatalogList from '$lib/components/app/skills/SkillCatalogList.svelte';
import {
	CONFIG_LOCALSTORAGE_KEY,
	DISABLED_SKILL_IDS_LOCALSTORAGE_KEY,
	SETTINGS_KEYS,
	SKILLS_PANE_SIZES_LOCALSTORAGE_KEY
} from '$lib/constants';
import { buildSkillRunSnapshot, serializeSkillCatalogEnvelope } from '$lib/services/skills.service';
import { isMobile } from '$lib/stores';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { modelsStore } from '$lib/stores/models.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { skillAvailabilityStore } from '$lib/stores/skill-availability.svelte';
import { skillsStore } from '$lib/stores/skills.svelte';
import type { SkillCatalogEntry, SkillCatalogResponse, SkillReadResult } from '$lib/types';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		headers: { 'content-type': 'application/json' },
		status
	});
}

function makeEntry(name: string, overrides: Partial<SkillCatalogEntry> = {}): SkillCatalogEntry {
	return {
		// Server-owned opaque XML; the UI must never render it (it may hold
		// host paths the catalog presentation is forbidden from showing).
		catalog_xml: `<skill><name>${name}</name></skill>`,
		description: `description of ${name}`,
		id: `opaque-${name}`,
		instruction: { bytes: 16, lines: 1, modified_at: null, tokens: 4, tokens_estimated: true },
		name,
		provider: 'agents',
		resources: { count: 0, truncated: false },
		scope: 'project',
		...overrides
	};
}

const LONG_DESCRIPTION =
	'This description contains enough repeated words to wrap across more than three lines in the catalog card at the default viewport width. '.repeat(
		4
	);
const SHORT_DESCRIPTION = 'A short skill description.';

function makeCatalog(...entries: SkillCatalogEntry[]): SkillCatalogResponse {
	return {
		catalog_instruction_xml:
			'<available_skills>Call read_skill(name) when matching.</available_skills>',
		diagnostics: [],
		skills: entries
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
	const base = { catalog_instruction_xml: instructionXml, diagnostics: [], skills: entries };
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

	it.each([
		['defaults to 2000 when missing', undefined, 2000],
		['keeps zero', 0, 0],
		['clamps negative values', -5, 0],
		['rounds fractional values', 3.7, 4],
		['falls back for non-numeric values', 'huge', 2000]
	])('%s', (_label, value, expected) => {
		if (value !== undefined) {
			localStorage.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify({ maxSkillBudget: value }));
		}

		settingsStore.initialize();

		expect(settingsStore.config[SETTINGS_KEYS.MAX_SKILL_BUDGET]).toBe(expected);
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
		// timestamp, resource count / lower bound. The `agents` API provider
		// renders as the provider-agnostic `generic` label and the card shows
		// a tilde-prefixed token count instead of an `estimated` chip.
		expect(text).toContain('demo-skill');
		expect(text).toContain('A skill that does things.');
		expect(text).toContain('global');
		expect(text).toContain('generic');
		expect(text).not.toContain('agents');
		expect(text).toContain('~512 tokens');
		expect(text).toContain('42');
		expect(text).toContain('2024');
		// `Resources:` and the count are sibling text nodes, so the browser
		// serializes the inter-node whitespace into textContent; assert the
		// label/count contract tolerantly of that whitespace.
		// Truncated resource listing renders as a lower bound.
		expect(text).toMatch(/Resources:\s*3\+/);
		// A complete resource listing renders the exact count.
		expect(text).toMatch(/Resources:\s*2\b/);
		// Exact counts render plainly with no count-mode chip.
		expect(text).toContain('2 tokens');
		expect(text).not.toContain('exact');
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

	it('renders every diagnostic identity field with labels and keeps duplicate-code rows independent', async () => {
		const catalog: SkillCatalogResponse = {
			...makeCatalog(makeEntry('demo-skill')),
			diagnostics: [
				{
					code: 'overlapping-skill',
					message: 'first diagnostic message',
					name: 'Alpha Skill',
					provider: 'agents',
					scope: 'global',
					severity: 'warning'
				},
				{
					code: 'overlapping-skill',
					message: 'second diagnostic message',
					name: 'Beta Skill',
					provider: 'local',
					scope: 'project',
					severity: 'error'
				}
			]
		};

		mockFetchOnce(catalog);
		render(SkillsPage);

		await vi.waitFor(() => expect(bodyText()).toContain('first diagnostic message'));

		const text = bodyText();

		// Code plus every identity field, each under its own readable label.
		expect(text).toContain('overlapping-skill');
		expect(text).toContain('Skill: Alpha Skill');
		expect(text).toContain('Skill: Beta Skill');
		expect(text).toContain('Scope: global');
		expect(text).toContain('Scope: project');
		expect(text).toContain('Provider: generic');
		expect(text).not.toContain('Provider: agents');
		expect(text).toContain('Provider: local');

		// Both messages render; duplicate codes stay two independent rows.
		expect(text).toContain('first diagnostic message');
		expect(text).toContain('second diagnostic message');
		expect(text.match(/Skill:/g)).toHaveLength(2);
		expect(text.match(/Scope:/g)).toHaveLength(2);
		expect(text.match(/Provider:/g)).toHaveLength(2);
	});

	it('renders the collapsed shadowed providers list of one skill_shadowed diagnostic', async () => {
		const catalog: SkillCatalogResponse = {
			...makeCatalog(makeEntry('demo-skill')),
			diagnostics: [
				{
					code: 'skill_shadowed',
					message: 'Skill is shadowed by a higher-precedence entry',
					name: 'demo-skill',
					provider: 'claude',
					providers: ['claude', 'gemini', 'opencode'],
					scope: 'project',
					severity: 'warning'
				}
			]
		};

		mockFetchOnce(catalog);
		render(SkillsPage);

		await vi.waitFor(() => expect(bodyText()).toContain('skill_shadowed'));

		const text = bodyText();

		// The singular first provider and the full collapsed list both render.
		expect(text).toContain('Provider: claude');
		expect(text).toContain('Providers: claude, gemini, opencode');
		expect(text.match(/Providers:/g)).toHaveLength(1);
		expect(text.match(/skill_shadowed/g)).toHaveLength(1);
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

	it('renders complete budget copy from the measured full token count and drops the old Budget line', async () => {
		modelsStore.selectedModelName = 'test-model';
		const catalog = makePaddedCatalog(
			[
				makeEntry('demo-skill', { catalog_xml: '<s/>' }),
				makeEntry('second-skill', { catalog_xml: '<s/>' })
			],
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
		const text = bodyText().replace(/\s+/g, ' ');

		expect(text).toContain('3 of 8 skills are included');
		expect(text).toContain('list_skill() is available');
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
		const catalog = makePaddedCatalog(
			[makeEntry('demo-skill', { catalog_xml: '<s/>' })],
			'<inst/>',
			80
		);
		const tokenizeResolvers: Array<(response: Response) => void> = [];

		vi.mocked(fetch).mockImplementation(async (url, init) => {
			if (String(url).includes('/tokenize')) {
				const signal = (init as RequestInit).signal;
				const { promise, reject, resolve } = Promise.withResolvers<Response>();

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
			body_markdown: `# Content of ${name}\n\nBody text.\n`,
			content_xml: `<skill_content name="${name}">body</skill_content>`,
			diagnostics: [],
			kind: 'skill',
			resources: { paths: [], truncated: false },
			skill: {
				id: `opaque-${name}`,
				metadata: { description: `Structured description of ${name}` },
				name,
				provider: 'agents',
				scope: 'project'
			},
			source: `---\nname: ${name}\ndescription: raw frontmatter\n---\n# Content of ${name}\n\nBody text.\n`
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

	function stubDescriptionMeasurement() {
		vi.stubGlobal(
			'ResizeObserver',
			class {
				constructor(private readonly callback: ResizeObserverCallback) {}

				observe(node: Element) {
					const overflowing = (node.textContent?.length ?? 0) > 100;

					Object.defineProperties(node, {
						clientHeight: { configurable: true, value: overflowing ? 50 : 20 },
						scrollHeight: { configurable: true, value: overflowing ? 200 : 20 }
					});
					this.callback([], this as unknown as ResizeObserver);
				}

				disconnect() {}
			}
		);
	}
	async function renderCatalogWithReads(catalog: SkillCatalogResponse) {
		mockCatalogWithReads(catalog);
		const screen = await render(SkillsPageWrapper);
		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));
		return screen;
	}

	/**
	 * A controllable /skills/read fetch: records the request signal and, like
	 * a real fetch, rejects the in-flight request when that signal aborts.
	 */
	function deferredRead() {
		const state: { signal?: AbortSignal } = {};
		const { promise, reject, resolve } = Promise.withResolvers<Response>();

		return {
			attach(init?: RequestInit) {
				state.signal = init?.signal ?? undefined;
				init?.signal?.addEventListener(
					'abort',
					() => reject(new DOMException('The operation was aborted.', 'AbortError')),
					{ once: true }
				);
			},
			promise,
			reject,
			resolve,
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
		localStorage.removeItem(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY);
		localStorage.removeItem(SKILLS_PANE_SIZES_LOCALSTORAGE_KEY);
		// The availability store is a process-wide singleton; clear any IDs
		// left by a prior test so each catalog toggle test starts enabled.
		for (const id of [...skillAvailabilityStore.disabledIds]) {
			skillAvailabilityStore.setEnabled(id, true);
		}
		settingsStore.initialize();
		skillsStore.invalidate(undefined);
		modelsStore.selectedModelName = null;
		conversationsStore.pendingCwd = null;
		vi.mocked(fetch).mockClear();
	});

	afterEach(async () => {
		vi.unstubAllGlobals();
		await page.viewport(414, 896);
	});

	it('restores a valid persisted desktop split', async () => {
		await useDesktopViewport();
		localStorage.setItem(SKILLS_PANE_SIZES_LOCALSTORAGE_KEY, JSON.stringify([40, 60]));
		const screen = await renderCatalogWithReads(makeCatalog(makeEntry('demo-skill')));
		await screen.getByRole('button', { name: /demo-skill/ }).click();
		await vi.waitFor(() => expect(panes()).toHaveLength(2));
		await vi.waitFor(() => {
			expect(parseFloat(panes()[0].style.flexGrow)).toBeCloseTo(40, 0);
			expect(parseFloat(panes()[1].style.flexGrow)).toBeCloseTo(60, 0);
		});
	});

	it('normalizes persisted pane sizes and falls back for malformed values', async () => {
		await useDesktopViewport();
		localStorage.setItem(SKILLS_PANE_SIZES_LOCALSTORAGE_KEY, JSON.stringify([40, 40]));
		const first = await renderCatalogWithReads(makeCatalog(makeEntry('demo-skill')));
		await first.getByRole('button', { name: /demo-skill/ }).click();
		await vi.waitFor(() => expect(panes()).toHaveLength(2));
		await vi.waitFor(() => expect(parseFloat(panes()[0].style.flexGrow)).toBeCloseTo(50, 0));
		first.unmount();

		localStorage.setItem(SKILLS_PANE_SIZES_LOCALSTORAGE_KEY, JSON.stringify([10, 90]));
		const second = await renderCatalogWithReads(makeCatalog(makeEntry('demo-skill')));
		await second.getByRole('button', { name: /demo-skill/ }).click();
		await vi.waitFor(() => expect(panes()).toHaveLength(2));
		await vi.waitFor(() => expect(parseFloat(panes()[0].style.flexGrow)).toBeCloseTo(35, 0));
		second.unmount();

		localStorage.setItem(SKILLS_PANE_SIZES_LOCALSTORAGE_KEY, JSON.stringify({ left: 40 }));
		const third = await renderCatalogWithReads(makeCatalog(makeEntry('demo-skill')));
		await third.getByRole('button', { name: /demo-skill/ }).click();
		await vi.waitFor(() => expect(panes()).toHaveLength(2));
		await vi.waitFor(() => expect(parseFloat(panes()[0].style.flexGrow)).toBeCloseTo(55, 0));
	});

	it('persists normalized sizes after resizing the desktop split', async () => {
		await useDesktopViewport();
		const screen = await renderCatalogWithReads(makeCatalog(makeEntry('demo-skill')));
		await screen.getByRole('button', { name: /demo-skill/ }).click();
		await vi.waitFor(() => expect(panes()).toHaveLength(2));

		const handle = document.querySelector<HTMLElement>('[data-pane-resizer]')!;
		handle.focus();
		handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowLeft' }));

		await vi.waitFor(() => {
			const stored = JSON.parse(
				localStorage.getItem(SKILLS_PANE_SIZES_LOCALSTORAGE_KEY) ?? 'null'
			) as unknown;

			expect(Array.isArray(stored)).toBe(true);
			expect(stored).toHaveLength(2);
			expect((stored as number[]).reduce((total, value) => total + value, 0)).toBeCloseTo(100);
		});
	});

	it.each([
		['Enter', 'Enter'],
		['Space', ' ']
	])('selects a focused card with %s and shows the detail', async (_label, key) => {
		await useDesktopViewport();
		const screen = await renderCatalogWithReads(
			makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill'))
		);

		const card = screen.getByRole('button', { name: /demo-skill/ });
		const element = card.element();

		element.focus();
		element.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key }));

		await vi.waitFor(() => expect(bodyText()).toContain('Content of demo-skill'));
		await expect
			.element(screen.getByRole('button', { name: /demo-skill/ }))
			.toHaveAttribute('aria-pressed', 'true');
		await expect
			.element(screen.getByRole('button', { name: /second-skill/ }))
			.toHaveAttribute('aria-pressed', 'false');
	});

	it('toggles long card descriptions with a chevron disclosure', async () => {
		stubDescriptionMeasurement();
		const longEntry = makeEntry('long-skill', { description: LONG_DESCRIPTION });
		const shortEntry = makeEntry('short-skill', { description: SHORT_DESCRIPTION });
		const emptyEntry = makeEntry('empty-skill', { description: '' });

		mockCatalogWithReads(makeCatalog(longEntry, shortEntry, emptyEntry));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('long-skill'));

		const description = screen.getByTestId(`skill-description-${longEntry.id}`).element();
		const showMore = screen.getByTestId(`skill-description-toggle-${longEntry.id}`);

		expect(description.classList.contains('line-clamp-3')).toBe(true);
		expect(showMore.query()).not.toBeNull();
		expect(screen.getByTestId(`skill-description-toggle-${shortEntry.id}`).query()).toBeNull();
		expect(screen.getByTestId(`skill-description-${emptyEntry.id}`).query()).toBeNull();

		await showMore.click();
		await vi.waitFor(() => expect(bodyText()).toContain('Show less'));
		expect(description.classList.contains('line-clamp-3')).toBe(false);
		expect(showMore.element().getAttribute('aria-expanded')).toBe('true');

		await showMore.click();
		await vi.waitFor(() => expect(bodyText()).toContain('Show more'));
		expect(description.classList.contains('line-clamp-3')).toBe(true);
		expect(showMore.element().getAttribute('aria-expanded')).toBe('false');
	});

	it('renders multiline catalog descriptions as one paragraph', async () => {
		const entry = makeEntry('multiline-skill', {
			description: 'This is line one.\n  This is line two.\n\nThis is the final line.'
		});

		mockCatalogWithReads(makeCatalog(entry));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('multiline-skill'));

		expect(screen.getByTestId(`skill-description-${entry.id}`).element().textContent).toBe(
			'This is line one. This is line two. This is the final line.'
		);
	});

	it('does not select a card when its description disclosure is clicked', async () => {
		stubDescriptionMeasurement();
		const longEntry = makeEntry('long-skill', { description: LONG_DESCRIPTION });

		mockCatalogWithReads(makeCatalog(longEntry));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('long-skill'));
		await screen.getByTestId(`skill-description-toggle-${longEntry.id}`).click();
		await vi.waitFor(() => expect(bodyText()).toContain('Show less'));
		expect(screen.getByTestId('skill-detail').query()).toBeNull();

		await screen.getByText('long-skill').click();
		await vi.waitFor(() => expect(bodyText()).toContain('Content of long-skill'));
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
			handle!.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowLeft' }));
		}

		await vi.waitFor(() => expect(parseFloat(panes()[0].style.flexGrow)).toBeCloseTo(35, 0));
		expect(parseFloat(panes()[1].style.flexGrow)).toBeCloseTo(65, 0);

		// Now the other way: the detail pane clamps at its 30 minimum.
		for (let i = 0; i < 10; i++) {
			handle!.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }));
		}

		await vi.waitFor(() => expect(parseFloat(panes()[1].style.flexGrow)).toBeCloseTo(30, 0));
		expect(parseFloat(panes()[0].style.flexGrow)).toBeCloseTo(70, 0);
	});

	it('does not re-read or replay detail selection for the already-selected card', async () => {
		await useDesktopViewport();
		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));
		const card = screen.getByRole('button', { name: /demo-skill/ });

		await card.click();
		await vi.waitFor(() => expect(bodyText()).toContain('Content of demo-skill'));

		const readCount = () =>
			vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('/skills/read')).length;

		expect(readCount()).toBe(1);

		await card.click();
		await tick();
		await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

		expect(readCount()).toBe(1);
		expect(bodyText()).toContain('Content of demo-skill');
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
			handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowLeft' }));
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
			handle.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowLeft' }));
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

	it('does not read or write the pane preference on mobile', async () => {
		const getItem = vi.spyOn(Storage.prototype, 'getItem');
		const setItem = vi.spyOn(Storage.prototype, 'setItem');

		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));
		expect(getItem).not.toHaveBeenCalledWith(SKILLS_PANE_SIZES_LOCALSTORAGE_KEY);

		await screen.getByRole('button', { name: /demo-skill/ }).click();
		await vi.waitFor(() => expect(bodyText()).toContain('Content of demo-skill'));

		expect(setItem).not.toHaveBeenCalledWith(
			SKILLS_PANE_SIZES_LOCALSTORAGE_KEY,
			expect.any(String)
		);
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
				readCwdHeaders.push(new Headers((init as RequestInit).headers).get('x-skill-cwd') ?? '');
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

	it('disabling the selected card leaves its detail open and shows a Disabled badge', async () => {
		await useDesktopViewport();
		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));
		await screen.getByRole('button', { name: /demo-skill/ }).click();
		await vi.waitFor(() => expect(bodyText()).toContain('Content of demo-skill'));

		// Toggling availability never selects or deselects the card.
		await screen.getByRole('switch', { name: 'Disable demo-skill' }).click();
		await vi.waitFor(() => expect(bodyText()).toContain('Disabled'));

		// The detail stays open and the card stays selected.
		expect(bodyText()).toContain('Content of demo-skill');
		await expect
			.element(screen.getByRole('button', { name: /demo-skill/ }))
			.toHaveAttribute('aria-pressed', 'true');
	});

	it('keeps the budget preview and does not retokenize when a skill is disabled and re-enabled', async () => {
		await useDesktopViewport();
		modelsStore.selectedModelName = 'test-model';
		const catalog = makePaddedCatalog(
			[
				makeEntry('demo-skill', { catalog_xml: '<s/>' }),
				makeEntry('second-skill', { catalog_xml: '<s/>' })
			],
			'<inst/>',
			120
		);

		// Catalog GET, per-name read, and a one-token-per-character
		// tokenizer, so the initial pack measures the complete envelope.
		vi.mocked(fetch).mockImplementation(async (url, init) => {
			if (String(url).includes('/tokenize')) {
				const body = JSON.parse((init as RequestInit).body as string) as { content: string };

				return jsonResponse({ tokens: Array.from({ length: body.content.length }, (_, i) => i) });
			}

			if (String(url).includes('/skills/read')) {
				const body = JSON.parse((init as RequestInit).body as string) as { name: string };

				return jsonResponse(previewResult(body.name));
			}

			return jsonResponse(catalog);
		});
		vi.mocked(fetch).mockClear();

		const screen = await render(SkillsPageWrapper);

		const fullSnapshot = buildSkillRunSnapshot(undefined, catalog);
		const budgetCopy = `The full Skills catalog uses ${fullSnapshot.envelope.length.toLocaleString()} of 2,000 budget tokens`;
		const tokenizeCalls = () =>
			vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('/tokenize'));

		await vi.waitFor(() => expect(bodyText()).toContain(budgetCopy));
		// The initial render measured the complete envelope exactly once.
		expect(tokenizeCalls()).toHaveLength(1);

		// Toggling availability only flips the card's Disabled badge; the
		// loaded catalog content is unchanged, so the pack/budget preview is
		// neither aborted nor remeasured.
		await screen.getByRole('switch', { name: 'Disable demo-skill' }).click();
		await vi.waitFor(() => expect(bodyText()).toContain('Disabled'));
		expect(bodyText()).toContain(budgetCopy);
		expect(tokenizeCalls()).toHaveLength(1);

		await screen.getByRole('switch', { name: 'Enable demo-skill' }).click();
		await vi.waitFor(() => expect(bodyText()).not.toContain('Disabled'));
		expect(bodyText()).toContain(budgetCopy);
		expect(tokenizeCalls()).toHaveLength(1);

		// The card/detail interaction stays usable: selecting the card opens
		// its detail, and toggling from there keeps the detail open while the
		// badge appears and disappears, still without a new tokenizer call.
		await screen.getByRole('button', { name: /demo-skill/ }).click();
		await vi.waitFor(() => expect(bodyText()).toContain('Content of demo-skill'));

		await screen.getByRole('switch', { name: 'Disable demo-skill' }).click();
		await vi.waitFor(() => expect(bodyText()).toContain('Disabled'));
		expect(bodyText()).toContain('Content of demo-skill');
		expect(tokenizeCalls()).toHaveLength(1);

		await screen.getByRole('switch', { name: 'Enable demo-skill' }).click();
		await vi.waitFor(() => expect(bodyText()).not.toContain('Disabled'));
		expect(bodyText()).toContain('Content of demo-skill');
		expect(tokenizeCalls()).toHaveLength(1);
	});

	it('labels every card switch with the Enable/Disable action for its state', async () => {
		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		// Every card exposes a switch whose accessible name names the action
		// its current state implies.
		await expect
			.element(screen.getByRole('switch', { name: 'Disable demo-skill' }))
			.toBeInTheDocument();
		await expect
			.element(screen.getByRole('switch', { name: 'Disable second-skill' }))
			.toBeInTheDocument();

		// Disabling flips only that card's switch to the re-enable action.
		await screen.getByRole('switch', { name: 'Disable demo-skill' }).click();

		await vi.waitFor(() => expect(bodyText()).toContain('Disabled'));
		await expect
			.element(screen.getByRole('switch', { name: 'Enable demo-skill' }))
			.toBeInTheDocument();
		await expect
			.element(screen.getByRole('switch', { name: 'Disable second-skill' }))
			.toBeInTheDocument();
	});

	it('keeps switch and action-label clicks from selecting the card', async () => {
		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		await screen.getByRole('switch', { name: 'Disable demo-skill' }).click();
		await vi.waitFor(() => expect(bodyText()).toContain('Disabled'));

		const assertUnselected = () => {
			expect(screen.getByTestId('skill-detail').query()).toBeNull();
			expect(bodyText()).not.toContain('Content of demo-skill');
			expect(
				vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('/skills/read'))
			).toHaveLength(0);
		};

		await expect
			.element(screen.getByRole('button', { name: /demo-skill/ }))
			.toHaveAttribute('aria-pressed', 'false');
		assertUnselected();

		const label = document.querySelector<HTMLElement>(
			'label[for="skill-enabled-opaque-demo-skill"]'
		);

		expect(label).not.toBeNull();
		label!.click();

		await vi.waitFor(() => expect(bodyText()).not.toContain('Disabled'));
		assertUnselected();
	});

	it.each(['Enter', ' '])('keeps %s on the switch from selecting the card', async (key) => {
		mockCatalogWithReads(makeCatalog(makeEntry('demo-skill'), makeEntry('second-skill')));

		const screen = await render(SkillsPageWrapper);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		const switchElement = screen.getByRole('switch', { name: 'Disable demo-skill' }).element();

		switchElement.focus();
		switchElement.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key }));

		await vi.waitFor(() => expect(bodyText()).toContain('Disabled'));
		expect(screen.getByTestId('skill-detail').query()).toBeNull();
		expect(bodyText()).not.toContain('Content of demo-skill');
	});
});

describe('SkillCatalogList manual-only badge', () => {
	it('renders a Manual only badge for flagged entries', async () => {
		const screen = await render(SkillCatalogList, {
			props: {
				entries: [makeEntry('manual', { disable_model_invocation: true })],
				selectedId: null,
				open: false,
				onSelect: vi.fn()
			}
		});

		await expect.element(screen.getByText('Manual only')).toBeInTheDocument();
	});

	it('omits the badge for model-visible entries', async () => {
		await render(SkillCatalogList, {
			props: {
				entries: [
					makeEntry('normal'),
					makeEntry('legacy', { disable_model_invocation: false }),
					makeEntry('older', { disable_model_invocation: undefined })
				],
				selectedId: null,
				open: false,
				onSelect: vi.fn()
			}
		});

		expect(bodyText()).not.toContain('Manual only');
	});
});

describe('SkillCatalogList availability switch', () => {
	it('shows both Manual only and Disabled badges while keeping the card readable and selectable', async () => {
		const screen = await render(SkillCatalogList, {
			props: {
				entries: [makeEntry('gated', { disable_model_invocation: true })],
				selectedId: null,
				open: false,
				onSelect: vi.fn(),
				isDisabled: () => true,
				onEnabledChange: vi.fn()
			}
		});

		await expect.element(screen.getByText('Manual only')).toBeInTheDocument();
		await expect.element(screen.getByText('Disabled')).toBeInTheDocument();

		// Full legibility: description, instruction facts, and the switch all
		// remain, and the card itself is still selectable.
		expect(bodyText()).toContain('description of gated');
		expect(bodyText()).toContain('4 tokens');
		expect(bodyText()).toContain('1 lines');
		expect(bodyText()).toContain('16 bytes');
		await expect.element(screen.getByRole('switch', { name: 'Enable gated' })).toBeInTheDocument();
		expect(screen.getByRole('button', { name: /gated/ }).query()).not.toBeNull();
	});
});
