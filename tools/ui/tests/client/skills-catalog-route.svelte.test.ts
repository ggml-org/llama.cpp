// Guards read-only `/skills` states, safe fields, packing, retry, and settings.

import SkillsPage from '../../src/routes/skills/+page.svelte';
import SkillsPageWrapper from './components/SkillsPageWrapper.svelte';
import SkillCatalogList from '$lib/components/app/skills/SkillCatalogList.svelte';
import SkillDiagnosticsPanel from '$lib/components/app/skills/SkillDiagnosticsPanel.svelte';
import SkillCatalogSearchToolbar from '$lib/components/app/skills/SkillCatalogSearchToolbar.svelte';
import { CONFIG_LOCALSTORAGE_KEY } from '$lib/constants';
import { serializeSkillCatalogEnvelope } from '$lib/services/skills.service';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { modelsStore } from '$lib/stores/models.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { skillsStore } from '$lib/stores/skills.svelte';
import type { SkillCatalogEntry, SkillCatalogResponse, SkillDiagnostic } from '$lib/types';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { catalogOf, deferredRead, jsonResponse, makeCatalog, makeEntry } from '../fixtures/skills';

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

beforeEach(() => {
	localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
	settingsStore.initialize();
	skillsStore.invalidate(undefined);
	modelsStore.selectedModelName = null;
	conversationsStore.pendingCwd = null;
	vi.mocked(fetch).mockClear();
});

describe('/skills route states', () => {
	it('shows a loading state while the catalog request is in flight', async () => {
		vi.mocked(fetch).mockImplementation(() => new Promise(() => {}));

		render(SkillsPage);

		await vi.waitFor(() => expect(bodyText()).toContain('Loading catalog'));
	});

	it('renders only safe catalog fields and never opaque XML or host paths', async () => {
		const catalog = makeCatalog([
			makeEntry('demo-skill', {
				catalog_xml:
					'<skill><name>demo-skill</name><path>/srv/secret/skills/demo/SKILL.md</path></skill>',
				description: 'A skill that does things.',
				instruction: {
					bytes: 1024,
					lines: 42,
					modified_at: '2024-01-02T03:04:05Z',
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
		]);

		mockFetchOnce(catalog);
		render(SkillsPage);

		await vi.waitFor(() => expect(bodyText()).toContain('demo-skill'));

		const text = bodyText();

		// Provider mapping, estimated tokens, and instruction facts render.
		expect(text).toContain('A skill that does things.');
		expect(text).toContain('global');
		expect(text).toContain('generic');
		expect(text).not.toContain('agents');
		expect(text).toContain('~512 tokens');
		expect(text).toContain('42');
		expect(text).toContain('2024');
		// Truncated resources render a lower bound; complete ones the total.
		expect(text).toMatch(/Resources:\s*3\+/);
		expect(text).toMatch(/Resources:\s*2\b/);
		expect(text).toContain('2 tokens');
		expect(text).not.toContain('exact');
		// Opaque catalog XML and host paths never reach the DOM.
		expect(text).not.toContain('/srv/secret/skills/demo/SKILL.md');
		expect(text).not.toContain('<skill>');
		expect(text).not.toContain('catalog_instruction_xml');
	});

	it('renders diagnostics with duplicate-code rows and a collapsed shadowed providers list', async () => {
		const catalog: SkillCatalogResponse = {
			...catalogOf('demo-skill'),
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
				},
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

		await vi.waitFor(() => expect(bodyText()).toContain('first diagnostic message'));

		const text = bodyText();

		// Duplicate codes stay two independent rows: the code renders once per row.
		expect(text).toContain('Skill: Alpha Skill');
		expect(text).toContain('Skill: Beta Skill');
		expect(text).toContain('first diagnostic message');
		expect(text).toContain('second diagnostic message');
		expect(text.match(/overlapping-skill/g)).toHaveLength(2);
		// The shadowed diagnostic renders the singular first provider and the
		// collapsed full list exactly once.
		expect(text).toContain('Provider: claude');
		expect(text).toContain('Providers: claude, gemini, opencode');
		expect(text.match(/Providers:/g)).toHaveLength(1);
		expect(text.match(/skill_shadowed/g)).toHaveLength(1);
	});

	it.each([
		[
			'shows the empty state for a server-empty catalog',
			() => makeCatalog([]),
			200,
			(text: string) => expect(text).toContain('No skills found')
		],
		[
			'keeps a zero budget distinct from a server-empty catalog',
			() => {
				settingsStore.updateConfig('maxSkillBudget', 0);

				return catalogOf('demo-skill');
			},
			200,
			(text: string) => {
				expect(text).toContain('demo-skill');
				expect(text).toContain(
					'Skills tools are disabled because the catalog budget is 0 tokens'
				);
				expect(text).not.toContain('No skills found');
			}
		],
		[
			'shows the generic error state with a retry action',
			() => ({ error: { code: 503, message: 'catalog temporarily unavailable' } }),
			503,
			(text: string) => {
				expect(text).toContain('catalog temporarily unavailable');
				expect(text).toContain('Retry');
			}
		],
		[
			'distinguishes a missing skills route from a request error',
			() => 'Not Found',
			404,
			(text: string) => {
				expect(text).toContain('not enabled');
				expect(text).not.toContain('Retry');
			}
		]
	])('%s', async (_label, makeBody, status, assert) => {
		mockFetchOnce(makeBody(), status);
		render(SkillsPage);

		await vi.waitFor(() => assert(bodyText()));
	});
});

describe('/skills budget copy', () => {
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
		mockFetchOnce(catalogOf('demo-skill'));
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

			return jsonResponse(catalogOf('demo-skill', 'second-skill'));
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

	it('exposes the desktop splitter as a named resize control', async () => {
		mockFetchOnce(catalogOf('demo-skill'));
		const screen = await render(SkillsPageWrapper);

		await screen.getByRole('button', { name: /demo-skill/ }).click();

		await expect.element(screen.getByRole('separator')).toBeInTheDocument();
		await expect
			.element(screen.getByRole('separator', { name: 'Resize catalog and detail panels' }))
			.toBeInTheDocument();
	});
});

describe('SkillCatalogList badges', () => {
	it('renders the Manual only badge and keeps a Disabled card readable and selectable', async () => {
		const screen = await render(SkillCatalogList, {
			props: {
				entries: [makeEntry('gated', { disable_model_invocation: true })],
				isDisabled: () => true,
				onEnabledChange: vi.fn(),
				onSelect: vi.fn(),
				open: false,
				selectedId: null
			}
		});

		await expect.element(screen.getByText('Manual only')).toBeInTheDocument();
		await expect.element(screen.getByText('Disabled')).toBeInTheDocument();

		expect(bodyText()).toContain('description of gated');
		expect(bodyText()).toContain('4 tokens');
		expect(bodyText()).toContain('1 lines');
		expect(bodyText()).toContain('16 bytes');
		await expect.element(screen.getByRole('switch', { name: 'Enable gated' })).toBeInTheDocument();
		expect(screen.getByRole('button', { name: /gated/ }).query()).not.toBeNull();
	});
});

describe('SkillDiagnosticsPanel', () => {
	function warning(code: string, message: string): SkillDiagnostic {
		return { code, message, severity: 'warning' };
	}

	function error(code: string, message: string): SkillDiagnostic {
		return { code, message, severity: 'error' };
	}

	it('always renders errors and a single warning inline, with no collapse chip', async () => {
		render(SkillDiagnosticsPanel, {
			props: {
				budgetChip: null,
				diagnostics: [error('e1', 'error one'), warning('w1', 'warning one')],
				dismissed: false,
				onDismiss: vi.fn()
			}
		});

		const text = bodyText();

		expect(text).toContain('error one');
		expect(text).toContain('warning one');
		expect(text).not.toContain('warnings');
	});

	it('collapses 2+ warnings behind a summary chip, expandable to the full list', async () => {
		const screen = await render(SkillDiagnosticsPanel, {
			props: {
				budgetChip: null,
				diagnostics: [warning('w1', 'warning one'), warning('w2', 'warning two')],
				dismissed: false,
				onDismiss: vi.fn()
			}
		});

		expect(bodyText()).toContain('2 warnings');
		expect(bodyText()).not.toContain('warning one');

		await screen.getByTestId('skill-diagnostics-warnings-toggle').click();

		expect(bodyText()).toContain('warning one');
		expect(bodyText()).toContain('warning two');
	});

	it('renders the budget chip above errors and warnings, excluded from the warnings count', async () => {
		const screen = await render(SkillDiagnosticsPanel, {
			props: {
				budgetChip: { detail: 'exceeds budget detail text', label: 'Partial fit' },
				diagnostics: [
					error('e1', 'error one'),
					warning('w1', 'warning one'),
					warning('w2', 'warning two')
				],
				dismissed: false,
				onDismiss: vi.fn()
			}
		});

		const text = bodyText();
		const order = [text.indexOf('Partial fit'), text.indexOf('error one'), text.indexOf('2 warnings')];

		expect(order[0]).toBeGreaterThanOrEqual(0);
		expect(order[0]).toBeLessThan(order[1]);
		expect(order[1]).toBeLessThan(order[2]);
		await expect
			.element(screen.getByTitle('exceeds budget detail text'))
			.toBeInTheDocument();
	});

	it('renders nothing when dismissed, and calls onDismiss from the dismiss control', async () => {
		const onDismiss = vi.fn();
		const screen = await render(SkillDiagnosticsPanel, {
			props: { budgetChip: null, diagnostics: [error('e1', 'error one')], dismissed: false, onDismiss }
		});

		await screen.getByRole('button', { name: 'Dismiss diagnostics' }).click();
		expect(onDismiss).toHaveBeenCalledOnce();

		screen.unmount();

		await render(SkillDiagnosticsPanel, {
			props: { budgetChip: null, diagnostics: [error('e1', 'error one')], dismissed: true, onDismiss }
		});
		expect(bodyText()).not.toContain('error one');
describe('SkillCatalogSearchToolbar', () => {
	it('debounces search input before calling onQueryChange', async () => {
		vi.useFakeTimers();
		const onQueryChange = vi.fn();
		const screen = await render(SkillCatalogSearchToolbar, {
			props: {
				excludedProviders: new Set(),
				includeProject: true,
				onIncludeProjectChange: vi.fn(),
				onProvidersChange: vi.fn(),
				onQueryChange,
				providers: ['agents', 'local']
			}
		});

		await screen.getByLabelText('Search skills').fill('canvas');
		expect(onQueryChange).not.toHaveBeenCalled();

		vi.advanceTimersByTime(150);
		expect(onQueryChange).toHaveBeenCalledWith('canvas');
		vi.useRealTimers();
	});

	it('lists one checkbox per provider, toggling it via onProvidersChange', async () => {
		const onProvidersChange = vi.fn();
		const screen = await render(SkillCatalogSearchToolbar, {
			props: {
				excludedProviders: new Set(),
				includeProject: true,
				onIncludeProjectChange: vi.fn(),
				onProvidersChange,
				onQueryChange: vi.fn(),
				providers: ['agents', 'local']
			}
		});

		await screen.getByRole('button', { name: 'Filter skills' }).click();
		await screen.getByText('local').click();

		expect(onProvidersChange).toHaveBeenCalledWith(new Set(['local']));
	});

	it('shows the active-filter dot when a filter deviates from default, and Reset restores both facets', async () => {
		const onProvidersChange = vi.fn();
		const onIncludeProjectChange = vi.fn();
		const screen = await render(SkillCatalogSearchToolbar, {
			props: {
				excludedProviders: new Set(['local']),
				includeProject: true,
				onIncludeProjectChange,
				onProvidersChange,
				onQueryChange: vi.fn(),
				providers: ['local']
			}
		});

		await expect.element(screen.getByTestId('skill-filter-active-dot')).toBeInTheDocument();

		await screen.getByRole('button', { name: 'Filter skills' }).click();
		await screen.getByText('Include project skills').click();
		expect(onIncludeProjectChange).toHaveBeenCalledWith(false);

		await screen.getByRole('button', { name: 'Reset' }).click();
		expect(onProvidersChange).toHaveBeenCalledWith(new Set());
		expect(onIncludeProjectChange).toHaveBeenCalledWith(true);
	});
});
