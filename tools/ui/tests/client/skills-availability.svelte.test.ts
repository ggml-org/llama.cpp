// Guards Skills navigation, availability probing, and settings-tool visibility.

import SidebarNavigationActions from '../../src/lib/components/app/navigation/SidebarNavigation/SidebarNavigationActions.svelte';
import SettingsChatToolsTab from '../../src/lib/components/app/settings/SettingsChat/SettingsChatToolsTab.svelte';
import {
	DISABLED_SKILL_IDS_LOCALSTORAGE_KEY,
	DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY,
	SKILL_LIST_TOOL,
	SKILL_READ_TOOL,
	SKILL_SERVER_LABEL
} from '$lib/constants';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { SkillAvailabilityStore } from '$lib/stores/skill-availability.svelte';
import { skillsStore } from '$lib/stores/skills.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import type { SkillCatalogEntry, SkillCatalogResponse } from '$lib/types';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

// localStorage polyfill for SkillAvailabilityStore unit tests (Node has no DOM).
const storageFixture = vi.hoisted(() => {
	const state = new Map<string, string>();
	const writes = { count: 0 };

	const polyfill: Storage = {
		clear: () => state.clear(),
		getItem: (key) => state.get(key) ?? null,
		key: (index) => [...state.keys()][index] ?? null,
		get length() {
			return state.size;
		},
		removeItem: (key) => {
			state.delete(key);
		},
		setItem: (key, value) => {
			writes.count += 1;
			state.set(key, String(value));
		}
	};

	return { state, writes, polyfill };
});

vi.mock('$app/environment', () => ({ browser: true }));

const nodeGlobal = globalThis as unknown as { localStorage: Storage };
nodeGlobal.localStorage = storageFixture.polyfill;

function storeCatalogEntry(id: string, name: string): SkillCatalogEntry {
	return {
		id,
		name,
		description: `description for ${name}`,
		scope: 'global',
		provider: 'test-provider',
		instruction: {
			bytes: 10,
			lines: 1,
			tokens: 4,
			tokens_estimated: false,
			modified_at: null
		},
		resources: { count: 0, truncated: false },
		catalog_xml: `<skills_catalog><skill name="${name}"/></skills_catalog>`
	};
}

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		headers: { 'content-type': 'application/json' },
		status
	});
}

function makeEntry(name: string): SkillCatalogEntry {
	return {
		catalog_xml: `<skill><name>${name}</name></skill>`,
		description: `description of ${name}`,
		id: `opaque-${name}`,
		instruction: { bytes: 16, lines: 1, modified_at: null, tokens: 4, tokens_estimated: true },
		name,
		provider: 'agents',
		resources: { count: 0, truncated: false },
		scope: 'project'
	};
}

function makeCatalog(...names: string[]): SkillCatalogResponse {
	return {
		catalog_instruction_xml:
			'<available_skills>Call read_skill(name) when matching.</available_skills>',
		diagnostics: [],
		skills: names.map(makeEntry)
	};
}

/** Item tooltips in DOM order; each sidebar action renders one `span.truncate`. */
function visibleItemOrder(): string[] {
	return Array.from(document.querySelectorAll('span.truncate')).map(
		(span) => span.textContent?.trim() ?? ''
	);
}

const BASE_ACTIONS = ['New chat', 'Search', 'MCP Servers', 'Settings'] as const;

/** Built-in listing shape consumed by `ToolsService.list`. */
function makeReadFileListing() {
	return [
		{
			definition: {
				function: {
					name: 'read_file',
					parameters: { properties: {}, type: 'object' }
				},
				type: 'function'
			},
			display_name: 'Read File',
			permissions: { write: false },
			tool: 'read_file',
			type: 'builtin',
			uses_cwd: false
		}
	];
}

interface ApiRoutes {
	skills?: (url: string, init?: RequestInit) => Response | Promise<Response>;
	tools?: (url: string, init?: RequestInit) => Response | Promise<Response>;
}

/** URL-routed fetch mock; counts `/skills` requests for the no-request guard. */
function mockApi(routes: ApiRoutes = {}): () => number {
	let skillsCalls = 0;

	vi.mocked(fetch).mockImplementation(async (input, init) => {
		const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;

		if (url.includes('/skills')) {
			skillsCalls += 1;

			return (routes.skills ?? (() => jsonResponse(makeCatalog())))(url, init);
		}

		if (url.includes('/tools')) {
			return (routes.tools ?? (() => jsonResponse([])))(url, init);
		}

		return jsonResponse({}, 404);
	});

	return () => skillsCalls;
}

beforeEach(() => {
	storageFixture.state.clear();
	storageFixture.writes.count = 0;
});

afterEach(() => {
	vi.restoreAllMocks();
	conversationsStore.activeConversation = null;
	conversationsStore.pendingCwd = null;
	skillsStore.invalidate(undefined);
});

// ---------------------------------------------------------------------------
// SkillAvailabilityStore unit tests (ported from skill-availability-store.test)
// ---------------------------------------------------------------------------

describe('SkillAvailabilityStore persistence', () => {
	it('loads a valid stored string array of opaque IDs into the disabled set', () => {
		storageFixture.state.set(
			DISABLED_SKILL_IDS_LOCALSTORAGE_KEY,
			JSON.stringify(['entry-a', 'entry-c'])
		);

		const store = new SkillAvailabilityStore();

		expect(store.isDisabled('entry-a')).toBe(true);
		expect(store.isDisabled('entry-c')).toBe(true);
		expect(store.isDisabled('entry-b')).toBe(false);
		expect([...store.disabledIds]).toEqual(['entry-a', 'entry-c']);
	});

	it('falls back to an empty set on malformed JSON, matching persisted() semantics', () => {
		storageFixture.state.set(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY, '{not-json');

		const store = new SkillAvailabilityStore();

		expect(store.disabledIds.size).toBe(0);
		expect(store.isDisabled('entry-a')).toBe(false);
	});

	it('normalizes non-array values and arrays containing non-strings to an empty set', () => {
		const badShapes = ['{"a":1}', '[1, 2]', '[null]', '7', '"entry-a"'];

		for (const serialized of badShapes) {
			storageFixture.state.set(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY, serialized);

			const store = new SkillAvailabilityStore();

			expect(store.disabledIds.size, `shape ${serialized}`).toBe(0);
			expect(store.isDisabled('entry-a'), `shape ${serialized}`).toBe(false);
		}
	});

	it('disabling and re-enabling persist the opaque ID, never the name/scope/provider/path', () => {
		const store = new SkillAvailabilityStore();

		store.setEnabled('opaque-id-1', false);

		expect(store.isDisabled('opaque-id-1')).toBe(true);
		expect(
			JSON.parse(storageFixture.state.get(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY) ?? '[]')
		).toEqual(['opaque-id-1']);

		store.setEnabled('opaque-id-1', true);

		expect(store.isDisabled('opaque-id-1')).toBe(false);
		expect(
			JSON.parse(storageFixture.state.get(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY) ?? '[]')
		).toEqual([]);
	});

	it('keeps same-name entries with different IDs independent', () => {
		const store = new SkillAvailabilityStore();

		store.setEnabled('id-shared-a', false);
		store.setEnabled('id-shared-b', false);
		store.setEnabled('id-shared-a', true);

		expect(store.isDisabled('id-shared-a')).toBe(false);
		expect(store.isDisabled('id-shared-b')).toBe(true);
		expect(
			JSON.parse(storageFixture.state.get(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY) ?? '[]')
		).toEqual(['id-shared-b']);
	});

	it('leaves unknown stored IDs harmlessly persisted', () => {
		storageFixture.state.set(
			DISABLED_SKILL_IDS_LOCALSTORAGE_KEY,
			JSON.stringify(['known-id', 'mystery-id'])
		);

		const store = new SkillAvailabilityStore();

		store.setEnabled('known-id', true);
		expect(
			JSON.parse(storageFixture.state.get(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY) ?? '[]')
		).toEqual(['mystery-id']);

		store.setEnabled('other-id', false);
		expect(
			JSON.parse(storageFixture.state.get(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY) ?? '[]')
		).toEqual(['mystery-id', 'other-id']);
	});
});

describe('SkillAvailabilityStore write discipline', () => {
	it('does not rewrite localStorage on repeated no-op setEnabled calls', () => {
		const store = new SkillAvailabilityStore();

		store.setEnabled('entry-a', false);
		expect(storageFixture.writes.count).toBe(1);
		expect(store.isDisabled('entry-a')).toBe(true);

		store.setEnabled('entry-a', false);
		expect(storageFixture.writes.count).toBe(1);

		store.setEnabled('entry-a', true);
		expect(storageFixture.writes.count).toBe(2);
		expect(store.isDisabled('entry-a')).toBe(false);

		store.setEnabled('entry-a', true);
		expect(storageFixture.writes.count).toBe(2);
	});

	it('does not write when enabling an id that was never disabled', () => {
		const store = new SkillAvailabilityStore();

		store.setEnabled('never-disabled', true);

		expect(storageFixture.writes.count).toBe(0);
		expect(storageFixture.state.has(DISABLED_SKILL_IDS_LOCALSTORAGE_KEY)).toBe(false);
	});
});

// ---------------------------------------------------------------------------
// Component-level availability tests
// ---------------------------------------------------------------------------

describe('SidebarNavigationActions Skills entry', () => {
	it('hides the Skills entry while availability is unknown/loading and shows it after a successful probe', async () => {
		let resolveCatalog!: (response: Response) => void;

		vi.mocked(fetch).mockImplementation(() => {
			const { promise, resolve } = Promise.withResolvers<Response>();

			resolveCatalog = resolve;

			return promise;
		});

		const screen = await render(SidebarNavigationActions, {
			class: 'w-64',
			isExpandedMode: true,
			isSearchModeActive: false,
			searchQuery: ''
		});

		// unknown/loading: the entry is filtered out.
		expect(screen.getByText('Skills').query()).toBeNull();
		expect(visibleItemOrder()).toEqual([...BASE_ACTIONS]);

		resolveCatalog(jsonResponse(makeCatalog('demo')));

		await vi.waitFor(() => expect(screen.getByText('Skills').query()).toBeTruthy());
		expect(visibleItemOrder()).toEqual(['New chat', 'Search', 'MCP Servers', 'Skills', 'Settings']);
	});

	it('keeps the Skills entry hidden when the probe confirms a 404 (disabled)', async () => {
		vi.mocked(fetch).mockImplementation(async () =>
			jsonResponse({ error: { code: 404, message: 'no skills route' } }, 404)
		);

		const screen = await render(SidebarNavigationActions, {
			class: 'w-64',
			isExpandedMode: true,
			isSearchModeActive: false,
			searchQuery: ''
		});

		await vi.waitFor(() => expect(skillsStore.availability).toBe('disabled'));

		expect(screen.getByText('Skills').query()).toBeNull();
		expect(visibleItemOrder()).toEqual([...BASE_ACTIONS]);
	});

	it('keeps the Skills entry visible when the probe fails with a non-404 error', async () => {
		vi.mocked(fetch).mockImplementation(async () =>
			jsonResponse({ error: { code: 503, message: 'unavailable' } }, 503)
		);

		const screen = await render(SidebarNavigationActions, {
			class: 'w-64',
			isExpandedMode: true,
			isSearchModeActive: false,
			searchQuery: ''
		});

		await vi.waitFor(() => expect(skillsStore.availability).toBe('error'));

		expect(screen.getByText('Skills').query()).toBeTruthy();
		expect(visibleItemOrder()).toEqual(['New chat', 'Search', 'MCP Servers', 'Skills', 'Settings']);
	});

	it('probes with the active conversation CWD', async () => {
		const calls: Array<{ headers: Record<string, string> }> = [];

		vi.mocked(fetch).mockImplementation(async (_input, init) => {
			calls.push({ headers: (init?.headers ?? {}) as Record<string, string> });

			return jsonResponse(makeCatalog('demo'));
		});

		conversationsStore.activeConversation = {
			currNode: null,
			cwd: '/proj/a',
			id: 'conv-1',
			lastModified: 0,
			name: 'Demo'
		};

		await render(SidebarNavigationActions, {
			class: 'w-64',
			isExpandedMode: true,
			isSearchModeActive: false,
			searchQuery: ''
		});

		await vi.waitFor(() => expect(skillsStore.availability).toBe('available'));
		expect(calls[0].headers['x-skill-cwd']).toBe('/proj/a');
	});

	it('falls back to the pending CWD when no conversation is active', async () => {
		const calls: Array<{ headers: Record<string, string> }> = [];

		vi.mocked(fetch).mockImplementation(async (_input, init) => {
			calls.push({ headers: (init?.headers ?? {}) as Record<string, string> });

			return jsonResponse(makeCatalog('demo'));
		});

		conversationsStore.pendingCwd = '/pending';

		await render(SidebarNavigationActions, {
			class: 'w-64',
			isExpandedMode: true,
			isSearchModeActive: false,
			searchQuery: ''
		});

		await vi.waitFor(() => expect(skillsStore.availability).toBe('available'));
		expect(calls[0].headers['x-skill-cwd']).toBe('/pending');
	});

	it('omits the CWD header when no working directory is known', async () => {
		const calls: Array<{ headers: Record<string, string> }> = [];

		vi.mocked(fetch).mockImplementation(async (_input, init) => {
			calls.push({ headers: (init?.headers ?? {}) as Record<string, string> });

			return jsonResponse(makeCatalog('demo'));
		});

		await render(SidebarNavigationActions, {
			class: 'w-64',
			isExpandedMode: true,
			isSearchModeActive: false,
			searchQuery: ''
		});

		await vi.waitFor(() => expect(skillsStore.availability).toBe('available'));
		expect('x-skill-cwd' in calls[0].headers).toBe(false);
	});
});

describe('SettingsChatToolsTab Skills group', () => {
	beforeEach(() => {
		localStorage.removeItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY);
		toolsStore.setToolEnabled('skill:read_skill', true);
		toolsStore.setToolEnabled('skill:list_skill', true);
	});

	it('renders both Skills toggles from the centralized registry with independent persistence', async () => {
		mockApi({ skills: () => jsonResponse(makeCatalog()) });
		await skillsStore.probeAvailability(undefined);

		const screen = await render(SettingsChatToolsTab);

		// One group (no generic tools served): the Skills group.
		expect(document.querySelectorAll('[data-slot="collapsible"]')).toHaveLength(1);
		expect(screen.getByText(SKILL_SERVER_LABEL, { exact: true }).query()).toBeTruthy();
		expect(screen.getByText('2 tools').query()).toBeTruthy();

		// Skills stay out of the generic tool collections consumed by chat.
		expect(toolsStore.allTools.some((entry) => toolsStore.isSkillToolKey(entry.key))).toBe(false);
		expect(toolsStore.toolGroups.some((group) => group.source === 'skills')).toBe(false);

		await screen.getByRole('button', { name: /Skills/ }).click();
		await vi.waitFor(() => expect(screen.getByText('Read skill').query()).toBeTruthy());
		expect(screen.getByText('List skills').query()).toBeTruthy();

		const readCheckbox = screen.getByRole('checkbox', { name: 'Enable Read skill' });
		const listCheckbox = screen.getByRole('checkbox', { name: 'Enable List skills' });

		await expect.element(readCheckbox).toHaveAttribute('data-state', 'checked');
		await expect.element(listCheckbox).toHaveAttribute('data-state', 'checked');

		// Independent toggle: disabling read_skill leaves list_skill enabled and
		// persists the stable `skill:` key through the store.
		await readCheckbox.click();

		expect(toolsStore.isToolEnabled('skill:read_skill')).toBe(false);
		expect(toolsStore.isToolEnabled('skill:list_skill')).toBe(true);
		expect([...toolsStore.getEnabledSkillToolNames()]).toEqual([SKILL_LIST_TOOL]);
		expect(JSON.parse(localStorage.getItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY) ?? '[]')).toEqual([
			'skill:read_skill'
		]);

		await expect.element(readCheckbox).toHaveAttribute('data-state', 'unchecked');
		await expect.element(listCheckbox).toHaveAttribute('data-state', 'checked');

		// Re-enabling restores both.
		await readCheckbox.click();

		expect(toolsStore.isToolEnabled('skill:read_skill')).toBe(true);
		expect([...toolsStore.getEnabledSkillToolNames()]).toEqual([SKILL_READ_TOOL, SKILL_LIST_TOOL]);
	});

	it('issues no catalog request from the settings tab', async () => {
		const skillsCalls = mockApi({ skills: () => jsonResponse(makeCatalog()) });

		// The startup probe is the only owner of the catalog request.
		await skillsStore.probeAvailability(undefined);

		expect(skillsCalls()).toBe(1);

		await render(SettingsChatToolsTab);

		expect(skillsCalls()).toBe(1);
	});

	it('shows the Skills group during loading', async () => {
		let resolveCatalog!: (response: Response) => void;

		const skillsCalls = mockApi({
			skills: () =>
				new Promise<Response>((resolve) => {
					resolveCatalog = resolve;
				})
		});
		// Probe stays pending: availability is `loading`.
		const probe = skillsStore.probeAvailability(undefined);

		expect(skillsCalls()).toBe(1);

		const screen = await render(SettingsChatToolsTab);

		expect(screen.getByText(SKILL_SERVER_LABEL, { exact: true }).query()).toBeTruthy();

		resolveCatalog(jsonResponse(makeCatalog()));
		await probe;
	});

	it('shows the Skills group for a retryable availability error', async () => {
		mockApi({
			skills: () => jsonResponse({ error: { code: 503, message: 'unavailable' } }, 503)
		});
		await skillsStore.probeAvailability(undefined);

		expect(skillsStore.availability).toBe('error');

		const screen = await render(SettingsChatToolsTab);

		expect(screen.getByText(SKILL_SERVER_LABEL, { exact: true }).query()).toBeTruthy();
	});

	it('hides the Skills group only for a confirmed disabled availability', async () => {
		mockApi({
			skills: () => jsonResponse({ error: { code: 404, message: 'no skills route' } }, 404)
		});
		await skillsStore.probeAvailability(undefined);

		expect(skillsStore.availability).toBe('disabled');

		const screen = await render(SettingsChatToolsTab);

		expect(screen.getByText(SKILL_SERVER_LABEL).query()).toBeNull();
		expect(screen.getByText('No tools available').query()).toBeTruthy();
		expect(document.querySelectorAll('[data-slot="collapsible"]')).toHaveLength(0);
	});

	it('keeps generic groups and their Always allow controls while Skills rows get none', async () => {
		mockApi({
			skills: () => jsonResponse(makeCatalog()),
			tools: () => jsonResponse(makeReadFileListing())
		});
		await toolsStore.fetchBuiltinTools();
		await skillsStore.probeAvailability(undefined);

		const screen = await render(SettingsChatToolsTab);

		// Both groups present: generic Built-in first, Skills appended after.
		await screen.getByRole('button', { name: /Built-in/ }).click();
		await screen.getByRole('button', { name: /Skills/ }).click();
		await vi.waitFor(() => expect(screen.getByText('Read file').query()).toBeTruthy());

		// Generic row keeps the full Enabled + Always allow flow.
		const contents = document.querySelectorAll('[data-slot="collapsible-content"]');

		expect(contents[0]?.querySelectorAll('[data-slot="checkbox"]').length ?? 0).toBe(2);

		// Skills rows: exactly the two Enabled toggles, no Always allow control,
		// and the per-skill consent note.
		expect(contents[1]?.querySelectorAll('[data-slot="checkbox"]').length ?? 0).toBe(2);
		expect(document.querySelectorAll('[title*="per resolved skill identity"]').length).toBe(2);
		const skillCheckboxes = [...document.querySelectorAll<HTMLElement>('[role="checkbox"]')].filter(
			(el) => /^Enable (Read skill|List skills)$/.test(el.getAttribute('aria-label') ?? '')
		);

		expect(skillCheckboxes).toHaveLength(2);

		// The generic permission flow still resolves read_file.
		expect(toolsStore.getPermissionKey('read_file')).toBe('builtin:read_file');
	});
});
