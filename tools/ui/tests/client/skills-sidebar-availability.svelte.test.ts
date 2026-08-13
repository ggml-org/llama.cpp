// Guards the startup Skills navigation gate: the sidebar keeps the Skills
// entry hidden while availability is unknown/loading, shows it after a
// successful catalog probe, keeps it hidden for a confirmed 404 (disabled),
// and keeps it visible for every other failure so the route stays reachable
// with retry. The probe uses the active conversation CWD, then the pending
// CWD, then no header. Ordinary sidebar actions never reorder.
//
// The one-request contract (sidebar probe + route initial load share a single
// catalog request per CWD, while Retry forces a new one) is guarded in the
// unit refresh-controller suite, which drives the same store entry.

import SidebarNavigationActions from '../../src/lib/components/app/navigation/SidebarNavigation/SidebarNavigationActions.svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { skillsStore } from '$lib/stores/skills.svelte';
import type { SkillCatalogEntry, SkillCatalogResponse } from '$lib/types';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		headers: { 'content-type': 'application/json' },
		status
	});
}

function makeEntry(name: string): SkillCatalogEntry {
	return {
		id: `opaque-${name}`,
		name,
		description: `description of ${name}`,
		scope: 'project',
		provider: 'agents',
		instruction: { bytes: 16, lines: 1, tokens: 4, tokens_estimated: true, modified_at: null },
		resources: { count: 0, truncated: false },
		catalog_xml: `<skill><name>${name}</name></skill>`
	};
}

function makeCatalog(...names: string[]): SkillCatalogResponse {
	return {
		skills: names.map(makeEntry),
		catalog_instruction_xml: '<available_skills>Call read_skill(name) when matching.</available_skills>',
		diagnostics: []
	};
}

/** Item tooltips in DOM order; each sidebar action renders one `span.truncate`. */
function visibleItemOrder(): string[] {
	return Array.from(document.querySelectorAll('span.truncate')).map(
		(span) => span.textContent?.trim() ?? ''
	);
}

const BASE_ACTIONS = ['New chat', 'Search', 'MCP Servers', 'Settings'] as const;

describe('Skills sidebar availability', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		conversationsStore.activeConversation = null;
		conversationsStore.pendingCwd = null;
		skillsStore.invalidate(undefined);
	});

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
		vi.mocked(fetch).mockImplementation(
			async () => jsonResponse({ error: { code: 404, message: 'no skills route' } }, 404)
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
		vi.mocked(fetch).mockImplementation(
			async () => jsonResponse({ error: { code: 503, message: 'unavailable' } }, 503)
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

		await render(SidebarNavigationActions, { class: 'w-64', isExpandedMode: true, isSearchModeActive: false, searchQuery: '' });

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

		await render(SidebarNavigationActions, { class: 'w-64', isExpandedMode: true, isSearchModeActive: false, searchQuery: '' });

		await vi.waitFor(() => expect(skillsStore.availability).toBe('available'));
		expect(calls[0].headers['x-skill-cwd']).toBe('/pending');
	});

	it('omits the CWD header when no working directory is known', async () => {
		const calls: Array<{ headers: Record<string, string> }> = [];

		vi.mocked(fetch).mockImplementation(async (_input, init) => {
			calls.push({ headers: (init?.headers ?? {}) as Record<string, string> });

			return jsonResponse(makeCatalog('demo'));
		});

		await render(SidebarNavigationActions, { class: 'w-64', isExpandedMode: true, isSearchModeActive: false, searchQuery: '' });

		await vi.waitFor(() => expect(skillsStore.availability).toBe('available'));
		expect('x-skill-cwd' in calls[0].headers).toBe(false);
	});
});
