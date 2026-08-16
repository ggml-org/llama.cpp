// Guards the Skills-only group in the Chat tool settings tab: both
// model-facing adapters render as independent persisted Enabled toggles from
// the centralized Skills settings registry; the group is gated on availability
// (shown for `available`, `loading`, and retryable `error`, hidden only for a
// confirmed 404 `disabled`); the tab issues no catalog request of its own; and
// no generic Always allow control appears for Skills rows - Skills consent
// stays per resolved skill identity during execution. Generic groups keep
// their existing Enabled/Always allow flow untouched.

import SettingsChatToolsTab from '../../src/lib/components/app/settings/SettingsChat/SettingsChatToolsTab.svelte';
import {
	DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY,
	SKILL_LIST_TOOL,
	SKILL_READ_TOOL,
	SKILL_SERVER_LABEL
} from '$lib/constants';
import { skillsStore } from '$lib/stores/skills.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import type { SkillCatalogResponse } from '$lib/types';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		headers: { 'content-type': 'application/json' },
		status
	});
}

function makeCatalog(): SkillCatalogResponse {
	return {
		catalog_instruction_xml: '<available_skills/>',
		diagnostics: [],
		skills: []
	};
}

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

describe('SettingsChatToolsTab Skills group', () => {
	beforeEach(() => {
		localStorage.removeItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY);
		toolsStore.setToolEnabled('skill:read_skill', true);
		toolsStore.setToolEnabled('skill:list_skill', true);
	});

	afterEach(() => {
		vi.restoreAllMocks();
		skillsStore.invalidate(undefined);
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
