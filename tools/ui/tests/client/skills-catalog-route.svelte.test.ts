// Guards the read-only /skills route presentation: distinct loading / error /
// unavailable / empty / success states, safe catalog fields only (no host
// paths, opaque XML never rendered), estimate labels, truncated resource
// lower bounds, the zero-budget note (distinct from a server-empty catalog),
// retry wiring, and persisted maxSkillBudget validation through the settings
// store.

import SkillsPage from '../../src/routes/skills/+page.svelte';
import { CONFIG_LOCALSTORAGE_KEY, SETTINGS_KEYS } from '$lib/constants';
import { skillsStore } from '$lib/stores/skills.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import type { SkillCatalogEntry, SkillCatalogResponse } from '$lib/types';
import { beforeEach, describe, expect, it, vi } from 'vitest';
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
		// Truncated resource listing renders as a lower bound.
		expect(text).toContain('Resources: 3+');
		// A complete resource listing renders the exact count.
		expect(text).toContain('Resources: 2');
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
		expect(bodyText()).toContain('Budget is 0');
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
});
