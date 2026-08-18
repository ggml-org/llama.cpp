// Guards the shared page shell used by MCP Servers and Skills.

import McpServersPage from '../../src/routes/mcp-servers/+page.svelte';
import SkillsPage from '../../src/routes/skills/+page.svelte';
import { goto } from '$app/navigation';
import { ROUTES, SETTINGS_KEYS } from '$lib/constants';
import { mcpStore } from '$lib/stores/mcp.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { skillsStore } from '$lib/stores/skills.svelte';
import type { SkillCatalogEntry, SkillCatalogResponse } from '$lib/types';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

vi.mock('$app/navigation', async (importOriginal) => {
	const actual = await importOriginal<typeof import('$app/navigation')>();

	return {
		...actual,
		goto: vi.fn()
	};
});

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

function makeCatalog(...entries: SkillCatalogEntry[]): SkillCatalogResponse {
	return {
		catalog_instruction_xml: '',
		diagnostics: [],
		skills: entries
	};
}

describe('shared standalone page shell', () => {
	beforeEach(() => {
		localStorage.clear();
		settingsStore.initialize();
		settingsStore.updateConfig(SETTINGS_KEYS.MCP_SERVERS, '[]');
		skillsStore.invalidate(undefined);
		vi.mocked(fetch).mockImplementation(async () =>
			jsonResponse(makeCatalog(makeEntry('demo-skill')))
		);
		vi.mocked(goto).mockClear();
		// A fresh page uses the start route for Close.
		Object.defineProperty(window.history, 'length', { configurable: true, value: 1 });
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	it('renders the MCP Servers route with the shared title and a Close action', async () => {
		const mcp = await render(McpServersPage);

		await expect
			.element(mcp.getByRole('heading', { exact: true, name: 'MCP Servers' }))
			.toBeVisible();
		await expect
			.element(mcp.getByTestId('standalone-page-shell').getByRole('button', { name: 'Close' }))
			.toBeVisible();
	});

	it('renders the Skills route with the shared title and a Close action', async () => {
		const skills = await render(SkillsPage);

		await expect
			.element(skills.getByRole('heading', { exact: true, name: 'Skills' }))
			.toBeVisible();
		await expect
			.element(skills.getByTestId('standalone-page-shell').getByRole('button', { name: 'Close' }))
			.toBeVisible();
	});

	it('keeps the MCP Servers content under the shared heading', async () => {
		mcpStore.addServer({
			displayName: 'GitHub',
			enabled: false,
			url: 'https://mcp.example.com/github'
		});

		const mcp = await render(McpServersPage);

		await vi.waitFor(() => expect(document.body.textContent ?? '').toContain('GitHub'));
		await expect.element(mcp.getByRole('heading', { name: 'MCP Servers' })).toBeVisible();
		expect(document.body.textContent ?? '').toContain('Add another MCP server');
	});

	it('keeps the Skills catalog content under the shared heading', async () => {
		const skills = await render(SkillsPage);

		await vi.waitFor(() => expect(document.body.textContent ?? '').toContain('demo-skill'));
		await expect.element(skills.getByRole('heading', { name: 'Skills' })).toBeVisible();
		expect(document.body.textContent ?? '').toContain('description of demo-skill');
	});

	it('closes MCP Servers through the history fallback when a previous route exists', async () => {
		Object.defineProperty(window.history, 'length', { configurable: true, value: 2 });
		const back = vi.spyOn(History.prototype, 'back').mockImplementation(() => {});
		const mcp = await render(McpServersPage);

		await mcp.getByRole('button', { name: 'Close' }).click();

		expect(back).toHaveBeenCalledOnce();
		expect(goto).not.toHaveBeenCalled();
	});

	it('closes MCP Servers to the start route when history is shallow', async () => {
		const mcp = await render(McpServersPage);

		await mcp.getByRole('button', { name: 'Close' }).click();

		expect(goto).toHaveBeenCalledWith(ROUTES.START);
	});

	it('closes Skills through the history fallback when a previous route exists', async () => {
		Object.defineProperty(window.history, 'length', { configurable: true, value: 2 });
		const back = vi.spyOn(History.prototype, 'back').mockImplementation(() => {});
		const skills = await render(SkillsPage);

		await skills.getByRole('button', { name: 'Close' }).click();

		expect(back).toHaveBeenCalledOnce();
		expect(goto).not.toHaveBeenCalled();
	});

	it('closes Skills to the start route when history is shallow', async () => {
		const skills = await render(SkillsPage);

		await skills.getByRole('button', { name: 'Close' }).click();

		expect(goto).toHaveBeenCalledWith(ROUTES.START);
	});
});
