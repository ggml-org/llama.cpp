import { beforeEach, describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { McpServerForm } from '$lib/components/app/mcp';
import { mcpStore } from '$lib/stores/mcp.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';

describe('mcp server display name', () => {
	beforeEach(() => {
		settingsStore.updateConfig('mcpServers', '[]');
	});

	it('custom display name wins over the url fallback', () => {
		const server = mcpStore.addServer({
			enabled: false,
			url: 'https://mcp.example.com/a',
			displayName: 'My Tools'
		});
		expect(mcpStore.getServerLabel(server)).toBe('My Tools');
	});

	it('without a custom name the url is the label', () => {
		const server = mcpStore.addServer({ enabled: false, url: 'https://mcp.example.com/a' });
		expect(mcpStore.getServerLabel(server)).toBe('https://mcp.example.com/a');
	});

	it('identical labels get positional suffixes', () => {
		const a = mcpStore.addServer({
			enabled: false,
			url: 'https://mcp.example.com/a',
			displayName: 'GitHub'
		});
		const b = mcpStore.addServer({
			enabled: false,
			url: 'https://mcp.example.com/b',
			displayName: 'GitHub'
		});
		expect(mcpStore.getServerLabel(a)).toBe('GitHub (1)');
		expect(mcpStore.getServerLabel(b)).toBe('GitHub (2)');
	});

	it('renaming one twin dissolves the suffixes', () => {
		const a = mcpStore.addServer({
			enabled: false,
			url: 'https://mcp.example.com/a',
			displayName: 'GitHub'
		});
		const b = mcpStore.addServer({
			enabled: false,
			url: 'https://mcp.example.com/b',
			displayName: 'GitHub'
		});
		mcpStore.updateServer(b.id, { displayName: 'GitHub Work' });
		expect(mcpStore.getServerLabel(a)).toBe('GitHub');
		expect(mcpStore.getServerLabel(mcpStore.getServerById(b.id)!)).toBe('GitHub Work');
	});

	it('the form exposes an editable display name field', async () => {
		let captured = '';
		const screen = await render(McpServerForm, {
			url: 'https://mcp.example.com/a',
			headers: '',
			name: '',
			onUrlChange: () => {},
			onHeadersChange: () => {},
			onNameChange: (v: string) => (captured = v)
		});

		const input = screen.getByLabelText('Display name');
		await expect.element(input).toBeVisible();
		await input.fill('My Custom Server');
		expect(captured).toBe('My Custom Server');
	});
});
