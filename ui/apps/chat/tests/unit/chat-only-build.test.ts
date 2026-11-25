// A chat-only build (VITE_PUBLIC_AGENTIC=false) offers no tools, no MCP and no
// agentic settings, and a full build offers them only for a server that has
// them. The flag is read when the constants load, so each case loads them fresh.

import { SidebarAction } from '#lib/enums/index.ts';
import { afterEach, describe, expect, it, vi } from 'vitest';

async function loadConstants(agentic?: string) {
	vi.resetModules();

	if (agentic !== undefined) vi.stubEnv('VITE_PUBLIC_AGENTIC', agentic);

	return import('#lib/constants/index.ts');
}

afterEach(() => {
	vi.unstubAllEnvs();
	vi.resetModules();
});

describe('chat-only build', () => {
	it('is off unless the build sets the flag to false', async () => {
		expect((await loadConstants()).AGENTIC_ENABLED).toBe(true);
		expect((await loadConstants('true')).AGENTIC_ENABLED).toBe(true);
		expect((await loadConstants('false')).AGENTIC_ENABLED).toBe(false);
	});

	it('gives the tools store no tools', async () => {
		vi.resetModules();
		vi.stubEnv('VITE_PUBLIC_AGENTIC', 'false');

		const { agenticMode } = await import('#lib/stores/agentic-mode.svelte.ts');
		const { toolsStore } = await import('#lib/stores/tools.svelte.ts');

		// even for a server that offers its tools
		agenticMode.serverHasTools = true;

		expect(agenticMode.enabled).toBe(false);
		expect(toolsStore.allTools).toEqual([]);
		expect(toolsStore.getEnabledToolsForLLM()).toEqual([]);
	});
});

describe('agentic mode', () => {
	it('is on only for a server that offers its tools or its proxy', async () => {
		vi.resetModules();

		const { agenticMode } = await import('#lib/stores/agentic-mode.svelte.ts');
		const { serverStore } = await import('#lib/stores/llama-server.svelte.ts');
		const { toolsStore } = await import('#lib/stores/tools.svelte.ts');

		expect(agenticMode.enabled).toBe(false);
		expect(toolsStore.allTools).toEqual([]);

		agenticMode.serverHasTools = true;

		expect(agenticMode.enabled).toBe(true);
		expect(toolsStore.allTools.length).toBeGreaterThan(0);

		agenticMode.serverHasTools = false;
		serverStore.props = { cors_proxy_enabled: true } as typeof serverStore.props;

		expect(agenticMode.enabled).toBe(true);
	});

	it('is on with the browser tools alone for a server started with --ui-agent', async () => {
		vi.resetModules();

		const { agenticMode } = await import('#lib/stores/agentic-mode.svelte.ts');
		const { ToolSource } = await import('#lib/enums/index.ts');
		const { serverStore } = await import('#lib/stores/llama-server.svelte.ts');
		const { toolsStore } = await import('#lib/stores/tools.svelte.ts');

		serverStore.props = { ui_agent: true } as typeof serverStore.props;

		expect(agenticMode.enabled).toBe(true);
		expect(toolsStore.allTools.length).toBeGreaterThan(0);
		expect(toolsStore.allTools.every((tool) => tool.source === ToolSource.BROWSER)).toBe(true);
	});

	it('leaves the Tools and Agentic settings and the MCP entry out when it is off', async () => {
		const c = await loadConstants();
		const slugs = (agentic: boolean) => c.settingsChatSections(agentic).map((s) => s.slug);
		const keys = (agentic: boolean) =>
			c.settingsChatSections(agentic).flatMap((s) => (s.fields ?? []).map((field) => field.key));
		const actions = (agentic: boolean) => c.sidebarFooterItems(agentic).map((item) => item.action);
		const { AGENTIC, TOOLS } = c.SETTINGS_SECTION_SLUGS;

		expect(slugs(true)).toEqual(expect.arrayContaining([TOOLS, AGENTIC]));
		expect(slugs(false)).not.toContain(TOOLS);
		expect(slugs(false)).not.toContain(AGENTIC);
		expect(slugs(false).length).toBeGreaterThan(0);

		expect(keys(true)).toContain(c.SETTINGS_KEYS.SHOW_MCP);
		expect(keys(false)).not.toContain(c.SETTINGS_KEYS.SHOW_MCP);

		expect(actions(true)).toContain(SidebarAction.MCP);
		expect(actions(false)).not.toContain(SidebarAction.MCP);

		expect(c.attachmentTooltipText(false)).not.toMatch(/tools|MCP/i);
	});
});
