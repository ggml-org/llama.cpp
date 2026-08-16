import {
	DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY,
	SKILL_LIST_TOOL,
	SKILL_READ_TOOL,
	SKILL_SERVER_LABEL,
	SKILL_TOOL_SETTINGS
} from '$lib/constants';
import { ToolSource } from '$lib/enums';
import type { toolsStore as ToolsStoreValue } from '$lib/stores/tools.svelte';
import type { ToolEntry } from '$lib/types';
import { beforeEach, describe, expect, it, vi } from 'vitest';

// The node unit project has no DOM or localStorage; install a Map-backed
// Storage and force `browser: true` for this module graph so the constructor
// path (disabled-tool localStorage restore) is exercised. The viewport store
// reads window.innerWidth at module scope when browser is true, so it is
// stubbed, and the /tools fetch is replaced with a deterministic listing.
const storageState = vi.hoisted(() => new Map<string, string>());
const storagePolyfill = vi.hoisted(() => {
	const storage: Storage = {
		clear: () => storageState.clear(),
		getItem: (key) => storageState.get(key) ?? null,
		key: (index) => [...storageState.keys()][index] ?? null,
		get length() {
			return storageState.size;
		},
		removeItem: (key) => {
			storageState.delete(key);
		},
		setItem: (key, value) => {
			storageState.set(key, String(value));
		}
	};

	return storage;
});

vi.mock('$app/environment', () => ({ browser: true }));
vi.mock('$lib/stores/viewport.svelte', () => ({
	isMobile: { current: false },
	viewport: { width: 0 }
}));
vi.mock('$lib/services/tools.service', () => ({
	ToolsService: {
		executeToolRaw: vi.fn(),
		list: vi.fn().mockResolvedValue([
			{
				definition: {
					type: 'function',
					function: {
						name: 'read_file',
						parameters: { type: 'object', properties: {} }
					}
				},
				display_name: 'Read File',
				permissions: { write: false },
				tool: 'read_file',
				type: ToolSource.BUILTIN,
				uses_cwd: false
			}
		])
	}
}));

let toolsStore: typeof ToolsStoreValue;

beforeEach(async () => {
	storageState.clear();
	// node env lacks localStorage; reuse the polyfill for the browser path
	const nodeGlobal = globalThis as unknown as { localStorage: Storage };

	nodeGlobal.localStorage = storagePolyfill;
	// Static import cannot be used: a fresh store instance is needed per test
	// so the constructor re-reads the persisted disabled-tool set.
	vi.resetModules();
	({ toolsStore } = await import('$lib/stores/tools.svelte'));
});

describe('ToolsStore Skills settings group', () => {
	it('defaults both Skill settings to enabled', () => {
		expect([...toolsStore.getEnabledSkillToolNames()].sort()).toEqual([
			SKILL_LIST_TOOL,
			SKILL_READ_TOOL
		]);
		expect(toolsStore.isToolEnabled('skill:read_skill')).toBe(true);
		expect(toolsStore.isToolEnabled('skill:list_skill')).toBe(true);
	});

	it('builds a settings-only Skills group from the centralized registry', () => {
		expect(toolsStore.skillToolGroups).toHaveLength(1);

		const group = toolsStore.skillToolGroups[0];

		expect(group).toMatchObject({
			key: ToolSource.SKILLS,
			label: SKILL_SERVER_LABEL,
			source: ToolSource.SKILLS
		});
		expect(group.tools).toHaveLength(2);

		const byKey = new Map(group.tools.map((tool) => [tool.key, tool]));

		expect(byKey.get('skill:read_skill')).toMatchObject({
			key: 'skill:read_skill',
			source: ToolSource.SKILLS
		});
		expect(byKey.get('skill:list_skill')).toMatchObject({
			key: 'skill:list_skill',
			source: ToolSource.SKILLS
		});

		// The group reuses the centralized settings definitions unchanged
		// (deep-equal; `vi.resetModules()` creates a fresh registry instance
		// per test, so object identity cannot span the boundary).
		const expected = new Map(SKILL_TOOL_SETTINGS.map((setting) => [setting.key, setting.definition]));

		for (const tool of group.tools) {
			expect(tool.definition).toEqual(expected.get(tool.key));
		}
	});

	it('restores disabled Skill settings from persisted localStorage', async () => {
		storageState.set(
			DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY,
			JSON.stringify(['skill:list_skill'])
		);

		vi.resetModules();
		({ toolsStore } = await import('$lib/stores/tools.svelte'));

		expect(toolsStore.isToolEnabled('skill:list_skill')).toBe(false);
		expect(toolsStore.isToolEnabled('skill:read_skill')).toBe(true);
		expect([...toolsStore.getEnabledSkillToolNames()]).toEqual([SKILL_READ_TOOL]);
	});

	it('keeps old generic disabled settings compatible with Skill keys', async () => {
		storageState.set(
			DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY,
			JSON.stringify(['builtin:read_file', 'skill:read_skill'])
		);

		vi.resetModules();
		({ toolsStore } = await import('$lib/stores/tools.svelte'));

		expect(toolsStore.isToolEnabled('builtin:read_file')).toBe(false);
		expect(toolsStore.isToolEnabled('skill:read_skill')).toBe(false);
		expect(toolsStore.isToolEnabled('skill:list_skill')).toBe(true);
	});

	it('toggles Skill settings independently and persists them to localStorage', () => {
		toolsStore.toggleTool('skill:read_skill');

		expect(toolsStore.isToolEnabled('skill:read_skill')).toBe(false);
		expect(toolsStore.isToolEnabled('skill:list_skill')).toBe(true);
		expect([...toolsStore.getEnabledSkillToolNames()]).toEqual([SKILL_LIST_TOOL]);
		expect(
			JSON.parse(localStorage.getItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY) ?? '[]')
		).toEqual(['skill:read_skill']);

		toolsStore.toggleTool('skill:read_skill');

		expect(toolsStore.isToolEnabled('skill:read_skill')).toBe(true);
		expect(
			JSON.parse(localStorage.getItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY) ?? '[]')
		).toEqual([]);
	});

	it('round-trips independently toggled Skill settings through a fresh store', async () => {
		toolsStore.toggleTool('skill:list_skill');
		toolsStore.toggleTool('skill:read_skill');

		vi.resetModules();
		({ toolsStore } = await import('$lib/stores/tools.svelte'));

		expect(toolsStore.isToolEnabled('skill:read_skill')).toBe(false);
		expect(toolsStore.isToolEnabled('skill:list_skill')).toBe(false);
		expect([...toolsStore.getEnabledSkillToolNames()]).toEqual([]);
	});

	it('isSkillToolKey recognizes only the stable skill: settings keys', () => {
		expect(toolsStore.isSkillToolKey('skill:read_skill')).toBe(true);
		expect(toolsStore.isSkillToolKey('skill:list_skill')).toBe(true);
		expect(toolsStore.isSkillToolKey('builtin:read_file')).toBe(false);
		expect(toolsStore.isSkillToolKey('mcp-abc:read_file')).toBe(false);
		expect(toolsStore.isSkillToolKey('custom:read_skill')).toBe(false);
		expect(toolsStore.isSkillToolKey('frontend:run_javascript')).toBe(false);
		expect(toolsStore.isSkillToolKey('skill:unknown')).toBe(false);
	});

	it('keeps Skills out of the generic tool collections and LLM tool assembly', () => {
		// A generic builtin is served so the collections are non-empty.
		expect(toolsStore.allTools.map((entry) => entry.definition.function.name)).toEqual([
			'read_file'
		]);
		expect(toolsStore.allTools.some((entry) => toolsStore.isSkillToolKey(entry.key))).toBe(false);
		expect(toolsStore.allTools.map((entry) => entry.key)).not.toContain('skill:read_skill');
		expect(toolsStore.allTools.map((entry) => entry.definition.function.name)).not.toContain(
			SKILL_READ_TOOL
		);
		expect(toolsStore.allTools.map((entry) => entry.definition.function.name)).not.toContain(
			SKILL_LIST_TOOL
		);

		expect(toolsStore.toolGroups).toHaveLength(1);
		expect(toolsStore.toolGroups[0].source).toBe(ToolSource.BUILTIN);
		expect(toolsStore.toolGroups.some((group) => group.source === ToolSource.SKILLS)).toBe(false);
		expect(
			toolsStore.toolGroups.flatMap((group) => group.tools.map((tool) => tool.key))
		).not.toContain('skill:read_skill');

		const llmNames = toolsStore.getEnabledToolsForLLM().map((def) => def.function.name);

		expect(llmNames).toEqual(['read_file']);
		expect(llmNames).not.toContain(SKILL_READ_TOOL);
		expect(llmNames).not.toContain(SKILL_LIST_TOOL);
		expect(toolsStore.allToolDefinitions.map((def) => def.function.name)).not.toContain(
			SKILL_READ_TOOL
		);
	});

	it('never resolves Skills names to generic permission keys or sources', () => {
		expect(toolsStore.getPermissionKey(SKILL_READ_TOOL)).toBeNull();
		expect(toolsStore.getPermissionKey(SKILL_LIST_TOOL)).toBeNull();
		expect(toolsStore.getToolSource(SKILL_READ_TOOL)).toBeNull();
		expect(toolsStore.getToolSource(SKILL_LIST_TOOL)).toBeNull();
		expect(toolsStore.getToolServerLabel(SKILL_READ_TOOL)).toBe('');
	});
});
