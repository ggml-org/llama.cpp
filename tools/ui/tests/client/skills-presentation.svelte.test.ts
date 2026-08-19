// Merged Skills presentation smoke: provider label, chat result block, page shell, toolsStore settings.

import ChatMessageActionCardPermissionRequest from '$lib/components/app/chat/ChatMessages/ChatMessageActions/ChatMessageActionCard/ChatMessageActionCardPermissionRequest.svelte';
import ChatMessageToolCallBlock from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlock.svelte';
import SkillProviderLabel from '$lib/components/app/skills/SkillProviderLabel.svelte';
import SkillsPage from '../../src/routes/skills/+page.svelte';
import { goto } from '$app/navigation';
import { DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY, ROUTES, SKILL_LIST_TOOL, SKILL_READ_TOOL } from '$lib/constants';
import { AgenticSectionType, AttachmentType } from '$lib/enums';
import { settingsStore } from '$lib/stores/settings.svelte';
import { skillsStore } from '$lib/stores/skills.svelte';
import type { toolsStore as ToolsStoreValue } from '$lib/stores/tools.svelte';
import type { AgenticSection, DatabaseMessageExtraSkill, SkillConsentInfo } from '$lib/types';
import { jsonResponse, makeCatalog, makeEntry } from '../fixtures/skills';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

vi.mock('$app/navigation', async (importOriginal) => {
	const actual = await importOriginal<typeof import('$app/navigation')>();

	return { ...actual, goto: vi.fn() };
});

function skillExtra(overrides: Partial<DatabaseMessageExtraSkill> = {}): DatabaseMessageExtraSkill {
	return {
		kind: 'base',
		name: 'add-new-model',
		provider: 'agents',
		scope: 'project',
		skillId: 'opaque-id-1',
		state: 'approved',
		type: AttachmentType.SKILL,
		...overrides
	};
}

function section(overrides: Partial<AgenticSection> = {}): AgenticSection {
	return {
		content: '',
		toolName: 'read_skill',
		toolResult: '<skill_content name="add-new-model">body &amp; more</skill_content>',
		type: AgenticSectionType.TOOL_CALL,
		...overrides
	};
}

async function renderBlock(sectionData: AgenticSection) {
	const { container } = render(ChatMessageToolCallBlock, {
		isStreaming: false,
		open: true,
		section: sectionData
	});

	await tick();

	return container;
}

async function renderCard(skill?: SkillConsentInfo) {
	const { container } = render(ChatMessageActionCardPermissionRequest, {
		onDecision: vi.fn(),
		serverLabel: 'llama-server',
		skill,
		toolName: 'read_skill'
	});

	await tick();

	return container;
}

function textOf(container: HTMLElement): string {
	return (container.textContent ?? '').replace(/\s+/g, ' ').trim();
}

afterEach(() => {
	vi.restoreAllMocks();
});

describe('SkillProviderLabel', () => {
	it.each([
		['agents', 'generic'],
		['claude', 'claude']
	] as const)('renders provider %s as %s', (provider, expected) => {
		const { container } = render(SkillProviderLabel, { props: { provider } });

		expect(container.textContent).toContain(expected);
	});
});

describe('read_skill result rendering', () => {
	it('renders a base activation with typed labels and opaque XML text', async () => {
		const container = await renderBlock(section({ toolResultExtras: [skillExtra()] }));
		const text = textOf(container);

		expect(text).toContain('Skill · add-new-model');
		expect(text).toContain('generic · project');
		expect(text).not.toContain('agents · project');
		// XML remains literal text and is never parsed into DOM markup.
		expect(text).toContain('<skill_content name="add-new-model">body &amp; more</skill_content>');
		expect(container.querySelector('skill_content')).toBeNull();
	});

	it('falls back to the generic tool card for a read_skill section without valid metadata', async () => {
		const container = await renderBlock(section({ toolResultExtras: [] }));
		const text = textOf(container);

		expect(text).toContain('read_skill');
		expect(text).not.toContain('Skill ·');
	});
});

describe('permission request card skill identity', () => {
	it('shows the safe skill identity for a base consent pause', async () => {
		const container = await renderCard({
			name: 'add-new-model',
			provider: 'agents',
			scope: 'project'
		});
		const text = textOf(container);

		expect(text).toContain('Allow use of read_skill from llama-server?');
		expect(text).toContain('Skill: add-new-model (project · generic)');
		expect(text).not.toContain('(project · agents)');
	});
});

describe('shared standalone page shell (Skills route)', () => {
	beforeEach(() => {
		localStorage.clear();
		settingsStore.initialize();
		skillsStore.invalidate(undefined);
		vi.mocked(fetch).mockImplementation(async () =>
			jsonResponse(makeCatalog([makeEntry('demo-skill')]))
		);
		vi.mocked(goto).mockClear();
		// A fresh page uses the start route for Close.
		Object.defineProperty(window.history, 'length', { configurable: true, value: 1 });
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

	it('closes Skills to the start route when history is shallow', async () => {
		const skills = await render(SkillsPage);

		await skills.getByRole('button', { name: 'Close' }).click();

		expect(goto).toHaveBeenCalledWith(ROUTES.START);
	});
});

// The browser env has real localStorage, but a per-test shadow keeps the
// persisted disabled-tool set deterministic and isolated from other suites.
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

let toolsStore: typeof ToolsStoreValue;

describe('ToolsStore Skills settings group', () => {
	beforeEach(async () => {
		storageState.clear();
		Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: storagePolyfill });
		// A fresh store instance re-reads the persisted disabled-tool set.
		vi.resetModules();
		({ toolsStore } = await import('$lib/stores/tools.svelte'));
	});

	it('defaults both Skill settings to enabled', () => {
		expect([...toolsStore.getEnabledSkillToolNames()].sort()).toEqual([
			SKILL_LIST_TOOL,
			SKILL_READ_TOOL
		]);
		expect(toolsStore.isToolEnabled('skill:read_skill')).toBe(true);
		expect(toolsStore.isToolEnabled('skill:list_skill')).toBe(true);
	});

	it('persists a toggled Skill setting under its skill: key', () => {
		toolsStore.toggleTool('skill:read_skill');

		expect(toolsStore.isToolEnabled('skill:read_skill')).toBe(false);
		expect(
			JSON.parse(localStorage.getItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY) ?? '[]')
		).toEqual(['skill:read_skill']);

		toolsStore.toggleTool('skill:read_skill');

		expect(
			JSON.parse(localStorage.getItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY) ?? '[]')
		).toEqual([]);
	});

	it('reflects the enabled set in getEnabledSkillToolNames', () => {
		toolsStore.toggleTool('skill:list_skill');

		expect([...toolsStore.getEnabledSkillToolNames()]).toEqual([SKILL_READ_TOOL]);
	});
});
