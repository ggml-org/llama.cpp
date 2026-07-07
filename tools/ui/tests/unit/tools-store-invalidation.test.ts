import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

// node env unit project has no DOM, install a minimal localStorage backed by a Map
beforeAll(() => {
	const store = new Map<string, string>();
	const polyfill: Storage = {
		get length() {
			return store.size;
		},
		clear: () => store.clear(),
		getItem: (k) => (store.has(k) ? store.get(k)! : null),
		key: (i) => Array.from(store.keys())[i] ?? null,
		removeItem: (k) => {
			store.delete(k);
		},
		setItem: (k, v) => {
			store.set(k, String(v));
		}
	};
	(globalThis as unknown as { localStorage: Storage }).localStorage = polyfill;
});

// Hoist mock-state references so vi.mock factories (which Vitest runs at the
// top of the file before imports) can read mutable backing vars.
const mockState = vi.hoisted(() => {
	const makeBuiltIn = (name: string) => ({
		display_name: name,
		tool: name,
		type: 'builtin' as const,
		permissions: { write: false },
		definition: {
			type: 'function' as const,
			function: {
				name,
				description: `${name} tool`,
				parameters: { type: 'object' as const, properties: {} }
			}
		}
	});
	return {
		tokenizeResult: 18 as number | null,
		builtins: [
			makeBuiltIn('web_search').definition,
			makeBuiltIn('calculator').definition
		] as unknown[]
	};
});

vi.mock('$lib/services/tools.service', () => ({
	ToolsService: {
		list: () => Promise.resolve(mockState.builtins)
	}
}));

vi.mock('$lib/services/tokenize.service', () => ({
	TokenizeService: {
		count: () => Promise.resolve(mockState.tokenizeResult)
	}
}));

// Import after the mocks so toolsStore picks them up on construction.
import type { OpenAIToolDefinition } from '$lib/types';
const { toolsStore } = await import('$lib/stores/tools.svelte');
type StoreInternals = {
	_builtinTools: unknown[];
	_enabledToolsTokenCount: number | null;
	_enabledToolsTokenHash: string;
};

const readCount = (): number | null =>
	(toolsStore as unknown as StoreInternals)._enabledToolsTokenCount;

describe('toolsStore tool-toggle cache invalidation', () => {
	beforeEach(() => {
		// Reset cached count and disabled-list between tests so each start clean.
		const s = toolsStore as unknown as StoreInternals;
		s._enabledToolsTokenCount = null;
		s._enabledToolsTokenHash = '';
		localStorage.clear();
		// Re-seed built-ins (the constructor already populated them via the
		// mocked /tools endpoint, but enforcing them here makes each test
		// independent of any other).
		s._builtinTools = [...mockState.builtins];
	});

	afterEach(() => {
		mockState.tokenizeResult = 18;
	});

	it('resets cached token count to null after toggleTool', async () => {
		await toolsStore.refreshEnabledToolsTokenCount();
		expect(readCount()).toBe(18);

		toolsStore.toggleTool('builtin:web_search');
		expect(readCount()).toBeNull();
	});

	it('resets cached token count to null after setToolEnabled(false)', async () => {
		await toolsStore.refreshEnabledToolsTokenCount();
		expect(readCount()).toBe(18);

		toolsStore.setToolEnabled('builtin:web_search', false);
		expect(readCount()).toBeNull();
	});

	it('resets cached token count to null after toggleGroup', async () => {
		await toolsStore.refreshEnabledToolsTokenCount();
		expect(readCount()).toBe(18);

		toolsStore.toggleGroup({
			source: 'builtin' as never,
			label: 'Built-in',
			tools: [
				{
					source: 'builtin' as never,
					key: 'builtin:web_search',
					definition: mockState.builtins[0] as OpenAIToolDefinition
				},
				{
					source: 'builtin' as never,
					key: 'builtin:calculator',
					definition: mockState.builtins[1] as OpenAIToolDefinition
				}
			]
		});
		expect(readCount()).toBeNull();
	});

	it('after disabling the last tool, a refresh commits 0 not the stale count', async () => {
		await toolsStore.refreshEnabledToolsTokenCount();
		expect(readCount()).toBe(18);

		toolsStore.toggleTool('builtin:web_search');
		toolsStore.toggleTool('builtin:calculator');
		// Cache is invalidated; the empty-list branch must commit 0 without
		// ever calling the tokenizer service for the new (empty) list.
		mockState.tokenizeResult = 99; // would be wrong if reached
		const result = await toolsStore.refreshEnabledToolsTokenCount();
		expect(result).toBe(0);
		expect(readCount()).toBe(0);
	});
});
