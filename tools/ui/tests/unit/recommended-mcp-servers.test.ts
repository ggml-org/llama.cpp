import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
	RECOMMENDED_MCP_SERVER_IDS,
	RECOMMENDED_MCP_SERVERS
} from '$lib/constants/recommended-mcp-servers';
import { parseMcpServerSettings } from '$lib/utils/mcp';
import { DEFAULT_MCP_CONFIG, MCP_SERVER_ID_PREFIX } from '$lib/constants/mcp';

// Tell the opt-in helpers they are running in a browser context so they
// read/write localStorage. Hoisted by vitest before the module under test
// is imported.
vi.mock('$app/environment', () => ({ browser: true }));

const RECOMMENDATIONS_OPTIN_KEY = 'LlamaUi.mcpRecommendationsOptedIn';
type OptInModule = typeof import('$lib/utils/recommended-mcp-servers.svelte');

// Polyfill localStorage - the unit project runs in node with no DOM
function installLocalStoragePolyfill(): void {
	if (typeof globalThis.localStorage !== 'undefined') return;

	const store = new Map<string, string>();
	const polyfill: Storage = {
		get length() {
			return store.size;
		},
		clear: () => store.clear(),
		getItem: (k) => (store.has(k) ? (store.get(k) as string) : null),
		key: (i) => Array.from(store.keys())[i] ?? null,
		removeItem: (k) => {
			store.delete(k);
		},
		setItem: (k, v) => {
			store.set(k, String(v));
		}
	};
	(globalThis as unknown as { localStorage: Storage }).localStorage = polyfill;
}

beforeEach(() => {
	installLocalStoragePolyfill();
	localStorage.clear();
});

// Each test re-imports the helper module so its module-level SvelteSet
// starts empty. vi.resetModules drops the cached evaluation while
// vi.doMock re-asserts the browser flag for the fresh module load.
async function loadOptInModule(): Promise<OptInModule> {
	vi.resetModules();
	vi.doMock('$app/environment', () => ({ browser: true }));

	return import('$lib/utils/recommended-mcp-servers.svelte');
}

/**
 * Tests for the predefined recommended MCP servers.
 *
 * These are surfaced to first-time users via
 * DialogMcpServerRecommendations and used as the default value of the MCP
 * servers setting, so a regression that breaks the round-trip through the
 * settings parser would silently break onboarding for new users.
 */
describe('RECOMMENDED_MCP_SERVERS', () => {
	it('lists at least one entry and uses stable, unique ids', () => {
		expect(RECOMMENDED_MCP_SERVERS.length).toBeGreaterThan(0);

		const ids = RECOMMENDED_MCP_SERVERS.map((server) => server.id);
		expect(new Set(ids).size).toBe(ids.length);

		for (const id of ids) {
			expect(id).toMatch(/^[a-z0-9-]+$/);
			expect(MCP_SERVER_ID_PREFIX).not.toContain(id);
		}
	});

	it('requires a name, description and url for every entry', () => {
		for (const server of RECOMMENDED_MCP_SERVERS) {
			expect(server.name?.trim().length ?? 0).toBeGreaterThan(0);
			expect(server.description.trim().length).toBeGreaterThan(0);
			expect(server.url.trim().length).toBeGreaterThan(0);
			expect(() => new URL(server.url)).not.toThrow();
		}
	});
});

describe('RECOMMENDED_MCP_SERVER_IDS', () => {
	it('matches the ids declared in RECOMMENDED_MCP_SERVERS', () => {
		expect(RECOMMENDED_MCP_SERVER_IDS.size).toBe(RECOMMENDED_MCP_SERVERS.length);

		for (const server of RECOMMENDED_MCP_SERVERS) {
			expect(RECOMMENDED_MCP_SERVER_IDS.has(server.id)).toBe(true);
		}
	});
});

describe('recommended-mcp-servers default value', () => {
	it('round-trips cleanly through parseMcpServerSettings', () => {
		const serialized = JSON.stringify(RECOMMENDED_MCP_SERVERS);
		const parsed = parseMcpServerSettings(serialized);

		expect(parsed).toHaveLength(RECOMMENDED_MCP_SERVERS.length);

		for (let index = 0; index < RECOMMENDED_MCP_SERVERS.length; index++) {
			const source = RECOMMENDED_MCP_SERVERS[index];
			const entry = parsed[index];

			expect(entry).toBeDefined();
			expect(entry?.id).toBe(source.id);
			expect(entry?.url).toBe(source.url);
			expect(entry?.enabled).toBe(source.enabled);
			expect(entry?.requestTimeoutSeconds).toBe(source.requestTimeoutSeconds);
			expect(entry?.name).toBe(source.name);

			// Headers and useProxy are not set on recommended servers; the
			// parser must fall back to the inactive defaults rather than
			// surfacing undefined-boundary states.
			expect(entry?.headers).toBeUndefined();
			expect(entry?.useProxy).toBe(false);
		}
	});

	it('uses the global default timeout when one is not specified on an entry', () => {
		const sourceOnlyRequired = {
			id: 'roundtrip-only',
			name: 'Only required fields',
			url: 'https://example.test/mcp',
			description: 'Smoke entry for parser roundtrip with default timeout.',
			enabled: true
		};

		const parsed = parseMcpServerSettings(JSON.stringify([sourceOnlyRequired]));
		const entry = parsed[0];

		expect(entry?.requestTimeoutSeconds).toBe(DEFAULT_MCP_CONFIG.requestTimeoutSeconds);
	});
});

describe('parseOptedInRecommendationIds', () => {
	it('returns an empty array for null and empty inputs', async () => {
		const { parseOptedInRecommendationIds } = await loadOptInModule();
		expect(parseOptedInRecommendationIds(null)).toEqual([]);
		expect(parseOptedInRecommendationIds('')).toEqual([]);
	});

	it('returns an empty array for non-JSON payloads without throwing', async () => {
		const { parseOptedInRecommendationIds } = await loadOptInModule();
		expect(parseOptedInRecommendationIds('not-json')).toEqual([]);
		expect(parseOptedInRecommendationIds('{')).toEqual([]);
	});

	it('returns an empty array for non-array JSON values', async () => {
		const { parseOptedInRecommendationIds } = await loadOptInModule();
		expect(parseOptedInRecommendationIds('"a"')).toEqual([]);
		expect(parseOptedInRecommendationIds('{"id":"a"}')).toEqual([]);
		expect(parseOptedInRecommendationIds('42')).toEqual([]);
	});

	it('keeps only string entries from a mixed-type array', async () => {
		const { parseOptedInRecommendationIds } = await loadOptInModule();
		expect(parseOptedInRecommendationIds('["a", 1, null, "b", true]')).toEqual(['a', 'b']);
	});

	it('preserves valid entries verbatim', async () => {
		const { parseOptedInRecommendationIds } = await loadOptInModule();
		expect(parseOptedInRecommendationIds(JSON.stringify(['exa-web-search']))).toEqual([
			'exa-web-search'
		]);
	});
});

describe('mcp-recommendation opt-in helpers', () => {
	const firstRecommendation = RECOMMENDED_MCP_SERVERS[0]?.id;
	const secondRecommendation = RECOMMENDED_MCP_SERVERS[1]?.id ?? firstRecommendation;

	if (!firstRecommendation) {
		throw new Error('test setup: missing predefined recommendations');
	}

	it('hydrates the reactive set from a valid localStorage payload at module load', async () => {
		localStorage.setItem(RECOMMENDATIONS_OPTIN_KEY, JSON.stringify([firstRecommendation]));

		const { getOptedInRecommendationIds } = await loadOptInModule();
		expect(getOptedInRecommendationIds().has(firstRecommendation)).toBe(true);
	});

	it('starts with an empty set when localStorage is empty', async () => {
		const { getOptedInRecommendationIds } = await loadOptInModule();
		expect(getOptedInRecommendationIds().size).toBe(0);
	});

	it('starts with an empty set when localStorage holds a malformed payload', async () => {
		localStorage.setItem(RECOMMENDATIONS_OPTIN_KEY, 'not-json');

		const { getOptedInRecommendationIds } = await loadOptInModule();
		expect(getOptedInRecommendationIds().size).toBe(0);
	});

	it('adds accepted IDs to the reactive set and mirrors them to localStorage', async () => {
		const { addOptedInRecommendationIds, getOptedInRecommendationIds, isOptedInRecommendation } =
			await loadOptInModule();

		addOptedInRecommendationIds([firstRecommendation]);

		expect(isOptedInRecommendation(firstRecommendation)).toBe(true);
		expect(getOptedInRecommendationIds().has(firstRecommendation)).toBe(true);

		const stored = localStorage.getItem(RECOMMENDATIONS_OPTIN_KEY);
		expect(stored).not.toBeNull();

		const parsed = JSON.parse(stored as string) as unknown;
		expect(parsed).toEqual([firstRecommendation]);
	});

	it('drops IDs that are not part of RECOMMENDED_MCP_SERVER_IDS', async () => {
		const { addOptedInRecommendationIds, getOptedInRecommendationIds } = await loadOptInModule();

		addOptedInRecommendationIds(['not-a-real-recommendation', 123, null] as unknown as string[]);

		expect(getOptedInRecommendationIds().size).toBe(0);
		// No valid IDs were accepted, so no localStorage write should happen
		expect(localStorage.getItem(RECOMMENDATIONS_OPTIN_KEY)).toBeNull();
	});

	it('accepts valid IDs while ignoring invalid ones in the same call', async () => {
		const { addOptedInRecommendationIds, getOptedInRecommendationIds } = await loadOptInModule();

		addOptedInRecommendationIds([
			'not-a-real-recommendation',
			firstRecommendation,
			123,
			'another-fake'
		] as unknown as string[]);

		const ids = getOptedInRecommendationIds();
		expect(ids.size).toBe(1);
		expect(ids.has(firstRecommendation)).toBe(true);
	});

	it('is a no-op on the server (browser=false)', async () => {
		// Override the import-time mock for this single module load so we hit
		// the `if (!browser) return;` short-circuit. localStorage is still
		// writable from the polyfill; if the guard regresses we'll see writes
		// leaking into it.
		vi.resetModules();
		vi.doMock('$app/environment', () => ({ browser: false }));
		const { addOptedInRecommendationIds, getOptedInRecommendationIds } =
			await import('$lib/utils/recommended-mcp-servers.svelte');

		addOptedInRecommendationIds([firstRecommendation]);

		expect(getOptedInRecommendationIds().size).toBe(0);
		expect(localStorage.getItem(RECOMMENDATIONS_OPTIN_KEY)).toBeNull();
	});

	it('does not hydrate from localStorage when browser=false, even if a payload exists', async () => {
		// Pre-seed the polyfill with a payload that would normally hydrate the
		// set on the client. Under SSR (browser=false), the module should
		// ignore it entirely and start empty.
		localStorage.setItem(RECOMMENDATIONS_OPTIN_KEY, JSON.stringify([firstRecommendation]));

		vi.resetModules();
		vi.doMock('$app/environment', () => ({ browser: false }));
		const { getOptedInRecommendationIds } =
			await import('$lib/utils/recommended-mcp-servers.svelte');

		expect(getOptedInRecommendationIds().size).toBe(0);
	});

	it('merges new accepted IDs without dropping previously stored ones', async () => {
		// Pre-seed localStorage with a previous session's accepted id
		localStorage.setItem(RECOMMENDATIONS_OPTIN_KEY, JSON.stringify([firstRecommendation]));

		const { addOptedInRecommendationIds, getOptedInRecommendationIds } = await loadOptInModule();

		addOptedInRecommendationIds([secondRecommendation]);

		const ids = getOptedInRecommendationIds();
		expect(ids.has(firstRecommendation)).toBe(true);
		expect(ids.has(secondRecommendation)).toBe(true);
	});

	it('does not duplicate IDs that are already opted in', async () => {
		const { addOptedInRecommendationIds, getOptedInRecommendationIds } = await loadOptInModule();

		addOptedInRecommendationIds([firstRecommendation, firstRecommendation]);

		expect(getOptedInRecommendationIds().size).toBe(1);
		expect(JSON.parse(localStorage.getItem(RECOMMENDATIONS_OPTIN_KEY) as string)).toEqual([
			firstRecommendation
		]);
	});
});
