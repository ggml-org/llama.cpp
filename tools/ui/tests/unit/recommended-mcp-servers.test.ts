import { describe, expect, it } from 'vitest';
import {
	RECOMMENDED_MCP_SERVER_IDS,
	RECOMMENDED_MCP_SERVERS
} from '$lib/constants/recommended-mcp-servers';
import { parseMcpServerSettings } from '$lib/utils/mcp';
import { DEFAULT_MCP_CONFIG, MCP_SERVER_ID_PREFIX } from '$lib/constants/mcp';

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
			expect(id.toLowerCase()).not.toContain(MCP_SERVER_ID_PREFIX.toLowerCase());
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
