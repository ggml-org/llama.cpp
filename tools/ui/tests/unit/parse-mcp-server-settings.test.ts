import { describe, expect, it, vi } from 'vitest';
import { parseMcpServerSettings } from '$lib/utils/mcp';
import { DEFAULT_MCP_CONFIG, MCP_SERVER_ID_PREFIX } from '$lib/constants/mcp';

/**
 * Tests for the mcpServers settings parser.
 *
 * The branch seeds the MCP servers setting with a default value of
 * `JSON.stringify(RECOMMENDED_MCP_SERVERS)`, so the parser has to be
 * resilient to anything that may live in the user's localStorage: malformed
 * JSON, wrong shapes, missing fields, falsy-but-not-zero numbers, and entry
 * arrays that have been mutated by the user via the settings form.
 */
describe('parseMcpServerSettings', () => {
	it('returns an empty array for falsy or whitespace-only input', () => {
		expect(parseMcpServerSettings(null)).toEqual([]);
		expect(parseMcpServerSettings(undefined)).toEqual([]);
		expect(parseMcpServerSettings('')).toEqual([]);
		expect(parseMcpServerSettings('   ')).toEqual([]);
	});

	it('returns an empty array and logs a warning for invalid JSON strings', () => {
		const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

		expect(parseMcpServerSettings('{not-json')).toEqual([]);
		expect(warn).toHaveBeenCalled();

		warn.mockRestore();
	});

	it('returns an empty array for valid JSON that is not an array', () => {
		expect(parseMcpServerSettings('"plain-string"')).toEqual([]);
		expect(parseMcpServerSettings('{"id":"foo"}')).toEqual([]);
		expect(parseMcpServerSettings('42')).toEqual([]);
		expect(parseMcpServerSettings('null')).toEqual([]);
	});

	it('drops entries with no parseable id and substitutes a stable fallback', () => {
		const parsed = parseMcpServerSettings(
			JSON.stringify([{ url: 'https://a.test', enabled: true }, { url: 'https://b.test' }])
		);

		expect(parsed).toHaveLength(2);
		expect(parsed[0]?.id).toBe(`${MCP_SERVER_ID_PREFIX}-1`);
		expect(parsed[1]?.id).toBe(`${MCP_SERVER_ID_PREFIX}-2`);
	});

	it('reuses the first id when it is present and falls back only for missing ones', () => {
		const parsed = parseMcpServerSettings(
			JSON.stringify([
				{ id: 'custom-1', url: 'https://a.test' },
				{ url: 'https://b.test' },
				{ id: 'custom-3', url: 'https://c.test' }
			])
		);

		expect(parsed[0]?.id).toBe('custom-1');
		expect(parsed[1]?.id).toBe(`${MCP_SERVER_ID_PREFIX}-2`);
		expect(parsed[2]?.id).toBe('custom-3');
	});

	it('falls back to the configured default requestTimeoutSeconds only for nullish values', () => {
		const fallback = DEFAULT_MCP_CONFIG.requestTimeoutSeconds;

		const parsed = parseMcpServerSettings(
			JSON.stringify([
				{ id: 'a', url: 'https://a.test' },
				{ id: 'b', url: 'https://b.test', requestTimeoutSeconds: undefined },
				{ id: 'c', url: 'https://c.test', requestTimeoutSeconds: 0 },
				{ id: 'd', url: 'https://d.test', requestTimeoutSeconds: 45 }
			])
		);

		// The parser uses ?? for timeout fallback, which only triggers on
		// null/undefined. Explicit 0 is preserved at face value.
		expect(parsed[0]?.requestTimeoutSeconds).toBe(fallback);
		expect(parsed[1]?.requestTimeoutSeconds).toBe(fallback);
		expect(parsed[2]?.requestTimeoutSeconds).toBe(0);
		expect(parsed[3]?.requestTimeoutSeconds).toBe(45);
	});

	it('treats whitespace-only headers strings as undefined', () => {
		const parsed = parseMcpServerSettings(
			JSON.stringify([
				{ id: 'a', url: 'https://a.test', headers: '   ' },
				{ id: 'b', url: 'https://b.test', headers: '{"X-Foo":"bar"}' }
			])
		);

		// The parser trims headers and coerces empty/whitespace to undefined.
		expect(parsed[0]?.headers).toBeUndefined();
		expect(parsed[1]?.headers).toBe('{"X-Foo":"bar"}');
	});

	it('defaults coercion for booleans (undefined -> false, true -> true)', () => {
		const parsed = parseMcpServerSettings(
			JSON.stringify([
				{ id: 'a', url: 'https://a.test' },
				{ id: 'b', url: 'https://b.test', enabled: true },
				{ id: 'c', url: 'https://c.test', enabled: false },
				{ id: 'd', url: 'https://d.test', useProxy: true }
			])
		);

		expect(parsed[0]?.enabled).toBe(false);
		expect(parsed[1]?.enabled).toBe(true);
		expect(parsed[2]?.enabled).toBe(false);
		expect(parsed[0]?.useProxy).toBe(false);
		expect(parsed[3]?.useProxy).toBe(true);
	});

	it('preserves input order when mapping entries', () => {
		const source = [
			{ id: 'gamma', url: 'https://c.test' },
			{ id: 'alpha', url: 'https://a.test' },
			{ id: 'beta', url: 'https://b.test' }
		];

		const parsed = parseMcpServerSettings(JSON.stringify(source));

		expect(parsed.map((entry) => entry.id)).toEqual(['gamma', 'alpha', 'beta']);
	});

	it('passes non-string raw input through the JSON-equality path', () => {
		const parsed = parseMcpServerSettings([
			{ id: 'a', url: 'https://a.test' },
			{ id: 'b', url: 'https://b.test', enabled: true }
		]);

		expect(parsed).toHaveLength(2);
		expect(parsed[0]?.id).toBe('a');
		expect(parsed[1]?.enabled).toBe(true);
	});

	it('coerces non-string url values to an empty string rather than throwing', () => {
		const parsed = parseMcpServerSettings(
			JSON.stringify([{ id: 'a', url: 42 }, { id: 'b' }, { id: 'c', url: 'https://c.test' }])
		);

		expect(parsed[0]?.url).toBe('');
		expect(parsed[1]?.url).toBe('');
		expect(parsed[2]?.url).toBe('https://c.test');
	});
});
