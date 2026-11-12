import { MCP_SERVER_ID_PREFIX } from '$lib/constants';
import { parseMcpServerSettings } from '$lib/utils/mcp';
import { describe, expect, it, vi } from 'vitest';

/**
 * Tests for the mcpServers settings parser.
 *
 * The parser has to be resilient to anything that may live in the
 * user's localStorage: malformed JSON, wrong shapes, missing fields,
 * falsy-but-not-zero numbers, and entry arrays that have been mutated
 * by the user via the settings form.
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
			JSON.stringify([{ enabled: true, url: 'https://a.test' }, { url: 'https://b.test' }])
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

	it('does not emit a per-server timeout, the request timeout is a live global setting', () => {
		// A stored per-server requestTimeoutSeconds was never editable in
		// any UI and froze the global setting at server creation time,
		// making the Settings value a no-op for existing servers. The
		// parser drops the field so the global applies live everywhere.
		const parsed = parseMcpServerSettings(
			JSON.stringify([{ id: 'a', requestTimeoutSeconds: 45, url: 'https://a.test' }])
		);

		expect(parsed[0]).not.toHaveProperty('requestTimeoutSeconds');
	});

	it('treats whitespace-only headers strings as undefined', () => {
		const parsed = parseMcpServerSettings(
			JSON.stringify([
				{ headers: '   ', id: 'a', url: 'https://a.test' },
				{ headers: '{"X-Foo":"bar"}', id: 'b', url: 'https://b.test' }
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
				{ enabled: true, id: 'b', url: 'https://b.test' },
				{ enabled: false, id: 'c', url: 'https://c.test' },
				{ id: 'd', url: 'https://d.test', useProxy: true }
			])
		);

		expect(parsed[0]?.enabled).toBe(false);
		expect(parsed[1]?.enabled).toBe(true);
		expect(parsed[2]?.enabled).toBe(false);
		expect(parsed[0]?.useProxy).toBe(false);
		expect(parsed[3]?.useProxy).toBe(true);
	});

	it('keeps disabled entries in the list, enabled is state and never a visibility filter', () => {
		// Regression guard for issue #25625: filtering the server list on
		// `enabled` hides a toggled-off server from every UI surface with
		// no way to re-enable it. Any list derived from this parser must
		// contain disabled entries.
		const parsed = parseMcpServerSettings(
			JSON.stringify([
				{ enabled: true, id: 'on', url: 'https://on.test' },
				{ enabled: false, id: 'off', url: 'https://off.test' }
			])
		);

		expect(parsed.map((entry) => entry.id)).toEqual(['on', 'off']);
		expect(parsed[1]?.enabled).toBe(false);
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
			{ enabled: true, id: 'b', url: 'https://b.test' }
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
