import { describe, expect, it } from 'vitest';
import { parseHeadersToArray, serializeHeaders } from '$lib/utils/headers';

/**
 * Tests for the header serialization helpers used by the MCP server form
 * (custom header rows) and the new Authorization/Bearer-token flow.
 */
describe('parseHeadersToArray', () => {
	it('returns an empty array for empty or whitespace-only input', () => {
		expect(parseHeadersToArray('')).toEqual([]);
		expect(parseHeadersToArray('   ')).toEqual([]);
		expect(parseHeadersToArray(undefined as unknown as string)).toEqual([]);
	});

	it('returns an empty array for invalid JSON input', () => {
		expect(parseHeadersToArray('{not-json')).toEqual([]);
		expect(parseHeadersToArray('[]')).toEqual([]);
		expect(parseHeadersToArray('"plain-string"')).toEqual([]);
	});

	it('converts an object into ordered key/value pairs', () => {
		expect(parseHeadersToArray('{"X-Foo":"bar","Authorization":"Bearer abc"}')).toEqual([
			{ key: 'X-Foo', value: 'bar' },
			{ key: 'Authorization', value: 'Bearer abc' }
		]);
	});

	it('stringifies non-string values', () => {
		expect(parseHeadersToArray('{"count":"42","flag":"true"}')).toEqual([
			{ key: 'count', value: '42' },
			{ key: 'flag', value: 'true' }
		]);
	});
});

describe('serializeHeaders', () => {
	it('returns an empty string when there are no valid pairs', () => {
		expect(serializeHeaders([])).toBe('');
		expect(serializeHeaders([{ key: '', value: 'value' }])).toBe('');
		expect(serializeHeaders([{ key: '   ', value: 'value' }])).toBe('');
	});

	it('returns an empty string when every pair has a blank key', () => {
		expect(
			serializeHeaders([
				{ key: '', value: 'drop-me' },
				{ key: '   ', value: 'drop-me-too' },
				{ key: '\t', value: 'tab-key' }
			])
		).toBe('');
	});

	it('drops pairs with empty keys but keeps the rest', () => {
		expect(
			serializeHeaders([
				{ key: '', value: 'drop-me' },
				{ key: 'X-Keep', value: 'ok' }
			])
		).toBe('{"X-Keep":"ok"}');
	});

	it('trims keys before serializing', () => {
		expect(serializeHeaders([{ key: '  X-Space  ', value: 'ok' }])).toBe('{"X-Space":"ok"}');
	});

	it('preserves the input order of surviving pairs', () => {
		const serialized = serializeHeaders([
			{ key: 'X-C', value: '3' },
			{ key: 'X-A', value: '1' },
			{ key: 'X-B', value: '2' }
		]);

		// Object key order follows insertion order in modern JS engines, so
		// the serialized JSON writes keys in our input order.
		expect(JSON.parse(serialized)).toEqual({ 'X-C': '3', 'X-A': '1', 'X-B': '2' });
	});
});

describe('parseHeadersToArray / serializeHeaders roundtrip', () => {
	it('serializes back to an equal header object after a parse', () => {
		const original = JSON.stringify({
			'Content-Type': 'application/json',
			'X-Trace-Id': 'abc-123'
		});

		const roundtrip = serializeHeaders(parseHeadersToArray(original));

		expect(JSON.parse(roundtrip)).toEqual(JSON.parse(original));
	});

	it('drops rows whose keys are blank after trimming during serialization', () => {
		const pairs = parseHeadersToArray('{"X-Keep":"ok","":"drop-me"}');

		// parseHeadersToArray keeps raw key strings (the consumer is expected to
		// filter blanks, not the parser); serialization must strip them.
		expect(pairs).toEqual([
			{ key: 'X-Keep', value: 'ok' },
			{ key: '', value: 'drop-me' }
		]);
		expect(serializeHeaders(pairs)).toBe('{"X-Keep":"ok"}');
	});

	it('preserves upstream keys untouched (does not lowercase them)', () => {
		const upperCased = '{"Authorization":"Bearer xyz"}';

		const parsed = parseHeadersToArray(upperCased);

		expect(parsed).toEqual([{ key: 'Authorization', value: 'Bearer xyz' }]);
	});

	it('bearer-token write survives a re-parse when paired with regular custom headers', () => {
		// The McpServerForm bearer UI writes {Authorization: `Bearer <token>`}
		// into the same headers string as the custom KV section. The round
		// trip below mirrors the exact shape the form produces so a future
		// refactor of either code path cannot silently change the on-disk key.
		const pairs = [
			{ key: 'X-Trace-Id', value: 'abc-123' },
			{ key: 'Authorization', value: 'Bearer super-secret' }
		];

		const serialized = serializeHeaders(pairs);

		expect(serialized).toBe('{"X-Trace-Id":"abc-123","Authorization":"Bearer super-secret"}');
		expect(parseHeadersToArray(serialized)).toEqual(pairs);
	});
});
