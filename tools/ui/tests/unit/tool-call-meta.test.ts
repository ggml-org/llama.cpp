import { describe, expect, it } from 'vitest';
import { tryParseToolResultObject } from '$lib/utils';

describe('tryParseToolResultObject', () => {
	it('returns null when no result is provided', () => {
		expect(tryParseToolResultObject(undefined)).toBeNull();
		expect(tryParseToolResultObject('')).toBeNull();
	});

	it('returns the parsed object when the result is JSON', () => {
		expect(tryParseToolResultObject('{"result":"ok","bytes":42}')).toEqual({
			result: 'ok',
			bytes: 42
		});
	});

	it('returns null for JSON arrays (only objects are useful to callers)', () => {
		expect(tryParseToolResultObject('[1,2,3]')).toBeNull();
	});

	it('returns null for JSON primitives', () => {
		expect(tryParseToolResultObject('"raw string"')).toBeNull();
		expect(tryParseToolResultObject('42')).toBeNull();
	});

	it('returns null for invalid JSON', () => {
		expect(tryParseToolResultObject('not json')).toBeNull();
		expect(tryParseToolResultObject('{bad')).toBeNull();
	});
});
