import { describe, expect, it } from 'vitest';
import { isAbortError } from '$lib/utils/abort';

describe('isAbortError', () => {
	it('returns false for null, undefined and non-error values', () => {
		expect(isAbortError(null)).toBe(false);
		expect(isAbortError(undefined)).toBe(false);
		expect(isAbortError('string error')).toBe(false);
		expect(isAbortError({ name: 'AbortError' })).toBe(false);
		expect(isAbortError(42)).toBe(false);
	});

	it('returns true for DOMException with AbortError name', () => {
		const err = new DOMException('Operation was aborted', 'AbortError');
		expect(isAbortError(err)).toBe(true);
	});

	it('returns true for plain Error with AbortError name', () => {
		const err = new Error('aborted');
		err.name = 'AbortError';
		expect(isAbortError(err)).toBe(true);
	});

	it('returns false for unrelated Error instances', () => {
		expect(isAbortError(new Error('something failed'))).toBe(false);
		expect(isAbortError(new TypeError('not related'))).toBe(false);
		expect(isAbortError(new RangeError('out of range'))).toBe(false);
	});

	it('recognizes Firefox TypeError "Error in input stream" emitted at page unload', () => {
		expect(isAbortError(new TypeError('Error in input stream'))).toBe(true);
		expect(isAbortError(new TypeError('TypeError: Error in input stream'))).toBe(true);
	});

	it('recognizes Safari "The network connection was lost" during transient drop', () => {
		expect(isAbortError(new TypeError('The network connection was lost.'))).toBe(true);
	});

	it('recognizes Safari "Load failed" during page navigation', () => {
		expect(isAbortError(new TypeError('Load failed'))).toBe(true);
	});

	it('does NOT recognize generic TypeError messages as aborts', () => {
		// matching too broadly would hide real bugs, the predicate must stay conservative
		expect(isAbortError(new TypeError('Failed to fetch'))).toBe(false);
		expect(isAbortError(new TypeError('Cannot read property of undefined'))).toBe(false);
		expect(isAbortError(new TypeError('NetworkError when attempting to fetch resource'))).toBe(
			false
		);
	});

	it('is case insensitive on the matched substrings', () => {
		expect(isAbortError(new TypeError('error in INPUT STREAM'))).toBe(true);
		expect(isAbortError(new TypeError('the network connection WAS LOST'))).toBe(true);
	});
});
