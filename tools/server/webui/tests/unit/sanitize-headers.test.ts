import { describe, expect, it } from 'vitest';
import { sanitizeHeaders } from '$lib/utils/api-headers';

describe('sanitizeHeaders', () => {
	it('returns empty object for undefined input', () => {
		expect(sanitizeHeaders()).toEqual({});
	});

	it('passes through non-sensitive headers', () => {
		const headers = new Headers({ 'content-type': 'application/json', accept: 'text/html' });
		expect(sanitizeHeaders(headers)).toEqual({
			'content-type': 'application/json',
			accept: 'text/html'
		});
	});

	it('redacts known sensitive headers', () => {
		const headers = new Headers({
			authorization: 'Bearer secret',
			'x-api-key': 'key-123',
			'content-type': 'application/json'
		});
		const result = sanitizeHeaders(headers);
		expect(result.authorization).toBe('[redacted]');
		expect(result['x-api-key']).toBe('[redacted]');
		expect(result['content-type']).toBe('application/json');
	});

	it('partially redacts headers specified in partialRedactHeaders', () => {
		const headers = new Headers({ 'mcp-session-id': 'session-12345' });
		const partial = new Map([['mcp-session-id', 5]]);
		expect(sanitizeHeaders(headers, undefined, partial)['mcp-session-id']).toBe('....12345');
	});

	it('fully redacts mcp-session-id when no partialRedactHeaders is given', () => {
		const headers = new Headers({ 'mcp-session-id': 'session-12345' });
		expect(sanitizeHeaders(headers)['mcp-session-id']).toBe('[redacted]');
	});

	it('redacts extra headers provided by the caller', () => {
		const headers = new Headers({
			'x-vendor-key': 'vendor-secret',
			'content-type': 'application/json'
		});
		const result = sanitizeHeaders(headers, ['x-vendor-key']);
		expect(result['x-vendor-key']).toBe('[redacted]');
		expect(result['content-type']).toBe('application/json');
	});

	it('handles case-insensitive extra header names', () => {
		const headers = new Headers({ 'X-Custom-Token': 'token-value' });
		const result = sanitizeHeaders(headers, ['X-CUSTOM-TOKEN']);
		expect(result['x-custom-token']).toBe('[redacted]');
	});
});
