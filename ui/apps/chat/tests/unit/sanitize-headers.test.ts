import { CORS_PROXY } from '$lib/constants';
import { sanitizeHeaders } from '$lib/utils/api-headers';
import { describe, expect, it } from 'vitest';

describe('sanitizeHeaders', () => {
	it('returns empty object for undefined input', () => {
		expect(sanitizeHeaders()).toEqual({});
	});

	it('passes through non-sensitive headers', () => {
		const headers = new Headers({ accept: 'text/html', 'content-type': 'application/json' });

		expect(sanitizeHeaders(headers)).toEqual({
			accept: 'text/html',
			'content-type': 'application/json'
		});
	});

	it('redacts known sensitive headers', () => {
		const headers = new Headers({
			authorization: 'Bearer secret',
			'content-type': 'application/json',
			'x-api-key': 'key-123'
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
			'content-type': 'application/json',
			'x-vendor-key': 'vendor-secret'
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

	it('redacts proxied sensitive and custom target headers', () => {
		const proxiedAuthorization = `${CORS_PROXY.HEADER_PREFIX}authorization`;
		const proxiedSessionId = `${CORS_PROXY.HEADER_PREFIX}mcp-session-id`;
		const proxiedVendorKey = `${CORS_PROXY.HEADER_PREFIX}x-vendor-key`;
		const headers = new Headers({
			[proxiedAuthorization]: 'Bearer secret',
			[proxiedSessionId]: 'session-12345',
			[proxiedVendorKey]: 'vendor-secret'
		});
		const partial = new Map([['mcp-session-id', 5]]);
		const result = sanitizeHeaders(headers, ['x-vendor-key'], partial);

		expect(result[proxiedAuthorization]).toBe('[redacted]');
		expect(result[proxiedSessionId]).toBe('....12345');
		expect(result[proxiedVendorKey]).toBe('[redacted]');
	});
});
