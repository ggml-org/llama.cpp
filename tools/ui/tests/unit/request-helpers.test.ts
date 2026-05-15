import { describe, expect, it } from 'vitest';
import {
	getRequestUrl,
	getRequestMethod,
	getRequestBody,
	summarizeRequestBody,
	formatDiagnosticErrorMessage,
	extractJsonRpcMethods
} from '$lib/utils/request-helpers';

describe('getRequestUrl', () => {
	it('returns a plain string input as-is', () => {
		expect(getRequestUrl('https://example.com/mcp')).toBe('https://example.com/mcp');
	});

	it('returns href from a URL object', () => {
		expect(getRequestUrl(new URL('https://example.com/mcp'))).toBe('https://example.com/mcp');
	});

	it('returns url from a Request object', () => {
		const req = new Request('https://example.com/mcp');
		expect(getRequestUrl(req)).toBe('https://example.com/mcp');
	});
});

describe('getRequestMethod', () => {
	it('prefers method from init', () => {
		expect(getRequestMethod('https://example.com', { method: 'POST' })).toBe('POST');
	});

	it('falls back to Request.method', () => {
		const req = new Request('https://example.com', { method: 'PUT' });
		expect(getRequestMethod(req)).toBe('PUT');
	});

	it('falls back to baseInit.method', () => {
		expect(getRequestMethod('https://example.com', undefined, { method: 'DELETE' })).toBe('DELETE');
	});

	it('defaults to GET', () => {
		expect(getRequestMethod('https://example.com')).toBe('GET');
	});
});

describe('getRequestBody', () => {
	it('returns body from init', () => {
		expect(getRequestBody('https://example.com', { body: 'payload' })).toBe('payload');
	});

	it('returns undefined when no body is present', () => {
		expect(getRequestBody('https://example.com')).toBeUndefined();
	});
});

describe('summarizeRequestBody', () => {
	it('returns empty for null', () => {
		expect(summarizeRequestBody(null)).toEqual({ kind: 'empty' });
	});

	it('returns empty for undefined', () => {
		expect(summarizeRequestBody(undefined)).toEqual({ kind: 'empty' });
	});

	it('returns string kind with size', () => {
		expect(summarizeRequestBody('hello')).toEqual({ kind: 'string', size: 5 });
	});

	it('returns blob kind with size', () => {
		const blob = new Blob(['abc']);
		expect(summarizeRequestBody(blob)).toEqual({ kind: 'blob', size: 3 });
	});

	it('returns formdata kind', () => {
		expect(summarizeRequestBody(new FormData())).toEqual({ kind: 'formdata' });
	});

	it('returns arraybuffer kind with size', () => {
		expect(summarizeRequestBody(new ArrayBuffer(8))).toEqual({ kind: 'arraybuffer', size: 8 });
	});
});

describe('formatDiagnosticErrorMessage', () => {
	it('appends CORS hint for Failed to fetch', () => {
		expect(formatDiagnosticErrorMessage(new TypeError('Failed to fetch'))).toBe(
			'Failed to fetch (check CORS?)'
		);
	});

	it('passes through other error messages unchanged', () => {
		expect(formatDiagnosticErrorMessage(new Error('timeout'))).toBe('timeout');
	});

	it('handles non-Error values', () => {
		expect(formatDiagnosticErrorMessage('some string')).toBe('some string');
	});
});

describe('extractJsonRpcMethods', () => {
	it('extracts methods from a JSON-RPC array', () => {
		const body = JSON.stringify([
			{ jsonrpc: '2.0', id: 1, method: 'initialize' },
			{ jsonrpc: '2.0', method: 'notifications/initialized' }
		]);
		expect(extractJsonRpcMethods(body)).toEqual(['initialize', 'notifications/initialized']);
	});

	it('extracts method from a single JSON-RPC message', () => {
		const body = JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/list' });
		expect(extractJsonRpcMethods(body)).toEqual(['tools/list']);
	});

	it('returns undefined for non-string body', () => {
		expect(extractJsonRpcMethods(null)).toBeUndefined();
		expect(extractJsonRpcMethods(undefined)).toBeUndefined();
	});

	it('returns undefined for invalid JSON', () => {
		expect(extractJsonRpcMethods('not json')).toBeUndefined();
	});

	it('returns undefined when no methods found', () => {
		expect(extractJsonRpcMethods(JSON.stringify({ foo: 'bar' }))).toBeUndefined();
	});
});
