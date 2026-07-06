import { afterEach, describe, expect, it, vi } from 'vitest';
import { Client } from '@modelcontextprotocol/sdk/client';
import { MCPService } from '$lib/services/mcp.service';
import { MCPConnectionPhase, MCPTransportType } from '$lib/enums';
import type { MCPConnectionLog, MCPServerConfig } from '$lib/types';
import { CORS_PROXY_HEADER_PREFIX } from '$lib/constants';

type DiagnosticFetchFactory = (
	serverName: string,
	config: MCPServerConfig,
	baseInit: RequestInit,
	targetUrl: URL,
	useProxy: boolean,
	onLog?: (log: MCPConnectionLog) => void
) => { fetch: typeof fetch; disable: () => void };

const createDiagnosticFetch = (
	config: MCPServerConfig,
	onLog?: (log: MCPConnectionLog) => void,
	baseInit: RequestInit = {},
	useProxy = false
) =>
	(
		MCPService as unknown as { createDiagnosticFetch: DiagnosticFetchFactory }
	).createDiagnosticFetch('test-server', config, baseInit, new URL(config.url), useProxy, onLog);

describe('MCPService', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it('stops transport phase logging after handshake diagnostics are disabled', async () => {
		const logs: MCPConnectionLog[] = [];
		const response = new Response('{}', {
			status: 200,
			headers: { 'content-type': 'application/json' }
		});

		vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response));

		const config: MCPServerConfig = {
			url: 'https://example.com/mcp',
			transport: MCPTransportType.STREAMABLE_HTTP
		};

		const controller = createDiagnosticFetch(config, (log) => logs.push(log));

		await controller.fetch(config.url, { method: 'POST', body: '{}' });
		expect(logs).toHaveLength(2);
		expect(logs.every((log) => log.message.includes('https://example.com/mcp'))).toBe(true);

		controller.disable();
		await controller.fetch(config.url, { method: 'POST', body: '{}' });

		expect(logs).toHaveLength(2);
	});

	it('redacts all configured custom headers in diagnostic request logs', async () => {
		const logs: MCPConnectionLog[] = [];
		const response = new Response('{}', {
			status: 200,
			headers: { 'content-type': 'application/json' }
		});

		vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response));

		const config: MCPServerConfig = {
			url: 'https://example.com/mcp',
			transport: MCPTransportType.STREAMABLE_HTTP,
			headers: {
				'x-auth-token': 'secret-token',
				'x-vendor-api-key': 'secret-key'
			}
		};

		const controller = createDiagnosticFetch(config, (log) => logs.push(log), {
			headers: config.headers
		});

		await controller.fetch(config.url, {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: '{}'
		});

		expect(logs).toHaveLength(2);
		expect(logs[0].details).toMatchObject({
			request: {
				headers: {
					'x-auth-token': '[redacted]',
					'x-vendor-api-key': '[redacted]',
					'content-type': 'application/json'
				}
			}
		});
	});

	it('wraps dynamic request headers when using the CORS proxy', async () => {
		const logs: MCPConnectionLog[] = [];
		const proxiedAuthToken = `${CORS_PROXY_HEADER_PREFIX}x-auth-token`;
		const proxiedContentType = `${CORS_PROXY_HEADER_PREFIX}content-type`;
		const proxiedSessionId = `${CORS_PROXY_HEADER_PREFIX}mcp-session-id`;
		const response = new Response('{}', {
			status: 200,
			headers: { 'content-type': 'application/json' }
		});
		const fetchMock = vi.fn().mockResolvedValue(response);

		vi.stubGlobal('fetch', fetchMock);

		const config: MCPServerConfig = {
			url: 'https://example.com/mcp',
			transport: MCPTransportType.STREAMABLE_HTTP,
			useProxy: true
		};

		const controller = createDiagnosticFetch(
			config,
			(log) => logs.push(log),
			{
				headers: {
					authorization: 'Bearer llama-server-key',
					[proxiedAuthToken]: 'target-token'
				}
			},
			true
		);

		await controller.fetch('http://localhost:8080/cors-proxy?url=https%3A%2F%2Fexample.com%2Fmcp', {
			method: 'POST',
			headers: {
				'content-type': 'application/json',
				'mcp-session-id': 'session-request-12345'
			},
			body: '{}'
		});

		const sentHeaders = fetchMock.mock.calls[0]?.[1]?.headers as Headers;
		expect(sentHeaders.get('authorization')).toBe('Bearer llama-server-key');
		expect(sentHeaders.get(proxiedAuthToken)).toBe('target-token');
		expect(sentHeaders.get(proxiedContentType)).toBe('application/json');
		expect(sentHeaders.get(proxiedSessionId)).toBe('session-request-12345');
		expect(sentHeaders.has('content-type')).toBe(false);
		expect(sentHeaders.has('mcp-session-id')).toBe(false);
		expect(logs[0].details).toMatchObject({
			request: {
				headers: {
					authorization: '[redacted]',
					[proxiedAuthToken]: '[redacted]',
					[proxiedSessionId]: '....12345'
				}
			}
		});
	});

	it('DELETE request with CORS proxy should return a fake 200 response', async () => {
		const logs: MCPConnectionLog[] = [];
		const fetchMock = vi.fn();

		vi.stubGlobal('fetch', fetchMock);

		const config: MCPServerConfig = {
			url: 'https://example.com/mcp',
			transport: MCPTransportType.STREAMABLE_HTTP,
			useProxy: true
		};

		const controller = createDiagnosticFetch(config, (log) => logs.push(log), {}, true);

		const response = await controller.fetch(
			'http://localhost:8080/cors-proxy?url=https%3A%2F%2Fexample.com%2Fmcp',
			{ method: 'DELETE' }
		);

		expect(fetchMock).not.toHaveBeenCalled();
		expect(response.status).toBe(200);
		expect(logs.at(-1)?.details).toMatchObject({
			response: { status: 200, isFake: true }
		});
	});

	it('partially redacts mcp-session-id in diagnostic request and response logs', async () => {
		const logs: MCPConnectionLog[] = [];
		const response = new Response('{}', {
			status: 200,
			headers: {
				'content-type': 'application/json',
				'mcp-session-id': 'session-response-67890'
			}
		});

		vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response));

		const config: MCPServerConfig = {
			url: 'https://example.com/mcp',
			transport: MCPTransportType.STREAMABLE_HTTP
		};

		const controller = createDiagnosticFetch(config, (log) => logs.push(log));

		await controller.fetch(config.url, {
			method: 'POST',
			headers: {
				'content-type': 'application/json',
				'mcp-session-id': 'session-request-12345'
			},
			body: '{}'
		});

		expect(logs).toHaveLength(2);
		expect(logs[0].details).toMatchObject({
			request: {
				headers: {
					'content-type': 'application/json',
					'mcp-session-id': '....12345'
				}
			}
		});
		expect(logs[1].details).toMatchObject({
			response: {
				headers: {
					'content-type': 'application/json',
					'mcp-session-id': '....67890'
				}
			}
		});
	});

	it('extracts JSON-RPC methods without logging the raw request body', async () => {
		const logs: MCPConnectionLog[] = [];
		const response = new Response('{}', {
			status: 200,
			headers: { 'content-type': 'application/json' }
		});

		vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response));

		const config: MCPServerConfig = {
			url: 'https://example.com/mcp',
			transport: MCPTransportType.STREAMABLE_HTTP
		};

		const controller = createDiagnosticFetch(config, (log) => logs.push(log));

		await controller.fetch(config.url, {
			method: 'POST',
			body: JSON.stringify([
				{ jsonrpc: '2.0', id: 1, method: 'initialize' },
				{ jsonrpc: '2.0', method: 'notifications/initialized' }
			])
		});

		expect(logs[0].details).toMatchObject({
			request: {
				method: 'POST',
				body: {
					kind: 'string',
					size: expect.any(Number)
				},
				jsonRpcMethods: ['initialize', 'notifications/initialized']
			}
		});
	});

	it('adds a CORS hint to Failed to fetch diagnostic log messages', async () => {
		const logs: MCPConnectionLog[] = [];
		const fetchError = new TypeError('Failed to fetch');

		vi.stubGlobal('fetch', vi.fn().mockRejectedValue(fetchError));

		const config: MCPServerConfig = {
			url: 'http://localhost:8000/mcp',
			transport: MCPTransportType.STREAMABLE_HTTP
		};

		const controller = createDiagnosticFetch(config, (log) => logs.push(log));

		await expect(controller.fetch(config.url, { method: 'POST', body: '{}' })).rejects.toThrow(
			'Failed to fetch'
		);

		expect(logs).toHaveLength(2);
		expect(logs[1].message).toBe(
			'HTTP POST http://localhost:8000/mcp failed: Failed to fetch (check CORS?)'
		);
	});

	it('detaches phase error logging after the initialize handshake completes', async () => {
		const phaseLogs: Array<{ phase: MCPConnectionPhase; log: MCPConnectionLog }> = [];
		const stopPhaseLogging = vi.fn();
		let emitClientError: ((error: Error) => void) | undefined;

		vi.spyOn(MCPService, 'createTransport').mockReturnValue({
			transport: {} as never,
			type: MCPTransportType.WEBSOCKET,
			stopPhaseLogging
		});
		vi.spyOn(MCPService, 'listTools').mockResolvedValue([]);
		vi.spyOn(Client.prototype, 'getServerVersion').mockReturnValue(undefined);
		vi.spyOn(Client.prototype, 'getServerCapabilities').mockReturnValue(undefined);
		vi.spyOn(Client.prototype, 'getInstructions').mockReturnValue(undefined);
		vi.spyOn(Client.prototype, 'connect').mockImplementation(async function (this: Client) {
			emitClientError = (error: Error) => this.onerror?.(error);
			this.onerror?.(new Error('handshake protocol error'));
		});

		await MCPService.connect(
			'test-server',
			{
				url: 'ws://example.com/mcp',
				transport: MCPTransportType.WEBSOCKET
			},
			undefined,
			undefined,
			(phase, log) => phaseLogs.push({ phase, log })
		);

		expect(stopPhaseLogging).toHaveBeenCalledTimes(1);
		expect(
			phaseLogs.filter(
				({ phase, log }) =>
					phase === MCPConnectionPhase.ERROR &&
					log.message === 'Protocol error: handshake protocol error'
			)
		).toHaveLength(1);

		emitClientError?.(new Error('runtime protocol error'));

		expect(
			phaseLogs.filter(
				({ phase, log }) =>
					phase === MCPConnectionPhase.ERROR &&
					log.message === 'Protocol error: runtime protocol error'
			)
		).toHaveLength(0);
	});
});
