/**
 * Fixture MCP servers for demo recordings.
 *
 * Serves a few stateless Streamable HTTP MCP servers with canned tools, so
 * demos show MCP groups without depending on real third-party servers.
 *
 * Usage: node demo/fixtures/mcp-servers.ts
 */

import { DEMO_MCP_PORT, DEMO_MCP_SERVERS, type DemoMcpServer } from './demo-config.ts';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { z } from 'zod';

function buildServer(fixture: DemoMcpServer): McpServer {
	const server = new McpServer({ name: fixture.name, title: fixture.title, version: '1.0.0' });

	for (const tool of fixture.tools) {
		server.registerTool(
			tool.name,
			{ description: tool.description, inputSchema: { query: z.string().optional() } },
			async () => ({ content: [{ text: tool.result, type: 'text' }] })
		);
	}

	return server;
}

function setCorsHeaders(res: ServerResponse): void {
	res.setHeader('Access-Control-Allow-Origin', '*');
	res.setHeader('Access-Control-Allow-Headers', '*');
	res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
	res.setHeader('Access-Control-Expose-Headers', 'mcp-session-id, mcp-protocol-version');
	// the UI runs with COEP require-corp
	res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
}

async function readBody(req: IncomingMessage): Promise<unknown> {
	const chunks: Buffer[] = [];

	for await (const chunk of req) chunks.push(chunk as Buffer);

	const raw = Buffer.concat(chunks).toString();

	return raw ? JSON.parse(raw) : undefined;
}

const httpServer = createServer(async (req, res) => {
	setCorsHeaders(res);

	if (req.method === 'OPTIONS') {
		res.writeHead(204).end();

		return;
	}

	const path = new URL(req.url ?? '/', 'http://localhost').pathname;
	const fixture = DEMO_MCP_SERVERS.find((s) => path === `/${s.id}/mcp`);

	if (!fixture) {
		res.writeHead(path === '/health' ? 200 : 404).end();

		return;
	}

	if (req.method !== 'POST') {
		// stateless: no standalone SSE stream or session teardown
		res.writeHead(405).end();

		return;
	}

	const server = buildServer(fixture);
	const transport = new StreamableHTTPServerTransport({
		enableJsonResponse: true,
		sessionIdGenerator: undefined
	});

	res.on('close', () => {
		transport.close();
		server.close();
	});

	await server.connect(transport);
	await transport.handleRequest(req, res, await readBody(req));
});

httpServer.listen(DEMO_MCP_PORT, () => {
	console.log(`demo MCP servers listening on http://localhost:${DEMO_MCP_PORT}`);
});
