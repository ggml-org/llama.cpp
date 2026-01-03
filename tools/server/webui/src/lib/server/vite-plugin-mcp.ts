// [AI] Vite plugin for MCP integration - HTTP API middleware
import type { Plugin } from 'vite';
import { MCPBridge } from './mcp-bridge';
import type {
	MCPServer,
	MCPServerListResponse,
	MCPToolsListResponse,
	MCPToolExecutionRequest,
	MCPToolExecutionResult
} from '../types';

/**
 * Vite plugin for MCP server management
 *
 * Provides HTTP endpoints for MCP server and tool management:
 * - GET  /mcp/servers - List all configured MCP servers
 * - POST /mcp/servers - Add a new MCP server
 * - PUT  /mcp/servers/:id - Update MCP server configuration
 * - DELETE /mcp/servers/:id - Remove MCP server
 * - GET  /mcp/tools - List all available tools from enabled servers
 * - POST /mcp/execute - Execute a tool call
 */
export function mcpPlugin(): Plugin {
	let bridge: MCPBridge | null = null;

	return {
		name: 'llama.cpp:mcp',
		apply: 'serve' as const,

		configureServer(server) {
			bridge = new MCPBridge();

			// Cleanup on server close
			server.httpServer?.on('close', async () => {
				await bridge?.shutdown();
			});

			// Add middleware for MCP endpoints
			server.middlewares.use(async (req, res, next) => {
				if (!req.url?.startsWith('/mcp/')) {
					return next();
				}

				const url = new URL(req.url, `http://${req.headers.host}`);
				const path = url.pathname;

				try {
					// GET /mcp/servers - List servers
					if (path === '/mcp/servers' && req.method === 'GET') {
						const servers = await bridge!.listServers();
						const response: MCPServerListResponse = { servers };
						res.setHeader('Content-Type', 'application/json');
						res.end(JSON.stringify(response));
						return;
					}

					// POST /mcp/servers - Add server
					if (path === '/mcp/servers' && req.method === 'POST') {
						const body = await readBody(req);
						const serverConfig = JSON.parse(body) as Omit<MCPServer, 'status'>;
						const server = await bridge!.addServer(serverConfig);
						res.setHeader('Content-Type', 'application/json');
						res.end(JSON.stringify(server));
						return;
					}

					// PUT /mcp/servers/:id - Update server
					const updateMatch = path.match(/^\/mcp\/servers\/([^/]+)$/);
					if (updateMatch && req.method === 'PUT') {
						const serverId = decodeURIComponent(updateMatch[1]);
						const body = await readBody(req);
						const updates = JSON.parse(body) as Partial<MCPServer>;
						const server = await bridge!.updateServer(serverId, updates);
						res.setHeader('Content-Type', 'application/json');
						res.end(JSON.stringify(server));
						return;
					}

					// DELETE /mcp/servers/:id - Remove server
					const deleteMatch = path.match(/^\/mcp\/servers\/([^/]+)$/);
					if (deleteMatch && req.method === 'DELETE') {
						const serverId = decodeURIComponent(deleteMatch[1]);
						await bridge!.removeServer(serverId);
						res.setHeader('Content-Type', 'application/json');
						res.end(JSON.stringify({ success: true }));
						return;
					}

					// GET /mcp/tools - List tools
					if (path === '/mcp/tools' && req.method === 'GET') {
						const tools = await bridge!.listTools();
						const response: MCPToolsListResponse = { tools };
						res.setHeader('Content-Type', 'application/json');
						res.end(JSON.stringify(response));
						return;
					}

					// POST /mcp/execute - Execute tool
					if (path === '/mcp/execute' && req.method === 'POST') {
						const body = await readBody(req);
						const request = JSON.parse(body) as MCPToolExecutionRequest;
						const result = await bridge!.executeTool(request);
						res.setHeader('Content-Type', 'application/json');
						res.end(JSON.stringify(result));
						return;
					}

					// Unknown MCP endpoint
					res.statusCode = 404;
					res.end(JSON.stringify({ error: 'Not found' }));
				} catch (error) {
					console.error('MCP endpoint error:', error);
					res.statusCode = 500;
					res.setHeader('Content-Type', 'application/json');
					res.end(JSON.stringify({
						error: error instanceof Error ? error.message : 'Internal server error'
					}));
				}
			});
		}
	};
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

function readBody(req: NodeJS.ReadableStream): Promise<string> {
	return new Promise((resolve, reject) => {
		const chunks: Buffer[] = [];
		req.on('data', (chunk) => chunks.push(chunk));
		req.on('end', () => resolve(Buffer.concat(chunks).toString()));
		req.on('error', reject);
	});
}
