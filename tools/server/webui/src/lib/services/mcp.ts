// [AI] MCP Service - Client-side service layer for MCP server management
import { base } from '$app/paths';
import { getJsonHeaders } from '$lib/utils';
import type {
	MCPServer,
	MCPServerListResponse,
	MCPToolsListResponse,
	MCPTool,
	MCPToolExecutionRequest,
	MCPToolExecutionResult
} from '$lib/types';

/**
 * MCPService - Stateless service for MCP server and tool management
 *
 * This service handles communication with MCP-related endpoints:
 * - `/mcp/servers` - MCP server configuration and management
 * - `/mcp/tools` - Tool listing from enabled servers
 * - `/mcp/execute` - Tool execution requests
 *
 * **Responsibilities:**
 * - List/add/update/remove MCP servers
 * - Fetch available tools from running servers
 * - Execute tool calls and return results
 *
 * **Used by:**
 * - mcpStore: Primary consumer for MCP state management
 * - chatStore: For tool calling during conversations
 */
export class MCPService {
	// ─────────────────────────────────────────────────────────────────────────────
	// Server Management
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * List all configured MCP servers
	 */
	static async listServers(): Promise<MCPServer[]> {
		const response = await fetch(`${base}/mcp/servers`, {
			headers: getJsonHeaders()
		});

		if (!response.ok) {
			throw new Error(`Failed to fetch MCP servers (status ${response.status})`);
		}

		const data = (await response.json()) as MCPServerListResponse;
		return data.servers;
	}

	/**
	 * Add a new MCP server
	 * @param server - Server configuration (without status field)
	 */
	static async addServer(server: Omit<MCPServer, 'status'>): Promise<MCPServer> {
		const response = await fetch(`${base}/mcp/servers`, {
			method: 'POST',
			headers: getJsonHeaders(),
			body: JSON.stringify(server)
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => ({}));
			throw new Error(errorData.error || `Failed to add MCP server (status ${response.status})`);
		}

		return response.json() as Promise<MCPServer>;
	}

	/**
	 * Update an existing MCP server
	 * @param serverId - Server identifier
	 * @param updates - Partial server configuration to update
	 */
	static async updateServer(
		serverId: string,
		updates: Partial<MCPServer>
	): Promise<MCPServer> {
		const response = await fetch(`${base}/mcp/servers/${encodeURIComponent(serverId)}`, {
			method: 'PUT',
			headers: getJsonHeaders(),
			body: JSON.stringify(updates)
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => ({}));
			throw new Error(
				errorData.error || `Failed to update MCP server (status ${response.status})`
			);
		}

		return response.json() as Promise<MCPServer>;
	}

	/**
	 * Remove an MCP server
	 * @param serverId - Server identifier
	 */
	static async removeServer(serverId: string): Promise<void> {
		const response = await fetch(`${base}/mcp/servers/${encodeURIComponent(serverId)}`, {
			method: 'DELETE',
			headers: getJsonHeaders()
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => ({}));
			throw new Error(
				errorData.error || `Failed to remove MCP server (status ${response.status})`
			);
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Tool Management
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Fetch all available tools from enabled MCP servers
	 * Returns tools in OpenAI-compatible format
	 */
	static async listTools(): Promise<MCPTool[]> {
		const response = await fetch(`${base}/mcp/tools`, {
			headers: getJsonHeaders()
		});

		if (!response.ok) {
			throw new Error(`Failed to fetch MCP tools (status ${response.status})`);
		}

		const data = (await response.json()) as MCPToolsListResponse;
		return data.tools;
	}

	/**
	 * Execute a tool call on an MCP server
	 * @param request - Tool execution request with serverId, toolName, and arguments
	 */
	static async executeTool(request: MCPToolExecutionRequest): Promise<MCPToolExecutionResult> {
		const response = await fetch(`${base}/mcp/execute`, {
			method: 'POST',
			headers: getJsonHeaders(),
			body: JSON.stringify(request)
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => ({}));
			throw new Error(
				errorData.error || `Failed to execute tool (status ${response.status})`
			);
		}

		return response.json() as Promise<MCPToolExecutionResult>;
	}
}
