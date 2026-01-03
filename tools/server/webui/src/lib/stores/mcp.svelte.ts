// [AI] MCP Store - Reactive Svelte 5 store for MCP servers and tools
import { MCPService } from '$lib/services';
import type { MCPServer, MCPTool } from '$lib/types';
import { MCPServerStatus } from '$lib/types';

const MCP_SERVERS_STORAGE_KEY = 'mcp_servers';

/**
 * mcpStore - MCP server and tool state management
 *
 * Manages:
 * - List of configured MCP servers
 * - Available tools from enabled servers
 * - Server enable/disable state
 * - Persistence to localStorage
 */
class MCPStore {
	servers = $state<MCPServer[]>([]);
	tools = $state<MCPTool[]>([]);
	isLoading = $state(false);
	error = $state<string | null>(null);

	constructor() {
		// Load servers from localStorage on construction
		this.loadFromLocalStorage();
	}

	/**
	 * Load servers from localStorage
	 */
	private loadFromLocalStorage(): void {
		if (typeof window === 'undefined') return;

		try {
			const stored = localStorage.getItem(MCP_SERVERS_STORAGE_KEY);
			if (stored) {
				const parsed = JSON.parse(stored) as Array<Omit<MCPServer, 'status'>>;
				// Don't set servers directly - they'll be synced with backend during initialize
				// Just store them for use during initialization
				this.pendingServers = parsed;
			}
		} catch (error) {
			console.error('Failed to load MCP servers from localStorage:', error);
		}
	}

	/**
	 * Save servers to localStorage
	 */
	private saveToLocalStorage(): void {
		if (typeof window === 'undefined') return;

		try {
			// Store servers without status (status is runtime only)
			const toStore = this.servers.map(({ status, ...rest }) => rest);
			localStorage.setItem(MCP_SERVERS_STORAGE_KEY, JSON.stringify(toStore));
		} catch (error) {
			console.error('Failed to save MCP servers to localStorage:', error);
		}
	}

	private pendingServers: Array<Omit<MCPServer, 'status'>> = [];

	/**
	 * Initialize store by syncing with backend and loading persisted servers
	 */
	async initialize(): Promise<void> {
		console.log("[MCPStore] Initializing, pending servers:", this.pendingServers.length);
		this.isLoading = true;
		this.error = null;

		try {
			// First, restore any persisted servers to the backend
			if (this.pendingServers.length > 0) {
				// Fetch existing servers from backend to avoid duplicates
				const existingServers = await MCPService.listServers();
				const existingIds = new Set(existingServers.map(s => s.id));
				
				for (const server of this.pendingServers) {
					try {
						// Only add if server doesn't already exist in backend
						if (!existingIds.has(server.id)) {
							await MCPService.addServer(server);
						} else {
							console.log('[MCPStore] Server already exists in backend:', server.id);
						}
					} catch (error) {
						// Server might already exist, that's okay
						console.warn('Failed to restore server:', server.id, error);
					}
				}
				console.log("[MCPStore] Restored servers from localStorage");
				this.pendingServers = [];
			}

			// Then fetch current state from backend
			await this.fetchServers();
			console.log("[MCPStore] Fetched servers from backend:", this.servers.length, "servers");

			// [AI] Ensure all enabled servers are running (handles page reload case)
			for (const server of this.servers) {
				if (server.enabled && server.status !== 'running' && server.status !== 'starting') {
					console.log('[MCPStore] Starting enabled server after reload:', server.id);
					try {
						await MCPService.updateServer(server.id, { enabled: true });
					} catch (error) {
						console.error('[MCPStore] Failed to start server:', server.id, error);
					}
				}
			}
			
			await this.fetchTools();
		} catch (error) {
			this.error = error instanceof Error ? error.message : 'Failed to initialize MCP';
			console.error('MCP initialization error:', error);
		} finally {
			this.isLoading = false;
		}
	}

	/**
	 * Fetch list of configured MCP servers
	 */
	async fetchServers(): Promise<void> {
		try {
			this.servers = await MCPService.listServers();
		} catch (error) {
			console.error('Failed to fetch MCP servers:', error);
			throw error;
		}
	}

	/**
	 * Fetch available tools from enabled servers
	 */
	async fetchTools(): Promise<void> {
		try {
		console.log("Fetching MCP tools...");
			this.tools = await MCPService.listTools();
			console.log("Fetched MCP tools:", this.tools);
		} catch (error) {
			console.error('Failed to fetch MCP tools:', error);
			// Don't throw - tools list can be empty
			this.tools = [];
		}
	}

	/**
	 * Add a new MCP server
	 */
	async addServer(server: Omit<MCPServer, 'status'>): Promise<void> {
		try {
			const newServer = await MCPService.addServer(server);
			this.servers = [...this.servers, newServer];
			this.saveToLocalStorage();

			if (newServer.enabled) {
				await this.fetchTools();
			}
		} catch (error) {
			this.error = error instanceof Error ? error.message : 'Failed to add server';
			console.error('Failed to add MCP server:', error);
			throw error;
		}
	}

	/**
	 * Update an existing MCP server
	 */
	async updateServer(serverId: string, updates: Partial<MCPServer>): Promise<void> {
		try {
			const updatedServer = await MCPService.updateServer(serverId, updates);
			this.servers = this.servers.map((s) => (s.id === serverId ? updatedServer : s));
			this.saveToLocalStorage();

			// Refetch tools if enabled state changed
			if ('enabled' in updates) {
				await this.fetchTools();
			}
		} catch (error) {
			this.error = error instanceof Error ? error.message : 'Failed to update server';
			console.error('Failed to update MCP server:', error);
			throw error;
		}
	}

	/**
	 * Remove an MCP server
	 */
	async removeServer(serverId: string): Promise<void> {
		try {
			await MCPService.removeServer(serverId);
			this.servers = this.servers.filter((s) => s.id !== serverId);
			this.saveToLocalStorage();
			await this.fetchTools();
		} catch (error) {
			this.error = error instanceof Error ? error.message : 'Failed to remove server';
			console.error('Failed to remove MCP server:', error);
			throw error;
		}
	}

	/**
	 * Get tools in OpenAI-compatible format for chat completion requests
	 */
	getToolsForAPI(): Array<{ type: 'function'; function: { name: string; description?: string; parameters?: Record<string, unknown> } }> {
		return this.tools.map((tool) => ({
			type: 'function' as const,
			function: {
				name: tool.function.name,
				description: tool.function.description,
				parameters: tool.function.parameters
			}
		}));
	}

	/**
	 * Check if any servers are enabled
	 */
	hasEnabledServers(): boolean {
		return this.servers.some((s) => s.enabled);
	}

	/**
	 * Check if tools are available
	 */
	hasTools(): boolean {
		return this.tools.length > 0;
	}
}

export const mcpStore = new MCPStore();
