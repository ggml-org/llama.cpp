import type { OpenAIToolDefinition, ToolEntry, ToolGroup } from '$lib/types';
import { ToolsService } from '$lib/services/tools.service';
import { mcpStore } from '$lib/stores/mcp.svelte';
import { HealthCheckStatus, JsonSchemaType, ToolCallType, ToolSource } from '$lib/enums';
import { config } from '$lib/stores/settings.svelte';
import {
	DISABLED_TOOLS_LOCALSTORAGE_KEY,
	TOOL_GROUP_LABELS,
	TOOL_SERVER_LABELS
} from '$lib/constants';
import { SvelteSet } from 'svelte/reactivity';

class ToolsStore {
	private _builtinTools = $state<OpenAIToolDefinition[]>([]);
	private _loading = $state(false);
	private _error = $state<string | null>(null);
	private _disabledTools = $state(new SvelteSet<string>());
	private _toolsEndpointUnreachable = $state(false);

	constructor() {
		try {
			const stored = localStorage.getItem(DISABLED_TOOLS_LOCALSTORAGE_KEY);
			if (stored) {
				const parsed = JSON.parse(stored);
				if (Array.isArray(parsed)) {
					for (const name of parsed) {
						if (typeof name === 'string') this._disabledTools.add(name);
					}
				}
			}
		} catch (err) {
			console.error('[ToolsStore] Failed to load disabled tools from localStorage:', err);
		}

		// Initialize builtin tools on startup
		this.fetchBuiltinTools();
	}

	private persistDisabledTools(): void {
		try {
			localStorage.setItem(
				DISABLED_TOOLS_LOCALSTORAGE_KEY,
				JSON.stringify([...this._disabledTools])
			);
		} catch {
			// ignore storage errors
		}
	}

	get builtinTools(): OpenAIToolDefinition[] {
		return this._builtinTools;
	}

	get mcpTools(): OpenAIToolDefinition[] {
		return mcpStore.getToolDefinitionsForLLM();
	}

	get customTools(): OpenAIToolDefinition[] {
		const raw = config().custom;
		if (!raw || typeof raw !== 'string') return [];

		try {
			const parsed = JSON.parse(raw);
			if (!Array.isArray(parsed)) return [];

			return parsed.filter(
				(t: unknown): t is OpenAIToolDefinition =>
					typeof t === 'object' &&
					t !== null &&
					'type' in t &&
					(t as OpenAIToolDefinition).type === 'function' &&
					'function' in t &&
					typeof (t as OpenAIToolDefinition).function?.name === 'string'
			);
		} catch {
			return [];
		}
	}

	/** Flat list of all tool entries with source metadata */
	get allTools(): ToolEntry[] {
		const entries: ToolEntry[] = [];

		for (const def of this._builtinTools) {
			entries.push({ source: ToolSource.BUILTIN, definition: def });
		}

		// Use live connections when available (full schema), fall back to health check data
		const connections = mcpStore.getConnections();
		if (connections.size > 0) {
			for (const [serverId, connection] of connections) {
				const serverName = mcpStore.getServerDisplayName(serverId);
				for (const tool of connection.tools) {
					const rawSchema = (tool.inputSchema as Record<string, unknown>) ?? {
						type: JsonSchemaType.OBJECT,
						properties: {},
						required: []
					};
					entries.push({
						source: ToolSource.MCP,
						serverName,
						serverId,
						definition: {
							type: ToolCallType.FUNCTION,
							function: {
								name: tool.name,
								description: tool.description,
								parameters: rawSchema
							}
						}
					});
				}
			}
		} else {
			for (const { serverId, serverName, tools } of this.getMcpToolsFromHealthChecks()) {
				for (const tool of tools) {
					entries.push({
						source: ToolSource.MCP,
						serverName,
						serverId,
						definition: {
							type: ToolCallType.FUNCTION,
							function: {
								name: tool.name,
								description: tool.description,
								parameters: { type: JsonSchemaType.OBJECT, properties: {}, required: [] }
							}
						}
					});
				}
			}
		}

		for (const def of this.customTools) {
			entries.push({ source: ToolSource.CUSTOM, definition: def });
		}

		return entries;
	}

	/** Tools grouped by category for tree display */
	get toolGroups(): ToolGroup[] {
		const groups: ToolGroup[] = [];

		if (this._builtinTools.length > 0) {
			groups.push({
				source: ToolSource.BUILTIN,
				label: TOOL_GROUP_LABELS[ToolSource.BUILTIN],
				tools: this._builtinTools
			});
		}

		// Use live connections when available, fall back to health check data
		const connections = mcpStore.getConnections();
		if (connections.size > 0) {
			for (const [serverId, connection] of connections) {
				if (connection.tools.length === 0) continue;
				const label = mcpStore.getServerDisplayName(serverId);
				const tools: OpenAIToolDefinition[] = connection.tools.map((tool) => {
					const rawSchema = (tool.inputSchema as Record<string, unknown>) ?? {
						type: JsonSchemaType.OBJECT,
						properties: {},
						required: []
					};
					return {
						type: ToolCallType.FUNCTION,
						function: {
							name: tool.name,
							description: tool.description,
							parameters: rawSchema
						}
					};
				});
				groups.push({ source: ToolSource.MCP, label, serverId, tools });
			}
		} else {
			for (const { serverId, serverName, tools } of this.getMcpToolsFromHealthChecks()) {
				if (tools.length === 0) continue;
				const defs: OpenAIToolDefinition[] = tools.map((tool) => ({
					type: ToolCallType.FUNCTION,
					function: {
						name: tool.name,
						description: tool.description,
						parameters: { type: JsonSchemaType.OBJECT, properties: {}, required: [] }
					}
				}));
				groups.push({ source: ToolSource.MCP, label: serverName, serverId, tools: defs });
			}
		}

		const custom = this.customTools;
		if (custom.length > 0) {
			groups.push({
				source: ToolSource.CUSTOM,
				label: TOOL_GROUP_LABELS[ToolSource.CUSTOM],
				tools: custom
			});
		}

		return groups;
	}

	/** Only enabled tool definitions (for sending to the API) */
	get enabledToolDefinitions(): OpenAIToolDefinition[] {
		return this.allTools
			.filter((t) => !this._disabledTools.has(t.definition.function.name))
			.map((t) => t.definition);
	}

	/**
	 * Returns enabled tool definitions for sending to the LLM.
	 * MCP tools use properly normalized schemas from mcpStore.
	 * Filters out tools disabled via the UI checkboxes.
	 */
	getEnabledToolsForLLM(): OpenAIToolDefinition[] {
		const disabled = this._disabledTools;
		const result: OpenAIToolDefinition[] = [];

		for (const tool of this._builtinTools) {
			if (!disabled.has(tool.function.name)) {
				result.push(tool);
			}
		}

		// MCP tools with properly normalized schemas
		for (const tool of mcpStore.getToolDefinitionsForLLM()) {
			if (!disabled.has(tool.function.name)) {
				result.push(tool);
			}
		}

		for (const tool of this.customTools) {
			if (!disabled.has(tool.function.name)) {
				result.push(tool);
			}
		}

		return result;
	}

	get allToolDefinitions(): OpenAIToolDefinition[] {
		return this.allTools.map((t) => t.definition);
	}

	get loading(): boolean {
		return this._loading;
	}

	get error(): string | null {
		return this._error;
	}

	get isToolsEndpointUnreachable(): boolean {
		return this._toolsEndpointUnreachable;
	}

	get disabledTools(): SvelteSet<string> {
		return this._disabledTools;
	}

	isToolEnabled(toolName: string): boolean {
		return !this._disabledTools.has(toolName);
	}

	toggleTool(toolName: string): void {
		if (this._disabledTools.has(toolName)) {
			this._disabledTools.delete(toolName);
		} else {
			this._disabledTools.add(toolName);
		}
		this.persistDisabledTools();
	}

	setToolEnabled(toolName: string, enabled: boolean): void {
		if (enabled) {
			this._disabledTools.delete(toolName);
		} else {
			this._disabledTools.add(toolName);
		}
	}

	/**
	 * Enable all tools belonging to a specific MCP server.
	 * Called when a server is enabled for a conversation.
	 */
	enableAllToolsForServer(serverId: string): void {
		const connection = mcpStore.getConnections().get(serverId);
		if (!connection) return;
		for (const tool of connection.tools) {
			this._disabledTools.delete(tool.name);
		}
		this.persistDisabledTools();
	}

	toggleGroup(group: ToolGroup): void {
		const allEnabled = group.tools.every((t) => this.isToolEnabled(t.function.name));
		for (const tool of group.tools) {
			this.setToolEnabled(tool.function.name, !allEnabled);
		}
		this.persistDisabledTools();
	}

	isGroupFullyEnabled(group: ToolGroup): boolean {
		return group.tools.length > 0 && group.tools.every((t) => this.isToolEnabled(t.function.name));
	}

	isGroupPartiallyEnabled(group: ToolGroup): boolean {
		const enabledCount = group.tools.filter((t) => this.isToolEnabled(t.function.name)).length;
		return enabledCount > 0 && enabledCount < group.tools.length;
	}

	/**
	 * Get MCP tools from health check data (reactive).
	 * Used when live connections aren't established yet.
	 */
	private getMcpToolsFromHealthChecks(): {
		serverId: string;
		serverName: string;
		tools: { name: string; description?: string }[];
	}[] {
		const result: ReturnType<ToolsStore['getMcpToolsFromHealthChecks']> = [];
		for (const server of mcpStore.getServersSorted().filter((s) => s.enabled)) {
			const health = mcpStore.getHealthCheckState(server.id);
			if (health.status === HealthCheckStatus.SUCCESS && health.tools.length > 0) {
				result.push({
					serverId: server.id,
					serverName: mcpStore.getServerLabel(server),
					tools: health.tools
				});
			}
		}
		return result;
	}

	/** Determine the source of a tool by its name. */
	getToolSource(toolName: string): ToolSource | null {
		if (this._builtinTools.some((t) => t.function.name === toolName)) {
			return ToolSource.BUILTIN;
		}
		for (const entry of this.allTools) {
			if (entry.definition.function.name === toolName) {
				return entry.source;
			}
		}
		return null;
	}

	/** Get the display label for the server that owns a given tool. */
	getToolServerLabel(toolName: string): string {
		for (const entry of this.allTools) {
			if (entry.definition.function.name === toolName) {
				if (entry.serverName) {
					return mcpStore.getServerDisplayName(entry.serverName);
				}
				if (entry.source === ToolSource.BUILTIN) {
					return TOOL_SERVER_LABELS[ToolSource.BUILTIN];
				}
				if (entry.source === ToolSource.CUSTOM) {
					return TOOL_SERVER_LABELS[ToolSource.CUSTOM];
				}
			}
		}
		return '';
	}

	/** Build a permission key with category prefix, e.g. "mcp-<serverId>:tool_name" */
	getPermissionKey(toolName: string): string | null {
		for (const entry of this.allTools) {
			if (entry.definition.function.name === toolName) {
				switch (entry.source) {
					case ToolSource.BUILTIN:
						return `builtin:${toolName}`;
					case ToolSource.CUSTOM:
						return `custom:${toolName}`;
					case ToolSource.MCP:
						if (entry.serverId) {
							return `mcp-${entry.serverId}:${toolName}`;
						}
						return `mcp:${toolName}`;
					default:
						return null;
				}
			}
		}
		return null;
	}

	/** Check if there are any enabled tools available (builtin, MCP, or custom). */
	get hasEnabledTools(): boolean {
		return this.getEnabledToolsForLLM().length > 0;
	}

	async fetchBuiltinTools(): Promise<void> {
		if (this._loading) return;

		this._loading = true;
		this._error = null;
		this._toolsEndpointUnreachable = false;

		try {
			const toolInfos = await ToolsService.list();
			this._builtinTools = toolInfos.map((info) => info.definition);
		} catch (err) {
			const errorMessage = err instanceof Error ? err.message : String(err);
			this._error = errorMessage;
			// 404 from /tools means the server was started without --tools
			if (errorMessage.includes('404') || errorMessage.toLowerCase().includes('not found')) {
				this._toolsEndpointUnreachable = true;
			}
			console.error('[ToolsStore] Failed to fetch built-in tools:', err);
		} finally {
			this._loading = false;
		}
	}
}

export const toolsStore = new ToolsStore();

export const allTools = () => toolsStore.allTools;
export const allToolDefinitions = () => toolsStore.allToolDefinitions;
export const enabledToolDefinitions = () => toolsStore.enabledToolDefinitions;
export const toolGroups = () => toolsStore.toolGroups;
