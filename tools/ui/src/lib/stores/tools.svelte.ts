import {
	buildSandboxToolDefinition,
	DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY,
	HOME_TILDE,
	TOOL_GROUP_LABELS,
	TOOL_SERVER_LABELS
} from '$lib/constants';
import {
	BuiltInTool,
	GlobSearchType,
	HealthCheckStatus,
	JsonSchemaType,
	ToolCallType,
	ToolSource
} from '$lib/enums';
import { ToolsService } from '$lib/services/tools.service';
import { mcpStore } from '$lib/stores/mcp.svelte';
import { config } from '$lib/stores/settings.svelte';
import type { OpenAIToolDefinition, ToolEntry, ToolGroup } from '$lib/types';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';

/** Stable selection identity for a tool, shared by the disabled set and the permission store */

class ToolsStore {
	private _builtinTools = $state<OpenAIToolDefinition[]>([]);
	private _loading = $state(false);
	private _error = $state<string | null>(null);
	private _disabledTools = $state(new SvelteSet<string>());
	// builtin tools that resolve their paths against the working directory,
	// as declared by the server in its `/tools` listing
	private _cwdAwareTools = $state(new SvelteSet<string>());
	private _toolsEndpointUnreachable = $state(false);
	private _serverHome = $state<string | null | undefined>(undefined);

	constructor() {
		try {
			const stored = localStorage.getItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY);

			if (stored) {
				const parsed = JSON.parse(stored);

				if (Array.isArray(parsed)) {
					for (const key of parsed) {
						if (typeof key === 'string') this._disabledTools.add(key);
					}
				}
			}
		} catch (err) {
			console.error('[ToolsStore] Failed to load disabled tools from localStorage:', err);
		}

		this.fetchBuiltinTools();
	}

	private persistDisabledTools(): void {
		try {
			localStorage.setItem(
				DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY,
				JSON.stringify([...this._disabledTools])
			);
		} catch {
			// ignore storage errors
		}
	}

	private toolKey(source: ToolSource, name: string, serverId?: string): string {
		switch (source) {
			case ToolSource.MCP:
				return serverId ? `mcp-${serverId}:${name}` : `mcp:${name}`;
			case ToolSource.CUSTOM:
				return `custom:${name}`;
			case ToolSource.FRONTEND:
				return `frontend:${name}`;
			default:
				return `builtin:${name}`;
		}
	}

	private inferTypeFromDefault(value: unknown): string | undefined {
		if (typeof value === 'string') return 'string';

		if (typeof value === 'boolean') return 'boolean';

		if (typeof value === 'number') return Number.isInteger(value) ? 'integer' : 'number';

		if (Array.isArray(value)) return 'array';

		if (value !== null && typeof value === 'object') return 'object';

		return undefined;
	}

	/**
	 * Recursively normalize a JSON Schema object: infers `type` from `default`
	 * for properties / items that omit it, and descends into nested `properties`
	 * and `items`. Returns a new object -- does not mutate the input.
	 */
	private normalizeJsonSchema(schema: Record<string, unknown>): Record<string, unknown> {
		if (!schema || typeof schema !== 'object') return schema;

		const normalized: Record<string, unknown> = { ...schema };

		if (normalized.properties && typeof normalized.properties === 'object') {
			const props = normalized.properties as Record<string, Record<string, unknown>>;
			const normalizedProps: Record<string, Record<string, unknown>> = {};

			for (const [key, prop] of Object.entries(props)) {
				if (!prop || typeof prop !== 'object') {
					normalizedProps[key] = prop;

					continue;
				}

				const normalizedProp: Record<string, unknown> = { ...prop };

				if (!normalizedProp.type && normalizedProp.default !== undefined) {
					const inferred = this.inferTypeFromDefault(normalizedProp.default);

					if (inferred) normalizedProp.type = inferred;
				}

				if (normalizedProp.properties) {
					Object.assign(
						normalizedProp,
						this.normalizeJsonSchema(normalizedProp as Record<string, unknown>)
					);
				}

				if (normalizedProp.items && typeof normalizedProp.items === 'object') {
					normalizedProp.items = this.normalizeJsonSchema(
						normalizedProp.items as Record<string, unknown>
					);
				}

				normalizedProps[key] = normalizedProp;
			}
			normalized.properties = normalizedProps;
		}

		return normalized;
	}

	private mcpDefinition(
		name: string,
		description: string | undefined,
		schema?: Record<string, unknown>
	): OpenAIToolDefinition {
		return {
			function: {
				description,
				name,
				parameters: schema ?? { properties: {}, required: [], type: JsonSchemaType.OBJECT }
			},
			type: ToolCallType.FUNCTION
		};
	}

	get builtinTools(): OpenAIToolDefinition[] {
		return this._builtinTools;
	}

	get serverHome(): string | null {
		return this._serverHome ?? null;
	}

	get mcpTools(): OpenAIToolDefinition[] {
		return this.mcpEntries().map((e) => e.definition);
	}

	get frontendTools(): OpenAIToolDefinition[] {
		return config().jsSandboxEnabled
			? [buildSandboxToolDefinition(!!config().symbolicMathEnabled)]
			: [];
	}

	get customTools(): OpenAIToolDefinition[] {
		const raw = config().customJson;

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

	/** Normalize MCP tools from live connections when available, fall back to health check data */
	private mcpEntries(): {
		serverId: string;
		serverName: string;
		definition: OpenAIToolDefinition;
	}[] {
		const out: { serverId: string; serverName: string; definition: OpenAIToolDefinition }[] = [];
		const connections = mcpStore.getConnections();

		if (connections.size > 0) {
			for (const [serverId, connection] of connections) {
				const serverName = mcpStore.getServerDisplayName(serverId);

				for (const tool of connection.tools) {
					const rawSchema = (tool.inputSchema as Record<string, unknown>) ?? {
						properties: {},
						required: [],
						type: JsonSchemaType.OBJECT
					};

					out.push({
						definition: {
							function: {
								description: tool.description,
								name: tool.name,
								parameters: this.normalizeJsonSchema(rawSchema)
							},
							type: ToolCallType.FUNCTION
						},
						serverId,
						serverName
					});
				}
			}
		} else {
			for (const { serverId, serverName, tools } of this.getMcpToolsFromHealthChecks()) {
				for (const tool of tools) {
					out.push({
						definition: this.mcpDefinition(tool.name, tool.description),
						serverId,
						serverName
					});
				}
			}
		}

		return out;
	}

	/** Canonical flat list of tool entries with source metadata and stable keys, deduped by key */
	get allTools(): ToolEntry[] {
		const entries: ToolEntry[] = [];
		const seen = new SvelteSet<string>();
		const push = (entry: ToolEntry) => {
			if (seen.has(entry.key)) return;

			seen.add(entry.key);
			entries.push(entry);
		};

		for (const def of this._builtinTools) {
			const name = def.function.name;

			push({
				definition: def,
				key: this.toolKey(ToolSource.BUILTIN, name),
				source: ToolSource.BUILTIN
			});
		}

		for (const def of this.frontendTools) {
			const name = def.function.name;

			push({
				definition: def,
				key: this.toolKey(ToolSource.FRONTEND, name),
				source: ToolSource.FRONTEND
			});
		}

		for (const { definition, serverId, serverName } of this.mcpEntries()) {
			const name = definition.function.name;

			push({
				definition,
				key: this.toolKey(ToolSource.MCP, name, serverId),
				serverId,
				serverName,
				source: ToolSource.MCP
			});
		}

		for (const def of this.customTools) {
			const name = def.function.name;

			push({
				definition: def,
				key: this.toolKey(ToolSource.CUSTOM, name),
				source: ToolSource.CUSTOM
			});
		}

		return entries;
	}

	/** Tools grouped by category for tree display, derived from the canonical entries */
	get toolGroups(): ToolGroup[] {
		const groups: ToolGroup[] = [];
		const byKey = new SvelteMap<string, ToolGroup>();

		for (const entry of this.allTools) {
			const groupKey =
				entry.source === ToolSource.MCP ? `mcp:${entry.serverId ?? ''}` : entry.source;

			let group = byKey.get(groupKey);

			if (!group) {
				group = {
					key: groupKey,
					label: this.groupLabel(entry),
					serverId: entry.serverId,
					source: entry.source,
					tools: []
				};
				byKey.set(groupKey, group);
				groups.push(group);
			}

			group.tools.push(entry);
		}

		return groups;
	}

	private groupLabel(entry: ToolEntry): string {
		switch (entry.source) {
			case ToolSource.MCP:
				return entry.serverName ?? '';
			case ToolSource.CUSTOM:
				return TOOL_GROUP_LABELS[ToolSource.CUSTOM];
			case ToolSource.FRONTEND:
				return TOOL_GROUP_LABELS[ToolSource.FRONTEND];
			default:
				return TOOL_GROUP_LABELS[ToolSource.BUILTIN];
		}
	}

	/**
	 * Enabled tool definitions for sending to the LLM.
	 * MCP tool schemas are normalized here so the wire payload is consistent
	 * across all four sources (built-in, frontend/sandbox, MCP, custom JSON).
	 * The API identifies tools by name, so a name is sent at most once.
	 */
	getEnabledToolsForLLM(): OpenAIToolDefinition[] {
		const enabledNames = new SvelteSet<string>();

		for (const entry of this.allTools) {
			if (!this._disabledTools.has(entry.key)) {
				enabledNames.add(entry.definition.function.name);
			}
		}

		const result: OpenAIToolDefinition[] = [];
		const seen = new SvelteSet<string>();
		const take = (def: OpenAIToolDefinition) => {
			const name = def.function.name;

			if (!enabledNames.has(name) || seen.has(name)) return;

			seen.add(name);
			result.push(def);
		};

		for (const def of this._builtinTools) take(def);
		for (const def of this.frontendTools) take(def);
		// mcpEntries() over mcpStore directly so wire shape stays normalized and aligned with the tools UI.
		for (const entry of this.mcpEntries()) take(entry.definition);
		for (const def of this.customTools) take(def);

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

	isToolEnabled(key: string): boolean {
		return !this._disabledTools.has(key);
	}

	toggleTool(key: string): void {
		if (this._disabledTools.has(key)) {
			this._disabledTools.delete(key);
		} else {
			this._disabledTools.add(key);
		}

		this.persistDisabledTools();
	}

	setToolEnabled(key: string, enabled: boolean): void {
		if (enabled) {
			this._disabledTools.delete(key);
		} else {
			this._disabledTools.add(key);
		}
	}

	/** Enable all tools belonging to a specific MCP server */
	enableAllToolsForServer(serverId: string): void {
		const connection = mcpStore.getConnections().get(serverId);

		if (!connection) return;

		for (const tool of connection.tools) {
			this._disabledTools.delete(this.toolKey(ToolSource.MCP, tool.name, serverId));
		}
		this.persistDisabledTools();
	}

	toggleGroup(group: ToolGroup): void {
		const allEnabled = group.tools.every((t) => this.isToolEnabled(t.key));
		const target = !allEnabled;

		for (const tool of group.tools) {
			if (target) this._disabledTools.delete(tool.key);
			else this._disabledTools.add(tool.key);
		}
		this.persistDisabledTools();
	}

	isGroupFullyEnabled(group: ToolGroup): boolean {
		return group.tools.length > 0 && group.tools.every((t) => this.isToolEnabled(t.key));
	}

	/** Get MCP tools from health check data, used when live connections aren't established yet */
	private getMcpToolsFromHealthChecks(): {
		serverId: string;
		serverName: string;
		tools: { name: string; description?: string }[];
	}[] {
		const result: ReturnType<ToolsStore['getMcpToolsFromHealthChecks']> = [];

		for (const server of mcpStore.getServers()) {
			if (!server.enabled) continue;

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

	/** First canonical entry matching a tool name, runtime tool calls resolve by name */
	private findEntryByName(toolName: string): ToolEntry | null {
		for (const entry of this.allTools) {
			if (entry.definition.function.name === toolName) return entry;
		}

		return null;
	}

	/** Determine the source of a tool by its name */
	getToolSource(toolName: string): ToolSource | null {
		return this.findEntryByName(toolName)?.source ?? null;
	}

	/** Get the display label for the server that owns a given tool */
	getToolServerLabel(toolName: string): string {
		const entry = this.findEntryByName(toolName);

		if (!entry) return '';

		if (entry.serverName) return mcpStore.getServerDisplayName(entry.serverName);

		if (entry.source === ToolSource.BUILTIN) return TOOL_SERVER_LABELS[ToolSource.BUILTIN];

		if (entry.source === ToolSource.CUSTOM) return TOOL_SERVER_LABELS[ToolSource.CUSTOM];

		if (entry.source === ToolSource.FRONTEND) return TOOL_SERVER_LABELS[ToolSource.FRONTEND];

		return '';
	}

	/** Permission key for a tool name, identical to the selection key */
	getPermissionKey(toolName: string): string | null {
		return this.findEntryByName(toolName)?.key ?? null;
	}

	/** Check if there are any enabled tools available (builtin, MCP, or custom) */
	get hasEnabledTools(): boolean {
		return this.getEnabledToolsForLLM().length > 0;
	}

	/**
	 * Check if a working directory is worth setting: at least one builtin tool
	 * that reads it is both served and left enabled by the user.
	 */
	get hasEnabledCwdTools(): boolean {
		return this._builtinTools.some((def) => {
			const name = def.function.name;

			return (
				this._cwdAwareTools.has(name) &&
				!this._disabledTools.has(this.toolKey(ToolSource.BUILTIN, name))
			);
		});
	}

	async fetchBuiltinTools(): Promise<void> {
		if (this._loading) return;

		this._loading = true;
		this._error = null;
		this._toolsEndpointUnreachable = false;

		try {
			const toolInfos = await ToolsService.list();

			this._builtinTools = toolInfos.map((info) => info.definition);
			this._cwdAwareTools = new SvelteSet(
				toolInfos.filter((info) => info.uses_cwd).map((info) => info.tool)
			);
		} catch (err) {
			const errorMessage = err instanceof Error ? err.message : String(err);

			this._error = errorMessage;

			// 403 from /tools means the server was started without --tools
			// TODO: check status code instead of relying on message
			if (errorMessage.includes('this feature is disabled')) {
				this._toolsEndpointUnreachable = true;
				console.info('[ToolsStore] Built-in tools are disabled on the server');
			} else {
				console.error('[ToolsStore] Failed to fetch built-in tools:', err);
			}
		} finally {
			this._loading = false;
		}
	}

	/**
	 * Absolute home directory on the server, resolved once per session via
	 * file_glob_search's `base` field (the server expands `~`). Anchors the
	 * directory picker's search scope and the `~` abbreviation of cwd
	 * displays. Returns null when tools are unavailable.
	 */
	async resolveServerHome(): Promise<string | null> {
		if (this._serverHome !== undefined) return this._serverHome;

		try {
			const res = await ToolsService.executeToolRaw(BuiltInTool.FILE_GLOB_SEARCH, {
				limit: 1,
				max_depth: 1,
				path: HOME_TILDE,
				type: GlobSearchType.DIR
			});

			this._serverHome = typeof res.base === 'string' ? res.base : null;
		} catch {
			// searches still work via a literal `~`, only `~` abbreviation degrades
			this._serverHome = null;
		}

		return this._serverHome;
	}
}

export const toolsStore = new ToolsStore();

export const allTools = () => toolsStore.allTools;
export const allToolDefinitions = () => toolsStore.allToolDefinitions;
export const toolGroups = () => toolsStore.toolGroups;
