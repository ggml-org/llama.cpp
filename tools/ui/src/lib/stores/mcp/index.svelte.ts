/**
 * mcpStore - MCP host: server connections and tool operations
 *
 * Implements the MCP "Host" role, coordinating multiple server connections
 * and exposing a unified tool interface: lifecycle, name-conflict detection
 * and automatic tool-to-server routing. Owns connection state and raw
 * `Tool[]` per server; the OpenAI-compatible wire format is built in
 * toolsStore. Composes the health-check manager; uses MCPService for the
 * protocol layer.
 */

import type { ListChangedHandlers } from '@modelcontextprotocol/sdk/types.js';
import { browser } from '$app/environment';
import { SETTINGS_KEYS } from '$lib/constants';
import { CACHE, DEFAULT_MCP_CONFIG, MCP_RECONNECT, MCP_SERVER_ID_PREFIX } from '$lib/constants';
import { ColorMode, HealthCheckStatus, MCPConnectionPhase, MCPRefType } from '$lib/enums';
import { MCPService } from '$lib/services/mcp.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { MCPHealthCheckManager, type McpHealthHost } from '$lib/stores/mcp/health.svelte';
import { mcpResourceStore } from '$lib/stores/mcp/resources.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type {
	GetPromptResult,
	HealthCheckParams,
	HealthCheckState,
	MCPClientConfig,
	MCPConnection,
	MCPPromptInfo,
	MCPResourceAttachment,
	MCPResourceContent,
	MCPServerConfig,
	MCPServerDisplayInfo,
	MCPServerSettingsEntry,
	MCPToolCall,
	ServerStatus,
	Tool,
	ToolExecutionResult
} from '$lib/types';
import type { DatabaseMessageExtraMcpResource } from '$lib/types/database';
import type { SettingsConfigType } from '$lib/types/settings';
import {
	detectMcpTransportFromUrl,
	getMcpIconUrl,
	getMcpServerFaviconFallback,
	getMcpServerLabel,
	parseMcpServerSettings,
	uuid
} from '$lib/utils';
import { mode } from 'mode-watcher';

class MCPStore implements McpHealthHost {
	private _error = $state<string | null>(null);
	private _isInitializing = $state(false);
	private _toolCount = $state(0);
	private activeFlowCount = 0;

	private configSignature: string | null = null;
	private connectedServers = $state<string[]>([]);
	private connections = new Map<string, MCPConnection>();
	// health checks: per-server connectivity probes with optional promotion to active connections
	private health = new MCPHealthCheckManager(this);
	private initPromise: Promise<boolean> | null = null;
	private reconnectingServers = new Set<string>(); // Guard against concurrent reconnections
	private serverConfigs = new Map<string, MCPServerConfig>(); // Store configs for reconnection
	private serversCache: { raw: unknown; servers: MCPServerSettingsEntry[] } | null = null;
	private toolsIndex = new Map<string, string>();

	get availableTools(): string[] {
		return Array.from(this.toolsIndex.keys());
	}

	get connectedServerCount(): number {
		return this.connectedServers.length;
	}

	get connectedServerNames(): string[] {
		return this.connectedServers;
	}

	get error(): string | null {
		return this._error;
	}

	get isEnabled(): boolean {
		const mcpConfig = this.buildMcpClientConfig(settingsStore.config);

		return (
			mcpConfig !== null && mcpConfig !== undefined && Object.keys(mcpConfig.servers).length > 0
		);
	}

	get isInitialized(): boolean {
		return this.connections.size > 0;
	}

	get isInitializing(): boolean {
		return this._isInitializing;
	}

	get isProxyAvailable(): boolean {
		return serverStore.props?.cors_proxy_enabled ?? false;
	}

	/** Resource state, composed here so consumers have a single MCP scope. */
	get resources() {
		return mcpResourceStore;
	}

	get toolCount(): number {
		return this._toolCount;
	}

	acquireConnection(): void {
		this.activeFlowCount++;
	}

	addServer(
		serverData: Omit<MCPServerSettingsEntry, 'id'> & { id?: string }
	): MCPServerSettingsEntry {
		const servers = this.getServers();
		const newServer: MCPServerSettingsEntry = {
			displayName: serverData.displayName,
			enabled: serverData.enabled,
			headers: serverData.headers?.trim() || undefined,
			id: serverData.id || (uuid() ?? `server-${Date.now()}`),
			name: serverData.name,
			url: serverData.url.trim(),
			useProxy: serverData.useProxy
		};

		settingsStore.updateConfig(SETTINGS_KEYS.MCP_SERVERS, JSON.stringify([...servers, newServer]));

		return newServer;
	}

	/**
	 * Add a resource as attachment to chat context.
	 * Automatically fetches content if not cached.
	 */
	async attachResource(uri: string): Promise<MCPResourceAttachment | null> {
		const resourceInfo = mcpResourceStore.findResourceByUri(uri);

		if (!resourceInfo) {
			console.error(`[MCPStore] Resource not found: ${uri}`);

			return null;
		}

		if (mcpResourceStore.isAttached(uri)) {
			return null;
		}

		const attachment = mcpResourceStore.addAttachment(resourceInfo);

		try {
			const content = await this.readResource(uri);

			if (content) {
				mcpResourceStore.updateAttachmentContent(attachment.id, content);
			} else {
				mcpResourceStore.updateAttachmentError(attachment.id, 'Failed to read resource');
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);

			mcpResourceStore.updateAttachmentError(attachment.id, message);
		}

		return mcpResourceStore.getAttachment(attachment.id) ?? null;
	}

	/**
	 * Auto-reconnect to a server with exponential backoff.
	 * Continues indefinitely until successful.
	 *
	 * Race-condition safety: when the phase callback fires a DISCONNECTED event
	 * while we are still inside this function (e.g., the server drops right after
	 * a successful connect()), a naive inner `autoReconnect()` call would be
	 * swallowed by the `reconnectingServers` guard, leaving the server
	 * permanently disconnected once the outer call exits. We solve this by
	 * deferring the new reconnection via the `needsReconnect` flag: the flag is
	 * set inside the phase callback and honoured in the `finally` block after
	 * the guard entry has been removed.
	 */
	async autoReconnect(serverName: string): Promise<void> {
		// Guard against concurrent reconnections
		if (this.reconnectingServers.has(serverName)) {
			console.log(`[MCPStore][${serverName}] Reconnection already in progress, skipping`);

			return;
		}

		const serverConfig = this.serverConfigs.get(serverName);

		if (!serverConfig) {
			console.error(`[MCPStore] No config found for ${serverName}, cannot reconnect`);

			return;
		}

		this.reconnectingServers.add(serverName);
		let backoff = MCP_RECONNECT.INITIAL_DELAY;
		// Flag set by the phase callback when a DISCONNECTED event fires while
		// reconnectingServers still holds this server (see JSDoc above).
		let needsReconnect = false;

		try {
			while (true) {
				await new Promise((resolve) => setTimeout(resolve, backoff));

				console.log(`[MCPStore][${serverName}] Auto-reconnecting...`);

				try {
					// Per-attempt timeout: reject if the server doesn't respond in time,
					// then fall through to backoff logic as with any other failure.
					const timeoutPromise = new Promise<never>((_, reject) =>
						setTimeout(
							() =>
								reject(
									new Error(
										`Reconnect attempt timed out after ${MCP_RECONNECT.ATTEMPT_TIMEOUT_MS}ms`
									)
								),
							MCP_RECONNECT.ATTEMPT_TIMEOUT_MS
						)
					);

					needsReconnect = false;
					const listChangedHandlers = this.createListChangedHandlers(serverName);
					const connectPromise = MCPService.connect(
						serverName,
						serverConfig,
						DEFAULT_MCP_CONFIG.clientInfo,
						DEFAULT_MCP_CONFIG.capabilities,
						(phase) => {
							if (phase === MCPConnectionPhase.DISCONNECTED) {
								if (this.reconnectingServers.has(serverName)) {
									// Reconnect loop is active; defer to after it exits.
									needsReconnect = true;
								} else {
									console.log(
										`[MCPStore][${serverName}] Connection lost, restarting auto-reconnect`
									);
									this.autoReconnect(serverName);
								}
							}
						},
						listChangedHandlers
					);
					const connection = await Promise.race([connectPromise, timeoutPromise]);

					this.connections.set(serverName, connection);

					// Rebuild tool index for this server
					this.indexServerTools(serverName, connection.tools);

					console.log(`[MCPStore][${serverName}] Reconnected successfully`);

					break;
				} catch (error) {
					console.warn(`[MCPStore][${serverName}] Reconnection failed:`, error);
					backoff = Math.min(backoff * MCP_RECONNECT.BACKOFF_MULTIPLIER, MCP_RECONNECT.MAX_DELAY);
				}
			}
		} finally {
			this.reconnectingServers.delete(serverName);

			// If the phase callback signalled a disconnect while this function held
			// the guard, kick off a fresh reconnect now that the guard is released.
			if (needsReconnect) {
				console.log(
					`[MCPStore][${serverName}] Deferred disconnect detected, restarting auto-reconnect`
				);
				this.autoReconnect(serverName);
			}
		}
	}

	clearError(): void {
		this._error = null;
	}

	clearHealthCheck(serverId: string): void {
		this.health.clear(serverId);
	}

	/**
	 * Clear all resource attachments.
	 */
	clearResourceAttachments(): void {
		mcpResourceStore.clearAttachments();
	}

	/**
	 * Convert current resource attachments to DatabaseMessageExtra[] and clear them.
	 * Called during message send to persist resources with the user message.
	 */
	consumeResourceAttachmentsAsExtras(): DatabaseMessageExtraMcpResource[] {
		const extras = mcpResourceStore.toMessageExtras();

		if (extras.length > 0) {
			mcpResourceStore.clearAttachments();
		}

		return extras;
	}

	/**
	 * Initialize every settings-enabled server. Policy filtering happens at tool
	 * collection time, so switching conversation policies never re-initializes.
	 */
	async ensureInitialized(): Promise<boolean> {
		if (!browser) {
			return false;
		}

		const mcpConfig = this.buildMcpClientConfig(settingsStore.config);
		const signature = mcpConfig ? JSON.stringify(mcpConfig) : null;

		if (!signature) {
			await this.shutdown();

			return false;
		}

		if (this.isInitialized && this.configSignature === signature) {
			return true;
		}

		if (this.initPromise && this.configSignature === signature) {
			return this.initPromise;
		}

		if (this.connections.size > 0 || this.initPromise) await this.shutdown();

		return this.initialize(signature, mcpConfig!);
	}

	async executeTool(toolCall: MCPToolCall, signal?: AbortSignal): Promise<ToolExecutionResult> {
		return this.executeToolByName(
			toolCall.function.name,
			this.parseToolArguments(toolCall.function.arguments),
			signal
		);
	}

	async executeToolByName(
		toolName: string,
		args: Record<string, unknown>,
		signal?: AbortSignal
	): Promise<ToolExecutionResult> {
		const serverName = this.toolsIndex.get(toolName);

		if (!serverName) throw new Error(`Unknown tool: ${toolName}`);

		const connection = this.connections.get(serverName);

		if (!connection) throw new Error(`Server "${serverName}" is not connected`);

		try {
			return await MCPService.callTool(connection, { arguments: args, name: toolName }, signal);
		} catch (error) {
			if (MCPService.isSessionExpiredError(error)) {
				await this.reconnectServer(serverName);

				const newConnection = this.connections.get(serverName);

				if (!newConnection) throw new Error(`Failed to reconnect to "${serverName}"`);

				return MCPService.callTool(newConnection, { arguments: args, name: toolName }, signal);
			}

			throw error;
		}
	}

	/**
	 * Fetch resources from all connected servers that support them.
	 * Updates mcpResourceStore with the results.
	 * @param forceRefresh - If true, bypass cache and fetch fresh data
	 */
	async fetchAllResources(forceRefresh: boolean = false): Promise<void> {
		const serversWithResources = this.getServersWithResources();

		if (serversWithResources.length === 0) {
			return;
		}

		// Check if we have cached resources and they're recent (unless force refresh)
		if (!forceRefresh) {
			const allServersCached = serversWithResources.every((serverName) => {
				const serverRes = mcpResourceStore.getServerResources(serverName);

				if (!serverRes || !serverRes.lastFetched) {
					return false;
				}

				// Cache is valid for 5 minutes
				const age = Date.now() - serverRes.lastFetched.getTime();

				return age < CACHE.DEFAULT_TTL_MS;
			});

			if (allServersCached) {
				console.log('[MCPStore] Using cached resources');

				return;
			}
		}

		mcpResourceStore.setLoading(true);

		try {
			await Promise.all(
				serversWithResources.map((serverName) => this.fetchServerResources(serverName))
			);
		} finally {
			mcpResourceStore.setLoading(false);
		}
	}

	/**
	 * Fetch resources from a specific server.
	 * Updates mcpResourceStore with the results.
	 */
	async fetchServerResources(serverName: string): Promise<void> {
		const connection = this.connections.get(serverName);

		if (!connection) {
			console.warn(`[MCPStore] No connection found for server: ${serverName}`);

			return;
		}

		if (!MCPService.supportsResources(connection)) {
			return;
		}

		mcpResourceStore.setServerLoading(serverName, true);

		try {
			const [resources, templates] = await Promise.all([
				MCPService.listAllResources(connection),
				MCPService.listAllResourceTemplates(connection)
			]);

			mcpResourceStore.setServerResources(serverName, resources, templates);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);

			mcpResourceStore.setServerError(serverName, message);
			console.error(`[MCPStore][${serverName}] Failed to fetch resources:`, error);
		}
	}

	/**
	 * Resolve which configured MCP server owns a given tool name. Looks at
	 * active connections first (fast path), then falls back to per-server
	 * health-check data so server-side MCP proxies (where llama-server
	 * executes MCP tools but the browser does not hold a direct connection)
	 * still resolve tool names to their owning server.
	 */
	findServerForTool(toolName: string): string | undefined {
		const fromIndex = this.toolsIndex.get(toolName);

		if (fromIndex) return fromIndex;

		for (const server of this.getServers()) {
			const health = this.health.checks[server.id];

			if (!health || health.status !== HealthCheckStatus.SUCCESS) continue;

			if (health.tools.some((tool) => tool.name === toolName)) {
				return server.id;
			}
		}

		return undefined;
	}
	getActiveFlowCount(): number {
		return this.activeFlowCount;
	}

	async getAllPrompts(): Promise<MCPPromptInfo[]> {
		const results: MCPPromptInfo[] = [];

		for (const [serverName, connection] of this.connections) {
			if (!connection.serverCapabilities?.prompts) continue;

			const prompts = await MCPService.listPrompts(connection);

			for (const prompt of prompts) {
				results.push({
					arguments: prompt.arguments?.map((arg) => ({
						description: arg.description,
						name: arg.name,
						required: arg.required
					})),
					description: prompt.description,
					name: prompt.name,
					serverName,
					title: prompt.title
				});
			}
		}

		return results;
	}

	/**
	 * Get all active MCP connections.
	 * @returns Map of server names to connections
	 */
	getConnections(): Map<string, MCPConnection> {
		return this.connections;
	}

	/**
	 * Check if a server already has an active connection that can be reused.
	 * Returns the existing connection if available.
	 */
	getExistingConnection(serverId: string): MCPConnection | undefined {
		return this.connections.get(serverId);
	}

	/**
	 * Get server instructions from health check results (for display before active connection).
	 * Useful for showing instructions in settings UI.
	 */
	getHealthCheckInstructions(): Array<{
		serverId: string;
		serverTitle?: string;
		instructions: string;
	}> {
		const results: Array<{ serverId: string; serverTitle?: string; instructions: string }> = [];

		for (const [serverId, state] of Object.entries(this.health.checks)) {
			if (state.status === HealthCheckStatus.SUCCESS && state.instructions) {
				results.push({
					instructions: state.instructions,
					serverId,
					serverTitle: state.serverInfo?.title || state.serverInfo?.name
				});
			}
		}

		return results;
	}

	/**
	 * Health checks live in MCPHealthCheckManager; these delegate so
	 * consumers keep a single entry point.
	 */
	getHealthCheckState(serverId: string): HealthCheckState {
		return this.health.getState(serverId);
	}

	async getPrompt(
		serverName: string,
		promptName: string,
		args?: Record<string, string>
	): Promise<GetPromptResult> {
		const connection = this.connections.get(serverName);

		if (!connection) throw new Error(`Server "${serverName}" not found for prompt "${promptName}"`);

		return MCPService.getPrompt(connection, promptName, args);
	}

	async getPromptCompletions(
		serverName: string,
		promptName: string,
		argumentName: string,
		argumentValue: string
	): Promise<{ values: string[]; total?: number; hasMore?: boolean } | null> {
		const connection = this.connections.get(serverName);

		if (!connection) {
			console.warn(`[MCPStore] Server "${serverName}" is not connected`);

			return null;
		}

		if (!connection.serverCapabilities?.completions) {
			return null;
		}

		return MCPService.complete(
			connection,
			{ name: promptName, type: MCPRefType.PROMPT },
			{ name: argumentName, value: argumentValue }
		);
	}

	/**
	 * Request timeout in milliseconds, read live from the global setting
	 * so a change in Settings applies to every server immediately.
	 */
	getRequestTimeoutMs(): number {
		const seconds =
			Number(settingsStore.config.mcpRequestTimeoutSeconds) ||
			DEFAULT_MCP_CONFIG.requestTimeoutSeconds;

		return Math.round(seconds * 1000);
	}

	/**
	 * Get completions for a resource template argument.
	 * Uses the MCP Completion API with ref/resource.
	 */
	async getResourceCompletions(
		serverName: string,
		uriTemplate: string,
		argumentName: string,
		argumentValue: string
	): Promise<{ values: string[]; total?: number; hasMore?: boolean } | null> {
		const connection = this.connections.get(serverName);

		if (!connection) {
			console.warn(`[MCPStore] Server "${serverName}" is not connected`);

			return null;
		}

		if (!connection.serverCapabilities?.completions) {
			return null;
		}

		return MCPService.complete(
			connection,
			{ type: MCPRefType.RESOURCE, uri: uriTemplate },
			{ name: argumentName, value: argumentValue }
		);
	}

	/**
	 * Get formatted resource context for chat.
	 */
	getResourceContextForChat(): string {
		return mcpResourceStore.formatAttachmentsForContext();
	}

	getServerById(serverId: string): MCPServerSettingsEntry | undefined {
		return this.getServers().find((s) => s.id === serverId);
	}

	/**
	 * Get display name for an MCP server by its ID.
	 * Falls back to the server ID if server is not found.
	 */
	getServerDisplayName(serverId: string): string {
		const server = this.getServerById(serverId);

		return server ? this.getServerLabel(server) : serverId;
	}

	/**
	 * Get icon URL for an MCP server by its ID.
	 * Returns the best icon from the MCP server's `icons` array
	 * (see MCP spec: spec.modelcontextprotocol.io).
	 * Returns null if no icon is available.
	 */
	getServerFavicon(serverId: string): string | null {
		const server = this.getServerById(serverId);

		if (!server) {
			return null;
		}

		const isDark = mode.current === ColorMode.DARK;
		const healthState = this.health.getState(serverId);

		if (healthState.status === HealthCheckStatus.SUCCESS && healthState.serverInfo?.icons) {
			const mcpIconUrl = getMcpIconUrl(healthState.serverInfo.icons, isDark);

			if (mcpIconUrl) {
				return mcpIconUrl;
			}
		}

		return getMcpServerFaviconFallback(server.url);
	}

	/**
	 * Resolve the favicon URL for an MCP server by one of its tool names.
	 * Returns `null` if the tool is not provided by any configured MCP server,
	 * or if the owning server has no icon to show.
	 * Pair with {@link getServerFavicon} for direct server-id lookup.
	 */
	getServerFaviconForTool(toolName: string | undefined): string | null {
		if (!toolName) return null;

		const serverId = this.findServerForTool(toolName);

		if (!serverId) return null;

		return this.getServerFavicon(serverId);
	}

	/**
	 * Get aggregated server instructions from all connected servers.
	 * Returns an array of { serverName, serverTitle, instructions } objects.
	 */
	getServerInstructions(): Array<{
		serverName: string;
		serverTitle?: string;
		instructions: string;
	}> {
		const results: Array<{ serverName: string; serverTitle?: string; instructions: string }> = [];

		for (const [serverName, connection] of this.connections) {
			if (connection.instructions) {
				results.push({
					instructions: connection.instructions,
					serverName,
					serverTitle: connection.serverInfo?.title || connection.serverInfo?.name
				});
			}
		}

		return results;
	}

	getServerLabel(server: MCPServerDisplayInfo): string {
		return getMcpServerLabel(server, this.getServers(), this.health.checks);
	}

	getServers(): MCPServerSettingsEntry[] {
		const raw = settingsStore.config.mcpServers;

		// cache the parse: the config string rarely changes and getServers is
		// called from hot paths (per-tool display lookups, capability checks)
		if (this.serversCache && this.serversCache.raw === raw) {
			return this.serversCache.servers;
		}

		const servers = parseMcpServerSettings(raw);

		this.serversCache = { raw, servers };

		return servers;
	}

	getServersStatus(): ServerStatus[] {
		const statuses: ServerStatus[] = [];

		for (const [name, connection] of this.connections) {
			statuses.push({
				error: undefined,
				isConnected: true,
				name,
				toolCount: connection.tools.length
			});
		}

		return statuses;
	}

	/**
	 * Get list of enabled servers that support resources.
	 * Checks active connections first, then health check state as fallback.
	 */
	getServersWithResources(): string[] {
		const enabledServerIds = new Set(
			this.getServers()
				.filter((s) => s.enabled)
				.map((s) => s.id)
		);
		const servers: string[] = [];

		for (const [name, connection] of this.connections) {
			if (!enabledServerIds.has(name)) continue;

			if (MCPService.supportsResources(connection) && !servers.includes(name)) {
				servers.push(name);
			}
		}

		// Also check health check states for servers not yet connected
		for (const [serverId, state] of Object.entries(this.health.checks)) {
			if (!enabledServerIds.has(serverId)) continue;

			if (
				!servers.includes(serverId) &&
				state.status === HealthCheckStatus.SUCCESS &&
				state.capabilities?.server?.resources !== undefined
			) {
				servers.push(serverId);
			}
		}

		return servers;
	}

	getToolNames(): string[] {
		return Array.from(this.toolsIndex.keys());
	}

	getToolServer(toolName: string): string | undefined {
		return this.toolsIndex.get(toolName);
	}

	hasAvailableServers(): boolean {
		return parseMcpServerSettings(settingsStore.config.mcpServers).some(
			(s) => s.enabled && s.url.trim()
		);
	}

	hasEnabledServers(): boolean {
		return Boolean(this.buildMcpClientConfig(settingsStore.config));
	}

	/**
	 * Check if any connected server has instructions.
	 */
	hasServerInstructions(): boolean {
		for (const connection of this.connections.values()) {
			if (connection.instructions) {
				return true;
			}
		}

		return false;
	}

	hasTool(toolName: string): boolean {
		return this.toolsIndex.has(toolName);
	}

	/**
	 * Promote a health check connection to an active connection.
	 * This avoids the need to reconnect when the server is needed for agentic flows.
	 */
	promoteHealthCheckToConnection(serverId: string, connection: MCPConnection): void {
		this.indexServerTools(serverId, connection.tools);

		this.connections.set(serverId, connection);

		this.updateState({
			connectedServers: Array.from(this.connections.keys()),
			toolCount: this.toolsIndex.size
		});
	}

	/**
	 * Read resource content from a server.
	 * Caches the result in mcpResourceStore.
	 */
	async readResource(uri: string): Promise<MCPResourceContent[] | null> {
		const cached = mcpResourceStore.getCachedContent(uri);

		if (cached) {
			return cached.content;
		}

		// Find which server has this resource
		const serverName = mcpResourceStore.findServerForUri(uri);

		if (!serverName) {
			console.error(`[MCPStore] No server found for resource URI: ${uri}`);

			return null;
		}

		const connection = this.connections.get(serverName);

		if (!connection) {
			console.error(`[MCPStore] No connection found for server: ${serverName}`);

			return null;
		}

		try {
			const result = await MCPService.readResource(connection, uri);
			const resourceInfo = mcpResourceStore.findResourceByUri(uri);

			if (resourceInfo) {
				mcpResourceStore.cacheResourceContent(resourceInfo, result.contents);
			}

			return result.contents;
		} catch (error) {
			console.error(`[MCPStore] Failed to read resource ${uri}:`, error);

			return null;
		}
	}

	/**
	 * Read a resource by an arbitrary URI (e.g., one expanded from a template).
	 * Unlike readResource(), this does not require the URI to be in the resources list.
	 */
	async readResourceByUri(serverName: string, uri: string): Promise<MCPResourceContent[] | null> {
		const connection = this.connections.get(serverName);

		if (!connection) {
			console.error(`[MCPStore] No connection found for server: ${serverName}`);

			return null;
		}

		try {
			const result = await MCPService.readResource(connection, uri);

			return result.contents;
		} catch (error) {
			console.error(`[MCPStore] Failed to read resource ${uri}:`, error);

			return null;
		}
	}

	/** Store a server config so auto-reconnect can rebuild the session. */
	registerServerConfig(name: string, config: MCPServerConfig): void {
		this.serverConfigs.set(name, config);
	}

	/**
	 * Release a connection reference.
	 * By default, keeps connections alive for reuse (shutdownIfUnused=false).
	 * MCP spec encourages long-lived sessions to avoid reconnection overhead.
	 */
	async releaseConnection(shutdownIfUnused = false): Promise<void> {
		this.activeFlowCount = Math.max(0, this.activeFlowCount - 1);

		if (shutdownIfUnused && this.activeFlowCount === 0) {
			await this.shutdown();
		}
	}

	/**
	 * Drop a connection without disconnecting, e.g. when a health check finds
	 * it stale and recreates it.
	 */
	removeConnection(serverId: string): void {
		this.connections.delete(serverId);
	}

	/**
	 * Remove a resource attachment from chat context.
	 */
	removeResourceAttachment(attachmentId: string): void {
		mcpResourceStore.removeAttachment(attachmentId);
	}

	removeServer(id: string): void {
		const servers = this.getServers();

		settingsStore.updateConfig(
			SETTINGS_KEYS.MCP_SERVERS,
			JSON.stringify(servers.filter((s) => s.id !== id))
		);
		this.clearHealthCheck(id);
	}

	async runHealthCheck(server: HealthCheckParams, promoteToActive = false): Promise<void> {
		return this.health.run(server, promoteToActive);
	}

	async runHealthChecksForServers(
		servers: {
			id: string;
			enabled: boolean;
			url: string;
			headers?: string;
		}[],
		skipIfChecked = true,
		promoteToActive = false
	): Promise<void> {
		return this.health.runForServers(servers, skipIfChecked, promoteToActive);
	}

	async shutdown(): Promise<void> {
		if (this.initPromise) {
			await this.initPromise.catch(() => {});
			this.initPromise = null;
		}

		if (this.connections.size === 0) {
			return;
		}

		await Promise.all(
			Array.from(this.connections.values()).map((conn) =>
				MCPService.disconnect(conn).catch((error) =>
					console.warn(`[MCPStore] Error disconnecting ${conn.serverName}:`, error)
				)
			)
		);

		this.connections.clear();
		this.toolsIndex.clear();
		this.serverConfigs.clear();
		this.configSignature = null;
		this.updateState({
			connectedServers: [],
			error: null,
			isInitializing: false,
			toolCount: 0
		});
	}

	/**
	 * Subscribe to resource updates.
	 */
	async subscribeToResource(uri: string): Promise<boolean> {
		const serverName = mcpResourceStore.findServerForUri(uri);

		if (!serverName) {
			console.error(`[MCPStore] No server found for resource URI: ${uri}`);

			return false;
		}

		const connection = this.connections.get(serverName);

		if (!connection) {
			console.error(`[MCPStore] No connection found for server: ${serverName}`);

			return false;
		}

		if (!MCPService.supportsResourceSubscriptions(connection)) {
			return false;
		}

		try {
			await MCPService.subscribeResource(connection, uri);
			mcpResourceStore.addSubscription(uri, serverName);

			return true;
		} catch (error) {
			console.error(`[MCPStore] Failed to subscribe to resource ${uri}:`, error);

			return false;
		}
	}

	/**
	 * Unsubscribe from resource updates.
	 */
	async unsubscribeFromResource(uri: string): Promise<boolean> {
		const serverName = mcpResourceStore.findServerForUri(uri);

		if (!serverName) {
			console.error(`[MCPStore] No server found for resource URI: ${uri}`);

			return false;
		}

		const connection = this.connections.get(serverName);

		if (!connection) {
			console.error(`[MCPStore] No connection found for server: ${serverName}`);

			return false;
		}

		try {
			await MCPService.unsubscribeResource(connection, uri);
			mcpResourceStore.removeSubscription(uri);

			return true;
		} catch (error) {
			console.error(`[MCPStore] Failed to unsubscribe from resource ${uri}:`, error);

			return false;
		}
	}

	updateServer(id: string, updates: Partial<MCPServerSettingsEntry>): void {
		const servers = this.getServers();

		settingsStore.updateConfig(
			SETTINGS_KEYS.MCP_SERVERS,
			JSON.stringify(
				servers.map((server) => (server.id === id ? { ...server, ...updates } : server))
			)
		);
	}

	/**
	 * Builds MCP client configuration from settings.
	 */
	private buildMcpClientConfig(cfg: SettingsConfigType): MCPClientConfig | undefined {
		const rawServers = parseMcpServerSettings(cfg.mcpServers);

		if (!rawServers.length) {
			return undefined;
		}

		const servers: Record<string, MCPServerConfig> = {};

		for (const [index, entry] of rawServers.entries()) {
			if (!entry.enabled) continue;

			const normalized = this.buildServerConfig(entry);

			if (normalized) servers[this.generateServerId(entry.id, index)] = normalized;
		}

		if (Object.keys(servers).length === 0) {
			return undefined;
		}

		return {
			capabilities: DEFAULT_MCP_CONFIG.capabilities,
			clientInfo: DEFAULT_MCP_CONFIG.clientInfo,
			protocolVersion: DEFAULT_MCP_CONFIG.protocolVersion,
			requestTimeoutMs: this.getRequestTimeoutMs(),
			servers
		};
	}

	/**
	 * Builds server configuration from a settings entry.
	 */
	private buildServerConfig(
		entry: MCPServerSettingsEntry,
		connectionTimeoutMs = DEFAULT_MCP_CONFIG.connectionTimeoutMs
	): MCPServerConfig | undefined {
		if (!entry?.url) {
			return undefined;
		}

		let headers: Record<string, string> | undefined;

		if (entry.headers) {
			try {
				const parsed = JSON.parse(entry.headers);

				if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed))
					headers = parsed as Record<string, string>;
			} catch {
				console.warn('[MCP] Failed to parse custom headers JSON:', entry.headers);
			}
		}

		return {
			handshakeTimeoutMs: connectionTimeoutMs,
			headers,
			requestTimeoutMs: this.getRequestTimeoutMs(),
			transport: detectMcpTransportFromUrl(entry.url),
			url: entry.url,
			useProxy: entry.useProxy
		};
	}

	private createListChangedHandlers(serverName: string): ListChangedHandlers {
		return {
			prompts: {
				onChanged: (error: Error | null) => {
					if (error) {
						console.warn(`[MCPStore][${serverName}] Prompts list changed error:`, error);

						return;
					}
				}
			},
			tools: {
				onChanged: (error: Error | null, tools: Tool[] | null) => {
					if (error) {
						console.warn(`[MCPStore][${serverName}] Tools list changed error:`, error);

						return;
					}

					this.handleToolsListChanged(serverName, tools ?? []);
				}
			}
		};
	}

	private async doInitialize(
		signature: string,
		mcpConfig: MCPClientConfig,
		serverEntries: [string, MCPClientConfig['servers'][string]][]
	): Promise<boolean> {
		const clientInfo = mcpConfig.clientInfo ?? DEFAULT_MCP_CONFIG.clientInfo;
		const capabilities = mcpConfig.capabilities ?? DEFAULT_MCP_CONFIG.capabilities;
		const results = await Promise.allSettled(
			serverEntries.map(async ([name, serverConfig]) => {
				this.serverConfigs.set(name, serverConfig);

				const listChangedHandlers = this.createListChangedHandlers(name);
				const connection = await MCPService.connect(
					name,
					serverConfig,
					clientInfo,
					capabilities,
					(phase) => {
						if (phase === MCPConnectionPhase.DISCONNECTED) {
							console.log(`[MCPStore][${name}] Connection lost, starting auto-reconnect`);
							this.autoReconnect(name);
						}
					},
					listChangedHandlers
				);

				return { connection, name };
			})
		);

		if (this.configSignature !== signature) {
			for (const result of results) {
				if (result.status === 'fulfilled')
					await MCPService.disconnect(result.value.connection).catch(console.warn);
			}

			return false;
		}

		for (const result of results) {
			if (result.status === 'fulfilled') {
				const { connection, name } = result.value;

				this.connections.set(name, connection);

				this.indexServerTools(name, connection.tools);
			} else {
				console.error(`[MCPStore] Failed to connect:`, result.reason);
			}
		}

		const successCount = this.connections.size;

		if (successCount === 0 && serverEntries.length > 0) {
			this.updateState({
				connectedServers: [],
				error: 'All MCP server connections failed',
				isInitializing: false,
				toolCount: 0
			});
			this.initPromise = null;

			return false;
		}

		this.updateState({
			connectedServers: Array.from(this.connections.keys()),
			error: null,
			isInitializing: false,
			toolCount: this.toolsIndex.size
		});
		this.initPromise = null;

		return true;
	}

	/**
	 * Generates a unique server ID from an optional ID string or index.
	 */
	private generateServerId(id: unknown, index: number): string {
		if (typeof id === 'string' && id.trim()) {
			return id.trim();
		}

		return `${MCP_SERVER_ID_PREFIX}-${index + 1}`;
	}

	/** Server ids that are usable right now: globally enabled ones. */
	private globalEnabledServerIds(): Set<string> {
		return new Set(
			this.getServers()
				.filter((s) => s.enabled)
				.map((s) => s.id)
		);
	}

	private handleToolsListChanged(serverName: string, tools: Tool[]): void {
		const connection = this.connections.get(serverName);

		if (!connection) {
			return;
		}

		for (const [toolName, ownerServer] of this.toolsIndex.entries()) {
			if (ownerServer === serverName) this.toolsIndex.delete(toolName);
		}

		connection.tools = tools;

		for (const tool of tools) {
			if (this.toolsIndex.has(tool.name))
				console.warn(
					`[MCPStore] Tool name conflict after list change: "${tool.name}" exists in "${this.toolsIndex.get(tool.name)}" and "${serverName}". Using tool from "${serverName}".`
				);

			this.toolsIndex.set(tool.name, serverName);
		}
		this.updateState({ toolCount: this.toolsIndex.size });
	}

	/**
	 * Registers the tools exposed by a server into the global name->server index,
	 * warning on conflicts. Shared by connect, reconnect and auto-reconnect.
	 */
	private indexServerTools(serverName: string, tools: Tool[]): void {
		for (const tool of tools) {
			if (this.toolsIndex.has(tool.name))
				console.warn(
					`[MCPStore] Tool name conflict: "${tool.name}" exists in "${this.toolsIndex.get(tool.name)}" and "${serverName}". Using tool from "${serverName}".`
				);

			this.toolsIndex.set(tool.name, serverName);
		}
	}

	private async initialize(signature: string, mcpConfig: MCPClientConfig): Promise<boolean> {
		this.updateState({ error: null, isInitializing: true });
		this.configSignature = signature;

		const serverEntries = Object.entries(mcpConfig.servers);

		if (serverEntries.length === 0) {
			this.updateState({ connectedServers: [], isInitializing: false, toolCount: 0 });

			return false;
		}

		this.initPromise = this.doInitialize(signature, mcpConfig, serverEntries);

		return this.initPromise;
	}

	private parseToolArguments(args: string | Record<string, unknown>): Record<string, unknown> {
		if (typeof args === 'string') {
			const trimmed = args.trim();

			if (trimmed === '') {
				return {};
			}

			try {
				const parsed = JSON.parse(trimmed);

				if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed))
					throw new Error(
						`Tool arguments must be an object, got ${Array.isArray(parsed) ? 'array' : typeof parsed}`
					);

				return parsed as Record<string, unknown>;
			} catch (error) {
				throw new Error(`Failed to parse tool arguments as JSON: ${(error as Error).message}`);
			}
		}

		if (typeof args === 'object' && args !== null && !Array.isArray(args)) {
			return args;
		}

		throw new Error(`Invalid tool arguments type: ${typeof args}`);
	}

	/**
	 * Immediately reconnect to a server by creating a fresh transport and session.
	 * Used when a session-expired error (HTTP 404) is detected during tool execution.
	 * Per MCP spec 2025-11-25: client MUST discard session ID and re-initialize.
	 *
	 * Unlike autoReconnect (which uses exponential backoff for connectivity issues),
	 * this performs a single immediate reconnection attempt since the server is known
	 * to be reachable (it responded with 404).
	 */
	private async reconnectServer(serverName: string): Promise<void> {
		const serverConfig = this.serverConfigs.get(serverName);

		if (!serverConfig) {
			throw new Error(`[MCPStore] No config found for ${serverName}, cannot reconnect`);
		}

		// Disconnect stale connection (clears old transport + session ID)
		const oldConnection = this.connections.get(serverName);

		if (oldConnection) {
			await MCPService.disconnect(oldConnection).catch(console.warn);
			this.connections.delete(serverName);
		}

		console.log(`[MCPStore][${serverName}] Session expired, reconnecting with fresh session...`);

		const listChangedHandlers = this.createListChangedHandlers(serverName);
		const connection = await MCPService.connect(
			serverName,
			serverConfig,
			DEFAULT_MCP_CONFIG.clientInfo,
			DEFAULT_MCP_CONFIG.capabilities,
			(phase) => {
				if (phase === MCPConnectionPhase.DISCONNECTED) {
					console.log(`[MCPStore][${serverName}] Connection lost, starting auto-reconnect`);
					this.autoReconnect(serverName);
				}
			},
			listChangedHandlers
		);

		this.connections.set(serverName, connection);
		this.indexServerTools(serverName, connection.tools);

		console.log(`[MCPStore][${serverName}] Session recovered successfully`);
	}

	private updateState(state: {
		isInitializing?: boolean;
		error?: string | null;
		toolCount?: number;
		connectedServers?: string[];
	}): void {
		if (state.isInitializing !== undefined) {
			this._isInitializing = state.isInitializing;
		}

		if (state.error !== undefined) {
			this._error = state.error;
		}

		if (state.toolCount !== undefined) {
			this._toolCount = state.toolCount;
		}

		if (state.connectedServers !== undefined) {
			this.connectedServers = state.connectedServers;
		}
	}
}

export const mcpStore = new MCPStore();
