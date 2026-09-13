/**
 * MCPHealthCheckManager - Health checks for MCP servers
 *
 * Owns per-server connectivity probes: connection reuse, capability
 * snapshots, and promotion of a successful check to an active connection.
 * Created and owned by mcpStore; the host owns the connection registry the
 * probes draw from and promote into.
 */

import { DEFAULT_MCP_CONFIG } from '$lib/constants';
import { HealthCheckStatus, MCPConnectionPhase, MCPLogLevel } from '$lib/enums';
import { MCPService } from '$lib/services/mcp.service';
import type {
	ClientCapabilities,
	HealthCheckParams,
	HealthCheckState,
	MCPCapabilitiesInfo,
	MCPConnection,
	MCPConnectionLog,
	MCPServerConfig,
	ServerCapabilities
} from '$lib/types';
import { detectMcpTransportFromUrl } from '$lib/utils';

// module-level so the timestamp is not flagged as reactive state by prefer-svelte-reactivity
function createConnectionErrorLog(message: string): MCPConnectionLog {
	return {
		level: MCPLogLevel.ERROR,
		message: `Connection failed: ${message}`,
		phase: MCPConnectionPhase.ERROR,
		timestamp: new Date()
	};
}

/**
 * The slice of mcpStore the probes drive. Kept narrow on purpose so the
 * probes cannot reach around the host's full surface; mcpStore implements
 * this structurally.
 */
export interface McpHealthHost {
	autoReconnect(serverName: string): Promise<void>;
	getExistingConnection(serverId: string): MCPConnection | undefined;
	getRequestTimeoutMs(): number;
	promoteHealthCheckToConnection(serverId: string, connection: MCPConnection): void;
	registerServerConfig(name: string, config: MCPServerConfig): void;
	removeConnection(serverId: string): void;
}

export class MCPHealthCheckManager {
	private _checks = $state<Record<string, HealthCheckState>>({});

	/** Raw per-server check states, for host-side capability scans. */
	get checks(): Record<string, HealthCheckState> {
		return this._checks;
	}

	clear(serverId: string): void {
		const { [serverId]: _removed, ...rest } = this._checks;

		this._checks = rest;
	}

	constructor(private host: McpHealthHost) {}

	getState(serverId: string): HealthCheckState {
		return this._checks[serverId] ?? { status: HealthCheckStatus.IDLE };
	}

	hasState(serverId: string): boolean {
		return serverId in this._checks && this._checks[serverId].status !== HealthCheckStatus.IDLE;
	}

	/**
	 * Run a health check for a server.
	 * If the server already has an active connection, reuses it instead of creating a new one.
	 * If promoteToActive is true and server is enabled, the connection will be kept
	 * and promoted to an active connection instead of being disconnected.
	 */
	async run(server: HealthCheckParams, promoteToActive = false): Promise<void> {
		const existingConnection = this.host.getExistingConnection(server.id);

		if (existingConnection) {
			// Reuse existing connection - just refresh tools list
			try {
				const tools = await MCPService.listTools(existingConnection);
				const capabilities = this.buildCapabilitiesInfo(
					existingConnection.serverCapabilities,
					existingConnection.clientCapabilities
				);

				this.setState(server.id, {
					capabilities,
					connectionTimeMs: existingConnection.connectionTimeMs,
					instructions: existingConnection.instructions,
					logs: [],
					protocolVersion: existingConnection.protocolVersion,
					serverInfo: existingConnection.serverInfo,
					status: HealthCheckStatus.SUCCESS,
					tools: tools.map((tool) => ({
						description: tool.description,
						name: tool.name,
						title: tool.title
					})),
					transportType: existingConnection.transportType
				});

				return;
			} catch (error) {
				console.warn(
					`[MCPStore] Failed to reuse connection for ${server.id}, creating new one:`,
					error
				);
				// Connection may be stale, remove it and create new one
				this.host.removeConnection(server.id);
			}
		}

		const trimmedUrl = server.url.trim();
		const logs: MCPConnectionLog[] = [];

		let currentPhase: MCPConnectionPhase = MCPConnectionPhase.IDLE;

		if (!trimmedUrl) {
			this.setState(server.id, {
				logs: [],
				message: 'Please enter a server URL first.',
				status: HealthCheckStatus.ERROR
			});

			return;
		}

		this.setState(server.id, {
			logs: [],
			phase: MCPConnectionPhase.TRANSPORT_CREATING,
			status: HealthCheckStatus.CONNECTING
		});

		const timeoutMs = this.host.getRequestTimeoutMs();
		const headers = this.parseHeaders(server.headers);

		try {
			const serverConfig: MCPServerConfig = {
				handshakeTimeoutMs: DEFAULT_MCP_CONFIG.connectionTimeoutMs,
				headers,
				requestTimeoutMs: timeoutMs,
				transport: detectMcpTransportFromUrl(trimmedUrl),
				url: trimmedUrl,
				useProxy: server.useProxy
			};

			this.host.registerServerConfig(server.id, serverConfig);

			const connection = await MCPService.connect(
				server.id,
				serverConfig,
				DEFAULT_MCP_CONFIG.clientInfo,
				DEFAULT_MCP_CONFIG.capabilities,
				(phase, log) => {
					currentPhase = phase;
					logs.push(log);
					this.setState(server.id, {
						logs: [...logs],
						phase,
						status: HealthCheckStatus.CONNECTING
					});

					if (phase === MCPConnectionPhase.DISCONNECTED && promoteToActive) {
						console.log(
							`[MCPStore][${server.id}] Connection lost during health check, starting auto-reconnect`
						);
						this.host.autoReconnect(server.id);
					}
				}
			);
			const tools = connection.tools.map((tool) => ({
				description: tool.description,
				name: tool.name,
				title: tool.title
			}));
			const capabilities = this.buildCapabilitiesInfo(
				connection.serverCapabilities,
				connection.clientCapabilities
			);

			this.setState(server.id, {
				capabilities,
				connectionTimeMs: connection.connectionTimeMs,
				instructions: connection.instructions,
				logs,
				protocolVersion: connection.protocolVersion,
				serverInfo: connection.serverInfo,
				status: HealthCheckStatus.SUCCESS,
				tools,
				transportType: connection.transportType
			});

			if (promoteToActive && server.enabled) {
				this.host.promoteHealthCheckToConnection(server.id, connection);
			} else {
				await MCPService.disconnect(connection);
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Unknown error occurred';

			if (logs.at(-1)?.phase !== MCPConnectionPhase.ERROR) {
				logs.push(createConnectionErrorLog(message));
			}

			this.setState(server.id, {
				logs,
				message,
				phase: currentPhase,
				status: HealthCheckStatus.ERROR
			});
		}
	}

	async runForServers(
		servers: {
			id: string;
			enabled: boolean;
			url: string;
			headers?: string;
		}[],
		skipIfChecked = true,
		promoteToActive = false
	): Promise<void> {
		const serversToCheck = skipIfChecked
			? servers.filter((s) => !this.hasState(s.id) && s.url.trim())
			: servers.filter((s) => s.url.trim());

		if (serversToCheck.length === 0) {
			return;
		}

		const BATCH_SIZE = 5;

		for (let i = 0; i < serversToCheck.length; i += BATCH_SIZE) {
			const batch = serversToCheck.slice(i, i + BATCH_SIZE);

			await Promise.allSettled(batch.map((server) => this.run(server, promoteToActive)));
		}
	}

	/**
	 * Builds capabilities info from server and client capabilities.
	 */
	private buildCapabilitiesInfo(
		serverCaps?: ServerCapabilities,
		clientCaps?: ClientCapabilities
	): MCPCapabilitiesInfo {
		return {
			client: {
				elicitation: clientCaps?.elicitation
					? { form: !!clientCaps.elicitation.form, url: !!clientCaps.elicitation.url }
					: undefined,
				roots: clientCaps?.roots ? { listChanged: clientCaps.roots.listChanged } : undefined,
				sampling: !!clientCaps?.sampling,
				tasks: !!clientCaps?.tasks
			},
			server: {
				completions: !!serverCaps?.completions,
				logging: !!serverCaps?.logging,
				prompts: serverCaps?.prompts ? { listChanged: serverCaps.prompts.listChanged } : undefined,
				resources: serverCaps?.resources
					? {
							listChanged: serverCaps.resources.listChanged,
							subscribe: serverCaps.resources.subscribe
						}
					: undefined,
				tasks: !!serverCaps?.tasks,
				tools: serverCaps?.tools ? { listChanged: serverCaps.tools.listChanged } : undefined
			}
		};
	}

	private parseHeaders(headersJson?: string): Record<string, string> | undefined {
		if (!headersJson?.trim()) {
			return undefined;
		}

		try {
			const parsed = JSON.parse(headersJson);

			if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed))
				return parsed as Record<string, string>;
		} catch {
			console.warn('[MCPStore] Failed to parse custom headers JSON:', headersJson);
		}

		return undefined;
	}

	private setState(serverId: string, state: HealthCheckState): void {
		this._checks = { ...this._checks, [serverId]: state };
	}
}
