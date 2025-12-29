<script lang="ts">
	import { Loader2, Plus, Trash2 } from '@lucide/svelte';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Input } from '$lib/components/ui/input';
	import Label from '$lib/components/ui/label/label.svelte';
	import { Button } from '$lib/components/ui/button';
	import { parseMcpServerSettings } from '$lib/config/mcp';
	import { detectMcpTransportFromUrl } from '$lib/utils/mcp';
	import type { MCPServerSettingsEntry } from '$lib/types/mcp';
	import { MCPClient } from '$lib/mcp';
	import type { SettingsConfigType } from '$lib/types/settings';
	import { DEFAULT_MCP_CONFIG } from '$lib/constants/mcp';

	interface Props {
		localConfig: SettingsConfigType;
		onConfigChange: (key: string, value: string | boolean) => void;
	}

	let { localConfig, onConfigChange }: Props = $props();

	type HealthCheckState =
		| { status: 'idle' }
		| { status: 'loading' }
		| { status: 'error'; message: string }
		| { status: 'success'; tools: { name: string; description?: string }[] };

	let healthChecks: Record<string, HealthCheckState> = $state({});

	function serializeServers(servers: MCPServerSettingsEntry[]) {
		onConfigChange('mcpServers', JSON.stringify(servers));
	}

	function getServers(): MCPServerSettingsEntry[] {
		return parseMcpServerSettings(localConfig.mcpServers);
	}

	function addServer() {
		const servers = getServers();
		const newServer: MCPServerSettingsEntry = {
			id: crypto.randomUUID ? crypto.randomUUID() : `server-${Date.now()}`,
			enabled: true,
			url: '',
			requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds
		};

		serializeServers([...servers, newServer]);
	}

	function updateServer(id: string, updates: Partial<MCPServerSettingsEntry>) {
		const servers = getServers();
		const nextServers = servers.map((server) =>
			server.id === id
				? {
						...server,
						...updates
					}
				: server
		);

		serializeServers(nextServers);
	}

	function removeServer(id: string) {
		const servers = getServers().filter((server) => server.id !== id);
		serializeServers(servers);
	}

	function getHealthState(id: string): HealthCheckState {
		return healthChecks[id] ?? { status: 'idle' };
	}

	function isErrorState(state: HealthCheckState): state is { status: 'error'; message: string } {
		return state.status === 'error';
	}

	function isSuccessState(
		state: HealthCheckState
	): state is { status: 'success'; tools: { name: string; description?: string }[] } {
		return state.status === 'success';
	}

	function setHealthState(id: string, state: HealthCheckState) {
		healthChecks = { ...healthChecks, [id]: state };
	}

	async function runHealthCheck(server: MCPServerSettingsEntry) {
		const trimmedUrl = server.url.trim();

		if (!trimmedUrl) {
			setHealthState(server.id, {
				status: 'error',
				message: 'Please enter a server URL before running a health check.'
			});
			return;
		}

		setHealthState(server.id, { status: 'loading' });

		const timeoutMs = Math.round(server.requestTimeoutSeconds * 1000);

		const mcpClient = new MCPClient({
			protocolVersion: DEFAULT_MCP_CONFIG.protocolVersion,
			capabilities: DEFAULT_MCP_CONFIG.capabilities,
			clientInfo: DEFAULT_MCP_CONFIG.clientInfo,
			requestTimeoutMs: timeoutMs,
			servers: {
				[server.id]: {
					url: trimmedUrl,
					transport: detectMcpTransportFromUrl(trimmedUrl),
					handshakeTimeoutMs: DEFAULT_MCP_CONFIG.connectionTimeoutMs,
					requestTimeoutMs: timeoutMs
				}
			}
		});

		try {
			await mcpClient.initialize();
			const tools = (await mcpClient.getToolsDefinition()).map((tool) => ({
				name: tool.function.name,
				description: tool.function.description
			}));

			setHealthState(server.id, { status: 'success', tools });
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Unknown error occurred';
			setHealthState(server.id, { status: 'error', message });
		} finally {
			try {
				await mcpClient.shutdown();
			} catch (shutdownError) {
				console.warn('[MCP] Failed to cleanly shutdown client', shutdownError);
			}
		}
	}
</script>

<div class="space-y-4">
	<div class="flex items-center justify-between gap-4">
		<div>
			<h4 class="text-base font-semibold">MCP Servers</h4>
			<p class="text-sm text-muted-foreground">
				Configure one or more MCP Servers. Only enabled servers with a URL are used.
			</p>
		</div>

		<Button variant="outline" class="shrink-0" onclick={addServer}>
			<Plus class="mr-2 h-4 w-4" />
			Add MCP Server
		</Button>
	</div>

	{#if getServers().length === 0}
		<div class="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
			No MCP Servers configured yet. Add one to enable agentic features.
		</div>
	{/if}

	<div class="space-y-3">
		{#each getServers() as server, index (server.id)}
			{@const healthState = getHealthState(server.id)}

			<div class="space-y-3 rounded-lg border p-4 shadow-sm">
				<div class="flex flex-wrap items-center gap-3">
					<div class="flex items-center gap-2">
						<Checkbox
							id={`mcp-enabled-${server.id}`}
							checked={server.enabled}
							onCheckedChange={(checked) =>
								updateServer(server.id, {
									enabled: Boolean(checked)
								})}
						/>
						<div class="space-y-1">
							<Label for={`mcp-enabled-${server.id}`} class="cursor-pointer text-sm font-medium">
								MCP Server {index + 1}
							</Label>
							<p class="text-xs text-muted-foreground">
								{detectMcpTransportFromUrl(server.url) === 'websocket'
									? 'WebSocket'
									: 'Streamable HTTP'}
							</p>
						</div>
					</div>

					<div class="ml-auto flex items-center gap-2">
						<Button
							variant="ghost"
							size="icon"
							class="text-muted-foreground hover:text-foreground"
							onclick={() => removeServer(server.id)}
							aria-label={`Remove MCP Server ${index + 1}`}
						>
							<Trash2 class="h-4 w-4" />
						</Button>
					</div>
				</div>

				<div class="space-y-3">
					<div class="space-y-2">
						<Label class="text-sm font-medium">Endpoint URL</Label>
						<Input
							value={server.url}
							placeholder="http://127.0.0.1:8080 or ws://..."
							class="w-full"
							oninput={(event) =>
								updateServer(server.id, {
									url: event.currentTarget.value
								})}
						/>
					</div>

					<div class="space-y-2 md:min-w-[14rem]">
						<Label class="text-sm font-medium">Request timeout (seconds)</Label>
						<div class="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
							<Input
								type="number"
								min="1"
								inputmode="numeric"
								value={String(server.requestTimeoutSeconds ?? '')}
								class="w-20 sm:w-28"
								oninput={(event) => {
									const parsed = Number(event.currentTarget.value);
									updateServer(server.id, {
										requestTimeoutSeconds:
											Number.isFinite(parsed) && parsed > 0
												? parsed
												: DEFAULT_MCP_CONFIG.requestTimeoutSeconds
									});
								}}
							/>

							<Button
								variant="secondary"
								size="sm"
								class="w-full sm:ml-auto sm:w-auto"
								onclick={() => runHealthCheck(server)}
								disabled={healthState.status === 'loading'}
							>
								Health Check
							</Button>
						</div>
					</div>
				</div>

				{#if healthState.status !== 'idle'}
					<div class="space-y-2 text-sm">
						{#if healthState.status === 'loading'}
							<div class="flex items-center gap-2 text-muted-foreground">
								<Loader2 class="h-4 w-4 animate-spin" />
								<span>Running health check...</span>
							</div>
						{:else if isErrorState(healthState)}
							<p class="text-destructive">
								Health check failed: {healthState.message}
							</p>
						{:else if isSuccessState(healthState)}
							{#if healthState.tools.length === 0}
								<p class="text-muted-foreground">No tools returned by this server.</p>
							{:else}
								<div class="space-y-2">
									<p class="font-medium">
										Available tools ({healthState.tools.length})
									</p>
									<ul class="space-y-2">
										{#each healthState.tools as tool (tool.name)}
											<li class="leading-relaxed">
												<span
													class="mr-2 inline-flex items-center rounded bg-accent/70 px-2 py-0.5 font-semibold text-accent-foreground"
												>
													{tool.name}
												</span>
												<span class="text-muted-foreground"
													>{tool.description ?? 'No description provided.'}</span
												>
											</li>
										{/each}
									</ul>
								</div>
							{/if}
						{/if}
					</div>
				{/if}
			</div>
		{/each}
	</div>
</div>
