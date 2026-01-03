<script lang="ts">
	import { Plus, X, Check } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { parseMcpServerSettings } from '$lib/config/mcp';
	import type { MCPServerSettingsEntry } from '$lib/types/mcp';
	import type { SettingsConfigType } from '$lib/types/settings';
	import { DEFAULT_MCP_CONFIG } from '$lib/constants/mcp';
	import McpServerCard from './McpServerCard.svelte';
	import McpServerForm from './McpServerForm.svelte';

	interface Props {
		localConfig: SettingsConfigType;
		onConfigChange: (key: string, value: string | boolean) => void;
	}

	let { localConfig, onConfigChange }: Props = $props();

	// Get servers from localConfig
	let servers = $derived<MCPServerSettingsEntry[]>(parseMcpServerSettings(localConfig.mcpServers));

	// New server form state
	let isAddingServer = $state(false);
	let newServerUrl = $state('');
	let newServerHeaders = $state('');

	// Validation for new server URL
	let newServerUrlError = $derived.by(() => {
		if (!newServerUrl.trim()) return 'URL is required';
		try {
			new URL(newServerUrl);
			return null;
		} catch {
			return 'Invalid URL format';
		}
	});

	function serializeServers(updatedServers: MCPServerSettingsEntry[]) {
		onConfigChange('mcpServers', JSON.stringify(updatedServers));
	}

	function showAddServerForm() {
		isAddingServer = true;
		newServerUrl = '';
		newServerHeaders = '';
	}

	function cancelAddServer() {
		isAddingServer = false;
		newServerUrl = '';
		newServerHeaders = '';
	}

	function saveNewServer() {
		if (newServerUrlError) return;
		const newServer: MCPServerSettingsEntry = {
			id: crypto.randomUUID ? crypto.randomUUID() : `server-${Date.now()}`,
			enabled: true,
			url: newServerUrl.trim(),
			headers: newServerHeaders.trim() || undefined,
			requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds
		};
		serializeServers([...servers, newServer]);
		isAddingServer = false;
		newServerUrl = '';
		newServerHeaders = '';
	}

	function updateServer(id: string, updates: Partial<MCPServerSettingsEntry>) {
		const nextServers = servers.map((server) =>
			server.id === id ? { ...server, ...updates } : server
		);
		serializeServers(nextServers);
	}

	function removeServer(id: string) {
		serializeServers(servers.filter((server) => server.id !== id));
	}

	// Get display name for server
	function getServerDisplayName(server: MCPServerSettingsEntry): string {
		if (server.name) return server.name;
		try {
			const url = new URL(server.url);
			const host = url.hostname.replace(/^(www\.|mcp\.)/, '');
			const name = host.split('.')[0] || 'Unknown';
			return name.charAt(0).toUpperCase() + name.slice(1);
		} catch {
			return 'New Server';
		}
	}

	// Get favicon URL for server
	function getFaviconUrl(server: MCPServerSettingsEntry): string | null {
		try {
			const url = new URL(server.url);
			const hostnameParts = url.hostname.split('.');
			const rootDomain =
				hostnameParts.length >= 2 ? hostnameParts.slice(-2).join('.') : url.hostname;
			return `https://www.google.com/s2/favicons?domain=${rootDomain}&sz=32`;
		} catch {
			return null;
		}
	}
</script>

<div class="space-y-4">
	<div class="flex items-center justify-between gap-4">
		<div>
			<h4 class="text-base font-semibold">Manage Servers</h4>
		</div>

		{#if !isAddingServer}
			<Button variant="outline" size="sm" class="shrink-0" onclick={showAddServerForm}>
				<Plus class="h-4 w-4" />
				Add New Server
			</Button>
		{/if}
	</div>

	<!-- Add New Server Form -->
	{#if isAddingServer}
		<Card.Root class="bg-muted/30 p-4">
			<div class="mb-3 flex items-center justify-between">
				<p class="font-medium">Add New Server</p>
				<div class="flex items-center gap-1">
					<Button
						variant="ghost"
						size="icon"
						class="h-7 w-7"
						onclick={cancelAddServer}
						aria-label="Cancel"
					>
						<X class="h-3.5 w-3.5" />
					</Button>
					<Button
						variant="ghost"
						size="icon"
						class="h-7 w-7 text-green-600 hover:bg-green-100 hover:text-green-700 dark:text-green-500 dark:hover:bg-green-950"
						onclick={saveNewServer}
						disabled={!!newServerUrlError}
						aria-label="Save"
					>
						<Check class="h-3.5 w-3.5" />
					</Button>
				</div>
			</div>
			<McpServerForm
				url={newServerUrl}
				headers={newServerHeaders}
				onUrlChange={(v) => (newServerUrl = v)}
				onHeadersChange={(v) => (newServerHeaders = v)}
				urlError={newServerUrl ? newServerUrlError : null}
				id="new-server"
			/>
		</Card.Root>
	{/if}

	{#if servers.length === 0 && !isAddingServer}
		<div class="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
			No MCP Servers configured yet. Add one to enable agentic features.
		</div>
	{/if}

	<!-- Server Cards -->
	{#if servers.length > 0}
		<div class="space-y-3">
			{#each servers as server (server.id)}
				<McpServerCard
					{server}
					displayName={getServerDisplayName(server)}
					faviconUrl={getFaviconUrl(server)}
					onToggle={(enabled) => updateServer(server.id, { enabled })}
					onUpdate={(updates) => updateServer(server.id, updates)}
					onDelete={() => removeServer(server.id)}
				/>
			{/each}
		</div>
	{/if}
</div>
