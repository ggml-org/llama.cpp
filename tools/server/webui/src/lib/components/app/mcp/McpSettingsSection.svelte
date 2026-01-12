<script lang="ts">
	import { Plus, X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { getServerDisplayName, getFaviconUrl } from '$lib/utils/mcp';
	import type { MCPServerSettingsEntry } from '$lib/types/mcp';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { McpServerCard } from '$lib/components/app/mcp/McpServerCard';
	import McpServerForm from './McpServerForm.svelte';

	// Get servers from store
	let servers = $derived<MCPServerSettingsEntry[]>(mcpStore.getServers());

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
		mcpStore.addServer({
			enabled: true,
			url: newServerUrl.trim(),
			headers: newServerHeaders.trim() || undefined
		});
		isAddingServer = false;
		newServerUrl = '';
		newServerHeaders = '';
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
			<div class="space-y-4">
				<div class="flex items-center justify-between">
					<p class="font-medium">Add New Server</p>
					<Button
						variant="ghost"
						size="icon"
						class="h-7 w-7"
						onclick={cancelAddServer}
						aria-label="Cancel"
					>
						<X class="h-3.5 w-3.5" />
					</Button>
				</div>

				<McpServerForm
					url={newServerUrl}
					headers={newServerHeaders}
					onUrlChange={(v) => (newServerUrl = v)}
					onHeadersChange={(v) => (newServerHeaders = v)}
					urlError={newServerUrl ? newServerUrlError : null}
					id="new-server"
				/>

				<div class="flex items-center justify-end">
					<Button
						variant="default"
						size="sm"
						onclick={saveNewServer}
						disabled={!!newServerUrlError}
						aria-label="Save"
					>
						Add
					</Button>
				</div>
			</div>
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
					faviconUrl={getFaviconUrl(server.url)}
					onToggle={(enabled) => mcpStore.updateServer(server.id, { enabled })}
					onUpdate={(updates) => mcpStore.updateServer(server.id, updates)}
					onDelete={() => mcpStore.removeServer(server.id)}
				/>
			{/each}
		</div>
	{/if}
</div>
