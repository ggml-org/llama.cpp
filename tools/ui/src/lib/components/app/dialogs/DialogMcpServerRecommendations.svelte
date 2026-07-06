<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import * as Dialog from '$lib/components/ui/dialog';
	import { fly } from 'svelte/transition';
	import { McpServerCardCompact, McpServerForm } from '$lib/components/app/mcp';
	import { RECOMMENDED_MCP_SERVERS } from '$lib/constants';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { uuid } from '$lib/utils';
	import { MCP_SERVERS_ADDED_TO_CHAT_LOCALSTORAGE_KEY, MCP_SERVER_ID_PREFIX } from '$lib/constants';
	import type { MCPServerSettingsEntry } from '$lib/types';
	import { Plus } from '@lucide/svelte';

	interface Props {
		open: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), onOpenChange }: Props = $props();

	let selected = $state<Record<string, boolean>>(
		Object.fromEntries(RECOMMENDED_MCP_SERVERS.map((server) => [server.id, false]))
	);

	let addedServers = $state<MCPServerSettingsEntry[]>([]);

	let showAddForm = $state(false);
	let newServerUrl = $state('');
	let newServerHeaders = $state('');
	let newServerUrlError = $derived.by(() => {
		if (!newServerUrl.trim()) return 'URL is required';
		try {
			new URL(newServerUrl);

			return null;
		} catch {
			return 'Invalid URL format';
		}
	});

	function handleOpenChange(value: boolean) {
		if (!value) {
			showAddForm = false;
			newServerUrl = '';
			newServerHeaders = '';
			addedServers = [];

			localStorage.setItem(MCP_SERVERS_ADDED_TO_CHAT_LOCALSTORAGE_KEY, 'true');
		}
		open = value;
		onOpenChange?.(value);
	}

	function resetAddForm() {
		showAddForm = false;
		newServerUrl = '';
		newServerHeaders = '';
	}

	function enableSelected() {
		localStorage.setItem(MCP_SERVERS_ADDED_TO_CHAT_LOCALSTORAGE_KEY, 'true');

		for (const server of RECOMMENDED_MCP_SERVERS) {
			if (selected[server.id]) {
				const existing = mcpStore.getServerById(server.id);
				if (existing) {
					mcpStore.updateServer(server.id, { enabled: true });
				} else {
					mcpStore.addServer({
						id: server.id,
						enabled: true,
						url: server.url,
						name: server.name
					});
				}
				conversationsStore.setMcpServerOverride(server.id, true);
			}
		}
		handleOpenChange(false);
	}

	function saveNewServer() {
		if (newServerUrlError) return;

		const newServerId = uuid() ?? `${MCP_SERVER_ID_PREFIX}-${Date.now()}`;

		localStorage.setItem(MCP_SERVERS_ADDED_TO_CHAT_LOCALSTORAGE_KEY, 'true');

		const newServer = mcpStore.addServer({
			id: newServerId,
			enabled: true,
			url: newServerUrl.trim(),
			headers: newServerHeaders.trim() || undefined
		});

		conversationsStore.setMcpServerOverride(newServerId, true);

		if (newServer) {
			addedServers = [...addedServers, newServer];
		}

		resetAddForm();
	}
</script>

<Dialog.Root bind:open onOpenChange={handleOpenChange}>
	<Dialog.Content class="sm:max-w-lg">
		<Dialog.Header>
			<Dialog.Title>Do more with MCP</Dialog.Title>
			<Dialog.Description>
				Power-up your experience by adding tools, resources and more capabilities provided by MCP
				servers.
			</Dialog.Description>
		</Dialog.Header>

		<div class="max-h-[60vh] space-y-4 overflow-y-auto py-4" in:fly={{ y: 16, duration: 300 }}>
			<h3 class="text-sm font-semibold">Quickly get started with</h3>

			{#each RECOMMENDED_MCP_SERVERS as server (server.id)}
				<McpServerCardCompact
					{server}
					enabled={selected[server.id]}
					onToggle={(enabled) => (selected[server.id] = enabled)}
				/>
			{/each}

			{#if addedServers.length > 0}
				{#each addedServers as server (server.id)}
					<McpServerCardCompact {server} enabled={true} />
				{/each}
			{/if}

			{#if showAddForm}
				<Card.Root class="gap-3! bg-muted/30 p-4">
					<McpServerForm
						url={newServerUrl}
						headers={newServerHeaders}
						onUrlChange={(v) => (newServerUrl = v)}
						onHeadersChange={(v) => (newServerHeaders = v)}
						urlError={newServerUrl ? newServerUrlError : null}
						id="recommendation-new-server"
					/>

					<div class="flex justify-end gap-2 pt-2">
						<Button variant="secondary" size="sm" onclick={resetAddForm}>Cancel</Button>

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
				</Card.Root>
			{:else}
				<Card.Root class="gap-0 border-dashed bg-muted/30 p-0 transition-colors hover:bg-muted/50">
					<button
						type="button"
						class="flex w-full items-center justify-center gap-2 rounded-lg p-6 text-sm text-muted-foreground transition-colors hover:text-foreground"
						onclick={() => (showAddForm = true)}
						aria-label="Add your own MCP server"
					>
						<Plus class="h-4 w-4" />
						<span>Add your own server</span>
					</button>
				</Card.Root>
			{/if}
		</div>

		<Dialog.Footer>
			<Button variant="secondary" size="sm" onclick={() => handleOpenChange(false)}>Not now</Button>

			<Button variant="default" size="sm" onclick={enableSelected}>Add selected</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
