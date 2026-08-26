<script lang="ts">
	import { Plus } from '@lucide/svelte';
	import { replaceState } from '$app/navigation';
	import { page } from '$app/state';
	import { McpServerCard, McpServerCardSkeleton } from '$lib/components/app';
	import { DialogMcpResourcesBrowser, DialogMcpServerAddNew } from '$lib/components/app/dialogs';
	import { Button } from '$lib/components/ui/button';
	import * as Empty from '$lib/components/ui/empty';
	import { HealthCheckStatus } from '$lib/enums';
	import { conversationsStore, mcpStore, toolsStore } from '$lib/stores';
	import { onMount } from 'svelte';
	import { fade } from 'svelte/transition';

	interface Props {
		class?: string;
	}

	let { class: className }: Props = $props();

	let servers = $derived(mcpStore.getServers());

	let isAddingServer = $state(false);
	let isResourcesDialogOpen = $state(false);

	onMount(() => {
		if (page.url.searchParams.has('add')) {
			isAddingServer = true;

			const newUrl = new URL(page.url);

			newUrl.searchParams.delete('add');

			replaceState(newUrl, {});
		}
	});

	// Each card decides for itself whether to render based on its own
	// health-check state, so adding a server only flashes the new card
	// (not every other already-loaded card) until its health check resolves.
	// Disabled servers never receive a startup health check, so IDLE only
	// counts as pending when the server is enabled; otherwise the real card
	// renders and keeps the enable toggle reachable.
	function isServerPending(serverId: string, enabled: boolean): boolean {
		const status = mcpStore.getHealthCheckState(serverId).status;

		return (
			status === HealthCheckStatus.CONNECTING || (status === HealthCheckStatus.IDLE && enabled)
		);
	}
</script>

<div in:fade={{ duration: 150 }} class="flex flex-col h-full">
	<DialogMcpServerAddNew bind:open={isAddingServer} />

	<DialogMcpResourcesBrowser bind:open={isResourcesDialogOpen} />

	{#if servers.length === 0}
		<div class="flex flex-1 items-center justify-center pb-20 pt-10 my-auto">
			<Empty.Root class="max-w-md">
				<Empty.Header>
					<Empty.Media variant="icon">
						<Plus />
					</Empty.Media>

					<Empty.Title>Add your first MCP server</Empty.Title>

					<Empty.Description>Connect a remote MCP server by URL.</Empty.Description>
				</Empty.Header>

				<Empty.Content>
					<Button onclick={() => (isAddingServer = true)} size="sm">
						<Plus />

						Add New Server
					</Button>
				</Empty.Content>
			</Empty.Root>
		</div>
	{:else}
		<div
			class="grid gap-4 {className}"
			style="grid-template-columns: repeat(auto-fill, minmax(min(25rem, calc(100dvw - 4rem)), 1fr));"
		>
			{#each servers as server (server.id)}
				{#if isServerPending(server.id, server.enabled)}
					<McpServerCardSkeleton />
				{:else}
					<McpServerCard
						enabled={conversationsStore.preferences.isMcpServerEnabledForChat(server.id)}
						onBrowseResources={() => (isResourcesDialogOpen = true)}
						onDelete={() => mcpStore.removeServer(server.id)}
						onToggle={async () => {
							const wasEnabled = conversationsStore.preferences.isMcpServerEnabledForChat(
								server.id
							);

							await conversationsStore.preferences.toggleMcpServerForChat(server.id);

							if (!wasEnabled) {
								// Promote the connection so tools/prompts/resources become
								// available right away instead of waiting for the next chat-init.
								await mcpStore.runHealthCheck(server, true);
								toolsStore.enableAllToolsForServer(server.id);
							}
						}}
						onUpdate={(updates) => mcpStore.updateServer(server.id, updates)}
						{server}
					/>
				{/if}
			{/each}

			{#if !isAddingServer}
				<Empty.Root class="border">
					<Empty.Header>
						<Empty.Media variant="icon">
							<Plus />
						</Empty.Media>

						<Empty.Title>Add another MCP server</Empty.Title>

						<Empty.Description>Connect a remote MCP server by URL.</Empty.Description>
					</Empty.Header>

					<Empty.Content>
						<Button onclick={() => (isAddingServer = true)} size="sm">
							<Plus />

							Add New Server
						</Button>
					</Empty.Content>
				</Empty.Root>
			{/if}
		</div>
	{/if}
</div>
