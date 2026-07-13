<script lang="ts">
	import { X, Plus } from '@lucide/svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Empty from '$lib/components/ui/empty';
	import { ActionIcon, McpServerCard, McpServerCardSkeleton } from '$lib/components/app';
	import { DialogMcpServerAddNew } from '$lib/components/app/dialogs';
	import { HealthCheckStatus } from '$lib/enums';
	import { ROUTES } from '$lib/constants';
	import { fade } from 'svelte/transition';
	import { onMount } from 'svelte';
	import McpLogo from '../mcp/McpLogo.svelte';
	import { browser } from '$app/environment';
	import { page } from '$app/state';
	import { goto, replaceState } from '$app/navigation';

	interface Props {
		class?: string;
	}

	let { class: className }: Props = $props();

	let servers = $derived(mcpStore.visibleMcpServers);

	let isAddingServer = $state(false);

	let previousRouteId = $state<string | null>(null);

	$effect(() => {
		const currentId = page.route.id;
		return () => {
			previousRouteId = currentId;
		};
	});

	function handleClose() {
		const prevIsMcpServers = previousRouteId === '/mcp-servers';
		if (browser && window.history.length > 1 && !prevIsMcpServers) {
			history.back();
		} else {
			goto(ROUTES.START);
		}
	}

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
	function isServerPending(serverId: string): boolean {
		const status = mcpStore.getHealthCheckState(serverId).status;
		return status === HealthCheckStatus.IDLE || status === HealthCheckStatus.CONNECTING;
	}
</script>

<div in:fade={{ duration: 150 }} class="flex min-h-[calc(100dvh-4rem)] flex-col">
	<div class="fixed top-4.5 right-4 z-50 md:hidden">
		<ActionIcon icon={X} tooltip="Close" onclick={handleClose} />
	</div>

	<div
		class="sticky top-0 z-10 mt-4 mb-2 flex items-start gap-4 md:p-4 p-0 px-4 md:justify-between md:px-8"
	>
		<div class="flex items-center gap-2">
			<McpLogo class="h-5 w-5 md:h-6 md:w-6" />

			<h1 class="text-lg font-semibold md:text-2xl">MCP Servers</h1>
		</div>
	</div>

	<DialogMcpServerAddNew bind:open={isAddingServer} />

	{#if servers.length === 0}
		<div class="flex flex-1 items-center justify-center py-16">
			<Empty.Root class="max-w-md">
				<Empty.Header>
					<Empty.Media variant="icon">
						<Plus />
					</Empty.Media>

					<Empty.Title>Add your first MCP server</Empty.Title>

					<Empty.Description>Connect a remote MCP server by URL.</Empty.Description>
				</Empty.Header>

				<Empty.Content>
					<Button size="sm" onclick={() => (isAddingServer = true)}>
						<Plus />

						Add New Server
					</Button>
				</Empty.Content>
			</Empty.Root>
		</div>
	{:else}
		<div
			class="grid gap-3 {className}"
			style="grid-template-columns: repeat(auto-fill, minmax(min(32rem, calc(100dvw - 2rem)), 1fr));"
		>
			{#each servers as server (server.id)}
				{#if isServerPending(server.id)}
					<McpServerCardSkeleton />
				{:else}
					<McpServerCard
						{server}
						enabled={conversationsStore.isMcpServerEnabledForChat(server.id)}
						onToggle={async () => {
							const wasEnabled = conversationsStore.isMcpServerEnabledForChat(server.id);
							await conversationsStore.toggleMcpServerForChat(server.id);
							if (!wasEnabled) {
								toolsStore.enableAllToolsForServer(server.id);
							}
						}}
						onUpdate={(updates) => mcpStore.updateServer(server.id, updates)}
						onDelete={() => mcpStore.removeServer(server.id)}
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
						<Button size="sm" onclick={() => (isAddingServer = true)}>
							<Plus />

							Add New Server
						</Button>
					</Empty.Content>
				</Empty.Root>
			{/if}
		</div>
	{/if}
</div>
