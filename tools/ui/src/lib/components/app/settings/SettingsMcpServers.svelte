<script lang="ts">
	import McpLogo from '../mcp/McpLogo.svelte';
	import { Plus } from '@lucide/svelte';
	import { browser } from '$app/environment';
	import { goto, replaceState } from '$app/navigation';
	import { page } from '$app/state';
	import { McpServerCard, McpServerCardSkeleton, StandalonePageShell } from '$lib/components/app';
	import { DialogMcpServerAddNew } from '$lib/components/app/dialogs';
	import { Button } from '$lib/components/ui/button';
	import * as Empty from '$lib/components/ui/empty';
	import { ROUTES } from '$lib/constants';
	import { HealthCheckStatus } from '$lib/enums';
	import { conversationsStore, mcpStore, toolsStore } from '$lib/stores';
	import { onMount } from 'svelte';

	interface Props {
		class?: string;
	}

	let { class: className }: Props = $props();

	let servers = $derived(mcpStore.getServers());

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

	// Keep loaded cards visible while new or enabled servers await health checks.
	function isServerPending(serverId: string, enabled: boolean): boolean {
		const status = mcpStore.getHealthCheckState(serverId).status;

		return (
			status === HealthCheckStatus.CONNECTING || (status === HealthCheckStatus.IDLE && enabled)
		);
	}
</script>

<StandalonePageShell icon={McpLogo} title="MCP Servers" onClose={handleClose}>
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
				{#if isServerPending(server.id, server.enabled)}
					<McpServerCardSkeleton />
				{:else}
					<McpServerCard
						{server}
						enabled={conversationsStore.isMcpServerEnabledForChat(server.id)}
						onToggle={async () => {
							const wasEnabled = conversationsStore.isMcpServerEnabledForChat(server.id);

							await conversationsStore.toggleMcpServerForChat(server.id);

							if (!wasEnabled) {
								// Promote the connection so tools/prompts/resources become
								// available right away instead of waiting for the next chat-init.
								await mcpStore.runHealthCheck(server, true);
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
</StandalonePageShell>
