<script lang="ts">
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { McpServerCard, McpServerCardSkeleton } from '$lib/components/app';
	import { HealthCheckStatus } from '$lib/enums';

	interface Props {
		class?: string;
	}

	let { class: className }: Props = $props();

	let servers = $derived(mcpStore.getServersSorted());

	let initialLoadComplete = $state(false);

	$effect(() => {
		if (initialLoadComplete) return;

		const allChecked =
			servers.length > 0 &&
			servers.every((server) => {
				const state = mcpStore.getHealthCheckState(server.id);

				return (
					state.status === HealthCheckStatus.SUCCESS || state.status === HealthCheckStatus.ERROR
				);
			});

		if (allChecked) {
			initialLoadComplete = true;
		}
	});
</script>

<div class="grid gap-3 {className}">
	{#if servers.length === 0}
		<div class="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
			No MCP Servers configured yet. Add one to enable agentic features.
		</div>
	{/if}

	{#if servers.length > 0}
		<div
			class="grid gap-3"
			style="grid-template-columns: repeat(auto-fill, minmax(min(32rem, calc(100dvw - 2rem)), 1fr));"
		>
			{#each servers as server (server.id)}
				{#if !initialLoadComplete}
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
						onDelete={() => {
							void mcpStore.removeServer(server.id);
						}}
					/>
				{/if}
			{/each}
		</div>
	{/if}
</div>
