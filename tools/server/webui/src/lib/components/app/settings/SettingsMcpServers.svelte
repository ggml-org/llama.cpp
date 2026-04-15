<script lang="ts">
	import { Plus } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { McpServerCard, McpServerCardSkeleton } from '$lib/components/app/mcp';
	import { DialogMcpServerAddNew } from '$lib/components/app/dialogs';
	import { HealthCheckStatus } from '$lib/enums';
	import { fade } from 'svelte/transition';
	import McpLogo from '../mcp/McpLogo.svelte';

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

	let isAddingServer = $state(false);
</script>

<div in:fade={{ duration: 150 }} class="max-h-full overflow-auto">
	<div class="flex items-center gap-2 p-4 md:absolute md:top-8 md:left-8 md:p-0">
		<McpLogo class="h-5 w-5 md:h-6 md:w-6" />

		<h1 class="text-xl font-semibold md:text-2xl">MCP Servers</h1>
	</div>

	<div class="sticky top-0 z-10 mt-4 flex items-start justify-end gap-4 px-8 py-4">
		<Button variant="outline" size="sm" class="shrink-0" onclick={() => (isAddingServer = true)}>
			<Plus class="h-4 w-4" />

			Add New Server
		</Button>
	</div>

	<DialogMcpServerAddNew bind:open={isAddingServer} />

	<div class="grid gap-5 md:space-y-4 {className}">
		{#if servers.length === 0 && !isAddingServer}
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
							faviconUrl={mcpStore.getServerFavicon(server.id)}
							enabled={conversationsStore.isMcpServerEnabledForChat(server.id)}
							onToggle={async () => await conversationsStore.toggleMcpServerForChat(server.id)}
							onUpdate={(updates) => mcpStore.updateServer(server.id, updates)}
							onDelete={() => mcpStore.removeServer(server.id)}
						/>
					{/if}
				{/each}
			</div>
		{/if}
	</div>
</div>
