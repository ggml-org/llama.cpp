<script lang="ts">
	import { Plus } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { McpServerCard, McpServerCardSkeleton } from '$lib/components/app/mcp';
	import { DialogMcpServerAddNew } from '$lib/components/app/dialogs';
	import { HealthCheckStatus } from '$lib/enums';
	import { fade } from 'svelte/transition';
	import { onMount } from 'svelte';
	import McpLogo from '../mcp/McpLogo.svelte';
	import { page } from '$app/state';
	import { replaceState } from '$app/navigation';

	interface Props {
		class?: string;
	}

	let { class: className }: Props = $props();

	let servers = $derived(mcpStore.getServersSorted());

	let initialLoadComplete = $state(false);
	let isAddingServer = $state(false);

	onMount(() => {
		if (page.url.searchParams.has('add')) {
			isAddingServer = true;

			const newUrl = new URL(page.url);
			newUrl.searchParams.delete('add');

			replaceState(newUrl, {});
		}
	});

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

<div in:fade={{ duration: 150 }} class="h-full max-h-[100dvh] overflow-y-auto">
	<div class="flex items-center gap-2 p-4 md:absolute md:top-8 md:left-8 md:px-0 md:py-2">
		<McpLogo class="h-5 w-5 md:h-6 md:w-6" />

		<h1 class="text-xl font-semibold md:text-2xl">MCP Servers</h1>
	</div>

	<div class="sticky top-0 z-10 mt-4 flex items-start gap-4 p-4 md:justify-end md:px-8">
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
			</div>
		{/if}
	</div>
</div>
