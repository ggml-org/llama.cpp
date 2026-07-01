<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { McpServerCard } from '$lib/components/app/mcp';
	import { RECOMMENDED_MCP_SERVERS } from '$lib/constants';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';

	interface Props {
		open: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), onOpenChange }: Props = $props();

	let selected = $state<Record<string, boolean>>(
		Object.fromEntries(RECOMMENDED_MCP_SERVERS.map((server) => [server.id, false]))
	);

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}

	function enableSelected() {
		for (const server of RECOMMENDED_MCP_SERVERS) {
			if (selected[server.id]) {
				conversationsStore.setMcpServerOverride(server.id, true);
			}
		}
		handleOpenChange(false);
	}
</script>

<Dialog.Root bind:open onOpenChange={handleOpenChange}>
	<Dialog.Content class="sm:max-w-lg">
		<Dialog.Header>
			<Dialog.Title>Enable suggested MCP servers?</Dialog.Title>
			<Dialog.Description>
				These servers can provide extra tools for your conversations. Choose the ones you want to
				enable.
			</Dialog.Description>
		</Dialog.Header>

		<div class="max-h-[60vh] space-y-4 overflow-y-auto py-4">
			{#each RECOMMENDED_MCP_SERVERS as server (server.id)}
				<McpServerCard
					{server}
					enabled={selected[server.id]}
					onToggle={(enabled) => (selected[server.id] = enabled)}
					onUpdate={(updates) => mcpStore.updateServer(server.id, updates)}
					onDelete={() => mcpStore.removeServer(server.id)}
				/>
			{/each}
		</div>

		<Dialog.Footer>
			<Button variant="secondary" size="sm" onclick={() => handleOpenChange(false)}>Not now</Button>

			<Button variant="default" size="sm" onclick={enableSelected}>Enable selected</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
