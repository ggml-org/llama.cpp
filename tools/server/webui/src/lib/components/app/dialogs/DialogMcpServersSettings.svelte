<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { McpLogo, McpServersSettings } from '$lib/components/app';
	import { mcpStore } from '$lib/stores/mcp.svelte';

	interface Props {
		onOpenChange?: (open: boolean) => void;
		open?: boolean;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	$effect(() => {
		if (open) {
			mcpStore.runHealthChecksForServers(mcpStore.getServers());
		}
	});

	function handleClose() {
		onOpenChange?.(false);
	}
</script>

<Dialog.Root {open} onOpenChange={handleClose}>
	<Dialog.Content
		class="z-999999 flex h-[100dvh] max-h-[100dvh] min-h-[100dvh] flex-col gap-0 rounded-none p-0
			md:h-[80dvh] md:h-auto md:max-h-[80dvh] md:min-h-0 md:rounded-lg"
		style="max-width: 56rem;"
	>
		<div class="grid gap-2 border-b border-border/30 p-4 md:p-6">
			<Dialog.Title class="inline-flex items-center text-lg font-semibold">
				<McpLogo class="mr-2 inline h-4 w-4" />

				MCP Servers
			</Dialog.Title>

			<Dialog.Description class="text-sm text-muted-foreground">
				Add and configure MCP servers to enable agentic tool execution capabilities.
			</Dialog.Description>
		</div>

		<div class="flex-1 overflow-y-auto px-4 py-6">
			<McpServersSettings />
		</div>
	</Dialog.Content>
</Dialog.Root>
