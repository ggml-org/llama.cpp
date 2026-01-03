<!-- [AI] MCP Settings dialog wrapper -->
<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import MCPSettings from '../mcp/MCPSettings.svelte';

	interface Props {
		onOpenChange?: (open: boolean) => void;
		open?: boolean;
	}

	let { onOpenChange, open = false }: Props = $props();

	function handleClose() {
		onOpenChange?.(false);
	}

	function handleSave() {
		onOpenChange?.(false);
	}
</script>

<Dialog.Root {open} onOpenChange={handleClose}>
	<Dialog.Content
		class="z-999999 flex h-[100dvh] max-h-[100dvh] min-h-[100dvh] flex-col gap-0 rounded-none p-0
			md:h-[70vh] md:max-h-[70vh] md:min-h-0 md:rounded-lg"
		style="max-width: 48rem;"
	>
		<MCPSettings onSave={handleSave} />
	</Dialog.Content>
</Dialog.Root>
