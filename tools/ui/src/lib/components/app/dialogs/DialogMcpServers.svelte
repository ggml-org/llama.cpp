<script lang="ts">
	import { McpLogo } from '$lib/components/app';
	import { SettingsMcpServers } from '$lib/components/app/settings';
	import * as Dialog from '$lib/components/ui/dialog';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content
		class="h-[calc(100vh-4rem)]! max-h-240! w-[calc(100vw-4rem)]! max-w-360! flex flex-col"
	>
		<Dialog.Header>
			<Dialog.Title class="flex items-center gap-2">
				<McpLogo class="h-5 w-5" />

				<span>MCP Servers</span>
			</Dialog.Title>
		</Dialog.Header>

		<SettingsMcpServers class="mt-4" />
	</Dialog.Content>
</Dialog.Root>
