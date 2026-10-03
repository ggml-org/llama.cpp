<script lang="ts">
	import { McpLogo } from '$lib/components/app';
	import { SettingsMcpServers } from '$lib/components/app/settings';
	import * as Dialog from '$lib/components/ui/dialog';
	import { deviceStore } from '$lib/stores';

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

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="max-md:h-[100dvh]! max-md:w-screen! max-md:max-w-none! max-md:rounded-none! md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-360! flex flex-col p-4"
		onOpenAutoFocus={(event) => deviceStore.isMobile && event.preventDefault()}
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
