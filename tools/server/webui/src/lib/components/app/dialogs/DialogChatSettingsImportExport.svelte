<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { ChatSettingsImportExportTab } from '$lib/components/app/chat';

	interface Props {
		onOpenChange?: (open: boolean) => void;
		open?: boolean;
	}

	let { onOpenChange, open = false }: Props = $props();

	function handleClose() {
		onOpenChange?.(false);
	}
</script>

<Dialog.Root {open} onOpenChange={handleClose}>
	<Dialog.Content
		class="z-999999 flex h-[100dvh] max-h-[100dvh] min-h-[100dvh] max-w-4xl! flex-col gap-0 rounded-none
			p-0 md:h-[80vh] md:max-h-[80vh] md:min-h-0 md:rounded-lg"
	>
		<div class="grid gap-2 border-b border-border/30 p-4 md:p-6">
			<Dialog.Title class="text-lg font-semibold">Import/Export Conversations</Dialog.Title>
			<Dialog.Description class="text-sm text-muted-foreground">
				Export your conversations to a JSON file or import previously exported conversations.
			</Dialog.Description>
		</div>

		<div class="flex-1 overflow-y-auto px-4 py-6">
			<ChatSettingsImportExportTab />
		</div>
	</Dialog.Content>
</Dialog.Root>
