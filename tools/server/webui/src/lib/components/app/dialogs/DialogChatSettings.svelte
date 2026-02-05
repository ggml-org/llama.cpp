<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { ChatSettings } from '$lib/components/app';

	interface Props {
		onOpenChange?: (open: boolean) => void;
		open?: boolean;
	}

	let { onOpenChange, open = false }: Props = $props();

	let chatSettingsRef: ChatSettings | undefined = $state();

	function handleClose() {
		onOpenChange?.(false);
	}

	function handleSave() {
		onOpenChange?.(false);
	}

	$effect(() => {
		if (open && chatSettingsRef) {
			chatSettingsRef.reset();
		}
	});
</script>

<Dialog.Root {open} onOpenChange={handleClose}>
	<Dialog.Content
		class="z-999999 flex h-[100dvh] max-h-[100dvh] min-h-[100dvh] max-w-4xl! flex-col gap-0 rounded-none
			p-0 md:h-[64vh] md:max-h-[64vh] md:min-h-0 md:rounded-lg"
	>
		<ChatSettings bind:this={chatSettingsRef} onSave={handleSave} />
	</Dialog.Content>
</Dialog.Root>
