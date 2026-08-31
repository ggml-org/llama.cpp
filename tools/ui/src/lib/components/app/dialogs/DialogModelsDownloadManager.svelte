<script lang="ts">
	import ModelsDownloadManager from '$lib/components/app/models/download-manager/ModelsDownloadManager.svelte';
	import * as Dialog from '$lib/components/ui/dialog';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	function handleOpenChange(next: boolean) {
		open = next;
		onOpenChange?.(next);
	}
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-200!"
	>
		<Dialog.Header>
			<Dialog.Title class="text-sm font-semibold">Download manager</Dialog.Title>
		</Dialog.Header>

		<ModelsDownloadManager />
	</Dialog.Content>
</Dialog.Root>
