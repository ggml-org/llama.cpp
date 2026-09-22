<script lang="ts">
	import { Logo } from '$lib/components/app/misc';
	import ModelsManager from '$lib/components/app/models/ModelsManager/ModelsManager.svelte';
	import * as Dialog from '$lib/components/ui/dialog';

	interface Props {
		onOpenDiscover?: () => void;
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, onOpenDiscover, open = $bindable(false) }: Props = $props();

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="flex flex-col md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-380! md:overflow-hidden"
	>
		<Dialog.Header>
			<Dialog.Title class="flex items-center gap-2">
				<Logo class="h-5 w-5" style="--size: 1.25rem" />

				<span>Manage models</span>
			</Dialog.Title>

			<Dialog.Description>Model management goes here.</Dialog.Description>
		</Dialog.Header>

		<ModelsManager {onOpenDiscover} />
	</Dialog.Content>
</Dialog.Root>
