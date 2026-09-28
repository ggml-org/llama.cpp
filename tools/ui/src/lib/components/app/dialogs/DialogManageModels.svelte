<script lang="ts">
	import ModelsManager from '$lib/components/app/models/ModelsManager/ModelsManager.svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import { MODEL_ICON } from '$lib/constants';

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
		class="md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-380! flex flex-col p-4 pb-0"
		onCloseAutoFocus={(event) => event.preventDefault()}
		onOpenAutoFocus={(event) => event.preventDefault()}
	>
		<Dialog.Header class="flex flex-row items-center p-2 pr-8">
			<Dialog.Title class="flex min-h-7 items-center gap-2">
				<MODEL_ICON class="h-5 w-5 shrink-0" />

				<span>Models</span>
			</Dialog.Title>
		</Dialog.Header>

		<div class="min-h-0 flex-1">
			<ModelsManager class="h-full" />
		</div>
	</Dialog.Content>
</Dialog.Root>
