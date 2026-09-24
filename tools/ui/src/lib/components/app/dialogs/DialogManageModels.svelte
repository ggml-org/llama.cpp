<script lang="ts">
	import { Box, PackageSearch } from '@lucide/svelte';
	import ModelsManager from '$lib/components/app/models/ModelsManager/ModelsManager.svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { uiStore } from '$lib/stores';

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
		class="md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-360! flex flex-col p-4"
		onOpenAutoFocus={(event) => event.preventDefault()}
	>
		<Dialog.Header class="flex flex-row items-center justify-between p-2 pr-8">
			<Dialog.Title class="flex items-center gap-2">
				<Box class="h-5 w-5" />

				<span>Manage models</span>
			</Dialog.Title>

			<Button onclick={() => uiStore.openDiscoverModels()} size="sm" variant="outline">
				<PackageSearch class="h-3.5 w-3.5" />

				Discover models
			</Button>
		</Dialog.Header>

		<ModelsManager class="mt-4" />
	</Dialog.Content>
</Dialog.Root>
