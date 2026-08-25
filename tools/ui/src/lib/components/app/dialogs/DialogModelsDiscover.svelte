<script lang="ts">
	import { ModelsDiscover } from '$lib/components/app/models/discover';
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

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="grid gap-0 p-0 md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-360!"
		style="grid-template-columns: auto 1fr;"
	>
		<ModelsDiscover />
	</Dialog.Content>
</Dialog.Root>
