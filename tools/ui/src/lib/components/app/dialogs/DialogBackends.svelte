<script lang="ts">
	import { SettingsBackends } from '$lib/components/app/backends';
	import { Logo } from '$lib/components/app/misc';
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
	<Dialog.Content class="md:max-h-[80vh]! md:w-[calc(100vw-4rem)]! md:max-w-164! flex flex-col">
		<Dialog.Header>
			<Dialog.Title class="flex items-center gap-2">
				<Logo class="h-5 w-5" style="--size: 1.25rem" />

				<span>Backends</span>
			</Dialog.Title>

			<Dialog.Description>
				Endpoints this UI can talk to. The built-in backend is the llama.cpp server serving it.
			</Dialog.Description>
		</Dialog.Header>

		<SettingsBackends class="mt-4 overflow-y-auto" />
	</Dialog.Content>
</Dialog.Root>
