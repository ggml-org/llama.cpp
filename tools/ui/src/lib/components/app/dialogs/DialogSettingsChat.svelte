<script lang="ts">
	import { Settings, X } from '@lucide/svelte';
	import { SettingsChat } from '$lib/components/app/settings';
	import * as Dialog from '$lib/components/ui/dialog';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
		initialSection?: string;
	}

	let { initialSection, onOpenChange, open = $bindable(false) }: Props = $props();

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="max-md:h-[100dvh]! max-md:w-screen! max-md:max-w-none! max-md:rounded-none! md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-6xl! flex flex-col p-0 md:p-6 gap-0"
	>
		<Dialog.Header class="md:p-0 p-4" showCloseButton={false}>
			<Dialog.Title class="flex items-center gap-2">
				<Settings class="h-5 w-5" />

				<span>Settings</span>
			</Dialog.Title>

			<!-- the body is flush with the dialog, so the corner close needs its own inset -->
			<Dialog.Close
				class="absolute top-4 right-4 rounded-xs opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:outline-hidden md:top-0 md:right-0"
			>
				<X class="size-4" />

				<span class="sr-only">Close</span>
			</Dialog.Close>
		</Dialog.Header>

		<SettingsChat {initialSection} onClose={() => (open = false)} onSectionChange={() => {}} />
	</Dialog.Content>
</Dialog.Root>
