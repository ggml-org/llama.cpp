<script lang="ts">
	import { Settings } from '@lucide/svelte';
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
		class="md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-6xl! flex flex-col p-0 md:p-6 gap-0"
	>
		<Dialog.Header class="md:p-0 p-4">
			<Dialog.Title class="flex items-center gap-2">
				<Settings class="h-5 w-5" />

				<span>Settings</span>
			</Dialog.Title>
		</Dialog.Header>

		<SettingsChat {initialSection} onClose={() => (open = false)} onSectionChange={() => {}} />
	</Dialog.Content>
</Dialog.Root>
