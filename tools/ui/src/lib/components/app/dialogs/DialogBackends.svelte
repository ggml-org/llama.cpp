<script lang="ts">
	import { SettingsBackends } from '$lib/components/app/backends';
	import { Logo } from '$lib/components/app/misc';
	import * as Dialog from '$lib/components/ui/dialog';
	import * as Tabs from '$lib/components/ui/tabs';
	import type { BackendProtocol } from '$lib/types';

	const PROTOCOL_TABS: Array<{ label: string; value: BackendProtocol }> = [
		{ label: 'Llama-compat', value: 'llama.cpp' },
		{ label: 'OpenAI-compat', value: 'openai' }
	];

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	let protocol = $state<BackendProtocol>('llama.cpp');

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content class="flex flex-col md:max-h-[80vh]! md:w-[calc(100vw-4rem)]! md:max-w-164!">
		<Dialog.Header>
			<Dialog.Title class="flex items-center gap-2">
				<Logo class="h-5 w-5" style="--size: 1.25rem" />

				<span>Backends</span>
			</Dialog.Title>

			<Dialog.Description>Endpoints this UI can talk to.</Dialog.Description>
		</Dialog.Header>

		<Tabs.Root
			class="min-h-0 flex-1 gap-3"
			onValueChange={(value) => (protocol = value as BackendProtocol)}
			value={protocol}
		>
			<Tabs.List>
				{#each PROTOCOL_TABS as tab (tab.value)}
					<Tabs.Trigger value={tab.value}>{tab.label}</Tabs.Trigger>
				{/each}
			</Tabs.List>

			{#each PROTOCOL_TABS as tab (tab.value)}
				<Tabs.Content class="min-h-0 flex-1 overflow-y-auto" value={tab.value}>
					<SettingsBackends protocol={tab.value} />
				</Tabs.Content>
			{/each}
		</Tabs.Root>
	</Dialog.Content>
</Dialog.Root>
