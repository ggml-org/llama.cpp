<script lang="ts">
	import { SettingsBackends } from '$lib/components/app/backends';
	import { Logo } from '$lib/components/app/misc';
	import { ModelsDiscover } from '$lib/components/app/models/discover';
	import * as Dialog from '$lib/components/ui/dialog';
	import * as Tabs from '$lib/components/ui/tabs';
	import { SETTINGS_KEYS } from '$lib/constants';
	import { serverStore, settingsStore } from '$lib/stores';
	import type { BackendProtocol } from '$lib/types';

	type ManageTab = 'discover' | 'providers';

	const PROTOCOL_TABS: Array<{ label: string; value: BackendProtocol }> = [
		{ label: 'Llama-compat', value: 'llama.cpp' },
		{ label: 'OpenAI-compat', value: 'openai' }
	];

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	// discovery needs the local router's download endpoints
	let hasDiscover = $derived(
		serverStore.localIsRouter && settingsStore.config[SETTINGS_KEYS.ENABLE_DISCOVER_MODELS] === true
	);
	let tab = $state<ManageTab>('discover');
	let providerProtocol = $state<BackendProtocol>('llama.cpp');

	// land on the first available tab each time the dialog opens
	$effect(() => {
		if (!open) return;

		tab = hasDiscover ? 'discover' : 'providers';
	});

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="flex flex-col md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-360! md:overflow-hidden"
	>
		<Dialog.Header>
			<Dialog.Title class="flex items-center gap-2">
				<Logo class="h-5 w-5" style="--size: 1.25rem" />

				<span>Manage models</span>
			</Dialog.Title>

			<Dialog.Description>
				Download models from the catalog and manage the providers this UI can talk to.
			</Dialog.Description>
		</Dialog.Header>

		<Tabs.Root
			class="min-h-0 flex-1 gap-3"
			onValueChange={(value) => (tab = value as ManageTab)}
			value={tab}
		>
			<Tabs.List>
				{#if hasDiscover}
					<Tabs.Trigger value="discover">Discover</Tabs.Trigger>
				{/if}

				<Tabs.Trigger value="providers">Providers</Tabs.Trigger>
			</Tabs.List>

			{#if hasDiscover}
				<Tabs.Content
					class="grid min-h-0 flex-1 overflow-hidden"
					style="grid-template-columns: auto 1fr;"
					value="discover"
				>
					<ModelsDiscover />
				</Tabs.Content>
			{/if}

			<Tabs.Content class="min-h-0 flex-1 overflow-y-auto" value="providers">
				<Tabs.Root
					class="gap-3"
					onValueChange={(value) => (providerProtocol = value as BackendProtocol)}
					value={providerProtocol}
				>
					<Tabs.List>
						{#each PROTOCOL_TABS as protocolTab (protocolTab.value)}
							<Tabs.Trigger value={protocolTab.value}>{protocolTab.label}</Tabs.Trigger>
						{/each}
					</Tabs.List>

					{#each PROTOCOL_TABS as protocolTab (protocolTab.value)}
						<Tabs.Content value={protocolTab.value}>
							<SettingsBackends protocol={protocolTab.value} />
						</Tabs.Content>
					{/each}
				</Tabs.Root>
			</Tabs.Content>
		</Tabs.Root>
	</Dialog.Content>
</Dialog.Root>
