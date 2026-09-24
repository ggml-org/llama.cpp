<script lang="ts">
	import { Box, Compass, Server } from '@lucide/svelte';
	import SettingsBackends from '$lib/components/app/backends/SettingsBackends.svelte';
	import ModelsDiscover from '$lib/components/app/models/discover/ModelsDiscover.svelte';
	import ModelsManager from '$lib/components/app/models/ModelsManager/ModelsManager.svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import * as Tabs from '$lib/components/ui/tabs';
	import { uiStore } from '$lib/stores';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	let tab = $state('manage');

	// the sidebar's Discover entry opens this dialog on its Discover tab
	$effect(() => {
		if (!uiStore.discoverModelsOpen) return;

		uiStore.discoverModelsOpen = false;
		tab = 'discover';
		handleOpenChange(true);
	});

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-380! flex flex-col p-4"
		onCloseAutoFocus={(event) => event.preventDefault()}
		onOpenAutoFocus={(event) => event.preventDefault()}
	>
		<Dialog.Header class="flex flex-row items-center justify-between p-2 pr-8">
			<Dialog.Title class="flex items-center gap-2">
				<Box class="h-5 w-5" />

				<span>Models</span>
			</Dialog.Title>
		</Dialog.Header>

		<Tabs.Root bind:value={tab} class="mt-2 min-h-0 flex-1 gap-0">
			<div class="px-2">
				<Tabs.List>
					<Tabs.Trigger value="manage">
						<Box class="h-3.5 w-3.5" />

						Manage
					</Tabs.Trigger>

					<Tabs.Trigger value="discover">
						<Compass class="h-3.5 w-3.5" />

						Discover
					</Tabs.Trigger>

					<Tabs.Trigger value="providers">
						<Server class="h-3.5 w-3.5" />

						Providers
					</Tabs.Trigger>
				</Tabs.List>
			</div>

			<Tabs.Content class="flex min-h-0 flex-1 flex-col pt-4" value="manage">
				<ModelsManager />
			</Tabs.Content>

			<Tabs.Content class="min-h-0 flex-1 overflow-hidden pt-4" value="discover">
				<div class="grid h-full min-h-0 gap-0" style="grid-template-columns: auto 1fr;">
					<ModelsDiscover />
				</div>
			</Tabs.Content>

			<Tabs.Content class="min-h-0 flex-1 overflow-y-auto px-2 pt-4" value="providers">
				<SettingsBackends />
			</Tabs.Content>
		</Tabs.Root>
	</Dialog.Content>
</Dialog.Root>
