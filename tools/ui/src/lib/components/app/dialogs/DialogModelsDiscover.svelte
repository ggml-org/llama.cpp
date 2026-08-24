<script lang="ts">
	import { Boxes } from '@lucide/svelte';
	import { SearchInput } from '$lib/components/app';
	import { ModelsDiscoverList, ModelsDiscoverDetails } from '$lib/components/app/models/discover';
	import * as Dialog from '$lib/components/ui/dialog';
	import { modelsHubStore } from '$lib/stores';
	import { untrack } from 'svelte';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	let selectedId = $state<string | null>(null);
	let searchQuery = $state('');
	let searchTimeout: ReturnType<typeof setTimeout> | null = null;

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}

	// Load the sidebar list when the dialog opens, and reset browsing state.
	// Store writes are untracked so this effect only depends on `open`.
	$effect(() => {
		if (open) {
			untrack(() => {
				void modelsHubStore.fetch();
				void modelsHubStore.search('');
			});
			searchQuery = '';
			selectedId = null;
		}
	});

	// Auto-select the first model only when the dialog opens (selection reset).
	$effect(() => {
		const first = modelsHubStore.firstModel;

		if (!selectedId && first) {
			selectedId = first.id;
		}
	});

	function handleSearchInput(value: string) {
		searchQuery = value;

		if (searchTimeout) clearTimeout(searchTimeout);

		searchTimeout = setTimeout(() => {
			void modelsHubStore.search(value);
		}, 300);
	}
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content
		class="flex flex-col md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-360!"
	>
		<Dialog.Header>
			<Dialog.Title class="flex items-center gap-2">
				<Boxes class="h-4 w-4" />
				Discover Models
			</Dialog.Title>
		</Dialog.Header>

		<div class="flex min-h-0 flex-1">
			<aside class="flex w-88 shrink-0 flex-col border-r border-border/40">
				<div class="p-2">
					<SearchInput
						bind:value={searchQuery}
						placeholder="Search models..."
						onInput={handleSearchInput}
					/>
				</div>

				<div class="min-h-0 flex-1 overflow-y-auto">
					{#if modelsHubStore.loading}
						<p class="p-4 text-sm text-muted-foreground">Loading models...</p>
					{:else if modelsHubStore.error}
						<p class="p-4 text-sm text-destructive">{modelsHubStore.error}</p>
					{:else if modelsHubStore.models.length === 0}
						<p class="p-4 text-sm text-muted-foreground">No models found</p>
					{:else}
						<ModelsDiscoverList
							models={modelsHubStore.models}
							activeId={selectedId}
							showBaseModelAvatar
							onSelect={(id) => (selectedId = id)}
						/>
					{/if}
				</div>
			</aside>

			<main class="min-w-0 flex-1 overflow-y-auto">
				{#if selectedId}
					<ModelsDiscoverDetails modelId={selectedId} />
				{/if}
			</main>
		</div>
	</Dialog.Content>
</Dialog.Root>
