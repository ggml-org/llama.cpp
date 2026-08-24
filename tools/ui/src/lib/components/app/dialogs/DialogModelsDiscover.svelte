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
		class="grid gap-0 p-0 md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-360!" style="grid-template-columns: auto 1fr;"
	>
			<aside
				class="sticky top-0 w-100 shrink-0 self-start border-r border-border/40 bg-background overflow-y-auto md:p-4 h-full space-y-1 md:max-h-239.5!"
			>
				<div class="p-2 sticky top-0 z-99">
					<SearchInput
					    class=""
						bind:value={searchQuery}
						placeholder="Search models..."
						onInput={handleSearchInput}
					/>
				</div>

				<div>
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

			<main>
				{#if selectedId}
					<ModelsDiscoverDetails modelId={selectedId} />
				{/if}
			</main>
	</Dialog.Content>
</Dialog.Root>
