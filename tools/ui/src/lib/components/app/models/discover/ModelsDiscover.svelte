<script lang="ts">
	import { SearchInput } from '$lib/components/app';
	import { ModelsDiscoverDetails, ModelsDiscoverList } from '$lib/components/app/models/discover';
	import { modelsHubStore } from '$lib/stores';

	let selectedId = $state<string | null>(null);
	let searchQuery = $state('');
	let searchTimeout: ReturnType<typeof setTimeout> | null = null;

	// Load the sidebar list on mount (the component is mounted when the dialog opens).
	$effect(() => {
		void modelsHubStore.fetch();
		void modelsHubStore.search('');
	});

	// Auto-select the first model.
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

<aside
	class="w-108 shrink-0 self-start border-r border-border/40 bg-background overflow-y-auto md:p-4 h-full space-y-1"
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

<main class="overflow-y-auto">
	{#if selectedId}
		<ModelsDiscoverDetails modelId={selectedId} />
	{/if}
</main>
