<script lang="ts">
	import {
		ModelsDiscoverList,
		ModelsDiscoverListSearch,
		ModelsDiscoverModelDetails
	} from '$lib/components/app/models/discover';
	import { HuggingFaceService } from '$lib/services';
	import { modelsHubStore } from '$lib/stores';
	import type { HfModelDetailInfo, HfModelSibling } from '$lib/types';

	let selectedId = $state<string | null>(null);
	let searchQuery = $state('');
	let searchTimeout: ReturnType<typeof setTimeout> | null = null;

	// Detail pane state, reloaded when the selection changes.
	let details = $state<HfModelDetailInfo | null>(null);
	let files = $state<HfModelSibling[]>([]);
	let readme = $state<string | null>(null);
	let detailLoading = $state(false);
	let detailError = $state<string | null>(null);

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

	// Load the detail pane for the selected model (component is reused across
	// selections, so this re-fetches on every change).
	$effect(() => {
		const id = selectedId;

		if (!id) return;

		let cancelled = false;

		detailLoading = true;
		detailError = null;

		void (async () => {
			try {
				const [info, tree, readmeText] = await Promise.all([
					HuggingFaceService.getDetails(id),
					HuggingFaceService.getTree(id),
					HuggingFaceService.getReadme(id)
				]);

				if (cancelled) return;

				if (!info) {
					detailError = 'Model not found';

					return;
				}

				details = info;
				files = HuggingFaceService.filterByExtension(
					HuggingFaceService.collapseGgufShards(tree),
					'.gguf'
				);
				readme = readmeText;
			} catch (err) {
				if (cancelled) return;

				detailError = err instanceof Error ? err.message : 'Failed to load model';
			} finally {
				if (!cancelled) detailLoading = false;
			}
		})();

		return () => {
			cancelled = true;
		};
	});
</script>

<aside
	class="w-md shrink-0 self-start border-r border-border/40 bg-background overflow-y-auto md:p-4 h-full space-y-1"
>
	<ModelsDiscoverListSearch bind:value={searchQuery} onSearch={handleSearchInput} />

	<!-- One list instance, so the rows keep their state across search round trips;
		 skeleton rows replace them while the initial catalog or a query loads. -->
	<div>
		{#if modelsHubStore.error}
			<p class="p-4 text-sm text-destructive">{modelsHubStore.error}</p>
		{:else if !modelsHubStore.loading && !modelsHubStore.searching && modelsHubStore.models.length === 0}
			<p class="p-4 text-sm text-muted-foreground">No models found</p>
		{:else}
			<ModelsDiscoverList
				activeId={selectedId}
				loading={modelsHubStore.loading || modelsHubStore.searching}
				models={modelsHubStore.models}
				onSelect={(id) => (selectedId = id)}
				showBaseModelAvatar
			/>
		{/if}
	</div>
</aside>

<main class="overflow-y-auto">
	{#if selectedId}
		<ModelsDiscoverModelDetails
			{details}
			error={detailError}
			{files}
			loading={detailLoading}
			modelId={selectedId}
			{readme}
		/>
	{/if}
</main>
