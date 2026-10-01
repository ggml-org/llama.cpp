<script lang="ts">
	import ModelsDiscoverListItem from './ModelsDiscoverList/ModelsDiscoverListItem.svelte';
	import { Download } from '@lucide/svelte';
	import { ModelsSection } from '$lib/components/app';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import {
		ModelsDiscoverDetails,
		ModelsDiscoverList,
		ModelsDiscoverListSearch
	} from '$lib/components/app/models/discover';
	import { Button } from '$lib/components/ui/button';
	import { MODEL_ID } from '$lib/constants';
	import { ModelDownloadConfirmAction } from '$lib/enums';
	import { HuggingFaceService } from '$lib/services';
	import { modelsDiscoverStore, modelsStore, uiStore } from '$lib/stores';
	import type { HfModelDetailInfo, HfModelInfo, HfModelSibling } from '$lib/types';

	let selectedId = $state<string | null>(null);
	let searchQuery = $state('');
	// running and paused downloads both lead the list; a search owns the pane, so the
	// section steps aside while a query is active
	let downloadEntries = $derived(modelsStore.status.getDownloadEntries());
	let isSearching = $derived(searchQuery.trim().length > 0);

	/** Repo id of a download tag, e.g. `org/repo:Q8_0-mtp` reports `org/repo`. */
	function repoOf(repoWithTag: string): string {
		return repoWithTag.split(MODEL_ID.QUANTIZATION_SEPARATOR)[0] ?? repoWithTag;
	}

	/** Hub record for a repo the list does not carry, so its row still renders. */
	function hubRecord(id: string): HfModelInfo {
		return {
			_id: id,
			downloads: 0,
			id,
			library_name: null,
			likes: 0,
			modelId: id,
			pipeline_tag: null,
			private: false,
			tags: []
		};
	}

	// Hub records of repos the list does not carry, e.g. a download started from a
	// search hit: fetched once so the row shows the same badges and icons as a result.
	let fetchedRecords = $state<Record<string, HfModelInfo>>({});

	/** Record of a repo: what the list fetched, else its own details, else a bare one. */
	function recordOf(repo: string): HfModelInfo {
		return (
			modelsDiscoverStore.models.find((model) => model.id === repo) ??
			fetchedRecords[repo] ??
			hubRecord(repo)
		);
	}

	$effect(() => {
		for (const entry of downloadEntries) {
			const repo = repoOf(entry.repoWithTag);

			if (fetchedRecords[repo] || modelsDiscoverStore.models.some((m) => m.id === repo)) continue;

			void HuggingFaceService.getDetails(repo).then((details) => {
				if (details) fetchedRecords = { ...fetchedRecords, [repo]: details };
			});
		}
	});

	// cancel is confirmed once, so a single dialog serves every download row
	// The target is kept while the dialog closes so its copy stays rendered.
	let pendingCancel = $state('');
	let cancelOpen = $state(false);

	function requestCancel(repoWithTag: string) {
		pendingCancel = repoWithTag;
		cancelOpen = true;
	}

	let searchTimeout: ReturnType<typeof setTimeout> | null = null;

	// Detail pane state, reloaded when the selection changes.
	let details = $state<HfModelDetailInfo | null>(null);
	let files = $state<HfModelSibling[]>([]);
	let readme = $state<string | null>(null);
	let detailLoading = $state(false);
	let detailError = $state<string | null>(null);

	// Load the sidebar list on mount (the component is mounted when the dialog opens).
	$effect(() => {
		void modelsDiscoverStore.fetch();
		void modelsDiscoverStore.search('');
	});

	// Auto-select the first model.
	$effect(() => {
		const first = modelsDiscoverStore.firstModel;

		if (!selectedId && first) {
			selectedId = first.id;
		}
	});

	// a caller can ask for one repo to be opened, the manager's rows do
	$effect(() => {
		const focus = uiStore.discoverModelFocus;

		if (!focus) return;

		selectedId = focus;
		uiStore.discoverModelFocus = null;
	});

	function handleSearchInput(value: string) {
		searchQuery = value;

		if (searchTimeout) clearTimeout(searchTimeout);

		searchTimeout = setTimeout(() => {
			void modelsDiscoverStore.search(value);
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

{#snippet catalog()}
	<ModelsDiscoverList
		activeId={selectedId}
		loading={modelsDiscoverStore.loading || modelsDiscoverStore.searching}
		models={modelsDiscoverStore.models}
		onSelect={(id) => (selectedId = id)}
		showBaseModelAvatar
	/>
{/snippet}

<aside
	class="w-md shrink-0 self-start border-r border-border/40 bg-background overflow-y-auto md:pr-4 h-full space-y-4"
>
	<ModelsDiscoverListSearch bind:value={searchQuery} onSearch={handleSearchInput} />

	{#if downloadEntries.length > 0 && !isSearching}
		<ModelsSection
			count={downloadEntries.length}
			label="Downloading"
			persistKey="discover-downloads"
			sticky
			stickyClass="sticky z-10 bg-background"
		>
			{#snippet icon()}
				<Download class="h-3.5 w-3.5 shrink-0" />
			{/snippet}

			{#each downloadEntries as entry (entry.repoWithTag)}
				<!-- the row is the list's own, so a download opens the same details pane a
				     result does; only the state line and the trailing control differ -->
				{@const repo = repoOf(entry.repoWithTag)}

				<ModelsDiscoverListItem
					active={repo === selectedId}
					download={{
						isPaused: entry.isPaused,
						onRequestCancel: requestCancel,
						progress: entry.progress,
						repoWithTag: entry.repoWithTag
					}}
					model={recordOf(repo)}
					onSelect={(id) => (selectedId = id)}
					showBaseModelAvatar
				/>
			{/each}
		</ModelsSection>
	{/if}

	<!-- One list instance, so the rows keep their state across search round trips;
		 skeleton rows replace them while the initial catalog or a query loads. -->
	<div>
		{#if modelsDiscoverStore.error}
			<div class="flex flex-col items-start gap-2 p-4">
				<p class="text-sm text-destructive">{modelsDiscoverStore.error}</p>

				<Button onclick={() => void modelsDiscoverStore.fetch()} size="sm" variant="outline">
					Retry
				</Button>
			</div>
		{:else if !modelsDiscoverStore.loading && !modelsDiscoverStore.searching && modelsDiscoverStore.models.length === 0}
			<p class="p-4 text-sm text-muted-foreground">No models found</p>
		{:else}
			{#if isSearching}
				{@render catalog()}
			{:else}
				<ModelsSection
					count={modelsDiscoverStore.models.length}
					label="Suggested models"
					persistKey="discover-suggested"
					sticky
					stickyClass="sticky z-10 bg-background"
				>
					{@render catalog()}
				</ModelsSection>
			{/if}
		{/if}
	</div>
</aside>

<main class="overflow-y-auto">
	{#if selectedId}
		<ModelsDiscoverDetails
			{details}
			error={detailError}
			{files}
			loading={detailLoading}
			modelId={selectedId}
			{readme}
		/>
	{/if}
</main>

<DialogConfirmDownload
	action={ModelDownloadConfirmAction.CANCEL}
	onClose={() => (cancelOpen = false)}
	open={cancelOpen}
	repoWithTag={pendingCancel}
/>
