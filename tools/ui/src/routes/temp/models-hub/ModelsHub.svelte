<script lang="ts">
	import IconFromName from './IconFromName.svelte';
	import ModelDetail from './ModelDetail.svelte';
	import { Download, Heart, Sparkles } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { SearchInput } from '$lib/components/app';
	import { HuggingFaceService, RouterService } from '$lib/services';
	import type { HfModelInfo } from '$lib/types/huggingface';
	import { SvelteMap } from 'svelte/reactivity';

	type SortOption = (typeof HuggingFaceService.SORT_OPTIONS)[number];

	interface Props {
		/** Selected model id from the route (`#/temp/models-hub/<id>`), empty for the list-only view. */
		modelId?: string;
		class?: string;
	}

	let { class: className, modelId = '' }: Props = $props();

	let trendingModels: HfModelInfo[] = $state([]);
	let recommendedModels: HfModelInfo[] = $state([]);
	let searchResults: HfModelInfo[] = $state([]);
	let searchQuery = $state('');
	let searchLoading = $state(false);
	let searchError: string | null = $state(null);
	let loading = $state(true);
	let error: string | null = $state(null);
	let activeFilter = $state<string | null>(null);
	let sortBy = $state<SortOption>('downloads');
	let feed = $state<'trending' | 'recommended'>('trending');

	let selectedModelId = $derived(modelId);
	let isSearching = $derived(searchQuery.trim().length > 0);

	let availableFilters = $derived.by(() => {
		const counts = new SvelteMap<string, number>();

		for (const m of trendingModels) {
			if (!m.pipeline_tag) continue;

			counts.set(m.pipeline_tag, (counts.get(m.pipeline_tag) ?? 0) + 1);
		}

		return Array.from(counts.entries())
			.map(([tag, count]) => ({ count, tag }))
			.sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag));
	});

	let baseModels = $derived(
		isSearching ? searchResults : feed === 'recommended' ? recommendedModels : trendingModels
	);

	let sortedModels = $derived.by(() => {
		const list = [...baseModels];

		list.sort((a, b) => {
			const key = sortBy === 'lastModified' ? 'createdAt' : sortBy;

			return (
				((b as unknown as Record<string, number>)[key] ?? 0) -
				((a as unknown as Record<string, number>)[key] ?? 0)
			);
		});

		return list;
	});

	let filteredModels = $derived.by(() => {
		let list = sortedModels;

		if (activeFilter) list = list.filter((m) => m.pipeline_tag === activeFilter);

		return list;
	});

	async function loadInitial() {
		loading = true;
		error = null;

		try {
			const [trending, recommended] = await Promise.all([
				HuggingFaceService.getTrending(100),
				HuggingFaceService.search({ author: 'ggml-org', limit: 100, sort: 'downloads' })
			]);

			trendingModels = trending.filter((m) => m.tags.includes('gguf'));
			recommendedModels = recommended.filter((m) => m.tags.includes('gguf')).slice(0, 12);
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to fetch models';
		} finally {
			loading = false;
		}
	}

	async function performSearch(query: string) {
		const trimmed = query.trim();

		if (!trimmed) {
			await loadInitial();

			return;
		}

		searchLoading = true;
		searchError = null;

		try {
			searchResults = await HuggingFaceService.searchByQuery(trimmed, { limit: 100 });
		} catch (err) {
			searchError = err instanceof Error ? err.message : 'Search failed';
			searchResults = [];
		} finally {
			searchLoading = false;
		}
	}

	let timeout: ReturnType<typeof setTimeout> | null = null;

	function handleSearchInput(value: string) {
		searchQuery = value;

		if (timeout) clearTimeout(timeout);

		timeout = setTimeout(() => {
			performSearch(value);
		}, 300);
	}

	function openModelDetails(model: HfModelInfo) {
		goto(RouterService.model(model.id));
	}

	$effect(() => {
		loadInitial();

		return () => {
			if (timeout) clearTimeout(timeout);
		};
	});
</script>

<div class="flex min-h-0 flex-col lg:h-full lg:flex-row lg:overflow-hidden {className}">
	<div
		class="{selectedModelId
			? 'hidden lg:flex'
			: 'flex'} h-full min-h-0 w-full flex-col lg:w-80 lg:shrink-0 lg:border-r lg:border-border/40"
	>
		<!-- List header: controls stay pinned while the results scroll below -->
		<div class="shrink-0 space-y-3 border-b border-border/40 bg-background/80 p-3 backdrop-blur">
			<div class="flex items-center justify-between gap-2">
				<h1 class="text-base font-semibold">Models</h1>

				{#if !loading}
					<span class="text-xs text-muted-foreground">{filteredModels.length}</span>
				{/if}
			</div>

			{#if !isSearching}
				<div class="flex rounded-md border bg-muted/40 p-0.5 text-xs font-medium" role="tablist">
					<button
						aria-selected={feed === 'trending'}
						class="flex flex-1 cursor-pointer items-center justify-center gap-1.5 rounded px-2 py-1 transition-colors {feed ===
						'trending'
							? 'bg-background text-foreground shadow-sm'
							: 'text-muted-foreground hover:text-foreground'}"
						onclick={() => (feed = 'trending')}
						role="tab"
						type="button"
					>
						Trending
					</button>

					<button
						aria-selected={feed === 'recommended'}
						class="flex flex-1 cursor-pointer items-center justify-center gap-1.5 rounded px-2 py-1 transition-colors {feed ===
						'recommended'
							? 'bg-background text-foreground shadow-sm'
							: 'text-muted-foreground hover:text-foreground'}"
						onclick={() => (feed = 'recommended')}
						role="tab"
						type="button"
					>
						<Sparkles class="h-3 w-3 text-primary" />
						ggml-org
					</button>
				</div>
			{/if}

			<SearchInput
				bind:value={searchQuery}
				onInput={handleSearchInput}
				placeholder="Search models..."
			/>

			{#if !isSearching}
				<label class="flex items-center gap-2 text-xs text-muted-foreground">
					<span>Sort</span>

					<select
						bind:value={sortBy}
						class="flex-1 rounded-md border bg-background px-2 py-1 text-xs font-medium text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/50"
					>
						{#each HuggingFaceService.SORT_OPTIONS as opt (opt)}
							{#if opt !== 'trendingScore'}
								<option value={opt}>{HuggingFaceService.SORT_LABELS[opt]}</option>
							{/if}
						{/each}
					</select>
				</label>
			{/if}

			{#if !isSearching && availableFilters.length > 0}
				<div class="flex flex-wrap items-center gap-1">
					<button
						class="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium transition-colors {activeFilter ===
						null
							? 'border-primary bg-primary/10 text-primary'
							: 'border-border bg-background text-muted-foreground hover:border-primary/40 hover:text-foreground'}"
						onclick={() => (activeFilter = null)}
						type="button"
					>
						All
					</button>

					{#each availableFilters as f (f.tag)}
						{@const isActive = activeFilter === f.tag}
						<button
							class="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium transition-colors {isActive
								? 'border-primary bg-primary/10 text-primary'
								: 'border-border bg-background text-muted-foreground hover:border-primary/40 hover:text-foreground'}"
							onclick={() => (activeFilter = isActive ? null : f.tag)}
							type="button"
						>
							<IconFromName class="h-3 w-3" name={HuggingFaceService.pipelineTagIcon(f.tag)} />
							{HuggingFaceService.pipelineTagLabel(f.tag)}
							<span class="text-muted-foreground">{f.count}</span>
						</button>
					{/each}
				</div>
			{/if}
		</div>

		<!-- Scrollable results -->
		<div class="min-h-0 flex-1 overflow-y-auto p-2">
			{#if error}
				<div class="mx-1 rounded-lg border border-destructive/50 bg-destructive/5 p-4 text-center">
					<p class="text-xs text-destructive">{error}</p>
				</div>
			{/if}

			{#if searchError}
				<div class="mx-1 rounded-lg border border-destructive/50 bg-destructive/5 p-4 text-center">
					<p class="text-xs text-destructive">{searchError}</p>
				</div>
			{/if}

			{#if loading || searchLoading}
				<div class="flex items-center justify-center py-16">
					<p class="text-xs text-muted-foreground">
						{isSearching ? 'Searching...' : 'Loading models...'}
					</p>
				</div>
			{:else if filteredModels.length === 0}
				<div class="py-16 text-center">
					<p class="text-xs text-muted-foreground">
						{#if isSearching}
							No models found matching "{searchQuery}".
						{:else}
							No models found.
						{/if}
					</p>
				</div>
			{:else}
				<div>
					{#each filteredModels as model (model.id)}
						{@const isActive = model.id === selectedModelId}
						<button
							class="flex w-full cursor-pointer items-start gap-2.5 rounded-lg p-2.5 text-left transition-colors {isActive
								? 'bg-primary/10 hover:bg-primary/15'
								: 'hover:bg-muted/60'}"
							onclick={() => openModelDetails(model)}
							type="button"
						>
							<div
								class="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary"
							>
								<IconFromName
									class="h-4 w-4"
									name={HuggingFaceService.pipelineTagIcon(model.pipeline_tag)}
								/>
							</div>

							<div class="min-w-0 flex-1">
								<div class="flex items-center justify-between gap-2">
									<span class="truncate text-sm font-medium">{model.id}</span>

									<span class="shrink-0 text-[10px] text-muted-foreground">
										{HuggingFaceService.formatRelativeTime(model.createdAt)}
									</span>
								</div>

								{#if model.pipeline_tag}
									<p class="mt-0.5 truncate text-xs text-muted-foreground">
										{HuggingFaceService.pipelineTagLabel(model.pipeline_tag)}
									</p>
								{/if}

								<div class="mt-1 flex items-center gap-3 text-xs text-muted-foreground">
									<span class="flex items-center gap-1">
										<Download class="h-3 w-3" />
										{HuggingFaceService.formatDownloads(model.downloads)}
									</span>

									<span class="flex items-center gap-1">
										<Heart class="h-3 w-3" />
										{HuggingFaceService.formatLikes(model.likes)}
									</span>
								</div>
							</div>
						</button>
					{/each}
				</div>
			{/if}
		</div>
	</div>

	<!-- RIGHT: detail pane -->
	<main
		class="min-w-0 flex-col {selectedModelId
			? 'flex'
			: 'hidden lg:flex'} h-full flex-1 lg:border-l lg:border-border/40"
	>
		{#if selectedModelId}
			<ModelDetail class="h-full" modelId={selectedModelId} />
		{:else}
			<div
				class="flex h-full items-center justify-center px-8 text-center text-sm text-muted-foreground"
			>
				<div class="flex flex-col items-center gap-3">
					<div class="flex h-14 w-14 items-center justify-center rounded-2xl bg-muted/60">
						<Download class="h-6 w-6 text-muted-foreground/60" />
					</div>

					<p class="max-w-xs">
						Select a model from the list to see its details and download options.
					</p>
				</div>
			</div>
		{/if}
	</main>
</div>
