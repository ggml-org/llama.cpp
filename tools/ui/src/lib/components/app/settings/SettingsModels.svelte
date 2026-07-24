<script lang="ts">
	import { Download, Heart, Search as SearchIcon, Sparkles, X } from '@lucide/svelte';
	import { HuggingFaceService } from '$lib/services';
	import { RouterService } from '$lib/services/router.service';
	import type { HfModelInfo } from '$lib/types/huggingface';
	import { goto } from '$app/navigation';
	import { browser } from '$app/environment';
	import { ActionIcon, IconFromName, SearchInput } from '$lib/components/app';
	import { SvelteMap } from 'svelte/reactivity';
	import { ROUTES } from '$lib/constants';
	import { fade } from 'svelte/transition';

	type SortOption = (typeof HuggingFaceService.SORT_OPTIONS)[number];

	interface Props {
		class?: string;
	}

	let { class: className }: Props = $props();

	let models: HfModelInfo[] = $state([]);
	let recommendedModels: HfModelInfo[] = $state([]);
	let searchQuery = $state('');
	let searchLoading = $state(false);
	let searchError: string | null = $state(null);
	let searchTimeout: ReturnType<typeof setTimeout> | null = null;
	let loading = $state(true);
	let error: string | null = $state(null);

	// Trending filter chips: pipeline tags actually present in the loaded
	// models. `null` means "All". `pipeline_tag` is the raw HF API value.
	let activeFilter = $state<string | null>(null);
	let sortBy = $state<SortOption>('downloads');

	let availableFilters = $derived.by(() => {
		const counts = new SvelteMap<string, number>();
		for (const m of models) {
			if (!m.pipeline_tag) continue;
			counts.set(m.pipeline_tag, (counts.get(m.pipeline_tag) ?? 0) + 1);
		}
		return Array.from(counts.entries())
			.map(([tag, count]) => ({ tag, count }))
			.sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag));
	});

	let sortedModels = $derived.by(() => {
		const list = [...models];
		list.sort((a, b) => {
			// HfModelSort includes 'lastModified' but HfModelInfo lists 'createdAt' only;
			// narrowing keeps us safe without changing the type contract.
			const key = sortBy === 'lastModified' ? 'createdAt' : sortBy;
			return (
				((b as unknown as Record<string, number>)[key] ?? 0) -
				((a as unknown as Record<string, number>)[key] ?? 0)
			);
		});
		return list;
	});

	let filteredRecommendedModels = $derived.by(() => {
		const query = searchQuery.trim().toLowerCase();
		if (!query) return recommendedModels;
		return recommendedModels.filter((m) => m.id.toLowerCase().includes(query));
	});

	let filteredModels = $derived.by(() => {
		const query = searchQuery.trim().toLowerCase();
		let list = sortedModels;
		if (activeFilter) list = list.filter((m) => m.pipeline_tag === activeFilter);
		if (query) list = list.filter((m) => m.id.toLowerCase().includes(query));
		return list;
	});

	async function performSearch(query: string) {
		const trimmed = query.trim();
		if (!trimmed) {
			return;
		}

		searchLoading = true;
		searchError = null;

		try {
			const results = await HuggingFaceService.searchByQuery(trimmed, { limit: 100 });
			models = results;
			recommendedModels = results.filter(
				(m) => m.id.toLowerCase().includes('gemma-4') || m.id.toLowerCase().includes('qwen3.6')
			);
		} catch (err) {
			searchError = err instanceof Error ? err.message : 'Search failed';
			models = [];
			recommendedModels = [];
		} finally {
			searchLoading = false;
		}
	}

	function handleSearchInput(value: string) {
		searchQuery = value;

		if (searchTimeout) {
			clearTimeout(searchTimeout);
		}

		searchTimeout = setTimeout(() => {
			performSearch(value);
		}, 300);
	}

	function handleClose() {
		if (browser && window.history.length > 1) {
			history.back();
		} else {
			goto(ROUTES.START);
		}
	}

	function formatDownloads(count: number): string {
		return HuggingFaceService.formatDownloads(count);
	}

	function formatLikes(count: number): string {
		return HuggingFaceService.formatLikes(count);
	}

	function formatRelativeTime(dateStr: string): string {
		return HuggingFaceService.formatRelativeTime(dateStr);
	}

	function pipelineTagLabel(tag: string | null): string | null {
		return HuggingFaceService.pipelineTagLabel(tag);
	}

	function pipelineTagIcon(tag: string | null): string | null {
		return HuggingFaceService.pipelineTagIcon(tag);
	}

	function getLibraryLabel(lib: string | null): string | null {
		return lib ? HuggingFaceService.LIBRARIES[lib] || lib : null;
	}

	async function openModelDetails(model: HfModelInfo) {
		goto(RouterService.model(model.id));
	}

	async function loadModels() {
		try {
			const [trending, recommended] = await Promise.all([
				HuggingFaceService.getTrending(100),
				HuggingFaceService.search({ author: 'ggml-org', limit: 1000 })
			]);

			models = trending.filter((m) => m.tags.includes('gguf'));
			recommendedModels = recommended.filter(
				(m) => m.id.toLowerCase().includes('gemma-4') || m.id.toLowerCase().includes('qwen3.6')
			);
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to fetch models';
		} finally {
			loading = false;
		}
	}

	// Run on mount
	$effect(() => {
		loadModels();

		return () => {
			if (searchTimeout) {
				clearTimeout(searchTimeout);
			}
		};
	});
</script>

<div in:fade={{ duration: 150 }} class="flex min-h-[calc(100dvh-4rem)] flex-col {className}">
	<div class="fixed top-4.5 right-4 z-50 md:hidden">
		<ActionIcon icon={X} tooltip="Close" onclick={handleClose} />
	</div>

	<div class="sticky top-0 z-10 mt-4 mb-2 flex items-start gap-4 p-4 md:justify-between md:px-8">
		<div class="flex-1">
			<h1 class="text-lg font-semibold md:text-2xl">Models</h1>
			<p class="text-sm text-muted-foreground">Manage and download your models here</p>
		</div>
		<div class="w-64 shrink-0 md:w-72">
			<SearchInput
				bind:value={searchQuery}
				placeholder="Search models..."
				onInput={handleSearchInput}
			/>
		</div>
	</div>

	<!-- Error -->
	{#if error}
		<div class="rounded-lg border border-destructive/50 bg-destructive/5 p-4 text-center">
			<p class="text-destructive">{error}</p>
		</div>
	{/if}
	{#if searchError}
		<div class="rounded-lg border border-destructive/50 bg-destructive/5 p-4 text-center">
			<p class="text-destructive">{searchError}</p>
		</div>
	{/if}

	<!-- Loading -->
	{#if loading}
		<div class="flex items-center justify-center py-20">
			<p class="text-muted-foreground">Loading models...</p>
		</div>
	{/if}
	{#if searchLoading}
		<div class="flex items-center justify-center py-20">
			<p class="text-muted-foreground">Searching...</p>
		</div>
	{/if}

	<!-- Recommended Section (kept as-is, just shrunk collection) -->
	{#if !loading && !searchLoading && filteredRecommendedModels.length > 0}
		<section class="mb-8 px-4 md:px-8">
			{#if !searchQuery.trim()}
				<header class="mb-4 flex items-center gap-2">
					<Sparkles class="h-4 w-4 text-primary" />
					<h2 class="text-lg font-semibold">Recommended from ggml-org</h2>
				</header>
			{/if}
			<div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
				{#each filteredRecommendedModels as model (model.id)}
					<div
						role="button"
						tabindex="0"
						onclick={() => openModelDetails(model)}
						onkeydown={(e) => {
							if (e.key === 'Enter' || e.key === ' ') {
								e.preventDefault();
								openModelDetails(model);
							}
						}}
						class="flex cursor-pointer flex-col rounded-lg border bg-card p-5 transition-all hover:border-primary/50 hover:shadow-md focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/50"
					>
						<h3 class="mb-2 truncate text-base font-semibold">{model.id}</h3>
						<div class="mb-3 flex flex-wrap gap-1">
							{#if pipelineTagLabel(model.pipeline_tag)}
								<span class="rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
									{pipelineTagLabel(model.pipeline_tag)}
								</span>
							{/if}
							{#if getLibraryLabel(model.library_name)}
								<span
									class="rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground"
								>
									{getLibraryLabel(model.library_name)}
								</span>
							{/if}
							{#if model.tags.includes('gguf')}
								<span
									class="rounded bg-orange-500/10 px-2 py-0.5 text-xs font-medium text-orange-600 dark:text-orange-400"
								>
									GGUF
								</span>
							{/if}
						</div>
						<div class="mt-auto flex items-center gap-4 text-xs text-muted-foreground">
							<span title="Downloads">↓ {formatDownloads(model.downloads)}</span>
							<span title="Likes">♥ {formatLikes(model.likes)}</span>
						</div>
					</div>
				{/each}
			</div>
		</section>
	{/if}

	<!-- Trending GGUF Models - polished, OpenRouter-inspired -->
	{#if !loading && !searchLoading && models.length > 0}
		<section class="px-4 md:px-8">
			{#if !searchQuery.trim()}
				<header class="mb-4 flex items-center justify-between gap-3">
					<h2 class="text-lg font-semibold">Trending GGUF Models</h2>
					<label class="flex items-center gap-2 text-xs text-muted-foreground">
						<span>Sort by</span>
						<select
							bind:value={sortBy}
							class="rounded-md border bg-background px-2 py-1 text-xs font-medium text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/50"
						>
							{#each HuggingFaceService.SORT_OPTIONS as opt (opt)}
								{#if opt !== 'trendingScore'}
									<option value={opt}>{HuggingFaceService.SORT_LABELS[opt]}</option>
								{/if}
							{/each}
						</select>
					</label>
				</header>
			{/if}

			<!-- Pipeline-tag filter chips. Each chip lists the count of models
			     currently loaded under that tag; the "All" chip resets. -->
			{#if !searchQuery.trim() && availableFilters.length > 0}
				<div class="mb-4 flex flex-wrap items-center gap-1.5">
					<button
						type="button"
						onclick={() => (activeFilter = null)}
						class="inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium transition-colors {activeFilter ===
						null
							? 'border-primary bg-primary/10 text-primary'
							: 'border-border bg-background text-muted-foreground hover:border-primary/40 hover:text-foreground'}"
					>
						<SearchIcon class="h-3 w-3" />
						All
						<span class="text-muted-foreground">{models.length}</span>
					</button>
					{#each availableFilters as f (f.tag)}
						{@const isActive = activeFilter === f.tag}
						<button
							type="button"
							onclick={() => (activeFilter = isActive ? null : f.tag)}
							class="inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium transition-colors {isActive
								? 'border-primary bg-primary/10 text-primary'
								: 'border-border bg-background text-muted-foreground hover:border-primary/40 hover:text-foreground'}"
						>
							<IconFromName name={pipelineTagIcon(f.tag)} class="h-3 w-3" />
							{pipelineTagLabel(f.tag)}
							<span class="text-muted-foreground">{f.count}</span>
						</button>
					{/each}
				</div>
			{/if}

			{#if filteredModels.length === 0}
				<div class="rounded-lg border bg-card p-8 text-center">
					<p class="text-sm text-muted-foreground">
						No trending models matching the current filter.
					</p>
					<button
						type="button"
						onclick={() => (activeFilter = null)}
						class="mt-3 inline-flex items-center rounded-md border px-3 py-1 text-xs font-medium hover:border-primary/40"
					>
						Clear filter
					</button>
				</div>
			{:else}
				<div class="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
					{#each filteredModels as model (model.id)}
						{@const tagLabel = pipelineTagLabel(model.pipeline_tag)}
						{@const tagIcon = pipelineTagIcon(model.pipeline_tag)}
						{@const libLabel = getLibraryLabel(model.library_name)}
						<div
							role="button"
							tabindex="0"
							onclick={() => openModelDetails(model)}
							onkeydown={(e) => {
								if (e.key === 'Enter' || e.key === ' ') {
									e.preventDefault();
									openModelDetails(model);
								}
							}}
							class="group flex cursor-pointer flex-col gap-3 rounded-lg border bg-card p-4 transition-all hover:border-primary/50 hover:shadow-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-primary/50"
						>
							<!-- Header: pipeline-tag icon + model id -->
							<header class="flex items-start gap-3">
								<div
									class="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary"
								>
									<IconFromName name={tagIcon} class="h-4 w-4" />
								</div>
								<div class="min-w-0 flex-1">
									<h3 class="truncate text-sm font-semibold leading-tight">{model.id}</h3>
									{#if tagLabel}
										<p class="mt-0.5 truncate text-xs text-muted-foreground">{tagLabel}</p>
									{/if}
								</div>
							</header>

							<!-- Tag badges -->
							<div class="flex flex-wrap gap-1">
								{#if model.tags.includes('gguf')}
									<span
										class="rounded bg-orange-500/10 px-1.5 py-0.5 text-xs font-medium text-orange-600 dark:text-orange-400"
									>
										GGUF
									</span>
								{/if}
								{#if libLabel && libLabel !== 'GGUF'}
									<span
										class="rounded bg-secondary px-1.5 py-0.5 text-xs font-medium text-secondary-foreground"
									>
										{libLabel}
									</span>
								{/if}
								{#if model.tags.includes('safetensors')}
									<span
										class="rounded bg-blue-500/10 px-1.5 py-0.5 text-xs font-medium text-blue-600 dark:text-blue-400"
									>
										Safetensors
									</span>
								{/if}
							</div>

							<!-- Metrics: downloads, likes, created -->
							<div
								class="mt-auto flex items-center justify-between gap-2 border-t pt-3 text-xs text-muted-foreground"
							>
								<div class="flex items-center gap-3">
									<span title="Downloads" class="flex items-center gap-1">
										<Download class="h-3 w-3" />
										{formatDownloads(model.downloads)}
									</span>
									<span title="Likes" class="flex items-center gap-1">
										<Heart class="h-3 w-3" />
										{formatLikes(model.likes)}
									</span>
								</div>
								<span title="Created">{formatRelativeTime(model.createdAt)}</span>
							</div>
						</div>
					{/each}
				</div>
			{/if}
		</section>
	{/if}

	<!-- No Results -->
	{#if !loading && !searchLoading && !error && !searchError}
		{#if searchQuery.trim() && filteredModels.length === 0 && filteredRecommendedModels.length === 0}
			<div class="py-20 text-center">
				<p class="text-muted-foreground">No models found matching "{searchQuery}".</p>
			</div>
		{:else if !searchQuery.trim() && !loading && models.length === 0}
			<div class="py-20 text-center">
				<p class="text-muted-foreground">No trending models found.</p>
			</div>
		{/if}
	{/if}
</div>

<style>
</style>
