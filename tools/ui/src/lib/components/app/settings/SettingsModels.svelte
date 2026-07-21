<script lang="ts">
	import { X } from '@lucide/svelte';
	import { HuggingFaceService } from '$lib/services';
	import { RouterService } from '$lib/services/router.service';
	import type { HfModelInfo } from '$lib/types/huggingface';
	import { goto } from '$app/navigation';
	import { browser } from '$app/environment';
	import { ActionIcon, SearchInput } from '$lib/components/app';
	import { ROUTES } from '$lib/constants';
	import { fade } from 'svelte/transition';

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

	let filteredRecommendedModels = $derived.by(() => {
		const query = searchQuery.trim().toLowerCase();
		if (!query) return recommendedModels;
		return recommendedModels.filter((m) => m.id.toLowerCase().includes(query));
	});

	let filteredModels = $derived.by(() => {
		const query = searchQuery.trim().toLowerCase();
		if (!query) return models;
		return models.filter((m) => m.id.toLowerCase().includes(query));
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

	function getTaskLabel(task: string | null): string | null {
		return task ? HuggingFaceService.TASKS[task] || task : null;
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
			<h1 class="text-lg font-semibold md:text-2xl">Model Hub</h1>
			<p class="text-sm text-muted-foreground">Trending models from Hugging Face</p>
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

	<!-- Recommended Section -->
	{#if !loading && !searchLoading && filteredRecommendedModels.length > 0}
		<section class="mb-8 px-4 md:px-8">
			{#if !searchQuery.trim()}
				<h2 class="mb-4 text-lg font-semibold">Recommended from ggml-org</h2>
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
							{#if getTaskLabel(model.pipeline_tag)}
								<span class="rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
									{getTaskLabel(model.pipeline_tag)}
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

	<!-- Model List -->
	{#if !loading && !searchLoading && filteredModels.length > 0}
		<section class="px-4 md:px-8">
			{#if !searchQuery.trim()}
				<h2 class="mb-4 text-lg font-semibold">Trending GGUF Models</h2>
			{/if}
			<div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
				{#each filteredModels as model (model.id)}
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
							{#if getTaskLabel(model.pipeline_tag)}
								<span class="rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
									{getTaskLabel(model.pipeline_tag)}
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
							{#if model.tags.includes('safetensors')}
								<span
									class="rounded bg-blue-500/10 px-2 py-0.5 text-xs font-medium text-blue-600 dark:text-blue-400"
								>
									SafeTensors
								</span>
							{/if}
						</div>
						<div class="mt-auto flex items-center gap-4 text-xs text-muted-foreground">
							<span title="Downloads">↓ {formatDownloads(model.downloads)}</span>
							<span title="Likes">♥ {formatLikes(model.likes)}</span>
							<span title="Created">{formatRelativeTime(model.createdAt)}</span>
						</div>
					</div>
				{/each}
			</div>
		</section>
	{/if}

	<!-- No Results -->
	{#if !loading && !searchLoading && !error && !searchError}
		{#if searchQuery.trim() && filteredModels.length === 0 && filteredRecommendedModels.length === 0}
			<div class="py-20 text-center">
				<p class="text-muted-foreground">No models found matching "{searchQuery}".</p>
			</div>
		{:else}
			<div class="py-20 text-center">
				<p class="text-muted-foreground">No trending models found.</p>
			</div>
		{/if}
	{/if}
</div>

<style>
</style>
