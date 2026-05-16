<script lang="ts">
	import { HuggingFaceService } from '$lib/services';
	import type { HfModelInfo, HfModelDetailInfo } from '$lib/types/huggingface';
	import { onMount } from 'svelte';

	// Dialog state
	let selectedModel: HfModelInfo | null = $state(null);
	let modelDetails: HfModelDetailInfo | null = $state(null);
	let modelFiles: { path: string; size: number }[] = $state([]);
	let detailsLoading = $state(false);
	let showDetails = $state(false);

	let models: HfModelInfo[] = $state([]);
	let recommendedModels: HfModelInfo[] = $state([]);
	let loading = $state(true);
	let error: string | null = $state(null);

	function formatDownloads(count: number): string {
		return HuggingFaceService.formatDownloads(count);
	}

	function formatLikes(count: number): string {
		return HuggingFaceService.formatLikes(count);
	}

	function formatRelativeTime(dateStr: string): string {
		return HuggingFaceService.formatRelativeTime(dateStr);
	}

	function formatFileSize(bytes: number): string {
		return HuggingFaceService.formatFileSize(bytes);
	}

	function formatParams(params: number): string {
		if (params >= 1_000_000_000_000) {
			return `${(params / 1_000_000_000_000).toFixed(2)}T`;
		}
		if (params >= 1_000_000_000) {
			return `${(params / 1_000_000_000).toFixed(2)}B`;
		}
		if (params >= 1_000_000) {
			return `${(params / 1_000_000).toFixed(2)}M`;
		}
		return params.toString();
	}

	function formatContextLength(tokens: number): string {
		if (tokens >= 1_000_000) {
			return `${(tokens / 1_000_000).toFixed(1)}M`;
		}
		if (tokens >= 1_000) {
			return `${(tokens / 1_000).toFixed(1)}K`;
		}
		return tokens.toString();
	}

	function getTaskLabel(task: string | null): string | null {
		return task ? HuggingFaceService.TASKS[task] || task : null;
	}

	function getLibraryLabel(lib: string | null): string | null {
		return lib ? HuggingFaceService.LIBRARIES[lib] || lib : null;
	}

	async function openModelDetails(model: HfModelInfo) {
		selectedModel = model;
		modelDetails = null;
		modelFiles = [];
		detailsLoading = true;
		showDetails = true;

		try {
			const [details, files] = await Promise.all([
				HuggingFaceService.getDetails(model.id),
				HuggingFaceService.getTree(model.id)
			]);
			modelDetails = details;
			modelFiles = files.sort((a, b) => b.size - a.size);
		} catch (err) {
			console.error('Failed to load model details:', err);
		} finally {
			detailsLoading = false;
		}
	}

	function closeDialog() {
		showDetails = false;
		selectedModel = null;
		modelDetails = null;
		modelFiles = [];
	}

	onMount(async () => {
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
	});
</script>

<!-- <div class="min-h-screen bg-background p-6"> -->
<!-- Header -->
<header class="mb-6">
	<h1 class="text-2xl font-bold">Model Hub</h1>
	<p class="text-sm text-muted-foreground">Trending models from Hugging Face</p>
</header>

<!-- Error -->
{#if error}
	<div class="rounded-lg border border-destructive/50 bg-destructive/5 p-4 text-center">
		<p class="text-destructive">{error}</p>
	</div>
{/if}

<!-- Loading -->
{#if loading}
	<div class="flex items-center justify-center py-20">
		<p class="text-muted-foreground">Loading models...</p>
	</div>
{/if}

<!-- Recommended Section -->
{#if !loading && recommendedModels.length > 0}
	<section class="mb-8">
		<h2 class="mb-4 text-lg font-semibold">Recommended from ggml-org</h2>
		<div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
			{#each recommendedModels as model (model.id)}
				<div
					onclick={() => openModelDetails(model)}
					class="flex cursor-pointer flex-col rounded-lg border bg-card p-5 transition-all hover:border-primary/50 hover:shadow-md"
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
{#if !loading && models.length > 0}
	<h2 class="mb-4 text-lg font-semibold">Trending GGUF Models</h2>
	<div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
		{#each models as model (model.id)}
			<div
				onclick={() => openModelDetails(model)}
				class="flex cursor-pointer flex-col rounded-lg border bg-card p-5 transition-all hover:border-primary/50 hover:shadow-md"
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
{/if}

<!-- No Results -->
{#if !loading && models.length === 0 && !error}
	<div class="py-20 text-center">
		<p class="text-muted-foreground">No trending models found.</p>
	</div>
{/if}

<!-- Model Details Dialog -->
{#if showDetails && selectedModel}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
		onclick={(e) => {
			if (e.target === e.currentTarget) closeDialog();
		}}
	>
		<div class="max-h-[90vh] w-full max-w-2xl overflow-hidden rounded-xl border bg-card shadow-2xl">
			<!-- Dialog Header -->
			<div class="flex items-center justify-between border-b px-6 py-4">
				<h2 class="truncate text-xl font-semibold">{selectedModel.id}</h2>
				<button onclick={closeDialog} class="rounded-md p-1 transition-colors hover:bg-muted">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						width="20"
						height="20"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"
						class="lucide lucide-x"><path d="M18 6 6 18" /><path d="m6 6 12 12" /></svg
					>
				</button>
			</div>

			<!-- Dialog Content -->
			<div class="max-h-[70vh] overflow-y-auto p-6">
				{#if detailsLoading}
					<div class="flex items-center justify-center py-12">
						<p class="text-muted-foreground">Loading details...</p>
					</div>
				{:else}
					<!-- Basic Info Grid -->
					<div class="mb-6 grid grid-cols-2 gap-4">
						{#if modelDetails?.downloads}
							<div>
								<p class="text-xs text-muted-foreground">Downloads</p>
								<p class="text-lg font-semibold">{formatDownloads(modelDetails.downloads)}</p>
							</div>
						{/if}
						{#if modelDetails?.likes}
							<div>
								<p class="text-xs text-muted-foreground">Likes</p>
								<p class="text-lg font-semibold">{formatLikes(modelDetails.likes)}</p>
							</div>
						{/if}
						{#if modelDetails?.lastModified}
							<div>
								<p class="text-xs text-muted-foreground">Last Modified</p>
								<p class="text-lg font-semibold">
									{formatRelativeTime(modelDetails.lastModified)}
								</p>
							</div>
						{/if}
						{#if modelDetails?.size}
							<div>
								<p class="text-xs text-muted-foreground">Model Size</p>
								<p class="text-lg font-semibold">{formatFileSize(modelDetails.size)}</p>
							</div>
						{/if}
						{#if modelDetails?.gated}
							<div>
								<p class="text-xs text-muted-foreground">Status</p>
								<span
									class="inline-flex items-center rounded-md bg-yellow-500/10 px-2 py-0.5 text-xs font-medium text-yellow-600 dark:text-yellow-400"
								>
									Gated
								</span>
							</div>
						{/if}
					</div>

					<!-- Model Specs -->
					{#if modelDetails?.gguf}
						<div class="mb-6 rounded-lg border bg-muted/30 p-4">
							<h3 class="mb-3 text-sm font-semibold text-muted-foreground">Model Specs</h3>
							<div class="grid grid-cols-2 gap-3">
								{#if modelDetails.gguf.architecture}
									<div>
										<p class="text-xs text-muted-foreground">Architecture</p>
										<p class="text-sm font-medium capitalize">
											{modelDetails.gguf.architecture.replace(/_/g, ' ')}
										</p>
									</div>
								{/if}
								{#if modelDetails.gguf.total}
									<div>
										<p class="text-xs text-muted-foreground">Parameters</p>
										<p class="text-sm font-medium">{formatParams(modelDetails.gguf.total)}</p>
									</div>
								{/if}
								{#if modelDetails.gguf.context_length}
									<div>
										<p class="text-xs text-muted-foreground">Context Length</p>
										<p class="text-sm font-medium">
											{formatContextLength(modelDetails.gguf.context_length)} tokens
										</p>
									</div>
								{/if}
								{#if modelDetails.cardData?.license}
									<div>
										<p class="text-xs text-muted-foreground">License</p>
										{#if modelDetails.cardData.license_link}
											<a
												href={modelDetails.cardData.license_link}
												target="_blank"
												class="inline-flex items-center text-sm font-medium text-primary underline-offset-4 hover:underline"
											>
												{modelDetails.cardData.license}
												<span class="ml-1">
													<svg
														xmlns="http://www.w3.org/2000/svg"
														width="12"
														height="12"
														viewBox="0 0 24 24"
														fill="none"
														stroke="currentColor"
														stroke-width="2"
														stroke-linecap="round"
														stroke-linejoin="round"
														><path
															d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"
														/><polyline points="15 3 21 3 21 9" /><line
															x1="10"
															y1="14"
															x2="21"
															y2="3"
														/></svg
													>
												</span>
											</a>
										{:else}
											<span class="text-sm font-medium">{modelDetails.cardData.license}</span>
										{/if}
									</div>
								{/if}
							</div>
						</div>
					{/if}

					<!-- Available GGUF Quantizations -->
					{#if modelFiles.length > 0}
						<div>
							<h3 class="mb-3 text-sm font-semibold text-muted-foreground">
								Available GGUF Quantizations ({modelFiles.length})
							</h3>
							<div class="space-y-2">
								{#each modelFiles as file (file.path)}
									<a
										href={`https://huggingface.co/${selectedModel.id}/resolve/main/${file.path}`}
										target="_blank"
										class="flex items-center justify-between rounded-lg border bg-muted/50 p-3 transition-colors hover:bg-muted/80"
									>
										<span class="truncate font-mono text-sm">{file.path}</span>
										<span class="ml-4 text-xs text-muted-foreground"
											>{formatFileSize(file.size)}</span
										>
									</a>
								{/each}
							</div>
						</div>
					{/if}
				{/if}
			</div>

			<!-- Dialog Footer -->
			<div class="flex items-center justify-end gap-3 border-t px-6 py-4">
				<a
					href={`https://huggingface.co/${selectedModel.id}`}
					target="_blank"
					class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
				>
					View on Hugging Face
				</a>
			</div>
		</div>
	</div>
{/if}

<!-- </div> -->

<style>
	.prose :global(h1) {
		font-size: 1.2em;
		margin-bottom: 0.5em;
		font-weight: 600;
	}
	.prose :global(h2) {
		font-size: 1.1em;
		margin-bottom: 0.5em;
		margin-top: 1em;
		font-weight: 600;
	}
	.prose :global(p) {
		margin-bottom: 0.75em;
	}
	.prose :global(ul) {
		margin-bottom: 0.75em;
		padding-left: 1.5em;
		list-style-type: disc;
	}
	.prose :global(li) {
		margin-bottom: 0.25em;
	}
	.prose :global(a) {
		color: hsl(var(--primary));
		text-decoration: underline;
	}
</style>
