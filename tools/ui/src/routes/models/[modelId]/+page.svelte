<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { browser } from '$app/environment';
	import { X } from '@lucide/svelte';
	import { ActionIcon, Breadcrumb } from '$lib/components/app';
	import { ROUTES } from '$lib/constants';
	import { HuggingFaceService } from '$lib/services';
	import type { HfModelDetailInfo } from '$lib/types/huggingface';
	import { fade } from 'svelte/transition';

	interface HfModelGgufMeta {
		architecture?: string;
		total?: number;
		context_length?: number;
	}

	let modelId = $derived(decodeURIComponent(page.params.modelId ?? ''));
	let modelInfo: HfModelDetailInfo | null = $state(null);
	let modelFiles: { path: string; size: number }[] = $state([]);
	let loading = $state(true);
	let error: string | null = $state(null);

	let details = $derived.by(() => modelInfo?.details);
	let gguf = $derived.by(() => {
		const d = details ?? undefined;
		return d?.gguf as HfModelGgufMeta | undefined;
	});

	async function loadModel() {
		loading = true;
		error = null;
		try {
			const [info, files] = await Promise.all([
				HuggingFaceService.getDetails(modelId),
				HuggingFaceService.getTree(modelId)
			]);
			modelInfo = info;
			modelFiles = files.sort((a, b) => b.size - a.size);
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load model';
		} finally {
			loading = false;
		}
	}

	function handleClose() {
		if (browser && window.history.length > 1) {
			history.back();
		} else {
			goto(ROUTES.NEW_CHAT);
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

	$effect(() => {
		loadModel();
	});
</script>

<svelte:head>
	<title>{modelId} · llama.cpp</title>
</svelte:head>

<div in:fade={{ duration: 150 }} class="flex min-h-[calc(100dvh-4rem)] flex-col">
	<div class="fixed top-4.5 right-4 z-50 md:hidden">
		<ActionIcon icon={X} tooltip="Close" onclick={handleClose} />
	</div>

	<!-- Header -->
	<div class="sticky top-0 z-10 mt-4 mb-2 flex items-start gap-4 p-4 md:justify-between md:px-8">
		<div class="min-w-0 flex-1">
			<Breadcrumb items={[{ label: 'Models', href: ROUTES.MANAGE_MODELS }, { label: modelId }]} />
			<h1 class="mt-1 truncate text-lg font-semibold md:text-2xl">{modelId}</h1>
		</div>
		<a
			href={`https://huggingface.co/${modelId}`}
			target="_blank"
			class="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
		>
			View on Hugging Face
		</a>
	</div>

	<!-- Error -->
	{#if error}
		<div class="rounded-lg border border-destructive/50 bg-destructive/5 p-4 text-center">
			<p class="text-destructive">{error}</p>
		</div>
	{/if}

	<!-- Loading -->
	{#if loading}
		<div class="flex items-center justify-center py-20">
			<p class="text-muted-foreground">Loading model details...</p>
		</div>
	{/if}

	<!-- Model Details -->
	{#if !loading && modelInfo}
		<div class="px-4 md:px-8">
			<!-- Basic Info Grid -->
			<div class="mb-6 grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4">
				{#if modelInfo.downloads}
					<div>
						<p class="text-xs text-muted-foreground">Downloads</p>
						<p class="text-lg font-semibold">{formatDownloads(modelInfo.downloads)}</p>
					</div>
				{/if}
				{#if modelInfo.likes}
					<div>
						<p class="text-xs text-muted-foreground">Likes</p>
						<p class="text-lg font-semibold">{formatLikes(modelInfo.likes)}</p>
					</div>
				{/if}
				{#if details?.lastModified}
					<div>
						<p class="text-xs text-muted-foreground">Last Modified</p>
						<p class="text-lg font-semibold">{formatRelativeTime(details.lastModified)}</p>
					</div>
				{/if}
				{#if details?.size}
					<div>
						<p class="text-xs text-muted-foreground">Model Size</p>
						<p class="text-lg font-semibold">{formatFileSize(details.size)}</p>
					</div>
				{/if}
				{#if details?.gated}
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
			{#if gguf}
				<div class="mb-6 rounded-lg border bg-muted/30 p-4">
					<h2 class="mb-3 text-sm font-semibold text-muted-foreground">Model Specs</h2>
					<div class="grid grid-cols-2 gap-3 md:grid-cols-4">
						{#if gguf.architecture}
							<div>
								<p class="text-xs text-muted-foreground">Architecture</p>
								<p class="text-sm font-medium capitalize">
									{gguf.architecture.replace(/_/g, ' ')}
								</p>
							</div>
						{/if}
						{#if gguf.total}
							<div>
								<p class="text-xs text-muted-foreground">Parameters</p>
								<p class="text-sm font-medium">{formatParams(gguf.total)}</p>
							</div>
						{/if}
						{#if gguf.context_length}
							<div>
								<p class="text-xs text-muted-foreground">Context Length</p>
								<p class="text-sm font-medium">
									{formatContextLength(gguf.context_length)} tokens
								</p>
							</div>
						{/if}
						{#if details?.cardData?.license}
							<div>
								<p class="text-xs text-muted-foreground">License</p>
								{#if details.cardData?.license_link}
									<a
										href={String(details.cardData.license_link)}
										target="_blank"
										class="inline-flex items-center text-sm font-medium text-primary underline-offset-4 hover:underline"
									>
										{details.cardData.license}
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
											>
												<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
												<polyline points="15 3 21 3 21 9" />
												<line x1="10" y1="14" x2="21" y2="3" />
											</svg>
										</span>
									</a>
								{:else}
									<span class="text-sm font-medium">{details.cardData.license}</span>
								{/if}
							</div>
						{/if}
					</div>
				</div>
			{/if}

			<!-- Tags -->
			{#if details?.cardData?.tags?.length}
				<div class="mb-6 flex flex-wrap gap-1.5">
					{#each details.cardData.tags.slice(0, 20) as tag (tag)}
						<span class="rounded bg-secondary px-2 py-0.5 text-xs text-secondary-foreground">
							{tag}
						</span>
					{/each}
				</div>
			{/if}

			<!-- Tags from model info -->
			<div class="mb-6 flex flex-wrap gap-1">
				{#if getTaskLabel(modelInfo.pipeline_tag)}
					<span class="rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
						{getTaskLabel(modelInfo.pipeline_tag)}
					</span>
				{/if}
				{#if getLibraryLabel(modelInfo.library_name)}
					<span
						class="rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground"
					>
						{getLibraryLabel(modelInfo.library_name)}
					</span>
				{/if}
			</div>

			<!-- Description -->
			{#if details?.cardData?.description}
				<div class="mb-8 rounded-lg border bg-muted/30 p-4">
					<h2 class="mb-3 text-sm font-semibold text-muted-foreground">Description</h2>
					<div class="prose-sm max-w-none text-sm text-muted-foreground">
						{@html details.cardData.description}
					</div>
				</div>
			{/if}

			<!-- Available GGUF Quantizations -->
			{#if modelFiles.length > 0}
				<div>
					<h2 class="mb-4 text-lg font-semibold">
						Available GGUF Quantizations ({modelFiles.length})
					</h2>
					<div class="space-y-2">
						{#each modelFiles as file (file.path)}
							<a
								href={`https://huggingface.co/${modelId}/resolve/main/${file.path}`}
								target="_blank"
								class="flex items-center justify-between rounded-lg border bg-muted/50 p-3 transition-colors hover:bg-muted/80"
							>
								<span class="truncate font-mono text-sm">{file.path}</span>
								<span class="ml-4 shrink-0 text-xs text-muted-foreground"
									>{formatFileSize(file.size)}</span
								>
							</a>
						{/each}
					</div>
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.prose-sm :global(p) {
		margin-bottom: 0.5em;
	}
	.prose-sm :global(h1) {
		font-size: 1.2em;
		margin-bottom: 0.5em;
	}
	.prose-sm :global(h2) {
		font-size: 1.1em;
		margin-top: 1em;
		margin-bottom: 0.5em;
	}
	.prose-sm :global(ul) {
		margin-bottom: 0.5em;
		padding-left: 1.5em;
		list-style-type: disc;
	}
	.prose-sm :global(li) {
		margin-bottom: 0.25em;
	}
	.prose-sm :global(a) {
		color: hsl(var(--primary));
		text-decoration: underline;
	}
</style>
