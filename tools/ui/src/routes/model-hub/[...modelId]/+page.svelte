<script lang="ts">
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { browser } from '$app/environment';
	import { Cpu, Download, ExternalLink, GitBranch, Heart, Package, X, Check } from '@lucide/svelte';
	import {
		ActionIcon,
		Breadcrumb,
		DialogModelDownload,
		DownloadProgressBar,
		MarkdownContent
	} from '$lib/components/app';
	import { onMount } from 'svelte';
	import { ROUTES } from '$lib/constants';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore } from '$lib/stores/models.svelte';
	import type { HfModelDetailInfo, HfModelSibling } from '$lib/types/huggingface';
	import type { DraftVariant } from '$lib/constants/model-id';
	import { fade } from 'svelte/transition';
	import { SvelteMap } from 'svelte/reactivity';

	interface HfModelGgufMeta {
		architecture?: string;
		total?: number;
		context_length?: number;
		chat_template?: string;
	}

	interface PendingDownload {
		filePath: string;
		sizeBytes: number | null;
		quant: string | null;
		variant: DraftVariant | null;
	}

	let modelId = $derived(page.params.modelId ?? '');

	// Warm the router-models cache so already-downloaded quants show their
	// checkmark on hard refresh (otherwise `/v1/models` is never queried and
	// `isModelDownloaded(...)` returns false for every chip). The dropdown
	// already does this lazily, but landing on this page directly should
	// behave the same.
	onMount(() => {
		void modelsStore.fetchRouterModels();
	});
	let modelInfo: HfModelDetailInfo | null = $state(null);
	let siblings: HfModelSibling[] = $state([]);
	let readme: string | null = $state(null);
	let loading = $state(true);
	let error: string | null = $state(null);

	let pendingDownload = $state<PendingDownload | null>(null);

	let details = $derived.by(() => modelInfo?.details);
	let gguf = $derived.by(() => {
		const d = details ?? undefined;
		return d?.gguf as HfModelGgufMeta | undefined;
	});
	let ggufFiles = $derived(HuggingFaceService.filterByExtension(siblings, '.gguf'));
	let baseModels = $derived(HuggingFaceService.getBaseModels(modelInfo));
	let licenseTag = $derived.by(() => {
		const tags = modelInfo?.tags ?? [];
		return tags.find((t) => t.startsWith('license:'))?.replace('license:', '') ?? null;
	});
	let author = $derived.by(() => modelInfo?.author ?? modelId.split('/')[0] ?? '');

	type BitDepthRow = { bitDepth: number; files: HfModelSibling[] };
	let bitDepthRows = $derived.by<BitDepthRow[]>(() => {
		const rows = new SvelteMap<number, HfModelSibling[]>();
		for (const file of ggufFiles) {
			const meta = HuggingFaceService.extractQuantMeta(file.path);
			// mmproj-* sidecars carry the vision/audio projector for multimodal
			// models. Their existence is already conveyed by `pipeline_tag`
			// (e.g. "image-text-to-test") in the metadata chip row, so we hide
			// them here and only show real weight quantizations + mtp/dflash drafts.
			if (meta?.variant === 'mmproj') continue;
			const depth = meta?.quant ? HuggingFaceService.getBitDepth(meta.quant) : null;
			// unknown quants get bucket 99 so they surface as "Other" at the end
			const bucket = depth ?? 99;
			const list = rows.get(bucket) ?? [];
			list.push(file);
			rows.set(bucket, list);
		}
		return Array.from(rows.entries())
			.map(([bitDepth, files]) => ({ bitDepth, files }))
			.sort((a, b) => a.bitDepth - b.bitDepth);
	});

	// Best-effort label for the user machine, falls back to a generic placeholder.
	// navigator is SSR-safe (gated by typeof) so this never throws during build.
	let machineLabel = $derived.by(() => {
		if (typeof navigator === 'undefined') return 'This machine';
		const cores = navigator.hardwareConcurrency;
		const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
		if (!cores && !mem) return 'This machine';
		const cpu = cores ? `${cores}-core` : 'unknown CPU';
		const memory = mem ? ` · ${mem} GB RAM` : '';
		return `${cpu}${memory}`;
	});

	async function loadModel() {
		loading = true;
		error = null;
		try {
			const [info, tree, readmeText] = await Promise.all([
				HuggingFaceService.getDetails(modelId),
				HuggingFaceService.getTree(modelId),
				HuggingFaceService.getReadme(modelId)
			]);
			modelInfo = info;
			siblings = tree;
			readme = readmeText;
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

	$effect(() => {
		loadModel();
	});
</script>

<svelte:head>
	<title>{modelId} · llama.cpp</title>
</svelte:head>

<div in:fade={{ duration: 150 }} class="flex min-h-[calc(100dvh-4rem)] flex-col">
	<!-- Floating close on mobile -->
	<div class="fixed top-4.5 right-4 z-50 md:hidden">
		<ActionIcon icon={X} tooltip="Close" onclick={handleClose} />
	</div>

	<div
		class="flex items-start gap-4 border-b border-border/40 bg-background/70 backdrop-blur md:justify-between"
	>
		<div class="min-w-0 flex-1">
			<Breadcrumb items={[{ label: 'Models', href: ROUTES.MANAGE_MODELS }, { label: modelId }]} />
			<h1 class="mt-1 truncate text-lg font-semibold md:text-2xl">{modelId}</h1>
		</div>
		<div class="hidden items-start gap-2 md:flex">
			<ActionIcon icon={GitBranch} tooltip="Use this model" onclick={() => goto(ROUTES.NEW_CHAT)} />
			<a
				href={`https://huggingface.co/${modelId}`}
				target="_blank"
				rel="noopener noreferrer"
				class="inline-flex items-center gap-1.5 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
			>
				<ExternalLink size={14} />
				View on Hugging Face
			</a>
		</div>
	</div>

	{#if error}
		<div
			class="mx-4 mt-4 rounded-lg border border-destructive/50 bg-destructive/5 p-4 text-center md:mx-8"
		>
			<p class="text-destructive">{error}</p>
		</div>
	{/if}

	{#if loading}
		<div class="flex items-center justify-center py-20">
			<p class="text-muted-foreground">Loading model details...</p>
		</div>
	{/if}

	{#if !loading && modelInfo}
		<!-- Metadata chip row -->
		<div class="flex flex-wrap items-center gap-1.5 px-4 pt-4 md:px-8">
			{#if author}
				<span
					class="rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground"
				>
					{author}
				</span>
			{/if}
			{#if modelInfo.pipeline_tag}
				<span class="rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
					{modelInfo.pipeline_tag}
				</span>
			{/if}
			{#if modelInfo.library_name}
				<span
					class="rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground"
				>
					{modelInfo.library_name}
				</span>
			{/if}
			{#if modelInfo.gated === true}
				<span
					class="rounded bg-yellow-500/10 px-2 py-0.5 text-xs font-medium text-yellow-600 dark:text-yellow-400"
				>
					gated
				</span>
			{/if}
			{#if licenseTag}
				<span class="rounded bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
					{licenseTag}
				</span>
			{/if}
		</div>

		<!-- Two-column HF layout: main + sticky aside -->
		<div class="grid gap-6 px-4 py-6 md:px-8 lg:grid-cols-3">
			<main class="min-w-0 space-y-6 lg:col-span-2">
				<!-- README -->
				{#if readme}
					<article class="rounded-lg border bg-card p-5">
						<MarkdownContent content={readme} allowRawHtml class="prose-sm max-w-none" />
					</article>
				{:else if details?.cardData?.description}
					<article class="rounded-lg border bg-card p-5">
						<!-- eslint-disable-next-line no-at-html-tags -->
						{@html details.cardData.description}
					</article>
				{/if}

				<!-- GGUF Specs -->
				{#if gguf}
					<section class="rounded-lg border bg-card p-5">
						<h2 class="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
							GGUF Specs
						</h2>
						<dl class="grid grid-cols-2 gap-x-4 gap-y-3 md:grid-cols-3">
							{#if gguf.architecture}
								<div>
									<dt class="text-xs text-muted-foreground">Architecture</dt>
									<dd class="text-sm font-medium capitalize">
										{gguf.architecture.replace(/_/g, ' ')}
									</dd>
								</div>
							{/if}
							{#if gguf.total}
								<div>
									<dt class="text-xs text-muted-foreground">Total params</dt>
									<dd class="text-sm font-medium tabular-nums">
										{HuggingFaceService.formatFileSize(gguf.total).replace(' B', '')}B
									</dd>
								</div>
							{/if}
							{#if gguf.context_length}
								<div>
									<dt class="text-xs text-muted-foreground">Context length</dt>
									<dd class="text-sm font-medium tabular-nums">
										{gguf.context_length.toLocaleString()}
									</dd>
								</div>
							{/if}
						</dl>
					</section>
				{/if}
			</main>

			<aside class="space-y-4 lg:sticky lg:top-20 lg:self-start">
				<!-- Stats card -->
				<section class="rounded-lg border bg-card p-4">
					<h2 class="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
						Stats
					</h2>
					<dl class="space-y-2.5 text-sm">
						{#if typeof modelInfo.downloads === 'number'}
							<div class="flex items-center justify-between">
								<dt class="flex items-center gap-1.5 text-muted-foreground">
									<Download size={13} />
									Downloads
								</dt>
								<dd class="font-medium tabular-nums">{formatDownloads(modelInfo.downloads)}</dd>
							</div>
						{/if}
						{#if typeof modelInfo.likes === 'number'}
							<div class="flex items-center justify-between">
								<dt class="flex items-center gap-1.5 text-muted-foreground">
									<Heart size={13} />
									Likes
								</dt>
								<dd class="font-medium tabular-nums">{formatLikes(modelInfo.likes)}</dd>
							</div>
						{/if}
						{#if details?.lastModified}
							<div class="flex items-center justify-between">
								<dt class="text-muted-foreground">Last modified</dt>
								<dd class="font-medium">{formatRelativeTime(details.lastModified)}</dd>
							</div>
						{/if}
						{#if details?.size}
							<div class="flex items-center justify-between">
								<dt class="text-muted-foreground">Total size</dt>
								<dd class="font-medium tabular-nums">{formatFileSize(details.size)}</dd>
							</div>
						{/if}
					</dl>
				</section>

				<!-- Base model + Library combo -->
				{#if baseModels.length || ggufFiles.length}
					<section class="rounded-lg border bg-card p-4">
						{#if baseModels.length}
							<h2 class="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
								Base model
							</h2>
							<ul class="mb-4 space-y-1.5 text-sm">
								{#each baseModels as base (base)}
									<li>
										<a
											href={`https://huggingface.co/${base}`}
											target="_blank"
											rel="noopener noreferrer"
											class="flex items-center justify-between rounded-md px-2 py-1 transition-colors hover:bg-muted"
										>
											<span class="truncate font-mono text-xs">{base}</span>
											<ExternalLink size={11} class="shrink-0 opacity-60" />
										</a>
									</li>
								{/each}
							</ul>
						{/if}

						{#if modelInfo.library_name}
							<h2 class="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
								Library
							</h2>
							<p class="text-sm font-medium">{modelInfo.library_name}</p>
						{/if}
					</section>
				{/if}

				<!-- GGUF Quantizations / Hardware compatibility -->
				{#if ggufFiles.length}
					<section class="rounded-lg border bg-card p-4">
						<header class="mb-3 flex items-center justify-between gap-2">
							<h2
								class="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground"
							>
								<Cpu size={13} />
								Hardware compatibility
							</h2>
							<span
								class="rounded bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground"
								title={machineLabel}
							>
								×1
							</span>
						</header>

						<div class="space-y-2">
							{#each bitDepthRows as row (row.bitDepth)}
								<div class="grid grid-cols-[6rem_1fr] items-center gap-3">
									<div class="text-xs font-semibold tabular-nums text-muted-foreground">
										{#if row.bitDepth === 99}
											Other
										{:else}
											{row.bitDepth}-bit
										{/if}
									</div>
									<div class="flex flex-wrap gap-1.5">
										{#each row.files as file (file.path)}
											{@const meta = HuggingFaceService.extractQuantMeta(file.path)}
											{@const basename = file.path.split('/').pop() ?? file.path}
											{@const fallbackLabel = basename
												.replace(/\.gguf$/i, '')
												.replace(/^(?:mtp|dflash|mmproj)-/i, '')
												.replace(/-mtp$/i, '')}
											{@const label = meta?.quant ?? fallbackLabel}
											{@const downloadTagInput = meta?.quant
												? { quant: meta.quant, variant: meta.variant ?? null }
												: null}
											{@const hfRepoWithTag = ModelsService.buildDownloadTag(
												modelId,
												downloadTagInput
											)}
											{@const downloadProgress = modelsStore.getDownloadProgress(hfRepoWithTag)}
											{@const isDownloading = modelsStore.isDownloadInProgress(hfRepoWithTag)}
											{@const isFullyDownloaded = modelsStore.isModelDownloaded(hfRepoWithTag)}
											{@const isFailed = modelsStore.hasFailedDownload(hfRepoWithTag)}
											{@const chipState = isDownloading
												? 'downloading'
												: isFullyDownloaded
													? 'downloaded'
													: isFailed
														? 'failed'
														: 'idle'}
											<button
												type="button"
												onclick={() =>
													(pendingDownload = {
														filePath: file.path,
														sizeBytes: file.size ?? null,
														quant: meta?.quant ?? null,
														variant: meta?.variant ?? null
													})}
												class="relative inline-flex cursor-pointer items-center gap-1 overflow-hidden rounded-md border bg-background px-2 py-1 text-left font-mono text-xs transition-colors hover:border-primary/60 hover:bg-primary/5"
												class:border-foreground={isFullyDownloaded && !isDownloading && !isFailed}
												class:bg-muted={isFullyDownloaded && !isDownloading && !isFailed}
												class:border-destructive={isFailed && !isDownloading}
												title={chipState === 'downloading'
													? `In progress: ${file.path}. Click to view cancel options.`
													: chipState === 'downloaded'
														? `Already downloaded: ${file.path}`
														: chipState === 'failed'
															? `Last attempt failed: ${file.path}. Click to delete partial files and retry.`
															: `Download ${file.path}`}
											>
												{#if isFullyDownloaded && !isDownloading}
													<Check class="h-3 w-3 text-foreground/70" />
												{/if}
												{#if isFailed && !isDownloading && !isFullyDownloaded}
													<span
														class="rounded bg-destructive px-1 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-destructive-foreground"
													>
														Failed
													</span>
												{/if}
												{#if meta?.variant && meta.variantForm === 'prefix'}
													<span
														class="rounded bg-primary px-1 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary-foreground"
													>
														{meta.variant}
													</span>
												{/if}
												<span class="font-medium">{label}</span>
												{#if meta?.variant && meta.variantForm === 'suffix'}
													<span
														class="rounded bg-primary px-1 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary-foreground"
													>
														{meta.variant}
													</span>
												{/if}
												<span class="text-muted-foreground">
													{#if isDownloading && downloadProgress && downloadProgress.totalBytes > 0}
														{Math.round(
															(downloadProgress.downloadedBytes / downloadProgress.totalBytes) * 100
														)}%
													{:else}
														{formatFileSize(file.size ?? 0)}
													{/if}
												</span>
												{#if isDownloading && downloadProgress}
													<DownloadProgressBar
														overlay
														downloadedBytes={downloadProgress.downloadedBytes}
														totalBytes={downloadProgress.totalBytes}
													/>
												{/if}
											</button>
										{/each}
									</div>
								</div>
							{/each}
						</div>
					</section>
				{/if}

				<!-- License link -->
				{#if details?.cardData?.license}
					<section class="rounded-lg border bg-card p-4">
						<h2 class="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
							License
						</h2>
						{#if details.cardData.license_link}
							<a
								href={String(details.cardData.license_link)}
								target="_blank"
								rel="noopener noreferrer"
								class="inline-flex items-center gap-1 text-sm font-medium text-primary underline-offset-4 hover:underline"
							>
								{details.cardData.license}
								<ExternalLink size={11} />
							</a>
						{:else}
							<span class="text-sm font-medium">{details.cardData.license}</span>
						{/if}
					</section>
				{/if}

				<!-- Mobile View on HF link -->
				<a
					href={`https://huggingface.co/${modelId}`}
					target="_blank"
					rel="noopener noreferrer"
					class="flex items-center justify-center gap-1.5 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 md:hidden"
				>
					<Package size={14} />
					View on Hugging Face
				</a>
			</aside>
		</div>
	{/if}
</div>

{#if pendingDownload}
	<DialogModelDownload
		bind:open={
			() => pendingDownload !== null,
			(v) => {
				if (!v) pendingDownload = null;
			}
		}
		repoId={modelId}
		filePath={pendingDownload.filePath}
		quant={pendingDownload.quant}
		variant={pendingDownload.variant}
		formattedSize={pendingDownload.sizeBytes !== null
			? formatFileSize(pendingDownload.sizeBytes)
			: ''}
		onConfirm={() => (pendingDownload = null)}
		onCancel={() => (pendingDownload = null)}
	/>
{/if}

<style>
</style>
