<script lang="ts">
	import Breadcrumb from './Breadcrumb.svelte';
	import DialogModelDownload from './DialogModelDownload.svelte';
	import DownloadProgressBar from './DownloadProgressBar.svelte';
	import { ArrowLeft, Check, Copy, Cpu, Download, ExternalLink, Heart } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { ActionIcon } from '$lib/components/app';
	import type { DraftVariant } from '$lib/constants';
	import { ROUTES } from '$lib/constants';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type { HfModelDetailInfo, HfModelSibling } from '$lib/types/huggingface';
	import { copyToClipboard } from '$lib/utils';
	import { onMount } from 'svelte';
	import { SvelteMap } from 'svelte/reactivity';

	interface Props {
		modelId: string;
		class?: string;
	}

	let { class: className, modelId }: Props = $props();

	let modelInfo: HfModelDetailInfo | null = $state(null);
	let siblings: HfModelSibling[] = $state([]);
	let loading = $state(true);
	let error: string | null = $state(null);
	let copied = $state(false);

	let pendingDownload = $state<{
		filePath: string;
		sizeBytes: number | null;
		quant: string | null;
		variant: DraftVariant | null;
	} | null>(null);

	let details = $derived.by(() => modelInfo);
	let gguf = $derived.by(() => modelInfo?.gguf);
	let ggufFiles = $derived(HuggingFaceService.filterByExtension(siblings, '.gguf'));
	let description = $derived.by(() => modelInfo?.cardData?.description ?? null);
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

			if (meta?.variant === 'mmproj') continue;

			const depth = meta?.quant ? HuggingFaceService.getBitDepth(meta.quant) : null;
			const bucket = depth ?? 99;
			const list = rows.get(bucket) ?? [];

			list.push(file);
			rows.set(bucket, list);
		}

		return Array.from(rows.entries())
			.map(([bitDepth, files]) => ({ bitDepth, files }))
			.sort((a, b) => a.bitDepth - b.bitDepth);
	});

	function handleBack() {
		goto(ROUTES.MANAGE_MODELS);
	}

	async function handleCopy() {
		await copyToClipboard(modelId);
		copied = true;
		setTimeout(() => (copied = false), 1500);
	}

	async function loadModel() {
		loading = true;
		error = null;

		try {
			const [info, tree] = await Promise.all([
				HuggingFaceService.getDetails(modelId),
				HuggingFaceService.getTree(modelId)
			]);

			if (!info) {
				error = 'Model not found on Hugging Face.';

				return;
			}

			modelInfo = info;
			siblings = tree;
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load model';
		} finally {
			loading = false;
		}
	}

	// Warm the router-models cache so already-downloaded quants show their
	// checkmark (otherwise isModelDownloaded(...) stays false on hard refresh).
	onMount(() => {
		void modelsStore.fetchRouterModels();
	});

	$effect(() => {
		loadModel();
	});
</script>

<div class="flex h-full flex-col {className}">
	<!-- Pane header: back, model id + copy, View on HF -->
	<header
		class="flex shrink-0 items-center gap-2 border-b border-border/40 bg-background/90 px-4 py-3 backdrop-blur"
	>
		<div class="lg:hidden">
			<ActionIcon icon={ArrowLeft} tooltip="Back to models" onclick={handleBack} />
		</div>

		<h1 class="min-w-0 flex-1 truncate font-semibold">{modelId}</h1>

		<button
			type="button"
			onclick={handleCopy}
			class="inline-flex shrink-0 items-center gap-1.5 rounded-md border px-2 py-1 text-xs font-medium text-muted-foreground transition-colors hover:border-border hover:text-foreground"
			aria-label="Copy model id"
		>
			{#if copied}
				<Check class="h-3.5 w-3.5 text-primary" />
			{:else}
				<Copy class="h-3.5 w-3.5" />
			{/if}
			{copied ? 'Copied' : 'Copy'}
		</button>

		<a
			href={`https://huggingface.co/${modelId}`}
			target="_blank"
			rel="noopener noreferrer"
			class="inline-flex shrink-0 items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90"
		>
			<ExternalLink size={14} />
			<span class="hidden sm:inline">View on HF</span>
		</a>
	</header>

	<div class="min-h-0 flex-1 overflow-y-auto">
		<Breadcrumb
			items={[{ href: ROUTES.MANAGE_MODELS, label: 'Models' }, { label: modelId }]}
			class="px-4 pt-4 md:px-6"
		/>

		{#if error}
			<div
				class="mx-4 mt-4 rounded-lg border border-destructive/50 bg-destructive/5 p-4 text-center md:mx-6"
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
			<div class="flex flex-wrap items-center gap-1.5 px-4 pt-4 md:px-6">
				{#if author}
					<span
						class="rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground"
					>
						{author}
					</span>
				{/if}
				{#if modelInfo.pipeline_tag}
					<span class="rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
						{HuggingFaceService.pipelineTagLabel(modelInfo.pipeline_tag)}
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

			<!-- Two-column content -->
			<div class="grid gap-6 px-4 py-6 md:px-6 xl:grid-cols-3">
				<main class="min-w-0 space-y-6 xl:col-span-2">
					{#if description}
						<section class="rounded-lg border bg-card p-5">
							<h2 class="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
								About
							</h2>
							<p class="text-sm whitespace-pre-line text-foreground/90">{description}</p>
						</section>
					{/if}

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

				<aside class="space-y-4 xl:sticky xl:top-4 xl:self-start">
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
									<dd class="font-medium tabular-nums">
										{HuggingFaceService.formatDownloads(modelInfo.downloads)}
									</dd>
								</div>
							{/if}
							{#if typeof modelInfo.likes === 'number'}
								<div class="flex items-center justify-between">
									<dt class="flex items-center gap-1.5 text-muted-foreground">
										<Heart size={13} />
										Likes
									</dt>
									<dd class="font-medium tabular-nums">
										{HuggingFaceService.formatLikes(modelInfo.likes)}
									</dd>
								</div>
							{/if}
							{#if details?.lastModified}
								<div class="flex items-center justify-between">
									<dt class="text-muted-foreground">Last modified</dt>
									<dd class="font-medium">
										{HuggingFaceService.formatRelativeTime(details.lastModified)}
									</dd>
								</div>
							{/if}
							{#if gguf?.totalFileSize}
								<div class="flex items-center justify-between">
									<dt class="text-muted-foreground">Total size</dt>
									<dd class="font-medium tabular-nums">
										{HuggingFaceService.formatFileSize(gguf.totalFileSize)}
									</dd>
								</div>
							{/if}
						</dl>
					</section>

					{#if ggufFiles.length}
						<section class="rounded-lg border bg-card p-4">
							<header class="mb-3 flex items-center gap-1.5">
								<Cpu size={13} class="text-muted-foreground" />
								<h2 class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
									Available files
								</h2>
							</header>

							<div class="space-y-2">
								{#each bitDepthRows as row (row.bitDepth)}
									<div class="grid grid-cols-[5rem_1fr] items-center gap-3">
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
												{@const downloadProgress =
													modelsStore.status.getDownloadProgress(hfRepoWithTag)}
												{@const isDownloading =
													modelsStore.status.isDownloadInProgress(hfRepoWithTag)}
												{@const isFullyDownloaded =
													modelsStore.status.isModelDownloaded(hfRepoWithTag)}
												{@const isFailed = modelsStore.status.hasFailedDownload(hfRepoWithTag)}
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
															quant: meta?.quant ?? null,
															sizeBytes: file.size ?? null,
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
																(downloadProgress.downloadedBytes / downloadProgress.totalBytes) *
																	100
															)}%
														{:else}
															{HuggingFaceService.formatFileSize(file.size ?? 0)}
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
				</aside>
			</div>
		{/if}
	</div>
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
			? HuggingFaceService.formatFileSize(pendingDownload.sizeBytes)
			: ''}
		onConfirm={() => (pendingDownload = null)}
		onCancel={() => (pendingDownload = null)}
	/>
{/if}
