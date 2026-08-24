<script lang="ts">
	import {
		Check,
		Copy,
		Download,
		ExternalLink,
		Eye,
		Heart,
		MessageSquareCode,
		Sparkles,
		Wrench
	} from '@lucide/svelte';
	import { MarkdownContent } from '$lib/components/app';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import type { GgufVariantTagInput } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type { HfModelDetailInfo, HfModelSibling } from '$lib/types/huggingface';
	import { copyToClipboard } from '$lib/utils';
	import { SvelteMap } from 'svelte/reactivity';
	import DialogModelDownload from './DialogModelDownload.svelte';
	import DownloadProgressBar from './DownloadProgressBar.svelte';

	interface Props {
		/** Full HuggingFace model id, e.g. `ggml-org/gemma-3-4b-it-GGUF`. */
		modelId: string;
	}

	interface PendingDownload {
		filePath: string;
		sizeBytes: number | null;
		quant: string | null;
		variant: GgufVariantTagInput['variant'];
	}

	let { modelId }: Props = $props();

	let details = $state<HfModelDetailInfo | null>(null);
	let files = $state<HfModelSibling[]>([]);
	let readme = $state<string | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let pendingDownload = $state<PendingDownload | null>(null);

	let gguf = $derived(details?.gguf);
	let baseModels = $derived(HuggingFaceService.getBaseModels(details));
	let licenseTag = $derived.by(() => {
		const tags = details?.tags ?? [];

		return tags.find((t) => t.startsWith('license:'))?.replace('license:', '') ?? null;
	});

	// Capabilities derived from HF metadata. Vision comes from an mmproj sidecar
	// or a multimodal pipeline tag; tool use / reasoning from the chat template.
	let hasMmproj = $derived(
		files.some((f) => HuggingFaceService.extractQuantMeta(f.path)?.variant === 'mmproj')
	);
	let hasVision = $derived(hasMmproj || details?.pipeline_tag === 'image-text-to-text');
	let hasTools = $derived(Boolean(gguf?.chat_template && /tools?[_\s}]/i.test(gguf.chat_template)));
	let hasReasoning = $derived(
		Boolean(gguf?.chat_template && /think|reasoning/i.test(gguf.chat_template))
	);

	type BitDepthRow = { bitDepth: number; files: HfModelSibling[] };
	let bitDepthRows = $derived.by<BitDepthRow[]>(() => {
		const rows = new SvelteMap<number, HfModelSibling[]>();

		for (const file of files) {
			const meta = HuggingFaceService.extractQuantMeta(file.path);

			// mmproj sidecars are already conveyed by the Vision capability badge.
			if (meta?.variant === 'mmproj') continue;

			const depth = meta?.quant ? HuggingFaceService.getBitDepth(meta.quant) : null;
			const bucket = depth ?? 99;
			const list = rows.get(bucket) ?? [];

			list.push(file);
			rows.set(bucket, list);
		}

		return Array.from(rows.entries())
			.map(([bitDepth, rowFiles]) => ({ bitDepth, files: rowFiles }))
			.sort((a, b) => a.bitDepth - b.bitDepth);
	});

	async function load(id: string) {
		loading = true;
		error = null;

		try {
			const [info, tree, readmeText] = await Promise.all([
				HuggingFaceService.getDetails(id),
				HuggingFaceService.getTree(id),
				HuggingFaceService.getReadme(id)
			]);

			if (!info) {
				error = 'Model not found';

				return;
			}

			details = info;
			files = HuggingFaceService.filterByExtension(tree, '.gguf');
			readme = readmeText;
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load model';
		} finally {
			loading = false;
		}
	}

	// Re-fetch when the dialog selects a different model (component is reused).
	$effect(() => {
		void load(modelId);
	});
</script>

{#if loading}
	<div class="flex h-full items-center justify-center py-20">
		<p class="text-sm text-muted-foreground">Loading model...</p>
	</div>
{:else if error}
	<div class="flex h-full items-center justify-center py-20">
		<p class="text-sm text-destructive">{error}</p>
	</div>
{:else if details}
	<div class="space-y-6 p-6">
		<!-- Header -->
		<header class="space-y-3">
			<div class="flex items-start justify-between gap-3">
				<div class="flex min-w-0 items-center gap-2">
					<h1 class="truncate text-lg font-semibold">{details.id}</h1>
					<button
						type="button"
						onclick={() => copyToClipboard(details?.id ?? modelId)}
						class="shrink-0 text-muted-foreground transition-colors hover:text-foreground"
						aria-label="Copy model id"
					>
						<Copy class="h-4 w-4" />
					</button>
				</div>
				<a
					href={HuggingFaceService.getModelUrl(modelId)}
					target="_blank"
					rel="noopener noreferrer"
					class="inline-flex shrink-0 items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors hover:bg-muted"
				>
					<ExternalLink class="h-3.5 w-3.5" />
					View on HF
				</a>
			</div>

			<div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-muted-foreground">
				{#if typeof details.downloads === 'number'}
					<span class="inline-flex items-center gap-1.5">
						<Download class="h-3.5 w-3.5" />
						{HuggingFaceService.formatDownloads(details.downloads)}
					</span>
				{/if}
				{#if typeof details.likes === 'number'}
					<span class="inline-flex items-center gap-1.5">
						<Heart class="h-3.5 w-3.5" />
						{HuggingFaceService.formatLikes(details.likes)}
					</span>
				{/if}
				{#if details.lastModified}
					<span>Updated {HuggingFaceService.formatRelativeTime(details.lastModified)}</span>
				{/if}
			</div>

			{#if details.cardData?.description}
				<p class="text-sm text-muted-foreground">{details.cardData.description}</p>
			{/if}

			<!-- Metadata chips -->
			<div class="flex flex-wrap items-center gap-1.5">
				{#if gguf?.total}
					<span class="rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
						{HuggingFaceService.formatFileSize(gguf.total).replace(' B', '')}B params
					</span>
				{/if}
				{#if gguf?.architecture}
					<span class="rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground capitalize">
						{gguf.architecture.replace(/_/g, ' ')}
					</span>
				{/if}
				{#if gguf?.context_length}
					<span class="rounded bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground">
						{gguf.context_length.toLocaleString()} ctx
					</span>
				{/if}
				{#if licenseTag}
					<span class="rounded bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
						{licenseTag}
					</span>
				{/if}
				{#if details.gated === true}
					<span class="rounded bg-yellow-500/10 px-2 py-0.5 text-xs font-medium text-yellow-600 dark:text-yellow-400">
						gated
					</span>
				{/if}
			</div>

			<!-- Capability badges -->
			{#if hasVision || hasTools || hasReasoning}
				<div class="flex flex-wrap items-center gap-1.5">
					{#if hasVision}
						<span class="inline-flex items-center gap-1 rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
							<Eye class="h-3 w-3" />
							Vision
						</span>
					{/if}
					{#if hasTools}
						<span class="inline-flex items-center gap-1 rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
							<Wrench class="h-3 w-3" />
							Tool use
						</span>
					{/if}
					{#if hasReasoning}
						<span class="inline-flex items-center gap-1 rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
							<Sparkles class="h-3 w-3" />
							Reasoning
						</span>
					{/if}
				</div>
			{/if}
		</header>

		<!-- Download options -->
		{#if bitDepthRows.length}
			<section class="space-y-3">
				<h2 class="flex items-center gap-1.5 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
					<MessageSquareCode class="h-3.5 w-3.5" />
					Download options
				</h2>
				<div class="space-y-2">
					{#each bitDepthRows as row (row.bitDepth)}
						<div class="grid grid-cols-[5rem_1fr] items-start gap-3">
							<div class="pt-1 text-xs font-semibold tabular-nums text-muted-foreground">
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
									{@const label = meta?.quant ?? basename.replace(/\.gguf$/i, '')}
									{@const tagInput = meta?.quant
										? { quant: meta.quant, variant: meta.variant ?? null }
										: null}
									{@const hfRepoWithTag = ModelsService.buildDownloadTag(modelId, tagInput)}
									{@const progress = modelsStore.status.getDownloadProgress(hfRepoWithTag)}
									{@const isDownloading = modelsStore.status.isDownloadInProgress(hfRepoWithTag)}
									{@const isDownloaded = meta?.variant
										? modelsStore.status.isDraftDownloaded(modelId, file.path)
										: modelsStore.status.isModelDownloaded(hfRepoWithTag)}
									{@const isFailed = modelsStore.status.hasFailedDownload(hfRepoWithTag)}
									<button
										type="button"
										onclick={() =>
											(pendingDownload = {
												filePath: file.path,
												sizeBytes: file.size ?? null,
												quant: meta?.quant ?? null,
												variant: meta?.variant ?? null
											})}
										title={isDownloading
											? `Downloading ${file.path}`
											: isDownloaded
												? `Already downloaded: ${file.path}`
												: isFailed
													? `Last attempt failed: ${file.path}. Click to retry.`
													: `Download ${file.path}`}
										class="relative inline-flex cursor-pointer items-center gap-1 overflow-hidden rounded-md border bg-background px-2 py-1 text-left font-mono text-xs transition-colors hover:border-primary/60 hover:bg-primary/5"
										class:border-foreground={isDownloaded && !isDownloading && !isFailed}
										class:bg-muted={isDownloaded && !isDownloading && !isFailed}
										class:border-destructive={isFailed && !isDownloading}
									>
										{#if isDownloaded && !isDownloading}
											<Check class="h-3 w-3 text-foreground/70" />
										{/if}
										{#if isFailed && !isDownloading && !isDownloaded}
											<span class="rounded bg-destructive px-1 py-0.5 text-[10px] font-semibold tracking-wide text-destructive-foreground uppercase">
												Failed
											</span>
										{/if}
										{#if meta?.variant}
											<span class="rounded bg-primary px-1 py-0.5 text-[10px] font-semibold tracking-wide text-primary-foreground uppercase">
												{meta.variant}
											</span>
										{/if}
										<span class="font-medium">{label}</span>
										<span class="text-muted-foreground">
											{#if isDownloading && progress && progress.totalBytes > 0}
												{Math.round((progress.downloadedBytes / progress.totalBytes) * 100)}%
											{:else}
												{HuggingFaceService.formatFileSize(file.size ?? 0)}
											{/if}
										</span>
										{#if isDownloading && progress}
											<DownloadProgressBar
												overlay
												downloadedBytes={progress.downloadedBytes}
												totalBytes={progress.totalBytes}
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

		<!-- Base model -->
		{#if baseModels.length}
			<section class="space-y-2">
				<h2 class="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
					Base model
				</h2>
				<ul class="space-y-1">
					{#each baseModels as base (base)}
						<li>
							<a
								href={HuggingFaceService.getModelUrl(base)}
								target="_blank"
								rel="noopener noreferrer"
								class="inline-flex items-center gap-1.5 rounded-md px-1 py-0.5 font-mono text-xs transition-colors hover:bg-muted"
							>
								{base}
								<ExternalLink class="h-3 w-3 opacity-60" />
							</a>
						</li>
					{/each}
				</ul>
			</section>
		{/if}

		<!-- README -->
		{#if readme}
			<section class="space-y-2">
				<h2 class="text-xs font-semibold tracking-wide text-muted-foreground uppercase">README</h2>
				<article class="rounded-lg border bg-card p-4">
					<MarkdownContent content={readme} class="prose-sm max-w-none" />
				</article>
			</section>
		{/if}
	</div>
{/if}

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
		formattedSize={pendingDownload.sizeBytes != null
			? HuggingFaceService.formatFileSize(pendingDownload.sizeBytes)
			: undefined}
		onConfirm={() => (pendingDownload = null)}
		onCancel={() => (pendingDownload = null)}
	/>
{/if}
