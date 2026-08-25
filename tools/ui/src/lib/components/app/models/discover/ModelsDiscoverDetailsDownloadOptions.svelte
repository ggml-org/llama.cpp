<script lang="ts">
	import DialogModelDownload from './DialogModelDownload.svelte';
	import DownloadProgressBar from './DownloadProgressBar.svelte';
	import { Check, Cpu, Download, MessageSquareCode, Monitor, TriangleAlert, X } from '@lucide/svelte';
	import { browser } from '$app/environment';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import type { GgufVariantTagInput } from '$lib/services';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore, settingsStore } from '$lib/stores';
	import type { HfModelSibling } from '$lib/types/huggingface';
	import { computeFileCompatibilityTiers, detectOs, resolveDeviceMemoryGb } from '$lib/utils';

	interface Props {
		modelId: string;
		files: HfModelSibling[];
		bitDepthRows: BitDepthRow[];
		nativeCtxTokens: number;
	}

	interface PendingDownload {
		filePath: string;
		sizeBytes: number | null;
		quant: string | null;
		variant: GgufVariantTagInput['variant'];
	}

	type BitDepthRow = { bitDepth: number; files: HfModelSibling[] };

	let { bitDepthRows, files, modelId, nativeCtxTokens }: Props = $props();

	let pendingDownload = $state<PendingDownload | null>(null);

	let deviceMemoryGb = $derived(
		resolveDeviceMemoryGb(Number(settingsStore.config.deviceMemoryGb) || 0)
	);
	let osLabel = $derived(browser ? detectOs(navigator.userAgent) : 'unknown');
	let tiers = $derived(computeFileCompatibilityTiers(files, nativeCtxTokens, deviceMemoryGb));

	function buttonClass(parts: {
		isDownloaded: boolean;
		isFailed: boolean;
		isUnavailable: boolean;
	}): string {
		const { isDownloaded, isFailed, isUnavailable } = parts;
		const classes = [
			'relative inline-flex items-center gap-1 overflow-hidden rounded-md border bg-muted px-2 py-1 text-left font-mono text-xs transition-colors'
		];

		// Buttons stay neutral; only the leading compatibility badge carries
		// color (green/yellow/red). Unavailable quants are greyed + disabled.
		if (isUnavailable) {
			classes.push('cursor-not-allowed opacity-50');
		} else {
			classes.push('cursor-pointer hover:border-primary/60 hover:bg-primary/5');
		}

		if (isDownloaded && !isFailed) {
			classes.push('border-foreground bg-muted');
		} else if (isFailed) {
			classes.push('border-destructive');
		}

		return classes.join(' ');
	}
</script>

{#if bitDepthRows.length}
	<section class="rounded-xl border">
		<div class="flex flex-wrap items-center justify-between gap-2 px-4 pt-3 pb-1">
			<h2 class="flex items-center gap-1.5 text-sm font-medium text-muted-foreground">
				<Download class="h-4 w-4" />
				Downloadable options
			</h2>

			<span
				class="inline-flex items-center gap-1.5 rounded-full border bg-background px-2.5 py-1 text-xs font-medium"
			>
				<Cpu class="h-3 w-3 text-muted-foreground" />
				{osLabel}
				{#if deviceMemoryGb > 0}
					<span class="text-muted-foreground">({deviceMemoryGb} GB)</span>
				{/if}
			</span>
		</div>
		<div class="divide-y px-4 pb-1">
			{#each bitDepthRows as row (row.bitDepth)}
				<div class="grid grid-cols-[5rem_1fr] items-start gap-3 py-3">
					<div class="pt-1 text-sm tabular-nums text-muted-foreground">
						{#if row.bitDepth === 99}
							Other
						{:else}
							{row.bitDepth}-bit
						{/if}
					</div>
					<div class="flex flex-wrap justify-end gap-1.5">
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
							{@const tier = tiers.get(file.path)}
							{@const isUnavailable =
								tier === 'none' && !isDownloaded && !isDownloading && !isFailed}
							{@const isAvailable = tier === 'full' && !isDownloaded && !isDownloading && !isFailed}
							{@const isLimited =
								tier === 'limited' && !isDownloaded && !isDownloading && !isFailed}
							{@const tooltipText = isDownloading
								? `Downloading ${file.path}`
								: isDownloaded
									? `Already downloaded: ${file.path}`
									: isFailed
										? `Last attempt failed: ${file.path}. Click to retry.`
										: isUnavailable
											? `Does not fit this device: ${file.path}`
											: `Download ${file.path}`}
							<Tooltip.Root>
								<Tooltip.Trigger
									type="button"
									onclick={() => {
										if (isUnavailable) return;

										pendingDownload = {
											filePath: file.path,
											quant: meta?.quant ?? null,
											sizeBytes: file.size ?? null,
											variant: meta?.variant ?? null
										};
									}}
									aria-disabled={isUnavailable}
									class={buttonClass({
										isDownloaded,
										isFailed,
										isUnavailable
									})}
								>
									{#if isAvailable}
										<span
											class="flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full bg-green-600"
										>
											<Check class="h-2.5 w-2.5 text-white" />
										</span>
									{:else if isLimited}
										<span
											class="flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full bg-yellow-500"
										>
											<TriangleAlert class="h-2.5 w-2.5 text-white" />
										</span>
									{:else if isUnavailable}
										<span
											class="flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full bg-red-600"
										>
											<X class="h-2.5 w-2.5 text-white" />
										</span>
									{/if}
									{#if isFailed && !isDownloading && !isDownloaded}
										<span
											class="rounded bg-destructive px-1 py-0.5 text-[10px] font-semibold tracking-wide text-destructive-foreground uppercase"
										>
											Failed
										</span>
									{/if}
									{#if meta?.variant}
										<span
											class="rounded bg-primary px-1 py-0.5 text-[10px] font-semibold tracking-wide text-primary-foreground uppercase"
										>
											{meta.variant}
										</span>
									{/if}
									<span class="font-medium {isDownloaded ? '' : 'text-muted-foreground/80'}"
										>{label}</span
									>
									<span class="-my-1 w-px self-stretch bg-border"></span>
									<span class={isDownloaded ? '' : 'text-muted-foreground/80'}>
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
								</Tooltip.Trigger>

								<Tooltip.Content>
									<p>{tooltipText}</p>
								</Tooltip.Content>
							</Tooltip.Root>
						{/each}
					</div>
				</div>
			{/each}
		</div>
	</section>
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
