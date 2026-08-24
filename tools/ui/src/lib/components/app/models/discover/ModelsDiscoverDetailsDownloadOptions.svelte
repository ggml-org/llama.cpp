<script lang="ts">
	import { Check, MessageSquareCode } from '@lucide/svelte';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import type { GgufVariantTagInput } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type { HfModelSibling } from '$lib/types/huggingface';
	import DialogModelDownload from './DialogModelDownload.svelte';
	import DownloadProgressBar from './DownloadProgressBar.svelte';

	interface Props {
		modelId: string;
		bitDepthRows: BitDepthRow[];
	}

	interface PendingDownload {
		filePath: string;
		sizeBytes: number | null;
		quant: string | null;
		variant: GgufVariantTagInput['variant'];
	}

	type BitDepthRow = { bitDepth: number; files: HfModelSibling[] };

	let { bitDepthRows, modelId }: Props = $props();

	let pendingDownload = $state<PendingDownload | null>(null);
</script>

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
