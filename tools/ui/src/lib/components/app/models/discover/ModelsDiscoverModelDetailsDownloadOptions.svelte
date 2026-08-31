<script lang="ts">
	import DialogModelDownload from './DialogModelDownload.svelte';
	import DownloadProgressBar from './DownloadProgressBar.svelte';
	import { Download } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { isAuxSidecar, type ModelSidecar } from '$lib/constants';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type { ModelDownloadProgress } from '$lib/types';
	import type { HfModelSibling } from '$lib/types/huggingface';
	import { estimateModelMemoryBytes } from '$lib/utils';

	/** Download state of a single repo entry, injected by the integration layer. */
	export interface DownloadEntryState {
		isDownloading: boolean;
		progress: ModelDownloadProgress | null;
		isDownloaded: boolean;
		isFailed: boolean;
	}

	type BitDepthRow = { bitDepth: number; files: HfModelSibling[] };

	interface PendingDownload {
		filePath: string;
		sizeBytes: number | null;
		quant: string | null;
		sidecar: ModelSidecar | null;
	}

	interface Props {
		/** Full HuggingFace repo id, e.g. `ggml-org/gemma-3-4b-it-GGUF`. */
		modelId: string;
		/** GGUF files grouped by bit depth. */
		bitDepthRows: BitDepthRow[];
		/** Download state lookup; defaults to the models store status feed. */
		getDownloadState?: (
			repoWithTag: string,
			filePath: string,
			isSidecar: boolean
		) => DownloadEntryState;
	}

	let { bitDepthRows, getDownloadState, modelId }: Props = $props();

	let pendingDownload: PendingDownload | null = $state(null);

	function stateFor(repoWithTag: string, filePath: string, isSidecar: boolean): DownloadEntryState {
		if (getDownloadState) return getDownloadState(repoWithTag, filePath, isSidecar);

		return {
			isDownloaded: isSidecar
				? modelsStore.status.isDraftDownloaded(modelId, filePath)
				: modelsStore.status.isModelDownloaded(repoWithTag),
			isDownloading: modelsStore.status.isDownloadInProgress(repoWithTag),
			isFailed: modelsStore.status.hasFailedDownload(repoWithTag),
			progress: modelsStore.status.getDownloadProgress(repoWithTag)
		};
	}

	function buttonClass(parts: { isDownloaded: boolean; isFailed: boolean }): string {
		const classes = [
			'relative inline-flex items-center gap-1 overflow-hidden rounded-md border bg-muted px-2 py-1 text-left font-mono text-xs transition-colors'
		];

		if (parts.isDownloaded && !parts.isFailed) {
			classes.push('border-foreground bg-muted');
		} else if (parts.isFailed) {
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
							{@const hfRepoWithTag = ModelsService.buildDownloadTag(
								modelId,
								meta?.quant ?? null,
								meta?.sidecar ?? null
							)}
							{@const state = stateFor(hfRepoWithTag, file.path, Boolean(meta?.sidecar))}
							{@const isDownloading = state.isDownloading}
							{@const progress = state.progress}
							{@const isDownloaded = state.isDownloaded}
							{@const isFailed = state.isFailed}
							{@const memoryGb = Math.ceil(estimateModelMemoryBytes(file.size ?? 0) / 1024 ** 3)}
							{@const tooltipText = isDownloading
								? `Downloading ${file.path}`
								: isDownloaded
									? `Already downloaded: ${file.path}`
									: isFailed
										? `Last attempt failed: ${file.path}. Click to retry.`
										: `Download ${file.path} (requires ~${memoryGb} GB of memory)`}
							<Tooltip.Root>
								<Tooltip.Trigger
									class={buttonClass({ isDownloaded, isFailed })}
									onclick={() => {
										pendingDownload = {
											filePath: file.path,
											quant: meta?.quant ?? null,
											sidecar: meta?.sidecar ?? null,
											sizeBytes: file.size ?? null
										};
									}}
									type="button"
								>
									{#if isFailed && !isDownloading && !isDownloaded}
										<span
											class="rounded bg-destructive px-1 py-0.5 text-[10px] font-semibold tracking-wide text-destructive-foreground uppercase"
										>
											Failed
										</span>
									{/if}

									{#if meta?.sidecar && !isAuxSidecar(meta.sidecar)}
										<span
											class="rounded bg-primary px-1 py-0.5 text-[10px] font-semibold tracking-wide text-primary-foreground uppercase"
										>
											{meta.sidecar}
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
											downloadedBytes={progress.downloadedBytes}
											overlay
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
		filePath={pendingDownload.filePath}
		formattedSize={pendingDownload.sizeBytes != null
			? HuggingFaceService.formatFileSize(pendingDownload.sizeBytes)
			: undefined}
		onClose={() => (pendingDownload = null)}
		quant={pendingDownload.quant}
		repoId={modelId}
		sidecar={pendingDownload.sidecar}
	/>
{/if}
