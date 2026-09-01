<script lang="ts">
	import { type DownloadEntryState, labelFor } from './download-options.utils';
	import DownloadProgressBar from './DownloadProgressBar.svelte';
	import { Check } from '@lucide/svelte';
	import { ToggleGroupItem } from '$lib/components/ui/toggle-group';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { isAuxSidecar } from '$lib/constants';
	import { HuggingFaceService } from '$lib/services';
	import type { HfModelSibling } from '$lib/types/huggingface';

	interface Props {
		/** GGUF file the chip stands for. */
		file: HfModelSibling;
		/** Download state of the file, from the parent's status feed. */
		state: DownloadEntryState;
	}

	let { file, state }: Props = $props();

	let meta = $derived(HuggingFaceService.extractQuantMeta(file.path));
	let label = $derived(labelFor(file.path));

	let tooltipText = $derived(
		state.isDownloading
			? `Downloading ${file.path}`
			: state.isDownloaded
				? `Already downloaded: ${file.path}`
				: state.isFailed
					? `Last attempt failed: ${file.path}`
					: `Download ${file.path}`
	);
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		{#if state.isDownloaded}
			<!-- downloaded files are not selectable, just marked as done -->
			<div
				aria-disabled="true"
				aria-label={tooltipText}
				class="inline-flex cursor-default items-center gap-1 rounded-md border border-green-600/25 bg-green-500/5 px-2 py-1 font-mono text-xs dark:border-green-500/30 dark:bg-green-500/10"
			>
				{#if meta?.sidecar && !isAuxSidecar(meta.sidecar)}
					<span
						class="rounded-md bg-primary px-1 py-0.5 text-[10px] font-semibold tracking-wide text-primary-foreground uppercase"
					>
						{meta.sidecar}{meta.shared ? '-shared' : ''}
					</span>
				{/if}

				<span class="font-medium">{label}</span>

				<span class="-my-1 w-px mx-0.75 self-stretch bg-green-600/25 dark:bg-green-600/30"></span>

				<span>{HuggingFaceService.formatFileSize(file.size ?? 0)}</span>

				<Check class="h-3.5 w-3.5 shrink-0 text-green-500" />
			</div>
		{:else}
			<ToggleGroupItem
				aria-label={tooltipText}
				class="relative inline-flex h-auto items-center gap-1 overflow-hidden rounded-md! border border-border/30 bg-background px-2 py-1 text-left font-mono text-xs shadow-xs transition-colors hover:data-[state=off]:bg-muted-foreground/10 data-[state=on]:border-primary data-[state=on]:bg-primary/10 data-[state=on]:hover:bg-primary/15 dark:border-border/20 dark:bg-muted-foreground/15 dark:text-secondary-foreground dark:data-[state=on]:border-primary dark:data-[state=on]:bg-primary/15 dark:data-[state=on]:hover:bg-primary/25 {state.isFailed
					? 'border-destructive!'
					: ''}"
				value={file.path}
			>
				{#if state.isFailed && !state.isDownloading}
					<span
						class="rounded-md bg-destructive px-1 py-0.5 text-[10px] font-semibold tracking-wide text-destructive-foreground uppercase"
					>
						Failed
					</span>
				{/if}

				{#if meta?.sidecar && !isAuxSidecar(meta.sidecar)}
					<span
						class="rounded-md bg-primary px-1 py-0.5 text-[10px] font-semibold tracking-wide text-primary-foreground uppercase"
					>
						{meta.sidecar}{meta.shared ? '-shared' : ''}
					</span>
				{/if}

				<span class="font-medium">{label}</span>

				<span class="-my-1 mx-0.75 w-px self-stretch bg-border"></span>

				<span>
					{#if state.isDownloading && state.progress && state.progress.totalBytes > 0}
						{Math.round((state.progress.downloadedBytes / state.progress.totalBytes) * 100)}%
					{:else}
						{HuggingFaceService.formatFileSize(file.size ?? 0)}
					{/if}
				</span>

				{#if state.isDownloading && state.progress}
					<DownloadProgressBar
						downloadedBytes={state.progress.downloadedBytes}
						overlay
						totalBytes={state.progress.totalBytes}
					/>
				{/if}
			</ToggleGroupItem>
		{/if}
	</Tooltip.Trigger>

	<Tooltip.Content>
		<p>{tooltipText}</p>
	</Tooltip.Content>
</Tooltip.Root>
