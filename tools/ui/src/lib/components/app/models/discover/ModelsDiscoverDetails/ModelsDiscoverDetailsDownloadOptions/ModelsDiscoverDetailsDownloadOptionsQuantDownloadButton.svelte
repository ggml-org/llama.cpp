<script lang="ts">
	import ModelDownloadProgressBar from '../../../ModelDownloadProgressBar.svelte';
	import { labelFor } from './download-options.utils';
	import { Check, Download, Loader2, Pause, Play, RotateCw, X } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { HuggingFaceService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type { ModelQuantChip } from '$lib/types';
	import type { Snippet } from 'svelte';

	interface Props {
		/** The main quant chip: its file, its download state and its joined drafts. */
		file: ModelQuantChip;
		/** Ask the parent to confirm deleting a downloaded model; the chip owns no dialog. */
		onRequestDelete?: (repoWithTag: string) => void;
		/** Ask the parent to confirm cancelling an in-flight download. */
		onRequestCancel?: (repoWithTag: string) => void;
	}

	let { file, onRequestCancel, onRequestDelete }: Props = $props();

	let entry = $derived(file.state);

	// one source, so the idle, in-flight and downloaded chip states stay identical
	const SIDECAR_BADGE_CLASS =
		'rounded-md bg-primary px-1 py-0.5 text-[10px] font-semibold tracking-wide text-primary-foreground uppercase';

	/** Queue the download; a failed attempt leaves partial files, drop them first. */
	async function startDownload() {
		try {
			if (entry.isFailed) await modelsStore.status.cancelDownload(entry.repoWithTag);

			// the joined drafts and the mmproj ride along: the server picks their files
			await modelsStore.status.downloadModel(entry.repoWithTag, file.sidecars);
		} catch {
			// the store already toasted the failure
		}
	}

	let label = $derived(labelFor(file.path));

	let percent = $derived(
		entry.progress && entry.progress.totalBytes > 0
			? Math.round((entry.progress.downloadedBytes / entry.progress.totalBytes) * 100)
			: null
	);

	/** Draft the chip pulls with its quant, e.g. `MTP`; null when none matches. */
	let companion = $derived(file.companion);

	/**
	 * The joined draft reads once it rides along or lands: a downloaded model
	 * missing its draft keeps to its own weights, and the draft gets its own chip.
	 */
	let showsCompanion = $derived(companion ? companion.onDisk || !entry.isDownloaded : false);

	/** What the chip's actions name: the quant file and, when it rides along, its draft. */
	let downloadLabel = $derived(
		showsCompanion && companion ? `${file.path} + ${companion.path}` : file.path
	);

	let tooltipText = $derived(
		entry.isDownloading
			? 'Pause downloading'
			: entry.isPaused
				? 'Resume downloading'
				: entry.isDownloaded
					? `Delete ${downloadLabel}`
					: entry.isFailed
						? `Retry download: ${downloadLabel}`
						: `Download ${downloadLabel}`
	);

	let downloadBytes = $derived(
		(file.size ?? 0) + (showsCompanion && companion ? companion.size : 0)
	);
</script>

{#snippet chipBody(dividerClass: string)}
	<!-- label, joined draft and divider: identical on every chip state. The draft
	     rides with this quant, so it reads as part of the same chip:
	     `Q4_K_M + [MTP:Q8_0] | 20.0 GB`, the size covering both files. -->
	<span class="font-medium">{label}</span>

	{#if showsCompanion && companion}
		<span class="text-muted-foreground/60">+</span>

		<span class="{SIDECAR_BADGE_CLASS} inline-flex items-center gap-0.5">
			<!-- the draft's own quant, so a 4-bit model paired with a 16-bit draft head reads at a glance -->
			{companion.kind}{companion.quant ? `:${companion.quant}` : ''}

			{#if companion.onDisk}
				<Check class="size-2.5" />
			{/if}
		</span>
	{/if}

	<span class={dividerClass}></span>
{/snippet}

<!-- every chip is a tooltip trigger; the button renders as the trigger's child
     so no nested button element is created -->
{#snippet tooltipTrigger(tip: string, button: Snippet<[Record<string, unknown>]>)}
	<Tooltip.Root>
		<Tooltip.Trigger>
			{#snippet child({ props })}
				{@render button(props)}
			{/snippet}
		</Tooltip.Trigger>

		<Tooltip.Content>
			<p>{tip}</p>
		</Tooltip.Content>
	</Tooltip.Root>
{/snippet}

{#snippet deleteChip(props: Record<string, unknown>)}
	<!-- downloaded chips are delete actions: green check by default, red X on hover -->
	<button
		{...props}
		aria-label={tooltipText}
		class="group relative inline-flex h-auto cursor-pointer items-center gap-1 rounded-md! border px-2 py-1 text-left font-mono text-xs shadow-sm transition-[background-color,border-color,transform] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97]
					border-green-600/25 bg-green-500/5 hover:border-destructive/50 hover:bg-destructive/10 dark:border-green-500/30 dark:bg-green-500/10 dark:hover:border-destructive/50 dark:hover:bg-destructive/15"
		onclick={() => onRequestDelete?.(entry.repoWithTag)}
		type="button"
	>
		{@render chipBody(
			'-my-1 mx-0.75 w-px self-stretch bg-green-600/25 transition-colors duration-200 group-hover:bg-destructive/30 dark:bg-green-600/30 dark:group-hover:bg-destructive/30'
		)}

		<span>{HuggingFaceService.formatFileSize(downloadBytes)}</span>

		<!-- icon slot: crossfade check -> x; touch devices show the delete affordance directly -->
		<span class="relative inline-flex size-3.5 shrink-0 items-center justify-center">
			<Check
				class="absolute size-icon-sm text-green-500 transition-[opacity,transform] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:scale-75 group-hover:opacity-0 [@media(pointer:coarse)]:hidden"
			/>

			<X
				class="absolute size-icon-sm scale-75 text-destructive opacity-0 transition-[opacity,transform] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:scale-100 group-hover:opacity-100 [@media(pointer:coarse)]:scale-100 [@media(pointer:coarse)]:opacity-100"
			/>
		</span>
	</button>
{/snippet}

{#snippet pauseResumeChip(props: Record<string, unknown>)}
	<button
		{...props}
		aria-label={tooltipText}
		class="flex h-auto min-w-0 flex-1 cursor-pointer items-center gap-1 text-left"
		onclick={() => {
			if (entry.isDownloading) void modelsStore.status.pauseDownload(entry.repoWithTag);
			else void modelsStore.status.downloadModel(entry.repoWithTag).catch(() => {});
		}}
		type="button"
	>
		{@render chipBody('-my-1 mx-0.75 w-px self-stretch bg-border')}

		{#if percent !== null}
			<span class="mr-1 tabular-nums">{percent}%</span>
		{:else if entry.isPaused}
			<span>Paused</span>
		{:else}
			<span>{HuggingFaceService.formatFileSize(downloadBytes)}</span>
		{/if}

		{#if entry.isDownloading}
			<!-- spinner fades into the pause affordance on hover; opacity only, the
			     spin keyframes own the transform so scale would fight them -->
			<span class="relative inline-flex size-3.5 shrink-0 items-center justify-center">
				<Loader2
					class="absolute size-icon-sm animate-spin text-muted-foreground transition-opacity duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:opacity-0 [@media(pointer:coarse)]:hidden"
				/>

				<Pause
					class="absolute size-icon-sm opacity-0 transition-opacity duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:opacity-100 [@media(pointer:coarse)]:opacity-100"
				/>
			</span>
		{:else}
			<!-- paused: the state at rest, the resume affordance taking its place on
			     hover, the way the spinner gives way to the pause action above -->
			<span class="relative inline-flex size-3.5 shrink-0 items-center justify-center">
				<Pause
					class="absolute size-icon-sm text-muted-foreground transition-opacity duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:opacity-0 [@media(pointer:coarse)]:hidden"
				/>

				<Play
					class="absolute size-icon-sm opacity-0 transition-opacity duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:opacity-100 [@media(pointer:coarse)]:opacity-100"
				/>
			</span>
		{/if}
	</button>
{/snippet}

{#snippet cancelChip(props: Record<string, unknown>)}
	<!-- cancel slot: same fixed slot and fade as the pause / play affordances
	     so the two icons line up exactly; the X turns destructive on its hover -->
	<button
		{...props}
		aria-label="Cancel downloading"
		class="relative grid size-3.5 shrink-0 cursor-pointer items-center justify-center text-muted-foreground/70"
		onclick={() => onRequestCancel?.(entry.repoWithTag)}
		type="button"
	>
		<X
			class="size-icon-sm transition-[opacity,transform,color] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] text-destructive [@media(pointer:coarse)]:scale-100 [@media(pointer:coarse)]:opacity-100"
		/>
	</button>
{/snippet}

{#snippet downloadChip(props: Record<string, unknown>)}
	<!-- idle chips download on click (retry when the last attempt failed) -->
	<button
		{...props}
		aria-label={tooltipText}
		class="group relative inline-flex h-auto cursor-pointer items-center gap-1 overflow-hidden rounded-md! border px-2 py-1 text-left font-mono text-xs shadow-sm transition-[background-color,border-color,transform] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97]
				border-border/30 bg-background hover:bg-muted-foreground/10 dark:border-border/20 dark:bg-muted-foreground/15 dark:text-secondary-foreground dark:hover:bg-muted-foreground/25
				{entry.isFailed ? 'border-destructive!' : ''}"
		onclick={() => void startDownload()}
		type="button"
	>
		{#if entry.isFailed}
			<span
				class="rounded-md bg-destructive px-1 py-0.5 text-[10px] font-semibold tracking-wide text-destructive-foreground uppercase"
			>
				Failed
			</span>
		{/if}

		{@render chipBody('-my-1 mx-0.75 w-px self-stretch bg-border')}

		<span>{HuggingFaceService.formatFileSize(downloadBytes)}</span>

		<span class="inline-flex size-3.5 shrink-0 items-center justify-center">
			{#if entry.isFailed}
				<RotateCw class="size-icon-sm text-destructive" />
			{:else}
				<Download
					class="size-icon-sm text-muted-foreground transition-colors duration-150 group-hover:text-foreground"
				/>
			{/if}
		</span>
	</button>
{/snippet}

{#if entry.isDownloaded}
	{@render tooltipTrigger(tooltipText, deleteChip)}
{:else if entry.isDownloading || entry.isPaused}
	<!-- in-flight / paused chips: the chip body pauses / resumes on click, the X
	     inside the chip cancels the download. The X slot is reserved, so the chip
	     never reflows when the affordance fades in -->
	<div
		class="group relative inline-flex h-auto items-center gap-1 overflow-hidden rounded-md! border px-2 py-1 text-left font-mono text-xs shadow-sm transition-[background-color,border-color,transform] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97]
			{entry.isPaused
			? 'border-yellow-600/40 bg-yellow-500/10 hover:bg-yellow-500/20 dark:border-yellow-500/30 dark:bg-yellow-500/10'
			: 'border-border/30 bg-background hover:bg-muted-foreground/10 dark:border-border/20 dark:bg-muted-foreground/15'}"
	>
		{@render tooltipTrigger(tooltipText, pauseResumeChip)}

		{@render tooltipTrigger('Cancel downloading', cancelChip)}

		{#if percent !== null}
			<ModelDownloadProgressBar
				downloadedBytes={entry.progress?.downloadedBytes ?? 0}
				overlay
				totalBytes={entry.progress?.totalBytes ?? 0}
			/>
		{/if}
	</div>
{:else}
	{@render tooltipTrigger(tooltipText, downloadChip)}
{/if}
