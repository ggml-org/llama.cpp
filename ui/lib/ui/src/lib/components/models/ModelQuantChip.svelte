<!--
	One quant of a model as a chip: its label, any draft riding with it, its size and the action its state allows.
	The props-in version of ui/apps/chat's ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton: the app passes the state
	and gets each action back as a callback; it never downloads anything itself. An idle chip with `href` is a link,
	for pages that hand the install to another app (llama.app's Install, a deep link).
-->
<script lang="ts">
	import ModelDownloadProgressBar from './ModelDownloadProgressBar.svelte';
	import { Check, Download, Loader2, Pause, Play, RotateCw, X } from '@lucide/svelte';
	import * as Tooltip from '../ui/tooltip';
	import { SIDECAR_BADGE_CLASS } from '../../models/constants';
	import { formatFileSize } from '../../models/format';
	import type {
		ModelDownloadProgress,
		ModelQuantCompanion,
		ModelQuantState
	} from '../../models/types';
	import type { Snippet } from 'svelte';

	interface Props {
		/** The quant, e.g. `Q4_K_M`. */
		label: string;
		/** The quant file's size in bytes. */
		sizeBytes?: number;
		state?: ModelQuantState;
		/** Live while downloading, frozen while paused. */
		progress?: ModelDownloadProgress | null;
		/** A draft model pulled with this quant, e.g. an MTP head; its size counts in the chip's. */
		companion?: ModelQuantCompanion | null;
		/** Makes an idle chip a link instead of a download button. */
		href?: string;
		/** Overrides the idle chip's tooltip and label, e.g. `Install Q4_K_M`. */
		actionLabel?: string;
		onDownload?: () => void;
		onPause?: () => void;
		onResume?: () => void;
		/** Ask the app to confirm cancelling; the chip owns no dialog. */
		onCancel?: () => void;
		/** Ask the app to confirm deleting; the chip owns no dialog. */
		onDelete?: () => void;
	}

	let {
		actionLabel,
		companion = null,
		href,
		label,
		onCancel,
		onDelete,
		onDownload,
		onPause,
		onResume,
		progress = null,
		sizeBytes = 0,
		state = 'idle'
	}: Props = $props();

	let isDownloaded = $derived(state === 'downloaded');
	let isDownloading = $derived(state === 'downloading');
	let isPaused = $derived(state === 'paused');
	let isFailed = $derived(state === 'failed');

	let percent = $derived(
		progress && progress.totalBytes > 0
			? Math.round((progress.downloadedBytes / progress.totalBytes) * 100)
			: null
	);

	/**
	 * The joined draft reads once it rides along or lands: a downloaded model
	 * missing its draft keeps to its own weights, and the draft gets its own chip.
	 */
	let showsCompanion = $derived(companion ? (companion.onDisk ?? false) || !isDownloaded : false);

	/** What the chip's actions name: the quant and, when it rides along, its draft. */
	let downloadLabel = $derived(
		showsCompanion && companion
			? `${label} + ${companion.kind.toUpperCase()}${companion.quant ? `:${companion.quant}` : ''}`
			: label
	);

	let tooltipText = $derived(
		isDownloading
			? 'Pause downloading'
			: isPaused
				? 'Resume downloading'
				: isDownloaded
					? `Delete ${downloadLabel}`
					: isFailed
						? `Retry download: ${downloadLabel}`
						: (actionLabel ?? `Download ${downloadLabel}`)
	);

	let downloadBytes = $derived(
		(sizeBytes ?? 0) + (showsCompanion && companion ? (companion.sizeBytes ?? 0) : 0)
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
		class="group relative inline-flex h-auto cursor-pointer items-center gap-1 rounded-md! border px-2 py-1 text-left font-mono text-xs shadow-(--elevation-control) transition-[background-color,border-color,transform] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97]
					border-green-600/25 bg-green-500/5 hover:border-destructive/50 hover:bg-destructive/10 dark:border-green-500/30 dark:bg-green-500/10 dark:hover:border-destructive/50 dark:hover:bg-destructive/15"
		onclick={() => onDelete?.()}
		type="button"
	>
		{@render chipBody(
			'-my-1 mx-0.75 w-px self-stretch bg-green-600/25 transition-colors duration-200 group-hover:bg-destructive/30 dark:bg-green-600/30 dark:group-hover:bg-destructive/30'
		)}

		<span>{formatFileSize(downloadBytes)}</span>

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
		onclick={() => (isDownloading ? onPause?.() : onResume?.())}
		type="button"
	>
		{@render chipBody('-my-1 mx-0.75 w-px self-stretch bg-border')}

		{#if percent !== null}
			<span class="mr-1 tabular-nums">{percent}%</span>
		{:else if isPaused}
			<span>Paused</span>
		{:else}
			<span>{formatFileSize(downloadBytes)}</span>
		{/if}

		{#if isDownloading}
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
		onclick={() => onCancel?.()}
		type="button"
	>
		<X
			class="size-icon-sm transition-[opacity,transform,color] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] text-destructive [@media(pointer:coarse)]:scale-100 [@media(pointer:coarse)]:opacity-100"
		/>
	</button>
{/snippet}

{#snippet idleChipContent()}
	{#if isFailed}
		<span
			class="rounded-md bg-(--badge-danger-background) px-1 py-0.5 text-[10px] font-semibold tracking-wide text-(--badge-danger-text) uppercase"
		>
			Failed
		</span>
	{/if}

	{@render chipBody('-my-1 mx-0.75 w-px self-stretch bg-border')}

	<span>{formatFileSize(downloadBytes)}</span>

	<span class="inline-flex size-3.5 shrink-0 items-center justify-center">
		{#if isFailed}
			<RotateCw class="size-icon-sm text-destructive" />
		{:else}
			<Download
				class="size-icon-sm text-muted-foreground transition-colors duration-150 group-hover:text-foreground"
			/>
		{/if}
	</span>
{/snippet}

{#snippet downloadChip(props: Record<string, unknown>)}
	{@const chipClass = `group relative inline-flex h-auto cursor-pointer items-center gap-1 overflow-hidden rounded-md! border px-2 py-1 text-left font-mono text-xs shadow-(--elevation-control) transition-[background-color,border-color,transform] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97]
				border-border/30 bg-background hover:bg-muted-foreground/10 dark:border-border/20 dark:bg-muted-foreground/15 dark:text-secondary-foreground dark:hover:bg-muted-foreground/25
				${isFailed ? 'border-destructive!' : ''}`}

	<!-- idle chips download on click (retry when the last attempt failed), or follow their link -->
	{#if href && !isFailed}
		<a {...props} aria-label={tooltipText} class={chipClass} {href} onclick={() => onDownload?.()}>
			{@render idleChipContent()}
		</a>
	{:else}
		<button
			{...props}
			aria-label={tooltipText}
			class={chipClass}
			onclick={() => onDownload?.()}
			type="button"
		>
			{@render idleChipContent()}
		</button>
	{/if}
{/snippet}

{#if isDownloaded}
	{@render tooltipTrigger(tooltipText, deleteChip)}
{:else if isDownloading || isPaused}
	<!-- in-flight / paused chips: the chip body pauses / resumes on click, the X
	     inside the chip cancels the download. The X slot is reserved, so the chip
	     never reflows when the affordance fades in -->
	<div
		class="group relative inline-flex h-auto items-center gap-1 overflow-hidden rounded-md! border px-2 py-1 text-left font-mono text-xs shadow-(--elevation-control) transition-[background-color,border-color,transform] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97]
			{isPaused
			? 'border-yellow-600/40 bg-yellow-500/10 hover:bg-yellow-500/20 dark:border-yellow-500/30 dark:bg-yellow-500/10'
			: 'border-border/30 bg-background hover:bg-muted-foreground/10 dark:border-border/20 dark:bg-muted-foreground/15'}"
	>
		{@render tooltipTrigger(tooltipText, pauseResumeChip)}

		{@render tooltipTrigger('Cancel downloading', cancelChip)}

		{#if percent !== null}
			<ModelDownloadProgressBar
				downloadedBytes={progress?.downloadedBytes ?? 0}
				overlay
				totalBytes={progress?.totalBytes ?? 0}
			/>
		{/if}
	</div>
{:else}
	{@render tooltipTrigger(tooltipText, downloadChip)}
{/if}
