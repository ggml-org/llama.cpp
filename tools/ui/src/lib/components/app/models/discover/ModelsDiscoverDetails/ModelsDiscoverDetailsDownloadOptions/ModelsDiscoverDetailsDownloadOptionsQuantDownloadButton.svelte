<script lang="ts">
	import DownloadProgressBar from '../../DownloadProgressBar.svelte';
	import { labelFor } from './download-options.utils';
	import { Check, Download, Loader2, Pause, Play, RotateCw, X } from '@lucide/svelte';
	import DialogConfirmation from '$lib/components/app/dialogs/DialogConfirmation.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { HuggingFaceService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type { DownloadEntryState, HfModelSibling } from '$lib/types';

	interface Props {
		/** GGUF file the chip stands for. */
		file: HfModelSibling;
		/** Download state of the file, from the parent's status feed. */
		entry: DownloadEntryState;
	}

	let { entry, file }: Props = $props();

	// delete / cancel confirmation state
	let confirmDeleteOpen = $state(false);
	let confirmCancelOpen = $state(false);

	/** Queue the download; a failed attempt leaves partial files, drop them first. */
	async function startDownload() {
		try {
			if (entry.isFailed) await modelsStore.status.cancelDownload(entry.repoWithTag);

			await modelsStore.status.downloadModel(entry.repoWithTag);
		} catch {
			// the store already toasted the failure
		}
	}

	let meta = $derived(HuggingFaceService.extractQuantMeta(file.path));
	let label = $derived(labelFor(file.path));

	let percent = $derived(
		entry.progress && entry.progress.totalBytes > 0
			? Math.round((entry.progress.downloadedBytes / entry.progress.totalBytes) * 100)
			: null
	);

	let tooltipText = $derived(
		entry.isDownloading
			? 'Pause downloading'
			: entry.isPaused
				? 'Resume downloading'
				: entry.isDownloaded
					? 'Delete model'
					: entry.isFailed
						? `Retry download: ${file.path}`
						: `Download ${file.path}`
	);
</script>

{#if entry.isDownloaded}
	<Tooltip.Root>
		<Tooltip.Trigger>
			<!-- downloaded chips are delete actions: green check by default, red X on hover -->
			<button
				aria-label={tooltipText}
				class="group relative inline-flex h-auto cursor-pointer items-center gap-1 rounded-md! border px-2 py-1 text-left font-mono text-xs shadow-xs transition-[background-color,border-color,transform] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97]
					border-green-600/25 bg-green-500/5 hover:border-destructive/50 hover:bg-destructive/10 dark:border-green-500/30 dark:bg-green-500/10 dark:hover:border-destructive/50 dark:hover:bg-destructive/15"
				onclick={() => (confirmDeleteOpen = true)}
				type="button"
			>
				{#if meta?.sidecar}
					<span
						class="rounded-md bg-primary px-1 py-0.5 text-[10px] font-semibold tracking-wide text-primary-foreground uppercase"
					>
						{meta.sidecar}
					</span>
				{/if}

				<span class="font-medium">{label}</span>

				<span
					class="-my-1 mx-0.75 w-px self-stretch bg-green-600/25 transition-colors duration-200 group-hover:bg-destructive/30 dark:bg-green-600/30 dark:group-hover:bg-destructive/30"
				></span>

				<span>{HuggingFaceService.formatFileSize(file.size ?? 0)}</span>

				<!-- icon slot: crossfade check -> x; touch devices show the delete affordance directly -->
				<span class="relative inline-flex h-3.5 w-3.5 shrink-0 items-center justify-center">
					<Check
						class="absolute h-3.5 w-3.5 text-green-500 transition-[opacity,transform] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:scale-75 group-hover:opacity-0 [@media(pointer:coarse)]:hidden"
					/>

					<X
						class="absolute h-3.5 w-3.5 scale-75 text-destructive opacity-0 transition-[opacity,transform] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:scale-100 group-hover:opacity-100 [@media(pointer:coarse)]:scale-100 [@media(pointer:coarse)]:opacity-100"
					/>
				</span>
			</button>
		</Tooltip.Trigger>

		<Tooltip.Content>
			<p>{tooltipText}</p>
		</Tooltip.Content>
	</Tooltip.Root>

	<DialogConfirmation
		confirmText="Delete"
		description={`This permanently removes ${modelsStore.toDisplayName(entry.repoWithTag)} from disk. You can download it again later.`}
		onCancel={() => (confirmDeleteOpen = false)}
		onConfirm={() => {
			confirmDeleteOpen = false;

			void modelsStore.status.cancelDownload(entry.repoWithTag);
		}}
		open={confirmDeleteOpen}
		title="Delete model"
		variant="destructive"
	/>
{:else if entry.isDownloading || entry.isPaused}
	<!-- in-flight / paused chips: the chip body pauses / resumes on click, the X
	     inside the chip cancels (stops and discards the partial files). The X slot
	     is reserved, so the chip never reflows when the affordance fades in -->
	<div
		class="group relative inline-flex h-auto items-center gap-1 overflow-hidden rounded-md! border px-2 py-1 text-left font-mono text-xs shadow-xs transition-[background-color,border-color,transform] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97]
			{entry.isPaused
			? 'border-yellow-600/40 bg-yellow-500/10 hover:bg-yellow-500/20 dark:border-yellow-500/30 dark:bg-yellow-500/10'
			: 'border-border/30 bg-background hover:bg-muted-foreground/10 dark:border-border/20 dark:bg-muted-foreground/15'}"
	>
		<Tooltip.Root>
			<Tooltip.Trigger>
				<button
					aria-label={tooltipText}
					class="flex h-auto min-w-0 flex-1 cursor-pointer items-center gap-1 text-left"
					onclick={() => {
						if (entry.isDownloading) void modelsStore.status.pauseDownload(entry.repoWithTag);
						else void modelsStore.status.downloadModel(entry.repoWithTag).catch(() => {});
					}}
					type="button"
				>
					{#if meta?.sidecar}
						<span
							class="rounded-md bg-primary px-1 py-0.5 text-[10px] font-semibold tracking-wide text-primary-foreground uppercase"
						>
							{meta.sidecar}
						</span>
					{/if}

					<span class="font-medium">{label}</span>

					<span class="-my-1 mx-0.75 w-px self-stretch bg-border"></span>

					{#if percent !== null}
						<span class="mr-1 tabular-nums">{percent}%</span>
					{:else if entry.isPaused}
						<span>Paused</span>
					{:else}
						<span>{HuggingFaceService.formatFileSize(file.size ?? 0)}</span>
					{/if}

					{#if entry.isDownloading}
						<!-- spinner fades into the pause affordance on hover; opacity only, the
							 spin keyframes own the transform so scale would fight them -->
						<span class="relative inline-flex h-3.5 w-3.5 shrink-0 items-center justify-center">
							<Loader2
								class="absolute h-3.5 w-3.5 animate-spin text-muted-foreground transition-opacity duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:opacity-0 [@media(pointer:coarse)]:hidden"
							/>

							<Pause
								class="absolute h-3.5 w-3.5 opacity-0 transition-opacity duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:opacity-100 [@media(pointer:coarse)]:opacity-100"
							/>
						</span>
					{:else}
						<!-- paused: the play affordance fades in on hover; visible directly on touch -->
						<span class="relative inline-flex h-3.5 w-3.5 shrink-0 items-center justify-center">
							<Play
								class="absolute h-3.5 w-3.5 scale-75 opacity-0 transition-[opacity,transform] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:scale-100 group-hover:opacity-100 [@media(pointer:coarse)]:scale-100 [@media(pointer:coarse)]:opacity-100"
							/>
						</span>
					{/if}
				</button>
			</Tooltip.Trigger>

			<Tooltip.Content>
				<p>{tooltipText}</p>
			</Tooltip.Content>
		</Tooltip.Root>

		<Tooltip.Root>
			<Tooltip.Trigger class="grid">
				<!-- cancel slot: same fixed slot and fade as the pause / play affordances
					 so the two icons line up exactly; the X turns destructive on its hover -->
				<button
					aria-label="Cancel downloading"
					class="relative inline-flex h-3.5 w-3.5 shrink-0 cursor-pointer items-center justify-center text-muted-foreground/70"
					onclick={() => (confirmCancelOpen = true)}
					type="button"
				>
					<X
						class="h-3.5 w-3.5 transition-[opacity,transform,color] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] text-destructive [@media(pointer:coarse)]:scale-100 [@media(pointer:coarse)]:opacity-100"
					/>
				</button>
			</Tooltip.Trigger>

			<Tooltip.Content>
				<p>Cancel downloading</p>
			</Tooltip.Content>
		</Tooltip.Root>

		{#if percent !== null}
			<DownloadProgressBar
				downloadedBytes={entry.progress?.downloadedBytes ?? 0}
				overlay
				totalBytes={entry.progress?.totalBytes ?? 0}
			/>
		{/if}
	</div>
{:else}
	<Tooltip.Root>
		<Tooltip.Trigger>
			<!-- idle chips download on click (retry when the last attempt failed) -->
			<button
				aria-label={tooltipText}
				class="group relative inline-flex h-auto cursor-pointer items-center gap-1 overflow-hidden rounded-md! border px-2 py-1 text-left font-mono text-xs shadow-xs transition-[background-color,border-color,transform] duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] active:scale-[0.97]
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

				{#if meta?.sidecar}
					<span
						class="rounded-md bg-primary px-1 py-0.5 text-[10px] font-semibold tracking-wide text-primary-foreground uppercase"
					>
						{meta.sidecar}
					</span>
				{/if}

				<span class="font-medium">{label}</span>

				<span class="-my-1 mx-0.75 w-px self-stretch bg-border"></span>

				<span>{HuggingFaceService.formatFileSize(file.size ?? 0)}</span>

				<span class="inline-flex h-3.5 w-3.5 shrink-0 items-center justify-center">
					{#if entry.isFailed}
						<RotateCw class="h-3.5 w-3.5 text-destructive" />
					{:else}
						<Download
							class="h-3.5 w-3.5 text-muted-foreground transition-colors duration-150 group-hover:text-foreground"
						/>
					{/if}
				</span>
			</button>
		</Tooltip.Trigger>

		<Tooltip.Content>
			<p>{tooltipText}</p>
		</Tooltip.Content>
	</Tooltip.Root>
{/if}

<DialogConfirmation
	cancelText="Keep downloading"
	confirmText="Cancel download"
	description={`This stops the download of ${modelsStore.toDisplayName(entry.repoWithTag)} and removes the partial files. Pause it instead to keep the progress.`}
	onCancel={() => (confirmCancelOpen = false)}
	onConfirm={() => {
		confirmCancelOpen = false;

		void modelsStore.status.cancelDownload(entry.repoWithTag);
	}}
	open={confirmCancelOpen}
	title="Cancel download"
	variant="destructive"
/>
