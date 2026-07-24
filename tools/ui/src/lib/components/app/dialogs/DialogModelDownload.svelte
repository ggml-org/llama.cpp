<script lang="ts">
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Download, Loader2, Trash2, TriangleAlert } from '@lucide/svelte';
	import { KeyboardKey } from '$lib/enums';
	import { DialogConfirmation } from '$lib/components/app';
	import { ModelsService, type GgufVariantTagInput } from '$lib/services/models.service';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { DownloadProgressBar } from '$lib/components/app/models';
	import type { DraftVariant } from '$lib/constants/model-id';

	interface Props {
		open: boolean;
		repoId: string;
		filePath: string;
		quant: string | null;
		variant: DraftVariant | null;
		formattedSize?: string;
		onConfirm: () => void;
		onCancel: () => void;
	}

	let {
		open = $bindable(),
		repoId,
		filePath,
		quant,
		variant,
		formattedSize,
		onConfirm,
		onCancel
	}: Props = $props();

	type Phase = 'pending' | 'starting' | 'downloading' | 'finished';
	let phase = $state<Phase>('pending');
	let hasSeenProgress = $state(false);
	let lastError: string | null = $state(null);

	let tagInput = $derived<GgufVariantTagInput | null>(
		quant || variant ? { quant: quant ?? '', variant } : null
	);
	let hfRepoWithTag = $derived(ModelsService.buildDownloadTag(repoId, tagInput));
	let tagDisplay = $derived.by(() => {
		if (quant && variant) return `${quant}-${variant.toUpperCase()}`;
		if (quant) return quant;
		if (variant) return variant.toUpperCase();
		return 'default';
	});

	let inFlight = $derived(phase === 'starting' || phase === 'downloading');
	// True when the previous SSE `download_failed` event left a recorded
	// failure for the same <repo>:<tag>. The dialog swaps the Download button
	// for a Delete-&-retry flow because POST /models rejects already-existing
	// partial entries.
	let previousFailure = $derived(modelsStore.hasFailedDownload(hfRepoWithTag));
	let cancelling = $state(false);
	let lastCancelError: string | null = $state(null);

	// Whether the Delete button makes sense. Only show it when the model is
	// registered with the server (i.e. it's a fully downloaded entry that
	// shows up in /v1/models) — the dialog otherwise represents an in-flight
	// download whose partial files are cleaned up automatically by the Retry
	// path, so an explicit delete would just be redundant noise.
	let canDelete = $derived(phase === 'finished' && modelsStore.isModelDownloaded(hfRepoWithTag));
	let showDeleteConfirm = $state(false);
	async function handleConfirmDelete() {
		showDeleteConfirm = false;
		await modelsStore.cancelDownload(hfRepoWithTag);
		// Close the download dialog too — cancelling the underlying entry
		// makes the rest of the wizard irrelevant.
		onCancel();
	}

	// Reactive: while the SSE feed reports progress for our download, surface it.
	// The downloadProgress map is deleted on download_finished/download_failed.
	let progress = $derived(modelsStore.getDownloadProgress(hfRepoWithTag));
	let progressPercent = $derived.by(() => {
		if (!progress || progress.totalBytes <= 0) return 0;
		return Math.round((progress.downloadedBytes / progress.totalBytes) * 100);
	});

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === KeyboardKey.ENTER && phase === 'pending') {
			event.preventDefault();
			void trigger();
		}
	}

	function handleOpenChange(newOpen: boolean) {
		if (newOpen) {
			lastError = null;
			lastCancelError = null;
			showDeleteConfirm = false;
			phase = 'pending';
			hasSeenProgress = false;
			return;
		}
		if (!inFlight) onCancel();
	}

	async function trigger() {
		if (inFlight) return;
		phase = 'starting';
		hasSeenProgress = false;
		lastError = null;
		lastCancelError = null;
		showDeleteConfirm = false;
		// Detect a previously-recorded failed attempt for the same <repo>:<tag>.
		// The server's POST /models rejects already-existing entries, so we
		// must remove the partial cached entry first before we can POST again.
		if (modelsStore.hasFailedDownload(hfRepoWithTag)) {
			await ModelsService.cancelDownload(hfRepoWithTag);
		}
		try {
			await modelsStore.downloadModel(hfRepoWithTag, filePath);
			phase = 'downloading';
		} catch (error) {
			lastError = error instanceof Error ? error.message : 'Failed to start download';
			phase = 'pending';
		}
	}

	async function cancel() {
		if (cancelling) return;
		cancelling = true;
		lastCancelError = null;
		try {
			const ok = await modelsStore.cancelDownload(hfRepoWithTag);
			if (!ok) {
				lastCancelError = 'Cancel request failed. Try again in a moment.';
			}
		} finally {
			cancelling = false;
		}
	}

	// Latch once we've seen real progress so we know what 'no longer in flight'
	// actually means. Without this the dialog auto-closed 600ms after the POST
	// resolved because no SSE event had landed yet, leaving the user staring at
	// an unmounted dialog during the actual download.
	$effect(() => {
		if (phase !== 'downloading') return;
		if (progress) hasSeenProgress = true;
	});

	// Promote to 'finished' only after progress was actually observed and the
	// SSE feed subsequently drops our entry. Auto-close after a short pause.
	$effect(() => {
		if (phase !== 'downloading') return;
		if (!hasSeenProgress) return;
		const stillInFlight = modelsStore.isDownloadInProgress(hfRepoWithTag);
		if (!stillInFlight) {
			phase = 'finished';
			const timer = setTimeout(() => onConfirm(), 600);
			return () => clearTimeout(timer);
		}
	});
</script>

<AlertDialog.Root {open} onOpenChange={handleOpenChange}>
	<AlertDialog.Content onkeydown={handleKeydown} class="max-w-md">
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<Download class="h-5 w-5 text-primary" />
				{#if phase === 'pending'}
					Download this model?
				{:else}
					Downloading {tagDisplay}
				{/if}
			</AlertDialog.Title>
			<AlertDialog.Description>
				{#if phase === 'pending'}
					llama-server will download this file (and related sidecar weights such as multimodal
					projectors or draft models) from Hugging Face into your local model cache.
				{:else}
					Download runs in the background; this dialog tracks live progress.
				{/if}
			</AlertDialog.Description>

			{#if previousFailure && phase === 'pending'}
				<div
					class="mt-2 flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/5 p-2 text-xs text-destructive"
					role="status"
				>
					<TriangleAlert class="mt-0.5 h-4 w-4 shrink-0" />
					<span>
						A previous attempt for this tag failed and left partial files on disk. The server will
						reject a fresh download until those files are removed. The Retry button below deletes
						the partial files automatically.
					</span>
				</div>
			{/if}
		</AlertDialog.Header>

		{#if canDelete}
			<div class="flex justify-end">
				<button
					type="button"
					onclick={() => (showDeleteConfirm = true)}
					class="inline-flex items-center gap-1.5 rounded-md border border-destructive/40 px-2 py-1 text-xs font-medium text-destructive transition-colors hover:bg-destructive/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-destructive/50"
					aria-label="Delete model from cache"
				>
					<Trash2 class="h-3.5 w-3.5" />
					Delete from cache
				</button>
			</div>
		{/if}

		<div class="space-y-3 rounded-md border bg-muted/40 p-3 text-xs">
			<div class="flex flex-col gap-1">
				<span class="text-muted-foreground">Request</span>
				<code class="break-all font-mono"
					>POST /models&nbsp;·&nbsp;{`{ model: "${hfRepoWithTag}" }`}</code
				>
			</div>
			<div class="flex flex-col gap-1">
				<span class="text-muted-foreground">File</span>
				<code class="break-all font-mono">{filePath}</code>
			</div>
			<div class="flex flex-wrap items-center gap-2">
				<span class="rounded bg-primary/15 px-2 py-0.5 font-mono font-semibold text-primary">
					{tagDisplay}
				</span>
				{#if formattedSize}
					<span class="text-muted-foreground">{formattedSize}</span>
				{/if}
				{#if variant}
					<span
						class="rounded bg-primary px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary-foreground"
					>
						{variant}
					</span>
				{/if}
			</div>

			{#if phase === 'downloading' || phase === 'finished'}
				<div class="flex flex-col gap-1.5">
					<div class="flex items-center justify-between text-muted-foreground">
						<span>
							{#if phase === 'finished'}
								Complete
							{:else if progress && progress.totalBytes > 0}
								Downloading
							{:else}
								Preparing download
							{/if}
						</span>
						<span class="font-mono tabular-nums">{progressPercent}%</span>
					</div>
					<DownloadProgressBar
						downloadedBytes={progress?.downloadedBytes ?? 0}
						totalBytes={progress?.totalBytes ?? 0}
					/>
				</div>
			{/if}
		</div>

		{#if lastError}
			<p class="text-xs text-destructive">{lastError}</p>
		{/if}
		{#if lastCancelError}
			<p class="text-xs text-destructive">{lastCancelError}</p>
		{/if}

		<AlertDialog.Footer>
			{#if phase === 'downloading'}
				<AlertDialog.Action disabled={cancelling} onclick={cancel}>
					{#if cancelling}
						<Loader2 class="mr-1.5 h-4 w-4 animate-spin" />
						Cancelling...
					{:else}
						Cancel download
					{/if}
				</AlertDialog.Action>
			{:else}
				<AlertDialog.Cancel disabled={inFlight} onclick={onCancel}>
					{#if phase === 'finished'}Close{:else}Cancel{/if}
				</AlertDialog.Cancel>
			{/if}
			{#if phase === 'pending'}
				<AlertDialog.Action disabled={inFlight} onclick={trigger}>
					<Download class="mr-1.5 h-4 w-4" />
					{previousFailure ? 'Retry download' : 'Download'}
				</AlertDialog.Action>
			{:else if phase === 'starting'}
				<AlertDialog.Action disabled>
					<Loader2 class="mr-1.5 h-4 w-4 animate-spin" />
					Starting...
				</AlertDialog.Action>
			{/if}
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>

<DialogConfirmation
	bind:open={showDeleteConfirm}
	title="Delete model"
	description={`Remove "${hfRepoWithTag}" from your cache? Any cached files will be deleted from disk.`}
	confirmText="Delete"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleConfirmDelete}
	onCancel={() => (showDeleteConfirm = false)}
/>
