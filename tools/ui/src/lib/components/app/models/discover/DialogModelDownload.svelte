<script lang="ts">
	import DownloadProgressBar from './DownloadProgressBar.svelte';
	import { Download, LoaderCircle, Trash2, TriangleAlert } from '@lucide/svelte';
	import { DialogConfirmation } from '$lib/components/app';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { isAuxSidecar, type ModelSidecar } from '$lib/constants';
	import { KeyboardKey } from '$lib/enums';
	import { ModelsService } from '$lib/services';
	import type { ModelDownloadProgress } from '$lib/types';

	interface Props {
		open: boolean;
		/** Full HuggingFace repo id, e.g. `ggml-org/gemma-3-4b-it-GGUF`. */
		repoId: string;
		/** Repo-relative path of the file this download resolves to. */
		filePath: string;
		/** Quantization token of the selected file, when known. */
		quant: string | null;
		/** Sidecar type pulled alongside the main weights, when any. */
		sidecar: ModelSidecar | null;
		/** Human-readable size of the download, when known. */
		formattedSize?: string;
		/** True when a previous attempt for this tag failed and left partial files. */
		previousFailure?: boolean;
		/** True while the server reports this download as in flight. */
		inFlight?: boolean;
		/** Live progress from the /models/sse feed; null before the first event. */
		progress?: ModelDownloadProgress | null;
		/** True when the model is fully downloaded and registered with the server. */
		isDownloaded?: boolean;
		/** Error message from a failed start attempt, shown above the footer. */
		error?: string | null;
		/** Fire the download (POST /models). */
		onDownload: () => void;
		/** Cancel the in-flight download (DELETE /models). */
		onCancelDownload?: () => void;
		/** Delete the model from the server cache; offered once finished. */
		onDelete?: () => void;
		/** Dialog was dismissed or the download completed. */
		onClose: () => void;
	}

	let {
		error = null,
		filePath,
		formattedSize,
		inFlight = false,
		isDownloaded = false,
		onCancelDownload,
		onClose,
		onDelete,
		onDownload,
		open = $bindable(false),
		previousFailure = false,
		progress = null,
		quant,
		repoId,
		sidecar
	}: Props = $props();

	let started = $state(false);
	let sawProgress = $state(false);
	let cancelling = $state(false);
	let showDeleteConfirm = $state(false);

	let hfRepoWithTag = $derived(ModelsService.buildDownloadTag(repoId, quant, sidecar));

	let tagDisplay = $derived.by(() => {
		if (quant && sidecar) return `${quant}-${sidecar.toUpperCase()}`;

		if (quant) return quant;

		if (sidecar) return sidecar.toUpperCase();

		return 'default';
	});

	let phase = $derived(
		started && inFlight ? 'downloading' : started && sawProgress ? 'finished' : 'confirm'
	);

	let progressPercent = $derived.by(() => {
		if (!progress || progress.totalBytes <= 0) return 0;

		return Math.round((progress.downloadedBytes / progress.totalBytes) * 100);
	});

	// Delete is offered once the download completed and the model is registered.
	let canDelete = $derived(phase === 'finished' && isDownloaded);

	function reset() {
		started = false;
		sawProgress = false;
		cancelling = false;
		showDeleteConfirm = false;
	}

	function handleOpenChange(next: boolean) {
		if (next) {
			reset();

			return;
		}

		if (!inFlight) onClose();
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === KeyboardKey.ENTER && phase === 'confirm') {
			event.preventDefault();
			start();
		}
	}

	function start() {
		if (inFlight) return;

		started = true;
		sawProgress = false;

		onDownload();
	}

	async function cancel() {
		if (cancelling || !onCancelDownload) return;

		cancelling = true;

		try {
			onCancelDownload();
		} finally {
			cancelling = false;
		}
	}

	function handleDelete() {
		showDeleteConfirm = false;

		onDelete?.();
		onClose();
	}

	// Latch progress: 'in-flight ending' only means finished once the feed has
	// reported progress; otherwise the POST resolving alone proves nothing.
	$effect(() => {
		if (inFlight && progress) sawProgress = true;
	});

	// Auto-close shortly after the download completes.
	$effect(() => {
		if (phase !== 'finished') return;

		const timer = setTimeout(() => onClose(), 600);

		return () => clearTimeout(timer);
	});
</script>

<AlertDialog.Root onOpenChange={handleOpenChange} {open}>
	<AlertDialog.Content class="max-w-md" onkeydown={handleKeydown}>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<Download class="h-5 w-5 text-primary" />

				{#if phase === 'confirm'}
					Download this model?
				{:else}
					Downloading {tagDisplay}
				{/if}
			</AlertDialog.Title>

			<AlertDialog.Description>
				{#if phase === 'confirm'}
					llama-server will download this file (and related sidecar weights such as multimodal
					projectors or draft models) from Hugging Face into your local model cache.
				{:else}
					Download runs in the background; this dialog tracks live progress.
				{/if}
			</AlertDialog.Description>

			{#if previousFailure && phase === 'confirm'}
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
					aria-label="Delete model from cache"
					class="inline-flex items-center gap-1.5 rounded-md border border-destructive/40 px-2 py-1 text-xs font-medium text-destructive transition-colors hover:bg-destructive/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-destructive/50"
					onclick={() => (showDeleteConfirm = true)}
					type="button"
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
					>POST /models&nbsp;&middot;&nbsp;{`{ model: "${hfRepoWithTag}" }`}</code
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

				{#if sidecar && !isAuxSidecar(sidecar)}
					<span
						class="rounded bg-primary px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary-foreground"
					>
						{sidecar}
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

		{#if error}
			<p class="text-xs text-destructive">{error}</p>
		{/if}

		<AlertDialog.Footer>
			{#if phase === 'downloading'}
				<AlertDialog.Action disabled={cancelling} onclick={cancel}>
					{#if cancelling}
						<LoaderCircle class="mr-1.5 h-4 w-4 animate-spin" />
						Cancelling...
					{:else}
						Cancel download
					{/if}
				</AlertDialog.Action>
			{:else}
				<AlertDialog.Cancel disabled={inFlight} onclick={() => onClose()}>
					{#if phase === 'finished'}Close{:else}Cancel{/if}
				</AlertDialog.Cancel>
			{/if}

			{#if phase === 'confirm'}
				<AlertDialog.Action disabled={inFlight} onclick={start}>
					<Download class="mr-1.5 h-4 w-4" />
					{previousFailure ? 'Retry download' : 'Download'}
				</AlertDialog.Action>
			{/if}
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>

<DialogConfirmation
	bind:open={showDeleteConfirm}
	cancelText="Cancel"
	confirmText="Delete"
	description={`Remove "${hfRepoWithTag}" from your cache? Any cached files will be deleted from disk.`}
	icon={Trash2}
	onCancel={() => (showDeleteConfirm = false)}
	onConfirm={handleDelete}
	title="Delete model"
	variant="destructive"
/>
