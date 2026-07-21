<script lang="ts">
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Download, Loader2 } from '@lucide/svelte';
	import { KeyboardKey } from '$lib/enums';
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
		try {
			await modelsStore.downloadModel(hfRepoWithTag, filePath);
			phase = 'downloading';
		} catch (error) {
			lastError = error instanceof Error ? error.message : 'Failed to start download';
			phase = 'pending';
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
		</AlertDialog.Header>

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

		<AlertDialog.Footer>
			<AlertDialog.Cancel disabled={inFlight} onclick={onCancel}>
				{#if phase === 'finished'}Close{:else}Cancel{/if}
			</AlertDialog.Cancel>
			{#if phase === 'pending'}
				<AlertDialog.Action disabled={inFlight} onclick={trigger}>
					<Download class="mr-1.5 h-4 w-4" />
					Download
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
