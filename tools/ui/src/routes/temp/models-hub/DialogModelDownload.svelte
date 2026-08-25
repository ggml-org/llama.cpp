<script lang="ts">
	import DownloadToast from './DownloadToast.svelte';
	import { Download, LoaderCircle, TriangleAlert } from '@lucide/svelte';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import type { DraftVariant } from '$lib/constants';
	import { KeyboardKey } from '$lib/enums';
	import { type GgufVariantTagInput, ModelsService } from '$lib/services/models.service';
	import { modelsStore } from '$lib/stores';
	import { toast } from 'svelte-sonner';

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
		filePath,
		formattedSize,
		onCancel,
		onConfirm,
		open = $bindable(),
		quant,
		repoId,
		variant
	}: Props = $props();

	type Phase = 'pending' | 'starting';
	let phase = $state<Phase>('pending');
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

	// True when a previous SSE `download_failed` left a recorded failure for the
	// same <repo>:<tag>. The dialog swaps Download for a delete-&-retry flow
	// because POST /models rejects already-existing partial entries.
	let previousFailure = $derived(modelsStore.status.hasFailedDownload(hfRepoWithTag));

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === KeyboardKey.ENTER && phase === 'pending') {
			event.preventDefault();
			void trigger();
		}
	}

	// The dialog is always closable - downloads run in the background and are
	// tracked by a toast, so closing mid-flight never aborts the download.
	function handleOpenChange(newOpen: boolean) {
		if (newOpen) {
			lastError = null;
			phase = 'pending';

			return;
		}

		onCancel();
	}

	async function trigger() {
		if (phase === 'starting') return;

		phase = 'starting';
		lastError = null;

		// A recorded failure for the same <repo>:<tag> means the server still holds
		// a partial entry that POST /models would reject; remove it before retrying.
		if (modelsStore.status.hasFailedDownload(hfRepoWithTag)) {
			await ModelsService.cancelDownload(hfRepoWithTag);
		}

		try {
			await modelsStore.status.downloadModel(hfRepoWithTag, filePath);
			// Download runs on the server; hand progress off to a toast and close.
			showDownloadToast();
			onConfirm();
		} catch (error) {
			lastError = error instanceof Error ? error.message : 'Failed to start download';
			phase = 'pending';
		}
	}

	function showDownloadToast() {
		toast.custom(DownloadToast, {
			componentProps: {
				displayName: filePath,
				repoId,
				repoWithTag: hfRepoWithTag
			},
			dismissible: true,
			duration: Infinity
		});
	}
</script>

<AlertDialog.Root onOpenChange={handleOpenChange} {open}>
	<AlertDialog.Content class="max-w-md" onkeydown={handleKeydown}>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<Download class="h-5 w-5 text-primary" />
				Download this model?
			</AlertDialog.Title>

			<AlertDialog.Description>
				llama-server will download this file (and related sidecar weights such as multimodal
				projectors or draft models) from Hugging Face into your local model cache. The download runs
				in the background and its progress is shown in a notification.
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

				{#if variant}
					<span
						class="rounded bg-primary px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary-foreground"
					>
						{variant}
					</span>
				{/if}
			</div>
		</div>

		{#if lastError}
			<p class="text-xs text-destructive">{lastError}</p>
		{/if}

		<AlertDialog.Footer>
			<AlertDialog.Cancel disabled={phase === 'starting'} onclick={onCancel}>
				Cancel
			</AlertDialog.Cancel>

			<AlertDialog.Action disabled={phase === 'starting'} onclick={trigger}>
				{#if phase === 'starting'}
					<LoaderCircle class="mr-1.5 h-4 w-4 animate-spin" />
					Starting...
				{:else}
					<Download class="mr-1.5 h-4 w-4" />
					{previousFailure ? 'Retry download' : 'Download'}
				{/if}
			</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
