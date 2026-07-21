<script lang="ts">
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Download, Loader2 } from '@lucide/svelte';
	import { KeyboardKey } from '$lib/enums';
	import { ModelsService, type GgufVariantTagInput } from '$lib/services/models.service';
	import { modelsStore } from '$lib/stores/models.svelte';
	import type { DraftVariant } from '$lib/constants/model-id';

	interface Props {
		open: boolean;
		repoId: string;
		filePath: string;
		quant: string | null;
		variant: DraftVariant | null;
		sizeBytes: number | null;
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
		sizeBytes,
		formattedSize,
		onConfirm,
		onCancel
	}: Props = $props();

	let submitting = $state(false);
	let lastError: string | null = $state(null);

	let tagDisplay = $derived.by(() => {
		if (quant && variant) return `${quant}-${variant.toUpperCase()}`;
		if (quant) return quant;
		if (variant) return variant.toUpperCase();
		return 'default';
	});

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === KeyboardKey.ENTER) {
			event.preventDefault();
			void trigger();
		}
	}

	function handleOpenChange(newOpen: boolean) {
		if (newOpen) {
			lastError = null;
			return;
		}
		if (!submitting) onCancel();
	}

	async function trigger() {
		if (submitting) return;
		submitting = true;
		lastError = null;
		try {
			await modelsStore.downloadModel(repoId, filePath);
			onConfirm();
		} catch (error) {
			lastError = error instanceof Error ? error.message : 'Failed to start download';
		} finally {
			submitting = false;
		}
	}
</script>

<AlertDialog.Root {open} onOpenChange={handleOpenChange}>
	<AlertDialog.Content onkeydown={handleKeydown} class="max-w-md">
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<Download class="h-5 w-5 text-primary" />
				Download this model?
			</AlertDialog.Title>
			<AlertDialog.Description>
				llama-server will download this file (and related sidecar weights such as multimodal
				projectors or draft models) from Hugging Face into your local model cache.
			</AlertDialog.Description>
		</AlertDialog.Header>

		<div class="space-y-3 rounded-md border bg-muted/40 p-3 text-xs">
			<div class="flex flex-col gap-1">
				<span class="text-muted-foreground">Repository</span>
				<code class="break-all font-mono">{repoId}</code>
			</div>
			<div class="flex flex-col gap-1">
				<span class="text-muted-foreground">File</span>
				<code class="break-all font-mono">{filePath}</code>
			</div>
			<div class="flex flex-wrap items-center gap-2">
				<span class="rounded bg-primary/15 px-2 py-0.5 font-mono font-semibold text-primary">
					{tagDisplay}
				</span>
				{#if (sizeBytes !== null && sizeBytes > 0) || formattedSize}
					<span class="text-muted-foreground">{formattedSize ?? '&#x2014;'}</span>
				{/if}
				{#if variant}
					<span
						class="rounded bg-primary px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary-foreground"
					>
						{variant}
					</span>
				{/if}
			</div>
			<div class="text-muted-foreground">
				The download runs in the background; you'll see it appear in the models sidebar when it
				finishes.
			</div>
		</div>

		{#if lastError}
			<p class="text-xs text-destructive">{lastError}</p>
		{/if}

		<AlertDialog.Footer>
			<AlertDialog.Cancel disabled={submitting} onclick={onCancel}>Cancel</AlertDialog.Cancel>
			<AlertDialog.Action disabled={submitting} onclick={trigger}>
				{#if submitting}
					<Loader2 class="mr-1.5 h-4 w-4 animate-spin" />
					Starting...
				{:else}
					<Download class="mr-1.5 h-4 w-4" />
					Download
				{/if}
			</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
